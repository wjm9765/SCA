"""Qwen3-Omni model for Full Duplex training.

This module provides the Qwen3OmniDuplexModel that handles interleaved
audio-text training for full duplex conversational AI.

Key differences from the single-turn model:
1. Processes scattered audio tokens (not just prefix audio)
2. Handles multiple speaking segments per sample
3. Uses simplified Talker prefix for duplex segments (no im_start markers)
4. Batches Talker training across segments from different samples
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoFeatureExtractor,
    MimiModel,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from sca_train.data_collator_duplex import SegmentInfo

# Liger Kernel for memory-efficient fused linear cross-entropy
# Only available on Linux; falls back to standard loss on other platforms
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore

    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

# FSDP utilities for accessing sharded parameters
# When FSDP is active, lm_head.weight is sharded and we need summon_full_params
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    FSDP = None  # type: ignore

# Constants for validation
SPEAKER_EMBEDDING_DIM = 192
TARGET_AUDIO_SAMPLE_RATE = 24000


def _assert_valid_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: torch.dtype | None = None,
    name: str = "tensor",
    check_finite: bool = True,
) -> None:
    """Assert that a tensor is valid.

    Args:
        tensor: Tensor to validate.
        expected_shape: Expected shape (use -1 for any dimension).
        expected_dtype: Expected dtype.
        name: Name for error messages.
        check_finite: Whether to check for NaN/Inf (only for float tensors).
    """
    assert isinstance(tensor, torch.Tensor), (
        f"{name}: expected Tensor, got {type(tensor)}"
    )

    if expected_shape is not None:
        assert len(tensor.shape) == len(expected_shape), (
            f"{name}: expected {len(expected_shape)}D, got {len(tensor.shape)}D with shape {tensor.shape}"
        )
        for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
            if expected != -1:
                assert actual == expected, (
                    f"{name}: dim {i} expected {expected}, got {actual} (full shape: {tensor.shape})"
                )

    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, (
            f"{name}: expected {expected_dtype}, got {tensor.dtype}"
        )

    if check_finite and tensor.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ):
        assert torch.isfinite(tensor).all(), f"{name}: contains NaN or Inf values"


def _assert_finite_scalar(value: torch.Tensor, name: str) -> None:
    """Assert that a tensor is a finite scalar."""
    assert isinstance(value, torch.Tensor), (
        f"{name}: expected Tensor, got {type(value)}"
    )
    assert value.numel() == 1, (
        f"{name}: expected scalar (1 element), got {value.numel()} elements"
    )
    assert torch.isfinite(value).all(), f"{name}: is not finite (NaN or Inf)"


@dataclass
class DuplexSegmentBatch:
    """Batched segment data for Talker training.

    Attributes:
        hidden_states: [num_segments, max_seg_len, hidden_dim] - Projected text hidden states.
        segment_lengths: [num_segments] - Original length of each segment.
        target_codes: [num_segments, 16, max_code_len] - Mimi codes for target audio.
        code_lengths: [num_segments] - Original code length per segment.
        speaker_embeddings: [num_segments, 192] - Speaker embedding per segment.
        seg_to_batch: [num_segments] - Maps segment index to batch sample index.
    """

    hidden_states: torch.Tensor
    segment_lengths: torch.Tensor
    target_codes: torch.Tensor
    code_lengths: torch.Tensor
    speaker_embeddings: torch.Tensor
    seg_to_batch: torch.Tensor


class Qwen3OmniDuplexConfig(Qwen3OmniMoeConfig):
    """Config for Qwen3OmniDuplexModel."""

    model_type = "qwen3_omni_duplex"

    def __init__(
        self,
        train_mtp: bool = True,
        mtp_weight: float = 2.0,
        **kwargs,
    ):
        """Initialize duplex config.

        Args:
            train_mtp: Whether to train MTP (Multi-Token Prediction) layers.
            mtp_weight: Weight for MTP loss in total loss computation.
            **kwargs: Additional arguments for Qwen3OmniMoeConfig.
        """
        self.train_mtp = train_mtp
        self.mtp_weight = mtp_weight
        super().__init__(**kwargs)


class Qwen3OmniDuplexModel(Qwen3OmniMoeForConditionalGeneration):
    """Qwen3-Omni model for full duplex training.

    This model handles interleaved audio-text sequences where:
    - Audio tokens are scattered throughout the sequence (not just prefix)
    - Multiple speaking segments may occur per sample
    - The model learns to speak while listening

    The forward pass:
    1. Runs Thinker on the full sequence (text + scattered audio)
    2. Extracts hidden states at speaking segment positions
    3. Encodes target audio to Mimi codes
    4. Runs Talker + MTP training on batched segments
    """

    config_class = Qwen3OmniDuplexConfig

    def __init__(self, config: Qwen3OmniDuplexConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Mimi model loaded separately to avoid weight corruption
        self.mimi_model: Optional[MimiModel] = None
        self.mimi_feature_extractor = None

        # Speaker projection: 192 -> talker hidden size
        speaker_embed_dim = 192
        talker_hidden_size = self.config.talker_config.text_config.hidden_size
        self.speaker_projection = nn.Linear(speaker_embed_dim, talker_hidden_size)

        # Liger fused linear cross-entropy for memory-efficient Thinker loss
        # This avoids materializing the full [seq_len, vocab_size] logits tensor
        if LIGER_AVAILABLE:
            self.thinker_loss_fn = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
            )
            print("[Liger] Using fused linear cross-entropy for Thinker loss")
        else:
            self.thinker_loss_fn = None
            print(
                "[Liger] Not available, using standard cross-entropy for Thinker loss"
            )

    def load_mimi_model(self) -> None:
        """Load Mimi model and feature extractor.

        IMPORTANT: Must be called AFTER from_pretrained() to avoid weight corruption.
        """
        if self.mimi_model is not None:
            print("Warning: Mimi model already loaded, skipping...")
            return

        print("Loading Mimi model from kyutai/mimi...")
        self.mimi_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "kyutai/mimi"
        )
        self.mimi_model = MimiModel.from_pretrained(
            "kyutai/mimi",
            torch_dtype=self.talker.dtype,
        )

        self.mimi_model.eval()
        for param in self.mimi_model.parameters():
            param.requires_grad = False

        # Validate Mimi model loaded correctly
        assert self.mimi_model is not None, "Mimi model failed to load"
        assert self.mimi_feature_extractor is not None, (
            "Mimi feature extractor failed to load"
        )
        assert not self.mimi_model.training, "Mimi model should be in eval mode"

        # Verify all parameters are frozen
        for name, param in self.mimi_model.named_parameters():
            assert not param.requires_grad, (
                f"Mimi parameter {name} should be frozen (requires_grad=False)"
            )

        print("Mimi model loaded successfully!")

    def _get_unwrapped_code_predictor(self):
        """Get code_predictor unwrapped from PEFT wrapper if present."""
        code_predictor = self.talker.code_predictor

        if hasattr(code_predictor, "modules_to_save"):
            active_adapters = getattr(code_predictor, "active_adapters", ["thinker"])
            if active_adapters and active_adapters[0] in code_predictor.modules_to_save:
                code_predictor = code_predictor.modules_to_save[active_adapters[0]]
            elif "thinker" in code_predictor.modules_to_save:
                code_predictor = code_predictor.modules_to_save["thinker"]
            else:
                code_predictor = code_predictor.original_module

        return code_predictor

    def _encode_audio_to_codes(self, audios: list[np.ndarray]) -> torch.Tensor:
        """Encode audio waveforms to Mimi codes.

        Args:
            audios: List of audio waveforms at 24kHz.

        Returns:
            Codes tensor [batch, 16, num_frames].
        """
        assert isinstance(audios, list), (
            f"_encode_audio_to_codes: audios must be list, got {type(audios)}"
        )
        assert len(audios) > 0, "_encode_audio_to_codes: audios list is empty"

        if self.mimi_model is None or self.mimi_feature_extractor is None:
            raise RuntimeError("Mimi model not loaded. Call load_mimi_model() first.")

        # Ensure mimi_model is on correct device
        mimi_device = next(self.mimi_model.parameters()).device
        if mimi_device != self.device:
            self.mimi_model = self.mimi_model.to(self.device)  # type: ignore[arg-type]

        processed_audios = []
        for i, audio in enumerate(audios):
            assert isinstance(audio, np.ndarray), (
                f"_encode_audio_to_codes: audio[{i}] expected ndarray, got {type(audio)}"
            )
            assert np.isfinite(audio).all(), (
                f"_encode_audio_to_codes: audio[{i}] contains NaN or Inf"
            )
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert audio.ndim == 1, (
                f"_encode_audio_to_codes: audio[{i}] expected 1D after processing, got {audio.ndim}D"
            )
            processed_audios.append(audio)

        audio_inputs = self.mimi_feature_extractor(
            processed_audios,
            sampling_rate=TARGET_AUDIO_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        assert "input_values" in audio_inputs, (
            "_encode_audio_to_codes: feature_extractor did not return 'input_values'"
        )
        audio_tensor = audio_inputs["input_values"].to(
            device=self.device,
            dtype=self.mimi_model.dtype,
        )

        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(1)

        assert audio_tensor.ndim == 3, (
            f"_encode_audio_to_codes: audio_tensor expected 3D, got {audio_tensor.ndim}D with shape {audio_tensor.shape}"
        )

        with torch.no_grad():
            output = self.mimi_model.encode(audio_tensor)
            if isinstance(output, tuple):
                codes = output[0]
            elif hasattr(output, "audio_codes"):
                codes = output.audio_codes
            else:
                codes = output

        if not isinstance(codes, torch.Tensor):
            raise ValueError(f"Expected Tensor from mimi encode, got {type(codes)}")

        # Validate output codes shape: [batch, num_quantizers, num_frames]
        batch_size = len(audios)
        assert codes.ndim == 3, (
            f"_encode_audio_to_codes: codes expected 3D, got {codes.ndim}D with shape {codes.shape}"
        )
        assert codes.shape[0] == batch_size, (
            f"_encode_audio_to_codes: codes batch size {codes.shape[0]} != expected {batch_size}"
        )

        return codes

    def _align_codebook_dim(self, codes: torch.Tensor) -> torch.Tensor:
        """Align codebook dimension to expected number of quantizers."""
        assert codes.ndim == 3, (
            f"_align_codebook_dim: codes expected 3D, got {codes.ndim}D"
        )

        target_quantizers = self.code2wav.config.num_quantizers
        assert target_quantizers is not None, (
            "_align_codebook_dim: num_quantizers is None"
        )

        current = codes.shape[1]
        if current == target_quantizers:
            return codes
        if current < target_quantizers:
            raise ValueError(
                f"Mimi codes have {current} quantizers but model expects {target_quantizers}."
            )

        result = codes[:, :target_quantizers, :]
        assert result.shape[1] == target_quantizers, (
            f"_align_codebook_dim: result has {result.shape[1]} quantizers, expected {target_quantizers}"
        )
        return result

    def _extract_segment_hidden_states(
        self,
        input_embeddings: torch.Tensor,
        segment_info: list[list[SegmentInfo]],
    ) -> tuple[list[torch.Tensor], list[int], list[int]]:
        """Extract hidden states for speaking segments.

        For duplex training, we use the input embeddings (hidden_states[0])
        rather than deep hidden states, as per the architecture design.

        Args:
            input_embeddings: [batch, seq_len, hidden_dim] from hidden_states[0].
            segment_info: Per-sample list of SegmentInfo.

        Returns:
            Tuple of:
                - segment_hidden: List of [1, seg_len, hidden_dim] tensors.
                - segment_lengths: List of segment lengths.
                - seg_to_batch: List mapping segment to batch index.
        """
        # Validate inputs
        assert input_embeddings.ndim == 3, (
            f"_extract_segment_hidden_states: input_embeddings expected 3D, got {input_embeddings.ndim}D"
        )
        batch_size, seq_len, hidden_dim = input_embeddings.shape
        assert len(segment_info) == batch_size, (
            f"_extract_segment_hidden_states: segment_info length {len(segment_info)} != batch_size {batch_size}"
        )

        segment_hidden = []
        segment_lengths = []
        seg_to_batch = []

        for batch_idx, sample_segments in enumerate(segment_info):
            for seg_idx, seg in enumerate(sample_segments):
                # Extract hidden states at text_token_idxs
                idxs = seg.text_token_idxs
                if not idxs:
                    continue

                # Validate indices are within sequence bounds
                for idx_pos, idx_val in enumerate(idxs):
                    assert 0 <= idx_val < seq_len, (
                        f"_extract_segment_hidden_states: batch[{batch_idx}].segment[{seg_idx}].text_token_idxs[{idx_pos}] = {idx_val} out of bounds [0, {seq_len})"
                    )

                # Get embeddings at segment positions
                # idxs is a list of indices into the sequence
                seg_emb = input_embeddings[batch_idx, idxs, :]  # [seg_len, hidden_dim]
                seg_emb = seg_emb.unsqueeze(0)  # [1, seg_len, hidden_dim]

                assert seg_emb.shape == (1, len(idxs), hidden_dim), (
                    f"_extract_segment_hidden_states: seg_emb shape {seg_emb.shape} != expected (1, {len(idxs)}, {hidden_dim})"
                )

                segment_hidden.append(seg_emb)
                segment_lengths.append(len(idxs))
                seg_to_batch.append(batch_idx)

        return segment_hidden, segment_lengths, seg_to_batch

    def _encode_segment_audios(
        self,
        segment_info: list[list[SegmentInfo]],
    ) -> tuple[list[torch.Tensor], list[int]]:
        """Encode target audios for all segments.

        Args:
            segment_info: Per-sample list of SegmentInfo.

        Returns:
            Tuple of:
                - segment_codes: List of [1, 16, num_frames] code tensors.
                - code_lengths: List of code lengths per segment.
        """
        segment_codes = []
        code_lengths = []

        # Collect all audios
        all_audios = []
        for sample_segments in segment_info:
            for seg in sample_segments:
                all_audios.append(seg.audio_waveform)

        if not all_audios:
            return [], []

        # Batch encode
        codes = self._encode_audio_to_codes(all_audios).long()
        aligned_codes = self._align_codebook_dim(codes)

        # Validate aligned_codes
        assert aligned_codes.shape[0] == len(all_audios), (
            f"_encode_segment_audios: aligned_codes batch {aligned_codes.shape[0]} != num_audios {len(all_audios)}"
        )

        # Split back to individual segments
        for i in range(len(all_audios)):
            seg_codes = aligned_codes[i : i + 1]  # [1, 16, num_frames]
            assert seg_codes.ndim == 3, (
                f"_encode_segment_audios: seg_codes[{i}] expected 3D, got {seg_codes.ndim}D"
            )
            assert seg_codes.shape[0] == 1, (
                f"_encode_segment_audios: seg_codes[{i}] batch dim expected 1, got {seg_codes.shape[0]}"
            )
            segment_codes.append(seg_codes)
            code_lengths.append(seg_codes.shape[2])

        assert len(segment_codes) == len(all_audios), (
            f"_encode_segment_audios: output length {len(segment_codes)} != num_audios {len(all_audios)}"
        )

        return segment_codes, code_lengths

    def _expand_speaker_to_segments(
        self,
        speaker_embeddings: torch.Tensor,
        seg_to_batch: list[int],
    ) -> torch.Tensor:
        """Expand speaker embeddings to per-segment.

        Args:
            speaker_embeddings: [batch, 192]
            seg_to_batch: Maps segment index to batch index.

        Returns:
            [num_segments, 192] speaker embeddings.
        """
        # Validate inputs
        assert speaker_embeddings.ndim == 2, (
            f"_expand_speaker_to_segments: speaker_embeddings expected 2D, got {speaker_embeddings.ndim}D"
        )
        assert speaker_embeddings.shape[1] == SPEAKER_EMBEDDING_DIM, (
            f"_expand_speaker_to_segments: speaker_embeddings dim {speaker_embeddings.shape[1]} != {SPEAKER_EMBEDDING_DIM}"
        )

        if not seg_to_batch:
            return torch.empty(0, SPEAKER_EMBEDDING_DIM, device=self.device)

        # Validate indices are within bounds
        batch_size = speaker_embeddings.shape[0]
        for i, idx in enumerate(seg_to_batch):
            assert 0 <= idx < batch_size, (
                f"_expand_speaker_to_segments: seg_to_batch[{i}] = {idx} out of bounds [0, {batch_size})"
            )

        indices = torch.tensor(seg_to_batch, device=self.device)
        result = speaker_embeddings[indices]

        # Validate output
        num_segments = len(seg_to_batch)
        assert result.shape == (num_segments, SPEAKER_EMBEDDING_DIM), (
            f"_expand_speaker_to_segments: result shape {result.shape} != expected ({num_segments}, {SPEAKER_EMBEDDING_DIM})"
        )

        return result

    def _build_duplex_talker_prefix(
        self,
        segment_hidden: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build simplified Talker prefix for a duplex segment.

        Unlike the single-turn model which uses im_start/assistant markers,
        the duplex prefix is simpler:
        - Text side: [zeros × 3] + [tts_pad × 4] + [tts_bos] + [first_text_token]
        - Codec side: [zeros × 3] + [nothink, think_bos, think_eos, SPEAKER, pad, bos]

        Args:
            segment_hidden: [1, seg_len, hidden_dim] - Text hidden states.
            speaker_embedding: [1, 192] - Speaker embedding.

        Returns:
            Tuple of (prefix_embed, trailing_text_hidden, tts_pad_embed).
        """
        # Validate inputs
        assert segment_hidden.ndim == 3, (
            f"_build_duplex_talker_prefix: segment_hidden expected 3D, got {segment_hidden.ndim}D"
        )
        assert segment_hidden.shape[0] == 1, (
            f"_build_duplex_talker_prefix: segment_hidden batch expected 1, got {segment_hidden.shape[0]}"
        )
        assert speaker_embedding.ndim == 2, (
            f"_build_duplex_talker_prefix: speaker_embedding expected 2D, got {speaker_embedding.ndim}D"
        )
        assert speaker_embedding.shape == (1, SPEAKER_EMBEDDING_DIM), (
            f"_build_duplex_talker_prefix: speaker_embedding shape {speaker_embedding.shape} != expected (1, {SPEAKER_EMBEDDING_DIM})"
        )

        config = self.config
        hidden_dim = segment_hidden.shape[2]

        # Project segment hidden through text_projection
        projected_hidden = self.talker.text_projection(segment_hidden).to(
            self.talker.device
        )

        # Get thinker embeddings (handling LoRA wrapper)
        thinker_embeddings = self.thinker.get_input_embeddings()
        if hasattr(thinker_embeddings, "base_layer"):
            thinker_embeddings = thinker_embeddings.base_layer

        # Get TTS special token embeddings
        talker_special_tokens = torch.tensor(
            [
                [
                    config.tts_bos_token_id,
                    config.tts_eos_token_id,
                    config.tts_pad_token_id,
                ]
            ],
            device=self.thinker.device,
            dtype=torch.long,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            self.talker.text_projection(thinker_embeddings(talker_special_tokens))
            .to(self.talker.device)
            .chunk(3, dim=1)
        )

        # Build text side of prefix
        # Structure: [zeros × 3] + [tts_pad × 4] + [tts_bos] + [first_text_token]
        zeros_3 = torch.zeros(
            1, 3, hidden_dim, device=self.talker.device, dtype=self.talker.dtype
        )
        text_prefix = torch.cat(
            [
                zeros_3,
                tts_pad_embed.expand(-1, 4, -1),
                tts_bos_embed,
                projected_hidden[:, :1, :],  # First text token
            ],
            dim=1,
        )  # [1, 9, hidden_dim]

        # Build codec side of prefix
        # Structure: [zeros × 3] + [nothink, think_bos, think_eos, SPEAKER, pad, bos]
        codec_special_tokens = torch.tensor(
            [
                [
                    config.talker_config.codec_nothink_id,
                    config.talker_config.codec_think_bos_id,
                    config.talker_config.codec_think_eos_id,
                    config.talker_config.codec_pad_id,  # Placeholder for speaker
                    config.talker_config.codec_pad_id,
                    config.talker_config.codec_bos_id,
                ]
            ],
            device=self.talker.device,
            dtype=torch.long,
        )

        codec_embeds_raw = self.talker.get_input_embeddings()(codec_special_tokens).to(
            self.talker.device
        )

        # Replace speaker position with projected speaker embedding
        projected_speaker = self.speaker_projection(speaker_embedding).to(
            codec_embeds_raw.dtype
        )
        codec_embeds = torch.cat(
            [
                codec_embeds_raw[:, :3, :],  # nothink, think_bos, think_eos
                projected_speaker.unsqueeze(1),  # speaker
                codec_embeds_raw[:, 4:, :],  # pad, bos
            ],
            dim=1,
        )

        codec_prefix = torch.cat(
            [
                torch.zeros(
                    1, 3, hidden_dim, device=self.talker.device, dtype=self.talker.dtype
                ),
                codec_embeds,
            ],
            dim=1,
        )  # [1, 9, hidden_dim]

        # Combine text and codec prefix
        prefix_embed = text_prefix + codec_prefix

        # Trailing text hidden = remaining text tokens + tts_eos
        trailing_text_hidden = torch.cat(
            [projected_hidden[:, 1:, :], tts_eos_embed],
            dim=1,
        )

        # Validate outputs
        assert prefix_embed.shape[0] == 1, (
            f"_build_duplex_talker_prefix: prefix_embed batch expected 1, got {prefix_embed.shape[0]}"
        )
        assert prefix_embed.shape[1] == 9, (
            f"_build_duplex_talker_prefix: prefix_embed seq_len expected 9, got {prefix_embed.shape[1]}"
        )

        return prefix_embed, trailing_text_hidden, tts_pad_embed

    def _forward_talker_single_segment(
        self,
        segment_hidden: torch.Tensor,
        speaker_embedding: torch.Tensor,
        target_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Talker on a single segment.

        Args:
            segment_hidden: [1, seg_len, hidden_dim] - Text hidden states.
            speaker_embedding: [1, 192] - Speaker embedding.
            target_codes: [1, 16, num_frames] - Target Mimi codes.

        Returns:
            Tuple of (talker_loss, mtp_loss).
        """
        # Validate inputs
        assert segment_hidden.ndim == 3 and segment_hidden.shape[0] == 1, (
            f"_forward_talker_single_segment: segment_hidden expected [1, seg_len, hidden_dim], got {segment_hidden.shape}"
        )
        assert speaker_embedding.ndim == 2 and speaker_embedding.shape[0] == 1, (
            f"_forward_talker_single_segment: speaker_embedding expected [1, 192], got {speaker_embedding.shape}"
        )
        assert target_codes.ndim == 3 and target_codes.shape[0] == 1, (
            f"_forward_talker_single_segment: target_codes expected [1, num_q, num_frames], got {target_codes.shape}"
        )

        config = self.config

        # Build prefix
        prefix_embed, trailing_text_hidden, tts_pad_embed = (
            self._build_duplex_talker_prefix(segment_hidden, speaker_embedding)
        )

        # Get layer 0 codes
        layer0_codes = target_codes[:, 0, :]  # [1, num_frames]
        num_codec_tokens = layer0_codes.shape[1]
        assert num_codec_tokens > 0, (
            "_forward_talker_single_segment: num_codec_tokens is 0"
        )

        # Get embeddings for all layers (for teacher forcing)
        layer0_embeddings = self.talker.get_input_embeddings()(
            layer0_codes.to(self.device)
        )

        unwrapped_code_predictor = self._get_unwrapped_code_predictor()
        predictor_embeds = unwrapped_code_predictor.get_input_embeddings()

        # Sum embeddings from all layers
        all_layer_embeds_sum = layer0_embeddings.clone()
        for j in range(len(predictor_embeds)):
            layer_j_codes = target_codes[:, j + 1, :]
            emb = predictor_embeds[j](layer_j_codes.to(self.device))
            all_layer_embeds_sum = all_layer_embeds_sum + emb

        # Build codec input sequence (teacher forcing)
        trailing_len = trailing_text_hidden.shape[1]
        codec_input_embeds_list = []

        for pos in range(num_codec_tokens):
            if pos == 0:
                continue
            prev_pos = pos - 1
            if prev_pos < trailing_len:
                text_hidden = trailing_text_hidden[:, prev_pos : prev_pos + 1, :]
            else:
                text_hidden = tts_pad_embed
            pos_embed = (
                all_layer_embeds_sum[:, prev_pos : prev_pos + 1, :] + text_hidden
            )
            codec_input_embeds_list.append(pos_embed)

        # EOS token
        last_pos = num_codec_tokens - 1
        if last_pos < trailing_len:
            eos_text_hidden = trailing_text_hidden[:, last_pos : last_pos + 1, :]
        else:
            eos_text_hidden = tts_pad_embed
        eos_input_embed = (
            all_layer_embeds_sum[:, last_pos : last_pos + 1, :] + eos_text_hidden
        )
        codec_input_embeds_list.append(eos_input_embed)

        if codec_input_embeds_list:
            codec_input_embeds = torch.cat(codec_input_embeds_list, dim=1).to(
                self.talker.dtype
            )
            full_inputs_embeds = torch.cat([prefix_embed, codec_input_embeds], dim=1)
        else:
            full_inputs_embeds = prefix_embed

        # Build labels
        prefix_len = prefix_embed.shape[1]
        codec_eos_id = config.talker_config.codec_eos_token_id

        # Labels: -100 for prefix[:-1], code[0] at prefix[-1], code[1:] + EOS for rest
        prefix_mask = torch.full(
            (1, prefix_len - 1), -100, dtype=torch.long, device=self.device
        )
        label_code0 = layer0_codes[:, 0:1]
        label_code_rest = layer0_codes[:, 1:]
        label_eos = torch.full(
            (1, 1), codec_eos_id, dtype=torch.long, device=self.device
        )

        labels = torch.cat(
            [prefix_mask, label_code0, label_code_rest, label_eos], dim=1
        )

        # Validate labels length matches inputs
        assert labels.shape[1] == full_inputs_embeds.shape[1], (
            f"_forward_talker_single_segment: labels length {labels.shape[1]} != inputs length {full_inputs_embeds.shape[1]}"
        )

        # Attention mask (all ones for single segment)
        seq_len = full_inputs_embeds.shape[1]
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)

        # Forward through Talker
        talker_outputs = self.talker(
            inputs_embeds=full_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=None,
            audio_feature_lengths=None,
            image_grid_thw=None,
            video_grid_thw=None,
            video_second_per_grid=None,
            output_hidden_states=True,
        )
        talker_loss = talker_outputs.loss

        # Validate talker_loss is finite
        _assert_finite_scalar(
            talker_loss, "_forward_talker_single_segment: talker_loss"
        )

        # MTP training
        if not config.train_mtp:
            mtp_loss = torch.tensor(0.0, device=self.device)
        else:
            # Get last hidden state
            if isinstance(talker_outputs.hidden_states, tuple):
                talker_hidden = talker_outputs.hidden_states[0][-1]
            else:
                talker_hidden = talker_outputs.hidden_states[-1]

            # Extract codec hidden states
            # h0 at prefix[-1], h_rest at [prefix_len : prefix_len + num_codec_tokens - 1]
            h0 = talker_hidden[:, prefix_len - 1 : prefix_len, :]
            h_rest = talker_hidden[:, prefix_len : prefix_len + num_codec_tokens - 1, :]
            codec_hidden = torch.cat(
                [h0, h_rest], dim=1
            )  # [1, num_codec_tokens, hidden]

            hidden_dim = codec_hidden.shape[2]
            hidden_flat = codec_hidden.reshape(-1, 1, hidden_dim)
            layer0_flat = layer0_embeddings.reshape(-1, 1, hidden_dim)

            mtp_total_loss = torch.tensor(0.0, device=self.device)
            num_mtp_layers = 15

            for mtp_layer_idx in range(num_mtp_layers):
                embed_list = [hidden_flat, layer0_flat]

                for prev_layer in range(mtp_layer_idx):
                    prev_codes = target_codes[:, prev_layer + 1, :].to(self.device)
                    prev_embed = predictor_embeds[prev_layer](prev_codes)
                    prev_embed_flat = prev_embed.reshape(-1, 1, hidden_dim)
                    embed_list.append(prev_embed_flat)

                mtp_inputs = torch.cat(embed_list, dim=1).to(self.talker.dtype)

                target_layer_codes = target_codes[:, mtp_layer_idx + 1, :].to(
                    self.device
                )
                target_labels = target_layer_codes.reshape(-1)

                mtp_outputs = unwrapped_code_predictor(
                    input_ids=None,
                    inputs_embeds=mtp_inputs,
                    generation_steps=mtp_layer_idx,
                    use_cache=False,
                )

                mtp_logits = mtp_outputs.logits[:, -1, :]
                mtp_layer_loss = F.cross_entropy(mtp_logits, target_labels)
                mtp_total_loss = mtp_total_loss + mtp_layer_loss

            mtp_loss = mtp_total_loss / num_mtp_layers

            # Validate mtp_loss is finite
            _assert_finite_scalar(mtp_loss, "_forward_talker_single_segment: mtp_loss")

        return talker_loss, mtp_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        segment_info: list[list[SegmentInfo]],
        speaker_embeddings: torch.Tensor,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for duplex training.

        Args:
            input_ids: [batch, seq_len] with audio_token_id at audio positions.
            attention_mask: [batch, seq_len].
            labels: [batch, seq_len] with -100 at audio positions.
            input_features: [batch, mel_dim, mel_len] - Mel spectrograms.
            feature_attention_mask: [batch, mel_len].
            segment_info: Per-sample list of SegmentInfo.
            speaker_embeddings: [batch, 192].
            **kwargs: Additional arguments.

        Returns:
            CausalLMOutputWithPast with combined loss.
        """
        # Validate inputs
        assert input_ids.ndim == 2, (
            f"forward: input_ids expected 2D, got {input_ids.ndim}D"
        )
        batch_size, seq_len = input_ids.shape

        assert attention_mask.shape == (batch_size, seq_len), (
            f"forward: attention_mask shape {attention_mask.shape} != expected ({batch_size}, {seq_len})"
        )
        assert labels.shape == (batch_size, seq_len), (
            f"forward: labels shape {labels.shape} != expected ({batch_size}, {seq_len})"
        )
        assert input_features.ndim == 3 and input_features.shape[0] == batch_size, (
            f"forward: input_features expected [batch, mel_dim, mel_len], got {input_features.shape}"
        )
        mel_len = input_features.shape[2]
        assert feature_attention_mask.shape == (batch_size, mel_len), (
            f"forward: feature_attention_mask shape {feature_attention_mask.shape} != expected ({batch_size}, {mel_len})"
        )
        assert len(segment_info) == batch_size, (
            f"forward: segment_info length {len(segment_info)} != batch_size {batch_size}"
        )
        assert speaker_embeddings.shape == (batch_size, SPEAKER_EMBEDDING_DIM), (
            f"forward: speaker_embeddings shape {speaker_embeddings.shape} != expected ({batch_size}, {SPEAKER_EMBEDDING_DIM})"
        )

        # Filter kwargs to only pass safe parameters to thinker
        # Remove parameters that could conflict with our explicit settings or cause issues
        safe_kwargs = {}
        unsafe_keys = {
            "output_hidden_states",  # We always set this to True
            "use_cache",  # Not needed for training, can cause issues
            "past_key_values",  # Not needed for training
            "cache_position",  # Computed internally by thinker
            "return_dict",  # We expect dict outputs
            "num_items_in_batch",  # Trainer artifact
        }
        for key, value in kwargs.items():
            if key not in unsafe_keys:
                safe_kwargs[key] = value

        # 1. Run Thinker on full sequence
        # NOTE: We pass labels=None to skip the internal loss computation.
        # This is because the internal loss materializes the full [seq_len, vocab_size]
        # logits tensor which causes OOM with long sequences. Instead, we compute
        # the loss using Liger's fused linear cross-entropy which never materializes
        # the full logits tensor.
        thinker_outputs = self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Skip internal loss - we compute it ourselves
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **safe_kwargs,
        )

        # Validate thinker_outputs has expected attributes
        assert hasattr(thinker_outputs, "hidden_states"), (
            "forward: thinker_outputs missing hidden_states"
        )
        assert thinker_outputs.hidden_states is not None, (
            "forward: thinker_outputs.hidden_states is None"
        )
        assert len(thinker_outputs.hidden_states) > 0, (
            "forward: thinker_outputs.hidden_states is empty"
        )

        # Compute Thinker loss using Liger's fused linear cross-entropy
        # This avoids materializing the full [seq_len, vocab_size] logits tensor
        # hidden_states[-1] is the last layer output (before lm_head)
        last_hidden_state = thinker_outputs.hidden_states[
            -1
        ]  # [batch, seq_len, hidden]

        if self.thinker_loss_fn is not None:
            # Use Liger fused linear cross-entropy (memory efficient)
            # Shift hidden states and labels for next-token prediction
            shift_hidden = last_hidden_state[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for the loss function: [batch * (seq_len-1), hidden]
            shift_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
            shift_labels = shift_labels.view(-1)

            # LigerFusedLinearCrossEntropyLoss.forward(lin_weight, _input, target, bias=None)
            # With FSDP, lm_head.weight is sharded after forward pass completes.
            # We need to use summon_full_params to temporarily unshard the weight.
            lm_head = self.thinker.lm_head
            if FSDP_AVAILABLE and isinstance(lm_head, FSDP):
                # lm_head is directly wrapped in FSDP - unshard it
                with FSDP.summon_full_params(lm_head, writeback=False):
                    thinker_loss = self.thinker_loss_fn(
                        lm_head.weight,  # [vocab_size, hidden]
                        shift_hidden,  # [batch * (seq_len-1), hidden]
                        shift_labels,  # [batch * (seq_len-1)]
                    )
            elif FSDP_AVAILABLE and isinstance(self.thinker, FSDP):
                # lm_head is part of an FSDP-wrapped thinker - unshard the whole thinker
                with FSDP.summon_full_params(self.thinker, writeback=False):
                    thinker_loss = self.thinker_loss_fn(
                        lm_head.weight,  # [vocab_size, hidden]
                        shift_hidden,  # [batch * (seq_len-1), hidden]
                        shift_labels,  # [batch * (seq_len-1)]
                    )
            else:
                # Not FSDP wrapped - direct access
                thinker_loss = self.thinker_loss_fn(
                    lm_head.weight,  # [vocab_size, hidden]
                    shift_hidden,  # [batch * (seq_len-1), hidden]
                    shift_labels,  # [batch * (seq_len-1)]
                )
        else:
            # Fallback: standard loss computation (will materialize logits)
            # This path is only used on non-Linux systems where Liger is unavailable
            logits = self.thinker.lm_head(last_hidden_state)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            thinker_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        _assert_finite_scalar(thinker_loss, "forward: thinker_loss")

        # 2. Extract segment hidden states (from hidden_states[0] = input embeddings)
        input_embeddings = thinker_outputs.hidden_states[0]  # [batch, seq_len, hidden]
        assert input_embeddings.shape[0] == batch_size, (
            f"forward: input_embeddings batch {input_embeddings.shape[0]} != expected {batch_size}"
        )

        segment_hidden_list, segment_lengths, seg_to_batch = (
            self._extract_segment_hidden_states(input_embeddings, segment_info)
        )

        # 3. If no segments, return thinker loss only
        if len(segment_hidden_list) == 0:
            self._last_thinker_loss = thinker_loss.detach()
            self._last_talker_loss = torch.tensor(0.0, device=self.device)
            self._last_mtp_loss = torch.tensor(0.0, device=self.device)
            return CausalLMOutputWithPast(
                loss=thinker_loss,  # type: ignore[arg-type]
                logits=thinker_outputs.logits,
                past_key_values=thinker_outputs.past_key_values,
                hidden_states=thinker_outputs.hidden_states,
                attentions=thinker_outputs.attentions,
            )

        # 4. Encode target audios
        segment_codes_list, code_lengths = self._encode_segment_audios(segment_info)
        assert len(segment_codes_list) == len(segment_hidden_list), (
            f"forward: segment_codes_list length {len(segment_codes_list)} != segment_hidden_list length {len(segment_hidden_list)}"
        )

        # 5. Expand speaker embeddings to segments
        segment_speakers = self._expand_speaker_to_segments(
            speaker_embeddings, seg_to_batch
        )

        # 6. Process each segment through Talker + MTP
        # For stability, we process segments one at a time
        total_talker_loss = torch.tensor(0.0, device=self.device)
        total_mtp_loss = torch.tensor(0.0, device=self.device)
        num_segments = len(segment_hidden_list)

        for seg_idx in range(num_segments):
            seg_hidden = segment_hidden_list[seg_idx]
            seg_codes = segment_codes_list[seg_idx]
            seg_speaker = segment_speakers[seg_idx : seg_idx + 1]

            talker_loss_seg, mtp_loss_seg = self._forward_talker_single_segment(
                seg_hidden, seg_speaker, seg_codes
            )

            total_talker_loss = total_talker_loss + talker_loss_seg
            total_mtp_loss = total_mtp_loss + mtp_loss_seg

        # Average losses over segments
        avg_talker_loss = total_talker_loss / num_segments
        avg_mtp_loss = total_mtp_loss / num_segments

        # 7. Combine losses
        mtp_weight = getattr(self.config, "mtp_weight", 2.0)
        total_loss = thinker_loss + avg_talker_loss + (mtp_weight * avg_mtp_loss)

        # Validate final loss is finite
        _assert_finite_scalar(total_loss, "forward: total_loss")

        # Store for logging
        self._last_thinker_loss = thinker_loss.detach()
        self._last_talker_loss = avg_talker_loss.detach()
        self._last_mtp_loss = avg_mtp_loss.detach()

        return CausalLMOutputWithPast(
            loss=total_loss,  # type: ignore[arg-type]
            logits=thinker_outputs.logits,
            past_key_values=thinker_outputs.past_key_values,
            hidden_states=thinker_outputs.hidden_states,
            attentions=thinker_outputs.attentions,
        )

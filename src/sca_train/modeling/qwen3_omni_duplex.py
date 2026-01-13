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
        if self.mimi_model is None:
            raise RuntimeError("Mimi model not loaded. Call load_mimi_model() first.")

        # Ensure mimi_model is on correct device
        mimi_device = next(self.mimi_model.parameters()).device
        if mimi_device != self.device:
            self.mimi_model = self.mimi_model.to(self.device)

        processed_audios = []
        for audio in audios:
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            processed_audios.append(audio)

        audio_inputs = self.mimi_feature_extractor(
            processed_audios, sampling_rate=24000, return_tensors="pt", padding=True
        )

        audio_tensor = audio_inputs["input_values"].to(
            device=self.device,
            dtype=self.mimi_model.dtype,
        )

        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(1)

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

        return codes

    def _align_codebook_dim(self, codes: torch.Tensor) -> torch.Tensor:
        """Align codebook dimension to expected number of quantizers."""
        target_quantizers = self.code2wav.config.num_quantizers
        assert target_quantizers is not None

        current = codes.shape[1]
        if current == target_quantizers:
            return codes
        if current < target_quantizers:
            raise ValueError(
                f"Mimi codes have {current} quantizers but model expects {target_quantizers}."
            )
        return codes[:, :target_quantizers, :]

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
        segment_hidden = []
        segment_lengths = []
        seg_to_batch = []

        for batch_idx, sample_segments in enumerate(segment_info):
            for seg in sample_segments:
                # Extract hidden states at text_token_idxs
                idxs = seg.text_token_idxs
                if not idxs:
                    continue

                # Get embeddings at segment positions
                # idxs is a list of indices into the sequence
                seg_emb = input_embeddings[batch_idx, idxs, :]  # [seg_len, hidden_dim]
                seg_emb = seg_emb.unsqueeze(0)  # [1, seg_len, hidden_dim]

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

        # Split back to individual segments
        for i in range(len(all_audios)):
            seg_codes = aligned_codes[i : i + 1]  # [1, 16, num_frames]
            segment_codes.append(seg_codes)
            code_lengths.append(seg_codes.shape[2])

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
        if not seg_to_batch:
            return torch.empty(0, 192, device=self.device)

        indices = torch.tensor(seg_to_batch, device=self.device)
        return speaker_embeddings[indices]

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
        config = self.config

        # Build prefix
        prefix_embed, trailing_text_hidden, tts_pad_embed = (
            self._build_duplex_talker_prefix(segment_hidden, speaker_embedding)
        )

        # Get layer 0 codes
        layer0_codes = target_codes[:, 0, :]  # [1, num_frames]
        num_codec_tokens = layer0_codes.shape[1]

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
        # 1. Run Thinker on full sequence
        thinker_outputs = self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        thinker_loss = thinker_outputs.loss
        if thinker_loss is None:
            thinker_loss = torch.tensor(0.0, device=self.device)

        # 2. Extract segment hidden states (from hidden_states[0] = input embeddings)
        input_embeddings = thinker_outputs.hidden_states[0]  # [batch, seq_len, hidden]
        segment_hidden_list, segment_lengths, seg_to_batch = (
            self._extract_segment_hidden_states(input_embeddings, segment_info)
        )

        # 3. If no segments, return thinker loss only
        if len(segment_hidden_list) == 0:
            self._last_thinker_loss = thinker_loss.detach()
            self._last_talker_loss = torch.tensor(0.0, device=self.device)
            self._last_mtp_loss = torch.tensor(0.0, device=self.device)
            return CausalLMOutputWithPast(
                loss=thinker_loss,
                logits=thinker_outputs.logits,
                past_key_values=thinker_outputs.past_key_values,
                hidden_states=thinker_outputs.hidden_states,
                attentions=thinker_outputs.attentions,
            )

        # 4. Encode target audios
        segment_codes_list, code_lengths = self._encode_segment_audios(segment_info)

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

        # Store for logging
        self._last_thinker_loss = thinker_loss.detach()
        self._last_talker_loss = avg_talker_loss.detach()
        self._last_mtp_loss = avg_mtp_loss.detach()

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=thinker_outputs.logits,
            past_key_values=thinker_outputs.past_key_values,
            hidden_states=thinker_outputs.hidden_states,
            attentions=thinker_outputs.attentions,
        )

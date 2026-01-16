"""Data collator for Full Duplex training.

This module provides the FullDuplexCollator that processes DatasetRow instances
into batched tensors for training the duplex model.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from sca_data.dataset_utils import DatasetRow

if TYPE_CHECKING:
    from transformers import Qwen3OmniMoeProcessor


# Silence token ID for Qwen3-Omni
SILENCE_TOKEN_ID = 151646

# Expected dimensions
SPEAKER_EMBEDDING_DIM = 192
INPUT_AUDIO_SAMPLE_RATE = 16000
TARGET_AUDIO_SAMPLE_RATE = 24000


def _assert_valid_audio_waveform(
    waveform: np.ndarray, expected_sr: int, name: str
) -> None:
    """Assert that an audio waveform is valid.

    Args:
        waveform: Audio waveform as numpy array.
        expected_sr: Expected sample rate (for error message context).
        name: Name for error messages.
    """
    assert isinstance(waveform, np.ndarray), (
        f"{name}: expected ndarray, got {type(waveform)}"
    )
    assert waveform.ndim == 1, (
        f"{name}: expected 1D array, got {waveform.ndim}D with shape {waveform.shape}"
    )
    assert waveform.dtype in (np.float32, np.float64), (
        f"{name}: expected float32/64, got {waveform.dtype}"
    )
    assert np.isfinite(waveform).all(), f"{name}: contains NaN or Inf values"
    assert len(waveform) > 0, f"{name}: waveform is empty"


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


@dataclass
class SegmentInfo:
    """Information about a speaking segment for Talker training.

    This is a simplified representation passed to the model for each segment.

    Attributes:
        text_token_idxs: Indices into the input sequence where model speaks.
        audio_waveform: Target audio waveform at 24kHz as numpy array.
    """

    text_token_idxs: list[int]
    audio_waveform: np.ndarray


class FullDuplexCollator:
    """Collator for full duplex interleaved audio-text training.

    This collator processes DatasetRow instances from the HuggingFace dataset
    and produces batched tensors for the Qwen3OmniDuplexModel.

    The collator handles:
    1. Replacing -100 placeholders with audio_token_id
    2. Processing input audios through the processor's feature extractor
    3. Building labels with -100 at audio positions
    4. Extracting segment info for Talker training
    5. Batching speaker embeddings

    Attributes:
        audio_token_id: Token ID for audio placeholders (from model config).
        silence_token_id: Token ID for silence tokens (from global config).
        pad_token_id: Token ID for padding.
        max_length: Maximum sequence length (will truncate if exceeded).
        max_segments_per_batch: Maximum number of speaking segments per batch.
        feature_extractor: Feature extractor from the processor for mel spectrograms.
    """

    def __init__(
        self,
        processor: "Qwen3OmniMoeProcessor",
        audio_token_id: int,
        silence_token_id: int,
        pad_token_id: int,
        max_length: int = 32768,
        max_segments_per_batch: int = 8,
    ):
        """Initialize the collator.

        Args:
            processor: Qwen3OmniMoeProcessor with feature_extractor attribute.
            audio_token_id: Token ID for audio placeholders (e.g., 151675).
            silence_token_id: Token ID for silence tokens.
            pad_token_id: Token ID for padding.
            max_length: Maximum sequence length.
            max_segments_per_batch: Maximum speaking segments per batch.
        """
        if not hasattr(processor, "feature_extractor"):
            raise ValueError(
                "Processor must have a feature_extractor attribute. "
                "Use Qwen3OmniMoeProcessor.from_pretrained() to load the processor."
            )

        self.audio_token_id = audio_token_id
        self.silence_token_id = silence_token_id
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.max_segments_per_batch = max_segments_per_batch

        # Use the feature extractor from the processor (AuT-style)
        self.feature_extractor: Callable[..., dict[str, Any]] = (
            processor.feature_extractor
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of features into model inputs.

        Args:
            features: List of dicts with "dataset_row_obj" key containing DatasetRow.

        Returns:
            Batched dict with:
                - input_ids: [batch, seq_len] with audio_token_id at audio positions
                - attention_mask: [batch, seq_len]
                - labels: [batch, seq_len] with -100 at audio positions
                - input_features: [batch, mel_dim, max_mel_len] mel spectrograms
                - feature_attention_mask: [batch, max_mel_len]
                - segment_info: list[list[SegmentInfo]] per sample per segment
                - speaker_embeddings: [batch, 192]
        """
        # Validate inputs
        assert isinstance(features, list), (
            f"features must be a list, got {type(features)}"
        )
        assert len(features) > 0, "features list is empty"

        # Extract DatasetRow from each feature
        rows: list[DatasetRow] = []
        for i, feature in enumerate(features):
            assert "dataset_row_obj" in feature, (
                f"feature[{i}] missing 'dataset_row_obj' key"
            )
            row = feature["dataset_row_obj"]
            rows.append(row)

        # 1. Build input_ids and labels
        input_ids_list, labels_list, audio_positions_list = self._build_sequences(rows)

        # 2. Pad sequences
        input_ids, attention_mask, labels = self._pad_sequences(
            input_ids_list, labels_list
        )

        # 3. Process input audios to mel spectrograms
        input_features, feature_attention_mask = self._process_input_audios(rows)

        # 4. Extract segment info for Talker
        segment_info = self._extract_segment_info(rows)

        # 5. Stack speaker embeddings
        speaker_embeddings = self._stack_speaker_embeddings(rows)

        # Validate outputs
        batch_size = len(rows)
        seq_len = input_ids.shape[1]
        mel_len = input_features.shape[2]

        _assert_valid_tensor(
            input_ids,
            (batch_size, seq_len),
            torch.long,
            "input_ids",
            check_finite=False,
        )
        _assert_valid_tensor(
            attention_mask,
            (batch_size, seq_len),
            torch.long,
            "attention_mask",
            check_finite=False,
        )
        _assert_valid_tensor(
            labels, (batch_size, seq_len), torch.long, "labels", check_finite=False
        )
        _assert_valid_tensor(
            input_features,
            (batch_size, -1, mel_len),
            None,
            "input_features",
            check_finite=True,
        )
        _assert_valid_tensor(
            feature_attention_mask,
            (batch_size, mel_len),
            torch.long,
            "feature_attention_mask",
            check_finite=False,
        )
        _assert_valid_tensor(
            speaker_embeddings,
            (batch_size, SPEAKER_EMBEDDING_DIM),
            None,
            "speaker_embeddings",
            check_finite=True,
        )

        # Validate attention_mask values are 0 or 1
        assert ((attention_mask == 0) | (attention_mask == 1)).all(), (
            "attention_mask must contain only 0s and 1s"
        )
        assert ((feature_attention_mask == 0) | (feature_attention_mask == 1)).all(), (
            "feature_attention_mask must contain only 0s and 1s"
        )

        # Validate segment_info structure
        assert len(segment_info) == batch_size, (
            f"segment_info length {len(segment_info)} != batch_size {batch_size}"
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "segment_info": segment_info,
            "speaker_embeddings": speaker_embeddings,
        }

    def _build_sequences(
        self, rows: list[DatasetRow]
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """Build input_ids and labels from DatasetRows.

        Replaces -100 with audio_token_id in input_ids.
        Keeps -100 in labels at audio positions.

        Returns:
            Tuple of (input_ids_list, labels_list, audio_positions_list).
        """
        assert len(rows) > 0, "_build_sequences: rows list is empty"

        input_ids_list = []
        labels_list = []
        audio_positions_list = []

        for row_idx, row in enumerate(rows):
            # Validate input_sequence
            input_seq = row.input_sequence
            assert len(input_seq) > 0, f"row[{row_idx}]: input_sequence is empty"

            input_ids = []
            labels = []
            audio_positions = []

            for i, token in enumerate(input_seq):
                assert isinstance(token, int), (
                    f"row[{row_idx}] token[{i}]: expected int, got {type(token)}"
                )
                if token == -100:
                    # Replace with audio_token_id for input, keep -100 for labels
                    input_ids.append(self.audio_token_id)
                    labels.append(-100)
                    audio_positions.append(i)
                else:
                    input_ids.append(token)
                    labels.append(token)

            # Truncate if too long
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]
                audio_positions = [p for p in audio_positions if p < self.max_length]

            assert len(input_ids) > 0, (
                f"row[{row_idx}]: input_ids is empty after processing"
            )
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            audio_positions_list.append(audio_positions)

        return input_ids_list, labels_list, audio_positions_list

    def _pad_sequences(
        self, input_ids_list: list[list[int]], labels_list: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad sequences to same length and create attention mask.

        Returns:
            Tuple of (input_ids, attention_mask, labels) tensors.
        """
        assert len(input_ids_list) > 0, "_pad_sequences: input_ids_list is empty"
        assert len(input_ids_list) == len(labels_list), (
            f"_pad_sequences: input_ids_list length {len(input_ids_list)} != labels_list length {len(labels_list)}"
        )

        max_len = max(len(seq) for seq in input_ids_list)
        assert max_len > 0, "_pad_sequences: max sequence length is 0"

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            seq_len = len(input_ids)
            pad_len = max_len - seq_len

            # Pad input_ids with pad_token_id
            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)

            # Pad labels with -100 (ignore in loss)
            padded_labels.append(labels + [-100] * pad_len)

            # Attention mask: 1 for real tokens, 0 for padding
            attention_masks.append([1] * seq_len + [0] * pad_len)

        result_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        result_attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        result_labels = torch.tensor(padded_labels, dtype=torch.long)

        # Validate output shapes
        batch_size = len(input_ids_list)
        assert result_input_ids.shape == (batch_size, max_len), (
            f"_pad_sequences: input_ids shape {result_input_ids.shape} != expected ({batch_size}, {max_len})"
        )
        assert result_attention_mask.shape == (batch_size, max_len), (
            f"_pad_sequences: attention_mask shape {result_attention_mask.shape} != expected ({batch_size}, {max_len})"
        )
        assert result_labels.shape == (batch_size, max_len), (
            f"_pad_sequences: labels shape {result_labels.shape} != expected ({batch_size}, {max_len})"
        )

        return result_input_ids, result_attention_mask, result_labels

    def _process_input_audios(
        self, rows: list[DatasetRow]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input audios into mel spectrograms.

        Concatenates all input audio chunks per sample, then extracts features.
        Uses padding=False for individual samples to handle variable lengths,
        then manually pads the batch.

        Returns:
            Tuple of (input_features, feature_attention_mask).
            input_features: [batch, mel_dim, max_mel_len]
            feature_attention_mask: [batch, max_mel_len]
        """
        assert len(rows) > 0, "_process_input_audios: rows list is empty"

        all_features = []
        all_lengths = []

        for row_idx, row in enumerate(rows):
            if len(row.input_audios) == 0:
                # No audio - create minimal placeholder
                # Use a small silence chunk
                concat_audio = np.zeros(1600, dtype=np.float32)  # 100ms at 16kHz
            else:
                # Validate each audio chunk
                for audio_idx, audio in enumerate(row.input_audios):
                    _assert_valid_audio_waveform(
                        audio.waveform,
                        INPUT_AUDIO_SAMPLE_RATE,
                        f"row[{row_idx}].input_audios[{audio_idx}]",
                    )

                # Concatenate all input audio chunks
                concat_audio = np.concatenate(
                    [audio.waveform for audio in row.input_audios]
                )

            assert len(concat_audio) > 0, f"row[{row_idx}]: concatenated audio is empty"

            # Extract features without padding
            # Returns dict with "input_features" key
            features = self.feature_extractor(
                concat_audio,
                sampling_rate=INPUT_AUDIO_SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )

            assert "input_features" in features, (
                f"row[{row_idx}]: feature_extractor did not return 'input_features'"
            )
            mel = features["input_features"]  # [1, mel_dim, mel_len]

            assert mel.ndim == 3, (
                f"row[{row_idx}]: mel expected 3D, got {mel.ndim}D with shape {mel.shape}"
            )
            assert mel.shape[0] == 1, (
                f"row[{row_idx}]: mel batch dim expected 1, got {mel.shape[0]}"
            )

            all_features.append(mel.squeeze(0))  # [mel_dim, mel_len]
            all_lengths.append(mel.shape[-1])

        # Pad to max length
        max_mel_len = max(all_lengths)
        mel_dim = all_features[0].shape[0]

        # Validate mel_dim is reasonable (typically 80 or 128)
        assert mel_dim in (80, 128), (
            f"_process_input_audios: unexpected mel_dim {mel_dim}, expected 80 or 128"
        )

        padded_features = torch.zeros(
            len(rows), mel_dim, max_mel_len, dtype=all_features[0].dtype
        )
        attention_mask = torch.zeros(len(rows), max_mel_len, dtype=torch.long)

        for i, (mel, length) in enumerate(zip(all_features, all_lengths)):
            padded_features[i, :, :length] = mel
            attention_mask[i, :length] = 1

        # Validate output shapes
        batch_size = len(rows)
        assert padded_features.shape == (batch_size, mel_dim, max_mel_len), (
            f"_process_input_audios: padded_features shape {padded_features.shape} != expected ({batch_size}, {mel_dim}, {max_mel_len})"
        )
        assert attention_mask.shape == (batch_size, max_mel_len), (
            f"_process_input_audios: attention_mask shape {attention_mask.shape} != expected ({batch_size}, {max_mel_len})"
        )

        return padded_features, attention_mask

    def _extract_segment_info(self, rows: list[DatasetRow]) -> list[list[SegmentInfo]]:
        """Extract segment info for Talker training.

        Returns:
            List of segment info lists, one per sample in batch.
            Each segment contains text_token_idxs and target audio waveform.
        """
        assert len(rows) > 0, "_extract_segment_info: rows list is empty"

        batch_segment_info = []
        total_segments = 0

        for row_idx, row in enumerate(rows):
            sample_segments = []

            for seg_idx, seg in enumerate(row.target_audios):
                # Check max segments limit
                if total_segments >= self.max_segments_per_batch:
                    break

                # Validate text_token_idxs
                assert isinstance(seg.text_token_idxs, list), (
                    f"row[{row_idx}].target_audios[{seg_idx}]: text_token_idxs expected list, got {type(seg.text_token_idxs)}"
                )
                assert len(seg.text_token_idxs) > 0, (
                    f"row[{row_idx}].target_audios[{seg_idx}]: text_token_idxs is empty"
                )
                for idx_pos, idx_val in enumerate(seg.text_token_idxs):
                    assert isinstance(idx_val, int), (
                        f"row[{row_idx}].target_audios[{seg_idx}].text_token_idxs[{idx_pos}]: expected int, got {type(idx_val)}"
                    )
                    assert idx_val >= 0, (
                        f"row[{row_idx}].target_audios[{seg_idx}].text_token_idxs[{idx_pos}]: negative index {idx_val}"
                    )

                # Validate target audio waveform
                _assert_valid_audio_waveform(
                    seg.audio.waveform,
                    TARGET_AUDIO_SAMPLE_RATE,
                    f"row[{row_idx}].target_audios[{seg_idx}].audio",
                )

                sample_segments.append(
                    SegmentInfo(
                        text_token_idxs=seg.text_token_idxs,
                        audio_waveform=seg.audio.waveform,
                    )
                )
                total_segments += 1

            batch_segment_info.append(sample_segments)

            if total_segments >= self.max_segments_per_batch:
                break

        # Pad remaining samples with empty lists if we hit the limit
        while len(batch_segment_info) < len(rows):
            batch_segment_info.append([])

        assert len(batch_segment_info) == len(rows), (
            f"_extract_segment_info: output length {len(batch_segment_info)} != rows length {len(rows)}"
        )

        return batch_segment_info

    def _stack_speaker_embeddings(self, rows: list[DatasetRow]) -> torch.Tensor:
        """Stack speaker embeddings into a batch tensor.

        Returns:
            Tensor of shape [batch, 192].
        """
        assert len(rows) > 0, "_stack_speaker_embeddings: rows list is empty"

        embeddings = []
        for row_idx, row in enumerate(rows):
            emb = row.speaker_embedding
            assert isinstance(emb, np.ndarray), (
                f"row[{row_idx}].speaker_embedding: expected ndarray, got {type(emb)}"
            )
            assert emb.shape == (SPEAKER_EMBEDDING_DIM,), (
                f"row[{row_idx}].speaker_embedding: expected shape ({SPEAKER_EMBEDDING_DIM},), got {emb.shape}"
            )
            assert np.isfinite(emb).all(), (
                f"row[{row_idx}].speaker_embedding: contains NaN or Inf values"
            )
            embeddings.append(torch.from_numpy(emb).float())

        result = torch.stack(embeddings, dim=0)

        # Validate output shape
        batch_size = len(rows)
        assert result.shape == (batch_size, SPEAKER_EMBEDDING_DIM), (
            f"_stack_speaker_embeddings: output shape {result.shape} != expected ({batch_size}, {SPEAKER_EMBEDDING_DIM})"
        )

        return result


def create_collator_from_config(
    config: Any,
    processor: "Qwen3OmniMoeProcessor",
) -> FullDuplexCollator:
    """Create a FullDuplexCollator from a training config and processor.

    Args:
        config: Training config with audio_token_id, silence_token_id, etc.
        processor: Qwen3OmniMoeProcessor with feature_extractor attribute.

    Returns:
        Configured FullDuplexCollator instance.
    """
    return FullDuplexCollator(
        processor=processor,
        audio_token_id=config.audio_token_id,
        silence_token_id=config.silence_token_id,
        pad_token_id=config.pad_token_id,
        max_length=config.max_length,
        max_segments_per_batch=config.max_segments_per_batch,
    )

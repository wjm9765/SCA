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
        # Extract DatasetRow from each feature
        rows: list[DatasetRow] = []
        for feature in features:
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
        input_ids_list = []
        labels_list = []
        audio_positions_list = []

        for row in rows:
            input_ids = []
            labels = []
            audio_positions = []

            for i, token in enumerate(row.input_sequence):
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
        max_len = max(len(seq) for seq in input_ids_list)

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

        return (
            torch.tensor(padded_input_ids, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
            torch.tensor(padded_labels, dtype=torch.long),
        )

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
        all_features = []
        all_lengths = []

        for row in rows:
            if len(row.input_audios) == 0:
                # No audio - create minimal placeholder
                # Use a small silence chunk
                concat_audio = np.zeros(1600, dtype=np.float32)  # 100ms at 16kHz
            else:
                # Concatenate all input audio chunks
                concat_audio = np.concatenate(
                    [audio.waveform for audio in row.input_audios]
                )

            # Extract features without padding
            # Returns dict with "input_features" key
            features = self.feature_extractor(
                concat_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=False,
            )

            mel = features["input_features"]  # [1, mel_dim, mel_len]
            all_features.append(mel.squeeze(0))  # [mel_dim, mel_len]
            all_lengths.append(mel.shape[-1])

        # Pad to max length
        max_mel_len = max(all_lengths)
        mel_dim = all_features[0].shape[0]

        padded_features = torch.zeros(
            len(rows), mel_dim, max_mel_len, dtype=all_features[0].dtype
        )
        attention_mask = torch.zeros(len(rows), max_mel_len, dtype=torch.long)

        for i, (mel, length) in enumerate(zip(all_features, all_lengths)):
            padded_features[i, :, :length] = mel
            attention_mask[i, :length] = 1

        return padded_features, attention_mask

    def _extract_segment_info(self, rows: list[DatasetRow]) -> list[list[SegmentInfo]]:
        """Extract segment info for Talker training.

        Returns:
            List of segment info lists, one per sample in batch.
            Each segment contains text_token_idxs and target audio waveform.
        """
        batch_segment_info = []
        total_segments = 0

        for row in rows:
            sample_segments = []

            for seg in row.target_audios:
                # Check max segments limit
                if total_segments >= self.max_segments_per_batch:
                    break

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

        return batch_segment_info

    def _stack_speaker_embeddings(self, rows: list[DatasetRow]) -> torch.Tensor:
        """Stack speaker embeddings into a batch tensor.

        Returns:
            Tensor of shape [batch, 192].
        """
        embeddings = [torch.from_numpy(row.speaker_embedding).float() for row in rows]
        return torch.stack(embeddings, dim=0)


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

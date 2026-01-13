"""Tests for data_collator_duplex module."""

import numpy as np
import pytest
import torch

from sca_train.data_collator_duplex import FullDuplexCollator, SegmentInfo
from tests.mock_duplex_data import (
    PLACEHOLDER_SILENCE_TOKEN_ID,
    MockProcessor,
    make_batch,
    make_dataset_row,
    make_row_without_speaking,
)


# Test constants
AUDIO_TOKEN_ID = 151675
PAD_TOKEN_ID = 151643


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing."""
    return MockProcessor()


@pytest.fixture
def collator(mock_processor):
    """Create a collator for testing."""
    return FullDuplexCollator(
        processor=mock_processor,
        audio_token_id=AUDIO_TOKEN_ID,
        silence_token_id=PLACEHOLDER_SILENCE_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        max_length=1024,
        max_segments_per_batch=8,
    )


class TestFullDuplexCollator:
    """Tests for FullDuplexCollator."""

    def test_basic_collation(self, collator):
        """Test basic collation produces expected output keys."""
        batch = make_batch(batch_size=2, seed=42)
        output = collator(batch)

        assert "input_ids" in output
        assert "attention_mask" in output
        assert "labels" in output
        assert "input_features" in output
        assert "feature_attention_mask" in output
        assert "segment_info" in output
        assert "speaker_embeddings" in output

    def test_input_ids_shape(self, collator):
        """Test input_ids has correct shape."""
        batch = make_batch(batch_size=3, seed=42)
        output = collator(batch)

        assert output["input_ids"].ndim == 2
        assert output["input_ids"].shape[0] == 3

    def test_audio_token_replacement(self, collator):
        """Test -100 placeholders are replaced with audio_token_id."""
        row = make_dataset_row(num_audio_blocks=2, tokens_per_audio_block=4, seed=42)
        batch = [{"dataset_row": row}]
        output = collator(batch)

        # Check that audio_token_id appears in input_ids
        assert (output["input_ids"] == AUDIO_TOKEN_ID).any()

        # Check that -100 does NOT appear in input_ids
        assert not (output["input_ids"] == -100).any()

    def test_labels_keep_audio_masked(self, collator):
        """Test labels have -100 at audio positions."""
        row = make_dataset_row(num_audio_blocks=2, tokens_per_audio_block=4, seed=42)
        batch = [{"dataset_row": row}]
        output = collator(batch)

        # Find audio positions in input_ids
        audio_mask = output["input_ids"] == AUDIO_TOKEN_ID

        # Labels should be -100 at those positions
        assert (output["labels"][audio_mask] == -100).all()

    def test_attention_mask_shape(self, collator):
        """Test attention mask matches input_ids shape."""
        batch = make_batch(batch_size=2, seed=42)
        output = collator(batch)

        assert output["attention_mask"].shape == output["input_ids"].shape

    def test_input_features_shape(self, collator):
        """Test input_features has expected 3D shape."""
        batch = make_batch(batch_size=2, seed=42)
        output = collator(batch)

        # Should be [batch, mel_dim, mel_len]
        assert output["input_features"].ndim == 3
        assert output["input_features"].shape[0] == 2
        assert output["input_features"].shape[1] in (80, 128)

    def test_feature_attention_mask_shape(self, collator):
        """Test feature attention mask has correct shape."""
        batch = make_batch(batch_size=2, seed=42)
        output = collator(batch)

        # Should be [batch, mel_len]
        assert output["feature_attention_mask"].ndim == 2
        assert output["feature_attention_mask"].shape[0] == 2
        assert (
            output["feature_attention_mask"].shape[1]
            == output["input_features"].shape[2]
        )

    def test_speaker_embeddings_shape(self, collator):
        """Test speaker embeddings have correct shape."""
        batch = make_batch(batch_size=3, seed=42)
        output = collator(batch)

        assert output["speaker_embeddings"].shape == (3, 192)
        assert output["speaker_embeddings"].dtype == torch.float32

    def test_segment_info_structure(self, collator):
        """Test segment_info has correct structure."""
        batch = make_batch(batch_size=2, num_speaking_segments=2, seed=42)
        output = collator(batch)

        # Should be list of lists
        assert isinstance(output["segment_info"], list)
        assert len(output["segment_info"]) == 2

        # Each sample should have list of SegmentInfo
        for sample_segments in output["segment_info"]:
            assert isinstance(sample_segments, list)
            for seg in sample_segments:
                assert isinstance(seg, SegmentInfo)
                assert isinstance(seg.text_token_idxs, list)
                assert isinstance(seg.audio_waveform, np.ndarray)

    def test_no_speaking_segments(self, collator):
        """Test handling of rows with no speaking segments."""
        row = make_row_without_speaking()
        batch = [{"dataset_row": row}]
        output = collator(batch)

        # Should still work, just with empty segment list
        assert len(output["segment_info"]) == 1
        assert output["segment_info"][0] == []

    def test_max_segments_limit(self, mock_processor):
        """Test max_segments_per_batch is respected."""
        # Create collator with low limit
        limited_collator = FullDuplexCollator(
            processor=mock_processor,
            audio_token_id=AUDIO_TOKEN_ID,
            silence_token_id=PLACEHOLDER_SILENCE_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            max_length=1024,
            max_segments_per_batch=2,  # Very low limit
        )

        # Create batch with many segments
        batch = make_batch(batch_size=3, num_speaking_segments=3, seed=42)
        output = limited_collator(batch)

        # Count total segments
        total_segments = sum(len(segs) for segs in output["segment_info"])
        assert total_segments <= 2

    def test_padding_correctness(self, collator):
        """Test sequences are padded correctly."""
        # Create rows with different lengths
        row1 = make_dataset_row(num_audio_blocks=1, tokens_per_audio_block=2, seed=1)
        row2 = make_dataset_row(num_audio_blocks=3, tokens_per_audio_block=4, seed=2)

        batch = [{"dataset_row": row1}, {"dataset_row": row2}]
        output = collator(batch)

        # Both should have same length after padding
        assert output["input_ids"].shape[1] == output["attention_mask"].shape[1]

        # Longer sequence should have more 1s in attention mask
        sum1 = output["attention_mask"][0].sum().item()
        sum2 = output["attention_mask"][1].sum().item()
        # row2 has more audio blocks, so should be longer
        assert sum2 > sum1

    def test_truncation(self, mock_processor):
        """Test sequences are truncated to max_length."""
        # Create a collator with very short max_length
        short_collator = FullDuplexCollator(
            processor=mock_processor,
            audio_token_id=AUDIO_TOKEN_ID,
            silence_token_id=PLACEHOLDER_SILENCE_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            max_length=10,  # Very short
            max_segments_per_batch=8,
        )

        row = make_dataset_row(num_audio_blocks=5, tokens_per_audio_block=4, seed=42)
        batch = [{"dataset_row": row}]
        output = short_collator(batch)

        # Should be truncated to max_length
        assert output["input_ids"].shape[1] <= 10

    def test_invalid_row_type_raises(self, collator):
        """Test that invalid row type raises TypeError."""
        batch = [{"dataset_row": {"invalid": "data"}}]
        with pytest.raises(TypeError, match="Expected DatasetRow"):
            collator(batch)

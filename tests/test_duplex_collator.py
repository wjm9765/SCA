"""Tests for data_collator_duplex module using real dataset."""

import numpy as np
import pytest
import torch
from sca_data.dataset_utils import DatasetRow, easy_load

from sca_train.data_collator_duplex import (
    SILENCE_TOKEN_ID,
    FullDuplexCollator,
    SegmentInfo,
)


# Test constants
AUDIO_TOKEN_ID = 151675
PAD_TOKEN_ID = 151643


class MockFeatureExtractor:
    """Mock feature extractor for testing.

    Generates synthetic mel spectrogram tensors without loading any models.
    """

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def __call__(
        self,
        raw_speech: np.ndarray,
        sampling_rate: int = 16000,
        return_tensors: str = "pt",
        padding: bool = False,
        **kwargs,
    ) -> dict:
        """Convert raw audio to mock mel spectrogram features."""
        num_samples = len(raw_speech)
        time_frames = max(1, num_samples // self.hop_length)

        rng = np.random.default_rng(42)
        features = rng.standard_normal((1, self.feature_size, time_frames)).astype(
            np.float32
        )

        if return_tensors == "pt":
            return {"input_features": torch.from_numpy(features)}
        return {"input_features": features}


class MockProcessor:
    """Mock processor for testing FullDuplexCollator."""

    def __init__(self):
        self.feature_extractor = MockFeatureExtractor()


@pytest.fixture(scope="module")
def dataset():
    """Load real dataset once for all tests."""
    return easy_load(format="duplex")


@pytest.fixture
def sample_row(dataset) -> DatasetRow:
    """Get a single DatasetRow from the dataset."""
    return dataset[0]["dataset_row_obj"]


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
        silence_token_id=SILENCE_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        max_length=32768,
        max_segments_per_batch=8,
    )


def make_batch(dataset, indices: list[int]) -> list[dict]:
    """Create a batch from dataset indices."""
    return [dataset[i] for i in indices]


class TestFullDuplexCollator:
    """Tests for FullDuplexCollator with real data."""

    def test_basic_collation(self, collator, dataset):
        """Test basic collation produces expected output keys."""
        batch = make_batch(dataset, [0, 1])
        output = collator(batch)

        assert "input_ids" in output
        assert "attention_mask" in output
        assert "labels" in output
        assert "input_features" in output
        assert "feature_attention_mask" in output
        assert "segment_info" in output
        assert "speaker_embeddings" in output

    def test_input_ids_shape(self, collator, dataset):
        """Test input_ids has correct shape."""
        batch = make_batch(dataset, [0, 1, 2])
        output = collator(batch)

        assert output["input_ids"].ndim == 2
        assert output["input_ids"].shape[0] == 3

    def test_audio_token_replacement(self, collator, dataset):
        """Test -100 placeholders are replaced with audio_token_id."""
        batch = make_batch(dataset, [0])
        output = collator(batch)

        # Check that audio_token_id appears in input_ids
        assert (output["input_ids"] == AUDIO_TOKEN_ID).any()

        # Check that -100 does NOT appear in input_ids
        assert not (output["input_ids"] == -100).any()

    def test_labels_keep_audio_masked(self, collator, dataset):
        """Test labels have -100 at audio positions."""
        batch = make_batch(dataset, [0])
        output = collator(batch)

        # Find audio positions in input_ids
        audio_mask = output["input_ids"] == AUDIO_TOKEN_ID

        # Labels should be -100 at those positions
        assert (output["labels"][audio_mask] == -100).all()

    def test_attention_mask_shape(self, collator, dataset):
        """Test attention mask matches input_ids shape."""
        batch = make_batch(dataset, [0, 1])
        output = collator(batch)

        assert output["attention_mask"].shape == output["input_ids"].shape

    def test_input_features_shape(self, collator, dataset):
        """Test input_features has expected 3D shape."""
        batch = make_batch(dataset, [0, 1])
        output = collator(batch)

        # Should be [batch, mel_dim, mel_len]
        assert output["input_features"].ndim == 3
        assert output["input_features"].shape[0] == 2
        assert output["input_features"].shape[1] == 128  # MockFeatureExtractor size

    def test_feature_attention_mask_shape(self, collator, dataset):
        """Test feature attention mask has correct shape."""
        batch = make_batch(dataset, [0, 1])
        output = collator(batch)

        # Should be [batch, mel_len]
        assert output["feature_attention_mask"].ndim == 2
        assert output["feature_attention_mask"].shape[0] == 2
        assert (
            output["feature_attention_mask"].shape[1]
            == output["input_features"].shape[2]
        )

    def test_speaker_embeddings_shape(self, collator, dataset):
        """Test speaker embeddings have correct shape."""
        batch = make_batch(dataset, [0, 1, 2])
        output = collator(batch)

        assert output["speaker_embeddings"].shape == (3, 192)
        assert output["speaker_embeddings"].dtype == torch.float32

    def test_segment_info_structure(self, collator, dataset):
        """Test segment_info has correct structure."""
        batch = make_batch(dataset, [0, 1])
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

    def test_max_segments_limit(self, mock_processor, dataset):
        """Test max_segments_per_batch is respected."""
        # Create collator with low limit
        limited_collator = FullDuplexCollator(
            processor=mock_processor,
            audio_token_id=AUDIO_TOKEN_ID,
            silence_token_id=SILENCE_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            max_length=32768,
            max_segments_per_batch=2,  # Very low limit
        )

        batch = make_batch(dataset, [0, 1, 2])
        output = limited_collator(batch)

        # Count total segments
        total_segments = sum(len(segs) for segs in output["segment_info"])
        assert total_segments <= 2

    def test_padding_correctness(self, collator, dataset):
        """Test sequences are padded correctly."""
        batch = make_batch(dataset, [0, 1])
        output = collator(batch)

        # Both should have same length after padding
        assert output["input_ids"].shape[1] == output["attention_mask"].shape[1]
        assert output["input_ids"].shape[1] == output["labels"].shape[1]

    def test_truncation(self, mock_processor, dataset):
        """Test sequences are truncated to max_length."""
        # Create a collator with short max_length
        short_collator = FullDuplexCollator(
            processor=mock_processor,
            audio_token_id=AUDIO_TOKEN_ID,
            silence_token_id=SILENCE_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            max_length=1000,  # Shorter than real data
            max_segments_per_batch=8,
        )

        batch = make_batch(dataset, [0])
        output = short_collator(batch)

        # Should be truncated to max_length
        assert output["input_ids"].shape[1] <= 1000


class TestDatasetRowStructure:
    """Tests to verify real DatasetRow structure."""

    def test_dataset_row_has_required_fields(self, sample_row):
        """Test DatasetRow has all required fields."""
        assert hasattr(sample_row, "input_sequence")
        assert hasattr(sample_row, "target_audios")
        assert hasattr(sample_row, "input_audios")
        assert hasattr(sample_row, "speaker_embedding")

    def test_input_sequence_is_list_of_ints(self, sample_row):
        """Test input_sequence is a list of integers."""
        assert isinstance(sample_row.input_sequence, list)
        assert len(sample_row.input_sequence) > 0
        assert all(isinstance(t, int) for t in sample_row.input_sequence[:100])

    def test_target_audios_structure(self, sample_row):
        """Test target_audios have expected structure."""
        assert isinstance(sample_row.target_audios, list)
        if len(sample_row.target_audios) > 0:
            seg = sample_row.target_audios[0]
            assert hasattr(seg, "text_token_idxs")
            assert hasattr(seg, "audio")
            assert hasattr(seg.audio, "waveform")
            assert seg.audio.sampling_rate == 24000

    def test_input_audios_structure(self, sample_row):
        """Test input_audios have expected structure."""
        assert isinstance(sample_row.input_audios, list)
        if len(sample_row.input_audios) > 0:
            audio = sample_row.input_audios[0]
            assert hasattr(audio, "waveform")
            assert hasattr(audio, "sampling_rate")
            assert audio.sampling_rate == 16000

    def test_speaker_embedding_shape(self, sample_row):
        """Test speaker embedding has correct shape."""
        assert sample_row.speaker_embedding.shape == (192,)
        assert sample_row.speaker_embedding.dtype == np.float32

    def test_input_sequence_contains_audio_placeholders(self, sample_row):
        """Test input_sequence contains -100 audio placeholders."""
        num_placeholders = sum(1 for t in sample_row.input_sequence if t == -100)
        assert num_placeholders > 0, "Expected -100 audio placeholders in sequence"

    def test_input_audios_count_matches_placeholders(self, sample_row):
        """Test number of input audios matches -100 placeholder blocks."""
        # Count -100 tokens
        num_placeholders = sum(1 for t in sample_row.input_sequence if t == -100)
        num_audios = len(sample_row.input_audios)

        # Each input audio corresponds to one or more -100 tokens
        # The relationship should be: num_audios <= num_placeholders
        assert num_audios <= num_placeholders

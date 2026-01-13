"""Tests for data_types module."""

import numpy as np
import pytest

from sca_train.data_types import Audio, AudioSeg, DatasetRow
from tests.mock_duplex_data import (
    PLACEHOLDER_SILENCE_TOKEN_ID,
    make_audio,
    make_dataset_row,
    make_minimal_row,
    make_row_without_speaking,
    make_speaker_embedding,
)


class TestAudio:
    """Tests for Audio dataclass."""

    def test_valid_audio_16khz(self):
        """Test creating valid 16kHz audio."""
        waveform = np.zeros(16000, dtype=np.float32)
        audio = Audio(waveform=waveform, sample_rate=16000)
        assert audio.duration_seconds == 1.0
        assert audio.num_samples == 16000

    def test_valid_audio_24khz(self):
        """Test creating valid 24kHz audio."""
        waveform = np.zeros(24000, dtype=np.float32)
        audio = Audio(waveform=waveform, sample_rate=24000)
        assert audio.duration_seconds == 1.0
        assert audio.num_samples == 24000

    def test_invalid_sample_rate(self):
        """Test that invalid sample rates raise error."""
        waveform = np.zeros(44100, dtype=np.float32)
        with pytest.raises(ValueError, match="Unexpected sample_rate"):
            Audio(waveform=waveform, sample_rate=44100)

    def test_invalid_2d_waveform(self):
        """Test that 2D waveforms raise error."""
        waveform = np.zeros((2, 16000), dtype=np.float32)
        with pytest.raises(ValueError, match="must be 1D"):
            Audio(waveform=waveform, sample_rate=16000)


class TestAudioSeg:
    """Tests for AudioSeg dataclass."""

    def test_valid_audio_seg(self):
        """Test creating valid AudioSeg."""
        audio = make_audio(0.5, 24000)
        seg = AudioSeg(text_token_idxs=[10, 11, 12], audio=audio)
        assert seg.start_idx == 10
        assert seg.end_idx == 13
        assert seg.length == 3

    def test_empty_indices_raises(self):
        """Test that empty indices raise error."""
        audio = make_audio(0.5, 24000)
        with pytest.raises(ValueError, match="cannot be empty"):
            AudioSeg(text_token_idxs=[], audio=audio)

    def test_wrong_sample_rate_raises(self):
        """Test that 16kHz audio raises error (should be 24kHz)."""
        audio = make_audio(0.5, 16000)  # Wrong rate for target audio
        with pytest.raises(ValueError, match="must be 24kHz"):
            AudioSeg(text_token_idxs=[0, 1, 2], audio=audio)

    def test_non_contiguous_indices_raises(self):
        """Test that non-contiguous indices raise error."""
        audio = make_audio(0.5, 24000)
        with pytest.raises(ValueError, match="must be contiguous"):
            AudioSeg(text_token_idxs=[10, 11, 13], audio=audio)  # Gap at 12


class TestDatasetRow:
    """Tests for DatasetRow dataclass."""

    def test_minimal_row(self):
        """Test creating minimal valid row."""
        row = make_minimal_row()
        assert row.num_speaking_segments == 1
        assert row.total_audio_tokens >= 1

    def test_row_without_speaking(self):
        """Test row with no speaking segments."""
        row = make_row_without_speaking()
        assert row.num_speaking_segments == 0
        assert row.total_audio_tokens == 4

    def test_audio_block_counting(self):
        """Test audio block counting."""
        row = make_dataset_row(num_audio_blocks=3, tokens_per_audio_block=4)
        blocks = row.get_audio_block_indices()
        assert len(blocks) == 3
        for start, end in blocks:
            assert end - start == 4

    def test_validate_passes_for_valid_row(self):
        """Test validation passes for valid data."""
        row = make_dataset_row(seed=42)
        row.validate(silence_token_id=PLACEHOLDER_SILENCE_TOKEN_ID)

    def test_validate_mismatched_audio_count(self):
        """Test validation fails when audio count mismatches."""
        row = make_dataset_row(num_audio_blocks=2)
        # Remove one audio
        row.input_audios = row.input_audios[:1]
        with pytest.raises(ValueError, match="does not match"):
            row.validate(silence_token_id=PLACEHOLDER_SILENCE_TOKEN_ID)

    def test_validate_overlapping_segments(self):
        """Test validation fails for overlapping segments."""
        audio1 = make_audio(0.5, 24000)
        audio2 = make_audio(0.5, 24000)
        # Create overlapping segments
        seg1 = AudioSeg(text_token_idxs=[10, 11, 12], audio=audio1)
        seg2 = AudioSeg(
            text_token_idxs=[11, 12, 13], audio=audio2
        )  # Overlaps with seg1

        row = DatasetRow(
            input_sequence=[100] * 20,  # No -100 for simplicity
            target_audios=[seg1, seg2],
            input_audios=[],
            speaker_embedding=make_speaker_embedding(),
        )
        with pytest.raises(ValueError, match="overlap"):
            row.validate(silence_token_id=PLACEHOLDER_SILENCE_TOKEN_ID)

    def test_validate_segment_pointing_to_audio_placeholder(self):
        """Test validation fails when segment points to -100."""
        audio = make_audio(0.5, 24000)
        seg = AudioSeg(text_token_idxs=[3, 4, 5], audio=audio)

        row = DatasetRow(
            input_sequence=[100, 101, 102, -100, -100, -100, 200],
            target_audios=[seg],  # Points to -100 positions
            input_audios=[make_audio(0.32, 16000)],
            speaker_embedding=make_speaker_embedding(),
        )
        with pytest.raises(ValueError, match="audio placeholder"):
            row.validate(silence_token_id=PLACEHOLDER_SILENCE_TOKEN_ID)

    def test_invalid_speaker_embedding_shape(self):
        """Test that wrong speaker embedding shape raises error."""
        with pytest.raises(ValueError, match="speaker_embedding must be shape"):
            DatasetRow(
                input_sequence=[100, -100, 200],
                target_audios=[],
                input_audios=[make_audio(0.1, 16000)],
                speaker_embedding=np.zeros(128),  # Wrong size
            )


class TestMockDataGenerator:
    """Tests for mock data generator functions."""

    def test_make_audio_duration(self):
        """Test that make_audio creates correct duration."""
        audio = make_audio(0.5, 16000)
        assert abs(audio.duration_seconds - 0.5) < 0.001

    def test_make_speaker_embedding_normalized(self):
        """Test that speaker embeddings are normalized."""
        emb = make_speaker_embedding(seed=42)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.001

    def test_make_dataset_row_reproducible(self):
        """Test that same seed produces same data."""
        row1 = make_dataset_row(seed=42)
        row2 = make_dataset_row(seed=42)
        assert row1.input_sequence == row2.input_sequence
        np.testing.assert_array_equal(row1.speaker_embedding, row2.speaker_embedding)

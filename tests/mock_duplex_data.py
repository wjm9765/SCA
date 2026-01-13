"""Mock data generator for full duplex training tests.

This module provides utilities to generate synthetic DatasetRow instances
for testing the collator and model without requiring real audio data.
"""

from typing import Optional

import numpy as np
import torch

from sca_train.data_types import Audio, AudioSeg, DatasetRow


# TODO: Replace with actual silence token ID from global config once defined
PLACEHOLDER_SILENCE_TOKEN_ID = 151700


class MockFeatureExtractor:
    """Mock feature extractor for testing.

    Generates synthetic mel spectrogram tensors without loading any models.
    Matches the interface expected by FullDuplexCollator.
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
    ) -> dict[str, torch.Tensor]:
        """Convert raw audio to mock mel spectrogram features.

        Args:
            raw_speech: Audio waveform as 1D numpy array.
            sampling_rate: Expected sample rate (must be 16000).
            return_tensors: Output format ("pt" for PyTorch tensors).
            padding: Ignored in mock implementation.
            **kwargs: Additional arguments (ignored).

        Returns:
            Dict with "input_features" of shape [1, feature_size, time_frames].
        """
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}"
            )

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
    """Mock processor for testing FullDuplexCollator.

    Provides only the feature_extractor attribute needed by the collator.
    """

    def __init__(self):
        self.feature_extractor = MockFeatureExtractor()


def make_audio(
    duration_seconds: float,
    sample_rate: int = 16000,
    frequency_hz: float = 440.0,
) -> Audio:
    """Create a synthetic audio waveform (sine wave).

    Args:
        duration_seconds: Duration of the audio in seconds.
        sample_rate: Sample rate in Hz (16000 for input, 24000 for target).
        frequency_hz: Frequency of the sine wave.

    Returns:
        Audio object with a sine wave waveform.
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * frequency_hz * t).astype(np.float32)
    return Audio(waveform=waveform, sample_rate=sample_rate)


def make_speaker_embedding(seed: Optional[int] = None) -> np.ndarray:
    """Create a random speaker embedding.

    Args:
        seed: Optional random seed for reproducibility.

    Returns:
        192-dim numpy array (simulating ECAPA-TDNN output).
    """
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal(192).astype(np.float32)
    # Normalize to unit length (typical for speaker embeddings)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def make_dataset_row(
    num_audio_blocks: int = 2,
    tokens_per_audio_block: int = 4,
    num_speaking_segments: int = 1,
    tokens_per_speaking_segment: int = 5,
    audio_duration_seconds: float = 0.32,
    target_audio_duration_seconds: float = 0.5,
    silence_token_id: int = PLACEHOLDER_SILENCE_TOKEN_ID,
    seed: Optional[int] = None,
) -> DatasetRow:
    """Create a synthetic DatasetRow for testing.

    Creates an interleaved sequence with the pattern:
        [text tokens] [-100 × N] [silence] [-100 × N] [model speech tokens] ...

    Args:
        num_audio_blocks: Number of audio blocks (user speech chunks).
        tokens_per_audio_block: Number of -100 tokens per audio block.
        num_speaking_segments: Number of segments where model speaks.
        tokens_per_speaking_segment: Number of text tokens per speaking segment.
        audio_duration_seconds: Duration of each input audio chunk.
        target_audio_duration_seconds: Duration of target audio for each segment.
        silence_token_id: Token ID for silence tokens.
        seed: Random seed for reproducibility.

    Returns:
        A synthetic DatasetRow instance.

    Example sequence for num_audio_blocks=2, num_speaking_segments=1:
        [100, 101, 102]  # System prompt (fake token IDs)
        [-100, -100, -100, -100]  # Audio block 1
        [silence_token_id]  # Silence
        [-100, -100, -100, -100]  # Audio block 2
        [200, 201, 202, 203, 204]  # Model speaking segment
    """
    input_sequence: list[int] = []
    input_audios: list[Audio] = []
    target_audios: list[AudioSeg] = []

    # 1. System prompt (3 fake tokens in range 100-199)
    system_tokens = [100, 101, 102]
    input_sequence.extend(system_tokens)

    # 2. Interleave audio blocks with silence tokens
    for i in range(num_audio_blocks):
        # Audio block (-100 placeholders)
        audio_block = [-100] * tokens_per_audio_block
        input_sequence.extend(audio_block)

        # Create corresponding input audio
        input_audios.append(
            make_audio(
                duration_seconds=audio_duration_seconds,
                sample_rate=16000,
                frequency_hz=440.0 + i * 100,  # Vary frequency for each chunk
            )
        )

        # Silence token after audio (except potentially after last audio)
        if i < num_audio_blocks - 1:
            input_sequence.append(silence_token_id)

    # 3. Add speaking segments (model speech tokens)
    for seg_idx in range(num_speaking_segments):
        # Record start position for this segment
        start_idx = len(input_sequence)

        # Add text tokens for this segment (use range 200-299)
        segment_tokens = [
            200 + seg_idx * 10 + j for j in range(tokens_per_speaking_segment)
        ]
        input_sequence.extend(segment_tokens)

        # Record indices for this segment
        text_token_idxs = list(
            range(start_idx, start_idx + tokens_per_speaking_segment)
        )

        # Create target audio for this segment
        target_audio = make_audio(
            duration_seconds=target_audio_duration_seconds,
            sample_rate=24000,  # Mimi codec sample rate
            frequency_hz=880.0 + seg_idx * 100,
        )

        target_audios.append(
            AudioSeg(
                text_token_idxs=text_token_idxs,
                audio=target_audio,
            )
        )

        # Optionally add more audio after speaking (for interleaved pattern)
        # This creates a more realistic duplex scenario
        if seg_idx < num_speaking_segments - 1:
            # Add another audio block
            audio_block = [-100] * tokens_per_audio_block
            input_sequence.extend(audio_block)
            input_audios.append(
                make_audio(
                    duration_seconds=audio_duration_seconds,
                    sample_rate=16000,
                    frequency_hz=640.0 + seg_idx * 100,
                )
            )
            input_sequence.append(silence_token_id)

    # 4. Create speaker embedding
    speaker_embedding = make_speaker_embedding(seed=seed)

    return DatasetRow(
        input_sequence=input_sequence,
        target_audios=target_audios,
        input_audios=input_audios,
        speaker_embedding=speaker_embedding,
    )


def make_batch(
    batch_size: int = 2,
    seed: Optional[int] = None,
    **kwargs,
) -> list[dict[str, DatasetRow]]:
    """Create a batch of synthetic dataset rows.

    Args:
        batch_size: Number of rows in the batch.
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments passed to make_dataset_row.

    Returns:
        List of dicts with "dataset_row" key, mimicking HuggingFace dataset format.
    """
    rows = []
    for i in range(batch_size):
        row_seed = seed + i if seed is not None else None
        row = make_dataset_row(seed=row_seed, **kwargs)
        rows.append({"dataset_row": row})
    return rows


def make_minimal_row(
    silence_token_id: int = PLACEHOLDER_SILENCE_TOKEN_ID,
) -> DatasetRow:
    """Create a minimal valid DatasetRow for basic testing.

    This creates the simplest possible valid row:
    - 1 audio block with 1 token
    - 1 speaking segment with 1 token
    - Minimal audio durations
    """
    return make_dataset_row(
        num_audio_blocks=1,
        tokens_per_audio_block=1,
        num_speaking_segments=1,
        tokens_per_speaking_segment=1,
        audio_duration_seconds=0.1,
        target_audio_duration_seconds=0.1,
        silence_token_id=silence_token_id,
        seed=42,
    )


def make_row_without_speaking() -> DatasetRow:
    """Create a DatasetRow with no speaking segments.

    This tests the edge case where the model is only listening
    and not producing any speech output.
    """
    input_sequence = [100, 101, 102, -100, -100, -100, -100]
    input_audios = [make_audio(0.32, 16000)]
    speaker_embedding = make_speaker_embedding(seed=0)

    return DatasetRow(
        input_sequence=input_sequence,
        target_audios=[],  # No speaking segments
        input_audios=input_audios,
        speaker_embedding=speaker_embedding,
    )

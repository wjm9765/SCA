"""Data types for Full Duplex training.

This module defines the core data structures used for interleaved audio-text
training in the duplex setting.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Audio:
    """Audio waveform container.

    Attributes:
        waveform: 1D numpy array of audio samples. Shape: [num_samples].
        sample_rate: Sample rate in Hz. Expected values:
            - 16000 for input audio (user speech, Whisper/Qwen-Audio)
            - 24000 for target audio (model speech, Mimi codec)
    """

    waveform: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.waveform.ndim != 1:
            raise ValueError(
                f"Audio waveform must be 1D, got shape {self.waveform.shape}"
            )
        if self.sample_rate not in (16000, 24000):
            raise ValueError(
                f"Unexpected sample_rate {self.sample_rate}. Expected 16000 or 24000."
            )

    @property
    def duration_seconds(self) -> float:
        """Duration of the audio in seconds."""
        return len(self.waveform) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        return len(self.waveform)


@dataclass
class AudioSeg:
    """A segment where the model speaks during duplex conversation.

    Represents a contiguous span of text tokens where the model is producing
    speech output, along with the corresponding ground truth audio.

    Attributes:
        text_token_idxs: List of indices into the input_sequence where the model
            is speaking (i.e., generating text tokens that should produce audio).
            These indices refer to positions in DatasetRow.input_sequence.
        audio: Target audio at 24kHz that corresponds to this speaking segment.
    """

    text_token_idxs: list[int]
    audio: Audio

    def __post_init__(self) -> None:
        if not self.text_token_idxs:
            raise ValueError("text_token_idxs cannot be empty")
        if self.audio.sample_rate != 24000:
            raise ValueError(
                f"AudioSeg.audio must be 24kHz for Mimi codec, got {self.audio.sample_rate}"
            )
        # Verify indices are sorted and contiguous
        for i in range(1, len(self.text_token_idxs)):
            if self.text_token_idxs[i] != self.text_token_idxs[i - 1] + 1:
                raise ValueError(
                    f"text_token_idxs must be contiguous, but got gap between "
                    f"index {i - 1} ({self.text_token_idxs[i - 1]}) and "
                    f"index {i} ({self.text_token_idxs[i]})"
                )

    @property
    def start_idx(self) -> int:
        """Start index of this segment in the input_sequence."""
        return self.text_token_idxs[0]

    @property
    def end_idx(self) -> int:
        """End index (exclusive) of this segment in the input_sequence."""
        return self.text_token_idxs[-1] + 1

    @property
    def length(self) -> int:
        """Number of text tokens in this segment."""
        return len(self.text_token_idxs)


@dataclass
class DatasetRow:
    """A single training example for full duplex training.

    The input_sequence uses -100 as a placeholder for audio token positions.
    These will be replaced with audio_token_id during collation.

    Interleaved Sequence Format:
        <sys_start> text... <sys_end> [-100 × N] <sil> [-100 × N] <txt> <txt> [-100 × N] <txt> ...
        └────── System Prompt ──────┘ └─ Audio ──┘      └─ Audio ──┘    └─ Model ─┘ └─ Audio ──┘

    Attributes:
        input_sequence: Token IDs with -100 for audio placeholder positions.
            Regular tokens are text tokens (system prompt, silence tokens,
            model speech tokens). -100 positions will be replaced with
            audio_token_id and filled with audio embeddings.
        target_audios: List of AudioSeg objects representing segments where
            the model is speaking. Each segment has text token positions and
            corresponding ground truth audio.
        input_audios: List of Audio objects at 16kHz representing user speech.
            These are concatenated and processed through the audio tower.
            Order matches the order of -100 blocks in input_sequence.
        speaker_embedding: 192-dim speaker embedding from ECAPA-TDNN.
            Used for voice cloning in the Talker module.
    """

    input_sequence: list[int]
    target_audios: list[AudioSeg]
    input_audios: list[Audio]
    speaker_embedding: np.ndarray

    # Validation cache - computed once during validate()
    _audio_block_count: Optional[int] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Basic shape validation
        if self.speaker_embedding.ndim != 1 or self.speaker_embedding.shape[0] != 192:
            raise ValueError(
                f"speaker_embedding must be shape (192,), got {self.speaker_embedding.shape}"
            )

    def validate(self, silence_token_id: int) -> None:
        """Validate the dataset row for consistency.

        This performs comprehensive validation to catch data issues early.
        Should be called during collation with the global silence_token_id.

        Args:
            silence_token_id: The silence token ID from global config.
                Used to verify silence tokens in the sequence are correct.

        Raises:
            ValueError: If validation fails with a descriptive message.
        """
        # 1. Count audio blocks (contiguous -100 regions)
        audio_blocks = self._count_audio_blocks()
        self._audio_block_count = audio_blocks

        # 2. Verify input_audios count matches audio block count
        if len(self.input_audios) != audio_blocks:
            raise ValueError(
                f"Number of input_audios ({len(self.input_audios)}) does not match "
                f"number of audio blocks in input_sequence ({audio_blocks})"
            )

        # 3. Verify all input_audios are 16kHz
        for i, audio in enumerate(self.input_audios):
            if audio.sample_rate != 16000:
                raise ValueError(
                    f"input_audios[{i}] has sample_rate {audio.sample_rate}, expected 16000"
                )

        # 4. Verify target_audios segments are within bounds and non-overlapping
        if self.target_audios:
            seq_len = len(self.input_sequence)

            # Check bounds
            for i, seg in enumerate(self.target_audios):
                if seg.start_idx < 0 or seg.end_idx > seq_len:
                    raise ValueError(
                        f"target_audios[{i}] indices [{seg.start_idx}:{seg.end_idx}] "
                        f"out of bounds for sequence length {seq_len}"
                    )

            # Check non-overlapping and sorted
            sorted_segs = sorted(self.target_audios, key=lambda s: s.start_idx)
            for i in range(1, len(sorted_segs)):
                prev_end = sorted_segs[i - 1].end_idx
                curr_start = sorted_segs[i].start_idx
                if curr_start < prev_end:
                    raise ValueError(
                        f"target_audios segments overlap: segment ending at {prev_end} "
                        f"overlaps with segment starting at {curr_start}"
                    )

        # 5. Verify target_audios text_token_idxs point to non-audio positions
        #    (i.e., they should not be -100 placeholders)
        for i, seg in enumerate(self.target_audios):
            for idx in seg.text_token_idxs:
                if self.input_sequence[idx] == -100:
                    raise ValueError(
                        f"target_audios[{i}].text_token_idxs contains index {idx} "
                        f"which points to an audio placeholder (-100)"
                    )

        # 6. Verify silence tokens in sequence match the expected silence_token_id
        #    TODO: Once silence_token_id is defined, add assertion here
        #    For now, we just store it for future use
        _ = silence_token_id  # Placeholder for future validation

    def _count_audio_blocks(self) -> int:
        """Count the number of contiguous -100 blocks in input_sequence."""
        count = 0
        in_block = False
        for token in self.input_sequence:
            if token == -100:
                if not in_block:
                    count += 1
                    in_block = True
            else:
                in_block = False
        return count

    def get_audio_block_indices(self) -> list[tuple[int, int]]:
        """Get start and end indices of each -100 block.

        Returns:
            List of (start_idx, end_idx) tuples for each audio block.
            end_idx is exclusive.
        """
        blocks = []
        start = None
        for i, token in enumerate(self.input_sequence):
            if token == -100:
                if start is None:
                    start = i
            else:
                if start is not None:
                    blocks.append((start, i))
                    start = None
        # Handle trailing block
        if start is not None:
            blocks.append((start, len(self.input_sequence)))
        return blocks

    @property
    def total_audio_tokens(self) -> int:
        """Total number of audio token positions (-100) in input_sequence."""
        return sum(1 for t in self.input_sequence if t == -100)

    @property
    def num_speaking_segments(self) -> int:
        """Number of segments where the model is speaking."""
        return len(self.target_audios)

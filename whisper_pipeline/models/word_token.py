"""
WordToken — one word produced by faster-whisper with word-level timestamps.

This is the atomic output of Piece 1 (Transcription). Every word in the
recording has its own precise start/end pair, which is what allows Piece 3
(Alignment) to assign speaker IDs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class WordToken(BaseModel):
    """A single transcribed word with precise timestamps."""

    word: str  = Field(..., description="The transcribed word (may include leading/trailing spaces).")
    start: float = Field(..., ge=0.0, description="Word start time in seconds from the beginning of the audio.")
    end: float   = Field(..., ge=0.0, description="Word end time in seconds from the beginning of the audio.")
    is_segment_start: bool = Field(
        False,
        description=(
            "True for the first word of each Whisper segment. "
            "Used by the aligner to create readable paragraph breaks "
            "when speaker diarization is unavailable."
        ),
    )

    @model_validator(mode="after")
    def end_after_start(self) -> "WordToken":
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self

    @property
    def midpoint(self) -> float:
        """The centre of the word's time window — used by the aligner for speaker lookup."""
        return (self.start + self.end) / 2.0

    @property
    def clean_word(self) -> str:
        """Word with surrounding whitespace stripped."""
        return self.word.strip()

    def __repr__(self) -> str:
        return f"WordToken(word={self.word!r}, start={self.start:.2f}, end={self.end:.2f})"

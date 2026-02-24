"""
SpeakerSegment — one interval produced by Pyannote diarization.

This is the atomic output of Piece 2 (Diarization). Each segment says:
"from second X to second Y, speaker Z was speaking."

The timestamps reference the exact same timeline as the original audio file,
which is also what Whisper processed — this shared timeline is what makes
alignment in Piece 3 mathematically valid.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SpeakerSegment(BaseModel):
    """A time interval attributed to a single speaker."""

    start: float = Field(..., ge=0.0, description="Segment start time in seconds.")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds.")
    speaker_id: str = Field(..., description="Opaque speaker label (e.g. 'SPEAKER_00', 'SPEAKER_01').")

    @model_validator(mode="after")
    def end_after_start(self) -> "SpeakerSegment":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self

    def contains(self, timestamp: float) -> bool:
        """Return True if *timestamp* falls within this segment (inclusive boundaries)."""
        return self.start <= timestamp <= self.end

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"SpeakerSegment(speaker={self.speaker_id!r}, "
            f"start={self.start:.2f}, end={self.end:.2f})"
        )

"""
AlignedSegment — the output of Piece 3 (Timestamp Alignment).

Each segment is a sequence of consecutive words that were assigned the same
speaker by the aligner. It is the unified representation used throughout the
rest of the pipeline.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AlignedSegment(BaseModel):
    """A block of text attributed to a single speaker, with precise timestamps."""

    text: str = Field(..., description="The transcribed text for this segment.")
    speaker_id: str = Field(..., description="Speaker label from diarization (or 'UNKNOWN' if unresolved).")
    start: float = Field(..., ge=0.0, description="Segment start time in seconds.")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds.")

    @model_validator(mode="after")
    def end_after_start(self) -> "AlignedSegment":
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_timestamp(self) -> str:
        """Human-readable [MM:SS] timestamp string."""
        m, s = divmod(int(self.start), 60)
        return f"[{m:02d}:{s:02d}]"

    def to_transcript_line(self) -> str:
        """Single line representation suitable for the clean transcript."""
        return f"{self.format_timestamp()} [{self.speaker_id}]  {self.text}"

    def __repr__(self) -> str:
        return (
            f"AlignedSegment(speaker={self.speaker_id!r}, "
            f"start={self.start:.2f}, end={self.end:.2f}, "
            f"text={self.text[:40]!r})"
        )

"""
FinalOutput — the top-level result of the entire pipeline.

Produced by Piece 5 (Global Synthesis). Validated by Pydantic before being
passed to the OutputWriter. This is the only object written to disk.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class GlobalTopic(BaseModel):
    """A cohesive topic spanning one or more blocks."""

    title: str
    summary: str
    subtopics: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    definitions: dict[str, str] = Field(default_factory=dict)
    referenced_blocks: list[int] = Field(
        default_factory=list,
        description="Block indices where this topic appeared.",
    )


class SpeakerProfile(BaseModel):
    """Inferred speaker role and participation summary."""

    speaker_id: str
    inferred_role: str = Field(
        default="Unknown",
        description="E.g. 'Instructor', 'Student', 'Unknown'.",
    )
    speaking_time_seconds: float = Field(default=0.0, ge=0.0)


class FinalOutput(BaseModel):
    """Complete structured output of the pipeline — validated before writing to disk."""

    # Metadata
    audio_file: str = Field(..., description="Path to the original audio file that was processed.")
    processed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
        description="ISO-8601 UTC timestamp of when the pipeline completed.",
    )
    duration_seconds: float = Field(default=0.0, ge=0.0)
    language: str = Field(default="")

    # Speaker profiles
    speakers: list[SpeakerProfile] = Field(default_factory=list)

    # Structured content
    global_summary: str = Field(default="", description="2–5 sentence high-level summary of the entire lecture.")
    topics: list[GlobalTopic] = Field(default_factory=list)
    logistical_notices: list[str] = Field(
        default_factory=list,
        description="All logistical announcements aggregated across blocks.",
    )
    student_questions: list[dict] = Field(
        default_factory=list,
        description="All student questions aggregated across blocks.",
    )

    # Reference paths
    transcript_path: str = Field(default="", description="Path to the saved clean transcript file.")
    summary_path: str = Field(default="", description="Path to the saved summary file.")

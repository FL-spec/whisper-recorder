"""
BlockResult — output of Piece 4 (Map Phase / Block Extraction).

Each ~10-minute block of aligned transcript is processed by the LLM once.
The LLM corrects obvious transcription noise while simultaneously extracting
structured content — all in a single pass. This model captures that output.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Topic(BaseModel):
    """A topic or subtopic covered within the block."""

    title: str = Field(..., description="Brief topic title.")
    summary: str = Field(..., description="2–4 sentence explanation of what was covered.")
    key_points: list[str] = Field(default_factory=list, description="Bullet-point key takeaways.")
    definitions: dict[str, str] = Field(
        default_factory=dict,
        description="Technical terms defined in this block: {term: definition}.",
    )


class StudentQuestion(BaseModel):
    """A question asked by a student."""

    question: str = Field(..., description="The question as asked (lightly cleaned).")
    answer_summary: str = Field(
        default="",
        description="Summary of the instructor's answer, if one was given.",
    )
    timestamp_approx: float = Field(
        default=0.0,
        ge=0.0,
        description="Approximate timestamp (seconds) where the question occurred.",
    )


class BlockResult(BaseModel):
    """Structured extraction result for one time block of the lecture."""

    block_index: int = Field(..., ge=0, description="Zero-based block index.")
    start_time: float = Field(..., ge=0.0, description="Block start time in seconds.")
    end_time: float = Field(..., ge=0.0, description="Block end time in seconds.")

    topics: list[Topic] = Field(default_factory=list, description="Topics covered in this block.")
    student_questions: list[StudentQuestion] = Field(
        default_factory=list, description="Questions raised by students."
    )
    logistical_notices: list[str] = Field(
        default_factory=list,
        description="Announcements, deadlines, exam info, admin notices.",
    )

    raw_text: str = Field(
        default="",
        description="The block's transcript text as fed to the LLM (for reference/debugging).",
    )

    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time) / 60.0

    def has_content(self) -> bool:
        return bool(self.topics or self.student_questions or self.logistical_notices)

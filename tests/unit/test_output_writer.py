"""
Unit tests for OutputWriter.

Verifies that all three output formats (markdown, plaintext, json) produce
correct, non-empty files. Uses synthetic AlignedSegment and FinalOutput data.
No models are loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisper_pipeline.models.aligned_segment import AlignedSegment
from whisper_pipeline.models.final_output import FinalOutput, GlobalTopic, SpeakerProfile
from whisper_pipeline.output_writer import OutputWriter


@pytest.fixture
def sample_segments() -> list[AlignedSegment]:
    return [
        AlignedSegment(text="Good morning everyone.", speaker_id="SPEAKER_00", start=0.5, end=3.0),
        AlignedSegment(text="Today we cover thermodynamics.", speaker_id="SPEAKER_00", start=3.5, end=7.0),
        AlignedSegment(text="Can you repeat that?", speaker_id="SPEAKER_01", start=8.0, end=9.5),
        AlignedSegment(text="Of course. Thermodynamics studies heat.", speaker_id="SPEAKER_00", start=10.0, end=14.0),
    ]


@pytest.fixture
def sample_final_output() -> FinalOutput:
    return FinalOutput(
        audio_file="/tmp/lecture.wav",
        duration_seconds=900.0,
        language="pt",
        global_summary="This lecture covered the fundamentals of thermodynamics.",
        speakers=[
            SpeakerProfile(speaker_id="SPEAKER_00", inferred_role="Instructor", speaking_time_seconds=720.0),
            SpeakerProfile(speaker_id="SPEAKER_01", inferred_role="Student", speaking_time_seconds=180.0),
        ],
        topics=[
            GlobalTopic(
                title="Introduction to Thermodynamics",
                summary="The lecture introduced the basic concepts of thermodynamics.",
                key_points=["Heat is energy in transit", "Temperature measures average kinetic energy"],
                definitions={"entropy": "A measure of disorder in a system"},
                referenced_blocks=[0, 1],
            )
        ],
        logistical_notices=["Exam on Friday at 10am", "Assignment 2 due next Monday"],
        student_questions=[
            {"question": "What is entropy?", "answer_summary": "A measure of disorder.", "timestamp_approx": 450.0}
        ],
    )


@pytest.fixture
def minimal_config(tmp_path):
    from whisper_pipeline.config import PipelineConfig
    cfg = PipelineConfig.model_validate({
        "pipeline": {
            "output_dir": str(tmp_path),
            "output_format": "markdown",
        }
    })
    return cfg


class TestMarkdownOutput:
    def test_transcript_contains_speaker_labels(self, sample_segments, sample_final_output, minimal_config, tmp_path):
        writer = OutputWriter()
        result = writer.write(sample_final_output, sample_segments, "/tmp/lecture.wav", minimal_config)

        transcript = Path(result.transcript_path).read_text()
        assert "SPEAKER_00" in transcript
        assert "SPEAKER_01" in transcript
        assert "Good morning everyone." in transcript

    def test_summary_contains_key_sections(self, sample_segments, sample_final_output, minimal_config, tmp_path):
        writer = OutputWriter()
        result = writer.write(sample_final_output, sample_segments, "/tmp/lecture.wav", minimal_config)

        summary = Path(result.summary_path).read_text()
        assert "Introduction to Thermodynamics" in summary
        assert "fundamentals of thermodynamics" in summary
        assert "Exam on Friday" in summary
        assert "What is entropy?" in summary

    def test_output_paths_populated(self, sample_segments, sample_final_output, minimal_config):
        writer = OutputWriter()
        result = writer.write(sample_final_output, sample_segments, "/tmp/lecture.wav", minimal_config)
        assert result.transcript_path != ""
        assert result.summary_path != ""
        assert Path(result.transcript_path).exists()
        assert Path(result.summary_path).exists()


class TestPlaintextOutput:
    def test_plaintext_transcript(self, sample_segments, sample_final_output, minimal_config, tmp_path):
        minimal_config.pipeline.output_format = "plaintext"
        writer = OutputWriter()
        result = writer.write(sample_final_output, sample_segments, "/tmp/lecture.wav", minimal_config)

        transcript = Path(result.transcript_path).read_text()
        assert "SPEAKER_00" in transcript
        assert "SPEAKER_01" in transcript
        assert ".md" not in result.transcript_path
        assert result.transcript_path.endswith(".txt")

    def test_plaintext_summary(self, sample_segments, sample_final_output, minimal_config):
        minimal_config.pipeline.output_format = "plaintext"
        writer = OutputWriter()
        result = writer.write(sample_final_output, sample_segments, "/tmp/lecture.wav", minimal_config)

        summary = Path(result.summary_path).read_text()
        assert "Introduction to Thermodynamics" in summary
        assert "GLOBAL SUMMARY" in summary


class TestJSONOutput:
    def test_json_transcript_is_valid(self, sample_segments, sample_final_output, minimal_config):
        minimal_config.pipeline.output_format = "json"
        writer = OutputWriter()
        result = writer.write(sample_final_output, sample_segments, "/tmp/lecture.wav", minimal_config)

        transcript_data = json.loads(Path(result.transcript_path).read_text())
        assert isinstance(transcript_data, list)
        assert len(transcript_data) == len(sample_segments)
        assert transcript_data[0]["text"] == "Good morning everyone."
        assert transcript_data[0]["speaker_id"] == "SPEAKER_00"

    def test_json_summary_is_valid(self, sample_segments, sample_final_output, minimal_config):
        minimal_config.pipeline.output_format = "json"
        writer = OutputWriter()
        result = writer.write(sample_final_output, sample_segments, "/tmp/lecture.wav", minimal_config)

        summary_data = json.loads(Path(result.summary_path).read_text())
        assert "global_summary" in summary_data
        assert "topics" in summary_data
        assert len(summary_data["topics"]) == 1

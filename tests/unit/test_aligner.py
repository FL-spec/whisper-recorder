"""
Unit tests for TimestampAligner.

These tests use entirely synthetic data — no models are loaded,
no audio files are processed. All heavy dependencies are absent.

Test coverage:
- Basic alignment: word midpoints matched to speaker intervals
- Words before/after all intervals → UNKNOWN
- Empty words input → empty segments
- Empty speakers input → all UNKNOWN
- Consecutive same-speaker words are grouped
- Binary search correctness at segment boundaries
"""
from __future__ import annotations

import pytest

from whisper_pipeline.models.aligned_segment import AlignedSegment
from whisper_pipeline.models.speaker_segment import SpeakerSegment
from whisper_pipeline.models.word_token import WordToken
from whisper_pipeline.modules.aligner import TimestampAligner


def _make_config():
    """Return a minimal PipelineConfig-like object (avoid loading YAML in unit tests)."""
    from whisper_pipeline.config import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def aligner() -> TimestampAligner:
    return TimestampAligner()


@pytest.fixture
def config():
    return _make_config()


# ── Helpers ────────────────────────────────────────────────────────────────────

def word(text: str, start: float, end: float) -> WordToken:
    return WordToken(word=text, start=start, end=end)


def speaker(sid: str, start: float, end: float) -> SpeakerSegment:
    return SpeakerSegment(speaker_id=sid, start=start, end=end)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBasicAlignment:
    def test_single_word_single_speaker(self, aligner, config):
        words = [word("hello", 1.0, 1.5)]
        speakers = [speaker("SPK_00", 0.0, 5.0)]

        result = aligner.align(words, speakers, config)

        assert len(result) == 1
        assert result[0].speaker_id == "SPK_00"
        assert result[0].text == "hello"
        assert result[0].start == 1.0
        assert result[0].end == 1.5

    def test_two_speakers_no_overlap(self, aligner, config):
        words = [
            word("Good", 0.5, 1.0),
            word("morning", 1.0, 1.5),
            word("class", 1.5, 2.0),   # SPK_00
            word("Yes", 5.0, 5.5),     # SPK_01
            word("exactly", 5.5, 6.0), # SPK_01
        ]
        speakers = [
            speaker("SPK_00", 0.0, 4.0),
            speaker("SPK_01", 4.5, 10.0),
        ]

        result = aligner.align(words, speakers, config)

        assert len(result) == 2
        assert result[0].speaker_id == "SPK_00"
        assert result[0].text == "Good morning class"
        assert result[1].speaker_id == "SPK_01"
        assert result[1].text == "Yes exactly"

    def test_midpoint_determines_speaker(self, aligner, config):
        # Word spans two speaker intervals — midpoint decides
        # word midpoint = (3.8 + 4.2) / 2 = 4.0  →  falls in SPK_01 (4.0–10.0)
        words = [word("boundary", 3.8, 4.2)]
        speakers = [
            speaker("SPK_00", 0.0, 4.0),
            speaker("SPK_01", 4.0, 10.0),
        ]
        result = aligner.align(words, speakers, config)
        # midpoint=4.0, SPK_01 starts at 4.0 → contains(4.0) = True
        assert result[0].speaker_id == "SPK_01"

    def test_word_before_all_speakers(self, aligner, config):
        # Gap > 2.0s so it doesn't snap
        words = [word("pre", 0.1, 0.5)]
        speakers = [speaker("SPK_00", 3.0, 5.0)]
    
        result = aligner.align(words, speakers, config)
        assert result[0].speaker_id == "UNKNOWN"

    def test_word_after_all_speakers(self, aligner, config):
        words = [word("post", 10.0, 10.5)]
        speakers = [speaker("SPK_00", 0.0, 5.0)]

        result = aligner.align(words, speakers, config)
        assert result[0].speaker_id == "UNKNOWN"

    def test_word_in_gap_between_speakers(self, aligner, config):
        # Gap: 5.0 – 10.0 (> 4 sec gap); midpoint 7.5 is 2.5s from nearest speaker (> 2.0s tolerance)
        words = [word("gap", 7.0, 8.0)]
        speakers = [
            speaker("SPK_00", 0.0, 5.0),
            speaker("SPK_01", 10.0, 15.0),
        ]
        result = aligner.align(words, speakers, config)
        assert result[0].speaker_id == "UNKNOWN"

    def test_word_snaps_to_closest_speaker(self, aligner, config):
        # midpoint is 6.0; dist to SPK_00 is 1.0, dist to SPK_01 is 2.0. 
        # Both within 2.0 tolerance, but 1.0 is closer.
        words = [word("snap", 5.5, 6.5)]
        speakers = [
            speaker("SPK_00", 0.0, 5.0),
            speaker("SPK_01", 8.0, 10.0),
        ]
        result = aligner.align(words, speakers, config)
        assert result[0].speaker_id == "SPK_00"


class TestEdgeCases:
    def test_empty_words(self, aligner, config):
        result = aligner.align([], [speaker("SPK_00", 0.0, 10.0)], config)
        assert result == []

    def test_empty_speakers(self, aligner, config):
        words = [word("hello", 1.0, 2.0), word("world", 2.0, 3.0)]
        result = aligner.align(words, [], config)
        assert len(result) == 1
        assert result[0].speaker_id == "UNKNOWN"
        assert result[0].text == "hello world"

    def test_all_words_unknown(self, aligner, config):
        words = [word("a", 20.0, 20.5), word("b", 21.0, 21.5)]
        speakers = [speaker("SPK_00", 0.0, 5.0)]
        result = aligner.align(words, speakers, config)
        assert all(s.speaker_id == "UNKNOWN" for s in result)


class TestGrouping:
    def test_consecutive_same_speaker_grouped(self, aligner, config):
        words = [
            word("one", 0.0, 0.5),
            word("two", 0.5, 1.0),
            word("three", 1.0, 1.5),
        ]
        speakers = [speaker("SPK_00", 0.0, 10.0)]
        result = aligner.align(words, speakers, config)
        assert len(result) == 1
        assert result[0].text == "one two three"

    def test_speaker_change_splits_group(self, aligner, config):
        words = [
            word("I", 0.5, 1.0),
            word("agree", 1.0, 1.5),
            word("Thanks", 5.5, 6.0),
        ]
        speakers = [
            speaker("SPK_00", 0.0, 3.0),
            speaker("SPK_01", 5.0, 10.0),
        ]
        result = aligner.align(words, speakers, config)
        assert len(result) == 2
        assert result[0].speaker_id == "SPK_00"
        assert result[1].speaker_id == "SPK_01"

    def test_timestamps_of_groups(self, aligner, config):
        words = [
            word("first", 1.0, 1.5),
            word("second", 2.0, 2.5),
        ]
        speakers = [speaker("SPK_00", 0.0, 10.0)]
        result = aligner.align(words, speakers, config)
        assert result[0].start == 1.0
        assert result[0].end == 2.5

    def test_alternating_speakers(self, aligner, config):
        """Each word alternates speaker — should produce N separate segments."""
        words = [
            word("a", 0.5, 1.0),
            word("b", 2.5, 3.0),
            word("c", 4.5, 5.0),
        ]
        speakers = [
            speaker("SPK_00", 0.0, 2.0),
            speaker("SPK_01", 2.0, 4.0),
            speaker("SPK_00", 4.0, 6.0),
        ]
        result = aligner.align(words, speakers, config)
        assert len(result) == 3
        assert result[0].speaker_id == "SPK_00"
        assert result[1].speaker_id == "SPK_01"
        assert result[2].speaker_id == "SPK_00"

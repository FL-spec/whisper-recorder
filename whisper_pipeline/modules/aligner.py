"""
TimestampAligner — Piece 3 (Timestamp Alignment / Who Said What).

Pure Python — no ML models, no GPU. Fully testable in isolation.

Algorithm:
1. For every WordToken, compute its midpoint = (start + end) / 2.
2. Find which SpeakerSegment interval contains that midpoint.
3. Assign the speaker_id of that interval to the word.
4. Group consecutive words with the same speaker_id into AlignedSegments.
5. Words that fall outside all intervals get speaker_id = 'UNKNOWN'.

This module resolves the core v1 problem: without word-level timestamps,
it was impossible to know which speaker uttered which word. Now that
faster-whisper provides word_timestamps=True, this alignment is a simple
range-containment lookup.

Performance note: for a 90-minute lecture with ~15k words and ~2k speaker
intervals, a sorted binary-search approach is used (O(n log m)) instead of
a naive O(n*m) scan.
"""
from __future__ import annotations

import bisect
import logging

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.interfaces.aligner import AbstractAligner
from whisper_pipeline.models.aligned_segment import AlignedSegment
from whisper_pipeline.models.speaker_segment import SpeakerSegment
from whisper_pipeline.models.word_token import WordToken

logger = logging.getLogger(__name__)

_UNKNOWN_SPEAKER = "UNKNOWN"


class TimestampAligner(AbstractAligner):
    """Pure-Python timestamp aligner using binary search."""

    def align(
        self,
        words: list[WordToken],
        speakers: list[SpeakerSegment],
        config: PipelineConfig,
    ) -> list[AlignedSegment]:
        """
        Cross-reference word-level timestamps with speaker intervals.

        Args:
            words: Ordered word tokens from the transcriber.
            speakers: Ordered speaker intervals from the diarizer (may be empty).
            config: Validated pipeline configuration (not used computationally here).

        Returns:
            Chronologically ordered list of AlignedSegment objects.
        """
        if not words:
            logger.warning("No word tokens provided — returning empty aligned segments.")
            return []

        if not speakers:
            logger.info(
                "No speaker segments provided (diarization disabled or failed). "
                "All words will be labelled '%s'.",
                _UNKNOWN_SPEAKER,
            )
            return self._group_into_segments(
                [(w, _UNKNOWN_SPEAKER) for w in words]
            )

        # Build sorted index arrays for binary search:
        # speaker_starts[i] = start time of speakers[i]
        speaker_starts = [s.start for s in speakers]

        labelled: list[tuple[WordToken, str]] = []
        unknown_count = 0

        for word in words:
            midpoint = word.midpoint
            speaker_id = self._find_speaker(midpoint, speakers, speaker_starts)
            if speaker_id == _UNKNOWN_SPEAKER:
                unknown_count += 1
            labelled.append((word, speaker_id))

        if unknown_count:
            logger.info(
                "%d / %d words fell outside all speaker intervals → '%s'.",
                unknown_count,
                len(words),
                _UNKNOWN_SPEAKER,
            )

        segments = self._group_into_segments(labelled)
        logger.info(
            "Alignment complete: %d words → %d aligned segments.",
            len(words),
            len(segments),
        )
        return segments

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _find_speaker(
        midpoint: float,
        speakers: list[SpeakerSegment],
        speaker_starts: list[float],
    ) -> str:
        """
        Binary-search for the speaker segment that contains *midpoint*.

        Uses bisect_right to find the rightmost segment whose start is <= midpoint,
        then checks whether the midpoint also falls before that segment's end.

        Returns _UNKNOWN_SPEAKER if no segment contains the midpoint.
        """
        # Find the insertion point for midpoint in speaker_starts.
        # idx - 1 is the last segment that started at or before midpoint.
        idx = bisect.bisect_right(speaker_starts, midpoint) - 1

        if idx < 0:
            # midpoint is before every segment's start
            return _UNKNOWN_SPEAKER

        candidate = speakers[idx]
        if candidate.contains(midpoint):
            return candidate.speaker_id

        return _UNKNOWN_SPEAKER

    @staticmethod
    def _group_into_segments(
        labelled: list[tuple[WordToken, str]],
    ) -> list[AlignedSegment]:
        """
        Group consecutively same-speaker (word, speaker_id) pairs into
        AlignedSegment objects.

        Words within a group are concatenated with a single space. The group's
        start/end timestamps come from the first/last word respectively.
        """
        if not labelled:
            return []

        segments: list[AlignedSegment] = []
        current_words: list[WordToken] = []
        current_speaker: str = labelled[0][1]

        for word, speaker_id in labelled:
            if speaker_id == current_speaker:
                current_words.append(word)
            else:
                # Flush current group
                segments.append(
                    AlignedSegment(
                        text=" ".join(w.clean_word for w in current_words),
                        speaker_id=current_speaker,
                        start=current_words[0].start,
                        end=current_words[-1].end,
                    )
                )
                # Start new group
                current_words = [word]
                current_speaker = speaker_id

        # Flush the final group
        if current_words:
            segments.append(
                AlignedSegment(
                    text=" ".join(w.clean_word for w in current_words),
                    speaker_id=current_speaker,
                    start=current_words[0].start,
                    end=current_words[-1].end,
                )
            )

        return segments

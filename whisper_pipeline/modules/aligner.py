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
                "Splitting on Whisper segment boundaries — all speakers labelled '%s'.",
                _UNKNOWN_SPEAKER,
            )
            return self._split_by_whisper_segments(words)

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
    def _split_by_whisper_segments(words: list[WordToken]) -> list[AlignedSegment]:
        """
        Group words into AlignedSegments using Whisper's natural segment
        boundaries (is_segment_start=True) when diarization data is absent.

        This replaces the single giant UNKNOWN block with one segment per
        Whisper pause/sentence so the transcript is readable and the LLM
        receives sentence-level paragraphs rather than a wall of text.
        """
        if not words:
            return []

        segments: list[AlignedSegment] = []
        current: list[WordToken] = []

        for word in words:
            if word.is_segment_start and current:
                # Flush previous segment
                segments.append(AlignedSegment(
                    text=(" ".join(w.clean_word for w in current)).strip(),
                    speaker_id=_UNKNOWN_SPEAKER,
                    start=current[0].start,
                    end=current[-1].end,
                ))
                current = []
            current.append(word)

        # Flush final segment
        if current:
            segments.append(AlignedSegment(
                text=(" ".join(w.clean_word for w in current)).strip(),
                speaker_id=_UNKNOWN_SPEAKER,
                start=current[0].start,
                end=current[-1].end,
            ))

        logger.info(
            "No diarization: split %d words into %d Whisper segments.",
            len(words), len(segments),
        )
        return segments


    @staticmethod
    def _find_speaker(
        midpoint: float,
        speakers: list[SpeakerSegment],
        speaker_starts: list[float],
        tolerance: float = 2.0,
    ) -> str:
        """
        Binary-search for the speaker segment that contains *midpoint*.
        If the midpoint falls in an unlabelled gap, snap to the closest segment
        if it is within `tolerance` seconds.
        """
        if not speakers:
            return _UNKNOWN_SPEAKER

        idx = bisect.bisect_right(speaker_starts, midpoint) - 1

        # Check if it falls exactly inside the left candidate
        if idx >= 0 and speakers[idx].contains(midpoint):
            return speakers[idx].speaker_id

        # It's in a gap. Find the distance to the left and right segments.
        dist_left = float('inf')
        speaker_left = _UNKNOWN_SPEAKER
        if idx >= 0:
            # Distance from the end of the left segment to midpoint
            dist_left = midpoint - speakers[idx].end
            speaker_left = speakers[idx].speaker_id

        dist_right = float('inf')
        speaker_right = _UNKNOWN_SPEAKER
        if idx + 1 < len(speakers):
            # Distance from midpoint to the start of the right segment
            dist_right = speakers[idx + 1].start - midpoint
            speaker_right = speakers[idx + 1].speaker_id

        if min(dist_left, dist_right) <= tolerance:
            if dist_left <= dist_right:
                return speaker_left
            else:
                return speaker_right

        return _UNKNOWN_SPEAKER

    @staticmethod
    def _group_into_segments(
        labelled: list[tuple[WordToken, str]],
    ) -> list[AlignedSegment]:
        """
        Group consecutively same-speaker (word, speaker_id) pairs into
        AlignedSegment objects.

        We break a group and start a new one if:
        1. The speaker changes.
        2. Whisper marked this word as the start of a natural pause/segment
           (is_segment_start=True).

        This ensures long monologues remain broken into readable paragraphs.
        """
        if not labelled:
            return []

        segments: list[AlignedSegment] = []
        current_words: list[WordToken] = []
        current_speaker: str = labelled[0][1]

        for word, speaker_id in labelled:
            speaker_changed = (speaker_id != current_speaker)
            # Break if it's a new speaker OR it's a new Whisper segment
            if (speaker_changed or word.is_segment_start) and current_words:
                segments.append(
                    AlignedSegment(
                        text=" ".join(w.clean_word for w in current_words).strip(),
                        speaker_id=current_speaker,
                        start=current_words[0].start,
                        end=current_words[-1].end,
                    )
                )
                current_words = []
                current_speaker = speaker_id
                
            current_words.append(word)

        if current_words:
            segments.append(
                AlignedSegment(
                    text=" ".join(w.clean_word for w in current_words).strip(),
                    speaker_id=current_speaker,
                    start=current_words[0].start,
                    end=current_words[-1].end,
                )
            )

        return segments

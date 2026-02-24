"""
OutputWriter — writes the pipeline results to disk.

Supports three output formats (selected via config.pipeline.output_format):
  - markdown  (default) — human-readable .md with rich structure
  - plaintext           — clean .txt, no markup
  - json                — raw FinalOutput as pretty-printed JSON

Two files are always written per run:
  1. <output_dir>/<stem>_transcript.<ext>  — clean full transcript with
     speaker labels and timestamps
  2. <output_dir>/<stem>_summary.<ext>     — structured analysis/summary

The paths are stored back into the FinalOutput object so the orchestrator
can print them to the user.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.models.aligned_segment import AlignedSegment
from whisper_pipeline.models.final_output import FinalOutput

logger = logging.getLogger(__name__)


class OutputWriter:
    """Writes transcript and summary files in the configured format."""

    def write(
        self,
        final_output: FinalOutput,
        aligned_segments: list[AlignedSegment],
        audio_path: str,
        config: PipelineConfig,
    ) -> FinalOutput:
        """
        Write all output files and update final_output with their paths.

        Returns the updated FinalOutput (with transcript_path and summary_path set).
        """
        output_dir = Path(config.pipeline.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(audio_path).stem
        fmt = config.pipeline.output_format

        ext_map = {"markdown": "md", "plaintext": "txt", "json": "json"}
        ext = ext_map[fmt]

        transcript_path = output_dir / f"{stem}_transcript.{ext}"
        summary_path = output_dir / f"{stem}_summary.{ext}"

        # Write transcript
        if fmt == "markdown":
            transcript_content = self._transcript_markdown(aligned_segments, audio_path)
            summary_content = self._summary_markdown(final_output)
        elif fmt == "plaintext":
            transcript_content = self._transcript_plaintext(aligned_segments, audio_path)
            summary_content = self._summary_plaintext(final_output)
        else:  # json
            transcript_content = self._transcript_json(aligned_segments)
            summary_content = self._summary_json(final_output)

        transcript_path.write_text(transcript_content, encoding="utf-8")
        summary_path.write_text(summary_content, encoding="utf-8")

        logger.info("Transcript written to: %s", transcript_path)
        logger.info("Summary written to: %s", summary_path)

        final_output.transcript_path = str(transcript_path)
        final_output.summary_path = str(summary_path)
        return final_output

    # ── Markdown writers ───────────────────────────────────────────────────────

    @staticmethod
    def _transcript_markdown(segments: list[AlignedSegment], audio_path: str) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# Transcript — {Path(audio_path).name}",
            f"",
            f"> Generated: {now}",
            f"",
            "---",
            "",
        ]
        current_speaker = None
        for seg in segments:
            if seg.speaker_id != current_speaker:
                current_speaker = seg.speaker_id
                lines.append(f"\n**[{seg.speaker_id}]**\n")
            lines.append(f"{seg.format_timestamp()} {seg.text}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _summary_markdown(fo: FinalOutput) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        duration_min = int(fo.duration_seconds // 60)
        duration_sec = int(fo.duration_seconds % 60)

        lines = [
            f"# Lecture Summary — {Path(fo.audio_file).name}",
            f"",
            f"> Processed: {fo.processed_at}  |  Duration: {duration_min}m {duration_sec}s  |  Language: {fo.language.upper()}",
            "",
            "---",
            "",
            "## Global Summary",
            "",
            fo.global_summary or "_No summary generated._",
            "",
        ]

        if fo.speakers:
            lines += ["## Speakers", ""]
            for sp in fo.speakers:
                t = int(sp.speaking_time_seconds)
                lines.append(
                    f"- **{sp.speaker_id}** — {sp.inferred_role}"
                    + (f" ({t//60}m {t%60}s)" if t else "")
                )
            lines.append("")

        if fo.topics:
            lines += ["## Topics", ""]
            for i, topic in enumerate(fo.topics, 1):
                blocks_str = ", ".join(str(b) for b in topic.referenced_blocks) if topic.referenced_blocks else "—"
                lines += [
                    f"### {i}. {topic.title}",
                    f"*(Blocks: {blocks_str})*",
                    "",
                    topic.summary,
                    "",
                ]
                if topic.subtopics:
                    lines.append("**Subtopics:**")
                    for st in topic.subtopics:
                        lines.append(f"- {st}")
                    lines.append("")
                if topic.key_points:
                    lines.append("**Key Points:**")
                    for kp in topic.key_points:
                        lines.append(f"- {kp}")
                    lines.append("")
                if topic.definitions:
                    lines.append("**Definitions:**")
                    for term, defn in topic.definitions.items():
                        lines.append(f"- **{term}**: {defn}")
                    lines.append("")

        if fo.student_questions:
            lines += ["## Student Questions", ""]
            for q in fo.student_questions:
                ts = q.get("timestamp_approx", 0)
                m, s = divmod(int(ts), 60)
                lines += [
                    f"**Q [{m:02d}:{s:02d}]:** {q.get('question', '')}",
                    "",
                ]
                ans = q.get("answer_summary", "")
                if ans:
                    lines += [f"> **A:** {ans}", ""]

        if fo.logistical_notices:
            lines += ["## Logistical Notices", ""]
            for notice in fo.logistical_notices:
                lines.append(f"- {notice}")
            lines.append("")

        return "\n".join(lines)

    # ── Plaintext writers ──────────────────────────────────────────────────────

    @staticmethod
    def _transcript_plaintext(segments: list[AlignedSegment], audio_path: str) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"TRANSCRIPT — {Path(audio_path).name}",
            f"Generated: {now}",
            "=" * 60,
            "",
        ]
        current_speaker = None
        for seg in segments:
            if seg.speaker_id != current_speaker:
                current_speaker = seg.speaker_id
                lines += [f"\n[{seg.speaker_id}]"]
            lines.append(f"{seg.format_timestamp()} {seg.text}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _summary_plaintext(fo: FinalOutput) -> str:
        lines = [
            f"LECTURE SUMMARY — {Path(fo.audio_file).name}",
            f"Processed: {fo.processed_at}",
            "=" * 60,
            "",
            "GLOBAL SUMMARY",
            "-" * 40,
            fo.global_summary or "(No summary generated.)",
            "",
        ]
        if fo.topics:
            lines += ["TOPICS", "-" * 40, ""]
            for i, t in enumerate(fo.topics, 1):
                lines += [f"{i}. {t.title}", f"   {t.summary}", ""]
                for kp in t.key_points:
                    lines.append(f"   • {kp}")
                lines.append("")
        if fo.logistical_notices:
            lines += ["LOGISTICAL NOTICES", "-" * 40]
            for n in fo.logistical_notices:
                lines.append(f"• {n}")
        return "\n".join(lines) + "\n"

    # ── JSON writers ───────────────────────────────────────────────────────────

    @staticmethod
    def _transcript_json(segments: list[AlignedSegment]) -> str:
        data = [s.model_dump() for s in segments]
        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def _summary_json(fo: FinalOutput) -> str:
        return fo.model_dump_json(indent=2)

"""
OllamaSynthesizer — Piece 5 (Reduce Phase / Global Synthesis).

Context-window considerations for llama3.1:8b
───────────────────────────────────────────────
The naive approach — serialising all BlockResults as full JSON — explodes:
  18 blocks × ~800 tokens each = 14,400 tokens  →  blows 8K context entirely.

Solution: before sending to the LLM, each block is condensed to a compact
text summary: title + summary + key_points bullets only.
No raw_text, no full definitions dict, no embedded JSON nesting.

Compact summary per block ≈ 150–200 tokens.
18 blocks × 200 tokens = 3,600 tokens + 400 token prompt = ~4,000 tokens.
Output JSON ≈ 1,500 tokens.
Total ≈ 5,500 tokens  →  safely inside 8,192.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.interfaces.synthesizer import AbstractSynthesizer
from whisper_pipeline.models.block_result import BlockResult
from whisper_pipeline.models.final_output import FinalOutput, GlobalTopic, SpeakerProfile
from whisper_pipeline.modules.extractor import OllamaExtractor   # for _language_name

logger = logging.getLogger(__name__)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class OllamaSynthesizer(AbstractSynthesizer):
    """Concrete synthesizer using a local Ollama server."""

    def synthesize(
        self,
        blocks: list[BlockResult],
        audio_path: str,
        config: PipelineConfig,
    ) -> FinalOutput:
        """One final LLM call merges all block summaries into a FinalOutput."""
        import ollama

        prompt_template = self._load_prompt(config)

        # Build a compact, token-efficient representation of all blocks
        compact_summaries = self._build_compact_summaries(blocks)
        blocks_text = json.dumps(compact_summaries, ensure_ascii=False, indent=2)

        prompt = self._render_prompt(
            prompt_template,
            num_blocks=str(len(blocks)),
            blocks_json=blocks_text,
            language_name=OllamaExtractor._language_name(config.whisper.language),
        )

        logger.info(
            "Synthesizing %d blocks  (compact prompt: ~%d chars).",
            len(blocks), len(prompt),
        )

        ollama_options = {
            "temperature": config.llm.temperature,
            "num_ctx": config.llm.num_ctx,
            "num_predict": 2048,   # More headroom for the global summary
        }

        try:
            response = ollama.chat(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                options=ollama_options,
            )
            raw = response["message"]["content"].strip()
            raw = self._strip_fences(raw)
            raw = self._extract_json_object(raw)
            data = json.loads(raw)
            return self._build_final_output(data, audio_path, blocks, config)

        except Exception as exc:
            logger.error("Global synthesis failed: %s — using fallback aggregation.", exc)
            return self._fallback_aggregation(audio_path, blocks, config)

    @staticmethod
    def _render_prompt(template: str, **kwargs: str) -> str:
        """Replace <<key>> tokens — avoids str.format() KeyErrors on JSON braces."""
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"<<{key}>>", value)
        return result

    @staticmethod
    def _load_prompt(config: PipelineConfig) -> str:
        p = Path(config.pipeline.prompt_dir) / "synthesize_global.txt"
        if not p.exists():
            raise FileNotFoundError(
                f"Prompt template missing: {p.resolve()}\n"
                "Expected 'prompts/synthesize_global.txt'."
            )
        return p.read_text(encoding="utf-8")

    # ── Compact block summary builder ──────────────────────────────────────────

    @staticmethod
    def _build_compact_summaries(blocks: list[BlockResult]) -> list[dict]:
        """
        Convert each BlockResult into a short, token-efficient dict.

        Full BlockResult JSON → too large.
        This version keeps only what the LLM needs for synthesis:
        - block index + time range
        - topic titles + summaries + key points (no definitions, no raw_text)
        - student questions (question + answer only)
        - logistical notices
        """
        summary = []
        for b in blocks:
            entry: dict = {
                "block": b.block_index,
                "time": f"{_fmt(b.start_time)}–{_fmt(b.end_time)}",
            }
            if b.topics:
                entry["topics"] = [
                    {
                        "title": t.title,
                        "summary": t.summary,
                        "key_points": t.key_points[:5],  # cap at 5 bullets
                    }
                    for t in b.topics
                ]
            if b.student_questions:
                entry["questions"] = [
                    {"q": q.question, "a": q.answer_summary}
                    for q in b.student_questions
                ]
            if b.logistical_notices:
                entry["notices"] = b.logistical_notices
            summary.append(entry)
        return summary

    # ── Output builders ────────────────────────────────────────────────────────

    @staticmethod
    def _build_final_output(
        data: dict,
        audio_path: str,
        blocks: list[BlockResult],
        config: PipelineConfig,
    ) -> FinalOutput:
        topics = []
        for t in data.get("topics", []):
            topics.append(GlobalTopic(
                title=t.get("title", ""),
                summary=t.get("summary", ""),
                key_points=t.get("key_points", []),
                definitions=t.get("definitions", {}),
            ))

        speakers = []
        for s in data.get("speakers", []):
            speakers.append(SpeakerProfile(
                speaker_id=s.get("speaker_id", "UNKNOWN"),
                inferred_role=s.get("inferred_role", "Unknown"),
            ))

        duration = max((b.end_time for b in blocks), default=0.0)

        return FinalOutput(
            audio_file=audio_path,
            duration_seconds=duration,
            language=config.whisper.language,
            speakers=speakers,
            global_summary=data.get("global_summary", ""),
            topics=topics,
            logistical_notices=data.get("logistical_notices", []),
            student_questions=data.get("student_questions", []),
        )

    @staticmethod
    def _fallback_aggregation(
        audio_path: str,
        blocks: list[BlockResult],
        config: PipelineConfig,
    ) -> FinalOutput:
        """Simple aggregation when the LLM call fails — always completes."""
        all_notices: list[str] = []
        all_questions: list[dict] = []

        for b in blocks:
            all_notices.extend(b.logistical_notices)
            for q in b.student_questions:
                all_questions.append(q.model_dump())

        duration = max((b.end_time for b in blocks), default=0.0)

        return FinalOutput(
            audio_file=audio_path,
            duration_seconds=duration,
            language=config.whisper.language,
            global_summary="(Synthesis LLM call failed — see individual block data below.)",
            topics=[],
            logistical_notices=list(dict.fromkeys(all_notices)),
            student_questions=all_questions,
            speakers=[],
        )

    # ── JSON cleanup ───────────────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(text: str) -> str:
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("{"):
                    return stripped
                if stripped.startswith("json\n"):
                    return stripped[5:]
        return text

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return text
        return text[start : end + 1]

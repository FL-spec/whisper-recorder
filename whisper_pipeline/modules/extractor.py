"""
OllamaExtractor — Piece 4 (Map Phase / Block Extraction).

Processes the aligned transcript in ~5-minute blocks (configurable), sending
each block to the local Ollama instance with the extract_block.txt prompt.

Context-window considerations for llama3.1:8b
───────────────────────────────────────────────
- Ollama default num_ctx = 2048  →  WAY too small. We set it to 8192 explicitly.
- 5-minute block ≈ 750 words ≈ 1,000 tokens of transcript.
- Prompt template ≈ 300 tokens.
- Expected JSON output ≈ 400–600 tokens.
- Total per call: ≈ 2,000–2,500 tokens  →  safely inside 8 192.
- Hard cap of max_transcript_chars (default 6,000) prevents overflow on
  unusually dense audio (e.g. fast speakers, minimal silence).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.interfaces.extractor import AbstractExtractor
from whisper_pipeline.models.aligned_segment import AlignedSegment
from whisper_pipeline.models.block_result import BlockResult, StudentQuestion, Topic

logger = logging.getLogger(__name__)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class OllamaExtractor(AbstractExtractor):
    """Concrete extractor using a local Ollama server."""

    def extract_blocks(
        self,
        segments: list[AlignedSegment],
        config: PipelineConfig,
    ) -> list[BlockResult]:
        """Split segments into time blocks and run one LLM pass per block."""
        import ollama

        if not segments:
            logger.warning("No aligned segments provided — skipping extraction.")
            return []

        prompt_template = self._load_prompt(config)
        blocks = self._split_into_blocks(segments, config.pipeline.block_duration_minutes)
        glossary_section = self._build_glossary_section(config.glossary)

        results: list[BlockResult] = []
        for block_index, block_segs in enumerate(blocks):
            start_time = block_segs[0].start
            end_time   = block_segs[-1].end

            # Render transcript and hard-cap length BEFORE prompt assembly
            raw_transcript = self._render_transcript(block_segs)
            safe_transcript = self._safe_truncate(
                raw_transcript, config.pipeline.max_transcript_chars
            )

            logger.info(
                "Block %d/%d  [%s → %s]  %d segments  %d chars (cap: %d)",
                block_index + 1, len(blocks),
                _fmt(start_time), _fmt(end_time),
                len(block_segs), len(safe_transcript),
                config.pipeline.max_transcript_chars,
            )

            prompt = self._render_prompt(
                prompt_template,
                block_index=str(block_index),
                block_time=f"{_fmt(start_time)} \u2192 {_fmt(end_time)}",
                glossary_section=glossary_section,
                transcript_text=safe_transcript,
                language_name=self._language_name(config.whisper.language),
            )

            result = self._call_llm(
                ollama_client=ollama,
                prompt=prompt,
                block_index=block_index,
                start_time=start_time,
                end_time=end_time,
                raw_text=raw_transcript,
                config=config,
            )
            results.append(result)

        logger.info("Extraction complete: %d blocks.", len(results))
        return results

    # ── Prompt loading ─────────────────────────────────────────────────────────

    def _load_prompt(self, config: PipelineConfig) -> str:
        p = Path(config.pipeline.prompt_dir) / "extract_block.txt"
        if not p.exists():
            raise FileNotFoundError(
                f"Prompt template missing: {p.resolve()}\n"
                "Expected 'prompts/extract_block.txt'."
            )
        return p.read_text(encoding="utf-8")

    @staticmethod
    def _render_prompt(template: str, **kwargs: str) -> str:
        """
        Replace <<key>> tokens in the template with their values.

        We deliberately avoid str.format() because the prompt templates contain
        literal JSON examples with { } braces, which str.format() would try to
        interpret as format placeholders and raise a KeyError.
        """
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"<<{key}>>", value)
        return result

    @staticmethod
    def _language_name(iso_code: str) -> str:
        """Map ISO 639-1 code to a full language name for the LLM instruction."""
        _MAP = {
            "pt": "Portuguese",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "ar": "Arabic",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "sv": "Swedish",
        }
        return _MAP.get(iso_code.lower(), iso_code.upper())

    # ── Block splitting ────────────────────────────────────────────────────────

    @staticmethod
    def _split_into_blocks(
        segments: list[AlignedSegment],
        block_duration_minutes: int,
    ) -> list[list[AlignedSegment]]:
        """Group segments into time windows of block_duration_minutes."""
        if not segments:
            return []
        block_secs = block_duration_minutes * 60
        first_start = segments[0].start
        blocks: list[list[AlignedSegment]] = []
        current: list[AlignedSegment] = []
        boundary = first_start + block_secs

        for seg in segments:
            if seg.start >= boundary and current:
                blocks.append(current)
                current = []
                boundary = seg.start + block_secs
            current.append(seg)

        if current:
            blocks.append(current)
        return blocks

    # ── Transcript rendering ───────────────────────────────────────────────────

    @staticmethod
    def _render_transcript(segments: list[AlignedSegment]) -> str:
        return "\n".join(seg.to_transcript_line() for seg in segments)

    @staticmethod
    def _safe_truncate(text: str, max_chars: int) -> str:
        """
        Truncate transcript to max_chars, cutting at a newline boundary so
        we don't break a line in the middle and confuse the LLM.
        """
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.5:
            truncated = truncated[:last_newline]
        logger.warning(
            "Transcript truncated from %d to %d chars to stay inside context window.",
            len(text), len(truncated),
        )
        return truncated + "\n[... truncated — audio continues in next block ...]"

    @staticmethod
    def _build_glossary_section(glossary: list[str]) -> str:
        if not glossary:
            return "(No glossary — infer technical terms from context.)"
        items = "\n".join(f"- {t}" for t in glossary)
        return f"GLOSSARY (use these spellings when you recognise them in the audio):\n{items}"

    # ── LLM call with retry ────────────────────────────────────────────────────

    def _call_llm(
        self,
        *,
        ollama_client,
        prompt: str,
        block_index: int,
        start_time: float,
        end_time: float,
        raw_text: str,
        config: PipelineConfig,
    ) -> BlockResult:
        """
        Call Ollama once; retry once with a stripped-down prompt if JSON parsing fails.
        On second failure, return an empty BlockResult (raw_text preserved for debugging).
        """
        ollama_options = {
            "temperature": config.llm.temperature,
            "num_ctx": config.llm.num_ctx,   # CRITICAL: override Ollama's 2048 default
            "num_predict": 1024,              # Enough for the JSON output; prevents runaway
        }

        for attempt in (1, 2):
            try:
                response = ollama_client.chat(
                    model=config.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    options=ollama_options,
                )
                raw_json = response["message"]["content"].strip()

                # Strip markdown code fences if the model wraps output
                raw_json = self._strip_fences(raw_json)

                # Sometimes llama outputs extra commentary before/after the JSON.
                # Find the first '{' and last '}' to extract clean JSON.
                raw_json = self._extract_json_object(raw_json)

                data = json.loads(raw_json)

                topics    = [self._coerce_topic(t)    for t in data.get("topics", [])
                             if t]
                questions = [self._coerce_question(q) for q in data.get("student_questions", [])
                             if q]
                notices   = [str(n) for n in data.get("logistical_notices", []) if n]

                # Filter out None returns from coercions that couldn't salvage the item
                topics    = [t for t in topics    if t is not None]
                questions = [q for q in questions if q is not None]

                return BlockResult(
                    block_index=block_index,
                    start_time=start_time,
                    end_time=end_time,
                    topics=topics,
                    student_questions=questions,
                    logistical_notices=notices,
                    raw_text=raw_text,
                )

            except Exception as exc:
                if attempt == 1:
                    logger.warning(
                        "Block %d: parse failed (attempt 1): %s  → retrying with minimal prompt.",
                        block_index, exc,
                    )
                    # Fallback: even simpler prompt, tiny transcript sample
                    prompt = (
                        "Extract JSON from this lecture excerpt.\n"
                        "Respond with ONLY valid JSON, exactly this structure:\n"
                        '{"topics":[],"student_questions":[],"logistical_notices":[]}\n\n'
                        f"Excerpt:\n{raw_text[:1500]}"
                    )
                else:
                    logger.error(
                        "Block %d: failed after 2 attempts (%s). Storing raw text only.",
                        block_index, exc,
                    )
                    return BlockResult(
                        block_index=block_index,
                        start_time=start_time,
                        end_time=end_time,
                        raw_text=raw_text,
                    )

        return BlockResult(block_index=block_index, start_time=start_time, end_time=end_time)

    # ── LLM output coercion ────────────────────────────────────────────────────

    @staticmethod
    def _coerce_topic(raw) -> "Topic | None":
        """
        Normalise whatever llama returned for a topic item into a Topic.

        llama3.1:8b sometimes returns:
          - a proper dict: {"title": "...", "summary": "...", "key_points": [...]}
          - a plain string: "Primeira lei da termodinâmica"
          - a dict with extra/missing keys

        Returns None if the item is completely unsalvageable.
        """
        try:
            if isinstance(raw, str):
                return Topic(title=raw.strip(), summary="", key_points=[])
            if isinstance(raw, dict):
                allowed = {"title", "summary", "key_points", "definitions"}
                clean = {k: v for k, v in raw.items() if k in allowed}
                clean.setdefault("title", "Untitled")
                clean.setdefault("summary", "")
                kp = clean.get("key_points", [])
                if isinstance(kp, str):
                    clean["key_points"] = [kp]
                elif isinstance(kp, list):
                    clean["key_points"] = [str(k) for k in kp if k]
                else:
                    clean["key_points"] = []
                return Topic(**clean)
        except Exception as exc:
            logger.debug("Could not coerce topic item %r: %s", raw, exc)
        return None

    @staticmethod
    def _coerce_question(raw) -> "StudentQuestion | None":
        """
        Normalise whatever llama returned for a student_question item.
        Handles plain strings and dicts with missing/extra keys.
        """
        try:
            if isinstance(raw, str):
                return StudentQuestion(question=raw.strip(), answer_summary="")
            if isinstance(raw, dict):
                allowed = {"question", "answer_summary", "timestamp_approx"}
                clean = {k: v for k, v in raw.items() if k in allowed}
                clean.setdefault("question", "")
                clean.setdefault("answer_summary", "")
                if not clean["question"]:
                    return None
                return StudentQuestion(**clean)
        except Exception as exc:
            logger.debug("Could not coerce question item %r: %s", raw, exc)
        return None

    # ── JSON cleanup helpers ────────────────────────────────────────────────────


    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences that some models add despite instructions."""
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
        """Find the outermost JSON object in text, ignoring surrounding prose."""
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return text  # let json.loads raise a clear error
        return text[start : end + 1]

# Whisper Pipeline v2

A fully local pipeline that takes a lecture recording and returns a clean transcript with speaker labels and a structured summary — with no data leaving your machine.

```
python run.py
```

Press **Enter** to start recording. Press **Enter** again to stop. The full pipeline runs automatically.

---

## What it produces

For every recording, two files are written to `outputs/`:

| File | Contents |
|---|---|
| `recording_YYYY-MM-DD_transcript.md` | Full transcript with `[MM:SS] [SPEAKER_00]` labels per line |
| `recording_YYYY-MM-DD_summary.md` | Structured summary: global overview · topics · key points · student questions · logistical notices |

---

## How it works

Six steps run sequentially after recording stops. Each heavy model is fully released from memory before the next one loads.

```
[Your mic]
    ↓
Step 1 — faster-whisper (large-v3)
         Transcribes audio with word-level timestamps.
         Built-in silence filter (VAD) skips dead air automatically.
         Output: every word with its exact start/end time.
    ↓
[GPU/CPU memory released]
    ↓
Step 2 — Pyannote Audio 3.1
         Runs on the same original audio file.
         Produces a map of time intervals → speaker IDs.
         Output: "from 00:12 to 03:45 → SPEAKER_00"
    ↓
[GPU/CPU memory released]
    ↓
Step 3 — Timestamp Alignment (pure Python, no model)
         Each word's midpoint is matched to the nearest speaker interval.
         Consecutive same-speaker words are merged into segments.
         Output: text blocks labelled with speaker + timestamps.
    ↓
Step 4 — Ollama / llama3.1:8b  [Map phase]
         Transcript split into 5-minute blocks.
         One LLM call per block — extracts topics, key points,
         student questions, and logistical notices.
         Context window explicitly set to 8 192 tokens.
    ↓
Step 5 — Ollama / llama3.1:8b  [Reduce phase]
         Compact summaries of all blocks merged into one final structure.
         Infers speaker roles (Instructor / Student).
    ↓
Step 6 — Output written to outputs/
         Transcript + summary in your chosen format.
```

---

## Setup

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled
- A HuggingFace account with the Pyannote model licence accepted

### 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Configure your HuggingFace token

```bash
cp .env.example .env
# Edit .env and paste your HuggingFace token:
# HF_TOKEN=hf_...
```

### 3 — Accept the Pyannote model licence

Go to **[hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)** and click **"Agree and access repository"**.

The model (~400 MB) downloads automatically on first run.

### 4 — Pull the Ollama model

```bash
ollama pull llama3.1:8b
```

### 5 — Pre-download the Whisper model (optional but recommended)

```bash
python download_model.py
```

Downloads `large-v3` (~3 GB) into the HuggingFace cache. If you skip this, it downloads automatically on first run.

---

## Usage

### Record and process

```bash
python run.py
```

### Process an existing audio file (skip recording)

```bash
python run.py --file path/to/lecture.wav
```

### All options

```
python run.py --help

  --file PATH            Process an existing file instead of recording.
  --config PATH          Custom config file (default: config.yaml).
  --max-seconds N        Auto-stop recording after N seconds.
  --no-diarization       Skip speaker identification (labels all text UNKNOWN).
  --output-format        markdown | plaintext | json  (overrides config).
  --output-dir DIR       Override output directory.
  --save-intermediates   Save word tokens + speaker segments as JSON for debugging.
  --list-devices         Show available audio input devices.
  --verbose              Full debug logging.
```

---

## Configuration

All parameters are in `config.yaml`. Nothing is hardcoded.

```yaml
whisper:
  model: large-v3      # or large-v3-turbo, medium, small
  language: pt         # ISO 639-1 code: pt, en, es, fr …

diarization:
  enabled: true        # false = skip, all speakers labelled UNKNOWN
  device: cpu          # cpu | cuda

llm:
  model: llama3.1:8b   # any model available in: ollama list
  num_ctx: 8192        # context window — do not lower this

pipeline:
  block_duration_minutes: 5   # chunk size sent to LLM
  max_transcript_chars: 6000  # hard cap per block (overflow protection)
  output_format: markdown     # markdown | plaintext | json

glossary:              # optional — domain-specific terms for the LLM
  - ENEM
  - termodinâmica
```

### Tuning for your hardware

| Scenario | Recommendation |
|---|---|
| Mac with Apple Silicon | `device: cpu`, `compute_type: int8` — both Whisper and Pyannote run well |
| CUDA GPU (≥6 GB VRAM) | `whisper.device: cuda`, `compute_type: float16`, `diarization.device: cuda` |
| Slow machine | Use `model: medium` or `model: small` for Whisper |
| Slow LLM / long lectures | Lower `block_duration_minutes` to 3 |
| Technical domain | Fill in the `glossary` list with domain terms |

---

## Repository structure

```
run.py                          ← Single entrypoint (record + full pipeline)
pipeline.py                     ← Process an existing file (no recording)
config.yaml                     ← All parameters
download_model.py               ← Pre-download Whisper models
requirements.txt
.env.example

prompts/
  extract_block.txt             ← LLM prompt for block extraction (map phase)
  synthesize_global.txt         ← LLM prompt for global synthesis (reduce phase)

whisper_pipeline/
  config.py                     ← Pydantic config schema (validated on startup)
  memory.py                     ← Explicit GPU/CPU cleanup between stages
  orchestrator.py               ← Sequential pipeline coordinator
  output_writer.py              ← Writes transcript + summary to disk

  models/                       ← Internal data contracts (Pydantic)
    word_token.py
    speaker_segment.py
    aligned_segment.py
    block_result.py
    final_output.py

  interfaces/                   ← Abstract interfaces (pipeline only talks to these)
    transcriber.py
    diarizer.py
    aligner.py
    extractor.py
    synthesizer.py

  modules/                      ← Concrete implementations
    transcriber.py              ← faster-whisper with word_timestamps=True
    diarizer.py                 ← Pyannote 3.1
    aligner.py                  ← Pure Python binary-search alignment
    extractor.py                ← Ollama map-phase (one call per block)
    synthesizer.py              ← Ollama reduce-phase (one global call)

tests/
  unit/                         ← No models loaded — fast
    test_aligner.py
    test_config.py
    test_output_writer.py
  integration/                  ← Loads real models — slow
    test_transcriber.py
    test_diarizer.py
  fixtures/
    short_audio.wav
```

---

## Running tests

```bash
# Unit tests (fast, no models required):
pytest tests/unit/

# Integration tests (loads real models — requires HF token + Ollama):
pytest tests/integration/ -m integration
```

---

## Design principles

- **Nothing hardcoded** — every parameter lives in `config.yaml` or environment variables.
- **Each piece is independently replaceable** — swap `faster-whisper` for any other ASR model by changing only `modules/transcriber.py`.
- **Strictly sequential with explicit memory cleanup** — heavy models are explicitly released between stages. This prevents OOM crashes on consumer hardware.
- **All communication in structured JSON** — every module receives and returns typed Pydantic models.
- **Graceful degradation** — if diarization fails or the LLM returns malformed JSON, the pipeline continues and labels speakers `UNKNOWN` rather than crashing.
- **Context-window aware** — every Ollama call sets `num_ctx: 8192` explicitly (overriding Ollama's 2048 default). Transcript blocks are capped at `max_transcript_chars` characters before being sent to the LLM.

---

## Troubleshooting

**`ollama: connection refused`** — Start Ollama with `ollama serve`.

**`pyannote model not found`** — Accept the licence at [hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

**LLM returns empty topics** — The transcript block may be too short or contain only silence. This is expected for non-speech audio.

**Diarization runs but all speakers are UNKNOWN** — Check that `HF_TOKEN` is set in `.env` and the licence is accepted.

**Recording produces no audio** — Run `python run.py --list-devices` to see available input devices.

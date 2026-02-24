# 🎙 Whisper Recorder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

A 100% local, blazing-fast voice recorder and transcriber powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Designed specifically for **MacBook Pro with Apple Silicon (M1–M4)**, but runs anywhere that supports Python and `faster-whisper`. 

By default, it uses the highly intelligent `large-v3` model and is optimized for Portuguese (PT) dictation, but can transcribe any language supported by Whisper.

---

## ✨ Features

- **100% Local & Private:** No audio leaves your machine. Your data is yours.
- **Highly Accurate Models:** Pre-configured with Hugging Face's `faster-whisper-large-v3` for peak intelligence and precision.
- **Optimized Downloading:** Utilizes `hf_transfer` to saturate your network connection for model downloading.
- **Concurrency Bulletproof:** Fixes macOS Python thread segmentation faults by managing `huggingface_hub` concurrent workers correctly.
- **Live Incremental Transcripts:** Captures live audio and transcribes it incrementally in a beautifully formatted CLI using `Rich`.

---

## ⚡ Installation (One-Time Setup)

**1. Clone the repository**
```bash
git clone https://github.com/FL-spec/whisper-recorder.git
cd whisper-recorder
```

**2. Create and activate a Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up an Environment File (Optional, but highly recommended for fast downloads)**
Copy the provided `.env.example` file:
```bash
cp .env.example .env
```
Then, insert your Hugging Face token in `.env` if you have one. This guarantees maximum download speed using `hf_transfer`.

---

## 🚀 Usage

Execute `record.py` inside your active virtual environment.

```bash
# Start recording → Press Enter to stop → Output saved to 'transcricao.txt'
python record.py

# Stop automatically after 60 seconds
python record.py --max-seconds 60

# Save to a custom output file
python record.py --output my_meeting_notes.txt

# Disable timestamp prefixes in output
python record.py --no-timestamps

# Pre-download the model explicitly (Optional, record.py will also download it organically)
python download_model.py
```

> **Note:** On your very first run, the local model (e.g., `large-v3` ~3 GB) will be quickly downloaded from the Hugging Face Hub. All subsequent executions load instantly from your local cache!

---

## 📂 Output Format

Every recording is **appended** safely to your target `.txt` file, ensuring no data override:

```text
# Gravação — 2026-02-24 14:32:10

[00:02] Hello, this is a local recording test.
[00:08] The system transcribed this without the internet.

────────────────────────────────────────────────────────────
```

---

## 🔧 Deep-Dive Configuration

You can configure the properties using **Command Line Arguments** or by setting **Environment Variables** (e.g., in your `.env` file). CLI arguments always take precedence.

| Environment Variable | CLI Argument | Default | Description |
|---|---|---|---|
| `WHISPER_MODEL` | `--model` / `-m` | `large-v3` | Faster-Whisper model ID or target Hugging Face model |
| `WHISPER_LANGUAGE` | `--language` / `-l` | `pt` | Target language for dictation (forces Whisper to skip auto-detection) |
| `WHISPER_DEVICE` | `--device` / `-d` | `cpu` | Processing device. Options: `cpu`, `cuda`, `auto` |
| `WHISPER_COMPUTE_TYPE` | `--compute-type` / `-c`| `int8` | Compute quantization. Options: `int8`, `float16`, `float32` |
| `WHISPER_BEAM_SIZE` | `--beam-size` / `-b`  | `5` | Beam size for the decoder. Higher = more accurate but slower |
| - | `--output` / `-o` | `transcricao.txt` | Target text file |
| - | `--max-seconds` / `-s`| ∞ | Max duration of the recording before graceful stop |
| - | `--no-timestamps` | `False` | Omit `[mm:ss]` styling from the text file output |
| - | `--list-devices` | `False` | Display a list of available host audio devices |
| `WHISPER_MODELS_TO_DOWNLOAD`| `download_model.py [models]`| `large-v3,large-v3-turbo` | Comma-separated list of models to download by default when running `download_model.py` |

---

## 🛠 System Requirements

- macOS 12+ with Apple Silicon (M1/M2/M3/M4) is highly recommended for `int8`/`float16` optimizations.
- **Python 3.10+**
- Standard Microphone

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

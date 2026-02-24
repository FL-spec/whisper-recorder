#!/usr/bin/env python3
"""
Download the faster-whisper model with full verbose logging.
Run this ONCE before using record.py.

    python download_model.py
"""

import os
import sys
import time
import logging

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ── 1. Load .env token ────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    print("❌  HF_TOKEN não encontrado no .env — abortando.")
    sys.exit(1)

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# ── 2. Enable full HF Hub + urllib3 debug logging ────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  [%(levelname)s]  %(name)s  —  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Keep third-party noise manageable
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("filelock").setLevel(logging.INFO)
logging.getLogger("tqdm").setLevel(logging.WARNING)

log = logging.getLogger("download_model")

# ── 3. Models to download ──────────────────────────────────────────────────────
MODELS_TO_DOWNLOAD = [
    {"alias": "large-v3-turbo", "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"},
]
CACHE_DIR   = os.path.expanduser("~/.cache/huggingface/hub")

# ── 4. Pre-flight checks ──────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🔍  Pre-flight checks")
print("─" * 60)
print(f"  Token  : {HF_TOKEN[:12]}…")
print(f"  Models : {[m['alias'] for m in MODELS_TO_DOWNLOAD]}")
print(f"  Cache  : {CACHE_DIR}")
print("─" * 60 + "\n")

# Network check
import urllib.request
print("⏳  Testando conectividade com huggingface.co …", flush=True)
try:
    urllib.request.urlopen("https://huggingface.co", timeout=10)
    print("✅  Rede OK\n")
except Exception as e:
    print(f"❌  Não foi possível conectar ao HuggingFace: {e}")
    sys.exit(1)

# ── 5. Login ──────────────────────────────────────────────────────────────────
import huggingface_hub
print(f"🔑  Autenticando com HuggingFace Hub (versão {huggingface_hub.__version__}) …", flush=True)
huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
whoami = huggingface_hub.whoami()
print(f"✅  Logged in as: {whoami.get('name', whoami.get('fullname', '?'))}\n")

# ── 6/7/8. Loop and Download/Test ─────────────────────────────────────────────
for m in MODELS_TO_DOWNLOAD:
    MODEL_ALIAS = m["alias"]
    MODEL_NAME = m["repo"]
    
    print("\n" + "=" * 60)
    print(f"  ⬇️   Processando modelo: {MODEL_NAME}")
    print("=" * 60)
    
    try:
        cached = huggingface_hub.scan_cache_dir()
        for repo in cached.repos:
            if MODEL_NAME.lower() in repo.repo_id.lower():
                size_mb = repo.size_on_disk / 1_000_000
                print(f"ℹ️   Modelo já em cache: {repo.repo_id}  ({size_mb:.0f} MB)")
    except Exception as e:
        log.debug(f"Cache scan skipped: {e}")

    t0 = time.time()
    try:
        local_path = huggingface_hub.snapshot_download(
            repo_id=MODEL_NAME,
            token=HF_TOKEN,
            repo_type="model",
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
            max_workers=1,
        )
        elapsed = time.time() - t0
        print(f"\n✅  Download de {MODEL_ALIAS} concluído em {elapsed:.0f}s")
        print(f"📂  Salvo em: {local_path}\n")
    except Exception as e:
        print(f"\n❌  Erro no download: {e}")
        log.exception(f"Download falhou para {MODEL_NAME}")
        continue

    print(f"🧪  Testando carregamento do modelo {MODEL_ALIAS} …", flush=True)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(local_path, device="cpu", compute_type="int8")
        print(f"✅  Modelo {MODEL_ALIAS} carregado com sucesso!\n")
    except Exception as e:
        print(f"❌  Falha ao carregar modelo {MODEL_ALIAS}: {e}")
        log.exception(f"Carregamento falhou para {MODEL_NAME}")

print("\n" + "─" * 60)
print(f"  Tudo pronto. Execute:  python record.py --model large-v3 ou --model large-v3-turbo")
print("─" * 60 + "\n")

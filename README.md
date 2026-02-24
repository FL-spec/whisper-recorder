# 🎙 Whisper Recorder — Português

Gravador de voz 100% local com transcrição em Português via **faster-whisper** (`large-v3-turbo`).  
Desenvolvido para **MacBook Pro com Apple Silicon (M1–M4)**.

---

## ⚡ Instalação (uma vez só)

```bash
# 1. Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instale as dependências
pip install -r requirements.txt
```

> **Nota:** Na primeira execução o modelo `large-v3-turbo` (~800 MB) será baixado automaticamente do Hugging Face e cacheado localmente. As execuções seguintes são instantâneas.

---

## 🚀 Como usar

```bash
# Ative o ambiente (se ainda não ativado)
source .venv/bin/activate

# Gravar → pressione Enter para parar → transcrição salva em transcricao.txt
python record.py

# Parar automaticamente após 60 segundos
python record.py --max-seconds 60

# Salvar em arquivo personalizado
python record.py --output reuniao_2024.txt

# Sem marcações de tempo no arquivo de saída
python record.py --no-timestamps

# Usar o modelo maior (mais preciso, mais lento)
python record.py --model large-v3

# Ver dispositivos de áudio disponíveis
python record.py --list-devices
```

---

## 📂 Arquivos de saída

Cada gravação é **acrescentada** ao arquivo `.txt` (não substitui), com cabeçalho de data/hora:

```
# Gravação — 2025-01-15 14:32:10

[00:02] Olá, este é um teste de gravação.
[00:08] O sistema está funcionando corretamente.

────────────────────────────────────────────────────────────

# Gravação — 2025-01-15 15:10:45
...
```

---

## 🔧 Configuração rápida

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--model` | `large-v3-turbo` | Modelo a usar |
| `--output` | `transcricao.txt` | Arquivo de saída |
| `--max-seconds` | ∞ | Duração máxima |
| `--no-timestamps` | off | Remover marcações de tempo |

### Modelos disponíveis (do mais rápido ao mais preciso)

| Modelo | Tamanho | Velocidade | Precisão |
|---|---|---|---|
| `tiny` | ~75 MB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ |
| `base` | ~145 MB | ⚡⚡⚡⚡ | ★★★☆☆ |
| `small` | ~244 MB | ⚡⚡⚡ | ★★★★☆ |
| `medium` | ~769 MB | ⚡⚡ | ★★★★☆ |
| `large-v3-turbo` | ~809 MB | ⚡⚡⚡ | ★★★★★ ← **recomendado** |
| `large-v3` | ~1.5 GB | ⚡ | ★★★★★ |

---

## 🛠 Requisitos

- macOS 12+ com Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Microfone

---

## 💡 Dica — Criar atalho de terminal

Adicione ao seu `~/.zshrc`:

```bash
alias gravar='cd /Users/franciscolivraghi/Desktop/whisper-recordeer && source .venv/bin/activate && python record.py'
```

Depois, simplesmente execute:
```bash
gravar
```

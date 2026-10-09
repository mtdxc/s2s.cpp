#!/bin/bash
# Download the GGUF models used at runtime
#
# Usage: ./models.sh [options]
#   default:   Q8_0 everywhere
#   --asr X:   Qwen3-ASR size, 0.6B or 1.7B (default).
#   --quant X: ASR quant (Q4_K_M, Q8_0, BF16)
#   --tts X:   TTS quant (Q4_K_M, Q8_0, BF16). Q4_K_M breaks the talker end of
#              speech, so the default stays Q8_0.
#
# The talker is the 1.7b Base one: the voices in voices/ are references
# extracted for its hidden size. The server loads the best quant up to Q8_0.
# The VAD, the turn detector and the echo canceller are small and always ship
# as F32.

set -eu

DIR="models"
VAD_REPO="Serveurperso/Silero-VAD-GGUF"
TURN_REPO="Serveurperso/Smart-Turn-v3-GGUF"
TTS_REPO="Serveurperso/Qwen3-TTS-GGUF"
AEC_REPO="Serveurperso/LocalVQE-GGUF"
ASR_REPO="Serveurperso/Qwen3-ASR-GGUF"
ASR="1.7B"
QUANT="Q8_0"
TTS_QUANT="Q8_0"

while [ $# -gt 0 ]; do
    case "$1" in
        --asr)    ASR="$2"; shift ;;
        --quant)  QUANT="$2"; shift ;;
        --tts)    TTS_QUANT="$2"; shift ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

case "$ASR" in
    0.6B|1.7B) ASR_FILE="qwenasr-${ASR}" ;;
    *)     echo "Unknown ASR: $ASR"; exit 1 ;;
esac

mkdir -p "$DIR"

dl() {
    local repo="$1" file="$2"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file <- $repo"
    hf download --quiet "$repo" "$file" --local-dir "$DIR"
}

dl "$VAD_REPO"  "silero-vad-F32.gguf"
dl "$TURN_REPO" "smart-turn-v3.2-F32.gguf"
dl "$ASR_REPO"  "${ASR_FILE}-${QUANT}.gguf"
dl "$TTS_REPO"  "qwen-talker-1.7b-base-${TTS_QUANT}.gguf"
dl "$TTS_REPO"  "qwen-tokenizer-12hz-Q8_0.gguf"
dl "$AEC_REPO"  "localvqe-v1.3-F32.gguf"

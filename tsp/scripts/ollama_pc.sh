#!/usr/bin/env bash
# Private Ollama instance for the phase 5 planner, separate from the server's shared Ollama:
# own binary, own model store, own HOME (so ~/.ollama is never touched) and port 11435 (never 11434).
#
#   scripts/ollama_pc.sh serve            # start the server (run it in tmux "ollama_pc", under nice -n 10)
#   scripts/ollama_pc.sh list | pull X    # any ollama client command against this instance
set -euo pipefail

TOOLS=/scratch/pcanaste/tools
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=$TOOLS/ollama_models
export HOME=$TOOLS/ollama_home            # Ollama keeps its key pair in $HOME/.ollama
export OLLAMA_MAX_LOADED_MODELS=1         # one model in GPU memory at a time
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-4096}
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-q8_0}
mkdir -p "$HOME"
exec "$TOOLS/ollama/bin/ollama" "$@"

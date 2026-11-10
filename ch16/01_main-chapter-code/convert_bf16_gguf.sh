#!/usr/bin/env bash
# Convert the Hugging Face BF16 checkpoint of Qwen3-8B into a BF16 GGUF for llama-server, so
# llama.cpp and vLLM/SGLang serve bit-identical weights (models/ is gitignored, 16 GB).
#
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
#   uv pip install --python ch16/01_main-chapter-code/.venv-cuda /tmp/llama.cpp/gguf-py
#   LLAMA_CPP=/tmp/llama.cpp bash ch16/01_main-chapter-code/convert_bf16_gguf.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP="${LLAMA_CPP:?point LLAMA_CPP at a llama.cpp checkout}"
SNAPSHOT="$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)"
mkdir -p "$HERE/models"
"$HERE/.venv-cuda/bin/python" "$LLAMA_CPP/convert_hf_to_gguf.py" "$SNAPSHOT" \
  --outtype bf16 --outfile "$HERE/models/Qwen3-8B-BF16.gguf"

#!/usr/bin/env bash
# Build llama.cpp's llama-server with CUDA for an RTX 3090 (sm_86), pinned to the commit we ran.
# Takes about 9 minutes on 12 cores. Needs cmake, gcc and the CUDA toolkit (we used 12.4).
#
#   bash ch16/01_main-chapter-code/build_llama_cpp.sh       # -> $LLAMA_CPP_DIR/build/bin/llama-server
#
# The official image ghcr.io/ggml-org/llama.cpp:server-cuda is the no-compiler alternative; on our
# network one of its layers never finished downloading, so the chapter's numbers use this build.
set -euo pipefail
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
COMMIT="${LLAMA_CPP_COMMIT:-9f0e5098a5d5d8adb8dd5c59036ae2a38ab07db9}"  # 0.9.0-dev, 2026-11-07
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$COMMIT"
git -C "$LLAMA_CPP_DIR" checkout -q "$COMMIT"

export PATH="$CUDA_HOME/bin:$PATH" CUDACXX="$CUDA_HOME/bin/nvcc"
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build" --target llama-server -j "$(nproc)"
"$LLAMA_CPP_DIR/build/bin/llama-server" --version

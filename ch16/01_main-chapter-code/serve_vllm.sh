#!/usr/bin/env bash
# vLLM's OpenAI-compatible server on :8216, from the chapter's CUDA venv.
#   bash ch16/01_main-chapter-code/serve_vllm.sh                 # Qwen/Qwen3-8B, BF16
#   VLLM_API_KEY=secret bash ch16/01_main-chapter-code/serve_vllm.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"

# --gpu-memory-utilization is a fraction of the WHOLE card that vLLM may claim (weights +
# activations + KV cache). Leave room for whatever else lives on the GPU (2-3 GB on our shared box).
# --max-model-len caps one request's context; the KV pool is shared by all requests anyway.
# VLLM_USE_FLASHINFER_SAMPLER=0: vLLM 0.33 otherwise JIT-compiles a FlashInfer sampler on the
# first request, which needs nvcc and ninja on PATH (and stalls that request while it compiles).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Kernels compiled at runtime through NVRTC need libnvrtc-builtins from the pip CUDA 13 wheels.
export LD_LIBRARY_PATH="$HERE/.venv-cuda/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$HERE/.venv-cuda/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8216 \
  --served-model-name "${VLLM_SERVED_NAME:-$MODEL}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.86}" \
  --max-model-len "${VLLM_MAX_LEN:-8192}" \
  --max-num-seqs "${VLLM_MAX_SEQS:-64}" \
  --reasoning-parser qwen3 \
  ${VLLM_EXTRA_ARGS:-}

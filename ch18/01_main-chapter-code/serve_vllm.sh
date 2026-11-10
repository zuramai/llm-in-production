#!/usr/bin/env bash
# Serve a Qwen3 base model plus any number of LoRA adapters with vLLM on port 8218.
#
#   cd ch18/01_main-chapter-code
#   ./serve_vllm.sh Qwen/Qwen3-1.7B q17-lora q17-qlora
#
# Each adapter in adapters/<name>/ becomes a model name on the OpenAI-compatible API; the base
# model stays available under its own name. GPU_UTIL (default 0.85) caps vLLM's share of VRAM.
set -euo pipefail
cd "$(dirname "$0")"
base="${1:?usage: serve_vllm.sh BASE_MODEL [ADAPTER ...]}"
shift
modules=()
for name in "$@"; do
  modules+=("${name}=adapters/${name}")
done
lora_args=()
if [ "${#modules[@]}" -gt 0 ]; then
  lora_args=(--enable-lora --max-loras 4 --max-lora-rank 16 --lora-modules "${modules[@]}")
fi
# The FlashInfer sampler JIT-compiles a CUDA kernel on first use and needs nvcc + ninja; the
# PyTorch sampler needs neither and is plenty fast for this workload.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec .venv-cuda/bin/vllm serve "$base" \
  --host 127.0.0.1 --port 8218 \
  --max-model-len 4096 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  "${lora_args[@]}"

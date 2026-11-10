#!/usr/bin/env bash
# Every measured experiment of Chapter 17, in order, on one GPU.
#
#   bash ch17/01_main-chapter-code/run_gpu_benchmarks.sh                 # all phases
#   bash ch17/01_main-chapter-code/run_gpu_benchmarks.sh llama vllm_spec # just some phases
#
# Phases: ollama (quantization on Ollama), llama (llama-server prefix cache + draft model),
# vllm = vllm_bf16 vllm_noapc vllm_spec vllm_awq vllm_fp8 (quantization, prefix cache,
# speculative decoding). Each phase starts its server, runs the client scripts with the main uv
# environment, and stops the server again. Results accumulate in results/*.json.
# Needs: the GPU to yourself, Docker with --gpus (or LLAMA_BIN), .venv-cuda (vLLM), and the
# weights already downloaded (see the chapter README).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="$HERE/.venv-cuda"
NV_LIB="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
LOGS="$HERE/results/logs"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
LLAMA_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
# Set LLAMA_BIN=/path/to/llama-server to use a local CUDA build instead of the Docker image
# (the author's 2026-11-09 run used the Docker image).
LLAMA_BIN="${LLAMA_BIN:-}"
LLAMA_PORT=8117
VLLM_PORT=8217
# vLLM's FlashInfer top-k/top-p sampler JIT-compiles with nvcc and ninja; without them on PATH
# (the author's host) use the PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# vLLM claims a fixed share of the card at startup. 0.80 leaves room for a desktop or another
# small process; raise it on a dedicated GPU.
VLLM_UTIL="${VLLM_UTIL:-0.80}"
PHASES=("${@:-ollama llama vllm}")
PHASES=(${PHASES[*]})
mkdir -p "$LOGS"
cd "$HERE"

py() { (cd "$HERE" && uv run --project "$REPO" python "$@"); }
hf_path() { HF_HUB_OFFLINE=1 "$VENV/bin/hf" download "$@" --quiet; }

wait_http() {  # url, seconds
  for _ in $(seq 1 "$2"); do curl -sf "$1" >/dev/null && return 0; sleep 1; done
  echo "timeout waiting for $1" >&2; return 1
}

record_env() {
  {
    echo "=== $(date -Is) phases: ${PHASES[*]}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    echo "vllm $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
    echo "torch $("$VENV/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    if [[ -n "$LLAMA_BIN" ]]; then "$LLAMA_BIN" --version 2>&1 | grep -i version || true
    else docker run --rm --gpus all --entrypoint /app/llama-server "$LLAMA_IMAGE" --version \
      2>&1 | grep -i version || true; fi
    echo "ollama $(curl -s localhost:11434/api/version)"
  } | tee -a "$LOGS/environment.txt"
}

# ------------------------------------------------------------------------------- Ollama
phase_ollama() {
  for cfg in ollama-q4_K_M ollama-q8_0 ollama-fp16; do
    py ch17_quant_bench.py --config "$cfg"
  done
  # Unload only what is resident: keep_alive=0 on a model that isn't loaded loads it first.
  for m in $(curl -s localhost:11434/api/ps | python3 -c \
      'import json,sys; print(" ".join(m["name"] for m in json.load(sys.stdin)["models"]))'); do
    curl -s localhost:11434/api/generate -d "{\"model\": \"$m\", \"keep_alive\": 0}" >/dev/null
  done
  # keep_alive=0 returns before the runner exits; wait until Ollama reports nothing loaded.
  for _ in $(seq 60); do
    [[ "$(curl -s localhost:11434/api/ps)" == *'"models":[]'* ]] && break
    sleep 1
  done
  sleep 3
}

# ------------------------------------------------------------------------------- llama-server
gguf() {  # a GGUF path as llama-server sees it (inside the container, or as is)
  if [[ -n "$LLAMA_BIN" ]]; then echo "$1"; else echo "/hf/${1#"$HF_HOME_DIR/"}"; fi
}
llama_up() {  # name, extra args...
  local name="$1"; shift
  local args=(-m "$(gguf "$TARGET_GGUF")" -ngl 99 -c 32768 --jinja)
  if [[ -n "$LLAMA_BIN" ]]; then
    nohup "$LLAMA_BIN" "${args[@]}" --host 127.0.0.1 --port "$LLAMA_PORT" "$@" \
      > "$LOGS/llama-$name.log" 2>&1 &
    LLAMA_PID=$!
  else
    docker rm -f ch17-llama >/dev/null 2>&1 || true
    docker run -d --name ch17-llama --gpus all -p "127.0.0.1:$LLAMA_PORT:8080" \
      -v "$HF_HOME_DIR:/hf:ro" "$LLAMA_IMAGE" "${args[@]}" --host 0.0.0.0 --port 8080 "$@" \
      >/dev/null
  fi
  wait_http "http://localhost:$LLAMA_PORT/health" 300
}
llama_down() {
  if [[ -n "$LLAMA_BIN" ]]; then kill "$LLAMA_PID"; wait "$LLAMA_PID" 2>/dev/null || true
  else docker logs ch17-llama > "$LOGS/llama-$1.log" 2>&1; docker rm -f ch17-llama >/dev/null
  fi
}

phase_llama() {
  TARGET_GGUF="$(hf_path Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf)"
  DRAFT_GGUF="$(hf_path Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf)"

  llama_up baseline
  py ch17_prefix_bench.py --engine llama --base-url "http://localhost:$LLAMA_PORT"
  py ch17_spec_bench.py --base-url "http://localhost:$LLAMA_PORT" --label "llama/baseline"
  llama_down baseline

  llama_up draft -md "$(gguf "$DRAFT_GGUF")" -ngld 99 --spec-draft-n-max 8
  py ch17_spec_bench.py --base-url "http://localhost:$LLAMA_PORT" --label "llama/draft-0.6b"
  llama_down draft
}

# ------------------------------------------------------------------------------- vLLM
vllm_up() {  # name, model, extra args... ; returns non-zero if the server dies on startup
  local name="$1" model="$2"; shift 2
  # LD_LIBRARY_PATH: vLLM's FP8 kernels JIT-compile with NVRTC, which needs
  # libnvrtc-builtins.so.13.0 from the venv's nvidia wheel (not on the default loader path).
  LD_LIBRARY_PATH="$NV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" HF_HUB_OFFLINE=1 \
  nohup "$VENV/bin/vllm" serve "$model" --host 127.0.0.1 --port "$VLLM_PORT" \
    --gpu-memory-utilization "${UTIL_OVERRIDE:-$VLLM_UTIL}" --enable-prompt-tokens-details \
    "$@" > "$LOGS/vllm-$name.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 900); do
    curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null && return 0
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM $name died, see $LOGS/vllm-$name.log"; return 1; }
    sleep 1
  done
  return 1
}
vllm_down() { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; sleep 5; }
URL="http://localhost:$VLLM_PORT"

phase_vllm_bf16() {
  # 12288 tokens: at 0.80 of 24 GB, BF16 weights (16.4 GB) leave KV cache for ~12.9k tokens;
  # the longest prefix-cache prompt is 10.9k tokens.
  vllm_up bf16 Qwen/Qwen3-8B --max-model-len 12288 || { vllm_down; return 0; }
  py ch17_quant_bench.py --config vllm-bf16 --server-log "$LOGS/vllm-bf16.log" \
    --throughput 1 8 32
  py ch17_prefix_bench.py --engine vllm --base-url "$URL" --model Qwen/Qwen3-8B --apc on
  py ch17_spec_bench.py --base-url "$URL" --model Qwen/Qwen3-8B --label vllm/baseline \
    --concurrency 1 8
  vllm_down
}
phase_vllm_noapc() {
  vllm_up bf16-noapc Qwen/Qwen3-8B --max-model-len 12288 --no-enable-prefix-caching \
    || { vllm_down; return 0; }
  py ch17_prefix_bench.py --engine vllm --base-url "$URL" --model Qwen/Qwen3-8B --apc off
  vllm_down
}
spec_run() {  # label, speculative-config JSON, gpu-memory-utilization, max-model-len
  # Each method needs a different memory budget on a 24 GB card shared with a ~2.3 GB process
  # (measured on the author's run): n-gram adds no weights and OOMs during warm-up above 0.80;
  # EAGLE-3 needs 0.88 for any KV cache; the 0.6B draft model needs 0.84 and a 2k context.
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UTIL_OVERRIDE="$3" \
    vllm_up "$1" Qwen/Qwen3-8B --max-model-len "$4" --speculative-config "$2" \
    || { vllm_down; return 0; }
  py ch17_spec_bench.py --base-url "$URL" --model Qwen/Qwen3-8B --label "vllm/$1" \
    --concurrency 1 8 || echo "spec bench $1 failed"
  vllm_down
}
phase_vllm_spec() {
  spec_run ngram '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}' \
    0.80 4096
  spec_run eagle3 '{"method": "eagle3", "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
                    "num_speculative_tokens": 3}' 0.88 4096
  spec_run draft-0.6b '{"method": "draft_model", "model": "Qwen/Qwen3-0.6B",
                        "num_speculative_tokens": 4}' 0.84 2048
}
phase_vllm_awq() {
  vllm_up awq Qwen/Qwen3-8B-AWQ --max-model-len 16384 || { vllm_down; return 0; }
  py ch17_quant_bench.py --config vllm-awq-int4 --server-log "$LOGS/vllm-awq.log" \
    --throughput 1 8 32
  vllm_down
}
phase_vllm_fp8() {
  vllm_up fp8 RedHatAI/Qwen3-8B-FP8-dynamic --max-model-len 16384 || { vllm_down; return 0; }
  py ch17_quant_bench.py --config vllm-fp8-w8a16 --server-log "$LOGS/vllm-fp8.log" \
    --throughput 1 8 32
  vllm_down
}
phase_vllm() {
  phase_vllm_bf16; phase_vllm_noapc; phase_vllm_spec; phase_vllm_awq; phase_vllm_fp8
}

record_env
check_free() {  # vLLM refuses to start if less than UTIL x total is free; fail early instead
  local free total
  read -r free total < <(nvidia-smi --query-gpu=memory.free,memory.total \
    --format=csv,noheader,nounits | tr -d ',')
  echo "GPU free: $free / $total MiB"
  if (( free * 100 < total * ${VLLM_UTIL#0.} )); then
    echo "not enough free GPU memory for VLLM_UTIL=$VLLM_UTIL; stopping" >&2; exit 1
  fi
}
for p in "${PHASES[@]}"; do
  echo "=== phase $p ($(date +%T))"
  check_free
  "phase_$p"
done
echo "=== done ($(date +%T))"

#!/usr/bin/env bash
# Benchmark each engine in turn on ONE GPU. Needs the GPU to yourself (shared Ollama stopped).
#
#   bash ch16/01_main-chapter-code/run_gpu_benchmarks.sh ollama-default ollama-np32 \
#        llamacpp-q4 llamacpp-bf16 vllm-bf16 sglang-bf16
#
# Each step: start the server, wait until it answers, run ch16_bench.py, snapshot GPU memory and
# /metrics, run a prefill sweep, stop the server. Results land in results/<label>.json, logs in
# results/logs/. Prerequisites: build_llama_cpp.sh, convert_bf16_gguf.sh, .venv-cuda (vLLM) and
# the SGLang image (docker compose pull).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p results/logs results/prefill
PY="${PY:-$HERE/../../.venv/bin/python}"
LEVELS="${LEVELS:-1,2,4,8,16,32}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/ollama/bin/ollama}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}/build/bin/llama-server}"
# Ollama stores qwen3:8b (Q4_K_M) as a content-addressed GGUF blob; llama-server reads it as-is.
Q4_GGUF="${Q4_GGUF:-$(grep -m1 '^FROM ' <("$OLLAMA_BIN" show --modelfile qwen3:8b) | cut -d' ' -f2)}"
NOTHINK='{"chat_template_kwargs": {"enable_thinking": false}, "ignore_eos": true}'
NOTHINK_OPTS='"chat_template_kwargs": {"enable_thinking": false},'

wait_ready() {  # url timeout_s -> prints seconds until the url answered 200
  local t0=$SECONDS
  until curl -fsS -o /dev/null "$1"; do
    if (( SECONDS - t0 > $2 )); then echo "timeout waiting for $1" >&2; return 1; fi
    sleep 1
  done
  echo $(( SECONDS - t0 ))
}

record() {  # label startup_s metrics_url
  local mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "results/logs/$1.gpu.txt"
  printf '{"label": "%s", "startup_s": %s, "gpu_mem_used_mib": %s}\n' "$1" "$2" "$mem" \
    >> results/startup.jsonl
  if [[ -n "${3:-}" ]]; then curl -fsS "$3" > "results/logs/$1.metrics.txt" || true; fi
}

bench() {  # label base_url model [extra ch16_bench.py args]
  local label=$1 url=$2 model=$3; shift 3
  "$PY" ch16_bench.py --base-url "$url" --model "$model" --label "$label" \
    --concurrency "$LEVELS" --requests 32 --prompt-tokens 512 --max-tokens 256 \
    --out "results/$label.json" "$@" | tee "results/logs/$label.bench.txt"
}

extras() {  # label base_url model [request options]: one sample response + a prefill sweep
  local label=$1 url=$2 model=$3; shift 3
  curl -sS "$url/chat/completions" -H 'content-type: application/json' \
    -H "authorization: Bearer ${BENCH_API_KEY:-none}" \
    -d "{\"model\": \"$model\", \"max_tokens\": 60, \"temperature\": 0, $SAMPLE_OPTS
         \"messages\": [{\"role\": \"user\", \"content\": \"How long do I have to return an item at ShopFlow? Say you don't know if unsure.\"}]}" \
    > "results/logs/$label.sample.json" || true
  # TTFT vs prompt length, one request at a time (Chapter 1 promised this measurement).
  for p in 256 1024 4096; do
    "$PY" ch16_bench.py --base-url "$url" --model "$model" --label "$label" \
      --concurrency 1 --requests 4 --prompt-tokens "$p" --max-tokens 16 \
      --out "results/prefill/$label-$p.json" "$@" | tee -a "results/logs/$label.prefill.txt"
  done
}

run_ollama() {  # label num_parallel context_length ("" = Ollama's default)
  local label=$1
  env OLLAMA_HOST=127.0.0.1:8116 OLLAMA_KEEP_ALIVE=30m ${2:+OLLAMA_NUM_PARALLEL=$2} \
    ${3:+OLLAMA_CONTEXT_LENGTH=$3} "$OLLAMA_BIN" serve > "results/logs/$label.server.log" 2>&1 &
  local pid=$!
  wait_ready http://127.0.0.1:8116/api/version 60 > /dev/null
  # Ollama loads the model on the first request: time a 1-token call as its "startup".
  local t0=$SECONDS
  curl -fsS http://127.0.0.1:8116/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' > /dev/null
  local startup=$(( SECONDS - t0 ))
  # Ollama runs llama.cpp underneath: its runner's command line shows the slots and context.
  pgrep -a -P "$pid" > "results/logs/$label.runner.txt" || true
  # Ollama has no ignore_eos: output length is up to the model (we ask for 400+ words).
  bench "$label" http://127.0.0.1:8116/v1 qwen3:8b --reasoning-effort none
  record "$label" "$startup" ""
  SAMPLE_OPTS='"reasoning_effort": "none",' extras "$label" http://127.0.0.1:8116/v1 qwen3:8b \
    --reasoning-effort none
  kill "$pid"; wait "$pid" 2> /dev/null || true
}

run_llamacpp() {  # label gguf_path ctx_size parallel [extra llama-server flags]
  local label=$1
  # --parallel N slots split --ctx-size evenly: 32768 / 32 = 1,024 tokens of context per slot,
  # unless --kv-unified lets all slots share one pool.
  "$LLAMA_SERVER_BIN" --model "$2" --alias qwen3-8b --host 127.0.0.1 --port 8116 \
    --n-gpu-layers 999 --ctx-size "$3" --parallel "$4" ${5:-} \
    --flash-attn on --jinja --metrics > "results/logs/$label.server.log" 2>&1 &
  local pid=$!
  local startup
  startup=$(wait_ready http://127.0.0.1:8116/health 600)
  bench "$label" http://127.0.0.1:8116/v1 qwen3-8b --extra-body "$NOTHINK"
  record "$label" "$startup" http://127.0.0.1:8116/metrics
  SAMPLE_OPTS="$NOTHINK_OPTS" extras "$label" http://127.0.0.1:8116/v1 qwen3-8b \
    --extra-body "$NOTHINK"
  kill "$pid"; wait "$pid" 2> /dev/null || true
}

run_vllm() {
  local label=$1
  # With an API key: show what it protects (and what it doesn't) before benchmarking.
  export VLLM_API_KEY="ch16-$RANDOM$RANDOM"
  export BENCH_API_KEY=$VLLM_API_KEY
  bash serve_vllm.sh > "results/logs/$label.server.log" 2>&1 &
  local pid=$!
  local startup
  startup=$(wait_ready http://127.0.0.1:8216/health 900)
  {
    echo "GET /v1/models without key -> $(curl -s -o /dev/null -w '%{http_code}' \
      http://127.0.0.1:8216/v1/models)"
    echo "GET /v1/models with key    -> $(curl -s -o /dev/null -w '%{http_code}' \
      -H "authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8216/v1/models)"
    echo "GET /metrics without key   -> $(curl -s -o /dev/null -w '%{http_code}' \
      http://127.0.0.1:8216/metrics)"
  } | tee "results/logs/$label.auth.txt"
  bench "$label" http://127.0.0.1:8216/v1 Qwen/Qwen3-8B --extra-body "$NOTHINK"
  record "$label" "$startup" http://127.0.0.1:8216/metrics
  SAMPLE_OPTS="$NOTHINK_OPTS" extras "$label" http://127.0.0.1:8216/v1 Qwen/Qwen3-8B \
    --extra-body "$NOTHINK"
  kill "$pid"; wait "$pid" 2> /dev/null || true
  unset VLLM_API_KEY BENCH_API_KEY
}

run_sglang() {
  local label=$1
  docker compose up -d sglang
  local startup
  startup=$(wait_ready http://127.0.0.1:8316/health 900)
  bench "$label" http://127.0.0.1:8316/v1 Qwen/Qwen3-8B --extra-body "$NOTHINK"
  record "$label" "$startup" http://127.0.0.1:8316/metrics
  SAMPLE_OPTS="$NOTHINK_OPTS" extras "$label" http://127.0.0.1:8316/v1 Qwen/Qwen3-8B \
    --extra-body "$NOTHINK"
  docker compose logs sglang > "results/logs/$label.server.log" 2>&1
  docker compose down
}

check_gpu() {  # wait up to 60 s for memory to drop below MAX_FOREIGN_MIB, else refuse to start
  local used t0=$SECONDS
  while used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        (( used > ${MAX_FOREIGN_MIB:-4096} )); do
    if (( SECONDS - t0 > 60 )); then
      echo "refusing to start $1: ${used} MiB already in use" >&2
      return 1
    fi
    sleep 2
  done
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "results/logs/$1.gpu-before.txt"
  echo "GPU memory in use before $1: ${used} MiB"
}

for step in "$@"; do
  echo "=== $step ($(date -Is))"
  check_gpu "$step"
  case $step in
    ollama-default) run_ollama "$step" "" "" ;;
    ollama-np32) run_ollama "$step" 32 2048 ;;
    llamacpp-q4) run_llamacpp "$step" "$Q4_GGUF" 32768 32 ;;
    llamacpp-q4-kvu) run_llamacpp "$step" "$Q4_GGUF" 32768 32 --kv-unified ;;
    # BF16: 16.4 GB of weights + 4.5 GB of KV for 32 slots ~ 22 GB, next to a 2.3 GB co-tenant.
    llamacpp-bf16) run_llamacpp "$step" "$HERE/models/Qwen3-8B-BF16.gguf" "${BF16_CTX:-32768}" \
      "${BF16_PARALLEL:-32}" ;;
    vllm-bf16) run_vllm "$step" ;;
    sglang-bf16) SGLANG_MEM_FRACTION=0.86 run_sglang "$step" ;;
    sglang-bf16-mem080) SGLANG_MEM_FRACTION=0.80 run_sglang "$step" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done

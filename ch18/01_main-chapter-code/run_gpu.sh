#!/usr/bin/env bash
# Every GPU step of Chapter 18, in the order the author ran them (needs the whole GPU).
#
#   cd ch18/01_main-chapter-code && ./run_gpu.sh train       # 8 training runs -> adapters/, results/
#   cd ch18/01_main-chapter-code && ./run_gpu.sh eval-1.7b   # vLLM 1.7B + 4 adapters, evals
#   cd ch18/01_main-chapter-code && ./run_gpu.sh eval-4b     # vLLM 4B + 3 adapters, evals
#
# Evals run from the main venv (uv run) against vLLM on :8218. Results are cached per run in
# results/eval__*.json; delete a file to re-run it.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-cuda/bin/python
EVAL="uv run python ch18_eval.py --endpoint vllm"

train() {
  $PY ch18_train.py --model Qwen/Qwen3-1.7B --name q17-lora
  $PY ch18_train.py --model Qwen/Qwen3-1.7B --name q17-qlora --4bit
  $PY ch18_train.py --model Qwen/Qwen3-1.7B --name q17-lora-peft --backend peft
  $PY ch18_train.py --model Qwen/Qwen3-1.7B --name q17-qlora-peft --backend peft --4bit
  $PY ch18_train.py --model Qwen/Qwen3-1.7B --name q17-lora-v2 --train train.jsonl synthetic.jsonl
  $PY ch18_train.py --model Qwen/Qwen3-4B --name q4-lora
  $PY ch18_train.py --model Qwen/Qwen3-4B --name q4-qlora --4bit
  $PY ch18_train.py --model Qwen/Qwen3-4B --name q4-lora-v2 --train train.jsonl synthetic.jsonl
}

serve() {  # serve BASE ADAPTER...; waits until the server answers
  ./serve_vllm.sh "$@" > "results/vllm_$(basename "$1").log" 2>&1 &
  echo $! > .vllm.pid
  for _ in $(seq 1 180); do
    curl -sf localhost:8218/v1/models > /dev/null && return 0
    sleep 2
  done
  echo "vLLM did not come up; see results/vllm_$(basename "$1").log" >&2
  return 1
}

stop() {
  kill "$(cat .vllm.pid)" 2> /dev/null || true
  rm -f .vllm.pid
  sleep 5
}

eval_base() {  # eval_base BASE ADAPTER...
  local base="$1"; shift
  for c in 1 32; do
    $EVAL --model "$base" --prompt ch02 --concurrency "$c"
    $EVAL --model "$base" --prompt fewshot --concurrency "$c"
    for a in "$@"; do $EVAL --model "$a" --prompt ft --concurrency "$c"; done
  done
}

case "${1:-}" in
  train) train ;;
  eval-1.7b)
    serve Qwen/Qwen3-1.7B q17-lora q17-qlora q17-lora-peft q17-lora-v2
    trap stop EXIT
    eval_base Qwen/Qwen3-1.7B q17-lora q17-qlora q17-lora-peft q17-lora-v2 ;;
  eval-4b)
    serve Qwen/Qwen3-4B q4-lora q4-qlora q4-lora-v2
    trap stop EXIT
    eval_base Qwen/Qwen3-4B q4-lora q4-qlora q4-lora-v2 ;;
  *) echo "usage: run_gpu.sh train|eval-1.7b|eval-4b" >&2; exit 2 ;;
esac

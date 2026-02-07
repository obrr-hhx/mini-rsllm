#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/soak_metal.sh <model.gguf> [prompt]"
  echo "Env overrides: BIN, ITERATIONS, N_TOKENS, GPU_LAYERS, MAX_RSS_GROWTH_KB, TEMPERATURE, TOP_P, TOP_K, SEED"
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Metal soak test only supports macOS."
  exit 1
fi

MODEL_PATH="$1"
PROMPT="${2:-Hello}"
BIN="${BIN:-./target/release/mini-rsllm}"
ITERATIONS="${ITERATIONS:-20}"
N_TOKENS="${N_TOKENS:-32}"
GPU_LAYERS="${GPU_LAYERS:-9999}"
MAX_RSS_GROWTH_KB="${MAX_RSS_GROWTH_KB:-262144}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-40}"
SEED="${SEED:-42}"

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN"
  echo "Build first: cargo build --release --features metal"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH"
  exit 1
fi

run_once() {
  local out_file="$1"
  local metrics_file="$2"
  set +e
  /usr/bin/time -l "$BIN" "$MODEL_PATH" \
    --device metal \
    --gpu-layers "$GPU_LAYERS" \
    -p "$PROMPT" \
    -n "$N_TOKENS" \
    -t "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" >"$out_file" 2>"$metrics_file"
  local status=$?
  set -e
  return $status
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

baseline_out="$tmp_dir/baseline.out"
baseline_metrics="$tmp_dir/baseline.metrics"
if ! run_once "$baseline_out" "$baseline_metrics"; then
  echo "Baseline run failed:"
  cat "$baseline_metrics"
  exit 1
fi

baseline_rss="$(awk '/maximum resident set size/{print $1}' "$baseline_metrics")"
if [[ -z "$baseline_rss" ]]; then
  echo "Failed to parse baseline RSS."
  exit 1
fi

max_rss="$baseline_rss"
last_rss="$baseline_rss"
sum_real_s=0.0
sum_tps=0.0

for ((i = 1; i <= ITERATIONS; i++)); do
  out_file="$tmp_dir/run_${i}.out"
  metrics_file="$tmp_dir/run_${i}.metrics"
  if ! run_once "$out_file" "$metrics_file"; then
    echo "Run failed at iteration $i:"
    cat "$metrics_file"
    exit 1
  fi

  if ! cmp -s "$baseline_out" "$out_file"; then
    echo "Output mismatch at iteration $i (expected deterministic output with temperature=$TEMPERATURE)."
    exit 1
  fi

  rss="$(awk '/maximum resident set size/{print $1}' "$metrics_file")"
  real_s="$(awk '$2=="real"{print $1}' "$metrics_file" | head -n1)"
  if [[ -z "$rss" || -z "$real_s" ]]; then
    echo "Failed to parse metrics at iteration $i."
    exit 1
  fi

  tps="$(awk -v n="$N_TOKENS" -v t="$real_s" 'BEGIN { if (t > 0) printf "%.6f", n / t; else print "0" }')"
  sum_real_s="$(awk -v a="$sum_real_s" -v b="$real_s" 'BEGIN { printf "%.6f", a + b }')"
  sum_tps="$(awk -v a="$sum_tps" -v b="$tps" 'BEGIN { printf "%.6f", a + b }')"

  if (( rss > max_rss )); then
    max_rss="$rss"
  fi
  last_rss="$rss"
done

rss_growth="$((last_rss - baseline_rss))"
if (( rss_growth > MAX_RSS_GROWTH_KB )); then
  echo "RSS growth too high: ${rss_growth} KB (limit ${MAX_RSS_GROWTH_KB} KB)."
  exit 1
fi

avg_real_s="$(awk -v s="$sum_real_s" -v n="$ITERATIONS" 'BEGIN { printf "%.6f", s / n }')"
avg_tps="$(awk -v s="$sum_tps" -v n="$ITERATIONS" 'BEGIN { printf "%.6f", s / n }')"

echo "=== Metal soak test passed ==="
echo "model: $MODEL_PATH"
echo "iterations: $ITERATIONS"
echo "n_tokens: $N_TOKENS"
echo "gpu_layers: $GPU_LAYERS"
echo "baseline_rss_kb: $baseline_rss"
echo "last_rss_kb: $last_rss"
echo "max_rss_kb: $max_rss"
echo "rss_growth_kb: $rss_growth"
echo "avg_real_s: $avg_real_s"
echo "avg_tokens_per_second: $avg_tps"

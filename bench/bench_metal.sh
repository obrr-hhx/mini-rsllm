#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bench/bench_metal.sh <model.gguf> [prompt]"
  echo "Env overrides: BIN, N_TOKENS, TEMPERATURE, TOP_P, TOP_K, SEED, GPU_LAYERS, MIN_SPEEDUP, ENFORCE_MIN_SPEEDUP"
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Metal benchmark only supports macOS."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="$1"
PROMPT="${2:-The answer is}"
BIN="${BIN:-$REPO_ROOT/target/release/mini-rsllm}"
N_TOKENS="${N_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-40}"
SEED="${SEED:-42}"
GPU_LAYERS="${GPU_LAYERS:-9999}"
MIN_SPEEDUP="${MIN_SPEEDUP:-2.0}"
ENFORCE_MIN_SPEEDUP="${ENFORCE_MIN_SPEEDUP:-0}"
SKIP_METAL_PREFLIGHT="${SKIP_METAL_PREFLIGHT:-0}"

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN"
  echo "Build first: cargo build --release --features metal"
  exit 1
fi

CPU_OUT="$(mktemp)"
METAL_OUT="$(mktemp)"
trap 'rm -f "$CPU_OUT" "$METAL_OUT"' EXIT

run_case() {
  local label="$1"
  local out_file="$2"
  shift 2

  local time_report status real_s user_s sys_s
  set +e
  time_report="$(
    {
      /usr/bin/time -p "$BIN" "$MODEL_PATH" \
        -p "$PROMPT" \
        -n "$N_TOKENS" \
        -t "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --top-k "$TOP_K" \
        --seed "$SEED" \
        "$@" >"$out_file"
    } 2>&1
  )"
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    echo "[$label] failed" >&2
    printf "%s\n" "$time_report" >&2
    return $status
  fi

  real_s="$(printf "%s\n" "$time_report" | awk '/^real /{print $2}')"
  user_s="$(printf "%s\n" "$time_report" | awk '/^user /{print $2}')"
  sys_s="$(printf "%s\n" "$time_report" | awk '/^sys /{print $2}')"
  printf "%s,%s,%s\n" "$real_s" "$user_s" "$sys_s"
}

if [[ "$SKIP_METAL_PREFLIGHT" != "1" ]]; then
  preflight_log="$(mktemp)"
  trap 'rm -f "$CPU_OUT" "$METAL_OUT" "$preflight_log"' EXIT
  set +e
  "$BIN" "$MODEL_PATH" \
    -p "$PROMPT" \
    -n 1 \
    -t "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --seed "$SEED" \
    --device metal \
    --gpu-layers "$GPU_LAYERS" >/dev/null 2>"$preflight_log"
  preflight_status=$?
  set -e
  if [[ $preflight_status -ne 0 ]]; then
    echo "[preflight] metal backend unavailable:" >&2
    cat "$preflight_log" >&2
    echo "Hint: ensure binary was built with '--features metal' and this shell session can access Metal." >&2
    exit 1
  fi
fi

CPU_METRICS=""
if ! CPU_METRICS="$(run_case cpu "$CPU_OUT" --device cpu)"; then
  exit 1
fi
METAL_METRICS=""
if ! METAL_METRICS="$(run_case metal "$METAL_OUT" --device metal --gpu-layers "$GPU_LAYERS")"; then
  exit 1
fi

CPU_REAL="$(printf "%s" "$CPU_METRICS" | cut -d, -f1)"
CPU_USER="$(printf "%s" "$CPU_METRICS" | cut -d, -f2)"
CPU_SYS="$(printf "%s" "$CPU_METRICS" | cut -d, -f3)"
METAL_REAL="$(printf "%s" "$METAL_METRICS" | cut -d, -f1)"
METAL_USER="$(printf "%s" "$METAL_METRICS" | cut -d, -f2)"
METAL_SYS="$(printf "%s" "$METAL_METRICS" | cut -d, -f3)"

CPU_TPS="$(awk -v n="$N_TOKENS" -v t="$CPU_REAL" 'BEGIN { if (t > 0) printf "%.3f", n / t; else print "inf" }')"
METAL_TPS="$(awk -v n="$N_TOKENS" -v t="$METAL_REAL" 'BEGIN { if (t > 0) printf "%.3f", n / t; else print "inf" }')"
SPEEDUP="$(awk -v c="$CPU_TPS" -v m="$METAL_TPS" 'BEGIN { if (c + 0 > 0) printf "%.3f", (m + 0) / (c + 0); else print "inf" }')"

echo "=== mini-rsllm CPU vs Metal benchmark ==="
echo "model: $MODEL_PATH"
echo "prompt: $PROMPT"
echo "n_tokens: $N_TOKENS"
echo "gpu_layers: $GPU_LAYERS"
echo "min_speedup: $MIN_SPEEDUP"
echo "enforce_min_speedup: $ENFORCE_MIN_SPEEDUP"
echo
echo "[cpu]"
echo "real_s: $CPU_REAL"
echo "user_s: $CPU_USER"
echo "sys_s: $CPU_SYS"
echo "tokens_per_second: $CPU_TPS"
echo
echo "[metal]"
echo "real_s: $METAL_REAL"
echo "user_s: $METAL_USER"
echo "sys_s: $METAL_SYS"
echo "tokens_per_second: $METAL_TPS"
echo
echo "speedup_vs_cpu: ${SPEEDUP}x"

if [[ "$ENFORCE_MIN_SPEEDUP" == "1" ]]; then
  if ! awk -v got="$SPEEDUP" 'BEGIN { g = tolower(got); exit (g == "nan" || g == "inf" || g == "-inf") ? 1 : 0 }'; then
    echo "benchmark gate failed: invalid speedup value '${SPEEDUP}'"
    exit 2
  fi
  if ! awk -v got="$SPEEDUP" -v need="$MIN_SPEEDUP" 'BEGIN { exit !(got + 0 >= need + 0) }'; then
    echo "benchmark gate failed: speedup ${SPEEDUP}x < required ${MIN_SPEEDUP}x"
    exit 2
  fi
  echo "benchmark gate passed: speedup ${SPEEDUP}x >= required ${MIN_SPEEDUP}x"
fi

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bench/bench_cpu.sh <model.gguf> [prompt]"
  echo "Env overrides: BIN, N_TOKENS, TEMPERATURE, TOP_P, TOP_K, SEED"
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

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found or not executable: $BIN"
  echo "Build first: cargo build --release"
  exit 1
fi

OUT_FILE="$(mktemp)"
trap 'rm -f "$OUT_FILE"' EXIT

TIME_REPORT="$(
  {
    /usr/bin/time -p "$BIN" "$MODEL_PATH" \
      -p "$PROMPT" \
      -n "$N_TOKENS" \
      -t "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --seed "$SEED" \
      --device cpu >"$OUT_FILE"
  } 2>&1
)"

REAL_SEC="$(printf "%s\n" "$TIME_REPORT" | awk '/^real /{print $2}')"
USER_SEC="$(printf "%s\n" "$TIME_REPORT" | awk '/^user /{print $2}')"
SYS_SEC="$(printf "%s\n" "$TIME_REPORT" | awk '/^sys /{print $2}')"
TOKENS_PER_SEC="$(awk -v n="$N_TOKENS" -v t="$REAL_SEC" 'BEGIN { if (t > 0) printf "%.3f", n / t; else print "inf" }')"

echo "=== mini-rsllm CPU baseline ==="
echo "model: $MODEL_PATH"
echo "prompt: $PROMPT"
echo "n_tokens: $N_TOKENS"
echo "real_s: $REAL_SEC"
echo "user_s: $USER_SEC"
echo "sys_s: $SYS_SEC"
echo "tokens_per_second: $TOKENS_PER_SEC"
echo
echo "generated_text:"
cat "$OUT_FILE"

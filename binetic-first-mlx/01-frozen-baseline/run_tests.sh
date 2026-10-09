#!/usr/bin/env bash
# Phase 1+2: frozen Bonsai 2 temporal KV experiments
# Run from anywhere:  bash 01-frozen-baseline/run_tests.sh
set -uo pipefail

PY="/opt/homebrew/bin/python3.12"

# Resolve paths relative to this script (works from any CWD)
BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BONSAI_ROOT="$(dirname "$BASELINE_DIR")"
BINETIC_SRC="${BINETIC_SRC:-$HOME/Documents/binetic-ai/src}"

export PYTHONPATH="$BINETIC_SRC:$BONSAI_ROOT/bonsai2-27b-mlx/runtime:${PYTHONPATH:-}"

echo "============================================================"
echo "Phase 1+2: Frozen Baseline + KV Injection Experiments"
echo "Model: Ternary Bonsai 2 27B (MLX 2-bit)"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================================"
echo

# Sanity: the binetic package must be importable before we burn model time
if ! "$PY" -c "import binetic.temporal_injection" 2>/dev/null; then
  echo "ERROR: cannot import binetic.temporal_injection"
  echo "  Expected at: $BINETIC_SRC/binetic/"
  echo "  Set BINETIC_SRC to the binetic-ai/src dir and re-run."
  exit 1
fi
echo "OK: binetic package importable"
echo

# Phase 1: No-model tests first (fast)
"$PY" "$BASELINE_DIR/run_single_test.py" --test 4
echo
"$PY" "$BASELINE_DIR/run_single_test.py" --test 3
echo

# Phase 1: Baseline recall (requires model)
"$PY" "$BASELINE_DIR/run_single_test.py" --test 1 --num-turns 5 --tokens-per-turn 200
echo
"$PY" "$BASELINE_DIR/run_single_test.py" --test 2 --num-turns 5 --tokens-per-turn 200
echo

# Phase 2: KV injection (closes the recall gap from Test 2)
"$PY" "$BASELINE_DIR/run_single_test.py" --test 5 --num-turns 10 --tokens-per-turn 1000 --chunk-size 32768
echo
"$PY" "$BASELINE_DIR/run_single_test.py" --test 6 --num-turns 10 --tokens-per-turn 1000 --chunk-size 2048
echo

echo "============================================================"
echo "All Phase 1+2 tests complete."
echo "Results: $BASELINE_DIR/results/"
echo "============================================================"

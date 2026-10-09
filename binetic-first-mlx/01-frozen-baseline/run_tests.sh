#!/usr/bin/env bash
# Phase 1+2: frozen Bonsai 2 temporal KV experiments
# Run from repo root: bash experiments/01-frozen-baseline/run_tests.sh

PY="/opt/homebrew/bin/python3.12"
export PYTHONPATH="$(pwd)/src:$(pwd)/bonsai2-27b-mlx/runtime:${PYTHONPATH:-}"

echo "============================================================"
echo "Phase 1+2: Frozen Baseline + KV Injection Experiments"
echo "Model: Ternary Bonsai 2 27B (MLX 2-bit)"
echo "============================================================"
echo

# Phase 1: No-model tests first (fast)
${PY} experiments/01-frozen-baseline/run_single_test.py --test 4
echo
${PY} experiments/01-frozen-baseline/run_single_test.py --test 3
echo

# Phase 1: Baseline recall (requires model)
${PY} experiments/01-frozen-baseline/run_single_test.py --test 1 --num-turns 5 --tokens-per-turn 200
echo
${PY} experiments/01-frozen-baseline/run_single_test.py --test 2 --num-turns 5 --tokens-per-turn 200
echo

# Phase 2: KV injection (closes the recall gap from Test 2)
${PY} experiments/01-frozen-baseline/run_single_test.py --test 5 --num-turns 10 --tokens-per-turn 1000 --chunk-size 32768
echo
${PY} experiments/01-frozen-baseline/run_single_test.py --test 6 --num-turns 10 --tokens-per-turn 1000 --chunk-size 2048
echo

echo "============================================================"
echo "All Phase 1+2 tests complete."
echo "Results: experiments/01-frozen-baseline/results/"
echo "============================================================"

#!/usr/bin/env python3
"""Test 14: Spectral compression limit — find where recall breaks.

Test 13 showed spectral compression works at keep=0.25 (4x) with 100% recall.
This test pushes harder: keep=0.15, 0.10, 0.08, 0.05, 0.02, 0.01.

Goal: find the maximum compression ratio that preserves 100% recall.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
BASELINE_DIR = SCRIPT.parent
BONSAI_ROOT = BASELINE_DIR.parent
MODEL_DIR = BONSAI_ROOT / "bonsai2-27b-mlx"
BINETIC_SRC = Path.home() / "Documents" / "binetic-ai" / "src"

sys.path.insert(0, str(BINETIC_SRC))
sys.path.insert(0, str(MODEL_DIR / "runtime"))

from binetic.temporal_injection import FULL_ATTENTION_LAYERS, create_cache_with_temporal_kv  # noqa: E402

NUM_TURNS = 20
TOKENS_PER_TURN = 2000
BATCH = 512
EOS_TOKEN = 248044
KEEP_FRACS = [0.15, 0.10, 0.08, 0.05, 0.02, 0.01]


def make_context() -> list[dict]:
    np.random.seed(42)
    padding_words = ["the", "system", "configures", "values", "during", "initialization",
                     "process", "and", "maintains", "state", "across", "multiple",
                     "requests", "with", "proper", "handling", "of", "edge", "cases",
                     "including", "timeout", "recovery", "retry", "logic", "cache"]
    turns = []
    for t in range(NUM_TURNS):
        memory_value = np.random.randint(1000, 9999)
        padding = " ".join(np.random.choice(padding_words, size=max(1, TOKENS_PER_TURN // 2)))
        prompt = f"Turn {t}: The secret code is {memory_value}. System log: {padding}."
        turns.append({"turn": t, "memory_token": int(memory_value), "prompt": prompt})
    return turns


def _render(messages: list[dict]) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    template_path = MODEL_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_context(turns: list[dict]) -> str:
    full_prompt = " ".join(t["prompt"] for t in turns)
    return _render([{"role": "user", "content": full_prompt}])


def render_query() -> str:
    return _render([{"role": "user", "content": "What was the secret code from turn 0? Just the number."}])


def extract_kv(model, tok, rendered: str):
    import mlx.core as mx
    ids = tok.encode(rendered).ids
    cache = model.make_cache()
    for i in range(0, len(ids), BATCH):
        x = mx.array([ids[i:i + BATCH]], dtype=mx.int32)
        model(x, cache=cache)
        mx.eval()
    kv = {}
    for layer_idx in FULL_ATTENTION_LAYERS:
        c = cache[layer_idx]
        sk, sv = c.state
        kv[layer_idx] = (np.asarray(sk).astype(np.float32),
                         np.asarray(sv).astype(np.float32))
    return kv, len(ids)


def spectral_compress(kv: dict, keep_frac: float, mode: str = "topenergy") -> dict:
    compressed = {}
    for layer_idx, (k, v) in kv.items():
        L = k.shape[2]
        Fk = np.fft.rfft(k, axis=2)
        Fv = np.fft.rfft(v, axis=2)
        Pk = (Fk.real ** 2 + Fk.imag ** 2).mean(axis=(0, 3))
        Pv = (Fv.real ** 2 + Fv.imag ** 2).mean(axis=(0, 3))
        P = Pk + Pv
        n_keep = max(1, int(len(P) * keep_frac))
        if mode == "topenergy":
            keep_idx = np.argsort(P)[::-1][:n_keep]
        else:
            keep_idx = np.arange(n_keep)
        Fkz = np.zeros_like(Fk)
        Fvz = np.zeros_like(Fv)
        Fkz[:, :, keep_idx, :] = Fk[:, :, keep_idx, :]
        Fvz[:, :, keep_idx, :] = Fv[:, :, keep_idx, :]
        k_comp = np.fft.irfft(Fkz, n=L, axis=2)
        v_comp = np.fft.irfft(Fvz, n=L, axis=2)
        compressed[layer_idx] = (k_comp.astype(np.float16), v_comp.astype(np.float16))
    return compressed


def run_recall(model, tok, compressed_kv: dict, query_rendered: str) -> str:
    import mlx.core as mx
    cache = create_cache_with_temporal_kv(model, compressed_kv)
    query_ids = tok.encode(query_rendered).ids
    x = mx.array([query_ids], dtype=mx.int32)
    logits = model(x, cache=cache)
    mx.eval(logits)
    gen_tokens = []
    for _ in range(32):
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        gen_tokens.append(next_tok)
        if next_tok == EOS_TOKEN:
            break
        x = mx.array([[next_tok]], dtype=mx.int32)
        logits = model(x, cache=cache)
        mx.eval(logits)
    return tok.decode(gen_tokens)


def check_recall(decoded: str, expected: int) -> tuple[float, int | None]:
    numbers = re.findall(r'\b(\d{4})\b', decoded)
    actual = int(numbers[0]) if numbers else None
    return (1.0 if actual == expected else 0.0), actual


def main():
    t_start = time.time()
    print("=" * 64)
    print("Test 14: Spectral compression limit — find where recall breaks")
    print("=" * 64)

    print("[1/3] Loading model...")
    from artifact import load_model
    from tokenizers import Tokenizer
    t0 = time.time()
    model, _ = load_model(str(MODEL_DIR))
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"      loaded in {time.time() - t0:.1f}s")

    print("[2/3] Building context, extracting KV...")
    turns = make_context()
    rendered_context = render_context(turns)
    expected = turns[0]["memory_token"]
    kv, n_tokens = extract_kv(model, tok, rendered_context)
    L = n_tokens
    print(f"      {L} positions, {len(kv)} layers, expected code: {expected}")

    query_rendered = render_query()

    print("[3/3] Running recall tests at aggressive keep fractions...\n")
    results = []

    def run_one(name: str, compressed: dict, keep: float):
        t0 = time.time()
        decoded = run_recall(model, tok, compressed, query_rendered)
        dt = time.time() - t0
        recall, actual = check_recall(decoded, expected)
        status = "PASS" if recall == 1.0 else "FAIL"
        got = str(actual) if actual is not None else "None"
        ratio = 1.0 / keep
        print(f"  {name:<30} keep={keep:.2f}  {status}  got={got:>5}  {ratio:>6.1f}x  ({dt:.1f}s)")
        results.append({
            "method": name, "keep": keep, "recall": recall,
            "expected": expected, "actual": actual,
            "decoded": decoded[:80], "time_s": round(dt, 2),
            "compression_ratio": round(ratio, 1),
        })

    # Baseline
    run_one("baseline", {l: (k.astype(np.float16), v.astype(np.float16)) for l, (k, v) in kv.items()}, 1.0)

    # Aggressive spectral compression
    for keep in KEEP_FRACS:
        compressed = spectral_compress(kv, keep, "topenergy")
        run_one(f"spectral_topenergy", compressed, keep)

    # Save
    out = {
        "test": 14,
        "name": "spectral_compression_limit",
        "num_turns": NUM_TURNS,
        "tokens_per_turn": TOKENS_PER_TURN,
        "total_tokens": L,
        "expected_code": expected,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out_path = BASELINE_DIR / "results" / "test_14_limit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  {'Method':<30} {'Keep':>6} {'Recall':>8} {'Got':>8} {'Ratio':>8}")
    print(f"  {'-' * 30} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results:
        got = str(r["actual"]) if r["actual"] is not None else "None"
        print(f"  {r['method']:<30} {r['keep']:>6.2f} {r['recall']:>8.0%} {got:>8} {r['compression_ratio']:>7.1f}x")

    # Find the limit
    passing = [r for r in results if r["recall"] == 1.0 and r["keep"] < 1.0]
    failing = [r for r in results if r["recall"] == 0.0 and r["keep"] < 1.0]
    if passing:
        best = max(passing, key=lambda r: r["compression_ratio"])
        print(f"\nMAX VALIDATED COMPRESSION: {best['compression_ratio']:.1f}x (keep={best['keep']:.2f})")
    if failing:
        worst = min(failing, key=lambda r: r["compression_ratio"])
        print(f"FIRST FAILURE: {worst['compression_ratio']:.1f}x (keep={worst['keep']:.2f})")

    print(f"\nTotal elapsed: {out['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
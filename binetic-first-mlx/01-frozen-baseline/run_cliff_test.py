#!/usr/bin/env python3
"""Test 16: Push the hybrid further — aggressive spectral keep with small reference.

Test 15 showed keep_first=512 + spectral=0.25 → 3.73x, 100% recall.
This test pushes the spectral side much harder: keep=0.10, 0.05, 0.02, 0.01.

Also runs pure spectral (no reference) at the same keep fractions for
comparison — to see how much the reference is actually buying.

Goal: find the recall cliff. If the hybrid holds at keep=0.05 (20x on the
compressed part, ~1.9x overall), that's a strong result.
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
KEEP_FIRST = 512
AGGRESSIVE_KEEP = [0.10, 0.05, 0.02, 0.01]


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


def spectral_compress(kv: dict, keep_frac: float) -> dict:
    compressed = {}
    for layer_idx, (k, v) in kv.items():
        L = k.shape[2]
        Fk = np.fft.rfft(k, axis=2)
        Fv = np.fft.rfft(v, axis=2)
        Pk = (Fk.real ** 2 + Fk.imag ** 2).mean(axis=(0, 3))
        Pv = (Fv.real ** 2 + Fv.imag ** 2).mean(axis=(0, 3))
        P = Pk + Pv
        n_keep = max(1, int(len(P) * keep_frac))
        keep_idx = np.argsort(P)[::-1][:n_keep]
        Fkz = np.zeros_like(Fk)
        Fvz = np.zeros_like(Fv)
        Fkz[:, :, keep_idx, :] = Fk[:, :, keep_idx, :]
        Fvz[:, :, keep_idx, :] = Fv[:, :, keep_idx, :]
        k_comp = np.fft.irfft(Fkz, n=L, axis=2)
        v_comp = np.fft.irfft(Fvz, n=L, axis=2)
        compressed[layer_idx] = (k_comp.astype(np.float16), v_comp.astype(np.float16))
    return compressed


def hybrid_compress(kv: dict, keep_first: int, spectral_keep: float) -> dict:
    compressed = {}
    for layer_idx, (k, v) in kv.items():
        L = k.shape[2]
        if keep_first >= L:
            compressed[layer_idx] = (k.astype(np.float16), v.astype(np.float16))
            continue
        k_ref = k[:, :, :keep_first, :]
        v_ref = v[:, :, :keep_first, :]
        k_rest = k[:, :, keep_first:, :]
        v_rest = v[:, :, keep_first:, :]
        L_rest = L - keep_first
        Fk = np.fft.rfft(k_rest, axis=2)
        Fv = np.fft.rfft(v_rest, axis=2)
        Pk = (Fk.real ** 2 + Fk.imag ** 2).mean(axis=(0, 3))
        Pv = (Fv.real ** 2 + Fv.imag ** 2).mean(axis=(0, 3))
        P = Pk + Pv
        n_keep = max(1, int(len(P) * spectral_keep))
        keep_idx = np.argsort(P)[::-1][:n_keep]
        Fkz = np.zeros_like(Fk)
        Fvz = np.zeros_like(Fv)
        Fkz[:, :, keep_idx, :] = Fk[:, :, keep_idx, :]
        Fvz[:, :, keep_idx, :] = Fv[:, :, keep_idx, :]
        k_comp = np.fft.irfft(Fkz, n=L_rest, axis=2)
        v_comp = np.fft.irfft(Fvz, n=L_rest, axis=2)
        k_out = np.concatenate([k_ref, k_comp], axis=2)
        v_out = np.concatenate([v_ref, v_comp], axis=2)
        compressed[layer_idx] = (k_out.astype(np.float16), v_out.astype(np.float16))
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
    print("Test 16: Aggressive compression — find the recall cliff")
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

    print("[3/3] Running aggressive compression tests...\n")
    results = []

    def run_one(name: str, compressed: dict, ratio: float):
        t0 = time.time()
        decoded = run_recall(model, tok, compressed, query_rendered)
        dt = time.time() - t0
        recall, actual = check_recall(decoded, expected)
        status = "PASS" if recall == 1.0 else "FAIL"
        got = str(actual) if actual is not None else "None"
        print(f"  {name:<40} {status}  got={got:>5}  {ratio:>6.2f}x  ({dt:.1f}s)")
        results.append({
            "method": name, "recall": recall, "expected": expected,
            "actual": actual, "decoded": decoded[:80], "time_s": round(dt, 2),
            "compression_ratio": round(ratio, 2),
        })

    # Baseline
    run_one("baseline (full KV)",
            {l: (k.astype(np.float16), v.astype(np.float16)) for l, (k, v) in kv.items()}, 1.0)

    # Pure spectral, aggressive
    for keep in AGGRESSIVE_KEEP:
        compressed = spectral_compress(kv, keep)
        run_one(f"pure_spectral keep={keep}", compressed, 1.0 / keep)

    # Hybrid with small reference, aggressive
    for keep in AGGRESSIVE_KEEP:
        compressed = hybrid_compress(kv, KEEP_FIRST, keep)
        ref_vals = KEEP_FIRST * 1024 * 2
        rest_vals = (L - KEEP_FIRST) * 1024 * 2
        total_comp = ref_vals + rest_vals * keep
        ratio = (L * 1024 * 2) / total_comp
        run_one(f"hybrid ref={KEEP_FIRST} keep={keep}", compressed, ratio)

    # Save
    out = {
        "test": 16,
        "name": "aggressive_compression_cliff",
        "num_turns": NUM_TURNS,
        "tokens_per_turn": TOKENS_PER_TURN,
        "total_tokens": L,
        "expected_code": expected,
        "keep_first": KEEP_FIRST,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out_path = BASELINE_DIR / "results" / "test_16_cliff.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  {'Method':<40} {'Recall':>8} {'Got':>8} {'Ratio':>8}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results:
        got = str(r["actual"]) if r["actual"] is not None else "None"
        print(f"  {r['method']:<40} {r['recall']:>8.0%} {got:>8} {r['compression_ratio']:>7.2f}x")

    # Cliff analysis
    print(f"\n{'=' * 64}")
    print("CLIFF ANALYSIS")
    print(f"{'=' * 64}")
    for prefix in ("pure_spectral", "hybrid"):
        subset = [r for r in results if r["method"].startswith(prefix)]
        passing = [r for r in subset if r["recall"] == 1.0]
        failing = [r for r in subset if r["recall"] == 0.0]
        if passing:
            best = max(passing, key=lambda r: r["compression_ratio"])
            print(f"  {prefix}: best passing = {best['method']} ({best['compression_ratio']:.2f}x)")
        if failing:
            worst = min(failing, key=lambda r: r["compression_ratio"])
            print(f"  {prefix}: first failing = {worst['method']} ({worst['compression_ratio']:.2f}x)")
        if not passing and not failing:
            print(f"  {prefix}: no results")

    print(f"\nTotal elapsed: {out['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
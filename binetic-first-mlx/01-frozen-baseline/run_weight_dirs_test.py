#!/usr/bin/env python3
"""Test 13: Reconstruct KV from weight's directions + modulation — task-level recall.

The final gate: can we store the modulation (KV projected onto the weight's
dominant directions) instead of the full KV, and still get recall?

Method:
  1. SVD k_proj (1024×320) → U_w: weight's dominant directions in key space
  2. Project KV onto U_w: C = K @ U_w (L × n_w) — the modulation coefficients
  3. Reconstruct: K_rec = C @ U_w.T (L × 1024)
  4. Inject the reconstructed KV, measure recall

Compression ratios:
  Full KV: L × 1024 = 21.7M values
  Modulation: L × n_w (n_w = 123, 59, 30, 10)
  → 8.3x, 17.4x, 35.8x, 87x compression

If recall holds at n_w=59 (17x), the "store the modulation, not the KV"
bet is validated at the task level.
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
N_W_OPTIONS = [123, 59, 30, 10]


# ---------------------------------------------------------------- context

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


# ------------------------------------------------------------------ KV

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


def get_layer_weights(model, layer_idx: int) -> dict:
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    weights = {}
    li = str(layer_idx)
    for name, param in params:
        parts = name.split(".")
        if li not in parts:
            continue
        if not name.endswith(".weight"):
            continue
        joined = name.lower()
        for key, tag in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"),
                         ("o_proj", "o")):
            if key in joined:
                weights[tag] = np.asarray(param).astype(np.float32)
                break
    return weights


def project_kv_to_weight_dirs(kv: dict, model, n_w: int) -> dict:
    """Project KV onto the weight's dominant directions, reconstruct.

    For each layer:
      k_proj (1024×320) → SVD → U_w (1024×n_w)
      K (1, 4, L, 256) → flatten to (L, 1024)
      C = K_flat @ U_w (L×n_w) — modulation coefficients
      K_rec = C @ U_w.T (L×1024) → reshape to (1, 4, L, 256)

    Same for V with v_proj.
    """
    compressed = {}
    for layer_idx, (k, v) in kv.items():
        w = get_layer_weights(model, layer_idx)
        L = k.shape[2]

        # K: project onto k_proj's dominant directions
        if "k" in w:
            U_wk, s_wk, _ = np.linalg.svd(w["k"], full_matrices=False)
            U_wk = U_wk[:, :n_w]  # (1024, n_w)
            k_flat = k[0].transpose(1, 0, 2).reshape(L, -1)  # (L, 1024)
            C_k = k_flat @ U_wk  # (L, n_w)
            k_rec = C_k @ U_wk.T  # (L, 1024)
            k_rec = k_rec.reshape(1, 4, L, 256)
        else:
            k_rec = k

        # V: project onto v_proj's dominant directions
        if "v" in w:
            U_wv, s_wv, _ = np.linalg.svd(w["v"], full_matrices=False)
            U_wv = U_wv[:, :n_w]  # (1024, n_w)
            v_flat = v[0].transpose(1, 0, 2).reshape(L, -1)  # (L, 1024)
            C_v = v_flat @ U_wv  # (L, n_w)
            v_rec = C_v @ U_wv.T  # (L, 1024)
            v_rec = v_rec.reshape(1, 4, L, 256)
        else:
            v_rec = v

        compressed[layer_idx] = (k_rec.astype(np.float16), v_rec.astype(np.float16))
    return compressed


def spectral_compress(kv: dict, keep_frac: float, mode: str = "topenergy") -> dict:
    """Spectral compression (from test 10) for comparison."""
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


# ---------------------------------------------------------------- recall

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


# ------------------------------------------------------------------ main

def main():
    t_start = time.time()
    print("=" * 64)
    print("Test 13: Reconstruct KV from weight's directions — task-level recall")
    print("=" * 64)

    print("[1/4] Loading model...")
    from artifact import load_model
    from tokenizers import Tokenizer
    t0 = time.time()
    model, _ = load_model(str(MODEL_DIR))
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"      loaded in {time.time() - t0:.1f}s")

    print("[2/4] Building context, extracting KV...")
    turns = make_context()
    rendered_context = render_context(turns)
    expected = turns[0]["memory_token"]
    kv, n_tokens = extract_kv(model, tok, rendered_context)
    L = n_tokens
    print(f"      {L} positions, {len(kv)} layers, expected code: {expected}")

    query_rendered = render_query()

    print("[3/4] Running recall tests...\n")
    results = []

    def run_one(name: str, compressed: dict, n_w: int = 0):
        t0 = time.time()
        decoded = run_recall(model, tok, compressed, query_rendered)
        dt = time.time() - t0
        recall, actual = check_recall(decoded, expected)
        status = "PASS" if recall == 1.0 else "FAIL"
        got = str(actual) if actual is not None else "None"
        # Compression ratio
        full_vals = L * 1024 * 2  # K + V
        if n_w > 0:
            mod_vals = L * n_w * 2
            ratio = full_vals / mod_vals
            ratio_str = f"{ratio:.1f}x"
        else:
            ratio_str = "1.0x"
        print(f"  {name:<40} {status}  got={got:>5}  {ratio_str:>6}  ({dt:.1f}s)")
        results.append({
            "method": name, "n_w": n_w, "recall": recall,
            "expected": expected, "actual": actual,
            "decoded": decoded[:80], "time_s": round(dt, 2),
            "compression_ratio": round(full_vals / max(mod_vals, 1), 1) if n_w > 0 else 1.0,
        })

    # Baseline (no compression)
    run_one("baseline (full KV)",
            {l: (k.astype(np.float16), v.astype(np.float16)) for l, (k, v) in kv.items()})

    # Weight-direction projection at different n_w
    for n_w in N_W_OPTIONS:
        t0 = time.time()
        compressed = project_kv_to_weight_dirs(kv, model, n_w)
        proj_time = time.time() - t0
        run_one(f"weight_dirs n_w={n_w}", compressed, n_w)
        print(f"      (projection took {proj_time:.1f}s)")

    # Spectral compression for comparison
    for keep in (0.75, 0.50, 0.25):
        compressed = spectral_compress(kv, keep, "topenergy")
        run_one(f"spectral_topenergy keep={keep}", compressed)

    # Save
    out = {
        "test": 13,
        "name": "weight_direction_reconstruction_recall",
        "num_turns": NUM_TURNS,
        "tokens_per_turn": TOKENS_PER_TURN,
        "total_tokens": L,
        "expected_code": expected,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out_path = BASELINE_DIR / "results" / "test_13_weight_dirs.json"
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
        print(f"  {r['method']:<40} {r['recall']:>8.0%} {got:>8} {r['compression_ratio']:>7.1f}x")

    # Verdict
    print(f"\n{'=' * 64}")
    print("VERDICT")
    print(f"{'=' * 64}")
    wd_59 = next((r for r in results if r["n_w"] == 59), None)
    wd_30 = next((r for r in results if r["n_w"] == 30), None)
    wd_10 = next((r for r in results if r["n_w"] == 10), None)

    if wd_59 and wd_59["recall"] == 1.0:
        print(f"  VALIDATED. Weight-direction projection at n_w=59 (17x) preserves recall.")
        print(f"  The 'store the modulation, not the KV' bet is confirmed at the task level.")
    elif wd_30 and wd_30["recall"] == 1.0:
        print(f"  PARTIALLY VALIDATED. n_w=59 fails but n_w=30 (36x) preserves recall.")
        print(f"  The bet is alive — the modulation is even lower-rank than expected.")
    elif wd_10 and wd_10["recall"] == 1.0:
        print(f"  SURPRISE. Even n_w=10 (87x) preserves recall.")
        print(f"  The modulation is extremely low-rank. The bet is strongly validated.")
    else:
        print(f"  NOT VALIDATED at the task level.")
        print(f"  The weight-direction projection loses recall.")
        print(f"  The alignment (Test 12) is real but not sufficient for reconstruction.")
        print(f"  Fall back to the 2x spectral compression (Test 9/10).")

    print(f"\nTotal elapsed: {out['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
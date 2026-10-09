#!/usr/bin/env python3
"""Test 12: Alignment — do the KV's dominant directions match the weight's?

The final piece of the "gravitational relationship" test.

Test 11 proved:
  - KV is low-rank (59/1024 for K, 39/1024 for V)
  - KV energy is in the low-sequency Hadamard lanes (87.7%)

This test asks: are the KV's 59 dominant directions the SAME as the
weight's 123 dominant directions? If yes, the context is a modulation
of the weight field — it falls into the weight's lanes.

Method:
  1. SVD k_proj (1024×320) → U_w: 123 dominant directions in key space
  2. SVD KV (4096×1024) → Vt_kv: 59 dominant directions in key space
  3. Cosine similarity between each KV direction and each weight direction
  4. Null: random directions vs weight directions (expected ~0)

Win condition: mean best cosine >> null (alignment is real)
"""
from __future__ import annotations

import json
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

from binetic.temporal_injection import FULL_ATTENTION_LAYERS  # noqa: E402

NUM_TURNS = 20
TOKENS_PER_TURN = 2000
BATCH = 512
N_NULLS = 20
SVD_SUBSAMPLE = 4096


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


def render_context(turns: list[dict]) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    template_path = MODEL_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())
    full_prompt = " ".join(t["prompt"] for t in turns)
    return template.render(
        messages=[{"role": "user", "content": full_prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


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


# ---------------------------------------------------------------- align

def _normalize(X: np.ndarray, axis: int) -> np.ndarray:
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / (n + 1e-12)


def alignment_score(kv_dirs: np.ndarray, w_dirs: np.ndarray) -> dict:
    """Cosine similarity between KV directions and weight directions.

    kv_dirs: (n_kv, D) — KV's dominant directions in D-dim space
    w_dirs:  (D, n_w)  — weight's dominant directions in D-dim space

    Returns per-KV-direction best-match cosine + aggregate stats.
    """
    kv_n = _normalize(kv_dirs, axis=1)   # (n_kv, D)
    w_n = _normalize(w_dirs, axis=0)     # (D, n_w)
    cos = kv_n @ w_n                      # (n_kv, n_w)
    best = np.abs(cos).max(axis=1)        # (n_kv,) — best match per KV dir
    return {
        "mean_best": float(best.mean()),
        "median_best": float(np.median(best)),
        "min_best": float(best.min()),
        "max_best": float(best.max()),
        "std_best": float(best.std()),
        "per_direction": best.tolist(),
    }


def null_alignment(n_kv: int, n_w: int, D: int, n_nulls: int = N_NULLS) -> dict:
    """Expected alignment if KV directions were random."""
    scores = []
    for _ in range(n_nulls):
        rand = np.random.randn(n_kv, D)
        rand = _normalize(rand, axis=1)
        w = np.random.randn(D, n_w)
        w = _normalize(w, axis=0)
        cos = np.abs(rand @ w)
        scores.append(float(cos.max(axis=1).mean()))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
    }


def hadamard_lane_alignment(w: np.ndarray, kv_flat: np.ndarray, block: int = 1024) -> dict:
    """Compare per-lane energy profiles of weight and KV in the Hadamard domain.

    w: (D, hidden) — weight matrix, Hadamard along D (key dim)
    kv_flat: (L, D) — KV, Hadamard along L (seq dim)

    Returns correlation between the two per-lane energy profiles.
    """
    def _hadamard(n):
        H = np.array([[1.0]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H

    H = _hadamard(block)
    seq = (np.diff(H, axis=1) != 0).sum(axis=1)
    order = np.argsort(seq, kind="stable")

    # Weight: Hadamard along D (key dim, axis=0)
    D = w.shape[0]
    n_blocks = (D + block - 1) // block
    w_lane = np.zeros(block)
    for b in range(n_blocks):
        wb = w[b * block:(b + 1) * block]
        if len(wb) < block:
            wb = np.pad(wb, ((0, block - len(wb)), (0, 0)))
        w_lane += ((H @ wb) ** 2).mean(axis=1)
    w_lane /= n_blocks
    w_lane = w_lane[order]

    # KV: Hadamard along L (seq dim, axis=0)
    L = kv_flat.shape[0]
    n_blocks_kv = (L + block - 1) // block
    kv_lane = np.zeros(block)
    for b in range(n_blocks_kv):
        kb = kv_flat[b * block:(b + 1) * block]
        if len(kb) < block:
            kb = np.pad(kb, ((0, block - len(kb)), (0, 0)))
        kv_lane += ((H @ kb) ** 2).mean(axis=1)
    kv_lane /= n_blocks_kv
    kv_lane = kv_lane[order]

    # Correlation between the two profiles
    corr = float(np.corrcoef(w_lane, kv_lane)[0, 1])
    # Energy in low-sequency half for each
    half = block // 2
    w_low = float(w_lane[:half].sum() / w_lane.sum())
    kv_low = float(kv_lane[:half].sum() / kv_lane.sum())

    return {
        "lane_correlation": corr,
        "weight_low_half_energy": w_low,
        "kv_low_half_energy": kv_low,
        "weight_top10_lanes": np.argsort(w_lane)[::-1][:10].tolist(),
        "kv_top10_lanes": np.argsort(kv_lane)[::-1][:10].tolist(),
    }


# ------------------------------------------------------------------ main

def main():
    t_start = time.time()
    print("=" * 64)
    print("Test 12: Alignment — KV directions vs weight directions")
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
    rendered = render_context(turns)
    kv, n_tokens = extract_kv(model, tok, rendered)
    L = n_tokens
    print(f"      {L} positions, {len(kv)} layers")

    print("[3/4] SVD alignment (K and V, all 16 layers)...")
    layer_results = []
    for layer_idx in FULL_ATTENTION_LAYERS:
        w = get_layer_weights(model, layer_idx)
        k, v = kv[layer_idx]
        k_flat = k[0].transpose(1, 0, 2).reshape(L, -1)  # (L, 1024)
        v_flat = v[0].transpose(1, 0, 2).reshape(L, -1)  # (L, 1024)

        # Subsample for SVD speed
        idx = np.random.choice(L, SVD_SUBSAMPLE, replace=False)
        k_sub = k_flat[idx]
        v_sub = v_flat[idx]

        row = {"layer": layer_idx}

        # K alignment: k_proj (1024×320) vs KV keys (4096×1024)
        if "k" in w:
            t0 = time.time()
            U_w, s_w, _ = np.linalg.svd(w["k"], full_matrices=False)
            _, s_kv, Vt_kv = np.linalg.svd(k_sub, full_matrices=False)
            svd_time = time.time() - t0

            # rank90 for each
            cum_w = np.cumsum(s_w ** 2) / (s_w ** 2).sum()
            cum_kv = np.cumsum(s_kv ** 2) / (s_kv ** 2).sum()
            n_w = int(np.searchsorted(cum_w, 0.90)) + 1
            n_kv = int(np.searchsorted(cum_kv, 0.90)) + 1

            W_dirs = U_w[:, :n_w]          # (1024, n_w)
            K_dirs = Vt_kv[:n_kv, :]       # (n_kv, 1024)

            row["k"] = {
                "n_w": n_w, "n_kv": n_kv,
                "svd_time_s": round(svd_time, 2),
                "alignment": alignment_score(K_dirs, W_dirs),
            }

            # Hadamard lane alignment
            row["k"]["hadamard"] = hadamard_lane_alignment(w["k"], k_flat)

        # V alignment: v_proj (1024×320) vs KV values (4096×1024)
        if "v" in w:
            t0 = time.time()
            U_wv, s_wv, _ = np.linalg.svd(w["v"], full_matrices=False)
            _, s_kvv, Vt_kvv = np.linalg.svd(v_sub, full_matrices=False)
            svd_time = time.time() - t0

            cum_wv = np.cumsum(s_wv ** 2) / (s_wv ** 2).sum()
            cum_kvv = np.cumsum(s_kvv ** 2) / (s_kvv ** 2).sum()
            n_wv = int(np.searchsorted(cum_wv, 0.90)) + 1
            n_kvv = int(np.searchsorted(cum_kvv, 0.90)) + 1

            Wv_dirs = U_wv[:, :n_wv]
            Kv_dirs = Vt_kvv[:n_kvv, :]

            row["v"] = {
                "n_w": n_wv, "n_kv": n_kvv,
                "svd_time_s": round(svd_time, 2),
                "alignment": alignment_score(Kv_dirs, Wv_dirs),
            }

        layer_results.append(row)
        k_align = row.get("k", {}).get("alignment", {})
        v_align = row.get("v", {}).get("alignment", {})
        print(f"      layer {layer_idx:>2}: K best={k_align.get('mean_best', 0):.3f}  "
              f"V best={v_align.get('mean_best', 0):.3f}  "
              f"(K: {row.get('k', {}).get('n_kv', 0)}/{row.get('k', {}).get('n_w', 0)}  "
              f"V: {row.get('v', {}).get('n_kv', 0)}/{row.get('v', {}).get('n_w', 0)})")

    print("[4/4] Null alignment (random directions)...")
    # Use the mean n_kv and n_w from the layers
    mean_n_kv = int(np.mean([r.get("k", {}).get("n_kv", 59) for r in layer_results]))
    mean_n_w = int(np.mean([r.get("k", {}).get("n_w", 123) for r in layer_results]))
    D = 1024
    null = null_alignment(mean_n_kv, mean_n_w, D)
    print(f"      null mean={null['mean']:.4f}  std={null['std']:.4f}  max={null['max']:.4f}")

    # Save
    results = {
        "test": 12,
        "name": "kv_weight_alignment",
        "total_tokens": L,
        "layers": layer_results,
        "null": null,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out = BASELINE_DIR / "results" / "test_12_alignment.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    # Summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)

    k_means = [r["k"]["alignment"]["mean_best"] for r in layer_results if "k" in r]
    v_means = [r["v"]["alignment"]["mean_best"] for r in layer_results if "v" in r]
    k_had = [r["k"]["hadamard"]["lane_correlation"] for r in layer_results if "k" in r]

    print(f"\nSVD alignment (mean best cosine per KV direction):")
    print(f"  K: mean={np.mean(k_means):.4f}  min={min(k_means):.4f}  max={max(k_means):.4f}")
    print(f"  V: mean={np.mean(v_means):.4f}  min={min(v_means):.4f}  max={max(v_means):.4f}")
    print(f"  Null (random): {null['mean']:.4f} ± {null['std']:.4f}")

    print(f"\nHadamard lane correlation (weight vs KV per-lane energy):")
    print(f"  K: mean={np.mean(k_had):.4f}")

    # Verdict
    print(f"\n{'=' * 64}")
    print("VERDICT")
    print(f"{'=' * 64}")
    k_mean = np.mean(k_means)
    null_mean = null["mean"]
    z_score = (k_mean - null_mean) / null["std"] if null["std"] > 0 else 0

    if z_score > 5:
        print(f"  ALIGNED. KV directions match weight directions (z={z_score:.1f}).")
        print(f"  The context IS a modulation of the weight field.")
        print(f"  → The gravitational relationship is CONFIRMED.")
    elif z_score > 2:
        print(f"  PARTIALLY ALIGNED (z={z_score:.1f}).")
        print(f"  The context partially falls into the weight's lanes.")
        print(f"  → The bet is alive but needs refinement.")
    else:
        print(f"  NOT ALIGNED (z={z_score:.1f}).")
        print(f"  The KV's dominant directions are unrelated to the weight's.")
        print(f"  → The 'gravitational relationship' is NOT confirmed.")
        print(f"  Fall back to the 2x spectral compression (Test 9/10).")

    print(f"\nTotal elapsed: {results['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
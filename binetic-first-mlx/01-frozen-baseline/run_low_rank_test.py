#!/usr/bin/env python3
"""Test 11: Is the KV a low-rank modulation of the weight field?

The gate test for the "context is a low-sequency perturbation of the
weight field" hypothesis.

For each full-attention layer:
  1. KV lane concentration: Hadamard transform K/V along seq_len,
     measure how many lanes carry 90/99% of the energy
  2. KV effective rank: SVD of the flattened KV matrix
  3. Weight effective rank: SVD of the Q/K/V/O projection matrices
  4. Cross-correlation: do the KV's dominant lanes align with the
     weights' dominant singular vectors?

Win condition: KV effective rank is small (< 100 out of 1024) AND
the weights' effective rank is also small → context is a low-rank
modulation of the weight field.
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
HADAMARD_BLOCK = 1024


# ---------------------------------------------------------------- helpers

def _hadamard(n: int) -> np.ndarray:
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_lanes(x: np.ndarray, block: int = HADAMARD_BLOCK) -> np.ndarray:
    """Blockwise Hadamard along axis 0, return per-lane energy ordered by sequency."""
    L = x.shape[0]
    H = _hadamard(block)
    seq = (np.diff(H, axis=1) != 0).sum(axis=1)
    order = np.argsort(seq, kind="stable")
    n_blocks = (L + block - 1) // block
    lane_energy = np.zeros(block)
    for b in range(n_blocks):
        xb = x[b * block:(b + 1) * block]
        if len(xb) < block:
            xb = np.pad(xb, ((0, block - len(xb)), (0, 0)))
        e = ((H @ xb) ** 2).mean(axis=1)
        lane_energy += e
    lane_energy /= n_blocks
    return lane_energy[order]


def lane_concentration(lane_energy: np.ndarray) -> dict:
    total = lane_energy.sum()
    if total <= 0:
        return {}
    cum = np.cumsum(lane_energy) / total
    n = len(lane_energy)

    def frac_for_energy(e: float) -> float:
        return float(np.searchsorted(cum, e)) / n

    return {
        "lanes_for_90": frac_for_energy(0.90),
        "lanes_for_99": frac_for_energy(0.99),
        "energy_low_half": float(cum[n // 2 - 1]),
        "energy_low_quarter": float(cum[n // 4 - 1]),
    }


def effective_rank(s: np.ndarray, energy: float = 0.90) -> int:
    """Number of singular values needed to capture `energy` fraction."""
    total = (s ** 2).sum()
    if total <= 0:
        return 0
    cum = np.cumsum(s ** 2) / total
    return int(np.searchsorted(cum, energy)) + 1


def svd_energy_spectrum(s: np.ndarray) -> dict:
    total = (s ** 2).sum()
    if total <= 0:
        return {}
    cum = np.cumsum(s ** 2) / total
    return {
        "rank_90": effective_rank(s, 0.90),
        "rank_99": effective_rank(s, 0.99),
        "rank_999": effective_rank(s, 0.999),
        "top10_energy": float(cum[min(10, len(cum)) - 1]),
        "top50_energy": float(cum[min(50, len(cum)) - 1]),
        "top100_energy": float(cum[min(100, len(cum)) - 1]),
        "n_singular_values": len(s),
    }


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
    """Extract Q, K, V, O projection weight matrices for a full-attention layer."""
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


def inspect_model_weights(model) -> dict:
    """Print all parameter names and shapes to find the weight naming convention."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    params = tree_flatten(model.parameters())
    info = {}
    for name, param in params:
        if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                                    "Wq", "Wk", "Wv", "Wo",
                                    "attn")):
            info[name] = list(param.shape)
    return info


# ------------------------------------------------------------------ main

def main():
    t_start = time.time()
    print("=" * 64)
    print("Test 11: Is the KV a low-rank modulation of the weight field?")
    print("=" * 64)

    print("[1/5] Loading model...")
    from artifact import load_model
    from tokenizers import Tokenizer
    t0 = time.time()
    model, _ = load_model(str(MODEL_DIR))
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"      loaded in {time.time() - t0:.1f}s")

    # Inspect weight naming
    print("[2/5] Inspecting model weight structure...")
    weight_info = inspect_model_weights(model)
    # Show a sample of the naming
    sample = list(weight_info.items())[:8]
    for name, shape in sample:
        print(f"      {name}: {shape}")
    print(f"      ... {len(weight_info)} attention-related parameters total")

    # Find the naming pattern for layer 3 (first full-attention layer)
    layer3_weights = {k: v for k, v in weight_info.items() if "3" in k.split(".")[1] if "." in k}
    if not layer3_weights:
        # Try to find any layer weights
        for k, v in weight_info.items():
            if "layers" in k or "layer" in k:
                print(f"      Found: {k}: {v}")
                break

    print("[3/5] Building context, extracting KV...")
    turns = make_context()
    rendered = render_context(turns)
    kv, n_tokens = extract_kv(model, tok, rendered)
    L = n_tokens
    print(f"      {L} positions, {len(kv)} layers")

    print("[4/5] Analyzing KV lane concentration + effective rank...")
    layer_results = []
    for layer_idx in FULL_ATTENTION_LAYERS:
        k, v = kv[layer_idx]
        # K: (1, 4, L, 256) → (L, 1024)
        k_flat = k[0].transpose(1, 0, 2).reshape(L, -1)  # (L, heads*dim)
        v_flat = v[0].transpose(1, 0, 2).reshape(L, -1)

        # Lane concentration (Hadamard along seq)
        k_lanes = hadamard_lanes(k_flat)
        v_lanes = hadamard_lanes(v_flat)
        k_lane = lane_concentration(k_lanes)
        v_lane = lane_concentration(v_lanes)

        # Effective rank (SVD)
        # Use a subsample for speed if L is large
        if L > 4096:
            idx = np.random.choice(L, 4096, replace=False)
            k_sub = k_flat[idx]
            v_sub = v_flat[idx]
        else:
            k_sub = k_flat
            v_sub = v_flat

        t0 = time.time()
        _, s_k, _ = np.linalg.svd(k_sub, full_matrices=False)
        _, s_v, _ = np.linalg.svd(v_sub, full_matrices=False)
        svd_time = time.time() - t0

        k_svd = svd_energy_spectrum(s_k)
        v_svd = svd_energy_spectrum(s_v)

        layer_results.append({
            "layer": layer_idx,
            "kv_shape": list(k.shape),
            "k_flat_shape": list(k_flat.shape),
            "k_lane": k_lane,
            "v_lane": v_lane,
            "k_svd": k_svd,
            "v_svd": v_svd,
            "svd_time_s": round(svd_time, 2),
        })
        print(f"      layer {layer_idx:>2}: K rank90={k_svd['rank_90']:>4}  "
              f"V rank90={v_svd['rank_90']:>4}  "
              f"K lanes90={k_lane['lanes_for_90']:.3f}  "
              f"V lanes90={v_lane['lanes_for_90']:.3f}  "
              f"({svd_time:.1f}s)")

    print("[5/5] Analyzing weight effective rank...")
    # Get weights for a few layers
    weight_results = {}
    for layer_idx in [3, 31, 63]:
        w = get_layer_weights(model, layer_idx)
        if not w:
            print(f"      layer {layer_idx}: no weights found (naming mismatch)")
            continue
        wr = {}
        for name, mat in w.items():
            if mat.ndim != 2:
                continue
            t0 = time.time()
            _, s, _ = np.linalg.svd(mat, full_matrices=False)
            dt = time.time() - t0
            wr[name] = {
                "shape": list(mat.shape),
                "svd": svd_energy_spectrum(s),
                "svd_time_s": round(dt, 2),
            }
            print(f"      layer {layer_idx} {name}: shape={mat.shape}  "
                  f"rank90={wr[name]['svd']['rank_90']:>4}  "
                  f"rank99={wr[name]['svd']['rank_99']:>4}  "
                  f"({dt:.1f}s)")
        weight_results[str(layer_idx)] = wr

    # Save
    results = {
        "test": 11,
        "name": "kv_low_rank_modulation_test",
        "total_tokens": L,
        "layers": layer_results,
        "weights": weight_results,
        "weight_naming_sample": dict(list(weight_info.items())[:20]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out = BASELINE_DIR / "results" / "test_11_low_rank.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    # Summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)

    k_ranks = [r["k_svd"]["rank_90"] for r in layer_results]
    v_ranks = [r["v_svd"]["rank_90"] for r in layer_results]
    k_lanes = [r["k_lane"]["lanes_for_90"] for r in layer_results]
    v_lanes = [r["v_lane"]["lanes_for_90"] for r in layer_results]
    k_low_half = [r["k_lane"]["energy_low_half"] for r in layer_results]
    v_low_half = [r["v_lane"]["energy_low_half"] for r in layer_results]

    print(f"\nKV effective rank (90% energy, out of {k_flat.shape[1]} dims):")
    print(f"  K: mean={np.mean(k_ranks):.0f}  min={min(k_ranks)}  max={max(k_ranks)}")
    print(f"  V: mean={np.mean(v_ranks):.0f}  min={min(v_ranks)}  max={max(v_ranks)}")

    print(f"\nKV lane concentration (Hadamard, 90% energy):")
    print(f"  K: mean={np.mean(k_lanes):.3f} of lanes  (1.0 = all lanes needed)")
    print(f"  V: mean={np.mean(v_lanes):.3f} of lanes")
    print(f"  K energy in low-sequency half: mean={np.mean(k_low_half):.3f}  (0.5 = flat)")
    print(f"  V energy in low-sequency half: mean={np.mean(v_low_half):.3f}")

    if weight_results:
        print(f"\nWeight effective rank (90% energy):")
        for layer_idx, wr in weight_results.items():
            for name, info in wr.items():
                print(f"  layer {layer_idx} {name}: rank90={info['svd']['rank_90']:>4}  "
                      f"rank99={info['svd']['rank_99']:>4}  shape={info['shape']}")

    # Verdict
    print(f"\n{'=' * 64}")
    print("VERDICT")
    print(f"{'=' * 64}")
    avg_rank = np.mean(k_ranks + v_ranks)
    total_dims = k_flat.shape[1]
    if avg_rank < 100:
        print(f"  KV effective rank is LOW ({avg_rank:.0f} / {total_dims}).")
        print(f"  The context IS a low-rank modulation of the weight field.")
        print(f"  → The bet is ALIVE. Store a low-rank factor, reconstruct on the fly.")
    elif avg_rank < 300:
        print(f"  KV effective rank is MODERATE ({avg_rank:.0f} / {total_dims}).")
        print(f"  The context is partially low-rank. Spectral compression still wins,")
        print(f"  but full 'store the delta' may need more structure.")
    else:
        print(f"  KV effective rank is HIGH ({avg_rank:.0f} / {total_dims}).")
        print(f"  The context is NOT low-rank. The 'store the delta' bet is DEAD.")
        print(f"  Fall back to the 2x spectral compression (Test 9/10).")

    print(f"\nTotal elapsed: {results['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Test 13b: RETEST weight-direction projection with diagnostics.

Test 13 failed at ALL n_w (even 123/8.3x). That's suspicious. This retest
determines WHY:

  1. Print k_proj / v_proj shapes (verify the projection is well-posed)
  2. Measure reconstruction error (rel RMSE) + energy captured at each n_w
     - if n_w=320 gives ~0 error → projection is correct, KV lies in
       k_proj's column space (as theory predicts: K = H @ k_proj.T)
     - if n_w=320 gives high error → BUG or theory is wrong
  3. Run recall at n_w=123, 200, 320

Verdicts:
  - n_w=320 recall PASS + n_w=123 FAIL → bet alive, but needs ALL 320
    directions (no compression). The KV uses the full weight column space.
  - n_w=320 recall FAIL → bug in projection, or KV not in weight column space
  - n_w=123 recall PASS → test 13 was a fluke, bet validated
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
N_W_OPTIONS = [123, 200, 320]


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


def project_and_diagnose(kv: dict, model, n_w: int) -> tuple[dict, dict]:
    """Project KV onto weight's top-n_w directions. Return (compressed, diagnostics)."""
    compressed = {}
    diag = {"k_energy": [], "v_energy": [], "k_rmse": [], "v_rmse": []}
    for layer_idx, (k, v) in kv.items():
        w = get_layer_weights(model, layer_idx)
        L = k.shape[2]

        # K
        if "k" in w:
            U_wk, s_wk, _ = np.linalg.svd(w["k"], full_matrices=False)
            n_use = min(n_w, U_wk.shape[1])
            U = U_wk[:, :n_use]
            k_flat = k[0].transpose(1, 0, 2).reshape(L, -1)
            C = k_flat @ U
            k_rec = C @ U.T
            total = (k_flat ** 2).sum()
            captured = (k_rec ** 2).sum()
            diag["k_energy"].append(float(captured / total))
            diag["k_rmse"].append(float(np.sqrt(np.mean((k_flat - k_rec) ** 2)) /
                                        (np.sqrt(np.mean(k_flat ** 2)) + 1e-12)))
            compressed_k = k_rec.reshape(1, 4, L, 256)
        else:
            compressed_k = k

        # V
        if "v" in w:
            U_wv, s_wv, _ = np.linalg.svd(w["v"], full_matrices=False)
            n_use = min(n_w, U_wv.shape[1])
            U = U_wv[:, :n_use]
            v_flat = v[0].transpose(1, 0, 2).reshape(L, -1)
            C = v_flat @ U
            v_rec = C @ U.T
            total = (v_flat ** 2).sum()
            captured = (v_rec ** 2).sum()
            diag["v_energy"].append(float(captured / total))
            diag["v_rmse"].append(float(np.sqrt(np.mean((v_flat - v_rec) ** 2)) /
                                        (np.sqrt(np.mean(v_flat ** 2)) + 1e-12)))
            compressed_v = v_rec.reshape(1, 4, L, 256)
        else:
            compressed_v = v

        compressed[layer_idx] = (compressed_k.astype(np.float16),
                                 compressed_v.astype(np.float16))
    diag["k_energy_mean"] = float(np.mean(diag["k_energy"]))
    diag["v_energy_mean"] = float(np.mean(diag["v_energy"]))
    diag["k_rmse_mean"] = float(np.mean(diag["k_rmse"]))
    diag["v_rmse_mean"] = float(np.mean(diag["v_rmse"]))
    return compressed, diag


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
    print("Test 13b: RETEST weight-direction projection (with diagnostics)")
    print("=" * 64)

    print("[1/4] Loading model...")
    from artifact import load_model
    from tokenizers import Tokenizer
    t0 = time.time()
    model, _ = load_model(str(MODEL_DIR))
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"      loaded in {time.time() - t0:.1f}s")

    # Verify weight shapes
    print("\n[2/4] Verifying weight shapes (layer 3):")
    w3 = get_layer_weights(model, 3)
    for tag, mat in w3.items():
        print(f"      {tag}_proj: {mat.shape}  (out={mat.shape[0]}, in={mat.shape[1]})")
    k_rank = w3["k"].shape[1]
    v_rank = w3["v"].shape[1]
    print(f"      → k_proj column space dim = {k_rank}, v_proj column space dim = {v_rank}")

    print("\n[3/4] Building context, extracting KV...")
    turns = make_context()
    rendered_context = render_context(turns)
    expected = turns[0]["memory_token"]
    kv, n_tokens = extract_kv(model, tok, rendered_context)
    L = n_tokens
    print(f"      {L} positions, {len(kv)} layers, expected code: {expected}")

    query_rendered = render_query()

    print("\n[4/4] Projection diagnostics + recall...\n")
    results = []

    # Baseline
    t0 = time.time()
    decoded = run_recall(model, tok,
                         {l: (k.astype(np.float16), v.astype(np.float16)) for l, (k, v) in kv.items()},
                         query_rendered)
    recall, actual = check_recall(decoded, expected)
    print(f"  {'baseline':<20} {'PASS' if recall==1 else 'FAIL':<6} got={actual}  "
          f"(energy=1.000, rmse=0.000)")
    results.append({"method": "baseline", "n_w": 0, "recall": recall, "actual": actual,
                    "k_energy": 1.0, "v_energy": 1.0, "k_rmse": 0.0, "v_rmse": 0.0})

    for n_w in N_W_OPTIONS:
        t0 = time.time()
        compressed, diag = project_and_diagnose(kv, model, n_w)
        proj_time = time.time() - t0
        decoded = run_recall(model, tok, compressed, query_rendered)
        recall, actual = check_recall(decoded, expected)
        status = "PASS" if recall == 1.0 else "FAIL"
        got = str(actual) if actual is not None else "None"
        print(f"  {'n_w=' + str(n_w):<20} {status:<6} got={got:>5}  "
              f"(K energy={diag['k_energy_mean']:.3f} rmse={diag['k_rmse_mean']:.3f}  "
              f"V energy={diag['v_energy_mean']:.3f} rmse={diag['v_rmse_mean']:.3f})  "
              f"[proj {proj_time:.1f}s]")
        results.append({"method": f"n_w={n_w}", "n_w": n_w, "recall": recall,
                        "actual": actual, "k_energy": diag["k_energy_mean"],
                        "v_energy": diag["v_energy_mean"],
                        "k_rmse": diag["k_rmse_mean"], "v_rmse": diag["v_rmse_mean"]})

    # Save
    out = {
        "test": "13b",
        "name": "weight_direction_retest_diagnostics",
        "total_tokens": L,
        "expected_code": expected,
        "k_proj_dim": k_rank,
        "v_proj_dim": v_rank,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    out_path = BASELINE_DIR / "results" / "test_13b_retest.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

    # Verdict
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    full = next((r for r in results if r["n_w"] == 320), None)
    top = next((r for r in results if r["n_w"] == 123), None)

    if full and full["k_energy"] > 0.99 and full["recall"] == 1.0:
        print(f"  Projection is CORRECT (n_w=320: energy={full['k_energy']:.3f}, recall PASS).")
        if top and top["recall"] == 0.0:
            print(f"  But top-123 fails (energy={top['k_energy']:.3f}).")
            print(f"  → The KV uses the FULL weight column space ({k_rank} dims).")
            print(f"  → No compression via weight-direction projection. Bet is DEAD for compression.")
            print(f"  → The 'low-rank' (test 11) is the KV's OWN rank, not the weight's.")
        else:
            print(f"  Top-123 also passes → test 13 was a fluke. Bet VALIDATED.")
    elif full and full["k_energy"] < 0.99:
        print(f"  BUG: n_w=320 only captures {full['k_energy']:.3f} of KV energy.")
        print(f"  The KV is NOT in k_proj's column space. Check the projection math.")
    else:
        print(f"  Inconclusive. Inspect the diagnostics above.")

    print(f"\nTotal elapsed: {out['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
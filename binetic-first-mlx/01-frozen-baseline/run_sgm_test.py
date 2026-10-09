"""SGM — Salience-Gated Selective KV Memory.

First-principles premise:
  The context is NOT uniform. ~99% is boilerplate, ~1% is the actual fact
  ("the secret code is 8270"). Uniform KV compression (spectral SVD, positional
  subsampling) wastes the bit-budget on padding and starves the fact.

  The model computes a FREE salience signal during the forward pass: the
  per-token SURPRISAL  s_t = -log p(t_t | t_<t).
    - boilerplate / redundant token  -> model predicted it -> surprisal ~ 0
    - surprising fact (a code in a sea of padding) -> surprisal is large

  SGM keeps the EXACT full-attention KV only for the top-k highest-surprisal
  tokens, dumps the rest, and re-injects only the salient KV at query time.

Design (same synthetic secret-code task as the recurrent-state test):
  Sanity:  in-place (all KV)                          -> expect 100%
  THE BET: surprisal top-k                            -> does the model's own
                                                          surprise find the fact?
  Controls (same k): random-k, first-k, last-k        -> isolates whether it's
                                                          the SELECTION RULE, not
                                                          just "having k tokens"

  We sweep k in {16, 64, 256}. One context forward pass; each variant only
  re-runs the short query (KV subsets are sliced from the already-computed
  full cache, so selection is free).

Diagnostics (so a 0% is still informative):
  - full surprisal curve (downsampled)
  - top-k selected positions + their decoded tokens (did we grab "8270"?)
  - the surprisal rank of the actual code tokens

Usage:
  /opt/homebrew/bin/python3.12 run_sgm_test.py
  /opt/homebrew/bin/python3.12 run_sgm_test.py --ks 16 64 256
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parent
BONSAI_ROOT = BASELINE_DIR.parent
PACK_DIR = BONSAI_ROOT / "bonsai2-27b-mlx"
sys.path.insert(0, str(PACK_DIR / "runtime"))

EOS = 248044
FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]


def load_model_and_tokenizer():
    from artifact import load_model
    from tokenizers import Tokenizer

    model, config = load_model(str(PACK_DIR))
    tok = Tokenizer.from_file(str(PACK_DIR / "tokenizer.json"))
    return model, config, tok


def make_synthetic_recall_data(num_turns: int, tokens_per_turn: int) -> list[dict]:
    """Same synthetic dialogue as run_single_test.py (seed 42, same padding)."""
    np.random.seed(42)
    turns = []
    padding_words = ["the", "system", "configures", "values", "during", "initialization",
                     "process", "and", "maintains", "state", "across", "multiple",
                     "requests", "with", "proper", "handling", "of", "edge", "cases",
                     "including", "timeout", "recovery", "retry", "logic", "cache"]
    for t in range(num_turns):
        memory_value = int(np.random.randint(1000, 9999))
        padding = " ".join(np.random.choice(padding_words, size=max(1, tokens_per_turn // 2)))
        prompt = f"Turn {t}: The secret code is {memory_value}. System log: {padding}."
        turns.append({"turn": t, "memory_token": memory_value, "prompt": prompt})
    return turns


def render_chat(template, text):
    return template.render(
        messages=[{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def compute_surprisal(logits, token_ids):
    """Per-token surprisal s_t = -log p(t_t | t_<t).

    logits: (1, n, vocab) — logits[i] predicts token i+1.
    Returns np array of length n; surprisal[0] = 0 (first token, no context).
    """
    import mlx.core as mx
    import mlx.nn as mnn

    log_probs = mnn.log_softmax(logits, axis=-1)          # (1, n, vocab)
    pred = log_probs[:, :-1, :]                            # (1, n-1, vocab)
    target = mx.array([token_ids[1:]], dtype=mx.int32)     # (1, n-1)
    gathered = mx.take_along_axis(pred, target[:, :, None], axis=-1)  # (1, n-1, 1)
    s = np.asarray(gathered[0, :, 0], dtype=np.float32)               # (n-1,)
    out = np.zeros(len(token_ids), dtype=np.float32)
    out[1:] = -s
    return out


def select_positions(surprisal, k, strategy, n, rng):
    """Return sorted list of k positions to keep, per strategy."""
    if strategy == "surprisal":
        # top-k by surprisal (position 0 has surprisal 0, rarely selected)
        idx = np.argsort(-surprisal)[:k]
    elif strategy == "random":
        idx = rng.choice(n, size=k, replace=False)
    elif strategy == "first":
        idx = np.arange(min(k, n))
    elif strategy == "last":
        idx = np.arange(max(0, n - k), n)
    else:
        raise ValueError(strategy)
    return sorted(int(i) for i in idx)


def extract_kv_at_positions(cache, positions):
    """Slice the full-attention KV at the given positions -> {layer: (k, v)}."""
    import mlx.core as mx

    sel = mx.array(positions, dtype=mx.int32)
    out = {}
    for layer_idx in FULL_ATTENTION_LAYERS:
        c = cache[layer_idx]
        if c.keys is None:
            continue
        k, v = c.state  # (1, heads, n, dim)
        k_sel = mx.take(k, sel, axis=2)
        v_sel = mx.take(v, sel, axis=2)
        out[layer_idx] = (np.asarray(k_sel).astype(np.float16),
                          np.asarray(v_sel).astype(np.float16))
    return out


def build_cache_with_kv(model, kv_by_layer):
    """Fresh cache with the given KV injected into full-attention layers."""
    import mlx.core as mx

    cache = model.make_cache()
    for layer_idx, (keys, values) in kv_by_layer.items():
        c = cache[layer_idx]
        c.state = (mx.array(keys), mx.array(values))
        mx.eval(c.keys, c.values)
    return cache


def generate(model, tok, cache, query_ids, max_tokens=32):
    import mlx.core as mx

    x = mx.array([query_ids], dtype=mx.int32)
    logits = model(x, cache=cache)
    mx.eval(logits)
    gen = []
    for _ in range(max_tokens):
        nxt = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        gen.append(nxt)
        if nxt == EOS:
            break
        x = mx.array([[nxt]], dtype=mx.int32)
        logits = model(x, cache=cache)
        mx.eval(logits)
    return tok.decode(gen)


def check_recall(decoded, expected):
    numbers = re.findall(r"\b(\d{4})\b", decoded)
    actual = int(numbers[0]) if numbers else None
    return (actual == expected), actual


def main():
    import mlx.core as mx
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-turns", type=int, default=10)
    ap.add_argument("--tokens-per-turn", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 64, 256])
    args = ap.parse_args()

    print("[sgm] loading model...")
    model, config, tok = load_model_and_tokenizer()

    template_path = PACK_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())

    turns = make_synthetic_recall_data(args.num_turns, args.tokens_per_turn)
    full_prompt = " ".join(t["prompt"] for t in turns)
    query_text = "What was the secret code from turn 0? Just the number."
    expected = turns[0]["memory_token"]

    context_ids = tok.encode(render_chat(template, full_prompt)).ids
    query_ids = tok.encode(render_chat(template, query_text)).ids
    n = len(context_ids)
    print(f"[sgm] {args.num_turns} turns, context={n} tok, query={len(query_ids)} tok, "
          f"expected={expected}, ks={args.ks}")

    # ---- ONE context forward pass: get logits + full KV cache ----
    t0 = time.time()
    cache_full = model.make_cache()
    x = mx.array([context_ids], dtype=mx.int32)
    logits = model(x, cache=cache_full)
    mx.eval(logits)
    ctx_time = time.time() - t0
    print(f"[sgm] context forward: {ctx_time:.1f}s")

    surprisal = compute_surprisal(logits, context_ids)
    print(f"[sgm] surprisal: mean={surprisal.mean():.3f} max={surprisal.max():.3f} "
          f"p99={np.percentile(surprisal, 99):.3f}")

    # Diagnostic: where do the code tokens sit in the surprisal ranking?
    # Find token positions whose decoded form contains the expected code.
    code_ranks = []
    for pos in range(n):
        dec = tok.decode([context_ids[pos]])
        if str(expected) in dec:
            rank = int((surprisal > surprisal[pos]).sum()) + 1
            code_ranks.append({"pos": pos, "token": dec, "surprisal": float(surprisal[pos]),
                               "rank": rank, "of": n})
    print(f"[sgm] code-token positions (value {expected}): {code_ranks}")

    # Downsampled surprisal curve for the record
    step = max(1, n // 200)
    curve = {"positions": list(range(0, n, step)),
             "surprisal": [float(surprisal[i]) for i in range(0, n, step)]}

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_sgm_result.json"
    prior = {}
    if out_path.is_file():
        try:
            prior = json.loads(out_path.read_text()).get("variants", {})
        except Exception:
            prior = {}

    rng = np.random.default_rng(0)
    variants = []

    # Sanity: in-place (all KV)
    if "inplace" not in prior:
        print("\n[sgm] === sanity: in-place (all KV) ===")
        decoded = generate(model, tok, cache_full, query_ids)
        correct, actual = check_recall(decoded, expected)
        r = {"strategy": "inplace", "k": n, "recall_correct": correct,
             "recall_accuracy": 1.0 if correct else 0.0, "expected": expected,
             "actual": actual, "text": decoded[:100]}
        prior["inplace"] = r
        variants.append(r)
        print(f"  recall={correct} actual={actual} text={decoded[:60]!r}")

    # Sweep: strategies x k
    for k in args.ks:
        for strategy in ("surprisal", "random", "first", "last"):
            key = f"{strategy}_k{k}"
            if key in prior:
                continue
            print(f"\n[sgm] === {strategy} k={k} ===")
            positions = select_positions(surprisal, k, strategy, n, rng)
            kv = extract_kv_at_positions(cache_full, positions)
            cache_sel = build_cache_with_kv(model, kv)
            decoded = generate(model, tok, cache_sel, query_ids)
            correct, actual = check_recall(decoded, expected)
            # which of the selected positions are code tokens?
            sel_set = set(positions)
            code_in_sel = [c for c in code_ranks if c["pos"] in sel_set]
            r = {"strategy": strategy, "k": k, "recall_correct": correct,
                 "recall_accuracy": 1.0 if correct else 0.0, "expected": expected,
                 "actual": actual, "text": decoded[:100],
                 "code_tokens_in_selection": code_in_sel,
                 "top_selected_tokens": [tok.decode([context_ids[p]]) for p in positions[:12]]}
            prior[key] = r
            variants.append(r)
            print(f"  recall={correct} actual={actual} code_in_sel={len(code_in_sel)} "
                  f"text={decoded[:50]!r}")
            # checkpoint
            out_path.write_text(json.dumps({
                "test": "sgm", "name": "salience_gated_selective_kv",
                "num_turns": args.num_turns, "context_tokens": n,
                "expected": expected, "surprisal_curve": curve,
                "code_token_positions": code_ranks,
                "variants": prior,
                "complete": False,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2))

    # Verdict
    print("\n[sgm] === VERDICT ===")
    print(f"  {'strategy':<12} " + " ".join(f"k={k:<5}" for k in args.ks))
    for strategy in ("surprisal", "random", "first", "last"):
        row = [f"{prior.get(f'{strategy}_k{k}', {}).get('recall_accuracy', float('nan')):.0%}"
               for k in args.ks]
        print(f"  {strategy:<12} " + " ".join(f"{v:<6}" for v in row))

    # Is the surprisal rule better than the controls at the same k?
    best_k = args.ks[-1]
    s_acc = prior.get(f"surprisal_k{best_k}", {}).get("recall_accuracy", 0.0)
    ctrl_accs = [prior.get(f"{c}_k{best_k}", {}).get("recall_accuracy", 0.0)
                 for c in ("random", "first", "last")]
    if s_acc >= 0.75 and max(ctrl_accs) < 0.25:
        verdict = ("SURPRISAL IS THE SIGNAL — the model's own surprise selects the "
                   "fact; controls at the same k fail")
    elif s_acc >= 0.75:
        verdict = "surprisal works but controls also work — selection rule not the differentiator"
    else:
        verdict = ("surprisal selection insufficient at this k — inspect code_token_positions "
                   "and surprisal_curve to find the right mechanism")
    print(f"  -> {verdict}")

    out_path.write_text(json.dumps({
        "test": "sgm", "name": "salience_gated_selective_kv",
        "num_turns": args.num_turns, "context_tokens": n,
        "expected": expected, "surprisal_curve": curve,
        "code_token_positions": code_ranks,
        "variants": prior, "verdict": verdict, "complete": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"\n[sgm] wrote {out_path}")


if __name__ == "__main__":
    main()
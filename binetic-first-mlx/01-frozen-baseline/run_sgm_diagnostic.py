"""SGM diagnostic — isolate WHY the first SGM run failed.

The first SGM run had two confounds:
  1. Target fact was at turn 0 (the START of context) -> `first` selection
     trivially won (100%), so the test measured position, not salience.
  2. The 4-digit code is MULTI-TOKEN -> per-token "8270" search found nothing,
     so we never learned where the fact actually sits in the surprisal ranking.

This diagnostic fixes both:
  - Target fact is at the MIDDLE turn (turn 5 of 10), so `first`/`last`
    position controls should now FAIL. Only a correct signal should win.
  - Uses the tokenizer's offset mapping to find the EXACT token span of the
    code (multi-token), so we can measure its surprisal rank precisely.

Two questions, cleanly separated:
  (A) MECHANISM — is selective-KV re-injection sound?
      ORACLE: select exactly the code token span (and a ±window variant).
      If recall works with a handful of tokens, the mechanism is sound.
  (B) SIGNAL — is surprisal the right selector?
      Measure the surprisal RANK of each code token. If the code tokens are
      NOT in the top-k, surprisal is the wrong signal (and we see what
      actually dominates the top-k).

Usage:
  /opt/homebrew/bin/python3.12 run_sgm_diagnostic.py
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


def make_turns(num_turns: int, tokens_per_turn: int, target_turn: int) -> list[dict]:
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
    import mlx.core as mx
    import mlx.nn as mnn

    log_probs = mnn.log_softmax(logits, axis=-1)
    pred = log_probs[:, :-1, :]
    target = mx.array([token_ids[1:]], dtype=mx.int32)
    gathered = mx.take_along_axis(pred, target[:, :, None], axis=-1)
    s = np.asarray(gathered[0, :, 0], dtype=np.float32)
    out = np.zeros(len(token_ids), dtype=np.float32)
    out[1:] = -s
    return out


def find_code_span(rendered: str, code: int, tok):
    """Find the token indices covering the code string, via offset mapping.

    Uses the SAME default encoding as the forward pass so indices align with
    context_ids. The Encoding.offsets attribute is always available.
    """
    enc = tok.encode(rendered)
    ids, offsets = enc.ids, enc.offsets
    code_str = str(code)
    pos = rendered.find(code_str)
    if pos < 0:
        return None, ids, offsets
    end = pos + len(code_str)
    span = [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]
    return span, ids, offsets


def extract_kv_at_positions(cache, positions):
    import mlx.core as mx
    sel = mx.array(sorted(set(int(p) for p in positions)), dtype=mx.int32)
    out = {}
    for layer_idx in FULL_ATTENTION_LAYERS:
        c = cache[layer_idx]
        if c.keys is None:
            continue
        k, v = c.state
        out[layer_idx] = (np.asarray(mx.take(k, sel, axis=2)).astype(np.float16),
                          np.asarray(mx.take(v, sel, axis=2)).astype(np.float16))
    return out


def build_cache_with_kv(model, kv_by_layer):
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
    ap.add_argument("--target-turn", type=int, default=5, help="which turn holds the target fact")
    ap.add_argument("--window", type=int, default=12, help="context window around code for oracle")
    args = ap.parse_args()

    print("[diag] loading model...")
    model, config, tok = load_model_and_tokenizer()

    template_path = PACK_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())

    turns = make_turns(args.num_turns, args.tokens_per_turn, args.target_turn)
    target = turns[args.target_turn]["memory_token"]
    full_prompt = " ".join(t["prompt"] for t in turns)
    query_text = f"What was the secret code from turn {args.target_turn}? Just the number."

    rendered = render_chat(template, full_prompt)
    context_ids = tok.encode(rendered).ids
    query_ids = tok.encode(render_chat(template, query_text)).ids
    n = len(context_ids)
    print(f"[diag] {args.num_turns} turns, context={n} tok, target turn={args.target_turn} "
          f"code={target}, query={len(query_ids)} tok")

    # Find the exact code token span
    code_span, enc_ids, offsets = find_code_span(rendered, target, tok)
    if code_span is None:
        print(f"[diag] ERROR: code {target} not found in rendered text")
        return
    print(f"[diag] code '{target}' token span: {code_span} "
          f"({len(code_span)} tokens) at char {rendered.find(str(target))}")
    print(f"[diag] code tokens decode to: {[tok.decode([context_ids[i]]) for i in code_span]}")

    # ONE context forward pass
    t0 = time.time()
    cache_full = model.make_cache()
    x = mx.array([context_ids], dtype=mx.int32)
    logits = model(x, cache=cache_full)
    mx.eval(logits)
    print(f"[diag] context forward: {time.time()-t0:.1f}s")

    surprisal = compute_surprisal(logits, context_ids)

    # (B) SIGNAL: surprisal rank of each code token
    print(f"\n[diag] === SIGNAL: surprisal rank of code tokens ===")
    code_ranks = []
    for i in code_span:
        rank = int((surprisal > surprisal[i]).sum()) + 1
        code_ranks.append({"pos": i, "token": tok.decode([context_ids[i]]),
                           "surprisal": round(float(surprisal[i]), 3), "rank": rank, "of": n})
    for cr in code_ranks:
        print(f"    pos={cr['pos']:5d} tok={cr['token']!r:8} surprisal={cr['surprisal']:7.3f} "
              f"rank={cr['rank']:5d}/{n}")
    in_top = {k: sum(1 for cr in code_ranks if cr["rank"] <= k) for k in (16, 64, 256)}
    print(f"    code tokens in top-16: {in_top[16]}/{len(code_span)}, "
          f"top-64: {in_top[64]}/{len(code_span)}, top-256: {in_top[256]}/{len(code_span)}")

    # What actually dominates the top-16 by surprisal?
    top16 = sorted(range(n), key=lambda i: -surprisal[i])[:16]
    print(f"    top-16 surprisal tokens: "
          f"{[(i, tok.decode([context_ids[i]])) for i in top16]}")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_sgm_diagnostic_result.json"
    results = {"code_span": code_span, "code_ranks": code_ranks,
               "code_in_top": in_top, "top16_surprisal":
               [{"pos": i, "token": tok.decode([context_ids[i]]),
                 "surprisal": round(float(surprisal[i]), 3)} for i in top16]}

    # (A) MECHANISM: oracle selection
    print(f"\n[diag] === MECHANISM: oracle selection (exact code span) ===")
    variants = {
        "oracle_code_only": code_span,
        "oracle_code_window": list(range(max(0, min(code_span) - args.window),
                                          min(n, max(code_span) + args.window + 1))),
        "first_256": list(range(256)),
        "last_256": list(range(n - 256, n)),
    }
    for name, positions in variants.items():
        kv = extract_kv_at_positions(cache_full, positions)
        cache_sel = build_cache_with_kv(model, kv)
        decoded = generate(model, tok, cache_sel, query_ids)
        correct, actual = check_recall(decoded, target)
        results[name] = {"n_selected": len(positions), "recall_correct": correct,
                         "actual": actual, "text": decoded[:80]}
        print(f"    {name:<20} n={len(positions):5d} recall={correct} actual={actual} "
              f"text={decoded[:40]!r}")

    # Sanity: in-place
    decoded = generate(model, tok, cache_full, query_ids)
    correct, actual = check_recall(decoded, target)
    results["inplace"] = {"n_selected": n, "recall_correct": correct, "actual": actual,
                          "text": decoded[:80]}
    print(f"    {'inplace':<20} n={n:5d} recall={correct} actual={actual} (sanity)")

    # Verdict
    print(f"\n[diag] === VERDICT ===")
    mech_ok = results["oracle_code_only"]["recall_correct"] or results["oracle_code_window"]["recall_correct"]
    signal_ok = in_top[256] >= len(code_span)  # all code tokens in top-256
    if mech_ok and signal_ok:
        verdict = "BOTH sound — mechanism works AND surprisal finds the fact; SGM should work"
    elif mech_ok and not signal_ok:
        verdict = ("MECHANISM sound but SURPRISAL is the wrong signal — the fact is not "
                   "high-surprisal; find a better salience measure (see code_ranks)")
    elif not mech_ok and signal_ok:
        verdict = ("SURPRISAL finds the fact but the MECHANISM fails — selective KV re-injection "
                   "of a tiny span is insufficient (fact needs more context)")
    else:
        verdict = "both fail — inspect code_ranks and top16_surprisal for the right mechanism"
    print(f"  -> {verdict}")

    results["verdict"] = verdict
    results["target"] = target
    results["target_turn"] = args.target_turn
    results["context_tokens"] = n
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
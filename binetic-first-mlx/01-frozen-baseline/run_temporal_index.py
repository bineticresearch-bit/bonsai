"""Bet 8 — Temporal Index: fuse query-conditional cosine with the temporal signature.

Why Bet 7 (learned index) failed: it had no per-CHUNK signal to learn from. All
10 queries were nearly identical (off-diag cos 0.9987), and the chunk hidden
states are also very similar to each other (they're all "the secret code is N"
statements). So a linear projection had nothing to disambiguate on.

The fix (the user's point): the method is sound — we just need to PASS the
necessary signal. The temporal signature IS that per-chunk signal: it's
intrinsic (write-time), in the model's own state space, and DIFFERENT for each
chunk (target fact = high state-delta, padding = low). So we FUSE:

  cosine(query, chunk)      -> "relevant to this query?"   (read-time, Bet 6)
  temporal_signature(chunk) -> "distinctive?"              (write-time, Bet 4)

The target fact is the only chunk that is BOTH relevant AND distinctive. The
fusion should rank it higher than cosine alone.

Fusion methods (all scale-invariant or normalized):
  - RRF          : score = 1/(60+cos_rank) + 1/(60+sig_rank)   (rank fusion)
  - multiplicative: score = cos_norm * sig_norm
  - additive     : score = 0.5*cos_norm + 0.5*sig_norm

We compare the target chunk's rank under each method vs cosine alone. If the
temporal index ranks the target higher, we confirm with a mechanism test
(warm-cache injection at a smaller k).

Usage:
  /opt/homebrew/bin/python3.12 run_temporal_index.py
  /opt/homebrew/bin/python3.12 run_temporal_index.py --mechanism   # + recall test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_DIR))

# Reuse the Bet 6 machinery (model load, chunked forward, warm cache, generate)
from run_retrieval_test import (
    load_model_and_tokenizer, make_turns, render_chat, process_in_chunks,
    build_chunk_index, query_hidden_state, build_warm_cache, generate,
    check_recall, PACK_DIR,
)


def rank_of(scores, target):
    return int((scores > scores[target]).sum()) + 1


def rrf_rank(cos_scores, sig_scores, target, k=60):
    """Reciprocal-rank-fusion rank of the target chunk."""
    n = len(cos_scores)
    cos_rank = np.argsort(np.argsort(-cos_scores))  # rank per chunk (0-based)
    sig_rank = np.argsort(np.argsort(-sig_scores))
    rrf = 1.0 / (k + cos_rank + 1) + 1.0 / (k + sig_rank + 1)
    return rank_of(rrf, target)


def norm01(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def main():
    import mlx.core as mx
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-turns", type=int, default=10)
    ap.add_argument("--tokens-per-turn", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--mechanism", action="store_true", help="run the recall test")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--window", type=int, default=16)
    args = ap.parse_args()

    print("[tidx] loading model...")
    model, config, tok = load_model_and_tokenizer()

    template_path = PACK_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())

    turns = make_turns(args.num_turns, args.tokens_per_turn)
    full_prompt = " ".join(t["prompt"] for t in turns)
    rendered = render_chat(template, full_prompt)
    context_ids = tok.encode(rendered).ids
    n = len(context_ids)
    n_chunks = (n + args.chunk - 1) // args.chunk
    print(f"[tidx] {args.num_turns} turns, context={n} tok, n_chunks={n_chunks}")

    # Locate each turn's code chunk
    enc = tok.encode(rendered)
    offsets = enc.offsets
    targets = []
    for t in turns:
        code_str = str(t["memory_token"])
        pos = rendered.find(code_str)
        if pos < 0:
            print(f"[tidx] ERROR: code {code_str} not found")
            return
        end = pos + len(code_str)
        span = [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]
        targets.append(min(span) // args.chunk)
    print(f"[tidx] target chunks per turn: {targets}")

    # ONE forward pass: hidden states (index) + temporal signature
    t0 = time.time()
    cache_full, hidden_chunks, signature = process_in_chunks(model, context_ids, args.chunk)
    print(f"[tidx] chunked forward: {time.time()-t0:.1f}s")
    C = build_chunk_index(hidden_chunks, args.chunk, n)  # (n_chunks, hidden)
    sig = np.asarray(signature, dtype=np.float64)  # (n_chunks,)

    # Query hidden states
    Q = []
    for t in turns:
        qtext = f"What was the secret code from turn {t['turn']}? Just the number."
        qids = tok.encode(render_chat(template, qtext)).ids
        Q.append(query_hidden_state(model, qids))
    Q = np.stack(Q, axis=0)  # (nq, hidden)

    # RANK COMPARISON
    print(f"\n[tidx] === RANK COMPARISON (target chunk rank, lower=better) ===")
    print(f"  {'turn':>4} {'code':>5} {'chunk':>5} {'cosine':>7} {'sig':>6} "
          f"{'RRF':>6} {'mult':>6} {'add':>6}")
    cos_ranks, sig_ranks, rrf_ranks, mult_ranks, add_ranks = [], [], [], [], []
    for i, tgt in enumerate(targets):
        cos_scores = C @ Q[i]
        cos_r = rank_of(cos_scores, tgt)
        sig_r = rank_of(sig, tgt)
        rrf_r = rrf_rank(cos_scores, sig, tgt)
        # multiplicative + additive (normalized)
        cos_n = norm01(cos_scores)
        sig_n = norm01(sig)
        mult_r = rank_of(cos_n * sig_n, tgt)
        add_r = rank_of(0.5 * cos_n + 0.5 * sig_n, tgt)
        cos_ranks.append(cos_r); sig_ranks.append(sig_r); rrf_ranks.append(rrf_r)
        mult_ranks.append(mult_r); add_ranks.append(add_r)
        print(f"  {turns[i]['turn']:>4} {turns[i]['memory_token']:>5} {tgt:>5} "
              f"{cos_r:>7} {sig_r:>6} {rrf_r:>6} {mult_r:>6} {add_r:>6}")

    def m(x): return float(np.mean(x))
    print(f"\n[tidx] mean rank:  cosine={m(cos_ranks):.1f}  sig={m(sig_ranks):.1f}  "
          f"RRF={m(rrf_ranks):.1f}  mult={m(mult_ranks):.1f}  add={m(add_ranks):.1f}")

    # Pick the best fusion method
    methods = {"cosine": cos_ranks, "RRF": rrf_ranks, "mult": mult_ranks, "add": add_ranks}
    best = min(methods, key=lambda k: m(methods[k]))
    print(f"[tidx] best method: {best} (mean rank {m(methods[best]):.1f} vs cosine {m(cos_ranks):.1f})")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_temporal_index_result.json"
    results = {
        "test": "temporal_index", "name": "temporal_index_fusion",
        "num_turns": args.num_turns, "context_tokens": n, "n_chunks": n_chunks,
        "targets": targets,
        "cos_ranks": cos_ranks, "sig_ranks": sig_ranks, "rrf_ranks": rrf_ranks,
        "mult_ranks": mult_ranks, "add_ranks": add_ranks,
        "cos_mean": round(m(cos_ranks), 2), "sig_mean": round(m(sig_ranks), 2),
        "rrf_mean": round(m(rrf_ranks), 2), "mult_mean": round(m(mult_ranks), 2),
        "add_mean": round(m(add_ranks), 2), "best_method": best,
    }

    # MECHANISM TEST (optional): does the best fusion recall at a smaller k?
    if args.mechanism:
        print(f"\n[tidx] === MECHANISM: warm-cache injection (best={best}) ===")
        # sanity: in-place
        target = turns[args.num_turns // 2]["memory_token"]
        tgt = targets[args.num_turns // 2]
        decoded = generate(model, tok, cache_full,
                           tok.encode(render_chat(template,
                               f"What was the secret code from turn {args.num_turns//2}? Just the number.")).ids)
        correct, actual = check_recall(decoded, target)
        results["inplace"] = {"recall": correct, "actual": actual}
        print(f"  inplace (sanity): recall={correct} actual={actual}")

        for k in args.ks:
            for label, topk in (("cosine", None), (best, None)):
                # recompute top-k for this method
                i = args.num_turns // 2
                cos_scores = C @ Q[i]
                if label == "cosine":
                    topk = sorted(range(n_chunks), key=lambda c: -cos_scores[c])[:k]
                elif label == "RRF":
                    cos_rank = np.argsort(np.argsort(-cos_scores))
                    sig_rank = np.argsort(np.argsort(-sig))
                    rrf = 1.0/(60+cos_rank+1) + 1.0/(60+sig_rank+1)
                    topk = sorted(range(n_chunks), key=lambda c: -rrf[c])[:k]
                elif label == "mult":
                    topk = sorted(range(n_chunks), key=lambda c: -(norm01(cos_scores)[c]*norm01(sig)[c]))[:k]
                elif label == "add":
                    topk = sorted(range(n_chunks), key=lambda c: -(0.5*norm01(cos_scores)[c]+0.5*norm01(sig)[c]))[:k]
                for sub, positions in (
                    (f"{label}_k{k}", [p for c in topk for p in range(c*args.chunk, min(n,(c+1)*args.chunk))]),
                    (f"{label}_k{k}_win{args.window}", sorted(
                        set(p for c in topk for p in range(max(0,c*args.chunk+args.chunk//2-args.window),
                                                            min(n,c*args.chunk+args.chunk//2+args.window))))),
                ):
                    cache_sel = build_warm_cache(model, cache_full, positions)
                    qids = tok.encode(render_chat(template,
                        f"What was the secret code from turn {args.num_turns//2}? Just the number.")).ids
                    decoded = generate(model, tok, cache_sel, qids)
                    correct, actual = check_recall(decoded, target)
                    code_in_sel = sum(1 for c in topk if c == tgt)
                    results[sub] = {"n": len(positions), "recall": correct, "actual": actual,
                                    "target_in_topk": code_in_sel}
                    print(f"  {sub:<24} n={len(positions):5d} target_in_topk={code_in_sel} "
                          f"recall={correct} actual={actual}")
                    del cache_sel
                    mx.metal.clear_cache()

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[tidx] wrote {out_path}")


if __name__ == "__main__":
    main()
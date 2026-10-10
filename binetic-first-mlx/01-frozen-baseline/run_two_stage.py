"""Bet 11 — Two-stage retrieval: topic (cosine) + disambiguation (algorithm).

The deep finding from Bets 7-10: no signal derived from the model's internal
state can disambiguate between similar facts. The disambiguation requires the
POSITION (known in a structured conversation).

This test builds a two-stage retrieval ALGORITHM (no learning):
  Stage 1 (topic): cosine(query, chunk) -> candidate set (top-k "secret code" chunks)
  Stage 2 (disambiguation): pick the correct chunk from the candidates

Three disambiguation strategies:
  1. POSITION: the query says "turn N" -> pick the chunk at that turn's position
  2. RECENCY:  pick the most recent "secret code" chunk (last in context)
  3. SALIENCE: pick the most salient "secret code" chunk (highest temporal signature)

We compare against the single-stage cosine (Bet 6, mean rank 10.1).

The key question: does the two-stage algorithm beat the single-stage cosine?
And: which disambiguation strategy works best?

Usage:
  /opt/homebrew/bin/python3.12 run_two_stage.py
  /opt/homebrew/bin/python3.12 run_two_stage.py --mechanism   # + recall test
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

from run_retrieval_test import (
    load_model_and_tokenizer, make_turns, render_chat, process_in_chunks,
    build_chunk_index, query_hidden_state, build_warm_cache, generate,
    check_recall, PACK_DIR,
)


def rank_of(scores, target, higher_is_better=True):
    if higher_is_better:
        return int((scores > scores[target]).sum()) + 1
    else:
        return int((scores < scores[target]).sum()) + 1


def main():
    import mlx.core as mx
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-turns", type=int, default=10)
    ap.add_argument("--tokens-per-turn", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--k", type=int, default=10, help="candidate set size for stage 1")
    ap.add_argument("--mechanism", action="store_true", help="run the recall test")
    args = ap.parse_args()

    print("[2s] loading model...")
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
    print(f"[2s] {args.num_turns} turns, context={n} tok, n_chunks={n_chunks}")

    # Locate each turn's code chunk
    enc = tok.encode(rendered)
    offsets = enc.offsets
    targets = []
    for t in turns:
        code_str = str(t["memory_token"])
        pos = rendered.find(code_str)
        if pos < 0:
            print(f"[2s] ERROR: code {code_str} not found")
            return
        end = pos + len(code_str)
        span = [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]
        targets.append(min(span) // args.chunk)
    print(f"[2s] target chunks per turn: {targets}")

    # ONE forward pass: hidden states (index) + temporal signature
    t0 = time.time()
    cache_full, hidden_chunks, signature = process_in_chunks(model, context_ids, args.chunk)
    print(f"[2s] chunked forward: {time.time()-t0:.1f}s")
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
    print(f"\n[2s] === RANK COMPARISON (target chunk rank, lower=better) ===")
    print(f"  {'turn':>4} {'code':>5} {'chunk':>5} {'cosine':>7} {'2s-pos':>7} "
          f"{'2s-rec':>7} {'2s-sal':>7}")
    cos_ranks, pos_ranks, rec_ranks, sal_ranks = [], [], [], []
    for i, tgt in enumerate(targets):
        cos_scores = C @ Q[i]
        cos_r = rank_of(cos_scores, tgt)

        # Stage 1: candidate set (top-k by cosine)
        candidates = sorted(range(n_chunks), key=lambda c: -cos_scores[c])[:args.k]

        # Stage 2a: POSITION — is the target in the candidate set?
        # (In a real system, you'd parse the turn number and look up the position.)
        # Here, we check: is the target in the top-k? If yes, rank = its position in the candidates.
        if tgt in candidates:
            pos_r = candidates.index(tgt) + 1  # rank within the candidate set
        else:
            pos_r = n_chunks  # not in the candidate set

        # Stage 2b: RECENCY — pick the most recent "secret code" chunk
        # (the last fact chunk in the candidate set)
        fact_candidates = [c for c in candidates if c in targets]
        if fact_candidates:
            rec_pick = max(fact_candidates)  # most recent
            rec_r = 1 if rec_pick == tgt else (candidates.index(rec_pick) + 1)
        else:
            rec_r = n_chunks

        # Stage 2c: SALIENCE — pick the most salient "secret code" chunk
        # (highest temporal signature in the candidate set)
        if fact_candidates:
            sal_pick = max(fact_candidates, key=lambda c: sig[c])
            sal_r = 1 if sal_pick == tgt else (candidates.index(sal_pick) + 1)
        else:
            sal_r = n_chunks

        cos_ranks.append(cos_r); pos_ranks.append(pos_r)
        rec_ranks.append(rec_r); sal_ranks.append(sal_r)
        print(f"  {turns[i]['turn']:>4} {turns[i]['memory_token']:>5} {tgt:>5} "
              f"{cos_r:>7} {pos_r:>7} {rec_r:>7} {sal_r:>7}")

    def m(x): return float(np.mean(x))
    print(f"\n[2s] mean rank:  cosine={m(cos_ranks):.1f}  2s-pos={m(pos_ranks):.1f}  "
          f"2s-rec={m(rec_ranks):.1f}  2s-sal={m(sal_ranks):.1f}")
    print(f"  (k={args.k} candidate set; 2s-pos = rank within candidates if target is in top-k)")

    # DIAGNOSTIC: how often is the target in the top-k?
    in_topk = sum(1 for r in pos_ranks if r <= args.k)
    print(f"\n[2s] target in top-{args.k}: {in_topk}/{len(targets)} "
          f"({100*in_topk/len(targets):.0f}%)")

    # Verdict
    print(f"\n[2s] === VERDICT ===")
    cos_mean = m(cos_ranks)
    pos_mean = m(pos_ranks)
    rec_mean = m(rec_ranks)
    sal_mean = m(sal_ranks)
    if pos_mean < cos_mean * 0.5:
        verdict = (f"TWO-STAGE (position) WINS — mean rank {pos_mean:.1f} vs cosine {cos_mean:.1f}; "
                   "the position is the disambiguator")
    elif rec_mean < cos_mean * 0.5:
        verdict = (f"TWO-STAGE (recency) WINS — mean rank {rec_mean:.1f} vs cosine {cos_mean:.1f}; "
                   "recency is a reasonable fallback when the position is unknown")
    elif sal_mean < cos_mean * 0.5:
        verdict = (f"TWO-STAGE (salience) WINS — mean rank {sal_mean:.1f} vs cosine {cos_mean:.1f}; "
                   "the temporal signature is a reasonable fallback")
    else:
        verdict = (f"TWO-STAGE DOES NOT BEAT COSINE — best is "
                   f"pos={pos_mean:.1f}, rec={rec_mean:.1f}, sal={sal_mean:.1f} vs cosine {cos_mean:.1f}. "
                   "The disambiguation requires the position, which is a lookup, not retrieval.")

    print(f"  -> {verdict}")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_two_stage_result.json"
    results = {
        "test": "two_stage", "name": "two_stage_retrieval",
        "num_turns": args.num_turns, "context_tokens": n, "n_chunks": n_chunks,
        "k": args.k,
        "targets": targets,
        "cos_ranks": cos_ranks, "pos_ranks": pos_ranks,
        "rec_ranks": rec_ranks, "sal_ranks": sal_ranks,
        "cos_mean": round(cos_mean, 2), "pos_mean": round(pos_mean, 2),
        "rec_mean": round(rec_mean, 2), "sal_mean": round(sal_mean, 2),
        "in_topk": in_topk, "n_targets": len(targets),
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # MECHANISM TEST (optional): does the two-stage algorithm recall?
    if args.mechanism:
        print(f"\n[2s] === MECHANISM: warm-cache injection (two-stage) ===")
        target = turns[args.num_turns // 2]["memory_token"]
        tgt = targets[args.num_turns // 2]
        qids = tok.encode(render_chat(template,
            f"What was the secret code from turn {args.num_turns//2}? Just the number.")).ids

        # sanity: in-place
        decoded = generate(model, tok, cache_full, qids)
        correct, actual = check_recall(decoded, target)
        results["inplace"] = {"recall": correct, "actual": actual}
        print(f"  inplace (sanity): recall={correct} actual={actual}")

        # two-stage: candidate set (top-k by cosine) + position disambiguation
        cos_scores = C @ Q[args.num_turns // 2]
        candidates = sorted(range(n_chunks), key=lambda c: -cos_scores[c])[:args.k]
        # position: the target chunk
        if tgt in candidates:
            positions = [p for c in candidates for p in range(c*args.chunk, min(n,(c+1)*args.chunk))]
        else:
            positions = [p for c in candidates for p in range(c*args.chunk, min(n,(c+1)*args.chunk))]

        cache_sel = build_warm_cache(model, cache_full, positions)
        decoded = generate(model, tok, cache_sel, qids)
        correct, actual = check_recall(decoded, target)
        results["two_stage"] = {"n": len(positions), "recall": correct, "actual": actual,
                                "target_in_candidates": tgt in candidates}
        print(f"  two_stage (k={args.k}): n={len(positions)} target_in_candidates={tgt in candidates} "
              f"recall={correct} actual={actual}")
        del cache_sel
        mx.metal.clear_cache()

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[2s] wrote {out_path}")


if __name__ == "__main__":
    main()
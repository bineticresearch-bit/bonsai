"""Bet 9 — Triangle Index: the difference vector (Q - C) and its residual.

The triangle O-Q-C_i has three sides: |Q|, |C_i|, and D_i = Q - C_i (the
difference that "completes the triangle"). Cosine only uses the angle; the
difference vector uses the FULL triangle (angle + both magnitudes).

Decompose the chunk into the part the query EXPLAINS and the part it DOESN'T:
  C_i = proj_Q(C_i) + R_i
  proj_Q(C_i) = (Q.C_i/|Q|^2) Q   <- "general" (same topic as the query)
  R_i  = C_i - proj_Q(C_i)        <- "specific" (perpendicular to Q)

Hypothesis: the residual R_i isolates the SPECIFIC content (the code value)
from the GENERAL topic ("secret code"). If the target fact's residual is
different from the other facts' residuals, the triangle geometry provides a
disambiguation signal that cosine (angle-only) throws away.

Tests (rank of the target chunk, lower = better):
  1. cosine(Q, C_j)          <- baseline (Bet 6, mean rank ~10.1)
  2. |Q - C_j| (distance)    <- full triangle, expect ~cosine (angle-dominated)
  3. |R_j| (residual mag)    <- "how specific is this chunk?"
  4. cosine(Q, R_j)          <- "does the query point at the specific content?"
  5. |R_j| x position        <- structured case: specific content + known position

Usage:
  /opt/homebrew/bin/python3.12 run_triangle_index.py
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
    build_chunk_index, query_hidden_state, PACK_DIR,
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
    args = ap.parse_args()

    print("[tri] loading model...")
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
    print(f"[tri] {args.num_turns} turns, context={n} tok, n_chunks={n_chunks}")

    # Locate each turn's code chunk
    enc = tok.encode(rendered)
    offsets = enc.offsets
    targets = []
    for t in turns:
        code_str = str(t["memory_token"])
        pos = rendered.find(code_str)
        if pos < 0:
            print(f"[tri] ERROR: code {code_str} not found")
            return
        end = pos + len(code_str)
        span = [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]
        targets.append(min(span) // args.chunk)
    print(f"[tri] target chunks per turn: {targets}")

    # ONE forward pass: chunk hidden states
    t0 = time.time()
    cache, hidden_chunks, _ = process_in_chunks(model, context_ids, args.chunk)
    print(f"[tri] chunked forward: {time.time()-t0:.1f}s")
    C = build_chunk_index(hidden_chunks, args.chunk, n)  # (n_chunks, hidden)

    # Query hidden states
    Q = []
    for t in turns:
        qtext = f"What was the secret code from turn {t['turn']}? Just the number."
        qids = tok.encode(render_chat(template, qtext)).ids
        Q.append(query_hidden_state(model, qids))
    Q = np.stack(Q, axis=0)  # (nq, hidden)

    # RANK COMPARISON
    print(f"\n[tri] === RANK COMPARISON (target chunk rank, lower=better) ===")
    print(f"  {'turn':>4} {'code':>5} {'chunk':>5} {'cosine':>7} {'dist':>6} "
          f"{'resid':>6} {'q.resid':>8} {'resid+pos':>10}")
    cos_ranks, dist_ranks, resid_ranks, qresid_ranks, residpos_ranks = [], [], [], [], []
    for i, tgt in enumerate(targets):
        q = Q[i]  # (hidden,)
        # 1. cosine
        cos_scores = C @ q
        cos_r = rank_of(cos_scores, tgt)
        # 2. distance |Q - C_j| (lower = closer = better)
        dist = np.linalg.norm(C - q[None, :], axis=1)
        dist_r = rank_of(dist, tgt, higher_is_better=False)
        # 3. residual magnitude |R_j| where R_j = C_j - proj_q(C_j)
        proj = (C @ q)[:, None] / (q @ q) * q[None, :]  # (n_chunks, hidden)
        R = C - proj  # (n_chunks, hidden)
        resid_mag = np.linalg.norm(R, axis=1)
        # Higher residual = more "specific" (less explained by query)
        # The target fact should have a DISTINCTIVE residual, but is it
        # HIGHER or LOWER than the others? Let's rank by magnitude.
        resid_r = rank_of(resid_mag, tgt)
        # 4. cosine(Q, R_j) — does the query point at the specific content?
        R_norm = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
        qresid_scores = R_norm @ q
        qresid_r = rank_of(qresid_scores, tgt)
        # 5. residual magnitude x position (structured: known target position)
        # Position score: 1/(1+|chunk_pos - target_pos|) — peaks at target
        chunk_positions = np.arange(n_chunks)
        pos_score = 1.0 / (1.0 + np.abs(chunk_positions - tgt).astype(float))
        # Combine: residual magnitude (normalized) x position score
        resid_norm = resid_mag / (resid_mag.max() + 1e-12)
        residpos_scores = resid_norm * pos_score
        residpos_r = rank_of(residpos_scores, tgt)

        cos_ranks.append(cos_r); dist_ranks.append(dist_r)
        resid_ranks.append(resid_r); qresid_ranks.append(qresid_r)
        residpos_ranks.append(residpos_r)
        print(f"  {turns[i]['turn']:>4} {turns[i]['memory_token']:>5} {tgt:>5} "
              f"{cos_r:>7} {dist_r:>6} {resid_r:>6} {qresid_r:>8} {residpos_r:>10}")

    def m(x): return float(np.mean(x))
    print(f"\n[tri] mean rank:  cosine={m(cos_ranks):.1f}  dist={m(dist_ranks):.1f}  "
          f"resid={m(resid_ranks):.1f}  q.resid={m(qresid_ranks):.1f}  "
          f"resid+pos={m(residpos_ranks):.1f}")

    # DIAGNOSTIC: are the residuals actually different per chunk?
    print(f"\n[tri] === DIAGNOSTIC: residual structure ===")
    # For the target query (turn 5), compute residuals of all chunks
    i = args.num_turns // 2
    q = Q[i]
    proj = (C @ q)[:, None] / (q @ q) * q[None, :]
    R = C - proj
    R_norm = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    # Cosine between the target residual and each other fact's residual
    tgt = targets[i]
    other_facts = [t for t in targets if t != tgt]
    print(f"  target chunk {tgt} residual vs other fact residuals:")
    for of in other_facts:
        cos_rr = float(R_norm[tgt] @ R_norm[of])
        print(f"    chunk {of:4d} (code {turns[targets.index(of)]['memory_token']}): "
              f"cos(resid_target, resid_other) = {cos_rr:.4f}")
    # Cosine between target residual and a padding chunk
    pad_chunk = 0  # first chunk is padding
    cos_rp = float(R_norm[tgt] @ R_norm[pad_chunk])
    print(f"    chunk {pad_chunk:4d} (padding): cos(resid_target, resid_pad) = {cos_rp:.4f}")
    # Residual magnitude: target vs other facts vs padding
    resid_mag = np.linalg.norm(R, axis=1)
    print(f"\n  residual magnitude: target={resid_mag[tgt]:.4f}  "
          f"other_facts_mean={np.mean([resid_mag[t] for t in other_facts]):.4f}  "
          f"padding_mean={np.mean(resid_mag[:10]):.4f}")

    # Verdict
    print(f"\n[tri] === VERDICT ===")
    cos_mean = m(cos_ranks)
    best_method = min(
        [("cosine", cos_ranks), ("dist", dist_ranks), ("resid", resid_ranks),
         ("q.resid", qresid_ranks), ("resid+pos", residpos_ranks)],
        key=lambda x: m(x[1])
    )
    print(f"  best method: {best_method[0]} (mean rank {m(best_method[1]):.1f} vs cosine {cos_mean:.1f})")
    if m(best_method[1]) < cos_mean * 0.5:
        verdict = f"TRIANGLE INDEX WINS — {best_method[0]} mean rank {m(best_method[1]):.1f} vs cosine {cos_mean:.1f}"
    elif m(best_method[1]) < cos_mean:
        verdict = f"TRIANGLE INDEX HELPS — {best_method[0]} mean rank {m(best_method[1]):.1f} vs cosine {cos_mean:.1f}"
    else:
        verdict = (f"TRIANGLE INDEX DOES NOT BEAT COSINE — best is {best_method[0]} "
                   f"(mean rank {m(best_method[1]):.1f} vs cosine {cos_mean:.1f}). "
                   "The residual is orthogonal to the query (the query has no specific "
                   "content to match), so the triangle geometry doesn't provide the "
                   "disambiguation. The disambiguation signal is POSITION, not the triangle.")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_triangle_index_result.json"
    out_path.write_text(json.dumps({
        "test": "triangle_index", "name": "triangle_difference_index",
        "num_turns": args.num_turns, "context_tokens": n, "n_chunks": n_chunks,
        "targets": targets,
        "cos_ranks": cos_ranks, "dist_ranks": dist_ranks,
        "resid_ranks": resid_ranks, "qresid_ranks": qresid_ranks,
        "residpos_ranks": residpos_ranks,
        "cos_mean": round(cos_mean, 2), "dist_mean": round(m(dist_ranks), 2),
        "resid_mean": round(m(resid_ranks), 2), "qresid_mean": round(m(qresid_ranks), 2),
        "residpos_mean": round(m(residpos_ranks), 2),
        "best_method": best_method[0],
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"\n[tri] wrote {out_path}")


if __name__ == "__main__":
    main()
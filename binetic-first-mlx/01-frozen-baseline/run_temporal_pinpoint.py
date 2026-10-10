"""Bet 10 — Temporal Pinpoint: the state-delta VECTOR (direction + magnitude).

The temporal signature (Bet 4) only uses the MAGNITUDE of the state-delta
(||ΔS||) — "how much changed." The temporal PINPOINT uses the full ΔS VECTOR:
  - magnitude ||ΔS|| = "how much changed" (salience, fact-vs-padding)
  - direction ΔS/||ΔS|| = "what kind of change" (the event's fingerprint)

Hypothesis: the direction is a per-event fingerprint. Two different facts
produce different state-delta directions (different "kinks" in the state
trajectory), even if their magnitudes are similar. So the direction is a
richer signal than the magnitude.

Tests:
  1. Fact-vs-padding: is the ΔS DIRECTION a better fact-vs-padding signal
     than the ΔS MAGNITUDE? (Rank the target by direction distinctiveness
     vs by magnitude.)
  2. Target-vs-other-facts: is the target's ΔS direction DIFFERENT from the
     other facts' ΔS directions? (Cosine between target's ΔS and other
     facts' ΔS.)
  3. Fusion: does cosine (content) × ΔS direction beat Bet 8's temporal
     index (cosine × ΔS magnitude, mean rank 9.6)?

Usage:
  /opt/homebrew/bin/python3.12 run_temporal_pinpoint.py
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
    load_model_and_tokenizer, make_turns, render_chat,
    build_chunk_index, query_hidden_state, PACK_DIR,
    snapshot_linear_states,
)


def flatten_snap(snap):
    """Flatten a snapshot {i: [arrays]} into one big 1-D vector."""
    parts = []
    for i in sorted(snap.keys()):
        for a in snap[i]:
            if a is not None:
                parts.append(np.asarray(a, dtype=np.float32).ravel())
    return np.concatenate(parts) if parts else np.zeros(1)


def process_with_delta_vectors(model, context_ids, chunk_size, proj_dim=2048, seed=0):
    """One chunked forward pass -> (cache, hidden_chunks, delta_proj, delta_mags).

    delta_proj: (n_chunks, proj_dim) — the ΔS vector projected to a random
    subspace (preserves cosine similarity, avoids OOM on the 75M-dim state).
    delta_mags: (n_chunks,) — ||ΔS|| per chunk (the temporal signature).
    """
    import mlx.core as mx

    # Fixed random subset of state dims (the projection)
    # We don't know the state_dim until the first chunk, so use a lazy init.
    cache = model.make_cache()
    hidden_chunks = []
    delta_proj_list = []
    delta_mags = []
    prev_snap = snapshot_linear_states(cache)
    prev_flat = flatten_snap(prev_snap)
    proj_indices = None
    rng = np.random.default_rng(seed)
    for i in range(0, len(context_ids), chunk_size):
        chunk = context_ids[i:i + chunk_size]
        x = mx.array([chunk], dtype=mx.int32)
        h = model.model(x, cache=cache)
        mx.eval(h)
        hidden_chunks.append(np.asarray(h, dtype=np.float32))
        snap = snapshot_linear_states(cache)
        cur_flat = flatten_snap(snap)
        delta = cur_flat - prev_flat
        mag = float(np.linalg.norm(delta))
        delta_mags.append(mag)
        if proj_indices is None:
            state_dim = len(delta)
            proj_indices = rng.choice(state_dim, size=proj_dim, replace=False)
        delta_proj_list.append(delta[proj_indices])
        prev_flat = cur_flat
    return cache, hidden_chunks, np.stack(delta_proj_list), np.array(delta_mags)


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

    print("[pin] loading model...")
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
    print(f"[pin] {args.num_turns} turns, context={n} tok, n_chunks={n_chunks}")

    # Locate each turn's code chunk
    enc = tok.encode(rendered)
    offsets = enc.offsets
    targets = []
    for t in turns:
        code_str = str(t["memory_token"])
        pos = rendered.find(code_str)
        if pos < 0:
            print(f"[pin] ERROR: code {code_str} not found")
            return
        end = pos + len(code_str)
        span = [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]
        targets.append(min(span) // args.chunk)
    print(f"[pin] target chunks per turn: {targets}")

    # ONE forward pass: hidden states + delta vectors (projected)
    t0 = time.time()
    cache, hidden_chunks, delta_vecs, delta_mags = process_with_delta_vectors(
        model, context_ids, args.chunk)
    print(f"[pin] chunked forward: {time.time()-t0:.1f}s, "
          f"proj_dim={delta_vecs.shape[1]}")
    C = build_chunk_index(hidden_chunks, args.chunk, n)  # (n_chunks, hidden)

    # Query hidden states
    Q = []
    for t in turns:
        qtext = f"What was the secret code from turn {t['turn']}? Just the number."
        qids = tok.encode(render_chat(template, qtext)).ids
        Q.append(query_hidden_state(model, qids))
    Q = np.stack(Q, axis=0)  # (nq, hidden)

    # Normalize delta vectors (direction)
    delta_norms = np.linalg.norm(delta_vecs, axis=1, keepdims=True)
    delta_norms[delta_norms == 0] = 1.0
    delta_dirs = delta_vecs / delta_norms  # (n_chunks, state_dim)

    # Mean delta direction (the "average event")
    mean_dir = delta_dirs.mean(axis=0)
    mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)

    # Distinctiveness: 1 - cosine(chunk_dir, mean_dir)
    # Higher = more distinctive (further from the average event)
    distinctiveness = 1.0 - (delta_dirs @ mean_dir)

    # RANK COMPARISON
    print(f"\n[pin] === RANK COMPARISON (target chunk rank, lower=better) ===")
    print(f"  {'turn':>4} {'code':>5} {'chunk':>5} {'cosine':>7} {'mag':>6} "
          f"{'dir':>6} {'cos×mag':>8} {'cos×dir':>8}")
    cos_ranks, mag_ranks, dir_ranks, cosmags_ranks, cosdirs_ranks = [], [], [], [], []
    for i, tgt in enumerate(targets):
        cos_scores = C @ Q[i]
        cos_r = rank_of(cos_scores, tgt)
        mag_r = rank_of(delta_mags, tgt)
        dir_r = rank_of(distinctiveness, tgt)
        # Fusion: cosine (normalized) × magnitude (normalized)
        cos_n = (cos_scores - cos_scores.min()) / (cos_scores.max() - cos_scores.min() + 1e-12)
        mag_n = (delta_mags - delta_mags.min()) / (delta_mags.max() - delta_mags.min() + 1e-12)
        dir_n = (distinctiveness - distinctiveness.min()) / (distinctiveness.max() - distinctiveness.min() + 1e-12)
        cosmags_r = rank_of(cos_n * mag_n, tgt)
        cosdirs_r = rank_of(cos_n * dir_n, tgt)
        cos_ranks.append(cos_r); mag_ranks.append(mag_r); dir_ranks.append(dir_r)
        cosmags_ranks.append(cosmags_r); cosdirs_ranks.append(cosdirs_r)
        print(f"  {turns[i]['turn']:>4} {turns[i]['memory_token']:>5} {tgt:>5} "
              f"{cos_r:>7} {mag_r:>6} {dir_r:>6} {cosmags_r:>8} {cosdirs_r:>8}")

    def m(x): return float(np.mean(x))
    print(f"\n[pin] mean rank:  cosine={m(cos_ranks):.1f}  mag={m(mag_ranks):.1f}  "
          f"dir={m(dir_ranks):.1f}  cos×mag={m(cosmags_ranks):.1f}  "
          f"cos×dir={m(cosdirs_ranks):.1f}")
    print(f"  (Bet 8 temporal index cos×mag was 9.6; cosine was 10.1)")

    # DIAGNOSTIC: is the direction a per-event fingerprint?
    print(f"\n[pin] === DIAGNOSTIC: direction structure ===")
    i = args.num_turns // 2
    tgt = targets[i]
    other_facts = [t for t in targets if t != tgt]
    print(f"  target chunk {tgt} ΔS direction vs other fact ΔS directions:")
    for of in other_facts:
        cos_dd = float(delta_dirs[tgt] @ delta_dirs[of])
        print(f"    chunk {of:4d} (code {turns[targets.index(of)]['memory_token']}): "
              f"cos(ΔS_target, ΔS_other) = {cos_dd:.4f}")
    # vs padding
    pad_chunk = 0
    cos_dp = float(delta_dirs[tgt] @ delta_dirs[pad_chunk])
    print(f"    chunk {pad_chunk:4d} (padding): cos(ΔS_target, ΔS_pad) = {cos_dp:.4f}")
    # Magnitude comparison
    print(f"\n  ΔS magnitude: target={delta_mags[tgt]:.4f}  "
          f"other_facts_mean={np.mean([delta_mags[t] for t in other_facts]):.4f}  "
          f"padding_mean={np.mean(delta_mags[:10]):.4f}")
    # Distinctiveness comparison
    print(f"  distinctiveness: target={distinctiveness[tgt]:.4f}  "
          f"other_facts_mean={np.mean([distinctiveness[t] for t in other_facts]):.4f}  "
          f"padding_mean={np.mean(distinctiveness[:10]):.4f}")

    # Verdict
    print(f"\n[pin] === VERDICT ===")
    cos_mean = m(cos_ranks)
    methods = [("cosine", cos_ranks), ("mag", mag_ranks), ("dir", dir_ranks),
               ("cos×mag", cosmags_ranks), ("cos×dir", cosdirs_ranks)]
    best = min(methods, key=lambda x: m(x[1]))
    print(f"  best method: {best[0]} (mean rank {m(best[1]):.1f} vs cosine {cos_mean:.1f})")
    if m(best[1]) < cos_mean * 0.5:
        verdict = f"TEMPORAL PINPOINT WINS — {best[0]} mean rank {m(best[1]):.1f} vs cosine {cos_mean:.1f}"
    elif m(best[1]) < cos_mean:
        verdict = f"TEMPORAL PINPOINT HELPS — {best[0]} mean rank {m(best[1]):.1f} vs cosine {cos_mean:.1f}"
    else:
        verdict = (f"TEMPORAL PINPOINT DOES NOT BEAT COSINE — best is {best[0]} "
                   f"(mean rank {m(best[1]):.1f} vs cosine {cos_mean:.1f}). "
                   "The ΔS direction is a fact-vs-padding signal, not a "
                   "target-vs-other-facts signal (all facts are 'fact-like').")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_temporal_pinpoint_result.json"
    out_path.write_text(json.dumps({
        "test": "temporal_pinpoint", "name": "temporal_pinpoint_delta_vector",
        "num_turns": args.num_turns, "context_tokens": n, "n_chunks": n_chunks,
        "state_dim": int(delta_vecs.shape[1]),
        "targets": targets,
        "cos_ranks": cos_ranks, "mag_ranks": mag_ranks, "dir_ranks": dir_ranks,
        "cosmags_ranks": cosmags_ranks, "cosdirs_ranks": cosdirs_ranks,
        "cos_mean": round(cos_mean, 2), "mag_mean": round(m(mag_ranks), 2),
        "dir_mean": round(m(dir_ranks), 2),
        "cosmags_mean": round(m(cosmags_ranks), 2),
        "cosdirs_mean": round(m(cosdirs_ranks), 2),
        "best_method": best[0],
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"\n[pin] wrote {out_path}")


if __name__ == "__main__":
    main()
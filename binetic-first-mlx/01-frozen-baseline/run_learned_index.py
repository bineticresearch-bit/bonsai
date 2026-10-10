"""Bet 7 — Can a LEARNED retrieval index beat the cosine index?

Bet 6 showed query-conditional retrieval (cosine in hidden space) ranks the
target fact chunk 18/677 and recalls at k=8 (with a window). The next question:
is the index LEARNABLE? I.e., can a small projection trained to align
query<->fact push the target rank to ~1 and let us drop the window?

Design (one 27B forward pass, then all index work in numpy):
  - Build a 10-turn context. Each turn N has a code; the target chunk for turn
    N is the chunk containing "The secret code is X".
  - Extract: 677 chunk hidden states (mean per chunk) + 10 query hidden states
    (mean over "What was the secret code from turn N?").
  - Baseline: cosine(query_N, chunk) -> rank of target chunk N.
  - Learned: train a small projection P (hidden -> d) with InfoNCE:
        for query N: positive = target chunk N, negatives = all other chunks
        loss = -log( exp(sim(q_N, c_N)/tau) / sum_j exp(sim(q_N, c_j)/tau) )
    sim = cosine of projected, L2-normalized vectors.
  - Evaluate: rank of each target chunk under the learned index vs cosine.

DIAGNOSTIC (the crux): the 10 queries differ ONLY in the turn number
("turn 0" vs "turn 5"). If their hidden states are nearly identical, a linear
projection has no signal to disambiguate which code chunk to pick — and that
itself is a finding (the query must carry the disambiguating signal, or the
index needs the turn number as an explicit feature).

Usage:
  /opt/homebrew/bin/python3.12 run_learned_index.py
  /opt/homebrew/bin/python3.12 run_learned_index.py --dim 128 --steps 400
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

N_LAYERS = 64
FA_INTERVAL = 4
FULL_ATTENTION_LAYERS = [i for i in range(N_LAYERS) if (i + 1) % FA_INTERVAL == 0]
LINEAR_LAYERS = [i for i in range(N_LAYERS) if (i + 1) % FA_INTERVAL != 0]


def load_model_and_tokenizer():
    from artifact import load_model
    from tokenizers import Tokenizer

    model, config = load_model(str(PACK_DIR))
    tok = Tokenizer.from_file(str(PACK_DIR / "tokenizer.json"))
    return model, config, tok


def make_turns(num_turns: int, tokens_per_turn: int) -> list[dict]:
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


def process_in_chunks(model, context_ids, chunk_size):
    """One chunked forward pass -> (cache, per-chunk mean hidden state)."""
    import mlx.core as mx

    cache = model.make_cache()
    hidden_chunks = []
    for i in range(0, len(context_ids), chunk_size):
        chunk = context_ids[i:i + chunk_size]
        x = mx.array([chunk], dtype=mx.int32)
        h = model.model(x, cache=cache)
        mx.eval(h)
        hidden_chunks.append(np.asarray(h, dtype=np.float32))
    return cache, hidden_chunks


def build_chunk_index(hidden_chunks, n_total):
    vecs = [h[0].mean(axis=0) for h in hidden_chunks]
    idx = np.stack(vecs, axis=0)
    norms = np.linalg.norm(idx, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return idx / norms


def query_hidden_state(model, query_ids):
    import mlx.core as mx
    cache = model.make_cache()
    x = mx.array([query_ids], dtype=mx.int32)
    h = model.model(x, cache=cache)
    mx.eval(h)
    v = np.asarray(h, dtype=np.float32)[0].mean(axis=0)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def info_nce_train(Q, C, targets, dim, steps, lr, tau, seed=0):
    """Train a linear projection P (hidden->dim) with InfoNCE.

    Q: (nq, hidden) query vectors (L2-normalized)
    C: (nc, hidden) chunk vectors (L2-normalized)
    targets: (nq,) target chunk index per query
    Returns: trained P (hidden, dim), loss curve.
    """
    rng = np.random.default_rng(seed)
    hidden = Q.shape[1]
    # small init
    P = (rng.standard_normal((hidden, dim)) * (1.0 / np.sqrt(hidden))).astype(np.float32)
    # Adam state
    m = np.zeros_like(P)
    v = np.zeros_like(P)
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss_curve = []
    for step in range(steps):
        # projected, normalized
        Qp = l2norm(Q @ P)  # (nq, dim)
        Cp = l2norm(C @ P)  # (nc, dim)
        # logits: (nq, nc)
        logits = Qp @ Cp.T / tau
        # InfoNCE loss (mean over queries)
        lmax = logits.max(axis=1, keepdims=True)
        logsumexp = lmax.squeeze(1) + np.log(np.sum(np.exp(logits - lmax), axis=1))
        correct = logits[np.arange(len(Q)), targets]
        loss = np.mean(logsumexp - correct)
        loss_curve.append(float(loss))
        # grad of loss w.r.t. logits: (nq, nc)
        probs = np.exp(logits - logsumexp[:, None])
        grad_logits = probs.copy()
        grad_logits[np.arange(len(Q)), targets] -= 1.0
        grad_logits /= (len(Q) * tau)
        # backprop through logits = (Qp @ Cp.T)/tau
        # dL/dQp = grad_logits @ Cp ; dL/dCp = grad_logits.T @ Qp
        dQp = grad_logits @ Cp
        dCp = grad_logits.T @ Qp
        # dL/dP = Q.T @ dQp + C.T @ dCp  (ignoring the l2norm Jacobian for a
        # first-order step — good enough for a small index; we re-normalize each
        # step so the scale is stable)
        dP = Q.T @ dQp + C.T @ dCp
        # Adam
        m = b1 * m + (1 - b1) * dP
        v = b2 * v + (1 - b2) * (dP ** 2)
        mhat = m / (1 - b1 ** (step + 1))
        vhat = v / (1 - b2 ** (step + 1))
        P -= lr * mhat / (np.sqrt(vhat) + eps)
    return P, loss_curve


def rank_of(scores, target):
    return int((scores > scores[target]).sum()) + 1


def main():
    import mlx.core as mx
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-turns", type=int, default=10)
    ap.add_argument("--tokens-per-turn", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    print("[learn] loading model...")
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
    print(f"[learn] {args.num_turns} turns, context={n} tok, n_chunks={n_chunks}")

    # Locate each turn's code chunk
    enc = tok.encode(rendered)
    offsets = enc.offsets
    targets = []
    for t in turns:
        code_str = str(t["memory_token"])
        pos = rendered.find(code_str)
        if pos < 0:
            print(f"[learn] ERROR: code {code_str} not found")
            return
        end = pos + len(code_str)
        span = [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]
        targets.append(min(span) // args.chunk)
    print(f"[learn] target chunks per turn: {targets}")

    # ONE forward pass: chunk hidden states
    t0 = time.time()
    cache, hidden_chunks = process_in_chunks(model, context_ids, args.chunk)
    print(f"[learn] chunked forward: {time.time()-t0:.1f}s")
    C = build_chunk_index(hidden_chunks, n)  # (n_chunks, hidden)

    # Query hidden states (one per turn)
    Q = []
    for t in turns:
        qtext = f"What was the secret code from turn {t['turn']}? Just the number."
        qids = tok.encode(render_chat(template, qtext)).ids
        Q.append(query_hidden_state(model, qids))
    Q = np.stack(Q, axis=0)  # (nq, hidden)
    print(f"[learn] extracted Q={Q.shape} C={C.shape}")

    # DIAGNOSTIC: are the queries distinct?
    qcos = Q @ Q.T
    off_diag = (qcos.sum() - np.trace(qcos)) / (len(Q) * (len(Q) - 1))
    print(f"\n[learn] === DIAGNOSTIC: query distinctiveness ===")
    print(f"  mean off-diag cosine between queries: {off_diag:.4f}")
    print(f"  (1.0 = identical queries -> no disambiguation signal; lower = more distinct)")

    # BASELINE: cosine rank per target
    print(f"\n[learn] === BASELINE: cosine(query, chunk) rank per target ===")
    cos_ranks = []
    for i, tgt in enumerate(targets):
        scores = C @ Q[i]
        r = rank_of(scores, tgt)
        cos_ranks.append(r)
        print(f"  turn {turns[i]['turn']:2d} code={turns[i]['memory_token']} "
              f"chunk {tgt:4d} cos_rank={r:4d}/{n_chunks}")
    print(f"  mean cos_rank = {np.mean(cos_ranks):.1f}, max = {max(cos_ranks)}")

    # LEARNED: InfoNCE projection with a TRAIN/TEST SPLIT.
    # The queries are nearly identical (off-diag cos ~0.9987), so training on all
    # 10 and testing on all 10 would only measure MEMORIZATION. The honest test:
    # train on 8 turns, test on 2 HELD-OUT turns. If the learned index generalizes,
    # the held-out targets should rank better than cosine.
    print(f"\n[learn] === LEARNED: InfoNCE projection (dim={args.dim}, steps={args.steps}) ===")
    nq = len(Q)
    test_idx = [nq - 2, nq - 1]          # hold out the last 2 turns
    train_idx = [i for i in range(nq) if i not in test_idx]
    Qtr = Q[train_idx]
    tgt_tr = np.array([targets[i] for i in train_idx])
    print(f"  train turns: {[turns[i]['turn'] for i in train_idx]}")
    print(f"  test  turns: {[turns[i]['turn'] for i in test_idx]} (held out)")
    P, loss_curve = info_nce_train(Qtr, C, tgt_tr, args.dim, args.steps, args.lr, args.tau)
    print(f"  loss: start={loss_curve[0]:.4f} end={loss_curve[-1]:.4f}")

    def learned_rank(i):
        Qp = l2norm(Q[i:i+1] @ P)
        Cp = l2norm(C @ P)
        return rank_of((Qp @ Cp.T)[0], targets[i])

    train_ranks = [learned_rank(i) for i in train_idx]
    test_ranks = [learned_rank(i) for i in test_idx]
    for i in train_idx:
        print(f"  [train] turn {turns[i]['turn']:2d} code={turns[i]['memory_token']} "
              f"chunk {targets[i]:4d} learn_rank={train_ranks[train_idx.index(i)]:4d}/{n_chunks}")
    for i in test_idx:
        print(f"  [test ] turn {turns[i]['turn']:2d} code={turns[i]['memory_token']} "
              f"chunk {targets[i]:4d} learn_rank={test_ranks[test_idx.index(i)]:4d}/{n_chunks}")
    print(f"  mean TRAIN  learn_rank = {np.mean(train_ranks):.1f}  (memorization)")
    print(f"  mean TEST   learn_rank = {np.mean(test_ranks):.1f}  (generalization)")

    # Verdict — based on the HELD-OUT (test) ranks vs cosine
    print(f"\n[learn] === VERDICT ===")
    cos_mean = float(np.mean(cos_ranks))
    test_mean = float(np.mean(test_ranks))
    train_mean = float(np.mean(train_ranks))
    if test_mean < cos_mean * 0.5:
        verdict = (f"LEARNED INDEX GENERALIZES — held-out mean rank {test_mean:.1f} vs cosine "
                   f"{cos_mean:.1f} ({cos_mean/test_mean:.1f}x better); the index is learnable")
    elif test_mean < cos_mean:
        verdict = (f"LEARNED INDEX HELPS ON HELD-OUT — mean rank {test_mean:.1f} vs cosine "
                   f"{cos_mean:.1f}; partial generalization, needs more data or a deeper index")
    else:
        verdict = (f"LEARNED INDEX DOES NOT GENERALIZE — held-out mean rank {test_mean:.1f} vs "
                   f"cosine {cos_mean:.1f} (train {train_mean:.1f}). The near-identical queries "
                   f"(off-diag cos {off_diag:.4f}) give a linear projection no disambiguation "
                   f"signal; it memorizes train but can't pick the right held-out fact.")
    print(f"  -> {verdict}")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_learned_index_result.json"
    out_path.write_text(json.dumps({
        "test": "learned_index", "name": "learned_retrieval_index",
        "num_turns": args.num_turns, "context_tokens": n, "n_chunks": n_chunks,
        "dim": args.dim, "steps": args.steps, "lr": args.lr, "tau": args.tau,
        "query_offdiag_cosine": round(float(off_diag), 4),
        "targets": targets,
        "train_turns": [turns[i]['turn'] for i in train_idx],
        "test_turns": [turns[i]['turn'] for i in test_idx],
        "cos_ranks": cos_ranks,
        "train_learn_ranks": train_ranks, "test_learn_ranks": test_ranks,
        "cos_mean_rank": round(cos_mean, 2),
        "train_mean_rank": round(train_mean, 2), "test_mean_rank": round(test_mean, 2),
        "loss_start": round(loss_curve[0], 4), "loss_end": round(loss_curve[-1], 4),
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"\n[learn] wrote {out_path}")


if __name__ == "__main__":
    main()
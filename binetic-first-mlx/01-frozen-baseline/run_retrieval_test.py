"""Bet 6 — Query-conditional retrieval over a bank of the model's own KV.

The deep finding from the experiment log: salience is NOT a property of the
token — it is a property of the (token, query) pair. Every INTRINSIC
(write-time) salience signal failed:
  - per-token surprisal        -> ranks structural surprises, not facts
  - temporal signature (dS)    -> better, but still query-agnostic
  - linear recurrent state     -> lossy gist, not a fact store

The only frame that escapes that dead end is QUERY-CONDITIONAL retrieval:
  - store the model's OWN KV (already in its computation space) as a bank
  - at query time, retrieve the chunks most relevant to the QUERY
  - inject the retrieved KV into a WARM cache (warm linear state)

The crux risk (stated plainly): it all hinges on whether
  cosine(query_hidden_state, fact_chunk_hidden_state)
ranks the fact chunk highly. The query is a QUESTION, the fact is a
STATEMENT — their representations may not be close. That is the whole bet.

This script is a clean A/B against the temporal-signature baseline:
  - retrieval  : top-k chunks by cosine(query_hidden, chunk_mean_hidden)
  - tsig       : top-k chunks by temporal signature (the intrinsic control)
  - random     : top-k random chunks (the null)
  - inplace    : full KV (sanity, should be 100%)

If retrieval recalls the fact at a SMALLER k than tsig, it is a win. If it
cannot beat tsig, the index must be LEARNED (the next bet).

First thing verified (the log flagged this): model.model(x, cache=...) returns
the normalized residual stream (1, n, hidden_size) and populates the same
cache. model(x, cache=...) applies lm_head -> logits. We use model.model for
hidden states.

Usage:
  /opt/homebrew/bin/python3.12 run_retrieval_test.py
  /opt/homebrew/bin/python3.12 run_retrieval_test.py --chunk 8 --ks 8 16 32
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


def snapshot_linear_states(cache):
    """Copy the linear (ArraysCache) state for all linear layers -> {i: [arrays]}."""
    snap = {}
    for i in LINEAR_LAYERS:
        c = cache[i]
        state = c.state
        if state is None:
            snap[i] = None
        else:
            snap[i] = [np.asarray(a) if a is not None else None for a in state]
    return snap


def delta_norm(snap_a, snap_b):
    """Total Frobenius norm of the state delta across all linear layers."""
    total = 0.0
    for i in LINEAR_LAYERS:
        a, b = snap_a[i], snap_b[i]
        if a is None and b is None:
            continue
        if a is None:
            a = [np.zeros_like(x) for x in b if x is not None]
            b = [x for x in b if x is not None]
        elif b is None:
            b = [np.zeros_like(x) for x in a if x is not None]
            a = [x for x in a if x is not None]
        else:
            a = [x for x in a if x is not None]
            b = [x for x in b if x is not None]
        for x, y in zip(a, b):
            total += float(np.sum((x - y) ** 2))
    return float(np.sqrt(total))


def process_in_chunks(model, context_ids, chunk_size):
    """Process in chunks; return (cache, per-chunk temporal signature).

    Uses model.model (hidden states) so we ALSO get the residual stream for
    the retrieval index. We keep the hidden states per chunk.
    """
    import mlx.core as mx

    cache = model.make_cache()
    hidden_chunks = []
    signature = []
    prev_snap = snapshot_linear_states(cache)
    for i in range(0, len(context_ids), chunk_size):
        chunk = context_ids[i:i + chunk_size]
        x = mx.array([chunk], dtype=mx.int32)
        h = model.model(x, cache=cache)  # (1, len(chunk), hidden)
        mx.eval(h)
        hidden_chunks.append(np.asarray(h, dtype=np.float32))
        snap = snapshot_linear_states(cache)
        signature.append(delta_norm(prev_snap, snap))
        prev_snap = snap
    return cache, hidden_chunks, signature


def build_chunk_index(hidden_chunks, chunk_size, n_total):
    """Per-chunk mean hidden state -> (n_chunks, hidden). L2-normalized rows."""
    vecs = []
    for h in hidden_chunks:
        vecs.append(h[0].mean(axis=0))  # (hidden,)
    idx = np.stack(vecs, axis=0)  # (n_chunks, hidden)
    norms = np.linalg.norm(idx, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return idx / norms


def query_hidden_state(model, query_ids):
    """Mean hidden state over the query tokens (the 'query vector')."""
    import mlx.core as mx

    cache = model.make_cache()
    x = mx.array([query_ids], dtype=mx.int32)
    h = model.model(x, cache=cache)  # (1, nq, hidden)
    mx.eval(h)
    v = np.asarray(h, dtype=np.float32)[0].mean(axis=0)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def build_warm_cache(model, cache_full, token_positions):
    """Fresh cache: WARM final linear state + only the selected full-attn KV."""
    import mlx.core as mx

    cache = model.make_cache()
    for i in LINEAR_LAYERS:
        cache[i].state = cache_full[i].state
    sel = mx.array(sorted(set(int(p) for p in token_positions)), dtype=mx.int32)
    for i in FULL_ATTENTION_LAYERS:
        c = cache_full[i]
        if c.keys is None:
            continue
        k, v = c.state
        cache[i].state = (mx.take(k, sel, axis=2), mx.take(v, sel, axis=2))
    for i in FULL_ATTENTION_LAYERS:
        mx.eval(cache[i].keys, cache[i].values)
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
    ap.add_argument("--target-turn", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=8, help="chunk size for index/signature")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--window", type=int, default=16, help="window around top chunks")
    args = ap.parse_args()

    print("[retr] loading model...")
    model, config, tok = load_model_and_tokenizer()

    template_path = PACK_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())

    turns = make_turns(args.num_turns, args.tokens_per_turn)
    target = turns[args.target_turn]["memory_token"]
    full_prompt = " ".join(t["prompt"] for t in turns)
    query_text = f"What was the secret code from turn {args.target_turn}? Just the number."

    rendered = render_chat(template, full_prompt)
    context_ids = tok.encode(rendered).ids
    query_ids = tok.encode(render_chat(template, query_text)).ids
    n = len(context_ids)
    n_chunks = (n + args.chunk - 1) // args.chunk
    print(f"[retr] {args.num_turns} turns, context={n} tok, target turn={args.target_turn} "
          f"code={target}, chunk={args.chunk}, n_chunks={n_chunks}")

    # Locate the code chunk (for reporting its retrieval rank)
    enc = tok.encode(rendered)
    offsets = enc.offsets
    code_str = str(target)
    pos = rendered.find(code_str)
    if pos < 0:
        print(f"[retr] ERROR: code {target} not found")
        return
    end = pos + len(code_str)
    code_span = [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]
    code_chunk = min(code_span) // args.chunk
    print(f"[retr] code '{target}' span={code_span} -> chunk {code_chunk}")

    # ONE context pass: hidden states (for index) + temporal signature + full cache
    t0 = time.time()
    cache_full, hidden_chunks, signature = process_in_chunks(model, context_ids, args.chunk)
    print(f"[retr] chunked forward: {time.time()-t0:.1f}s, {len(signature)} chunks")

    # Retrieval index: per-chunk mean hidden state
    chunk_index = build_chunk_index(hidden_chunks, args.chunk, n)
    # Query vector
    qv = query_hidden_state(model, query_ids)
    # Retrieval scores: cosine(query, chunk)
    scores = chunk_index @ qv  # (n_chunks,)
    ret_rank = int((scores > scores[code_chunk]).sum()) + 1
    sig = np.asarray(signature)
    sig_rank = int((sig > sig[code_chunk]).sum()) + 1
    print(f"\n[retr] === SIGNAL: code chunk {code_chunk} ranking ===")
    print(f"  retrieval (cosine q~chunk): score={scores[code_chunk]:.4f} rank={ret_rank}/{n_chunks}")
    print(f"  temporal signature:         value={sig[code_chunk]:.4f} rank={sig_rank}/{n_chunks}")

    # Top-10 retrieval chunks and their content
    top10 = sorted(range(n_chunks), key=lambda i: -scores[i])[:10]
    print(f"\n[retr] top-10 retrieval chunks:")
    for i in top10:
        lo, hi = i * args.chunk, min(n, (i + 1) * args.chunk)
        content = tok.decode(context_ids[lo:hi])[:40]
        marker = "  <-- CODE" if i == code_chunk else ""
        print(f"  chunk {i:4d} cos={scores[i]:.4f} {content!r}{marker}")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_retrieval_result.json"
    results = {
        "code_chunk": code_chunk, "code_span": code_span,
        "ret_rank": ret_rank, "sig_rank": sig_rank, "n_chunks": n_chunks,
        "top10_retrieval": [{"chunk": i, "cos": round(float(scores[i]), 4),
                             "content": tok.decode(context_ids[i * args.chunk:min(n, (i + 1) * args.chunk)])[:40]}
                            for i in top10],
    }

    # MECHANISM: warm state + KV of selected chunks, for each selector x k
    print(f"\n[retr] === MECHANISM: warm state + selected KV ===")
    # sanity: in-place
    decoded = generate(model, tok, cache_full, query_ids)
    correct, actual = check_recall(decoded, target)
    results["inplace"] = {"n": n, "recall": correct, "actual": actual, "text": decoded[:80]}
    print(f"  inplace (sanity): recall={correct} actual={actual}")

    rng = np.random.default_rng(42)
    for k in args.ks:
        # retrieval selection
        ret_topk = sorted(range(n_chunks), key=lambda i: -scores[i])[:k]
        # tsig selection
        sig_topk = sorted(range(n_chunks), key=lambda i: -sig[i])[:k]
        # random selection
        rand_topk = sorted(rng.choice(n_chunks, size=k, replace=False).tolist())

        for label, topk in (("retrieval", ret_topk), ("tsig", sig_topk), ("random", rand_topk)):
            # plain: the top-k chunks' tokens
            plain_pos = [p for i in topk for p in range(i * args.chunk, min(n, (i + 1) * args.chunk))]
            # window: +/-window tokens around each top chunk's center
            win_pos = set()
            for i in topk:
                center = i * args.chunk + args.chunk // 2
                win_pos.update(range(max(0, center - args.window), min(n, center + args.window)))
            for sub, positions in ((f"{label}_k{k}", plain_pos),
                                   (f"{label}_k{k}_win{args.window}", sorted(win_pos))):
                cache_sel = build_warm_cache(model, cache_full, positions)
                decoded = generate(model, tok, cache_sel, query_ids)
                correct, actual = check_recall(decoded, target)
                code_in_sel = sum(1 for i in code_span if i in set(int(p) for p in positions))
                results[sub] = {"n": len(positions), "recall": correct, "actual": actual,
                                "code_in_selection": code_in_sel, "text": decoded[:80]}
                print(f"  {sub:<22} n={len(positions):5d} code_in_sel={code_in_sel}/{len(code_span)} "
                      f"recall={correct} actual={actual} text={decoded[:40]!r}")
                del cache_sel
                mx.metal.clear_cache()

    # Verdict: does retrieval beat tsig at the same k?
    print(f"\n[retr] === VERDICT ===")
    def best_k(method):
        for k in args.ks:
            for sub in (f"{method}_k{k}", f"{method}_k{k}_win{args.window}"):
                if results.get(sub, {}).get("recall"):
                    return k
        return None
    ret_k = best_k("retrieval")
    sig_k = best_k("tsig")
    if ret_k is not None and (sig_k is None or ret_k <= sig_k):
        verdict = (f"RETRIEVAL WINS — recalls at k={ret_k} (tsig best k={sig_k}); "
                   "query-conditional retrieval is a usable selector")
    elif ret_k is not None and sig_k is not None and ret_k > sig_k:
        verdict = (f"RETRIEVAL WORKS but tsig is better (retrieval k={ret_k} vs tsig k={sig_k}) — "
                   "cosine index is not the right similarity; consider a learned index")
    elif ret_k is None and sig_k is not None:
        verdict = ("RETRIEVAL FAILS, tsig works — cosine(query, chunk) does not rank the fact; "
                   "the index must be learned")
    else:
        verdict = "both fail — inspect ret_rank and top10_retrieval"
    print(f"  -> {verdict}")

    results["verdict"] = verdict
    results["retrieval_best_k"] = ret_k
    results["tsig_best_k"] = sig_k
    results["target"] = target
    results["target_turn"] = args.target_turn
    results["context_tokens"] = n
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[retr] wrote {out_path}")


if __name__ == "__main__":
    main()
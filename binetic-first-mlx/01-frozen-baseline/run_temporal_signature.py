"""Temporal-Signature selective KV.

First-principles premise:
  A fact's signature is NOT in any single token (per-token surprisal failed:
  the code '5211' = 4 common digit-tokens ranked in the BOTTOM third of
  surprisal). It is in HOW THE MODEL'S MEMORY STATE EVOLVES over the segment.

  The 48 GatedDeltaNet layers carry a recurrent state S that is the model's
  literal temporal memory. The TEMPORAL SIGNATURE of a segment is the delta it
  induces in that state:  dS = S_after - S_before.
    - padding / boilerplate -> small, unremarkable state change
    - a distinctive fact    -> a larger, more distinctive state change

  This is a TEMPORAL signal (a change over time), computed in the model's OWN
  representational space, and it is FREE (the state is already computed).

Two questions, cleanly separated (same as the SGM diagnostic):
  (SIGNAL)   Does the state-delta signature rank the fact chunk HIGHLY?
             (compare against surprisal, which we know fails)
  (MECHANISM) With a WARM linear state + the full-attn KV of the top-signature
             chunks, does recall work?

Design:
  - Process the context in chunks of `chunk` tokens, snapshotting the linear
    state after each chunk. signature[i] = ||S_after_i - S_before_i||_F.
  - Locate the code chunk (via offset mapping). Report its signature rank AND
    its surprisal rank (the comparison that matters).
  - Mechanism: build a cache with the WARM final linear state + only the KV of
    the top-k signature chunks (and a +/-window variant). Query.

Usage:
  /opt/homebrew/bin/python3.12 run_temporal_signature.py
  /opt/homebrew/bin/python3.12 run_temporal_signature.py --chunk 8 --ks 8 16 32
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
    """Process in chunks; return (cache, concatenated logits, per-chunk signature)."""
    import mlx.core as mx

    cache = model.make_cache()
    all_logits = []
    signature = []
    prev_snap = snapshot_linear_states(cache)  # initial (empty) state
    for i in range(0, len(context_ids), chunk_size):
        chunk = context_ids[i:i + chunk_size]
        x = mx.array([chunk], dtype=mx.int32)
        logits = model(x, cache=cache)
        mx.eval(logits)
        all_logits.append(logits)
        snap = snapshot_linear_states(cache)
        signature.append(delta_norm(prev_snap, snap))
        prev_snap = snap
    all_logits = mx.concatenate(all_logits, axis=1)
    return cache, all_logits, signature


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
    enc = tok.encode(rendered)
    ids, offsets = enc.ids, enc.offsets
    code_str = str(code)
    pos = rendered.find(code_str)
    if pos < 0:
        return None
    end = pos + len(code_str)
    return [i for i, (s, e) in enumerate(offsets) if s < end and e > pos]


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
    ap.add_argument("--chunk", type=int, default=8, help="chunk size for the signature")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--window", type=int, default=16, help="window around top chunks")
    args = ap.parse_args()

    print("[tsig] loading model...")
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
    print(f"[tsig] {args.num_turns} turns, context={n} tok, target turn={args.target_turn} "
          f"code={target}, chunk={args.chunk}")

    code_span = find_code_span(rendered, target, tok)
    if code_span is None:
        print(f"[tsig] ERROR: code {target} not found")
        return
    code_chunk = min(code_span) // args.chunk
    print(f"[tsig] code '{target}' span={code_span} -> chunk {code_chunk} "
          f"({[tok.decode([context_ids[i]]) for i in code_span]})")

    # Process in chunks: signature curve + logits
    t0 = time.time()
    cache_full, logits, signature = process_in_chunks(model, context_ids, args.chunk)
    print(f"[tsig] chunked forward: {time.time()-t0:.1f}s, {len(signature)} chunks")

    surprisal = compute_surprisal(logits, context_ids)

    # (SIGNAL) rank of the code chunk by signature vs surprisal
    sig = np.asarray(signature)
    sig_rank = int((sig > sig[code_chunk]).sum()) + 1
    # surprisal rank of the code chunk = mean surprisal over its tokens, ranked
    code_surp = float(np.mean([surprisal[i] for i in code_span]))
    # rank the chunk by its mean token surprisal among all chunks
    chunk_surp = np.array([float(np.mean(surprisal[i * args.chunk:(i + 1) * args.chunk]))
                           for i in range(len(signature))])
    surp_rank = int((chunk_surp > chunk_surp[code_chunk]).sum()) + 1
    print(f"\n[tsig] === SIGNAL: code chunk {code_chunk} ranking ===")
    print(f"  signature:  value={sig[code_chunk]:.4f}  rank={sig_rank}/{len(signature)} "
          f"(1=highest)")
    print(f"  surprisal:  value={code_surp:.4f}  rank={surp_rank}/{len(signature)} "
          f"(1=highest)")

    # Top-10 signature chunks and their content
    top10 = sorted(range(len(signature)), key=lambda i: -sig[i])[:10]
    print(f"\n[tsig] top-10 signature chunks:")
    for i in top10:
        lo, hi = i * args.chunk, min(n, (i + 1) * args.chunk)
        content = tok.decode(context_ids[lo:hi])[:40]
        marker = "  <-- CODE" if i == code_chunk else ""
        print(f"  chunk {i:4d} sig={sig[i]:.4f} {content!r}{marker}")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_temporal_signature_result.json"
    results = {
        "code_chunk": code_chunk, "code_span": code_span,
        "sig_rank": sig_rank, "surp_rank": surp_rank,
        "n_chunks": len(signature),
        "top10_signature": [{"chunk": i, "sig": round(float(sig[i]), 4),
                             "content": tok.decode(context_ids[i * args.chunk:min(n, (i + 1) * args.chunk)])[:40]}
                            for i in top10],
    }

    # (MECHANISM) warm state + KV of top-k signature chunks
    print(f"\n[tsig] === MECHANISM: warm state + top-k signature KV ===")
    # sanity: in-place
    decoded = generate(model, tok, cache_full, query_ids)
    correct, actual = check_recall(decoded, target)
    results["inplace"] = {"n": n, "recall": correct, "actual": actual, "text": decoded[:80]}
    print(f"  inplace (sanity): recall={correct} actual={actual}")

    for k in args.ks:
        topk = sorted(range(len(signature)), key=lambda i: -sig[i])[:k]
        # plain: the top-k chunks' tokens
        plain_pos = [p for i in topk for p in range(i * args.chunk, min(n, (i + 1) * args.chunk))]
        # window: +/-window tokens around each top chunk's center
        win_pos = set()
        for i in topk:
            center = i * args.chunk + args.chunk // 2
            win_pos.update(range(max(0, center - args.window), min(n, center + args.window)))
        for label, positions in ((f"topk{k}", plain_pos), (f"topk{k}_win{args.window}", sorted(win_pos))):
            cache_sel = build_warm_cache(model, cache_full, positions)
            decoded = generate(model, tok, cache_sel, query_ids)
            correct, actual = check_recall(decoded, target)
            code_in_sel = sum(1 for i in code_span if i in set(int(p) for p in positions))
            results[label] = {"n": len(positions), "recall": correct, "actual": actual,
                              "code_in_selection": code_in_sel, "text": decoded[:80]}
            print(f"  {label:<16} n={len(positions):5d} code_in_sel={code_in_sel}/{len(code_span)} "
                  f"recall={correct} actual={actual} text={decoded[:40]!r}")
            del cache_sel
            mx.metal.clear_cache()

    # Verdict
    print(f"\n[tsig] === VERDICT ===")
    sig_is_better = sig_rank < surp_rank
    mech_ok = any(results[l]["recall"] for l in results
                  if l.startswith("topk") and "win" not in l)
    mech_win_ok = any(results[l]["recall"] for l in results if l.startswith("topk") and "win" in l)
    if sig_is_better and (mech_ok or mech_win_ok):
        verdict = ("TEMPORAL SIGNATURE WORKS — the state-delta ranks the fact higher than "
                   "surprisal AND selecting by it (warm state) supports recall")
    elif sig_is_better:
        verdict = ("signature is a BETTER signal than surprisal, but the mechanism still fails "
                   "— the fact needs more than its chunk's KV (inspect code_in_selection)")
    elif mech_ok or mech_win_ok:
        verdict = ("mechanism works but the signature isn't the best selector — the fact was "
                   "selected by luck/position, not by signature rank")
    else:
        verdict = ("both fail — the state-delta is not a usable salience signal at this "
                   "granularity; inspect top10_signature to see what dominates")
    print(f"  signature rank {sig_rank} vs surprisal rank {surp_rank} "
          f"({'signature better' if sig_is_better else 'surprisal better'})")
    print(f"  -> {verdict}")

    results["verdict"] = verdict
    results["target"] = target
    results["target_turn"] = args.target_turn
    results["context_tokens"] = n
    results["chunk"] = args.chunk
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[tsig] wrote {out_path}")


if __name__ == "__main__":
    main()
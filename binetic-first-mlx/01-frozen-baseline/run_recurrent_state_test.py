"""Recurrent-state memory bet: can the 48 GatedDeltaNet layers' carried
recurrent state support long-range exact recall when the 16 full-attention
layers' KV is discarded between chunks?

First-principles premise:
  Bonsai 2 is a hybrid — 48 linear-attention (GatedDeltaNet) layers with a
  FIXED-SIZE recurrent state (~157 MB total, sequence-length-independent) +
  16 full-attention layers with a GROWING KV cache. The existing harness
  (tests 1-16) only ever touches the full-attention KV and discards the linear
  state between chunks. This test asks whether the linear state — the model's
  own native long-term memory — is sufficient for recall on its own.

Three-way design (same synthetic secret-code task as test 1/5):
  A (in-place):      process all context in one cache, no resets.
                     Full-attn KV has everything. Sanity — expect 100%.
  B (THE BET):       process in chunks; CARRY the linear state across chunks
                     (it accumulates naturally) but RESET the full-attention KV
                     to empty after each chunk. After the context, the linear
                     state summarizes ALL chunks; the full-attn KV only has the
                     LAST chunk. The turn-0 secret is only reachable via the
                     carried linear state.
  C (control):       process in chunks; reset BOTH linear state and full-attn KV
                     after each chunk. Nothing from early chunks survives.
                     Expect ~0% (establishes the gap B must close).

If B >> C and B ~ A, the linear recurrent state is the long-term memory
substrate — and the architecture's 48/64 layers were doing the remembering all
along.

Usage:
  /opt/homebrew/bin/python3.12 run_recurrent_state_test.py            # 10 turns (~5.4K tok)
  /opt/homebrew/bin/python3.12 run_recurrent_state_test.py --long     # 20 turns (~10.8K tok)
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
sys.path.insert(0, str(Path.home() / "Documents" / "binetic-ai" / "src"))

EOS = 248044
LINEAR_STATE_MB = 156.89  # measured by probe_recurrent_state.py (48 layers, fixed)


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


def generate(model, tok, cache, query_ids, max_tokens=32):
    """Greedy argmax generation from a cache, returning decoded text + tokens."""
    import mlx.core as mx

    x = mx.array([query_ids], dtype=mx.int32)
    logits = model(x, cache=cache)
    mx.eval(logits)
    gen_tokens = []
    for _ in range(max_tokens):
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        gen_tokens.append(next_tok)
        if next_tok == EOS:
            break
        x = mx.array([[next_tok]], dtype=mx.int32)
        logits = model(x, cache=cache)
        mx.eval(logits)
    return tok.decode(gen_tokens), gen_tokens


def check_recall(decoded, expected):
    numbers = re.findall(r"\b(\d{4})\b", decoded)
    actual = int(numbers[0]) if numbers else None
    return (actual == expected), actual


def reset_full_attn_kv(cache, full_layers):
    """Replace full-attention KV caches with fresh empty ones (keep linear state)."""
    from mlx_lm.models.cache import KVCache
    for i in full_layers:
        cache[i] = KVCache()


def reset_linear_state(cache, linear_layers):
    """Replace linear-attention caches with fresh empty ones (keep full-attn KV)."""
    from mlx_lm.models.cache import ArraysCache
    for i in linear_layers:
        cache[i] = ArraysCache(2)


def run_variant(model, tok, template, context_ids, query_ids, expected,
                chunk_size, mode, full_layers, linear_layers):
    """Run one variant. mode in {'inplace', 'carry_linear', 'reset_both'}."""
    import mlx.core as mx

    t0 = time.time()
    cache = model.make_cache()
    batch_size = 256

    if mode == "inplace":
        # Process the whole context in one cache, no resets.
        for i in range(0, len(context_ids), chunk_size):
            chunk = context_ids[i:i + chunk_size]
            for j in range(0, len(chunk), batch_size):
                batch = chunk[j:j + batch_size]
                x = mx.array([batch], dtype=mx.int32)
                logits = model(x, cache=cache)
                mx.eval(logits)
    else:
        # Chunked: process each chunk, then reset per mode.
        for i in range(0, len(context_ids), chunk_size):
            chunk = context_ids[i:i + chunk_size]
            for j in range(0, len(chunk), batch_size):
                batch = chunk[j:j + batch_size]
                x = mx.array([batch], dtype=mx.int32)
                logits = model(x, cache=cache)
                mx.eval(logits)
            if mode == "carry_linear":
                reset_full_attn_kv(cache, full_layers)
            elif mode == "reset_both":
                reset_full_attn_kv(cache, full_layers)
                reset_linear_state(cache, linear_layers)

    context_time = time.time() - t0

    # Query: feed only the query tokens; the cache holds whatever survived.
    decoded, _ = generate(model, tok, cache, query_ids)
    correct, actual = check_recall(decoded, expected)

    return {
        "mode": mode,
        "chunk_size": chunk_size,
        "context_tokens": len(context_ids),
        "context_time_s": round(context_time, 2),
        "recall_correct": correct,
        "recall_accuracy": 1.0 if correct else 0.0,
        "expected_value": expected,
        "actual_value": actual,
        "generated_text": decoded[:100],
    }


def main():
    import mlx.core as mx
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="20 turns (~10.8K tok) instead of 10")
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--variant", choices=["inplace", "carry_linear", "reset_both", "all"],
                    default="all", help="run a single variant (for re-runs after OOM)")
    args = ap.parse_args()

    num_turns = 20 if args.long else 10
    tokens_per_turn = 1000
    chunk_size = args.chunk_size

    print(f"[recurrent-state] loading model...")
    model, config, tok = load_model_and_tokenizer()
    tc = config["text_config"]
    fa_interval = tc["full_attention_interval"]
    n_layers = tc["num_hidden_layers"]
    full_layers = [i for i in range(n_layers) if (i + 1) % fa_interval == 0]
    linear_layers = [i for i in range(n_layers) if (i + 1) % fa_interval != 0]

    template_path = PACK_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())

    turns = make_synthetic_recall_data(num_turns, tokens_per_turn)
    full_prompt = " ".join(t["prompt"] for t in turns)
    query_text = "What was the secret code from turn 0? Just the number."
    expected = turns[0]["memory_token"]

    context_ids = tok.encode(render_chat(template, full_prompt)).ids
    query_ids = tok.encode(render_chat(template, query_text)).ids
    print(f"[recurrent-state] {num_turns} turns, context={len(context_ids)} tok, "
          f"query={len(query_ids)} tok, chunk={chunk_size}, expected={expected}")
    print(f"[recurrent-state] linear state footprint (fixed): {LINEAR_STATE_MB} MB")

    results_dir = BASELINE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "test_recurrent_state_result.json"

    # Load any prior checkpoint so single-variant re-runs merge cleanly
    prior = {}
    if out_path.is_file():
        try:
            prior = json.loads(out_path.read_text()).get("variants", {})
        except Exception:
            prior = {}

    modes = ["inplace", "carry_linear", "reset_both"] if args.variant == "all" else [args.variant]
    for mode in modes:
        print(f"\n[recurrent-state] === variant {mode} ===")
        r = run_variant(model, tok, template, context_ids, query_ids, expected,
                        chunk_size, mode, full_layers, linear_layers)
        prior[mode] = r
        # Checkpoint after every variant so an OOM mid-run doesn't lose work
        out_path.write_text(json.dumps({
            "test": "recurrent_state",
            "name": "recurrent_state_memory_bet",
            "num_turns": num_turns,
            "tokens_per_turn": tokens_per_turn,
            "context_tokens": len(context_ids),
            "chunk_size": chunk_size,
            "linear_state_mb": LINEAR_STATE_MB,
            "variants": prior,
            "complete": set(prior) == {"inplace", "carry_linear", "reset_both"},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        print(f"  recall={r['recall_correct']} expected={r['expected_value']} "
              f"actual={r['actual_value']} time={r['context_time_s']}s")
        print(f"  text: {r['generated_text']!r}")
        print(f"  [checkpointed -> {out_path.name}]")

    # Verdict (only when all three are present)
    if set(prior) == {"inplace", "carry_linear", "reset_both"}:
        a = prior["inplace"]; b = prior["carry_linear"]; c = prior["reset_both"]
        print("\n[recurrent-state] === VERDICT ===")
        print(f"  A in-place (sanity):      {a['recall_accuracy']:.0%}")
        print(f"  B carry-linear (THE BET): {b['recall_accuracy']:.0%}")
        print(f"  C reset-both (control):   {c['recall_accuracy']:.0%}")
        if b["recall_accuracy"] >= 0.75 and c["recall_accuracy"] < 0.25:
            verdict = "LINEAR STATE IS THE SUBSTRATE — carried recurrent state supports recall"
        elif b["recall_accuracy"] >= 0.75:
            verdict = "carry-linear works but control also works — inconclusive"
        else:
            verdict = "linear state alone insufficient — full-attn KV required"
        print(f"  -> {verdict}")
        out_path.write_text(json.dumps({
            "test": "recurrent_state",
            "name": "recurrent_state_memory_bet",
            "num_turns": num_turns,
            "tokens_per_turn": tokens_per_turn,
            "context_tokens": len(context_ids),
            "chunk_size": chunk_size,
            "linear_state_mb": LINEAR_STATE_MB,
            "variants": prior,
            "verdict": verdict,
            "complete": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))
        print(f"\n[recurrent-state] wrote {out_path}")
    else:
        print(f"\n[recurrent-state] partial run — {sorted(prior)} done, "
              f"{sorted(set(modes) - set(prior))} pending. Re-run with --variant to continue.")


if __name__ == "__main__":
    main()
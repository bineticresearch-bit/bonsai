"""Memory-as-weights (LoRA) bet: can a fact be written into the model's WEIGHTS
so it's recalled with an EMPTY cache (no KV, no context)?

The 27B Bonsai uses custom `Packed` ternary modules that mlx_lm.lora cannot
wrap (it only handles nn.Linear / nn.QuantizedLinear / nn.Embedding), so this
test uses a small standard model (Qwen2.5-0.5B-Instruct) as the substrate.
The question is mechanistic, not about Bonsai specifically: does gradient
training write a specific fact into the weights?

Decisive design — two-code A/B/C:
  base:   untrained model, query "What is the secret code?" -> expect no code
  A:      LoRA trained on "the secret code is 8270" -> query -> expect 8270
  B:      LoRA trained on "the secret code is 1234" -> query -> expect 1234

If A recalls 8270 AND B recalls 1234 (and base recalls neither), the weight is
provably carrying the SPECIFIC fact — not a generic "answer with a number"
prior. That's the memory-as-weights mechanism working.

The loss mask trains ONLY on the answer tokens (the code), so the model learns
to predict the code given the question — exactly the recall we test.

Usage:
  /opt/homebrew/bin/python3.12 run_lora_memory_test.py            # train + query
  /opt/homebrew/bin/python3.12 run_lora_memory_test.py --skip-train  # query only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

BASELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASELINE_DIR / "results"
ADAPTER_DIR = BASELINE_DIR / "lora_adapters"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTION = "What is the secret code?"
CODE_A = 8270
CODE_B = 1234


def load_base():
    import mlx_lm
    model, tok = mlx_lm.load(MODEL_NAME)
    return model, tok


def make_dataset(tok, code: int, n_examples: int = 16):
    """Each example: (input_ids, loss_start) — a 2-tuple where the 2nd element is
    a SINGLE INT (loss_start). iterate_batches zips (loss_start, token_length)
    into the (batch, 2) lengths array that default_loss masks on, so loss_end is
    implicitly the full sequence length.

    FIX (2026-10-09): loss_start now masks the ENTIRE assistant turn, not just
    the last 8 tokens. The old mask (len(ids)-8) gave the model "The secret
    code is" for free and only trained it to predict "8270" — a trivial
    completion. At inference the model must generate the WHOLE answer from the
    question, which was out-of-distribution -> degenerate collapse. Masking the
    full assistant turn makes training match inference (in-distribution)."""
    # Prompt = everything up to (and including) the assistant turn's opening
    # marker. loss_start = len(prompt) masks the entire assistant turn.
    prompt_ids = list(tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}], add_generation_prompt=True))
    loss_start = len(prompt_ids)
    ds = []
    for _ in range(n_examples):
        messages = [
            {"role": "user", "content": QUESTION},
            {"role": "assistant", "content": f"The secret code is {code}."},
        ]
        ids = list(tok.apply_chat_template(messages, add_generation_prompt=False))
        ds.append((ids, loss_start))
    return ds


def train_adapter(tok, code: int, iters: int = 200, lr: float = 1e-3,
                  adapter_file: str = "adapters.safetensors"):
    """Train a fresh LoRA adapter on one code. Loads a fresh base each call so
    LoRA layers are never double-wrapped. Returns the TRAINED model (in-process)
    — we query it directly rather than save/load, because mlx_lm.train only
    writes raw weights (no adapter_config.json) so load_adapters can't reload it."""
    import mlx_lm
    from mlx_lm.lora import linear_to_lora_layers, train
    from mlx_lm.tuner.trainer import TrainingArgs

    model, _ = mlx_lm.load(MODEL_NAME)
    lora_config = {"rank": 8, "scale": 10.0, "dropout": 0.0,
                   "pattern": {"q_proj", "k_proj", "v_proj", "o_proj"}}
    # NOTE: linear_to_lora_layers mutates `model` in place and returns None —
    # do NOT reassign.
    linear_to_lora_layers(model, num_layers=24, config=lora_config)
    ds = make_dataset(tok, code)
    opt = optim.AdamW(learning_rate=lr)
    args = TrainingArgs(batch_size=4, iters=iters, steps_per_report=50,
                        steps_per_eval=10**9, steps_per_save=10**9,
                        max_seq_length=256, adapter_file=adapter_file)
    t0 = time.time()
    train(model, opt, ds, args=args)
    print(f"[lora] trained code={code} in {time.time()-t0:.1f}s")
    return model


def query(model, tok, max_tokens=24):
    import mlx_lm
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}], add_generation_prompt=True)
    # Greedy: no sampler -> argmax. (generate_step takes a `sampler`, not `temperature`.)
    return mlx_lm.generate(model, tok, prompt=prompt, max_tokens=max_tokens)


def check_code(text, code):
    nums = re.findall(r"\b(\d{4})\b", text)
    actual = int(nums[0]) if nums else None
    return (actual == code), actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    ADAPTER_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "test_lora_memory_result.json"

    print("[lora] loading base model...")
    base_model, tok = load_base()

    results = {}
    if out_path.is_file():
        try:
            results = json.loads(out_path.read_text()).get("results", {})
        except Exception:
            results = {}

    # Query: base, A, B — all with EMPTY cache.
    # Train + query each adapter IN-PROCESS (train returns the trained model),
    # then free it before the next, to avoid holding two 0.5B models at once.
    import mlx_lm

    if "base" not in results:
        print("\n[lora] === base (untrained) ===")
        text = query(base_model, tok)
        correct, actual = check_code(text, CODE_A)
        results["base"] = {"text": text[:100], "actual": actual,
                           "matches_A": correct, "matches_B": check_code(text, CODE_B)[0]}
        print(f"  text={text[:60]!r} actual={actual}")

    for label, code in (("A", CODE_A), ("B", CODE_B)):
        key = f"adapter_{label}"
        if key in results:
            continue
        print(f"\n[lora] === adapter {label} (trained on {code}) ===")
        afile = str(ADAPTER_DIR / f"adapter_{code}.safetensors")
        if args.skip_train:
            # Without a loadable adapter dir we can't skip-train; just note it.
            print(f"  [skip-train] no loadable adapter for {code}; skipping")
            continue
        m = train_adapter(tok, code, args.iters, args.lr, afile)
        text = query(m, tok)
        correct, actual = check_code(text, code)
        results[key] = {"trained_code": code, "text": text[:100], "actual": actual,
                        "recalls_own_code": correct,
                        "recalls_other_code": check_code(text, CODE_A if label == "B" else CODE_B)[0]}
        print(f"  text={text[:60]!r} actual={actual} recalls_own={correct}")
        del m
        mx.metal.clear_cache()

    # Verdict
    print("\n[lora] === VERDICT ===")
    a = results.get("adapter_A", {})
    b = results.get("adapter_B", {})
    base = results.get("base", {})
    print(f"  base:  actual={base.get('actual')}")
    print(f"  A:     trained={CODE_A} actual={a.get('actual')} recalls_own={a.get('recalls_own_code')}")
    print(f"  B:     trained={CODE_B} actual={b.get('actual')} recalls_own={b.get('recalls_own_code')}")
    if a.get("recalls_own_code") and b.get("recalls_own_code"):
        verdict = ("MEMORY-AS-WEIGHTS WORKS — each adapter recalls its OWN specific "
                   "code with an empty cache; the fact is in the weights")
    elif a.get("recalls_own_code") or b.get("recalls_own_code"):
        verdict = "partial — one adapter recalls its code; check the other's training"
    else:
        verdict = ("memory-as-weights insufficient at this scale/iters — the weight "
                   "did not carry the specific fact (or the query didn't trigger it)")
    print(f"  -> {verdict}")

    out_path.write_text(json.dumps({
        "test": "lora_memory", "name": "memory_as_weights_lora",
        "model": MODEL_NAME, "question": QUESTION,
        "code_A": CODE_A, "code_B": CODE_B,
        "iters": args.iters, "lr": args.lr,
        "results": results, "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"\n[lora] wrote {out_path}")


if __name__ == "__main__":
    main()
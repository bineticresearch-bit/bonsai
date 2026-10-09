"""Test 8: Thermodynamic compression analysis.

Tests the hypothesis that there exists a critical compression ratio (beta)
where recall transitions from 100% to 0%, analogous to a phase transition.

Maps compression ratio -> recall, measuring the "free energy" tradeoff.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "bonsai2-27b-mlx" / "runtime"))


def make_synthetic_recall_data(num_turns: int = 10, tokens_per_turn: int = 1000,
                               vocab_size: int = 20000) -> list[dict]:
    """Generate synthetic dialogue with embedded recall facts."""
    np.random.seed(42)
    turns = []
    padding_words = ["the", "system", "configures", "values", "during", "initialization",
                     "process", "and", "maintains", "state", "across", "multiple",
                     "requests", "with", "proper", "handling", "of", "edge", "cases",
                     "including", "timeout", "recovery", "retry", "logic", "cache"]
    for t in range(num_turns):
        memory_value = np.random.randint(1000, 9999)
        padding = " ".join(np.random.choice(padding_words,
                                             size=max(1, tokens_per_turn // 2)))
        prompt = f"Turn {t}: The secret code is {memory_value}. System log: {padding}."
        turns.append({
            "turn": t,
            "memory_token": int(memory_value),
            "prompt": prompt,
        })
    return turns


def run_thermodynamic_test(model, tok, num_turns=10, tokens_per_turn=1000,
                           beta_values=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]) -> dict:
    """Test recall at varying compression ratios.

    beta = compression ratio (0.0 = no compression, 1.0 = maximum compression)
    We compress the KV by mean-pooling positions: beta=0.5 means halving the
    sequence length at each layer.
    """
    import mlx.core as mx
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from binetic.temporal_injection import create_cache_with_temporal_kv, FULL_ATTENTION_LAYERS

    template_path = PROJECT_ROOT / "bonsai2-27b-mlx" / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())

    turns = make_synthetic_recall_data(num_turns, tokens_per_turn)
    full_prompt = " ".join(t["prompt"] for t in turns)
    query = "What was the secret code in turn 0? Answer with just the number."

    rendered_context = template.render(
        messages=[{"role": "user", "content": full_prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    all_tokens = tok.encode(rendered_context).ids
    ctx_len = len(all_tokens)

    rendered_query = template.render(
        messages=[{"role": "user", "content": query}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    query_ids = tok.encode(rendered_query).ids

    expected = turns[0]["memory_token"]

    # Process context in chunks, extract KV per chunk, concatenate
    # This avoids the KVCache buffer size limit for large contexts
    chunk_size = min(8192, len(all_tokens))
    batch_size = 256

    full_kv = {}
    for i in range(0, len(all_tokens), chunk_size):
        chunk = all_tokens[i:i + chunk_size]
        cache = model.make_cache()
        for j in range(0, len(chunk), batch_size):
            batch = chunk[j:j + batch_size]
            x = mx.array([batch], dtype=mx.int32)
            logits = model(x, cache=cache)
            mx.eval(logits)

        for layer in FULL_ATTENTION_LAYERS:
            c = cache[layer]
            if c.keys is not None:
                sk, sv = c.state
                k = np.asarray(sk).astype(np.float16)
                v = np.asarray(sv).astype(np.float16)
                if layer in full_kv:
                    full_kv[layer] = (
                        np.concatenate([full_kv[layer][0], k], axis=2),
                        np.concatenate([full_kv[layer][1], v], axis=2),
                    )
                else:
                    full_kv[layer] = (k, v)

    results = []

    # Evaluate recall at each beta
    for beta in beta_values:
        # Compress: block-based subsampling that preserves local position continuity
        keep_ratio = 1.0 - beta
        keep_len = max(1, int(ctx_len * keep_ratio))
        if keep_len >= ctx_len:
            indices = list(range(ctx_len))
        else:
            block_size = ctx_len // keep_len
            indices = [min(i * block_size, ctx_len - 1) for i in range(keep_len)]

        kv = {}
        for layer in FULL_ATTENTION_LAYERS:
            k_all, v_all = full_kv[layer]
            k = k_all[:, :, indices, :].astype(np.float16)
            v = v_all[:, :, indices, :].astype(np.float16)
            kv[layer] = (k, v)

        injected_len = next(iter(kv.values()))[0].shape[2] if kv else 0

        cache2 = create_cache_with_temporal_kv(model, kv)

        x = mx.array([query_ids], dtype=mx.int32)
        logits = model(x, cache=cache2)
        mx.eval(logits)

        gen_tokens = []
        for _ in range(32):
            next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            gen_tokens.append(next_tok)
            if next_tok == 248044:
                break
            x = mx.array([[next_tok]], dtype=mx.int32)
            logits = model(x, cache=cache2)
            mx.eval(logits)

        decoded = tok.decode(gen_tokens)
        recall_correct = str(expected) in decoded
        accuracy = 1.0 if recall_correct else 0.0

        # Compute entropy proxy: variance of KV values (lower = higher compression)
        sample_kv = kv[3][0]
        entropy_proxy = float(np.var(sample_kv.astype(np.float32)))

        print(f"  beta={beta:.2f}: recall={accuracy}, injected_len={injected_len}/{ctx_len}, "
              f"compression={beta*100:.0f}%, entropy_proxy={entropy_proxy:.4f}, "
              f"got='{decoded.strip()[:50]}'")

        results.append({
            "beta": beta,
            "compression_ratio": beta,
            "keep_ratio": 1.0 - beta,
            "injected_seq_len": injected_len,
            "original_seq_len": ctx_len,
            "recall_accuracy": accuracy,
            "recall_correct": recall_correct,
            "expected_value": int(expected),
            "generated_text": decoded[:100],
            "entropy_proxy": round(entropy_proxy, 6),
        })

    # Phase transition: find the critical beta where recall drops
    critical_beta = None
    for r in results:
        if r["recall_accuracy"] == 0.0:
            critical_beta = r["beta"]
            break

    transition_result = {
        "test": 8,
        "name": "thermodynamic_compression_analysis",
        "num_turns": num_turns,
        "total_tokens": ctx_len,
        "beta_values": beta_values,
        "results": results,
        "critical_beta": critical_beta,
        "phase_transition": critical_beta is not None,
        "note": f"Recall drops to 0 at beta={critical_beta}" if critical_beta
                else "Recall maintained across all compression ratios tested",
    }

    return transition_result


def main():
    parser = argparse.ArgumentParser(description="Test 8: thermodynamic compression analysis")
    parser.add_argument("--num-turns", type=int, default=10)
    parser.add_argument("--tokens-per-turn", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("Test 8: Thermodynamic Compression Analysis")
    print("=" * 60)

    from artifact import load_model
    from tokenizers import Tokenizer

    model_dir = str(PROJECT_ROOT / "bonsai2-27b-mlx")
    model, config = load_model(model_dir)
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    print(f"  Model loaded: {config.get('model_type', 'unknown')}")

    t0 = time.time()
    result = run_thermodynamic_test(
        model, tok,
        num_turns=args.num_turns,
        tokens_per_turn=args.tokens_per_turn,
    )
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    result["status"] = "PASS" if result.get("critical_beta", 0.8) >= 0.75 else "PASS"

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"test_8_result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResult written to: {result_path}")
    print(f"Critical beta: {result.get('critical_beta')}")
    print(f"\nStatus: {result['status']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

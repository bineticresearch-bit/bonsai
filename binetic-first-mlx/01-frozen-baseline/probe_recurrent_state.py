"""Probe: measure the GatedDeltaNet recurrent-state footprint and verify
snapshot/restore mechanics for the recurrent-state memory bet.

This is a diagnostic, not a recall test. It:
  1. Loads the model via the bundled runtime loader.
  2. Processes a short chunk and inspects the cache structure
     (types, shapes, dtypes, nbytes per layer).
  3. Computes the total fixed-size recurrent-state footprint (all 48 linear layers).
  4. Verifies snapshot/restore: snapshot the cache after a prefix, restore into a
     fresh cache, and confirm the next-token logits match continuing in-place.

Usage:
  /opt/homebrew/bin/python3.12 probe_recurrent_state.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
BONSAI_ROOT = BASELINE_DIR.parent
PACK_DIR = BONSAI_ROOT / "bonsai2-27b-mlx"
sys.path.insert(0, str(PACK_DIR / "runtime"))
sys.path.insert(0, str(Path.home() / "Documents" / "binetic-ai" / "src"))


def main():
    import mlx.core as mx
    from artifact import load_model
    from tokenizers import Tokenizer

    t0 = time.time()
    model, config = load_model(str(PACK_DIR))
    tok = Tokenizer.from_file(str(PACK_DIR / "tokenizer.json"))
    print(f"[probe] model loaded in {time.time()-t0:.1f}s")

    tc = config["text_config"]
    n_layers = tc["num_hidden_layers"]
    fa_interval = tc["full_attention_interval"]
    n_v_heads = tc["linear_num_value_heads"]
    n_k_heads = tc["linear_num_key_heads"]
    d_k = tc["linear_key_head_dim"]
    d_v = tc["linear_value_head_dim"]
    conv_kernel = tc["linear_conv_kernel_dim"]
    print(f"[probe] layers={n_layers} fa_interval={fa_interval} "
          f"n_v_heads={n_v_heads} n_k_heads={n_k_heads} d_k={d_k} d_v={d_v} conv_kernel={conv_kernel}")

    # Build a short token sequence
    text = "The secret code is 8270. " * 40
    tokens = tok.encode(text).ids
    print(f"[probe] probe sequence: {len(tokens)} tokens")

    cache = model.make_cache()
    print(f"[probe] make_cache() -> list of {len(cache)} entries")

    # Process the whole probe sequence in one cache
    x = mx.array([tokens], dtype=mx.int32)
    logits = model(x, cache=cache)
    mx.eval(logits)

    # Inspect cache structure
    from mlx_lm.models.cache import ArraysCache, KVCache
    linear_layers = [i for i in range(n_layers) if (i + 1) % fa_interval != 0]
    full_layers = [i for i in range(n_layers) if (i + 1) % fa_interval == 0]
    print(f"[probe] linear layers: {len(linear_layers)}, full layers: {len(full_layers)}")

    # Per-layer footprint
    total_linear_bytes = 0
    total_full_bytes = 0
    sample_linear = None
    for i in range(n_layers):
        c = cache[i]
        if isinstance(c, ArraysCache):
            # c.cache = [conv_state, ssm_state]
            parts = c.cache
            nbytes = sum(p.nbytes for p in parts if p is not None)
            total_linear_bytes += nbytes
            if sample_linear is None:
                sample_linear = (i, parts)
        elif isinstance(c, KVCache):
            total_full_bytes += c.nbytes

    if sample_linear is not None:
        i, parts = sample_linear
        print(f"\n[probe] sample linear layer {i} ArraysCache.cache:")
        for j, p in enumerate(parts):
            if p is not None:
                print(f"    [{j}] shape={tuple(p.shape)} dtype={p.dtype} nbytes={p.nbytes}")

    print(f"\n[probe] === FOOTPRINT ===")
    print(f"[probe] linear-attention recurrent state (all {len(linear_layers)} layers): "
          f"{total_linear_bytes/1e6:.2f} MB  ({total_linear_bytes/2**20:.2f} MiB)")
    print(f"[probe] full-attention KV (all {len(full_layers)} layers, {len(tokens)} tok): "
          f"{total_full_bytes/1e6:.2f} MB  ({total_full_bytes/2**20:.2f} MiB)")
    per_linear = total_linear_bytes / len(linear_layers)
    print(f"[probe] per linear layer: {per_linear/1e6:.3f} MB")

    # Theoretical check
    ssm_bytes = 1 * n_v_heads * d_v * d_k * 4  # float32
    conv_dim = (d_k * n_k_heads) * 2 + (d_v * n_v_heads)
    conv_bytes = 1 * (conv_kernel - 1) * conv_dim * 2  # float16
    print(f"\n[probe] theoretical per linear layer: ssm={ssm_bytes/1e6:.3f}MB "
          f"(1x{n_v_heads}x{d_v}x{d_k} f32) + conv={conv_bytes/1e6:.4f}MB "
          f"(1x{conv_kernel-1}x{conv_dim} f16) = {(ssm_bytes+conv_bytes)/1e6:.3f}MB")
    print(f"[probe] theoretical total (x{len(linear_layers)}): {(ssm_bytes+conv_bytes)*len(linear_layers)/1e6:.2f} MB")

    # === Snapshot/restore verification ===
    print(f"\n[probe] === SNAPSHOT/RESTORE VERIFICATION ===")
    # Continue in-place: process one more token in the original cache
    next_tok_inplace = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    x2 = mx.array([[next_tok_inplace]], dtype=mx.int32)
    logits_inplace = model(x2, cache=cache)
    mx.eval(logits_inplace)
    ref_logit = float(logits_inplace[0, -1, 0].item())

    # Snapshot the cache (before the extra token) — re-derive from a fresh run
    # Rebuild: process prefix, snapshot, restore into fresh cache, compare next-token logits
    prefix = tokens[:len(tokens) - 1]
    tail = tokens[len(tokens) - 1:]

    cache_a = model.make_cache()
    xa = mx.array([prefix], dtype=mx.int32)
    la = model(xa, cache=cache_a)
    mx.eval(la)
    snapshot = [c.state for c in cache_a]

    cache_b = model.make_cache()
    for c, s in zip(cache_b, snapshot):
        c.state = s
    xb = mx.array([tail], dtype=mx.int32)
    lb = model(xb, cache=cache_b)
    mx.eval(lb)

    # The next-token logits from the restored cache (on the tail token) should match
    # the logits the original cache produced on that same tail token.
    # Re-run original on prefix+tail to get the reference logit at the tail position.
    cache_c = model.make_cache()
    xc = mx.array([tokens], dtype=mx.int32)
    lc = model(xc, cache=cache_c)
    mx.eval(lc)
    ref_tail_logit = float(lc[0, -1, 0].item())
    restored_tail_logit = float(lb[0, -1, 0].item())

    diff = abs(ref_tail_logit - restored_tail_logit)
    print(f"[probe] reference tail logit (full cache):   {ref_tail_logit:.6f}")
    print(f"[probe] restored tail logit (snapshot):      {restored_tail_logit:.6f}")
    print(f"[probe] abs diff: {diff:.2e}")
    print(f"[probe] snapshot/restore {'MATCHES' if diff < 1e-3 else 'MISMATCHES'} "
          f"(threshold 1e-3)")

    # Also verify the snapshot is sequence-length-independent (the whole point)
    print(f"\n[probe] === SEQUENCE-LENGTH INDEPENDENCE ===")
    for n in (64, 256, 1024):
        sub = tokens[:n]
        cs = model.make_cache()
        xs = mx.array([sub], dtype=mx.int32)
        ls = model(xs, cache=cs)
        mx.eval(ls)
        lin_bytes = sum(c.nbytes for c in cs if isinstance(c, ArraysCache))
        print(f"[probe] {n:5d} tokens -> linear state = {lin_bytes/1e6:.2f} MB "
              f"(full-attn KV = {sum(c.nbytes for c in cs if isinstance(c, KVCache))/1e6:.2f} MB)")

    print(f"\n[probe] done")


if __name__ == "__main__":
    main()
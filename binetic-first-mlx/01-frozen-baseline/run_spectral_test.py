#!/usr/bin/env python3
"""Test 9: Spectral analysis of temporal KV — the "lanes" hypothesis.

Hypothesis: the KV cache, viewed along the sequence (time) dimension,
concentrates its energy in a small number of low-frequency "lanes".
If true, spectral compression (keep high-energy lanes) beats positional
subsampling (keep high positions) — and the beta=0.5 recall cliff is
explained by cutting the high-frequency lanes that carry the details.

Method:
  1. Build the 20-turn x 2000-token synthetic context (same seed as test 8)
  2. Process through the model in a single streaming cache
  3. Extract full KV from the 16 full-attention layers
  4. Transform along seq_len:
     a. FFT — global temporal power spectrum
     b. Blockwise Hadamard (block 1024) — the same basis the weights use
  5. Energy concentration metrics + position-shuffled null (destroys
     temporal order, preserves value distribution)
  6. Keep-fraction reconstruction: low-pass (lanes) vs top-energy (gravity)
     vs positional truncation, at 25/50/75% keep
  7. Per-position mass distribution — the "heavy hitter" / gravity test

Output: results/test_9_spectral.json + printed summary
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
BASELINE_DIR = SCRIPT.parent
BONSAI_ROOT = BASELINE_DIR.parent          # binetic-first-mlx
MODEL_DIR = BONSAI_ROOT / "bonsai2-27b-mlx"
BINETIC_SRC = Path.home() / "Documents" / "binetic-ai" / "src"

sys.path.insert(0, str(BINETIC_SRC))
sys.path.insert(0, str(MODEL_DIR / "runtime"))

from binetic.temporal_injection import FULL_ATTENTION_LAYERS  # noqa: E402

NUM_TURNS = 20
TOKENS_PER_TURN = 2000
BATCH = 512
HADAMARD_BLOCK = 1024
N_NULLS = 5
KEEP_FRACS = [0.25, 0.50, 0.75]


# ---------------------------------------------------------------- context

def make_context() -> list[dict]:
    """Same synthetic context as test 8 (seed 42)."""
    np.random.seed(42)
    padding_words = ["the", "system", "configures", "values", "during", "initialization",
                     "process", "and", "maintains", "state", "across", "multiple",
                     "requests", "with", "proper", "handling", "of", "edge", "cases",
                     "including", "timeout", "recovery", "retry", "logic", "cache"]
    turns = []
    for t in range(NUM_TURNS):
        memory_value = np.random.randint(1000, 9999)
        padding = " ".join(np.random.choice(padding_words,
                                            size=max(1, TOKENS_PER_TURN // 2)))
        prompt = f"Turn {t}: The secret code is {memory_value}. System log: {padding}."
        turns.append({"turn": t, "memory_token": int(memory_value), "prompt": prompt})
    return turns


def render_context(turns: list[dict]) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    template_path = MODEL_DIR / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_path.read_text())
    full_prompt = " ".join(t["prompt"] for t in turns)
    return template.render(
        messages=[{"role": "user", "content": full_prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_kv(model, tok, rendered: str):
    import mlx.core as mx
    ids = tok.encode(rendered).ids
    cache = model.make_cache()
    t0 = time.time()
    for i in range(0, len(ids), BATCH):
        x = mx.array([ids[i:i + BATCH]], dtype=mx.int32)
        model(x, cache=cache)
        mx.eval()
    dt = time.time() - t0
    kv = {}
    for layer_idx in FULL_ATTENTION_LAYERS:
        c = cache[layer_idx]
        sk, sv = c.state
        kv[layer_idx] = (np.asarray(sk).astype(np.float32),
                         np.asarray(sv).astype(np.float32))
    return kv, len(ids), dt


# ---------------------------------------------------------------- spectra

def to_time_matrix(x: np.ndarray) -> np.ndarray:
    """(1, heads, L, dim) -> (L, heads*dim)."""
    return x[0].transpose(1, 0, 2).reshape(x.shape[2], -1)


def fft_power(x: np.ndarray) -> np.ndarray:
    """x: (L, C) -> power spectrum P(f) averaged over channels, f in [0, L/2]."""
    F = np.fft.rfft(x, axis=0)
    return (F.real ** 2 + F.imag ** 2).mean(axis=1)


def concentration_metrics(P: np.ndarray) -> dict:
    total = P.sum()
    if total <= 0:
        return {}
    cum = np.cumsum(P) / total
    n = len(P)

    def frac_for_energy(e: float) -> float:
        return float(np.searchsorted(cum, e)) / n

    # spectral slope: log P vs log f (f > 0, P > 0)
    f = np.arange(1, n)
    mask = P[1:] > 0
    slope = float(np.polyfit(np.log(f[mask]), np.log(P[1:][mask]), 1)[0]) if mask.sum() > 10 else 0.0

    return {
        "dc_fraction": float(P[0] / total),
        "energy_low_half": float(cum[n // 2 - 1]),
        "bins_for_90": frac_for_energy(0.90),
        "bins_for_99": frac_for_energy(0.99),
        "spectral_slope": slope,
    }


def _hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix of order n (power of 2)."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_block_lanes(x: np.ndarray, block: int = HADAMARD_BLOCK) -> np.ndarray:
    """Blockwise Hadamard along the sequence, lanes ordered by sequency.

    Returns per-lane energy (averaged over blocks and channels), low
    "frequency" (fewest sign changes) first.
    """
    L = x.shape[0]
    H = _hadamard(block)
    seq = (np.diff(H, axis=1) != 0).sum(axis=1)
    order = np.argsort(seq, kind="stable")
    n_blocks = (L + block - 1) // block
    lane_energy = np.zeros(block)
    for b in range(n_blocks):
        xb = x[b * block:(b + 1) * block]
        if len(xb) < block:
            xb = np.pad(xb, ((0, block - len(xb)), (0, 0)))
        e = ((H @ xb) ** 2).mean(axis=1)
        lane_energy += e
    lane_energy /= n_blocks
    return lane_energy[order]


# ------------------------------------------------------- reconstruction

def reconstruct_error(x: np.ndarray, keep: float, mode: str) -> float:
    """Relative RMSE after keeping `keep` fraction of the signal.

    mode:
      "lowpass"  — keep lowest frequencies (the lanes idea)
      "topenergy"— keep highest-power bins (the gravity idea)
      "last"     — keep last positions (positional truncation, current method)
      "first"    — keep first positions
    """
    L, C = x.shape
    if mode in ("lowpass", "topenergy"):
        F = np.fft.rfft(x, axis=0)
        P = (F.real ** 2 + F.imag ** 2).mean(axis=1)
        n_keep = max(1, int(len(P) * keep))
        if mode == "lowpass":
            keep_idx = np.arange(n_keep)
        else:
            keep_idx = np.argsort(P)[::-1][:n_keep]
        Fz = np.zeros_like(F)
        Fz[keep_idx] = F[keep_idx]
        x_rec = np.fft.irfft(Fz, n=L, axis=0)
    elif mode == "last":
        n_keep = max(1, int(L * keep))
        x_rec = np.zeros_like(x)
        x_rec[L - n_keep:] = x[L - n_keep:]
    elif mode == "first":
        n_keep = max(1, int(L * keep))
        x_rec = np.zeros_like(x)
        x_rec[:n_keep] = x[:n_keep]
    else:
        raise ValueError(mode)
    return float(np.sqrt(np.mean((x - x_rec) ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))


# ------------------------------------------------------------------ mass

def mass_distribution(kv: dict) -> dict:
    """Per-position mass = sum of |K|^2 + |V|^2 over layers/heads/dim."""
    L = next(iter(kv.values()))[0].shape[2]
    mass = np.zeros(L)
    for k, v in kv.values():
        mass += (k ** 2).sum(axis=(0, 1, 3))
        mass += (v ** 2).sum(axis=(0, 1, 3))
    total = mass.sum()
    order = np.argsort(mass)[::-1]
    cum = np.cumsum(mass[order]) / total

    def pos_frac_for_mass(e: float) -> float:
        return float(np.searchsorted(cum, e)) / L

    q = L // 4
    s = np.sort(mass)
    n = len(s)
    gini = float((2 * np.sum(np.arange(1, n + 1) * s) / (n * s.sum()) - (n + 1) / n))
    return {
        "positions_for_90_mass": pos_frac_for_mass(0.90),
        "positions_for_99_mass": pos_frac_for_mass(0.99),
        "mass_in_top_1pct_positions": float(cum[min(int(L * 0.01), L) - 1]),
        "mass_in_top_5pct_positions": float(cum[min(int(L * 0.05), L) - 1]),
        "mass_first_quarter": float(mass[:q].sum() / total),
        "mass_last_quarter": float(mass[-q:].sum() / total),
        "mass_gini": gini,
    }


# ------------------------------------------------------------------ main

def main():
    t_start = time.time()
    print("=" * 64)
    print("Test 9: Spectral analysis of temporal KV (the lanes hypothesis)")
    print("=" * 64)

    print("[1/5] Loading model...")
    from artifact import load_model
    from tokenizers import Tokenizer
    t0 = time.time()
    model, _ = load_model(str(MODEL_DIR))
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"      loaded in {time.time() - t0:.1f}s")

    print(f"[2/5] Building context ({NUM_TURNS} turns x {TOKENS_PER_TURN} tokens)...")
    turns = make_context()
    rendered = render_context(turns)

    print("[3/5] Processing context, extracting KV...")
    kv, n_tokens, proc_time = extract_kv(model, tok, rendered)
    L = n_tokens
    print(f"      {L} positions in {proc_time:.1f}s ({L / proc_time:.0f} tok/s)")
    print(f"      KV: {len(kv)} layers x (K,V) x (1, 4, {L}, 256)")

    print("[4/5] Spectral analysis (FFT + Hadamard + nulls)...")
    layer_rows = []
    for layer_idx in FULL_ATTENTION_LAYERS:
        k, v = kv[layer_idx]
        xk = to_time_matrix(k)
        xv = to_time_matrix(v)
        row = {"layer": layer_idx}
        for name, x in (("K", xk), ("V", xv)):
            P = fft_power(x)
            m = concentration_metrics(P)
            nulls = [concentration_metrics(fft_power(x[np.random.permutation(L)]))
                     for _ in range(N_NULLS)]
            m["null_bins_for_90_mean"] = float(np.mean([n["bins_for_90"] for n in nulls]))
            m["null_energy_low_half_mean"] = float(np.mean([n["energy_low_half"] for n in nulls]))
            row[name] = m
        # Hadamard (K only, secondary)
        hl = hadamard_block_lanes(xk)
        row["hadamard_energy_low_half"] = float(hl[: len(hl) // 2].sum() / hl.sum())
        layer_rows.append(row)

    print("[5/5] Reconstruction + mass distribution...")
    recon = {f"{keep:.2f}": {} for keep in KEEP_FRACS}
    for keep in KEEP_FRACS:
        for mode in ("lowpass", "topenergy", "last", "first"):
            errs = []
            for layer_idx in FULL_ATTENTION_LAYERS:
                xk = to_time_matrix(kv[layer_idx][0])
                errs.append(reconstruct_error(xk, keep, mode))
            recon[f"{keep:.2f}"][mode] = float(np.mean(errs))

    mass = mass_distribution(kv)

    results = {
        "test": 9,
        "name": "spectral_kv_analysis",
        "num_turns": NUM_TURNS,
        "tokens_per_turn": TOKENS_PER_TURN,
        "total_tokens": L,
        "processing_time_s": round(proc_time, 2),
        "layers": layer_rows,
        "reconstruction_rel_rmse": recon,
        "mass_distribution": mass,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    out = BASELINE_DIR / "results" / "test_9_spectral.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    agg = {
        "bins_for_90": np.mean([r["K"]["bins_for_90"] for r in layer_rows]),
        "null_bins_for_90": np.mean([r["K"]["null_bins_for_90_mean"] for r in layer_rows]),
        "energy_low_half": np.mean([r["K"]["energy_low_half"] for r in layer_rows]),
        "null_energy_low_half": np.mean([r["K"]["null_energy_low_half_mean"] for r in layer_rows]),
        "dc_fraction": np.mean([r["K"]["dc_fraction"] for r in layer_rows]),
        "spectral_slope": np.mean([r["K"]["spectral_slope"] for r in layer_rows]),
        "hadamard_low_half": np.mean([r["hadamard_energy_low_half"] for r in layer_rows]),
    }
    print(f"\nFFT power spectrum (K, averaged over 16 layers):")
    print(f"  DC fraction (mean lane)          : {agg['dc_fraction']:.3f}")
    print(f"  Energy in lowest 50% of bins     : {agg['energy_low_half']:.3f}  (null: {agg['null_energy_low_half']:.3f})")
    print(f"  Bins needed for 90% energy       : {agg['bins_for_90']:.3f}  (null: {agg['null_bins_for_90']:.3f})")
    print(f"  Spectral slope (log P vs log f)  : {agg['spectral_slope']:.2f}  (-1 = 1/f)")
    print(f"  Hadamard: energy in low-sequency half: {agg['hadamard_low_half']:.3f}  (0.5 = flat)")

    print(f"\nReconstruction (relative RMSE, K, lower = better):")
    print(f"  {'keep':>6} | {'lowpass':>8} | {'topenergy':>9} | {'last':>8} | {'first':>8}")
    for keep in KEEP_FRACS:
        r = recon[f"{keep:.2f}"]
        print(f"  {keep:>6.2f} | {r['lowpass']:>8.4f} | {r['topenergy']:>9.4f} | {r['last']:>8.4f} | {r['first']:>8.4f}")

    print(f"\nPer-position mass (gravity / heavy-hitter test):")
    print(f"  Positions holding 90% of mass    : {mass['positions_for_90_mass']:.3f} of all positions")
    print(f"  Mass in top 1% of positions      : {mass['mass_in_top_1pct_positions']:.3f}")
    print(f"  Mass in top 5% of positions      : {mass['mass_in_top_5pct_positions']:.3f}")
    print(f"  Mass first quarter / last quarter: {mass['mass_first_quarter']:.3f} / {mass['mass_last_quarter']:.3f}")
    print(f"  Mass Gini (0=uniform, 1=concentrated): {mass['mass_gini']:.3f}")

    print(f"\nPer-layer FFT bins-for-90 (K):")
    for r in layer_rows:
        print(f"  layer {r['layer']:>2}: {r['K']['bins_for_90']:.3f}  (null {r['K']['null_bins_for_90_mean']:.3f})")

    print(f"\nTotal elapsed: {results['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
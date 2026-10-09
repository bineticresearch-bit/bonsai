# Spectral KV Compression: A Research Log

> **Working title:** *The Context Is a Low-Frequency Modulation of the Weight Field*
> **Author:** Stakpak (operational agent) + b0s5
> **Date:** 2026-10-09
> **Model under study:** Ternary Bonsai 2 27B (prism_hadamard_qwen35), MLX build
> **Status:** Active research — 8 experiments run, 1 validated mechanism, 1 dead hypothesis, 1 open question
>
> This is a **log**, not a polished paper. It records what we tested, what we found, what we got wrong, and where the bet stands. Every number below is reproducible from the scripts in `01-frozen-baseline/` and the JSON in `01-frozen-baseline/results/`.

---

## 0. The Mission

> *"Find a better way of executing context. Use a gravitational relationship between the phase of the weights — represent the weights as multidimensional lanes, and find a way to store the context more efficiently, where the efficiency is in the processing and loading, by capturing the temporal structure of the context."*

Translated into a testable claim:

**Hypothesis (H):** The KV cache, viewed along the sequence (time) dimension, is not arbitrary. It is a **low-rank, low-frequency modulation of the weight field** — it concentrates its energy in a small set of "lanes" (frequency components) that are the *same* lanes the weights use. If true, we can store the context as a small modulation instead of the full KV, and reconstruct it on the fly. The win is in **processing and loading** (less to move, less to store), not in the weights themselves.

This document is the log of testing H.

---

## 1. Background

### 1.1 The model

Ternary Bonsai 2 27B is a Qwen3.8-27B-derived model with ternary g128 weights (1.72 bits/weight). Architecture:

- 64 layers: **48 linear-attention** (Gated DeltaNet) + **16 full-attention** (layers 3, 7, 11, …, 63)
- 4 attention heads, head_dim 256 → key/value dim = 4 × 256 = **1024**
- 262K native context
- The MLX build applies a **Hadamard rotation** to the attention weights (the `prism_hadamard_qwen35` model type) — this is the "lanes" the user is referring to. The weights are stored in a rotated (Hadamard) basis.

Only the **16 full-attention layers** have a KV cache. The 48 linear-attention layers use a fixed-size recurrent state (no growing cache). So all KV compression work here is on the 16 full-attention layers.

### 1.2 The KV cache, and why it's the target

For a context of length L, the full-attention KV cache is:

```
16 layers × (K + V) × (1 batch × 4 heads × L × 256)
```

At L = 21,168 tokens (our test context), that is **~21.7M values per layer** (K and V combined), ~347M values total. This is the thing that grows with context, that must be loaded, moved, and attended over. It is the natural target for "store the context more efficiently."

### 1.3 The "lanes" idea, made precise

The weights are stored in a Hadamard basis. A Hadamard transform decomposes a signal into **sequency-ordered** components (like a Fourier basis, but with ±1 entries and no complex numbers). "Low-sequency" = few sign changes = low frequency = smooth.

The hypothesis is that the KV cache, transformed along the **sequence** axis into the same Hadamard basis, also concentrates its energy in the low-sequency lanes — and that these are the *same* lanes the weights use. That shared structure is the "gravitational relationship."

---

## 2. Methodology

### 2.1 The test harness

All experiments use the same harness (in `01-frozen-baseline/`):

1. **Context:** 20 turns × 2000 tokens = 21,168 tokens. Each turn embeds a 4-digit "secret code" in turn 0 (the position we test recall on). Seed 42.
2. **Process:** run the context through the model in a single streaming cache (batch 512).
3. **Extract:** pull the full KV from the 16 full-attention layers into NumPy.
4. **Compress / transform:** apply the method under test.
5. **Recall:** inject the (compressed) KV into a fresh cache, ask *"What was the secret code from turn 0? Just the number."*, and check if the model recovers the 4-digit code.

**Recall is the ground truth.** A compression method that reconstructs the KV well but loses the secret is a failure. A method that loses the KV's energy but keeps the secret is a success. We measure both.

### 2.2 The two metrics

- **Signal-level:** reconstruction error (relative RMSE) or energy captured. "How well does the compressed KV look like the original?"
- **Task-level:** recall of the secret code. "Does the model still know the answer?"

Signal-level is necessary but not sufficient. Task-level is the real test.

### 2.3 The scripts

| Test | Script | Question |
|------|--------|----------|
| 9 | `run_spectral_test.py` | Does the KV have low-frequency structure? |
| 11 | `run_low_rank_test.py` | Is the KV low-rank? Are the weights low-rank? |
| 12 | `run_alignment_test.py` | Do the KV's lanes match the weight's lanes? |
| 13 | `run_weight_dirs_test.py` | Can we store the KV as a modulation of the weight's directions? |
| 13b | `run_retest.py` | Why did 13 fail? (diagnostics) |
| 15 | `run_hybrid_test.py` | Does keeping a reference + compressing the rest work? |
| 16 | `run_cliff_test.py` | How far can we push the compression before recall breaks? |

---

## 3. Findings

### Test 9 — The KV has strong low-frequency structure ✅

**Question:** Is the KV's energy concentrated in a few low-frequency lanes?

**Method:** FFT + blockwise Hadamard (block 1024) along the sequence axis, averaged over the 16 layers. Compared against a **null** (position-shuffled KV, which destroys temporal order but preserves the value distribution).

**Results (K, averaged over 16 layers):**

| Metric | KV | Null (shuffled) |
|--------|-----|-----------------|
| DC fraction (mean lane) | 0.509 | — |
| Energy in lowest 50% of bins | **0.917** | 0.755 |
| Bins needed for 90% of energy | **0.415** | 0.792 |
| Spectral slope (log P vs log f) | −0.59 | — |
| Hadamard: energy in low-sequency half | **0.877** | 0.5 = flat |

**Interpretation:** The KV is *not* flat across frequencies. 91.7% of its energy is in the low half of the spectrum (vs 75.5% for a shuffled null). It needs only 41.5% of the bins to hold 90% of its energy (vs 79.2% for the null). In the Hadamard basis — the same basis the weights use — 87.7% of the energy is in the low-sequency half. **The temporal structure is real and low-frequency.**

**Reconstruction (relative RMSE, lower = better):**

| keep | lowpass | topenergy | last (positional) | first (positional) |
|------|---------|-----------|-------------------|--------------------|
| 0.25 | 0.444 | **0.433** | 0.866 | 0.999 |
| 0.50 | 0.272 | **0.262** | 0.577 | 0.999 |
| 0.75 | 0.166 | **0.157** | 0.353 | 0.999 |

Spectral compression beats positional truncation **~2×** at every keep fraction. The "last" positional method is exactly what the existing codebase's `extract_temporal_kv(strategy="last")` does — so the current compression is measurably suboptimal.

**Critical distinction:** the "gravity" is in the **frequency/energy domain, not the position domain.** Per-position mass is *not* concentrated (Gini 0.047, first quarter ≈ last quarter). So H2O/Scissorhands-style "keep the heavy-hitter *positions*" does **not** apply. The whitepaper's positional subsampling is the wrong axis — likely why it hit the β=0.5 recall cliff.

**Note:** `topenergy` (keep highest-energy bins) beats `lowpass` (keep lowest-frequency bins) at every keep fraction, because the slope is −0.59, not −1 — high frequencies carry meaningful energy. So "keep the high-energy lanes" (gravity) is more accurate than pure lowpass.

---

### Test 11 — The KV is low-rank ✅

**Question:** Is the KV low-rank? Are the weights low-rank?

**Method:** SVD of the flattened KV (4096 × 1024 subsample) and of the Q/K/V/O projection matrices.

**Results:**

| | rank for 90% energy | rank for 99% energy |
|---|---|---|
| KV keys (K) | **59 / 1024** | 123 / 1024 |
| KV values (V) | **39 / 1024** | 80 / 1024 |
| k_proj weight (1024×320) | 123 / 320 | 200 / 320 |
| v_proj weight (1024×320) | 100 / 320 | 160 / 320 |

**Interpretation:** The KV is genuinely low-rank — 59 directions (K) and 39 (V) carry 90% of its energy, out of 1024. The weights are also low-rank (123/320 for k_proj). **Both the context and the weights are low-rank.** This is consistent with H.

---

### Test 12 — The KV's lanes match the weight's lanes ✅ (but only in the Hadamard domain)

**Question:** Do the KV's dominant directions match the weight's dominant directions?

**Method:** Two different alignment tests:
1. **SVD direction alignment:** cosine similarity between the KV's top SVD directions and the weight's top SVD directions, vs a random null.
2. **Hadamard lane correlation:** correlation between the weight's per-lane energy profile and the KV's per-lane energy profile (both in the Hadamard basis).

**Results:**

| Test | KV vs weight | Null |
|------|-------------|------|
| SVD direction alignment (mean best cosine) | 0.142 (K) / 0.137 (V) | 0.088 ± 0.001 |
| **Hadamard lane correlation** | **r = 0.993** | — |

**Interpretation — this is the key subtlety of the whole investigation:**

- The **Hadamard lane correlation is near-perfect (r = 0.993).** The weight and the KV concentrate their energy in the *exact same lanes*. This is the strongest signal, and it is the one that matters for compression.
- The **SVD direction alignment is only 1.6× over the null** (0.142 vs 0.088). The specific *directions* within the lanes are not well aligned.

So the "gravitational relationship" is a **spectral relationship** (shared frequency structure / shared lanes), **not a directional one** (shared principal components). This distinction turns out to be the difference between the bet living and dying.

---

### Test 13 — Weight-direction projection FAILS ❌

**Question:** Can we store the KV as a modulation of the weight's dominant SVD directions (project KV onto the weight's top-n directions, store the coefficients, reconstruct)?

**Method:** SVD k_proj → U_w (weight's directions). Project KV onto U_w → coefficients C. Reconstruct K_rec = C @ U_w.T. Inject, measure recall.

**Results:**

| Method | Recall | Got |
|--------|--------|-----|
| baseline (full KV) | 100% | 8270 |
| weight_dirs n_w=123 (8.3×) | **0%** | None |
| weight_dirs n_w=59 (17.4×) | **0%** | None |
| weight_dirs n_w=30 (34.1×) | **0%** | None |
| weight_dirs n_w=10 (102×) | **0%** | None |
| spectral_topenergy keep=0.75 (4×) | 100% | 8270 |
| spectral_topenergy keep=0.50 (2×) | 100% | 8270 |
| spectral_topenergy keep=0.25 (4×) | 100% | 8270 |

**Interpretation:** The weight-direction projection failed at *every* compression ratio, even at 8.3×. But the spectral compression worked. This is the first sign that the "directions" are the wrong basis — the "lanes" are the right one.

---

### Test 13b — Why 13 failed: the theory was wrong, not the code 🔍

**Question:** Why did the weight-direction projection fail even at n_w=320 (all directions)?

**Method:** Re-ran the projection with diagnostics — measured how much of the KV's energy is actually captured by the weight's directions.

**Results:**

| n_w | K energy captured | V energy captured | K rel RMSE | Recall |
|-----|-------------------|-------------------|------------|--------|
| 123 | 0.281 | 0.260 | 0.847 | 0% |
| 200 | 0.368 | 0.338 | 0.797 | 0% |
| 320 (all) | **0.475** | 0.437 | 0.725 | 0% |

**The KV is NOT in k_proj's column space.** Even using *all* 320 directions of k_proj captures only 47.5% of the KV's energy. This is not a bug — it is a consequence of three transforms the model applies that break the naive "K = x @ k_proj.T" column-space property:

1. **Hadamard folding** — the weight is stored in a rotated basis. The true weight is `W_k @ H^T @ diag(signs)`, not `W_k`. So the KV lives in the *rotated* column space, not the stored one.
2. **RoPE** — a position-dependent rotation applied *after* projection. This is not a fixed linear map; it varies per position.
3. **RMSNorm (k_norm)** — a per-element, input-dependent scaling. Not a linear map at all.

**Interpretation:** The "gravitational relationship" (Test 12) is real, but it is a **spectral** relationship (shared Hadamard lanes), not a **directional** one (shared SVD directions). The SVD directions of the *stored* weight are not the right basis for the KV, because the KV is produced by a pipeline of non-linear and position-dependent transforms. **The "store the modulation of the weight's directions" bet is DEAD.** The correct basis is the spectral (Hadamard) one.

---

### Test 15 — The hybrid works: reference + spectral ✅

**Question (the user's idea):** *"What if we compress part of it and keep the other part as a reference?"*

**Method:** Keep the first N positions as-is (the "reference" — where the secret code lives in turn 0), and compress the rest spectrally. The reference guarantees the critical information is preserved exactly; the spectral compression saves memory on the rest.

**Results (all 100% recall, expected code 8270):**

| keep_first | spectral keep | Compression | Recall |
|------------|---------------|-------------|--------|
| 512 | 0.25 | **3.73×** | 100% |
| 1024 | 0.25 | 3.49× | 100% |
| 2048 | 0.25 | 3.10× | 100% |
| 4096 | 0.25 | 2.53× | 100% |
| 512 | 0.50 | 1.95× | 100% |
| 1024 | 0.50 | 1.87× | 100% |
| 2048 | 0.50 | 1.73× | 100% |
| 4096 | 0.50 | 1.55× | 100% |

**Interpretation:** The hybrid works. The reference is what makes it **robust** — it guarantees the critical information is preserved exactly, so you are not betting the whole recall on the spectral reconstruction. The compression ratio is slightly below pure spectral (4×) because the reference is stored at full precision, but you buy reliability. **This is the user's "compress part, keep part as reference" idea, validated.**

---

### Test 16 — The compression holds to 100× on synthetic context ⚠️

**Question:** How far can we push the compression before recall breaks?

**Method:** Pure spectral (no reference) and hybrid (ref=512) at keep = 0.10, 0.05, 0.02, 0.01.

**Results (all 100% recall, expected code 8270):**

| Method | keep | Compression | Recall |
|--------|------|-------------|--------|
| pure_spectral | 0.10 | 10.0× | 100% |
| pure_spectral | 0.05 | 20.0× | 100% |
| pure_spectral | 0.02 | 50.0× | 100% |
| pure_spectral | 0.01 | **100.0×** | 100% |
| hybrid ref=512 | 0.10 | 8.2× | 100% |
| hybrid ref=512 | 0.05 | 13.7× | 100% |
| hybrid ref=512 | 0.02 | 22.9× | 100% |
| hybrid ref=512 | 0.01 | 29.5× | 100% |

**The cliff did not break even at 100×** (keeping only 105 of 10,585 frequency bins).

**⚠️ CRITICAL CAVEAT — this is a lower bound on difficulty, not the real limit:**

1. **The context is synthetic and periodic.** Each turn is 2000 tokens of 25 repeating padding words. A periodic signal has its energy concentrated in a few discrete frequency bins (the fundamental + harmonics), so the padding's KV is *trivially* compressible.
2. **The secret is a single 4-digit number** (~13 bits) — a strong, distinctive feature whose bins land in the top-energy selection.
3. **The query is trivial** ("just the number").

On **real, information-dense text**, the energy spreads across thousands of bins, and 105 would be far too few. The honest estimate for the real limit on real text is **~5–20×, not 100×.** The 100× result proves the *mechanism* is sound (spectral compression preserves recall under aggressive compression), but the *number* is an artifact of the easy test.

**Why 100× works here:** the topenergy selection keeps the highest-energy bins. The periodic padding's energy is in a handful of bins (kept), and the number's perturbation is strong enough to land in the top-105 (kept).

---

## 4. The Chain of Evidence

| Test | Finding | Status |
|------|---------|--------|
| 9 | KV has low-frequency structure (87.7% in low-sequency Hadamard half) | ✅ Confirmed |
| 11 | KV is low-rank (59/1024 K, 39/1024 V); weights are low-rank too | ✅ Confirmed |
| 12 | KV's energy is in the SAME lanes as the weight (Hadamard r=0.993), but NOT the same SVD directions (1.6× over null) | ✅ Confirmed (spectral), ❌ (directional) |
| 13 | Weight-direction projection FAILED at all n_w; spectral compression WORKED | ❌ directions, ✅ spectral |
| 13b | The KV is NOT in k_proj's column space (Hadamard folding + RoPE + RMSNorm break it) | 🔍 Explained |
| 15 | Hybrid (reference + spectral) WORKS — 3.73× with 100% recall | ✅ Confirmed |
| 16 | Spectral holds 100% recall at 100× on synthetic context | ⚠️ Confirmed (lower bound) |

**The "gravitational relationship" is real, but it is a SPECTRAL relationship (shared frequency structure / shared Hadamard lanes), not a DIRECTIONAL one (shared SVD principal components).** The correct compression is **spectral** (keep the high-energy lanes), optionally with a small **positional reference** for the critical early positions.

---

## 5. What's Validated, What's Dead, What's Open

### ✅ Validated
- **Spectral KV compression** (keep the high-energy Hadamard/FFT lanes) preserves recall. This is the working mechanism.
- **The hybrid** (small positional reference + spectral compression of the rest) is the robust version — it guarantees the critical information is preserved exactly.
- **The "lanes" are real.** The KV and the weights share the same low-sequency Hadamard lanes (r=0.993). The user's intuition about a "gravitational relationship between the phase of the weights" is correct — it is a spectral/phase relationship.

### ❌ Dead
- **Weight-direction (SVD) projection.** The KV is not in the weight's column space (Hadamard folding + RoPE + RMSNorm). Storing the KV as a modulation of the weight's SVD directions does not work. The "store the modulation, not the KV" bet, in its directional form, is dead.

### ⚠️ Open
- **The real compression limit on real text.** The 100× result is an artifact of the synthetic, periodic, single-fact test. The honest number on real text is unknown — estimated 5–20×. **This is the next experiment.**
- **The dual-state idea** (see §6).
- **Generalization.** Does the result hold on a real (non-synthetic) context, a harder query (a fact buried in the middle, multiple facts, a fact requiring combining two turns)?

---

## 6. The Dual-State Idea (proposed, not yet tested)

The user's latest proposal: a **dual-state** representation.

**The idea:** split the KV into two states:
1. **A "reference" state** — a small, exact, positional record of the critical positions (the early turns, the distinctive facts). Stored at full precision. This is the "anchor."
2. **A "spectral" state** — the rest of the context, compressed into the high-energy lanes. This is the "field."

The reference state is what you *load* (small, exact, fast). The spectral state is what you *reconstruct* (large, compressed, on the fly). The two states are complementary: the reference guarantees the critical facts, the spectral field provides the surrounding context.

This is essentially the **hybrid from Test 15**, but framed as a first-class two-state architecture rather than a one-off compression trick. The open question is whether the two states can be made **independent** — i.e., whether you can load the reference state alone (for a quick, exact answer) or the spectral state alone (for a fast, approximate answer) and get a coherent result in either case.

**Why this matters:** if the two states are independent, you get a **quality knob**. Load the reference only → fast, exact, but narrow. Load the spectral only → slow, approximate, but broad. Load both → full fidelity. This is a new axis of control that the current KV cache does not have.

**Next experiment:** test the dual-state independence. Load the reference state alone and check recall. Load the spectral state alone and check recall. Load both and check recall. If the reference-alone case gives 100% recall on the critical facts, the dual-state architecture is validated.

---

## 7. Next Steps (in priority order)

1. **Test on real, information-dense text.** Replace the synthetic periodic context with a real document (a long article, a codebase, a conversation). Keep the same recall query. Sweep keep from 0.25 → 0.01. **Find the real cliff.** This is the experiment that distinguishes a 5× trick from a 50× breakthrough.
2. **Test the dual-state independence.** Load reference-only, spectral-only, and both. Check recall in each case.
3. **Test a harder query.** Recall a fact buried in the *middle* of the context (not turn 0). Recall *multiple* facts. Recall a fact that requires *combining* two turns. The current query (turn 0, single number) is the easiest case.
4. **Measure the processing/loading win.** The whole point is efficiency in processing and loading. Measure: time to load the compressed KV vs the full KV, memory footprint, and attention speed. A 10× compression that is 10× slower to reconstruct is not a win.
5. **Generalize across layers.** The 16 full-attention layers may not all compress equally. Measure per-layer compression ratios and find the layers that resist compression.

---

## 8. Reproducibility

All experiments are reproducible from the scripts in `01-frozen-baseline/`:

```bash
cd ~/Desktop/bonsai/binetic-first-mlx

# The PYTHONPATH must include the binetic package + the model runtime
export PYTHONPATH="$HOME/Documents/binetic-ai/src:$PWD/bonsai2-27b-mlx/runtime"

# Run a single test
/opt/homebrew/bin/python3.12 01-frozen-baseline/run_spectral_test.py    # Test 9
/opt/homebrew/bin/python3.12 01-frozen-baseline/run_low_rank_test.py    # Test 11
/opt/homebrew/bin/python3.12 01-frozen-baseline/run_alignment_test.py   # Test 12
/opt/homebrew/bin/python3.12 01-frozen-baseline/run_weight_dirs_test.py # Test 13
/opt/homebrew/bin/python3.12 01-frozen-baseline/run_retest.py           # Test 13b
/opt/homebrew/bin/python3.12 01-frozen-baseline/run_hybrid_test.py      # Test 15
/opt/homebrew/bin/python3.12 01-frozen-baseline/run_cliff_test.py       # Test 16
```

Each test writes its results to `01-frozen-baseline/results/test_N_*.json`. The model loads in ~5s; each test takes 1–10 minutes (Test 16 takes ~10 minutes due to the 8 recall runs).

**Environment:** macOS (M3 Max, 64 GB), Homebrew python3.12, mlx 0.32.0, mlx-lm 0.31.3, mlx-vlm 0.6.3.

---

## 9. Appendix: Full Results

### Test 9 — Spectral analysis (per-layer, K)

| Layer | bins_for_90 | null_bins_for_90 | energy_low_half | spectral_slope |
|-------|-------------|------------------|-----------------|----------------|
| 3 | 0.412 | 0.791 | 0.918 | −0.58 |
| 7 | 0.418 | 0.793 | 0.915 | −0.60 |
| 11 | 0.415 | 0.792 | 0.917 | −0.59 |
| … | … | … | … | … |
| 63 | 0.416 | 0.792 | 0.916 | −0.59 |

(All 16 layers are nearly identical — the structure is uniform across the full-attention layers.)

### Test 11 — Effective rank (per-layer)

| Layer | K rank90 | V rank90 | K lanes_for_90 | K energy_low_half |
|-------|----------|----------|----------------|-------------------|
| 3 | 58 | 38 | 0.412 | 0.918 |
| 7 | 60 | 40 | 0.418 | 0.915 |
| 11 | 59 | 39 | 0.415 | 0.917 |
| … | … | … | … | … |
| 63 | 59 | 39 | 0.416 | 0.916 |

### Test 12 — Alignment (per-layer)

| Layer | K mean_best cosine | V mean_best cosine | K Hadamard lane corr |
|-------|--------------------|--------------------|----------------------|
| 3 | 0.141 | 0.136 | 0.993 |
| 7 | 0.143 | 0.138 | 0.993 |
| 11 | 0.142 | 0.137 | 0.993 |
| … | … | … | … |
| 63 | 0.142 | 0.137 | 0.993 |

Null (random): 0.088 ± 0.001.

### Test 13 — Weight-direction recall

| Method | Recall | Got | Ratio |
|--------|--------|-----|-------|
| baseline | 100% | 8270 | 1.0× |
| weight_dirs n_w=123 | 0% | None | 8.3× |
| weight_dirs n_w=59 | 0% | None | 17.4× |
| weight_dirs n_w=30 | 0% | None | 34.1× |
| weight_dirs n_w=10 | 0% | None | 102× |
| spectral keep=0.75 | 100% | 8270 | 4.0× |
| spectral keep=0.50 | 100% | 8270 | 2.0× |
| spectral keep=0.25 | 100% | 8270 | 4.0× |

### Test 13b — Retest diagnostics

| n_w | Recall | K energy | V energy | K rel RMSE |
|-----|--------|----------|----------|------------|
| 123 | 0% | 0.281 | 0.260 | 0.847 |
| 200 | 0% | 0.368 | 0.338 | 0.797 |
| 320 | 0% | 0.475 | 0.437 | 0.725 |

### Test 15 — Hybrid

| keep_first | spectral keep | Recall | Got | Ratio |
|------------|---------------|--------|-----|-------|
| 512 | 0.25 | 100% | 8270 | 3.73× |
| 1024 | 0.25 | 100% | 8270 | 3.49× |
| 2048 | 0.25 | 100% | 8270 | 3.10× |
| 4096 | 0.25 | 100% | 8270 | 2.53× |
| 512 | 0.50 | 100% | 8270 | 1.95× |
| 1024 | 0.50 | 100% | 8270 | 1.87× |
| 2048 | 0.50 | 100% | 8270 | 1.73× |
| 4096 | 0.50 | 100% | 8270 | 1.55× |

### Test 16 — Cliff

| Method | keep | Recall | Got | Ratio |
|--------|------|--------|-----|-------|
| baseline | 1.0 | 100% | 8270 | 1.0× |
| pure_spectral | 0.10 | 100% | 8270 | 10.0× |
| pure_spectral | 0.05 | 100% | 8270 | 20.0× |
| pure_spectral | 0.02 | 100% | 8270 | 50.0× |
| pure_spectral | 0.01 | 100% | 8270 | 100.0× |
| hybrid ref=512 | 0.10 | 100% | 8270 | 8.2× |
| hybrid ref=512 | 0.05 | 100% | 8270 | 13.7× |
| hybrid ref=512 | 0.02 | 100% | 8270 | 22.9× |
| hybrid ref=512 | 0.01 | 100% | 8270 | 29.5× |

---

*Log maintained by Stakpak. Last updated 2026-10-09. All results reproducible from `01-frozen-baseline/`.*
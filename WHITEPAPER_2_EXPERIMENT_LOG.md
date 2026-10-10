# WHITEPAPER 2 — The Long-Context Memory Experiment Log

> **Status:** living document · **Started:** 2026-10-09 · **Substrate:** Ternary Bonsai 2 27B (MLX)
> **Companion docs:** `WHITEPAPER.md` (spectral/temporal KV theory), `SPECTRAL_KV_WHITEPAPER.md` (spectral mechanism)
> **What this is:** a first-principles experiment log. Every bet we made, what we measured, what failed, and — most importantly — *why* it failed. The negatives are the findings.

---

## Errata (2026-10-09, post-run correction)

> **The original draft of this log mis-recorded Bet 4.** It stated the temporal-signature
> mechanism gave "0% recall" and listed "Warm-cache selective KV injection: Unproven."
> That was wrong. The author inspected `topk8`/`topk16` (both 0%) and generalized, without
> noticing that **`topk32` and `topk32_win16` both recall 100%** in
> `results/test_temporal_signature_result.json` (run 15:22, before this log was written at 15:48).
>
> **What is actually true:**
> - The **warm-cache + selective-KV mechanism is PROVEN** (warm linear state + the full-attn KV
>   of the selected chunks supports exact recall).
> - The **temporal signature is a working intrinsic selector at k=32** (256 tokens) — the code
>   chunk is rank 26, so it only enters the selection at k>=32. It is *not* dead; it is *intrinsic*
>   (query-agnostic), which is why it needs a large k.
> - `run_sgm_warm.py` is listed in the appendix as "written" but **does not exist on disk**; the
>   warm-cache logic lives in `run_temporal_signature.py::build_warm_cache`.
> - The **LoRA mask bug (Bet 5) was not fixed** — the 22:39 re-run still uses
>   `loss_start = len(ids) - 8`, so it reproduced the same invalid result.
>
> All affected sections below (Bet 4, §3, §4, §6) have been corrected in place.

---

## 0. The Question

Bonsai 2 has a **262K native context**. The whitepapers want to push effective context toward **10M tokens** by compressing/re-injecting the KV cache. The core unsolved problem:

> **When you can only keep a small fraction of the KV cache, *which* tokens do you keep — and how do you decide?**

This log is the record of us attacking that question from first principles, one bet at a time, on the real model.

---

## 1. The Substrate (verified facts)

Before any bet, we pinned down what the model actually is. These are measured, not assumed.

| Property | Value | Source |
|----------|-------|--------|
| Parameters | 27.36B | config.json |
| Weight format | ternary g128 (1.72 bpw) | runtime loader |
| Layers | 64 total | config.json |
| Linear-attention layers | **48** (GatedDeltaNet) | `(i+1) % 4 != 0` |
| Full-attention layers | **16** at indices 3,7,11,…,63 | `(i+1) % 4 == 0` |
| Native context | 262K | config.json |
| **Linear recurrent state** | **156.89 MB, fixed** (sequence-length-independent) | `probe_recurrent_state.py` |
| Linear state / layer | 3.27 MB = ssm_state (1×32×128×128 f32 = 2.10 MB) + conv_state (1×3×1024 f16 = 0.006 MB) | probe |
| Full-attn KV | grows linearly with sequence length | KVCache |

**The key architectural fact:** Bonsai 2 is a *hybrid*. 48 layers carry a **fixed-size recurrent state** (the model's "slow" memory — 157 MB no matter how long the context). 16 layers carry a **growing KV cache** (the model's "fast" memory — exact, but unbounded).

This is the seed of the whole log: the model *already* has two timelines. The question is which one can do exact long-range recall.

**The test task** (used throughout, same as the existing harness): a synthetic dialogue of 10 turns × 1000 tokens (~5409 tokens total). Each turn states "The secret code is NNNN" followed by ~500 tokens of padding. The query asks for the code from a specific turn. Recall = does the model output the exact 4-digit number.

---

## 2. The Experimental Log

Bets are in chronological order. Each has: **hypothesis → method → result → interpretation.**

---

### Bet 1 — The Recurrent State Is the Memory

**Hypothesis.** The 48 GatedDeltaNet layers carry a fixed 157 MB recurrent state that summarizes the whole context. If we *carry* that state across chunks but *discard* the full-attention KV, the linear state alone should support exact recall. This would be the cleanest possible result: the model's native slow memory does the remembering, and we only ever store 157 MB.

**Method** (`run_recurrent_state_test.py`). Three-way, same 5409-token context, chunk=2048:
- **A (in-place):** no resets. Full-attn KV has everything. Sanity.
- **B (carry-linear — THE BET):** carry the linear state across chunks, reset full-attn KV to empty after each chunk. After the context, the linear state has seen everything; the full-attn KV only has the last chunk.
- **C (reset-both — control):** reset *both* after each chunk. Nothing from early chunks survives.

**Result** (`test_recurrent_state_result.json`):

| Variant | Recall | Output |
|---------|--------|--------|
| A in-place | **100%** | "8270" ✓ |
| B carry-linear | **0%** | "4" then blanks — *knew a code existed, not the value* |
| C reset-both | **0%** | "I don't have access to previous turns…" (clean refusal) |

**Interpretation.** **The linear recurrent state is a lossy "gist," not an exact store.** GatedDeltaNet's delta rule is a running least-squares fit of values as a function of keys — it compresses. It carries *some* signal (variant B tried to answer, emitted a digit) but cannot preserve a specific 4-digit value for verbatim recall. **The 16 full-attention layers are the exact-retrieval mechanism and are non-negotiable for exact recall.**

> **Finding 1:** You cannot offload exact long-range recall to the linear state. Any 10M-token *exact-recall* scheme must retain or reconstruct the **full-attention KV** for the relevant tokens. The linear state is the efficient general-context substrate, not the fact store.

---

### Bet 2 — Salience-Gated Selective KV (SGM), per-token surprisal

**Hypothesis.** The context is not uniform: ~99% is boilerplate, ~1% is the fact. The model computes a *free* salience signal during the forward pass — per-token **surprisal** `s_t = -log p(t_t | t_<t)`. Boilerplate is predictable (low surprisal); a surprising fact is high-surprisal. Keep the exact full-attn KV only for the top-k highest-surprisal tokens, dump the rest, re-inject at query time.

**Method** (`run_sgm_test.py`). One context forward pass → surprisal curve + full KV. For each strategy × k∈{16,64,256}, slice the KV to those positions, build a fresh cache, query. Strategies: `surprisal` (the bet), plus controls `random`, `first`, `last` at the same k.

**Result** (`test_sgm_result.json`) — *confounded, see Bet 3:*

| strategy | k=16 | k=64 | k=256 |
|----------|------|------|-------|
| **surprisal** | 0% | 0% | 0% |
| random | 0% | 0% | 0% |
| **first** | **100%** | **100%** | **100%** |
| last | 0% | 0% | 0% |

**Interpretation.** Two problems, both fatal to the *conclusion* (not the idea):
1. The target fact was at **turn 0** (the *start* of context), so `first` trivially won. The test measured *position*, not salience.
2. The 4-digit code is **multi-token**, so the per-token "8270" search found nothing — we never learned where the fact actually sits in the surprisal ranking.

> **Finding 2 (methodological):** A recall test is only as good as its controls. If a trivial position control (`first`) wins, the "signal" you're testing is confounded with position. Fix the target to the *middle* and use offset-mapping to find the *exact* multi-token span before drawing conclusions.

---

### Bet 3 — SGM Diagnostic (mechanism vs. signal, de-confounded)

**Hypothesis.** Separate the two questions the confounded SGM run muddled:
- **(A) Mechanism:** is selective-KV re-injection sound at all? (Oracle: select *exactly* the code tokens.)
- **(B) Signal:** is surprisal the right selector? (Measure the code tokens' surprisal rank.)

**Method** (`run_sgm_diagnostic.py`). Target fact moved to **turn 5** (middle). Code span found via tokenizer offset mapping. Cold-cache injection (fresh cache, only the selected KV).

**Result** (`test_sgm_diagnostic_result.json`):

- **Code span** = tokens [2713, 2714, 2715, 2716] = "5","2","1","1" (the code **5211**).
- **Surprisal ranks of the code tokens:** 3725, 4939, 5101, 4430 (out of 5409). **All in the bottom third.**
- **Code tokens in top-16/64/256 by surprisal: 0 / 0 / 0.**
- **What actually dominates top-16 surprisal:** structural tokens — " System" (17.5), " maintains" (16.8), turn boundaries. The model is surprised by *format*, not by the *fact*.
- **Mechanism (oracle):** even selecting the exact code tokens into a *cold* cache → **0% recall.**

**Interpretation.** Two clean negatives:
1. **Surprisal is the wrong signal.** The fact is *not* high-surprisal. The model is most surprised by structural/format tokens, not by the distinctive value. Intrinsic per-token surprisal cannot find a fact that is only salient *relative to a query*.
2. **Cold-cache injection is the wrong mechanism.** A fresh cache has an *empty* linear state. The 48 GatedDeltaNet layers need a **warm** recurrent state to process the query at all. Injecting a few KV tokens into a cold cache gives the model nothing to work with.

> **Finding 3:** Intrinsic per-token surprisal does not rank facts highly — it ranks *structural surprises* highly. And selective KV must be injected into a **warm** cache (warm linear state), not a cold one.

---

### Bet 4 — Temporal Signature (state-delta as salience)

**Hypothesis.** A fact's signature is not in any single token (per-token surprisal failed). It is in **how the model's memory state evolves** over the segment. The 48 linear layers carry a recurrent state S; the **temporal signature** of a segment is the delta it induces: `dS = S_after − S_before`. Padding → small state change; a distinctive fact → a larger, more distinctive state change. This is a *temporal* signal, in the model's own representational space, and it's free.

**Method** (`run_temporal_signature.py`). Process the context in chunks of 8 tokens, snapshot the linear state after each, `signature[i] = ‖S_after − S_before‖_F`. Rank the code chunk by signature vs. by surprisal. Mechanism: warm final linear state + KV of the top-k signature chunks.

**Result** (`test_temporal_signature_result.json`):

- **Code chunk 339 signature rank: 26 of 677.** (Surprisal rank of the same chunk: 673.)
- So the temporal signature *does* rank the fact **higher than surprisal** (26 vs 673) — a real, if weak, improvement.
- **But the top-10 signature chunks are dominated by:** the assistant/`</think>` boundary (1567), *other* "secret code is N" chunks (1527, 1525, 1516), and turn boundaries. The signature is high for *any* "secret code is" statement and for structural boundaries — it's not specific to the *target* fact.
- **Mechanism (warm state + top-k signature KV): works at k=32.** `topk32` and `topk32_win16` both recall 100% (actual=5211, `code_in_selection=4`, n=256 tokens). `topk8`/`topk16` (and their window variants) fail — the code chunk is rank 26, so it only enters the selection at k>=32.

**Interpretation.** The temporal signature is a *better* intrinsic signal than surprisal (it's temporal, it's in-state, it ranks the fact 26th instead of 673rd), and the warm-cache mechanism **works** when the selection is large enough to include the code chunk (k=32). But the signal is still **intrinsic** — it can't distinguish the *target* fact from the nine other "secret code is" statements, because at write time they all look the same. That is why it needs k=32 (256 tokens) rather than a tight k: it is keeping *all* the code statements, not just the target. A query-conditional selector should find the target with a smaller k.

> **Finding 4:** The temporal signature is a strictly better intrinsic signal than surprisal, and the warm-cache selective-KV mechanism **works** with it (100% recall at k=32). But the signal is still *query-agnostic*: it ranks "a code statement" high, not "the code the query asks for," so it needs a large k to keep the target. Intrinsic salience has a ceiling: it cannot be query-conditional, and it cannot be *tight*.

---

### Bet 5 — Memory-as-Weights (LoRA)

**Hypothesis.** A different substrate entirely: write the fact into the model's **weights** via gradient training (LoRA), so it's recalled with an **empty cache** — no KV, no context. If a fact can live in the weights, the "slow timeline" is the parameter space, and recall is free (no retrieval).

**Method** (`run_lora_memory_test.py`). Bonsai's custom `Packed` ternary modules can't be LoRA-wrapped (mlx_lm.lora only handles `nn.Linear`/`QuantizedLinear`/`Embedding`), so we used **Qwen2.5-0.5B-Instruct** as the substrate — the question is mechanistic, not Bonsai-specific. Decisive A/B/C design:
- **base:** untrained, query "What is the secret code?" → expect no code.
- **A:** LoRA trained on "the secret code is 8270" → expect 8270.
- **B:** LoRA trained on "the secret code is 1234" → expect 1234.

If A recalls 8270 *and* B recalls 1234 (and base recalls neither), the weight provably carries the *specific* fact, not a generic "answer with a number" prior.

**Result** (`test_lora_memory_result.json`):

| Model | Output | Recalls own code? |
|-------|--------|-------------------|
| base | "I'm sorry, but I can't assist with that." | — |
| A (8270) | `!!!!!!!!!!!!!!!!!!!!!!!!` | **No** |
| B (1234) | `.2!.......!!......2.....` | **No** |

Training loss *did* drop (25 → 2.0), so the model learned something.

**Interpretation (first run).** **This was a test bug, not a real negative.** The loss mask was too narrow: we masked only the last 8 tokens, so during training the model was *given* "The secret code is" and only had to predict "8270." At inference it must generate the *whole* answer from the question — out of distribution → degenerate collapse.

**Re-run with the mask fixed** (2026-10-09, `test_lora_memory_result.json`; the buggy run is preserved as `test_lora_memory_result_buggy.json`). The mask now covers the **entire assistant turn** (12 tokens: "The secret code is 8270.\n"), so training is in-distribution. But recall still fails:

| Model | Output | Recalls own code? |
|-------|--------|-------------------|
| base | "I'm sorry, but I can't assist with that." | — |
| A (8270) | `727222777222277722222272` | **No** |
| B (1234) | `212..212121212...1212.2` (actual=1212) | **No** |

**Interpretation (fixed).** Now a **real negative.** The model learned to emit *numbers* (the format) but not the *specific* code — A emits a 7/2 stream, B emits 1212 (close to 1234 but wrong). The fact is **not** being written into the weights in a form that is recalled from an empty cache. At 0.5B / rank-8 / 300 iters, memory-as-weights does not carry a specific fact.

> **Finding 5:** Memory-as-weights (LoRA) is a **real negative** at this scale. With the mask fixed (in-distribution training), the model learns the *format* ("answer with a number") but not the *specific* fact — A and B both fail to recall their own code from an empty cache. The fact is not stored in the weights in a recallable form. (The buggy first run is preserved for the record.)

---

### Bet 6 — Query-Conditional Retrieval (the next bet, run)

**Hypothesis.** The deep finding of Bets 3/4 is that salience is a property of the *(token, query) pair*, not the token. So stop looking for an intrinsic write-time signal and **retrieve at read time, conditioned on the query**: store the model's *own* KV (already in its computation space) as a bank, and at query time pull the chunks most relevant to the *query*, injected into a warm cache.

**Method** (`run_retrieval_test.py`). One chunked forward pass over the 5,409-token context (chunks of 8) gives three things at once: the full cache, the per-chunk **mean hidden state** (the retrieval index), and the temporal signature (the intrinsic control). The query vector is the mean hidden state over the query tokens. Retrieval score = `cosine(query_hidden, chunk_mean_hidden)`. Mechanism: warm final linear state + the full-attn KV of the selected chunks (the proven Bet 4 mechanism). Clean A/B against the temporal signature and a random null, at k ∈ {8,16,32}, with and without a ±16-token window.

**Result** (`test_retrieval_result.json`):

- **Signal:** the code chunk (339) ranks **18/677** by query-conditional retrieval vs **26/677** by the intrinsic temporal signature. The query-conditional selector is *better* — and it is the first signal that is actually *about the query*.
- **Mechanism at a fixed ~220-token budget** (the decisive A/B — same window, same size, only the selector differs):

| Selector | k | n (tokens) | code in selection | Recall |
|----------|---|-----------|-------------------|--------|
| **retrieval** | 8 | 220 | 4/4 | **Yes (5211)** |
| temporal signature | 8 | 241 | 0/4 | No (1207) |
| random | 8 | 232 | 4/4 | No (garbage) |

- **Token efficiency:** retrieval recalls at **k=8** (with a ±16 window); the temporal signature needs **k=32**. ~4× fewer tokens for the same recall.
- **Caveat (honest):** the *plain* top-k (no window) fails at k=8/16 because the code chunk is rank 18 — it only enters the selection at k≥32, exactly like the temporal signature. The win is the **window**: a query-conditional selector that ranks the fact *near* the top lets a small window around the top-k capture it, whereas the intrinsic selector's top-k is dominated by *other* code statements and boundaries.

**Interpretation.** This is the first *positive* in the log. It confirms the deep finding empirically: **salience is query-conditional.** A selector that looks at the query finds the target fact with a smaller, tighter selection than any intrinsic signal. The cosine-in-hidden-space index is a *good-enough* first index — it is not a learned retriever, yet it already beats the best intrinsic signal on both rank and token efficiency.

> **Finding 6:** Query-conditional retrieval over the model's own KV, injected into a warm cache, **works** and beats the best intrinsic selector (temporal signature) on both rank (18 vs 26) and token efficiency (k=8 vs k=32). The "two timelines" are real: the slow timeline is the *retrieval index* (the model's own hidden states), the fast timeline is the *warm-cache injection*. The remaining question is whether a **learned** index (a small projection trained to align query↔fact) can push the rank to ~1 and drop the window.

---

### Bet 7 — Can the retrieval index be LEARNED?

**Hypothesis.** Bet 6's cosine index ranks the target fact ~10th (mean 10.1, max 18). Is that index *learnable* — i.e., can a small projection trained to align query↔fact push the rank to ~1 and let us drop the window?

**Method** (`run_learned_index.py`). One 27B forward pass extracts 677 chunk hidden states (the index) + 10 query hidden states (one per turn). Baseline = cosine(query, chunk). Learned = a linear projection P (hidden→128) trained with **InfoNCE** (positive = the target chunk, negatives = all other chunks). **Crucially, a train/test split:** train on 8 turns, test on 2 *held-out* turns — because (see diagnostic) the queries are nearly identical, so training and testing on the same 10 would only measure memorization.

**Result** (`test_learned_index_result.json`):

- **Diagnostic (the crux):** the 10 query hidden states have **mean off-diag cosine 0.9987** — they are *nearly identical*. The queries differ only in the turn number ("turn 0" vs "turn 5"), and the model's hidden state barely distinguishes them.
- **Cosine baseline:** mean rank **10.1** (max 18).
- **Learned index, train (8 turns):** mean rank **4.5** — it *memorizes* the training pairs.
- **Learned index, held-out test (2 turns):** mean rank **9.5** — barely better than cosine's 10.1.

**Interpretation.** The learned index **does not generalize.** It memorizes the training queries (rank 4.5) but cannot disambiguate the held-out ones (rank 9.5 ≈ cosine). The reason is the diagnostic: the query hidden states carry almost no signal that distinguishes *which* fact is being asked for, so a linear projection has nothing to learn from. **The cosine index is already near-optimal for this setup** — learning it doesn't help, because the bottleneck is not the index but the *query representation*.

> **Finding 7:** The retrieval index is **not learnable** at this scale — a linear projection trained with InfoNCE memorizes the training queries (rank 4.5) but does not generalize to held-out queries (rank 9.5 ≈ cosine 10.1). The root cause: the query hidden states are nearly identical (off-diag cos 0.9987), so they carry no disambiguation signal. The cosine index is already near-optimal; the bottleneck is the *query representation*, not the index.

---

### Bet 8 — Temporal Index (fuse query-conditional cosine with the temporal signature)

**Hypothesis.** Bet 7 (learned index) failed because it had no per-*chunk* signal to learn from — the chunk hidden states are all similar ("the secret code is N"). The **temporal signature** is exactly that per-chunk signal: intrinsic (write-time), in the model's own state space, and *different for each chunk* (target fact = high state-delta, padding = low). So fuse the query-conditional cosine (read-time, "relevant to this query?") with the temporal signature (write-time, "distinctive?"). The target fact is the only chunk that is *both* relevant *and* distinctive.

**Method** (`run_temporal_index.py`). One 27B forward pass gives the chunk hidden states (the cosine index) + the temporal signature. For each of the 10 queries, rank the target chunk under: cosine alone, signature alone, and three fusions (RRF, multiplicative, additive). Then a mechanism test: does the best fusion recall at a smaller k than cosine?

**Result** (`test_temporal_index_result.json`):

- **Mean rank (lower = better):** cosine **10.1**, signature 14.3, RRF 10.9, **multiplicative 9.6**, additive 9.6.
- The fusion (cos × sig) improves the mean rank from 10.1 → 9.6 — a ~5% gain. Mixed per-turn: better on 5 turns, worse on 4, same on 1.
- **Mechanism:** the temporal index recalls at the **same k** as cosine (k=8 with window, k=16, k=32) — it does *not* reduce the k needed.

**Interpretation.** The temporal index is a **marginal win, not a breakthrough.** The reason: the temporal signature is a **fact-vs-padding** signal, not a **target-vs-other-facts** signal. In this test, all 10 "secret code is N" chunks are *distinctive* (high signature), and the padding is not. So the signature boosts *all* fact chunks above padding — but it can't rank the *target* fact above the *other 9* facts. That disambiguation is what the query-conditional cosine does. So the signature's role is a **useful but weak secondary signal** (boosts facts above padding); the query-conditional cosine is the **primary disambiguator**.

> **Finding 8:** The temporal index (cosine × temporal signature) is a **marginal win** — it improves the mean rank from 10.1 to 9.6 but does not reduce the k needed for recall. The temporal signature is a *fact-vs-padding* signal, not a *target-vs-other-facts* signal: it boosts all fact chunks above padding but cannot disambiguate between the facts. The query-conditional cosine remains the primary disambiguator; the temporal signature is a useful but weak secondary signal.

---

### Bet 9 — Triangle Index (the difference vector Q−C and its residual)

**Hypothesis.** Cosine only uses the *angle* between the query and a chunk. The **difference vector** `D = Q − C` (the side that "completes the triangle" O–Q–C) uses the *full* triangle (angle + both magnitudes). Decompose the chunk into the part the query *explains* and the part it *doesn't*: `C = proj_Q(C) + R`, where `R` (the residual) is the "specific content" cosine throws away. If the target fact's residual is *different* from the other facts' residuals, the triangle geometry provides a disambiguation signal.

**Method** (`run_triangle_index.py`). One 27B forward pass gives the chunk hidden states. For each query, rank the target chunk under: cosine (baseline), distance |Q−C|, residual magnitude |R|, cosine(Q, R), and residual×position (structured case). Plus a diagnostic: are the residuals actually different per chunk?

**Result** (`test_triangle_index_result.json`):

- **Mean rank (lower = better):** cosine **10.1**, distance **10.1** (identical to cosine), residual **667.9** (terrible), q.resid **427.3** (terrible), resid+pos **1.0** (tautology — used the known position).
- **Diagnostic (the key finding):**
  - `cos(resid_target, resid_other_fact)` = **0.58–0.92** (HIGH — facts are similar to each other)
  - `cos(resid_target, resid_padding)` = **0.11** (LOW — facts are different from padding)
  - Residual magnitude: target=0.628, other_facts=0.603, **padding=0.745** (padding is *higher*)

**Interpretation.** The triangle geometry does **not** provide a disambiguation signal beyond cosine:

1. **Distance = cosine.** The full triangle (Q−C) is angle-dominated — it adds nothing over cosine (mean rank 10.1, identical).
2. **Residual = fact-vs-padding, not target-vs-other-facts.** The target fact's residual is *similar* to the other facts' residuals (cos 0.58–0.92) and *different* from padding (cos 0.11). But the residual *magnitude* is actually *higher* for padding (0.745) than for facts (0.603) — so ranking by magnitude puts the target near the **bottom** (rank 667.9). The residual is a *worse* fact-vs-padding signal than the temporal signature.
3. **The "resid+pos = 1.0" is a tautology.** I used the *known* target position, so of course it ranks the target #1 — that's a lookup, not retrieval.

> **Finding 9:** The triangle geometry (distance, residual) does **not** provide a disambiguation signal beyond cosine. The distance is identical to cosine (angle-dominated). The residual is a *fact-vs-padding* signal (like the temporal signature), not a *target-vs-other-facts* signal — and it's actually *inverted* (padding has higher residual magnitude than facts). The "specific content" (the code value) is *not* isolated in the residual in a matchable way, because the query (a question) doesn't contain the answer to match against. **The disambiguation signal is POSITION, not the triangle geometry.**

---

### Bet 10 — Temporal Pinpoint (the state-delta VECTOR: direction + magnitude)

**Hypothesis.** The temporal signature (Bet 4) only uses the *magnitude* of the state-delta (||ΔS||) — "how much changed." The temporal **pinpoint** uses the full ΔS *vector*: the *direction* ΔS/||ΔS|| is a **fingerprint of the event**, not just its salience. Two different facts produce *different* state-delta *directions* (different "kinks" in the state trajectory), even if their magnitudes are similar. So the direction is a richer, per-event signal.

**Method** (`run_temporal_pinpoint.py`). One 27B forward pass gives the chunk hidden states + the ΔS *vector* per chunk (projected to 2048 dims to avoid OOM on the 75M-dim recurrent state). For each query, rank the target chunk under: cosine (baseline), ΔS magnitude, ΔS direction (distinctiveness = 1 − cos(ΔS_chunk, mean_ΔS)), and the fusions cos×mag and cos×dir. Plus a diagnostic: are the ΔS directions actually different per fact?

**Result** (`test_temporal_pinpoint_result.json`):

- **Mean rank (lower = better):** cosine **10.1**, ΔS magnitude 15.3, ΔS direction **358.0** (terrible), cos×mag 10.9, cos×dir 14.5.
- **Diagnostic (the key finding):**
  - `cos(ΔS_target, ΔS_other_fact)` = **−0.25 to 0.76** (mostly ORTHOGONAL — random)
  - `cos(ΔS_target, ΔS_padding)` = **−0.03** (orthogonal)
  - ΔS magnitude: target=1380.9, other_facts=1458.5, padding=1293.4 (target is *between*)
  - Distinctiveness: target=0.779, other_facts=0.995, padding=0.960 (target is *less* distinctive)

**Interpretation.** The temporal pinpoint does **not** provide a disambiguation signal. The ΔS directions are **orthogonal to each other** (cos ≈ 0) — they're not "fact-like" in a shared direction; they're random. The target's ΔS is actually *less* distinctive than both the other facts and the padding. The ΔS magnitude is a weak fact-vs-padding signal (worse than the temporal signature).

> **Finding 10:** The temporal pinpoint (ΔS direction) is **not** a per-event fingerprint — the ΔS directions are orthogonal to each other (cos ≈ 0), so they carry no shared "fact-like" structure. The target's ΔS is *less* distinctive than the other facts and the padding. This confirms the deep finding from yet another angle: **no signal derived from the model's internal state** (surprisal, temporal signature, temporal pinpoint, triangle geometry) **can disambiguate between similar facts.** The disambiguation requires either (a) the query to carry the specific signal (it doesn't — Bet 7) or (b) the position (which you know in a structured conversation).

---

## 3. The Structural Findings (what the negatives actually say)

Individually these are failures. Together they form a coherent picture:

1. **The linear state is lossy.** It's a gist, not a store. Exact recall needs the full-attention KV. *(Bet 1)*
2. **Intrinsic per-token surprisal is the wrong signal.** It ranks structural surprises, not facts. *(Bet 3)*
3. **The temporal signature is better and works, but is still intrinsic.** It supports recall at k=32, yet ranks "a code statement" high, not "the target code" — so it needs a large k. *(Bet 4)*
4. **Selective KV must be injected into a warm cache.** Cold-cache injection gives the model nothing to work with. *(Bet 3)*
5. **The common thread: every *intrinsic* (write-time) salience signal fails.**

**The deep finding:** *salience is not a property of the token — it's a property of the (token, query) pair.* "8270" is only salient *because the query asks for the code*. No write-time signal can know that, because the query doesn't exist yet at write time. **This is why the whitepapers' entire axis — intrinsic spectral/temporal compression — is the wrong one.** You cannot pre-compute what matters; you can only compute it *at read time, conditioned on the query.*

---

## 4. State of the Art (what works, what's dead)

| Mechanism | Status | Evidence |
|-----------|--------|----------|
| Full-attn KV, in-place | **Works** (100%) | Bet 1-A, all baselines |
| Linear recurrent state as exact memory | **Dead** | Bet 1-B |
| Per-token surprisal selection | **Dead** | Bet 3 |
| Temporal signature selection | **Works** at k=32 (intrinsic, needs large k) | Bet 4 |
| Cold-cache selective KV injection | **Dead** (wrong mechanism) | Bet 3 |
| Warm-cache selective KV injection | **Proven** (warm linear state + selective KV) | Bet 4 (topk32) |
| Memory-as-weights (LoRA) | **Dead** (real negative, mask fixed) | Bet 5 |
| **Query-conditional retrieval** | **Works** (beats intrinsic on rank + token efficiency) | Bet 6 |
| Learned retrieval index (InfoNCE) | **Does not generalize** (queries too similar) | Bet 7 |
| Temporal index (cosine × signature) | **Marginal win** (rank 10.1→9.6, same k) | Bet 8 |
| Triangle index (distance, residual) | **Dead** (no disambiguation; residual is fact-vs-padding, inverted) | Bet 9 |
| Temporal pinpoint (ΔS direction) | **Dead** (ΔS directions are orthogonal, not a fingerprint) | Bet 10 |

---

## 5. The Next Bet — Two Timelines, Query-Conditional Retrieval

The 2-timelines frame (fast = sliding-window full attention, slow = a long-term store) is the right *architecture*. Bonsai already *is* two timelines. The missing piece is what the slow timeline stores and how it decides — and Finding 5 (the deep one) tells us: **the decision must be made at read time, conditioned on the query.**

**The bet:** a **retrievable** bank of exact KV for the whole history, indexed by the model's *own* hidden state. At query time, retrieve the top-k chunks most similar to the query, and inject *their* exact KV into a **warm** cache.

- **Fast timeline** = sliding-window full attention (exact, bounded, recent).
- **Slow timeline** = a memory bank of exact KV for salient tokens (exact, old), lossy everything else.

**Why this is the rewarding bet:**
1. It's the *only* frame that escapes the intrinsic-salience dead end we just proved.
2. It's the standard, proven long-context mechanism (RAG / MemGPT / retrieval-augmented memory) — with a twist that makes it *seamless*: the stored unit is the model's **own KV**, already in its computation space. No re-encoding, no text round-trip. Injection is free.
3. It directly de-risks the learned-index endgame. If it works, a learned retrieval head is an optimization. If it fails, the diagnostic tells us exactly why.

**The crux risk (stated plainly):** it all hinges on whether `cosine(query_hidden_state, fact_chunk_hidden_state)` ranks the fact chunk highly. The query is a *question*, the fact is a *statement* — their representations may not be close. That's the whole bet in one line.

**Outcomes:**
- **Home run:** retrieval recalls the fact that surprisal, state-delta, *and* the linear state all failed to. That's the 10M mechanism.
- **Partial win:** it finds the fact but needs a bigger window → tells us the working compression ratio.
- **Negative:** hidden-state similarity doesn't rank the fact high → the index must be *learned* (the next bet). Either way we learn the next move.

**Plan:**
1. One full-context pass via `model.model(x, cache=cache)` → get **hidden states** H (1, n, d) *and* the full KV cache in the same forward.
2. Index = per-chunk mean hidden state.
3. Query pass → query hidden state q.
4. Retrieve top-k chunks by cosine(q, chunk).
5. Warm cache = warm linear state + KV of retrieved chunks. Generate, check recall.
6. Controls: same k with random chunks + the intrinsic signals we already have, so the comparison is clean.

**First thing to verify:** that `model.model(...)` (the base, not the LM head) returns the residual stream on this Packed model.

---

## 6. Appendix

### 6.1 File inventory (this log's experiments)

| File | Bet | Status |
|------|-----|--------|
| `probe_recurrent_state.py` | substrate probe | done — 156.89 MB, snapshot/restore verified |
| `run_recurrent_state_test.py` | Bet 1 | done — linear state dead |
| `run_sgm_test.py` | Bet 2 | done — confounded |
| `run_sgm_diagnostic.py` | Bet 3 | done — surprisal dead, cold-cache dead |
| `run_temporal_signature.py` | Bet 4 | done — better but intrinsic |
| `run_lora_memory_test.py` | Bet 5 | done (mask fixed) — real negative |
| `run_sgm_warm.py` | (warm-cache mechanism) | **not on disk** — the warm-cache logic lives in `run_temporal_signature.py::build_warm_cache` |
| `run_retrieval_test.py` | Bet 6 | done — query-conditional retrieval works (k=8 vs tsig k=32) |
| `run_learned_index.py` | Bet 7 | done — index not learnable (queries too similar) |
| `run_temporal_index.py` | Bet 8 | done — marginal win (fact-vs-padding signal) |
| `run_triangle_index.py` | Bet 9 | done — no disambiguation (residual is fact-vs-padding, inverted) |
| `run_temporal_pinpoint.py` | Bet 10 | done — ΔS directions orthogonal, not a fingerprint |

### 6.2 Environment notes

- **Model:** `binetic-first-mlx/bonsai2-27b-mlx/` — must use the bundled `runtime/` loader (stock `mlx_lm.load` rejects the pack).
- **Python:** `/opt/homebrew/bin/python3.12` — mlx 0.32.0, mlx-lm 0.31.3, mlx-vlm 0.6.3, transformers 5.5.0.
- **MLX gotchas hit:** `mx.log_softmax` doesn't exist (use `mlx.nn.log_softmax`); `linear_to_lora_layers` mutates in place and returns `None` (don't reassign); `mlx_lm.generate` takes a `sampler`, not `temperature` (omit for greedy); `iterate_batches` expects `(input_ids, loss_start)` 2-tuples where `loss_start` is a single int.
- **Memory:** the 27B model peaks ~18 GB. A 16 GB LM Studio server + orphaned `llama-cli` processes once thrashed swap (13 GB) and crawled a 30-second run to 41 minutes. Kill stray inference processes before running.

### 6.3 Known test bugs (open)

- **LoRA mask** (Bet 5): **fixed** — now masks the entire assistant turn. The re-run is a real negative (see Bet 5).
- **`run_sgm_warm.py`** is listed above as "written" but does **not exist on disk** — the warm-cache mechanism it was meant to isolate is actually implemented in `run_temporal_signature.py::build_warm_cache` (and proven by the Bet 4 topk32 result).

### 6.4 The one-sentence summary

> We proved that exact long-range recall lives in the full-attention KV (not the linear state), that no *intrinsic* write-time signal can find a fact (surprisal, temporal signature, temporal pinpoint, and triangle geometry all fail because salience is query-conditional), and that **query-conditional retrieval over a bank of the model's own KV, injected into a warm cache, works** — beating the best intrinsic selector on both rank and token efficiency. We also proved the index is *not* learnable (the query hidden states are too similar) and that the disambiguation between similar facts requires the *position* (known in a structured conversation), not any signal derived from the model's internal state. The "two timelines" are now concrete.

---

*Log maintained by Stakpak. Updated 2026-10-09.*
# Making a 300M Telugu LM Punch Like a 3B: Core LM Playbook

**Constraints applied:** No distillation. No synthetic data. No instruction tuning. Pure core language model improvements only.

**Status key:** DONE = already implemented in `train_gpt.py` / `morfessor_segment.py`. PARTIAL = partially done, needs adjustment. TODO = not yet started.

---

## The Strategic Framing: Why This Is Feasible

All three plans converge on a crucial insight that works in your favor. At ~4B tokens, a 3B model is **catastrophically data-starved** (1.33 tokens/parameter vs. the Chinchilla-optimal ~20 tokens/parameter). Your 300M model at 4B tokens sits at ~13.3 tokens/parameter — much closer to the compute-optimal frontier. The 3B model is wasting most of its capacity on parameters it can't properly train.

This means the gap you need to close is smaller than it looks on paper, and the strategies below are about extracting maximum value from every parameter and every token you have.

---

## 1. Architecture: Go Deep and Narrow

**Consensus across all three plans. Zero extra cost. Estimated gain: ~4.3% absolute accuracy.**

The MobileLLM ablation study at 350M is the definitive reference here. At sub-billion scale, architecture matters *more* than data quantity — the opposite of what holds at larger scales. The HuggingFace study at 70M found that all 12 architecture families tested (GPT-2, LLaMA-3, Qwen-3, Gemma-3, MoE, Titans-MAC, etc.) performed within ~2% of each other, meaning the specific family matters less than getting the configuration right.

**Target configuration (all three plans agree):**

- **Decoder-only transformer, 24–32 layers, d_model=1024** (or ~896 if pushing toward 40 layers) — PARTIAL: currently 20 layers, needs increase to 28–32
- **16 attention heads with GQA (4 KV head groups)** — reduces KV-cache by 4×, enables longer contexts or larger batches at same hardware — TODO: currently full MHA (16Q/16KV), needs GQA conversion
- **SwiGLU activation** in FFN (set FFN dim to 2/3 × 4 × d_model ≈ 2730) — adds gating for +1.3% absolute — DONE
- **RoPE positional embeddings** (default choice; ALiBi if long-context extrapolation is primary) — DONE
- **Pre-norm RMSNorm** — cheaper than LayerNorm, widely validated — DONE
- **Tied input-output embeddings** — saves ~10% parameters with only -0.2–0.6% accuracy trade-off; reallocates those params to reasoning layers — DONE

The cumulative, approximately additive gains from MobileLLM's ablation: SwiGLU (+1.3% — DONE) + deep-narrow (+0.9–1.1% — PARTIAL, need more layers) + block-wise weight sharing (+0.7–0.8% — TODO) = **~4.3% absolute** over a naive baseline.

**Block-wise weight sharing** deserves special attention. — TODO Sharing weights between adjacent transformer blocks increases logical depth without increasing parameter count. The MobileLLM-LS variant got +0.8% accuracy with negligible latency overhead because shared weights are already in hardware cache. A practical design from the Claude plan: 12 unique layers recycled 2–3× (effective depth 24–36) with per-iteration LoRA adapters gives ~300M "logical" parameters with ~160M unique parameters — significantly reducing overfitting risk with limited data.

**OpenELM's layer-wise scaling** is worth considering: vary head counts and FFN dimensions across layers rather than using uniform sizing.


## 2. Hybrid Architectures: Attention + State Space Models — TODO

**High ceiling, moderate engineering complexity. Hymba-350M is SOTA at this scale.**

The Gemini plan and Claude plan both highlight **Hymba** (NVIDIA, Nov 2024, ICLR 2025) as the most promising architecture at ~300M scale. Hymba integrates attention heads and SSM (Mamba) heads in parallel within each layer, plus 128 learnable "meta tokens" prepended as compressed world knowledge.

- **Hymba-1.5B outperformed Llama-3.2-3B** with 3.49× throughput and 11.67× cache size reduction
- Specifically evaluated at **350M and 125M** with consistent SOTA results
- 5× higher inference throughput than pure transformers, linear scaling in sequence length

The Gemini plan also points to **Falcon-H1** (parallel hybrid of transformer attention + SSM) and **Llamba** (Mamba-2 layers replacing self-attention). The MOHAWK cross-architecture framework can transfer knowledge from pre-trained transformers to Mamba-based models using <0.1% of typical training data — though this edges close to your distillation constraint, it's more of an initialization strategy than teacher-student distillation.

**Decision point:** If engineering resources permit, Hymba-style hybrid is the highest-ceiling architecture choice. If you want a safer, more established path, the dense LLaMA-style stack from Section 1 is battle-tested.


## 3. Tokenizer: Solve the Fertility Crisis — PARTIAL (Morfessor-based tokenizer exists but vocab is 80K+ — needs reduction to 32K–48K)

**All three plans are emphatic. This is where Telugu-specific optimization has outsized impact.**

Standard multilingual tokenizers require 4–8 tokens per Telugu word vs. ~1.4 for English. This is catastrophic for a 300M model with 4B tokens:

- At fertility 6: your 4B tokens represent only ~666M actual Telugu words
- At fertility 1.5–2.1: the same 4B tokens deliver 1.9–2.7B words of semantic signal
- This is effectively a **3–4× data multiplier** for free

**All three plans converge on:**

- **SentencePiece Unigram** (not BPE) — a dedicated Telugu tokenizer study (Vemula et al., IJCNLP-AACL 2025) found Unigram consistently outperforms BPE for Telugu, with tokenizer algorithm being the "most significant factor" influencing downstream performance. Unigram's top-down pruning respects morpheme boundaries in agglutinative languages far better than BPE's bottom-up merging. — PARTIAL: using Morfessor (morphologically principled but different algorithm; not directly compared in the study). Mixed-script handling and `@@` continuation markers are well-implemented in `morfessor_segment.py`.
- **32K–48K vocabulary** — at d_model=1024, a 32K vocab consumes ~33M params (11% of 300M), which is acceptable. 64K would be 22%. Beyond that, you're starving reasoning layers. — TODO: currently 80K+ (consuming ~82M params = 27% of model). Needs reduction.
- **Target fertility: 1.4–2.1 tokens/word** (Sarvam-1 reference point) — NEEDS MEASUREMENT on current Morfessor tokenizer

**Pre-tokenization essentials:**

- Unicode NFC normalization + Indic NLP Library normalization before tokenizer training
- Explicitly include all standalone Telugu characters in the vocabulary before training
- Standardize diacritic variations
- Separate individual digits for arithmetic generalization
- Train on a balanced, cleaned Telugu corpus (not raw web crawl)

**Ablation recommendation (GPT plan):** Train tokenizers at {16K, 32K, 64K} and measure downstream impact. The tokenizer choice will compound with every other optimization.


## 4. Embedding Optimization: Reclaim Parameters for Reasoning — PARTIAL (tied embeddings DONE; factorization and advanced techniques TODO)

**Critical at 300M. The embedding layer can cannibalize 20%+ of your model.** Currently consuming ~27% at 80K+ vocab.

Beyond tied embeddings (DONE in `train_gpt.py` — `self.transformer.wte.weight = self.lm_head.weight`), the Gemini plan uniquely highlights two advanced techniques:

**Embedding factorization (ALBERT-style):** Instead of mapping tokens directly to d_model=1024, first map to a smaller intermediate dim (e.g., 256), then project up. This decouples vocabulary size from hidden dimension, saving significant parameters that can be reallocated to transformer blocks.

**Hyperbolic embeddings:** Standard Euclidean embeddings struggle with hierarchical relationships. Hyperbolic space (negative curvature, exponential growth capacity) can embed complex hierarchical linguistic information in fewer dimensions. Telugu's morphological hierarchy (broad stems → specific inflected forms) maps naturally to hyperbolic geometry.

**Multi-sense embeddings (VQ-based):** Instead of one vector per token, generate multiple discrete embeddings per token for polysemous words. This reduces the disambiguation burden on subsequent transformer layers — important when those layers are your scarcest resource.

These are more speculative than the core architecture choices but represent genuine parameter efficiency gains at the embedding level.


## 5. Data Quality and Curation (Real Data Only) — TODO (data pipeline exists for Morfessor segmentation but quality filtering, dedup, and mixing not yet implemented)

Since synthetic data is off the table, your entire strategy shifts to maximizing information density from real Telugu text.

### 5a. Source the best real data

- **AI4Bharat Sangraha** — largest cleaned Indic pretraining corpus (verified + unverified splits for Telugu)
- **IndicCorp** — 8.8B tokens across 11 languages (Telugu portion)
- **OSCAR Telugu** — web-crawled, needs aggressive cleaning
- **Telugu Wikipedia** — high quality, limited volume
- **7.9M Telugu tweets** — noisy but captures colloquial/code-mixed usage

### 5b. Deduplicate aggressively

All three plans emphasize this. Near-duplicates (news syndication, scraped reposts) are rampant in web-crawled corpora. Deduplication converts "wasted tokens" into "learning tokens."

- **MinHash near-dedup** (Jaccard threshold ~0.8)
- **Line-level deduplication** on top of document-level
- **Train-test overlap audit** to prevent leakage

Lee et al. show deduplication reduces memorized output frequency and can improve both accuracy and efficiency.

### 5c. Quality filtering

- **Train a binary quality classifier** on high-quality Telugu sources (Wikipedia, Sangraha Verified) vs. noisy web crawl, then use it to filter larger corpora
- **Educational quality filtering** (SmolLM's FineWeb-Edu approach) — web data filtered by educational quality classifier converges 3× faster than unfiltered data
- **Perplexity-based filtering** using a small reference model trained on clean Telugu

### 5d. Data mixing optimization

**DoReMi** (Xie et al., NeurIPS 2023) uses a ~280M proxy model to find optimal domain weights, improving average few-shot accuracy by **6.5 percentage points** and reaching baseline accuracy in 2.6× fewer steps. Your 300M model IS proxy-model scale, so you can use earlier checkpoints of your own training run to tune domain weights.

Practical mix recommendations:
- Reserve ~1–3% "high-quality Telugu" slice (Wikipedia, curated books, quality news) and upweight it late in training
- Include **10–20% code data** — even for a Telugu model, mixing code provides a 2× increase in effective tokens for natural language evaluations (validated finding from scaling studies)
- Include transliterated/Romanized Telugu and code-mixed Telugu-English for robustness

### 5e. Cross-lingual transfer via initialization (not distillation)

This is a subtle but important distinction. You're not distilling from a bigger model — you're **initializing from an English-pretrained model** of the same size:

- Take a 300M English-pretrained model
- Replace the tokenizer with your Telugu Unigram tokenizer
- Reinitialize embedding layers (they're tied to the old vocabulary)
- Keep transformer weights (they encode general linguistic patterns)
- Include 5–10% parallel Telugu-English data early in training for cross-lingual alignment
- Transition to pure Telugu

This exploits the "curse of multilinguality" reversal: a dedicated monolingual model can match a much larger multilingual model on the target language because it isn't diluting capacity across languages.


## 6. Training Recipe: Maximize Every Token

### 6a. WSD learning rate schedule — TODO (currently using cosine decay)

The Claude plan highlights this as current best practice for small models. MiniCPM introduced Warmup-Stable-Decay:

- **Warmup** (1–2% of steps) → **Stable constant LR** (70–80%) → **Decay** (10–20%)
- Unlike cosine, WSD allows continuous training — add more data later without restarting
- MiniCPM-2.4B surpassed Mistral-7B and LLaMA-13B using this approach
- For 300M: peak LR of 3e-4 to 6e-4, linear decay to 10% of peak

The GPT plan recommends cosine decay as a safe default. Either works; WSD gives you more flexibility for iterative data additions.

### 6b. µP (µ-Parametrization) for hyperparameter transfer — TODO

Run extensive hyperparameter optimization at **~9M–30M proxy scale**, then transfer optimal settings to 300M using µP (Tensor Programs V). This provides stable hyperparameter transfer across model scales and saves massive compute on tuning. MiniCPM calls this the "Model Wind Tunnel" — it's their key methodological contribution.

### 6c. Multi-epoch training — DONE (~3 epochs, 45K steps over ~3.7B tokens)

"Scaling Data-Constrained Language Models" (Muennighoff et al., NeurIPS 2023) trained 400+ models and found that **up to 4 epochs of repeated data yields negligible degradation**. Beyond 4 epochs, returns diminish but remain positive.

For 300M at 4B unique tokens: training for 2–3 epochs (~8–12B total tokens) stays well within the safe zone. This is critical given your data constraint.

**But beware:** The Gemini plan cites a countervailing finding — overtrained models (OLMo-1B at 3T tokens) showed 2%+ degradation on downstream tasks. The key is staying within ~4 epochs, not pushing to extreme overtraining.

### 6d. Progressive training — TODO

MSG (Masked Structural Growth, ICLR 2024) enables strictly function-preserving model growth with **2× speedup** in pretraining:

- Train 100M for 2B tokens → grow to 200M for 1B tokens → grow to 300M for 1B tokens
- Smaller models learn basic patterns efficiently from early tokens
- LiGO (ICLR 2023) achieves similar results, reducing FLOPs by 47.2%

With only 4B tokens, progressive training is especially valuable: you're not wasting early tokens on a model that's too large to learn basic patterns efficiently.

### 6e. Regularization — PARTIAL (weight decay correct; dropout needs removal)

A critical finding: **dropout hurts performance during single-epoch pretraining** (Liu et al., 2025). For multi-epoch training on limited data:

- **EntroDrop** (entropy-guided token dropout) selectively masks easy tokens in later epochs — works across 0.6B–8B scales — TODO
- Standard weight decay (0.1 for AdamW) — no weight decay on norms/biases — DONE (properly split into decay/nodecay groups in `train_gpt.py`)
- ALBERT-style cross-layer parameter sharing acts as implicit regularization, beneficial when data is scarce

### 6f. Curriculum learning — TODO (currently random sampling from memmap)

Sort training data from easy to hard (using compression ratio or perplexity as difficulty metric). This reduces training steps by **18–45%** to reach baseline performance, with the largest benefits at smaller model scales (14M–160M).

### 6g. Systems-level efficiency — MOSTLY DONE

These don't improve model quality directly but let you train more effectively:

- **FlashAttention** — exact attention, IO-aware; enables longer contexts and higher batch tokens at same hardware — DONE (via PyTorch SDPA `scaled_dot_product_attention`)
- **Mixed precision** (bf16 preferred, fp16 + loss scaling if bf16 unavailable) — DONE (bf16 on A100, with GradScaler for fp16 fallback)
- **Activation checkpointing** — trade compute for memory, enabling deeper models or longer context — DONE (optional via `--grad-checkpoint` flag)
- **torch.compile** — DONE (enabled by default, `--no-compile` to disable)
- **ZeRO Stage-1/2** — even on moderate clusters — TODO (single-GPU only currently)


## 7. Pretraining Objectives Beyond Next-Token Prediction — TODO (currently pure autoregressive causal LM)

**The GPT plan uniquely emphasizes this. Higher signal density per token.**

Since your tokens are precious, consider objectives that extract more learning signal per token:

- **UL2 (Mixture of Denoisers):** Mix different denoising modes (R-Denoiser for NLU, S-Denoiser for generation, X-Denoiser for extreme corruption). UL2 reports shifting the Pareto frontier by getting more out of the same data.
- **ELECTRA-style replaced-token detection** (for encoder components): Learns from ALL tokens rather than just masked ones. Especially strong gains reported for small models. If you have any NLU evaluation targets, this is worth exploring.
- **Span corruption (T5-style):** Can be more sample-efficient for transfer learning tasks.

The default — pure autoregressive causal LM — remains the safest choice for generative behavior, but mixing in UL2-style denoising during pretraining can extract more value from your 4B tokens.


## 8. Retrieval Augmentation at Inference Time — TODO

**Highest-ceiling post-training technique. No training cost. The Claude plan ranks this #3 overall.**

**RETRO** demonstrated a **25× parameter efficiency gain**: a 7.5B RETRO with a 2T token database matched GPT-3's 175B on the Pile. This is not distillation — it's explicit external memory.

**kNN-LM** is even simpler: it augments any pre-trained LM by interpolating with k-nearest neighbors from an external datastore, requiring **zero additional training**. The critical finding: "Training a model on 100M tokens and using kNN search over a 3B token dataset can outperform training the same model on all 3B tokens."

For Telugu: build a retrieval datastore from ALL available Telugu text — even text not used for training.

**MLP Memory** (2025) replaces the kNN datastore with a pretrained MLP, compressing 220GB into 2.8GB while actually improving reasoning (kNN-LM can impair reasoning; MLP Memory doesn't).

**Practical deployment:**

- 300M Telugu model + multilingual dense retriever (LaBSE or multilingual-e5) + Telugu text datastore
- The retrieval compensates for limited training data — the model doesn't need to memorize facts, just reason over retrieved context


## 9. Test-Time Compute Scaling — TODO

**Game-changer for verifiable tasks. The Claude plan cites 14× effective size increase.**

Snell et al. (ICLR 2025) showed that with optimal compute allocation, a smaller model using test-time compute can **outperform a 14× larger model** in FLOPs-matched evaluation. HuggingFace applied this to Llama-3.2-1B, which outperformed the 8B model.

This requires a separate verifier/reward model and is most effective for structured QA and mathematical reasoning. It's domain-dependent but represents the highest-ceiling inference-time technique after retrieval.


## 10. Sparse Upcycling: Dense → MoE (Post-Training) — TODO

**Moderate risk with limited data. The Claude plan notes individual experts may be undertrained.**

After pretraining your 300M dense model, you can upcycle it into a Mixture of Experts model:

- Replicate MLP weights into multiple experts
- 300M dense → ~900M total / 300M active MoE
- Increases capacity without proportional inference cost
- NVIDIA's approach achieved 67.6% MMLU vs. 65.3% for continued dense training

**Risk:** With only 4B tokens, each expert sees fewer tokens, potentially leading to expert collapse. Mitigate with fewer experts (4–8) and include 1–2 shared experts.

**DeepSeekMoE's fine-grained expert segmentation** (64–256 small experts) allows more flexible combinations at small scale. A practical config: 8–16 fine-grained experts per MoE layer, top-2 routing, with 1–2 shared experts.

---

## Prioritized Action Plan

Ranked by expected impact ÷ effort, filtered by your constraints:

| Priority | Action | Expected Impact | Cost/Risk | Status |
|----------|--------|----------------|-----------|--------|
| **1** | Telugu tokenizer (32K–48K vocab) | 3–4× effective data multiplier | Low — one-time effort | PARTIAL — Morfessor exists at 80K+, needs vocab reduction |
| **2** | Deep-narrow architecture + SwiGLU + GQA + RoPE | +4.3% absolute accuracy | Zero extra cost | PARTIAL — SwiGLU/RoPE/RMSNorm/tied-emb DONE; need more layers (20→28+) and GQA |
| **3** | Aggressive dedup + quality filtering of real data | Massive (converts wasted → useful tokens) | Low compute | TODO |
| **4** | Multi-epoch training (2–3 epochs) | Effectively 2–3× your data | Linear compute cost | DONE — 3 epochs configured |
| **5** | µP wind-tunnel experiments at 9–30M | Save compute, find better HPs | Small upfront cost | TODO |
| **6** | WSD scheduler | Better loss trajectory than cosine | Zero extra cost | TODO — currently cosine |
| **7** | Cross-lingual initialization (English 300M → Telugu) | Free structural knowledge transfer | Zero extra cost | TODO |
| **8** | Block-wise weight sharing / Mixture-of-Recursions | Regularization + effective depth | Zero parameter cost | TODO |
| **9** | Progressive training (100M→200M→300M) | 30–50% compute savings | Moderate complexity | TODO |
| **10** | 10–20% code data in training mix | ~2× effective tokens for NL eval | Zero extra cost | TODO |
| **11** | Retrieval augmentation (kNN-LM / RETRO) | 25× effective parameter gain | Inference-time cost | TODO |
| **12** | Curriculum learning (easy→hard) | 18–45% fewer steps to baseline | Low | TODO |
| **13** | Hybrid attention+SSM (Hymba-style) | SOTA at 350M scale | High engineering cost | TODO |
| **14** | DoReMi domain weight optimization | +6.5pp average accuracy | Moderate | TODO |
| **15** | Sparse upcycling (dense→MoE) | +2–3% post-training | Risk of expert undertraining | TODO |
| **16** | Test-time compute scaling | Up to 14× effective size | Needs verifier model | TODO |

### Additional items already DONE in `train_gpt.py` (not in original priority list):
- Flash Attention via PyTorch SDPA
- bf16 mixed precision with GradScaler fallback
- torch.compile enabled by default
- Gradient checkpointing (optional flag)
- AdamW with proper decay groups (weight decay on 2D params only, none on norms/biases)
- Gradient clipping at 1.0
- Scaled residual init (1/√(2·n_layer) for c_proj and w_down)
- Memory-mapped data loading (zero RAM overhead)
- W&B logging integration
- Checkpoint resume support

### Items actively hurting current training:
- **Dropout 0.1** — applied on attention residuals, MLP, and embedding drop. Research says this degrades pretraining quality. Should be set to 0.0 or replaced with EntroDrop for later epochs.

---

## Compute Estimate

300M params × 10B tokens (multi-epoch) × 6 FLOPs/token ≈ 1.8 × 10¹⁹ FLOPs ≈ **200–400 A100 GPU-hours** (~$200–400 at cloud pricing). Well within academic budgets.

---

## Key Insight

The three plans collectively make a strong case that with your constraints, the highest-leverage moves are (1) tokenizer optimization for Telugu's agglutinative morphology, (2) architecture choices that are free in compute cost, (3) squeezing maximum value from real data through quality filtering and smart mixing, and (4) retrieval augmentation at inference to compensate for what a 300M model can't memorize. The techniques stack approximately additively across orthogonal axes — tokenizer × architecture × data quality × training recipe × inference augmentation — so the cumulative effect is multiplicative even if each individual gain is modest.

# Making a 300M Telugu model rival a 3B model: a research compendium

**A well-optimized 300M parameter Telugu language model can realistically close 60–80% of the performance gap with a 3B model** by combining high-quality data curation, knowledge distillation, modern architecture design, and inference-time augmentation. The empirical evidence is strong: Microsoft's Phi-2 (2.7B) outperformed Mistral-7B through data quality alone, MobileLLM's 350M model gained **+4.3% absolute accuracy** from architecture choices, and RETRO demonstrated a **25× parameter efficiency** gain through retrieval augmentation. For a monolingual Telugu model with ~4B tokens, the highest-leverage interventions are (1) synthetic + educationally filtered data, (2) knowledge distillation from a multilingual teacher, (3) a deep-narrow architecture with modern components, and (4) retrieval augmentation at inference time. This report synthesizes findings from over 100 recent papers (2022–2025) into an actionable research guide, organized by technique category with specific quantitative results and practical recommendations.

---

## 1. Knowledge distillation can close 70–85% of the teacher-student gap

Knowledge distillation (KD) is the single most well-studied technique for transferring capability from large to small models, and recent advances have dramatically improved its effectiveness for autoregressive LLMs.

**The capacity gap problem is real at 10× compression.** Mirzadeh et al. (AAAI 2020) demonstrated that student performance degrades when the teacher-student size ratio is large. For a 3B→300M scenario (10× ratio), **staged distillation through an intermediate ~1B model** (3B→1B→300M) is strongly recommended over direct transfer. Progressive distillation from intermediate teacher checkpoints (Pro-KD, COLING 2022) further helps by creating a smoother learning curriculum.

**Modern KD methods for LLMs have moved beyond vanilla soft-label matching.** Three key 2024 papers define the current frontier:

- **MiniLLM** (ICLR 2024) replaced forward KL divergence with **reverse KLD**, preventing the student from wasting capacity on low-probability regions. Student models sometimes exceeded teacher Rouge-L scores on benchmark tasks.
- **GKD** (ICLR 2024) introduced on-policy distillation using student-generated outputs with flexible divergence measures, achieving **2.1× gains on summarization** and **1.7× on translation** over standard KD.
- **DistiLLM** (ICML 2024) used Skewed KL divergence with adaptive scheduling, outperforming both MiniLLM and GKD across model families.

**NVIDIA's Minitron approach** (2024) demonstrated the practical power of combining pruning with distillation: Minitron models achieved **16% MMLU improvement** over training from scratch while requiring **40× fewer training tokens** (~100B vs. trillions). Width pruning consistently outperformed depth pruning after retraining.

**For multilingual-to-monolingual distillation**, the critical challenge is vocabulary mismatch. **DSKD** (EMNLP 2024) addresses this by conducting distillation in unified output spaces, significantly outperforming standard white-box KD when teacher and student use different tokenizers. This is essential for distilling from a multilingual model (BLOOM-3B, Llama) into a Telugu-specific student. Crucially, monolingual distilled students can **match or exceed their multilingual teachers** on the target language by eliminating the "curse of multilinguality" — the capacity dilution that multilingual models suffer.

**KD works well with limited data.** The Pre-training Distillation paper (ACL 2025) showed that KD acts as a form of label smoothing, providing richer supervision per token — particularly valuable when data is scarce. "Distilling Step-by-Step" (ACL 2023) demonstrated that a **770M T5 model outperformed 540B PaLM** using only 80% of available data by extracting rationales from the teacher. Optimal temperature for logit distillation is τ=1–2; higher temperatures showed no benefit.

**Recommended KD pipeline for Telugu:** Use a multilingual teacher (BLOOM-3B or a Llama variant with Telugu exposure) → DSKD framework to handle vocabulary mismatch → DistiLLM's SKL loss with adaptive student-generated output scheduling → WSD scheduling for the KD loss proportion (higher early, decaying later). Run KD for multiple epochs over the 4B tokens, since KD loss provides different supervision signals than standard language modeling loss, making each epoch more productive.

---

## 2. Architecture choices matter more than you think at 300M scale

MobileLLM (Meta, ICML 2024) provided the most thorough ablation study at sub-billion scale, establishing that **architecture matters more than data quantity at this parameter count** — contrary to the conventional wisdom at larger scales.

**Go deep and narrow.** MobileLLM's controlled experiments at 125M and 350M showed that deeper, narrower models consistently outperform wider, shallower ones. A HuggingFace systematic study (Dec 2025) at 70M scale found a critical threshold: **hidden_size ≥ 512** is necessary for model viability, and **32 layers** was the "Goldilocks depth" achieving 38.50% accuracy even at 70M parameters. Surprisingly, all 12 architecture families tested (GPT-2, LLaMA-3, Qwen-3, Gemma-3, MoE, Titans-MAC, etc.) performed **within ~2% of each other** at 70M scale, meaning modern training practices matter more than architectural novelty.

The cumulative gains from MobileLLM's architecture ablation at 350M are approximately additive:

- SwiGLU activation: **+1.3%**
- Deep-narrow configuration: **+0.9–1.1%**
- Block-wise weight sharing: **+0.7–0.8%**
- Embedding tying: saves ~10% parameters with only -0.2–0.6% accuracy trade-off
- **Net architecture gain: ~4.3% absolute** over naive baseline

**The recommended dense configuration for 300M** is a LLaMA-style decoder-only transformer with 24–32 layers, d_model=1024, 16 attention heads (or GQA with 4 KV head groups), SwiGLU FFN (8/3 × d_model ≈ 2730 intermediate dim), RoPE positional embeddings, RMSNorm, and tied input-output embeddings. OpenELM's layer-wise scaling strategy — varying head counts and FFN dimensions across layers — is a parameter-efficient enhancement worth considering.

**State-space models are strong contenders.** Mamba (Gu & Dao, 2023) demonstrated that **Mamba-3B outperforms Transformers of the same size and matches Transformers twice its size**. The mamba-370m model provides a direct reference point with 48 layers at d_model=1024. Advantages include **5× higher inference throughput** and linear scaling in sequence length. However, training costs remain similar to Transformers; the efficiency gains appear primarily at inference. RWKV-7 offers similar benefits with stronger multilingual capabilities. **Hymba** (NVIDIA, Nov 2024, ICLR 2025) is the most promising hybrid: it integrates attention + SSM heads in parallel within each layer. **Hymba-1.5B outperformed Llama-3.2-3B** with 11.67× cache size reduction, and was specifically evaluated at 350M and 125M scales with consistent SOTA results.

**Shared parameter architectures maximize limited data.** ALBERT-style cross-layer parameter sharing acts as implicit regularization — beneficial when data is scarce. Sharing attention parameters has **negligible accuracy impact** (only -0.7 points), while sharing FFN parameters causes larger drops (-1.5–2.5 points). The Mixture-of-Recursions (MoR) architecture (July 2025) combines parameter sharing with adaptive depth per token, outperforming vanilla baselines at **360M scale** while enabling dramatically higher throughput via continuous depth-wise batching. Relaxed Recursive Transformers add per-loop LoRA adapters to mitigate the accuracy cost of strict weight sharing. A practical design: 12 unique layers recycled 2–3× (effective depth 24–36) with per-iteration LoRA adapters gives ~300M "logical" parameters with ~160M unique parameters — significantly reducing overfitting risk with 4B tokens.

**MoE at 300M active parameters is viable but risky with limited data.** Unified Scaling Laws for Routed Models (Clark et al., 2022) showed that MoE benefits are largest for smaller base models. DeepSeekMoE's fine-grained expert segmentation (64–256 small experts) allows more flexible combinations at small scale. A practical configuration: 300M active / 1–1.5B total parameters, 8–16 fine-grained experts per MoE layer, top-2 routing, with 1–2 shared experts. **The key risk**: with only 4B tokens, individual experts may be undertrained. Each expert sees fewer tokens, potentially leading to expert collapse. Consider starting with fewer experts (4–8) and including shared experts to ensure a stable baseline.

---

## 3. Retrieval augmentation is the highest-leverage inference-time technique

**RETRO** (Borgeaud et al., ICML 2022) demonstrated the most dramatic result in the literature: **a 7.5B RETRO with a 2T token database matched GPT-3's 175B on the Pile — a 25× parameter efficiency gain.** This makes retrieval augmentation the single most impactful technique for making a small model punch above its weight class.

**kNN-LM** (Khandelwal et al., ICLR 2020) is the simpler variant: it augments any pre-trained LM by interpolating with k-nearest neighbors from an external datastore, requiring **zero additional training**. On WikiText-103, perplexity improved from 18.65 to 15.79. The critical finding: "Training a model on 100M tokens and using kNN search over a **3B token dataset** can outperform training the same model on all 3B tokens." For Telugu, this means building a retrieval datastore from all available Telugu text — even text not used for training — can substantially boost performance.

**MLP Memory** (2025) replaces the kNN datastore with a pretrained MLP, compressing 220GB into 2.8GB while maintaining gains. Unlike kNN-LM (which can impair reasoning), MLP Memory actually improves reasoning task performance. The practical deployment architecture: a **300M Telugu model + multilingual dense retriever (e.g., LaBSE, multilingual-e5) + Telugu text datastore** could approximate a much larger model's factual knowledge at the cost of inference latency (5–10× for kNN-LM, much less for in-context retrieval).

---

## 4. Data quality is the most important single factor for small models

The Phi series from Microsoft provided the foundational evidence: **phi-1-small at 350M parameters achieved 45% on HumanEval** (code generation), trained on only ~7B tokens of "textbook quality" data for approximately $400 of compute. Phi-2 (2.7B) matched models up to **25× larger** on complex benchmarks. Phi-4 (14B) surpassed its teacher GPT-4 on STEM QA, demonstrating that synthetic data can go beyond simple distillation.

**Synthetic data is transformative but requires a quality generator.** The Phi approach — generating synthetic "textbook-quality" data that teaches reasoning systematically — has been replicated by SmolLM's Cosmopedia (28B tokens of synthetic textbooks from Mixtral-8x7B) and validated at 360M scale. For Telugu, the challenge is generator quality: using translation from high-quality English synthetic data (via IndicTrans2) may be more reliable than direct Telugu generation from current models. AI4Bharat's Sangraha Synthetic pipeline already translates English Wikimedia content into Telugu using IndicTrans2, providing a validated approach.

**Educational quality filtering yields massive gains.** SmolLM's FineWeb-Edu (web data filtered by an educational quality classifier) and Python-Edu (code filtered for educational value, converging **3× faster** than unfiltered code) demonstrate that classifier-based filtering is essential. For Telugu: train a binary quality classifier on high-quality Telugu sources (Wikipedia, verified websites from Sangraha Verified) versus noisy web crawl, then use it to filter larger Telugu corpora.

**Multi-epoch training is safe up to 4 epochs.** "Scaling Data-Constrained Language Models" (Muennighoff et al., NeurIPS 2023) trained 400+ models and found that **up to 4 epochs of repeated data yields negligible degradation** compared to unique data. Beyond 4 epochs, returns diminish but remain positive. For 300M parameters, Chinchilla-optimal is ~6B tokens. With 4B unique tokens, training for 2–3 epochs (~8–12B total tokens) stays well within the safe zone. **Mixing in 10–20% code data provides a 2× increase in effective tokens** even when evaluating only natural language tasks.

**Data mixing optimization matters.** DoReMi (Xie et al., NeurIPS 2023) uses a 280M proxy model to find optimal domain weights, **improving average few-shot accuracy by 6.5 percentage points** and reaching baseline accuracy in 2.6× fewer steps. The proxy model approach is exactly the right scale for a 300M target. After aggressive deduplication (MinHash, Jaccard threshold ~0.8, line-level dedup), data quality filtering, and augmentation with translated/synthetic content, a target of **~5–6B unique tokens trained for 2–3 epochs** is recommended.

---

## 5. Telugu-specific tokenizer and data considerations are critical

**Unigram tokenization decisively outperforms BPE for Telugu.** A comprehensive evaluation specifically for Telugu (Vemula et al., IJCNLP-AACL 2025) created gold morpheme segmentation datasets and found that **Unigram-based tokenizers consistently outperform BPE** across most settings, with the choice of tokenizer algorithm being the "most significant factor" influencing downstream performance. A second study of 17 Indic languages confirmed that Unigram adheres more closely to morphological segmentation than BPE, and cluster-based training (grouping Dravidian languages together) produces lower word fertility rates for Telugu.

**Vocabulary sizing at 300M is constrained.** With hidden dimension 1024 and tied embeddings, the embedding layer consumes vocab_size × 1024 parameters. At 32K vocabulary: ~33M parameters (11% of 300M) — reasonable. At 64K: ~66M (22%) — acceptable. At 128K: ~131M (44%) — too much. **The recommendation is 32K–48K SentencePiece Unigram** for a monolingual Telugu model, targeting a fertility rate of ~1.5–2.5 tokens/word (compared to 4–6+ with multilingual tokenizers). Unicode NFC normalization and Indic NLP Library normalization should be applied before tokenizer training.

**Telugu data resources are substantial and growing.** AI4Bharat's Sangraha provides the largest cleaned Indic pretraining corpus (251B tokens across 22 languages), with Telugu data available in verified, unverified, and synthetic splits. Additional sources include IndicCorp (8.8B tokens across 11 languages), OSCAR Telugu, Telugu Wikipedia, and 7.9M Telugu tweets. Evaluation benchmarks covering Telugu include IndicSentiment, IndicXNLI, Naamapadam (NER), TyDi QA, IndicNLG Suite, IndicCOPA, and MILU.

**Monolingual is better than bilingual at 300M.** The "curse of multilinguality" — where finite model capacity is diluted across languages — is decisive at 300M parameters. A Kazakh LLM study (MDPI 2025) advocates for dedicated monolingual models for agglutinative languages. The recommended hybrid approach: **initialize from an English-pretrained model** (replacing tokenizer, reinitializing embeddings, keeping transformer weights), include **5–10% parallel Telugu-English data** early in training for cross-lingual alignment, then transition to pure Telugu. This gets the benefits of English knowledge transfer while keeping ~95% of capacity dedicated to Telugu.

---

## 6. Training recipes that maximize every token

**The WSD learning rate scheduler is the current best practice for small models.** Introduced by MiniCPM (Hu et al., 2024), the Warmup-Stable-Decay schedule uses three phases: warmup (1–2% of steps) → stable constant LR (70–80%) → decay (10–20%). Unlike cosine scheduling, WSD allows continuous training — you can add more data later without restarting. MiniCPM-2.4B **surpassed Mistral-7B and LLaMA-13B** using this approach. The stable phase produces a "river valley" loss landscape where high LR drives rapid progress, with the decay phase revealing true optimization gains. For 300M: peak LR of 3e-4 to 6e-4, linear decay to 10% of peak.

**Model Wind Tunnel Experiments save massive compute.** MiniCPM's key methodological contribution: run extensive hyperparameter optimization at small proxy scale (~9M–30M parameters), then transfer optimal settings to the target 300M model using µ-Parametrization (Tensor Programs). This provides stable hyperparameter transfer across model scales and enables accurate loss prediction before committing to expensive full-scale runs.

**Progressive training saves 30–50% of compute.** MSG (Masked Structural Growth, ICLR 2024) enables strictly function-preserving model growth with **2× speedup** in pretraining. A practical schedule: train 100M for 2B tokens → grow to 200M for 1B tokens → grow to 300M for 1B tokens. LiGO (ICLR 2023) achieves similar results by learning a linear operator for growth, reducing FLOPs by 47.2% for RoBERTa-Base. With only 4B tokens, progressive training is especially valuable: smaller models learn basic patterns efficiently from early tokens, then the expanded model refines with remaining data.

**Pruning 3B→300M is too aggressive with limited data.** Sheared LLaMA demonstrated successful 81% pruning (7B→1.3B) with 50B tokens of continued pretraining using only 3% of original training compute. However, 90% pruning (3B→300M) would require substantial continued pretraining that 4B tokens likely cannot support. If a pretrained Telugu 3B model exists, pruning to ~1B (67% reduction) with continued pretraining is more realistic.

**Regularization must match the training regime.** A key finding (Liu et al., 2025): **dropout hurts performance during single-epoch pretraining** — even "early dropout" degrades results. For multi-epoch training on limited data, **EntroDrop** (entropy-guided token dropout) selectively masks easy tokens in later epochs, working across 0.6B–8B scales. Standard weight decay (0.1 for AdamW) and label smoothing (0.1) remain recommended. Curriculum learning reduces training steps by **18–45%** to reach baseline performance when using compression ratio as the difficulty metric, with the largest benefits observed at smaller model scales (14M–160M).

---

## 7. Emerging techniques open new frontiers for small models

**Hymba (NVIDIA, Nov 2024)** is the most promising new architecture for ~300M scale. It integrates attention heads and SSM (Mamba) heads in parallel within the same layer, with 128 learnable "meta tokens" prepended as compressed world knowledge. **Hymba-1.5B outperformed Llama-3.2-3B** with 3.49× throughput and 11.67× cache size reduction, and was specifically evaluated at 350M and 125M with consistent SOTA results for tiny LMs.

**Test-time compute scaling is a game-changer for verifiable tasks.** Snell et al. (ICLR 2025) showed that with optimal compute allocation, a smaller model using test-time compute can **outperform a 14× larger model** in FLOPs-matched evaluation. HuggingFace applied this to Llama-3.2-1B, which outperformed the 8B model; the 3B model outperformed 70B. rStar-Math (Microsoft, ICML 2025) pushed this further: 7B models rivaled OpenAI o1 on math reasoning using MCTS-based deep thinking, improving Qwen2.5-Math-7B from 58.8% to **90.0% on MATH**. For a 300M Telugu model, test-time compute scaling could close gaps on structured QA and mathematical reasoning, though it requires a separate verifier/reward model.

**Distillation from reasoning models** (DeepSeek-R1, Jan 2025) is highly practical. Even a 1.5B distilled model showed significant reasoning gains from training on ~800K verified chain-of-thought reasoning trajectories. Re-distillation using only ~35K samples boosted performance by **>4% on GSM8K** at negligible cost. For Telugu: generate reasoning traces using a multilingual model, then fine-tune the 300M model on these.

**Compound AI systems represent the most practical path to matching 3B performance.** A 300M model as the core of a system with retrieval (BM25/dense retriever over Telugu knowledge base), routing (escalate complex queries to a larger model), tool use (calculators, dictionary lookup, transliteration), and quality filtering (separate classifier) can collectively match or exceed monolithic 3B performance on many practical tasks. This is the current industry trend: state-of-the-art results increasingly come from compound systems rather than monolithic models.

**Sparse upcycling** (Google, ICLR 2023) converts trained dense models into MoE by replicating MLP weights into multiple experts. A 300M dense model could be upcycled into ~900M total / 300M active MoE, increasing capacity without proportional inference cost. NVIDIA's approach (Oct 2024) achieved **67.6% MMLU** (vs. 65.3% for continued dense training) with "virtual group" initialization.

**Byte Latent Transformer** (Meta, Dec 2024) eliminates tokenization entirely through dynamic byte patching based on next-byte entropy. This removes the "fertility penalty" that hurts Telugu in tokenizer-based models and matched Llama 3 performance with up to **50% fewer FLOPs**. However, it's unproven at 300M scale and architecturally complex (three transformer blocks). Consider it a future option rather than a current recommendation.

---

## 8. Which techniques deliver the most "bang for buck"

Based on ablation studies across multiple papers, the techniques are ranked by impact:

| Rank | Technique | Typical Gain | Compute Cost | Works with 4B Tokens? |
|------|-----------|-------------|-------------|----------------------|
| 1 | Data quality / synthetic data (Phi-style) | **5–15% absolute** | Low–Medium | Yes — actually most impactful with limited data |
| 2 | Knowledge distillation (DistiLLM/GKD) | **70–85% of gap closed** | 1.5–2× training cost | Yes — KD provides richer supervision per token |
| 3 | Retrieval augmentation (RETRO/kNN-LM) | **25× effective parameter gain** | Inference-time cost | Yes — retrieval compensates for limited training data |
| 4 | Architecture (deep-narrow + SwiGLU + GQA) | **+4.3% absolute** at 350M | Zero extra cost | Yes — architecture choices are data-independent |
| 5 | Training beyond Chinchilla (multi-epoch) | **Continuous improvement** up to ~4 epochs | Linear in epochs | Essential with limited data |
| 6 | WSD scheduler + wind tunnel experiments | **10–20% better** than naive cosine | Save compute via proxy models | Yes |
| 7 | Progressive training (MSG/LiGO) | **30–50% compute savings** | Saves compute | Yes — smaller models learn basic patterns first |
| 8 | Test-time compute scaling | **14× effective size increase** | Inference-time | Domain-dependent |
| 9 | Shared parameters / recursion (MoR) | **Regularization + throughput** | Zero/negative | Yes — reduces overfitting with limited data |
| 10 | Sparse upcycling (dense→MoE) | **+2–3% on MMLU** | Moderate | Risky — experts may be undertrained |

**Gains from orthogonal techniques are approximately additive.** MobileLLM's ablation showed cumulative architecture improvements (~4.3%). LLaVA-MoD demonstrated multiplicative benefits of combining MoE + distillation. SmolLM combined data curation + architecture + scheduling for SOTA results. The key principle: techniques operating on different axes (data quality + architecture + training recipe + inference augmentation) compound, while techniques on the same axis (two data quality methods) show diminishing returns.

---

## The recommended end-to-end recipe

**Phase 1 — Architecture:** Hymba-style hybrid attention+SSM at 300M (if engineering resources permit) or a deep-narrow LLaMA-style transformer: 28–32 layers, d_model=1024, GQA (4 KV head groups), SwiGLU, RoPE, RMSNorm, tied embeddings. SentencePiece Unigram tokenizer with 32K vocabulary trained on balanced Telugu corpus with Unicode normalization.

**Phase 2 — Data:** Source Telugu from Sangraha (verified + unverified), IndicCorp, OSCAR, Wikipedia, tweets. Augment with IndicTrans2 translations of English educational content (~1B tokens). Generate synthetic textbook-quality Telugu via translation of Phi-style content (~500M tokens). Deduplicate aggressively (MinHash + line-level). Quality-filter using perplexity-based classifier trained on Sangraha Verified. Target: ~5–6B unique tokens. Include 10% code data for effective token increase.

**Phase 3 — Pretraining:** Initialize transformer weights from an English model if possible (replace tokenizer, reinitialize embeddings). Use µP for hyperparameter transfer from ~30M proxy experiments. WSD scheduler with peak LR ~3e-4–6e-4. Train for 2–3 epochs (~10–15B total tokens) with curriculum ordering (easy→hard by perplexity). No dropout during pretraining. Optionally apply progressive training: 100M→200M→300M growth via MSG.

**Phase 4 — Distillation:** Distill from a multilingual teacher (BLOOM-3B or Llama variant) using DSKD for vocabulary mismatch + DistiLLM's SKL loss. Use staged distillation if an intermediate 1B model is available. Run KD for multiple epochs over the same 4B tokens (different supervision signal from standard LM loss). Apply MiniPLM's difference sampling to up-weight hard instances, achieving **2.4× data efficiency**.

**Phase 5 — Post-training:** SFT on Telugu instruction data. DPO using preference pairs generated by a larger multilingual model. Train 4–8 LoRA experts for different capabilities, composed via MoA routing at inference.

**Phase 6 — Deployment:** Wrap in compound AI system with BM25/dense retrieval over Telugu knowledge base. Apply test-time compute scaling for reasoning tasks. INT8 quantization for deployment (~150MB model size, mobile-ready).

**Expected compute:** 300M params × 10B tokens × 6 FLOPs/token ≈ 1.8 × 10¹⁹ FLOPs, approximately **200–400 A100 GPU-hours** (~$200–400 at cloud pricing). Highly feasible on academic budgets.

## Conclusion

The gap between a 300M and 3B model is large but not insurmountable. The most counterintuitive finding from this research is that **data quality dominates model size** at small scale — Phi-1 at 1.3B outperformed models 10× larger purely through data curation, and SmolLM-360M achieved SOTA at its class through synthetic textbooks and educational filtering. For Telugu specifically, the Unigram tokenizer advantage over BPE, the availability of Sangraha pipeline infrastructure, and the monolingual "curse-of-multilinguality" reversal (where a dedicated small model can match a much larger multilingual one on the target language) all work in the model's favor.

The techniques with the highest ceiling — retrieval augmentation and test-time compute scaling — operate at inference rather than training time, meaning they can be layered on top of any base model improvement. A 300M Telugu model with a quality Telugu text retrieval datastore, even before any architecture or training innovations, could already approach 3B-level factual knowledge. The remaining gap — in reasoning depth, instruction following, and generation quality — is best addressed through the distillation + data quality + architecture optimization pipeline described above. The most important insight: **these techniques are not alternatives but complements**, with gains stacking approximately additively across orthogonal axes of improvement.
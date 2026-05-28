# Methods and Transformer Innovations to Make a 300M Telugu LM Behave Like a 3B Model Without Distillation

## Executive summary

A 300M-parameter monolingual Telugu model can **plausibly match—or exceed—a 3B-parameter Telugu model** on many Telugu-centric evaluations **if both are constrained to ~4B Telugu tokens**, because 3B parameters at 4B tokens is *extremely under-trained* relative to compute-optimal scaling, while 300M at 4B tokens is much closer to the compute-optimal regime. Compute-optimal studies (Kaplan scaling laws; Chinchilla) quantify that **training compute scales roughly as \(C \approx 6 N\) FLOPs per token** and that optimal allocation requires scaling tokens with model size; Chinchilla reports token budgets on the compute-optimal frontier that correspond to **~20 tokens/parameter** (e.g., 400M params → ~8B tokens; 1B params → ~20B tokens). citeturn26view2turn17view1turn16view0

To close the remaining behavior/performance gap without distillation, the highest-leverage strategy is to **stack multiple “small wins” that are now standard in strong open LLMs** (normalization, activations, positional encoding, KV-sharing, and fast attention kernels), then use **data efficiency and objective choices** that improve sample efficiency per token, plus **parameter-efficient adaptation** for post-training (instruction following, domain alignment) without retraining the full model for each capability.

A recommended “dense-core” recipe, tailored to **~300M params and ~4B tokens**, is:

- **Architecture (dense, LLaMA-like core)**: Pre-norm **RMSNorm**, **SwiGLU** FFN, **RoPE** (or ALiBi if you prioritize extrapolation to longer context), and **Grouped-Query Attention (GQA)** to reduce KV-cache/memory and enable longer contexts or larger batches at the same hardware. citeturn1search9turn0search2turn2search1turn3search1  
- **Systems efficiency**: Use **FlashAttention** (exact attention, IO-aware) and/or **xFormers memory-efficient attention** so you can afford longer contexts and/or higher batch tokens with the same hardware budget. citeturn0search3turn12search0turn14search3  
- **Optimization**: **AdamW** (decoupled weight decay) with cosine decay + warmup; use mixed precision (bf16/fp16 with loss scaling) and activation checkpointing to increase effective depth/sequence length. Chinchilla explicitly reports AdamW improvements vs Adam in their ablations. citeturn8search0turn8search1turn11search3turn18view0  
- **Data efficiency for Telugu (~4B tokens)**: aggressive **deduplication** (reduces memorization and can improve efficiency/accuracy), Telugu-aware normalization, and augmentation via **translation/transliteration pairs** (MuRIL-style) and parallel corpora like **Samanantar** for synthetic Telugu (carefully filtered). citeturn28search3turn4search2turn5search2  
- **Evaluation**: Use Telugu-inclusive benchmarks and proxies (IndicGLUE/IndicXTREME, TyDi QA Telugu, UD Telugu treebank, WikiAnn NER) plus intrinsic perplexity and code-mixed robustness slices. citeturn4search0turn5search1turn37view0turn32search2turn32search5  

Finally, if “behavior/performance” includes **knowledge-heavy interactive QA**, consider **retrieval augmentation (RETRO / RAG-style)** as a non-distillation lever: RETRO reports GPT-3–comparable performance on the Pile with **25× fewer parameters** by retrieving from a large text database. This changes the system (model+index) but is one of the few proven ways to substitute parameters with external memory. citeturn29search0  

## Problem framing and assumptions

**Goal interpretation**: “Match the behavior and performance of a 3B-parameter model” is not a single number; it depends on:
1) **Intrinsic modeling quality** (Telugu perplexity / cross-entropy).  
2) **Downstream task performance** (classification, QA, NER, etc.).  
3) **Interactive “assistant behavior”** (instruction adherence, safety/harmlessness style, helpfulness), which is dominated by post-training data and alignment methods.

**Key scaling-law constraint**: If the 3B model and 300M model both only see ~4B Telugu tokens, the 3B model is *data-starved* by compute-optimal standards. Chinchilla shows that many large models were “significantly undertrained” and that compute-optimal training requires increasing training tokens substantially with model size. citeturn16view0 Chinchilla’s table of compute-optimal token counts implies ~20 tokens/parameter on the frontier (e.g., 400M → ~8B; 1B → ~20B). citeturn17view1 Under that lens:
- 300M params at 4B tokens ≈ **13.3 tokens/param** (closer to 20).  
- 3B params at 4B tokens ≈ **1.33 tokens/param** (very far from 20).  

So, **under a fixed Telugu data budget**, a well-trained 300M can be surprisingly competitive.

**Compute model used for estimates**: Kaplan et al. provide a widely used approximation of transformer training compute: after accounting for forward/backward, **non-embedding compute is \(C \approx 6N\) floating-point operations per training token**, where \(N\) is the number of non-embedding parameters. citeturn26view2turn26view3  

**Assumptions (explicit)**:
- You can train dense transformers end-to-end (no distillation).  
- You can afford standard LLM infra (bf16/fp16, checkpointing, distributed training if needed).  
- Telugu data is ~4B tokens *after* cleaning/dedup; you may add synthetic/augmented text but will track it separately.

## Candidate techniques and evidence

The tables below focus on techniques with the strongest evidence that they improve **parameter efficiency** (quality per parameter or quality per token/compute) and are implementable in modern training stacks.

### Architectural innovations for parameter efficiency

| Technique | What it changes | Why it helps at fixed ~300M params | Empirical evidence (from primary sources) | Trade-offs / limitations | Implementation notes |
|---|---|---|---|---|---|
| **RMSNorm** | Replaces LayerNorm with RMS-based normalization | Cheaper normalization; widely adopted in efficient LLM stacks; can improve training/inference efficiency | RMSNorm proposes dropping mean-centering and argues LayerNorm’s re-centering invariance is dispensable; it is computationally simpler and more efficient. citeturn1search9 | May require retuning eps and LR slightly when swapping from LayerNorm | Use Pre-Norm blocks; typical eps ~1e-6 to 1e-5 (tune). |
| **SwiGLU / GLU variants** | Replaces GELU MLP with gated MLP | Often improves quality per parameter by increasing expressivity of the FFN at similar cost | “GLU Variants Improve Transformer” (SwiGLU family) is a primary reference motivating gated FFNs. citeturn0search2 | Slightly more compute in FFN; needs careful FFN dimension choice to keep params in budget | In LLaMA-style designs, set FFN dim near \(\frac{2}{3}\cdot 4d\) to hold params constant vs standard FFN. |
| **RoPE** | Rotary positional embedding applied to Q/K | Works well for decoder-only LMs; encodes relative position structure; strong default in modern stacks | RoFormer (RoPE) describes the method and its benefits (relative dependency properties, length flexibility). citeturn2search1turn2search13 | Long-context extrapolation may need RoPE scaling variants (separate research line) | Implement by rotating Q/K in each layer; verify numerical stability in bf16/fp16. |
| **ALiBi** | Adds linear distance biases to attention logits instead of positional embeddings | Better length extrapolation; can train shorter sequences and test longer → effectively reallocates compute | ALiBi shows a 1.3B model trained at length 1024 extrapolates to 2048 with similar perplexity to a sinusoidal model trained at 2048, while training ~11% faster and using ~11% less memory. citeturn2search2turn12search1 | Strong recency bias may hurt tasks needing precise absolute positions; less common than RoPE in current LLMs | Easy to implement; head-specific slopes; official repo provides reference implementation. citeturn12search1 |
| **GQA / MQA (KV sharing)** | Shares K/V across head groups (GQA) or all heads (MQA) | Reduces KV-cache size and memory bandwidth → enables longer context or larger batches; can indirectly raise quality by allowing more training tokens per wall-clock | MQA proposes sharing keys/values to reduce decoding bottlenecks with minor quality degradation. citeturn3search0 GQA generalizes MQA and reports quality close to multi-head while retaining speed benefits. citeturn3search1turn3search5 | Too few KV heads can reduce quality; best benefit is memory/speed, not direct perplexity | For 300M, GQA is a safe middle ground: choose e.g. 16 Q heads with 4 KV heads. |
| **Reversible layers** | Recompute activations instead of storing them | Memory savings can be traded for depth, sequence length, or batch tokens | Reformer combines reversible residual layers with LSH attention; reversible layers allow storing activations only once instead of per-layer. citeturn3search2turn3search10 | Recompute overhead; reversibility constraints; more complex debugging | Consider using reversible blocks only if memory is your binding constraint. |
| **Sparse / long-context attention** | Replaces full attention with sparse patterns or approximations | Allows longer contexts at similar compute/memory; could improve downstream QA/summarization | BigBird reduces quadratic attention to linear, preserving strong theoretical properties and improves long-context tasks. citeturn14search4turn14search0 Longformer is a drop-in attention replacement enabling long documents. citeturn14search5turn14search1 Performer approximates softmax attention with linear complexity. citeturn13search7turn14search2 | Many sparse/linear attentions can underperform full attention on standard LM perplexity; more hyperparameter surface | For a 300M Telugu LM, start with **full attention + FlashAttention** unless long-doc tasks are primary. citeturn12search0 |
| **MoE / conditional computation** | Adds experts; routes tokens to a subset | Best-known way to increase quality at fixed FLOPs (but usually increases total params) | Switch Transformer reports up to ~7× pretraining speed increases with same compute in certain settings and enables very large sparse models. citeturn12search3 | Typically violates a strict “total params = 300M” cap; training complexity and stability | Only consider if your constraint is *active params / FLOPs*, not total params. |

**Takeaway**: For a strict 300M total-parameter cap, the highest ROI architectural bundle is the **modern dense stack**: **Pre-Norm RMSNorm + RoPE + SwiGLU + GQA + FlashAttention**. These choices primarily improve *training stability, throughput, and usable context/batch*, which lets you turn a fixed data budget into stronger effective learning. citeturn1search9turn2search1turn0search2turn3search1turn12search0  

### Training and optimization techniques

| Technique | Why it helps parameter efficiency | Empirical evidence | Trade-offs | Implementation notes |
|---|---|---|---|---|
| **AdamW (decoupled weight decay)** | Better generalization than Adam+L2; common default for transformers | AdamW introduces decoupled weight decay and shows improved behavior for adaptive optimizers. citeturn8search0 Chinchilla trained with AdamW (vs Adam in Gopher) and reports AdamW-trained models outperform Adam across schedules in their ablations. citeturn18view0 | Needs tuning of weight decay and LR together | Typical LLM practice: no weight decay for norms/bias; decay for matrices. |
| **Cosine LR decay + warmup** | Stable training and good final loss; widely used in scaling-law experiments | Kaplan reports using a 3000-step linear warmup followed by cosine decay to zero as the default schedule in their scaling-law runs. citeturn26view2turn26view3 Chinchilla emphasizes matching schedule length to training tokens for best loss. citeturn16view0 | Short-training regimes require careful warmup fraction | For 4B tokens, pick warmup as % of total steps (e.g., 1–5%), not a fixed step count. |
| **Mixed precision + loss scaling** | Nearly 2× memory reduction; often higher throughput | Mixed Precision Training describes fp16 training with loss scaling and maintaining fp32 master weights, reducing memory consumption. citeturn8search1 | Numerical stability pitfalls (overflow/underflow) | Prefer bf16 when available; otherwise fp16 + dynamic loss scaling. |
| **Activation checkpointing** | Trades compute for memory; enables longer contexts/deeper models at same hardware | “Training Deep Nets with Sublinear Memory Cost” introduces sublinear-memory checkpointing. citeturn11search3 DeepSpeed documents practical activation checkpointing utilities. citeturn11search11 | Extra forward recompute cost | Especially valuable if you want 4k–8k context at 300M. |
| **ZeRO optimizer sharding** | Cuts optimizer-state memory overhead; scales training efficiently | ZeRO “eliminates memory redundancies” and enables training far larger models with better memory efficiency. citeturn8search2 | Distributed complexity; comms overhead | Use ZeRO Stage-1/2 even on moderate clusters for stability and throughput. |
| **μP / μTransfer** | Hyperparameters transfer across sizes; reduces tuning cost; can improve final quality by enabling better HPs | Tensor Programs V shows μP stabilizes optimal HPs across scales and enables zero-shot HP transfer to large models. citeturn8search3turn8search11 | Requires μP-compatible parameterization and setup | Very useful if your organization already tuned a smaller Telugu prototype and wants to scale to 300M reliably. |

**Takeaway**: For a 300M Telugu LM, most “wins” come from **getting to a clean, stable, high-throughput pretraining loop** and then using the saved wall-clock/memory to (a) increase context length modestly, (b) increase effective batch tokens, and (c) run more ablations on data/tokenizer quality without blowing budgets. citeturn8search1turn11search3turn12search0  

### Parameter-efficiency methods (not distillation)

This category splits into (A) methods that **reduce parameters wasted on embeddings/layers** so you can reallocate capacity, and (B) methods that **adapt behavior cheaply** after pretraining.

**Core methods to consider**:
- **Weight tying (input embedding = output embedding)**: reduces parameters and can improve perplexity. Press & Wolf recommend tying and show perplexity gains plus large parameter savings in translation models. citeturn7search3turn7search7  
- **Factorized embeddings / parameter sharing (ALBERT-style)**: ALBERT introduces factorized embeddings and cross-layer parameter sharing to reduce memory and improve scaling behavior. citeturn11search1  
  - Practical note: full cross-layer sharing often degrades quality for generative decoder-only LMs; “partial sharing” (share only some submodules or share in cycles) is a compromise explored in later parameter-sharing work. citeturn11search0  
- **Adapters (Houlsby)**: add small per-task modules; near full fine-tuning quality for many tasks at a fraction of trainable parameters. citeturn9search1turn9search5  
- **LoRA (low-rank adaptation)**: injects trainable low-rank matrices into attention/MLP projections; drastically reduces trainable parameters and can match full fine-tuning on multiple models. citeturn9search0  
- **IA³**: rescales activations with learned vectors; designed for parameter-efficient fine-tuning and shown competitive in few-shot settings. citeturn10search0  
- **Prefix-tuning**: optimizes continuous “virtual tokens” to steer the model while keeping base weights frozen. citeturn9search3turn9search7  

**Why this matters for “behavior matching”**: if your target “3B behavior” includes instruction following and domain norms, you rarely want to full-finetune a 300M base repeatedly. PEFT lets you:
- maintain one strong Telugu base,
- attach (LoRA/adapters) for domain/instruction styles,
- iterate on behavior quickly using limited supervised Telugu data (or translated instruction corpora), without distillation.

## Data efficiency and tokenization for low-resource Telugu (~4B tokens)

### Tokenizer choices and vocabulary sizing

**SentencePiece (Unigram or BPE)** is the most deployable default for Telugu training because it trains directly from raw text and supports Unigram LM and BPE segmentation. citeturn6search0turn6search16  
Classic BPE subword methods were introduced for open-vocabulary handling in NMT and remain a strong baseline. citeturn6search1turn6search17  

**Key Telugu-specific considerations**:
- Telugu is an agglutinating, suffixing Dravidian language with rich morphology; TyDi QA’s typological discussion explicitly highlights Telugu’s orthography and productive transitive/causative formation, implying a high surface-form variety that stresses subword choices. citeturn37view0  
- Vocab size trade-off: larger vocab reduces sequence length (more characters per token) but increases embedding/softmax cost; at 300M, **~32k SentencePiece Unigram** is a practical starting point, with ablations at 16k and 64k.

**Tokenizer decision table (practical)**

| Option | Best for | Why it may help at 4B tokens | Risks |
|---|---|---|---|
| SentencePiece **Unigram**, 32k | General Telugu LM | Robust segmentation; good compression; easy training from raw | Needs careful normalization; may under-segment rare morphology |
| SentencePiece **BPE**, 32k–64k | Slightly more deterministic merges | Often strong for NMT and LMs; can reduce fertility | Merge rules can overfit frequent scripts/domains |
| **Byte/char models** (ByT5-style) | Noisy text, mixed scripts, transliteration | Token-free modeling is robust to noise and avoids vocab dilution; parameter-matched ByT5 is competitive with subword mT5 and better on noise-sensitive tasks. citeturn6search2turn6search10 | Sequence lengths explode → compute increases; decoder-only byte LMs are harder to make efficient |

### Data quality, deduplication, and augmentation

**Deduplicate aggressively**. Lee et al. show near-duplicates are common; deduplication reduces memorized output frequency and can improve accuracy/efficiency, and they release code for dataset deduplication. citeturn28search3  
For a 4B-token Telugu corpus, near-duplication (news syndication, scraped reposts) can be substantial; dedup is a direct way to convert “wasted tokens” into “learning tokens.”

**Augment with translation + transliteration pairs (MuRIL-style)**. MuRIL explicitly trains on Indian-language corpora and augments monolingual text with translated and transliterated document pairs as supervised signals; it reports strong gains over mBERT on cross-lingual XTREME and improved handling of transliterated data. citeturn4search2  
Even if your final model is monolingual Telugu, this suggests a productive strategy: **generate parallel Telugu variants** (native script + romanized + lightly normalized) and train objectives that force consistency.

**Use parallel corpora to create synthetic Telugu**, with filters:
- **Samanantar** provides a large English–Indic parallel resource (tens of millions of sentence pairs) and was built for high-quality MT training. citeturn5search2  
You can translate high-quality English instruction/data into Telugu using strong MT models trained on Samanantar (or other public MT systems) and then filter for fluency and script correctness.

**Multilingual transfer as an initializer (optional)**: Multilingual pretraining can help low-resource languages but suffers from “capacity dilution.” XLM-R shows strong cross-lingual transfer and explicitly analyzes trade-offs (positive transfer vs dilution) at scale. citeturn27search0 For Indian languages specifically, MuRIL argues 100+ language multilingual LMs underrepresent Indian languages in vocab/data and underperform in resource-lean settings, motivating Indic-focused pretraining. citeturn4search2  
Pragmatically: if you can initialize from an Indic-focused model family (or train a small multilingual Indic decoder and then continue on Telugu), you may get better generalization at the same Telugu token budget.

### Pretraining objectives beyond pure next-token prediction

Because Telugu tokens are limited, consider objectives with better **token-level learning signal density**:

- **Pure autoregressive (decoder-only)** remains the best fit if you want a chatty assistant and strong free-form generation.
- **Span corruption / text-infilling (T5)**: T5 systematically studies objectives and shows strong transfer via text-to-text; span corruption can be more sample-efficient for some tasks. citeturn6search3turn6search7  
- **Mixture-of-denoisers (UL2)**: UL2 proposes mixing objectives (different denoising modes) and reports strong performance across tasks, emphasizing that objective choice can shift the Pareto frontier. citeturn7search1  
- **ELECTRA-style replaced-token detection** (mostly for encoders): ELECTRA argues it is more compute/sample-efficient than MLM because it learns from all tokens, and reports especially strong gains for small models. citeturn7search0turn7search8  
  - Telugu implication: if your priority is Telugu NLU representations, an ELECTRA-like objective can help. If you need generative behavior, keep a decoder-only base and optionally add an encoder/bi-objective later.

## Evaluation metrics and benchmarks for Telugu and proxies

A robust evaluation suite should mix **intrinsic LM metrics**, Telugu-centric downstream tasks, and proxy tasks that correlate with real assistant behavior.

**Intrinsic**
- **Perplexity / cross-entropy** on a strictly held-out Telugu set (post-dedup). Scaling-law work uses cross-entropy as the core metric and shows predictable power-law behavior. citeturn7search2turn26view2  
- **Memorization / leakage checks** via train–test overlap audits; dedup reduces train–test overlap and memorized generations. citeturn28search3  

**Telugu-inclusive downstream benchmarks**
- **IndicGLUE (AI4Bharat)**: a benchmark suite for Indian-language NLU tasks, designed to cover multiple Indian languages across tasks. citeturn4search0turn4search3  
- **IndicXTREME / IndicCorpV2 ecosystem**: the “Leaving No Indic Language Behind” work introduces IndicCorp (20.9B tokens, 24 languages) and IndicXTREME (9 tasks, 105 eval sets, 20 languages), and trains IndicBERT v2 with measurable improvements. citeturn5search1turn4search4  
- **TyDi QA (Telugu)**: TyDi QA is explicitly multilingual and includes Telugu; the TyDi QA paper’s typological section includes Telugu and motivates evaluation on genuine information-seeking QA. citeturn36view0turn37view0  
- **UD Telugu (MTG)**: a small, manually annotated dependency treebank for Telugu; provides syntax-focused evaluation/finetuning targets. citeturn32search2turn32search6  
- **WikiAnn (PAN-X) NER**: multilingual NER supporting many languages; provides a standardized NER proxy for Telugu. citeturn32search5  

**Behavior / assistant proxies**
- **Instruction-following eval**: create a Telugu instruction set (human + translated), evaluate with exact-match for constrained tasks and LLM-as-judge only if you can control judge bias; track refusal/helpfulness policies separately.  
- **Robustness slices**: native Telugu script vs romanized Telugu; spelling noise; code-mixed Telugu-English (MuRIL highlights transliteration/code-mixing as important). citeturn4search2  

## Recommended combined model design and training plan

### Recommended 300M model design (dense, Telugu-optimized)

A concrete, parameter-budget-respecting target that aligns with best-practice dense LLM ingredients:

- **Decoder-only transformer**, ~**24 layers**, **d_model ≈ 1024**, **n_heads = 16**, **GQA with n_kv = 4**, **SwiGLU FFN**, Pre-Norm **RMSNorm**. (This configuration is close to ~300M parameters when paired with a ~32k vocab and tied embeddings; exact count depends on vocab and FFN rounding.)  
- **Positional encoding**: **RoPE** by default; consider **ALiBi** if long-context extrapolation is a primary requirement. citeturn2search1turn2search2  
- **Attention kernel**: **FlashAttention** for training speed and memory efficiency. citeturn0search3turn12search0  
- **Weight tying**: tie input embeddings and output head. citeturn7search3  

Mermaid sketch (block-level):

```mermaid
flowchart TB
  A[Telugu text] --> B[SentencePiece Unigram ~32k]
  B --> C[Token embedding]
  C --> D[Transformer block × 24]

  subgraph Block
    D1[RMSNorm (pre-norm)] --> D2[GQA self-attn\n(Q heads=16, KV heads=4)\nRoPE on Q/K\nFlashAttention kernel]
    D2 --> D3[Residual add]
    D3 --> D4[RMSNorm]
    D4 --> D5[SwiGLU MLP\n(ffn_dim ≈ 2/3·4d)]
    D5 --> D6[Residual add]
  end

  D --> E[Final RMSNorm]
  E --> F[LM head (tied to embedding)]
```

### Training pipeline (data → pretrain → post-train), with key knobs

```mermaid
flowchart LR
  A[Raw Telugu corpora\n~4B tokens] --> B[Cleaning & normalization\n(langID, Unicode normalize)]
  B --> C[Dedup\n(doc+near-dup)]
  C --> D[Train tokenizer\nSP Unigram 16k/32k/64k ablation]
  D --> E[Pretraining\nCausal LM (primary)\n+ optional UL2-style denoising mix]
  E --> F[Base eval\nPPL + IndicGLUE/IndicXTREME\nTyDiQA-te + NER + UD]
  F --> G[Post-training\nSFT (Telugu instructions)\n+ PEFT (LoRA/adapters)]
  G --> H[Behavior eval\nInstruction adherence\nRobustness slices]
```

### Practical hyperparameter ranges to try (starting grid)

These are meant as *starting ranges*; μP can reduce tuning cost if you already have smaller pilots. citeturn8search3turn8search11

- **Tokenizer**: SentencePiece Unigram {16k, 32k, 64k}. citeturn6search0  
- **Context length**: 2048 (baseline) → 4096 (if memory allows; benefit depends on Telugu task mix).  
- **Optimizer**: AdamW. citeturn8search0turn18view0  
  - betas: (0.9, 0.95) or (0.9, 0.98); eps 1e-8  
  - weight decay: 0.05–0.2 (exclude norms/bias)  
  - grad clip: 1.0  
- **LR schedule**: warmup + cosine decay. Kaplan used 3000-step warmup + cosine decay (in their setting); for 4B tokens you should scale warmup to a % of total steps. citeturn26view2turn26view3  
  - peak LR (for 300M): 1e-4 to 5e-4 (tune by stability + validation loss)  
  - warmup fraction: 1–5% of steps  
  - cosine to near-zero or to ~10% of peak (both are used in practice; choose by held-out loss)  
- **Batching**: choose global batch tokens so you have enough steps for smooth optimization. Kaplan-scale runs used fixed batch sizes (example: 512×1024 tokens ≈ 0.5M tokens) in their setup; but at 4B tokens this yields only ~8k steps. citeturn26view2  
  - suggested global batch tokens: 64k–512k tokens (gives ~62.5k to ~7.8k steps for 4B tokens)  
- **Precision & memory**: bf16/fp16 mixed precision; activation checkpointing if needed. citeturn8search1turn11search3turn11search11  
- **Data mixing**:
  - reserve ~1–3% “high-quality Telugu” slice (Wikipedia, curated books, high-quality news) and upweight it late in training (curriculum).  
  - include a controlled % of transliterated/code-mixed Telugu to improve robustness (MuRIL motivation). citeturn4search2  

### Compute and data budget estimates

Using Kaplan’s approximation **\(C \approx 6N\)** FLOPs per token and token count \(T\): total training compute is approximately \(6NT\). citeturn26view2turn26view3  

- **300M params, 4B tokens**:  
  \(C \approx 6 \cdot 3\times 10^8 \cdot 4\times 10^9 \approx 7.2\times 10^{18}\) FLOPs (order-of-magnitude).  
- **3B params, 4B tokens**:  
  \(C \approx 7.2\times 10^{19}\) FLOPs (≈10× more compute than 300M at same tokens).  

**Data-optimality sanity check**: Chinchilla’s compute-optimal frontier implies ~20 tokens/parameter (e.g., 400M→8B; 1B→20B). citeturn17view1  
So 300M “wants” ~6B tokens for compute-optimality; you have 4B, which is moderately data-limited but not catastrophically so. A 3B model would “want” ~60B tokens—far beyond your 4B—making it heavily data-limited and therefore a realistic target for a strong 300M model to match on many metrics.

### Expected performance vs compute (interpretive chart)

Below is an interpretive plot of the **Chinchilla-like compute-optimal line** (tokens ≈ 20× parameters) versus your operating points:

```mermaid
xychart-beta
  title "Tokens vs Parameters (compute-optimal frontier ~20 tokens/param)"
  x-axis "Parameters (B)" [0.3, 3.0]
  y-axis "Tokens (B)" 0 --> 80
  line "Compute-optimal ~20×" [6, 60]
  point "Your 300M @ 4B" 0.3 4
  point "3B @ 4B" 3.0 4
```

Interpretation: the 3B@4B point is far below the compute-optimal line (severely data-starved), while 300M@4B is closer—supporting the feasibility of “matching” under these constraints. citeturn17view1turn16view0  

### Optional “system-level” lever for knowledge-heavy behavior

If the 3B baseline’s advantage is mostly **factual recall and QA**, you can substitute parameters with retrieval:
- **RETRO** retrieves from a huge database and reports comparable performance to GPT-3/Jurassic-1 on the Pile with **25× fewer parameters**, translating to downstream knowledge-intensive gains after fine-tuning. citeturn29search0  
This is not distillation; it is *explicit memory*. Trade-off: you must build and serve a Telugu retrieval index (e.g., Telugu Wikipedia + curated corpora) and evaluate as a system.

## Prioritized list of papers and primary sources with official repos

The list below is grouped by implementable idea; priority is based on (a) evidence strength, (b) applicability to a 300M Telugu LM, and (c) implementation maturity.

**Scaling / compute planning**
- Kaplan et al., *Scaling Laws for Neural Language Models* (compute model \(C \approx 6N\) per token; weak dependence on depth/width in wide ranges). citeturn25view0turn26view3  
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla; tokens/params scaling; modern “undertrained” diagnosis). citeturn16view0turn17view1  

**Core dense architecture stack**
- Zhang & Sennrich, *RMSNorm* (NeurIPS) citeturn1search9  
- Su et al., *RoFormer / RoPE* + official RoFormer repo citeturn2search1turn2search13  
- Shazeer, *GLU Variants Improve Transformer* (SwiGLU) citeturn0search2  
- Shazeer, *Multi-Query Attention* citeturn3search0  
- Ainslie et al., *GQA* citeturn3search1turn3search5  

**Efficient attention kernels / throughput**
- Dao et al., *FlashAttention* + official implementation repo citeturn0search3turn12search0  
- xFormers memory-efficient attention repo citeturn14search3  

**Long context (optional)**
- Press et al., *ALiBi* + official repo citeturn2search2turn12search1  
- Zaheer et al., *BigBird* + official repo citeturn14search4turn14search0  
- Beltagy et al., *Longformer* + official repo citeturn14search5turn14search1  
- Choromanski et al., *Performer* + Google Research code citeturn13search7turn14search2  

**Deep / memory-efficient training (optional)**
- Kitaev et al., *Reformer* (reversible layers) citeturn3search2  
- Chen et al., *Training Deep Nets with Sublinear Memory Cost* (activation checkpointing) citeturn11search3  
- Rajbhandari et al., *ZeRO* citeturn8search2  
- Wang et al., *DeepNet / DeepNorm* (if exploring much deeper-than-standard models) citeturn3search3  

**Optimization and tuning transfer**
- Loshchilov & Hutter, *AdamW* + repo citeturn8search0turn8search12  
- Micikevicius et al., *Mixed Precision Training* citeturn8search1  
- Yang et al., *Tensor Programs V / μTransfer* + official μP repo citeturn8search3turn8search11  

**Tokenization**
- Kudo & Richardson, *SentencePiece* + official repo citeturn6search0turn6search16  
- Sennrich et al., BPE subwords for rare words citeturn6search1  
- Xue et al., *ByT5* + official repo (for robustness/noise ablations) citeturn6search2turn6search10  

**Data quality**
- Lee et al., *Deduplicating Training Data Makes Language Models Better* + code citeturn28search3  

**Objectives / sample efficiency**
- Raffel et al., *T5* (span corruption; systematic objective study) citeturn6search3turn6search7  
- Tay et al., *UL2* (mixture of denoisers objective) citeturn7search1  
- Clark et al., *ELECTRA* + official repo (especially if building Telugu encoders) citeturn7search0turn7search8  

**Telugu / Indic corpora and evaluation**
- AI4Bharat, IndicGLUE + IndicNLPSuite paper citeturn4search0turn4search3  
- Doddapaneni et al., *Leaving No Indic Language Behind* (IndicCorp 20.9B tokens; IndicXTREME benchmark) citeturn5search1  
- Khanuja et al., *MuRIL* (Indic-focused multilingual LM; translation/transliteration augmentation) citeturn4search2  
- Ramesh et al., *Samanantar* (parallel corpora for synthetic Telugu) citeturn5search2  
- Clark et al., *TyDi QA* (includes Telugu; typological notes on Telugu) citeturn36view0turn37view0  
- Universal Dependencies: UD Telugu MTG treebank + repo citeturn32search2turn32search6  
- WikiAnn NER dataset card citeturn32search5  

**Retrieval augmentation (system-level, non-distillation)**
- Borgeaud et al., *RETRO* (25× fewer parameters with retrieval; comparable to GPT-3/Jurassic-1 on Pile) citeturn29search0turn29search12  
- Guu et al., *REALM* citeturn29search1  
- Lewis et al., *RAG* citeturn29search2turn29search6  


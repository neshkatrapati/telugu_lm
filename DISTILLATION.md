# Telugu LLaMA — Distillation Plan

## Goal

Compress the 300M teacher model into a smaller student model that retains as much knowledge as possible, for faster inference and lighter deployment.

---

## Teacher Model (Reference)

| Property | Value |
|----------|-------|
| Architecture | LLaMA-style (RoPE + SwiGLU + RMSNorm) |
| Parameters | ~300M |
| Layers | 16 |
| Heads | 16 |
| Embedding dim | 1024 |
| MLP hidden dim | 2816 |
| Context length | 2048 |
| Vocab size | ~51.8K |
| Val loss | ~2.58 |
| Perplexity | ~13 |

---

## Knowledge Distillation — Mechanism

### Core Idea

Train a smaller student model to mimic the teacher's full output probability distribution, not just the hard ground-truth labels. The teacher's soft probabilities carry "dark knowledge" — information about which wrong answers are more plausible, relationships between morphemes, and structural patterns that the small model couldn't discover on its own from raw data.

### Distillation Loss

```
Loss = α * CE(student_logits, ground_truth)
     + (1 - α) * KL_div(student_softmax(logits/T), teacher_softmax(logits/T)) * T²
```

- **α = 0.5** — balance between hard labels (ground truth) and soft labels (teacher distribution)
- **T = 2-4** — temperature that softens probability distributions. Higher T spreads probability mass across more tokens, giving the student richer training signal
- **KL divergence** forces the student to assign similar probabilities to similar morphemes
- **T² scaling** compensates for the gradient magnitude reduction from temperature scaling

### Training Procedure

1. Freeze the teacher (300M trained model, no gradients)
2. For each training batch:
   - Run teacher forward pass (no grad) → get teacher logits
   - Run student forward pass → get student logits
   - Compute combined loss (hard CE + soft KL)
   - Backprop through student only
3. Same training data, same 3 epochs
4. Student learning rate slightly higher: 5e-4 (vs 3e-4 for teacher) — smaller models tolerate higher LR
5. Same cosine decay schedule with warmup

---

## Approach 1: Direct Distillation

### Student Architecture Options

All options use the same LLaMA-style architecture (RoPE + SwiGLU + RMSNorm) and the same tokenizer.

#### Option A: 50M Student

| Property | Teacher (300M) | Student (50M) |
|----------|---------------|---------------|
| Layers | 16 | 8 |
| Heads | 16 | 8 |
| Embedding dim | 1024 | 512 |
| Head dim | 64 | 64 |
| MLP hidden dim | 2816 | 1408 |
| Context length | 2048 | 2048 |
| Vocab size | ~51.8K | ~51.8K |
| Compression | 1x | 6x |

- Val loss estimate: 3.0-3.3
- Quality retention: ~75-80%
- Training time on B200: ~3-4 hours

#### Option B: 80M Student

| Property | Teacher (300M) | Student (80M) |
|----------|---------------|---------------|
| Layers | 16 | 10 |
| Heads | 16 | 10 |
| Embedding dim | 1024 | 640 |
| Head dim | 64 | 64 |
| MLP hidden dim | 2816 | 1792 |
| Context length | 2048 | 2048 |
| Vocab size | ~51.8K | ~51.8K |
| Compression | 1x | 3.75x |

- Val loss estimate: 2.8-3.0
- Quality retention: ~85-88%
- Training time on B200: ~4-5 hours

#### Option C: 120M Student (Recommended for direct distillation)

| Property | Teacher (300M) | Student (120M) |
|----------|---------------|----------------|
| Layers | 16 | 12 |
| Heads | 16 | 12 |
| Embedding dim | 1024 | 768 |
| Head dim | 64 | 64 |
| MLP hidden dim | 2816 | 2048 |
| Context length | 2048 | 2048 |
| Vocab size | ~51.8K | ~51.8K |
| Compression | 1x | 2.5x |

- Val loss estimate: 2.7-2.8
- Quality retention: ~90-93%
- Training time on B200: ~5-6 hours

### Quality Comparison — Direct Distillation

| Model | Params | Val Loss (est.) | Perplexity (est.) | Quality |
|-------|--------|----------------|-------------------|---------|
| Teacher 300M | 300M | 2.58 | ~13 | 100% |
| Distilled 120M | 120M | 2.7-2.8 | ~15-16 | ~90-93% |
| Distilled 80M | 80M | 2.8-3.0 | ~16-20 | ~85-88% |
| Distilled 50M | 50M | 3.0-3.3 | ~20-27 | ~75-80% |
| 50M from scratch (no distillation) | 50M | 3.5-3.8 | ~33-45 | ~60-65% |

Note: Distillation saves ~0.3-0.5 val loss over training the same small model from scratch.

### What Students Retain vs Lose

**Retained well**:
- Telugu morphological patterns (word formation, suffixes, postpositions)
- Common sentence structures
- Frequent vocabulary usage
- Basic grammatical patterns

**Degraded**:
- Rare word combinations — less capacity to memorise the long tail
- Longer-range coherence — fewer layers = shallower reasoning
- Diversity in generation — more repetitive outputs
- Non-Telugu handling — English/numbers degrade more
- Nuanced word choices in low-frequency contexts

---

## Approach 2: Progressive Distillation

### Idea

Instead of one large compression step, distill in stages where each student becomes the teacher for the next:

```
300M → 150M → 75M
```

Each step only compresses 2x. Each intermediate student is a better teacher for the next stage than jumping directly from 300M, because its internal representations are closer in scale to the next student.

### Stages

| Stage | Teacher | Student | Compression |
|-------|---------|---------|-------------|
| 1 | 300M | 150M | 2x |
| 2 | 150M | 75M | 2x |
| Total | 300M | 75M | 4x |

### Expected Quality

| Model | Params | Val Loss (est.) | Quality |
|-------|--------|----------------|---------|
| Direct 300M → 75M | 75M | 2.9-3.1 | ~82-85% |
| Progressive 300M → 150M → 75M | 75M | 2.8-2.9 | ~87-90% |

Progressive distillation gains ~0.1-0.2 val loss over direct at the cost of 2x training time.

### Training Time on B200

| Stage | Time (est.) |
|-------|-------------|
| 300M → 150M (3 epochs) | ~5-6 hours |
| 150M → 75M (3 epochs) | ~4-5 hours |
| **Total** | **~10-12 hours** |

---

## Approach 3: Distillation + LoRA Replay (Recommended for maximum quality at small size)

### Idea

The 50M base model handles common patterns (80-90% of text). A tiny LoRA adapter (~2-5M params) is trained to recover the hard/rare knowledge that was lost during compression. After training, the LoRA is permanently merged into the base weights — zero runtime overhead.

### Step 1: Standard Distillation → 50M Base

Distill 300M → 50M using standard distillation (see Approach 1, Option A). This produces a model that handles frequent patterns well but struggles with rare morpheme combinations, complex dependencies, and low-frequency vocabulary.

### Step 2: Identify Hard Examples

Run both teacher and student on the full training data. Collect examples where the student significantly underperforms the teacher:

```python
for each batch in training_data:
    teacher_loss = CE(teacher(batch), ground_truth)
    student_loss = CE(student(batch), ground_truth)
    difficulty = student_loss - teacher_loss

    if difficulty > threshold:
        hard_set.append(batch)
```

Typically 15-25% of the data falls into the hard set. These are the examples where the student lost knowledge during distillation — rare words, complex structures, unusual morpheme patterns.

### Step 3: Train LoRA Adapter on Hard Set

Freeze the 50M base model entirely. Attach LoRA (Low-Rank Adaptation) matrices to the key layers:

```
Base model: 50M (frozen)
LoRA rank: 16-32
LoRA targets: Q, K, V, output projection, gate, up, down (all linear layers)
LoRA params: ~2-5M
Training data: hard set only (15-25% of full data)
Loss: KL divergence from teacher on hard examples
Learning rate: 1e-4
Epochs: 3-5 on the hard set
```

LoRA adds two small matrices A (d × r) and B (r × d) to each targeted weight matrix W:
```
W_effective = W_base + α * B @ A
```
Where r (rank) = 16-32 is much smaller than d (512), so the total parameter overhead is tiny.

The LoRA learns specifically what the base model is missing — it doesn't waste capacity on patterns the base already knows.

### Step 4: Merge and Deploy

Permanently merge LoRA weights into the base model:

```python
for each layer:
    W_merged = W_base + α * B @ A
```

This produces a single 55M model with:
- No runtime overhead (no adapter switching)
- Same inference speed as the 50M base
- Significantly better quality on rare/hard patterns

### Why This Beats Direct 55M Distillation

| Approach | What happens |
|----------|-------------|
| Direct 55M distillation | All 55M params try to learn everything — capacity spread thin across both common and rare patterns |
| 50M base + 5M LoRA | 50M params specialise on common patterns (80% of data), 5M params specialise on the hard tail (20% of data) |

The LoRA approach gives targeted capacity allocation. The base doesn't waste parameters on rare cases, and the adapter doesn't waste parameters on things the base already knows. This division of labour results in better overall quality per parameter.

### Expected Quality

| Model | Params | Val Loss (est.) | Quality |
|-------|--------|----------------|---------|
| Teacher 300M | 300M | 2.58 | 100% |
| Direct distill 50M | 50M | 3.0-3.3 | ~75-80% |
| **50M + LoRA merged** | **55M** | **2.8-2.9** | **~87-90%** |
| Direct distill 120M | 120M | 2.7-2.8 | ~90-93% |

The 55M with LoRA gets close to 120M quality at half the size.

### Training Timeline on B200

| Stage | Time (est.) |
|-------|-------------|
| Distill 300M → 50M (3 epochs) | ~3-4 hours |
| Hard set identification (inference on full data through both models) | ~2-3 hours |
| LoRA training on hard set (3-5 epochs) | ~1-2 hours |
| Merge + evaluate | minutes |
| **Total** | **~7-10 hours** |

### Hyperparameter Sensitivity

- **Difficulty threshold**: Controls hard set size. Too low = LoRA trains on easy examples it doesn't need. Too high = misses important hard examples. Start with median difficulty, tune based on hard set size (target 15-25%).
- **LoRA rank**: 16 is sufficient for 50M model. 32 adds more capacity but diminishing returns. Going below 8 loses too much expressiveness.
- **α (LoRA scaling)**: Start at 1.0, tune based on val loss. Higher α = stronger LoRA influence.

---

## All Approaches Summary

| Approach | Final Size | Val Loss | Quality | Training Time | Best For |
|----------|-----------|----------|---------|--------------|----------|
| Direct 120M | 120M | 2.7-2.8 | ~90-93% | ~5-6 hrs | Simplest path to high quality |
| Direct 50M | 50M | 3.0-3.3 | ~75-80% | ~3-4 hrs | Maximum compression, quality not critical |
| Progressive 300→150→75M | 75M | 2.8-2.9 | ~87-90% | ~10-12 hrs | Balanced, no LoRA complexity |
| **50M + LoRA replay** | **55M** | **2.8-2.9** | **~87-90%** | **~7-10 hrs** | **Best quality-per-parameter** |

---

## Inference Speed After Distillation

### Raw PyTorch (CPU, no optimisations)

| Model | Tokens/sec |
|-------|------------|
| Teacher 300M | 5-10 |
| Student 120M | 15-25 |
| Student 50-55M | 30-60 |

### With Phase 1 Optimisations (KV Cache + INT8 + torch.compile)

| Model | Tokens/sec |
|-------|------------|
| Teacher 300M | 60-120 |
| Student 120M | 150-250 |
| Student 50-55M | 300-500 |

### With Phase 2 (GGUF + llama.cpp Q4)

| Model | Tokens/sec | Model Size on Disk |
|-------|------------|-------------------|
| Teacher 300M | 150-300 | ~150MB |
| Student 120M | 300-500 | ~60MB |
| Student 50-55M | 500-1000+ | ~30MB |

---

## Recommendation

For maximum quality retention at the smallest size:

**→ Approach 3: Distillation + LoRA Replay (50M + 5M LoRA → 55M merged)**

Then apply inference optimisations from INFERENCE_OPTIMISATION.md:
- Phase 1 for development: 300-500 tok/s on CPU
- Phase 2 (GGUF) for deployment: 500-1000+ tok/s, ~30MB model file

This gives a model that:
- Is 5.5x smaller than the teacher
- Retains ~87-90% of quality
- Runs at 500+ tok/s on CPU with GGUF
- Ships as a ~30MB file

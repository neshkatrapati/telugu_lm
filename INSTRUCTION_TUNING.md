# Telugu LLaMA — Instruction Tuning Plan

## Goal

Turn the 300M pretrained base model into a basic Telugu chatbot that can follow instructions and respond in a conversational format.

---

## Current State: Base Model

The pretrained 300M model is a **next-token predictor**. It was trained on raw Telugu text (articles, web pages, documents). Given a prompt, it will continue the text as if writing more of the same — it does not understand questions, instructions, or conversational turn-taking.

Example (base model behaviour):
```
Input:  "తెలుగులో రాజధాని ఏమిటి?"
Output: "అనే ప్రశ్న చాలా మందికి ఉంటుంది. భారతదేశంలో అనేక రాష్ట్రాలు..."
        (continues as if writing an article about the question, doesn't answer it)
```

---

## Step 1: Supervised Fine-Tuning (SFT)

### What SFT Does

Fine-tune the base model on instruction-response pairs so it learns the pattern: when a user asks something, generate a relevant answer.

### Chat Format

```
<bos>user: {instruction or question}
assistant: {response}<eos>
```

For multi-turn conversations:

```
<bos>user: {first message}
assistant: {first response}
user: {follow-up}
assistant: {follow-up response}<eos>
```

The model learns to:
- Recognise the `user:`/`assistant:` turn-taking pattern
- Generate responses after seeing `assistant:`
- Stop generating at `<eos>`

### Loss Masking

During SFT, the loss is computed **only on the assistant's response tokens**. The user's instruction tokens are fed in but not trained on. This prevents the model from learning to generate user-like text and focuses all capacity on generating good responses.

```
<bos>user: తెలుగులో రాజధాని ఏమిటి?
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  (masked — no loss)
assistant: ఆంధ్రప్రదేశ్ రాజధాని అమరావతి.<eos>
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  (loss computed here)
```

---

## Step 2: Instruction Data

This is the primary bottleneck. Need thousands of Telugu instruction-response pairs.

### Option 1: Translate Existing English Datasets

Take established English instruction datasets and translate to Telugu.

**Source datasets**:
- Stanford Alpaca (~52K instructions)
- Databricks Dolly (~15K instructions)
- OpenAssistant Conversations (~10K multi-turn)
- FLAN Collection (diverse tasks)

**Translation methods**:
- IndicTrans2 (AI4Bharat's translation model — free, good for Telugu)
- GPT-4 / Gemini API (higher quality, costs money)
- Google Translate API (cheapest, lowest quality)

**Pros**: Fast, large volume, covers diverse instruction types
**Cons**: Translation artifacts, unnatural Telugu phrasing, cultural mismatches (references to Western concepts)

**Recommendation**: Translate 10K Alpaca examples using IndicTrans2, then filter for quality. Expect ~7-8K usable pairs after filtering.

### Option 2: Existing Telugu Instruction Datasets

Search HuggingFace and other sources for pre-existing Telugu instruction data.

**Potential sources**:
- AI4Bharat datasets (may have instruction-formatted data)
- Dhenu Telugu instruction datasets on HuggingFace
- IndicInstruct / IndicNLP resources
- Community-created Telugu chat datasets
- Search: `telugu instruction`, `telugu chat`, `telugu sft`, `indic instruction`

**Pros**: Native Telugu, culturally appropriate
**Cons**: Limited quantity, variable quality, may need cleaning

### Option 3: Synthetic Generation

Use a strong multilingual LLM to generate Telugu Q&A pairs from scratch.

**Method**:
1. Define topic categories: history, geography, science, literature, daily life, grammar, culture, politics, technology, health, education, agriculture
2. For each topic, prompt GPT-4/Gemini:
   ```
   Generate 50 question-answer pairs in Telugu about {topic}.
   Questions should be natural and varied.
   Answers should be 1-3 sentences, factually accurate.
   Write entirely in Telugu script.
   ```
3. Review and filter outputs

**Pros**: High quality, controlled topics, natural Telugu
**Cons**: Costs money (API calls), may reflect LLM biases, needs verification

**Recommendation**: Generate 2-3K high-quality pairs across diverse topics. Budget: ~$20-50 in API costs.

### Option 4: Mixed (Recommended)

Combine all sources for maximum coverage:

| Source | Quantity (est.) | Quality |
|--------|----------------|---------|
| Translated Alpaca (via IndicTrans2) | 7-8K | Medium |
| Existing Telugu datasets (HuggingFace) | 2-5K | Variable |
| Synthetic generation (GPT-4/Gemini) | 2-3K | High |
| **Total** | **~12-15K pairs** | Mixed |

Post-processing:
- Deduplicate
- Filter out low-quality translations (too short, broken Telugu, English leakage)
- Ensure diverse instruction types: questions, commands, creative tasks, summarisation, translation
- Final usable set: ~10K pairs

---

## Step 3: SFT Training Configuration

### Hyperparameters

| Property | Value | Notes |
|----------|-------|-------|
| Data | 5K-15K instruction-response pairs | More is better, quality matters more than quantity |
| Epochs | 3-5 | Small dataset needs multiple passes |
| Learning rate | 2e-5 | 10x lower than pretraining — preserve pretrained knowledge |
| LR schedule | Cosine decay with warmup (100 steps) | |
| Batch size (B200) | 32-64 | Sequences are shorter than pretraining |
| Gradient accumulation | 1-2 | Small dataset, don't need large effective batch |
| Weight decay | 0.01 | Lower than pretraining (0.1) to reduce forgetting |
| Max sequence length | 1024 | Most instruction pairs are short |
| Precision | BF16 | |
| Loss | CE on assistant tokens only (masked) | |
| Training time (B200) | ~30-60 minutes | Small dataset = fast |

### Two Approaches to SFT

#### Approach A: Full Fine-Tune

Update all 300M parameters. Simple, maximum expressiveness.

**Risk**: Catastrophic forgetting — the model may lose some pretrained Telugu knowledge, especially with a small dataset and too many epochs.

**Mitigation**: Low learning rate (2e-5), early stopping on val loss, keep pretraining checkpoint for comparison.

#### Approach B: LoRA Fine-Tune (Recommended)

Freeze the base model. Train only LoRA adapters on the instruction data.

| Property | Value |
|----------|-------|
| LoRA rank | 16-32 |
| LoRA targets | Q, K, V, output, gate, up, down |
| LoRA α | 32 |
| Trainable params | ~2-5M (< 2% of total) |

**Pros**:
- No catastrophic forgetting — base knowledge fully preserved
- Can store multiple adapter versions (different instruction styles)
- Merge into base when satisfied: `W = W_base + α * BA`
- Much faster training, less VRAM

**Cons**:
- Slightly less expressive than full fine-tune
- For a 300M model the difference is small

**Recommendation**: Start with LoRA. If quality is insufficient, try full fine-tune with careful learning rate.

---

## Step 4: Evaluation

### Automated Metrics

- **Perplexity on held-out instruction pairs**: Compare base model vs SFT model on instruction-formatted text. SFT model should have much lower perplexity on the assistant portions.
- **BLEU/ROUGE against reference responses**: Rough quality indicator, not highly reliable for open-ended generation.
- **Response format compliance**: % of outputs that follow the `assistant: ...` format and terminate with `<eos>`.

### Manual Evaluation

Create a test set of ~100 diverse Telugu questions across categories:

| Category | Example |
|----------|---------|
| Factual | తెలంగాణ రాజధాని ఏది? |
| Instruction | ఈ వాక్యాన్ని సరళంగా రాయండి: ... |
| Creative | వర్షం గురించి నాలుగు వాక్యాలు రాయండి |
| Conversational | నమస్కారం, మీరు ఎలా ఉన్నారు? |
| Summarisation | ఈ పేరా సారాంశం రాయండి: ... |
| Translation | "Good morning" తెలుగులో ఏమిటి? |

Rate each response on:
1. **Relevance** (0-3): Does it answer the question?
2. **Fluency** (0-3): Is the Telugu natural and grammatical?
3. **Completeness** (0-3): Is the answer sufficient?
4. **Factuality** (0-3): Are stated facts correct?

---

## Realistic Expectations

### What a 300M SFT Model Will Do Well

- Understand the turn-taking format (user asks, assistant answers)
- Give short, roughly relevant Telugu responses
- Handle simple factual questions seen in training data
- Follow basic instructions (translate, summarise, list)
- Generate grammatical Telugu in most cases
- Respond appropriately to greetings and simple conversation

### What It Will NOT Do

- Reason or think step-by-step
- Handle complex multi-turn conversations with context tracking
- Be reliably factually accurate (will hallucinate heavily)
- Refuse harmful or inappropriate requests (requires RLHF/DPO)
- Handle nuanced or ambiguous instructions
- Generate long, coherent responses (>100 words)
- Perform mathematical reasoning
- Understand code or structured data

### Honest Assessment

Think of it as a **pattern-matching Telugu responder**, not a real assistant. It will produce Telugu text in the right format, often on-topic, but it will hallucinate facts, repeat itself, and fail on anything requiring actual reasoning. For a 300M model this is expected — even English models at this scale have similar limitations.

---

## Beyond SFT: Future Alignment Steps

SFT is the minimum viable step. For a better chatbot, the following steps would be needed (in order):

### DPO (Direct Preference Optimisation)

Train the model to prefer good responses over bad ones using human preference data. Requires pairs of (chosen response, rejected response) for the same prompt.

- Reduces hallucination
- Improves response quality and helpfulness
- Needs ~5K preference pairs in Telugu
- Much harder to source than SFT data

### Safety Tuning

Fine-tune or apply guardrails so the model:
- Refuses harmful requests
- Doesn't generate offensive content
- Acknowledges when it doesn't know something
- Doesn't pretend to be human

For a 300M model, a simple keyword/pattern-based safety filter may be more practical than training-based alignment.

### RLHF (Reinforcement Learning from Human Feedback)

The gold standard for alignment. Requires:
1. A reward model (itself trained on preference data)
2. PPO or similar RL training
3. Large amounts of human feedback

Likely overkill for a 300M Telugu model. DPO is the practical alternative.

---

## Implementation Timeline

| Step | Task | Time (est.) |
|------|------|-------------|
| 1 | Source and prepare instruction data | 1-3 days |
| 2 | Format data (chat template, tokenize) | 2-3 hours |
| 3 | SFT training (LoRA) on B200 | 30-60 minutes |
| 4 | Manual evaluation on test set | 2-3 hours |
| 5 | Iterate (adjust data, retrain) | 1-2 days |
| **Total** | | **~3-5 days** |

The bottleneck is data, not compute.

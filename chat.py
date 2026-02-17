#!/usr/bin/env python3
"""
Telugu LLaMA — Interactive Chat CLI
=====================================
Chat with an SFT-tuned Pothana model using the same chat template
it was fine-tuned on.

Supports multi-turn conversation with sliding context window,
system instructions, and generation controls.

The SFT checkpoint includes special token IDs in its metadata.
If using a base (pretrained) checkpoint, pass --base to use
text-completion mode instead of chat mode.

Usage:
    # Chat with SFT model
    python chat.py --checkpoint ./sft_checkpoints/best.pt

    # With custom system instruction
    python chat.py --checkpoint ./sft_checkpoints/best.pt \\
        --system "మీరు తెలుగు సహాయకుడు. సంక్షిప్తంగా సమాధానం ఇవ్వండి."

    # Adjust generation
    python chat.py --checkpoint ./sft_checkpoints/best.pt \\
        --temperature 0.6 --top-k 40 --max-tokens 300

    # Single-shot (non-interactive)
    python chat.py --checkpoint ./sft_checkpoints/best.pt \\
        --prompt "తెలంగాణ రాజధాని ఏది?"

    # Base model (text completion, no chat template)
    python chat.py --checkpoint ./checkpoints/best.pt --base
"""

import sys
import re
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")

# Existing special tokens from base tokenizer
BOS_ID = 2
EOS_ID = 3

# Default system instruction (Telugu)
DEFAULT_SYSTEM = (
    "మీరు ఒక సహాయకరమైన తెలుగు AI అసిస్టెంట్. "
    "వినియోగదారుని ప్రశ్నలకు స్పష్టంగా మరియు సరిగ్గా సమాధానం ఇవ్వండి."
)


# ============================================================================
# Morfessor Segmentation
# ============================================================================
def load_morfessor_model(model_path: Path):
    """Load the Morfessor model for segmentation."""
    import morfessor
    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))
    return model


def segment_text(text: str, morf_model, separator: str = "@@") -> str:
    """Segment raw text using Morfessor with @@ continuation markers."""
    tokens = text.split()
    seg_tokens = []

    for token in tokens:
        if TELUGU_WORD_RE.fullmatch(token):
            segments = morf_model.viterbi_segment(token)[0]
            for i, seg in enumerate(segments):
                if i < len(segments) - 1:
                    seg_tokens.append(seg + separator)
                else:
                    seg_tokens.append(seg)
        elif TELUGU_WORD_RE.search(token):
            parts = re.split(r"([\u0C00-\u0C7F]+)", token)
            parts = [p for p in parts if p]
            for part_idx, part in enumerate(parts):
                is_last_part = (part_idx == len(parts) - 1)
                if TELUGU_WORD_RE.fullmatch(part):
                    segments = morf_model.viterbi_segment(part)[0]
                    for i, seg in enumerate(segments):
                        if i < len(segments) - 1:
                            seg_tokens.append(seg + separator)
                        else:
                            if not is_last_part:
                                seg_tokens.append(seg + separator)
                            else:
                                seg_tokens.append(seg)
                else:
                    if not is_last_part:
                        seg_tokens.append(part + separator)
                    else:
                        seg_tokens.append(part)
        else:
            seg_tokens.append(token)

    return " ".join(seg_tokens)


# ============================================================================
# Model Loading
# ============================================================================
def load_sft_model(checkpoint_path: Path, device: str):
    """Load SFT checkpoint — includes special token IDs in metadata."""
    sys.path.insert(0, str(Path(__file__).parent))
    from train_gpt import GPTConfig, build_model

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

    special_tokens = checkpoint.get("special_tokens", {})
    step = checkpoint.get("step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    training_type = checkpoint.get("training_type", "unknown")

    logger.info("Loaded checkpoint from %s", checkpoint_path)
    logger.info("  Type: %s | Step: %s | Val loss: %s", training_type, step, val_loss)
    logger.info("  Config: %d layers, %d heads, %d dim, vocab=%d",
                config.n_layer, config.n_head, config.n_embd, config.vocab_size)
    if special_tokens:
        logger.info("  Special tokens: %s", special_tokens)

    return model, config, special_tokens


# ============================================================================
# Generation
# ============================================================================
@torch.no_grad()
def generate(
    model,
    token_ids: list[int],
    max_new_tokens: int,
    stop_ids: set[int],
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    device: str = "cuda",
) -> list[int]:
    """Autoregressive generation with stop tokens, top-p, and repetition penalty."""
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    block_size = model.config.block_size
    generated = []

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, V)

        # Repetition penalty
        if repetition_penalty != 1.0 and len(generated) > 0:
            prev_tokens = set(generated[-64:])  # penalise recent tokens
            for tid in prev_tokens:
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Top-p (nucleus)
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = -float("Inf")
            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        next_id = idx_next.item()
        generated.append(next_id)

        if next_id in stop_ids:
            break

    return generated


# ============================================================================
# Chat Template
# ============================================================================
class ChatSession:
    """Manages multi-turn conversation with chat template formatting."""

    def __init__(
        self,
        tokenizer,
        morf_model,
        special_tokens: dict,
        system_instruction: str = DEFAULT_SYSTEM,
        context_window: int = 4,  # turns (not pairs) — matches SFT training
    ):
        self.tokenizer = tokenizer
        self.morf_model = morf_model
        self.special = special_tokens
        self.system_instruction = system_instruction
        self.context_window = context_window

        # Conversation history: [(user_text, assistant_text), ...]
        self.history: list[tuple[str, str]] = []

        # Reverse map for decoding
        self.id_to_special = {v: k for k, v in special_tokens.items()}

    def _encode_text(self, text: str) -> list[int]:
        """Morfessor segment → tokenize (no BOS/EOS)."""
        segmented = segment_text(text, self.morf_model)
        return self.tokenizer.encode(segmented, add_bos=False, add_eos=False)

    def build_prompt(self, user_input: str) -> list[int]:
        """Build the full token sequence for the current turn.

        Format (matches SFT training):
            <bos> <|system|> {sys} <|end|>
            <|user|> {ctx_u1} <|end|> <|assistant|> {ctx_a1} <|end|>
            ...
            <|user|> {current} <|end|> <|assistant|>

        The model generates after <|assistant|>.
        """
        ids = []

        # <bos>
        ids.append(BOS_ID)

        # System instruction
        ids.append(self.special["<|system|>"])
        ids.extend(self._encode_text(self.system_instruction))
        ids.append(self.special["<|end|>"])

        # Context turns from history (last N turns)
        context_pairs = self.context_window // 2  # 4 turns = 2 pairs
        start = max(0, len(self.history) - context_pairs)
        for user_text, asst_text in self.history[start:]:
            ids.append(self.special["<|user|>"])
            ids.extend(self._encode_text(user_text))
            ids.append(self.special["<|end|>"])
            ids.append(self.special["<|assistant|>"])
            ids.extend(self._encode_text(asst_text))
            ids.append(self.special["<|end|>"])

        # Current user input
        ids.append(self.special["<|user|>"])
        ids.extend(self._encode_text(user_input))
        ids.append(self.special["<|end|>"])

        # Generation starts here
        ids.append(self.special["<|assistant|>"])

        return ids

    def decode_response(self, token_ids: list[int]) -> str:
        """Decode generated token IDs back to text, stripping special tokens."""
        regular_ids = []
        for tid in token_ids:
            if tid in self.id_to_special:
                continue  # skip special tokens in output
            regular_ids.append(tid)
        if not regular_ids:
            return ""
        return self.tokenizer.decode(regular_ids)

    def add_turn(self, user_text: str, assistant_text: str):
        """Record a completed turn in conversation history."""
        self.history.append((user_text, assistant_text))

    def reset(self):
        """Clear conversation history."""
        self.history.clear()


# ============================================================================
# Decode helper for base model (no special tokens)
# ============================================================================
def decode_plain(token_ids: list[int], tokenizer) -> str:
    """Decode token IDs from base model (strip BOS/EOS only)."""
    filtered = [tid for tid in token_ids if tid not in (BOS_ID, EOS_ID)]
    return tokenizer.decode(filtered)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Telugu LLaMA — Interactive Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", "-c", type=Path, required=True,
        help="Path to model checkpoint (.pt) — SFT or base",
    )
    parser.add_argument(
        "--tokenizer", "-t", type=Path, default=Path("./tokenizer"),
        help="Tokenizer directory (default: ./tokenizer)",
    )
    parser.add_argument(
        "--morfessor", "-m", type=Path, default=Path("./morfessor_telugu.bin"),
        help="Morfessor model path (default: ./morfessor_telugu.bin)",
    )

    # Generation parameters
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    gen.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    gen.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    gen.add_argument("--top-p", type=float, default=0.0, help="Top-p / nucleus sampling (default: 0.0 = off)")
    gen.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")

    # Chat options
    chat = parser.add_argument_group("chat")
    chat.add_argument("--system", type=str, default=None, help="System instruction (default: built-in Telugu)")
    chat.add_argument("--context-window", type=int, default=4, help="Context turns for multi-turn (default: 4)")
    chat.add_argument("--prompt", "-p", type=str, default=None, help="Single prompt (non-interactive)")
    chat.add_argument("--base", action="store_true", help="Use base model (text completion, no chat template)")

    # Misc
    parser.add_argument("--verbose", "-v", action="store_true", help="Show token IDs and debug info")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ---- Load components ----
    logger.info("Loading Morfessor model...")
    morf_model = load_morfessor_model(args.morfessor)

    logger.info("Loading tokenizer...")
    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer
    tokenizer = MorfessorTokenizer(args.tokenizer)
    logger.info("  Vocab size: %d", tokenizer.vocab_size)

    logger.info("Loading model...")
    model, config, special_tokens = load_sft_model(args.checkpoint, device)

    if not args.no_compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ---- Determine mode ----
    is_chat = not args.base and bool(special_tokens)
    if args.base:
        logger.info("Mode: BASE (text completion)")
    elif special_tokens:
        logger.info("Mode: CHAT (SFT model with special tokens)")
    else:
        logger.info("Mode: BASE (no special tokens in checkpoint — falling back to completion)")
        is_chat = False

    # ---- Stop token IDs ----
    if is_chat:
        stop_ids = {EOS_ID}
        end_id = special_tokens.get("<|end|>")
        if end_id is not None:
            stop_ids.add(end_id)
    else:
        stop_ids = {EOS_ID}

    # ---- Chat session ----
    system_instruction = args.system or DEFAULT_SYSTEM
    session = None
    if is_chat:
        session = ChatSession(
            tokenizer=tokenizer,
            morf_model=morf_model,
            special_tokens=special_tokens,
            system_instruction=system_instruction,
            context_window=args.context_window,
        )

    # ---- Single-shot mode ----
    if args.prompt:
        if is_chat:
            prompt_ids = session.build_prompt(args.prompt)
            if args.verbose:
                logger.info("Prompt tokens: %d", len(prompt_ids))

            gen_ids = generate(
                model, prompt_ids, args.max_tokens, stop_ids,
                args.temperature, args.top_k, args.top_p,
                args.repetition_penalty, device,
            )
            response = session.decode_response(gen_ids)
        else:
            segmented = segment_text(args.prompt, morf_model)
            prompt_ids = tokenizer.encode(segmented, add_bos=True, add_eos=False)
            gen_ids = generate(
                model, prompt_ids, args.max_tokens, stop_ids,
                args.temperature, args.top_k, args.top_p,
                args.repetition_penalty, device,
            )
            response = decode_plain(gen_ids, tokenizer)

        print(f"\n{response}")
        return

    # ---- Interactive mode ----
    print()
    print("=" * 60)
    if is_chat:
        print("  Pothana 300M — Telugu Chat")
        print("  (SFT model, multi-turn conversation)")
    else:
        print("  Pothana 300M — Telugu Text Completion")
        print("  (base model, text continuation)")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /reset    — clear conversation history")
    print("  /system   — change system instruction")
    print("  /config   — show current generation settings")
    print("  /set K V  — change setting (e.g. /set temperature 0.5)")
    print("  /history  — show conversation history")
    print("  /quit     — exit")
    print()

    # Mutable generation params
    gen_params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ---- Commands ----
        if user_input.startswith("/"):
            cmd = user_input.split()
            cmd_name = cmd[0].lower()

            if cmd_name in ("/quit", "/exit", "/q"):
                print("Bye!")
                break

            elif cmd_name == "/reset":
                if session:
                    session.reset()
                print("[Conversation history cleared]")
                continue

            elif cmd_name == "/system":
                new_sys = " ".join(cmd[1:]).strip()
                if new_sys:
                    if session:
                        session.system_instruction = new_sys
                        session.reset()
                    print(f"[System instruction updated. History cleared.]")
                else:
                    current = session.system_instruction if session else "(N/A — base mode)"
                    print(f"[Current system: {current}]")
                    print("[Usage: /system <new instruction>]")
                continue

            elif cmd_name == "/config":
                print("[Generation config]")
                for k, v in gen_params.items():
                    print(f"  {k}: {v}")
                if session:
                    print(f"  context_window: {session.context_window}")
                    print(f"  history_turns: {len(session.history)}")
                continue

            elif cmd_name == "/set":
                if len(cmd) >= 3:
                    key = cmd[1]
                    if key in gen_params:
                        try:
                            val_type = type(gen_params[key])
                            gen_params[key] = val_type(cmd[2])
                            print(f"[{key} = {gen_params[key]}]")
                        except ValueError:
                            print(f"[Invalid value for {key}]")
                    else:
                        print(f"[Unknown setting: {key}]")
                        print(f"[Available: {', '.join(gen_params.keys())}]")
                else:
                    print("[Usage: /set <key> <value>]")
                continue

            elif cmd_name == "/history":
                if session and session.history:
                    for i, (u, a) in enumerate(session.history, 1):
                        print(f"  [{i}] You: {u}")
                        print(f"      Asst: {a}")
                else:
                    print("[No history]")
                continue

            else:
                print(f"[Unknown command: {cmd_name}]")
                continue

        # ---- Generate response ----
        if is_chat:
            prompt_ids = session.build_prompt(user_input)

            if args.verbose:
                logger.info("Prompt tokens: %d | History: %d turns",
                            len(prompt_ids), len(session.history))

            gen_ids = generate(
                model, prompt_ids, gen_params["max_tokens"], stop_ids,
                gen_params["temperature"], gen_params["top_k"],
                gen_params["top_p"], gen_params["repetition_penalty"],
                device,
            )
            response = session.decode_response(gen_ids)

            # Record turn
            session.add_turn(user_input, response)

        else:
            # Base model — text completion
            segmented = segment_text(user_input, morf_model)
            prompt_ids = tokenizer.encode(segmented, add_bos=True, add_eos=False)

            gen_ids = generate(
                model, prompt_ids, gen_params["max_tokens"], stop_ids,
                gen_params["temperature"], gen_params["top_k"],
                gen_params["top_p"], gen_params["repetition_penalty"],
                device,
            )
            response = decode_plain(gen_ids, tokenizer)

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()

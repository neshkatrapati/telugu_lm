"""
Pothana tokenizer — handles `@@` continuation prefix used by morfessor_bpe_telugu_v4.

The vocab contains both regular tokens (e.g., "మా", "ఇది") and continuation
tokens prefixed with `@@` (e.g., "@@కు", "@@లు"). A `@@` prefix means the
morpheme attaches to the previous token without a space:

    "మా @@కు" → "మాకు"
    "ఫలితాల @@ను" → "ఫలితాలను"

Inputs to encode are expected to be morfessor-segmented (whitespace-separated
morphemes). The included `morfessor_telugu.bin` model can produce this
segmentation from raw Telugu text; see README.
"""
from transformers import PreTrainedTokenizerFast


class PothanaTokenizer(PreTrainedTokenizerFast):
    """v4 tokenizer: bare morphemes + `@@` continuation prefix on suffixes."""

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        # Space-then-@@ means "join to previous, drop @@"
        text = text.replace(" @@", "")
        # Any leftover @@ (start of string, after special token, etc.) — strip
        text = text.replace("@@", "")
        return text

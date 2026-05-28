"""
Pothana — Telugu LLaMA-style decoder with QK-norm.

Thin extension of HuggingFace's LlamaForCausalLM that:
  - Adds QK-norm (RMSNorm on q and k per-head before RoPE) — Llama 3.1 / Cosmos style.
  - Untied lm_head (set via tie_word_embeddings=False in config).
  - Otherwise stock LLaMA: SwiGLU, RoPE, RMSNorm, GQA.

Loads cleanly via:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("neshmailsu/pothana-base-v2-225M",
                                                  trust_remote_code=True)
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack


class PothanaConfig(LlamaConfig):
    """Llama config + `use_qk_norm` flag."""
    model_type = "pothana"

    def __init__(self, use_qk_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_qk_norm = use_qk_norm


class PothanaAttention(LlamaAttention):
    """LlamaAttention + optional RMSNorm on Q and K per-head before RoPE."""

    def __init__(self, config: PothanaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if getattr(config, "use_qk_norm", False):
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ----- QK-norm (Pothana addition) -----
        # RMSNorm on the head_dim axis BEFORE RoPE.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PothanaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: PothanaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = PothanaAttention(config, layer_idx)


class PothanaModel(LlamaModel):
    config_class = PothanaConfig

    def __init__(self, config: PothanaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [PothanaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # Re-run post_init so the new layers get proper weight init
        self.post_init()


class PothanaForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM with PothanaModel (QK-norm in attention)."""
    config_class = PothanaConfig

    def __init__(self, config: PothanaConfig):
        super().__init__(config)
        self.model = PothanaModel(config)
        # tie_word_embeddings is read from config — set False at conversion time
        self.post_init()

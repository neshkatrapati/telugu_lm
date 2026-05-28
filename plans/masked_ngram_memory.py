"""
Masked N-Gram Memory Module
============================
Learnable attention-derived masked n-gram patterns with MDL-regularized
span embeddings and an EMA-updated cache.

Core idea:
  At each position, the model looks at a contiguous window of the preceding
  tokens. A small learned head predicts which positions in that window are
  "structurally important" (unmasked) vs. ignorable (masked). The resulting
  sparse pattern — a masked n-gram — is embedded and blended into the
  residual stream via a learned gate.

  An MDL penalty on the mask forces the model to retain tokens only when
  the information gain justifies the description cost, producing sparse,
  recurring patterns that cache well.

Plugs into CausalSelfAttention: after QK computation, before/alongside
the normal attention output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedNgramMemory(nn.Module):
    """
    Learnable masked n-gram memory with MDL regularisation.

    For each position p in the sequence, this module:
      1. Extracts QK logits over a local contiguous window of size `window_size`
      2. A pattern predictor head converts these to per-position mask probs
      3. Gumbel-sigmoid sampling produces a differentiable hard mask
      4. The masked token pattern is embedded via a small span encoder
      5. The span embedding is blended into the residual stream via a learned gate
      6. An MDL loss penalises retaining too many tokens (encourages sparse patterns)
      7. (Optional) An EMA cache stores span embeddings keyed by discretised pattern

    Parameters
    ----------
    n_embd : int
        Model hidden dimension (d_model).
    n_head : int
        Number of attention heads (used to infer head_dim for QK projection).
    window_size : int
        Length of the contiguous n-gram window (default: 7).
    mask_prior : float
        Prior probability of a position being *unmasked* (π in MDL loss).
        Lower = sparser patterns = more cache hits.  Default: 0.15 (~1 of 7).
    mdl_weight : float
        Weight λ for the MDL loss term.  Default: 0.01.
    cache_size : int
        Maximum entries in the EMA span-embedding cache.  0 = no cache.
    cache_ema_decay : float
        Exponential moving average decay for cached embeddings.
    gumbel_tau : float
        Temperature for Gumbel-sigmoid sampling.  Lower = harder masks.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        window_size: int = 7,
        mask_prior: float = 0.15,
        mdl_weight: float = 0.01,
        cache_size: int = 0,
        cache_ema_decay: float = 0.99,
        gumbel_tau: float = 0.5,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.window_size = window_size
        self.mask_prior = mask_prior
        self.mdl_weight = mdl_weight
        self.cache_size = cache_size
        self.cache_ema_decay = cache_ema_decay
        self.gumbel_tau = gumbel_tau

        # --- Pattern predictor ---
        # Takes the mean QK logits (averaged over heads) for the local window
        # and predicts a mask probability per window position.
        self.pattern_predictor = nn.Sequential(
            nn.Linear(window_size, window_size * 2),
            nn.SiLU(),
            nn.Linear(window_size * 2, window_size),
            # output: raw logits, sigmoid applied later for Gumbel-sigmoid
        )

        # --- Span encoder ---
        # Embeds the masked n-gram pattern into d_model.
        # Input: window_size token embeddings (masked positions get a learned mask embedding).
        # Output: single d_model vector.
        self.mask_token_embedding = nn.Parameter(torch.randn(n_embd) * 0.02)
        self.span_encoder = nn.Sequential(
            nn.Linear(window_size * n_embd, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

        # --- Gating ---
        # Learned per-layer scalar gate (initialised near zero so the module
        # is ~no-op at the start of training).
        self.gate_logit = nn.Parameter(torch.tensor(-3.0))

        # --- Cache (optional) ---
        # Stores pattern_hash -> span_embedding via EMA updates.
        if cache_size > 0:
            self.register_buffer(
                "cache_keys", torch.zeros(cache_size, dtype=torch.long)
            )
            self.register_buffer(
                "cache_values", torch.zeros(cache_size, n_embd)
            )
            self.register_buffer(
                "cache_counts", torch.zeros(cache_size, dtype=torch.long)
            )
        else:
            self.cache_keys = None

        # Store last MDL loss for the training loop to pick up
        self._last_mdl_loss = None

    # -----------------------------------------------------------------
    # Gumbel-sigmoid: differentiable hard mask
    # -----------------------------------------------------------------
    def _gumbel_sigmoid(self, logits: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """
        Sample from Gumbel-sigmoid distribution.
        Returns values in [0, 1]. If hard=True, applies straight-through
        to get binary {0, 1} in forward but smooth gradients in backward.
        """
        if self.training:
            # Gumbel noise
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            y_soft = torch.sigmoid((logits + gumbel) / self.gumbel_tau)
        else:
            y_soft = torch.sigmoid(logits / self.gumbel_tau)

        if hard:
            y_hard = (y_soft > 0.5).float()
            # Straight-through: forward uses hard, backward uses soft
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    # -----------------------------------------------------------------
    # MDL loss: KL(Bernoulli(p) || Bernoulli(π))
    # -----------------------------------------------------------------
    def _compute_mdl_loss(self, mask_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute MDL penalty as sum of per-position KL divergences
        from the sparse Bernoulli prior.

        mask_probs: (B, T, window_size) — sigmoid probabilities of unmasking
        """
        pi = self.mask_prior
        eps = 1e-7
        p = mask_probs.clamp(eps, 1 - eps)

        kl = p * (torch.log(p) - math.log(pi)) + \
             (1 - p) * (torch.log(1 - p) - math.log(1 - pi))

        # Mean over batch and positions, sum over window
        return kl.sum(dim=-1).mean()

    # -----------------------------------------------------------------
    # Pattern hashing (for cache)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _hash_patterns(self, token_ids: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
        """
        Hash discretised masked n-gram patterns for cache lookup.
        token_ids: (B, T, window_size) — int64
        hard_mask: (B, T, window_size) — binary {0, 1}

        Returns: (B, T) int64 hash values
        """
        # Masked positions get token_id = 0 (MASK), unmasked keep their id
        masked_ids = token_ids * hard_mask.long()
        # Simple polynomial hash
        B, T, W = masked_ids.shape
        # Use position-dependent primes for mixing
        primes = torch.tensor(
            [31, 37, 41, 43, 47, 53, 59][:W], device=masked_ids.device, dtype=torch.long
        )
        hashed = (masked_ids * primes.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return hashed % self.cache_size if self.cache_size > 0 else hashed

    # -----------------------------------------------------------------
    # Cache lookup and update
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _cache_lookup(self, hashes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Look up cached span embeddings.
        Returns: (cached_embeddings, hit_mask)
        """
        if self.cache_keys is None:
            B, T = hashes.shape
            return torch.zeros(B, T, self.n_embd, device=hashes.device), \
                   torch.zeros(B, T, dtype=torch.bool, device=hashes.device)

        flat = hashes.reshape(-1)
        cached = self.cache_values[flat].reshape(*hashes.shape, -1)
        hits = (self.cache_counts[flat] > 0).reshape(hashes.shape)
        return cached, hits

    @torch.no_grad()
    def _cache_update(self, hashes: torch.Tensor, embeddings: torch.Tensor):
        """Update cache with EMA of new span embeddings."""
        if self.cache_keys is None:
            return

        flat_h = hashes.reshape(-1)
        flat_e = embeddings.reshape(-1, self.n_embd)
        alpha = self.cache_ema_decay

        for i in range(flat_h.shape[0]):
            idx = flat_h[i].item()
            if self.cache_counts[idx] == 0:
                self.cache_values[idx] = flat_e[i]
            else:
                self.cache_values[idx] = alpha * self.cache_values[idx] + (1 - alpha) * flat_e[i]
            self.cache_counts[idx] += 1

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        qk_logits: torch.Tensor,
        token_ids: torch.Tensor,
        tok_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, d_model)
            Current hidden states (for residual addition).
        qk_logits : (B, n_head, T, T)
            Raw QK dot products BEFORE softmax (pre-mask, pre-scale).
            We use these to derive the local window attention signal.
        token_ids : (B, T)
            Input token IDs (int64) for constructing the masked n-gram.
        tok_embeddings : (B, T, d_model)
            Token embeddings from the embedding layer (for span encoding).

        Returns
        -------
        out : (B, T, d_model)
            Gated span embedding to ADD to the residual stream.
            Caller does: x = x + attn_out + masked_ngram_memory(...)
        """
        B, T, D = x.shape
        W = self.window_size

        # Positions with insufficient context get zero output
        if T <= W:
            self._last_mdl_loss = torch.tensor(0.0, device=x.device)
            return torch.zeros_like(x)

        # ---- 1. Extract local QK logits for the window ----
        # For each position p, we want QK logits over [p-W, p-1]
        # qk_logits shape: (B, n_head, T, T)
        # Average across heads to get (B, T, T)
        qk_avg = qk_logits.mean(dim=1)  # (B, T, T)

        # Extract contiguous window: for position p (p >= W), gather [p-W : p]
        # Build indices: for each valid position p in [W, T), the window is [p-W, p)
        valid_T = T - W
        # Positions we compute patterns for
        pos_idx = torch.arange(W, T, device=x.device)  # (valid_T,)
        # Window offsets
        win_idx = torch.arange(W, device=x.device)  # (W,)
        # Gather indices: (valid_T, W)
        gather_idx = pos_idx.unsqueeze(1) - W + win_idx.unsqueeze(0)  # (valid_T, W)

        # Extract QK logits for the window at each position
        # qk_avg[:, pos_idx, :] -> (B, valid_T, T), then gather along last dim
        qk_window = qk_avg[:, pos_idx, :]  # (B, valid_T, T)
        gather_expanded = gather_idx.unsqueeze(0).expand(B, -1, -1)  # (B, valid_T, W)
        local_qk = torch.gather(qk_window, dim=2, index=gather_expanded)  # (B, valid_T, W)

        # ---- 2. Pattern predictor ----
        mask_logits = self.pattern_predictor(local_qk)  # (B, valid_T, W)
        mask_probs = torch.sigmoid(mask_logits)

        # ---- 3. MDL loss ----
        self._last_mdl_loss = self._compute_mdl_loss(mask_probs) * self.mdl_weight

        # ---- 4. Gumbel-sigmoid hard mask ----
        hard_mask = self._gumbel_sigmoid(mask_logits, hard=True)  # (B, valid_T, W) binary

        # ---- 5. Build masked n-gram embeddings ----
        # Gather token embeddings for the window positions
        gather_emb_idx = gather_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D)
        tok_emb_expanded = tok_embeddings.unsqueeze(1).expand(-1, valid_T, -1, -1)
        # Simpler: gather from tok_embeddings
        window_emb = torch.gather(
            tok_embeddings.unsqueeze(1).expand(-1, valid_T, -1, -1),
            dim=2,
            index=gather_idx.unsqueeze(0).unsqueeze(-1).expand(B, valid_T, W, D),
        )  # (B, valid_T, W, D)

        # Apply mask: unmasked positions keep token embedding, masked get mask_token
        mask_expanded = hard_mask.unsqueeze(-1)  # (B, valid_T, W, 1)
        mask_tok = self.mask_token_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,D)
        masked_window_emb = mask_expanded * window_emb + (1 - mask_expanded) * mask_tok
        # (B, valid_T, W, D)

        # ---- 6. Span encoder ----
        span_input = masked_window_emb.reshape(B, valid_T, W * D)
        span_emb = self.span_encoder(span_input)  # (B, valid_T, D)

        # ---- 7. Cache lookup/update (optional, inference optimisation) ----
        if self.cache_keys is not None:
            window_token_ids = torch.gather(
                token_ids.unsqueeze(1).expand(-1, valid_T, -1),
                dim=2,
                index=gather_idx.unsqueeze(0).expand(B, -1, -1),
            )  # (B, valid_T, W)
            hashes = self._hash_patterns(window_token_ids, hard_mask)
            cached_emb, hits = self._cache_lookup(hashes)

            if not self.training:
                # At inference, use cache where available
                span_emb = torch.where(hits.unsqueeze(-1), cached_emb, span_emb)
            else:
                # During training, always use fresh computation but update cache
                self._cache_update(hashes, span_emb.detach())

        # ---- 8. Gate and pad to full sequence length ----
        gate = torch.sigmoid(self.gate_logit)

        # Pad: first W positions get zero (no complete window available)
        output = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        output[:, W:, :] = gate * span_emb

        return output

    def get_mdl_loss(self) -> torch.Tensor:
        """Return the MDL loss from the last forward pass."""
        if self._last_mdl_loss is None:
            return torch.tensor(0.0)
        return self._last_mdl_loss


# =========================================================================
# Integration example: how this plugs into CausalSelfAttention
# =========================================================================
"""
In your Block.forward(), the integration looks like:

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = RMSNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = RMSNorm(config.n_embd)
            self.mlp = SwiGLUMLP(config)
            self.ngram_memory = MaskedNgramMemory(
                n_embd=config.n_embd,
                n_head=config.n_head,
                window_size=7,
                mask_prior=0.15,     # expect ~1 of 7 positions unmasked
                mdl_weight=0.01,     # tune this
            )

        def forward(self, x, freqs_cis, token_ids, tok_embeddings):
            # Normal attention (modified to also return QK logits)
            attn_out, qk_logits = self.attn(self.ln_1(x), freqs_cis, return_qk=True)

            # Masked n-gram memory (additive, gated)
            ngram_out = self.ngram_memory(x, qk_logits, token_ids, tok_embeddings)

            x = x + attn_out + ngram_out
            x = x + self.mlp(self.ln_2(x))
            return x

In CausalSelfAttention, add a `return_qk` flag:

    def forward(self, x, freqs_cis, return_qk=False):
        ...
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Raw QK logits (before softmax) — needed for pattern predictor
        qk_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, ...)
        ...
        if return_qk:
            return y, qk_logits
        return y

In the training loop, accumulate MDL losses across layers:

    mdl_loss = sum(
        block.ngram_memory.get_mdl_loss()
        for block in model.transformer.h
    )
    total_loss = lm_loss + mdl_loss
"""

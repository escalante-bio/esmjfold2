# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""AttentionPairBias + DiffusionTransformer."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import AbstractFromTorch, from_torch
from .adaln import AdaptiveLayerNorm, ConditionedTransitionBlock
from .primitives import Linear, LayerNorm


class AttentionPairBias(AbstractFromTorch):
    """Gated multi-head attention with pair bias conditioning."""

    q_proj: Linear
    kv_proj: Linear
    g_proj: Linear
    out_proj: Linear
    pair_norm: LayerNorm | None = None
    pair_bias_proj: Linear | None = None
    adaln: AdaptiveLayerNorm | None = None
    pre_norm: LayerNorm | None = None
    out_gate: Linear | None = None
    d_model: int = 0
    num_heads: int = 0
    head_dim: int = 0

    def __call__(self, a, s, z, beta=0.0, attention_mask=None, pair_bias=None):
        """Single-batch attention with pair bias.

        a:    [B, L, d_model] — queries / keys / values
        s:    [B, L, d_cond] or None (no conditioning → pre_norm)
        z:    [B, L, L, d_pair]
        attention_mask: [B, L] bool (1 = valid)
        pair_bias: [B, L, L, H] precomputed ``compute_pair_bias(z)``; z is then unused
        """
        bsz, n, d_model = a.shape

        if s is not None:
            assert self.adaln is not None
            x = self.adaln(a, s)
        else:
            assert self.pre_norm is not None
            x = self.pre_norm(a)

        q = self.q_proj(x).reshape(bsz, n, self.num_heads, self.head_dim)
        kv = self.kv_proj(x)
        k, v = jnp.split(kv, 2, axis=-1)
        k = k.reshape(bsz, n, self.num_heads, self.head_dim)
        v = v.reshape(bsz, n, self.num_heads, self.head_dim)

        scale = self.head_dim ** -0.5
        # logits [B, Lq, Lk, H]
        logits = jnp.einsum("biha,bjha->bijh", q, k) * scale

        if pair_bias is None and z is not None and self.pair_bias_proj is not None:
            pair_bias = self.compute_pair_bias(z)
        if pair_bias is not None:
            logits = logits + pair_bias

        if attention_mask is not None:
            min_val = jnp.finfo(logits.dtype).min
            mask_bias = jnp.where(attention_mask[:, None, :, None].astype(jnp.bool_), 0.0, min_val)
            logits = logits + mask_bias

        attn = jax.nn.softmax(logits, axis=-2).astype(v.dtype)
        ctx = jnp.einsum("bijh,bjhd->bihd", attn, v)

        g = jax.nn.sigmoid(self.g_proj(x)).reshape(bsz, n, self.num_heads, self.head_dim)
        ctx = g * ctx
        out = self.out_proj(ctx.reshape(bsz, n, d_model))
        if s is not None:
            assert self.out_gate is not None
            out = jax.nn.sigmoid(self.out_gate(s)) * out
        return out

    def compute_pair_bias(self, z):
        return self.pair_bias_proj(self.pair_norm(z))  # [B, L, L, H]


class _AttentionTransitionPair(eqx.Module):
    """One DiffusionTransformer block — attention + transition."""

    attn: AttentionPairBias
    transition: ConditionedTransitionBlock

    def __call__(self, x, s, z, attention_mask=None, pair_bias=None):
        x = x + self.attn(x, s, z, attention_mask=attention_mask, pair_bias=pair_bias)
        x = x + self.transition(x, s)
        return x


class DiffusionTransformer(eqx.Module):
    """Stack of (AttentionPairBias + ConditionedTransitionBlock) → scan."""

    block_params: _AttentionTransitionPair
    block_static: _AttentionTransitionPair

    def compute_pair_bias(self, z):
        """Per-block pair bias of z, stacked: [n_blocks, B, L, L, H]."""
        return jax.vmap(
            lambda params: eqx.combine(self.block_static, params).attn.compute_pair_bias(z)
        )(self.block_params)

    def __call__(self, a, s, z, attention_mask=None, pair_bias=None):
        """``pair_bias``: optional precomputed ``compute_pair_bias(z)``; z is then unused."""
        @jax.checkpoint
        def body(state, inputs):
            params, block_pair_bias = inputs
            block = eqx.combine(self.block_static, params)
            return block(state, s, z, attention_mask=attention_mask, pair_bias=block_pair_bias), None

        out, _ = jax.lax.scan(body, a, (self.block_params, pair_bias))
        return out

    @classmethod
    def from_torch(cls, model):
        attn_list = [from_torch(b) for b in model.attn_blocks]
        trans_list = [from_torch(b) for b in model.transition_blocks]
        pairs = [_AttentionTransitionPair(attn=a, transition=t) for a, t in zip(attn_list, trans_list)]
        params0, static = eqx.partition(pairs[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *vs: jnp.stack(vs, 0),
            *[eqx.filter(p, eqx.is_inexact_array) for p in pairs],
        )
        return cls(block_params=stacked, block_static=static)


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    from_torch.register(common.AttentionPairBias, AttentionPairBias.from_torch)
    from_torch.register(common.DiffusionTransformer, DiffusionTransformer.from_torch)

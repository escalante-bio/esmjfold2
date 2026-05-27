# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""JAX/Equinox translation of ESMC (the language-model backbone of ESMFold2).

Translates the pure-PyTorch fallback path in ``transformers.models.esmc.modeling_esmc``:
``_PyTorchLayerNormLinear`` (LN+Linear), ``_PyTorchLayerNormMLP`` (LN+SwiGLU FFN),
``RotaryEmbedding``, ``MultiHeadAttention``, ``UnifiedTransformerBlock``,
``TransformerStack``, ``ESMCModel``. The accelerated Transformer-Engine fused
modules share the same state-dict layout as the pure-PyTorch fallback, so the
same converter handles both checkpoints.
"""

from __future__ import annotations

import math

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .backend import AbstractFromTorch, from_torch
from .primitives import Embedding, LayerNorm, Linear


class LayerNormLinear(eqx.Module):
    """``layer_norm(x) @ weight^T``, sharing the state-dict layout of the
    PyTorch fused TE module and its ``_PyTorchLayerNormLinear`` fallback.
    """

    layer_norm_weight: Float[Array, "D_in"]
    layer_norm_bias: Float[Array, "D_in"]
    weight: Float[Array, "D_out D_in"]
    d_in: int
    eps: float = 1e-5

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        x = x * self.layer_norm_weight + self.layer_norm_bias
        return einops.einsum(x, self.weight, "... In, Out In -> ... Out")

    @classmethod
    def from_torch(cls, model):
        return cls(
            layer_norm_weight=from_torch(model.layer_norm_weight),
            layer_norm_bias=from_torch(model.layer_norm_bias),
            weight=from_torch(model.weight),
            d_in=int(model.d_in if hasattr(model, "d_in") else model.layer_norm_weight.shape[0]),
            eps=float(getattr(model, "eps", 1e-5)),
        )


class LayerNormMLP(eqx.Module):
    """LN → fc1 (packed 2*ffn) → silu(x1) * x2 → fc2.

    State-dict layout matches the PyTorch / TE fused module.
    """

    layer_norm_weight: Float[Array, "D"]
    layer_norm_bias: Float[Array, "D"]
    fc1_weight: Float[Array, "2H D"]
    fc2_weight: Float[Array, "D H"]
    hidden_size: int
    ffn_hidden_size: int
    eps: float = 1e-5

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        x = x * self.layer_norm_weight + self.layer_norm_bias
        x = einops.einsum(x, self.fc1_weight, "... D, H D -> ... H")
        x1, x2 = jnp.split(x, 2, axis=-1)
        x = jax.nn.silu(x1) * x2
        return einops.einsum(x, self.fc2_weight, "... H, D H -> ... D")

    @classmethod
    def from_torch(cls, model):
        return cls(
            layer_norm_weight=from_torch(model.layer_norm_weight),
            layer_norm_bias=from_torch(model.layer_norm_bias),
            fc1_weight=from_torch(model.fc1_weight),
            fc2_weight=from_torch(model.fc2_weight),
            hidden_size=int(model.hidden_size if hasattr(model, "hidden_size") else model.layer_norm_weight.shape[0]),
            ffn_hidden_size=int(model.ffn_hidden_size if hasattr(model, "ffn_hidden_size") else model.fc2_weight.shape[1]),
            eps=float(getattr(model, "eps", 1e-5)),
        )


# Same conversion handles transformer_engine modules ("TE LayerNormLinear" /
# "TE LayerNormMLP") since they expose the same parameter names. Registration
# happens at the bottom of this file.


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rope(x, cos, sin):
    """x: [B, S, H, D]. cos, sin: [S, D/2]. Returns same shape."""
    ro_dim = cos.shape[-1] * 2
    S = x.shape[1]
    cos = cos[:S][:, None, :]                 # [S, 1, D/2]
    sin = sin[:S][:, None, :]
    cos = jnp.concatenate([cos, cos], axis=-1)  # [S, 1, D]
    sin = jnp.concatenate([sin, sin], axis=-1)
    rot = x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim]) * sin
    return jnp.concatenate([rot, x[..., ro_dim:]], axis=-1)


def _build_rope_cache(seqlen: int, dim: int, base: float, dtype):
    """Compute (cos, sin) of shape [seqlen, dim/2]."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(seqlen, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


class RotaryEmbedding(eqx.Module):
    """ESMC RoPE — rotate_half style, no XPos."""

    dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True, default=10000.0)

    def __call__(self, q, k):
        """q, k: [B, S, H, D]. Cos/sin built per-call (jit caches by shape)."""
        S = q.shape[1]
        cos, sin = _build_rope_cache(S, self.dim, self.base, q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

    @classmethod
    def from_torch(cls, model):
        return cls(dim=int(model.dim), base=float(model.base))


def _attention(q, k, v, n_heads, d_head, sequence_id):
    """SDPA matching the torch fallback path.

    q, k, v: [B, S, n_heads*d_head]
    sequence_id: [B, S] int (mask: same id can attend) or None.
    Returns: [B, S, n_heads*d_head]
    """
    B, S, _ = q.shape
    q = q.reshape(B, S, n_heads, d_head)
    k = k.reshape(B, S, n_heads, d_head)
    v = v.reshape(B, S, n_heads, d_head)

    # JAX dot_product_attention expects (B, S, H, D). Build optional bias.
    if sequence_id is not None:
        # mask[b, i, j] = sequence_id[b, i] == sequence_id[b, j]
        mask = sequence_id[:, :, None] == sequence_id[:, None, :]
        bias = jnp.where(mask, 0.0, jnp.finfo(q.dtype).min).astype(q.dtype)
        bias = bias[:, None, :, :]  # broadcast over heads
        out = jax.nn.dot_product_attention(q, k, v, bias=bias)
    else:
        out = jax.nn.dot_product_attention(q, k, v)

    return out.reshape(B, S, n_heads * d_head)


class MultiHeadAttention(AbstractFromTorch):
    """LayerNormLinear (LN + QKV) → q_ln / k_ln → RoPE → SDPA → out_proj."""

    layernorm_qkv: LayerNormLinear
    out_proj: Linear
    rotary: RotaryEmbedding
    q_ln: LayerNorm | eqx.Module
    k_ln: LayerNorm | eqx.Module
    n_heads: int
    d_head: int

    def __call__(self, x, sequence_id=None):
        qkv = self.layernorm_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = self.q_ln(q).astype(q.dtype)
        k = self.k_ln(k).astype(k.dtype)

        B, S = q.shape[:2]
        q = q.reshape(B, S, self.n_heads, self.d_head)
        k = k.reshape(B, S, self.n_heads, self.d_head)
        q, k = self.rotary(q, k)
        q = q.reshape(B, S, self.n_heads * self.d_head)
        k = k.reshape(B, S, self.n_heads * self.d_head)

        ctx = _attention(q, k, v, self.n_heads, self.d_head, sequence_id)
        return self.out_proj(ctx)


class UnifiedTransformerBlock(AbstractFromTorch):
    attn: MultiHeadAttention
    ffn: LayerNormMLP
    scaling_factor: float

    def __call__(self, x, sequence_id=None):
        x = x + self.attn(x, sequence_id) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class TransformerStack(eqx.Module):
    """80-layer ESMC stack — scan-stacked for compile speed."""

    block_params: UnifiedTransformerBlock
    block_static: UnifiedTransformerBlock
    norm: LayerNorm

    def __call__(self, x, sequence_id=None, *, collect_hidden_states: bool = True):
        """Run the stack. When ``collect_hidden_states`` is True (default), returns
        ``(last_hidden_state, all_hiddens)`` where ``all_hiddens`` is shape
        ``[n_layers + 1, B, S, D]`` containing pre-block inputs for layers
        0..n_layers-1 followed by the final post-norm state. This matches the
        layout that ESMFold2's LanguageModelShim expects.

        The scan body is wrapped in ``jax.checkpoint`` so per-block activations
        are rematerialised on the backward pass instead of stored. At inference
        XLA folds the checkpoint away; under autograd it caps the 80-layer
        stack's residual-stream memory at one block's worth.
        """
        @jax.checkpoint
        def body(state, params):
            block = eqx.combine(self.block_static, params)
            new_state = block(state, sequence_id)
            return new_state, state  # ``state`` is the pre-block input

        last_x, pre_block = jax.lax.scan(body, x, self.block_params)
        norm_x = self.norm(last_x)
        if not collect_hidden_states:
            return norm_x, None

        # pre_block is [n_layers, B, S, D]; final hidden is norm_x. Concat.
        all_hiddens = jnp.concatenate([pre_block, norm_x[None]], axis=0)
        return norm_x, all_hiddens

    @classmethod
    def from_torch(cls, model):
        blocks = [from_torch(b) for b in model.blocks]
        _, static = eqx.partition(blocks[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *vs: jnp.stack(vs, 0),
            *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
        )
        return cls(
            block_params=stacked, block_static=static, norm=from_torch(model.norm)
        )


class ESMC(eqx.Module):
    """Top-level ESMC model: ``embed → transformer``.

    Provides the same return signature as ``ESMCModel.forward``: a tensor of
    ``hidden_states`` of shape ``[n_layers + 1, B, S, d_model]``, which is
    what ``LanguageModelShim`` expects for feature aggregation.
    """

    embed: Embedding
    transformer: TransformerStack

    def __call__(self, input_ids, sequence_id=None, *, collect_hidden_states: bool = True):
        """Args:
            input_ids: [B, S] int32 token indices
            sequence_id: [B, S] int chain-id (same id = can attend; -1 = padding).
                Pass ``None`` to disable chain-aware masking entirely.

        Returns:
            ``(last_hidden_state, hidden_states)`` where ``hidden_states`` is
            ``[n_layers + 1, B, S, D]`` (or ``None`` when not collecting).
        """
        x = self.embed(input_ids)
        return self.transformer(x, sequence_id, collect_hidden_states=collect_hidden_states)

    @classmethod
    def from_torch(cls, model):
        return cls(embed=from_torch(model.embed), transformer=from_torch(model.transformer))


def _convert_te_layernorm_linear(m):
    """Transformer Engine ``LayerNormLinear`` exposes ``layer_norm_weight``,
    ``layer_norm_bias`` and ``weight`` — same as the pure-PyTorch fallback —
    so the same converter handles both.
    """
    return LayerNormLinear.from_torch(m)


def _convert_te_layernorm_mlp(m):
    return LayerNormMLP.from_torch(m)


def register():
    """Register ESMC submodule converters."""
    import torch.nn as tnn
    from transformers.models.esmc.modeling_esmc import (
        ESMCModel,
        MultiHeadAttention as PTMultiHeadAttention,
        RotaryEmbedding as PTRotaryEmbedding,
        TransformerStack as PTTransformerStack,
        UnifiedTransformerBlock as PTUnifiedTransformerBlock,
        _PyTorchLayerNormLinear,
        _PyTorchLayerNormMLP,
    )

    from_torch.register(_PyTorchLayerNormLinear, LayerNormLinear.from_torch)
    from_torch.register(_PyTorchLayerNormMLP, LayerNormMLP.from_torch)
    from_torch.register(PTRotaryEmbedding, RotaryEmbedding.from_torch)
    from_torch.register(PTMultiHeadAttention, MultiHeadAttention.from_torch)
    from_torch.register(PTUnifiedTransformerBlock, UnifiedTransformerBlock.from_torch)
    from_torch.register(PTTransformerStack, TransformerStack.from_torch)
    from_torch.register(ESMCModel, ESMC.from_torch)

    # Also register the TE fused modules when TE is installed: they share the
    # same parameter names as the PyTorch fallback.
    try:
        import transformer_engine.pytorch as te  # type: ignore[import-untyped]
        from_torch.register(te.LayerNormLinear, _convert_te_layernorm_linear)
        from_torch.register(te.LayerNormMLP, _convert_te_layernorm_mlp)
        from_torch.register(te.Linear, lambda m: _te_linear_to_eqx(m))
    except ImportError:
        pass


def _te_linear_to_eqx(m):
    """Transformer Engine ``te.Linear`` has ``.weight`` and optional ``.bias`` —
    convert to our pair primitives.Linear.
    """
    return Linear(
        weight=from_torch(m.weight),
        bias=from_torch(m.bias) if m.bias is not None else None,
    )

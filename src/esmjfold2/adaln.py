# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""Adaptive layer norm + transition + Fourier embedding + conditioned transition."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .backend import AbstractFromTorch, from_torch
from .primitives import Linear


class AdaptiveLayerNorm(AbstractFromTorch):
    """adaLN-Zero: sigmoid(s_gate(LN(s))) * LN(a) + s_shift(LN(s))."""

    s_scale: Float[Array, "Cond"]
    s_gate: Linear
    s_shift: Linear
    d_model: int
    d_cond: int
    eps: float = 1e-5

    def __call__(self, a, s):
        # LN(a) with no affine
        a_mean = a.mean(axis=-1, keepdims=True)
        a_var = jnp.mean(jnp.square(a - a_mean), axis=-1, keepdims=True)
        a_norm = (a - a_mean) * jax.lax.rsqrt(a_var + self.eps)
        # LN(s) with s_scale weight (no bias)
        s_mean = s.mean(axis=-1, keepdims=True)
        s_var = jnp.mean(jnp.square(s - s_mean), axis=-1, keepdims=True)
        s_norm = (s - s_mean) * jax.lax.rsqrt(s_var + self.eps) * self.s_scale
        return jax.nn.sigmoid(self.s_gate(s_norm)) * a_norm + self.s_shift(s_norm)


class TransitionLayer(AbstractFromTorch):
    """LN → a_proj, b_proj → silu(a)*b → out_proj."""

    norm: eqx.Module  # LayerNorm
    a_proj: Linear
    b_proj: Linear
    out_proj: Linear

    def __call__(self, x):
        x = self.norm(x)
        a = self.a_proj(x)
        b = self.b_proj(x)
        return self.out_proj(jax.nn.silu(a) * b)


class FourierEmbedding(eqx.Module):
    """cos(2π (t * w + b)) — w, b are non-trainable buffers."""

    w: Float[Array, "C"]
    b: Float[Array, "C"]

    def __call__(self, t):
        t = jnp.asarray(t).reshape(-1).astype(self.w.dtype)
        return jnp.cos(
            2.0 * jnp.pi * (t[:, None] * self.w[None, :] + self.b[None, :])
        )

    @classmethod
    def from_torch(cls, model):
        return cls(w=from_torch(model.w), b=from_torch(model.b))


class ConditionedTransitionBlock(AbstractFromTorch):
    """adaLN + lin_swish (2x hidden) → silu*gate → lin_out → output_gate."""

    adaln: AdaptiveLayerNorm | None = None
    pre_norm: eqx.Module | None = None  # LayerNorm if no conditioning
    output_gate: Linear | None = None
    lin_swish: Linear = None  # type: ignore[assignment]
    lin_out: Linear = None  # type: ignore[assignment]

    def __call__(self, a, s):
        if s is not None:
            x = self.adaln(a, s)
        else:
            x = self.pre_norm(a)
        sw_a, sw_b = jnp.split(self.lin_swish(x), 2, axis=-1)
        out = self.lin_out(jax.nn.silu(sw_a) * sw_b)
        if s is not None:
            out = jax.nn.sigmoid(self.output_gate(s)) * out
        return out


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    from_torch.register(common.AdaptiveLayerNorm, AdaptiveLayerNorm.from_torch)
    from_torch.register(common.TransitionLayer, TransitionLayer.from_torch)
    from_torch.register(common.FourierEmbedding, FourierEmbedding.from_torch)
    from_torch.register(common.ConditionedTransitionBlock, ConditionedTransitionBlock.from_torch)

# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""Leaf modules: Linear, LayerNorm, RMSNorm, Embedding, Sequential, SwiGLU."""

from __future__ import annotations

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .backend import AbstractFromTorch, from_torch, identity


class Linear(AbstractFromTorch):
    """PyTorch nn.Linear semantics: weight is (Out, In)."""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None = None

    def __call__(self, x):
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + self.bias
        return o


class LayerNorm(AbstractFromTorch):
    weight: Float[Array, "D"] | None = None
    bias: Float[Array, "D"] | None = None
    eps: float = 1e-5

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class RMSNorm(AbstractFromTorch):
    weight: Float[Array, "D"] | None = None
    eps: float = 1e-6

    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x


class Embedding(AbstractFromTorch):
    weight: Float[Array, "Vocab D"]

    def __call__(self, indices):
        return self.weight[indices]


class Sequential(eqx.Module):
    """Equivalent of nn.Sequential, storing children in an ordered dict."""

    _modules: dict[str, eqx.Module]

    def __call__(self, x):
        for i in range(len(self._modules)):
            x = self._modules[str(i)](x)
        return x


def _convert_linear(m):
    return Linear(weight=from_torch(m.weight), bias=from_torch(m.bias) if m.bias is not None else None)


def _convert_layernorm(m):
    return LayerNorm(
        weight=from_torch(m.weight) if m.weight is not None else None,
        bias=from_torch(m.bias) if m.bias is not None else None,
        eps=m.eps,
    )


def _convert_rmsnorm(m):
    eps = m.eps
    if eps is None:
        # PyTorch's RMSNorm defaults eps to None and computes finfo(float32).eps lazily.
        import torch
        eps = torch.finfo(torch.float32).eps
    return RMSNorm(
        weight=from_torch(m.weight) if getattr(m, "weight", None) is not None else None,
        eps=float(eps),
    )


def _convert_embedding(m):
    return Embedding(weight=from_torch(m.weight))


def _convert_sequential(m):
    children = {name: from_torch(child) for name, child in m.named_children()}
    # nn.Sequential children are keyed "0", "1", ...
    return Sequential(_modules=children)


def register():
    """Register converters for PyTorch primitives. Call after register_base_types()."""
    import torch
    import torch.nn as tnn

    from_torch.register(tnn.Linear, _convert_linear)
    from_torch.register(tnn.LayerNorm, _convert_layernorm)
    if hasattr(tnn, "RMSNorm"):
        from_torch.register(tnn.RMSNorm, _convert_rmsnorm)
    from_torch.register(tnn.Embedding, _convert_embedding)
    from_torch.register(tnn.Sequential, _convert_sequential)

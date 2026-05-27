# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""TriangleMultiplicativeUpdate / TriangleMultiplicativeBlock and OuterProductMean."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .backend import AbstractFromTorch, from_torch
from .primitives import Linear, LayerNorm


class TriangleMultiplicativeBlock(AbstractFromTorch):
    """Triangle multiplicative engine (inner block of TriangleMultiplicativeUpdate).

    Field names mirror the PyTorch class.
    """

    norm_start: LayerNorm
    norm_mix: LayerNorm
    proj_bundle: Linear        # in -> 4*latent
    proj_emit: Linear           # latent -> in
    proj_gate: Linear           # in -> in
    flow: str = "outgoing"
    input_channels: int = 0
    latent_channels: int = 0

    def __call__(self, pair_grid, mask=None):
        if mask is None:
            mask = jnp.ones(pair_grid.shape[:-1], dtype=pair_grid.dtype)

        normalized = self.norm_start(pair_grid)
        bundled = self.proj_bundle(normalized)
        # Split into [signal (2*latent), gate (2*latent)]
        signal, gate_logits = jnp.split(bundled, 2, axis=-1)
        routed = signal * jax.nn.sigmoid(gate_logits)
        routed = routed * mask[..., None]

        left, right = jnp.split(routed, 2, axis=-1)
        left = left.astype(jnp.float32)
        right = right.astype(jnp.float32)
        if self.flow == "outgoing":
            contracted = jnp.einsum("bikd,bjkd->bijd", left, right)
        else:
            contracted = jnp.einsum("bkid,bkjd->bijd", left, right)
        contracted = contracted.astype(pair_grid.dtype)

        mixed = self.proj_emit(self.norm_mix(contracted))
        output_gate = jax.nn.sigmoid(self.proj_gate(normalized))
        return mixed * output_gate

    @classmethod
    def from_torch(cls, model):
        return cls(
            norm_start=from_torch(model.norm_start),
            norm_mix=from_torch(model.norm_mix),
            proj_bundle=from_torch(model.proj_bundle),
            proj_emit=from_torch(model.proj_emit),
            proj_gate=from_torch(model.proj_gate),
            flow=str(model.flow),
            input_channels=int(model.input_channels),
            latent_channels=int(model.latent_channels),
        )


class TriangleMultiplicativeUpdate(AbstractFromTorch):
    """Thin wrapper: just forwards to _engine."""

    _engine: TriangleMultiplicativeBlock

    def __call__(self, z, mask=None):
        return self._engine(z, mask=mask)


class OuterProductMean(AbstractFromTorch):
    """MSA -> pair outer-product mean.

    PyTorch behavior depends on ``divide_outer_before_proj``. We translate the
    default (False): ``Wout(outer) / n_valid``.
    """

    norm: LayerNorm
    W: Linear
    Wout: Linear
    d_hidden: int
    divide_outer_before_proj: bool = False

    def __call__(self, m, msa_attention_mask):
        m_norm = self.norm(m)
        mask = msa_attention_mask[..., None].astype(m_norm.dtype)
        x = self.W(m_norm) * mask
        a, b = jnp.split(x, 2, axis=-1)
        mask_f = msa_attention_mask.astype(a.dtype)
        # n_valid[b, i, j] = sum_m mask[b,i,m] * mask[b,j,m]
        n_valid = jnp.einsum("bim,bjm->bij", mask_f, mask_f)[..., None]
        n_valid = jnp.clip(n_valid, min=1.0)
        outer = jnp.einsum("bimc,bjmd->bijcd", a, b)
        outer = outer.reshape(*outer.shape[:-2], -1)
        if self.divide_outer_before_proj:
            return self.Wout(outer / n_valid)
        return self.Wout(outer) / n_valid


class MSAPairWeightedAveraging(AbstractFromTorch):
    """AF3 Algorithm 10 — pair-biased MSA row update."""

    norm_single: LayerNorm
    compute_bias: eqx.Module  # Sequential(LayerNorm, Linear)
    Wv: Linear
    Wgate: Linear
    Wout: Linear
    n_heads: int
    head_width: int

    def __call__(self, msa_repr, pair_repr, pair_attention_mask):
        B, L, M, _ = msa_repr.shape
        h, dh = self.n_heads, self.head_width

        msa_normed = self.norm_single(msa_repr)
        bias = self.compute_bias(pair_repr)  # [B, L, L, n_heads]
        bias = jnp.where(pair_attention_mask[..., None].astype(jnp.bool_), bias, -1e5)
        attn = jax.nn.softmax(bias, axis=-2)  # softmax over j

        v = self.Wv(msa_normed).reshape(B, L, M, h, dh)
        gate = jax.nn.sigmoid(self.Wgate(msa_normed)).reshape(B, L, M, h, dh)
        output = jnp.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, gate)
        return self.Wout(output.reshape(B, L, M, h * dh))


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    from_torch.register(common.TriangleMultiplicativeBlock, TriangleMultiplicativeBlock.from_torch)

    @from_torch.register(common.TriangleMultiplicativeUpdate)
    def _tmu(m):
        return TriangleMultiplicativeUpdate(_engine=from_torch(m._engine))

    from_torch.register(common.OuterProductMean, OuterProductMean.from_torch)
    from_torch.register(common.MSAPairWeightedAveraging, MSAPairWeightedAveraging.from_torch)

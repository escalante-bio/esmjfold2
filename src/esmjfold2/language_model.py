# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""LanguageModelShim + SingleToPair."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .backend import AbstractFromTorch, from_torch, gelu
from .primitives import Linear, LayerNorm, Sequential


class SingleToPair(AbstractFromTorch):
    """downproject → outer-product (sym + diff) → MLP."""

    downproject: Linear
    output_mlp: Sequential

    def __call__(self, x):
        x = self.downproject(x)
        # Match torch ``x.unsqueeze(2) * x.unsqueeze(1)`` / ``x.unsqueeze(2) - x.unsqueeze(1)``:
        # at [b, i, j, d] -> x[b, i, d] (op) x[b, j, d]. Use [:, :, None, :] (i kept) and
        # [:, None, :, :] (j broadcast) to get that convention.
        outer = jnp.concatenate(
            [
                x[:, :, None, :] * x[:, None, :, :],
                x[:, :, None, :] - x[:, None, :, :],
            ],
            axis=3,
        )
        return self.output_mlp(outer)


class LanguageModelShim(eqx.Module):
    """Project ESMC pre-computed hidden states to pair representation.

    Stored as the same parameters as the PyTorch shim:
      - base_z_mlp: SingleToPair, LayerNorm  (Sequential)
      - base_z_linear: LayerNorm, Linear     (Sequential)
      - base_z_combine: [num_layers + 1] weights
    """

    base_z_mlp: Sequential
    base_z_linear: Sequential
    base_z_combine: Float[Array, "L+1"]

    def __call__(self, hidden_states):
        """hidden_states: [B, L, num_layers+1, d_model] → [B, L, L, d_pair]."""
        lm_z = self.base_z_linear(hidden_states)  # [B, L, 81, d_z]
        weights = jax.nn.softmax(self.base_z_combine, axis=0)
        # weighted sum over layer axis (-2)
        lm_z = jnp.einsum("blnd,n->bld", lm_z, weights)
        return self.base_z_mlp(lm_z)

    @classmethod
    def from_torch(cls, model):
        return cls(
            base_z_mlp=from_torch(model.base_z_mlp),
            base_z_linear=from_torch(model.base_z_linear),
            base_z_combine=from_torch(model.base_z_combine),
        )


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    from_torch.register(common.SingleToPair, SingleToPair.from_torch)
    from_torch.register(common.LanguageModelShim, LanguageModelShim.from_torch)

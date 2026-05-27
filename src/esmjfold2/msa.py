# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""MSAEncoder + MSAEncoderBlock."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import AbstractFromTorch, from_torch
from .primitives import Linear
from .triangle import (
    MSAPairWeightedAveraging,
    OuterProductMean,
    TriangleMultiplicativeUpdate,
)
from .trunk import PairTransition


class MSAEncoderBlock(AbstractFromTorch):
    outer_product_mean: OuterProductMean
    tri_mul_out: TriangleMultiplicativeUpdate
    tri_mul_in: TriangleMultiplicativeUpdate
    pair_transition: PairTransition
    msa_pair_weighted_averaging: MSAPairWeightedAveraging | None = None
    msa_transition: PairTransition | None = None
    is_final_block: bool = False

    def __call__(self, m, pair, msa_attention_mask, pair_attention_mask):
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        if not self.is_final_block:
            assert self.msa_pair_weighted_averaging is not None
            assert self.msa_transition is not None
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = m + self.msa_transition(m)
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair


class MSAEncoder(eqx.Module):
    embed: Linear
    project_inputs: Linear
    blocks: list  # heterogeneous: last block has no msa updates

    def __call__(
        self,
        x_pair, x_inputs, msa_oh, has_deletion, deletion_value, msa_attention_mask,
    ):
        m_feat = jnp.concatenate(
            [msa_oh, has_deletion[..., None], deletion_value[..., None]],
            axis=-1,
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs)[:, :, None, :]
        tok_mask = msa_attention_mask[:, :, 0].astype(jnp.bool_)
        pair_attention_mask = tok_mask[:, :, None] & tok_mask[:, None, :]
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_attention_mask, pair_attention_mask)
        return x_pair

    @classmethod
    def from_torch(cls, model):
        return cls(
            embed=from_torch(model.embed),
            project_inputs=from_torch(model.project_inputs),
            blocks=[from_torch(b) for b in model.blocks],
        )


def register():
    from .modeling_refs import _esm
    _, modeling = _esm()
    from_torch.register(modeling.MSAEncoderBlock, MSAEncoderBlock.from_torch)
    from_torch.register(modeling.MSAEncoder, MSAEncoder.from_torch)

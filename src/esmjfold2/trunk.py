# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""FoldingTrunk + PairUpdateBlock + Transition + PairTransition."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import AbstractFromTorch, from_torch
from .primitives import LayerNorm
from .swiglu import SwiGLU
from .triangle import TriangleMultiplicativeUpdate


class Transition(AbstractFromTorch):
    """LN -> SwiGLU residual."""

    norm: LayerNorm
    ffn: SwiGLU

    def __call__(self, x):
        return x + self.ffn(self.norm(x))


class PairTransition(AbstractFromTorch):
    """Same structure as Transition but without residual (used in MSA encoder)."""

    norm: LayerNorm
    ffn: SwiGLU

    def __call__(self, x):
        return self.ffn(self.norm(x))


class PairUpdateBlock(AbstractFromTorch):
    """One folding-trunk block: trimul-out, trimul-in, pair transition residual."""

    tri_mul_out: TriangleMultiplicativeUpdate
    tri_mul_in: TriangleMultiplicativeUpdate
    pair_transition: Transition

    def __call__(self, pair, pair_attention_mask=None):
        # row_drop has r=0 in inference → identity residual sum.
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = self.pair_transition(pair)
        return pair


class FoldingTrunk(eqx.Module):
    """Stack of identical PairUpdateBlocks → scan.

    Stores (block_params, block_static) where block_params has arrays stacked
    along the leading dim. Built via from_torch.
    """

    block_params: PairUpdateBlock
    block_static: PairUpdateBlock

    def __call__(self, pair, pair_attention_mask=None):
        @jax.checkpoint
        def body(p, params):
            block = eqx.combine(self.block_static, params)
            return block(p, pair_attention_mask=pair_attention_mask), None

        pair, _ = jax.lax.scan(body, pair, self.block_params)
        return pair

    @classmethod
    def from_torch(cls, m):
        blocks = [from_torch(b) for b in m.blocks]
        if not blocks:
            raise ValueError("Empty FoldingTrunk.blocks")
        # Partition arrays/static using the first block as the static skeleton.
        params0, static = eqx.partition(blocks[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *vs: jnp.stack(vs, 0),
            *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
        )
        return cls(block_params=stacked, block_static=static)


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    # Transition / PairTransition both wrap a LN + SwiGLU. Tell apart by class.
    from_torch.register(common.Transition, Transition.from_torch)
    # PairTransition lives in modeling_esmfold2.py
    from transformers.models.esmfold2 import modeling_esmfold2
    from_torch.register(modeling_esmfold2.PairTransition, PairTransition.from_torch)
    from_torch.register(common.PairUpdateBlock, PairUpdateBlock.from_torch)
    from_torch.register(common.FoldingTrunk, FoldingTrunk.from_torch)

# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""SwiGLU MLPs from modeling_esmfold2_common."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import AbstractFromTorch, from_torch
from .primitives import Linear


class SwiGLU(AbstractFromTorch):
    """SwiGLU: x12 = w12(x); silu(x12[:,:H]) * x12[:,H:] -> w3."""

    w12: Linear
    w3: Linear
    hidden_features: int

    def __call__(self, x):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = jax.nn.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFN(AbstractFromTorch):
    """Atom transformer FFN: w_up (2*H), w_down."""

    w_up: Linear
    w_down: Linear

    def __call__(self, x):
        x1, x2 = jnp.split(self.w_up(x), 2, axis=-1)
        return self.w_down(jax.nn.silu(x1) * x2)


def register():
    from .modeling_refs import _esm

    common, _ = _esm()
    from_torch.register(common.SwiGLU, SwiGLU.from_torch)
    from_torch.register(common.SwiGLUMLP, SwiGLU.from_torch)  # subclass of SwiGLU
    from_torch.register(common.SwiGLUFFN, SwiGLUFFN.from_torch)

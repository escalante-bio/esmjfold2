"""Orchestrate from_torch registrations.

Importing this module performs the side-effect of registering all of esmjfold2's
JAX modules into the singledispatch. Requires torch + Biohub transformers fork.
"""

from __future__ import annotations

from .backend import from_torch, register_base_types
from . import adaln, attention, atom_encoder, confidence, diffusion, esmc, inputs
from . import language_model, model, msa, primitives, swa, swiglu, triangle, trunk
from . import experimental


def _register_all() -> None:
    register_base_types()
    primitives.register()
    swiglu.register()
    adaln.register()
    triangle.register()
    swa.register()
    atom_encoder.register()
    attention.register()
    inputs.register()
    language_model.register()
    trunk.register()
    msa.register()
    confidence.register()
    esmc.register()
    model.register()
    experimental.register()


_register_all()


__all__ = ["from_torch"]

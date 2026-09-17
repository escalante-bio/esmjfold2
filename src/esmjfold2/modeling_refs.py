"""Lazy imports of Biohub's native Torch modules for converter registration."""

from __future__ import annotations


def _esm():
    """Return the native ESMFold2 layers and model modules."""
    from esm.models.esmfold2 import layers, model

    return layers, model

"""Lazy import of the PyTorch ESMFold2 modules for converter registration.

Importing this module requires torch + the Biohub transformers fork. Only
called when from_torch conversion is actually invoked.
"""

from __future__ import annotations


def _esm():
    """Return (modeling_esmfold2_common, modeling_esmfold2)."""
    from transformers.models.esmfold2 import (
        modeling_esmfold2,
        modeling_esmfold2_common,
    )

    return modeling_esmfold2_common, modeling_esmfold2

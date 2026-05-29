"""JAX/Equinox translation of ESMFold2.

The pretrained model is loaded via ``from_torch``; this requires torch and
the Biohub transformers fork. Pure inference (after conversion + save) does
not require torch.
"""

from .esmc import ESMC, ESMCForMaskedLM
from .experimental import ESMFold2Experimental
from .features import Features
from .lm_features import compute_lm_hidden_states
from .model import ESMFold2
from .prediction import Prediction
from .serialization import load_model, save_model
from .structure_output import (
    output_to_mmcif,
    output_to_mmcif_multi,
    output_to_molecular_complex,
)


def from_torch(x):
    """Convert a PyTorch ESMFold2Model (or any registered submodule) to JAX/Equinox."""
    from .convert import from_torch as _from_torch
    return _from_torch(x)


def prepare_protein_features(sequence: str):
    """Featurize a protein sequence to JAX arrays.

    Uses the torch implementation from the Biohub fork to ensure tensor-exact
    parity with the reference model.
    """
    import jax.numpy as jnp
    from transformers.models.esmfold2.protein_utils import prepare_protein_features as _ppf
    feats = _ppf(sequence)
    return {k: jnp.asarray(v.cpu().numpy()) for k, v in feats.items()}


__all__ = [
    "ESMC",
    "ESMCForMaskedLM",
    "ESMFold2",
    "ESMFold2Experimental",
    "Features",
    "Prediction",
    "compute_lm_hidden_states",
    "from_torch",
    "load_model",
    "output_to_mmcif",
    "output_to_mmcif_multi",
    "output_to_molecular_complex",
    "prepare_protein_features",
    "save_model",
]

# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""Typed output container for the ESMFold2 forward pass.

Mirrors the dict that ``ESMFold2.__call__`` used to return, with named
attributes + jaxtyping shapes. Same field names so downstream consumers
can swap ``out["plddt"]`` → ``out.plddt`` mechanically.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float


class Prediction(eqx.Module):
    """All tensors produced by a single ``ESMFold2`` forward pass.

    Shape conventions:
      B = batch (input batch × ``num_diffusion_samples``)
      L = number of tokens
      A = number of atoms (padded)
    """

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    #: All-atom predicted coordinates. Use with ``Features.atom_to_token`` and
    #: ``Features.atom_attention_mask`` to filter / aggregate per-token.
    sample_atom_coords: Float[Array, "B A 3"]

    #: 64-bin pair-distance distogram, symmetrized over (i, j).
    distogram_logits: Float[Array, "B L L Dbins"]

    # ------------------------------------------------------------------
    # Per-token / per-atom confidence
    # ------------------------------------------------------------------

    #: Mean pLDDT averaged across the atoms of each token. 0–1.
    plddt: Float[Array, "B L"]

    #: Per-atom pLDDT used as the B-factor when writing mmCIF / PDB. 0–1.
    plddt_per_atom: Float[Array, "B A"]

    #: Categorical pLDDT logits (50 bins) — softmax then bin-centre to get
    #: ``plddt_per_atom``. Useful for downstream uncertainty estimates.
    plddt_logits: Float[Array, "B A 50"]

    #: Mask-weighted average of ``plddt_per_atom`` across the whole complex.
    complex_plddt: Float[Array, "B"]

    #: Per-atom 2-way logit (unresolved, resolved). Probability of the atom
    #: being well-defined under the model.
    resolved_logits: Float[Array, "B A 2"]

    # ------------------------------------------------------------------
    # Pairwise confidence (PAE / PDE / pTM)
    # ------------------------------------------------------------------

    #: Predicted Aligned Error logits (64 bins over 0–32 Å).
    pae_logits: Float[Array, "B L L 64"]

    #: Expected PAE per token pair, in Ångströms.
    pae: Float[Array, "B L L"]

    #: Predicted Distance Error logits (64 bins over 0–32 Å).
    pde_logits: Float[Array, "B L L 64"]

    #: Expected PDE per token pair, in Ångströms.
    pde: Float[Array, "B L L"]

    #: Predicted TM-score, max over the per-row aggregate (AF2 / AF3 convention).
    ptm: Float[Array, "B"]

    #: Interface pTM, computed over inter-chain pairs only. Equal to ``ptm``
    #: shape but informative only for multi-chain inputs.
    iptm: Float[Array, "B"]

    # ------------------------------------------------------------------
    # Input metadata pass-through (handy for downstream PDB / mmCIF writers
    # that need to label chains / residues)
    # ------------------------------------------------------------------

    #: Same as ``Features.residue_index``.
    residue_index: Float[Array, "B L"]

    #: Same as ``Features.entity_id``.
    entity_id: Float[Array, "B L"]

# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""Typed input container for the ESMFold2 forward pass.

The torch reference (and ``ESMFold2InputBuilder.prepare_input``) hands the model
a ``dict[str, torch.Tensor]`` with ~22 entries. About half are unused at
inference: ``gt_coords`` / ``is_resolved`` / ``frames_idx`` are training-time,
and ``pocket_feature`` / ``disto_cond`` / ``disto_cond_mask`` are accepted by
the public input API but silently dropped by the model side. ``input_ids`` is
consumed by ``compute_lm_hidden_states`` *outside* the model proper.

This module strips the input down to just what ``ESMFold2.__call__`` reads,
gives every field a typed name + jaxtyping shape, and provides a
``from_input_builder`` classmethod to convert the torch / numpy dict the
``esm`` SDK returns.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class Features(eqx.Module):
    """All inputs the JAX ESMFold2 model actually consumes at inference.

    Shape conventions:
      B = batch (usually 1 in inference)
      L = number of tokens (≈ residues for proteins, + per-atom tokens for ligands)
      A = number of atoms (padded to a multiple of 32)
      M = MSA depth (rows; row 0 is always the query)

    Construct via :meth:`from_input_builder` from the dict returned by
    ``ESMFold2InputBuilder.prepare_input(spi)``, or hand-build by attribute.
    """

    # ------------------------------------------------------------------
    # Token-level features  (shape ``[B, L]`` unless noted)
    # ------------------------------------------------------------------

    #: Per-chain residue numbering (resets at chain boundaries). Used by the
    #: relative-position encoding.
    residue_index: Int[Array, "B L"]

    #: Sequential token index, 0..L-1. Used by the relative-position encoding to
    #: distinguish per-atom tokens within the same residue (HYP, MSE, ...).
    token_index: Int[Array, "B L"]

    #: Numeric chain ID (0..n_chains - 1). Same-chain attention bins in the
    #: relative-position encoding; chain-aware ipTM calculation in the
    #: confidence head.
    asym_id: Int[Array, "B L"]

    #: Numeric symmetry-copy index within an entity (e.g. 0 / 1 for a
    #: homodimer's two chains). Relative-chain bin.
    sym_id: Int[Array, "B L"]

    #: Unique-sequence ID — homodimer chains share an entity_id. Used by the
    #: ``same_entity`` bit of the relative-position encoding; also returned in
    #: ``output["entity_id"]`` for downstream metadata.
    entity_id: Int[Array, "B L"]

    #: 0 = protein, 1 = DNA, 2 = RNA, 3 = non-polymer. Used by the confidence
    #: head's ipTM (ligand contacts are weighted differently).
    mol_type: Int[Array, "B L"]

    #: Residue-type index (2..22 canonical, 23..32 nucleic + UNK). One-hot
    #: expanded by the model and concatenated into the atom-encoder input.
    res_type: Int[Array, "B L"]

    #: Dense covalent-bond bit matrix. Off-diagonal entries flag ligand-atom
    #: or explicit ``CovalentBond`` pairs. Embedded by ``self.token_bonds``.
    token_bonds: Float[Array, "B L L 1"]

    #: True = real token, False = padding. Defines the pair-attention mask
    #: used throughout the trunk and confidence head.
    token_attention_mask: Bool[Array, "B L"]

    #: For each token, the atom index used as its representative when
    #: computing the distogram (typically CB for proteins, P for nucleotides,
    #: the heavy atom itself for ligands).
    distogram_atom_idx: Int[Array, "B L"]

    # ------------------------------------------------------------------
    # Atom-level features  (shape ``[B, A]`` or ``[B, A, ...]``)
    # ------------------------------------------------------------------

    #: CCD-conformer reference positions, used as the atom encoder's initial
    #: geometry (the model never sees the GT coords at inference).
    ref_pos: Float[Array, "B A 3"]

    #: Atomic number per atom. One-hot expanded into the atom-feature vector.
    ref_element: Int[Array, "B A"]

    #: Formal charge per atom. Concatenated into the atom-feature vector.
    ref_charge: Int[Array, "B A"]

    #: Atom name encoded as 4 small integers (``ord(c) - 32``, 0 = pad). Used
    #: by the atom encoder and decoded back to letters when writing mmCIF.
    ref_atom_name_chars: Int[Array, "B A 4"]

    #: Grouping ID per CCD instance — atoms in the same residue / ligand
    #: instance share a uid. The SWA 3D RoPE uses this to keep intra-residue
    #: positional encodings consistent with the reference conformer.
    ref_space_uid: Int[Array, "B A"]

    #: True = real atom, False = padding atom.
    atom_attention_mask: Bool[Array, "B A"]

    #: Atom → owning token index. Multiplied by ``atom_attention_mask`` inside
    #: the model so masked atoms get bucketed to token 0 (and ignored by the
    #: scatter that aggregates per-atom features to per-token).
    atom_to_token: Int[Array, "B A"]

    # ------------------------------------------------------------------
    # MSA features (depth axis M; ``None`` to disable MSA conditioning)
    # ------------------------------------------------------------------

    #: Per-(sequence, position) residue-type index. Row 0 is always the query.
    msa: Optional[Int[Array, "B M L"]] = None

    #: True where a homolog is in-alignment at this position. Drives the
    #: MSA-attention masks and the row-validity counts in ``OuterProductMean``.
    msa_attention_mask: Optional[Bool[Array, "B M L"]] = None

    #: A3M lowercase-flag per cell (an insertion occurred before this column).
    has_deletion: Optional[Bool[Array, "B M L"]] = None

    #: Number of consecutive insertions before this column (rescaled). Fed
    #: into the MSA encoder alongside ``msa_oh`` / ``has_deletion``.
    deletion_value: Optional[Float[Array, "B M L"]] = None

    #: Per-token mean of ``deletion_value`` across the MSA depth. Concatenated
    #: into the atom-encoder input. Defaults to zeros when no MSA was loaded.
    deletion_mean: Optional[Float[Array, "B L"]] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_input_builder(cls, features_dict) -> "Features":
        """Wrap the dict returned by ``ESMFold2InputBuilder.prepare_input``.

        Accepts torch tensors, JAX arrays, or numpy arrays; everything is
        normalised to JAX arrays. Drops the keys the model doesn't use
        (``gt_coords``, ``is_resolved``, ``frames_idx``, ``pocket_feature``,
        ``disto_cond``, ``disto_cond_mask``, ``input_ids`` — that last one is
        consumed by ``compute_lm_hidden_states`` outside the model).
        """
        def _arr(x):
            if hasattr(x, "cpu") and hasattr(x, "numpy"):
                return jnp.asarray(x.detach().cpu().numpy())
            return jnp.asarray(x)

        kwargs = dict(
            residue_index=_arr(features_dict["residue_index"]),
            token_index=_arr(features_dict["token_index"]),
            asym_id=_arr(features_dict["asym_id"]),
            sym_id=_arr(features_dict["sym_id"]),
            entity_id=_arr(features_dict["entity_id"]),
            mol_type=_arr(features_dict["mol_type"]),
            res_type=_arr(features_dict["res_type"]),
            token_bonds=_arr(features_dict["token_bonds"]).astype(jnp.float32),
            token_attention_mask=_arr(features_dict["token_attention_mask"]).astype(jnp.bool_),
            distogram_atom_idx=_arr(features_dict["distogram_atom_idx"]),
            ref_pos=_arr(features_dict["ref_pos"]).astype(jnp.float32),
            ref_element=_arr(features_dict["ref_element"]),
            ref_charge=_arr(features_dict["ref_charge"]),
            ref_atom_name_chars=_arr(features_dict["ref_atom_name_chars"]),
            ref_space_uid=_arr(features_dict["ref_space_uid"]),
            atom_attention_mask=_arr(features_dict["atom_attention_mask"]).astype(jnp.bool_),
            atom_to_token=_arr(features_dict["atom_to_token"]),
        )
        # Optional MSA block (depth-1 MSA from a single-sequence input also
        # populates these; passing them through is harmless).
        for k in ("msa", "msa_attention_mask", "has_deletion",
                  "deletion_value", "deletion_mean"):
            if k in features_dict and features_dict[k] is not None:
                kwargs[k] = _arr(features_dict[k])
        return cls(**kwargs)

# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""Compute ESMFold2 LM features in JAX.

Mirrors ``ESMFold2Model._compute_lm_hidden_states`` (which mirrors
``transformers.models.esmfold2.modeling_esmfold2_common.compute_lm_hidden_states``):

1. Per-batch-element, dedup protein tokens by (asym_id, residue_index) so
   multi-atom-tokenized residues collapse to a single LM token.
2. Insert ``[BOS] chain1 [EOS BOS] chain2 ... [EOS]``.
3. Pad to max length across the batch (optionally to a multiple of N).
4. Build ``sequence_id`` from cumulative BOS markers (PAD positions get -1).
5. Run ESMC.
6. Scatter the resulting per-LM-position hidden states back to per-token
   layout via the ``expand_map``.

Steps 1-4 and 6 have dynamic shapes per batch element — implemented in
numpy. Step 5 is a single jittable ESMC call.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

BOS_ID = 0
PAD_ID = 1
EOS_ID = 2


def _prepare_lm_inputs(
    input_ids: np.ndarray,        # [B, L]
    asym_id: np.ndarray,           # [B, L]
    residue_index: np.ndarray,     # [B, L]
    mol_type: np.ndarray,          # [B, L]
    token_mask: np.ndarray,        # [B, L] bool
    pad_to_multiple: int | None = None,
):
    """Per-element chain-aware LM tokenization. Returns
    ``(lm_input_ids, sequence_id, expand_maps)`` as numpy arrays of shape
    ``([B, max_len], [B, max_len], [B, L])``.
    """
    B, L = input_ids.shape
    protein_mask = (mol_type == 0) & token_mask

    lm_input_list = []
    lm_lengths = []
    expand_maps = np.full((B, L), -1, dtype=np.int64)

    for b in range(B):
        mb = protein_mask[b]
        prot_pos_b = np.nonzero(mb)[0]
        ids_b = input_ids[b][mb]
        asym_b = asym_id[b][mb]
        res_b = residue_index[b][mb]

        # Dedup by (asym_id, residue_index), preserve original order.
        keys = np.stack([asym_b, res_b], axis=1)
        # ``np.unique(axis=0)`` sorts lexically; we want to preserve the
        # order of first appearance instead.
        _, first_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        n_unique = len(first_idx)
        # ``first_idx`` is sorted by the unique-key sort. We want them sorted
        # by first appearance position instead:
        order = np.argsort(first_idx)
        # Build remap: original-unique-index → ordered-index
        remap = np.empty(n_unique, dtype=np.int64)
        remap[order] = np.arange(n_unique)
        inverse_ordered = remap[inverse]

        ids_collapsed = ids_b[first_idx[order]]
        asym_collapsed = asym_b[first_idx[order]]

        chain_ids = np.unique(asym_collapsed)
        parts: list[np.ndarray] = [np.array([BOS_ID], dtype=ids_b.dtype)]
        per_token_lm_pos = np.empty(n_unique, dtype=np.int64)
        cursor = 1
        for i, cid in enumerate(chain_ids):
            in_chain = np.nonzero(asym_collapsed == cid)[0]
            parts.append(ids_collapsed[in_chain])
            per_token_lm_pos[in_chain] = np.arange(cursor, cursor + len(in_chain))
            cursor += len(in_chain)
            if i < len(chain_ids) - 1:
                parts.append(np.array([EOS_ID, BOS_ID], dtype=ids_b.dtype))
                cursor += 2
        parts.append(np.array([EOS_ID], dtype=ids_b.dtype))
        lm_seq = np.concatenate(parts)
        lm_input_list.append(lm_seq)
        lm_lengths.append(len(lm_seq))

        expand_maps[b, prot_pos_b] = per_token_lm_pos[inverse_ordered]

    max_len = max(lm_lengths) if lm_lengths else 1
    if pad_to_multiple is not None and pad_to_multiple > 1:
        max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    lm_input_ids = np.full((B, max_len), PAD_ID, dtype=input_ids.dtype)
    for b in range(B):
        lm_input_ids[b, : lm_lengths[b]] = lm_input_list[b]

    # sequence_id: cumsum of BOS markers minus 1; PAD positions become -1
    seqid = (lm_input_ids == BOS_ID).cumsum(axis=1) - 1
    seqid = np.where(lm_input_ids == PAD_ID, -1, seqid)
    return lm_input_ids, seqid, expand_maps


@eqx.filter_jit
def _run_esmc(esmc, input_ids, sequence_id):
    """JITted ESMC forward returning all hidden states."""
    _, hs = esmc(input_ids, sequence_id, collect_hidden_states=True)
    return hs  # [n_layers+1, B, max_len, D]


def compute_lm_hidden_states(
    esmc,
    input_ids,             # [B, L] jax or np int
    asym_id,
    residue_index,
    mol_type,
    token_mask,
    pad_to_multiple: int | None = None,
):
    """JAX equivalent of ESMFold2Model._compute_lm_hidden_states.

    Returns ``[B, L, n_layers+1, D]`` features ready to pass into
    ``LanguageModelShim``.
    """
    # Convert any JAX arrays to numpy for the Python-side prep.
    arrs = []
    for a in (input_ids, asym_id, residue_index, mol_type, token_mask):
        arrs.append(np.asarray(a))
    lm_input_ids, sequence_id, expand_maps = _prepare_lm_inputs(
        *arrs, pad_to_multiple=pad_to_multiple,
    )

    lm_in = jnp.asarray(lm_input_ids)
    seqid = jnp.asarray(sequence_id)
    hs = _run_esmc(esmc, lm_in, seqid)  # [n_layers+1, B, max_len, D]

    # Scatter back: result[b, l, n, d] = hs[n, b, expand_maps[b, l], d]
    # Positions with expand_maps == -1 stay zero.
    n_plus1, B, _max_len, D = hs.shape
    L = expand_maps.shape[1]
    em = jnp.asarray(expand_maps)
    # gather over axis=2 of hs (the max_len dimension):
    # hs_perm: [B, max_len, n_plus1, D]
    hs_perm = jnp.transpose(hs, (1, 2, 0, 3))
    safe_em = jnp.where(em < 0, 0, em)  # avoid OOB; we'll zero out below
    gathered = jnp.take_along_axis(
        hs_perm, safe_em[:, :, None, None].astype(jnp.int32), axis=1
    )  # [B, L, n_plus1, D]
    valid = (em >= 0)[..., None, None]
    return jnp.where(valid, gathered, 0.0)

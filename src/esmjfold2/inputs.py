# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""InputsEmbedder, ResIdxAsymIdSymIdEntityIdEncoding, RowAttentionPooling."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .atom_encoder import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    MAX_CHARS,
    ESMFold2AtomEncoder,
)
from .backend import AbstractFromTorch, from_torch
from .primitives import Linear


NUM_RES_TYPES = 33


class InputsEmbedder(AbstractFromTorch):
    """Atom encoder (structure_prediction=False) → concat with aatype / profile / deletion."""

    atom_attention_encoder: ESMFold2AtomEncoder

    def __call__(
        self,
        aatype, profile, deletion_mean,
        ref_pos, atom_attention_mask,
        ref_space_uid, ref_charge,
        ref_element_oh, ref_atom_name_chars_oh,
        atom_to_token, n_tokens,
    ):
        a, _, _, _ = self.atom_attention_encoder(
            ref_pos=ref_pos,
            atom_attention_mask=atom_attention_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element_oh=ref_element_oh,
            ref_atom_name_chars_oh=ref_atom_name_chars_oh,
            atom_to_token=atom_to_token,
            n_tokens=n_tokens,
        )
        return jnp.concatenate(
            [a, aatype, profile, deletion_mean[..., None]], axis=-1
        )


class ResIdxAsymIdSymIdEntityIdEncoding(AbstractFromTorch):
    embed: Linear
    n_relative_residx_bins: int
    n_relative_chain_bins: int
    d_pair: int

    def __call__(self, residue_index, asym_id, sym_id, entity_id, token_index):
        r = self.n_relative_residx_bins
        c = self.n_relative_chain_bins

        # Match torch: ``a.unsqueeze(2) - a.unsqueeze(1)`` gives ``a[i] - a[j]``
        # at index [b, i, j]. With ``a[..., :, None]`` (i axis kept) and
        # ``a[..., None, :]`` (j axis broadcast), the same convention holds.
        same_chain = asym_id[..., :, None] == asym_id[..., None, :]
        same_residue = residue_index[..., :, None] == residue_index[..., None, :]
        same_entity = entity_id[..., :, None] == entity_id[..., None, :]

        dij = jnp.clip(
            residue_index[..., :, None] - residue_index[..., None, :] + r,
            0, 2 * r,
        )
        dij = jnp.where(same_chain, dij, 2 * r + 1)
        aij_rel_pos = jax.nn.one_hot(dij, 2 * r + 2)

        dij_tok = jnp.clip(
            token_index[..., :, None] - token_index[..., None, :] + r,
            0, 2 * r,
        )
        dij_tok = jnp.where(same_chain & same_residue, dij_tok, 2 * r + 1)
        aij_rel_token = jax.nn.one_hot(dij_tok, 2 * r + 2)

        dij_chain = jnp.clip(sym_id[..., :, None] - sym_id[..., None, :] + c, 0, 2 * c)
        dij_chain = jnp.where(same_chain, 2 * c + 1, dij_chain)
        aij_rel_chain = jax.nn.one_hot(dij_chain, 2 * c + 2)

        feats = jnp.concatenate(
            [
                aij_rel_pos.astype(jnp.float32),
                aij_rel_token.astype(jnp.float32),
                same_entity[..., None].astype(jnp.float32),
                aij_rel_chain.astype(jnp.float32),
            ],
            axis=-1,
        )
        return self.embed(feats)


class RowAttentionPooling(AbstractFromTorch):
    attn_proj: Linear
    out_proj: Linear

    def __call__(self, z, mask):
        scores = self.attn_proj(z).squeeze(-1)  # [B, L, L]
        # mask is [B, L] — mask out keys (last axis)
        mask_bias = jnp.where(mask[:, None, :].astype(jnp.bool_), 0.0, -1e9)
        scores = scores + mask_bias
        weights = jax.nn.softmax(scores, axis=-1)
        pooled = jnp.einsum("bnm,bnmd->bnd", weights, z)
        return self.out_proj(pooled)


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    from_torch.register(common.InputsEmbedder, InputsEmbedder.from_torch)
    from_torch.register(
        common.ResIdxAsymIdSymIdEntityIdEncoding,
        ResIdxAsymIdSymIdEntityIdEncoding.from_torch,
    )
    from_torch.register(common.RowAttentionPooling, RowAttentionPooling.from_torch)

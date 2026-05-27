# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""ConfidenceHead."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .atom_encoder import _gather_token_to_atom
from .backend import AbstractFromTorch, from_torch
from .inputs import RowAttentionPooling
from .primitives import Embedding, Linear, LayerNorm
from .trunk import FoldingTrunk


_NONPOLYMER_ID = 4
_EPS = 1e-6


def _categorical_mean(logits, start, end):
    n_bins = logits.shape[-1]
    edges = jnp.linspace(start, end, n_bins + 1, dtype=jnp.float32)
    v = (edges[:-1] + edges[1:]) / 2
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return (probs * v).sum(axis=-1)


def _compute_intra_token_idx(atom_to_token):
    """Local atom index within each token. atom_to_token: [B, A]."""
    # group-relative cumulative index. Atoms belonging to a token are contiguous.
    B, A = atom_to_token.shape
    same_as_prev = jnp.concatenate(
        [jnp.zeros((B, 1), dtype=jnp.bool_), atom_to_token[:, 1:] == atom_to_token[:, :-1]],
        axis=1,
    )
    ones = jnp.ones_like(atom_to_token)
    cumsum = jnp.cumsum(ones, axis=-1)
    group_start = jnp.where(same_as_prev, 0, cumsum)
    group_start = jax.lax.cummax(group_start, axis=group_start.ndim - 1)
    return cumsum - group_start


def _gather_rep_atom_coords(coords, rep_idx):
    """coords: [B, A, 3]; rep_idx: [B, L]; returns [B, L, 3]."""
    return jnp.take_along_axis(coords, rep_idx[..., None], axis=1)


class ConfidenceHead(eqx.Module):
    boundaries: Float[Array, "Bins-1"]
    dist_bin_pairwise_embed: Embedding
    s_norm: LayerNorm
    s_inputs_to_single: Linear
    s_to_z: Linear
    s_to_z_transpose: Linear
    s_to_z_prod_in1: Linear
    s_to_z_prod_in2: Linear
    s_to_z_prod_out: Linear
    s_input_to_s: Linear
    s_inputs_norm: LayerNorm
    z_norm: LayerNorm
    row_attention_pooling: RowAttentionPooling
    folding_trunk: FoldingTrunk
    plddt_ln: LayerNorm
    plddt_weight: Float[Array, "23 D B"]
    pae_ln: LayerNorm
    pae_head: Linear
    pde_ln: LayerNorm
    pde_head: Linear
    resolved_ln: LayerNorm
    resolved_weight: Float[Array, "23 D 2"]

    def __call__(
        self,
        s_inputs, z, x_pred,
        distogram_atom_idx, token_attention_mask, atom_to_token,
        atom_attention_mask, asym_id, mol_type,
        relative_position_encoding=None, token_bonds_encoding=None,
    ):
        s_inputs_normed = self.s_inputs_norm(s_inputs)
        z_base = self.z_norm(z)
        if relative_position_encoding is not None:
            z_base = z_base + relative_position_encoding
        if token_bonds_encoding is not None:
            z_base = z_base + token_bonds_encoding
        z_base = z_base + self.s_to_z(s_inputs_normed)[:, :, None, :]
        z_base = z_base + self.s_to_z_transpose(s_inputs_normed)[:, None, :, :]
        z_base = z_base + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :]
            * self.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :]
        )

        pair = z_base
        x_pred_flat = x_pred
        rep_idx = distogram_atom_idx.astype(jnp.int32)

        rep_coords = _gather_rep_atom_coords(x_pred_flat, rep_idx)
        diffs = rep_coords[:, :, None, :] - rep_coords[:, None, :, :]
        rep_distances = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1))
        distogram_bins = (rep_distances[..., None] > self.boundaries).sum(axis=-1).astype(jnp.int32)
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        mask_f = token_attention_mask.astype(jnp.float32)
        pair_mask = mask_f[:, :, None] * mask_f[:, None, :]
        pair = pair + self.folding_trunk(pair, pair_attention_mask=pair_mask)
        single = self.row_attention_pooling(pair, token_attention_mask)

        atom_mask_f = atom_attention_mask.astype(jnp.float32)
        s_at_atoms = _gather_token_to_atom(single, atom_to_token)
        s_at_atoms_ln = self.plddt_ln(s_at_atoms)
        intra = jnp.clip(_compute_intra_token_idx(atom_to_token), 0, self.plddt_weight.shape[0] - 1)
        w_plddt = self.plddt_weight[intra]
        plddt_logits = jnp.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)
        plddt_per_atom = _categorical_mean(plddt_logits, 0.0, 1.0)

        # pae / pde
        pae_logits = self.pae_head(self.pae_ln(pair))
        pae = _categorical_mean(pae_logits, 0.0, 32.0)
        pde_logits = self.pde_head(self.pde_ln(pair))
        pde = _categorical_mean(pde_logits, 0.0, 32.0)

        s_at_atoms_res = self.resolved_ln(s_at_atoms)
        w_res = self.resolved_weight[intra]
        resolved_logits = jnp.einsum("...c,...cb->...b", s_at_atoms_res, w_res)

        # Aggregate plddt: per-atom → per-token mean (mask-aware)
        B = single.shape[0]
        L = single.shape[1]
        # scatter mean
        plddt_sum = jnp.zeros((B, L), dtype=plddt_per_atom.dtype)
        atom_mask_dt = atom_mask_f.astype(plddt_per_atom.dtype)
        plddt_sum = plddt_sum.at[
            jnp.arange(B)[:, None], atom_to_token
        ].add(plddt_per_atom * atom_mask_dt)
        atom_count = jnp.zeros((B, L), dtype=plddt_per_atom.dtype)
        atom_count = atom_count.at[
            jnp.arange(B)[:, None], atom_to_token
        ].add(atom_mask_dt)
        plddt = plddt_sum / jnp.clip(atom_count, min=_EPS)

        complex_plddt = (plddt_per_atom * atom_mask_f).sum(-1) / (atom_mask_f.sum(-1) + _EPS)

        # pTM estimate (simplified — full PyTorch loops over chains for pair_chains_iptm)
        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = jnp.arange(0.5 * bin_width, 32.0, bin_width)
        n_res = mask_f.sum(-1, keepdims=True)
        d0 = 1.24 * jnp.clip(n_res, min=19.0) ** (1 / 3) - 1.8 - 1.24 * (15.0 ** (1 / 3))
        # Note: the exact formula in the source is `1.24 * ((N-15) ** 1/3) - 1.8` with clamp(N, min=19).
        # We compute it carefully here:
        d0 = 1.24 * (jnp.clip(n_res, min=19.0) - 15) ** (1.0 / 3.0) - 1.8
        tm_per_bin = 1.0 / (1.0 + (bin_centers / d0) ** 2)
        pae_probs = jax.nn.softmax(pae_logits, axis=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(-1)
        pair_mask_2d = mask_f[:, :, None] * mask_f[:, None, :]
        ptm_per_row = (tm_expected * pair_mask_2d).sum(-1) / (pair_mask_2d.sum(-1) + _EPS)
        ptm = ptm_per_row.max(-1)

        # ipTM — restrict to inter-chain pairs only.
        inter_chain_mask = (
            (asym_id[..., :, None] != asym_id[..., None, :]).astype(jnp.float32)
            * pair_mask_2d
        )
        iptm_per_row = (tm_expected * inter_chain_mask).sum(-1) / (
            inter_chain_mask.sum(-1) + _EPS
        )
        iptm = iptm_per_row.max(-1)

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt,
            "plddt_per_atom": plddt_per_atom,
            "complex_plddt": complex_plddt,
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm,
            "iptm": iptm,
        }

    @classmethod
    def from_torch(cls, model):
        return cls(
            boundaries=from_torch(model.boundaries),
            dist_bin_pairwise_embed=from_torch(model.dist_bin_pairwise_embed),
            s_norm=from_torch(model.s_norm),
            s_inputs_to_single=from_torch(model.s_inputs_to_single),
            s_to_z=from_torch(model.s_to_z),
            s_to_z_transpose=from_torch(model.s_to_z_transpose),
            s_to_z_prod_in1=from_torch(model.s_to_z_prod_in1),
            s_to_z_prod_in2=from_torch(model.s_to_z_prod_in2),
            s_to_z_prod_out=from_torch(model.s_to_z_prod_out),
            s_input_to_s=from_torch(model.s_input_to_s),
            s_inputs_norm=from_torch(model.s_inputs_norm),
            z_norm=from_torch(model.z_norm),
            row_attention_pooling=from_torch(model.row_attention_pooling),
            folding_trunk=from_torch(model.folding_trunk),
            plddt_ln=from_torch(model.plddt_ln),
            plddt_weight=from_torch(model.plddt_weight),
            pae_ln=from_torch(model.pae_ln),
            pae_head=from_torch(model.pae_head),
            pde_ln=from_torch(model.pde_ln),
            pde_head=from_torch(model.pde_head),
            resolved_ln=from_torch(model.resolved_ln),
            resolved_weight=from_torch(model.resolved_weight),
        )


def register():
    from .modeling_refs import _esm
    _, modeling = _esm()
    from_torch.register(modeling.ConfidenceHead, ConfidenceHead.from_torch)

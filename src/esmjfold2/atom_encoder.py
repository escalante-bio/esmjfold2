# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""ESMFold2AtomEncoder + ESMFold2AtomDecoder."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import AbstractFromTorch, from_torch
from .primitives import Linear, LayerNorm
from .swa import SWAAtomTransformer


MAX_ATOMIC_NUMBER = 128
CHAR_VOCAB_SIZE = 64
MAX_CHARS = 4
XYZ_DIMS = 3
ATOM_FEATURE_DIM = XYZ_DIMS + 1 + 1 + MAX_ATOMIC_NUMBER + CHAR_VOCAB_SIZE * MAX_CHARS


def _scatter_atom_mean(atom_features, atom_to_token_idx, n_tokens, atom_mask):
    """Aggregate per-atom features to per-token features (mean).

    atom_features: [B, A, d]; atom_to_token_idx: [B, A] int64; mask: [B, A] bool.
    n_tokens is a static int (Python int) — must be passed from caller.
    Uses overflow bin trick: bin n_tokens for masked atoms, then drop it.
    """
    B, A, d = atom_features.shape
    n_out = n_tokens + 1
    idx = jnp.where(atom_mask, atom_to_token_idx, n_tokens)
    # zero out masked atoms in the sum
    masked_feats = jnp.where(atom_mask[..., None], atom_features, 0.0)

    def _scatter_one(idx_row, feats_row, mask_row):
        sums = jnp.zeros((n_out, d), dtype=feats_row.dtype)
        sums = sums.at[idx_row].add(feats_row)
        counts = jnp.zeros((n_out,), dtype=jnp.float32)
        counts = counts.at[idx_row].add(mask_row.astype(jnp.float32))
        return sums / jnp.clip(counts[:, None], min=1.0)

    out = jax.vmap(_scatter_one)(idx, masked_feats, atom_mask)
    return out[:, :n_tokens, :]


def _gather_token_to_atom(token_features, atom_to_token_idx):
    """token_features: [B, L, d]; atom_to_token_idx: [B, A]; returns [B, A, d]."""
    return jnp.take_along_axis(
        token_features, atom_to_token_idx[..., None], axis=1
    )


class ESMFold2AtomEncoder(AbstractFromTorch):
    """Atom encoder: builds atom features → norm → SWA transformer → scatter to tokens."""

    atom_linear: Linear
    atom_norm: LayerNorm
    atom_transformer: SWAAtomTransformer
    atom_to_token_linear: Linear
    coords_linear: Linear | None = None  # only when structure_prediction=True
    structure_prediction: bool = False
    d_atom: int = 0
    d_token: int = 0

    def __call__(
        self,
        ref_pos,
        atom_attention_mask,
        ref_space_uid,
        ref_charge,
        ref_element_oh,
        ref_atom_name_chars_oh,
        atom_to_token,
        r_l=None,
        pred_r1=None,
        n_tokens=None,
    ):
        """Returns (a, q, c, cos_sin_mask).

        ref_element_oh: [B, A, MAX_ATOMIC_NUMBER]
        ref_atom_name_chars_oh: [B, A, MAX_CHARS, CHAR_VOCAB_SIZE]
        n_tokens: static Python int (caller must supply).
        """
        B, N = ref_pos.shape[:2]
        atom_feats = jnp.concatenate(
            [
                ref_pos,
                ref_charge[..., None].astype(ref_pos.dtype),
                atom_attention_mask[..., None].astype(ref_pos.dtype),
                ref_element_oh,
                ref_atom_name_chars_oh.reshape(B, N, MAX_CHARS * CHAR_VOCAB_SIZE),
            ],
            axis=-1,
        )
        c_base = self.atom_norm(self.atom_linear(atom_feats))
        cos, sin = self.atom_transformer.build_rope(ref_pos, ref_space_uid)

        q = c_base
        c = c_base
        if self.structure_prediction and r_l is not None:
            if pred_r1 is None:
                pred_r1 = jnp.zeros_like(r_l)
            r_input = jnp.concatenate([r_l, pred_r1], axis=-1)
            q = q + self.coords_linear(r_input)

        q = self.atom_transformer(q, c, cos, sin, mask=atom_attention_mask.astype(jnp.bool_))

        q_to_a = jax.nn.relu(self.atom_to_token_linear(q))
        a = _scatter_atom_mean(q_to_a, atom_to_token, n_tokens, atom_attention_mask.astype(jnp.bool_))
        return a, q, c, (cos, sin)


class ESMFold2AtomDecoder(AbstractFromTorch):
    token_to_atom_linear: Linear
    atom_transformer: SWAAtomTransformer
    norm: LayerNorm
    output_linear: Linear

    def __call__(
        self,
        a_i,             # [B, L, d_token]
        q_l,             # [B, A, d_atom]
        c_l,             # [B, A, d_atom]
        cos,
        sin,
        atom_to_token,
        atom_attention_mask,
    ):
        a_to_q = self.token_to_atom_linear(a_i)
        a_to_q = _gather_token_to_atom(a_to_q, atom_to_token)
        q_l = q_l + a_to_q
        q_l = self.atom_transformer(q_l, c_l, cos, sin, mask=atom_attention_mask.astype(jnp.bool_))
        return self.output_linear(self.norm(q_l))


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    from_torch.register(common.ESMFold2AtomEncoder, ESMFold2AtomEncoder.from_torch)
    from_torch.register(common.ESMFold2AtomDecoder, ESMFold2AtomDecoder.from_torch)

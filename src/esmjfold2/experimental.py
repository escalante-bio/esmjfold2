# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""JAX port of ``ESMFold2ExperimentalModel`` (the ``-Experimental*`` checkpoints).

Differences from the released ``ESMFold2`` model translated in ``model.py``:

* **No parcae state.** The refinement loop simply re-injects ``z`` via a
  zero-initialised ``Sequential(LayerNorm, Linear)`` projection. No
  ``parcae_log_a`` / ``parcae_log_delta`` / ``parcae_b_cont`` /
  ``parcae_input_norm`` / ``parcae_readout`` / ``parcae_coda`` fields.
* **No ``lm_encoder``.** The LM contribution is added once to ``z_init``
  outside the loop, rather than refined per iteration.
* **MSA encoder runs every iteration** (when enabled), with a slightly
  different ``MSAEncoderBlock`` ordering and an explicit ``msa_track_mask``
  that zeros out the contribution for samples with no real (non-query)
  MSA rows.
* **Slimmer ``ConfidenceHead``** (no PDE, no resolved head; otherwise
  identical machinery).
* **No per-loop LM dropout.** A single ``lm_dropout`` may be applied to
  the projected LM contribution before it's added to ``z_init``.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .atom_encoder import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    _gather_token_to_atom,
)
from .backend import AbstractFromTorch, from_torch
from .confidence import (
    _categorical_mean,
    _compute_intra_token_idx,
    _gather_rep_atom_coords,
    _EPS,
    _NONPOLYMER_ID,
)
from .diffusion import DiffusionStructureHead
from .features import Features
from .inputs import (
    InputsEmbedder,
    NUM_RES_TYPES,
    ResIdxAsymIdSymIdEntityIdEncoding,
    RowAttentionPooling,
)
from .language_model import LanguageModelShim
from .model import _Context  # shared embedding context (same shape as release)
from .prediction import Prediction
from .primitives import Embedding, LayerNorm, Linear, Sequential
from .triangle import (
    MSAPairWeightedAveraging,
    OuterProductMean,
    TriangleMultiplicativeUpdate,
)
from .trunk import FoldingTrunk, PairTransition


# ---------------------------------------------------------------------------
# Experimental MSA encoder
# ---------------------------------------------------------------------------


class MSAEncoderBlockExperimental(AbstractFromTorch):
    """Experimental block: MSA-PWA → MSA-transition → OPM → tri-out → tri-in → pair-transition.

    Differs from the release block in (1) ordering — release runs OPM first
    then the MSA updates; experimental runs MSA updates first then OPM —
    and (2) the ``msa_track_mask`` per-batch gate that zeros out a sample's
    contribution when its MSA has no real (non-query) rows.

    The msa/pair transitions are ``PairTransition`` (LN + SwiGLU, no
    residual) — the same class the upstream calls ``_TransitionFFN``.
    """

    outer_product_mean: OuterProductMean
    msa_pair_weighted_averaging: MSAPairWeightedAveraging
    msa_transition: PairTransition
    tri_mul_out: TriangleMultiplicativeUpdate
    tri_mul_in: TriangleMultiplicativeUpdate
    pair_transition: PairTransition

    def __call__(
        self,
        msa_repr,
        pair_repr,
        msa_attention_mask,
        pair_attention_mask,
        msa_track_mask=None,
    ):
        if msa_track_mask is not None:
            gate = msa_track_mask[:, None, None, None].astype(msa_repr.dtype)
            mask_fn = lambda x: x * gate
        else:
            mask_fn = lambda x: x

        msa_repr = msa_repr + mask_fn(
            self.msa_pair_weighted_averaging(msa_repr, pair_repr, pair_attention_mask)
        )
        msa_repr = msa_repr + mask_fn(self.msa_transition(msa_repr))

        pair_repr = pair_repr + mask_fn(
            self.outer_product_mean(msa_repr, msa_attention_mask)
        )
        pair_repr = pair_repr + mask_fn(
            self.tri_mul_out(pair_repr, mask=pair_attention_mask)
        )
        pair_repr = pair_repr + mask_fn(
            self.tri_mul_in(pair_repr, mask=pair_attention_mask)
        )
        pair_repr = pair_repr + mask_fn(self.pair_transition(pair_repr))

        return msa_repr, pair_repr


class MSAEncoderExperimental(eqx.Module):
    """Experimental MSA encoder. Same field names + shapes as release; different
    block class and an outer zero-gating by ``msa_track_mask``.
    """

    embed: Linear
    project_inputs: Linear
    blocks: list  # heterogeneous? actually homogeneous, but a Python list is fine

    def __call__(
        self,
        x_pair,
        x_inputs,
        msa_oh,
        has_deletion,
        deletion_value,
        msa_attention_mask,
    ):
        B, L, M = msa_attention_mask.shape

        m_feat = jnp.concatenate(
            [msa_oh, has_deletion[..., None], deletion_value[..., None]], axis=-1
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs)[:, :, None, :]

        # Per-sample MSA-track gate: True iff ANY non-query (M>1) row of the
        # MSA has a real position at any token. Samples with no real MSA rows
        # get their pair update zeroed.
        if M > 1:
            msa_track_mask = jnp.any(
                msa_attention_mask[:, :, 1:].astype(jnp.bool_), axis=(1, 2)
            )
        else:
            msa_track_mask = jnp.zeros((B,), dtype=jnp.bool_)

        tok_mask = msa_attention_mask[:, :, 0]
        pair_attention_mask = tok_mask[:, :, None] * tok_mask[:, None, :]

        for block in self.blocks:
            m, x_pair = block(
                m, x_pair, msa_attention_mask, pair_attention_mask, msa_track_mask
            )

        x_pair = x_pair * msa_track_mask[:, None, None, None].astype(x_pair.dtype)
        return x_pair

    @classmethod
    def from_torch(cls, model):
        return cls(
            embed=from_torch(model.embed),
            project_inputs=from_torch(model.project_inputs),
            blocks=[from_torch(b) for b in model.blocks],
        )


# ---------------------------------------------------------------------------
# Experimental confidence head — slimmer than release (no PDE, no resolved)
# ---------------------------------------------------------------------------


class ConfidenceHeadExperimental(eqx.Module):
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
    pae_ln: LayerNorm | None
    pae_head: Linear

    def __call__(
        self,
        s_inputs,
        z,
        x_pred,
        distogram_atom_idx,
        token_attention_mask,
        atom_to_token,
        atom_attention_mask,
        asym_id,
        mol_type,
        relative_position_encoding=None,
        token_bonds_encoding=None,
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
        rep_idx = distogram_atom_idx.astype(jnp.int32)
        rep_coords = _gather_rep_atom_coords(x_pred, rep_idx)
        diffs = rep_coords[:, :, None, :] - rep_coords[:, None, :, :]
        rep_distances = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1))
        distogram_bins = (
            (rep_distances[..., None] > self.boundaries).sum(axis=-1).astype(jnp.int32)
        )
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        mask_f = token_attention_mask.astype(jnp.float32)
        pair_mask = mask_f[:, :, None] * mask_f[:, None, :]
        pair = pair + self.folding_trunk(pair, pair_attention_mask=pair_mask)
        single = self.row_attention_pooling(pair, token_attention_mask)

        atom_mask_f = atom_attention_mask.astype(jnp.float32)
        s_at_atoms = _gather_token_to_atom(single, atom_to_token)
        s_at_atoms = self.plddt_ln(s_at_atoms)
        intra = jnp.clip(
            _compute_intra_token_idx(atom_to_token), 0, self.plddt_weight.shape[0] - 1
        )
        w = self.plddt_weight[intra]
        plddt_logits = jnp.einsum("...c,...cb->...b", s_at_atoms, w)
        plddt_per_atom = _categorical_mean(plddt_logits, 0.0, 1.0)

        # per-token mean of plddt_per_atom (scatter_mean)
        B = single.shape[0]
        L = single.shape[1]
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

        complex_plddt = (plddt_per_atom * atom_mask_f).sum(-1) / (
            atom_mask_f.sum(-1) + _EPS
        )

        # PAE / pTM / ipTM
        pae_logits = self.pae_head(pair)
        pae = _categorical_mean(pae_logits, 0.0, 32.0)
        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = jnp.arange(0.5 * bin_width, 32.0, bin_width)
        n_res = mask_f.sum(-1, keepdims=True)
        d0 = 1.24 * (jnp.clip(n_res, min=19.0) - 15) ** (1.0 / 3.0) - 1.8
        tm_per_bin = 1.0 / (1.0 + (bin_centers / d0) ** 2)
        pae_probs = jax.nn.softmax(pae_logits, axis=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(-1)

        pair_mask_2d = mask_f[:, :, None] * mask_f[:, None, :]
        ptm_per_row = (tm_expected * pair_mask_2d).sum(-1) / (
            pair_mask_2d.sum(-1) + _EPS
        )
        ptm = ptm_per_row.max(-1)

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
            pae_ln=None,  # not present in experimental
            pae_head=from_torch(model.pae_head),
        )


# ---------------------------------------------------------------------------
# Top-level experimental model
# ---------------------------------------------------------------------------


class ESMFold2Experimental(eqx.Module):
    inputs_embedder: InputsEmbedder
    z_init_1: Linear
    z_init_2: Linear
    rel_pos: ResIdxAsymIdSymIdEntityIdEncoding
    token_bonds: Linear
    language_model: LanguageModelShim
    folding_trunk: FoldingTrunk
    pair_loop_proj: Sequential  # Sequential(LayerNorm, Linear) zero-init
    structure_head: DiffusionStructureHead
    distogram_head: Linear
    confidence_head: ConfidenceHeadExperimental | None
    msa_encoder: MSAEncoderExperimental | None

    # static config
    d_pair: int = eqx.field(static=True)
    num_loops: int = eqx.field(static=True)
    num_diffusion_samples: int = eqx.field(static=True)
    lm_dropout: float = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Phase 1: featurization
    # ------------------------------------------------------------------

    def _prepare_embeddings(self, features: Features) -> _Context:
        """One-hot / mask / project all input features into a ``_Context``.

        Identical to ``ESMFold2._prepare_embeddings`` except the MSA-kwargs
        branch only handles the 3D-int input shape (the experimental code
        path doesn't take pre-one-hot MSA tensors).
        """
        res_type = features.res_type
        tok_mask = features.token_attention_mask
        atm_mask = features.atom_attention_mask
        atm_mask_f = atm_mask.astype(jnp.float32)

        if res_type.ndim == 2:
            res_type_oh = jax.nn.one_hot(res_type.astype(jnp.int32), NUM_RES_TYPES).astype(jnp.float32)
            res_type_oh = res_type_oh * tok_mask[..., None].astype(jnp.float32)
        else:
            res_type_oh = res_type.astype(jnp.float32)

        ref_element_oh = jax.nn.one_hot(
            features.ref_element.astype(jnp.int32), MAX_ATOMIC_NUMBER
        ).astype(jnp.float32) * atm_mask_f[..., None]
        ref_atom_name_chars_oh = jax.nn.one_hot(
            features.ref_atom_name_chars.astype(jnp.int32), CHAR_VOCAB_SIZE
        ).astype(jnp.float32) * atm_mask_f[..., None, None]
        atom_to_token = features.atom_to_token * atm_mask.astype(jnp.int32)

        msa = features.msa
        if msa is not None:
            if msa.ndim == 3:
                msa_oh_profile = jax.nn.one_hot(msa.astype(jnp.int32), NUM_RES_TYPES).astype(jnp.float32)
            else:
                msa_oh_profile = msa.astype(jnp.float32)
            msa_attn = features.msa_attention_mask
            if msa_attn is not None:
                mf = msa_attn.astype(jnp.float32)[..., None]
                msa_oh_profile = msa_oh_profile * mf
                valid = jnp.clip(msa_attn.astype(jnp.float32).sum(1), min=1)
                profile = msa_oh_profile.sum(1) / valid[..., None]
            else:
                profile = msa_oh_profile.mean(1)
        else:
            profile = res_type_oh

        deletion_mean = features.deletion_mean
        if deletion_mean is None:
            deletion_mean = jnp.zeros(res_type.shape, dtype=jnp.float32)

        L = int(res_type.shape[1])
        x_inputs = self.inputs_embedder(
            aatype=res_type_oh,
            profile=profile.astype(jnp.float32),
            deletion_mean=deletion_mean.astype(jnp.float32),
            ref_pos=features.ref_pos,
            atom_attention_mask=atm_mask,
            ref_space_uid=features.ref_space_uid,
            ref_charge=features.ref_charge,
            ref_element_oh=ref_element_oh,
            ref_atom_name_chars_oh=ref_atom_name_chars_oh,
            atom_to_token=atom_to_token,
            n_tokens=L,
        )

        z_init = (
            self.z_init_1(x_inputs)[:, :, None, :]
            + self.z_init_2(x_inputs)[:, None, :, :]
        )
        rel_pos_enc = self.rel_pos(
            residue_index=features.residue_index,
            asym_id=features.asym_id,
            sym_id=features.sym_id,
            entity_id=features.entity_id,
            token_index=features.token_index,
        )
        token_bonds_enc = self.token_bonds(features.token_bonds.astype(jnp.float32))
        z_init = z_init + rel_pos_enc + token_bonds_enc

        mask_f = tok_mask.astype(jnp.float32)
        pair_mask = mask_f[:, :, None] * mask_f[:, None, :]

        msa_kwargs = None
        if self.msa_encoder is not None and msa is not None:
            B_msa, M, _L = msa.shape
            msa_oh = jax.nn.one_hot(
                jnp.transpose(msa.astype(jnp.int32), (0, 2, 1)), NUM_RES_TYPES
            ).astype(jnp.float32)
            msa_attn = features.msa_attention_mask
            if msa_attn is not None:
                msa_attn_t = jnp.transpose(msa_attn.astype(jnp.float32), (0, 2, 1))
            else:
                msa_attn_t = jnp.broadcast_to(
                    tok_mask[:, :, None].astype(jnp.float32), (B_msa, _L, M)
                )
            msa_oh = msa_oh * msa_attn_t[..., None]
            has_deletion = features.has_deletion
            hd = (
                jnp.transpose(has_deletion.astype(jnp.float32), (0, 2, 1))
                if has_deletion is not None
                else jnp.zeros((B_msa, _L, M), dtype=jnp.float32)
            )
            deletion_value = features.deletion_value
            dv = (
                jnp.transpose(deletion_value.astype(jnp.float32), (0, 2, 1))
                if deletion_value is not None
                else jnp.zeros((B_msa, _L, M), dtype=jnp.float32)
            )
            msa_kwargs = dict(
                x_inputs=x_inputs, msa_oh=msa_oh,
                has_deletion=hd, deletion_value=dv,
                msa_attention_mask=msa_attn_t,
            )

        return _Context(
            x_inputs=x_inputs,
            z_init=z_init,
            relative_position_encoding=rel_pos_enc,
            token_bonds_encoding=token_bonds_enc,
            pair_mask=pair_mask,
            tok_mask=tok_mask,
            ref_pos=features.ref_pos,
            ref_charge=features.ref_charge,
            atm_mask=atm_mask,
            ref_element_oh=ref_element_oh,
            ref_atom_name_chars_oh=ref_atom_name_chars_oh,
            ref_space_uid=features.ref_space_uid,
            atom_to_token=atom_to_token,
            distogram_atom_idx=features.distogram_atom_idx,
            asym_id=features.asym_id,
            mol_type=features.mol_type,
            residue_index=features.residue_index,
            entity_id=features.entity_id,
            msa_kwargs=msa_kwargs,
            n_tokens=L,
            n_atoms=int(features.atom_to_token.shape[1]),
        )

    # ------------------------------------------------------------------
    # Phase 2: trunk refinement loop
    # ------------------------------------------------------------------

    def _run_trunk(
        self,
        ctx: _Context,
        lm_z,
        key,
        *,
        num_loops: int,
        msa_max_depth: int | None = None,
    ):
        """Refinement loop: ``z_{i+1} = trunk(z_init + proj(z_i) [+ msa(z)])``.

        The LM contribution is added once to ``z_init`` before the loop
        (no per-loop refinement, no per-loop dropout — both are static in
        this architecture). Optional dropout is applied once to ``lm_z``.

        ``msa_max_depth`` controls per-loop MSA subsampling. The torch
        experimental reference doesn't subsample, so the default is
        ``None`` (parity with torch). Set a finite cap to opt into the
        paper's per-loop subsampling recipe — at every step, sample
        ``msa_max_depth`` rows from the full MSA (always keeping row 0,
        the query). Same semantics as ``ESMFold2._run_trunk``.

        Same weights every iteration → ``jax.lax.fori_loop``. Body wrapped
        in ``jax.checkpoint`` so reverse-mode autodiff recomputes per-step
        activations rather than materialising them (mirrors the release
        model's memory profile).
        """
        total_steps = max(1, num_loops + 1)

        if lm_z is not None and self.lm_dropout > 0.0:
            key, kdrop = jax.random.split(key)
            keep_prob = 1.0 - self.lm_dropout
            mask = jax.random.bernoulli(kdrop, p=keep_prob, shape=lm_z.shape)
            lm_z = jnp.where(mask, lm_z / keep_prob, 0.0)

        z_init = ctx.z_init + lm_z if lm_z is not None else ctx.z_init
        z0 = jnp.zeros_like(z_init)

        _do_msa_subsample = (
            self.msa_encoder is not None
            and ctx.msa_kwargs is not None
            and msa_max_depth is not None
            and msa_max_depth < ctx.msa_kwargs["msa_oh"].shape[2]
        )
        if _do_msa_subsample:
            _msa_max_depth_static = int(msa_max_depth)
            _full_msa_depth = int(ctx.msa_kwargs["msa_oh"].shape[2])
        else:
            _msa_max_depth_static = None
            _full_msa_depth = None

        def body(step_i, carry):
            z, k = carry
            z = z_init + self.pair_loop_proj(z)
            if self.msa_encoder is not None and ctx.msa_kwargs is not None:
                if _do_msa_subsample:
                    kmsa = jax.random.fold_in(k, step_i)
                    pick = jax.random.choice(
                        kmsa,
                        _full_msa_depth - 1,
                        shape=(_msa_max_depth_static - 1,),
                        replace=False,
                    ) + 1
                    idx = jnp.concatenate([jnp.zeros((1,), dtype=pick.dtype), pick])
                    msa_kwargs_step = {
                        "x_inputs": ctx.msa_kwargs["x_inputs"],
                        "msa_oh": jnp.take(ctx.msa_kwargs["msa_oh"], idx, axis=2),
                        "has_deletion": jnp.take(ctx.msa_kwargs["has_deletion"], idx, axis=2),
                        "deletion_value": jnp.take(ctx.msa_kwargs["deletion_value"], idx, axis=2),
                        "msa_attention_mask": jnp.take(ctx.msa_kwargs["msa_attention_mask"], idx, axis=2),
                    }
                else:
                    msa_kwargs_step = ctx.msa_kwargs
                z = z + self.msa_encoder(x_pair=z, **msa_kwargs_step)
            z = self.folding_trunk(z, pair_attention_mask=ctx.pair_mask)
            return (z, k)

        body = jax.checkpoint(body)
        z, _ = jax.lax.fori_loop(0, total_steps, body, (z0, key))
        return z

    # ------------------------------------------------------------------
    # Phase 3: diffusion structure sampler
    # ------------------------------------------------------------------

    def _sample_structure(
        self,
        ctx: _Context,
        z,
        key,
        *,
        num_sampling_steps: int,
        noise_scale: float | None,
        step_scale: float | None,
    ):
        """Run the Karras-style diffusion sampler in the structure head."""
        return self.structure_head.sample(
            key,
            z_trunk=z, s_inputs=ctx.x_inputs,
            relative_position_encoding=ctx.relative_position_encoding,
            ref_pos=ctx.ref_pos, ref_charge=ctx.ref_charge,
            ref_mask=ctx.atm_mask,
            ref_element_oh=ctx.ref_element_oh,
            ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh,
            ref_space_uid=ctx.ref_space_uid,
            atom_to_token=ctx.atom_to_token,
            token_attention_mask=ctx.tok_mask,
            n_atoms=ctx.n_atoms,
            n_tokens=ctx.n_tokens,
            num_sampling_steps=num_sampling_steps,
            noise_scale=noise_scale,
            step_scale=step_scale,
        )

    # ------------------------------------------------------------------
    # Phase 4: distogram + confidence heads
    # ------------------------------------------------------------------

    def _compute_distogram(self, z):
        """Symmetrize the pair representation and project to bin logits."""
        return self.distogram_head(z + jnp.swapaxes(z, -2, -3))

    def _compute_confidence(self, ctx: _Context, z, sample_atom_coords):
        """Run the confidence head. Returns zero-tensor placeholders when
        the head is absent (some experimental checkpoints ship without
        a trained confidence head)."""
        if self.confidence_head is not None:
            return self.confidence_head(
                s_inputs=ctx.x_inputs, z=z, x_pred=sample_atom_coords,
                distogram_atom_idx=ctx.distogram_atom_idx,
                token_attention_mask=ctx.tok_mask,
                atom_to_token=ctx.atom_to_token,
                atom_attention_mask=ctx.atm_mask,
                asym_id=ctx.asym_id,
                mol_type=ctx.mol_type,
                relative_position_encoding=ctx.relative_position_encoding,
                token_bonds_encoding=ctx.token_bonds_encoding,
            )
        B = ctx.x_inputs.shape[0]
        return {
            "plddt_logits":   jnp.zeros((B, ctx.n_atoms, 50), dtype=jnp.float32),
            "plddt":          jnp.zeros((B, ctx.n_tokens), dtype=jnp.float32),
            "plddt_per_atom": jnp.zeros((B, ctx.n_atoms), dtype=jnp.float32),
            "complex_plddt":  jnp.zeros((B,), dtype=jnp.float32),
            "pae_logits":     jnp.zeros((B, ctx.n_tokens, ctx.n_tokens, 64), dtype=jnp.float32),
            "pae":            jnp.zeros((B, ctx.n_tokens, ctx.n_tokens), dtype=jnp.float32),
            "ptm":            jnp.zeros((B,), dtype=jnp.float32),
            "iptm":           jnp.zeros((B,), dtype=jnp.float32),
        }

    # ------------------------------------------------------------------
    # Top-level forward — orchestration only.
    # ------------------------------------------------------------------

    def __call__(
        self,
        features: Features,
        lm_hidden_states=None,
        *,
        key,
        num_loops: int | None = None,
        num_sampling_steps: int = 14,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        msa_max_depth: int | None = None,
    ) -> Prediction:
        """Run a full forward pass.

        Same signature as :meth:`ESMFold2.__call__` so a caller can swap
        between the release and experimental models without touching the
        call site. The returned :class:`Prediction` has ``resolved_logits``,
        ``pde_logits``, ``pde`` set to ``None`` (the experimental
        confidence head does not produce them).

        ``msa_max_depth`` defaults to ``None`` here, matching the torch
        experimental reference (no per-loop MSA subsampling). Pass an
        integer to opt into the paper's recipe.
        """
        n_loops = self.num_loops if num_loops is None else num_loops

        ctx = self._prepare_embeddings(features)
        lm_z = (
            self.language_model(lm_hidden_states)
            if lm_hidden_states is not None
            else None
        )

        key, ktrunk, ksample = jax.random.split(key, 3)
        z = self._run_trunk(
            ctx, lm_z, ktrunk, num_loops=n_loops, msa_max_depth=msa_max_depth
        )
        sample_atom_coords = self._sample_structure(
            ctx, z, ksample,
            num_sampling_steps=num_sampling_steps,
            noise_scale=noise_scale,
            step_scale=step_scale,
        )

        confidence = self._compute_confidence(ctx, z, sample_atom_coords)
        return Prediction(
            sample_atom_coords=sample_atom_coords,
            distogram_logits=self._compute_distogram(z),
            plddt=confidence["plddt"],
            plddt_per_atom=confidence["plddt_per_atom"],
            plddt_logits=confidence["plddt_logits"],
            complex_plddt=confidence["complex_plddt"],
            resolved_logits=None,
            pae_logits=confidence["pae_logits"],
            pae=confidence["pae"],
            pde_logits=None,
            pde=None,
            ptm=confidence["ptm"],
            iptm=confidence["iptm"],
            residue_index=ctx.residue_index,
            entity_id=ctx.entity_id,
        )

    @classmethod
    def from_torch(cls, model):
        cfg = model.config
        return cls(
            inputs_embedder=from_torch(model.inputs_embedder),
            z_init_1=from_torch(model.z_init_1),
            z_init_2=from_torch(model.z_init_2),
            rel_pos=from_torch(model.rel_pos),
            token_bonds=from_torch(model.token_bonds),
            language_model=from_torch(model.language_model),
            folding_trunk=from_torch(model.folding_trunk),
            pair_loop_proj=from_torch(model.pair_loop_proj),
            structure_head=from_torch(model.structure_head),
            distogram_head=from_torch(model.distogram_head),
            confidence_head=(
                from_torch(model.confidence_head)
                if model.confidence_head is not None
                else None
            ),
            msa_encoder=(
                from_torch(model.msa_encoder)
                if model.msa_encoder is not None
                else None
            ),
            d_pair=int(cfg.d_pair),
            num_loops=int(cfg.num_loops),
            num_diffusion_samples=int(cfg.num_diffusion_samples),
            lm_dropout=float(getattr(cfg, "lm_dropout", 0.0)),
        )


# ---------------------------------------------------------------------------
# Registration entry point — called by convert.py
# ---------------------------------------------------------------------------


def register():
    from transformers.models.esmfold2 import modeling_esmfold2_experimental as ex

    from_torch.register(ex.MSAEncoderBlock, MSAEncoderBlockExperimental.from_torch)
    from_torch.register(ex.MSAEncoder, MSAEncoderExperimental.from_torch)
    from_torch.register(ex.ConfidenceHead, ConfidenceHeadExperimental.from_torch)
    from_torch.register(ex.ESMFold2ExperimentalModel, ESMFold2Experimental.from_torch)
    # _TransitionFFN has the same field layout as our PairTransition (LN + SwiGLU).
    from_torch.register(ex._TransitionFFN, PairTransition.from_torch)

# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""Top-level ESMFold2 model — assembles all components and runs inference.

This translation does NOT load the ESMC-6B language model itself. Instead,
the caller pre-computes ``lm_hidden_states`` in PyTorch via
``ESMFold2Model._compute_lm_hidden_states`` (or sets it to ``None`` to skip
LM conditioning) and passes them as a JAX array. The reason: the LM-feature
preparation contains Python loops over chains with dynamic shapes (e.g.
``torch.unique``, dynamic indexing for BOS/EOS insertion) that are
fundamentally incompatible with ``jax.jit``.

The top-level ``ESMFold2.__call__`` is split into four phases:

    _prepare_embeddings  — featurize inputs (one-hot, MSA, atom encoder,
                           z_init, rel_pos, token_bonds) → ``_Context``
    _run_trunk           — refinement loop + parcae readout + coda → final ``z``
    _sample_structure    — diffusion-head sampler → predicted coordinates
    _compute_confidence  — confidence head → plddt / pTM / ipTM / pae / pde

The first three are the conceptual "trunk + structure" pipeline; the last
runs the confidence head against the final ``z`` + predicted coords.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .atom_encoder import CHAR_VOCAB_SIZE, MAX_ATOMIC_NUMBER
from .backend import AbstractFromTorch, from_torch
from .confidence import ConfidenceHead
from .diffusion import DiffusionStructureHead
from .features import Features
from .prediction import Prediction
from .inputs import (
    InputsEmbedder,
    NUM_RES_TYPES,
    ResIdxAsymIdSymIdEntityIdEncoding,
    RowAttentionPooling,
)
from .language_model import LanguageModelShim
from .msa import MSAEncoder
from .primitives import Linear, LayerNorm
from .trunk import FoldingTrunk


def _inv_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


# ---------------------------------------------------------------------------
# Shared embedding / featurization context.
#
# Built once by ``_prepare_embeddings`` and threaded through the rest of the
# forward path. Holds everything the trunk, structure head, and confidence
# head consume more than once — avoids re-passing 10+ kwargs to each phase.
# ---------------------------------------------------------------------------


class _Context(eqx.Module):
    # Token-level features that downstream heads consume.
    x_inputs:            Float[Array, "B L D_in"]
    z_init:              Float[Array, "B L L D_pair"]
    relative_position_encoding: Float[Array, "B L L D_pair"]
    token_bonds_encoding:Float[Array, "B L L D_pair"]
    pair_mask:           Float[Array, "B L L"]
    tok_mask:            Int[Array,   "B L"]

    # Atom-level features for the structure head.
    ref_pos:             Float[Array, "B A 3"]
    ref_charge:          Int[Array,   "B A"]
    atm_mask:            Int[Array,   "B A"]
    ref_element_oh:      Float[Array, "B A E"]
    ref_atom_name_chars_oh: Float[Array, "B A C V"]
    ref_space_uid:       Int[Array,   "B A"]
    atom_to_token:       Int[Array,   "B A"]
    distogram_atom_idx:  Int[Array,   "B L"]
    asym_id:             Int[Array,   "B L"]
    mol_type:            Int[Array,   "B L"]
    residue_index:       Int[Array,   "B L"]
    entity_id:           Int[Array,   "B L"]

    # Pre-built MSA inputs for the MSA encoder (None if no MSA).
    msa_kwargs:          dict | None

    # Static integer shapes — eqx.field(static=True) keeps them out of pytrees.
    n_tokens:            int = eqx.field(static=True)
    n_atoms:             int = eqx.field(static=True)


class ESMFold2(eqx.Module):
    inputs_embedder: InputsEmbedder
    z_init_1: Linear
    z_init_2: Linear
    rel_pos: ResIdxAsymIdSymIdEntityIdEncoding
    token_bonds: Linear
    language_model: LanguageModelShim
    folding_trunk: FoldingTrunk
    lm_encoder: FoldingTrunk | None
    parcae_input_norm: LayerNorm
    parcae_log_a: Float[Array, "D"]
    parcae_log_delta: Float[Array, "D"]
    parcae_b_cont: Float[Array, "D D"]
    parcae_readout: Linear
    parcae_coda: FoldingTrunk
    structure_head: DiffusionStructureHead
    distogram_head: Linear
    confidence_head: ConfidenceHead
    msa_encoder: MSAEncoder | None

    # static config
    d_pair: int = eqx.field(static=True)
    num_loops: int = eqx.field(static=True)
    num_diffusion_samples: int = eqx.field(static=True)
    msa_encoder_overwrite: bool = eqx.field(static=True)
    lm_dropout: float = eqx.field(static=True)
    per_loop_lm_dropout: bool = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discretized_dynamics(self):
        """Parcae linear-recurrent gates: ``z_new = a · z + B · injected``."""
        delta = jax.nn.softplus(self.parcae_log_delta)
        a = jnp.exp(-delta * jnp.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    # ------------------------------------------------------------------
    # Phase 1: featurization + LM projection
    # ------------------------------------------------------------------

    def _prepare_embeddings(self, features: Features) -> _Context:
        """One-hot / mask / project all input features into a ``_Context``."""
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

        # Profile from MSA (or fall back to res_type for single sequence).
        msa = features.msa
        if msa is not None:
            msa_oh_profile = jax.nn.one_hot(msa.astype(jnp.int32), NUM_RES_TYPES).astype(jnp.float32)
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

        # MSA encoder inputs (pre-transposed to [B, L, M, *]).
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
    # Phase 2: trunk refinement loop + parcae readout/coda
    # ------------------------------------------------------------------

    def _run_trunk(
        self,
        ctx: _Context,
        lm_z,
        key,
        *,
        num_loops: int,
        msa_max_depth: int | None = 1024,
    ):
        """Run the refinement loop and parcae readout/coda. Returns final ``z``.

        Same weights every iteration → ``jax.lax.fori_loop`` so the body is
        compiled once instead of unrolled ``num_loops + 1`` times.

        ``msa_max_depth`` defaults to 1024 (the value from the ESMFold2
        paper). When the loaded MSA is deeper than this we follow the
        paper's "subsample the MSA at every loop" prescription: each
        iteration picks ``msa_max_depth`` random rows from the full MSA
        (always keeping row 0, the query). The paper notes this is
        "vital for improved performance across recurrences."

        Pass ``msa_max_depth=None`` to disable subsampling and exactly
        replicate the HF torch reference (which omits this step).
        """
        total_steps = max(1, num_loops + 1)

        # Random init for the linear-recurrent pair state.
        key, kinit = jax.random.split(key)
        std = math.sqrt(2.0 / (5.0 * ctx.z_init.shape[-1]))
        z = std * jax.random.truncated_normal(
            kinit, lower=-3, upper=3, shape=ctx.z_init.shape, dtype=jnp.float32
        )

        a, b = self._discretized_dynamics()
        a = a.reshape(1, 1, 1, -1)

        _per_loop_lm_dropout = (
            lm_z is not None and self.per_loop_lm_dropout and self.lm_dropout > 0.0
        )

        # Decide once (at trace time) whether we'll subsample the MSA.
        _do_msa_subsample = (
            ctx.msa_kwargs is not None
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
            if _per_loop_lm_dropout:
                # ESMFold2 keeps LM dropout active at inference; fold_in gives
                # each step a distinct deterministic key.
                kdrop = jax.random.fold_in(k, step_i)
                keep_prob = 1.0 - self.lm_dropout
                mask = jax.random.bernoulli(kdrop, p=keep_prob, shape=lm_z.shape)
                lm_z_i = jnp.where(mask, lm_z / keep_prob, 0.0)
            else:
                lm_z_i = lm_z

            refined_lm_z = None
            if lm_z_i is not None and self.lm_encoder is not None:
                refined_lm_z = self.lm_encoder(lm_z_i, pair_attention_mask=ctx.pair_mask)
            z_inject_pair = ctx.z_init
            if lm_z_i is not None and self.lm_encoder is None:
                z_inject_pair = z_inject_pair + lm_z_i
            if self.msa_encoder is not None and ctx.msa_kwargs is not None:
                if _do_msa_subsample:
                    # Per-loop MSA subsample (paper's "vital" inference recipe).
                    # Always keep row 0 (the query); sample max_depth-1 from
                    # rows 1..M-1 without replacement. fold_in offsets by a
                    # large constant so the MSA RNG doesn't collide with the
                    # LM-dropout RNG at the same step_i.
                    kmsa = jax.random.fold_in(k, step_i + 2**20)
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
                msa_pair = self.msa_encoder(x_pair=z_inject_pair, **msa_kwargs_step)
                z_inject_pair = msa_pair if self.msa_encoder_overwrite else z_inject_pair + msa_pair
            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z

            injected = self.parcae_input_norm(z_inject_pair)
            z_new = a * z + injected @ b.T
            z_new = self.folding_trunk(z_new, pair_attention_mask=ctx.pair_mask)
            return (z_new, k)

        z, _ = jax.lax.fori_loop(0, total_steps, body, (z, key))

        z = self.parcae_readout(z)
        z = self.parcae_coda(z, pair_attention_mask=ctx.pair_mask)
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
        msa_max_depth: int | None = 1024,
    ):
        """Run a full forward pass.

        ``features`` is a :class:`Features` instance. Build one from the dict
        returned by ``ESMFold2InputBuilder.prepare_input`` via
        :meth:`Features.from_input_builder`.

        ``lm_hidden_states``: ``[B, L, num_layers+1, d_model]`` from ESMC, or
        ``None`` to skip LM conditioning.
        """
        n_loops = self.num_loops if num_loops is None else num_loops

        ctx = self._prepare_embeddings(features)
        lm_z = (
            self.language_model(lm_hidden_states)
            if lm_hidden_states is not None
            else None
        )

        key, ktrunk, ksample = jax.random.split(key, 3)
        z = self._run_trunk(ctx, lm_z, ktrunk, num_loops=n_loops, msa_max_depth=msa_max_depth)
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
            resolved_logits=confidence["resolved_logits"],
            pae_logits=confidence["pae_logits"],
            pae=confidence["pae"],
            pde_logits=confidence["pde_logits"],
            pde=confidence["pde"],
            ptm=confidence["ptm"],
            iptm=confidence["iptm"],
            residue_index=ctx.residue_index,
            entity_id=ctx.entity_id,
        )

    # ------------------------------------------------------------------
    # PyTorch → JAX conversion
    # ------------------------------------------------------------------

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
            lm_encoder=from_torch(model.lm_encoder) if model.lm_encoder is not None else None,
            parcae_input_norm=from_torch(model.parcae_input_norm),
            parcae_log_a=from_torch(model.parcae_log_a),
            parcae_log_delta=from_torch(model.parcae_log_delta),
            parcae_b_cont=from_torch(model.parcae_b_cont),
            parcae_readout=from_torch(model.parcae_readout),
            parcae_coda=from_torch(model.parcae_coda),
            structure_head=from_torch(model.structure_head),
            distogram_head=from_torch(model.distogram_head),
            confidence_head=from_torch(model.confidence_head),
            msa_encoder=from_torch(model.msa_encoder) if model.msa_encoder is not None else None,
            d_pair=int(cfg.d_pair),
            num_loops=int(cfg.num_loops),
            num_diffusion_samples=int(cfg.num_diffusion_samples),
            msa_encoder_overwrite=bool(cfg.msa_encoder_overwrite),
            lm_dropout=float(getattr(cfg.lm_encoder, "lm_dropout", 0.0)),
            per_loop_lm_dropout=bool(getattr(cfg.lm_encoder, "per_loop_lm_dropout", False)),
        )


def register():
    from .modeling_refs import _esm
    from .diffusion import DiffusionConditioning, DiffusionModule, DiffusionStructureHead

    common, modeling = _esm()
    from_torch.register(common.DiffusionConditioning, DiffusionConditioning.from_torch)
    from_torch.register(common.DiffusionModule, DiffusionModule.from_torch)
    from_torch.register(common.DiffusionStructureHead, DiffusionStructureHead.from_torch)
    from_torch.register(modeling.ESMFold2Model, ESMFold2.from_torch)

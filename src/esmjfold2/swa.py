# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""Sliding-window 3D-RoPE atom transformer: SWA3DRoPEAttention, SWAAtomBlock, SWAAtomTransformer."""

from __future__ import annotations

import math

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .backend import AbstractFromTorch, from_torch
from .primitives import Linear
from .swiglu import SwiGLUFFN


def _qk_rms_norm(x):
    """rms_norm with no learned weight, matching torch.nn.functional.rms_norm."""
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(var + 1e-6)).astype(x.dtype)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb_3d(x, cos, sin):
    """Apply RoPE with batched cos/sin.
    x: [B, L, H, D]; cos, sin: [B, L, D/2]."""
    ro_dim = cos.shape[-1] * 2
    cos = jnp.broadcast_to(cos[:, :, None, :], (cos.shape[0], cos.shape[1], x.shape[2], cos.shape[-1]))
    sin = jnp.broadcast_to(sin[:, :, None, :], (sin.shape[0], sin.shape[1], x.shape[2], sin.shape[-1]))
    cos = einops.repeat(cos, "b l h d -> b l h (two d)", two=2)
    sin = einops.repeat(sin, "b l h d -> b l h (two d)", two=2)
    rot = x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim]) * sin
    return jnp.concatenate([rot, x[..., ro_dim:]], axis=-1)


def build_3d_rope(
    ref_pos,
    ref_space_uid,
    head_dim: int,
    n_spatial_per_axis: int,
    n_uid_pairs: int,
    spatial_base_freq: float,
    uid_base_freq: float,
):
    """Return (cos, sin) for SWA atom transformer.

    Matches `build_3d_rope` in modeling_esmfold2_common: per-axis pairs +
    UID frequencies, zero-padded to head_dim/2.
    """
    B, N = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    spatial_inv_freq = 1.0 / (
        spatial_base_freq
        ** (jnp.arange(0, n_spatial_per_axis, dtype=jnp.float32) / n_spatial_per_axis)
    )
    uid_inv_freq = 1.0 / (
        uid_base_freq
        ** (jnp.arange(0, n_uid_pairs, dtype=jnp.float32) / n_uid_pairs)
    )

    pos_f32 = ref_pos.astype(jnp.float32)
    spatial_freqs = jnp.einsum("bna,k->bnak", pos_f32, spatial_inv_freq).reshape(
        B, N, n_spatial_total
    )

    uid_f32 = ref_space_uid.astype(jnp.float32)
    uid_freqs = jnp.einsum("bn,k->bnk", uid_f32, uid_inv_freq)

    n_active = n_spatial_total + n_uid_pairs
    freqs = jnp.concatenate([spatial_freqs, uid_freqs], axis=-1)

    if n_active < half_dim:
        pad = jnp.zeros((B, N, half_dim - n_active), dtype=jnp.float32)
        freqs = jnp.concatenate([freqs, pad], axis=-1)
    elif n_active > half_dim:
        freqs = freqs[..., :half_dim]

    return jnp.cos(freqs), jnp.sin(freqs)


def _swa_attention(q, k, v, half_window, scale, mask, use_window: bool = True):
    """Sliding-window attention along the atom axis.

    q, k, v: ``[B, L, H, D]``. ``mask``: ``[B, L]`` bool (True = valid), or
    ``None``. ``half_window``: window radius — each query attends only to
    keys with ``|i - j| <= half_window``.

    ``use_window=True`` (the default) applies the SWA mask the released
    ESMFold2 weights were trained with. ``use_window=False`` recovers the
    torch ``no-flash_attn`` fallback (dense attention, no window, no
    padding mask) — useful when you want bit-similarity to that path
    rather than to what the model was trained on.
    """
    B, L, H, D = q.shape
    q_bhqd = jnp.swapaxes(q, 1, 2)
    k_bhqd = jnp.swapaxes(k, 1, 2)
    v_bhqd = jnp.swapaxes(v, 1, 2)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q_bhqd, k_bhqd) * scale

    if use_window:
        # SWA mask: |i - j| <= half_window. Broadcasts over (B, H).
        qi = jnp.arange(L)[:, None]
        ki = jnp.arange(L)[None, :]
        window_ok = jnp.abs(qi - ki) <= half_window
        # Padding mask: keep keys where mask[b, k] is True.
        if mask is not None:
            kkeep = mask[:, None, None, :].astype(jnp.bool_)
            keep = window_ok[None, None, :, :] & kkeep
        else:
            keep = window_ok[None, None, :, :]
        logits = jnp.where(keep, logits, jnp.finfo(logits.dtype).min)

    attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v_bhqd)
    return jnp.swapaxes(out, 1, 2)


class SWA3DRoPEAttention(AbstractFromTorch):
    n_heads: int
    head_dim: int
    half_window: int
    Wqkv: Linear
    out_proj: Linear
    gate_proj: Linear

    def __call__(self, x, cos, sin, mask=None, use_window: bool = True):
        B, N = x.shape[:2]
        qkv = self.Wqkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = _qk_rms_norm(q), _qk_rms_norm(k)
        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)
        scale = self.head_dim ** -0.5
        out = _swa_attention(q, k, v, self.half_window, scale, mask, use_window=use_window)
        out = out.reshape(B, N, -1)
        out = out * jax.nn.sigmoid(self.gate_proj(x))
        return self.out_proj(out)

    @classmethod
    def from_torch(cls, model):
        return cls(
            n_heads=int(model.n_heads),
            head_dim=int(model.head_dim),
            half_window=int(model.half_window),
            Wqkv=from_torch(model.Wqkv),
            out_proj=from_torch(model.out_proj),
            gate_proj=from_torch(model.gate_proj),
        )


class SWAAtomBlock(AbstractFromTorch):
    attn_norm: eqx.Module  # RMSNorm without affine
    ffn_norm: eqx.Module
    adaln_modulation: eqx.Module  # Sequential(SiLU, Linear)
    attn: SWA3DRoPEAttention
    ffn: SwiGLUFFN

    def __call__(self, x, c_l, cos, sin, mask=None, use_window: bool = True):
        mod = self.adaln_modulation(c_l)
        if mod.ndim == 2:
            mod = mod[:, None, :]
        # Split into 6 chunks
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(mod, 6, axis=-1)

        attn_input = self.attn_norm(x) * (1 + scale_a) + shift_a
        attn_out = self.attn(attn_input, cos, sin, mask=mask, use_window=use_window)
        x = x + gate_a * attn_out

        ffn_input = self.ffn_norm(x) * (1 + scale_f) + shift_f
        ffn_out = self.ffn(ffn_input)
        x = x + gate_f * ffn_out
        return x


class SWAAtomTransformer(eqx.Module):
    """Stack of SWAAtomBlocks → scan.

    ``use_swa_window`` (static, default True) toggles the sliding-window
    attention mask used during training. Set to False to recover the torch
    ``no-flash_attn`` fallback (dense atom attention, no padding mask) —
    matches that path bit-similarly. The trained weights expect the window
    so True is the right default; False is a knob for parity testing.
    """

    block_params: SWAAtomBlock
    block_static: SWAAtomBlock
    head_dim: int
    swa_window_size: int
    spatial_rope_base_frequency: float
    n_spatial_rope_pairs_per_axis: int
    n_uid_rope_pairs: int
    uid_rope_base_frequency: float
    use_swa_window: bool = eqx.field(static=True, default=True)

    def build_rope(self, ref_pos, ref_space_uid):
        return build_3d_rope(
            ref_pos,
            ref_space_uid,
            head_dim=self.head_dim,
            n_spatial_per_axis=self.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.n_uid_rope_pairs,
            spatial_base_freq=self.spatial_rope_base_frequency,
            uid_base_freq=self.uid_rope_base_frequency,
        )

    def __call__(self, q_l, c_l, cos, sin, mask=None):
        use_window = self.use_swa_window

        def body(state, params):
            block = eqx.combine(self.block_static, params)
            q = block(state, c_l, cos, sin, mask=mask, use_window=use_window)
            return q, None

        out, _ = jax.lax.scan(body, q_l, self.block_params)
        return out

    @classmethod
    def from_torch(cls, model):
        blocks = [from_torch(b) for b in model.blocks]
        params0, static = eqx.partition(blocks[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *vs: jnp.stack(vs, 0),
            *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
        )
        return cls(
            block_params=stacked,
            block_static=static,
            head_dim=int(model.head_dim),
            swa_window_size=int(model.swa_window_size),
            spatial_rope_base_frequency=float(model.spatial_rope_base_frequency),
            n_spatial_rope_pairs_per_axis=int(model.n_spatial_rope_pairs_per_axis),
            n_uid_rope_pairs=int(model.n_uid_rope_pairs),
            uid_rope_base_frequency=float(model.uid_rope_base_frequency),
        )


def register():
    from .modeling_refs import _esm
    common, _ = _esm()
    from_torch.register(common.SWA3DRoPEAttention, SWA3DRoPEAttention.from_torch)
    from_torch.register(common.SWAAtomBlock, SWAAtomBlock.from_torch)
    from_torch.register(common.SWAAtomTransformer, SWAAtomTransformer.from_torch)

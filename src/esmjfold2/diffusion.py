# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""DiffusionConditioning + DiffusionModule + DiffusionStructureHead (with sampler)."""

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .adaln import FourierEmbedding, TransitionLayer
from .atom_encoder import ESMFold2AtomDecoder, ESMFold2AtomEncoder
from .attention import DiffusionTransformer
from .backend import AbstractFromTorch, from_torch
from .primitives import Linear, LayerNorm


class DiffusionConditioning(AbstractFromTorch):
    """Conditions pair/single representations on noise timestep."""

    z_input_norm: LayerNorm
    z_proj: Linear
    z_transitions: list[TransitionLayer]
    s_input_norm: LayerNorm
    s_proj: Linear
    fourier: FourierEmbedding
    noise_norm: LayerNorm
    noise_proj: Linear
    s_transitions: list[TransitionLayer]
    sigma_data: float
    c_z: int
    c_s: int
    c_s_inputs: int

    def __call__(self, t_hat, s_inputs, z_trunk, relative_position_encoding):
        # z conditioning (only once across diffusion steps; here we always recompute)
        z = jnp.concatenate(
            [z_trunk.astype(jnp.float32), relative_position_encoding.astype(jnp.float32)],
            axis=-1,
        )
        z = self.z_proj(self.z_input_norm(z))
        for block in self.z_transitions:
            z = z + block(z)

        s = self.s_proj(self.s_input_norm(s_inputs.astype(jnp.float32)))

        t = jnp.asarray(t_hat, dtype=jnp.float32).reshape(-1)
        t_noise = 0.25 * jnp.log(jnp.clip(t / self.sigma_data, min=1e-20))
        n = self.fourier(t_noise)
        n = self.noise_proj(self.noise_norm(n))
        s = s + n[:, None, :]

        for block in self.s_transitions:
            s = s + block(s)

        return s, z

    @classmethod
    def from_torch(cls, model):
        return cls(
            z_input_norm=from_torch(model.z_input_norm),
            z_proj=from_torch(model.z_proj),
            z_transitions=[from_torch(b) for b in model.z_transitions],
            s_input_norm=from_torch(model.s_input_norm),
            s_proj=from_torch(model.s_proj),
            fourier=from_torch(model.fourier),
            noise_norm=from_torch(model.noise_norm),
            noise_proj=from_torch(model.noise_proj),
            s_transitions=[from_torch(b) for b in model.s_transitions],
            sigma_data=float(model.sigma_data),
            c_z=int(model.c_z),
            c_s=int(model.c_s),
            c_s_inputs=int(model.c_s_inputs),
        )


class DiffusionModule(AbstractFromTorch):
    """Diffusion denoising module."""

    conditioning: DiffusionConditioning
    atom_encoder: ESMFold2AtomEncoder
    atom_decoder: ESMFold2AtomDecoder
    s_to_token: Linear
    token_transformer: DiffusionTransformer
    s_step_norm: LayerNorm
    token_norm: LayerNorm
    sigma_data: float

    def __call__(
        self,
        x_noisy,                         # [B, A, 3]
        t_hat,                           # scalar or [B]
        ref_pos, ref_charge, ref_mask,
        ref_element_oh, ref_atom_name_chars_oh, ref_space_uid,
        tok_idx,                         # atom_to_token, [B, A]
        s_inputs, z_trunk,
        relative_position_encoding,
        n_tokens,                        # static int
        token_attention_mask=None,
    ):
        bsz = x_noisy.shape[0]
        t = jnp.broadcast_to(jnp.asarray(t_hat, dtype=jnp.float32).reshape(-1), (bsz,))
        sigma = self.sigma_data
        s, z = self.conditioning(t, s_inputs, z_trunk, relative_position_encoding)

        denom = jnp.sqrt(t * t + sigma * sigma)
        r_noisy = x_noisy / denom[:, None, None]

        a, q_skip, c_skip, (cos, sin) = self.atom_encoder(
            ref_pos=ref_pos,
            atom_attention_mask=ref_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element_oh=ref_element_oh,
            ref_atom_name_chars_oh=ref_atom_name_chars_oh,
            atom_to_token=tok_idx,
            r_l=r_noisy,
            n_tokens=n_tokens,
        )

        a = a + self.s_to_token(self.s_step_norm(s))
        a = self.token_transformer(a, s, z, attention_mask=token_attention_mask)
        a = self.token_norm(a)

        r_update = self.atom_decoder(
            a_i=a, q_l=q_skip, c_l=c_skip, cos=cos, sin=sin,
            atom_to_token=tok_idx, atom_attention_mask=ref_mask,
        )

        sigma2 = sigma * sigma
        t2 = t * t
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / jnp.sqrt(sigma2 + t2))[:, None, None] * r_update
        return out


def _random_rotation(key, dtype):
    """Single random rotation matrix from a Gaussian quaternion."""
    q = jax.random.normal(key, (4,), dtype=dtype)
    scale = jnp.sqrt(jnp.sum(q * q))
    q = q / jnp.where(q[0] < 0, -scale, scale)
    r, i, j, k = q[0], q[1], q[2], q[3]
    two_s = 2.0 / jnp.sum(q * q)
    R = jnp.stack(
        [
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ]
    ).reshape(3, 3)
    return R


def _center_random_augmentation(key, x, atom_mask, second=None):
    """Center coords, random rotate, random translate."""
    bsz = x.shape[0]
    mask = atom_mask[..., None]
    denom = jnp.clip(mask.sum(axis=1, keepdims=True), min=1.0)
    mean = (x * mask).sum(axis=1, keepdims=True) / denom
    x = x - mean
    if second is not None:
        second = second - mean

    krot, ktrans = jax.random.split(key)
    R = jax.vmap(_random_rotation, in_axes=(0, None))(
        jax.random.split(krot, bsz), x.dtype
    )  # [B, 3, 3]
    x = jnp.einsum("bmd,bds->bms", x, R)
    if second is not None:
        second = jnp.einsum("bmd,bds->bms", second, R)
    t = jax.random.normal(ktrans, (bsz, 1, 3), dtype=x.dtype)
    x = x + t
    if second is not None:
        second = second + t
    return x, second


def _weighted_rigid_align(x, x_gt, w, mask):
    """Kabsch alignment with weights."""
    w = (mask * w)[..., None]
    denom = jnp.clip(w.sum(axis=-2, keepdims=True), min=1e-8)
    mu = (x * w).sum(axis=-2, keepdims=True) / denom
    mu_gt = (x_gt * w).sum(axis=-2, keepdims=True) / denom
    x_c = x - mu
    xgt_c = x_gt - mu_gt
    H = jnp.einsum("bni,bnj->bij", w * xgt_c, x_c).astype(jnp.float32)
    U, _, Vh = jnp.linalg.svd(H, full_matrices=False)
    det = jnp.linalg.det(U @ Vh)
    ones = jnp.ones_like(det)
    diag = jnp.stack([ones, ones, det], axis=-1)
    R = (U @ (diag[..., None, :] * jnp.eye(3)) @ Vh).astype(x.dtype)
    return x_c @ jnp.swapaxes(R, -1, -2) + mu_gt


class DiffusionStructureHead(AbstractFromTorch):
    """Wrapper around DiffusionModule with the Karras-style diffusion sampler."""

    diffusion_module: DiffusionModule
    sigma_data: float
    gamma_0: float
    gamma_min: float
    noise_scale: float
    step_scale: float
    inference_s_max: float
    inference_s_min: float
    inference_p: float
    inference_num_steps: int

    def noise_schedule(self, num_steps: int):
        """Karras schedule + trailing 0; static num_steps. Returns numpy array."""
        import numpy as np
        if num_steps == 1:
            return np.asarray(
                [self.inference_s_max * self.sigma_data, 0.0], dtype=np.float32
            )
        p = self.inference_p
        inv_p = 1.0 / p
        k = np.arange(num_steps, dtype=np.float32)
        base = self.inference_s_max ** inv_p + (k / (num_steps - 1)) * (
            self.inference_s_min ** inv_p - self.inference_s_max ** inv_p
        )
        sched = self.sigma_data * (base ** p)
        return np.concatenate([sched, np.zeros((1,), dtype=np.float32)])

    def _truncated_schedule(self, num_steps: int, max_inference_sigma: float | None):
        """Build the schedule at the Python level so we can drop high-σ entries
        and prepend the cap. Returns a numpy array — the shape is then static.
        """
        import numpy as np
        sched = self.noise_schedule(num_steps)
        if max_inference_sigma is not None:
            mask = sched <= max_inference_sigma
            sched = sched[mask]
            sched = np.concatenate([[max_inference_sigma], sched])
        return sched

    def sample(
        self,
        key,
        z_trunk,
        s_inputs,
        relative_position_encoding,
        ref_pos, ref_charge, ref_mask,
        ref_element_oh, ref_atom_name_chars_oh, ref_space_uid,
        atom_to_token,
        token_attention_mask=None,
        n_atoms=None,
        n_tokens=None,
        num_sampling_steps: int = 14,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
    ):
        """Algorithm-18-style sampler. ``num_sampling_steps`` and ``n_atoms``/
        ``n_tokens`` must be Python ints (static).

        ``noise_scale`` (λ) and ``step_scale`` (η) override the model's
        defaults. Lower λ → less stochasticity at each step; lower η →
        smaller corrector steps.
        """
        B = s_inputs.shape[0]
        if n_atoms is None:
            n_atoms = atom_to_token.shape[1]
        if n_tokens is None:
            n_tokens = s_inputs.shape[1]

        sched = jnp.asarray(self._truncated_schedule(num_sampling_steps, max_inference_sigma))
        gammas = jnp.where(sched > self.gamma_min, self.gamma_0, 0.0)

        lam = self.noise_scale if noise_scale is None else float(noise_scale)
        eta = self.step_scale if step_scale is None else float(step_scale)

        # Initialize from gaussian at sched[0]
        key, kinit = jax.random.split(key)
        x = sched[0] * jax.random.normal(kinit, (B, n_atoms, 3), dtype=jnp.float32)
        atom_mask = ref_mask.astype(jnp.float32)

        # We pre-build (sigma_tm, sigma_t, gamma) tuples per step
        sigma_tms = sched[:-1]
        sigma_ts = sched[1:]
        gammas_t = gammas[1:]

        def step(carry, inputs):
            x, key = carry
            sigma_tm, sigma_t, gamma = inputs

            key, k_aug, k_noise = jax.random.split(key, 3)
            x, _ = _center_random_augmentation(k_aug, x, atom_mask)

            t_hat = sigma_tm * (1.0 + gamma)
            eps_std = lam * jnp.sqrt(jnp.clip(t_hat * t_hat - sigma_tm * sigma_tm, min=0.0))
            x_noisy = x + eps_std * jax.random.normal(k_noise, x.shape, dtype=x.dtype)

            x_denoised = self.diffusion_module(
                x_noisy=x_noisy,
                t_hat=jnp.full((B,), t_hat, dtype=jnp.float32),
                ref_pos=ref_pos, ref_charge=ref_charge, ref_mask=ref_mask,
                ref_element_oh=ref_element_oh,
                ref_atom_name_chars_oh=ref_atom_name_chars_oh,
                ref_space_uid=ref_space_uid,
                tok_idx=atom_to_token,
                s_inputs=s_inputs, z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                n_tokens=n_tokens,
                token_attention_mask=token_attention_mask,
            )

            # Rigid-align noisy onto denoised
            x_noisy_aligned = _weighted_rigid_align(
                x_noisy.astype(jnp.float32), x_denoised.astype(jnp.float32), atom_mask, atom_mask
            ).astype(x_denoised.dtype)

            denoised_over_sigma = (x_noisy_aligned - x_denoised) / t_hat
            x = x_noisy_aligned + eta * (sigma_t - t_hat) * denoised_over_sigma
            return (x, key), None

        (x, _), _ = jax.lax.scan(step, (x, key), (sigma_tms, sigma_ts, gammas_t))
        return x

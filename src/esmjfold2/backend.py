# SPDX-License-Identifier: Apache-2.0
# Translated from PyTorch reference Copyright 2026 Biohub. All rights reserved.
"""from_torch singledispatch + AbstractFromTorch helper.

Pattern adapted from the pytorch-to-equinox skill and the esmj package.
Conversion is opt-in: importing this module does *not* import torch.
"""

from __future__ import annotations

import dataclasses
from functools import singledispatch
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


@singledispatch
def from_torch(obj):
    raise TypeError(f"No from_torch converter registered for {type(obj)}")


def register_base_types():
    """Register basic Python and torch primitives. Call once when torch is imported."""
    import torch

    @from_torch.register(torch.Tensor)
    def _tensor(t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return jnp.asarray(t.cpu().numpy())

    from_torch.register(int, lambda x: x)
    from_torch.register(float, lambda x: x)
    from_torch.register(bool, lambda x: x)
    from_torch.register(str, lambda x: x)
    from_torch.register(type(None), lambda x: x)
    from_torch.register(tuple, lambda t: tuple(from_torch(v) for v in t))
    from_torch.register(list, lambda lst: [from_torch(v) for v in lst])
    from_torch.register(dict, lambda d: {k: from_torch(v) for k, v in d.items()})

    @from_torch.register(torch.nn.Identity)
    def _identity(_):
        return identity

    @from_torch.register(torch.nn.ReLU)
    def _relu(_):
        return relu

    @from_torch.register(torch.nn.SiLU)
    def _silu(_):
        return silu

    @from_torch.register(torch.nn.GELU)
    def _gelu(_):
        return gelu

    @from_torch.register(torch.nn.Sigmoid)
    def _sigmoid(_):
        return sigmoid

    @from_torch.register(torch.nn.Dropout)
    def _dropout(m):
        # Inference: dropout is identity. The HF model has all dropouts set to 0
        # for inference anyway (see DropoutResidual default r=0).
        return identity

    @from_torch.register(torch.nn.ModuleList)
    def _modulelist(m):
        return [from_torch(child) for child in m]


# Pre-bound activation modules (eqx.Module wrappers around jax.nn functions)
class _Identity(eqx.Module):
    def __call__(self, x):
        return x


class _ReLU(eqx.Module):
    def __call__(self, x):
        return jax.nn.relu(x)


class _SiLU(eqx.Module):
    def __call__(self, x):
        return jax.nn.silu(x)


class _GELU(eqx.Module):
    def __call__(self, x):
        # PyTorch nn.GELU defaults to "none" (no approx). Match it.
        return jax.nn.gelu(x, approximate=False)


class _Sigmoid(eqx.Module):
    def __call__(self, x):
        return jax.nn.sigmoid(x)


identity = _Identity()
relu = _ReLU()
silu = _SiLU()
gelu = _GELU()
sigmoid = _Sigmoid()


class AbstractFromTorch(eqx.Module):
    """Mixin: auto-implement ``from_torch`` by name-matching dataclass fields
    against the PyTorch module's ``named_children``, ``named_parameters``, and
    ``named_buffers``.
    """

    @classmethod
    def from_torch(cls, model):
        field_to_field = {f.name: f for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}

        for name, child in model.named_children():
            if name in field_to_field:
                kwargs[name] = from_torch(child)

        for name, param in model.named_parameters(recurse=False):
            if name in field_to_field:
                kwargs[name] = from_torch(param)

        for name, buf in model.named_buffers(recurse=False):
            if name in field_to_field:
                kwargs[name] = from_torch(buf)

        # Optional/default-having fields can be left unset. Plain attributes on
        # the torch module (e.g. config scalars not registered as parameters or
        # children) take priority over the dataclass default.
        for name, field in field_to_field.items():
            if name in kwargs:
                continue
            if hasattr(model, name):
                attr = getattr(model, name)
                # Skip if it's a callable / method / submodule we already processed.
                if not callable(attr) or isinstance(attr, (int, float, bool, str)):
                    kwargs[name] = from_torch(attr)
                    continue
            if field.default is not dataclasses.MISSING:
                kwargs[name] = field.default
            elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                kwargs[name] = field.default_factory()
            elif _is_optional(field.type):
                kwargs[name] = None
            else:
                raise ValueError(
                    f"{cls.__name__}: field {name!r} (type {field.type}) is "
                    f"required but missing from torch module {type(model).__name__}"
                )

        return cls(**kwargs)


def _is_optional(field_type) -> bool:
    """Best-effort 'None is allowed for this annotation' check."""
    s = str(field_type)
    return "None" in s or "Optional" in s or "| None" in s

"""Demonstration: load Biohub/ESMFold2-Fast, convert to JAX/Equinox, predict structure.

Mirrors the workflow shown in the project README. Requires:
- biohub/ESMFold2-Fast weights (downloads on first run)
- biohub/ESMC-6B if include_lm_features=True (downloads on first run; ~12GB)
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import torch

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.protein_utils import prepare_protein_features

import esmjfold2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ", help="Protein sequence")
    p.add_argument("--num-loops", type=int, default=3)
    p.add_argument("--num-sampling-steps", type=int, default=14)
    p.add_argument("--include-lm", action="store_true",
                   help="Load ESMC-6B and use its features (much higher quality)")
    p.add_argument("--save-to", help="Save converted JAX model to this path prefix")
    args = p.parse_args()

    print(f"Loading ESMFold2-Fast (include_lm={args.include_lm})...")
    t0 = time.time()
    model = ESMFold2Model.from_pretrained(
        "biohub/ESMFold2-Fast",
        load_esmc=args.include_lm,
        dtype=torch.float32,
        esmc_precision="bf16" if args.include_lm else None,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s. Trunk+head params: ", end="")
    print(f"{sum(p.numel() for n, p in model.named_parameters() if not n.startswith('_esmc')):,}")

    features = prepare_protein_features(args.seq)
    B, L = features["res_type"].shape

    print("Converting trunk + structure head to JAX/Equinox...")
    t0 = time.time()
    eqx_model = esmjfold2.from_torch(model)
    print(f"  {time.time()-t0:.1f}s")

    features_jax = {k: jnp.asarray(v.cpu().numpy()) for k, v in features.items()}

    if args.include_lm:
        print("Converting ESMC-6B to JAX/Equinox...")
        # Force fp32 weights for the JAX conversion.
        model._esmc = model._esmc.to(dtype=torch.float32)
        model._esmc_fp8 = False
        t0 = time.time()
        eqx_esmc = esmjfold2.from_torch(model._esmc)
        print(f"  {time.time()-t0:.1f}s")
        print("Computing LM features in JAX...")
        t0 = time.time()
        lm_jax = esmjfold2.compute_lm_hidden_states(
            eqx_esmc,
            features["input_ids"], features["asym_id"],
            features["residue_index"], features["mol_type"],
            features["token_attention_mask"],
        )
        lm_jax.block_until_ready()
        print(f"  {time.time()-t0:.1f}s shape={lm_jax.shape}")
    else:
        lm_jax = jnp.zeros((B, L, model.config.lm_num_layers + 1,
                            model.config.lm_d_model), dtype=jnp.float32)

    if args.save_to:
        print(f"Saving trunk to {args.save_to}.{{eqx, skeleton.pkl}}...")
        esmjfold2.save_model(eqx_model, args.save_to)
        if args.include_lm:
            print(f"Saving ESMC to {args.save_to}_esmc.{{eqx, skeleton.pkl}}...")
            esmjfold2.save_model(eqx_esmc, f"{args.save_to}_esmc")

    @eqx.filter_jit
    def predict(m, f, lm, key):
        return m(f, lm_hidden_states=lm, key=key, num_loops=args.num_loops,
                 num_sampling_steps=args.num_sampling_steps)

    print(f"\nPredicting structure for sequence (L={L}, sampling_steps="
          f"{args.num_sampling_steps}, loops={args.num_loops})...")
    t0 = time.time()
    out = predict(eqx_model, features_jax, lm_jax, jax.random.key(0))
    out["sample_atom_coords"].block_until_ready()
    print(f"  JIT compile + first run: {time.time()-t0:.1f}s")
    t0 = time.time()
    out = predict(eqx_model, features_jax, lm_jax, jax.random.key(0))
    out["sample_atom_coords"].block_until_ready()
    print(f"  cached run: {(time.time()-t0)*1000:.1f}ms")

    print("\n=== Output ===")
    print(f"sample_atom_coords: {out['sample_atom_coords'].shape}")
    print(f"plddt:              {out['plddt'].shape}, mean={float(out['plddt'].mean()):.3f}")
    print(f"ptm:                {float(out['ptm'][0]):.3f}")
    print(f"distogram_logits:   {out['distogram_logits'].shape}")
    print(f"pae:                {out['pae'].shape}, mean={float(out['pae'].mean()):.3f}")


if __name__ == "__main__":
    main()

"""Self-contained ESMFold2 prediction script (JAX/Equinox).

Loads ``biohub/ESMFold2-Fast`` + ``biohub/ESMC-6B`` (in PyTorch), converts
both to JAX/Equinox, runs structure prediction under ``eqx.filter_jit``, and
writes an mmCIF file plus a JSON sidecar with confidence metrics.

Usage
-----
Single chain (default = ubiquitin / 1UBQ):

    uv run python scripts/predict.py

Specify a sequence:

    uv run python scripts/predict.py --seq MQIFVKTLT... --out ubq.cif

Multi-chain protein complex (chain spec ``ID:SEQ`` separated by commas):

    uv run python scripts/predict.py \\
        --chains "A:GSHSMRY...,B:IQRTPK...,C:SLLMWITQC" \\
        --out complex.cif

Sampling knobs (defaults match ESMFold2-Fast inference defaults):

    --num-loops 3                Refinement loops in the trunk
    --num-sampling-steps 14      Karras diffusion steps
    --num-diffusion-samples 1    Parallel samples in the structure head
    --seed 0                     PRNG seed (deterministic on CPU)

GPU memory notes
----------------
ESMC-6B in fp32 takes ~25 GB GPU. After running it once we free its weights
before invoking the JIT-compiled trunk to leave headroom. Default JAX
preallocation is disabled so torch + JAX can share the GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Must be set before importing JAX.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch

import jax
import jax.numpy as jnp
import equinox as eqx

import esmjfold2

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


def _parse_chains(
    arg: str, msas: dict[str, "MSA"] | None = None
) -> list[ProteinInput]:
    chains = []
    msas = msas or {}
    for spec in arg.split(","):
        if ":" not in spec:
            raise SystemExit(f"Bad --chains entry {spec!r}; expected 'ID:SEQUENCE'.")
        cid, seq = spec.split(":", 1)
        cid = cid.strip()
        chains.append(
            ProteinInput(id=cid, sequence=seq.strip(), msa=msas.get(cid))
        )
    if not chains:
        raise SystemExit("--chains must produce at least one ProteinInput.")
    unused = set(msas) - {c.id for c in chains}
    if unused:
        raise SystemExit(f"--msa references unknown chain id(s): {sorted(unused)}")
    return chains


def _load_msas(specs: list[str], max_sequences: int) -> dict[str, "MSA"]:
    """Parse --msa entries (``CHAIN_ID=path/to.a3m``) into ``MSA`` objects."""
    from esm.utils.msa import MSA
    out: dict[str, MSA] = {}
    for s in specs:
        if "=" not in s:
            raise SystemExit(
                f"Bad --msa entry {s!r}; expected 'CHAIN_ID=path/to/file.a3m'."
            )
        cid, path = s.split("=", 1)
        cid = cid.strip()
        path = path.strip()
        m = MSA.from_a3m(path=path, remove_insertions=True, max_sequences=max_sequences)
        out[cid] = m
        print(f"  loaded MSA for chain {cid}: depth={m.depth} from {path}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--seq", help="Single protein sequence (chain 'A').")
    g.add_argument(
        "--chains",
        help="Comma-separated ID:SEQUENCE specs for multi-chain prediction.",
    )
    p.add_argument(
        "--out",
        default="prediction.cif",
        help="Output mmCIF path (default: prediction.cif).",
    )
    p.add_argument(
        "--checkpoint",
        default="biohub/ESMFold2-Fast",
        help="HuggingFace repo for ESMFold2 weights.",
    )
    p.add_argument("--num-loops", type=int, default=3)
    p.add_argument("--num-sampling-steps", type=int, default=14)
    p.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Draw N independent diffusion samples and pick the best by "
        "ranking_score (0.8·ipTM + 0.2·pTM for complexes, pTM for monomers).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="Override the diffusion sampler's λ (default from model config ≈1.003). "
        "Lower → less stochastic sampling.",
    )
    p.add_argument(
        "--step-scale",
        type=float,
        default=None,
        help="Override the diffusion sampler's η (default from model config ≈1.5). "
        "Lower → smaller corrector steps.",
    )
    p.add_argument(
        "--msa",
        action="append",
        default=[],
        metavar="CHAIN=PATH.a3m",
        help="Attach an MSA to a chain. Repeatable: --msa B=B.a3m --msa F=F.a3m. "
        "Only meaningful for the full biohub/ESMFold2 checkpoint; biohub/"
        "ESMFold2-Fast doesn't condition on the MSA encoder.",
    )
    p.add_argument(
        "--msa-load-depth",
        type=int,
        default=1024,
        help="Cap on the number of MSA rows loaded from each a3m file. The "
        "model sees up to this many; per-loop subsampling (--msa-max-depth) "
        "further narrows this each iteration.",
    )
    p.add_argument(
        "--msa-max-depth",
        type=int,
        default=1024,
        help="Subsample the MSA to this depth at every trunk refinement loop "
        "(paper's recipe — default 1024 matches the paper). No-op when the "
        "loaded MSA is shallower than this. Pass --msa-max-depth=-1 to "
        "disable subsampling and match the HF torch reference behaviour.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for ESMC and the conversion step.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    msas = _load_msas(args.msa, max_sequences=args.msa_load_depth) if args.msa else {}

    if args.chains:
        proteins = _parse_chains(args.chains, msas=msas)
    elif args.seq:
        proteins = [ProteinInput(id="A", sequence=args.seq, msa=msas.get("A"))]
    else:
        proteins = [ProteinInput(id="A", sequence=UBIQUITIN, msa=msas.get("A"))]
        print("No --seq / --chains given; defaulting to ubiquitin (1UBQ).")
    if msas and "ESMFold2-Fast" in args.checkpoint:
        print(
            "WARNING: --msa provided but checkpoint is the -Fast variant, which "
            "has no MSA encoder enabled. MSAs will be ignored."
        )

    spi = StructurePredictionInput(sequences=proteins)
    print(f"\nInput: {len(proteins)} chain(s)")
    for pi in proteins:
        print(f"  chain {pi.id}: L={len(pi.sequence)}")

    # ------------------------------------------------------------------
    # 1. Featurize via the official input builder.
    # ------------------------------------------------------------------
    builder = ESMFold2InputBuilder()
    features, chain_infos = builder.prepare_input(spi)
    L = features["res_type"].shape[1]
    A = int(features["atom_attention_mask"].sum().item())
    print(f"  tokens={L}, atoms={A}")

    # ------------------------------------------------------------------
    # 2. Load the PyTorch checkpoint (we need it to seed from_torch).
    #    ESMC is loaded on CPU here so it doesn't fight JAX for GPU memory
    #    while we convert; we'll only ever run it via the JAX path below.
    # ------------------------------------------------------------------
    print(f"\nLoading {args.checkpoint} (with ESMC-6B; this allocates ~12 GB of CPU RAM)...")
    t0 = time.time()
    torch_model = (
        ESMFold2Model.from_pretrained(
            args.checkpoint,
            load_esmc=True,
            dtype=torch.float32,
            esmc_precision="bf16",
        )
        .eval()
    )
    torch_model._esmc = torch_model._esmc.to(dtype=torch.float32)
    torch_model._esmc_fp8 = False
    print(f"  loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Convert both models to JAX/Equinox.
    # ------------------------------------------------------------------
    print("Converting ESMC + ESMFold2 trunk/head to JAX...")
    t0 = time.time()
    eqx_esmc = esmjfold2.from_torch(torch_model._esmc)
    eqx_trunk = esmjfold2.from_torch(torch_model)
    print(f"  converted in {time.time() - t0:.1f}s")
    # Free the torch reference now that we have JAX copies.
    del torch_model
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Compute LM hidden states (chain BOS/EOS prep in numpy, ESMC in JAX).
    # ------------------------------------------------------------------
    print("Computing LM features (ESMC under filter_jit)...")
    t0 = time.time()
    lm_hidden = esmjfold2.compute_lm_hidden_states(
        eqx_esmc,
        features["input_ids"].cpu().numpy(),
        features["asym_id"].cpu().numpy(),
        features["residue_index"].cpu().numpy(),
        features["mol_type"].cpu().numpy(),
        features["token_attention_mask"].cpu().numpy(),
    )
    lm_hidden.block_until_ready()
    print(f"  LM features: {time.time() - t0:.1f}s ({lm_hidden.shape})")

    # ESMC weights are no longer needed; free them so the trunk has headroom.
    del eqx_esmc
    gc.collect()

    # ------------------------------------------------------------------
    # 5. JIT-compiled trunk + diffusion sampler + confidence head.
    # ------------------------------------------------------------------
    features_jax = esmjfold2.Features.from_input_builder(features)

    @eqx.filter_jit
    def predict(model, feats, lm, key):
        return model(
            feats,
            lm_hidden_states=lm,
            key=key,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            noise_scale=args.noise_scale,
            step_scale=args.step_scale,
            msa_max_depth=None if args.msa_max_depth < 0 else args.msa_max_depth,
        )

    print(
        f"Running structure prediction (loops={args.num_loops}, "
        f"sampling_steps={args.num_sampling_steps}, samples={args.num_samples})..."
    )
    n_chains_input = len(proteins)

    sample_scores = []
    all_coords = []   # one entry per sample
    all_plddt = []
    best_idx = 0
    best_score = float("-inf")
    for s in range(args.num_samples):
        t0 = time.time()
        key = jax.random.key(args.seed + s)
        out = predict(eqx_trunk, features_jax, lm_hidden, key)
        out.sample_atom_coords.block_until_ready()
        ptm = float(out.ptm[0])
        iptm = float(out.iptm[0])
        plddt = np.asarray(out.plddt[0])
        score = 0.8 * iptm + 0.2 * ptm if n_chains_input > 1 else ptm
        elapsed = time.time() - t0
        sample_scores.append(
            {"sample": s, "seed": args.seed + s, "score": score,
             "ptm": ptm, "iptm": iptm, "plddt": float(plddt.mean()),
             "time_s": elapsed}
        )
        # Keep every sample's coords + plddt — only retain numpy copies so the
        # JAX arrays can free.
        all_coords.append(np.asarray(out.sample_atom_coords[0]))
        all_plddt.append(plddt)
        tag = "compile+1st" if s == 0 else "cached"
        print(f"  sample {s} ({tag} {elapsed:.1f}s): "
              f"plddt={plddt.mean():.3f}, pTM={ptm:.3f}, "
              f"ipTM={iptm:.3f}, ranking={score:.3f}")
        if score > best_score:
            best_idx = s
            best_score = score

    plddt = all_plddt[best_idx]
    ptm = sample_scores[best_idx]["ptm"]
    iptm = sample_scores[best_idx]["iptm"]
    print(f"\nSelected sample {best_idx} (ranking_score={best_score:.3f})")

    # ------------------------------------------------------------------
    # 6. Confidence summary.
    # ------------------------------------------------------------------
    print("\n=== Confidence ===")
    print(f"  plddt mean : {plddt.mean():.3f}")
    print(f"  plddt min  : {plddt.min():.3f}")
    print(f"  plddt max  : {plddt.max():.3f}")
    print(f"  pTM        : {ptm:.3f}")
    if n_chains_input > 1:
        print(f"  ipTM       : {iptm:.3f}")

    # Per-chain plddt
    asym_id = features["asym_id"][0].cpu().numpy()
    chain_lookup = {ci.asym_id: ci.chain_id for ci in chain_infos}
    per_chain = {}
    for aid in np.unique(asym_id):
        mask = asym_id == aid
        per_chain[chain_lookup.get(int(aid), str(int(aid)))] = float(plddt[mask].mean())
    if len(per_chain) > 1:
        print("  per-chain plddt:")
        for cid, val in per_chain.items():
            print(f"    {cid}: {val:.3f}")

    # ------------------------------------------------------------------
    # 7. Write multi-model mmCIF (one MODEL per sample). Models are ordered
    # best-first by ranking score so viewers default to the highest-confidence
    # pose.
    # ------------------------------------------------------------------
    order = sorted(range(args.num_samples), key=lambda i: -sample_scores[i]["score"])
    coords_ordered = [all_coords[i] for i in order]
    plddt_ordered = [all_plddt[i] for i in order]

    out_path = Path(args.out).resolve()
    cif = esmjfold2.output_to_mmcif_multi(
        coords_per_sample=coords_ordered,
        plddt_per_sample=plddt_ordered,
        atom_mask=features_jax.atom_attention_mask[0],
        ref_element=features_jax.ref_element[0],
        ref_atom_name_chars=features_jax.ref_atom_name_chars[0],
        chain_infos=chain_infos,
        complex_id=out_path.stem,
    )
    out_path.write_text(cif)
    print(f"\nWrote {out_path}  ({args.num_samples} MODEL blocks, best first)")

    # JSON sidecar with metrics
    json_path = out_path.with_suffix(".json")
    sidecar = {
        "checkpoint": args.checkpoint,
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "num_tokens": int(L),
        "num_atoms": A,
        "plddt_mean": float(plddt.mean()),
        "ptm": ptm,
        "iptm": iptm if n_chains_input > 1 else None,
        "ranking_score": best_score,
        "selected_sample": best_idx,
        # Models in the mmCIF are written best-first; this maps
        # MODEL N (1-indexed) → original sample index used as PRNG seed offset.
        "model_to_sample": {f"MODEL {n + 1}": int(s) for n, s in enumerate(order)},
        "all_samples": sample_scores,
        "per_chain_plddt": per_chain,
    }
    json_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

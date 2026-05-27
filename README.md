# esmjfold2

JAX/Equinox translation of [ESMFold2](https://huggingface.co/biohub/ESMFold2),
the Biohub structure prediction model, in the style of
[esmj](https://github.com/escalante-bio/esmj).

## License & attribution

`esmjfold2` is licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE).
This package contains code derived from two upstream projects; full attribution is
in [`NOTICE`](NOTICE):

- The model architecture (modules + forward logic) is ported from the
  [Biohub fork of HuggingFace Transformers](https://github.com/Biohub/transformers)
  (`src/transformers/models/esmfold2`, `…/esmc`) — Apache-2.0,
  Copyright 2026 Biohub.
- The structure-output writer (`src/esmjfold2/structure_output.py`) is ported
  from [biohub/esm](https://github.com/Biohub/esm)
  (`esm/models/esmfold2/output.py`) — MIT, Copyright 2026 Chan Zuckerberg Biohub.

## Usage

### Quickstart from the CLI

```bash
# Single protein, default model is ESMFold2-Fast (no MSA encoder)
uv run python scripts/predict.py --seq MQIFVKTLTGKT...

# Multi-chain complex, full ESMFold2 with MSAs and per-loop subsampling
uv run python scripts/predict.py \
    --checkpoint biohub/ESMFold2 \
    --chains "A:peptide_seq,B:hla_seq,F:vhh_seq" \
    --msa B=B_HLA.a3m --msa F=F_VHH.a3m \
    --num-loops 20 --num-sampling-steps 50 --num-samples 16
```

Output is a multi-model mmCIF (one `MODEL` per diffusion sample, best ordered
first by ranking score), plus a JSON sidecar with per-sample pTM / ipTM /
pLDDT.

### Programmatic use

```python
import jax, torch, equinox as eqx
import esmjfold2
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from esm.utils.msa import MSA
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

# 1. Load and convert the torch reference to JAX/Equinox.
torch_model = ESMFold2Model.from_pretrained(
    "biohub/ESMFold2", load_esmc=True, dtype=torch.float32,
).eval()
torch_model._esmc = torch_model._esmc.to(dtype=torch.float32)
torch_model._esmc_fp8 = False

eqx_esmc  = esmjfold2.from_torch(torch_model._esmc)    # 6.3B-param ESMC LM
eqx_trunk = esmjfold2.from_torch(torch_model)          # ~189M trunk+head

# 2. Featurize a complex.
spi = StructurePredictionInput(sequences=[
    ProteinInput(id="A", sequence="SLLMWITQC"),                       # peptide
    ProteinInput(id="B", sequence="GSHSMRY…", msa=MSA.from_a3m("B.a3m")),  # HLA
    ProteinInput(id="F", sequence="QVQLQE…", msa=MSA.from_a3m("F.a3m")),   # VHH
])
features_dict, chain_infos = ESMFold2InputBuilder().prepare_input(spi)
features = esmjfold2.Features.from_input_builder(features_dict)        # typed input

# 3. Pre-compute ESMC features once (numpy chain prep + jitted ESMC forward).
lm_hidden = esmjfold2.compute_lm_hidden_states(
    eqx_esmc,
    features_dict["input_ids"], features_dict["asym_id"],
    features_dict["residue_index"], features_dict["mol_type"],
    features_dict["token_attention_mask"],
)

# 4. JIT-compiled trunk + diffusion sampler + confidence head.
@eqx.filter_jit
def predict(model, features, lm, key):
    return model(
        features, lm_hidden_states=lm, key=key,
        num_loops=20, num_sampling_steps=50,
        msa_max_depth=1024,   # paper's "vital" per-loop MSA subsampling
    )

out: esmjfold2.Prediction = predict(eqx_trunk, features, lm_hidden, jax.random.key(0))
print(out.plddt.mean(), float(out.ptm[0]), float(out.iptm[0]))
print(out.sample_atom_coords.shape)                                    # (1, n_atoms, 3)

# 5. Write structure as mmCIF.
cif = esmjfold2.output_to_mmcif(
    coords=out.sample_atom_coords[0],
    plddt=out.plddt_per_atom[0],
    atom_mask=features.atom_attention_mask[0],
    ref_element=features.ref_element[0],
    ref_atom_name_chars=features.ref_atom_name_chars[0],
    chain_infos=chain_infos,
    complex_id="my_target",
)
open("my_target.cif", "w").write(cif)

# Or, write all N diffusion samples as one multi-model mmCIF:
# esmjfold2.output_to_mmcif_multi(coords_per_sample=[...], plddt_per_sample=[...],
#                                  atom_mask=..., ref_element=..., chain_infos=...)
```

### Typed I/O

`ESMFold2.__call__` takes an `esmjfold2.Features` and returns an
`esmjfold2.Prediction` — both are `eqx.Module`s with jaxtyping shape annotations
on every field. The fields on `Features` are exactly the ones the model
actually reads at inference; the fields on `Prediction` are exactly what the
model produces. Both are accessed via attributes (`features.res_type`,
`out.plddt`), no dict indexing.

To build a `Features` from the raw dict that `ESMFold2InputBuilder.prepare_input`
returns: `esmjfold2.Features.from_input_builder(features_dict)`.

### Save / load without torch

```python
esmjfold2.save_model(eqx_trunk, "esmfold2")
# In a new process — no torch needed:
reloaded = esmjfold2.load_model("esmfold2")
```

## Available inference knobs

Defined on `ESMFold2.__call__`; the CLI mirrors all of them in `scripts/predict.py`:

- `num_loops` — trunk refinement iterations (each runs the whole 24- or 48-layer trunk over the pair state via Mamba-style linear-recurrent gates).
- `num_sampling_steps` — Karras diffusion solver steps.
- `noise_scale`, `step_scale` — diffusion sampler hyperparameters (λ, η).
- `msa_max_depth=1024` — **per-loop MSA subsampling**. Each refinement iteration picks `msa_max_depth` random rows from the loaded MSA (always retaining the query). Matches the paper's "vital for improved performance across recurrences" prescription, which the HF torch inference release omits. Set to `None` to opt out.

`scripts/predict.py` adds `--num-samples N` (draw N independent diffusion
samples, write all to a multi-model mmCIF, rank by ipTM/pTM) and `--msa
CHAIN=path.a3m` (repeatable, attaches MSAs to chains).

## Notes on numerical parity

- The torch reference runs the trunk under `torch.amp.autocast(bfloat16)`. The JAX path is fp32. Outputs differ by a few percent on a numeric level.
- The SWA atom attention uses `flash_attn_varlen` + a sliding window mask at training time. The JAX `SWAAtomTransformer.use_swa_window=True` default reproduces the trained-with-window behavior. Setting it to `False` recovers the torch `no-flash_attn` fallback (dense attention, no window/padding mask), only useful for parity testing — diverges from training.
- The pair state is initialized with random truncated-normal noise; different RNG streams between JAX and torch produce different initial states and thus different sampler trajectories. Expected, not a bug.
- Same-key JIT inference is bit-exact on CPU. On GPU there can be ~ULP differences run-to-run from cuBLAS / cuDNN non-deterministic reductions.
- We discovered and fixed several upstream indexing bugs during this port: `rel_pos` and `SingleToPair` had a flipped `i-j` vs `j-i` broadcast, and the Karras noise-schedule cap was clipping in place instead of filtering + prepending. The JAX path produces correct outputs; see `NOTICE` for the full list.

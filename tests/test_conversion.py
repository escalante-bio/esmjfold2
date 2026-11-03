"""Native Biohub model loading, conversion, and CPU numerical parity."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import torch

from esm.models.esmc import EsmcConfig, EsmcForMaskedLM, EsmcModel
from esm.models.esmfold2 import EsmFold2Config, EsmFold2ExperimentalModel, EsmFold2Model
from esm.models.esmfold2.protein_utils import prepare_protein_features
from safetensors.torch import save_file

import esmjfold2


@pytest.fixture
def config():
    return EsmFold2Config(
        hidden_size=32,
        pairwise_hidden_size=16,
        num_loops=1,
        num_diffusion_samples=1,
        lm_d_model=32,
        lm_num_layers=2,
        folding_trunk_num_hidden_layers=1,
        folding_trunk_num_attention_heads=2,
        atom_encoder={"hidden_size": 64, "num_attention_heads": 2, "num_hidden_layers": 1},
        structure_head={
            "inference_num_steps": 2,
            "distogram_bins": 8,
            "diffusion_module": {
                "atom_encoder": {"hidden_size": 64, "num_attention_heads": 2, "num_hidden_layers": 1},
                "token_hidden_size": 32,
                "c_z": 16,
                "fourier_dim": 16,
                "token_num_blocks": 1,
                "token_num_heads": 2,
            },
        },
        confidence_head={
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "distogram_bins": 8,
            "num_plddt_bins": 8,
            "num_pae_bins": 8,
            "num_pde_bins": 8,
        },
        parcae_num_coda_layers=1,
        lm_encoder={"num_hidden_layers": 1},
        msa_encoder={"enabled": True, "hidden_size": 32, "outer_hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2, "head_width": 8},
    )


@pytest.mark.parametrize("model_class", [EsmcModel, EsmcForMaskedLM])
def test_esmc_parity(model_class):
    torch.manual_seed(4)
    model = model_class(EsmcConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2)).eval()
    converted = esmjfold2.from_torch(model)
    ids = np.array([[0, 5, 6, 7, 2]], dtype=np.int32)
    with torch.no_grad():
        expected = model(torch.from_numpy(ids), output_hidden_states=True)
    output, hidden = eqx.filter_jit(converted)(jnp.asarray(ids), collect_hidden_states=True)
    reference = expected.logits if model_class is EsmcForMaskedLM else expected.last_hidden_state
    np.testing.assert_allclose(output, reference.numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(hidden, expected.hidden_states.numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("model_class", [EsmFold2Model, EsmFold2ExperimentalModel])
def test_checkpoint_conversion_and_prediction(config, model_class, tmp_path):
    """Exercise the real loader, all converters, JIT inference and serialization."""
    torch.manual_seed(0)
    config.type = "experimental" if model_class is EsmFold2ExperimentalModel else "release"
    config.esmc_config = EsmcConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2)
    config.save_pretrained(tmp_path)
    save_file(model_class(config).eval().state_dict(), tmp_path / "model.safetensors")
    model = EsmFold2Model.from_pretrained(tmp_path, device="cpu", dtype=torch.float32, esmc_precision="fp32")
    assert isinstance(model, model_class)
    converted = esmjfold2.from_torch(model)
    esmc = esmjfold2.from_torch(model.esmc)
    raw = prepare_protein_features("GAG")
    features = esmjfold2.Features.from_input_builder(raw)
    lm = esmjfold2.compute_lm_hidden_states(
        esmc, raw["input_ids"], raw["asym_id"], raw["residue_index"],
        raw["mol_type"], raw["token_attention_mask"],
    )
    output = eqx.filter_jit(converted)(features, lm_hidden_states=lm, key=jax.random.key(0), num_loops=1, num_sampling_steps=2)
    assert output.sample_atom_coords.shape[-2:] == (raw["ref_pos"].shape[1], 3)
    for leaf in jax.tree.leaves(output):
        assert np.isfinite(np.asarray(leaf)).all()
    esmjfold2.save_model(converted, tmp_path / "converted")
    restored = esmjfold2.load_model(tmp_path / "converted")
    assert eqx.tree_equal(converted, restored)


def test_folding_components_parity(config):
    """Compare deterministic modules with identical inputs, avoiding RNG differences."""
    torch.manual_seed(7)
    model = EsmFold2Model(config).eval()
    model.set_kernel_backend(None)
    converted = esmjfold2.from_torch(model)
    raw = prepare_protein_features("GAG")
    features = esmjfold2.Features.from_input_builder(raw)
    ctx = converted._prepare_embeddings(features)

    def tensor(x):
        value = torch.from_numpy(np.array(x))
        return value.long() if value.dtype == torch.int32 else value

    def compare(jax_module, torch_module, **kwargs):
        with torch.no_grad():
            expected = torch_module(**{k: tensor(v) for k, v in kwargs.items()})
        actual = eqx.filter_jit(jax_module)(**kwargs)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)

    compare(converted.rel_pos, model.rel_pos, **{
        k: getattr(features, k)
        for k in ("residue_index", "asym_id", "sym_id", "entity_id", "token_index")
    })
    z = jax.random.normal(jax.random.key(1), (1, 3, 3, config.pairwise_hidden_size))
    compare(converted.folding_trunk, model.folding_trunk, pair=z, pair_attention_mask=ctx.pair_mask)

    x = jax.random.normal(jax.random.key(2), features.ref_pos.shape)
    args = dict(
        x_noisy=x, t_hat=jnp.array([1.5]), ref_pos=features.ref_pos,
        ref_charge=features.ref_charge, ref_mask=features.atom_attention_mask,
        ref_space_uid=features.ref_space_uid, tok_idx=ctx.atom_to_token,
        s_inputs=ctx.x_inputs, z_trunk=z,
        relative_position_encoding=ctx.relative_position_encoding,
        token_attention_mask=features.token_attention_mask,
    )
    torch_args = {k: tensor(v) for k, v in args.items()}
    torch_args.update(
        ref_element=tensor(ctx.ref_element_oh),
        ref_atom_name_chars=tensor(ctx.ref_atom_name_chars_oh),
        s_trunk=None,
        **{k: raw[k] for k in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")},
    )
    with torch.no_grad():
        expected = model.structure_head.diffusion_module(**torch_args)["x_denoised"]
    diffusion_module = converted.structure_head.diffusion_module
    actual = eqx.filter_jit(diffusion_module)(
        **args, ref_element_oh=ctx.ref_element_oh,
        ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh, n_tokens=3,
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)

    x_noisy, t_hat = args.pop("x_noisy"), args.pop("t_hat")
    cache = eqx.filter_jit(diffusion_module.prepare)(
        **args, ref_element_oh=ctx.ref_element_oh,
        ref_atom_name_chars_oh=ctx.ref_atom_name_chars_oh, n_tokens=3,
    )
    actual = eqx.filter_jit(diffusion_module.denoise)(cache, x_noisy, t_hat)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-5, atol=3e-5)


def _features_with_msa(sequence, depth, key):
    features = esmjfold2.Features.from_input_builder(prepare_protein_features(sequence))
    B, L = features.res_type.shape[:2]
    k_msa, k_del, k_val = jax.random.split(key, 3)
    msa = jax.random.randint(k_msa, (B, depth, L), 0, 20)
    msa = msa.at[:, 0].set(features.res_type.astype(jnp.int32))
    has_deletion = jax.random.bernoulli(k_del, 0.2, (B, depth, L))
    deletion_value = jax.random.uniform(k_val, (B, depth, L)) * has_deletion
    return dataclasses.replace(
        features,
        msa=msa,
        msa_attention_mask=jnp.ones((B, depth, L), dtype=bool),
        has_deletion=has_deletion,
        deletion_value=deletion_value,
        deletion_mean=deletion_value.mean(1),
    )


@pytest.mark.parametrize("model_class", [EsmFold2Model, EsmFold2ExperimentalModel])
@pytest.mark.parametrize("with_msa", [False, True])
@pytest.mark.parametrize("num_passes", [1, 3])
def test_single_pass_trunk_api_matches_run_trunk(config, model_class, with_msa, num_passes):
    """embed → init_state → step × k → readout reproduces _run_trunk(num_loops=k-1)."""
    torch.manual_seed(0)
    model = esmjfold2.from_torch(model_class(config).eval())
    experimental = model_class is EsmFold2ExperimentalModel
    if experimental:
        # Exercise the one-time LM dropout, and perturb the zero-initialised
        # loop projection so successive passes actually interact.
        model = dataclasses.replace(model, lm_dropout=0.25)
        proj = jax.tree.map(
            lambda x: x + 0.3 * jax.random.normal(jax.random.key(9), x.shape) if eqx.is_array(x) else x,
            model.pair_loop_proj,
        )
        model = eqx.tree_at(lambda m: m.pair_loop_proj, model, proj)
    else:
        assert model.per_loop_lm_dropout and model.lm_dropout > 0.0

    if with_msa:
        features = _features_with_msa("GAGK", 5, jax.random.key(11))
        msa_max_depth = 3  # shallower than the MSA → per-pass row subsampling
    else:
        features = esmjfold2.Features.from_input_builder(prepare_protein_features("GAGK"))
        msa_max_depth = None
    lm_hidden = jax.random.normal(jax.random.key(3), (1, 4, config.lm_num_layers + 1, config.lm_d_model))
    key = jax.random.key(5)

    @eqx.filter_jit
    def run_trunk(model, features, lm_hidden, key):
        ctx = model._prepare_embeddings(features)
        lm_z = model.language_model(lm_hidden)
        return model._run_trunk(ctx, lm_z, key, num_loops=num_passes - 1, msa_max_depth=msa_max_depth)

    embed = eqx.filter_jit(lambda m, f, h, k: m.embed(f, m.language_model(h), key=k))
    init_state = eqx.filter_jit(lambda m, inputs, k: m.init_state(inputs, key=k))
    step = eqx.filter_jit(lambda m, inputs, z, k: m.step(inputs, z, key=k, msa_max_depth=msa_max_depth))
    readout = eqx.filter_jit(lambda m, inputs, z: m.readout(inputs, z))

    if experimental:
        k_embed, k_init, k_steps = jax.random.split(key, 3)
    else:
        # ESMFold2.embed draws no randomness; any key works.
        (k_steps, k_init), k_embed = jax.random.split(key), jax.random.key(1)
    inputs = embed(model, features, lm_hidden, k_embed)
    assert isinstance(inputs, esmjfold2.TrunkInputs)
    z = init_state(model, inputs, k_init)
    for i in range(num_passes):
        z = step(model, inputs, z, jax.random.fold_in(k_steps, i))
    actual = readout(model, inputs, z)

    expected = run_trunk(model, features, lm_hidden, key)
    assert np.isfinite(np.asarray(expected)).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

"""Save/load JAX models without torch."""

import pickle

import equinox as eqx
import jax


def save_model(model, path: str) -> None:
    eqx.tree_serialise_leaves(f"{path}.eqx", model)
    skeleton = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if eqx.is_array(x) else x,
        model,
        is_leaf=eqx.is_array,
    )
    with open(f"{path}.skeleton.pkl", "wb") as f:
        pickle.dump(skeleton, f)


def load_model(path: str):
    with open(f"{path}.skeleton.pkl", "rb") as f:
        skeleton = pickle.load(f)
    return eqx.tree_deserialise_leaves(f"{path}.eqx", skeleton)

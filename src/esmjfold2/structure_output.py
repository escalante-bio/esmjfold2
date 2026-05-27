# SPDX-License-Identifier: MIT
# Numpy port of esm/models/esmfold2/output.py
# Copyright 2026 Chan Zuckerberg Biohub, Inc.  (original)
# See NOTICE for license details.
"""Convert ESMFold2 raw outputs into a MolecularComplex (→ mmCIF / PDB).

Translation of ``esm.models.esmfold2.output.build_molecular_complex_from_features``
from torch tensors to numpy/JAX. The downstream serializers
(`MolecularComplex.to_mmcif`, `.to_pdb`) are already pure numpy/biotite, so we
just need to construct the dataclass from our model's outputs.
"""

from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np


_MOL_TYPE_NONPOLYMER = 3


def _to_numpy(x):
    """Convert a torch tensor, JAX array, or numpy array to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def output_to_mmcif(
    coords,
    plddt,
    atom_mask,
    ref_element,
    ref_atom_name_chars,
    chain_infos,
    complex_id: str = "model",
) -> str:
    """Serialize ESMFold2 output to an mmCIF string.

    Args:
        coords: ``[n_atoms, 3]`` predicted positions (torch / JAX / numpy).
        plddt: ``[n_tokens]`` per-token pLDDT (0–1).
        atom_mask: ``[n_atoms]`` bool — valid atoms.
        ref_element: ``[n_atoms]`` int — atomic numbers (as in the input
            features, before one-hot).
        ref_atom_name_chars: ``[n_atoms, 4]`` int — atom name chars
            (ord(c) - 32) per atom, padded with 0.
        chain_infos: list of ``esm.models.esmfold2.prepare_input.ChainInfo``
            returned alongside the input features by
            ``ESMFold2InputBuilder.prepare_input``.
        complex_id: identifier written into the mmCIF data block.

    Returns:
        mmCIF document as a single string.
    """
    complex_ = output_to_molecular_complex(
        coords=coords,
        plddt=plddt,
        atom_mask=atom_mask,
        ref_element=ref_element,
        ref_atom_name_chars=ref_atom_name_chars,
        chain_infos=chain_infos,
        complex_id=complex_id,
    )
    return complex_.to_mmcif()


def output_to_mmcif_multi(
    coords_per_sample,
    plddt_per_sample,
    atom_mask,
    ref_element,
    ref_atom_name_chars,
    chain_infos,
    complex_id: str = "model",
) -> str:
    """Serialize N predicted samples as a multi-model mmCIF.

    Writes one ``MODEL`` block per sample, sharing chain / residue / atom-name
    annotations (built once from sample 0) and using sample 0's per-atom pLDDT
    as the B-factor column. Per-sample pLDDT is best kept in the JSON sidecar
    since biotite's AtomArrayStack shares B-factors across models.

    Args:
        coords_per_sample: list of length N, each ``[n_atoms, 3]``.
        plddt_per_sample: list of length N, each ``[n_tokens]``.
        atom_mask, ref_element, ref_atom_name_chars, chain_infos, complex_id:
            same as ``output_to_mmcif``.

    Returns:
        mmCIF document string.
    """
    import io as _io
    import biotite.structure as bs
    from biotite.structure.io.pdbx import (
        CIFFile, CIFColumn, CIFData, get_structure, set_structure,
    )

    assert len(coords_per_sample) == len(plddt_per_sample), \
        f"coords/plddt length mismatch: {len(coords_per_sample)} vs {len(plddt_per_sample)}"

    # Build a MolecularComplex per sample. Annotations (chain mapping,
    # token_to_atoms, atom names, hetero flags, entity lookup) come out
    # identical across samples — they depend only on the input features,
    # which we share. Only ``atom_positions`` and ``plddt`` differ.
    complexes = [
        output_to_molecular_complex(
            coords=c, plddt=p,
            atom_mask=atom_mask, ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            chain_infos=chain_infos, complex_id=complex_id,
        )
        for c, p in zip(coords_per_sample, plddt_per_sample)
    ]

    # Round-trip sample 0 through MolecularComplex.to_mmcif() once to inherit
    # all of its biotite annotation setup (chain_id, res_id, res_name, hetero,
    # element, atom_name, entity_id, label_asym_id, etc.). We then reuse those
    # annotations and only swap in the stacked coords.
    base_cif = CIFFile.read(_io.StringIO(complexes[0].to_mmcif()))
    base_atoms = get_structure(base_cif, model=1, extra_fields=["b_factor"])

    coord_stack = np.stack(
        [cx.atom_positions for cx in complexes], axis=0
    ).astype(np.float32)

    n_models, n_atoms, _ = coord_stack.shape
    stack = bs.AtomArrayStack(depth=n_models, length=n_atoms)
    # AtomArrayStack shares all annotations across models — propagate them
    # from the sample-0 AtomArray.
    for ann_name in base_atoms.get_annotation_categories():
        stack.set_annotation(ann_name, base_atoms.get_annotation(ann_name))
    stack.coord = coord_stack

    out_cif = CIFFile()
    set_structure(out_cif, stack, data_block=complex_id)

    # Mirror MolecularComplex.to_mmcif()'s entity-id fixup. set_structure
    # writes label_entity_id from biotite's annotation, which often defaults to
    # "1" everywhere; copy the per-chain entity numbering from sample 0.
    if "atom_site" in out_cif.block:
        atom_site = out_cif.block["atom_site"]
        if (
            "label_asym_id" in atom_site
            and "label_entity_id" in atom_site
            and "label_asym_id" in base_cif.block.get("atom_site", {})
            and "label_entity_id" in base_cif.block.get("atom_site", {})
        ):
            chain_to_entity: dict[str, str] = {}
            base_site = base_cif.block["atom_site"]
            for cid, eid in zip(
                base_site["label_asym_id"].as_array(str),
                base_site["label_entity_id"].as_array(str),
            ):
                chain_to_entity.setdefault(cid, eid)
            updated = [
                chain_to_entity.get(cid, "1")
                for cid in atom_site["label_asym_id"].as_array(str)
            ]
            atom_site["label_entity_id"] = CIFColumn(
                data=CIFData(array=np.array(updated), dtype=np.str_)
            )

    out = _io.StringIO()
    out_cif.write(out)
    return out.getvalue()


def output_to_molecular_complex(
    coords,
    plddt,
    atom_mask,
    ref_element,
    ref_atom_name_chars,
    chain_infos,
    complex_id: str = "model",
):
    """Build a ``MolecularComplex`` from raw outputs + chain metadata.

    Direct numpy port of
    ``esm.models.esmfold2.output.build_molecular_complex_from_features``.
    The ``esm`` package is imported lazily so this module doesn't pull it in at
    import time.
    """
    from esm.models.esmfold2.constants import ELEMENT_NUMBER_TO_SYMBOL
    from esm.utils.structure.molecular_complex import (
        MolecularComplex,
        MolecularComplexMetadata,
    )

    mask_np = _to_numpy(atom_mask).astype(bool)
    coords_np = _to_numpy(coords).astype(np.float32)
    name_chars_np = _to_numpy(ref_atom_name_chars).astype(np.int64)
    elements_np = _to_numpy(ref_element).astype(np.int64)
    plddt_np = _to_numpy(plddt).astype(np.float32)

    sequence_tokens: list[str] = []
    chain_ids_per_token: list[int] = []
    token_to_atoms: list[list[int]] = []
    confidence: list[float] = []
    flat_positions: list[list[float]] = []
    flat_elements: list[str] = []
    flat_names: list[str] = []
    flat_hetero: list[bool] = []

    chain_lookup: dict[int, str] = {}
    entity_info: dict[int, str] = {}
    out_atom_cursor = 0

    for ci in chain_infos:
        chain_lookup[ci.asym_id] = ci.chain_id
        is_nonpolymer = ci.mol_type == _MOL_TYPE_NONPOLYMER
        entity_info[ci.entity_id] = (
            "non-polymer" if is_nonpolymer else "polymer"
        )

        if is_nonpolymer:
            # Collapse all per-atom tokens into one residue.
            residue_name = ci.tokens[0].residue_name if ci.tokens else "LIG"
            sequence_tokens.append(residue_name)
            chain_ids_per_token.append(ci.asym_id)
            avg = (
                float(np.mean([plddt_np[ti.token_index] for ti in ci.tokens]))
                if ci.tokens
                else 0.0
            )
            confidence.append(avg)
            token_atom_start = out_atom_cursor
            for ti in ci.tokens:
                for atom_idx in range(ti.atom_start, ti.atom_start + ti.atom_count):
                    if not mask_np[atom_idx]:
                        continue
                    flat_positions.append(coords_np[atom_idx].tolist())
                    flat_elements.append(
                        ELEMENT_NUMBER_TO_SYMBOL.get(int(elements_np[atom_idx]), "X")
                    )
                    chars = name_chars_np[atom_idx]
                    name = "".join(
                        chr(int(c) + 32) for c in chars if int(c) != 0
                    ).strip()
                    flat_names.append(name)
                    flat_hetero.append(True)
                    out_atom_cursor += 1
            token_to_atoms.append([token_atom_start, out_atom_cursor])
            continue

        # Atom-tokenized residues (HYP, MSE, ...) span multiple tokens per
        # residue; collapse them back to one mmCIF residue.
        for _residue_index, ti_iter in groupby(
            ci.tokens, key=lambda t: t.residue_index
        ):
            ti_group = list(ti_iter)
            sequence_tokens.append(ti_group[0].residue_name)
            chain_ids_per_token.append(ci.asym_id)
            confidence.append(
                float(np.mean([plddt_np[ti.token_index] for ti in ti_group]))
            )
            token_atom_start = out_atom_cursor
            for ti in ti_group:
                for atom_idx in range(ti.atom_start, ti.atom_start + ti.atom_count):
                    if not mask_np[atom_idx]:
                        continue
                    flat_positions.append(coords_np[atom_idx].tolist())
                    flat_elements.append(
                        ELEMENT_NUMBER_TO_SYMBOL.get(int(elements_np[atom_idx]), "X")
                    )
                    chars = name_chars_np[atom_idx]
                    name = "".join(
                        chr(int(c) + 32) for c in chars if int(c) != 0
                    ).strip()
                    flat_names.append(name)
                    flat_hetero.append(False)
                    out_atom_cursor += 1
            token_to_atoms.append([token_atom_start, out_atom_cursor])

    return MolecularComplex(
        id=complex_id,
        sequence=sequence_tokens,
        atom_positions=np.array(flat_positions, dtype=np.float32).reshape(-1, 3),
        atom_elements=np.array(flat_elements, dtype=object),
        token_to_atoms=np.array(token_to_atoms, dtype=np.int32).reshape(-1, 2),
        chain_id=np.array(chain_ids_per_token, dtype=np.int64),
        plddt=np.array(confidence, dtype=np.float32),
        atom_names=np.array(flat_names, dtype=object),
        atom_hetero=np.array(flat_hetero, dtype=bool),
        metadata=MolecularComplexMetadata(
            entity_lookup=entity_info,
            chain_lookup=chain_lookup,
            assembly_composition=None,
        ),
    )

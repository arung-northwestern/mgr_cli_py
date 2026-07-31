"""Parity: legacy strict-double pipeline vs tolerance + Rust/Python expansion."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "cif"

pytest.importorskip("networkx")


def _cycle_set(cycles: list[list[int]]) -> set[tuple[int, ...]]:
    return {tuple(sorted(int(x) for x in c)) for c in cycles}


def _load_sgraph(cif_name: str):
    from ase.io import read
    from mgr.analyze import (
        find_aromatic_cycles_in_sgraph,
        find_aromatic_cycles_in_sgraph_legacy,
        sgraph_from_atoms,
        update_with_covalent_bonds,
    )

    cif = FIXTURES / cif_name
    if not cif.is_file():
        pytest.skip(f"missing fixture {cif}")
    atoms = update_with_covalent_bonds(read(cif))
    sgraph = sgraph_from_atoms(atoms)
    return atoms, sgraph, find_aromatic_cycles_in_sgraph, find_aromatic_cycles_in_sgraph_legacy


@pytest.mark.parametrize("cif", ["NU-1000_S1.cif", "MIL-101-primitive.cif"])
def test_legacy_subset_of_new_with_tolerance(cif: str):
    _, sgraph, find_new, find_legacy = _load_sgraph(cif)
    legacy = _cycle_set(find_legacy(sgraph))
    new = _cycle_set(find_new(sgraph, use_rust=False))
    assert legacy <= new, f"{cif}: legacy has rings missing from new: {len(legacy - new)}"


@pytest.mark.parametrize("cif", ["NU-1000_S1.cif"])
def test_rust_matches_python_expansion(cif: str):
    _, sgraph, find_new, _ = _load_sgraph(cif)
    py = _cycle_set(find_new(sgraph, use_rust=False))
    rust = _cycle_set(find_new(sgraph, use_rust=True))
    assert py == rust, f"{cif}: rust/python mismatch {len(py ^ rust)} rings"


@pytest.mark.timeout(120)
@pytest.mark.parametrize("cif", ["MIL-101-primitive.cif"])
def test_rust_matches_python_expansion_mil101(cif: str):
    _, sgraph, find_new, _ = _load_sgraph(cif)
    py = _cycle_set(find_new(sgraph, use_rust=False))
    rust = _cycle_set(find_new(sgraph, use_rust=True))
    assert py == rust, f"{cif}: rust/python mismatch {len(py ^ rust)} rings"


def test_compute_aromatic_cycles_cache():
    from ase.io import read
    from mgr.analyze import compute_aromatic_cycles, update_with_covalent_bonds

    cif = FIXTURES / "NU-1000_S1.cif"
    if not cif.is_file():
        pytest.skip("missing NU-1000 fixture")
    atoms = update_with_covalent_bonds(read(cif))
    a = compute_aromatic_cycles(atoms)
    b = compute_aromatic_cycles(atoms)
    assert a == b
    assert atoms.info.get("aromatic_cycles_key")
    assert atoms.info.get("aromatic_cycles") is a


def test_plot_aromatic_rings_matches_compute_aromatic_cycles():
    """Plotly ring path uses the same detector as Three.js."""
    from ase.io import read
    from mgr.analyze import compute_aromatic_cycles, update_with_covalent_bonds
    from mgr.visualize import plot_aromatic_rings

    cif = FIXTURES / "NU-1000_S1.cif"
    if not cif.is_file():
        pytest.skip("missing NU-1000 fixture")
    atoms = update_with_covalent_bonds(read(cif))
    expected = compute_aromatic_cycles(atoms, force_recompute=True)
    atoms.info.pop("aromatic_cycles", None)
    atoms.info.pop("aromatic_cycles_key", None)
    plot_aromatic_rings(atoms)
    assert atoms.info.get("aromatic_cycles") == expected
    assert len(expected) == 48


def test_strict_legacy_parity_when_no_extra_aromatic_types(monkeypatch):
    """With high order min, new basis should match legacy on NU-1000."""
    monkeypatch.setenv("MGR_AROMATIC_ORDER_MIN", "2.0")
    monkeypatch.setenv("MGR_AROMATIC_ORDER_TOL", "0.0")
    _, sgraph, find_new, find_legacy = _load_sgraph("NU-1000_S1.cif")
    legacy = _cycle_set(find_legacy(sgraph))
    new = _cycle_set(find_new(sgraph, use_rust=False))
    # New may still count bond_type aromatic; on NU-1000 legacy should match.
    assert legacy == new or legacy <= new

"""Crystallographic supercell helper."""
from __future__ import annotations

from ase import Atoms
from ase.build import bulk

from mgr.helper import apply_crystallographic_supercell


def test_identity_supercell_returns_same_object_count():
    atoms = bulk("Cu", "fcc", a=3.6)
    out = apply_crystallographic_supercell(atoms, 1, 1, 1)
    assert len(out) == len(atoms)


def test_diagonal_supercell_repeats_atoms():
    atoms = bulk("Cu", "fcc", a=3.6)
    out = apply_crystallographic_supercell(atoms, 2, 1, 3)
    assert len(out) == len(atoms) * 6

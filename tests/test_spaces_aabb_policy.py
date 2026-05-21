"""Hosted-demo auto-AABB policy (memory vs work)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python_cli"))

from mgrcli.interactive_app import (  # noqa: E402
    SPACES_DGRID_OPS_AUTO_AABB,
    spaces_should_auto_aabb_dgrid,
)


def _cubic_atoms(n_atoms: int, length_ang: float) -> Atoms:
    a = float(length_ang)
    return Atoms(
        symbols=["C"] * n_atoms,
        positions=np.zeros((n_atoms, 3)),
        cell=[(a, 0, 0), (0, a, 0), (0, 0, a)],
        pbc=True,
    )


def test_zif20_like_stays_on_rust_at_demo_spacing():
    """ZIF-20 (~4800 atoms, ~45 Å cell): Rust is fine; old formula falsely forced AABB."""
    atoms = _cubic_atoms(4800, 45.4725)
    for spacing in (0.5, 0.4):
        use_aabb, meta = spaces_should_auto_aabb_dgrid(atoms, spacing)
        assert not use_aabb, meta
        assert meta["ops"] < SPACES_DGRID_OPS_AUTO_AABB


def test_mil101_like_uses_aabb_at_fine_spacing():
    """Large grid × many atoms: prefer freud AABB on hosted demo."""
    atoms = _cubic_atoms(3604, 88.0)
    use_aabb, meta = spaces_should_auto_aabb_dgrid(atoms, 0.4)
    assert use_aabb
    assert meta.get("reason") == "work"

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


def test_zif20_like_uses_aabb_on_hosted_demo():
    """Hosted demo: ZIF-20 (~4800 atoms) is over the 1e9 work threshold, so AABB (faster)."""
    atoms = _cubic_atoms(4800, 45.4725)
    for spacing in (0.5, 0.4):
        use_aabb, meta = spaces_should_auto_aabb_dgrid(atoms, spacing)
        assert use_aabb and meta["reason"] == "work", meta
        assert meta["ops"] > SPACES_DGRID_OPS_AUTO_AABB
        assert meta["estimated_bytes"] < meta["budget_bytes"]


def test_mil101_like_uses_aabb_at_fine_spacing():
    """Large grid × many atoms: prefer freud AABB on hosted demo."""
    atoms = _cubic_atoms(3604, 88.0)
    use_aabb, meta = spaces_should_auto_aabb_dgrid(atoms, 0.4)
    assert use_aabb
    assert meta.get("reason") == "work"

def test_local_budget_constant_is_16_gib():
    from mgrcli.interactive_app import LOCAL_DGRID_RAM_BUDGET_BYTES

    assert LOCAL_DGRID_RAM_BUDGET_BYTES == int(16 * 1024**3)


def test_zif20_like_stays_on_rust_with_local_budget():
    from mgrcli.interactive_app import (
        LOCAL_DGRID_OPS_AUTO_AABB,
        LOCAL_DGRID_RAM_BUDGET_BYTES,
        spaces_should_auto_aabb_dgrid,
    )

    atoms = _cubic_atoms(4800, 45.0)
    for spacing in (0.5, 0.4):
        use_aabb, meta = spaces_should_auto_aabb_dgrid(
            atoms,
            spacing,
            ram_budget_bytes=LOCAL_DGRID_RAM_BUDGET_BYTES,
            ops_threshold=LOCAL_DGRID_OPS_AUTO_AABB,
        )
        assert not use_aabb, meta
        assert meta["budget_bytes"] == LOCAL_DGRID_RAM_BUDGET_BYTES



def test_cli_policy_has_no_work_threshold_by_default(monkeypatch):
    """CLI/local: exact dgrid for ZIF-20-like cases; env var opts in to a work limit."""
    from mgrcli.dgrid_policy import resolve_dgrid_method

    monkeypatch.delenv("MGR_DGRID_OPS_AUTO_AABB", raising=False)
    atoms = _cubic_atoms(4800, 45.4725)
    method, meta = resolve_dgrid_method(atoms, 0.4, "rust_exact")
    assert method == "rust_exact", meta
    monkeypatch.setenv("MGR_DGRID_OPS_AUTO_AABB", "1e9")
    method, meta = resolve_dgrid_method(atoms, 0.4, "rust_exact")
    assert method == "rust_aabb" and meta["reason"] == "work", meta

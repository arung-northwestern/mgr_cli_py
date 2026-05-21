"""Phase 0 covalent-bond profiling smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))
sys.path.insert(0, str(REPO / "scripts"))

import pytest

try:
    import mgrcli_rust
except ImportError:
    mgrcli_rust = None


@pytest.mark.skipif(mgrcli_rust is None, reason="mgrcli_rust not built")
def test_bonds_phase0_table_runs():
    from profile_covalent_bonds_phase0 import run_timing_table

    table = run_timing_table(repeats=2)
    assert "Covalent bonds Phase 0 profile" in table
    assert "ZIF-8" in table
    assert "marshal" in table


@pytest.mark.skipif(mgrcli_rust is None, reason="mgrcli_rust not built")
def test_bonds_phase0_bucket_counts_positive():
    from mgr.cif_io import read_cif_to_ase
    from profile_covalent_bonds_phase0 import _profile_buckets_once

    atoms = read_cif_to_ase(REPO / "tests/fixtures/cif/NU-1000_S1.cif")
    s = _profile_buckets_once(atoms)
    assert s.natoms > 0 and s.npairs > 0 and s.nbonds > 0 and s.e2e_s > 0

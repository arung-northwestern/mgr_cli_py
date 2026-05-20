"""Rust CIF parser parity vs ASE for Spaces upload fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read

try:
    import mgrcli_rust
except ImportError:
    mgrcli_rust = None  # type: ignore[assignment]

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "cif"

CASES = [
    ("ZIF-8-RASPA.cif", "ZIF-8"),
    ("Cu-BTC.cif", "CuBTC"),
    ("NU-1000_S1.cif", "NU-1000"),
    ("MIL-101-primitive.cif", "MIL-101-primitive"),
]

CELL_ATOL = 1e-6
POS_ATOL = 1e-4


@pytest.mark.skipif(mgrcli_rust is None, reason="mgrcli_rust not built")
@pytest.mark.parametrize("filename,label", CASES)
def test_read_cif_matches_ase(filename: str, label: str) -> None:
    path = FIXTURE_DIR / filename
    assert path.is_file(), f"missing fixture {path}"

    ase_atoms = ase_read(path)
    symbols, positions, cell, pbc = mgrcli_rust.read_cif(str(path))

    from ase import Atoms

    rust_atoms = Atoms(
        symbols=list(symbols),
        positions=positions,
        cell=cell,
        pbc=list(pbc),
    )

    assert list(ase_atoms.pbc) == list(rust_atoms.pbc), label
    assert np.allclose(ase_atoms.cell.array, rust_atoms.cell.array, atol=CELL_ATOL), label
    assert ase_atoms.get_chemical_symbols() == rust_atoms.get_chemical_symbols(), label

    pos_diff = np.linalg.norm(ase_atoms.get_positions() - rust_atoms.get_positions(), axis=1)
    assert float(pos_diff.max()) < POS_ATOL, f"{label}: max pos diff {pos_diff.max()}"


@pytest.mark.skipif(mgrcli_rust is None, reason="mgrcli_rust not built")
def test_cif_io_fast_path_matches_ase(monkeypatch: pytest.MonkeyPatch) -> None:
    from mgr.cif_io import read_cif_to_ase

    monkeypatch.setenv("MGR_FAST_CIF", "1")

    for filename, _ in CASES:
        path = FIXTURE_DIR / filename
        ase_atoms = ase_read(path)
        fast_atoms = read_cif_to_ase(path)
        assert np.allclose(ase_atoms.cell.array, fast_atoms.cell.array, atol=CELL_ATOL)
        assert ase_atoms.get_chemical_symbols() == fast_atoms.get_chemical_symbols()
        pos_diff = np.linalg.norm(ase_atoms.get_positions() - fast_atoms.get_positions(), axis=1)
        assert float(pos_diff.max()) < POS_ATOL

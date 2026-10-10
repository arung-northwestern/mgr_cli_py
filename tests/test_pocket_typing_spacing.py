"""Pocket typing must use segmentation grid spacing, not DEFAULT_SPACING_A."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))
sys.path.insert(0, str(REPO / "python_cli"))

from mgr.pocket_typing import compute_typing_dgrid, infer_segmentation_spacing


def _cubic_cell_atoms(length_a: float = 20.0) -> Atoms:
    return Atoms("C", positions=[[length_a / 2, length_a / 2, length_a / 2]], cell=[length_a] * 3, pbc=True)


def test_infer_segmentation_spacing_from_dgrid_shape():
    atoms = _cubic_cell_atoms(20.0)
    seg = {"dgrid": np.zeros((41, 41, 41), dtype=np.float32)}
    spacing = infer_segmentation_spacing(seg, atoms)
    assert abs(spacing - 0.5) < 0.02


def test_compute_typing_dgrid_matches_segmentation_shape_from_input_params():
    atoms = _cubic_cell_atoms(20.0)
    seg = {
        "dgrid": np.zeros((41, 41, 41), dtype=np.float32),
        "input_params": {"spacing": 0.5, "mask_thickness": 0.05},
    }
    typing = compute_typing_dgrid(atoms, seg, typing_radii="bondi_crc_fallback")
    assert typing is not None
    assert typing.shape == seg["dgrid"].shape

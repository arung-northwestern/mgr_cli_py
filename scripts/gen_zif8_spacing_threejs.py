#!/usr/bin/env python3
"""Generate ZIF-8 Three.js HTML for window/edge filter testing."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))

from ase.io import read

from mgr import dgrid_from_atoms_rust, do_periodic_segmentation_rust, update_with_covalent_bonds
from mgr.threejs_plotting import show_mof_threejs

CIF = REPO / "tests/fixtures/cif/ZIF-8-RASPA.cif"
OUT = REPO / "examples/viz_outputs/ZIF-8-RASPA_spacing0.25_threejs.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

SPACING = 0.25
MASK_THICKNESS = 0.05
H_VALUE = 0.2
MIN_DISTANCE = 1.5
THRESHOLD_ABS = 1.5
EXTRA_PADDING = 5.0

print(f"Loading {CIF.name}...")
mat_atoms = read(str(CIF))
mat_updated = update_with_covalent_bonds(mat_atoms)

print(f"Distance grid (spacing={SPACING} Å)...")
t0 = time.time()
dgrid = dgrid_from_atoms_rust(mat_atoms, spacing=SPACING).astype(np.float32)
print(f"  shape={dgrid.shape} ({time.time() - t0:.1f}s)")

print("Segmentation...")
t0 = time.time()
results = do_periodic_segmentation_rust(
    dgrid,
    mat_atoms,
    spacing=SPACING,
    mask_thickness=MASK_THICKNESS,
    h=H_VALUE,
    min_distance=MIN_DISTANCE,
    threshold_abs=THRESHOLD_ABS,
    extra_padding=EXTRA_PADDING,
    return_dgrid=True,
)
print(f"  done in {time.time() - t0:.1f}s")

results["ase_atoms"] = mat_atoms
results["dgrid"] = dgrid

print("Building Three.js scene...")
html = show_mof_threejs(
    results,
    stylize=True,
    options={
        "plot_framework_style": "instanced",
        "plot_layout": "white",
        "plot_spacing": SPACING,
        "plot_windows_style": {
            "color": 0xFF0000,
            "size": 8.0,
            "symbol": "circle",
        },
    },
)
OUT.write_text(html, encoding="utf-8")
print(f"Saved {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")

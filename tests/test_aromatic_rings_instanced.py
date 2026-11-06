"""Instanced aromatic ring trace payloads + merged Plotly highlight."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "cif"
if str(REPO / "mgr_src") not in sys.path:
    sys.path.insert(0, str(REPO / "mgr_src"))


def _mil101_atoms():
    from ase.io import read

    from mgr.analyze import compute_aromatic_cycles, update_with_covalent_bonds

    atoms = read(str(FIXTURES / "MIL-101-primitive.cif"))
    atoms = update_with_covalent_bonds(atoms)
    cycles = compute_aromatic_cycles(atoms)
    return atoms, cycles


@pytest.fixture(scope="module")
def mil101():
    return _mil101_atoms()


def test_instanced_trace_payload_shape(mil101):
    from mgr.movies import make_aromatic_rings_trace

    atoms, cycles = mil101
    traces = make_aromatic_rings_trace(atoms, cycles=cycles, use_instanced=True)
    circles = traces["circles"]
    assert circles["type"] == "instanced_ring_outlines"
    data = circles["data"]
    n = data["n_rings"]
    assert n > 0
    assert len(data["centers"]) == 3 * n
    assert len(data["normals"]) == 3 * n
    assert len(data["inner_radii"]) == n
    assert data["radial_segments"] == 16

    hex_trace = traces.get("fills_hex")
    pent_trace = traces.get("fills_pent")
    assert hex_trace is not None or pent_trace is not None
    for fill in (hex_trace, pent_trace):
        if fill is None:
            continue
        assert fill["type"] == "instanced_ring_fills"
        ni = fill["data"]["n_instances"]
        assert ni > 0
        assert len(fill["data"]["centers"]) == 3 * ni
        assert len(fill["data"]["scales"]) == ni


def test_instanced_payload_compact_vs_legacy_circles(mil101):
    from mgr.movies import make_aromatic_rings_trace

    atoms, cycles = mil101
    inst = make_aromatic_rings_trace(atoms, cycles=cycles, use_instanced=True)
    leg = make_aromatic_rings_trace(atoms, cycles=cycles, use_instanced=False)
    inst_bytes = len(json.dumps(inst["circles"]["data"]))
    leg_bytes = len(json.dumps(leg["circles"]["data"]))
    assert inst_bytes < leg_bytes * 0.5


def test_merged_highlight_payload(mil101):
    from mgr.movies import _merged_aromatic_highlight_payload

    atoms, cycles = mil101
    payload = _merged_aromatic_highlight_payload(cycles, atoms.get_positions())
    assert payload["n_vertices"] > 0
    # One fan triangle per ring vertex (hex → 6, pent → 5).
    assert payload["n_triangles"] == sum(len(c) for c in cycles)
    verts = np.asarray(payload["positions"], dtype=float).reshape(-1, 3)
    faces = np.asarray(payload["indices"], dtype=int).reshape(-1, 3)
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3


def test_plot_aromatic_rings_single_mesh(mil101):
    from mgr.visualize import plot_aromatic_rings

    atoms, cycles = mil101
    traces = plot_aromatic_rings(atoms, isomorphic_aromatic_cycles=cycles)
    assert len(traces) == 2
    assert traces[0].__class__.__name__ == "Scatter3d"
    assert traces[1].__class__.__name__ == "Mesh3d"

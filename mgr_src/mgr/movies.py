"""
Movie visualization module for Mofography.

Provides a Plotly-like API for building interactive Three.js HTML viewers
of framework + segmentation + MD snapshot animations.

Quick-start — fixed framework (framework + pores static, adsorbates animate)
--------------------------------------------------------------------------

```python
from pathlib import Path

# 1. Read a RASPA PDB movie into canonical snapshots
snapshots = mgr.read_raspa_pdb_movie(
    "mofography/examples/Movie_LTA96_1.1.1_298.150000_0.000000_component_K_0.pdb",
    max_frames=200,
)

# 2. Load the framework CIF
framework = mgr.read("examples/TestCIFs/LTA96.cif")

# 3. Build the movie figure — segmentation computed once, guests animate
fig = mgr.build_fixed_framework_movie(
    snapshots=snapshots,
    framework_atoms=framework,
    spacing=0.25,          # dgrid spacing
    isovalue=1.0,          # surface isosurface level
    surf_step_size=2,      # marching-cubes decimation
    fps=20.0,
    eye=(2.0, 0.0, 1.35),
    center=(0.0, 0.0, 0.0),
    up=(0.0, 0.0, 1.0),
)

# 4. Export interactive HTML
fig.to_html("my_movie.html")
```

Or use the one-liner convenience wrapper:

```python
mgr.make_movie_from_raspa_pdb(
    pdb_path="movie.pdb",
    cif_path="framework.cif",
    out_html="output.html",
    max_frames=200,
    spacing=0.25,
)
```

The exported HTML is self-contained (single file) and works offline once
cached — no Python runtime required to view.

Supported input formats (v1)
----------------------------
- RASPA PDB movies → ``read_raspa_pdb_movie(...)`` or ``read_movie_snapshots(..., movie_format="raspa_pdb")``
- Any ASE-readable trajectory (XYZ, extxyz, DCD, TRR, XTC, LAMMPS dump, …) →
  ``read_movie_snapshots(path, movie_format="ase")`` or ``movie_format="auto"``
  (non-``.pdb`` paths default to ASE indexed reads, one frame at a time for memory).

All adapters produce the same canonical ``list[snapshot]`` dict so downstream
code (``build_fixed_framework_movie``, ``compute_segmentation_for_snapshot``)
works identically regardless of source format.

Use ``frame_stride`` and ``position_decimals`` on ``read_movie_snapshots`` to shrink
embedded JSON and speed up browser playback for long trajectories.

Flexible framework (recompute pores per frame)
----------------------------------------------
Planned for Phase 2.  You will pass ``recompute_indices`` to control which
snapshots trigger segmentation recomputation (e.g. every 100th frame, or an
explicit list).  Uncomputed frames reuse the last known surface / graph.

Parallel / high-throughput workflows
------------------------------------
The module is intentionally split into pure-computation helpers
(``compute_segmentation_for_snapshot``) and the ``MovieFigure`` builder so
that the expensive steps can be wrapped in ``dask.delayed`` or a process pool:

```python
from dask import delayed, compute

tasks = [delayed(mgr.compute_segmentation_for_snapshot)(snap)
         for snap in snapshots]
results = compute(*tasks)   # parallel via Rust + Dask

# Then assemble traces into a single MovieFigure
fig = mgr.MovieFigure()
for seg, snap in zip(results, snapshots):
    fig.add_trace(mgr.make_surface_trace(seg["dgrid"], seg["regions"], snap["atoms"]))
```

Sidecar / streaming mode (large trajectories)
---------------------------------------------
For movies with thousands of frames, ``fig.to_html(..., sidecar=True)``
writes frame data to a ``.frames.jsonl`` file next to the HTML and the
JavaScript viewer fetches chunks on demand rather than embedding everything.
"""

from __future__ import annotations

import base64
import json
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh
from ase import Atoms
from ase.data import atomic_masses as atomic_masses_table
from ase.data import covalent_radii
from ase.geometry import cellpar_to_cell
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from matplotlib import colors as mcolors
from skimage.measure import marching_cubes
import colorsys

from .analyze import update_with_covalent_bonds
from .helper import get_fractional_coordinates, interpolate_labels

# ---------------------------------------------------------------------------
# Default styles registry (shared between Plotly and Three.js layers)
# ---------------------------------------------------------------------------

# Keys passed from ``update_traces(..., {"opacity": ...})`` etc. must merge into
# ``trace["style"]`` — the Three.js viewer's ``styleFor()`` only reads ``trace.style``.
_TRACE_STYLE_TOPLEVEL_KEYS = frozenset(
    {
        "opacity",
        "color",
        "roughness",
        "metalness",
        "side",
        "clearcoat",
        "clearcoatRoughness",
        "linewidth",
        "dashSize",
        "gapSize",
        "size",
        "shininess",
        "specular",
        "depthTest",
    }
)


DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    "framework": {
        "color": [0.7, 0.7, 0.7],
        "opacity": 1.0,
        "roughness": 0.38,
        "metalness": 0.0,
        "side": "FrontSide",
    },
    "guests": {
        "color": [1.0, 0.41, 0.71],
        "opacity": 1.0,
        "shininess": 10,
        "specular": 0x151515,
    },
    "guest_bonds": {
        "color": [0.88, 0.88, 0.90],
        "linewidth": 2,
    },
    "surface": {
        "opacity": 1.0,
        "roughness": 0.15,
        "metalness": 0.05,
        "side": "DoubleSide",
        "clearcoat": 0.8,
        "clearcoatRoughness": 0.05,
    },
    "cell": {
        "color": 0xFFA500,
        "linewidth": 2,
    },
    "internal_maxima": {
        "color": [0.85, 0.85, 0.85],
        "size": 5.0,
    },
    "periodic_maxima": {
        "color": [0.85, 0.85, 0.85],
        "size": 5.0,
    },
    "internal_windows": {
        "color": 0xFFE600,
        "size": 5.0,
    },
    "periodic_windows": {
        "color": 0xFFE600,
        "size": 5.0,
    },
    "internal_connections": {
        "color": 0xFF0000,
        "linewidth": 16,
    },
    "periodic_connections": {
        "color": 0xFF0000,
        "dashSize": 0.9,
        "gapSize": 0.5,
        "linewidth": 16,
    },
    "internal_nodes": {
        "color": 0x66CCFF,
        "size": 5.0,
    },
    "periodic_nodes": {
        "color": 0xFF9966,
        "size": 2.20,
    },
    "aromatic_rings": {
        "color": 0xFFFFFF,
        "linewidth": 4,
    },
    "aromatic_highlights": {
        "color": [0.20, 0.55, 0.95],
        "opacity": 0.3,
        "roughness": 0.5,
        "side": "DoubleSide",
    },
}

# CPK colors for hover text generation
CPK_COLORS = {
    "H": "#FFFFFF",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "P": "#FF8000",
    "F": "#90E050",
    "CL": "#1FF01F",
    "BR": "#A62929",
    "I": "#940094",
    "NA": "#AB5CF2",
    "K": "#8F40D4",
    "CA": "#3DFF00",
}


def _hex_to_rgb01(hex_color: str) -> list[float]:
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return [0.8, 0.8, 0.8]
    return [int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _resolve_style(style_ref: str | dict | None) -> dict[str, Any]:
    if isinstance(style_ref, dict):
        return style_ref
    if style_ref is None:
        return {}
    return DEFAULT_STYLES.get(style_ref, {})


# ---------------------------------------------------------------------------
# Scene model helpers
# ---------------------------------------------------------------------------

def _make_trace(
    trace_id: str,
    trace_type: str,
    name: str,
    data: dict[str, Any],
    style: str | dict | None = None,
    hover: list[str] | None = None,
    visible: bool = True,
) -> dict[str, Any]:
    return {
        "id": trace_id,
        "type": trace_type,
        "name": name,
        "visible": visible,
        "style": _resolve_style(style),
        "data": data,
        "hover": hover,
    }


def _serialize_trimesh(mesh: trimesh.Trimesh) -> dict[str, Any]:
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=int)
    colors = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(v):
            if vc.shape[1] >= 3:
                colors = vc[:, :3].astype(float) / 255.0
    if colors is None:
        colors = np.tile(np.asarray([[0.7, 0.7, 0.7]], dtype=float), (len(v), 1))
    return {
        "positions": v.reshape(-1).tolist(),
        "indices": f.reshape(-1).tolist(),
        "colors": colors.reshape(-1).tolist(),
        "n_vertices": int(len(v)),
        "n_triangles": int(len(f)),
    }


# Keys in trace.data that are large numerical arrays eligible for binary packing.
_BINARY_PACK_KEYS = frozenset({"positions", "indices", "colors", "radii", "sizes"})


def _pack_frame_data(frame: dict[str, Any]) -> dict[str, Any]:
    """Pack numerical arrays in frame trace data as base64-encoded binary blobs.

    Replaces large float/int lists with compact base64 strings (roughly 2x
    smaller than JSON text).  The JS viewer unpacks them back into typed
    arrays on demand.
    """
    if not frame or "traces" not in frame:
        return frame
    packed_traces: list[dict[str, Any]] = []
    for trace in frame["traces"]:
        packed_trace = dict(trace)
        data = trace.get("data")
        if isinstance(data, dict):
            packed_data: dict[str, Any] = {}
            for key, value in data.items():
                if key in _BINARY_PACK_KEYS and isinstance(value, list) and len(value) > 0:
                    if key == "indices":
                        max_val = max(value) if value else 0
                        dtype = np.uint16 if max_val < 65536 else np.uint32
                        arr = np.asarray(value, dtype=dtype)
                        packed_data[key] = {
                            "__binary__": True,
                            "dtype": "uint16" if dtype == np.uint16 else "uint32",
                            "count": len(value),
                            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
                        }
                    else:
                        arr = np.asarray(value, dtype=np.float32)
                        packed_data[key] = {
                            "__binary__": True,
                            "dtype": "float32",
                            "count": len(value),
                            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
                        }
                else:
                    packed_data[key] = value
            packed_trace["data"] = packed_data
        packed_traces.append(packed_trace)
    return {"frame_index": frame.get("frame_index"), "traces": packed_traces}


def _framework_vertex_hovertext(
    mesh: trimesh.Trimesh,
    ase_atoms: Atoms,
    bond_rgba: tuple[int, int, int, int] = (127, 128, 129, 255),
) -> list[str]:
    bonds = list(ase_atoms.info.get("covalent_bonds", []))
    n_atoms = int(ase_atoms.get_global_number_of_atoms())
    n_bonds = len(bonds)
    vc = np.asarray(mesh.visual.vertex_colors)
    if vc.size == 0 or vc.shape[0] != len(mesh.vertices):
        return [""] * len(mesh.vertices)
    if vc.shape[1] < 4:
        pad = np.full((vc.shape[0], 4 - vc.shape[1]), 255, dtype=vc.dtype)
        vc = np.hstack([vc, pad])
    vc_u8 = np.clip(
        np.rint(vc.astype(float) * 255.0) if vc.max() <= 1.0 else vc, 0, 255
    ).astype(np.uint8)
    colors = ["rgba(" + ",".join(c.astype(str)) + ")" for c in vc_u8]
    vertexcolors = np.asarray(colors, dtype=object)
    bond_rgba_string = "rgba(" + ",".join(str(x) for x in bond_rgba) + ")"
    n_verts = len(mesh.vertices)
    sphere_mask = vertexcolors != bond_rgba_string
    bond_mask = vertexcolors == bond_rgba_string
    n_sphere_verts = int(np.sum(sphere_mask))
    n_bond_verts = int(np.sum(bond_mask))
    if n_atoms <= 0 or n_sphere_verts % n_atoms != 0:
        return [""] * n_verts
    vertices_per_sphere = n_sphere_verts // n_atoms
    if n_bonds <= 0:
        vertices_per_bond = 0
    elif n_bond_verts % n_bonds != 0:
        return [""] * n_verts
    else:
        vertices_per_bond = n_bond_verts // n_bonds

    atomic_numbers = ase_atoms.get_atomic_numbers()
    chemical_symbols = ase_atoms.get_chemical_symbols()
    masses = atomic_masses_table[atomic_numbers]
    positions = np.round(ase_atoms.get_positions(), decimals=4)
    pos_str = [" ".join(map(str, row)) for row in positions]

    vertex_atom_ids = np.repeat(np.arange(n_atoms), vertices_per_sphere)
    vertex_symbols = np.repeat(chemical_symbols, vertices_per_sphere)
    vertex_masses = np.repeat(masses, vertices_per_sphere)
    vertex_pos = np.repeat(pos_str, vertices_per_sphere)

    hover_atoms = [
        f"Id: {i}\nElement: {s}\nMass: {m}\nPosition: {p}"
        for i, s, m, p in zip(
            vertex_atom_ids, vertex_symbols, vertex_masses, vertex_pos, strict=False
        )
    ]
    bond_atom_ids = np.repeat(np.asarray(bonds, dtype=int), vertices_per_bond, axis=0)
    hover_bonds = [
        "Bond between atoms " + " and ".join(v.astype(str)) for v in bond_atom_ids
    ]
    return hover_atoms + hover_bonds


def _serialize_framework_mesh(
    atoms: Atoms,
    bonds: list[list[int]] | None = None,
    representation: str = "ball_and_stick",
    kwargs_spheres: dict[str, Any] | None = None,
    kwargs_cylinders: dict[str, Any] | None = None,
    sphere_sections: int = 2,
    cylinder_sections: int = 15,
) -> dict[str, Any]:
    from .visualize import mesh_for_atoms

    mesh = mesh_for_atoms(
        atoms,
        bonds=bonds,
        kwargs_spheres=kwargs_spheres or {"scale": 0.65},
        kwargs_cylinders=kwargs_cylinders or {"radius": 0.15},
        representation=representation,
        sphere_sections=sphere_sections,
        cylinder_sections=cylinder_sections,
    )
    payload = _serialize_trimesh(mesh)
    payload["vertex_hover"] = _framework_vertex_hovertext(mesh, atoms)
    if len(payload["vertex_hover"]) != payload["n_vertices"]:
        payload["vertex_hover"] = [""] * int(payload["n_vertices"])
    payload["n_atoms"] = int(len(atoms))
    return payload


def _serialize_cell_lines(atoms: Atoms) -> dict[str, Any]:
    A = cellpar_to_cell(atoms.cell.cellpar()).T
    a, b, c = A[:, 0], A[:, 1], A[:, 2]
    o = np.array([0.0, 0.0, 0.0])
    corners = [o, a, b, c, a + b, a + c, b + c, a + b + c]
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (2, 4),
        (2, 6), (3, 6),
        (3, 5), (1, 5),
        (4, 7), (5, 7), (6, 7),
    ]
    pts = []
    for i, j in edges:
        pts.extend([corners[i], corners[j]])
    arr = np.asarray(pts, dtype=float)
    return {"positions": arr.reshape(-1).tolist(), "n_segments": len(edges)}


def _apply_taubin_smoothing(
    mesh: trimesh.Trimesh,
    enabled: bool = False,
    iterations: int = 10,
    lamb: float = 0.5,
    nu: float = -0.53,
) -> trimesh.Trimesh:
    if not enabled or mesh.vertices.size == 0 or mesh.faces.size == 0:
        return mesh
    try:
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=float(lamb),
            nu=float(nu),
            iterations=max(int(iterations), 1),
        )
    except Exception:
        return mesh
    return mesh


def _serialize_segmentation_surface_mesh(
    atoms: Atoms,
    seg: dict[str, Any],
    isovalue: float,
    step_size: int,
    cscale: list[list[float]] | None = None,
    taubin_smooth: bool = False,
    taubin_iterations: int = 10,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
) -> dict[str, Any]:
    dgrid = seg["dgrid"]
    regions = seg["regions"]
    verts, faces, _normals, _vals = marching_cubes(
        dgrid, level=isovalue, step_size=max(step_size, 1)
    )
    if len(faces) == 0:
        return {
            "positions": [],
            "indices": [],
            "colors": [],
            "labels": [],
            "n_vertices": 0,
            "n_triangles": 0,
        }
    cell = atoms.get_cell().array.T
    shape = np.asarray(dgrid.shape, dtype=float)
    # dgrid is sampled on inclusive [0, 1] fractional endpoints, so
    # voxel indices map with (n-1), not n.
    shape_den = np.maximum(shape - 1.0, 1.0)
    verts_cart = np.dot(cell, (verts / shape_den).T).T
    mesh = trimesh.Trimesh(vertices=verts_cart, faces=faces, process=False)
    mesh = _apply_taubin_smoothing(
        mesh,
        enabled=taubin_smooth,
        iterations=taubin_iterations,
        lamb=taubin_lambda,
        nu=taubin_nu,
    )

    # Sample region IDs in **grid index space** at marching-cubes vertices (same idea as
    # mgrcli.plotting.plot_surface_dgrid). Fractional-cell interpolation maps padded
    # vertices incorrectly into [0,1) and yields label 0 everywhere → flat grey meshes.
    from scipy.interpolate import RegularGridInterpolator

    regs = np.asarray(regions)
    nx, ny, nz = regs.shape
    gx = np.arange(nx, dtype=float)
    gy = np.arange(ny, dtype=float)
    gz = np.arange(nz, dtype=float)
    lbl_interp = RegularGridInterpolator(
        (gx, gy, gz),
        regs.astype(float, copy=False),
        method="nearest",
        bounds_error=False,
        fill_value=0,
    )
    cell_inv = np.linalg.inv(cell)
    v_world = np.asarray(mesh.vertices, dtype=float)
    idx_like = np.dot(v_world, cell_inv.T) * shape_den
    labels = lbl_interp(idx_like).astype(int)

    # Clamp negative labels to 0 (can arise from regions_cluster - 1 in pocket types)
    labels = np.maximum(labels, 0)

    # Backfill label-0 vertices with nearest non-zero label to prevent
    # uncolored gray patches on the mesh surface (k-NN via cKDTree).
    zero_mask = labels == 0
    if zero_mask.any() and not zero_mask.all():
        nonzero_idx = np.where(~zero_mask)[0]
        zero_idx = np.where(zero_mask)[0]
        from scipy.spatial import cKDTree

        tree = cKDTree(v_world[nonzero_idx])
        _, nn = tree.query(v_world[zero_idx], k=1)
        labels[zero_idx] = labels[nonzero_idx[nn]]

    _DEFAULT_CSCALE = [
        [102, 197, 204],
        [246, 207, 113],
        [248, 156, 116],
        [220, 176, 242],
        [135, 197, 95],
        [158, 185, 243],
        [254, 136, 177],
        [201, 219, 116],
        [139, 224, 164],
        [180, 151, 231],
        [179, 179, 179],
    ]
    if cscale is None:
        cscale = _DEFAULT_CSCALE
    else:
        cscale = [[int(c * 255) for c in rgb] for rgb in cscale]
    uniq = sorted([int(x) for x in np.unique(labels) if int(x) > 0])
    cmap = {rid: cscale[i % len(cscale)] for i, rid in enumerate(uniq)}
    colors = np.asarray(
        [cmap.get(int(r), [180, 180, 180]) for r in labels], dtype=float
    ) / 255.0
    return {
        "positions": np.asarray(mesh.vertices, dtype=float).reshape(-1).tolist(),
        "indices": np.asarray(mesh.faces, dtype=int).reshape(-1).tolist(),
        "colors": colors.reshape(-1).tolist(),
        "labels": labels.tolist(),
        "n_vertices": int(len(mesh.vertices)),
        "n_triangles": int(len(mesh.faces)),
    }


def _serialize_maxima_trace(seg: dict[str, Any], atoms: Atoms, cscale: list[list[float]] | None = None) -> dict[str, Any]:
    dgrid = np.asarray(seg["dgrid"])
    regions = np.asarray(seg.get("regions", np.zeros(dgrid.shape, dtype=int)))
    maxima = np.asarray(seg["all_maxima"], dtype=float)
    if maxima.size == 0:
        return {
            "internal_positions": [],
            "internal_sizes": [],
            "internal_colors": [],
            "internal_hover": [],
            "internal_n_points": 0,
            "outside_positions": [],
            "outside_sizes": [],
            "outside_colors": [],
            "outside_hover": [],
            "outside_n_points": 0,
            "pbc_positions": [],
            "pbc_sizes": [],
            "pbc_colors": [],
            "pbc_hover": [],
            "pbc_n_points": 0,
            "n_points": 0,
        }
    maxima_wrapped = np.mod(maxima, np.asarray(dgrid.shape, dtype=int)).astype(int)
    cage_radii = dgrid[tuple(maxima_wrapped.T)]
    cell = atoms.get_cell().array.T
    grid_shape = np.maximum(np.asarray(dgrid.shape, dtype=float) - 1.0, 1.0)
    frac = maxima / grid_shape
    pos = np.dot(cell, frac.T).T

    _DEFAULT_CSCALE = [
        [102, 197, 204],
        [246, 207, 113],
        [248, 156, 116],
        [220, 176, 242],
        [135, 197, 95],
        [158, 185, 243],
        [254, 136, 177],
        [201, 219, 116],
        [139, 224, 164],
        [180, 151, 231],
        [179, 179, 179],
    ]
    if cscale is None:
        cscale = _DEFAULT_CSCALE
    else:
        cscale = [[int(c * 255) for c in rgb] for rgb in cscale]

    # Color maxima by their region label for consistency with the surface mesh.
    # Each maxima is at a grid point; we look up its region in the regions array.
    region_labels = regions[tuple(maxima_wrapped.T)]
    uniq_regions = sorted([int(x) for x in np.unique(region_labels) if int(x) > 0])
    cmap = {rid: cscale[i % len(cscale)] for i, rid in enumerate(uniq_regions)}
    colors = [list(np.asarray(cmap.get(int(r), [180, 180, 180]), dtype=float) / 255.0) for r in region_labels]

    # Marker scale (half of former 6 + 5 * norm): 3 + 2.5 * norm from cage radii
    vals = np.asarray(cage_radii, dtype=float)
    vmin = float(vals.min()) if len(vals) else 0.0
    vmax = float(vals.max()) if len(vals) else 1.0
    denom = max(vmax - vmin, 1e-12)
    norm = (vals - vmin) / denom
    sizes = (6.0 + 5.0 * norm).astype(float).tolist()
    maxima_info = np.asarray(seg.get("maxima_info", []), dtype=int)
    maxima_pbc_dict_raw = seg.get("maxima_pbc_dict", {}) or {}

    def _lookup_pbc_label(coord: tuple[int, int, int]) -> int | None:
        # Preferred path: tuple-indexed mapping from base maxima coords -> pbc label.
        if coord in maxima_pbc_dict_raw:
            return int(maxima_pbc_dict_raw[coord])
        # Defensive fallbacks for stringified dict keys.
        key_csv = f"{coord[0]},{coord[1]},{coord[2]}"
        if key_csv in maxima_pbc_dict_raw:
            return int(maxima_pbc_dict_raw[key_csv])
        key_tuple_str = str(coord)
        if key_tuple_str in maxima_pbc_dict_raw:
            return int(maxima_pbc_dict_raw[key_tuple_str])
        return None

    def _maxima_info_for_idx(i: int) -> tuple[int | None, list[int] | None]:
        if maxima_info.ndim != 2 or maxima_info.shape[1] < 7 or i >= maxima_info.shape[0]:
            return None, None
        row = maxima_info[i]
        base_coord = (int(row[1]), int(row[2]), int(row[3]))
        pbc_label = _lookup_pbc_label(base_coord)
        offset = [int(row[4]), int(row[5]), int(row[6])]
        return pbc_label, offset

    hover = []
    for i, (g, p, r, rid, s) in enumerate(zip(maxima, pos, cage_radii, region_labels, sizes, strict=False)):
        pbc_label, offset = _maxima_info_for_idx(i)
        extra = ""
        if pbc_label is not None:
            extra += f"\nmaxima pbc label: {pbc_label}"
        if offset is not None:
            extra += f"\noffset: [{offset[0]}, {offset[1]}, {offset[2]}]"
        hover.append(
            f"trace: local maxima\n"
            f"index: {i + 1}\n"
            f"maxima label: {i + 1}"
            f"{extra}\n"
            f"grid: [{int(g[0])}, {int(g[1])}, {int(g[2])}]\n"
            f"xyz: ({float(p[0]):.3f}, {float(p[1]):.3f}, {float(p[2]):.3f})\n"
            f"pore radius (dgrid): {float(r):.3f} A\n"
            f"maxima size: {float(s):.3f}\n"
            f"region: {int(rid)}"
        )

    connections = list(seg.get("connections", []))
    periodic_nodes: set[int] = set()
    internal_nodes: set[int] = set()
    for c in connections:
        if len(c) < 3 or not c[1]:
            continue
        pair = {int(c[0][0]), int(c[0][1])}
        if str(c[2]) == "periodic":
            periodic_nodes.update(pair)
        elif str(c[2]) == "internal":
            internal_nodes.update(pair)
    outside_nodes = periodic_nodes - internal_nodes
    inside_nodes = {i + 1 for i in range(len(maxima)) if (i + 1) not in outside_nodes}

    def _pack(ids: set[int]):
        if not ids:
            return [], [], [], [], 0
        n_m = int(len(maxima))
        valid_ids = sorted(i for i in ids if 1 <= i <= n_m)
        if not valid_ids:
            return [], [], [], [], 0
        idx = np.asarray([i - 1 for i in valid_ids], dtype=int)
        p = pos[idx]
        s = np.asarray(sizes, dtype=float)[idx]
        c = np.asarray(colors, dtype=float)[idx]
        h = [hover[int(i)] for i in idx]
        return p.reshape(-1).tolist(), s.tolist(), c.reshape(-1).tolist(), h, int(len(valid_ids))

    maxima_pbc = np.asarray(seg.get("maxima_pbc", []), dtype=float)
    if maxima_pbc.size == 0 and maxima_info.ndim == 2 and maxima_info.shape[1] >= 7:
        zero_mask = np.all(maxima_info[:, 4:7] == 0, axis=1)
        maxima_pbc = np.asarray(maxima_info[zero_mask, 1:4], dtype=float)
    if maxima_pbc.ndim == 1:
        maxima_pbc = maxima_pbc.reshape(-1, 3) if maxima_pbc.size else np.zeros((0, 3), dtype=float)
    pbc_pos = []
    pbc_sizes = []
    pbc_cols = []
    pbc_hover = []
    pbc_n = 0
    if maxima_pbc.size:
        pbc_wrapped = np.mod(maxima_pbc, np.asarray(dgrid.shape, dtype=int)).astype(int)
        pbc_radii = dgrid[tuple(pbc_wrapped.T)]
        pbc_frac = maxima_pbc / grid_shape
        pbc_cart = np.dot(cell, pbc_frac.T).T
        pbc_regions = regions[tuple(pbc_wrapped.T)]
        pbc_colors = [list(np.asarray(cmap.get(int(r), [180, 180, 180]), dtype=float) / 255.0) for r in pbc_regions]
        pbc_norm = (np.asarray(pbc_radii, dtype=float) - vmin) / denom
        pbc_sizes_arr = (3.0 + 2.5 * pbc_norm).astype(float)
        pbc_hover = [
            (
                f"trace: maxima pbc\n"
                f"pbc index: {i + 1}\n"
                f"grid: [{int(g[0])}, {int(g[1])}, {int(g[2])}]\n"
                f"xyz: ({float(p[0]):.3f}, {float(p[1]):.3f}, {float(p[2]):.3f})\n"
                f"pore radius (dgrid): {float(r):.3f} A\n"
                f"maxima size: {float(s):.3f}\n"
                f"region: {int(rid)}"
            )
            for i, (g, p, r, rid, s) in enumerate(zip(maxima_pbc, pbc_cart, pbc_radii, pbc_regions, pbc_sizes_arr, strict=False))
        ]
        pbc_pos = pbc_cart.reshape(-1).tolist()
        pbc_sizes = pbc_sizes_arr.tolist()
        pbc_cols = np.asarray(pbc_colors, dtype=float).reshape(-1).tolist()
        pbc_n = int(len(maxima_pbc))

    in_pos, in_sizes, in_cols, in_hover, in_n = _pack(inside_nodes)
    out_pos, out_sizes, out_cols, out_hover, out_n = _pack(outside_nodes)
    return {
        "internal_positions": in_pos,
        "internal_sizes": in_sizes,
        "internal_colors": in_cols,
        "internal_hover": in_hover,
        "internal_n_points": in_n,
        "outside_positions": out_pos,
        "outside_sizes": out_sizes,
        "outside_colors": out_cols,
        "outside_hover": out_hover,
        "outside_n_points": out_n,
        "pbc_positions": pbc_pos,
        "pbc_sizes": pbc_sizes,
        "pbc_colors": pbc_cols,
        "pbc_hover": pbc_hover,
        "pbc_n_points": pbc_n,
        "n_points": int(len(pos)),
    }


def _serialize_windows_trace(
    seg: dict[str, Any],
    atoms = None,
    spacing: float | None = None,
) -> dict[str, Any]:
    conns = list(seg.get("connections", []))
    internal_windows = []
    internal_hover = []
    periodic_windows = []
    periodic_hover = []

    # Compute wall tolerance for "close or inside" criteria (same as maxima).
    wall_tol = None
    dgrid_shape = None
    inv_cell = None
    if atoms is not None and spacing is not None:
        from .helper import compute_wall_cells
        dgrid = np.asarray(seg.get("dgrid", []))
        if dgrid.size:
            dgrid_shape = np.asarray(dgrid.shape, dtype=float)
            wall_tol = compute_wall_cells(atoms, spacing, wall_thickness=0.05)
            cell = atoms.get_cell().array.T
            try:
                inv_cell = np.linalg.inv(cell)
            except np.linalg.LinAlgError:
                inv_cell = None

    def _window_grid_coord(cart: np.ndarray) -> np.ndarray | None:
        """Convert Cartesian window position to grid coordinates."""
        if inv_cell is None or dgrid_shape is None:
            return None
        frac = np.dot(cart, inv_cell)
        return frac * dgrid_shape

    for c in conns:
        if len(c) < 5 or not c[1]:
            continue
        ctype = str(c[2])
        pts = c[3] if c[3] is not None else []
        wr = c[4] if c[4] is not None else []
        n1, n2 = c[0]
        for p, r in zip(pts, wr, strict=False):
            wp = np.asarray(p, dtype=float)
            txt = (
                f"trace: {ctype} windows\n"
                f"connection maxima: {n1}-{n2}\n"
                f"type: {ctype}\n"
                f"size: {float(r):.3f} A"
            )
            if ctype == "internal" and wall_tol is not None and dgrid_shape is not None:
                # Reclassify far_outside windows as periodic
                grid_c = _window_grid_coord(wp)
                if grid_c is not None:
                    from .helper import is_inside_or_close_to_boundary
                    loc = is_inside_or_close_to_boundary(
                        [tuple(grid_c)], tuple(dgrid_shape.astype(int)), wall_tol
                    )[0]
                    if loc == "far_outside":
                        periodic_windows.append(wp)
                        periodic_hover.append(txt.replace("internal", "periodic (wrapped)"))
                        continue
            if ctype == "internal":
                internal_windows.append(wp)
                internal_hover.append(txt)
            elif ctype == "periodic":
                periodic_windows.append(wp)
                periodic_hover.append(txt)
    out: dict[str, Any] = {
        "internal_positions": [],
        "internal_hover": internal_hover,
        "internal_n_points": 0,
        "periodic_positions": [],
        "periodic_hover": periodic_hover,
        "periodic_n_points": 0,
    }
    if internal_windows:
        w = np.vstack(internal_windows)
        out["internal_positions"] = w.reshape(-1).tolist()
        out["internal_n_points"] = int(len(w))
    if periodic_windows:
        w = np.vstack(periodic_windows)
        out["periodic_positions"] = w.reshape(-1).tolist()
        out["periodic_n_points"] = int(len(w))
    return out


def _serialize_connection_traces(
    seg: dict[str, Any], atoms: Atoms
) -> dict[str, Any]:
    conns = list(seg.get("connections", []))
    maxima = np.asarray(seg.get("all_maxima", []), dtype=float)
    if len(conns) == 0 or maxima.size == 0:
        return {
            "internal_segments": [],
            "internal_hover": [],
            "periodic_segments": [],
            "periodic_hover": [],
            "internal_points": [],
            "internal_points_hover": [],
            "periodic_points": [],
            "periodic_points_hover": [],
        }
    shape = np.maximum(np.asarray(seg["dgrid"].shape, dtype=float) - 1.0, 1.0)
    cell = atoms.get_cell().array.T
    node_positions: dict[int, np.ndarray] = {}
    nodes: set[int] = set()
    for c in conns:
        nodes.update(c[0])
    for node in sorted(nodes):
        idx = int(node) - 1
        if idx < 0 or idx >= len(maxima):
            continue
        frac = maxima[idx] / shape
        node_positions[int(node)] = np.dot(cell, frac.T).T

    internal_segments: list[list[float]] = []
    internal_hover: list[str] = []
    periodic_segments: list[list[float]] = []
    periodic_hover: list[str] = []
    internal_nodes: set[int] = set()
    periodic_nodes: set[int] = set()
    for c in conns:
        if len(c) < 5 or not c[1]:
            continue
        n1, n2 = [int(x) for x in c[0]]
        p1 = node_positions.get(n1)
        p2 = node_positions.get(n2)
        if p1 is None or p2 is None:
            continue
        ctype = str(c[2])
        pts = c[3] if c[3] is not None else []
        wr = c[4] if c[4] is not None else []
        if ctype == "internal":
            internal_nodes.update([n1, n2])
            internal_segments.append(
                [
                    float(p1[0]), float(p1[1]), float(p1[2]),
                    float(p2[0]), float(p2[1]), float(p2[2]),
                ]
            )
            txt = f"trace: internal connections\nedge: {n1}-{n2}"
            if len(pts) and len(wr):
                txt += f"\nwindow radius: {float(wr[0]):.3f} A"
            internal_hover.append(txt)
        elif ctype == "periodic":
            periodic_nodes.update([n1, n2])
            periodic_segments.append(
                [
                    float(p1[0]), float(p1[1]), float(p1[2]),
                    float(p2[0]), float(p2[1]), float(p2[2]),
                ]
            )
            txt = f"trace: periodic connections\nedge: {n1}-{n2}"
            if len(pts) and len(wr):
                txt += f"\nwindow radius: {float(wr[0]):.3f} A"
            periodic_hover.append(txt)

    def _pack_nodes(node_ids: set[int], label: str) -> tuple[list[float], list[str]]:
        if not node_ids:
            return [], []
        arr: list[np.ndarray] = []
        hv: list[str] = []
        for n in sorted(node_ids):
            p = node_positions.get(n)
            if p is None:
                continue
            arr.append(p)
            hv.append(
                f"trace: {label}\n"
                f"node: {n}\n"
                f"xyz: ({float(p[0]):.3f}, {float(p[1]):.3f}, {float(p[2]):.3f})"
            )
        if not arr:
            return [], []
        a = np.vstack(arr)
        return a.reshape(-1).tolist(), hv

    internal_points, internal_points_hover = _pack_nodes(internal_nodes, "internal maxima")
    periodic_points, periodic_points_hover = _pack_nodes(periodic_nodes, "periodic maxima")
    return {
        "internal_segments": (
            np.asarray(internal_segments, dtype=float).reshape(-1).tolist()
            if internal_segments else []
        ),
        "internal_hover": internal_hover,
        "periodic_segments": (
            np.asarray(periodic_segments, dtype=float).reshape(-1).tolist()
            if periodic_segments else []
        ),
        "periodic_hover": periodic_hover,
        "internal_points": internal_points,
        "internal_points_hover": internal_points_hover,
        "periodic_points": periodic_points,
        "periodic_points_hover": periodic_points_hover,
    }


# ---------------------------------------------------------------------------
# I/O adapters
# ---------------------------------------------------------------------------

def _snapshot_dict(
    atoms: Atoms,
    *,
    path: str | Path,
    frame_index: int,
    source_format: str,
) -> dict[str, Any]:
    return {
        "atoms": atoms,
        "bonds": None,
        "cell": atoms.get_cell().array if atoms.cell else None,
        "dgrid": None,
        "regions": None,
        "connections": None,
        "meta": {
            "frame_index": frame_index,
            "source": str(path),
            "format": source_format,
        },
    }


def _normalize_element_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if not s:
        return "X"
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()


def read_movie_snapshots(
    path: str | Path,
    *,
    movie_format: str = "auto",
    max_frames: int | None = None,
    frame_stride: int = 1,
    position_decimals: int = 4,
) -> list[dict[str, Any]]:
    """Load MD snapshots as canonical ``snapshot`` dicts (each with an ``ase.Atoms``).

    Parameters
    ----------
    path
        Trajectory or RASPA multi-model PDB file.
    movie_format
        ``"auto"`` — ``.pdb`` uses RASPA parser; everything else uses ASE indexed reads
        (one frame at a time, low peak memory).
        ``"raspa_pdb"`` / ``"ase"`` force a specific backend.
    max_frames
        Maximum number of *kept* frames (after stride), not counting skipped indices.
    frame_stride
        Keep every Nth source frame (reduces output size and work in the browser).
    position_decimals
        Round Cartesian positions before building ``Atoms`` (smaller JSON when embedded).
    """
    from .data import read_raspa_pdb

    p = Path(path).expanduser().resolve()
    stride = max(int(frame_stride), 1)
    decimals = max(int(position_decimals), 0)
    fmt = (movie_format or "auto").lower()
    if fmt == "auto":
        fmt = "raspa_pdb" if p.suffix.lower() == ".pdb" else "ase"

    snapshots: list[dict[str, Any]] = []

    if fmt == "raspa_pdb":
        raw = read_raspa_pdb(str(p))
        coords = raw["coords"]
        symbols = raw["symbols"]
        cells = raw["cells"]
        for i in range(len(coords)):
            if i % stride != 0:
                continue
            xyz = np.round(np.asarray(coords[i], dtype=np.float64), decimals=decimals)
            syms = [_normalize_element_symbol(s) for s in symbols[i]]
            cell = cells[i] if i < len(cells) else cells[-1] if len(cells) else None
            atoms = Atoms(symbols=syms, positions=xyz, cell=cell, pbc=True)
            snapshots.append(_snapshot_dict(atoms, path=p, frame_index=i, source_format="raspa_pdb"))
            if max_frames is not None and len(snapshots) >= max_frames:
                break
    elif fmt == "ase":
        i = 0
        while True:
            try:
                at = read(str(p), index=i)
            except Exception:
                break
            if i % stride == 0:
                pos = np.round(np.asarray(at.get_positions(), dtype=np.float64), decimals=decimals)
                pbc = bool(np.all(at.pbc)) if getattr(at, "pbc", None) is not None else True
                atoms = Atoms(
                    symbols=at.get_chemical_symbols(),
                    positions=pos,
                    cell=at.cell,
                    pbc=pbc,
                )
                snapshots.append(_snapshot_dict(atoms, path=p, frame_index=i, source_format="ase"))
                if max_frames is not None and len(snapshots) >= max_frames:
                    break
            i += 1
    else:
        raise ValueError(f"Unsupported movie_format: {movie_format!r}")

    return snapshots


def read_raspa_pdb_movie(
    path: str | Path,
    max_frames: int | None = None,
    frame_stride: int = 1,
    position_decimals: int = 4,
) -> list[dict[str, Any]]:
    """Parse a RASPA PDB movie into canonical snapshot dictionaries."""
    return read_movie_snapshots(
        path,
        movie_format="raspa_pdb",
        max_frames=max_frames,
        frame_stride=frame_stride,
        position_decimals=position_decimals,
    )


def embedded_movie_dict_from_snapshots(
    snapshots: list[dict[str, Any]],
    *,
    source: str,
    inferred_format: str,
    frame_stride: int,
    position_decimals: int,
) -> dict[str, Any]:
    """Compact per-frame dict for the legacy devtools Three.js HTML (positions + radii only)."""
    frames: list[dict[str, Any]] = []
    for snap in snapshots:
        at = snap["atoms"]
        syms = [_normalize_element_symbol(s) for s in at.get_chemical_symbols()]
        xyz = np.asarray(at.get_positions(), dtype=np.float32)
        atomic_numbers = at.get_atomic_numbers()
        radii = (covalent_radii[atomic_numbers] * 0.8).astype(np.float32)
        frames.append(
            {
                "positions": xyz.reshape(-1).tolist(),
                "symbols": syms,
                "radii": radii.tolist(),
            }
        )
    return {
        "source": source,
        "format": inferred_format,
        "frame_stride": int(frame_stride),
        "position_decimals": int(position_decimals),
        "n_frames": int(len(frames)),
        "n_atoms": int(len(frames[0]["symbols"])) if frames else 0,
        "frames": frames,
        "n_frames_scanned": int(len(snapshots)),
    }


def embedded_movie_dict_from_path(
    path: str | Path,
    *,
    movie_format: str = "auto",
    max_frames: int | None = None,
    frame_stride: int = 1,
    position_decimals: int = 4,
) -> dict[str, Any]:
    """Load trajectory and return the JSON ``movie`` block used by ``md_pdb_threejs_viewer``."""
    p = Path(path).expanduser().resolve()
    snaps = read_movie_snapshots(
        p,
        movie_format=movie_format,
        max_frames=max_frames,
        frame_stride=frame_stride,
        position_decimals=position_decimals,
    )
    inferred = str(snaps[0]["meta"].get("format", "unknown")) if snaps else "none"
    return embedded_movie_dict_from_snapshots(
        snaps,
        source=str(p),
        inferred_format=inferred,
        frame_stride=frame_stride,
        position_decimals=position_decimals,
    )


def _coerce_snapshot_indices(
    snapshot_indices: Sequence[int] | None,
    *,
    n_frames: int,
    max_frames: int | None = None,
) -> list[int]:
    """Return validated frame indices (unique, in-range, ordered)."""
    if n_frames <= 0:
        return []
    if snapshot_indices is None:
        idx = list(range(n_frames))
        if max_frames is not None:
            idx = idx[: max(int(max_frames), 0)]
        return idx
    seen: set[int] = set()
    out: list[int] = []
    for raw in snapshot_indices:
        i = int(raw)
        if i < 0 or i >= n_frames or i in seen:
            continue
        seen.add(i)
        out.append(i)
    out.sort()
    if max_frames is not None:
        out = out[: max(int(max_frames), 0)]
    return out


def _component_palette_rgb01(i: int) -> tuple[float, float, float]:
    """Simple deterministic palette for component traces."""
    palette = [
        (1.00, 0.41, 0.71),  # pink
        (0.35, 0.78, 0.98),  # cyan
        (1.00, 0.77, 0.20),  # amber
        (0.60, 0.93, 0.45),  # green
        (0.79, 0.64, 1.00),  # violet
        (1.00, 0.55, 0.40),  # salmon
    ]
    return palette[i % len(palette)]


def compute_sparse_segmentation_series(
    framework_snapshots: Sequence[dict[str, Any]],
    snapshot_indices: Sequence[int],
    *,
    spacing: float = 0.25,
    mask_thickness: float = 0.5,
    h: float = 0.2,
    threshold_abs: float = 1.5,
    min_distance: float = 1.5,
    extra_padding: int = 5,
) -> list[dict[str, Any]]:
    """Compute segmentation payloads for sparse framework frame indices."""
    results: list[dict[str, Any]] = []
    for frame_index in snapshot_indices:
        snap = framework_snapshots[frame_index]
        seg = compute_segmentation_for_snapshot(
            snap,
            spacing=spacing,
            mask_thickness=mask_thickness,
            h=h,
            threshold_abs=threshold_abs,
            min_distance=min_distance,
            extra_padding=extra_padding,
        )
        dgrid = np.asarray(seg["dgrid"])
        regions = np.asarray(seg["regions"])
        results.append(
            {
                "frame_index": int(frame_index),
                "segmentation": seg,
                "summary": {
                    "dgrid_shape": [int(v) for v in dgrid.shape],
                    "dgrid_max": float(np.nanmax(dgrid)),
                    "dgrid_min": float(np.nanmin(dgrid)),
                    "n_regions": int(np.unique(regions).size),
                    "n_connections": int(len(seg.get("connections", []))),
                    "n_maxima": int(len(seg.get("all_maxima", []))),
                },
            }
        )
    return results


# ---------------------------------------------------------------------------
# Segmentation helper
# ---------------------------------------------------------------------------

def compute_segmentation_for_snapshot(
    snapshot: dict[str, Any],
    spacing: float = 0.25,
    mask_thickness: float = 0.5,
    h: float = 0.2,
    threshold_abs: float = 1.5,
    min_distance: float = 1.5,
    extra_padding: int = 5,
    use_rust: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute dgrid, regions, maxima, connections for a single snapshot."""
    from .rust_fun import do_periodic_segmentation_rust
    from .compute import dgrid_from_atoms_cpu

    atoms = snapshot["atoms"]
    dgrid = snapshot.get("dgrid")
    if dgrid is None:
        dgrid = dgrid_from_atoms_cpu(atoms, spacing=spacing, block_size=20000).astype(
            np.float32
        )

    if use_rust:
        seg = do_periodic_segmentation_rust(
            dgrid,
            atoms,
            spacing=spacing,
            mask_thickness=mask_thickness,
            h=h,
            threshold_abs=threshold_abs,
            min_distance=min_distance,
            return_dgrid=True,
            return_padded_regions=False,
            extra_padding=extra_padding,
            **kwargs,
        )
    else:
        raise NotImplementedError("Only Rust backend is supported for now.")

    return {
        "dgrid": seg.get("dgrid", dgrid),
        "regions": seg["regions"],
        "maxima_pbc": seg.get("maxima_pbc", []),
        "connections": seg.get("connections", []),
        "all_maxima": seg.get("all_maxima", []),
    }


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------

def make_framework_trace(
    atoms: Atoms,
    bonds: str | list | None = "auto",
    style: str | dict | None = "framework",
    name: str = "Framework",
    trace_id: str = "framework",
    representation: str = "ball_and_stick",
    kwargs_spheres: dict[str, Any] | None = None,
    kwargs_cylinders: dict[str, Any] | None = None,
    sphere_sections: int = 2,
    cylinder_sections: int = 15,
) -> dict[str, Any]:
    def _ase_covalent_bonds(a: Atoms, mult: float = 1.05) -> list[list[int]]:
        cutoffs = np.asarray(natural_cutoffs(a, mult=mult), dtype=float)
        i_idx, j_idx = neighbor_list("ij", a, cutoffs, self_interaction=False)
        seen: set[tuple[int, int]] = set()
        out: list[list[int]] = []
        for i, j in zip(i_idx, j_idx, strict=False):
            x = int(min(i, j))
            y = int(max(i, j))
            if x == y or (x, y) in seen:
                continue
            seen.add((x, y))
            out.append([x, y])
        return out

    mesh_bonds: list[list[int]] | None = None
    if bonds == "auto":
        if "covalent_bonds" in atoms.info and atoms.info["covalent_bonds"]:
            mesh_bonds = atoms.info["covalent_bonds"]
        else:
            atoms.info["covalent_bonds"] = _ase_covalent_bonds(atoms, mult=1.05)
            mesh_bonds = atoms.info["covalent_bonds"]
    elif bonds is None:
        mesh_bonds = None
    elif isinstance(bonds, list):
        mesh_bonds = bonds
    payload = _serialize_framework_mesh(
        atoms,
        bonds=mesh_bonds,
        representation=representation,
        kwargs_spheres=kwargs_spheres,
        kwargs_cylinders=kwargs_cylinders,
        sphere_sections=sphere_sections,
        cylinder_sections=cylinder_sections,
    )
    return _make_trace(
        trace_id=trace_id,
        trace_type="mesh",
        name=name,
        data=payload,
        style=style,
        hover=payload.get("vertex_hover"),
        visible=True,
    )


def make_framework_scatter_trace(
    atoms: Atoms,
    bonds: str | list | None = "auto",
    style: str | dict | None = "framework",
    name: str = "Framework",
    trace_id: str = "framework",
    atom_scale: float = 0.65,
    bond_color: list[float] | None = None,
    bond_linewidth: float = 8.0,
    depth_test: bool = True,
) -> dict[str, dict[str, Any]]:
    """Create framework as scatter points + line segments (fast, no mesh).

    Returns dict with ``"atoms"`` and ``"bonds"`` trace dicts suitable for
    ``MovieFigure.add_trace()``.  Renders atoms as ``THREE.Points`` and bonds
    as ``THREE.LineSegments`` — orders of magnitude fewer vertices than mesh.

    When *depth_test* is ``True`` (default), framework respects normal depth
    occlusion (hidden behind opaque surfaces).  Set ``False`` to render
    framework on top of everything.

    Parameters
    ----------
    atoms:
        ASE Atoms object (framework).
    bonds:
        ``"auto"`` (detect from covalent radii), a ``list[list[int]]`` of
        bond pairs, or ``None`` to skip bonds.
    style:
        Style name or dict (passed through to both traces).
    name:
        Base display name for the legend.
    trace_id:
        Base trace id (will produce ``trace_id + "_atoms"`` and ``"_bonds"``).
    atom_scale:
        Scale factor applied to covalent radii for point sizes.
    bond_color:
        RGB triple (0-1) for bond line colour.  Defaults to [0.5, 0.5, 0.5].
    bond_linewidth:
        Line width for bonds (8.0 = thick; WebGL may clamp on some
        platforms).
    depth_test:
        If ``True``, framework is hidden behind opaque surfaces.  Set
        ``False`` to always show framework on top.
    """
    from ase.data.colors import cpk_colors

    # ---- Resolve bonds ----
    if bonds == "auto":
        if "covalent_bonds" in atoms.info and atoms.info["covalent_bonds"]:
            bond_list = atoms.info["covalent_bonds"]
        else:
            cutoffs = np.asarray(natural_cutoffs(atoms, mult=1.05), dtype=float)
            i_idx, j_idx = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
            seen: set[tuple[int, int]] = set()
            bond_list = []
            for i, j in zip(i_idx, j_idx, strict=False):
                x, y = int(min(i, j)), int(max(i, j))
                if x == y or (x, y) in seen:
                    continue
                seen.add((x, y))
                bond_list.append([x, y])
            atoms.info["covalent_bonds"] = bond_list
    elif bonds is None:
        bond_list = []
    elif isinstance(bonds, list):
        bond_list = bonds
    else:
        bond_list = []

    n_atoms = int(len(atoms))
    atomic_numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    masses = atomic_masses_table[atomic_numbers]

    # ---- Atom colours from CPK ----
    atom_colors_rgb = np.array([cpk_colors[an] for an in atomic_numbers], dtype=float)
    atom_colors_flat = atom_colors_rgb.reshape(-1).tolist()

    # ---- Atom point sizes from covalent radii ----
    # Same scale as mesh spheres (covalent_radii * atom_scale), mapped to pixels.
    # At typical camera distance where the unit cell fills ~half viewport height,
    # a sphere of radius r Å occupies ~r/cell_size * viewport_height px.
    # Multiplier 14.0 approximates this for common MOF cell sizes (20-40 Å).
    radii = covalent_radii[atomic_numbers] * atom_scale
    sizes = (radii * 14.0).tolist()

    # ---- Positions flat ----
    positions_flat = positions.reshape(-1).tolist()

    # ---- Hover text ----
    hover = [
        f"Id: {i}\nElement: {s}\nMass: {m:.3f}\nPosition: {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
        for i, (s, m, p) in enumerate(zip(symbols, masses, positions))
    ]

    # Atom trace (points)
    atom_trace = _make_trace(
        trace_id=f"{trace_id}_atoms",
        trace_type="points",
        name=f"{name} (atoms)",
        data={
            "positions": positions_flat,
            "colors": atom_colors_flat,
            "sizes": sizes,
            "n_points": n_atoms,
        },
        style={
            "color": [0.7, 0.7, 0.7],
            "size": float(max(sizes)) if sizes else 2.5,
            "depthTest": depth_test,
            "opacity": 0.95,
        },
        hover=hover,
        visible=True,
    )

    # ---- Bond trace (line segments) ----
    _bond_color = bond_color if bond_color is not None else [0.5, 0.5, 0.5]
    if bond_list and len(bond_list) > 0:
        bond_positions_flat: list[float] = []
        for a_idx, b_idx in bond_list:
            if a_idx < n_atoms and b_idx < n_atoms:
                pa = positions[a_idx]
                pb = positions[b_idx]
                bond_positions_flat.extend([float(pa[0]), float(pa[1]), float(pa[2])])
                bond_positions_flat.extend([float(pb[0]), float(pb[1]), float(pb[2])])
        n_bond_segments = len(bond_list)
    else:
        bond_positions_flat = []
        n_bond_segments = 0

    bond_hex = int(round(_bond_color[0] * 255)) * 0x10000 + int(round(_bond_color[1] * 255)) * 0x100 + int(round(_bond_color[2] * 255))
    bond_trace = _make_trace(
        trace_id=f"{trace_id}_bonds",
        trace_type="line_segments",
        name=f"{name} (bonds)",
        data={
            "positions": bond_positions_flat,
            "n_segments": n_bond_segments,
        },
        style={
            "color": bond_hex,
            "linewidth": bond_linewidth,
            "depthTest": depth_test,
        },
        hover=None,
        visible=True,
    )

    return {"atoms": atom_trace, "bonds": bond_trace}


def make_surface_trace(
    dgrid: np.ndarray,
    regions: np.ndarray,
    atoms: Atoms,
    isovalue: float = 1.0,
    step_size: int = 2,
    style: str | dict | None = "surface",
    name: str = "Surface",
    trace_id: str = "surface",
    cscale: list[list[float]] | None = None,
    hover_prefix: str | None = None,
    taubin_smooth: bool = False,
    taubin_iterations: int = 10,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
) -> dict[str, Any]:
    seg = {"dgrid": dgrid, "regions": regions}
    payload = _serialize_segmentation_surface_mesh(
        atoms=atoms,
        seg=seg,
        isovalue=isovalue,
        step_size=step_size,
        cscale=cscale,
        taubin_smooth=taubin_smooth,
        taubin_iterations=taubin_iterations,
        taubin_lambda=taubin_lambda,
        taubin_nu=taubin_nu,
    )
    hover = None
    if hover_prefix and payload.get("labels"):
        hover = [f"{hover_prefix} {int(l)}" if int(l) > 0 else hover_prefix for l in payload["labels"]]
    return _make_trace(
        trace_id=trace_id,
        trace_type="mesh",
        name=name,
        data=payload,
        style=style,
        hover=hover,
        visible=True,
    )


def make_aromatic_rings_trace(
    atoms: Atoms,
    cycles: list[list[int]] | None = None,
    style_circles: str | dict | None = "aromatic_rings",
    style_highlights: str | dict | None = "aromatic_highlights",
    n_circle_points: int = 72,
) -> dict[str, dict[str, Any]]:
    """Build aromatic ring circle outlines (line_segments) + fill discs (mesh).

    Parameters
    ----------
    atoms
        ASE Atoms with covalent bonds (calls ``update_with_covalent_bonds`` if missing).
    cycles
        Precomputed aromatic cycles as list of atom-index lists.  If *None* the
        function detects them via ``analyze.find_aromatic_cycles_in_sgraph``.
    style_circles, style_highlights
        Style name from ``DEFAULT_STYLES`` or a raw style dict.
    n_circle_points
        Number of vertices per circle outline and per fill disc polygon.

    Returns
    -------
    dict  {"circles": trace_dict, "highlights": trace_dict}
    """
    from .analyze import find_aromatic_cycles_in_sgraph, sgraph_from_atoms
    from scipy.spatial.transform import Rotation as _Rot

    if "covalent_bonds" not in atoms.info:
        from .analyze import update_with_covalent_bonds
        atoms = update_with_covalent_bonds(atoms)

    if cycles is None:
        try:
            cycles = find_aromatic_cycles_in_sgraph(sgraph_from_atoms(atoms))
        except Exception:
            cycles = []

    if not cycles:
        empty_circles = _make_trace("aromatic_rings", "line_segments", "Aromatic rings",
                                   {"positions": [], "n_segments": 0}, style=style_circles)
        empty_highlights = _make_trace("aromatic_highlights", "mesh", "Aromatic highlights",
                                       {"positions": [], "indices": [],
                                        "n_vertices": 0, "n_triangles": 0}, style=style_highlights)
        return {"circles": empty_circles, "highlights": empty_highlights}

    positions = atoms.get_positions()
    theta = np.linspace(0, 2 * np.pi, n_circle_points, endpoint=False)
    unit_circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

    line_positions: list[float] = []
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []

    for ring in cycles:
        ring_pos = positions[ring]
        center = ring_pos.mean(axis=0)
        edge1 = ring_pos[1] - ring_pos[0]
        edge2 = ring_pos[2] - ring_pos[1]
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        radius = np.mean(np.linalg.norm(ring_pos - center, axis=1))

        rot, _ = _Rot.align_vectors(normal.reshape(-1, 3), np.array([[0.0, 0.0, 1.0]]))

        # Circle outline at ~40% of ring radius (chemistry textbook style)
        inner_radius = radius * 0.4
        circle_3d = rot.apply(unit_circle * inner_radius) + center
        for i in range(n_circle_points):
            j = (i + 1) % n_circle_points
            line_positions.extend(circle_3d[i].tolist())
            line_positions.extend(circle_3d[j].tolist())

        # Fill polygon — project ring atoms onto best-fit plane, fan-triangulate
        # from center.  Vertices are the actual ring atom positions (not a regular
        # n-gon), so they align perfectly with stick bond intersection points.
        ring_centered = ring_pos - center
        # Project onto ring plane (remove normal component)
        ring_proj = ring_centered - np.outer(np.dot(ring_centered, normal), normal)
        # Fan triangulation: center point + consecutive vertex pairs
        n_sides = len(ring)
        verts = np.vstack([ring_proj + center, center.reshape(1, -1)])
        faces = np.array(
            [[n_sides, i, (i + 1) % n_sides] for i in range(n_sides)],
            dtype=int,
        )
        all_verts.append(verts)
        all_faces.append(faces)

    circles_trace = _make_trace(
        "aromatic_rings", "line_segments", "Aromatic rings",
        {"positions": line_positions, "n_segments": len(cycles) * n_circle_points},
        style=style_circles,
    )

    if all_verts:
        v_offsets = np.cumsum([0] + [len(v) for v in all_verts[:-1]])
        merged_verts = np.vstack(all_verts)
        merged_faces = np.vstack([
            f + off for f, off in zip(all_faces, v_offsets)
        ])
        payload = {
            "positions": merged_verts.reshape(-1).tolist(),
            "indices": merged_faces.reshape(-1).tolist(),
            "n_vertices": int(len(merged_verts)),
            "n_triangles": int(len(merged_faces)),
        }
    else:
        payload = {"positions": [], "indices": [], "n_vertices": 0, "n_triangles": 0}

    highlights_trace = _make_trace(
        "aromatic_highlights", "mesh", "Aromatic highlights",
        payload, style=style_highlights,
    )

    return {"circles": circles_trace, "highlights": highlights_trace}


def make_maxima_trace(
    seg: dict[str, Any],
    atoms: Atoms,
    style: str | dict | None = "internal_maxima",
    name: str = "Maxima",
    trace_id: str = "maxima",
    cscale: list[list[float]] | None = None,
) -> dict[str, Any]:
    payload = _serialize_maxima_trace(seg, atoms, cscale=cscale)
    # Some systems classify all maxima as periodic; keep a visible local-maxima
    # scatter by mirroring periodic points into the internal trace as fallback.
    if payload["internal_n_points"] == 0 and payload["outside_n_points"] > 0:
        payload["internal_positions"] = payload["outside_positions"]
        payload["internal_colors"] = payload["outside_colors"]
        payload["internal_sizes"] = payload["outside_sizes"]
        payload["internal_n_points"] = payload["outside_n_points"]
        payload["internal_hover"] = payload["outside_hover"]
    # Internal maxima points
    internal = _make_trace(
        trace_id=f"{trace_id}_internal",
        trace_type="points",
        name=f"{name} (internal)",
        data={
            "positions": payload["internal_positions"],
            "colors": payload["internal_colors"],
            "sizes": payload["internal_sizes"],
            "n_points": payload["internal_n_points"],
        },
        style=style,
        hover=payload["internal_hover"],
        visible=True,
    )
    # Outside maxima points
    outside = _make_trace(
        trace_id=f"{trace_id}_outside",
        trace_type="points",
        name=f"{name} (periodic)",
        data={
            "positions": payload["outside_positions"],
            "colors": payload["outside_colors"],
            "sizes": payload["outside_sizes"],
            "n_points": payload["outside_n_points"],
        },
        style="periodic_maxima",
        hover=payload["outside_hover"],
        visible=True,
    )
    pbc = _make_trace(
        trace_id=f"{trace_id}_pbc",
        trace_type="points",
        name=f"{name} (PBC reps)",
        data={
            "positions": payload["pbc_positions"],
            "colors": payload["pbc_colors"],
            "sizes": payload["pbc_sizes"],
            "n_points": payload["pbc_n_points"],
        },
        style=style,
        hover=payload["pbc_hover"],
        visible=True,
    )
    return {"internal": internal, "outside": outside, "pbc": pbc, "payload": payload}


def make_windows_trace(
    seg: dict[str, Any],
    style: str | dict | None = "internal_windows",
    name: str = "Windows",
    trace_id: str = "windows",
    atoms = None,
    spacing: float | None = None,
) -> dict[str, Any]:
    payload = _serialize_windows_trace(seg, atoms=atoms, spacing=spacing)
    internal = _make_trace(
        trace_id=f"{trace_id}_internal",
        trace_type="points",
        name=f"{name} (internal)",
        data={
            "positions": payload["internal_positions"],
            "n_points": payload["internal_n_points"],
        },
        style="internal_windows",
        hover=payload["internal_hover"],
        visible=True,
    )
    periodic = _make_trace(
        trace_id=f"{trace_id}_periodic",
        trace_type="points",
        name=f"{name} (periodic)",
        data={
            "positions": payload["periodic_positions"],
            "n_points": payload["periodic_n_points"],
        },
        style="periodic_windows",
        hover=payload["periodic_hover"],
        visible=True,
    )
    return {"internal": internal, "periodic": periodic, "payload": payload}


def make_connections_trace(
    seg: dict[str, Any],
    atoms: Atoms,
    style: str | dict | None = "internal_connections",
    name: str = "Connections",
    trace_id: str = "connections",
) -> dict[str, Any]:
    payload = _serialize_connection_traces(seg, atoms)
    internal_lines = _make_trace(
        trace_id=f"{trace_id}_internal",
        trace_type="line_segments",
        name=f"{name} (internal)",
        data={
            "positions": payload["internal_segments"],
            "n_segments": len(payload["internal_segments"]) // 6,
        },
        style="internal_connections",
        hover=payload["internal_hover"],
        visible=True,
    )
    periodic_lines = _make_trace(
        trace_id=f"{trace_id}_periodic",
        trace_type="line_segments",
        name=f"{name} (periodic)",
        data={
            "positions": payload["periodic_segments"],
            "n_segments": len(payload["periodic_segments"]) // 6,
        },
        style="periodic_connections",
        hover=payload["periodic_hover"],
        visible=True,
    )
    internal_nodes = _make_trace(
        trace_id=f"{trace_id}_internal_nodes",
        trace_type="points",
        name=f"{name} internal nodes",
        data={
            "positions": payload["internal_points"],
            "n_points": len(payload["internal_points"]) // 3,
        },
        style="internal_nodes",
        hover=payload["internal_points_hover"],
        visible=True,
    )
    periodic_nodes = _make_trace(
        trace_id=f"{trace_id}_periodic_nodes",
        trace_type="points",
        name=f"{name} periodic nodes",
        data={
            "positions": payload["periodic_points"],
            "n_points": len(payload["periodic_points"]) // 3,
        },
        style="periodic_nodes",
        hover=payload["periodic_points_hover"],
        visible=True,
    )
    return {
        "internal_lines": internal_lines,
        "periodic_lines": periodic_lines,
        "internal_nodes": internal_nodes,
        "periodic_nodes": periodic_nodes,
        "payload": payload,
    }


def make_guest_trace(
    atoms: Atoms,
    style: str | dict | None = "guests",
    name: str = "Adsorbates",
    trace_id: str = "guests",
) -> dict[str, Any]:
    atomic_numbers = atoms.get_atomic_numbers()
    from ase.data.colors import cpk_colors

    syms = atoms.get_chemical_symbols()
    colors = np.asarray(cpk_colors[atomic_numbers], dtype=float)
    radii = covalent_radii[atomic_numbers] * 0.8
    positions = atoms.get_positions().reshape(-1).tolist()
    return _make_trace(
        trace_id=trace_id,
        trace_type="instanced_spheres",
        name=name,
        data={
            "positions": positions,
            "symbols": syms,
            "colors": colors.reshape(-1).tolist(),
            "radii": radii.astype(float).tolist(),
            "n_atoms": int(len(atoms)),
        },
        style=style,
        hover=None,
        visible=True,
    )


def make_framework_instanced_trace(
    atoms: Atoms,
    bonds: str | list | None = "auto",
    style: str | dict | None = "framework",
    name: str = "Framework",
    trace_id: str = "framework",
    atom_scale: float = 0.65,
    bond_color: list[float] | None = None,
    representation: str = "ball_and_stick",
) -> dict[str, dict[str, Any]]:
    """Create framework as InstancedMesh spheres + cylinder mesh bonds.

    Sends compact atom data (~30 KB) for instanced spheres + triangulated
    cylinders for bonds (~20 MB).  The browser builds sphere geometry locally.

    representation: "ball_and_stick" (default) or "stick" (two-tone bonds).
    """
    from ase.data.colors import cpk_colors

    # Resolve bonds
    if bonds == "auto":
        if "covalent_bonds" in atoms.info and atoms.info["covalent_bonds"]:
            bond_list = atoms.info["covalent_bonds"]
        else:
            cutoffs = np.asarray(natural_cutoffs(atoms, mult=1.05), dtype=float)
            i_idx, j_idx = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
            seen: set[tuple[int, int]] = set()
            bond_list = []
            for i, j in zip(i_idx, j_idx, strict=False):
                x, y = int(min(i, j)), int(max(i, j))
                if x == y or (x, y) in seen:
                    continue
                seen.add((x, y))
                bond_list.append([x, y])
            atoms.info["covalent_bonds"] = bond_list
    elif bonds is None:
        bond_list = []
    elif isinstance(bonds, list):
        bond_list = bonds
    else:
        bond_list = []

    n_atoms = int(len(atoms))
    atomic_numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Atom colors (CPK)
    atom_colors = np.array([cpk_colors[an] for an in atomic_numbers], dtype=float)

    # Atom radii (covalent radii × scale)
    radii = covalent_radii[atomic_numbers] * atom_scale

    # Atom trace (InstancedMesh — one draw call for all atoms)
    atom_trace = _make_trace(
        trace_id=f"{trace_id}_atoms",
        trace_type="instanced_spheres",
        name=f"{name} (atoms)",
        data={
            "positions": positions.reshape(-1).tolist(),
            "symbols": list(symbols),
            "colors": atom_colors.reshape(-1).tolist(),
            "radii": radii.astype(float).tolist(),
            "n_atoms": n_atoms,
        },
        style=style if style else "framework",
        hover=None,
        visible=True,
    )

    # Bond trace (triangulated cylinders for proper 3D appearance)
    if bond_list and len(bond_list) > 0:
        from .visualize import mesh_for_atoms
        cyl_radius = 0.08 if representation == "stick" else 0.15
        bond_mesh = mesh_for_atoms(
            atoms,
            bonds=bond_list,
            kwargs_spheres={"scale": 0.001},  # nearly invisible spheres
            kwargs_cylinders={"radius": cyl_radius},
            representation=representation,  # "stick" → two-tone CPK colors
            sphere_sections=1,
            cylinder_sections=8,
        )
        bond_payload = _serialize_trimesh(bond_mesh)
        # Filter to only cylinder vertices (bond-colored, not sphere-colored)
        # Keep all vertices — the tiny spheres are negligible
        bond_trace = _make_trace(
            trace_id=f"{trace_id}_bonds",
            trace_type="mesh",
            name=f"{name} (bonds)",
            data=bond_payload,
            style={"opacity": 1.0, "roughness": 0.38, "side": "DoubleSide"},
            hover=None,
            visible=True,
        )
    else:
        bond_trace = _make_trace(
            trace_id=f"{trace_id}_bonds",
            trace_type="mesh",
            name=f"{name} (bonds)",
            data={"positions": [], "indices": [], "n_vertices": 0, "n_triangles": 0},
            style={"opacity": 1.0},
            hover=None,
            visible=True,
        )

    return {"atoms": atom_trace, "bonds": bond_trace}


def _minimum_image_displacement(
    p: np.ndarray, q: np.ndarray, cell: np.ndarray | None
) -> np.ndarray:
    if cell is None or np.asarray(cell).size == 0:
        return np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    C = np.asarray(cell, dtype=float)
    invC = np.linalg.inv(C.T)
    df = np.dot(np.asarray(p, dtype=float) - np.asarray(q, dtype=float), invC.T)
    df -= np.round(df)
    return np.dot(df, C)


def _strip_framework_like_atoms(
    component_atoms: Atoms, framework_atoms: Atoms, tol: float = 0.08
) -> Atoms:
    """Remove framework-like atoms from an allcomponents snapshot.

    Matches atoms by element + minimum-image distance to framework atoms.
    """
    if len(component_atoms) == 0 or len(framework_atoms) == 0:
        return component_atoms
    comp_syms = component_atoms.get_chemical_symbols()
    fw_syms = framework_atoms.get_chemical_symbols()
    comp_pos = np.asarray(component_atoms.get_positions(), dtype=float)
    fw_pos = np.asarray(framework_atoms.get_positions(), dtype=float)
    cell = (
        framework_atoms.get_cell().array
        if framework_atoms.cell is not None and np.asarray(framework_atoms.get_cell().array).size
        else None
    )

    fw_by_sym: dict[str, list[int]] = {}
    for j, s in enumerate(fw_syms):
        fw_by_sym.setdefault(str(s), []).append(j)

    keep = np.ones(len(component_atoms), dtype=bool)
    matched = 0
    for i, s in enumerate(comp_syms):
        cands = fw_by_sym.get(str(s), [])
        if not cands:
            continue
        pi = comp_pos[i]
        best_j = -1
        best_d = float("inf")
        for j in cands:
            dvec = _minimum_image_displacement(pi, fw_pos[j], cell)
            d = float(np.linalg.norm(dvec))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_d <= tol:
            keep[i] = False
            matched += 1
            fw_by_sym[str(s)].remove(best_j)

    # Keep original trajectory if match confidence is low.
    if matched < max(4, int(0.4 * len(framework_atoms))):
        return component_atoms
    return component_atoms[keep]


def make_guest_bonds_trace(
    atoms: Atoms,
    style: str | dict | None = "guest_bonds",
    name: str = "Adsorbate bonds",
    trace_id: str = "guest_bonds",
) -> dict[str, Any]:
    """Build cylindrical bond mesh for polyatomic adsorbates."""
    if len(atoms) <= 1:
        return _make_trace(
            trace_id=trace_id,
            trace_type="mesh",
            name=name,
            data={"positions": [], "indices": [], "colors": [], "n_vertices": 0, "n_triangles": 0},
            style=style,
            hover=None,
            visible=True,
        )
    from .visualize import cylinders_for_bonds

    cutoffs = np.asarray(natural_cutoffs(atoms, mult=1.15), dtype=float)
    i_idx, j_idx = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    seen: set[tuple[int, int]] = set()
    bonds: list[list[int]] = []
    for i, j in zip(i_idx, j_idx, strict=False):
        a = int(min(i, j))
        b = int(max(i, j))
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        # Keep short bonds only (avoid occasional PBC-neighbor artifacts).
        if float(np.linalg.norm(pos[a] - pos[b])) <= 2.2:
            bonds.append([a, b])
    if not bonds:
        return _make_trace(
            trace_id=trace_id,
            trace_type="mesh",
            name=name,
            data={"positions": [], "indices": [], "colors": [], "n_vertices": 0, "n_triangles": 0},
            style=style,
            hover=None,
            visible=True,
        )
    cyl = cylinders_for_bonds(
        bonds,
        atoms,
        radius=0.12,
        bond_rgba_array=[127, 128, 129, 255],
        cylinder_resolution=12,
    )
    mesh = trimesh.util.concatenate(cyl)
    payload = _serialize_trimesh(mesh)
    return _make_trace(
        trace_id=trace_id,
        trace_type="mesh",
        name=name,
        data=payload,
        style=style,
        hover=None,
        visible=True,
    )


def make_cell_trace(
    atoms: Atoms,
    style: str | dict | None = "cell",
    name: str = "Unit cell",
    trace_id: str = "cell",
) -> dict[str, Any]:
    payload = _serialize_cell_lines(atoms)
    return _make_trace(
        trace_id=trace_id,
        trace_type="unit_cell_lines",
        name=name,
        data=payload,
        style=style,
        hover=None,
        visible=True,
    )


# ---------------------------------------------------------------------------
# MovieFigure class
# ---------------------------------------------------------------------------

_MARKER_SHAPES = frozenset({
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "cross", "x", "star", "pentagon", "hexagon", "ring",
})


class _MarkerProxy:
    """Plotly-like marker accessor: fig.traces['name'].marker.size = 12"""

    def __init__(self, trace: dict) -> None:
        self._trace = trace

    @property
    def size(self) -> float | None:
        s = self._trace.get("style", {})
        return s.get("size")

    @size.setter
    def size(self, val: float) -> None:
        self._trace.setdefault("style", {})["size"] = val

    @property
    def color(self):
        s = self._trace.get("style", {})
        return s.get("color")

    @color.setter
    def color(self, val) -> None:
        self._trace.setdefault("style", {})["color"] = val

    @property
    def opacity(self) -> float | None:
        s = self._trace.get("style", {})
        return s.get("opacity")

    @opacity.setter
    def opacity(self, val: float) -> None:
        self._trace.setdefault("style", {})["opacity"] = val

    @property
    def symbol(self) -> str | None:
        s = self._trace.get("style", {})
        return s.get("symbol", "circle")

    @symbol.setter
    def symbol(self, val: str) -> None:
        if val not in _MARKER_SHAPES:
            raise ValueError(f"Unknown marker symbol '{val}'. Choose from: {sorted(_MARKER_SHAPES)}")
        self._trace.setdefault("style", {})["symbol"] = val


class _LineProxy:
    """Plotly-like line accessor: fig.traces['name'].line.width = 3"""

    def __init__(self, trace: dict) -> None:
        self._trace = trace

    @property
    def width(self) -> float | None:
        s = self._trace.get("style", {})
        return s.get("linewidth")

    @width.setter
    def width(self, val: float) -> None:
        self._trace.setdefault("style", {})["linewidth"] = val

    @property
    def color(self):
        s = self._trace.get("style", {})
        return s.get("color")

    @color.setter
    def color(self, val) -> None:
        self._trace.setdefault("style", {})["color"] = val

    @property
    def dash(self) -> str | None:
        s = self._trace.get("style", {})
        if "dashSize" in s:
            return "dash"
        return "solid"

    @dash.setter
    def dash(self, val: str) -> None:
        s = self._trace.setdefault("style", {})
        if val == "dash":
            s.setdefault("dashSize", 0.9)
            s.setdefault("gapSize", 0.5)
        else:
            s.pop("dashSize", None)
            s.pop("gapSize", None)


class _MeshProxy:
    """Plotly-like mesh accessor: fig.traces['name'].mesh.roughness = 0.3"""

    def __init__(self, trace: dict) -> None:
        self._trace = trace

    @property
    def roughness(self) -> float | None:
        return self._trace.get("style", {}).get("roughness")

    @roughness.setter
    def roughness(self, val: float) -> None:
        self._trace.setdefault("style", {})["roughness"] = val

    @property
    def metalness(self) -> float | None:
        return self._trace.get("style", {}).get("metalness")

    @metalness.setter
    def metalness(self, val: float) -> None:
        self._trace.setdefault("style", {})["metalness"] = val

    @property
    def opacity(self) -> float | None:
        return self._trace.get("style", {}).get("opacity")

    @opacity.setter
    def opacity(self, val: float) -> None:
        self._trace.setdefault("style", {})["opacity"] = val

    @property
    def clearcoat(self) -> float | None:
        return self._trace.get("style", {}).get("clearcoat")

    @clearcoat.setter
    def clearcoat(self, val: float) -> None:
        self._trace.setdefault("style", {})["clearcoat"] = val

    @property
    def side(self) -> str | None:
        return self._trace.get("style", {}).get("side")

    @side.setter
    def side(self, val: str) -> None:
        self._trace.setdefault("style", {})["side"] = val


class _TraceProxy:
    """Plotly-like trace accessor returned by fig.traces['trace_name'].

    Usage::

        fig.traces['maxima_internal'].marker.size = 12
        fig.traces['maxima_internal'].marker.color = [1, 0, 0]
        fig.traces['maxima_internal'].marker.symbol = 'square'
        fig.traces['connections_internal'].line.width = 3
        fig.traces['surface'].mesh.roughness = 0.2
        fig.traces['surface'].visible = False
    """

    def __init__(self, trace: dict) -> None:
        self._trace = trace
        self.marker = _MarkerProxy(trace)
        self.line = _LineProxy(trace)
        self.mesh = _MeshProxy(trace)

    @property
    def name(self) -> str:
        return self._trace.get("name", "")

    @name.setter
    def name(self, val: str) -> None:
        self._trace["name"] = val

    @property
    def visible(self) -> bool:
        return self._trace.get("visible", True)

    @visible.setter
    def visible(self, val: bool) -> None:
        self._trace["visible"] = val

    @property
    def type(self) -> str:
        return self._trace.get("type", "")

    @property
    def id(self) -> str:
        return self._trace.get("id", "")


class _TraceCollection:
    """Dict-like accessor: fig.traces['trace_name'] returns a _TraceProxy."""

    def __init__(self, data: list[dict], frames: list[dict]) -> None:
        self._data = data
        self._frames = frames

    def _find(self, key: str) -> dict | None:
        for t in self._data:
            if t.get("id") == key or t.get("name") == key:
                return t
        return None

    def __getitem__(self, key: str) -> _TraceProxy:
        trace = self._find(key)
        if trace is None:
            raise KeyError(f"Trace '{key}' not found. Available: {[t.get('id') for t in self._data]}")
        return _TraceProxy(trace)

    def __contains__(self, key: str) -> bool:
        return self._find(key) is not None

    def keys(self) -> list[str]:
        return [t.get("id", "") for t in self._data]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        ids = [t.get("id", "?") for t in self._data]
        return f"TraceCollection({ids})"


def _pack_static_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack large numerical arrays in static traces as base64 binary blobs.

    Same logic as _pack_frame_data, applied to the static trace list.
    Reduces JSON size by ~40% for mesh-heavy payloads.
    """
    packed: list[dict[str, Any]] = []
    for trace in traces:
        pt = dict(trace)
        data = trace.get("data")
        if isinstance(data, dict):
            pd: dict[str, Any] = {}
            for key, value in data.items():
                if key in _BINARY_PACK_KEYS and isinstance(value, list) and len(value) > 0:
                    if key == "indices":
                        max_val = max(value) if value else 0
                        dtype = np.uint16 if max_val < 65536 else np.uint32
                        arr = np.asarray(value, dtype=dtype)
                        pd[key] = {
                            "__binary__": True,
                            "dtype": "uint16" if dtype == np.uint16 else "uint32",
                            "count": len(value),
                            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
                        }
                    else:
                        arr = np.asarray(value, dtype=np.float32)
                        pd[key] = {
                            "__binary__": True,
                            "dtype": "float32",
                            "count": len(value),
                            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
                        }
                else:
                    pd[key] = value
            pt["data"] = pd
        packed.append(pt)
    return packed


class MovieFigure:
    """Plotly-like API for building Three.js movie scenes."""

    def __init__(
        self,
        data: list[dict[str, Any]] | None = None,
        layout: dict[str, Any] | None = None,
        frames: list[dict[str, Any]] | None = None,
        groups: dict[str, Any] | None = None,
        styles: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self.data = list(data) if data is not None else []
        self.layout = dict(layout) if layout is not None else {}
        self.frames = list(frames) if frames is not None else []
        self.groups = dict(groups) if groups is not None else {}
        self.styles = dict(DEFAULT_STYLES)
        if styles is not None:
            self.styles.update(styles)
        self.meta = dict(meta) if meta is not None else {}

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_trace(self, trace: dict[str, Any], frame: int | None = None) -> "MovieFigure":
        if frame is None:
            self.data.append(trace)
        else:
            while len(self.frames) <= frame:
                self.frames.append({"frame_index": len(self.frames), "traces": []})
            self.frames[frame]["traces"].append(trace)
        return self

    def add_traces(self, traces: list[dict[str, Any]], frame: int | None = None) -> "MovieFigure":
        for t in traces:
            self.add_trace(t, frame=frame)
        return self

    def update_traces(
        self,
        patch: dict[str, Any],
        selector: dict[str, Any] | None = None,
        frame: int | None = None,
    ) -> "MovieFigure":
        targets = self.data if frame is None else self.frames[frame]["traces"]
        for t in targets:
            if selector is None or all(t.get(k) == v for k, v in selector.items()):
                merged = dict(patch)
                style_ov = {
                    k: merged.pop(k)
                    for k in list(merged.keys())
                    if k in _TRACE_STYLE_TOPLEVEL_KEYS
                }
                if style_ov:
                    st = t.get("style")
                    if isinstance(st, dict):
                        st.update(style_ov)
                    else:
                        base = dict(_resolve_style(st))
                        base.update(style_ov)
                        t["style"] = base
                if merged:
                    t.update(merged)
        return self

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = MovieFigure._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def update_layout(self, patch: dict[str, Any]) -> "MovieFigure":
        self.layout = self._deep_merge(self.layout, patch)
        return self

    def set_camera(
        self,
        eye: tuple[float, float, float] | None = None,
        center: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
        projection: str | None = None,
        keyframes: list[dict[str, Any]] | None = None,
    ) -> "MovieFigure":
        if "scene" not in self.layout:
            self.layout["scene"] = {}
        cam = self.layout["scene"].setdefault("camera", {})
        if eye is not None:
            cam["eye"] = list(eye)
        if center is not None:
            cam["center"] = list(center)
        if up is not None:
            cam["up"] = list(up)
        if projection is not None:
            cam["projection"] = projection
        if keyframes is not None:
            cam["keyframes"] = keyframes
        return self

    def set_appearance(
        self,
        background: str | None = None,
        scene_bg: str | int | None = None,
        font_family: str | None = None,
        label_font_size: int | None = None,
        title: str | None = None,
        theme: str | None = None,
    ) -> "MovieFigure":
        if "appearance" not in self.layout:
            self.layout["appearance"] = {}
        ap = self.layout["appearance"]
        if background is not None:
            ap["background"] = background
        if scene_bg is not None:
            ap["scene_bg"] = scene_bg
        if font_family is not None:
            ap["font_family"] = font_family
        if label_font_size is not None:
            ap["label_font_size"] = label_font_size
        if title is not None:
            ap["title"] = title
        if theme is not None:
            ap["theme"] = theme
        return self

    @staticmethod
    def parse_plotly_colors(colors: list[str] | str) -> list[list[float]]:
        import re
        if isinstance(colors, str):
            import plotly.express as px
            colors = getattr(px.colors.qualitative, colors, [colors])
        out = []
        for s in colors:
            s = s.strip()
            if s.startswith("#"):
                h = s.lstrip("#")
                out.append([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
            elif s.startswith("rgb("):
                m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", s)
                if m:
                    out.append([int(m.group(i)) / 255.0 for i in (1, 2, 3)])
            else:
                h = s.lstrip("#")
                try:
                    out.append([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
                except (ValueError, IndexError):
                    out.append([0.7, 0.7, 0.7])
        return out

    def set_colorscale(self, colors: list[str] | str, trace_type: str = "surface") -> "MovieFigure":
        rgb = self.parse_plotly_colors(colors)
        if "colorscale" not in self.layout:
            self.layout["colorscale"] = {}
        self.layout["colorscale"][trace_type] = rgb
        return self

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @property
    def traces(self) -> _TraceCollection:
        return _TraceCollection(self.data, self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "frames": self.frames,
            "static": {"traces": self.data},
            "groups": self.groups,
            "styles": self.styles,
            "layout": self.layout,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MovieFigure":
        return cls(
            data=d.get("static", {}).get("traces", []),
            layout=d.get("layout", {}),
            frames=d.get("frames", []),
            groups=d.get("groups", {}),
            styles=d.get("styles", {}),
            meta=d.get("meta", {}),
        )

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------

    def to_html(
        self,
        path: str | Path,
        template: str = "threejs_default",
        sidecar: bool = False,
        lazy_surfaces: bool = False,
        fps: float = 20.0,
        point_size: float = 0.16,
    ) -> Path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        scene = self.to_dict()

        # Lazy surfaces: extract surface trace data to separate file
        if lazy_surfaces and "static" in scene:
            surface_ids = {"surface", "surface_pbc", "surface_pocket"}
            lazy_data: dict[str, dict] = {}
            for trace in scene["static"].get("traces", []):
                tid = trace.get("id", "")
                if tid in surface_ids or (tid and tid.startswith("surface_")):
                    if "data" in trace:
                        lazy_data[tid] = trace["data"]
                        trace["data"] = {"__sidecar__": True, "id": tid}
            if lazy_data:
                sidecar_path = out.with_suffix(".surfaces.json")
                with open(sidecar_path, "w", encoding="utf-8") as fh:
                    json.dump(lazy_data, fh)
                scene["sidecar_surfaces"] = str(sidecar_path.name)

        if sidecar:
            frames_data = scene.pop("frames")
            sidecar_path = out.with_suffix(".frames.jsonl")
            with open(sidecar_path, "w", encoding="utf-8") as fh:
                for fr in frames_data:
                    fh.write(json.dumps(_pack_frame_data(fr)) + "\n")
            scene["sidecar"] = str(sidecar_path.name)
            # Build byte-offset index for per-frame HTTP Range streaming
            offsets: list[int] = []
            with open(sidecar_path, "rb") as fh:
                offset = 0
                for line in fh:
                    offsets.append(offset)
                    offset += len(line)
            idx_path = out.with_suffix(".frames.idx")
            with open(idx_path, "w", encoding="utf-8") as fh:
                json.dump(offsets, fh)
            scene["sidecar_idx"] = str(idx_path.name)

        # Pack large numerical arrays as base64 binary blobs (~29% smaller JSON)
        if "static" in scene and "traces" in scene["static"]:
            scene["static"]["traces"] = _pack_static_traces(scene["static"]["traces"])
        payload_json = json.dumps(scene)

        # Default camera
        cam = scene.get("layout", {}).get("scene", {}).get("camera", {})
        eye = tuple(cam.get("eye", [2.0, 0.0, 1.35]))
        center = tuple(cam.get("center", [0.0, 0.0, 0.0]))
        up = tuple(cam.get("up", [0.0, 0.0, 1.0]))
        projection = str(cam.get("projection", "perspective"))

        ap = scene.get("layout", {}).get("appearance", {})
        theme = ap.get("theme", "dark")
        is_light = theme == "light"
        bg_css = ap.get("background", "#ffffff" if is_light else "radial-gradient(circle at center, #7a7a7a 0%, #111111 100%)")
        scene_bg = ap.get("scene_bg", 0xffffff if is_light else 0x111111)
        font_family = ap.get("font_family", '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif')
        label_font_size = ap.get("label_font_size", 11)
        title = ap.get("title", "")
        text_color = "#222222" if is_light else "#efefef"
        panel_bg = "rgba(240, 240, 240, 0.9)" if is_light else "rgba(20, 20, 20, 0.85)"
        if isinstance(scene_bg, int):
            scene_bg = hex(scene_bg)

        html = _THREEJS_TEMPLATE.render(
            payload_json=payload_json,
            fps=fps,
            eye=eye,
            center=center,
            up=up,
            point_size=point_size,
            projection=projection,
            bg_css=bg_css,
            scene_bg=scene_bg,
            font_family=font_family,
            label_font_size=label_font_size,
            title=title,
            text_color=text_color,
            panel_bg=panel_bg,
        )
        out.write_text(html, encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    # Jupyter integration
    # ------------------------------------------------------------------

    def _render_html_string(self) -> str:
        """Render scene to HTML string (no file write)."""
        import json as _json
        scene = self.to_dict()
        # Pack large numerical arrays as base64 binary blobs (~40% smaller JSON)
        if "static" in scene and "traces" in scene["static"]:
            scene["static"]["traces"] = _pack_static_traces(scene["static"]["traces"])
        payload_json = _json.dumps(scene)
        cam = scene.get("layout", {}).get("scene", {}).get("camera", {})
        eye = tuple(cam.get("eye", [2.0, 0.0, 1.35]))
        center = tuple(cam.get("center", [0.0, 0.0, 0.0]))
        up = tuple(cam.get("up", [0.0, 0.0, 1.0]))
        projection = str(cam.get("projection", "perspective"))
        fps = float(scene.get("layout", {}).get("fps", 20.0))

        ap = scene.get("layout", {}).get("appearance", {})
        theme = ap.get("theme", "dark")
        is_light = theme == "light"
        bg_css = ap.get("background", "#ffffff" if is_light else "radial-gradient(circle at center, #7a7a7a 0%, #111111 100%)")
        scene_bg = ap.get("scene_bg", 0xffffff if is_light else 0x111111)
        font_family = ap.get("font_family", '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif')
        label_font_size = ap.get("label_font_size", 11)
        title = ap.get("title", "")
        text_color = "#222222" if is_light else "#efefef"
        panel_bg = "rgba(240, 240, 240, 0.9)" if is_light else "rgba(20, 20, 20, 0.85)"
        if isinstance(scene_bg, int):
            scene_bg = hex(scene_bg)

        return _THREEJS_TEMPLATE.render(
            payload_json=payload_json,
            fps=fps,
            eye=eye,
            center=center,
            up=up,
            point_size=0.16,
            projection=projection,
            bg_css=bg_css,
            scene_bg=scene_bg,
            font_family=font_family,
            label_font_size=label_font_size,
            title=title,
            text_color=text_color,
            panel_bg=panel_bg,
        )

    def _repr_html_(self) -> str:
        """Jupyter notebook integration — display via IFrame (avoids CSP issues).

        Jupyter auto-calls this method when the figure is the last expression
        in a cell.  Uses the ``srcdoc`` IFrame pattern (same approach as pydeck
        and Plotly's notebook renderer) to isolate Three.js/WebGL content from
        the notebook's Content Security Policy.
        """
        import html
        html_str = self._render_html_string()
        escaped = html.escape(html_str)
        return (
            f'<iframe width="100%" height="600" frameborder="0" '
            f'srcdoc="{escaped}"></iframe>'
        )

    def show(self, renderer: str = "notebook") -> None:
        """Display the figure.

        Parameters
        ----------
        renderer
            ``"notebook"`` — display inline in Jupyter output cell (default).
            ``"browser"`` — write a temporary HTML file and open it in the
            system's default web browser (like ``plotly.io.show(renderer="browser")``).
        """
        if renderer == "browser":
            import tempfile
            import webbrowser
            tmp = Path(tempfile.mktemp(suffix=".html"))
            self.to_html(tmp)
            webbrowser.open(str(tmp))
        else:
            try:
                from IPython.display import HTML, display
                display(HTML(self._repr_html_()))
            except ImportError:
                raise RuntimeError(
                    "IPython is required for notebook rendering. "
                    "Use renderer='browser' to open in a web browser instead."
                )

    # ------------------------------------------------------------------
    # Convenience trace builders (static scenes)
    # ------------------------------------------------------------------

    def add_framework(
        self,
        atoms: Atoms,
        bonds: str | list | None = "auto",
        representation: str = "ball_and_stick",
        kwargs_spheres: dict[str, Any] | None = None,
        kwargs_cylinders: dict[str, Any] | None = None,
        sphere_sections: int = 2,
        cylinder_sections: int = 15,
        generate_all_reps: bool = True,
        **style: Any,
    ) -> "MovieFigure":
        """Add framework atoms + bonds as a mesh trace (static).

        representation: "ball_and_stick" (default), "stick", or
        "space_filling".

        When *generate_all_reps* is ``True`` and representation is
        ``"ball_and_stick"``, stick and space-filling meshes are also
        generated (hidden, toggleable in viewer).  Set ``False`` for large
        structures to avoid 3× generation cost — precomputes ~2.6s for
        P1-Cu-BTC vs ~0.9s without alternates.
        """
        shared = {
            "sphere_sections": sphere_sections,
            "cylinder_sections": cylinder_sections,
        }
        if representation == "ball_and_stick":
            trace = make_framework_trace(
                atoms,
                bonds=bonds,
                style=style if style else "framework",
                kwargs_spheres=kwargs_spheres,
                kwargs_cylinders=kwargs_cylinders,
                **shared,
            )
            self.add_trace(trace)
            if generate_all_reps:
                # Also add stick and space-filling representations (start hidden)
                stick_trace = make_framework_trace(
                    atoms, bonds=bonds, trace_id="framework_stick",
                    name="Framework (stick)", representation="stick",
                    style={"visible": False, "side": "FrontSide"},
                    kwargs_spheres=kwargs_spheres,
                    kwargs_cylinders=kwargs_cylinders,
                    **shared,
                )
                self.add_trace(stick_trace)
                sf_trace = make_framework_trace(
                    atoms, bonds=bonds, trace_id="framework_spacefill",
                    name="Framework (spacefill)", representation="space_filling",
                    style={"visible": False, "side": "FrontSide"},
                    kwargs_spheres=kwargs_spheres,
                    kwargs_cylinders=kwargs_cylinders,
                    **shared,
                )
                self.add_trace(sf_trace)
        else:
            trace = make_framework_trace(
                atoms, bonds=bonds, representation=representation,
                style=style if style else "framework",
                kwargs_spheres=kwargs_spheres,
                kwargs_cylinders=kwargs_cylinders,
                **shared,
            )
            self.add_trace(trace)
        return self

    def add_aromatic_rings(
        self,
        atoms: Atoms,
        cycles: list[list[int]] | None = None,
        **style: Any,
    ) -> "MovieFigure":
        traces = make_aromatic_rings_trace(atoms, cycles=cycles)
        self.add_trace(traces["circles"])
        self.add_trace(traces["highlights"])
        return self

    def add_cell(self, atoms: Atoms, **style: Any) -> "MovieFigure":
        """Add unit-cell wireframe (static)."""
        trace = make_cell_trace(atoms, style=style if style else "cell")
        self.add_trace(trace)
        return self

    def add_surface(
        self,
        dgrid: np.ndarray,
        regions: np.ndarray,
        atoms: Atoms,
        isovalue: float = 1.0,
        step_size: int = 1,
        **style: Any,
    ) -> "MovieFigure":
        """Add segmentation isosurface mesh (static)."""
        cs = self.layout.get("colorscale", {}).get("surface")
        trace = make_surface_trace(
            dgrid, regions, atoms,
            isovalue=isovalue,
            step_size=step_size,
            style=style if style else "surface",
            cscale=cs,
        )
        self.add_trace(trace)
        return self

    def add_maxima(
        self,
        seg: dict[str, Any],
        atoms: Atoms,
        hide_periodic: bool = True,
        **style: Any,
    ) -> "MovieFigure":
        """Add pore-centre maxima points (internal always visible, periodic optionally hidden)."""
        cs = self.layout.get("colorscale", {}).get("surface")
        maxima = make_maxima_trace(seg, atoms, style=style if style else "internal_maxima", cscale=cs)
        self.add_trace(maxima["internal"])
        if hide_periodic:
            maxima["outside"]["visible"] = False
        self.add_trace(maxima["outside"])
        self.add_trace(maxima["pbc"])
        return self

    def add_connections(
        self,
        seg: dict[str, Any],
        atoms: Atoms,
        hide_periodic: bool = True,
        **style: Any,
    ) -> "MovieFigure":
        """Add pore-graph connection lines + nodes (internal + optional periodic)."""
        conns = make_connections_trace(seg, atoms, style=style if style else "internal_connections")
        self.add_trace(conns["internal_lines"])
        self.add_trace(conns["internal_nodes"])
        if hide_periodic:
            conns["periodic_lines"]["visible"] = False
            conns["periodic_nodes"]["visible"] = False
        self.add_trace(conns["periodic_lines"])
        self.add_trace(conns["periodic_nodes"])
        return self

    def add_windows(
        self,
        seg: dict[str, Any],
        hide_periodic: bool = True,
        atoms = None,
        spacing: float | None = None,
        **style: Any,
    ) -> "MovieFigure":
        """Add window points between connected pores (internal + optional periodic).

        When *atoms* and *spacing* are provided, windows inside internal
        connections that fall far outside the unit cell (beyond the wall
        tolerance) are reclassified as periodic, so they're hidden by default
        and don't appear as floating disconnected markers.
        """
        windows = make_windows_trace(seg, style=style if style else "internal_windows", atoms=atoms, spacing=spacing)
        self.add_trace(windows["internal"])
        if hide_periodic:
            windows["periodic"]["visible"] = False
        self.add_trace(windows["periodic"])
        return self

    def add_guests(
        self,
        atoms: Atoms,
        color: list[float] | None = None,
        **style: Any,
    ) -> "MovieFigure":
        """Add guest / adsorbate atoms as instanced spheres (single frame, static)."""
        trace = make_guest_trace(atoms, style=style if style else "guests")
        if color is not None:
            resolved = _resolve_style(style) if style else {}
            resolved["color"] = color
            trace["style"] = resolved
            trace["data"]["colors"] = []  # disable per-instance colors → use material base
        self.add_trace(trace)
        return self


# ---------------------------------------------------------------------------
# High-level builders
# ---------------------------------------------------------------------------

def build_fixed_framework_movie(
    snapshots: list[dict[str, Any]],
    framework_atoms: Atoms | None = None,
    spacing: float = 0.25,
    isovalue: float = 1.0,
    surf_step_size: int = 2,
    fps: float = 20.0,
    eye: tuple[float, float, float] = (1.5, 1.5, 1.5),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_frames: int | None = None,
) -> MovieFigure:
    """Build a MovieFigure for fixed-framework + animated adsorbates workflow."""
    n_frames = len(snapshots) if max_frames is None else min(len(snapshots), max_frames)
    snapshots = snapshots[:n_frames]

    # Framework from first snapshot (or provided atoms)
    if framework_atoms is None:
        framework_atoms = snapshots[0]["atoms"]

    fig = MovieFigure()
    fig.set_camera(eye=eye, center=center, up=up)
    fig.update_layout({"fps": fps, "n_frames": n_frames})

    # Static traces — framework in three representations (ball-and-stick default)
    fig.add_trace(make_framework_trace(framework_atoms))
    fig.add_trace(make_framework_trace(framework_atoms, trace_id="framework_stick", representation="stick", name="Framework (stick)", style={"visible": False, "side": "FrontSide"}))
    fig.add_trace(make_framework_trace(framework_atoms, trace_id="framework_spacefill", representation="space_filling", name="Framework (spacefill)", style={"visible": False, "side": "FrontSide"}))
    fig.add_trace(make_cell_trace(framework_atoms))

    # Compute segmentation once on the framework
    seg = compute_segmentation_for_snapshot(
        {"atoms": framework_atoms, "bonds": None, "cell": None, "dgrid": None, "regions": None, "connections": None, "meta": {}},
        spacing=spacing,
    )

    fig.add_trace(make_surface_trace(seg["dgrid"], seg["regions"], framework_atoms, isovalue=isovalue, step_size=surf_step_size))

    maxima = make_maxima_trace(seg, framework_atoms)
    fig.add_trace(maxima["internal"])
    fig.add_trace(maxima["outside"])

    windows = make_windows_trace(seg)
    fig.add_trace(windows["internal"])
    fig.add_trace(windows["periodic"])

    conns = make_connections_trace(seg, framework_atoms)
    fig.add_trace(conns["internal_lines"])
    fig.add_trace(conns["periodic_lines"])
    fig.add_trace(conns["internal_nodes"])
    fig.add_trace(conns["periodic_nodes"])

    # Groups for toggling
    fig.groups = {
        "internal_pore_graph": {
            "traces": ["maxima_internal", "windows_internal", "connections_internal", "connections_internal_nodes"],
            "label": "Internal Pore Graph",
        },
        "periodic_pore_graph": {
            "traces": ["maxima_outside", "windows_periodic", "connections_periodic", "connections_periodic_nodes"],
            "label": "Periodic Pore Graph",
            "default_off": True,
        },
        "guests": {
            "traces": ["guests"],
            "label": "Adsorbates",
        },
        "guest_bonds": {
            "traces": ["guestbonds_main"],
            "label": "Adsorbate bonds",
        },
    }

    # Per-frame guest traces
    for i, snap in enumerate(snapshots):
        guest = make_guest_trace(snap["atoms"])
        fig.add_trace(guest, frame=i)
        fig.add_trace(
            make_guest_bonds_trace(
                snap["atoms"],
                style="guest_bonds",
                name="Adsorbate bonds",
                trace_id="guestbonds_main",
            ),
            frame=i,
        )

    fig.meta = {
        "workflow": "fixed_framework",
        "n_frames": n_frames,
        "n_atoms_framework": int(len(framework_atoms)),
    }
    return fig


# ---------------------------------------------------------------------------
# Jinja2 template loader
# ---------------------------------------------------------------------------

class _TemplateLoader:
    """Minimal template loader that falls back to an embedded default."""

    def __init__(self) -> None:
        self._templates: dict[str, str] = {}

    def _default_threejs(self) -> str:
        return _DEFAULT_THREEJS_TEMPLATE

    def get(self, name: str) -> str:
        if name not in self._templates:
            if name == "threejs_default":
                self._templates[name] = self._default_threejs()
            else:
                raise ValueError(f"Unknown template: {name}")
        return self._templates[name]

    def render(self, **kwargs: Any) -> str:
        # Simple string substitution fallback
        template = self.get(kwargs.pop("template", "threejs_default"))
        for key, val in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(val, (tuple, list)):
                val = json.dumps(val)
            elif isinstance(val, (dict,)):
                val = json.dumps(val)
            template = template.replace(placeholder, str(val))
        return template


_THREEJS_TEMPLATE = _TemplateLoader()

# ---------------------------------------------------------------------------
# Embedded default Three.js template
# ---------------------------------------------------------------------------

_DEFAULT_THREEJS_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Framework + Segmentation + MD Snapshots</title>
  <style>
    html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
      background: {{bg_css}};
      font-family: {{font_family}}; color: {{text_color}};
      -webkit-text-size-adjust: 100%; touch-action: none; }
    #app { position: fixed; inset: 0; right: 200px; }
    #hud { position: fixed; right: 0; top: 0; width: 200px; height: 100vh;
      background: {{panel_bg}}; border-left: 1px solid rgba(128,128,128,0.2);
      padding: 10px; display: flex; flex-direction: column; gap: 4px;
      overflow-y: auto; overflow-x: hidden; resize: horizontal; z-index: 10; font-size: 11px; }
    #figTitle { font-size: 14px; font-weight: 600; margin-bottom: 4px; text-align: center; }
    #row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
    #frameSlider { width: 100%; }
    .checkrow { display: flex; flex-direction: column; gap: 3px; }
    label { user-select: none; font-size: 11px; white-space: nowrap; cursor: pointer; }
    #tooltip { position: fixed; pointer-events: none; display: none; z-index: 20;
      background: rgba(0,0,0,0.85); border: 1px solid #666; border-radius: 8px;
      padding: 6px 8px; font-size: 12px; white-space: pre; color: #efefef; }
    button { background: rgba(128,128,128,0.15); color: {{text_color}}; border: 1px solid rgba(128,128,128,0.3);
      border-radius: 4px; padding: 3px 6px; cursor: pointer; font-size: 11px; }
    button:hover { background: rgba(128,128,128,0.3); }
    #status, #meta { font-size: 11px; }
    #meta { display: block; opacity: 0.85; line-height: 1.3; }
    input[type="checkbox"] { accent-color: #b9d7ff; vertical-align: middle; }
    #axisLabels { position: fixed; bottom: 60px; left: 20px; z-index: 15; pointer-events: none; }
    #axisLabels span { font-size: 14px; font-weight: 700; text-shadow: 0 0 4px rgba(0,0,0,0.8); }
    @media (max-width: 600px) {
      #app { right: 0; bottom: 140px; }
      #hud { position: fixed; bottom: 0; left: 0; right: 0; top: auto;
        width: 100%; height: 140px; border-left: none; border-top: 1px solid rgba(128,128,128,0.2);
        flex-direction: row; flex-wrap: wrap; overflow-x: auto; overflow-y: hidden; }
      .checkrow { flex-direction: row; flex-wrap: wrap; gap: 2px 8px; }
    }
  </style>
</head>
<body>
  <div id="app"></div>
  <div id="tooltip"></div>
  <div id="axisLabels"></div>
  <div id="hud">
    <button id="toggleHud" style="position:absolute;top:4px;right:4px;font-size:16px;padding:0 4px;line-height:1;cursor:pointer;background:none;border:1px solid #666;border-radius:2px;color:inherit;" title="Toggle HUD panel">◧</button>
    <div id="figTitle">{{title}}</div>
    <div id="row">
      <button id="playPause">Play</button>
      <button id="prevFrame">Prev</button>
      <button id="nextFrame">Next</button>
      <span id="status">Frame 0</span>
      <button id="recordGif" style="display:none;">Record GIF</button>
    </div>
    <span id="meta"></span>
    <div class="checkrow">
      <button id="pivotScene">Pivot: Center</button>
      <button id="pivotOrigin">Pivot: Origin</button>
      <button id="toggleOrtho">Ortho</button>
      <button id="exportPng">Export PNG</button>
      <button id="recordVideo">Record</button>
      <button id="lightInfo">Lights</button>
    </div>
    <div class="checkrow" id="viewButtons" style="gap:2px;">
      <button id="viewX" title="View +X" style="font-size:9px;padding:1px 4px;min-width:24px;">+X</button>
      <button id="viewNegX" title="View -X" style="font-size:9px;padding:1px 4px;min-width:24px;">-X</button>
      <button id="viewY" title="View +Y" style="font-size:9px;padding:1px 4px;min-width:24px;">+Y</button>
      <button id="viewNegY" title="View -Y" style="font-size:9px;padding:1px 4px;min-width:24px;">-Y</button>
      <button id="viewZ" title="View +Z" style="font-size:9px;padding:1px 4px;min-width:24px;">+Z</button>
      <button id="viewNegZ" title="View -Z" style="font-size:9px;padding:1px 4px;min-width:24px;">-Z</button>
    </div>
    <div id="lightModal" style="display:none;position:fixed;inset:0;z-index:50;background:rgba(0,0,0,0.5);">
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#222;color:#eee;padding:16px;border-radius:10px;font-size:12px;max-width:320px;">
        <h3 style="margin:0 0 8px 0;font-size:14px;">Lighting Settings</h3>
        <pre id="lightDetails" style="margin:0;white-space:pre-wrap;font-family:monospace;font-size:11px;"></pre>
        <button id="closeLightModal" style="margin-top:10px;width:100%;">Close</button>
      </div>
    </div>
    <div class="checkrow">
      <button id="resetCam">Reset View</button>
      <button id="toggleRotate">Auto-Rotate</button>
      <button id="measureTool">Measure</button>
    </div>
    <div class="checkrow">
      <label>Rotate axis:
        <input id="rotAxisX" type="number" step="0.1" value="0" style="width:42px;vertical-align:middle;" />
        <input id="rotAxisY" type="number" step="0.1" value="0" style="width:42px;vertical-align:middle;" />
        <input id="rotAxisZ" type="number" step="0.1" value="1" style="width:42px;vertical-align:middle;" />
      </label>
    </div>
    <div class="checkrow" id="advancedControls" style="flex-direction:column;gap:6px;">
      <label>Representation:
        <select id="reprSelect" style="font-size:10px;width:100%;"><option value="ball_and_stick">Ball-and-stick</option><option value="stick">Stick</option><option value="space_filling">Space-filling</option></select>
      </label>
      <div id="meshOpacityContainer" style="display:flex;flex-direction:column;gap:2px;max-height:120px;overflow-y:auto;font-size:10px;"></div>
      <label>Point size: <input id="ptSizeSlider" type="range" min="0.5" max="4.5" step="0.1" value="1" style="width:80px;vertical-align:middle;" /></label>
      <label>Play speed: <input id="speedSlider" type="range" min="0.25" max="4" step="0.25" value="1" style="width:80px;vertical-align:middle;" /></label>
      <label>Ambient: <input id="ambSlider" type="range" min="0" max="1" step="0.05" value="0.28" style="width:80px;vertical-align:middle;" /></label>
      <label>Directional: <input id="dirSlider" type="range" min="0" max="2" step="0.05" value="1" style="width:80px;vertical-align:middle;" /></label>
      <label><input id="screenDoorToggle" type="checkbox" /> Screen-door transparency</label>
      <label>Dir light xyz:
        <input id="lightDirX" type="number" step="0.1" value="5" style="width:40px;vertical-align:middle;" />
        <input id="lightDirY" type="number" step="0.1" value="10" style="width:40px;vertical-align:middle;" />
        <input id="lightDirZ" type="number" step="0.1" value="7.5" style="width:40px;vertical-align:middle;" />
        <button id="applyLightDir" type="button" style="font-size:9px;padding:1px 4px;">Set</button>
      </label>
      <label>View dir xyz:
        <input id="viewDirX" type="number" step="0.1" value="1" style="width:40px;vertical-align:middle;" />
        <input id="viewDirY" type="number" step="0.1" value="1" style="width:40px;vertical-align:middle;" />
        <input id="viewDirZ" type="number" step="0.1" value="1" style="width:40px;vertical-align:middle;" />
        <button id="applyViewDir" type="button" style="font-size:9px;padding:1px 4px;">Set</button>
      </label>
      <label>Exploded: <input id="explodeSlider" type="range" min="0" max="6" step="0.05" value="0" style="width:140px;vertical-align:middle;" /></label>
      <label>Bg: <input id="bgPicker" type="color" value="#111111" style="width:40px;vertical-align:middle;border:none;height:16px;" /></label>
      <div style="margin-top:2px;display:flex;gap:2px;">
        <button id="bgGradientNone" title="Solid" style="font-size:8px;padding:1px 3px;">■</button>
        <button id="bgGradientLR" title="L→R gradient" style="font-size:8px;padding:1px 3px;">⬌</button>
        <button id="bgGradientTB" title="T→B gradient" style="font-size:8px;padding:1px 3px;">⬍</button>
        <button id="bgGradientRadial" title="Radial gradient" style="font-size:8px;padding:1px 3px;">◎</button>
        <button id="bgGradientTransparent" title="Transparent background" style="font-size:8px;padding:1px 3px;">⊡</button>
      </div>
      <label>PNG: <select id="pngRes" style="font-size:10px;vertical-align:middle;"><option value="1">1x</option><option value="2">2x</option><option value="4">4x</option></select></label>
      <label>REC: <select id="recRes" style="font-size:10px;vertical-align:middle;"><option value="1" selected>1x</option><option value="2">2x</option><option value="3">3x</option></select></label>
    </div>
    <div class="checkrow">
      <label id="labelShowCell"><input type="checkbox" id="showCell" checked /> Unit cell</label>
      <label id="labelShowFramework"><input type="checkbox" id="showFramework" checked /> Framework mesh</label>
      <label id="labelShowSurface"><input type="checkbox" id="showSurface" checked /> Segmentation surface (regions)</label>
      <label id="labelShowGuests"><input type="checkbox" id="showGuests" checked /> Snapshot adsorbates (mesh spheres)</label>
      <label id="labelShowGuestBonds"><input type="checkbox" id="showGuestBonds" checked /> Adsorbate bonds</label>
      <label id="labelShowWindows"><input type="checkbox" id="showWindows" checked /> Internal windows</label>
      <label id="labelShowPeriodicWindows"><input type="checkbox" id="showPeriodicWindows" /> Periodic windows</label>
      <label id="labelShowInternalConnections"><input type="checkbox" id="showInternalConnections" checked /> Internal connections</label>
      <label id="labelShowPeriodicConnections"><input type="checkbox" id="showPeriodicConnections" /> Periodic connections</label>
    </div>
    <div class="checkrow" id="groupChecks"></div>
    <button id="toggleLegend" style="font-size:10px;margin-top:2px;" title="Fold/unfold legend entries">Fold Legend ▲</button>
    <input id="frameSlider" type="range" min="0" max="0" step="1" value="0" />
  </div>
  <button id="showHudTab" style="display:none;position:fixed;top:8px;right:8px;z-index:30;background:rgba(20,20,20,0.85);color:{{text_color}};border:1px solid rgba(128,128,128,0.45);border-radius:6px;padding:5px 8px;font-size:11px;" title="Show controls">Controls</button>

  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/SSAOShader.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/math/SimplexNoise.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/MaskPass.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/SSAOPass.js"></script>
  <script src="https://unpkg.com/gif.js@0.2.0/dist/gif.js"></script>
  <script>
    const DATA = {{payload_json}};
    // Unpack binary arrays in static traces (positions, indices, colors)
    if (DATA.static && DATA.static.traces) {
      _unpackBinaryArrays(DATA.static);
    }
    const MOV = DATA.static && DATA.static.traces ? DATA.static.traces : [];
    const FRAMES = DATA.frames || [];

    // ---- Sidecar streaming (per-frame Range fetch, only 2-3 parsed frames in RAM) ----
    let frameOffsets = [];
    let parsedFrameCache = new Map();
    const FRAME_CACHE_SIZE = 3;
    let sidecarFullText = null;   // fallback when Range is unsupported
    let sidecarTextBlob = null;   // cached full response text when server ignores Range

    // Load the tiny byte-offset index (e.g. 200 frames -> 1.6 KB)
    if (DATA.sidecar && DATA.sidecar_idx) {
      try {
        const idxReq = new XMLHttpRequest();
        idxReq.open('GET', DATA.sidecar_idx, false);
        idxReq.send();
        if (idxReq.status === 200 || idxReq.status === 0) {
          frameOffsets = JSON.parse(idxReq.responseText);
        }
      } catch (e) {
        console.warn("Sidecar index load failed:", e);
      }
    }

    // Fallback: if no index, load the whole sidecar once (old behaviour)
    if (DATA.sidecar && frameOffsets.length === 0) {
      try {
        const req = new XMLHttpRequest();
        req.open('GET', DATA.sidecar, false);
        req.send();
        if (req.status === 200 || req.status === 0) {
          sidecarFullText = req.responseText.trim().split('\n').filter(l => l.trim());
        }
      } catch (e) {
        console.error("Sidecar load failed. Serve via HTTP (e.g. python -m http.server):", e);
      }
    }

    function getFrameCount() {
      if (DATA.sidecar) {
        if (frameOffsets.length > 0) return frameOffsets.length;
        if (sidecarFullText) return sidecarFullText.length;
      }
      return FRAMES.length;
    }

    function _unpackBinaryArrays(raw) {
      if (!raw.traces) return raw;
      raw.traces.forEach(trace => {
        if (trace.data) {
          for (const key in trace.data) {
            const val = trace.data[key];
            if (val && val.__binary__) {
              const binaryString = atob(val.data);
              const len = binaryString.length;
              const bytes = new Uint8Array(len);
              for (let j = 0; j < len; j++) bytes[j] = binaryString.charCodeAt(j);
              const view = new DataView(bytes.buffer);
              if (val.dtype === 'float32') {
                const out = new Array(len / 4);
                for (let j = 0; j < len / 4; j++) out[j] = view.getFloat32(j * 4, true);
                trace.data[key] = out;
              } else if (val.dtype === 'uint32') {
                const out = new Array(len / 4);
                for (let j = 0; j < len / 4; j++) out[j] = view.getUint32(j * 4, true);
                trace.data[key] = out;
              } else if (val.dtype === 'uint16') {
                const out = new Array(len / 2);
                for (let j = 0; j < len / 2; j++) out[j] = view.getUint16(j * 2, true);
                trace.data[key] = out;
              } else {
                trace.data[key] = Array.from(bytes);
              }
            }
          }
        }
      });
      return raw;
    }

    function _parseFrame(text) {
      const raw = JSON.parse(text);
      _unpackBinaryArrays(raw);
      return raw;
    }

    function getFrame(i) {
      if (!DATA.sidecar) return FRAMES[i] || null;
      const n = getFrameCount();
      if (i < 0 || i >= n) return null;
      if (parsedFrameCache.has(i)) return parsedFrameCache.get(i);

      let text = null;
      if (frameOffsets.length > 0) {
        // Streaming: fetch single frame line via HTTP Range
        const start = frameOffsets[i];
        const end = (i + 1 < frameOffsets.length) ? frameOffsets[i + 1] : null;
        // If we already discovered the server ignores Range, slice from cached blob
        if (sidecarTextBlob) {
          text = (end !== null) ? sidecarTextBlob.substring(start, end) : sidecarTextBlob.substring(start);
        } else {
          const req = new XMLHttpRequest();
          if (end !== null) {
            req.open('GET', DATA.sidecar, false);
            req.setRequestHeader('Range', 'bytes=' + start + '-' + (end - 1));
            req.send();
            if (req.status === 206) {
              text = req.responseText;
            } else if (req.status === 200) {
              // Server ignored Range header — cache full response once, then slice
              sidecarTextBlob = req.responseText;
              text = sidecarTextBlob.substring(start, end);
            }
          } else {
            // Last frame: open-ended range
            req.open('GET', DATA.sidecar, false);
            req.setRequestHeader('Range', 'bytes=' + start + '-');
            req.send();
            if (req.status === 206) {
              text = req.responseText;
            } else if (req.status === 200) {
              sidecarTextBlob = req.responseText;
              text = sidecarTextBlob.substring(start);
            }
          }
          if (text === null) {
            console.error("Range fetch failed for frame", i, "status:", req.status);
            return null;
          }
        }
      } else if (sidecarFullText) {
        text = sidecarFullText[i];
      }
      if (text === null) return null;

      const data = _parseFrame(text);
      parsedFrameCache.set(i, data);
      while (parsedFrameCache.size > FRAME_CACHE_SIZE) {
        const firstKey = parsedFrameCache.keys().next().value;
        parsedFrameCache.delete(firstKey);
      }
      return data;
    }
    // ----------------------------------------------------------------------

    const STYLES = DATA.styles || {};
    let guestSphereSegments = 14;
    const _gtProbe = getFrameCount() && getFrame(0) && getFrame(0).traces
      ? getFrame(0).traces.find(t => t.id === "guests" || t.id.startsWith("guests_"))
      : null;
    if (_gtProbe && _gtProbe.data && _gtProbe.data.n_atoms > 2000) guestSphereSegments = 6;
    else if (_gtProbe && _gtProbe.data && _gtProbe.data.n_atoms > 500) guestSphereSegments = 8;
    const app = document.getElementById("app");
    const slider = document.getElementById("frameSlider");
    const statusEl = document.getElementById("status");
    const metaEl = document.getElementById("meta");
    const playBtn = document.getElementById("playPause");
    const prevBtn = document.getElementById("prevFrame");
    const nextBtn = document.getElementById("nextFrame");
    const tooltip = document.getElementById("tooltip");

    const reprSelect = document.getElementById("reprSelect");
    const showCell = document.getElementById("showCell");
    const showFramework = document.getElementById("showFramework");
    const showSurface = document.getElementById("showSurface");
    const showGuests = document.getElementById("showGuests");
    const showGuestBonds = document.getElementById("showGuestBonds");
    const showWindows = document.getElementById("showWindows");
    const showPeriodicWindows = document.getElementById("showPeriodicWindows");
    const showInternalConnections = document.getElementById("showInternalConnections");
    const showPeriodicConnections = document.getElementById("showPeriodicConnections");
    const pivotSceneBtn = document.getElementById("pivotScene");
    const pivotOriginBtn = document.getElementById("pivotOrigin");
    const resetCamBtn = document.getElementById("resetCam");
    const toggleRotateBtn = document.getElementById("toggleRotate");
    const rotAxisXInput = document.getElementById("rotAxisX");
    const rotAxisYInput = document.getElementById("rotAxisY");
    const rotAxisZInput = document.getElementById("rotAxisZ");
    const measureBtn = document.getElementById("measureTool");
    const ptSizeSlider = document.getElementById("ptSizeSlider");
    const speedSlider = document.getElementById("speedSlider");
    const ambSlider = document.getElementById("ambSlider");
    const dirSlider = document.getElementById("dirSlider");
    const screenDoorToggle = document.getElementById("screenDoorToggle");
    const lightDirXInput = document.getElementById("lightDirX");
    const lightDirYInput = document.getElementById("lightDirY");
    const lightDirZInput = document.getElementById("lightDirZ");
    const applyLightDirBtn = document.getElementById("applyLightDir");
    const viewDirXInput = document.getElementById("viewDirX");
    const viewDirYInput = document.getElementById("viewDirY");
    const viewDirZInput = document.getElementById("viewDirZ");
    const applyViewDirBtn = document.getElementById("applyViewDir");
    const explodeSlider = document.getElementById("explodeSlider");
    const bgPicker = document.getElementById("bgPicker");
    const pngResSelect = document.getElementById("pngRes");
    const recResSelect = document.getElementById("recRes");
    let pointSizeMult = 1.0;
    pointSizeMult = parseFloat(ptSizeSlider.value) || 1.0;
    const screenDoorDefault = !!(DATA.layout && DATA.layout.surface_screen_door === true);
    if (screenDoorToggle) screenDoorToggle.checked = screenDoorDefault;

    function getRenderViewport() {
      const hudEl = document.getElementById("hud");
      const hudVisible = !!(hudEl && hudEl.style.display !== "none");
      const hudW = hudVisible ? (hudEl.offsetWidth || 200) : 0;
      const w = Math.max(window.innerWidth - hudW, 1);
      const h = Math.max(window.innerHeight, 1);
      return { w, h, hudW };
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    {
      const vp = getRenderViewport();
      renderer.setSize(vp.w, vp.h);
    }
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.setClearColor(0x000000, 1.0);
    app.appendChild(renderer.domElement);
    function makeSprite(drawFn) {
      const c = document.createElement('canvas');
      c.width = 64; c.height = 64;
      const ctx = c.getContext('2d');
      drawFn(ctx);
      return new THREE.CanvasTexture(c);
    }
    const spriteTextures = {
      circle: makeSprite(ctx => { ctx.beginPath(); ctx.arc(32, 32, 28, 0, Math.PI * 2); ctx.fillStyle = 'white'; ctx.fill(); }),
      square: makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.fillRect(4, 4, 56, 56); }),
      diamond: makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.beginPath(); ctx.moveTo(32, 2); ctx.lineTo(62, 32); ctx.lineTo(32, 62); ctx.lineTo(2, 32); ctx.closePath(); ctx.fill(); }),
      "triangle-up": makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.beginPath(); ctx.moveTo(32, 4); ctx.lineTo(60, 58); ctx.lineTo(4, 58); ctx.closePath(); ctx.fill(); }),
      "triangle-down": makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.beginPath(); ctx.moveTo(32, 60); ctx.lineTo(60, 4); ctx.lineTo(4, 4); ctx.closePath(); ctx.fill(); }),
      star: makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.beginPath(); for (let i = 0; i < 5; i++) { const a = (i * 72 - 90) * Math.PI / 180; ctx.lineTo(32 + 28 * Math.cos(a), 32 + 28 * Math.sin(a)); const b = ((i * 72 + 36) - 90) * Math.PI / 180; ctx.lineTo(32 + 12 * Math.cos(b), 32 + 12 * Math.sin(b)); } ctx.closePath(); ctx.fill(); }),
      cross: makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.fillRect(24, 4, 16, 56); ctx.fillRect(4, 24, 56, 16); }),
      x: makeSprite(ctx => { ctx.strokeStyle = 'white'; ctx.lineWidth = 10; ctx.beginPath(); ctx.moveTo(8, 8); ctx.lineTo(56, 56); ctx.moveTo(56, 8); ctx.lineTo(8, 56); ctx.stroke(); }),
      ring: makeSprite(ctx => { ctx.strokeStyle = 'white'; ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(32, 32, 24, 0, Math.PI * 2); ctx.stroke(); }),
      pentagon: makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.beginPath(); for (let i = 0; i < 5; i++) { const a = (i * 72 - 90) * Math.PI / 180; ctx.lineTo(32 + 28 * Math.cos(a), 32 + 28 * Math.sin(a)); } ctx.closePath(); ctx.fill(); }),
      hexagon: makeSprite(ctx => { ctx.fillStyle = 'white'; ctx.beginPath(); for (let i = 0; i < 6; i++) { const a = (i * 60 - 90) * Math.PI / 180; ctx.lineTo(32 + 28 * Math.cos(a), 32 + 28 * Math.sin(a)); } ctx.closePath(); ctx.fill(); }),
    };
    const circleTex = spriteTextures["circle"];

    const scene = new THREE.Scene();
    scene.background = new THREE.Color({{scene_bg}});

    const eyeArr = {{eye}};
    const centerArr = {{center}};
    const upArr = {{up}};
    const projection = "{{projection}}";

    const scaledEyeArr = eyeArr;

    let camera, perspectiveCam, orthoCam, frustumSize, isOrtho;
    const vp0 = getRenderViewport();
    const aspect = vp0.w / vp0.h;

    perspectiveCam = new THREE.PerspectiveCamera(45, aspect, 0.01, 2000);
    orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 2000);

    function updateOrthoFrustum() {
      var dist = camera.position.distanceTo(controls.target);
      frustumSize = 2 * dist * Math.tan(THREE.MathUtils.degToRad(22.5));
      var vp = getRenderViewport();
      var asp = vp.w / vp.h;
      orthoCam.left = frustumSize * asp / -2;
      orthoCam.right = frustumSize * asp / 2;
      orthoCam.top = frustumSize / 2;
      orthoCam.bottom = frustumSize / -2;
      orthoCam.updateProjectionMatrix();
    }

    isOrtho = projection === "orthographic";
    camera = isOrtho ? orthoCam : perspectiveCam;

    [perspectiveCam, orthoCam].forEach(function(cam) {
      cam.position.set(scaledEyeArr[0], scaledEyeArr[1], scaledEyeArr[2]);
      cam.up.set(upArr[0], upArr[1], upArr[2]);
      cam.lookAt(new THREE.Vector3(centerArr[0], centerArr[1], centerArr[2]));
    });

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    controls.target.set(centerArr[0], centerArr[1], centerArr[2]);
    controls.update();
    updateOrthoFrustum();  // set initial frustum from camera-to-target distance

    // Orthographic toggle
    const toggleOrthoBtn = document.getElementById("toggleOrtho");
    toggleOrthoBtn.textContent = isOrtho ? "Perspective" : "Ortho";
    function recreateComposer() {
      if (composer) {
        try { composer.passes.forEach(function(p) { if (p.dispose) p.dispose(); }); } catch(e) {}
      }
      composer = null; ssaoPass = null;
      if (!isOrtho) {
        try {
          composer = new THREE.EffectComposer(renderer);
          composer.addPass(new THREE.RenderPass(scene, camera));
          var vp = getRenderViewport();
          ssaoPass = new THREE.SSAOPass(scene, camera, vp.w, vp.h);
          ssaoPass.kernelRadius = 0.45;
          ssaoPass.kernelSize = 24;
          ssaoPass.minDistance = 0.003;
          ssaoPass.maxDistance = 0.03;
          composer.addPass(ssaoPass);
        } catch(e) { console.warn('SSAO unavailable:', e); composer = null; }
      }
    }

    // Store original light intensities for reversible ortho toggle
    const _lightOrig = {};
    const _defaultLightDir = new THREE.Vector3(5, 10, 7.5);

    function parseNormalizedVector(xInput, yInput, zInput, fallbackVec) {
      const x = parseFloat(xInput.value);
      const y = parseFloat(yInput.value);
      const z = parseFloat(zInput.value);
      if (!isFinite(x) || !isFinite(y) || !isFinite(z)) return fallbackVec.clone().normalize();
      const v = new THREE.Vector3(x, y, z);
      if (v.lengthSq() < 1e-12) return fallbackVec.clone().normalize();
      return v.normalize();
    }

    function applyDirectionalLightVector() {
      const dirNorm = parseNormalizedVector(lightDirXInput, lightDirYInput, lightDirZInput, _defaultLightDir);
      const mag = Math.max(dirKey.position.length(), 1e-6);
      dirKey.position.copy(dirNorm.multiplyScalar(mag));
    }

    toggleOrthoBtn.addEventListener("click", function() {
      isOrtho = !isOrtho;
      var oldPos = camera.position.clone();
      var oldTarget = controls.target.clone();
      camera = isOrtho ? orthoCam : perspectiveCam;
      camera.position.copy(oldPos);
      camera.lookAt(oldTarget);
      controls.object = camera;
      controls.target.copy(oldTarget);
      controls.update();
      updateOrthoFrustum();
      recreateComposer();

      if (isOrtho) {
        _lightOrig.ambient = ambient.intensity;
        _lightOrig.hemi = hemiLight.intensity;
        _lightOrig.key = dirKey.intensity;
        _lightOrig.fill = dirFill.intensity;
        ambient.intensity = 0.015;
        hemiLight.intensity = 0.10;
        dirKey.intensity = _lightOrig.key * 1.2;
        dirFill.intensity = _lightOrig.fill * 0.22;
      } else {
        ambient.intensity = _lightOrig.ambient;
        hemiLight.intensity = _lightOrig.hemi;
        dirKey.intensity = _lightOrig.key;
        dirFill.intensity = _lightOrig.fill;
      }

      renderer.domElement.style.filter = "";
      toggleOrthoBtn.textContent = isOrtho ? "Perspective" : "Ortho";

      Object.values(objectsById).forEach(function(obj) {
        if (obj && obj.material && obj.material.isPointsMaterial) {
          // Respect forceDepthTest (set by framework scatter) if present
          if (obj.material.userData && obj.material.userData.forceDepthTest !== undefined) {
            obj.material.depthTest = obj.material.userData.forceDepthTest;
          } else {
            obj.material.depthTest = true;
          }
          obj.material.depthWrite = false;
          var base = obj.material.userData && obj.material.userData.baseSize ? obj.material.userData.baseSize : obj.material.size;
          applyPointSize(obj, base);
        }
      });
    });

    const ambient = new THREE.AmbientLight(0xffffff, 0.28);
    scene.add(ambient);
    const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x3b3b3b, 0.25);
    scene.add(hemiLight);
    const dirKey = new THREE.DirectionalLight(0xffffff, 0.48);
    dirKey.position.set(5, 10, 7.5);
    scene.add(dirKey);
    const dirFill = new THREE.DirectionalLight(0xffffff, 0.32);
    dirFill.position.set(-5, -9, -7.5);
    scene.add(dirFill);
    // Initialize directional-light input fields from current key-light vector.
    lightDirXInput.value = String(dirKey.position.x.toFixed(2));
    lightDirYInput.value = String(dirKey.position.y.toFixed(2));
    lightDirZInput.value = String(dirKey.position.z.toFixed(2));

    // Ortho init: compensate for lack of SSAO with stronger directional contrast
    if (isOrtho) {
      _lightOrig.ambient = ambient.intensity;
      _lightOrig.hemi = hemiLight.intensity;
      _lightOrig.key = dirKey.intensity;
      _lightOrig.fill = dirFill.intensity;
      ambient.intensity = 0.015;
      hemiLight.intensity = 0.10;
      dirKey.intensity = _lightOrig.key * 1.2;
      dirFill.intensity = _lightOrig.fill * 0.22;
    }
    renderer.domElement.style.filter = "";

    // Post-processing: SSAO in perspective, contrast boost in ortho
    let composer, ssaoPass;
    recreateComposer();

    const objectsById = {};

    function styleFor(trace) {
      if (!trace) return {};
      const s = trace.style;
      let base = {};
      if (s && typeof s === "object" && !Array.isArray(s)) base = Object.assign({}, s);
      else if (typeof s === "string" && STYLES[s]) base = Object.assign({}, STYLES[s]);
      // Legacy / mistaken top-level keys (see MovieFigure.update_traces on Python side)
      if (trace.opacity !== undefined) base.opacity = trace.opacity;
      return base;
    }

    function hasColorsAttr(obj) {
      return obj && obj.geometry && obj.geometry.attributes && obj.geometry.attributes.color;
    }

    function threeColorFromStyle(c, fallbackHex) {
      if (c === undefined || c === null) return new THREE.Color(fallbackHex !== undefined ? fallbackHex : 0xffffff);
      if (typeof c === "number") return new THREE.Color(c);
      if (Array.isArray(c) && c.length >= 3) return new THREE.Color(c[0], c[1], c[2]);
      return new THREE.Color(fallbackHex !== undefined ? fallbackHex : 0xffffff);
    }

    function pointsBaseSize(trace) {
      const style = styleFor(trace);
      let size = style.size !== undefined ? style.size : 1.0;
      if (trace.data && trace.data.sizes && trace.data.sizes.length) {
        let mx = 0;
        for (let i = 0; i < trace.data.sizes.length; i++) {
          const v = Number(trace.data.sizes[i]);
          if (!isNaN(v)) mx = Math.max(mx, v);
        }
        if (mx > 0) size = mx;
      }
      return size;
    }

    function applyPointSize(pointsObj, baseSize) {
      if (!pointsObj || !pointsObj.material || !pointsObj.material.isPointsMaterial) return;
      const base = baseSize !== undefined ? baseSize : (
        pointsObj.material.userData && pointsObj.material.userData.baseSize
          ? pointsObj.material.userData.baseSize
          : pointsObj.material.size
      );
      pointsObj.material.size = base * pointSizeMult * (isOrtho ? 2.0 : 1.0);
      pointsObj.material.needsUpdate = true;
    }

    function isSurfaceMeshId(traceId) {
      return !!traceId && (
        traceId === "surface" ||
        traceId === "surface_pbc" ||
        traceId === "surface_pocket" ||
        traceId.indexOf("surface_") === 0
      );
    }

    function applySurfaceScreenDoorForObject(obj) {
      if (!obj || !obj.material) return;
      const mat = obj.material;
      if (!obj.userData) obj.userData = {};
      if (mat.userData.screenDoorOriginalOnBeforeCompile === undefined) {
        mat.userData.screenDoorOriginalOnBeforeCompile = mat.onBeforeCompile;
        mat.userData.screenDoorOriginalProgramKey = mat.customProgramCacheKey;
      }
      const enabled = !!(screenDoorToggle && screenDoorToggle.checked);
      if (enabled) {
        const clampedOpacity = Math.max(0.0, Math.min(1.0, mat.opacity !== undefined ? mat.opacity : 1.0));
        mat.userData.screenDoorOpacity = clampedOpacity;
        mat.transparent = false;
        mat.depthWrite = true;
        mat.alphaTest = 0.0;
        mat.dithering = true;
        mat.onBeforeCompile = function(shader) {
          const orig = mat.userData.screenDoorOriginalOnBeforeCompile;
          if (typeof orig === "function") orig(shader);
          shader.uniforms.uScreenDoorOpacity = { value: mat.userData.screenDoorOpacity !== undefined ? mat.userData.screenDoorOpacity : 1.0 };
          if (shader.fragmentShader.indexOf("uScreenDoorOpacity") === -1) {
            shader.fragmentShader = shader.fragmentShader.replace(
              "void main() {",
              "uniform float uScreenDoorOpacity;\nvoid main() {"
            );
            shader.fragmentShader = shader.fragmentShader.replace(
              "#include <dithering_fragment>",
              "float sdOpacity = clamp(uScreenDoorOpacity, 0.0, 1.0);\n" +
              "if (sdOpacity < 1.0) {\n" +
              "  vec2 p = mod(gl_FragCoord.xy, 4.0);\n" +
              "  float x = p.x;\n" +
              "  float y = p.y;\n" +
              "  float threshold = 0.0;\n" +
              "  if (y < 1.0) {\n" +
              "    threshold = (x < 1.0) ? 0.0/16.0 : ((x < 2.0) ? 8.0/16.0 : ((x < 3.0) ? 2.0/16.0 : 10.0/16.0));\n" +
              "  } else if (y < 2.0) {\n" +
              "    threshold = (x < 1.0) ? 12.0/16.0 : ((x < 2.0) ? 4.0/16.0 : ((x < 3.0) ? 14.0/16.0 : 6.0/16.0));\n" +
              "  } else if (y < 3.0) {\n" +
              "    threshold = (x < 1.0) ? 3.0/16.0 : ((x < 2.0) ? 11.0/16.0 : ((x < 3.0) ? 1.0/16.0 : 9.0/16.0));\n" +
              "  } else {\n" +
              "    threshold = (x < 1.0) ? 15.0/16.0 : ((x < 2.0) ? 7.0/16.0 : ((x < 3.0) ? 13.0/16.0 : 5.0/16.0));\n" +
              "  }\n" +
              "  if (sdOpacity <= threshold) discard;\n" +
              "  gl_FragColor.rgb *= sdOpacity;\n" +
              "}\n" +
              "#include <dithering_fragment>"
            );
          }
          mat.userData.screenDoorUniform = shader.uniforms.uScreenDoorOpacity;
        };
        // Distinct cache key per mesh instance so Three.js never reuses one compiled
        // program for multiple segmentation surfaces. Shared programs would alias
        // uScreenDoorOpacity and break screen-door + per-mesh opacity when several
        // surfaces exist (e.g. regions + PBC + pocket types).
        const screenDoorMeshKey = (obj && obj.uuid) ? String(obj.uuid) : String(mat.uuid || "");
        mat.customProgramCacheKey = function() {
          const orig = mat.userData.screenDoorOriginalProgramKey;
          const base = (typeof orig === "function") ? orig.call(this) : "";
          return base + "|screenDoorV3|" + screenDoorMeshKey;
        };
        if (!mat.userData.screenDoorEnabled) mat.needsUpdate = true;
        mat.userData.screenDoorEnabled = true;
        if (mat.userData.screenDoorUniform) {
          mat.userData.screenDoorUniform.value = clampedOpacity;
        }
      } else {
        mat.transparent = mat.opacity < 1.0;
        mat.depthWrite = mat.opacity >= 0.999;
        mat.alphaTest = 0.0;
        mat.dithering = false;
        if (mat.userData.screenDoorEnabled) {
          mat.onBeforeCompile = mat.userData.screenDoorOriginalOnBeforeCompile || function(shader) {};
          mat.customProgramCacheKey = mat.userData.screenDoorOriginalProgramKey || function() { return ""; };
          mat.userData.screenDoorUniform = null;
          mat.userData.screenDoorEnabled = false;
          mat.needsUpdate = true;
        }
      }
    }

    function applySurfaceScreenDoorAll() {
      Object.entries(objectsById).forEach(function(entry) {
        const tid = entry[0];
        const obj = entry[1];
        if (isSurfaceMeshId(tid)) applySurfaceScreenDoorForObject(obj);
      });
    }

    function addMesh(trace) {
      if (!trace.data.positions || !trace.data.positions.length || !trace.data.indices || !trace.data.indices.length) {
        console.warn("Skipping empty mesh trace:", trace.id);
        return null;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(trace.data.positions, 3));
      if (trace.data.colors && trace.data.colors.length) {
        geo.setAttribute("color", new THREE.Float32BufferAttribute(trace.data.colors, 3));
      }
      geo.setIndex(trace.data.indices);
      geo.computeVertexNormals();
      const style = styleFor(trace);
      const hasColors = !!(trace.data.colors && trace.data.colors.length);
      const clearcoat = style.clearcoat !== undefined ? style.clearcoat : 0.0;
      const opacity = style.opacity !== undefined ? style.opacity : 1.0;
      const isOpaque = opacity >= 1.0;
      let mat;
      if (typeof THREE.MeshPhysicalMaterial !== "undefined" && clearcoat > 0) {
        mat = new THREE.MeshPhysicalMaterial({
          vertexColors: hasColors,
          transparent: !isOpaque,
          opacity: opacity,
          roughness: style.roughness !== undefined ? style.roughness : 0.38,
          metalness: style.metalness !== undefined ? style.metalness : 0.0,
          side: style.side === "FrontSide" ? THREE.FrontSide : THREE.DoubleSide,
          clearcoat: clearcoat,
          clearcoatRoughness: style.clearcoatRoughness !== undefined ? style.clearcoatRoughness : 0.1,
          depthWrite: isOpaque,
        });
      } else {
        mat = new THREE.MeshStandardMaterial({
          vertexColors: hasColors,
          transparent: !isOpaque,
          opacity: opacity,
          roughness: style.roughness !== undefined ? style.roughness : 0.38,
          metalness: style.metalness !== undefined ? style.metalness : 0.0,
          side: style.side === "FrontSide" ? THREE.FrontSide : THREE.DoubleSide,
          depthWrite: isOpaque,
        });
      }
      // Apply style color when no per-vertex colors (e.g. aromatic highlights)
      if (!hasColors && style.color !== undefined) {
        mat.color.copy(threeColorFromStyle(style.color, 0xffffff));
      }
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = trace.id;
      mesh.userData.originalSide = mat.side;
      if (isSurfaceMeshId(trace.id)) applySurfaceScreenDoorForObject(mesh);
      scene.add(mesh);
      objectsById[trace.id] = mesh;
      return mesh;
    }

    function addLineSegments(trace, colorOverride) {
      const pts = trace.data.positions || [];
      if (!pts.length) { console.warn("Skipping empty line trace:", trace.id); return null; }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      const style = styleFor(trace);
      const raw = colorOverride !== undefined ? colorOverride : (style.color !== undefined ? style.color : 0xffffff);
      const color = threeColorFromStyle(raw, 0xffffff);
      let mat;
      if (style.dashSize !== undefined) {
        mat = new THREE.LineDashedMaterial({
          color: color,
          dashSize: style.dashSize,
          gapSize: style.gapSize !== undefined ? style.gapSize : 0.5,
          linewidth: style.linewidth || 1
        });
      } else {
        mat = new THREE.LineBasicMaterial({ color: color, linewidth: style.linewidth || 1 });
      }
      // Support style.depthTest (default true; set false for framework scatter).
      // Store original so ortho toggle does not stomp it.
      if (style.depthTest !== undefined) {
        mat.depthTest = !!style.depthTest;
        mat.userData.forceDepthTest = !!style.depthTest;
      }
      const lines = new THREE.LineSegments(geo, mat);
      if (style.dashSize !== undefined) lines.computeLineDistances();
      lines.name = trace.id;
      scene.add(lines);
      objectsById[trace.id] = lines;
      return lines;
    }

    function addPoints(trace, colorOverride, sizeOverride) {
      const pts = trace.data.positions || [];
      if (!pts.length) { console.warn("Skipping empty points trace:", trace.id); return null; }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      if (trace.data.colors && trace.data.colors.length) {
        geo.setAttribute("color", new THREE.Float32BufferAttribute(trace.data.colors, 3));
      }
      const style = styleFor(trace);
      const baseSize = (sizeOverride !== undefined ? sizeOverride : pointsBaseSize(trace)) * 1.3;
      const size = baseSize * pointSizeMult * (isOrtho ? 2.0 : 1.0);
      const sym = style.symbol || "circle";
      const tex = spriteTextures[sym] || spriteTextures["circle"];
      const mat = new THREE.PointsMaterial({
        size: size,
        map: tex,
        vertexColors: !!(trace.data.colors && trace.data.colors.length),
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.95,
      });
      mat.userData.baseSize = baseSize;
      // Support style.depthTest (default true; set false for framework scatter).
      // Store original so ortho toggle does not stomp it.
      if (style.depthTest !== undefined) {
        mat.depthTest = !!style.depthTest;
        mat.userData.forceDepthTest = !!style.depthTest;
      }
      if (colorOverride !== undefined && (!trace.data.colors || !trace.data.colors.length)) {
        mat.color.copy(threeColorFromStyle(colorOverride, 0xffffff));
      } else if (!trace.data.colors || !trace.data.colors.length) {
        mat.color.copy(threeColorFromStyle(style.color, 0xffffff));
      }
      const points = new THREE.Points(geo, mat);
      points.name = trace.id;
      scene.add(points);
      objectsById[trace.id] = points;
      return points;
    }

    function addInstancedSpheres(trace, segLat, segLon) {
      const n = trace.data.n_atoms || 0;
      if (!n) { console.warn("Skipping empty instanced trace:", trace.id); return null; }
      const sw = segLat || 14;
      const sh = segLon || 14;
      const geo = new THREE.SphereBufferGeometry(1.0, sw, sh);
      const style = styleFor(trace);
      const positions = trace.data.positions || [];
      const hasInstanceColors = !!(trace.data && trace.data.colors && trace.data.colors.length >= 3);
      const c = style.color;
      const col = hasInstanceColors
        ? new THREE.Color(1.0, 1.0, 1.0)
        : (Array.isArray(c) && c.length >= 3
            ? new THREE.Color(c[0], c[1], c[2])
            : (typeof c === "number" ? new THREE.Color(c) : new THREE.Color(0xff69b4)));
      const mat = new THREE.MeshPhongMaterial({
        color: col,
        shininess: style.shininess !== undefined ? style.shininess : 10,
        specular: style.specular !== undefined ? style.specular : 0x151515,
      });
      const mesh = new THREE.InstancedMesh(geo, mat, n);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      if (mesh.instanceColor) mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);

      // Populate instance transforms (position + scale per atom)
      const radii = trace.data.radii || [];
      const colors = trace.data.colors || [];
      const dummy = new THREE.Object3D();
      for (let a = 0; a < n; a++) {
        const x = positions[a * 3] || 0;
        const y = positions[a * 3 + 1] || 0;
        const z = positions[a * 3 + 2] || 0;
        const r = (radii[a]) || 0.6;
        dummy.position.set(x, y, z);
        dummy.scale.set(r, r, r);
        dummy.updateMatrix();
        mesh.setMatrixAt(a, dummy.matrix);
        if (mesh.setColorAt && colors.length >= (a * 3 + 3)) {
          mesh.setColorAt(a, new THREE.Color(colors[a * 3], colors[a * 3 + 1], colors[a * 3 + 2]));
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

      mesh.name = trace.id;
      scene.add(mesh);
      objectsById[trace.id] = mesh;
      return mesh;
    }

    function addUnitCellLines(trace) {
      const st = styleFor(trace);
      const c = st.color !== undefined ? st.color : 0xffa500;
      return addLineSegments(trace, c);
    }

    function replaceMeshGeometry(mesh, trace) {
      const oldGeo = mesh.geometry;
      const geo = new THREE.BufferGeometry();
      const pos = trace.data.positions || [];
      geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(pos), 3));
      if (trace.data.colors && trace.data.colors.length) {
        geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(trace.data.colors), 3));
      }
      geo.setIndex(trace.data.indices);
      geo.computeVertexNormals();
      mesh.geometry = geo;
      if (oldGeo) oldGeo.dispose();
      if (isSurfaceMeshId(trace.id)) applySurfaceScreenDoorForObject(mesh);
    }

    function replaceLineGeometry(lines, trace) {
      const oldGeo = lines.geometry;
      const geo = new THREE.BufferGeometry();
      const pts = trace.data.positions || [];
      geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(pts), 3));
      lines.geometry = geo;
      if (oldGeo) oldGeo.dispose();
      if (lines.material && lines.material.dashSize !== undefined) lines.computeLineDistances();
    }

    function replacePointsGeometry(points, trace) {
      const oldGeo = points.geometry;
      const geo = new THREE.BufferGeometry();
      const pts = trace.data.positions || [];
      geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(pts), 3));
      if (trace.data.colors && trace.data.colors.length) {
        geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(trace.data.colors), 3));
      }
      points.geometry = geo;
      if (oldGeo) oldGeo.dispose();
      const style = styleFor(trace);
      const baseSize = pointsBaseSize(trace) * 1.3;
      points.material.userData.baseSize = baseSize;
      applyPointSize(points, baseSize);
      points.material.vertexColors = !!(trace.data.colors && trace.data.colors.length);
      points.material.sizeAttenuation = false;
      if (!trace.data.colors || !trace.data.colors.length) {
        points.material.color.copy(threeColorFromStyle(style.color, 0xffffff));
      }
    }

    // Build static traces
    const staticTraces = DATA.static && DATA.static.traces ? DATA.static.traces : [];
    staticTraces.forEach(trace => {
      if (trace.type === "mesh") addMesh(trace);
      else if (trace.type === "line_segments") addLineSegments(trace);
      else if (trace.type === "points") addPoints(trace);
      else if (trace.type === "unit_cell_lines") {
        addUnitCellLines(trace);
        // Add A/B/C axis labels at cell corners
        if (trace.data && trace.data.positions && trace.data.positions.length >= 18) {
          const pos = trace.data.positions;
          const labels = [{ t: 'A', x: pos[3], y: pos[4], z: pos[5] }, { t: 'B', x: pos[9], y: pos[10], z: pos[11] }, { t: 'C', x: pos[15], y: pos[16], z: pos[17] }];
          labels.forEach(function(lbl) {
            const c = document.createElement('canvas');
            c.width = 32; c.height = 32;
            const ctx = c.getContext('2d');
            ctx.fillStyle = 'rgba(255,165,0,0.9)';
            ctx.font = 'bold 20px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(lbl.t, 16, 16);
            const tex = new THREE.CanvasTexture(c);
            const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
            const sprite = new THREE.Sprite(mat);
            sprite.position.set(lbl.x, lbl.y, lbl.z);
            sprite.scale.set(1.0, 1.0, 1);
            sprite.name = 'axisLabel_' + lbl.t;
            scene.add(sprite);
          });
        }
      }
      else if (trace.type === "instanced_spheres") {
        if (trace.id === "guests" || trace.id.startsWith("guests_")) {
          return;
        }
        addInstancedSpheres(trace, 14, 14);
      }
    });

    // ---- Lazy surface loading ----
    // When surfaces are stored in a separate sidecar file, fetch and load them
    // after the initial scene is rendered (framework, cell, maxima appear instantly).
    if (DATA.sidecar_surfaces) {
      const surfCount = staticTraces.filter(t => t.data && t.data.__sidecar__).length;
      if (surfCount > 0) {
        const statusEl = document.getElementById("status");
        if (statusEl) statusEl.textContent = "Loading surfaces…";
        fetch(DATA.sidecar_surfaces)
          .then(r => r.json())
          .then(surfaceData => {
            staticTraces.forEach(trace => {
              if (trace.data && trace.data.__sidecar__ && surfaceData[trace.data.id]) {
                trace.data = surfaceData[trace.data.id];
                _unpackBinaryArrays({traces: [trace]});
                const obj = objectsById[trace.id];
                if (obj && trace.type === "mesh") {
                  replaceMeshGeometry(obj, trace);
                }
              }
            });
            if (statusEl) statusEl.textContent = "";
            // Refresh legend/surface controls after lazy load
            applySurfaceScreenDoorAll();
          })
          .catch(e => {
            console.warn("Lazy surface load failed:", e);
            if (statusEl) statusEl.textContent = "";
          });
      }
    }

    // Traces that exist only on animation frames (e.g. flexible framework) — create once from frame 0
    const firstFrameTraces = (getFrameCount() > 0 && getFrame(0) && getFrame(0).traces) ? getFrame(0).traces : [];
    firstFrameTraces.forEach(trace => {
      if (objectsById[trace.id]) return;
      if (trace.type === "mesh") addMesh(trace);
      else if (trace.type === "line_segments") addLineSegments(trace);
      else if (trace.type === "points") addPoints(trace);
      else if (trace.type === "unit_cell_lines") addUnitCellLines(trace);
    });

    // Guest instanced meshes for animation (supports multi-component ids like guests_CO2)
    const guestMeshes = {};
    const lastRenderedGuestFrameById = {};
    const guestTraceIds = new Set();
    staticTraces.forEach(t => {
      if (t.type === "instanced_spheres" && (t.id === "guests" || t.id.startsWith("guests_"))) {
        guestTraceIds.add(t.id);
      }
    });
    firstFrameTraces.forEach(t => {
      if (t.type === "instanced_spheres" && (t.id === "guests" || t.id.startsWith("guests_"))) {
        guestTraceIds.add(t.id);
      }
    });
    guestTraceIds.forEach(id => {
      const staticTrace = staticTraces.find(t => t.id === id);
      const frameTrace = firstFrameTraces.find(t => t.id === id);
      const src = staticTrace || frameTrace;
      if (!src) return;
      const mesh = addInstancedSpheres(src, guestSphereSegments, guestSphereSegments);
      guestMeshes[id] = mesh;
      if (mesh) {
        mesh.userData.maxInstances = src.data.n_atoms || 0;
      }
      lastRenderedGuestFrameById[id] = -999;
    });

    // Sync initial visibility from trace data (honors visible=false for periodic elements)
    const allSetupTraces = [...staticTraces, ...(getFrameCount() && getFrame(0) && getFrame(0).traces ? getFrame(0).traces : [])];
    allSetupTraces.forEach(trace => {
      const obj = objectsById[trace.id];
      if (obj && trace.visible === false) {
        obj.visible = false;
      }
    });

    function setVisibleIfPresent(labelId, predicate) {
      const el = document.getElementById(labelId);
      if (!el) return;
      el.style.display = predicate ? "" : "none";
    }
    function hasStaticOrFrameTrace(pred) {
      if (staticTraces.some(pred)) return true;
      if (!DATA.sidecar) {
        return FRAMES.some(fr => (fr && fr.traces ? fr.traces : []).some(pred));
      }
      const first = getFrame(0);
      return first && first.traces ? first.traces.some(pred) : false;
    }
    setVisibleIfPresent("labelShowCell", hasStaticOrFrameTrace(t => t.id === "cell"));
    setVisibleIfPresent("labelShowFramework", hasStaticOrFrameTrace(t => t.id === "framework" || t.id === "framework_atoms" || t.id === "framework_bonds"));
    setVisibleIfPresent("labelShowSurface", hasStaticOrFrameTrace(t => t.id === "surface"));
    setVisibleIfPresent("labelShowGuests", hasStaticOrFrameTrace(t => t.id === "guests" || (t.id && t.id.startsWith("guests_"))));
    setVisibleIfPresent("labelShowGuestBonds", hasStaticOrFrameTrace(t => t.id && t.id.startsWith("guestbonds_")));
    setVisibleIfPresent("labelShowWindows", hasStaticOrFrameTrace(t => t.id === "windows_internal"));
    setVisibleIfPresent("labelShowPeriodicWindows", hasStaticOrFrameTrace(t => t.id === "windows_periodic"));
    setVisibleIfPresent("labelShowInternalConnections", hasStaticOrFrameTrace(t => t.id === "connections_internal" || t.id === "connections_internal_nodes" || t.id === "maxima_internal"));
    setVisibleIfPresent("labelShowPeriodicConnections", hasStaticOrFrameTrace(t => t.id === "connections_periodic" || t.id === "connections_periodic_nodes" || t.id === "maxima_outside"));

    // Populate dynamic group checkboxes from DATA.groups
    const groupChecksDiv = document.getElementById("groupChecks");
    const groups = DATA.groups || {};
    const mergedGroups = Object.assign({}, groups);
    const groupedTraceIds = new Set();
    Object.values(groups).forEach(function(g) {
      (g && g.traces ? g.traces : []).forEach(function(tid) { groupedTraceIds.add(tid); });
    });
    // Fallback: if surface traces exist but caller did not declare groups,
    // auto-register legend entries so legend and mesh-opacity controls stay in sync.
    [
      ["surface", "Surface"],
      ["surface_regions_raw", "Regions (raw)"],
      ["surface_pbc", "Surface (PBC)"],
      ["surface_regions_pbc", "Regions (PBC merged)"],
      ["surface_pocket", "Pocket types"],
      ["surface_regions_cluster", "Pocket types (cluster)"],
    ].forEach(function(defn) {
      const tid = defn[0];
      const label = defn[1];
      if (objectsById[tid] && !groupedTraceIds.has(tid)) {
        mergedGroups["__auto_" + tid] = { traces: [tid], label: label };
      }
    });
    const seenGroupKeys = new Set();
    Object.entries(mergedGroups).forEach(function(entry) {
      const groupId = entry[0];
      const group = entry[1];
      const traces = group.traces || [];
      if (!traces.some(function(tid) { return objectsById[tid]; })) return;
      const groupKey = (group.label || groupId) + "::" + traces.join(",");
      if (seenGroupKeys.has(groupKey)) return;
      seenGroupKeys.add(groupKey);
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      const startOff = !!group.default_off;
      cb.checked = !startOff;
      if (startOff) {
        (group.traces || []).forEach(function(tid) {
          const obj = objectsById[tid];
          if (obj) obj.visible = false;
        });
      }
      cb.addEventListener("change", function() {
        (group.traces || []).forEach(function(tid) {
          const obj = objectsById[tid];
          if (obj) obj.visible = cb.checked;
        });
        updateCellCopyVisibility();
        updateSurfaceCellVisibility();
        updateFrameworkCellVisibility();
      });
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + (group.label || groupId)));
      // Isolate button — shows only this group, hides everything else
      const isoBtn = document.createElement("button");
      isoBtn.type = "button";
      isoBtn.textContent = "⊕";
      isoBtn.title = "Isolate: show only this, hide everything else. Click again to restore.";
      isoBtn.style.cssText = "font-size:10px;padding:0 3px;margin-left:4px;cursor:pointer;background:none;border:1px solid #666;border-radius:2px;color:inherit;";
      (function(gid, traces) {
        var isoActive = false;
        var savedVisibility = null;
        function deactivateIsolate() {
          if (!isoActive) return;
          if (savedVisibility) {
            Object.entries(savedVisibility).forEach(function(e2) {
              var obj = objectsById[e2[0]];
              if (obj) obj.visible = e2[1];
            });
            savedVisibility = null;
          }
          isoActive = false;
          isoBtn.textContent = "⊕";
          updateCellCopyVisibility();
          updateSurfaceCellVisibility();
          updateFrameworkCellVisibility();
        }
        isoBtn.addEventListener("click", function(e) {
          e.stopPropagation();
          if (isoActive) {
            deactivateIsolate();
          } else {
            // Deactivate any other active isolate button
            var others = document.querySelectorAll("#groupChecks button.isolateActive");
            others.forEach(function(b) { b.click(); });
            // Save current visibility of all objects
            savedVisibility = {};
            Object.entries(objectsById).forEach(function(e2) {
              if (e2[1]) savedVisibility[e2[0]] = e2[1].visible;
            });
            // Hide everything
            Object.values(objectsById).forEach(function(obj) {
              if (obj) obj.visible = false;
            });
            // Show only this group
            (traces || []).forEach(function(tid) {
              var obj = objectsById[tid];
              if (obj) obj.visible = true;
            });
            isoActive = true;
            isoBtn.textContent = "⊖";
            isoBtn.classList.add("isolateActive");
            updateCellCopyVisibility();
            updateSurfaceCellVisibility();
            updateFrameworkCellVisibility();
          }
        });
        // Store deactivate ref for cross-button coordination
        isoBtn._deactivate = deactivateIsolate;
      })(groupId, group.traces || []);
      label.appendChild(isoBtn);
      groupChecksDiv.appendChild(label);
    });

    // Fallback: if nothing expanded the bounding box (no objects created),
    // force a minimal scene center so camera doesn't look at NaN.
    const sceneBox = new THREE.Box3();
    Object.values(objectsById).forEach(obj => {
      if (obj && obj.geometry) {
        try { sceneBox.expandByObject(obj); } catch (e) {}
      }
    });
    // Also expand by unit cell corners if cell trace exists
    const cellObj = objectsById["cell"];
    if (cellObj && cellObj.geometry && cellObj.geometry.attributes && cellObj.geometry.attributes.position) {
      const pos = cellObj.geometry.attributes.position.array;
      for (let i = 0; i < pos.length; i += 3) {
        sceneBox.expandByPoint(new THREE.Vector3(pos[i], pos[i+1], pos[i+2]));
      }
    }
    const sceneCenter = new THREE.Vector3();
    if (sceneBox.isEmpty()) {
      sceneCenter.set(0, 0, 0);
    } else {
      sceneBox.getCenter(sceneCenter);
    }

    var frame = 0;
    let playing = false;
    let lastT = 0;
    let frameDtMs = 1000.0 / Math.max({{fps}}, 0.01);
    const fps = Math.max({{fps}}, 0.01);

        slider.max = String(Math.max(getFrameCount() - 1, 0));
    metaEl.textContent = `Frames: ${getFrameCount()}`;
    controls.target.copy(sceneCenter);
    controls.update();

    // Override default camera to 1.5 * (la, lb, lc) from scene center along (1,1,1)
    let _camOverridePos = null;
    if (cellObj && cellObj.geometry && cellObj.geometry.attributes && cellObj.geometry.attributes.position) {
      const cpos = cellObj.geometry.attributes.position.array;
      const la = Math.sqrt(cpos[3]*cpos[3] + cpos[4]*cpos[4] + cpos[5]*cpos[5]);
      const lb = Math.sqrt(cpos[9]*cpos[9] + cpos[10]*cpos[10] + cpos[11]*cpos[11]);
      const lc = Math.sqrt(cpos[15]*cpos[15] + cpos[16]*cpos[16] + cpos[17]*cpos[17]);
      _camOverridePos = new THREE.Vector3(1.5 * la, 1.5 * lb, 1.5 * lc);
      [perspectiveCam, orthoCam].forEach(function(cam) {
        cam.position.copy(_camOverridePos);
        cam.lookAt(sceneCenter);
      });
      controls.target.copy(sceneCenter);
      controls.update();
      updateOrthoFrustum();
    }

    // Exploded view: multi-directional separation
    //   X-axis: framework+cell_framework left, pore graph+cell_copy right
    //   Y-axis (perpendicular): PBC surface+cell_pbc up, cluster surface+cell_pocket down, raw surface middle
    //   Keep local maxima in the central cell with raw surface (not in pore-graph copy).
    //   Original cell stays at center and is controlled independently by showCell.
    const _explodeGroups = [
      { ids: ["framework", "framework_stick", "framework_spacefill", "framework_atoms", "framework_bonds", "aromatic_rings", "aromatic_highlights", "cell_framework"], x: -1, y: 0 },
      { ids: ["windows_internal", "connections_internal", "connections_internal_nodes",
              "maxima_outside", "windows_periodic", "connections_periodic", "connections_periodic_nodes",
              "cell_copy"], x: 1, y: 0 },
      { ids: ["surface", "surface_regions_raw", "maxima_internal"], x: 0, y: 0 },
      { ids: ["surface_pbc", "surface_regions_pbc", "maxima_pbc", "cell_pbc"], x: 0, y: 1 },
      { ids: ["surface_pocket", "surface_regions_cluster", "cell_pocket"], x: 0, y: -1 },
    ];

    // Create unit-cell copies for exploded view — each group gets its own copy
    let cellCopy = null;
    let cellFramework = null;
    let cellPbc = null;
    let cellPocket = null;
    if (cellObj && cellObj.geometry) {
      cellCopy = new THREE.LineSegments(cellObj.geometry.clone(), cellObj.material.clone());
      cellCopy.name = "cell_copy";
      cellCopy.visible = false;
      scene.add(cellCopy);
      objectsById["cell_copy"] = cellCopy;

      cellFramework = new THREE.LineSegments(cellObj.geometry.clone(), cellObj.material.clone());
      cellFramework.name = "cell_framework";
      cellFramework.visible = false;
      scene.add(cellFramework);
      objectsById["cell_framework"] = cellFramework;

      cellPbc = new THREE.LineSegments(cellObj.geometry.clone(), cellObj.material.clone());
      cellPbc.name = "cell_pbc";
      cellPbc.visible = false;
      scene.add(cellPbc);
      objectsById["cell_pbc"] = cellPbc;

      cellPocket = new THREE.LineSegments(cellObj.geometry.clone(), cellObj.material.clone());
      cellPocket.name = "cell_pocket";
      cellPocket.visible = false;
      scene.add(cellPocket);
      objectsById["cell_pocket"] = cellPocket;
    }

    // Populate per-mesh opacity controls (skip alt-representation meshes)
    const meshOpacityContainer = document.getElementById("meshOpacityContainer");
    if (meshOpacityContainer) {
      const meshTraces = [];
      staticTraces.forEach(t => { if (t.type === "mesh") meshTraces.push(t); });
      if (getFrameCount() > 0 && getFrame(0) && getFrame(0).traces) {
        getFrame(0).traces.forEach(t => {
          if (t.type === "mesh" && !meshTraces.some(mt => mt.id === t.id)) meshTraces.push(t);
        });
      }
      meshTraces.forEach(trace => {
        // Skip alternate-representation meshes — their opacity is controlled by the main slider
        if (trace.id === "framework_stick" || trace.id === "framework_spacefill") return;
        // Skip percolation cluster meshes — grouped by direction in legend
        if (trace.id && trace.id.startsWith("percolation_clusters_")) return;
        // Skip aromatic traces — opacity is hard-coded in style
        if (trace.id === "aromatic_rings" || trace.id === "aromatic_highlights") return;
        const wrapper = document.createElement("div");
        wrapper.style.cssText = "display:flex;align-items:center;gap:4px;";
        const label = document.createElement("label");
        label.style.cssText = "font-size:10px;white-space:nowrap;width:70px;overflow:hidden;text-overflow:ellipsis;";
        label.textContent = (trace.name || trace.id);
        const opacitySlider = document.createElement("input");
        opacitySlider.type = "range";
        opacitySlider.min = "0";
        opacitySlider.max = "1";
        opacitySlider.step = "0.05";
        const st = styleFor(trace);
        var initOpacity = (st && st.opacity !== undefined) ? st.opacity : 1.0;
        opacitySlider.value = initOpacity;
        opacitySlider.style.cssText = "flex:1;";
        opacitySlider.dataset.traceId = trace.id;
        opacitySlider.addEventListener("input", () => {
          const val = parseFloat(opacitySlider.value);
          const targetIds = [trace.id];
          // If this is the main framework slider, also update alternate representations
          if (trace.id === "framework") {
            targetIds.push("framework_stick", "framework_spacefill", "aromatic_rings", "aromatic_highlights");
          }
          targetIds.forEach(tid => {
            const obj = objectsById[tid];
            if (obj && obj.material) {
              var isFw = tid === "framework" || tid === "framework_stick" || tid === "framework_spacefill";
              var hiddenByOpacity = val <= 0.001;
              if (obj.userData && obj.userData.originalCastShadow === undefined) {
                obj.userData.originalCastShadow = obj.castShadow;
                obj.userData.originalReceiveShadow = obj.receiveShadow;
              }
              const opaqueEnough = val >= 0.999;
              if (isFw) {
                // Opacity only — visibility follows Representation + Framework checkbox.
                // At exactly zero, hide to avoid ghost/shadow outlines from postprocessing.
                obj.material.opacity = val;
                obj.material.transparent = val < 1.0;
                // Keep framework depth writes on while fading so bonds remain occluded
                // by atom surfaces and do not appear to poke through.
                obj.material.depthWrite = true;
                obj.material.depthTest = true;
                obj.material.side = THREE.FrontSide;
                obj.visible = val > 0.001;
                obj.castShadow = val > 0.999;
                obj.receiveShadow = val > 0.999;
              } else {
                obj.visible = !hiddenByOpacity;
                obj.castShadow = hiddenByOpacity ? false : !!(obj.userData && obj.userData.originalCastShadow);
                obj.receiveShadow = hiddenByOpacity ? false : !!(obj.userData && obj.userData.originalReceiveShadow);
                obj.material.opacity = val;
                obj.material.transparent = val < 1.0;
                obj.material.depthWrite = opaqueEnough;
              }
              if (obj.userData && obj.userData.originalSide !== undefined) {
                obj.material.side = obj.userData.originalSide;
              }
              if (isSurfaceMeshId(tid)) {
                applySurfaceScreenDoorForObject(obj);
              }
              obj.material.needsUpdate = true;
            }
          });
          if (trace.id === "framework" || trace.id === "framework_atoms" || trace.id === "framework_bonds") {
            updateFrameworkVisibility();
            const activeId = reprSelect.value === "stick"
              ? "framework_stick"
              : (reprSelect.value === "space_filling" ? "framework_spacefill" : "framework");
            const active = objectsById[activeId];
            if (active) active.visible = showFramework.checked && val > 0.001;
            // Also handle scatter traces
            ["framework_atoms", "framework_bonds"].forEach(function(sid) {
              const sobj = objectsById[sid];
              if (sobj) sobj.visible = showFramework.checked && val > 0.001;
            });
          }
        });
        wrapper.appendChild(label);
        wrapper.appendChild(opacitySlider);
        meshOpacityContainer.appendChild(wrapper);
      });
      // Stacking-order note for overlapping mesh traces (e.g. multiple surfaces)
      if (meshTraces.length > 0) {
        const note = document.createElement("div");
        note.style.cssText = "font-size:9px;color:#aaa;margin-top:4px;font-style:italic;";
        note.textContent = "Tip: turn off upper surfaces to see through to lower ones";
        meshOpacityContainer.appendChild(note);
      }
    }

    function setFrame(i) {
      if (getFrameCount() <= 0) {
        statusEl.textContent = "No frames";
        slider.value = "0";
        return;
      }
      frame = Math.max(0, Math.min(i, getFrameCount() - 1));
      const fr = getFrame(frame);
      const traces = fr && fr.traces ? fr.traces : [];
      Object.keys(guestMeshes).forEach(guestId => {
        const guestTrace = traces.find(t => t.id === guestId);
        let guestMesh = guestMeshes[guestId];
        if (!guestTrace || !guestMesh) return;
        if (frame === lastRenderedGuestFrameById[guestId]) return;
        const positions = guestTrace.data.positions;
        const radii = guestTrace.data.radii;
        const n = guestTrace.data.n_atoms || 0;
        const cap = guestMesh.userData.maxInstances || guestMesh.count;
        if (n > cap) {
          scene.remove(guestMesh);
          if (guestMesh.geometry) guestMesh.geometry.dispose();
          guestMesh.material.dispose();
          guestMeshes[guestId] = addInstancedSpheres(guestTrace, guestSphereSegments, guestSphereSegments);
          guestMesh = guestMeshes[guestId];
          guestMesh.userData.maxInstances = n;
        }
        const dummy = new THREE.Object3D();
        for (let a = 0; a < n; a++) {
          const x = positions[a * 3], y = positions[a * 3 + 1], z = positions[a * 3 + 2];
          const r = (radii && radii[a]) || 0.6;
          dummy.position.set(x, y, z);
          dummy.scale.set(r, r, r);
          dummy.updateMatrix();
          guestMesh.setMatrixAt(a, dummy.matrix);
          if (guestMesh.setColorAt && guestTrace.data.colors && guestTrace.data.colors.length >= (a * 3 + 3)) {
            const rr = guestTrace.data.colors[a * 3];
            const gg = guestTrace.data.colors[a * 3 + 1];
            const bb = guestTrace.data.colors[a * 3 + 2];
            guestMesh.setColorAt(a, new THREE.Color(rr, gg, bb));
          }
        }
        guestMesh.count = n;
        guestMesh.instanceMatrix.needsUpdate = true;
        if (guestMesh.instanceColor) guestMesh.instanceColor.needsUpdate = true;
        if (guestMesh.material) {
          if (guestTrace.data.colors && guestTrace.data.colors.length > 0) {
            guestMesh.material.color.setRGB(1.0, 1.0, 1.0);
          }
          guestMesh.material.needsUpdate = true;
        }
        lastRenderedGuestFrameById[guestId] = frame;
      });
      traces.forEach(trace => {
        if (trace.id === "guests" || (trace.id && trace.id.startsWith("guests_"))) return;
        const obj = objectsById[trace.id];
        if (!obj) return;
        if (trace.type === "mesh") replaceMeshGeometry(obj, trace);
        else if (trace.type === "line_segments" || trace.type === "unit_cell_lines") replaceLineGeometry(obj, trace);
        else if (trace.type === "points") replacePointsGeometry(obj, trace);
      });
      statusEl.textContent = `Frame ${frame}`;
      slider.value = String(frame);
    }

    slider.addEventListener("input", () => setFrame(parseInt(slider.value, 10)));
    prevBtn.addEventListener("click", () => setFrame(frame - 1));
    nextBtn.addEventListener("click", () => setFrame(frame + 1));
    playBtn.addEventListener("click", () => {
      playing = !playing;
      playBtn.textContent = playing ? "Pause" : "Play";
    });

    showCell.addEventListener("change", () => {
      const obj = objectsById["cell"];
      if (obj) obj.visible = showCell.checked;
      scene.children.forEach(function(c) { if (c.name && c.name.startsWith("axisLabel_")) c.visible = showCell.checked; });
    });
    function updateFrameworkVisibility() {
      const val = reprSelect.value;
      const fw = objectsById["framework"];
      const stick = objectsById["framework_stick"];
      const sf = objectsById["framework_spacefill"];
      const arRings = objectsById["aromatic_rings"];
      const arHigh = objectsById["aromatic_highlights"];
      // Scatter / instanced traces (atoms + bonds split across two trace IDs)
      const scatterAtoms = objectsById["framework_atoms"];
      const scatterBonds = objectsById["framework_bonds"];
      const stickAtoms = objectsById["framework_stick_atoms"];
      const stickBonds = objectsById["framework_stick_bonds"];
      if (fw) fw.visible = showFramework.checked && val === "ball_and_stick";
      if (stick) stick.visible = showFramework.checked && val === "stick";
      if (sf) sf.visible = showFramework.checked && val === "space_filling";
      if (stickAtoms) stickAtoms.visible = showFramework.checked && val === "stick";
      if (stickBonds) stickBonds.visible = showFramework.checked && val === "stick";
      // Instanced/scatter ball-and-stick: show atoms+bonds for ball_and_stick selection
      if (scatterAtoms && !fw) scatterAtoms.visible = showFramework.checked && val === "ball_and_stick";
      if (scatterBonds && !fw) scatterBonds.visible = showFramework.checked && val === "ball_and_stick";
      if (arRings) arRings.visible = showFramework.checked;
      if (arHigh) arHigh.visible = showFramework.checked;
      updateFrameworkCellVisibility();
    }
    showFramework.addEventListener("change", updateFrameworkVisibility);
    reprSelect.addEventListener("change", updateFrameworkVisibility);
    // Hide representation selector when only scatter traces exist (no mesh variants)
    if (!objectsById["framework"] && !objectsById["framework_stick"] && !objectsById["framework_spacefill"]
        && !objectsById["framework_stick_atoms"]) {
      const reprLabel = reprSelect.parentNode;
      if (reprLabel) reprLabel.style.display = "none";
    }
    // Initialize visibility so only the selected representation is shown
    updateFrameworkVisibility();
    showSurface.addEventListener("change", () => {
      // "Segmentation surface" checkbox controls only the non-PBC/raw surface.
      // PBC/cluster overlays are controlled by their own legend group toggles.
      ["surface", "surface_regions_raw"].forEach(function(id) {
        const obj = objectsById[id];
        if (obj) obj.visible = showSurface.checked;
      });
      updateSurfaceCellVisibility();
    });
    showGuests.addEventListener("change", () => {
      Object.values(guestMeshes).forEach(m => { if (m) m.visible = showGuests.checked; });
    });
    showGuestBonds.addEventListener("change", () => {
      Object.keys(objectsById).forEach(id => {
        if (id && id.startsWith("guestbonds_")) {
          const obj = objectsById[id];
          if (obj) obj.visible = showGuestBonds.checked;
        }
      });
    });
    // Helpers to keep exploded cell-copy visibility in sync with actual trace visibility
    // (works for individual checkboxes, group checkboxes, and any other toggle mechanism)
    function updateCellCopyVisibility() {
      const exploded = parseFloat(explodeSlider.value) > 0;
      if (cellCopy) {
        const poreGraphIds = [
          "windows_internal", "connections_internal", "connections_internal_nodes",
          "maxima_outside", "windows_periodic", "connections_periodic", "connections_periodic_nodes"
        ];
        const anyVisible = poreGraphIds.some(id => {
          const obj = objectsById[id];
          return obj && obj.visible;
        });
        cellCopy.visible = exploded && anyVisible;
      }
    }
    function updateSurfaceCellVisibility() {
      const exploded = parseFloat(explodeSlider.value) > 0;
      if (cellPbc) {
        const ids = ["surface_pbc", "surface_regions_pbc", "maxima_pbc"];
        const anyVisible = ids.some(id => { const obj = objectsById[id]; return obj && obj.visible; });
        cellPbc.visible = exploded && anyVisible;
      }
      if (cellPocket) {
        const ids = ["surface_pocket", "surface_regions_cluster"];
        const anyVisible = ids.some(id => { const obj = objectsById[id]; return obj && obj.visible; });
        cellPocket.visible = exploded && anyVisible;
      }
    }
    function updateFrameworkCellVisibility() {
      const exploded = parseFloat(explodeSlider.value) > 0;
      if (cellFramework) {
        const ids = ["framework", "framework_stick", "framework_spacefill", "framework_atoms", "framework_bonds"];
        const anyVisible = ids.some(id => { const obj = objectsById[id]; return obj && obj.visible; });
        cellFramework.visible = exploded && anyVisible;
      }
    }

    showWindows.addEventListener("change", () => {
      const obj = objectsById["windows_internal"];
      if (obj) obj.visible = showWindows.checked;
      updateCellCopyVisibility();
    });
    showPeriodicWindows.addEventListener("change", () => {
      const obj = objectsById["windows_periodic"];
      if (obj) obj.visible = showPeriodicWindows.checked;
      updateCellCopyVisibility();
    });
    showInternalConnections.addEventListener("change", () => {
      const lineObj = objectsById["connections_internal"];
      const nodeObj = objectsById["connections_internal_nodes"];
      const maxObj = objectsById["maxima_internal"];
      if (lineObj) lineObj.visible = showInternalConnections.checked;
      if (nodeObj) nodeObj.visible = showInternalConnections.checked;
      if (maxObj) maxObj.visible = showInternalConnections.checked;
      updateCellCopyVisibility();
    });
    showPeriodicConnections.addEventListener("change", () => {
      const lineObj = objectsById["connections_periodic"];
      const nodeObj = objectsById["connections_periodic_nodes"];
      const maxObj = objectsById["maxima_outside"];
      if (lineObj) lineObj.visible = showPeriodicConnections.checked;
      if (nodeObj) nodeObj.visible = showPeriodicConnections.checked;
      if (maxObj) maxObj.visible = showPeriodicConnections.checked;
      updateCellCopyVisibility();
    });
    pivotSceneBtn.addEventListener("click", () => {
      controls.target.copy(sceneCenter);
      controls.update();
    });
    pivotOriginBtn.addEventListener("click", () => {
      controls.target.set(0, 0, 0);
      controls.update();
    });

    // ---- New dynamic controls ----
    const _camOrigPos = _camOverridePos ? _camOverridePos.clone() : new THREE.Vector3(scaledEyeArr[0], scaledEyeArr[1], scaledEyeArr[2]);
    const _camOrig = { pos: _camOrigPos, target: sceneCenter.clone() };
    resetCamBtn.addEventListener("click", () => {
      camera.position.copy(_camOrig.pos);
      controls.target.copy(_camOrig.target);
      camera.lookAt(_camOrig.target);
      controls.update();
      updateOrthoFrustum();
    });

    // Quick view along principal axes — preserves current zoom distance
    function setViewAxis(dir) {
      const target = controls.target.clone();
      const dist = camera.position.distanceTo(target);
      const newPos = target.clone().add(dir.clone().multiplyScalar(dist));
      camera.position.copy(newPos);
      camera.lookAt(target);
      controls.update();
      updateOrthoFrustum();
    }
    function setViewDirectionFromInputs() {
      const dirNorm = parseNormalizedVector(viewDirXInput, viewDirYInput, viewDirZInput, new THREE.Vector3(1, 1, 1));
      setViewAxis(dirNorm);
    }
    document.getElementById("viewX").addEventListener("click", () => setViewAxis(new THREE.Vector3(1, 0, 0)));
    document.getElementById("viewNegX").addEventListener("click", () => setViewAxis(new THREE.Vector3(-1, 0, 0)));
    document.getElementById("viewY").addEventListener("click", () => setViewAxis(new THREE.Vector3(0, 1, 0)));
    document.getElementById("viewNegY").addEventListener("click", () => setViewAxis(new THREE.Vector3(0, -1, 0)));
    document.getElementById("viewZ").addEventListener("click", () => setViewAxis(new THREE.Vector3(0, 0, 1)));
    document.getElementById("viewNegZ").addEventListener("click", () => setViewAxis(new THREE.Vector3(0, 0, -1)));
    applyLightDirBtn.addEventListener("click", applyDirectionalLightVector);
    applyViewDirBtn.addEventListener("click", setViewDirectionFromInputs);

    // HUD toggle — hide/show the legend sidebar
    var _hudVisible = true;
    var _hud = document.getElementById("hud");
    var _app = document.getElementById("app");
    var _showHudTab = document.getElementById("showHudTab");
    function setHudVisible(visible) {
      _hudVisible = visible;
      _hud.style.display = visible ? "" : "none";
      _app.style.right = visible ? "200px" : "0";
      if (_showHudTab) _showHudTab.style.display = visible ? "none" : "block";
      try { resizeRenderer(); } catch (e) {}
    }
    document.getElementById("toggleHud").addEventListener("click", function() {
      setHudVisible(!_hudVisible);
    });
    if (_showHudTab) {
      _showHudTab.addEventListener("click", function() {
        setHudVisible(true);
      });
    }

    // Legend fold/unfold
    var _legendVisible = true;
    var _legendSection = document.getElementById("groupChecks");
    var _legendToggleBtn = document.getElementById("toggleLegend");
    if (_legendToggleBtn && _legendSection) {
      _legendToggleBtn.addEventListener("click", function() {
        _legendVisible = !_legendVisible;
        _legendSection.style.display = _legendVisible ? "" : "none";
        _legendToggleBtn.textContent = _legendVisible ? "Fold Legend \u25B2" : "Show Legend \u25BC";
      });
    }

    let autoRotating = false;
    function getAutoRotateAxis() {
      var x = parseFloat(rotAxisXInput && rotAxisXInput.value);
      var y = parseFloat(rotAxisYInput && rotAxisYInput.value);
      var z = parseFloat(rotAxisZInput && rotAxisZInput.value);
      if (!isFinite(x)) x = 0;
      if (!isFinite(y)) y = 0;
      if (!isFinite(z)) z = 1;
      var axis = new THREE.Vector3(x, y, z);
      if (axis.lengthSq() < 1e-8) axis.set(0, 0, 1);
      axis.normalize();
      return axis;
    }
    toggleRotateBtn.addEventListener("click", () => {
      autoRotating = !autoRotating;
      controls.autoRotate = false;
      toggleRotateBtn.textContent = autoRotating ? "Stop Rotate" : "Auto-Rotate";
    });

    // Per-mesh opacity sliders are populated dynamically after scene init

    ptSizeSlider.addEventListener("input", () => {
      pointSizeMult = parseFloat(ptSizeSlider.value) || 1.0;
      Object.values(objectsById).forEach(function(obj) {
        if (obj && obj.material && obj.material.isPointsMaterial) {
          var base = obj.material.userData && obj.material.userData.baseSize ? obj.material.userData.baseSize : obj.material.size;
          applyPointSize(obj, base);
        }
      });
    });
    if (screenDoorToggle) {
      screenDoorToggle.addEventListener("change", () => {
        applySurfaceScreenDoorAll();
      });
    }

    let speedMult = 1.0;
    speedSlider.addEventListener("input", () => {
      speedMult = parseFloat(speedSlider.value);
      if (getFrameCount() > 1) {
        frameDtMs = 1000.0 / (fps * speedMult);
      }
    });

    ambSlider.addEventListener("input", () => {
      ambient.intensity = parseFloat(ambSlider.value);
    });
    dirSlider.addEventListener("input", () => {
      const mult = parseFloat(dirSlider.value);
      dirKey.intensity = (_lightOrig.key !== undefined ? _lightOrig.key : 0.48) * mult;
      dirFill.intensity = (_lightOrig.fill !== undefined ? _lightOrig.fill : 0.32) * mult;
    });

    const _explodeOrig = {};
    explodeSlider.addEventListener("input", () => {
      const dist = parseFloat(explodeSlider.value);
      const spacing = dist * 8.0;
      _explodeGroups.forEach(function(group) {
        const offsetX = group.x * spacing;
        const offsetY = group.y * spacing;
        group.ids.forEach(function(tid) {
          const obj = objectsById[tid];
          if (!obj) return;
          if (_explodeOrig[tid] === undefined) {
            _explodeOrig[tid] = obj.position.clone();
          }
          obj.position.copy(_explodeOrig[tid]).add(new THREE.Vector3(offsetX, offsetY, 0));
        });
      });
      // Show/hide the cell copies when exploded — each follows actual trace visibility
      updateFrameworkCellVisibility();
      updateCellCopyVisibility();
      updateSurfaceCellVisibility();
    });

    bgPicker.addEventListener("input", () => {
      updateSceneBackground();
    });

    // Gradient background modes
    var _bgGradientMode = "none";
    function setCheckerboardBackground(enabled) {
      if (enabled) {
        _app.style.backgroundImage =
          "linear-gradient(45deg, #d8d8d8 25%, transparent 25%), " +
          "linear-gradient(-45deg, #d8d8d8 25%, transparent 25%), " +
          "linear-gradient(45deg, transparent 75%, #d8d8d8 75%), " +
          "linear-gradient(-45deg, transparent 75%, #d8d8d8 75%)";
        _app.style.backgroundSize = "20px 20px";
        _app.style.backgroundPosition = "0 0, 0 10px, 10px -10px, -10px 0px";
      } else {
        _app.style.backgroundImage = "";
      }
    }
    function updateSceneBackground() {
      var color = new THREE.Color(bgPicker.value);
      if (_bgGradientMode === "transparent") {
        scene.background = null;
        renderer.setClearColor(0x000000, 0.0);
        setCheckerboardBackground(true);
        return;
      }
      renderer.setClearColor(0x000000, 1.0);
      setCheckerboardBackground(false);
      if (_bgGradientMode === "none") {
        scene.background = color;
        return;
      }
      var canvas = document.createElement("canvas");
      canvas.width = 256; canvas.height = 256;
      var ctx = canvas.getContext("2d");
      var dark = color.clone().multiplyScalar(0.3);
      var grad;
      if (_bgGradientMode === "lr") {
        grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
        grad.addColorStop(0, dark.getStyle());
        grad.addColorStop(0.5, color.getStyle());
        grad.addColorStop(1, dark.getStyle());
      } else if (_bgGradientMode === "tb") {
        grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
        grad.addColorStop(0, dark.getStyle());
        grad.addColorStop(0.5, color.getStyle());
        grad.addColorStop(1, dark.getStyle());
      } else if (_bgGradientMode === "radial") {
        grad = ctx.createRadialGradient(128, 128, 0, 128, 128, 180);
        grad.addColorStop(0, color.getStyle());
        grad.addColorStop(1, dark.getStyle());
      }
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      var tex = new THREE.CanvasTexture(canvas);
      scene.background = tex;
    }
    document.getElementById("bgGradientNone").addEventListener("click", () => { _bgGradientMode = "none"; updateSceneBackground(); });
    document.getElementById("bgGradientLR").addEventListener("click", () => { _bgGradientMode = "lr"; updateSceneBackground(); });
    document.getElementById("bgGradientTB").addEventListener("click", () => { _bgGradientMode = "tb"; updateSceneBackground(); });
    document.getElementById("bgGradientRadial").addEventListener("click", () => { _bgGradientMode = "radial"; updateSceneBackground(); });
    document.getElementById("bgGradientTransparent").addEventListener("click", () => { _bgGradientMode = "transparent"; updateSceneBackground(); });

    // Measurement tool
    let measuring = false;
    let measurePts = [];
    let measureLine = null;
    let measureLabel = null;
    measureBtn.addEventListener("click", () => {
      measuring = !measuring;
      measureBtn.textContent = measuring ? "Cancel Meas." : "Measure";
      if (!measuring) {
        const markers = measurePts.marker || [];
        measurePts = [];
        if (measureLine) { scene.remove(measureLine); measureLine.geometry.dispose(); measureLine.material.dispose(); measureLine = null; }
        if (measureLabel) { scene.remove(measureLabel); measureLabel = null; }
        markers.forEach(function(m) { scene.remove(m); m.geometry.dispose(); m.material.dispose(); });
      }
    });

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.3;

    function hoverHitPriority(obj, hit) {
      if (obj instanceof THREE.Points) return 6;
      if (hit && hit.face && obj instanceof THREE.Mesh) return 5;
      if (obj instanceof THREE.LineSegments) return 4;
      if (obj instanceof THREE.InstancedMesh) return 3;
      if (obj instanceof THREE.Mesh) return 2;
      return 1;
    }

    const mouse = new THREE.Vector2();
    let hoverRaf = null;
    let pendingHoverEv = null;
    function runHoverRaycast(ev) {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const pickables = Object.values(objectsById).filter(o => o && o.visible);
      const hits = raycaster.intersectObjects(pickables, false);
      if (!hits.length) {
        tooltip.style.display = "none";
        return;
      }
      // Pick the hit with the highest hover priority; tie-break by scene order (last rendered = on top)
      const hit = hits.reduce((best, h) => {
        const bestPri = hoverHitPriority(best.object, best);
        const hPri = hoverHitPriority(h.object, h);
        if (hPri !== bestPri) return hPri > bestPri ? h : best;
        const bestIdx = scene.children.indexOf(best.object);
        const hIdx = scene.children.indexOf(h.object);
        return hIdx > bestIdx ? h : best;
      });
      let txt = "";
      if (hit.object && hit.object.name && (hit.object.name === "guests" || hit.object.name.startsWith("guests_"))) {
        const idx = hit.instanceId ?? -1;
        const fr = getFrame(frame);
        const traces = fr && fr.traces ? fr.traces : [];
        const gt = traces.find(t => t.id === hit.object.name);
        if (gt) {
          const p = gt.data.positions.slice(idx * 3, idx * 3 + 3);
          const s = gt.data.symbols[idx] || "?";
          txt = `trace: ${hit.object.name}\nframe: ${frame}\natom: ${idx}\nelement: ${s}\nxyz: (${p[0].toFixed(3)}, ${p[1].toFixed(3)}, ${p[2].toFixed(3)})`;
        }
      } else if (hit.object.name && objectsById[hit.object.name]) {
        const frNow = getFrame(frame);
        const tracesNow = frNow && frNow.traces ? frNow.traces : [];
        const trace = tracesNow.find(t => t.id === hit.object.name) || staticTraces.find(t => t.id === hit.object.name);
        if (trace && trace.hover && hit.face) {
          const vi = hit.face.a;
          const hoverTxt = (vi >= 0 && vi < trace.hover.length) ? trace.hover[vi] : "";
          const traceName = trace.name || hit.object.name;
          txt = `trace: ${traceName}\n${hoverTxt}\nxyz: (${hit.point.x.toFixed(3)}, ${hit.point.y.toFixed(3)}, ${hit.point.z.toFixed(3)})`;
        } else if (trace && trace.hover && hit.index != null && hit.index >= 0 && hit.index < trace.hover.length) {
          txt = trace.hover[hit.index];
        } else if (hit.object instanceof THREE.LineSegments && trace && trace.hover && hit.index != null) {
          const segIdx = Math.floor(hit.index / 2);
          if (segIdx >= 0 && segIdx < trace.hover.length) txt = trace.hover[segIdx];
        } else if (hit.face) {
          const traceName = trace ? (trace.name || hit.object.name) : hit.object.name;
          txt = `trace: ${traceName}\nxyz: (${hit.point.x.toFixed(3)}, ${hit.point.y.toFixed(3)}, ${hit.point.z.toFixed(3)})`;
          const maxPts = staticTraces.filter(t => t.type === "points" && t.data && t.data.positions && t.data.sizes);
          let bestDist = Infinity, bestSize = null;
          maxPts.forEach(function(mt) {
            const mp = mt.data.positions;
            const ms = mt.data.sizes;
            for (let k = 0; k < mp.length; k += 3) {
              const dx = hit.point.x - mp[k], dy = hit.point.y - mp[k+1], dz = hit.point.z - mp[k+2];
              const d = dx*dx + dy*dy + dz*dz;
              if (d < bestDist && ms[k/3] !== undefined) { bestDist = d; bestSize = ms[k/3]; }
            }
          });
          if (bestSize !== null && bestDist < 100) txt += `\nnearest maxima size: ${bestSize.toFixed(3)} A`;
        }
      }
      if (!txt || !String(txt).trim()) {
        tooltip.style.display = "none";
        return;
      }
      tooltip.style.display = "block";
      tooltip.style.left = `${ev.clientX + 12}px`;
      tooltip.style.top = `${ev.clientY + 12}px`;
      tooltip.textContent = txt;
    }
    function onPointerMove(ev) {
      pendingHoverEv = ev;
      if (hoverRaf != null) return;
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = null;
        if (pendingHoverEv) runHoverRaycast(pendingHoverEv);
        pendingHoverEv = null;
      });
    }
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerleave", () => tooltip.style.display = "none");
    renderer.domElement.addEventListener("pointerdown", function(ev) {
      if (!measuring) return;
      const rc = new THREE.Raycaster();
      rc.setFromCamera(new THREE.Vector2((ev.clientX / window.innerWidth) * 2 - 1, -(ev.clientY / window.innerHeight) * 2 + 1), camera);
      const hits = rc.intersectObjects(Object.values(objectsById), false);
      if (!hits.length) return;
      const pt = hits[0].point.clone();
      // Keep only last point if already have 2
      if (measurePts.length >= 2) {
        measurePts = [measurePts[measurePts.length - 1]];
        if (measurePts.marker && measurePts.marker.length > 1) {
          const old = measurePts.marker.shift();
          scene.remove(old); old.geometry.dispose(); old.material.dispose();
        }
      }
      measurePts.push(pt);
      // Drop marker sphere
      const markerGeo = new THREE.SphereGeometry(0.08, 8, 8);
      const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
      const marker = new THREE.Mesh(markerGeo, markerMat);
      marker.position.copy(pt);
      scene.add(marker);
      measurePts.marker = measurePts.marker || [];
      measurePts.marker.push(marker);
      if (measurePts.length >= 2) {
        const p1 = measurePts[0];
        const p2 = measurePts[1];
        const dist = p1.distanceTo(p2).toFixed(3);
        if (measureLine) { scene.remove(measureLine); measureLine.geometry.dispose(); measureLine.material.dispose(); }
        const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 });
        measureLine = new THREE.Line(lineGeo, lineMat);
        scene.add(measureLine);
        if (measureLabel) { scene.remove(measureLabel); }
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(0,0,0,0.8)';
        ctx.fillRect(0, 0, 256, 64);
        ctx.fillStyle = '#0f0';
        ctx.font = 'bold 28px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(dist + ' Å', 128, 34);
        const tex = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
        measureLabel = new THREE.Sprite(spriteMat);
        measureLabel.position.copy(mid);
        measureLabel.scale.set(1.2, 0.3, 1);
        scene.add(measureLabel);
      }
    });

    let isRecordingVideo = false;
    function animate(t) {
      requestAnimationFrame(animate);
      const nowSec = (t || 0) * 0.001;
      const dt = Math.max(0.0, Math.min(0.1, nowSec - (animate._lastSec || nowSec)));
      animate._lastSec = nowSec;
      if (playing && getFrameCount() > 1 && (t - lastT) >= frameDtMs) {
        setFrame((frame + 1) % getFrameCount());
        lastT = t;
      }
      if (autoRotating) {
        const target = controls.target.clone();
        const axis = getAutoRotateAxis();
        const omega = 0.8 * (isRecordingVideo ? 0.55 : 1.0); // rad/s (slow slightly while recording)
        const rel = camera.position.clone().sub(target);
        rel.applyAxisAngle(axis, omega * dt);
        camera.position.copy(target.clone().add(rel));
        camera.lookAt(target);
      }
      controls.update();
      if (composer) {
        composer.render();
      } else {
        renderer.render(scene, camera);
      }
    }

    window.addEventListener("resize", () => {
      const vp = getRenderViewport();
      const w = vp.w;
      const h = vp.h;
      perspectiveCam.aspect = w / h;
      perspectiveCam.updateProjectionMatrix();
      renderer.setSize(w, h);
      document.getElementById("app").style.right = vp.hudW + "px";
      updateOrthoFrustum();
      if (composer) composer.setSize(w, h);
    });
    window.dispatchEvent(new Event("resize"));

    // Export PNG button
    document.getElementById("exportPng").addEventListener("click", function() {
      const resMult = parseInt(pngResSelect.value, 10) || 1;
      // Screen-door (shader-discard) can produce moire-like export artifacts
      // at high-resolution readback. Export with standard alpha path, then restore.
      const restoreScreenDoor = !!(screenDoorToggle && screenDoorToggle.checked);
      if (restoreScreenDoor) {
        screenDoorToggle.checked = false;
        applySurfaceScreenDoorAll();
      }
      const oldW = renderer.domElement.width;
      const oldH = renderer.domElement.height;
      if (resMult > 1) {
        renderer.setSize(oldW * resMult, oldH * resMult, false);
        camera.aspect = (oldW * resMult) / (oldH * resMult);
        camera.updateProjectionMatrix();
        updateOrthoFrustum();
      }
      if (composer) { composer.render(); } else { renderer.render(scene, camera); }
      const link = document.createElement("a");
      link.download = "figure3d_export_" + resMult + "x.png";
      link.href = renderer.domElement.toDataURL("image/png");
      link.click();
      if (resMult > 1) {
        renderer.setSize(oldW, oldH, false);
        camera.aspect = oldW / oldH;
        camera.updateProjectionMatrix();
        updateOrthoFrustum();
      }
      if (restoreScreenDoor) {
        screenDoorToggle.checked = true;
        applySurfaceScreenDoorAll();
        if (composer) { composer.render(); } else { renderer.render(scene, camera); }
      }
    });

    // Record video (WebM) via MediaRecorder + canvas.captureStream
    const recordBtn = document.getElementById("recordVideo");
    let mediaRecorder = null;
    let recordChunks = [];
    let recordTimer = null;
    let recordRenderState = null;
    function beginRecordingRenderScale() {
      const resMult = parseInt(recResSelect.value, 10) || 1;
      const oldW = renderer.domElement.width;
      const oldH = renderer.domElement.height;
      recordRenderState = { oldW, oldH };
      if (resMult > 1) {
        const newW = oldW * resMult;
        const newH = oldH * resMult;
        renderer.setSize(newW, newH, false);
        perspectiveCam.aspect = newW / newH;
        perspectiveCam.updateProjectionMatrix();
        updateOrthoFrustum();
        if (composer) composer.setSize(newW, newH);
      }
    }
    function restoreRecordingRenderScale() {
      if (!recordRenderState) return;
      const oldW = recordRenderState.oldW;
      const oldH = recordRenderState.oldH;
      renderer.setSize(oldW, oldH, false);
      perspectiveCam.aspect = oldW / oldH;
      perspectiveCam.updateProjectionMatrix();
      updateOrthoFrustum();
      if (composer) composer.setSize(oldW, oldH);
      recordRenderState = null;
    }
    recordBtn.addEventListener("click", function() {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        isRecordingVideo = false;
        recordBtn.textContent = "Record";
        recordBtn.style.background = "";
        recordBtn.style.color = "";
        if (recordTimer) { clearInterval(recordTimer); recordTimer = null; }
        return;
      }
      beginRecordingRenderScale();
      let stream;
      try {
        stream = renderer.domElement.captureStream(30);
      } catch(e) {
        restoreRecordingRenderScale();
        alert("Video recording not supported in this browser.");
        return;
      }
      let options = { mimeType: "video/webm;codecs=vp9", videoBitsPerSecond: 22000000 };
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options = { mimeType: "video/webm;codecs=vp8", videoBitsPerSecond: 16000000 };
      }
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options = { mimeType: "video/webm", videoBitsPerSecond: 12000000 };
      }
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        restoreRecordingRenderScale();
        alert("WebM recording not supported in this browser.");
        return;
      }
      mediaRecorder = new MediaRecorder(stream, options);
      isRecordingVideo = true;
      recordChunks = [];
      mediaRecorder.ondataavailable = function(e) {
        if (e.data && e.data.size > 0) recordChunks.push(e.data);
      };
      mediaRecorder.onstop = function() {
        isRecordingVideo = false;
        restoreRecordingRenderScale();
        const blob = new Blob(recordChunks, { type: options.mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "figure3d_movie.webm";
        a.click();
        setTimeout(function() { URL.revokeObjectURL(url); }, 2000);
      };
      mediaRecorder.onerror = function() {
        isRecordingVideo = false;
        restoreRecordingRenderScale();
      };
      mediaRecorder.start(100);
      recordBtn.textContent = "Stop";
      recordBtn.style.background = "#cc3333";
      recordBtn.style.color = "#fff";
    });

    // Record GIF (via gif.js) — only meaningful when there are frames
    const recordGifBtn = document.getElementById("recordGif");
    let gifRecorder = null;
    let gifRecording = false;
    let gifFrameTimer = null;
    let gifFramesCaptured = 0;
    let gifWorkerBlobUrl = null;
    const GIF_MAX_FRAMES = 100;   // ~5-10 seconds at 10-20 fps
    const GIF_FRAME_DELAY = 100;  // ms between captures

    async function getGifWorkerUrl() {
      if (gifWorkerBlobUrl) return gifWorkerBlobUrl;
      try {
        const resp = await fetch("https://unpkg.com/gif.js@0.2.0/dist/gif.worker.js");
        if (!resp.ok) throw new Error("fetch failed");
        const text = await resp.text();
        const blob = new Blob([text], { type: "application/javascript" });
        gifWorkerBlobUrl = URL.createObjectURL(blob);
        return gifWorkerBlobUrl;
      } catch(e) {
        return "https://unpkg.com/gif.js@0.2.0/dist/gif.worker.js";
      }
    }

    if (recordGifBtn && getFrameCount() > 0) {
      recordGifBtn.style.display = "";
      recordGifBtn.addEventListener("click", async function() {
        if (gifRecording) {
          gifRecording = false;
          if (gifFrameTimer) { clearTimeout(gifFrameTimer); gifFrameTimer = null; }
          recordGifBtn.textContent = "Record GIF";
          recordGifBtn.style.background = "";
          recordGifBtn.style.color = "";
          if (gifRecorder) { gifRecorder.render(); }
          return;
        }
        if (typeof GIF === "undefined") {
          alert("GIF encoder not loaded. Check your network connection.");
          return;
        }
        const workerUrl = await getGifWorkerUrl();
        gifRecording = true;
        gifFramesCaptured = 0;
        recordGifBtn.textContent = "Stop GIF";
        recordGifBtn.style.background = "#cc3333";
        recordGifBtn.style.color = "#fff";

        const w = renderer.domElement.width;
        const h = renderer.domElement.height;
        gifRecorder = new GIF({
          workers: 2,
          quality: 10,
          width: w,
          height: h,
          workerScript: workerUrl
        });
        gifRecorder.on("finished", function(blob) {
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "figure3d_movie.gif";
          a.click();
          setTimeout(function() { URL.revokeObjectURL(url); }, 2000);
          gifRecorder = null;
        });

        function captureNextGifFrame() {
          if (!gifRecording || !gifRecorder) return;
          if (gifFramesCaptured >= GIF_MAX_FRAMES) {
            gifRecording = false;
            recordGifBtn.textContent = "Record GIF";
            recordGifBtn.style.background = "";
            recordGifBtn.style.color = "";
            gifRecorder.render();
            return;
          }
          gifRecorder.addFrame(renderer.domElement, { delay: GIF_FRAME_DELAY, copy: true });
          gifFramesCaptured += 1;
          gifFrameTimer = setTimeout(captureNextGifFrame, GIF_FRAME_DELAY);
        }
        captureNextGifFrame();
      });
    }

    document.getElementById("lightInfo").addEventListener("click", function() {
      const proj = camera.isPerspectiveCamera ? "Perspective" : "Orthographic";
      const ssao = composer ? "ON" : "OFF";
      const info = [
        "Projection: " + proj,
        "SSAO: " + ssao,
        "",
        "Ambient:     " + ambient.intensity.toFixed(3) + "  (#ffffff)",
        "Hemisphere:  " + hemiLight.intensity.toFixed(3) + "  (sky: #ddeeff, ground: #3b3b3b)",
        "Dir Key:     " + dirKey.intensity.toFixed(3) + "  pos: (" + dirKey.position.x + ", " + dirKey.position.y + ", " + dirKey.position.z + ")",
        "Dir Fill:    " + dirFill.intensity.toFixed(3) + "  pos: (" + dirFill.position.x + ", " + dirFill.position.y + ", " + dirFill.position.z + ")",
        "",
        "Camera:  eye=(" + camera.position.x.toFixed(2) + ", " + camera.position.y.toFixed(2) + ", " + camera.position.z.toFixed(2) + ")",
        "Target: (" + controls.target.x.toFixed(2) + ", " + controls.target.y.toFixed(2) + ", " + controls.target.z.toFixed(2) + ")",
      ];
      document.getElementById("lightDetails").textContent = info.join("\n");
      document.getElementById("lightModal").style.display = "block";
    });
    document.getElementById("closeLightModal").addEventListener("click", function() {
      document.getElementById("lightModal").style.display = "none";
    });

    // Axis labels — project lattice vectors to screen
    const cell = DATA.static && DATA.static.traces ? DATA.static.traces.find(t => t.id === "cell") : null;
    if (cell && cell.data && cell.data.positions && cell.data.positions.length >= 18) {
      const p = cell.data.positions;
      const origin = new THREE.Vector3(p[0], p[1], p[2]);
      const aEnd = new THREE.Vector3(p[3], p[4], p[5]);
      const bEnd = new THREE.Vector3(p[9], p[10], p[11]);
      const cEnd = new THREE.Vector3(p[15], p[16], p[17]);
      const aLen = origin.distanceTo(aEnd);
      const bLen = origin.distanceTo(bEnd);
      const cLen = origin.distanceTo(cEnd);
      const axDiv = document.getElementById("axisLabels");
      let axisTick = 0;
      function updateAxisLabels() {
        if (++axisTick % 3 !== 0) return;
        const w = renderer.domElement.clientWidth;
        const h = renderer.domElement.clientHeight;
        function proj(v) {
          const p = v.clone().project(camera);
          return { x: (p.x * 0.5 + 0.5) * w, y: (-p.y * 0.5 + 0.5) * h };
        }
        const labels = [
          { text: "O", pos: proj(origin), color: "#aaaaaa" },
          { text: "a (" + aLen.toFixed(2) + " A)", pos: proj(aEnd), color: "#ff6666" },
          { text: "b (" + bLen.toFixed(2) + " A)", pos: proj(bEnd), color: "#66ff66" },
          { text: "c (" + cLen.toFixed(2) + " A)", pos: proj(cEnd), color: "#6688ff" },
        ];
        axDiv.innerHTML = "";
        labels.forEach(l => {
          if (l.pos.x < -50 || l.pos.x > w + 50 || l.pos.y < -50 || l.pos.y > h + 50) return;
          const span = document.createElement("span");
          span.textContent = l.text;
          span.style.position = "absolute";
          span.style.left = (l.pos.x + 4) + "px";
          span.style.top = (l.pos.y - 14) + "px";
          span.style.color = l.color;
          axDiv.appendChild(span);
        });
      }
      const origAnimate = animate;
      const wrappedAnimate = function(t) { origAnimate(t); updateAxisLabels(); };
      animate = wrappedAnimate;
    }

    if (getFrameCount() <= 0) {
      slider.disabled = true;
      playBtn.disabled = true;
      prevBtn.disabled = true;
      nextBtn.disabled = true;
      statusEl.textContent = "No frames";
    }
    Object.values(guestMeshes).forEach(m => { if (m) m.visible = showGuests.checked; });
    Object.keys(objectsById).forEach(id => {
      if (id && id.startsWith("guestbonds_")) {
        const obj = objectsById[id];
        if (obj) obj.visible = showGuestBonds.checked;
      }
    });
    setFrame(0);
    requestAnimationFrame(animate);
  </script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Convenience CLI / script entry-point (mirror of md_pdb_threejs_viewer.py)
# ---------------------------------------------------------------------------

def make_sparse_multicomponent_movie(
    *,
    component_movie_paths: Mapping[str, str | Path],
    cif_path: str | Path,
    out_html: str | Path = "movie_sparse_multicomponent.html",
    framework_movie_path: str | Path | None = None,
    snapshot_indices: Sequence[int] | None = None,
    movie_format: str = "auto",
    framework_movie_format: str | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
    position_decimals: int = 4,
    spacing: float = 0.25,
    isovalue: float = 1.0,
    surf_step_size: int = 2,
    fps: float = 20.0,
    eye: tuple[float, float, float] = (2.0, 0.0, 1.35),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    sidecar: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build sparse-index movie with separate guest traces per component.

    Notes
    -----
    - If ``framework_movie_path`` is set, **framework mesh, unit cell, surface, maxima,
      windows, and connection traces are stored per animation frame** so the viewer can
      show a flexible framework and recomputed segmentation for each sparse snapshot.
    - If ``framework_movie_path`` is omitted, the framework comes from ``cif_path`` and
      stays static (segmentation once); only guest traces animate.
    - Segmentation dictionaries are computed for every selected frame and returned in
      the second tuple element for downstream analysis.
    """
    if not component_movie_paths:
        raise ValueError("component_movie_paths must contain at least one component.")

    comp_snaps: dict[str, list[dict[str, Any]]] = {}
    n_frames_candidates: list[int] = []
    for label, path in component_movie_paths.items():
        snaps = read_movie_snapshots(
            path,
            movie_format=movie_format,
            max_frames=None,
            frame_stride=frame_stride,
            position_decimals=position_decimals,
        )
        if not snaps:
            raise ValueError(f"No snapshots parsed for component {label!r}: {path}")
        comp_snaps[str(label)] = snaps
        n_frames_candidates.append(len(snaps))

    if framework_movie_path is not None:
        fw_snaps = read_movie_snapshots(
            framework_movie_path,
            movie_format=framework_movie_format or movie_format,
            max_frames=None,
            frame_stride=frame_stride,
            position_decimals=position_decimals,
        )
        if not fw_snaps:
            raise ValueError(f"No framework snapshots parsed from: {framework_movie_path}")
        n_frames_candidates.append(len(fw_snaps))
    else:
        framework_atoms = read(str(cif_path))
        fw_snaps = [_snapshot_dict(framework_atoms, path=str(cif_path), frame_index=0, source_format="cif")]

    n_frames_common = int(min(n_frames_candidates))
    selected = _coerce_snapshot_indices(snapshot_indices, n_frames=n_frames_common, max_frames=max_frames)
    if not selected:
        raise ValueError("No valid snapshot indices selected.")

    seg_series = compute_sparse_segmentation_series(
        fw_snaps,
        selected,
        spacing=spacing,
    )
    seg_by_global_idx = {int(entry["frame_index"]): entry["segmentation"] for entry in seg_series}

    fig = MovieFigure()
    fig.set_camera(eye=eye, center=center, up=up)
    fig.update_layout({"fps": fps, "n_frames": len(selected)})

    component_labels = sorted(comp_snaps.keys())
    animate_framework = framework_movie_path is not None

    for local_frame, global_frame_idx in enumerate(selected):
        framework_atoms = fw_snaps[global_frame_idx]["atoms"]
        if animate_framework:
            framework_atoms = update_with_covalent_bonds(framework_atoms)
        seg = seg_by_global_idx[int(global_frame_idx)]

        if animate_framework:
            fig.add_trace(make_framework_trace(framework_atoms), frame=local_frame)
            fig.add_trace(make_cell_trace(framework_atoms), frame=local_frame)
            fig.add_trace(
                make_surface_trace(
                    seg["dgrid"],
                    seg["regions"],
                    framework_atoms,
                    isovalue=isovalue,
                    step_size=surf_step_size,
                ),
                frame=local_frame,
            )
            maxima = make_maxima_trace(seg, framework_atoms)
            fig.add_trace(maxima["internal"], frame=local_frame)
            fig.add_trace(maxima["outside"], frame=local_frame)
            windows = make_windows_trace(seg)
            fig.add_trace(windows["internal"], frame=local_frame)
            fig.add_trace(windows["periodic"], frame=local_frame)
            conns = make_connections_trace(seg, framework_atoms)
            fig.add_trace(conns["internal_lines"], frame=local_frame)
            fig.add_trace(conns["periodic_lines"], frame=local_frame)
            fig.add_trace(conns["internal_nodes"], frame=local_frame)
            fig.add_trace(conns["periodic_nodes"], frame=local_frame)
        elif local_frame == 0:
            fig.add_trace(make_framework_trace(framework_atoms))
            fig.add_trace(make_cell_trace(framework_atoms))
            fig.add_trace(
                make_surface_trace(
                    seg["dgrid"],
                    seg["regions"],
                    framework_atoms,
                    isovalue=isovalue,
                    step_size=surf_step_size,
                )
            )
            maxima = make_maxima_trace(seg, framework_atoms)
            fig.add_trace(maxima["internal"])
            fig.add_trace(maxima["outside"])
            windows = make_windows_trace(seg)
            fig.add_trace(windows["internal"])
            fig.add_trace(windows["periodic"])
            conns = make_connections_trace(seg, framework_atoms)
            fig.add_trace(conns["internal_lines"])
            fig.add_trace(conns["periodic_lines"])
            fig.add_trace(conns["internal_nodes"])
            fig.add_trace(conns["periodic_nodes"])

        for ci, label in enumerate(component_labels):
            atoms = comp_snaps[label][global_frame_idx]["atoms"]
            if animate_framework:
                atoms = _strip_framework_like_atoms(atoms, framework_atoms)
            trace = make_guest_trace(
                atoms,
                style={
                    "color": _component_palette_rgb01(ci),
                    "shininess": 10,
                    "specular": 0x151515,
                },
                name=f"Adsorbates ({label})",
                trace_id=f"guests_{label}",
            )
            fig.add_trace(trace, frame=local_frame)
            fig.add_trace(
                make_guest_bonds_trace(
                    atoms,
                    style="guest_bonds",
                    name=f"Adsorbate bonds ({label})",
                    trace_id=f"guestbonds_{label}",
                ),
                frame=local_frame,
            )

    fig.groups = {
        "guests": {"traces": [f"guests_{lbl}" for lbl in component_labels], "label": "Adsorbates (all components)"},
        "guest_bonds": {
            "traces": [f"guestbonds_{lbl}" for lbl in component_labels],
            "label": "Adsorbate bonds",
        },
    }
    fig.meta = {
        "workflow": "sparse_multicomponent",
        "component_labels": component_labels,
        "snapshot_indices": [int(i) for i in selected],
        "n_frames": int(len(selected)),
        "n_frames_common": int(n_frames_common),
        "flexible_framework_per_frame": bool(animate_framework),
    }

    out = fig.to_html(out_html, fps=fps, sidecar=sidecar)
    payload = {
        "html_path": str(out),
        "snapshot_indices": [int(i) for i in selected],
        "segmentation_series": seg_series,
    }
    return out, payload


def make_movie_from_trajectory(
    movie_path: str | Path,
    cif_path: str | Path,
    out_html: str | Path = "movie.html",
    *,
    movie_format: str = "auto",
    max_frames: int | None = 200,
    frame_stride: int = 1,
    position_decimals: int = 4,
    spacing: float = 0.25,
    isovalue: float = 1.0,
    surf_step_size: int = 2,
    fps: float = 20.0,
    eye: tuple[float, float, float] = (2.0, 0.0, 1.35),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    sidecar: bool = False,
) -> Path:
    """Build a fixed-framework movie from any ASE-readable trajectory (or RASPA PDB) + CIF."""
    snapshots = read_movie_snapshots(
        movie_path,
        movie_format=movie_format,
        max_frames=max_frames,
        frame_stride=frame_stride,
        position_decimals=position_decimals,
    )
    framework_atoms = read(str(cif_path))
    fig = build_fixed_framework_movie(
        snapshots=snapshots,
        framework_atoms=framework_atoms,
        spacing=spacing,
        isovalue=isovalue,
        surf_step_size=surf_step_size,
        fps=fps,
        eye=eye,
        center=center,
        up=up,
        max_frames=None,
    )
    return fig.to_html(out_html, fps=fps, sidecar=sidecar)


def make_movie_from_raspa_pdb(
    pdb_path: str | Path,
    cif_path: str | Path,
    out_html: str | Path = "movie.html",
    max_frames: int | None = 200,
    frame_stride: int = 1,
    position_decimals: int = 4,
    spacing: float = 0.25,
    isovalue: float = 1.0,
    surf_step_size: int = 2,
    fps: float = 20.0,
    eye: tuple[float, float, float] = (2.0, 0.0, 1.35),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    sidecar: bool = False,
) -> Path:
    """High-level convenience function to build a fixed-framework movie from RASPA PDB + CIF."""
    return make_movie_from_trajectory(
        movie_path=pdb_path,
        cif_path=cif_path,
        out_html=out_html,
        movie_format="raspa_pdb",
        max_frames=max_frames,
        frame_stride=frame_stride,
        position_decimals=position_decimals,
        spacing=spacing,
        isovalue=isovalue,
        surf_step_size=surf_step_size,
        fps=fps,
        eye=eye,
        center=center,
        up=up,
        sidecar=sidecar,
    )


# ---------------------------------------------------------------------------
# Public compute helpers (thin wrappers around _serialize_* internals)
# ---------------------------------------------------------------------------

def compute_framework_mesh(
    atoms: Atoms,
    bonds: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Compute raw framework mesh data (positions, indices, colors, hover)."""
    return _serialize_framework_mesh(atoms, bonds=bonds)


def compute_cell_lines(atoms: Atoms) -> dict[str, Any]:
    """Compute unit-cell wireframe line-segment data."""
    return _serialize_cell_lines(atoms)


def compute_surface_mesh(
    atoms: Atoms,
    dgrid: np.ndarray,
    regions: np.ndarray,
    isovalue: float = 1.0,
    step_size: int = 1,
) -> dict[str, Any]:
    """Compute marching-cubes isosurface mesh from distance grid and region labels."""
    seg = {"dgrid": dgrid, "regions": regions}
    return _serialize_segmentation_surface_mesh(
        atoms=atoms, seg=seg, isovalue=isovalue, step_size=max(step_size, 1),
    )


def compute_maxima_data(
    seg: dict[str, Any],
    atoms: Atoms,
) -> dict[str, Any]:
    """Compute maxima point-cloud data (positions, sizes, colors, hover)."""
    return _serialize_maxima_trace(seg, atoms)


def compute_windows_data(seg: dict[str, Any]) -> dict[str, Any]:
    """Compute window-point data from connection list."""
    return _serialize_windows_trace(seg)


def compute_connection_data(
    seg: dict[str, Any],
    atoms: Atoms,
) -> dict[str, Any]:
    """Compute connection line-segment + node-point data."""
    return _serialize_connection_traces(seg, atoms)


# ---------------------------------------------------------------------------
# One-shot static scene builder
# ---------------------------------------------------------------------------

def build_static_scene(
    atoms: Atoms,
    dgrid: np.ndarray | None = None,
    regions: np.ndarray | None = None,
    seg: dict[str, Any] | None = None,
    *,
    isovalue: float = 1.0,
    hide_periodic: bool = True,
    spacing: float = 0.25,
    bonds: str | list | None = "auto",
    show_aromatics: bool = True,
    eye: tuple[float, float, float] = (2.0, 0.0, 1.35),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> MovieFigure:
    """Build a static Three.js scene from framework atoms + optional segmentation.

    This is the simplest entry point for the common tutorial pattern:
    framework + cell + surface + maxima + connections + windows.

    Parameters
    ----------
    atoms
        ASE Atoms with unit cell (framework structure).
    dgrid
        Precomputed distance grid.  If *None* and *regions* / *seg* is not given
        the segmentation traces are skipped.
    regions
        Segmentation region labels (from ``do_periodic_segmentation_rust``).
    seg
        Full segmentation dict as returned by ``compute_segmentation_for_snapshot``
        or ``do_periodic_segmentation_rust``.  When provided the maxima, connections
        and windows traces are added too.
    isovalue
        Marching-cubes level for the surface.
    hide_periodic
        If True, periodic maxima / connections / windows start hidden.
    spacing
        Ignored unless *dgrid* must be computed on-the-fly (not yet implemented).
    bonds
        Passed to ``make_framework_trace`` (*"auto"* uses ASE neighbourlist).
    show_aromatics
        If True, detect and render aromatic ring outlines + fill discs.
    eye, center, up
        Initial camera position.

    Returns
    -------
    MovieFigure
        Ready to export via ``.to_html()`` or ``.show()``.
    """
    fig = MovieFigure()
    fig.set_camera(eye=eye, center=center, up=up)

    # --- Framework + cell (always) ---
    fig.add_framework(atoms, bonds=bonds)
    fig.add_cell(atoms)

    # --- Aromatic rings (grouped with framework) ---
    if show_aromatics:
        fig.add_aromatic_rings(atoms)

    # --- Surface (if we have dgrid + regions) ---
    if dgrid is not None and regions is not None:
        fig.add_surface(dgrid, regions, atoms, isovalue=isovalue)

    # --- Pore graph (if full segmentation available) ---
    if seg is not None:
        fig.add_maxima(seg, atoms, hide_periodic=hide_periodic)
        fig.add_connections(seg, atoms, hide_periodic=hide_periodic)
        fig.add_windows(seg, hide_periodic=hide_periodic, atoms=atoms, spacing=spacing)

    return fig


# ---------------------------------------------------------------------------
# User-friendly alias
# ---------------------------------------------------------------------------

Figure3D = MovieFigure



# ---------------------------------------------------------------------------
# Percolation cluster trace builder (added for mgr_cli_py Three.js backend)
# ---------------------------------------------------------------------------


def _parse_color_to_rgb(color, fallback=(99, 110, 250)):
    """Parse a color string or tuple to RGB tuple."""
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    if isinstance(color, str):
        color = color.strip()
        if color.startswith("#") and len(color) == 7:
            return (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
            )
        if color.startswith("rgb("):
            parts = color[4:-1].split(",")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
    return fallback


def _adjust_hls(rgb, light_factor=1.0, sat_factor=1.0, hue_deg=0.0):
    """Adjust RGB color via HLS space."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * light_factor))
    s = max(0.0, min(1.0, s * sat_factor))
    h = (h + hue_deg / 360.0) % 1.0
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def _serialize_percolation_cluster_mesh(
    dgrid,
    regions,
    atoms,
    component_regions,
    base_color,
    isovalue=1.6,
    step_size=1,
    taubin_smooth: bool = False,
    taubin_iterations: int = 10,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
):
    """Serialize a percolation cluster mesh for Three.js."""
    verts, faces, _normals, _vals = marching_cubes(
        dgrid, level=isovalue, step_size=max(step_size, 1)
    )
    if len(faces) == 0:
        return {
            "positions": [],
            "indices": [],
            "colors": [],
            "labels": [],
            "n_vertices": 0,
            "n_triangles": 0,
        }

    cell = atoms.get_cell().array.T
    shape = np.asarray(dgrid.shape, dtype=float)
    shape_den = np.maximum(shape - 1.0, 1.0)
    verts_cart = np.dot(cell, (verts / shape_den).T).T
    mesh = trimesh.Trimesh(vertices=verts_cart, faces=faces, process=False)

    # For face filtering, use grid-coordinate interpolation directly on marching-cubes
    # verts (matching Plotly's approach) to avoid fractional-coord round-trip errors
    # near cell boundaries that can cause patches of skipped faces.
    from scipy.interpolate import RegularGridInterpolator as RGI
    xx = np.linspace(0, regions.shape[0] - 1, regions.shape[0])
    yy = np.linspace(0, regions.shape[1] - 1, regions.shape[1])
    zz = np.linspace(0, regions.shape[2] - 1, regions.shape[2])
    rl = RGI((xx, yy, zz), regions, method='nearest', bounds_error=False, fill_value=0)

    comp_set = set(int(x) for x in component_regions)
    vert_labels = rl(verts).astype(int)
    face_mask = np.all(np.isin(vert_labels[faces], list(comp_set)), axis=1)
    faces_to_keep = np.where(face_mask)[0]

    if faces_to_keep.size == 0:
        return {
            "positions": [],
            "indices": [],
            "colors": [],
            "labels": [],
            "n_vertices": 0,
            "n_triangles": 0,
        }

    submesh = mesh.submesh([faces_to_keep], append=True)
    if isinstance(submesh, (list, tuple)):
        if not submesh:
            return {
                "positions": [],
                "indices": [],
                "colors": [],
                "labels": [],
                "n_vertices": 0,
                "n_triangles": 0,
            }
        submesh = trimesh.util.concatenate(submesh)

    if not hasattr(submesh, "vertices") or not hasattr(submesh, "faces"):
        return {
            "positions": [],
            "indices": [],
            "colors": [],
            "labels": [],
            "n_vertices": 0,
            "n_triangles": 0,
        }

    submesh = _apply_taubin_smoothing(
        submesh,
        enabled=taubin_smooth,
        iterations=taubin_iterations,
        lamb=taubin_lambda,
        nu=taubin_nu,
    )

    cell_inv = np.linalg.inv(cell)
    v_sub = np.asarray(submesh.vertices, dtype=float)
    idx_like = np.dot(v_sub, cell_inv.T) * shape_den
    gx = np.arange(regions.shape[0], dtype=float)
    gy = np.arange(regions.shape[1], dtype=float)
    gz = np.arange(regions.shape[2], dtype=float)
    lbl_paint = RGI(
        (gx, gy, gz),
        np.asarray(regions, dtype=float),
        method="nearest",
        bounds_error=False,
        fill_value=0,
    )
    sub_labels = lbl_paint(idx_like).astype(int)

    comp_sorted = sorted(comp_set)
    n_regions = len(comp_sorted)
    if n_regions == 1:
        shades = {comp_sorted[0]: base_color}
    else:
        light_low, light_high = 0.92, 1.06
        sat_low, sat_high = 0.95, 1.05
        hue_span = 10.0
        light_factors = np.linspace(light_low, light_high, n_regions)
        sat_factors = np.linspace(sat_low, sat_high, n_regions)
        hue_offsets = np.linspace(-hue_span / 2.0, hue_span / 2.0, n_regions)
        shades = {}
        for idx, label in enumerate(comp_sorted):
            shades[label] = _adjust_hls(
                base_color,
                light_factor=light_factors[idx],
                sat_factor=sat_factors[idx],
                hue_deg=hue_offsets[idx],
            )

    colors = np.asarray(
        [shades.get(int(r), base_color) for r in sub_labels], dtype=float
    ) / 255.0

    return {
        "positions": np.asarray(submesh.vertices, dtype=float).reshape(-1).tolist(),
        "indices": np.asarray(submesh.faces, dtype=int).reshape(-1).tolist(),
        "colors": colors.reshape(-1).tolist(),
        "labels": sub_labels.tolist(),
        "n_vertices": int(len(submesh.vertices)),
        "n_triangles": int(len(submesh.faces)),
    }


def make_percolation_clusters_trace(
    percolation_history,
    distance_grid,
    regions,
    atoms,
    probe_radius=None,
    isovalue=1.6,
    cscale=None,
    style="surface",
    trace_id="percolation_clusters",
    cluster_opacity: float = 1.0,
    step_size: int = 1,
    taubin_smooth: bool = False,
    taubin_iterations: int = 10,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
):
    """Build Three.js mesh traces for percolated clusters at a given probe radius."""
    if probe_radius is None:
        entry = percolation_history[-1]
    else:
        entry = next((e for e in percolation_history if e[0] >= probe_radius), None)
        if entry is None:
            raise ValueError(f"No percolation data found for probe radius {probe_radius}")

    threshold, components = entry

    unique_components = {}
    for axis, comp_list in components.items():
        for comp in comp_list:
            comp_key = tuple(sorted(comp))
            if comp_key in unique_components:
                unique_components[comp_key].append(axis)
            else:
                unique_components[comp_key] = [axis]

    try:
        import plotly.express as px
        default_palette = px.colors.qualitative.Plotly
    except Exception:
        default_palette = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3",
            "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        ]

    user_cscale = cscale
    traces = []

    for i, (comp, axes) in enumerate(unique_components.items(), 1):
        dir_labels = ['x' if 0 in axes else '', 'y' if 1 in axes else '', 'z' if 2 in axes else '']
        dir_str = ''.join(dir_labels)

        palette = None
        if user_cscale is not None:
            if isinstance(user_cscale, (list, tuple)) and len(user_cscale) > 0:
                palette = list(user_cscale)
            elif isinstance(user_cscale, str):
                try:
                    import plotly.express as px
                    palette = list(getattr(px.colors.qualitative, user_cscale))
                except Exception:
                    try:
                        palette = list(getattr(px.colors.sequential, user_cscale))
                    except Exception:
                        palette = None
        if not palette:
            palette = default_palette

        base_color_raw = palette[(i - 1) % len(palette)]
        base_color = _parse_color_to_rgb(base_color_raw, (99, 110, 250))

        payload = _serialize_percolation_cluster_mesh(
            dgrid=distance_grid,
            regions=regions,
            atoms=atoms,
            component_regions=list(comp),
            base_color=base_color,
            isovalue=isovalue,
            step_size=step_size,
            taubin_smooth=taubin_smooth,
            taubin_iterations=taubin_iterations,
            taubin_lambda=taubin_lambda,
            taubin_nu=taubin_nu,
        )

        if payload["n_vertices"] == 0:
            continue

        trace = _make_trace(
            trace_id=f"{trace_id}_{i}_{dir_str}",
            trace_type="mesh",
            name=f"Component {i}_{dir_str}",
            data=payload,
            style={"opacity": float(cluster_opacity), "side": "DoubleSide"} if style == "surface" else style,
            hover=None,
            visible=True,
        )
        traces.append(trace)

    return traces


def make_percolation_windows_trace(
    analysis: dict[str, Any],
    atoms,
    probe_radius: float | None = None,
    pld_tolerance: float = 0.25,
) -> dict[str, dict[str, Any]]:
    """Build Three.js traces for windows along percolating clusters.

    Returns two traces:
    - ``"windows"`` — yellow circles for all windows in percolating components
    - ``"pld_critical"`` — red diamonds for windows within ±*pld_tolerance* of PLD

    Parameters
    ----------
    analysis:
        Percolation analysis dict with keys: percolation_history, connections,
        dgrid, regions, pld.
    atoms:
        ASE Atoms for cell information.
    probe_radius:
        Selected probe radius. If None, uses the last entry in percolation_history.
    pld_tolerance:
        Radius tolerance for PLD-critical windows (default ±0.25 Å).
    """
    import numpy as np

    percolation_history = analysis.get("percolation_history", [])
    connections = analysis.get("connections", [])
    dgrid = np.asarray(analysis.get("dgrid", []))
    regions = np.asarray(analysis.get("regions", []))
    pld = float(analysis.get("pld", 0.0))
    pld_probe_radius = pld / 2.0  # PLD is a diameter; compare against radius

    if not percolation_history:
        return {"windows": {}, "pld_critical": {}}

    # Find the entry for the selected probe radius
    if probe_radius is None:
        entry = percolation_history[-1]
    else:
        entry = next((e for e in percolation_history if e[0] >= probe_radius), None)
        if entry is None:
            return {"windows": {}, "pld_critical": {}}

    _threshold, components = entry

    # Collect all unique percolating region sets
    unique_components: dict[tuple, list] = {}
    for axis, comp_list in components.items():
        for comp in comp_list:
            key = tuple(sorted(comp))
            if key not in unique_components:
                unique_components[key] = []
            unique_components[key].append(axis)

    # Collect all region labels in percolating components
    all_comp_regions: set[int] = set()
    for comp in unique_components:
        all_comp_regions.update(comp)

    if not all_comp_regions:
        return {"windows": {}, "pld_critical": {}}

    # Map maxima label → region label using the regions array
    maxima_info = np.asarray(analysis.get("maxima_info", []), dtype=int)
    maxima_to_region: dict[int, int] = {}
    if maxima_info.ndim == 2 and maxima_info.shape[1] >= 7:
        for row in maxima_info:
            label = int(row[0])
            # maxima_info columns: (label, z, y, x, offset_z, offset_y, offset_x)
            gz, gy, gx = int(row[1]), int(row[2]), int(row[3])
            oz, oy, ox = int(row[4]), int(row[5]), int(row[6])
            # Global grid position in the padded grid, then wrapped to primary cell
            gz_w = (gz + oz * regions.shape[0]) % regions.shape[0]
            gy_w = (gy + oy * regions.shape[1]) % regions.shape[1]
            gx_w = (gx + ox * regions.shape[2]) % regions.shape[2]
            if 0 <= gz_w < regions.shape[0] and 0 <= gy_w < regions.shape[1] and 0 <= gx_w < regions.shape[2]:
                rl = int(regions[gz_w, gy_w, gx_w])
                if rl > 0:
                    maxima_to_region[label] = rl

    # Collect windows from connections whose maxima are both in percolating regions
    all_window_positions: list[list[float]] = []
    all_window_radii: list[float] = []
    all_window_hover: list[str] = []
    pld_positions: list[list[float]] = []
    pld_radii: list[float] = []
    pld_hover: list[str] = []

    for c in connections:
        if not c[1] or len(c) < 5:
            continue
        n1, n2 = c[0]
        r1 = maxima_to_region.get(int(n1), -1)
        r2 = maxima_to_region.get(int(n2), -1)
        if r1 not in all_comp_regions or r2 not in all_comp_regions:
            continue
        pts = c[3] if c[3] is not None else []
        wr = c[4] if c[4] is not None else []
        ctype = str(c[2]) if len(c) > 2 else "internal"
        for p, r in zip(pts, wr if len(wr) == len(pts) else [0.0] * len(pts)):
            wp = np.asarray(p, dtype=float).tolist()
            radius = float(r)
            hover = (
                f"trace: {ctype} windows<br>"
                f"connection maxima: {n1}-{n2}<br>"
                f"type: {ctype}<br>"
                f"window radius: {radius:.3f} Å"
            )
            if pld > 0 and abs(radius - pld_probe_radius) <= pld_tolerance:
                pld_positions.append(wp)
                pld_radii.append(radius)
                pld_hover.append(hover + f"<br>⚠ PLD-critical (±{pld_tolerance} Å)")
            else:
                all_window_positions.append(wp)
                all_window_radii.append(radius)
                all_window_hover.append(hover)

    # Regular windows trace (yellow circles)
    window_trace = _make_trace(
        trace_id="percolation_windows",
        trace_type="points",
        name="Percolation windows",
        data={
            "positions": np.array(all_window_positions).reshape(-1).tolist() if all_window_positions else [],
            "colors": ([1.0, 1.0, 0.0] * len(all_window_positions)) if all_window_positions else [],  # yellow, flat
            "sizes": [12.0] * len(all_window_positions),
            "n_points": len(all_window_positions),
        },
        style={"color": [1.0, 1.0, 0.0], "size": 12.0, "symbol": "circle"},
        hover=all_window_hover,
        visible=True,
    )

    # PLD-critical windows trace (red diamonds)
    pld_trace = _make_trace(
        trace_id="percolation_windows_pld",
        trace_type="points",
        name="PLD-critical windows",
        data={
            "positions": np.array(pld_positions).reshape(-1).tolist() if pld_positions else [],
            "colors": ([1.0, 0.2, 0.2] * len(pld_positions)) if pld_positions else [],  # red, flat
            "sizes": [16.0] * len(pld_positions),
            "n_points": len(pld_positions),
        },
        style={"color": [1.0, 0.2, 0.2], "size": 16.0, "symbol": "diamond"},
        hover=pld_hover,
        visible=True,
    )

    return {"windows": window_trace, "pld_critical": pld_trace}

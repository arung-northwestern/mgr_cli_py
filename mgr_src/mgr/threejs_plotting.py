"""Three.js plotting adapter for mgr_cli_py Streamlit app.

Provides drop-in replacements for plotting.show_mof() and percolation
visualization that return self-contained HTML strings for embedding via
streamlit.components.v1.html().
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from .movies import (
    MovieFigure,
    build_static_scene,
    make_percolation_clusters_trace,
    make_surface_trace,
    _make_trace,
)


def show_mof_threejs(
    results: Dict[str, Any],
    stylize: bool = True,
    options: Optional[Dict[str, Any]] = None,
    on_step=None,
) -> str:
    """Build Three.js HTML string from MGR results dict.

    Mirrors the interface of mgrcli.plotting.show_mof().

    Returns
    -------
    str
        Self-contained HTML string ready for st.components.v1.html().
    """
    mat_atoms = results["ase_atoms"]
    dgrid = results["dgrid"]
    regions = results["regions"]
    regions_pbc = results.get("regions_pbc")
    all_maxima = results.get("all_maxima")
    connections = results.get("connections", [])

    defaults = {
        "plot_enable": True,
        "plot_cell": True,
        "plot_framework": True,
        "plot_framework_style": "mesh",
        "plot_framework_show_rings": True,
        "plot_framework_mesh_scale": 0.65,
        "plot_framework_bond_radius": 0.15,
        "plot_maxima": True,
        "plot_maxima_show_spheres": False,
        "plot_connections": True,
        "plot_windows": True,
        "plot_surface": True,
        "plot_surface_pbc": True,
        "plot_surface_isovalue": 0.75,
        "plot_surface_opacity": 1.0,
        "plot_surface_pbc_opacity": 1.0,
        "plot_surface_types_opacity": 1.0,
        "plot_surface_taubin_smooth": False,
        "plot_surface_taubin_iterations": 10,
        "plot_surface_taubin_lambda": 0.5,
        "plot_surface_taubin_nu": -0.53,
        "plot_projection": "perspective",
        "plot_layout": "dark",
    }
    if options:
        defaults.update({k: v for k, v in options.items() if v is not None})

    if not defaults.get("plot_enable", True):
        for key in ("plot_cell", "plot_framework", "plot_maxima",
                    "plot_connections", "plot_windows", "plot_surface", "plot_surface_pbc"):
            defaults[key] = False

    projection = str(defaults.get("plot_projection", "perspective")).lower()
    orthographic = projection == "orthographic"
    if orthographic:
        stylize = False

    framework_style = defaults.get("plot_framework_style", "mesh")

    # Progress bookkeeping
    _step_labels = []
    if defaults.get("plot_framework", True):
        _step_labels.append(f"Framework ({framework_style})")
    if defaults.get("plot_cell", True):
        _step_labels.append("Unit cell")
    if defaults.get("plot_surface", True):
        _step_labels.append("Regions surface")
    if defaults.get("plot_surface_pbc", True) and regions_pbc is not None:
        _step_labels.append("PBC surface")
    if results.get("pocket_types_detected", False) and "regions_cluster" in results:
        _step_labels.append("Pocket types")
    if defaults.get("plot_connections", True) and connections:
        _step_labels.append("Channel connections")
    if defaults.get("plot_windows", True) and connections:
        _step_labels.append("Windows")
    if defaults.get("plot_maxima", True) and all_maxima is not None and len(all_maxima):
        _step_labels.append("Local maxima")
    _total_steps = len(_step_labels)
    _step_idx = 0

    def _emit(label: str):
        nonlocal _step_idx
        _step_idx += 1
        if callable(on_step):
            try:
                on_step(_step_idx, _total_steps, label)
            except Exception:
                pass

    fig = MovieFigure()

    # Framework
    if defaults.get("plot_framework", True):
        _emit(_step_labels[_step_idx])
        bonds = "auto"
        representation = "ball_and_stick"
        if framework_style == "scatter":
            representation = "stick"
        kwargs = {}
        if framework_style == "mesh":
            kwargs["kwargs_spheres"] = {
                "scale": float(defaults.get("plot_framework_mesh_scale", 0.65)),
            }
            kwargs["kwargs_cylinders"] = {
                "radius": float(defaults.get("plot_framework_bond_radius", 0.15)),
            }
        fig.add_framework(
            mat_atoms,
            bonds=bonds,
            representation=representation,
            **kwargs,
        )

    # Cell
    if defaults.get("plot_cell", True):
        _emit(_step_labels[_step_idx])
        fig.add_cell(mat_atoms)

    # Surface (non-PBC) — trace_id must be "surface" for hardcoded showSurface checkbox
    if defaults.get("plot_surface", True):
        _emit(_step_labels[_step_idx])
        surface_trace = make_surface_trace(
            dgrid=dgrid,
            regions=regions,
            atoms=mat_atoms,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            step_size=1,
            name="Regions (raw)",
            trace_id="surface",
            style="surface",
            hover_prefix="Region:",
            taubin_smooth=bool(defaults.get("plot_surface_taubin_smooth", False)),
            taubin_iterations=int(defaults.get("plot_surface_taubin_iterations", 10)),
            taubin_lambda=float(defaults.get("plot_surface_taubin_lambda", 0.5)),
            taubin_nu=float(defaults.get("plot_surface_taubin_nu", -0.53)),
        )
        fig.add_trace(surface_trace)
        fig.update_traces(
            {"opacity": float(defaults.get("plot_surface_opacity", 1.0))},
            selector={"id": "surface"},
        )

    # Surface (PBC) — unique trace_id
    if defaults.get("plot_surface_pbc", True) and regions_pbc is not None:
        _emit(_step_labels[_step_idx])
        pbc_trace = make_surface_trace(
            dgrid=dgrid,
            regions=regions_pbc,
            atoms=mat_atoms,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            step_size=1,
            name="Regions (PBC merged)",
            trace_id="surface_regions_pbc",
            style="surface",
            hover_prefix="PBC:",
            taubin_smooth=bool(defaults.get("plot_surface_taubin_smooth", False)),
            taubin_iterations=int(defaults.get("plot_surface_taubin_iterations", 10)),
            taubin_lambda=float(defaults.get("plot_surface_taubin_lambda", 0.5)),
            taubin_nu=float(defaults.get("plot_surface_taubin_nu", -0.53)),
        )
        fig.add_trace(pbc_trace)
        fig.update_traces(
            {"opacity": float(defaults.get("plot_surface_pbc_opacity", 1.0))},
            selector={"id": "surface_regions_pbc"},
        )

    # Pocket types surface — unique trace_id
    if results.get("pocket_types_detected", False) and "regions_cluster" in results:
        _emit(_step_labels[_step_idx])
        regions_cluster = results["regions_cluster"]
        # Use regions_cluster directly (no -1 shift); label 0 = background,
        # label 1+ = pocket types. Backfill handles label-0 vertices.
        cluster_trace = make_surface_trace(
            dgrid=dgrid,
            regions=regions_cluster,
            atoms=mat_atoms,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            step_size=1,
            name="Pocket types (cluster)",
            trace_id="surface_regions_cluster",
            style="surface",
            hover_prefix="Pocket type:",
            taubin_smooth=bool(defaults.get("plot_surface_taubin_smooth", False)),
            taubin_iterations=int(defaults.get("plot_surface_taubin_iterations", 10)),
            taubin_lambda=float(defaults.get("plot_surface_taubin_lambda", 0.5)),
            taubin_nu=float(defaults.get("plot_surface_taubin_nu", -0.53)),
        )
        fig.add_trace(cluster_trace)
        fig.update_traces(
            {"opacity": float(defaults.get("plot_surface_types_opacity", 1.0))},
            selector={"id": "surface_regions_cluster"},
        )

    # Connections / Pore graph
    if defaults.get("plot_connections", True) and connections:
        _emit(_step_labels[_step_idx])
        # Build minimal seg dict for connections trace
        seg = {
            "dgrid": dgrid,
            "regions": regions,
            "all_maxima": all_maxima if all_maxima is not None else np.zeros((0, 3)),
            "connections": connections,
        }
        fig.add_connections(seg, mat_atoms, hide_periodic=True)

    # Windows
    if defaults.get("plot_windows", True) and connections:
        _emit(_step_labels[_step_idx])
        seg = {
            "dgrid": dgrid,
            "regions": regions,
            "all_maxima": all_maxima if all_maxima is not None else np.zeros((0, 3)),
            "connections": connections,
        }
        fig.add_windows(seg, hide_periodic=True)

    # Maxima
    if defaults.get("plot_maxima", True) and all_maxima is not None and len(all_maxima):
        _emit(_step_labels[_step_idx])
        seg = {
            "dgrid": dgrid,
            "regions": regions,
            "all_maxima": all_maxima,
            "connections": connections,
        }
        fig.add_maxima(seg, mat_atoms, hide_periodic=True)

    trace_ids = {t.get("id") for t in fig.data}

    # Keep framework/cell/raw-surface on dedicated HUD checkboxes only.
    # Avoid duplicating them in dynamic group legend.
    if (
        defaults.get("plot_surface_pbc", True)
        and regions_pbc is not None
        and "surface_regions_pbc" in trace_ids
    ):
        fig.groups["surface_regions_pbc"] = {
            "traces": ["surface_regions_pbc"],
            "label": "Regions (PBC merged)",
        }
    if (
        results.get("pocket_types_detected", False)
        and "regions_cluster" in results
        and defaults.get("plot_surface", True)
        and "surface_regions_cluster" in trace_ids
    ):
        fig.groups["surface_regions_cluster"] = {
            "traces": ["surface_regions_cluster"],
            "label": "Pocket types (cluster)",
        }
    # Keep a dedicated maxima legend entry so users can toggle the local-maxima
    # scatter trace directly from the trace legend.
    if defaults.get("plot_maxima", True) and "maxima_internal" in trace_ids:
        fig.groups["maxima_internal"] = {
            "traces": ["maxima_internal"],
            "label": "Local maxima",
        }

    # Camera
    fig.set_camera(
        eye=(2.0, 0.0, 1.35),
        center=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        projection="orthographic" if orthographic else "perspective",
    )

    # Appearance / layout
    layout = defaults.get("plot_layout", "dark")
    if layout == "white":
        fig.set_appearance(
            background="#ffffff",
            scene_bg=0xffffff,
        )
    else:
        fig.set_appearance(
            background="radial-gradient(circle at center, #7a7a7a 0%, #111111 100%)",
            scene_bg=0x111111,
        )

    # Stylize: tweak material properties
    if stylize:
        fig.update_traces(
            {"roughness": 0.05, "metalness": 0.3},
            selector={"type": "mesh"},
        )

    # Return HTML string (not file)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    fig.to_html(tmp.name)
    with open(tmp.name, "r") as f:
        html_str = f.read()
    import os
    os.unlink(tmp.name)
    return html_str


def show_percolation_threejs(
    analysis: Dict[str, Any],
    atoms,
    probe_radius: float,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Build Three.js HTML string for percolation cluster visualization.

    Parameters
    ----------
    analysis : dict
        Percolation analysis dict with keys:
        dgrid, regions, maxima, connections, percolation_history, pld_locations
    atoms : ase.Atoms
    probe_radius : float
        Selected probe radius for cluster visualization.
    options : dict, optional
        Currently supports ``layout`` ("dark" or "white").

    Returns
    -------
    str
        Self-contained HTML string.
    """
    fig = MovieFigure()
    opts = options or {}

    # Unit cell
    fig.add_cell(atoms)

    # Minimal seg dict for connections + maxima
    maxima = analysis.get("maxima")
    connections = analysis.get("connections", [])
    seg = {
        "dgrid": analysis["dgrid"],
        "regions": analysis["regions"],
        "all_maxima": maxima if maxima is not None else np.zeros((0, 3)),
        "connections": connections,
    }

    # Pore graph and maxima (wrap in try/except in case data is inconsistent)
    if connections:
        try:
            fig.add_connections(seg, atoms, hide_periodic=True)
        except Exception:
            pass

    if maxima is not None and len(maxima):
        try:
            fig.add_maxima(seg, atoms, hide_periodic=True)
        except Exception:
            pass

    # PLD locations
    pld_locs = analysis.get("pld_locations")
    if pld_locs is not None and len(pld_locs):
        # Convert grid coords to cartesian
        cell = atoms.get_cell().array.T
        shape = np.asarray(analysis["dgrid"].shape, dtype=float)
        frac = pld_locs / shape
        pos = np.dot(cell, frac.T).T

        pld_trace = _make_trace(
            trace_id="pld_locations",
            trace_type="points",
            name="PLD locations",
            data={
                "positions": pos.reshape(-1).tolist(),
                "colors": [[0.0, 1.0, 0.0]] * len(pos),  # limegreen
                "sizes": [5.0] * len(pos),
                "n_points": len(pos),
            },
            style="pld_locations",
            hover=[f"PLD location<br>grid: [{int(g[0])}, {int(g[1])}, {int(g[2])}]" for g in pld_locs],
            visible=True,
        )
        fig.add_trace(pld_trace)

    # Percolation clusters
    percolation_history = analysis.get("percolation_history", [])
    if percolation_history:
        cluster_traces = make_percolation_clusters_trace(
            percolation_history,
            analysis["dgrid"],
            analysis["regions"],
            atoms,
            probe_radius=probe_radius,
            isovalue=analysis.get("isovalue", 1.6),
            cluster_opacity=float(opts.get("plot_surface_opacity", 1.0)),
            # Keep percolation clusters unsmoothed to avoid patchy/miscolored domains.
            taubin_smooth=False,
        )
        fig.add_traces(cluster_traces)
        # Set up legend groups — one per component, labeled by direction
        for trace in cluster_traces:
            tid = trace["id"]
            fig.groups[tid] = {"traces": [tid], "label": trace["name"]}

    # Camera
    fig.set_camera(
        eye=(2.0, 0.0, 1.35),
        center=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        projection="perspective",
    )

    # Appearance
    layout = opts.get("layout", "dark")
    if layout == "white":
        fig.set_appearance(
            background="#ffffff",
            scene_bg=0xffffff,
        )
    else:
        fig.set_appearance(
            background="radial-gradient(circle at center, #7a7a7a 0%, #111111 100%)",
            scene_bg=0x111111,
        )

    # Export to HTML string
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    fig.to_html(tmp.name)
    with open(tmp.name, "r") as f:
        html_str = f.read()
    import os
    os.unlink(tmp.name)
    return html_str

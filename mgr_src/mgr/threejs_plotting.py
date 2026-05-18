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
from .visualize import resolve_plot_surface_taubin_params


def _dgrid_spacing(atoms, dgrid: np.ndarray) -> float:
    """Estimate grid spacing from cell vectors and dgrid shape."""
    cell = atoms.get_cell().array.T
    lengths = np.linalg.norm(cell, axis=0)
    shape = np.asarray(dgrid.shape, dtype=float)
    per_axis = lengths / np.maximum(shape, 1.0)
    return float(np.mean(per_axis))


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
    maxima_pbc = results.get("maxima_pbc")
    maxima_info = results.get("maxima_info")
    maxima_pbc_dict = results.get("maxima_pbc_dict")
    connections_raw = results.get("connections", [])
    unique_regions = np.asarray(results.get("unique_regions", []), dtype=int)
    if unique_regions.size:
        connections = [c for c in connections_raw if np.any(np.isin(c[0], unique_regions))]
    else:
        connections = list(connections_raw)

    # Match Plotly show_mof: only plot the first regions.max() maxima (excludes padding-only sites).
    regions_arr = np.asarray(regions)
    n_cap = int(regions_arr.max()) if regions_arr.size else 0
    am_full = (
        np.asarray(all_maxima, dtype=float)
        if all_maxima is not None
        else np.zeros((0, 3), dtype=float)
    )
    all_maxima_plot = am_full[:n_cap] if n_cap > 0 else am_full[:0]
    m_plot = len(all_maxima_plot)
    mi_src = maxima_info
    if mi_src is not None:
        mi_arr = np.asarray(mi_src, dtype=int)
        if mi_arr.ndim == 2 and mi_arr.shape[0] >= m_plot and m_plot > 0:
            maxima_info_plot = mi_arr[:m_plot]
        else:
            maxima_info_plot = np.zeros((0, 7), dtype=int)
    else:
        maxima_info_plot = np.zeros((0, 7), dtype=int)

    defaults = {
        "plot_enable": True,
        "plot_cell": True,
        "plot_framework": True,
        "plot_framework_style": "mesh",
        "plot_framework_show_rings": True,
        "plot_framework_mesh_scale": 0.65,
        "plot_framework_bond_radius": 0.15,
        "plot_framework_bond_linewidth": 8.0,
        "plot_framework_mesh_quality": "medium",
        "plot_maxima": True,
        "plot_maxima_show_spheres": False,
        "plot_connections": True,
        "plot_windows": True,
        "plot_surface": True,
        "plot_surface_pbc": True,
        "plot_surface_isovalue": 1.0,
        "plot_surface_opacity": 1.0,
        "plot_surface_pbc_opacity": 1.0,
        "plot_surface_types_opacity": 1.0,
        "plot_surface_taubin_smooth": True,
        "plot_surface_taubin_intensity": 0.05,
        "plot_surface_screen_door": False,
        "plot_projection": "perspective",
        "plot_layout": "dark",
    }
    if options:
        defaults.update({k: v for k, v in options.items() if v is not None})

    resolve_plot_surface_taubin_params(defaults, options)

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
    if defaults.get("plot_framework", True) and defaults.get("plot_framework_show_rings", True):
        _step_labels.append("Aromatic rings")
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
    fig.update_layout(
        {
            "surface_screen_door": bool(defaults.get("plot_surface_screen_door", False)),
        }
    )

    # Ultra-low quality: coarse surfaces (framework style from UI checkboxes)
    mesh_quality = str(defaults.get("plot_framework_mesh_quality", "medium")).lower()
    _surf_step = 2 if mesh_quality == "ultra-low" else 1  # surface step_size

    # Framework
    if defaults.get("plot_framework", True):
        _emit(_step_labels[_step_idx])
        if framework_style == "scatter":
            # True scatter: points + line segments (fast, no mesh geometry)
            from .movies import make_framework_scatter_trace
            scatter_traces = make_framework_scatter_trace(
                mat_atoms,
                bonds="auto",
                style="framework",
                atom_scale=float(defaults.get("plot_framework_mesh_scale", 0.65)),
                bond_linewidth=float(defaults.get("plot_framework_bond_linewidth", 8.0)),
            )
            fig.add_trace(scatter_traces["atoms"])
            fig.add_trace(scatter_traces["bonds"])
        elif framework_style == "instanced":
            # InstancedMesh: compact atom data + cylinder mesh for bonds
            from .movies import make_framework_instanced_trace
            instanced_traces = make_framework_instanced_trace(
                mat_atoms,
                bonds="auto",
                style="framework",
                atom_scale=float(defaults.get("plot_framework_mesh_scale", 0.65)),
            )
            fig.add_trace(instanced_traces["atoms"])
            fig.add_trace(instanced_traces["bonds"])
            # Stick variant: smaller spheres, hidden by default
            stick_traces = make_framework_instanced_trace(
                mat_atoms, bonds="auto", trace_id="framework_stick",
                name="Framework (stick)", atom_scale=float(defaults.get("plot_framework_mesh_scale", 0.65)) * 0.3,
            )
            fig.add_trace(stick_traces["atoms"])
            fig.add_trace(stick_traces["bonds"])
            fig.update_traces({"visible": False}, selector={"id": "framework_stick_atoms"})
            fig.update_traces({"visible": False}, selector={"id": "framework_stick_bonds"})
            # Spacefill variant: vdw radii, no bonds, hidden by default
            from ase.data import vdw_radii, colors as ase_colors
            sf_an = mat_atoms.get_atomic_numbers()
            sf_radii = vdw_radii[sf_an]
            sf_colors = np.array([ase_colors.cpk_colors[an] for an in sf_an], dtype=float)
            sf_trace = _make_trace(
                trace_id="framework_spacefill",
                trace_type="instanced_spheres",
                name="Framework (spacefill)",
                data={
                    "positions": mat_atoms.get_positions().reshape(-1).tolist(),
                    "colors": sf_colors.reshape(-1).tolist(),
                    "radii": sf_radii.astype(float).tolist(),
                    "n_atoms": int(len(mat_atoms)),
                },
                style={"visible": False},
                hover=None,
                visible=False,
            )
            fig.add_trace(sf_trace)
        else:
            # Mesh path — apply quality preset for faster rendering
            from .visualize import MESH_QUALITY_PRESETS
            mesh_quality = str(defaults.get("plot_framework_mesh_quality", "medium")).lower()
            quality = MESH_QUALITY_PRESETS.get(mesh_quality, MESH_QUALITY_PRESETS["medium"])
            kwargs: dict[str, Any] = {
                "sphere_sections": quality["sphere_sections"],
                "cylinder_sections": quality["cylinder_sections"],
                "generate_all_reps": True,  # always precompute stick + spacefill for toggle
            }
            if framework_style == "mesh":
                kwargs["kwargs_spheres"] = {
                    "scale": float(defaults.get("plot_framework_mesh_scale", 0.65)),
                }
                kwargs["kwargs_cylinders"] = {
                    "radius": float(defaults.get("plot_framework_bond_radius", 0.15)),
                }
            fig.add_framework(
                mat_atoms,
                bonds="auto",
                representation="ball_and_stick" if framework_style == "mesh" else "stick",
                **kwargs,
            )
        if defaults.get("plot_framework_show_rings", True):
            _emit(_step_labels[_step_idx])
            fig.add_aromatic_rings(mat_atoms)

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
            isovalue=float(defaults.get("plot_surface_isovalue", 1.0)),
            step_size=_surf_step,
            name="Regions (raw)",
            trace_id="surface",
            style="surface",
            hover_prefix="Region:",
            taubin_smooth=bool(defaults.get("plot_surface_taubin_smooth", True)),
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
            isovalue=float(defaults.get("plot_surface_isovalue", 1.0)),
            step_size=_surf_step,
            name="Regions (PBC merged)",
            trace_id="surface_regions_pbc",
            style="surface",
            hover_prefix="PBC:",
            taubin_smooth=bool(defaults.get("plot_surface_taubin_smooth", True)),
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
            isovalue=float(defaults.get("plot_surface_isovalue", 1.0)),
            step_size=_surf_step,
            name="Pocket types (cluster)",
            trace_id="surface_regions_cluster",
            style="surface",
            hover_prefix="Pocket type:",
            taubin_smooth=bool(defaults.get("plot_surface_taubin_smooth", True)),
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
            "maxima_pbc": maxima_pbc if maxima_pbc is not None else np.zeros((0, 3), dtype=int),
            "maxima_info": maxima_info if maxima_info is not None else np.zeros((0, 7), dtype=int),
            "maxima_pbc_dict": maxima_pbc_dict if maxima_pbc_dict is not None else {},
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
            "maxima_pbc": maxima_pbc if maxima_pbc is not None else np.zeros((0, 3), dtype=int),
            "maxima_info": maxima_info if maxima_info is not None else np.zeros((0, 7), dtype=int),
            "maxima_pbc_dict": maxima_pbc_dict if maxima_pbc_dict is not None else {},
            "connections": connections,
        }
        fig.add_windows(seg, hide_periodic=True, atoms=mat_atoms, spacing=_dgrid_spacing(mat_atoms, dgrid))

    # Maxima
    if defaults.get("plot_maxima", True) and len(all_maxima_plot):
        _emit(_step_labels[_step_idx])
        seg = {
            "dgrid": dgrid,
            "regions": regions,
            "all_maxima": all_maxima_plot,
            "maxima_pbc": maxima_pbc if maxima_pbc is not None else np.zeros((0, 3), dtype=int),
            "maxima_info": maxima_info_plot,
            "maxima_pbc_dict": maxima_pbc_dict if maxima_pbc_dict is not None else {},
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
    if defaults.get("plot_maxima", True) and "maxima_pbc" in trace_ids:
        fig.groups["maxima_pbc"] = {
            "traces": ["maxima_pbc"],
            "label": "Maxima (PBC reps)",
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
            step_size=2,  # 4× fewer vertices for speed on HF Spaces
            cluster_opacity=float(opts.get("plot_surface_opacity", 1.0)),
            # Keep percolation clusters unsmoothed to avoid patchy/miscolored domains.
            taubin_smooth=False,
        )
        fig.add_traces(cluster_traces)
        # Set up legend groups — one per component, labeled by direction
        for trace in cluster_traces:
            tid = trace["id"]
            fig.groups[tid] = {"traces": [tid], "label": trace["name"]}

    # Percolation windows — yellow circles + PLD-critical diamonds
    if connections and percolation_history:
        from .movies import make_percolation_windows_trace
        window_traces = make_percolation_windows_trace(
            analysis,
            atoms,
            probe_radius=probe_radius,
            pld_tolerance=0.25,
        )
        if window_traces.get("windows", {}).get("data", {}).get("n_points", 0):
            fig.add_trace(window_traces["windows"])
        if window_traces.get("pld_critical", {}).get("data", {}).get("n_points", 0):
            fig.add_trace(window_traces["pld_critical"])

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

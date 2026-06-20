"""Textural property analysis shared by CLI batch and Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

import mgr


def _results_with_distance_grid(results: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(results)
    if "dgrid" in out and "distance_grid" not in out:
        out["distance_grid"] = out["dgrid"]
    return out


def count_unique_percolated_clusters(components: Any) -> int:
    """Count distinct region sets across all percolation axes."""
    if not isinstance(components, dict):
        return 0
    seen: set[tuple[int, ...]] = set()
    for comp_list in components.values():
        for comp in comp_list or []:
            if comp:
                seen.add(tuple(sorted(int(x) for x in comp)))
    return len(seen)


def compute_pld_analysis_full(
    segmentation_results: Dict[str, Any],
    mat_atoms,
    *,
    spacing: float,
    mask_thickness: float,
    h: float,
    threshold_abs: float,
    min_distance: float,
    extra_padding: float,
    accessibility_method: str = "winding",
    lattice_variant: str = "v2",
) -> Dict[str, Any]:
    """PLD/LCD/DFSD + winding/lattice percolation history (matches Streamlit ``_compute_pld_analysis``)."""
    from mgrcli.plotting import prepare_results_for_plot

    prepared = prepare_results_for_plot(segmentation_results, mat_atoms)
    dgrid = np.asarray(prepared.get("dgrid"))
    regions = np.asarray(prepared.get("regions"))
    maxima_info = prepared.get("maxima_info")
    if dgrid is None or regions is None or dgrid.size == 0 or regions.size == 0:
        raise ValueError("Segmentation results are missing distance grid or regions.")

    seg_kw = dict(
        spacing=spacing,
        mask_thickness=mask_thickness,
        h=h,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
        extra_padding=extra_padding,
    )
    unique_regions = np.asarray(segmentation_results.get("unique_regions", []), dtype=int)
    connections_prepared = prepared.get("connections", [])
    connections_filtered = [
        c for c in connections_prepared if np.all(np.isin(c[0], unique_regions))
    ]
    percolation_input = dict(segmentation_results)
    percolation_input["connections"] = connections_filtered

    method = str(accessibility_method).lower()
    if method == "lattice":
        lcd, pld, dfsd, _ = mgr.get_pore_dia_lattice(
            mat_atoms,
            dgrid=dgrid,
            results=segmentation_results,
            variant=lattice_variant,
            kwargs=seg_kw,
        )
        from mgr.compute import compute_percolation_history_from_results_lattice

        seg_copy = dict(segmentation_results)
        compute_percolation_history_from_results_lattice(seg_copy, variant=lattice_variant)
        percolation_history = seg_copy.get("percolation_history_lattice", [])
        pld_locations = np.array([])
        final_clusters = None
    else:
        lcd, pld, dfsd, _ = mgr.winding_percolation.get_pore_dia_mgr_winding(
            mat_atoms=mat_atoms,
            dgrid=dgrid,
            kwargs=seg_kw,
            results=segmentation_results,
        )
        percolation_history, pld_locations, final_clusters = (
            mgr.winding_percolation.compute_percolation_history_winding(
                percolation_input,
                step=0.05,
                expand_components=True,
            )
        )
        pld_locations = np.asarray(pld_locations)

    maxima_raw = np.asarray(segmentation_results.get("all_maxima", []), dtype=int)
    if maxima_raw.ndim == 1:
        maxima_raw = maxima_raw.reshape(-1, 3) if maxima_raw.size else np.empty((0, 3), dtype=int)
    if maxima_raw.ndim != 2:
        maxima_raw = np.empty((0, 3), dtype=int)

    all_maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)]
    maxima = maxima_raw[: regions.max()]
    maxima_radii = all_maxima_radii[: regions.max()]
    rag = mgr.rag_from_connections(connections_filtered, maxima)
    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, regions.shape, mat_atoms)
    rag.graph["regions"] = regions
    rag.graph["ase_atoms"] = mat_atoms

    percolation_isovalue = float(np.clip(np.min([pld / 2.0 - 0.25, 1.75]), 0.1, None))

    return {
        "lcd": float(lcd),
        "pld": float(pld),
        "dfsd": float(dfsd),
        "percolation_history": percolation_history,
        "pld_locations": pld_locations,
        "final_clusters": final_clusters,
        "rag": rag,
        "maxima": maxima,
        "maxima_radii": maxima_radii,
        "maxima_info": segmentation_results.get("maxima_info", np.zeros((0, 7), dtype=int)),
        "maxima_pbc_dict": segmentation_results.get("maxima_pbc_dict", {}),
        "dgrid": dgrid,
        "regions": regions,
        "connections": connections_filtered,
        "isovalue": percolation_isovalue,
        "percolation_source": "segmentation_results",
        "accessibility_method": method,
    }


def compute_surface_area(
    results_dict: Dict[str, Any],
    mat_atoms,
    probe_radius: Optional[float],
    *,
    sa_method: str = "mc",
    accessibility_method: str = "winding",
    window_fp: bool = True,
    lattice_variant: str = "v2",
    mc_n_samples: int = 20000,
    mc_shell_thickness: float = 0.5,
    winding_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    modified = _results_with_distance_grid(results_dict)
    method = str(sa_method).lower()
    acc = str(accessibility_method).lower()
    try:
        if method == "mc":
            if acc == "lattice":
                from mgr.compute import compute_surface_area_mc_lattice

                return compute_surface_area_mc_lattice(
                    modified,
                    mat_atoms,
                    probe_radius,
                    n_samples=mc_n_samples,
                    shell_thickness=mc_shell_thickness,
                    variant=lattice_variant,
                )
            from mgr.compute import compute_surface_area_mc_winding

            return compute_surface_area_mc_winding(
                modified,
                mat_atoms,
                probe_radius,
                n_samples=mc_n_samples,
                shell_thickness=mc_shell_thickness,
                window_fp=window_fp,
                winding_kwargs=winding_kwargs,
            )
        if acc == "lattice":
            from mgr.compute import compute_surface_area_mesh_lattice

            return compute_surface_area_mesh_lattice(
                modified, mat_atoms, probe_radius, variant=lattice_variant
            )
        from mgr.compute import compute_surface_area_mesh_winding

        return compute_surface_area_mesh_winding(
            modified,
            mat_atoms,
            probe_radius,
            window_fp=window_fp,
            winding_kwargs=winding_kwargs,
        )
    except Exception as exc:
        print(f"Warning: Surface area computation failed: {exc}")
        return None


def compute_void_volume(
    results_dict: Dict[str, Any],
    mat_atoms,
    probe_radius: Optional[float],
    *,
    accessibility_method: str = "winding",
    window_fp: bool = True,
    lattice_variant: str = "v2",
    winding_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Void volume uses voxel integration; ``vol_method`` is accepted for API symmetry (only ``voxel``)."""
    modified = _results_with_distance_grid(results_dict)
    acc = str(accessibility_method).lower()
    try:
        if acc == "lattice":
            from mgr.compute import compute_void_volume_from_results_lattice

            return compute_void_volume_from_results_lattice(
                modified, mat_atoms, probe_radius, variant=lattice_variant
            )
        from mgr.compute import compute_void_volume_from_results_winding

        return compute_void_volume_from_results_winding(
            modified,
            mat_atoms,
            probe_radius,
            window_fp=window_fp,
            winding_kwargs=winding_kwargs,
        )
    except Exception as exc:
        print(f"Warning: Void volume computation failed: {exc}")
        return None


def save_percolation_plot_html(
    analysis: Dict[str, Any],
    atoms,
    path: Path | str,
    *,
    probe_radius: float,
    backend: str = "threejs",
    theme: str = "dark",
    cluster_isovalue: Optional[float] = None,
) -> None:
    out = Path(path)
    isovalue = float(cluster_isovalue if cluster_isovalue is not None else analysis.get("isovalue", 1.6))
    analysis_copy = dict(analysis)
    analysis_copy["isovalue"] = max(isovalue, 0.05)
    backend_norm = (backend or "threejs").lower()

    if backend_norm == "threejs":
        from mgr.threejs_plotting import show_percolation_threejs

        html_str = show_percolation_threejs(
            analysis_copy,
            atoms,
            probe_radius=probe_radius,
            options=dict(layout=theme, plot_surface_opacity=1.0, plot_surface_taubin_smooth=False),
        )
        out.write_text(html_str, encoding="utf-8")
        return

    import plotly.graph_objects as go
    import plotly.express as px

    from mgrcli.plotting import dark_layout, white_layout

    cluster_traces = mgr.plot_percolated_cluster_meshes(
        analysis_copy["percolation_history"],
        analysis_copy["dgrid"],
        analysis_copy["regions"],
        atoms,
        probe_radius=probe_radius,
        isovalue=analysis_copy["isovalue"],
        stepsize=2,
        stylize=True,
        opacity=1.0,
        taubin_smooth=False,
        cscale=px.colors.qualitative.Pastel,
    )
    plot_traces = [mgr.plot_cell(atoms)]
    rag_trace = mgr.plot_rag(analysis_copy["rag"], name="Pore Graph")
    if isinstance(rag_trace, (list, tuple)):
        plot_traces.extend(rag_trace)
    else:
        plot_traces.append(rag_trace)
    if analysis_copy.get("maxima") is not None and len(analysis_copy["maxima"]):
        maxima_trace = mgr.plot_maxima(
            analysis_copy["dgrid"],
            analysis_copy["maxima"],
            atoms,
            show_spheres=False,
            maxima_info=analysis_copy.get("maxima_info"),
            maxima_pbc_dict=analysis_copy.get("maxima_pbc_dict"),
        )
        if isinstance(maxima_trace, tuple):
            plot_traces.extend(maxima_trace)
        else:
            plot_traces.append(maxima_trace)
    plot_traces.extend(cluster_traces)
    if analysis_copy.get("percolation_history") and analysis_copy.get("connections"):
        from mgr.visualize import plot_percolation_windows

        window_traces = plot_percolation_windows(
            analysis_copy,
            atoms,
            probe_radius=probe_radius,
        )
        if isinstance(window_traces, (list, tuple)):
            plot_traces.extend(window_traces)
        elif window_traces:
            plot_traces.append(window_traces)

    fig = go.Figure(data=plot_traces)
    layout = white_layout if theme == "white" else dark_layout
    fig.update_layout(**layout)
    fig.write_html(str(out))


def save_textural_breakdown_html(
    results: Mapping[str, Any],
    path: Path | str,
    *,
    quantity: str,
    title: str,
    y_label: str,
) -> None:
    """Plotly bar chart for SA or void volume breakdown (fragment / pocket / cluster)."""
    import plotly.express as px

    breakdown_data = []
    by_region = results.get("by_region", {}) or results.get("by_region_fragments", {})
    for region_id, value in by_region.items():
        breakdown_data.append(
            {
                "Type": "Region (fragment)",
                "Category": "Segmented",
                "Label": f"region_{region_id}",
                y_label: value,
            }
        )
    for pocket_id, value in (results.get("by_pocket", {}) or {}).items():
        breakdown_data.append(
            {
                "Type": "Pocket (PBC)",
                "Category": "Grouped",
                "Label": f"pocket_{pocket_id}",
                y_label: value,
            }
        )
    for cluster_id, value in (results.get("by_percolated_cluster", {}) or {}).items():
        breakdown_data.append(
            {
                "Type": "Cluster (percolated)",
                "Category": "Accessible",
                "Label": str(cluster_id),
                y_label: value,
            }
        )
    if not breakdown_data:
        if quantity == "sa":
            total = results.get("total_surface_area", 0)
            acc = results.get("accessible_surface_area", 0)
        else:
            total = results.get("total_void_volume", 0)
            acc = results.get("accessible_void_volume", 0)
        breakdown_data = [
            {"Type": "Global", "Category": "Total", "Label": "Total", y_label: total},
            {"Type": "Global", "Category": "Accessible", "Label": "Accessible", y_label: acc},
            {
                "Type": "Global",
                "Category": "Inaccessible",
                "Label": "Inaccessible",
                y_label: total - acc,
            },
        ]
    df = __import__("pandas").DataFrame(breakdown_data)
    fig = px.bar(df, x="Label", y=y_label, color="Type", barmode="group", title=title)
    fig.write_html(str(path))

#!/usr/bin/env python3
"""
Standalone PLD / percolation figure matching the Streamlit interactive app
(Plotly backend): CIF → compute (dgrid + segmentation) → PLD analysis → figure.

Traces (main visualization + percolation tab, Plotly):
  framework mesh, unit cell, pore graph (RAG), windows (internal, red),
  maxima, percolated cluster meshes, percolation windows (yellow + PLD-critical).

Usage:
    python scripts/plot_pld_critical_windows.py
    python scripts/plot_pld_critical_windows.py tests/fixtures/cif/Cu-BTC.cif
    python scripts/plot_pld_critical_windows.py --output figures/ cifs/LTA96.cif
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[1]
MGR_SRC = REPO_ROOT / "mgr_src"
PYTHON_CLI = REPO_ROOT / "python_cli"
for p in (MGR_SRC, PYTHON_CLI):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ase.io import read  # noqa: E402

import mgr  # noqa: E402
from mgr.analyze import compute_aromatic_cycles, update_with_covalent_bonds  # noqa: E402
from mgrcli.cli import compute_dgrid_rust, python_do_periodic_segmentation_rust  # noqa: E402
from mgrcli.plotting import (  # noqa: E402
    STYLIZE_LIGHTING,
    STYLIZE_LIGHTPOSITION,
    dark_layout,
    plot_framework_as_mesh,
    prepare_results_for_plot,
)

DEFAULT_PARAMS: Dict[str, Any] = {
    "spacing": 0.5,
    "mask_thickness": 0.5,
    "h_value": 0.2,
    "min_distance": 1.5,
    "threshold_abs": 1.5,
    "extra_padding": 5.0,
    "watershed_backend": "rust_det",
    "surface_isovalue": 1.0,
    "pld_tolerance": 0.25,
    "percolation_history_step": 0.05,
    "cluster_stepsize": 2,
    "cluster_opacity": 1.0,
    "framework_mesh_scale": 0.65,
    "framework_bond_radius": 0.15,
    "framework_show_rings": True,
    "boundary_line_width": 2,
    "boundary_line_color": "black",
    "plot_windows": True,
    "windows_marker_color": "yellow",
    "windows_marker_size":15,
    "windows_opacity": 0.6,
    "stylize": True,
    "layout": "dark",
}

DEFAULT_CIFS: Tuple[str, ...] = (
    "tests/fixtures/cif/Cu-BTC.cif",
    "cifs/LTA96.cif",
)

CS = px.colors.qualitative.Pastel


def _run_segmentation(
    mat_atoms,
    dgrid: np.ndarray,
    *,
    spacing: float,
    mask_thickness: float,
    h_value: float,
    min_distance: float,
    threshold_abs: float,
    extra_padding: float,
    watershed_backend: str,
) -> Dict[str, Any]:
    return python_do_periodic_segmentation_rust(
        dgrid=dgrid,
        mat_atoms=mat_atoms,
        mask_thickness=mask_thickness,
        h=h_value,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        return_padded_regions=False,
        spacing=spacing,
        minimum_window_separation=1.0,
        wall_thickness=0.05,
        extra_padding=extra_padding,
        compactness=0,
        min_region_size=0.001,
        disable_tqdm=True,
        compute_connections=True,
        return_dgrid=True,
        watershed_backend=watershed_backend,
    )


def _compute_pld_analysis(
    results: Dict[str, Any],
    mat_atoms,
    *,
    spacing: float,
    mask_thickness: float,
    h_value: float,
    threshold_abs: float,
    min_distance: float,
    extra_padding: float,
    watershed_backend: str,
    percolation_history_step: float,
) -> Dict[str, Any]:
    analysis_results = results
    prepared = prepare_results_for_plot(results, mat_atoms)

    dgrid = np.asarray(prepared.get("dgrid"))
    regions = np.asarray(prepared.get("regions"))
    maxima_info = prepared.get("maxima_info")
    if dgrid is None or regions is None or dgrid.size == 0 or regions.size == 0:
        raise ValueError("Segmentation results are missing distance grid or regions.")

    maxima_raw = np.asarray(analysis_results.get("all_maxima", []), dtype=int)
    if maxima_raw.ndim == 1:
        maxima_raw = maxima_raw.reshape(-1, 3) if maxima_raw.size else np.empty((0, 3), dtype=int)
    if maxima_raw.ndim != 2:
        maxima_raw = np.empty((0, 3), dtype=int)

    unique_regions = np.asarray(analysis_results.get("unique_regions", []), dtype=int)

    seg_kwargs = dict(
        spacing=spacing,
        mask_thickness=mask_thickness,
        h=h_value,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
        extra_padding=extra_padding,
    )

    lcd, pld, dfsd, _ = mgr.winding_percolation.get_pore_dia_mgr_winding(
        mat_atoms=mat_atoms,
        dgrid=dgrid,
        kwargs=seg_kwargs,
        results=analysis_results,
    )

    percolation_history, pld_locations, _final_clusters = (
        mgr.winding_percolation.compute_percolation_history_winding(
            analysis_results,
            step=percolation_history_step,
            expand_components=False,
        )
    )
    pld_locations = np.asarray(pld_locations)

    connections_prepared = prepared.get("connections", [])
    connections_filtered = [
        conn
        for conn in connections_prepared
        if np.all(np.isin(conn[0], unique_regions))
    ]

    all_maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)]
    maxima = maxima_raw[: regions.max()]
    maxima_radii = all_maxima_radii[: regions.max()]
    rag = mgr.rag_from_connections(connections_filtered, maxima)
    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, regions.shape, mat_atoms)
    rag.graph["regions"] = regions
    rag.graph["ase_atoms"] = mat_atoms

    percolation_isovalue = float(
        np.clip(np.min([pld / 2.0 - 0.25, 1.75]), 0.1, None)
    )

    return {
        "lcd": float(lcd),
        "pld": float(pld),
        "dfsd": float(dfsd),
        "percolation_history": percolation_history,
        "pld_locations": pld_locations,
        "rag": rag,
        "maxima": maxima,
        "maxima_radii": maxima_radii,
        "maxima_info": analysis_results.get("maxima_info", np.zeros((0, 7), dtype=int)),
        "maxima_pbc_dict": analysis_results.get("maxima_pbc_dict", {}),
        "dgrid": dgrid,
        "regions": regions,
        "connections": connections_filtered,
        "isovalue": percolation_isovalue,
        "percolation_source": "segmentation_results",
        "ase_atoms": mat_atoms,
    }


def _estimate_spacing(mat_atoms, dgrid_shape: Tuple[int, ...]) -> float:
    """Match ``show_mof`` spacing estimate for ``plot_windows`` filtering."""
    cell = mat_atoms.get_cell().array.T
    lengths = np.linalg.norm(cell, axis=0)
    shape = np.asarray(dgrid_shape, dtype=float)
    return float(np.mean(lengths / np.maximum(shape, 1.0)))


def run_compute_pipeline(
    cif_path: Path,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Any]:
    path_to_cif = str(cif_path)
    mat_atoms = read(path_to_cif)

    mat_atoms = update_with_covalent_bonds(mat_atoms)
    compute_aromatic_cycles(mat_atoms)

    dgrid = compute_dgrid_rust(mat_atoms, float(params["spacing"]))
    results = _run_segmentation(
        mat_atoms,
        dgrid,
        spacing=float(params["spacing"]),
        mask_thickness=float(params["mask_thickness"]),
        h_value=float(params["h_value"]),
        min_distance=float(params["min_distance"]),
        threshold_abs=float(params["threshold_abs"]),
        extra_padding=float(params["extra_padding"]),
        watershed_backend=str(params["watershed_backend"]),
    )
    results["ase_atoms"] = mat_atoms
    results["dgrid"] = results.get("dgrid", dgrid)
    return results, mat_atoms


def _select_probe_radius(
    percolation_history: Sequence[Tuple[float, Any]],
    pld: float,
    probe_radius: Optional[float],
) -> float:
    radii = sorted({float(entry[0]) for entry in percolation_history})
    if not radii:
        raise ValueError("Empty percolation history")
    target = float(pld / 2.0) if probe_radius is None else float(probe_radius)
    return min(radii, key=lambda r: abs(r - target))


def _extend_traces(
    dest: List[Any],
    src: Union[Any, Sequence[Any], None],
) -> None:
    if src is None:
        return
    if isinstance(src, (list, tuple)):
        dest.extend(src)
    else:
        dest.append(src)


def build_percolation_figure(
    analysis: Dict[str, Any],
    mat_atoms,
    *,
    probe_radius: float,
    params: Dict[str, Any],
    material_label: str = "",
) -> go.Figure:
    mesh_isovalue = max(float(params["surface_isovalue"]), 0.05)
    stylize = bool(params["stylize"])

    plot_traces: List[Any] = []

    _extend_traces(
        plot_traces,
        plot_framework_as_mesh(
            mat_atoms,
            stylize=stylize,
            style="mesh",
            show_rings=bool(params["framework_show_rings"]),
            kwargs_mesh_for_atoms={
                "scale": float(params["framework_mesh_scale"]),
                "radius": float(params["framework_bond_radius"]),
            },
        ),
    )

    plot_traces.append(mgr.plot_cell(mat_atoms))

    _extend_traces(plot_traces, mgr.plot_rag(analysis["rag"], name="Pore Graph"))

    if params.get("plot_windows", True):
        connections = analysis.get("connections", [])
        internal_for_windows = [
            c
            for c in connections
            if c[1]
            and c[2] == "internal"
            and len(c) > 3
            and c[3] is not None
            and np.asarray(c[3]).size >= 3
        ]
        if internal_for_windows:
            est_spacing = _estimate_spacing(mat_atoms, analysis["dgrid"].shape)
            _extend_traces(
                plot_traces,
                mgr.plot_windows(
                    internal_for_windows,
                    atoms=mat_atoms,
                    spacing=est_spacing,
                    name="windows (internal)",
                    marker_color=str(params.get("windows_marker_color", "red")),
                    marker_size=int(params.get("windows_marker_size", 10)),
                    opacity=float(params.get("windows_opacity", 0.6)),
                ),
            )

    if analysis["maxima"].size:
        _extend_traces(
            plot_traces,
            mgr.plot_maxima(
                analysis["dgrid"],
                analysis["maxima"],
                mat_atoms,
                show_spheres=False,
                sphere_opacity=0.85,
                maxima_info=analysis.get("maxima_info"),
                maxima_pbc_dict=analysis.get("maxima_pbc_dict"),
            ),
        )

    cluster_traces = mgr.plot_percolated_cluster_meshes(
        analysis["percolation_history"],
        analysis["dgrid"],
        analysis["regions"],
        mat_atoms,
        probe_radius=probe_radius,
        isovalue=mesh_isovalue,
        stepsize=int(params["cluster_stepsize"]),
        stylize=stylize,
        opacity=float(params["cluster_opacity"]),
        taubin_smooth=False,
        cscale=CS,
        boundary_line_width=int(params["boundary_line_width"]),
        boundary_line_color=str(params["boundary_line_color"]),
    )
    _extend_traces(plot_traces, cluster_traces)

    if analysis.get("percolation_history"):
        _extend_traces(
            plot_traces,
            mgr.plot_percolation_windows(
                analysis,
                probe_radius=probe_radius,
                pld_tolerance=float(params["pld_tolerance"]),
            ),
        )

    fig = go.Figure(data=plot_traces)
    fig.update_layout(**dark_layout)

    plr = float(probe_radius)
    slug = (
        f"spacing_{params['spacing']:g}_h_{params['h_value']:g}_"
        f"threshAbs_{params['threshold_abs']:g}_mask_{params['mask_thickness']:g}"
    )
    title = f"Percolation probe r={plr:.2f}Å | {material_label or 'MOF'} | {slug}"
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff"), y=0.95),
        font=dict(color="#ffffff"),
        scene=dict(aspectmode="data"),
    )

    for trace in fig.data:
        if isinstance(trace, go.Mesh3d):
            trace.update(
                lighting=STYLIZE_LIGHTING,
                lightposition=STYLIZE_LIGHTPOSITION,
                showscale=False,
            )

    return fig


def process_material(
    cif_path: Path,
    *,
    output_dir: Path,
    params: Dict[str, Any],
    probe_radius: Optional[float],
    show: bool,
) -> Path:
    t0 = time.time()
    print(f"\n{'=' * 70}\n  {cif_path.name}\n{'=' * 70}")

    results, mat_atoms = run_compute_pipeline(cif_path, params)
    print(f"  Segmentation: {len(mat_atoms)} atoms, dgrid {np.asarray(results['dgrid']).shape}")

    analysis = _compute_pld_analysis(
        results,
        mat_atoms,
        spacing=float(params["spacing"]),
        mask_thickness=float(params["mask_thickness"]),
        h_value=float(params["h_value"]),
        threshold_abs=float(params["threshold_abs"]),
        min_distance=float(params["min_distance"]),
        extra_padding=float(params["extra_padding"]),
        watershed_backend=str(params["watershed_backend"]),
        percolation_history_step=float(params["percolation_history_step"]),
    )

    selected_radius = _select_probe_radius(
        analysis["percolation_history"],
        analysis["pld"],
        probe_radius,
    )
    print(
        f"  LCD={analysis['lcd']:.3f} Å  PLD={analysis['pld']:.3f} Å  "
        f"DFSD={analysis['dfsd']:.3f} Å  probe r={selected_radius:.3f} Å"
    )

    fig = build_percolation_figure(
        analysis,
        mat_atoms,
        probe_radius=selected_radius,
        params=params,
        material_label=cif_path.stem,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_html = output_dir / f"pld_percolation_{cif_path.stem}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"  Wrote {out_html} ({time.time() - t0:.1f}s)")

    if show:
        fig.show()

    return out_html


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PLD percolation figure (Plotly), matching interactive_app."
    )
    parser.add_argument("cifs", nargs="*", help="CIF paths (default: Cu-BTC + LTA96)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPO_ROOT / "figures" / "pld_percolation",
        help="Output directory for HTML figures",
    )
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=None,
        help="Probe radius (Å); default nearest history entry to PLD/2",
    )
    parser.add_argument("--show", action="store_true", help="Open Plotly viewer")
    parser.add_argument("--spacing", type=float, default=None)
    parser.add_argument("--mask-thickness", type=float, default=None)
    parser.add_argument("--h-value", type=float, default=None)
    parser.add_argument("--threshold-abs", type=float, default=None)
    parser.add_argument("--min-distance", type=float, default=None)
    parser.add_argument("--extra-padding", type=float, default=None)
    parser.add_argument("--watershed-backend", type=str, default=None)
    parser.add_argument("--surface-isovalue", type=float, default=None)
    parser.add_argument("--pld-tolerance", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    params = dict(DEFAULT_PARAMS)
    for key, arg_val in (
        ("spacing", args.spacing),
        ("mask_thickness", args.mask_thickness),
        ("h_value", args.h_value),
        ("threshold_abs", args.threshold_abs),
        ("min_distance", args.min_distance),
        ("extra_padding", args.extra_padding),
        ("watershed_backend", args.watershed_backend),
        ("surface_isovalue", args.surface_isovalue),
        ("pld_tolerance", args.pld_tolerance),
    ):
        if arg_val is not None:
            params[key] = arg_val

    cif_paths = [Path(p) for p in (args.cifs or DEFAULT_CIFS)]
    for cif in cif_paths:
        if not cif.is_file():
            print(f"Missing CIF: {cif}", file=sys.stderr)
            return 1

    for cif in cif_paths:
        process_material(
            cif.resolve(),
            output_dir=args.output.resolve(),
            params=params,
            probe_radius=args.probe_radius,
            show=args.show,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

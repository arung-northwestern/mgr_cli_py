"""Streamlit application for interactive MOF analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
from io import BytesIO

import plotly.express as px

import numpy as np
import streamlit as st
from ase.io import read

import plotly.graph_objects as go
import pandas as pd

import sys
from pathlib import Path
# Add mgr_src to path to use reference functions directly
REPO_ROOT = Path(__file__).resolve().parents[2]
MGR_SRC_PATH = REPO_ROOT / "mgr_src"
if str(MGR_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(MGR_SRC_PATH))

# Import the reference MGR library from mgr_src/mgr
try:
    import mgr as mgr
    from mgr.helper import estimate_h_from_percolation_history
except ImportError:
    # Fallback to mgr_visualize if mgr_src is not available
    import mgr_visualize.mgr as mgr
    from mgr_visualize.mgr.helper import estimate_h_from_percolation_history


def results_data_to_npz(results: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    serializable: Dict[str, Any] = {}
    for key, value in results.items():
        if key == "ase_atoms":
            continue
        if isinstance(value, np.ndarray):
            serializable[key] = value
        elif isinstance(value, (int, float, bool, str)):
            serializable[key] = value
    if not serializable:
        return b""
    np.savez(buffer, **serializable)
    buffer.seek(0)
    return buffer.getvalue()


def make_uniform_psd_dataframe(bins: np.ndarray, histogram: np.ndarray) -> pd.DataFrame:
    bin_starts = bins[:-1]
    bin_ends = bins[1:]
    bin_centers = 0.5 * (bin_starts + bin_ends)
    return pd.DataFrame(
        {
            "bin_start": bin_starts,
            "bin_end": bin_ends,
            "bin_center": bin_centers,
            "count": histogram,
        }
    )


def make_pocket_psd_dataframe(bins: np.ndarray, pocket_psds: Dict[Any, np.ndarray]) -> pd.DataFrame:
    rows = []
    bin_starts = bins[:-1]
    bin_ends = bins[1:]
    bin_centers = 0.5 * (bin_starts + bin_ends)
    for label, hist in pocket_psds.items():
        for start, end, center, count in zip(bin_starts, bin_ends, bin_centers, hist):
            rows.append(
                {
                    "region_label": int(label),
                    "bin_start": start,
                    "bin_end": end,
                    "bin_center": center,
                    "count": count,
                }
            )
    return pd.DataFrame(rows)

if __package__ is None or __package__ == "":  # pragma: no cover - streamlit entry
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from python_cli.mcli.cli import compute_dgrid_rust, python_do_periodic_segmentation_rust, get_pbc_groups
    from python_cli.mcli.plotting import (
        show_mof,
        STYLIZE_LIGHTING,
        STYLIZE_LIGHTPOSITION,
        dark_layout,
        white_layout,
        prepare_results_for_plot,
    )
else:  # pragma: no cover - package import path
    from .plotting import show_mof, STYLIZE_LIGHTING, STYLIZE_LIGHTPOSITION, dark_layout, white_layout, prepare_results_for_plot
    from .cli import compute_dgrid_rust, python_do_periodic_segmentation_rust, get_pbc_groups


ASCII_LOGO = r"""
███╗   ███╗ ██████╗ ███████╗ ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██╗   ██╗
████╗ ████║██╔═══██╗██╔════╝██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║╚██╗ ██╔╝
██╔████╔██║██║   ██║█████╗  ██║   ██║██║  ███╗██████╔╝███████║██████╔╝███████║ ╚████╔╝ 
██║╚██╔╝██║██║   ██║██╔══╝  ██║   ██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║  ╚██╔╝  
██║ ╚═╝ ██║╚██████╔╝██║     ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║   ██║   
╚═╝     ╚═╝ ╚═════╝ ╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   
"""


def _get_sample_cifs() -> Dict[str, Path]:
    sample_dir = Path(os.environ.get("MCLI_SAMPLE_CIF_DIR", Path(__file__).resolve().parents[2] / "cifs"))
    samples: Dict[str, Path] = {}
    if sample_dir.exists():
        for cif_path in sorted(sample_dir.glob("*.cif")):
            samples[cif_path.name] = cif_path
    return samples


def _auto_parameters(mat_atoms, spacing: float) -> Dict[str, float]:
    dgrid = compute_dgrid_rust(mat_atoms, spacing)
    pore_limiting_radius = mgr.find_pore_limiting_radius(dgrid)
    mask_thickness = float(np.clip(pore_limiting_radius - 0.25, 0.1, 0.5))
    h_value = float(max(estimate_h_from_percolation_history(dgrid) - 0.1, 0.25))
    surface_value = float(min(mask_thickness + 0.5, 2.0))
    return {
        "dgrid": dgrid,
        "mask_thickness": mask_thickness,
        "h": h_value,
        "surface_value": surface_value,
    }


def _run_segmentation(
    mat_atoms,
    dgrid: np.ndarray,
    spacing: float,
    mask_thickness: float,
    h_value: float,
    min_distance: float = 1.5,
    threshold_abs: float = 1.5,
    extra_padding: float = 5.0,
) -> Dict[str, np.ndarray]:
    segmentation_kwargs = dict(
        dgrid=dgrid,
        mat_atoms=mat_atoms,
        mask_thickness=mask_thickness,
        h=h_value,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        return_padded_regions=False,
        spacing=spacing,
        minimum_window_separation=None,
        wall_thickness=0.05,
        extra_padding=extra_padding,
        compactness=0,
        min_region_size=0.001,
        disable_tqdm=True,
        compute_connections=True,
        return_dgrid=True,
    )
    return python_do_periodic_segmentation_rust(**segmentation_kwargs)


def _compute_pld_analysis(
    results: Dict[str, Any],
    mat_atoms,
    spacing: float,
    mask_thickness: float,
) -> Dict[str, Any]:
    prepared = prepare_results_for_plot(results, mat_atoms)

    dgrid = np.asarray(prepared.get("dgrid"))
    print('Got grid')
    regions = np.asarray(prepared.get("regions"))
    print('Got regions')
    maxima_info = prepared.get("maxima_info")
    if dgrid is None or regions is None or dgrid.size == 0 or regions.size == 0:
        raise ValueError("Segmentation results are missing distance grid or regions.")

    maxima_raw = np.asarray(results.get("all_maxima", []), dtype=int)

    if maxima_raw.ndim == 1:
        maxima_raw = maxima_raw.reshape(-1, 3) if maxima_raw.size else np.empty((0, 3), dtype=int)
    if maxima_raw.ndim != 2:
        maxima_raw = np.empty((0, 3), dtype=int)

    unique_regions = np.asarray(results.get("unique_regions", []), dtype=int)

    # if unique_regions.size:
    #     max_label = int(unique_regions.max())
    # elif regions.size:
    #     max_label = int(np.max(regions))
    # else:
    #     max_label = maxima_raw.shape[0]
    #

    # max_label = min(max_label, maxima_raw.shape[0])
    # maxima = maxima_raw[:max_label] if max_label > 0 else np.empty((0, 3), dtype=int)

    lcd, pld, dfsd, _ = mgr.get_pore_dia_mgr(mat_atoms, dgrid=dgrid, kwargs=dict(spacing=spacing))
    print('got pore dia')

    percolation_history, pld_locations, _, _, _ = mgr.compute_percolation_history(
        distance_grid=dgrid,
        regions=regions,
        maxima=maxima_raw, 
        kwargs=dict(mask_thickness=mask_thickness, spacing=spacing),
    )
    print('got percolation hist')

    pld_locations = np.asarray(pld_locations)

    connections_prepared = prepared.get("connections", [])
    connections_filtered= [ conn for conn in connections_prepared if np.all(np.isin(conn[0], unique_regions) )]

    all_maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)]
    maxima = maxima_raw[:regions.max()]
    maxima_radii = all_maxima_radii[:regions.max()]
    rag = mgr.rag_from_connections(connections_filtered, maxima)
    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, regions.shape, mat_atoms)
    rag.graph["regions"] = regions
    rag.graph["ase_atoms"] = mat_atoms

    percolation_isovalue = float(
        np.clip(
            np.min([pld / 2.0 - 0.25, mask_thickness + 0.5, 1.75]),
            0.1,
            None,
        )
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
        "dgrid": dgrid,
        "regions": regions,
        "connections": connections_filtered,
        "isovalue": percolation_isovalue,
    }


def _make_plot(results: Dict[str, Any], *, stylize: bool, projection: str, layout: str,
               framework_style: str, show_surface: bool, show_surface_pbc: bool,
               surface_value: float, surface_opacity: float, surface_pbc_opacity: float,
               show_connections: bool, show_maxima_spheres: bool) -> go.Figure:
    options = {
        "plot_framework_style": framework_style,
        "plot_surface": show_surface,
        "plot_surface_pbc": show_surface_pbc,
        "plot_surface_isovalue": surface_value,
        "plot_surface_opacity": surface_opacity,
        "plot_surface_pbc_opacity": surface_pbc_opacity,
        "plot_connections": show_connections,
        "plot_maxima_show_spheres": show_maxima_spheres,
        "plot_projection": projection,
        "plot_layout": layout,
    }
    return show_mof(results, stylize=stylize, options=options)


def main() -> None:
    st.set_page_config(
        layout="wide",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )

    st.markdown(f"""
```
{ASCII_LOGO}
```
""")
    st.markdown("## Pore Analysis & Visualization Web App 📹")

    samples = _get_sample_cifs()

    sample_choice = None
    if samples:
        sample_choice = st.selectbox(
            "Or select a sample CIF from the repository",
            ["<None>"] + list(samples.keys()),
        )
        if sample_choice == "<None>":
            sample_choice = None

    uploaded_file = st.file_uploader("Upload CIF file", type=["cif"])

    atoms = None
    cif_path: Optional[Path] = None
    if uploaded_file is not None:
        cif_path = Path(uploaded_file.name)
        with open(cif_path, "wb") as buffer:
            buffer.write(uploaded_file.getbuffer())
        atoms = read(str(cif_path))
    elif sample_choice:
        cif_path = samples[sample_choice]
        atoms = read(str(cif_path))

    if atoms is None:
        st.info("Upload a CIF file or pick a sample to begin.")
        return

    st.markdown("### 🔧 Workflow Parameters")

    spacing = st.number_input(
        "Grid Spacing (Å)",
        value=0.5,
        min_value=0.1,
        step=0.1,
        help="Grid spacing in Angstroms. Smaller values increase resolution and runtime.",
    )

    auto_set_params = st.checkbox("Auto set parameters", value=True)
    
    # Minimum box thickness option
    min_box_thickness_enabled = st.checkbox(
        "Ensure minimum box thickness", 
        value=False,
        help="Create supercell to ensure minimum box dimensions for accurate analysis. Useful for small unit cells."
    )
    
    min_box_thickness = 6.0  # Default 6Å
    if min_box_thickness_enabled:
        min_box_thickness = st.number_input(
            "Minimum box thickness (Å)",
            value=6.0,
            min_value=1.0,
            max_value=50.0,
            step=0.5,
            help="Minimum dimension for the simulation box. Creates supercell if unit cell is smaller."
        )

    param_state = st.session_state.setdefault("param_cache", {})
    cache_key = f"{getattr(cif_path, 'name', 'uploaded')}::{spacing:.3f}"
    if auto_set_params and cache_key not in param_state:
        with st.spinner("Computing suggested parameters..."):
            param_state[cache_key] = _auto_parameters(atoms, spacing)

    current_params = param_state.get(cache_key)
    if current_params is None:
        current_params = {
            "dgrid": compute_dgrid_rust(atoms, spacing),
            "mask_thickness": 1.5,
            "h": 0.4,
            "surface_value": 2.0,
        }

    col1, col2, col3 = st.columns(3)
    with col1:
        mask_thickness = st.number_input(
            "Mask thickness (Å)",
            value=float(current_params["mask_thickness"]),
            min_value=0.05,
            help="Points closer than this to the framework are excluded.",
        )
    with col2:
        h_value = st.number_input(
            "h (Å)",
            value=float(current_params["h"]),
            min_value=0.05,
            help="Minimum separation between neighbouring maxima.",
        )
    with col3:
        surface_value = st.number_input(
            "Surface value (Å)",
            value=float(current_params["surface_value"]),
            min_value=0.1,
            help="Isosurface value for visualization.",
        )

    # Additional segmentation parameters
    with st.expander("Advanced Segmentation Parameters", expanded=False):
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        with col_adv1:
            min_distance = st.number_input(
                "Min distance between maxima (Å)",
                value=1.5,
                min_value=0.1,
                step=0.1,
                help="Minimum distance required between distinct local maxima.",
            )
        with col_adv2:
            threshold_abs = st.number_input(
                "Threshold absolute (Å)",
                value=1.5,
                min_value=0.1,
                step=0.1,
                help="Absolute threshold for maxima detection.",
            )
        with col_adv3:
            extra_padding = st.number_input(
                "Extra padding (Å)",
                value=5.0,
                min_value=0.0,
                step=0.5,
                help="Extra padding around the unit cell for periodic boundary conditions.",
            )

    st.markdown("### 🎨 Visualization Options")
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        mesh_framework = st.checkbox("Show framework as mesh", value=True)
        simple_framework = st.checkbox("Show simple framework scatter", value=False)
        if mesh_framework and simple_framework:
            st.warning("Select either mesh or simple framework.")
        projection_choice = st.selectbox("Projection", ["perspective", "orthographic"], index=0)
        stylize = st.checkbox("Stylize plot", value=True, disabled=(projection_choice == "orthographic"))
        if projection_choice == "orthographic":
            stylize = False
        layout_choice = st.selectbox("Background", ["dark", "white"], index=0)
    with col_vis2:
        show_surface = st.checkbox(f"Show pore surface at {surface_value:.2f} Å", value=True)
        show_surface_pbc = st.checkbox("Show periodic pore surface", value=True)
        show_connections = st.checkbox("Show pore graph", value=True)
        show_maxima_spheres = st.checkbox("Show maxima spheres", value=False)
        detect_pocket_types = st.checkbox("Detect Pocket Types", value=False)

        js_threshold = 0.15
        if detect_pocket_types:
            js_threshold = st.number_input(
                "Jensen-Shannon Threshold",
                min_value=0.01,
                max_value=1.0,
                value=0.15,
                step=0.01,
                help="Threshold for agglomerative clustering based on Jensen-Shannon distance between cavity size distributions"
            )

    surface_opacity = 1.0
    surface_pbc_opacity = 1.0

    compute_clicked = st.button("Compute and Visualize")

    if compute_clicked:
        status = st.empty()
        status.markdown("<div style='padding:8px;background:#332700;color:#FFD700;text-align:center;'>⚡ COMPUTING ⚡</div>", unsafe_allow_html=True)

        try:
            # Handle minimum box thickness - create supercell if needed
            working_atoms = atoms
            if min_box_thickness_enabled:
                # Get current cell dimensions
                cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
                min_current_length = min(cell_lengths)
                
                if min_current_length < min_box_thickness:
                    # Calculate repetitions needed
                    try:
                        repetitions = mgr.get_repetitions_to_meet_cutoff(atoms, min_box_thickness)
                        if hasattr(repetitions, '__len__') and len(repetitions) >= 3:
                            rep_str = f"{repetitions[0]}x{repetitions[1]}x{repetitions[2]}"
                        else:
                            rep_str = f"{repetitions}x{repetitions}x{repetitions}"
                        st.info(f"Creating {rep_str} supercell to meet minimum box thickness of {min_box_thickness:.1f} Å")
                        working_atoms = atoms * repetitions
                    except Exception as e:
                        st.warning(f"Could not create supercell automatically: {e}. Using original unit cell.")
            
            results = _run_segmentation(
                working_atoms,
                current_params["dgrid"],
                spacing=spacing,
                mask_thickness=mask_thickness,
                h_value=h_value,
                min_distance=min_distance,
                threshold_abs=threshold_abs,
                extra_padding=extra_padding,
            )
            results["ase_atoms"] = working_atoms
            results["dgrid"] = results.get("dgrid", current_params["dgrid"])

            # Add pocket type detection if requested
            if detect_pocket_types:
                from skimage.measure import regionprops
                import plotly.express as px

                # Get required data for clustering
                dgrid = results["dgrid"]
                regions = results["regions"]
                regions_pbc = results["regions_pbc"]
                maxima_info = results["maxima_info"]
                all_maxima = results["all_maxima"]
                all_maxima_radii = dgrid[tuple(maxima_info[:,1:4].T)]
                connections = results["connections"]
                unique_regions = results['unique_regions']
                mat_atoms = results["ase_atoms"]

                # Compute proper PBC groups using the reference library function
                pbc_groups_dict = get_pbc_groups(unique_regions, maxima_info)
                pbc_groups = [v for k, v in pbc_groups_dict.items()]

                # Trim maxima using regions.max() as shown in example
                maxima = all_maxima[:regions.max()]
                maxima_radii = all_maxima_radii[:regions.max()]
                connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]

                try:
                    if len(maxima) > 0:
                        # Debug output (can be removed in production)
                        print('regions shape:', regions.shape, 'unique:', np.unique(regions))
                        print('maxima shape:', maxima.shape, 'values:', maxima)
                        print('maxima_info shape:', maxima_info.shape, 'values:', maxima_info)
                        print('pbc_groups:', pbc_groups)
                        print('connections:', connections)
                        print('unique_regions:', unique_regions)
                        print('maxima_radii:', maxima_radii)
                        print('connections_trimmed:', connections_trimmed)

                        # Build RAG following the example pattern (from scratch)
                        rag = mgr.rag_from_connections(connections_trimmed, maxima)
                        print('rag nodes after connections:', rag.nodes())
                        print('rag edges after connections:', rag.edges())
                        rag = mgr.add_pbc_to_rag(rag, pbc_groups)
                        print('rag nodes after pbc:', rag.nodes())
                        print('rag edges after pbc:', rag.edges())
                        rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid.shape, mat_atoms)
                        print('rag nodes after maxima:', rag.nodes())
                        print('rag edges after maxima:', rag.edges())
                        print('pbc groups ', pbc_groups)

                        # Add PSD and clustering to RAG
                        props = regionprops(regions)
                        bins = np.linspace(0, 30, 50)
                        rag = mgr.add_psd_to_rag_with_pbc(rag, props, pbc_groups, bins, mat_atoms, dgrid.shape)
                        print('rag nodes after psd:', rag.nodes())
                        print('rag edges after psd:', rag.edges())
                        rag = mgr.add_clustering_to_rag_geometry_js(rag, props, pbc_groups, mat_atoms, dgrid.shape, use_pbc=True, js_threshold=js_threshold)
                        print('rag nodes after clustering:', rag.nodes())
                        print('rag edges after clustering:', rag.edges())

                        # Generate regions cluster
                        regions_cluster = mgr.clustering_grid_from_regions(regions, rag, cluster_key="cluster_geom_pbc")
                        print('regions_cluster:', regions_cluster)

                        # Store results
                        results["regions_cluster"] = regions_cluster
                        results["rag_clustered"] = rag
                        results["pocket_types_detected"] = True
                        results["js_threshold_used"] = js_threshold

                        print(f"Pocket types detected using JS threshold: {js_threshold}")
                        num_clusters = len(np.unique(regions_cluster)) - 1  # -1 for background
                        print(f"Found {num_clusters} different pocket types")

                    else:
                        print("Warning: Cannot perform pocket type detection - no maxima data found")
                        results["pocket_types_detected"] = False

                except Exception as e:
                    print(f"Warning: Pocket type detection failed: {e}")
                    results["pocket_types_detected"] = False

        finally:
            status.markdown("<div style='padding:8px;background:#003300;color:#00FF00;text-align:center;'>✓ COMPUTE FINISHED ✓</div>", unsafe_allow_html=True)

        st.session_state["latest_results"] = results
        st.session_state["latest_metadata"] = {
            "spacing": float(spacing),
            "mask_thickness": float(mask_thickness),
            "surface_value": float(surface_value),
        }
        st.session_state["analysis_key"] = cache_key
        pld_cache = st.session_state.setdefault("pld_cache", {})
        pld_cache.pop(cache_key, None)

    results_data = st.session_state.get("latest_results")
    metadata = st.session_state.get("latest_metadata", {})
    analysis_key = st.session_state.get("analysis_key")

    visualization_tab, percolation_tab, psd_tab, sa_vol_tab = st.tabs(["Visualization", "PLD & Percolation", "Pore Size Distribution", "Surface Areas & Void Volumes"])

    with visualization_tab:
        if results_data is None:
            st.info("Run the computation to visualize the structure.")
        else:
            framework_style = "scatter" if simple_framework and not mesh_framework else "mesh"
            main_plot_status = st.empty()
            main_plot_status.markdown("<div style='padding:6px;background:#102a43;color:#7FDBFF;text-align:center;'>🎨 Rendering main view…</div>", unsafe_allow_html=True)
            fig = show_mof(
                results_data,
                stylize=stylize,
                options=dict(
                    plot_framework_style=framework_style,
                    plot_framework_show_rings=True,
                    plot_surface=show_surface,
                    plot_surface_pbc=show_surface_pbc,
                    plot_surface_isovalue=surface_value,
                    plot_surface_opacity=surface_opacity,
                    plot_surface_pbc_opacity=surface_pbc_opacity,
                    plot_connections=show_connections,
                    plot_maxima_show_spheres=show_maxima_spheres,
                    plot_projection=projection_choice,
                    plot_layout=layout_choice,
                ),
            )

            plot_column, tips_column = st.columns([3, 1])
            with plot_column:
                st.plotly_chart(fig, use_container_width=True)
                html_path = Path("interactive_plot.html")
                fig.write_html(html_path)
                with open(html_path, "rb") as file_handle:
                    st.download_button(
                        label="Download interactive HTML",
                        data=file_handle,
                        file_name=f"mcli_{getattr(cif_path, 'stem', 'analysis')}.html",
                        mime="text/html",
                    )
                if results_data:
                    npz_bytes = results_data_to_npz(results_data)
                    if npz_bytes:
                        st.download_button(
                            label="Download results NPZ",
                            data=npz_bytes,
                            file_name=f"mcli_{getattr(cif_path, 'stem', 'analysis')}_results.npz",
                            mime="application/octet-stream",
                        )
            with tips_column:
                st.markdown("""
                ### Tips
                - Toggle traces via the legend to focus on specific features.
                - Use simple framework for a faster plot, or turn it off if not needed.
                - Performance mode speeds up surface generation on large unit cells.
                - Orthographic projection pairs best with stylize disabled.
                """)

            main_plot_status.markdown("<div style='padding:6px;background:#06310c;color:#6FE37F;text-align:center;'>✅ Plotting complete</div>", unsafe_allow_html=True)

    with percolation_tab:
        if results_data is None:
            st.info("Run the computation to enable PLD and percolation analysis.")
        else:
            pld_cache = st.session_state.setdefault("pld_cache", {})
            effective_key = analysis_key or cache_key
            compute_pld = st.button("Compute PLD & percolation metrics", key="compute_pld")
            if compute_pld:
                with st.spinner("Computing PLD and percolation history..."):
                    try:
                        pld_cache[effective_key] = _compute_pld_analysis(
                            results_data,
                            atoms,
                            spacing=float(metadata.get("spacing", spacing)),
                            mask_thickness=float(metadata.get("mask_thickness", mask_thickness)),
                        )
                    except Exception as exc:
                        st.error(f"Percolation analysis failed: {exc}")

            analysis = pld_cache.get(effective_key)
            if not analysis:
                st.info("Click the button above to derive PLD, DFSD, and percolation visuals.")
            else:
                metrics_cols = st.columns(3)
                metrics_cols[0].metric("LCD (Å)", f"{analysis['lcd']:.2f}")
                metrics_cols[1].metric("PLD (Å)", f"{analysis['pld']:.2f}")
                metrics_cols[2].metric("DFSD (Å)", f"{analysis['dfsd']:.2f}")

                # Layout selection for percolation plots
                percolation_layout_choice = st.selectbox("Background", ["dark", "white"], index=0, key="percolation_layout")

                history = analysis["percolation_history"]
                if not history:
                    st.warning("No percolating clusters detected for the computed grid.")
                else:
                    history_rows = []
                    for radius, components in history:
                        history_rows.append(
                            {
                                "Probe radius (Å)": round(float(radius), 3),
                                "Along a": len(components.get(0, [])),
                                "Along b": len(components.get(1, [])),
                                "Along c": len(components.get(2, [])),
                            }
                        )
                    st.dataframe(history_rows, use_container_width=True)

                    radius_options = sorted({float(entry[0]) for entry in history})
                    prev_radius = st.session_state.get("percolation_radius")
                    selected_radius = st.select_slider(
                        "Probe radius for visualization (Å)",
                        options=radius_options,
                        value=prev_radius if prev_radius in radius_options else radius_options[-1],
                        format_func=lambda r: f"{r:.2f}",
                        key="percolation_radius",
                    )

                    st.markdown(
                        f"Selected probe radius: **{selected_radius:.2f} Å** (diameter {2*selected_radius:.2f} Å)"
                    )

                    plot_status = st.empty()
                    plot_status.markdown("<div style='padding:6px;background:#102a43;color:#7FDBFF;text-align:center;'>🎨 Rendering percolation view…</div>", unsafe_allow_html=True)

                    try:
                        # Ensure plotly express is accessible for color scale
                        import plotly.express as px
                        cluster_traces = mgr.plot_percolated_cluster_meshes(
                            analysis["percolation_history"],
                            analysis["dgrid"],
                            analysis["regions"],
                            atoms,
                            probe_radius=selected_radius,
                            isovalue=analysis["isovalue"],
                            stylize=True,
                            opacity=1.0,
                            cscale = px.colors.qualitative.Pastel,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                        cluster_traces = []

                    plot_traces = []
                    plot_traces.append(mgr.plot_cell(atoms))
                    rag_trace = mgr.plot_rag(analysis["rag"], name="Pore Graph")
                    if isinstance(rag_trace, (list, tuple)):
                        plot_traces.extend(rag_trace)
                    else:
                        plot_traces.append(rag_trace)

                    if analysis["maxima"].size:
                        maxima_trace = mgr.plot_maxima(
                            analysis["dgrid"],
                            analysis["maxima"],
                            atoms,
                            show_spheres=False,
                            sphere_opacity=0.85,
                        )
                        if isinstance(maxima_trace, tuple):
                            plot_traces.extend(maxima_trace)
                        else:
                            plot_traces.append(maxima_trace)

                    if analysis["pld_locations"].size:
                        pld_traces = mgr.plot_pld_locations(
                            analysis["dgrid"],
                            analysis["pld_locations"],
                            atoms,
                            show_spheres=False,
                            sphere_opacity=0.85,
                            marker_size=10,
                            marker_color="limegreen",
                        )
                        if isinstance(pld_traces, tuple):
                            plot_traces.extend(pld_traces)
                        else:
                            plot_traces.append(pld_traces)

                    plot_traces.extend(cluster_traces)

                    percolation_fig = go.Figure(data=plot_traces)
                    # Use the selected layout for percolation plot
                    layout = white_layout if percolation_layout_choice == "white" else dark_layout
                    percolation_fig.update_layout(**layout)
                    percolation_fig.update_layout(
                        title=f"Percolation clusters at probe radius {selected_radius:.2f} Å"
                    )

                    for trace in percolation_fig.data:
                        if isinstance(trace, go.Mesh3d):
                            trace.update(
                                lighting=STYLIZE_LIGHTING,
                                lightposition=STYLIZE_LIGHTPOSITION,
                                showscale=False,
                            )

                    st.plotly_chart(percolation_fig, use_container_width=True)
                    plot_status.markdown("<div style='padding:6px;background:#06310c;color:#6FE37F;text-align:center;'>✅ Plotting complete</div>", unsafe_allow_html=True)

                    selected_components = next(
                        (entry[1] for entry in history if np.isclose(entry[0], selected_radius)),
                        {},
                    )
                    with st.expander("Percolation components at selected radius", expanded=False):
                        st.json({
                            "a-direction": selected_components.get(0, []),
                            "b-direction": selected_components.get(1, []),
                            "c-direction": selected_components.get(2, []),
                        })

    with psd_tab:
        if results_data is None:
            st.info("Run the computation to enable pore size distribution analysis.")
        else:
            psd_cache = st.session_state.setdefault("psd_cache", {})
            effective_key = analysis_key or cache_key
            regions_pbc = results_data.get("regions_pbc")
            with st.form("psd_settings"):
                st.subheader("Global PSD (uniform sampling)")
                num_samples_total = st.number_input(
                    "Total samples",
                    value=10000,
                    min_value=1000,
                    step=1000,
                    help="Number of uniformly sampled points across the unit cell.",
                )
                uniform_num_bins = st.number_input(
                    "Number of bins",
                    value=200,
                    min_value=10,
                    step=10,
                )
                uniform_max_diameter = st.number_input(
                    "Maximum pore diameter (Å)",
                    value=50.0,
                    min_value=5.0,
                    step=5.0,
                )

                st.subheader("Pocket-wise PSD")
                num_samples_per_region = st.number_input(
                    "Samples per region",
                    value=1000,
                    min_value=100,
                    step=100,
                    help="Number of samples drawn within each pore region.",
                )
                pocket_num_bins = st.number_input(
                    "Number of bins (pocket-wise)",
                    value=200,
                    min_value=10,
                    step=10,
                )
                pocket_max_diameter = st.number_input(
                    "Maximum pore diameter (Å) for pockets",
                    value=50.0,
                    min_value=5.0,
                    step=5.0,
                )

                compute_psd = st.form_submit_button("Compute PSDs")

            psd_entry = psd_cache.get(effective_key)

            if compute_psd:
                bins_uniform = np.linspace(0, float(uniform_max_diameter), int(uniform_num_bins) + 1)
                bins_pocket = np.linspace(0, float(pocket_max_diameter), int(pocket_num_bins) + 1)
                with st.spinner("Computing pore size distributions..."):
                    uniform_hist, uniform_fig = mgr.compute_psd_uniform_rust(
                        results_data["dgrid"],
                        results_data["ase_atoms"],
                        num_samples_total=int(num_samples_total),
                        bins=bins_uniform,
                        density=False,
                        return_fig=True,
                        material=getattr(cif_path, "stem", "Material"),
                    )
                    uniform_df = make_uniform_psd_dataframe(bins_uniform, uniform_hist)

                    pocket_results = None
                    if regions_pbc is not None:
                        pocket_psds, pocket_fig = mgr.compute_psds_pocketwise_rust(
                            results_data["dgrid"],
                            regions_pbc,
                            results_data["ase_atoms"],
                            num_samples_per_region=int(num_samples_per_region),
                            bins=bins_pocket,
                            return_fig=True,
                            material=getattr(cif_path, "stem", "Material"),
                        )
                        pocket_df = make_pocket_psd_dataframe(bins_pocket, pocket_psds)
                        pocket_results = {
                            "psds": pocket_psds,
                            "fig": pocket_fig,
                            "bins": bins_pocket,
                            "df": pocket_df,
                        }
                    else:
                        st.warning("Pocket-wise PSD requires segmented regions; skipping pocket computation.")
                        pocket_results = None

                psd_entry = {
                    "uniform": {
                        "hist": uniform_hist,
                        "fig": uniform_fig,
                        "bins": bins_uniform,
                        "df": uniform_df,
                    },
                    "pocket": pocket_results,
                }
                psd_cache[effective_key] = psd_entry
                st.success("Pore size distributions computed. Scroll down to view results.")

            if psd_entry:
                uniform_data = psd_entry.get("uniform")
                if uniform_data:
                    st.markdown("### Global PSD")
                    st.plotly_chart(uniform_data["fig"], use_container_width=True)
                    uniform_csv = uniform_data["df"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download global PSD (CSV)",
                        data=uniform_csv,
                        file_name=f"mcli_{getattr(cif_path, 'stem', 'analysis')}_psd_uniform.csv",
                        mime="text/csv",
                    )
                    st.dataframe(uniform_data["df"], use_container_width=True)

                pocket_data = psd_entry.get("pocket")
                if pocket_data:
                    st.markdown("### Pocket-wise PSD")
                    st.plotly_chart(pocket_data["fig"], use_container_width=True)
                    pocket_csv = pocket_data["df"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download pocket-wise PSDs (CSV)",
                        data=pocket_csv,
                        file_name=f"mcli_{getattr(cif_path, 'stem', 'analysis')}_psd_pockets.csv",
                        mime="text/csv",
                    )
                    st.dataframe(pocket_data["df"], use_container_width=True)

    with sa_vol_tab:
        if results_data is None:
            st.info("Run the computation to enable surface area and void volume analysis.")
        else:
            st.subheader("Surface Areas & Void Volumes")

            # Add probe radius input
            probe_radius = st.number_input(
                "Probe Radius (Å)",
                value=1.2,  # Default helium radius
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                help="Probe radius for surface area and volume calculations. Default is helium radius (1.2 Å)."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Surface Area Analysis")
                if st.button("Compute Surface Areas", key="compute_sa"):
                    try:
                        # Import the surface area computation function directly
                        from mgr.compute import compute_surface_area_from_results

                        # Convert results dict to expected format for reference functions
                        # Our app uses 'dgrid' but reference functions expect 'distance_grid'
                        modified_results = results_data.copy()
                        if 'dgrid' in modified_results:
                            modified_results['distance_grid'] = modified_results['dgrid']

                        with st.spinner("Computing surface area analysis..."):
                            atoms = results_data.get('ase_atoms')
                            if atoms is None:
                                st.error("Atoms object not found in results. Please re-run the computation.")
                                sa_results = None
                            else:
                                sa_results = compute_surface_area_from_results(modified_results, atoms, probe_radius)

                        if sa_results:
                            st.session_state["sa_results"] = sa_results
                            st.success("Surface area analysis completed!")
                        else:
                            st.error("Surface area analysis failed. Please ensure mgr_src is properly installed.")
                    except ImportError:
                        st.error("Surface area computation functions not available. Please ensure mgr_src is properly installed.")
                    except Exception as e:
                        st.error(f"Error computing surface areas: {e}")

            with col2:
                st.markdown("### Void Volume Analysis")
                if st.button("Compute Void Volumes", key="compute_vol"):
                    try:
                        # Import the void volume computation function directly
                        from mgr.compute import compute_void_volume_from_results

                        # Convert results dict to expected format for reference functions
                        # Our app uses 'dgrid' but reference functions expect 'distance_grid'
                        modified_results = results_data.copy()
                        if 'dgrid' in modified_results:
                            modified_results['distance_grid'] = modified_results['dgrid']

                        with st.spinner("Computing void volume analysis..."):
                            atoms = results_data.get('ase_atoms')
                            if atoms is None:
                                st.error("Atoms object not found in results. Please re-run the computation.")
                                vol_results = None
                            else:
                                vol_results = compute_void_volume_from_results(modified_results, atoms, probe_radius)

                        if vol_results:
                            st.session_state["vol_results"] = vol_results
                            st.success("Void volume analysis completed!")
                        else:
                            st.error("Void volume analysis failed. Please ensure mgr_src is properly installed.")
                    except ImportError:
                        st.error("Void volume computation functions not available. Please ensure mgr_src is properly installed.")
                    except Exception as e:
                        st.error(f"Error computing void volumes: {e}")

            # Display results if available
            sa_results = st.session_state.get("sa_results")
            vol_results = st.session_state.get("vol_results")

            if sa_results:
                st.markdown("### Surface Area Results")

                # Global metrics
                total_sa = sa_results.get('total_surface_area', 0)
                accessible_sa = sa_results.get('accessible_surface_area', 0)
                inaccessible_sa = total_sa - accessible_sa
                accessible_fraction = sa_results.get('accessible_fraction', 0)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Surface Area", f"{total_sa:.4f} m²/g")
                col2.metric("Accessible Surface Area", f"{accessible_sa:.4f} m²/g")
                col3.metric("Inaccessible Surface Area", f"{inaccessible_sa:.4f} m²/g")
                col4.metric("Accessible Fraction", f"{accessible_fraction:.4f}")

                # Create breakdown data
                breakdown_data = []

                # Region-wise breakdown
                by_region = sa_results.get('by_region', {})
                if by_region:
                    for region_id, area in by_region.items():
                        breakdown_data.append({
                            'Type': 'Region',
                            'Category': 'Individual',
                            'Label': f'region_{region_id}',
                            'Surface Area (m²/g)': area
                        })

                # Pocket-wise breakdown
                by_pocket = sa_results.get('by_pocket', {})
                if by_pocket:
                    for pocket_id, area in by_pocket.items():
                        breakdown_data.append({
                            'Type': 'Pocket',
                            'Category': 'Complete',
                            'Label': str(pocket_id),
                            'Surface Area (m²/g)': area
                        })

                # Percolated cluster breakdown
                by_percolated_cluster = sa_results.get('by_percolated_cluster', {})
                if by_percolated_cluster:
                    for cluster_id, area in by_percolated_cluster.items():
                        breakdown_data.append({
                            'Type': 'Cluster',
                            'Category': 'Percolated',
                            'Label': str(cluster_id),
                            'Surface Area (m²/g)': area
                        })

                if breakdown_data:
                    st.markdown("#### Detailed Breakdown")
                    sa_df = pd.DataFrame(breakdown_data)

                    # Create bar chart
                    import plotly.express as px
                    fig_sa = px.bar(sa_df, x='Label', y='Surface Area (m²/g)',
                                  color='Type', barmode='group',
                                  title="Surface Area Breakdown by Category")
                    st.plotly_chart(fig_sa, use_container_width=True)

                    # Show data table
                    st.dataframe(sa_df, use_container_width=True)

                    # Download CSV
                    csv_sa = sa_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download surface area results (CSV)",
                        data=csv_sa,
                        file_name=f"mcli_{getattr(cif_path, 'stem', 'analysis')}_surface_area.csv",
                        mime="text/csv",
                    )

            if vol_results:
                st.markdown("### Void Volume Results")

                # Global metrics
                total_vol = vol_results.get('total_void_volume', 0)
                accessible_vol = vol_results.get('accessible_void_volume', 0)
                inaccessible_vol = total_vol - accessible_vol
                accessible_fraction = vol_results.get('accessible_fraction', 0)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Void Volume", f"{total_vol:.4f} cm³/g")
                col2.metric("Accessible Void Volume", f"{accessible_vol:.4f} cm³/g")
                col3.metric("Inaccessible Void Volume", f"{inaccessible_vol:.4f} cm³/g")
                col4.metric("Accessible Fraction", f"{accessible_fraction:.4f}")

                # Create breakdown data
                breakdown_data = []

                # Region-wise breakdown
                by_region = vol_results.get('by_region', {})
                if by_region:
                    for region_id, volume in by_region.items():
                        breakdown_data.append({
                            'Type': 'Region',
                            'Category': 'Individual',
                            'Label': f'region_{region_id}',
                            'Void Volume (cm³/g)': volume
                        })

                # Pocket-wise breakdown
                by_pocket = vol_results.get('by_pocket', {})
                if by_pocket:
                    for pocket_id, volume in by_pocket.items():
                        breakdown_data.append({
                            'Type': 'Pocket',
                            'Category': 'Complete',
                            'Label': str(pocket_id),
                            'Void Volume (cm³/g)': volume
                        })

                # Percolated cluster breakdown
                by_percolated_cluster = vol_results.get('by_percolated_cluster', {})
                if by_percolated_cluster:
                    for cluster_id, volume in by_percolated_cluster.items():
                        breakdown_data.append({
                            'Type': 'Cluster',
                            'Category': 'Percolated',
                            'Label': str(cluster_id),
                            'Void Volume (cm³/g)': volume
                        })

                if breakdown_data:
                    st.markdown("#### Detailed Breakdown")
                    vol_df = pd.DataFrame(breakdown_data)

                    # Create bar chart
                    import plotly.express as px
                    fig_vol = px.bar(vol_df, x='Label', y='Void Volume (cm³/g)',
                                   color='Type', barmode='group',
                                   title="Void Volume Breakdown by Category")
                    st.plotly_chart(fig_vol, use_container_width=True)

                    # Show data table
                    st.dataframe(vol_df, use_container_width=True)

                    # Download CSV
                    csv_vol = vol_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download void volume results (CSV)",
                        data=csv_vol,
                        file_name=f"mcli_{getattr(cif_path, 'stem', 'analysis')}_void_volume.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()

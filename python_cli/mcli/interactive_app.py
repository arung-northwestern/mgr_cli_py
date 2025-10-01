"""Streamlit application for interactive MOF analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st
from ase.io import read

import plotly.graph_objects as go

import do_not_commit.mgr as mgr
from do_not_commit.mgr.helper import estimate_h_from_percolation_history

if __package__ is None or __package__ == "":  # pragma: no cover - streamlit entry
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from python_cli.mcli.cli import compute_dgrid_rust, python_do_periodic_segmentation_rust
    from python_cli.mcli.plotting import show_mof, DEFAULT_LIGHTING
else:  # pragma: no cover - package import path
    from .cli import compute_dgrid_rust, python_do_periodic_segmentation_rust
    from .plotting import show_mof, DEFAULT_LIGHTING


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
    h_value = float(max(estimate_h_from_percolation_history(dgrid) - 0.1, 0.1))
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
        extra_padding=5.0,
        compactness=0,
        min_region_size=0.001,
        disable_tqdm=True,
        compute_connections=True,
        return_dgrid=True,
    )
    return python_do_periodic_segmentation_rust(**segmentation_kwargs)


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

    param_state = st.session_state.setdefault("param_cache", {})
    cache_key = f"{getattr(cif_path, 'name', 'uploaded')}::{spacing:.3f}"
    if auto_set_params and cache_key not in param_state:
        with st.spinner("Computing suggested parameters..."):
            param_state[cache_key] = _auto_parameters(atoms, spacing)

    current_params = param_state.get(cache_key)
    if current_params is None:
        current_params = {
            "dgrid": compute_dgrid_rust(atoms, spacing),
            "mask_thickness": 0.5,
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

    surface_opacity = 0.8
    surface_pbc_opacity = 0.6

    compute_clicked = st.button("Compute and Visualize")

    if compute_clicked:
        status = st.empty()
        status.markdown("<div style='padding:8px;background:#332700;color:#FFD700;text-align:center;'>⚡ COMPUTING ⚡</div>", unsafe_allow_html=True)

        try:
            results = _run_segmentation(
                atoms,
                current_params["dgrid"],
                spacing=spacing,
                mask_thickness=mask_thickness,
                h_value=h_value,
            )
            results["ase_atoms"] = atoms
        finally:
            status.markdown("<div style='padding:8px;background:#003300;color:#00FF00;text-align:center;'>✓ COMPUTE FINISHED ✓</div>", unsafe_allow_html=True)

        # Adjust options and plotting
        framework_style = "scatter" if simple_framework and not mesh_framework else "mesh"
        results["dgrid"] = results.get("dgrid", current_params["dgrid"])
        fig = show_mof(
            results,
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

        if not stylize or projection_choice == "orthographic":
            for trace in fig.data:
                if isinstance(trace, go.Mesh3d):
                    trace.update(lighting=DEFAULT_LIGHTING)

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
        with tips_column:
            st.markdown("""
            ### Tips
            - Toggle traces via the legend to focus on specific features.
            - Orthographic projection pairs best with stylize disabled.
            - Performance mode speeds up surface generation on large cells.
            """)


if __name__ == "__main__":
    main()

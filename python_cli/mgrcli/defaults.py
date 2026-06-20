"""Shared analysis defaults aligned with the Streamlit web app."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

# Segmentation — matches interactive_app _run_segmentation / HTS worker
SEGMENTATION_DEFAULTS: Dict[str, Any] = {
    "spacing": 0.5,
    "h": 0.2,
    "mask_thickness": 0.05,
    "threshold_abs": 1.5,
    "min_distance": 1.5,
    "minimum_window_separation": 1.0,
    "wall_thickness": 0.05,
    "extra_padding": 5.0,
    "min_region_size": 0.001,
    "compactness": 0,
    "watershed_backend": "rust_det",
    "skip_endchannel_pairs": True,
    "compute_connections": True,
    "return_padded_regions": False,
    "return_dgrid": True,
    "disable_tqdm": False,
}

# Distance grid — web default VDW table + optional auto-AABB for large cells
DGRID_DEFAULTS: Dict[str, Any] = {
    "dgrid_method": "rust_exact",
    "dgrid_block_size": 50000,
    "dgrid_n_neighbors": 10,
    "dgrid_coarse_factor": 2,
    "vdw_radii_set": "bondi",
    "auto_dgrid": True,
}

# Textural property defaults (web app + batch CLI — MC SA + winding accessibility)
TEXTURAL_DEFAULTS: Dict[str, Any] = {
    "probe_radius": 1.2,
    "probe_radius_psd": 1.2,
    "window_fp": True,
    "accessibility_method": "winding",
    "lattice_variant": "v2",
    "pore_dia_method": "winding",
    "sa_method": "mc",
    "vol_method": "voxel",
    "mc_n_samples": 20000,
    "mc_shell_thickness": 0.5,
    "percolation_plot_backend": "threejs",
    "pocket_typing": "none",
}

# Batch CLI HTML plotting (web UI defaults to Three.js; CLI batch stays Plotly unless set)
PLOT_DEFAULTS: Dict[str, Any] = {
    "plot_backend": "plotly",
    "plot_enable": True,
    "plot_cell": True,
    "plot_framework": True,
    "plot_framework_style": "mesh",
    "plot_framework_show_rings": True,
    "plot_framework_mesh_scale": 0.65,
    "plot_framework_bond_radius": 0.15,
    "plot_maxima": True,
    "plot_maxima_show_spheres": False,
    "plot_show_periodic": False,
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

# Legacy batch profile (pre–web parity); kept for regression tests only
LEGACY_CLI_DEFAULTS: Dict[str, Any] = {
    "mask_thickness": 0.5,
    "watershed_backend": "rust_bucket",
    "minimum_window_separation": None,
    "vdw_radii_set": "crc",
    "auto_dgrid": False,
    "dgrid_method": "rust_exact",
}


def analysis_defaults(profile: str = "web") -> Dict[str, Any]:
    """Return a fresh defaults dict for ``profile`` ('web' or 'legacy')."""
    out: Dict[str, Any] = {}
    out.update(SEGMENTATION_DEFAULTS)
    out.update(DGRID_DEFAULTS)
    out.update(TEXTURAL_DEFAULTS)
    out.update(PLOT_DEFAULTS)
    if profile == "legacy":
        out.update(LEGACY_CLI_DEFAULTS)
    return out


def build_segmentation_kwargs(
    config: Mapping[str, Any],
    dgrid,
    mat_atoms,
    spacing: float,
    *,
    disable_tqdm: Optional[bool] = None,
) -> Dict[str, Any]:
    """Keyword args for ``python_do_periodic_segmentation_rust`` from a config dict."""
    return {
        "dgrid": dgrid,
        "mat_atoms": mat_atoms,
        "mask_thickness": float(config["mask_thickness"]),
        "h": float(config["h"]),
        "min_distance": float(config["min_distance"]),
        "threshold_abs": float(config["threshold_abs"]),
        "return_padded_regions": bool(config.get("return_padded_regions", False)),
        "spacing": spacing,
        "minimum_window_separation": config.get("minimum_window_separation"),
        "wall_thickness": float(config.get("wall_thickness", 0.05)),
        "extra_padding": float(config.get("extra_padding", 5.0)),
        "compactness": float(config.get("compactness", 0)),
        "min_region_size": float(config.get("min_region_size", 0.001)),
        "disable_tqdm": config.get("disable_tqdm", False)
        if disable_tqdm is None
        else bool(disable_tqdm),
        "compute_connections": bool(config.get("compute_connections", True)),
        "return_dgrid": bool(config.get("return_dgrid", True)),
        "watershed_backend": str(config.get("watershed_backend", "rust_det")),
        "skip_endchannel_pairs": bool(config.get("skip_endchannel_pairs", True)),
    }


def apply_config_overrides(
    base: MutableMapping[str, Any],
    args,
    *,
    arg_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Merge argparse namespace fields into *base* config."""
    mapping = arg_mapping or {
        "spacing": "spacing",
        "dgrid_method": "dgrid_method",
        "dgrid_block_size": "dgrid_block_size",
        "dgrid_n_neighbors": "dgrid_n_neighbors",
        "dgrid_coarse_factor": "dgrid_coarse_factor",
        "h": "h",
        "mask_thickness": "mask_thickness",
        "threshold_abs": "threshold_abs",
        "min_distance": "min_distance",
        "minimum_window_separation": "minimum_window_separation",
        "wall_thickness": "wall_thickness",
        "extra_padding": "extra_padding",
        "min_region_size": "min_region_size",
        "vdw_radii_set": "vdw_radii_set",
        "probe_radius": "probe_radius",
        "probe_radius_psd": "probe_radius_psd",
    }
    result = dict(base)
    for arg_name, config_key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            result[config_key] = value
    return result

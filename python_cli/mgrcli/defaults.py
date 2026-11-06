"""Shared analysis defaults aligned with the Streamlit web app."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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

# PSD sampling defaults (web PSD tab / HTS worker)
PSD_DEFAULTS: Dict[str, Any] = {
    "psd_num_samples_total": 10000,
    "psd_num_samples_per_region": 1000,
    "psd_uniform_num_bins": 200,
    "psd_pocket_num_bins": 200,
    "psd_uniform_max_diameter": 50.0,
    "psd_pocket_max_diameter": 50.0,
    "psd_uniform_log_bins": False,
}

# Textural property defaults (web app + batch CLI — MC SA + winding accessibility)
TEXTURAL_DEFAULTS: Dict[str, Any] = {
    "probe_radius": 1.2,
    "probe_radius_psd": 1.2,
    "psd_probe_radius": 1.2,
    "psd_accessible_probe_radius": 1.2,
    "sa_probe_radius": 1.2,
    "vol_probe_radius": 1.2,
    "percolation_probe_radius": 1.2,
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
    "plot_framework_style": "scatter",
    "plot_framework_show_rings": False,
    "plot_framework_mesh_scale": 0.65,
    "plot_framework_bond_radius": 0.15,
    "plot_framework_bond_linewidth": 8.0,
    "plot_framework_mesh_quality": "low",
    "plot_quality": "medium",
    "plot_maxima": True,
    "plot_maxima_show_spheres": False,
    "plot_show_periodic": False,
    "plot_connections": True,
    "plot_windows": True,
    "plot_windows_marker_color": "#FFFF00",
    "plot_windows_marker_size": 10,
    "plot_window_min_radius": None,
    "plot_hide_edges_below_window": True,
    "plot_explode": 0.0,
    "plot_explode_plane": "ab",
    "plot_surface": True,
    "plot_surface_pbc": True,
    "plot_surface_isovalue": 1.0,
    "plot_surface_opacity": 1.0,
    "plot_surface_pbc_opacity": 1.0,
    "plot_surface_types_opacity": 1.0,
    "plot_surface_taubin_smooth": True,
    "plot_surface_taubin_intensity": 0.05,
    "plot_surface_screen_door": True,
    "plot_projection": "perspective",
    "plot_layout": "dark",
    "percolation_cluster_isovalue": None,
    "percolation_boundary_color": "#000000",
    "percolation_boundary_width": 6,
    "percolation_filter_graph_at_probe": False,
    "percolation_window_marker_color": "#FFFF00",
    "percolation_window_marker_size": 10,
}

# Structure repeats, opt-in rings, and on-disk segmentation cache (web sidebar equivalents)
STRUCTURE_DEFAULTS: Dict[str, Any] = {
    "supercell_nx": 1,
    "supercell_ny": 1,
    "supercell_nz": 1,
    "compute_aromatic_rings": False,
    "use_result_cache": True,
    "cache_dir": None,
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
    out.update(PSD_DEFAULTS)
    out.update(TEXTURAL_DEFAULTS)
    out.update(PLOT_DEFAULTS)
    out.update(STRUCTURE_DEFAULTS)
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


SEGMENTATION_CACHE_KEYS = (
    "spacing",
    "h",
    "mask_thickness",
    "threshold_abs",
    "min_distance",
    "minimum_window_separation",
    "wall_thickness",
    "extra_padding",
    "min_region_size",
    "compactness",
    "watershed_backend",
    "skip_endchannel_pairs",
    "compute_connections",
    "return_padded_regions",
    "return_dgrid",
    "dgrid_method",
    "dgrid_block_size",
    "dgrid_n_neighbors",
    "dgrid_coarse_factor",
    "vdw_radii_set",
    "auto_dgrid",
    "supercell_nx",
    "supercell_ny",
    "supercell_nz",
    "min_box_thickness",
)


def segmentation_cache_token(cif_path, config: Mapping[str, Any]) -> str:
    """Stable id for a CIF plus the segmentation inputs that change the grid."""
    digest = hashlib.sha256()
    digest.update(Path(cif_path).read_bytes())
    payload = {key: config.get(key) for key in SEGMENTATION_CACHE_KEYS}
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:20]

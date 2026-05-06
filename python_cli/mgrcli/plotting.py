"""Plotting utilities for MCLI results."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple
import hashlib
from threading import Lock

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import hex_to_rgb
from ase import neighborlist
from ase.data import colors as ase_colors
from ase.data import covalent_radii
from ase.geometry import cellpar_to_cell, complete_cell
from skimage.measure import marching_cubes
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import to_rgba

import sys
from pathlib import Path
# Add mgr_src to path to use reference functions directly
REPO_ROOT = Path(__file__).resolve().parents[2]
MGR_SRC_PATH = REPO_ROOT / "mgr_src"
if str(MGR_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(MGR_SRC_PATH))

# Import the reference MGR library from mgr_src/mgr
try:
    import mgr.visualize as mgr_visualize
    from mgr.analyze import update_with_covalent_bonds
except ImportError:
    # Fallback to mgr_visualize if mgr_src is not available
    import mgr_visualize.mgr.visualize as mgr_visualize
    from mgr_visualize.mgr.analyze import update_with_covalent_bonds


DEFAULT_LIGHTING = dict(
    ambient=0.3,
    diffuse=1.0,
    fresnel=0.2,
    specular=0.05,
    roughness=0.5,
)


STYLIZE_LIGHTING = dict(
    ambient=0.18,
    diffuse=1.0,
    fresnel=0.1,
    specular=1.0,
    roughness=0.05,
    facenormalsepsilon=1e-15,
    vertexnormalsepsilon=1e-15,
)

STYLIZE_LIGHTPOSITION = dict(x=10, y=10, z=30)

# Match mgr.visualize.white_layout text/legend/hover colors (readable on white).
_WHITE_THEME_TEXT = "#1f1f1f"
_WHITE_SCENE_AXIS = dict(
    zeroline=False,
    visible=False,
    color=_WHITE_THEME_TEXT,
    tickfont=dict(color=_WHITE_THEME_TEXT),
)

white_layout = dict(
    template="plotly_white",
    font=dict(color=_WHITE_THEME_TEXT),
    title=dict(font=dict(color=_WHITE_THEME_TEXT)),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    scene=dict(
        bgcolor="#ffffff",
        xaxis=dict(_WHITE_SCENE_AXIS),
        yaxis=dict(_WHITE_SCENE_AXIS),
        zaxis=dict(_WHITE_SCENE_AXIS),
        aspectmode="data",
    ),
    margin=dict(t=20, b=20, l=20, r=20),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.05,
        xanchor="center",
        x=0.5,
        groupclick="toggleitem",
        font=dict(color=_WHITE_THEME_TEXT),
    ),
    hoverlabel=dict(
        font=dict(color=_WHITE_THEME_TEXT),
        bgcolor="#ffffff",
        bordercolor="#bdbdbd",
    ),
    height=800,
)

dark_layout = dict(
    template="plotly_dark",
    paper_bgcolor="#1f1f1f",
    plot_bgcolor="#1f1f1f",
    scene=dict(
        bgcolor="#1f1f1f",
        xaxis=dict(zeroline=False, visible=False),
        yaxis_visible=False,
        zaxis_visible=False,
        aspectmode="data",
    ),
    margin=dict(t=20, b=20, l=20, r=20),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.05,
        xanchor="center",
        x=0.5,
        groupclick="toggleitem",
    ),
    height=800,
)


def make_fig_orthographic(fig: go.Figure) -> go.Figure:
    """Ensure meshes use orthographic projection with softened lighting."""

    for trace in fig.data:
        if getattr(trace, "type", "") == "mesh3d":
            trace.update(lighting=DEFAULT_LIGHTING)
    fig.update_layout(scene_camera_projection_type="orthographic")
    return fig


def _normalize_scene_traces(traces: List[Any]) -> List[Any]:
    """Coerce traces that carry mesh topology (i,j,k + x,y,z) to mesh3d.

    Some code paths produce trace dicts or JSON blobs where ``type`` is missing
    or incorrectly set to ``scatter``; Plotly then validates against ``Scatter``
    and rejects ``i``/``j``/``k``. This keeps the hosted app from failing on figure build.
    """
    if not traces:
        return traces
    out: List[Any] = []
    for tr in traces:
        d: Dict[str, Any] | None
        if hasattr(tr, "to_plotly_json"):
            pj = tr.to_plotly_json()
            d = dict(pj) if isinstance(pj, dict) else None
        elif isinstance(tr, dict):
            d = dict(tr)
        else:
            out.append(tr)
            continue

        if d is None:
            out.append(tr)
            continue

        typ = str(d.get("type") or "").lower()
        has_mesh_topo = (
            all(k in d for k in ("x", "y", "z", "i", "j", "k"))
            and d.get("i") is not None
            and d.get("j") is not None
            and d.get("k") is not None
        )
        if has_mesh_topo and typ != "mesh3d":
            d["type"] = "mesh3d"
            out.append(d)
        else:
            out.append(tr)

    return out


def _configure_mesh_lighting(trace: go.BaseTraceType, stylize: bool, orthographic: bool) -> None:
    if not isinstance(trace, go.Mesh3d):
        return
    if orthographic:
        trace.update(lighting=DEFAULT_LIGHTING)
        return
    if stylize:
        trace.update(
            lighting=STYLIZE_LIGHTING,
            lightposition=STYLIZE_LIGHTPOSITION,
            showscale=False,
        )
    else:
        trace.update(lighting=DEFAULT_LIGHTING)


def plot_cell(ase_atoms, color: str = "purple", name: str = "cell") -> go.Scatter3d:
    """Return a wireframe trace of the unit cell."""

    cell_matrix = cellpar_to_cell(ase_atoms.cell.cellpar()).T
    origin = np.zeros(3)
    a_vec, b_vec, c_vec = cell_matrix[:, 0], cell_matrix[:, 1], cell_matrix[:, 2]

    segments = [
        origin,
        a_vec,
        [None, None, None],
        origin,
        b_vec,
        [None, None, None],
        origin,
        c_vec,
        [None, None, None],
        a_vec,
        a_vec + b_vec,
        [None, None, None],
        b_vec,
        a_vec + b_vec,
        [None, None, None],
        b_vec,
        b_vec + c_vec,
        [None, None, None],
        c_vec,
        b_vec + c_vec,
        [None, None, None],
        c_vec,
        c_vec + a_vec,
        [None, None, None],
        a_vec,
        c_vec + a_vec,
        [None, None, None],
        a_vec + b_vec,
        a_vec + b_vec + c_vec,
        [None, None, None],
        c_vec + a_vec,
        a_vec + b_vec + c_vec,
        [None, None, None],
        b_vec + c_vec,
        a_vec + b_vec + c_vec,
        [None, None, None],
    ]

    segments = np.asarray(segments)

    return go.Scatter3d(
        x=segments[:, 0],
        y=segments[:, 1],
        z=segments[:, 2],
        mode="lines",
        line=dict(color=color, width=6),
        name=name,
        legendgroup="cell",
    )


def _compute_cell_matrix(ase_atoms) -> np.ndarray:
    return complete_cell(ase_atoms.get_cell()).T


def _indices_to_cartesian(indices: np.ndarray, shape: Sequence[int], ase_atoms) -> np.ndarray:
    frac = np.asarray(indices, dtype=float) / np.asarray(shape, dtype=float)
    cell = _compute_cell_matrix(ase_atoms)
    return frac @ cell.T


def plot_maxima(dgrid: np.ndarray, maxima: np.ndarray, ase_atoms, **kwargs) -> go.Scatter3d:
    """Plot local maxima as 3D scatter points, optionally with spheres."""

    maxima = np.asarray(maxima, dtype=float)
    mod_indices = np.mod(maxima, dgrid.shape)
    coords = _indices_to_cartesian(mod_indices, dgrid.shape, ase_atoms)
    mod_indices = mod_indices.astype(int)
    radii = dgrid[tuple(mod_indices.T)]

    cage_labels = [f"{r:.2f} A" for r in np.round(radii, 2)]
    hovertext = [
        f"{idx + 1}  [{ ' '.join(map(str, maxima[idx])) }]  {label}"
        for idx, label in enumerate(cage_labels)
    ]

    name = kwargs.pop("name", "local maxima")
    marker_size = kwargs.pop("marker_size", 15)
    opacity = kwargs.pop("opacity", 0.85)
    marker_color = kwargs.pop("marker_color", np.arange(1, len(maxima) + 1))
    hovertext = kwargs.pop("hovertext", hovertext)
    colorscale = kwargs.pop("marker_colorscale", "Viridis")
    show_spheres = kwargs.pop("show_spheres", False)
    sphere_opacity = kwargs.pop("sphere_opacity", 0.5)
    # Never forward mesh-only keys into Scatter3d (would break validation).
    for _mesh_key in ("i", "j", "k"):
        kwargs.pop(_mesh_key, None)

    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=marker_color,
            opacity=opacity,
            colorscale=colorscale,
            line=dict(width=2, color="black"),
        ),
        hovertext=hovertext,
        name=name,
        legendgroup="maxima",
        **kwargs,
    )

    if not show_spheres or maxima.shape[0] == 0:
        return scatter

    colorscale_value = scatter.marker.colorscale
    data_values = np.array(scatter.marker.color, dtype=float)

    rgba_colors_with_opacity: List[List[int]]

    if isinstance(colorscale_value, list) and colorscale_value and isinstance(colorscale_value[0], (list, tuple)) and isinstance(colorscale_value[0][1], str) and colorscale_value[0][1].startswith("rgb"):
        def parse_rgb(rgb_str: str) -> List[int]:
            rgb = rgb_str.strip("rgb()").split(",")
            return [int(float(x)) for x in rgb] + [int(sphere_opacity * 255)]

        if np.issubdtype(data_values.dtype, np.number) and data_values.size > 0:
            denom = max(data_values.max() - data_values.min(), 1e-9)
            indices = ((data_values - data_values.min()) / denom * (len(colorscale_value) - 1)).astype(int)
            rgba_colors_with_opacity = [parse_rgb(colorscale_value[i][1]) for i in indices]
        else:
            base_colors = [parse_rgb(color[1]) for color in colorscale_value]
            rgba_colors_with_opacity = [base_colors[i % len(base_colors)] for i in range(len(maxima))]
    else:
        if isinstance(colorscale_value, str):
            colorscale_sequence = getattr(px.colors.sequential, colorscale_value, px.colors.sequential.Viridis)
        else:
            colorscale_sequence = colorscale_value

        if colorscale_sequence and not isinstance(colorscale_sequence[0], (list, tuple)):
            colorscale_sequence = list(zip(np.linspace(0, 1, len(colorscale_sequence)), colorscale_sequence))

        denom = max(data_values.max() - data_values.min(), 1e-9)
        normalized = (data_values - data_values.min()) / denom if data_values.size else np.zeros(0)

        def interpolate_color(colorscale_seq, value):
            for i in range(len(colorscale_seq) - 1):
                low, high = colorscale_seq[i], colorscale_seq[i + 1]
                low_pos, low_color = low
                high_pos, high_color = high
                if low_pos <= value <= high_pos:
                    t = (value - low_pos) / (high_pos - low_pos) if high_pos > low_pos else 0.0
                    color1 = np.array(to_rgba(low_color))
                    color2 = np.array(to_rgba(high_color))
                    blended = (1 - t) * color1 + t * color2
                    return blended
            return np.array(to_rgba(colorscale_seq[-1][1]))

        rgba_colors = [interpolate_color(colorscale_sequence, val) for val in normalized]
        rgba_colors_with_opacity = [
            [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), int(sphere_opacity * 255)]
            for c in rgba_colors
        ]

    spheres = spheres_for_positions(
        coords,
        radii,
        theta_resolution=30,
        phi_resolution=30,
    )

    for sphere, rgba in zip(spheres, rgba_colors_with_opacity):
        color_array = np.tile(np.array(rgba, dtype=float), (sphere.n_points, 1))
        sphere.point_data["rgba"] = color_array

    mesh = plot_mesh(
        spheres,
        stylize=True,
        opacity=sphere_opacity,
        name="maxima spheres",
    )

    return scatter, mesh


def _regular_grid(region_labels: np.ndarray) -> RegularGridInterpolator:
    x = np.linspace(0, region_labels.shape[0] - 1, region_labels.shape[0])
    y = np.linspace(0, region_labels.shape[1] - 1, region_labels.shape[1])
    z = np.linspace(0, region_labels.shape[2] - 1, region_labels.shape[2])
    return RegularGridInterpolator((x, y, z), region_labels, method="nearest", bounds_error=False, fill_value=None)


def _make_region_color_map(regions: np.ndarray, opacity: float) -> Dict[int, str]:
    positive = np.unique(regions[regions > 0])
    if positive.size == 0:
        positive = np.array([1])

    palette = px.colors.qualitative.Pastel or px.colors.qualitative.Light24
    colors: List[str] = []
    for i in range(positive.size):
        base = palette[i % len(palette)]
        if base.startswith("#"):
            r, g, b = hex_to_rgb(base)
        elif base.startswith("rgb"):
            vals = base[4:-1].split(",")
            r, g, b = [int(float(v)) for v in vals]
        else:
            r = g = b = 200
        colors.append(f"rgba({r},{g},{b},{opacity})")

    return {idx: col for idx, col in zip(positive, colors)}


def plot_surface_dgrid(
    dgrid: np.ndarray,
    isovalue: float,
    ase_atoms,
    region_labels: np.ndarray | None = None,
    select: Iterable[int] | str = "all",
    opacity: float = 1.0,
    color: str = "royalblue",
    stepsize: int = 1,
    hover_label: str = "Region:",
    cscale=None,
    stylize: bool = False,
    **kwargs: Any,
) -> go.Mesh3d:
    """Plot an isosurface of the distance grid, optionally colored by regions."""

    grid = np.asarray(dgrid, dtype=float)
    if region_labels is not None and select != "all":
        select = np.atleast_1d(select)
        mask = ~np.isin(region_labels, select)
        if np.any(mask):
            grid = grid.copy()
            grid[mask] = isovalue - 10.0

    verts, faces, normals, _ = marching_cubes(grid, level=isovalue, step_size=stepsize)

    cell = _compute_cell_matrix(ase_atoms)
    frac = verts / np.asarray(grid.shape)
    cart = frac @ cell.T

    name = kwargs.pop("name", f"Isosurface {isovalue:.2f} Å")
    kwargs.setdefault("showlegend", True)

    if region_labels is None:
        # Single color surface without regions
        mesh_trace = go.Mesh3d(
            x=cart[:, 0],
            y=cart[:, 1],
            z=cart[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=opacity,
            color=color,
            name=name,
            **kwargs,
        )
    else:
        # Handle region coloring with cscale support
        interpolator = _regular_grid(region_labels)
        region_ids = interpolator(verts).astype(int)
        hovertext = [f"{hover_label} {idx}" for idx in region_ids]

        if cscale is not None:
            # Use intensity-based coloring with colorscale (like reference library)
            # Map region IDs to intensity values (0-based)
            intensity_values = region_ids - 1

            mesh_trace = go.Mesh3d(
                x=cart[:, 0],
                y=cart[:, 1],
                z=cart[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=intensity_values,
                intensitymode='vertex',
                colorscale=cscale,
                hovertext=hovertext,
                opacity=opacity,
                name=name,
                showscale=False,
                **kwargs,
            )
        else:
            # Fallback to vertexcolor approach
            color_map = _make_region_color_map(region_labels, opacity)
            vertex_colors = [color_map.get(idx, "rgba(255,255,255,{:.2f})".format(opacity)) for idx in region_ids]

            mesh_trace = go.Mesh3d(
                x=cart[:, 0],
                y=cart[:, 1],
                z=cart[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertex_colors,
                hovertext=hovertext,
                opacity=opacity,
                name=name,
                showscale=False,
                **kwargs,
            )

    # Apply stylize lighting if requested (like reference library)
    if stylize:
        mesh_trace.update(
            lighting=STYLIZE_LIGHTING,
            lightposition=STYLIZE_LIGHTPOSITION,
            showscale=False,
        )

    return mesh_trace
def spheres_for_positions(
    positions: np.ndarray,
    radii: np.ndarray,
    scale: float = 0.8,
    theta_resolution: int = 20,
    phi_resolution: int = 20,
    colors: Sequence[Sequence[float]] | None = None,
):
    """Generate pyvista spheres at given positions and radii."""

    import pyvista as pv

    spheres = []
    positions = np.asarray(positions)
    radii = np.asarray(radii)
    color_array = None if colors is None else np.asarray(list(colors), dtype=float)

    for idx, (center, radius) in enumerate(zip(positions, radii)):
        scaled_radius = float(radius) * scale
        sphere = pv.Sphere(
            radius=max(scaled_radius, 1e-6),
            center=center,
            theta_resolution=theta_resolution,
            phi_resolution=phi_resolution,
        )
        if color_array is not None:
            rgba = color_array[min(idx, len(color_array) - 1)]
            sphere.point_data["rgba"] = np.tile(rgba, (sphere.n_points, 1))
        spheres.append(sphere)
    return spheres


def cylinders_for_bonds(
    ase_atoms,
    bonds: Sequence[Tuple[int, int, np.ndarray]],
    radius: float = 0.2,
    resolution: int = 16,
    rgba: Sequence[float] = (160, 160, 160, 255),
):
    import pyvista as pv

    cylinders = []
    rgba_arr = np.array(rgba, dtype=float)
    positions = ase_atoms.get_positions()

    for i, j, shifted in bonds:
        start = positions[i]
        end = shifted
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-8:
            continue
        center = start + direction / 2.0
        cylinder = pv.Cylinder(
            center=center,
            direction=direction,
            radius=radius,
            height=length,
            resolution=resolution,
        )
        cylinder.point_data["rgba"] = np.tile(rgba_arr, (cylinder.n_points, 1))
        cylinders.append(cylinder)

    return cylinders
    return spheres


def plot_mesh(
    meshes: Sequence[Any],
    stylize: bool = False,
    opacity: float = 0.5,
    name: str = "mesh",
    **kwargs: Any,
) -> go.Mesh3d:
    """Convert a sequence of pyvista meshes into a single Plotly Mesh3d."""

    if not meshes:
        return go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], name=name)

    vertices_list = []
    faces_list = []
    colors_list = []
    offset = 0

    for mesh in meshes:
        points = np.asarray(mesh.points)
        faces = mesh.faces.reshape((-1, 4))[:, 1:] + offset
        vertices_list.append(points)
        faces_list.append(faces)
        offset += len(points)

        color_data = mesh.point_data.get("rgba")
        if color_data is not None and len(color_data) == len(points):
            colors_list.append(color_data)
        else:
            default_color = np.array([255, 0, 0, int(opacity * 255)])
            colors_list.append(np.tile(default_color, (len(points), 1)))

    vertices = np.vstack(vertices_list)
    faces = np.vstack(faces_list)
    vertex_colors = np.vstack(colors_list)

    rgba_strings = [
        f"rgba({int(c[0])},{int(c[1])},{int(c[2])},{c[3] / 255:.3f})"
        for c in vertex_colors
    ]

    kwargs.setdefault("showlegend", True)
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=rgba_strings,
        opacity=opacity,
        name=name,
        legendgroup="maxima",
        **kwargs,
    )

    if stylize:
        mesh_trace.update(
            lighting=STYLIZE_LIGHTING,
            lightposition=STYLIZE_LIGHTPOSITION,
            showscale=False,
        )

    return mesh_trace


def _bond_pairs(ase_atoms) -> List[Tuple[int, int, np.ndarray]]:
    cutoffs = neighborlist.natural_cutoffs(ase_atoms, mult=1.1)
    nl = neighborlist.NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(ase_atoms)

def _draw_framework_scatter(ase_atoms, stylize: bool = True) -> Tuple[go.Scatter3d, go.Scatter3d]:
    positions = ase_atoms.get_positions()
    numbers = ase_atoms.get_atomic_numbers()
    symbols = ase_atoms.get_chemical_symbols()

    jmol = ase_colors.jmol_colors[numbers]
    atom_colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in jmol]
    radii = covalent_radii[numbers]
    sizes = np.interp(radii, (radii.min(), radii.max()), (12, 28))
    hovertext = [f"{sym} (index {idx})" for idx, sym in enumerate(symbols)]

    atoms_trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(size=sizes, color=atom_colors, opacity=0.9),
        name="Framework atoms",
        hovertext=hovertext,
        legendgroup="framework",
        showlegend=True,
    )

    bonds = _bond_pairs(ase_atoms)
    if bonds:
        x_vals: List[float] = []
        y_vals: List[float] = []
        z_vals: List[float] = []
        hover: List[str] = []
        positions = ase_atoms.get_positions()
        for i, j, shifted in bonds:
            start = positions[i]
            end = shifted
            x_vals.extend([start[0], end[0], None])
            y_vals.extend([start[1], end[1], None])
            z_vals.extend([start[2], end[2], None])
            hover.extend([f"Bond {i}-{j}", f"Bond {i}-{j}", None])

        bond_trace = go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="lines",
            line=dict(color="lightgray", width=4 if stylize else 2),
            hoverinfo="text",
            hovertext=hover,
            name="Framework bonds",
            legendgroup="framework",
            showlegend=True,
        )
    else:
        bond_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Framework bonds", visible=False)

    return atoms_trace, bond_trace


def plot_framework_as_mesh(
    ase_atoms,
    stylize: bool = True,
    style: str = "mesh",
    show_rings: bool = True,
    kwargs_mesh_for_atoms: Dict[str, Any] | None = None,
) -> Tuple[go.BaseTraceType, ...]:
    """Return framework traces, defaulting to mesh rendering with scatter fallback."""

    if style.lower() != "mesh":
        ase_atoms_updated = _safe_update_with_covalent_bonds(ase_atoms)
        # return _draw_framework_scatter(ase_atoms, stylize=stylize)
        return mgr_visualize.draw_framework(ase_atoms_updated)

    try:
        # Use safe bond update to prevent zero-length bonds
        safe_atoms = _safe_update_with_covalent_bonds(ase_atoms)
        kw_mesh = kwargs_mesh_for_atoms or {}
        traces = mgr_visualize.plot_framework_as_mesh(
            safe_atoms,
            show_rings=show_rings,
            stylize=stylize,
            kwargs_frame_mesh=dict(
                showlegend=True,
                showscale=False,
                name="Framework",
                legendgroup="framework",
                legendgrouptitle_text="Framework",
            ),
            kwargs_mesh_for_atoms=kw_mesh,
        )
        # Ensure tuple output for consistency
        if not isinstance(traces, (list, tuple)):
            traces = (traces,)
        else:
            traces = tuple(traces)

        if not stylize:
            for trace in traces:
                if isinstance(trace, go.Mesh3d):
                    trace.update(lighting=DEFAULT_LIGHTING)
        return traces
    except Exception:
        return _draw_framework_fallback(ase_atoms, stylize=stylize, show_rings=show_rings)


def _safe_update_with_covalent_bonds(ase_atoms):
    """Safe version of update_with_covalent_bonds that removes duplicate atoms after bond creation."""
    try:
        # If covalent bonds already exist, reuse them to avoid duplicating PBC images
        if 'covalent_bonds' in ase_atoms.info and ase_atoms.info['covalent_bonds'] is not None:
            mat_updated = ase_atoms.copy()
            mat_updated.info = dict(ase_atoms.info)
            return mat_updated

        # Use standard update_with_covalent_bonds function as-is
        mat_updated = update_with_covalent_bonds(ase_atoms)
        
        # Remove duplicate atoms that might have been created for PBC images
        positions = mat_updated.get_positions()
        atoms_to_keep = list(range(len(positions)))
        duplicate_atoms = set()
        
        # Find duplicate atoms (exact same coordinates)
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) < 1e-10:
                    duplicate_atoms.add(j)  # Mark second occurrence as duplicate
        
        # Remove duplicate atoms and update bonds
        if duplicate_atoms:
            print(f"Warning: Removing {len(duplicate_atoms)} duplicate atoms (PBC images) to prevent plotting issues")
            atoms_to_keep = [i for i in range(len(positions)) if i not in duplicate_atoms]
            
            from ase import Atoms
            unique_atoms = Atoms(
                symbols=mat_updated[atoms_to_keep].get_chemical_symbols(),
                positions=mat_updated[atoms_to_keep].get_positions(),
                cell=mat_updated.get_cell(),
                pbc=mat_updated.get_pbc()
            )
            
            # Update bond indices to account for removed atoms
            if 'covalent_bonds' in mat_updated.info:
                old_to_new_indices = {old: new for new, old in enumerate(atoms_to_keep)}
                updated_bonds = []
                updated_orders = []
                
                for i, bond in enumerate(mat_updated.info['covalent_bonds']):
                    if (bond[0] in old_to_new_indices and bond[1] in old_to_new_indices):
                        new_bond = [old_to_new_indices[bond[0]], old_to_new_indices[bond[1]]]
                        updated_bonds.append(new_bond)
                        if i < len(mat_updated.info.get('covalent_bond_orders', [])):
                            updated_orders.append(mat_updated.info['covalent_bond_orders'][i])
                
                unique_atoms.info['covalent_bonds'] = updated_bonds
                unique_atoms.info['covalent_bond_orders'] = updated_orders
            
            mat_updated = unique_atoms
        
        # Filter out zero-length bonds that cause division by zero
        if 'covalent_bonds' in mat_updated.info:
            positions = mat_updated.get_positions()
            original_bonds = mat_updated.info['covalent_bonds']
            original_orders = mat_updated.info.get('covalent_bond_orders', [])
            
            # Filter bonds to remove zero-length bonds
            valid_bonds = []
            valid_orders = []
            removed_count = 0
            
            for i, bond in enumerate(original_bonds):
                if len(bond) >= 2:
                    pos1, pos2 = positions[bond[0]], positions[bond[1]]
                    distance = np.linalg.norm(pos1 - pos2)
                    if distance > 1e-10:  # Only keep bonds with non-zero length
                        valid_bonds.append(bond)
                        if i < len(original_orders):
                            valid_orders.append(original_orders[i])
                    else:
                        removed_count += 1
            
            if removed_count > 0:
                print(f"Warning: Removed {removed_count} zero-length bonds to prevent division by zero error")
            
            mat_updated.info['covalent_bonds'] = valid_bonds
            if original_orders:
                mat_updated.info['covalent_bond_orders'] = valid_orders
        
        return mat_updated
    except Exception as e:
        print(f"Warning: Bond update failed with error: {e}")
        return ase_atoms

def _draw_framework_fallback(ase_atoms, stylize: bool = True, show_rings: bool = True) -> Tuple[go.BaseTraceType, ...]:
    mat_updated = _safe_update_with_covalent_bonds(ase_atoms)

    try:
        fallback = mgr_visualize.draw_framework(
            mat_updated,
            show_aromatics=show_rings,
            atoms_opacity=0.9 if stylize else 0.7,
        )
        if isinstance(fallback, (list, tuple)):
            return tuple(fallback)
        return (fallback,)
    except Exception as e:
        print(f"Warning: draw_framework failed with error: {e}")
        print("Falling back to simple scatter plot for framework")
        return _draw_framework_scatter(mat_updated, stylize=stylize)



def _decode_connections(raw: Any) -> List[Tuple[np.ndarray, bool, str, np.ndarray, np.ndarray]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        decoded = []
        for entry in raw:
            regions, exists, ctype, centers, radii = entry
            regions = np.asarray(regions, dtype=int)
            centers = np.asarray(centers, dtype=float).reshape(-1, 3)
            radii = np.asarray(radii, dtype=float).reshape(-1)
            decoded.append((regions, bool(exists), str(ctype), centers, radii))
        return decoded
    if isinstance(raw, dict):
        regions = np.asarray(raw.get("regions"))
        exists = np.asarray(raw.get("is_periodic"))
        ctype = np.asarray(raw.get("connection_type"))
        centers = raw.get("centers")
        radii = raw.get("radii")
        decoded = []
        for idx in range(len(regions)):
            reg = np.asarray(regions[idx], dtype=int)
            conn_type = str(ctype[idx]) if ctype is not None else ""
            center_flat = np.asarray(centers[idx]) if centers is not None else np.zeros(3)
            radius_flat = np.asarray(radii[idx]) if radii is not None else np.zeros(1)
            center = center_flat.reshape(-1, 3)
            radius = radius_flat.reshape(-1)
            decoded.append((reg, bool(exists[idx]), conn_type, center, radius))
        return decoded
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        if raw.size == 1 and raw.ndim <= 1:
            return _decode_connections(raw.item())
        if raw.ndim == 2 and raw.shape[1] == 5:
            decoded = []
            for row in raw:
                regions = np.asarray(row[0], dtype=int)
                exists = bool(row[1])
                ctype = str(row[2])
                centers = np.asarray(row[3], dtype=float).reshape(-1, 3)
                radii = np.asarray(row[4], dtype=float).reshape(-1)
                decoded.append((regions, exists, ctype, centers, radii))
            return decoded
    raise TypeError("Unsupported connections payload")





def prepare_results_for_plot(result: Dict[str, Any], ase_atoms) -> Dict[str, Any]:
    """Return a plotting-ready copy of the results dictionary."""

    prepared = dict(result)
    prepared.pop("cif_path", None)
    prepared["ase_atoms"] = ase_atoms

    if "connections" in prepared:
        prepared["connections"] = _decode_connections(prepared["connections"])

    for key in ("regions", "regions_pbc", "dgrid"):
        if key in prepared:
            prepared[key] = np.asarray(prepared[key])

    for key in ("all_maxima", "unique_regions", "maxima_info"):
        if key in prepared:
            prepared[key] = np.asarray(prepared[key])

    return prepared


# Cache for framework traces to avoid recomputation
_framework_cache: Dict[str, Tuple[go.BaseTraceType, ...]] = {}
_framework_cache_lock = Lock()


def _get_framework_cache_key(mat_atoms, style: str, show_rings: bool, stylize: bool, mesh_scale: float = 0.5, bond_radius: float = 0.25) -> str:
    """Generate a unique cache key for framework traces."""
    # Hash raw bytes directly (avoid expensive bytes->hex expansion).
    hasher = hashlib.md5()
    hasher.update(mat_atoms.get_positions().tobytes())
    hasher.update(mat_atoms.get_atomic_numbers().tobytes())
    hasher.update(mat_atoms.get_cell().array.tobytes())
    hasher.update(style.lower().encode("ascii", errors="ignore"))
    hasher.update(str(bool(show_rings)).encode("ascii"))
    hasher.update(str(bool(stylize)).encode("ascii"))
    hasher.update(f"{mesh_scale:.3f}".encode("ascii"))
    hasher.update(f"{bond_radius:.3f}".encode("ascii"))
    return hasher.hexdigest()


def clear_framework_cache():
    """Clear the framework cache to free memory."""
    global _framework_cache
    with _framework_cache_lock:
        _framework_cache.clear()


def prewarm_framework_cache(mat_atoms, stylize: bool = True, show_rings: bool = True) -> None:
    """Compute and cache mesh framework traces ahead of user style switch."""
    cache_key = _get_framework_cache_key(mat_atoms, "mesh", show_rings, stylize)
    with _framework_cache_lock:
        if cache_key in _framework_cache:
            return

    traces = plot_framework_as_mesh(
        mat_atoms,
        stylize=stylize,
        style="mesh",
        show_rings=show_rings,
    )
    with _framework_cache_lock:
        if len(_framework_cache) > 10:
            oldest_key = next(iter(_framework_cache))
            del _framework_cache[oldest_key]
        _framework_cache[cache_key] = traces


def show_mof(results: Dict[str, Any], stylize: bool = True, options: Dict[str, Any] | None = None, on_step=None) -> go.Figure:
    """Build the full MGR visualization figure.

    :param on_step: Optional callable(step_num, total_steps, label) called before
                    each major trace-building stage for progress reporting.
    """
    mat_atoms = results["ase_atoms"]
    dgrid = results["dgrid"]
    regions = results["regions"]
    regions_pbc = results["regions_pbc"]
    all_maxima = results["all_maxima"]
    defaults = {
        "plot_enable": True,
        "plot_cell": True,
        "plot_framework": True,
        "plot_framework_style": "mesh",
        "plot_framework_show_rings": True,
        "plot_framework_mesh_scale": 0.5,
        "plot_framework_bond_radius": 0.25,
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
        "plot_projection": "perspective",
        "plot_layout": "dark",
    }
    if options:
        defaults.update({k: v for k, v in options.items() if v is not None})

    if not defaults.get("plot_enable", True):
        for key in ("plot_cell", "plot_framework", "plot_maxima", "plot_connections", "plot_windows", "plot_surface", "plot_surface_pbc"):
            defaults[key] = False

    projection = str(defaults.get("plot_projection", "perspective")).lower()
    orthographic = projection == "orthographic"
    if orthographic:
        stylize = False

    framework_style = defaults.get("plot_framework_style", "mesh")

    # Pre-compute which steps will run so we can show accurate progress.
    _step_labels = []
    if defaults.get("plot_framework", True):
        _step_labels.append(f"Framework ({framework_style})")
    if defaults.get("plot_cell", True):
        _step_labels.append("Unit cell")
    if defaults.get("plot_surface", True):
        _step_labels.append("Regions surface")
    if defaults.get("plot_surface_pbc", True):
        _step_labels.append("PBC surface")
    if results.get("pocket_types_detected", False) and "regions_cluster" in results:
        _step_labels.append("Pocket types")
    if defaults.get("plot_connections", True):
        _step_labels.append("Channel connections")
    if defaults.get("plot_windows", True):
        _step_labels.append("Windows")
    if defaults.get("plot_maxima", True) and len(all_maxima[: int(regions.max())]):
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

    traces: List[go.BaseTraceType] = []
    if defaults.get("plot_framework", True):
        _emit(_step_labels[_step_idx])
        import time as _time
        _t0 = _time.time()
        cache_key = _get_framework_cache_key(
            mat_atoms,
            framework_style,
            defaults.get("plot_framework_show_rings", True),
            stylize,
            mesh_scale=float(defaults.get("plot_framework_mesh_scale", 0.5)),
            bond_radius=float(defaults.get("plot_framework_bond_radius", 0.25)),
        )
        with _framework_cache_lock:
            framework_traces = _framework_cache.get(cache_key)
        _t1 = _time.time()
        if framework_traces is None:
            kwargs_mesh = {}
            if framework_style == "mesh":
                kwargs_mesh["kwargs_mesh_for_atoms"] = dict(
                    kwargs_spheres=dict(
                        scale=float(defaults.get("plot_framework_mesh_scale", 0.5)),
                    ),
                    kwargs_cylinders=dict(
                        radius=float(defaults.get("plot_framework_bond_radius", 0.25)),
                    ),
                )
            framework_traces = plot_framework_as_mesh(
                mat_atoms,
                stylize=stylize,
                style=framework_style,
                show_rings=defaults.get("plot_framework_show_rings", True),
                **kwargs_mesh,
            )
            _t2 = _time.time()
            print(f"[TIMING] plot_framework_as_mesh ({len(mat_atoms)} atoms, style={framework_style}): {_t2-_t1:.2f}s")
            with _framework_cache_lock:
                if len(_framework_cache) > 10:
                    oldest_key = next(iter(_framework_cache))
                    del _framework_cache[oldest_key]
                _framework_cache[cache_key] = framework_traces
        else:
            print(f"[TIMING] framework cache hit: {_t1-_t0:.3f}s")
        traces.extend(framework_traces)

    if defaults.get("plot_cell", True):
        _emit(_step_labels[_step_idx])
        traces.append(plot_cell(mat_atoms))

    maxima = all_maxima[: int(regions.max())]
    connections = results.get("connections", [])
    unique_regions = np.asarray(results.get("unique_regions", []), dtype=int)
    if unique_regions.size:
        connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]
    else:
        connections_trimmed = list(connections)

    if defaults.get("plot_surface", True):
        _emit(_step_labels[_step_idx])
        surface_trace = plot_surface_dgrid(
            dgrid=dgrid,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            ase_atoms=mat_atoms,
            region_labels=regions,
            opacity=float(defaults.get("plot_surface_opacity", 1.0)),
            name="Regions Surface",
        )
        traces.append(surface_trace)

    if defaults.get("plot_surface_pbc", True):
        _emit(_step_labels[_step_idx])
        surface_trace_pbc = plot_surface_dgrid(
            dgrid=dgrid,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            ase_atoms=mat_atoms,
            region_labels=regions_pbc,
            opacity=float(defaults.get("plot_surface_pbc_opacity", 0.9)),
            name="Regions Surface (PBC)",
        )
        traces.append(surface_trace_pbc)

    # Add pocket type surface if available
    if results.get("pocket_types_detected", False) and "regions_cluster" in results:
        _emit(_step_labels[_step_idx])
        regions_cluster = results["regions_cluster"]
        cscale = px.colors.qualitative.Pastel
        surface_trace_cluster = plot_surface_dgrid(
            dgrid=dgrid,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            ase_atoms=mat_atoms,
            region_labels=regions_cluster - 1,
            opacity=float(defaults.get("plot_surface_types_opacity", 1.0)),
            name="Pocket Types",
            cscale=cscale,
            stylize=stylize,
            hover_label="Pocket type :"
        )
        traces.append(surface_trace_cluster)

    if defaults.get("plot_connections", True) and connections_trimmed:
        _emit(_step_labels[_step_idx])
        connections_traces = mgr_visualize.plot_connections_as_it_is(connections_trimmed, all_maxima, dgrid.shape, mat_atoms)
        traces.extend(connections_traces)

    # Windows for internal connections only (same reference as connections: unique_regions via connections_trimmed)
    if defaults.get("plot_windows", True) and connections_trimmed:
        _emit(_step_labels[_step_idx])
        internal_for_windows = [
            c for c in connections_trimmed
            if c[1] and c[2] == "internal" and len(c) > 3 and c[3] is not None and np.asarray(c[3]).size >= 3
        ]
        if internal_for_windows:
            window_trace = mgr_visualize.plot_windows(
                internal_for_windows,
                name="windows (internal)",
                marker_color="red",
                marker_size=10,
                opacity=0.6,
            )
            traces.append(window_trace)

    if defaults.get("plot_maxima", True) and len(maxima):
        _emit(_step_labels[_step_idx])
        maxima_trace = plot_maxima(
            dgrid,
            maxima,
            mat_atoms,
            marker_size=14,
            show_spheres=defaults.get("plot_maxima_show_spheres", False),
        )
        if isinstance(maxima_trace, tuple):
            traces.extend(maxima_trace)
        else:
            traces.append(maxima_trace)

    _plot_layout = str(defaults.get("plot_layout", "dark")).lower().strip()
    base_layout_dict = dict(white_layout if _plot_layout == "white" else dark_layout)
    traces = _normalize_scene_traces(traces)
    fig = go.Figure(data=traces, skip_invalid=True) if traces else go.Figure()
    fig.update_layout(**base_layout_dict)

    for trace in fig.data:
        _configure_mesh_lighting(trace, stylize=stylize, orthographic=orthographic)

    if orthographic:
        fig = make_fig_orthographic(fig)
    fig.update_layout(title="MGR Analysis")
    return fig


__all__ = [
    "dark_layout",
    "white_layout",
    "make_fig_orthographic",
    "plot_cell",
    "plot_surface_dgrid",
    "plot_maxima",
    "plot_framework_as_mesh",
    "spheres_for_positions",
    "plot_mesh",
    "prepare_results_for_plot",
    "prewarm_framework_cache",
    "show_mof",
]

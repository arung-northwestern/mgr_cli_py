"""Plotting utilities for MCLI results."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ase import neighborlist
from ase.data import colors as ase_colors
from ase.data import covalent_radii
from ase.geometry import cellpar_to_cell, complete_cell
from skimage.measure import marching_cubes
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import to_rgba



white_layout = dict(
    template="plotly_white",
    scene=dict(
        xaxis=dict(zeroline=False, visible=False),
        yaxis_visible=False,
        zaxis_visible=False,
        aspectmode="data",
    ),
    margin=dict(t=20, b=20, l=20, r=20),
    showlegend=True,
    legend=dict(x=1.1, y=0.75, groupclick="togglegroup"),
)

dark_layout = dict(
    template="plotly_dark",
    scene=dict(
        xaxis=dict(zeroline=False, visible=False),
        yaxis_visible=False,
        zaxis_visible=False,
        aspectmode="data",
    ),
    margin=dict(t=20, b=20, l=20, r=20),
    showlegend=True,
    legend=dict(x=0.1, y=1.1, groupclick="togglegroup"),
)


def make_fig_orthographic(fig: go.Figure) -> go.Figure:
    """Ensure meshes use orthographic projection with softened lighting."""

    for trace in fig.data:
        if getattr(trace, "type", "") == "mesh3d":
            trace.update(lighting=dict(ambient=0.3, diffuse=1.0, specular=0.1, roughness=0.5))
    fig.update_layout(scene_camera_projection_type="orthographic")
    return fig


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
    colors = px.colors.sample_colorscale("Turbo", np.linspace(0, 1, positive.size))

    def with_opacity(rgb: str) -> str:
        if rgb.startswith("rgb("):
            vals = rgb[4:-1].split(",")
            r, g, b = [int(float(v)) for v in vals]
            return f"rgba({r},{g},{b},{opacity})"
        return rgb

    return {idx: with_opacity(col) for idx, col in zip(positive, colors)}


def plot_surface_dgrid(
    dgrid: np.ndarray,
    isovalue: float,
    ase_atoms,
    region_labels: np.ndarray | None = None,
    select: Iterable[int] | str = "all",
    opacity: float = 0.5,
    color: str = "royalblue",
    stepsize: int = 1,
    hover_label: str = "Region:",
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
    kwargs.setdefault("legendgroup", "surface")

    if region_labels is None:
        return go.Mesh3d(
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

    interpolator = _regular_grid(region_labels)
    region_ids = interpolator(verts).astype(int)
    color_map = _make_region_color_map(region_labels, opacity)
    vertex_colors = [color_map.get(idx, "rgba(255,255,255,{:.2f})".format(opacity)) for idx in region_ids]
    hovertext = [f"{hover_label} {idx}" for idx in region_ids]

    return go.Mesh3d(
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
            lighting=dict(
                ambient=0.18,
                diffuse=1,
                specular=1,
                roughness=0.05,
            )
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


def plot_framework_as_mesh(ase_atoms, stylize: bool = True, style: str = "mesh") -> Tuple[go.BaseTraceType, ...]:
    """Return framework traces, defaulting to mesh rendering with scatter fallback."""

    if style.lower() != "mesh":
        return _draw_framework_scatter(ase_atoms, stylize=stylize)

    try:
        numbers = ase_atoms.get_atomic_numbers()
        positions = ase_atoms.get_positions()
        radii = covalent_radii[numbers]
        jmol = ase_colors.jmol_colors[numbers]
        rgba_colors = [[int(r * 255), int(g * 255), int(b * 255), 255] for r, g, b in jmol]

        spheres = spheres_for_positions(
            positions,
            radii,
            scale=0.8,
            theta_resolution=24,
            phi_resolution=24,
            colors=rgba_colors,
        )

        bonds = _bond_pairs(ase_atoms)
        cylinders = cylinders_for_bonds(ase_atoms, bonds, radius=0.18, resolution=20)

        mesh_trace = plot_mesh(
            [*spheres, *cylinders],
            stylize=stylize,
            opacity=0.9,
            name="Framework mesh",
            legendgroup="framework",
            showscale=False,
        )
        return (mesh_trace,)
    except Exception:
        return _draw_framework_scatter(ase_atoms, stylize=stylize)
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
        )
    else:
        bond_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines", name="Framework bonds", visible=False)

    return atoms_trace, bond_trace


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


def plot_connections_as_it_is(
    connections: Sequence[Tuple[np.ndarray, bool, str, np.ndarray, np.ndarray]],
    maxima: np.ndarray,
    shape: Sequence[int],
    ase_atoms,
    plot_periodic: bool = True,
    plot_disconnected: bool = False,
    **kwargs: Any,
) -> List[go.Scatter3d]:
    """Render connection graph traces, honoring periodic connections."""

    connections = _decode_connections(connections)
    if not connections:
        return []

    maxima = np.asarray(maxima, dtype=float)
    cell = _compute_cell_matrix(ase_atoms)
    shape = np.asarray(shape, dtype=float)

    node_ids = sorted({int(idx) for conn in connections for idx in conn[0]})
    node_positions = {node: np.dot(cell, (maxima[node - 1] / shape)) for node in node_ids if node - 1 < len(maxima)}

    cscale = kwargs.get("cscale", px.colors.sequential.Turbo)
    if isinstance(cscale, str):
        cscale = getattr(px.colors.sequential, cscale, px.colors.sequential.Turbo)

    node_to_color = {
        node: px.colors.sample_colorscale(cscale, (node - 1) / max(len(maxima) - 1, 1))[0]
        for node in node_ids
    }

    internal_nodes: List[np.ndarray] = []
    internal_hover: List[str] = []
    node_colors: List[str] = []
    periodic_nodes = set()
    internal_nodes_set = set()

    for regions, exists, ctype, centers, radii in connections:
        if exists and ctype == "periodic":
            periodic_nodes.update(regions)
        if exists and ctype == "internal":
            internal_nodes_set.update(regions)

    exclusive_periodic = periodic_nodes - internal_nodes_set

    for node in node_ids:
        if node in exclusive_periodic or node not in node_positions:
            continue
        pos = node_positions[node]
        internal_nodes.append(pos)
        node_colors.append(node_to_color[node])
        internal_hover.append(
            f"Index: {node}<br>Type: Internal<br>Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) Å"
        )

    traces: List[go.Scatter3d] = []
    if internal_nodes:
        coords = np.vstack(internal_nodes)
        traces.append(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(size=kwargs.get("marker_size", 10), color=node_colors, line=dict(width=2, color=node_colors)),
                hovertext=internal_hover,
                hoverinfo="text",
                name="Internal Maxima",
                legendgroup="connections",
            )
        )

    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_z: List[float] = []
    edge_hover: List[str] = []

    for regions, exists, ctype, centers, radii in connections:
        if not exists or ctype != "internal":
            continue
        n1, n2 = regions
        if n1 not in node_positions or n2 not in node_positions:
            continue
        pos1, pos2 = node_positions[n1], node_positions[n2]
        edge_x.extend([pos1[0], pos2[0], None])
        edge_y.extend([pos1[1], pos2[1], None])
        edge_z.extend([pos1[2], pos2[2], None])
        edge_hover.extend([f"Internal edge: {n1}-{n2}", f"Internal edge: {n1}-{n2}", None])

    if edge_x:
        traces.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(width=4, color=kwargs.get("line_color", "red")),
                hovertext=edge_hover,
                hoverinfo="text",
                name="Internal Connections",
                legendgroup="connections",
            )
        )

    if plot_periodic:
        dotted_x: List[float] = []
        dotted_y: List[float] = []
        dotted_z: List[float] = []
        periodic_markers: List[np.ndarray] = []
        periodic_hover: List[str] = []
        periodic_colors: List[str] = []

        for regions, exists, ctype, centers, radii in connections:
            if not exists or ctype != "periodic":
                continue
            n1, n2 = regions
            if n1 not in node_positions:
                continue
            start = node_positions[n1]
            for center in centers:
                dotted_points = np.linspace(start, center, kwargs.get("edge_resolution", 20))
                dotted_x.extend(np.concatenate([dotted_points[:, 0], [None]]))
                dotted_y.extend(np.concatenate([dotted_points[:, 1], [None]]))
                dotted_z.extend(np.concatenate([dotted_points[:, 2], [None]]))
                periodic_markers.append(center)
                periodic_hover.append(
                    f"Periodic edge: {n1}-{n2}<br>Window: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) Å"
                )
                periodic_colors.append(node_to_color.get(n2, "orange"))

        if dotted_x:
            traces.append(
                go.Scatter3d(
                    x=dotted_x,
                    y=dotted_y,
                    z=dotted_z,
                    mode="markers",
                    marker=dict(size=kwargs.get("dot_size", 1.0), color=kwargs.get("line_color", "red")),
                    hoverinfo="skip",
                    name="Periodic Connections",
                    legendgroup="connections",
                )
            )

        if periodic_markers:
            pm = np.vstack(periodic_markers)
            traces.append(
                go.Scatter3d(
                    x=pm[:, 0],
                    y=pm[:, 1],
                    z=pm[:, 2],
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=kwargs.get("marker_size", 10),
                        color=periodic_colors,
                        line=dict(width=3, color=periodic_colors),
                    ),
                    hovertext=periodic_hover,
                    hoverinfo="text",
                    name="Periodic Maxima",
                    legendgroup="connections",
                )
            )

    if plot_disconnected:
        maxima_coords = np.dot(cell, (maxima / shape).T).T
        linked = set(node_ids)
        all_nodes = set(range(1, len(maxima) + 1))
        disconnected = sorted(all_nodes - linked)
        if disconnected:
            coords = maxima_coords[[idx - 1 for idx in disconnected]]
            traces.append(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(size=kwargs.get("marker_size", 10), color="gray", opacity=0.3),
                    hovertext=[f"Index: {idx}" for idx in disconnected],
                    hoverinfo="text",
                    name="Disconnected Maxima",
                    legendgroup="connections",
                )
            )

    return traces


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


def show_mof(results: Dict[str, Any], stylize: bool = True, options: Dict[str, Any] | None = None) -> go.Figure:
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
        "plot_maxima": True,
        "plot_maxima_show_spheres": False,
        "plot_connections": True,
        "plot_surface": True,
        "plot_surface_pbc": True,
        "plot_surface_isovalue": 0.75,
        "plot_surface_opacity": 0.6,
        "plot_surface_pbc_opacity": 0.4,
    }
    if options:
        defaults.update({k: v for k, v in options.items() if v is not None})

    if not defaults.get("plot_enable", True):
        for key in ("plot_cell", "plot_framework", "plot_maxima", "plot_connections", "plot_surface", "plot_surface_pbc"):
            defaults[key] = False

    traces: List[go.BaseTraceType] = []

    framework_style = defaults.get("plot_framework_style", "mesh")
    if defaults.get("plot_framework", True):
        traces.extend(plot_framework_as_mesh(mat_atoms, stylize=stylize, style=framework_style))

    if defaults.get("plot_cell", True):
        traces.append(plot_cell(mat_atoms))

    maxima = all_maxima[: int(regions.max())]
    connections = results.get("connections", [])
    unique_regions = np.asarray(results.get("unique_regions", []), dtype=int)
    if unique_regions.size:
        connections_trimmed = [c for c in connections if np.all(np.isin(c[0], unique_regions))]
    else:
        connections_trimmed = list(connections)

    if defaults.get("plot_surface", True):
        surface_trace = plot_surface_dgrid(
            dgrid=dgrid,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            ase_atoms=mat_atoms,
            region_labels=regions,
            opacity=float(defaults.get("plot_surface_opacity", 0.6)),
            name="Regions Surface",
            legendgroup="surface",
        )
        traces.append(surface_trace)

    if defaults.get("plot_surface_pbc", True):
        surface_trace_pbc = plot_surface_dgrid(
            dgrid=dgrid,
            isovalue=float(defaults.get("plot_surface_isovalue", 0.75)),
            ase_atoms=mat_atoms,
            region_labels=regions_pbc,
            opacity=float(defaults.get("plot_surface_pbc_opacity", 0.4)),
            name="Regions Surface (PBC)",
            legendgroup="surface",
        )
        traces.append(surface_trace_pbc)

    if defaults.get("plot_connections", True) and connections_trimmed:
        connections_traces = plot_connections_as_it_is(connections_trimmed, maxima, regions.shape, mat_atoms)
        traces.extend(connections_traces)

    if defaults.get("plot_maxima", True) and len(maxima):
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

    if not traces:
        fig = go.Figure(layout=dark_layout)
    else:
        fig = go.Figure(data=traces, layout=dark_layout)
    fig = make_fig_orthographic(fig)
    fig.update_layout(title="MGR Analysis")
    return fig


__all__ = [
    "dark_layout",
    "white_layout",
    "make_fig_orthographic",
    "plot_cell",
    "plot_surface_dgrid",
    "plot_connections_as_it_is",
    "plot_maxima",
    "plot_framework_as_mesh",
    "spheres_for_positions",
    "plot_mesh",
    "prepare_results_for_plot",
    "show_mof",
]

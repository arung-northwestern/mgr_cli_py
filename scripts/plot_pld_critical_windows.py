"""
PLD Critical Window Visualization — Winding-Loop Constrained

Computes PLD and visualizes critical bottleneck windows using the winding
percolation method. This is the correct implementation that constrains
window search to winding-loop edges (not all connections in the padded domain).

Comparison with Chao's approach:
  Chao uses np.isclose(radii, plr) on ALL connections in percolating regions.
  This script uses ±0.10 Å tolerance on WINDING-LOOP edges only.

  Material    PLR(Å)   Chao(exact)   Winding ±0.00  ±0.10  ±0.25
  Cu-BTC      3.274        39             30          42     42
  NU-1000    12.633         0              1           1      1
  LTA         1.744         0             24          36     36
  ZIF-20      1.342         3              3           6     18

  Chao's issues:
    1. np.isclose(radii, plr, atol=0) misses windows that don't exactly equal PLR
       due to grid interpolation — gets 0 for NU-1000 and LTA
    2. Uses direct maxima labels as region labels — misses PBC-image connections
    3. Searches ALL connections, not just winding-loop edges — includes
       non-percolating paths

  Chao improvements:
    1. Use ±0.10 Å tolerance instead of np.isclose (handles grid discretization)
    2. Map maxima→region via grid lookup with PBC wrapping
    3. Constrain search to winding-loop edges from cluster_metadata

Usage:
    python scripts/plot_pld_critical_windows.py path/to/material.cif
    python scripts/plot_pld_critical_windows.py tests/fixtures/cif/Cu-BTC.cif
"""

import sys
import time
import numpy as np

# ── Setup paths ──────────────────────────────────────────────────────────
sys.path.insert(0, "mgr_src")
sys.path.insert(0, "python_cli")

import mgr
import plotly.graph_objects as go
import plotly.express as px
from ase.io import read as ase_read


# ── Parameters ───────────────────────────────────────────────────────────
SPACING = 0.5
THRESHOLD_ABS = 1.0
MIN_DISTANCE = 1.0
EXTRA_PADDING = 5
PLD_TOLERANCE = 0.10  # ±0.10 Å — sweet spot from benchmarking
CS = px.colors.qualitative.Pastel


def compute_pld_analysis(cif_path: str) -> dict:
    """Run full segmentation + PLD analysis, return everything needed for plotting."""
    t0 = time.time()
    atoms = ase_read(cif_path)
    print(f"Loaded {cif_path}: {len(atoms)} atoms")

    # Distance grid
    dgrid = mgr.dgrid_from_atoms_rust(atoms, SPACING)
    print(f"  dgrid: shape={dgrid.shape}, max={dgrid.max():.3f}")

    # Segmentation (computes maxima, regions, connections in one call)
    results = mgr.do_periodic_segmentation_rust(
        dgrid, atoms, spacing=SPACING, return_dgrid=True,
        threshold_abs=THRESHOLD_ABS, min_distance=MIN_DISTANCE,
        extra_padding=EXTRA_PADDING,
    )

    # Build directed pore graph for winding percolation
    graph = mgr.make_directed_graph_from_segmentation_results(results)
    base_labels = mgr.build_base_region_labels_from_results(results)

    # Find PLR via binary search on winding percolation
    plr = mgr.find_pore_limiting_radius_winding(
        results, graph=graph, base_region_labels=base_labels, include_dangling=True
    )
    if plr is None:
        raise ValueError("No percolating path found — material is non-porous at this probe size.")

    pld = plr * 2.0

    # Get percolating clusters at PLR with full metadata (includes winding-loop edges)
    axis_clusters, _, cluster_metadata = mgr.percolation_all_axes_winding(
        results, threshold=plr, graph=graph, base_region_labels=base_labels,
        include_dangling=True,
    )

    # Collect all percolating regions
    perc_regions = set()
    for axis, groups in axis_clusters.items():
        for g in groups:
            perc_regions.update(g)

    # Collect winding-loop edges
    perc_edges = set()
    for cid, meta in cluster_metadata.items():
        if not meta.get("axes"):
            continue
        perc_edges.update(meta["edges"])

    # Percolation history (for cluster mesh plotting)
    percolation_history, _, _ = mgr.winding_percolation.compute_percolation_history_winding(
        results, step=0.05, expand_components=False,
    )

    elapsed = time.time() - t0
    print(f"  PLR={plr:.4f} Å, PLD={pld:.4f} Å")
    print(f"  Percolating regions: {len(perc_regions)}")
    print(f"  Winding-loop edges: {len(perc_edges)}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "atoms": atoms,
        "dgrid": dgrid,
        "results": results,
        "graph": graph,
        "base_labels": base_labels,
        "plr": plr,
        "pld": pld,
        "perc_regions": perc_regions,
        "perc_edges": perc_edges,
        "cluster_metadata": cluster_metadata,
        "axis_clusters": axis_clusters,
        "percolation_history": percolation_history,
    }


def find_pld_critical_windows_from_winding(data: dict, tolerance: float = PLD_TOLERANCE):
    """Find PLD-critical windows from winding-loop edges only.

    This is the correct approach: only edges on the winding percolation path
    are relevant, not all connections in the padded domain.
    """
    graph = data["graph"]
    plr = data["plr"]
    perc_edges = data["perc_edges"]
    base_labels = data["base_labels"]
    results = data["results"]

    pld_windows = []   # (coord, radius, edge_info)
    other_windows = [] # windows in winding loops but not at PLR

    for u, v, key in perc_edges:
        if not graph.has_edge(u, v, key):
            continue
        edge_data = graph.edges[u, v, key]

        # Get window positions and radii from edge
        window_coords = edge_data.get("window_coords")
        window_radii = edge_data.get("window_radii")
        window_radius = edge_data.get("window_radius")

        # Also try to get from connections data for Cartesian coordinates
        region_u = base_labels.get(u, set())
        region_v = base_labels.get(v, set())

        if window_radii is not None:
            radii_list = list(window_radii) if hasattr(window_radii, '__iter__') else [float(window_radii)]
        elif window_radius is not None:
            radii_list = [float(window_radius)]
        else:
            continue

        # Get Cartesian coordinates from connections
        connections = results.get("connections", [])
        regions = results.get("regions_pbc", results.get("regions", np.array([])))

        for r_val in radii_list:
            r_val = float(r_val)
            delta = abs(r_val - plr)
            edge_info = {
                "radius": r_val,
                "delta": delta,
                "nodes": (u, v),
                "regions": (region_u, region_v),
            }
            if delta <= tolerance:
                pld_windows.append(edge_info)
            else:
                other_windows.append(edge_info)

    return pld_windows, other_windows


def build_pld_figure(data: dict) -> go.Figure:
    """Build the complete PLD & Percolation figure matching the interactive app."""
    atoms = data["atoms"]
    dgrid = data["dgrid"]
    plr = data["plr"]
    pld = data["pld"]
    results = data["results"]
    perc_regions = data["perc_regions"]
    percolation_history = data["percolation_history"]

    connections = results.get("connections", [])
    regions = results.get("regions_pbc", results.get("regions", np.array([])))
    maxima_info = results.get("maxima_info", np.array([]))
    all_maxima = results.get("all_maxima", np.array([]))
    maxima_raw = all_maxima[:regions.max()] if len(all_maxima) > 0 else np.array([])

    traces = []

    # 1. Unit cell wireframe
    traces.append(mgr.plot_cell(atoms))

    # 2. Framework as mesh
    frame_traces = mgr.plot_framework_as_mesh(atoms, stylize=True)
    if isinstance(frame_traces, (list, tuple)):
        traces.extend(frame_traces)
    else:
        traces.append(frame_traces)

    # 3. Percolated cluster meshes (isosurfaces at probe radius)
    #    Use last entry from percolation_history (closest to PLR that still percolates)
    try:
        probe_for_mesh = percolation_history[-1][0] if percolation_history else plr
        cluster_traces = mgr.plot_percolated_cluster_meshes(
            percolation_history,
            dgrid,
            regions,
            atoms,
            probe_radius=probe_for_mesh,
            isovalue=max(probe_for_mesh - 0.25, 0.05),
            stepsize=2,
            stylize=True,
            opacity=1.0,
            cscale=CS,
        )
        if isinstance(cluster_traces, (list, tuple)):
            traces.extend(cluster_traces)
        else:
            traces.append(cluster_traces)
    except Exception as e:
        print(f"  Warning: cluster mesh failed: {e}")

    # 4. Pore graph — internal connections only (RAG edges as lines)
    try:
        # Filter connections to unique regions only, matching RAG node count
        unique_regions_arr = np.unique(regions)
        unique_regions_arr = unique_regions_arr[unique_regions_arr > 0]
        connections_filtered = [
            c for c in connections
            if np.all(np.isin(np.asarray(c[0]), unique_regions_arr))
        ]
        rag = mgr.rag_from_connections(connections_filtered, maxima_raw)
        maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)][:regions.max()] if maxima_info.ndim == 2 and len(maxima_info) > 0 else np.array([])
        rag = mgr.add_maxima_to_rag(rag, maxima_raw, maxima_radii, regions.shape, atoms)

        rag_trace = mgr.plot_rag(rag, name="Pore Graph")
        if isinstance(rag_trace, (list, tuple)):
            traces.extend(rag_trace)
        else:
            traces.append(rag_trace)
    except Exception as e:
        print(f"  Warning: pore graph failed: {e}")

    # 5. Internal maxima (cage centers)
    if maxima_raw.size:
        maxima_trace = mgr.plot_maxima(
            dgrid, maxima_raw, atoms,
            show_spheres=False,
            maxima_info=maxima_info,
        )
        if isinstance(maxima_trace, tuple):
            traces.extend(maxima_trace)
        else:
            traces.append(maxima_trace)

    # 6. Percolation windows — use the plot_percolation_windows function
    #    which handles maxima→region mapping and tolerance correctly
    analysis_dict = {
        "percolation_history": percolation_history,
        "connections": connections,
        "regions": regions,
        "dgrid": dgrid,
        "pld": pld,
        "maxima_info": maxima_info,
    }

    window_traces = mgr.plot_percolation_windows(
        analysis_dict,
        probe_radius=plr,
        pld_tolerance=PLD_TOLERANCE,
    )
    if isinstance(window_traces, (list, tuple)):
        traces.extend(window_traces)
    elif window_traces:
        traces.append(window_traces)

    # Assemble figure — filter out empty trace lists
    flat_traces = []
    for t in traces:
        if isinstance(t, (list, tuple)):
            flat_traces.extend(t)
        elif t is not None:
            flat_traces.append(t)
    fig = go.Figure(data=flat_traces, layout=mgr.epic_layout)
    fig.update_layout(
        title=f"PLD = {pld:.2f} Å  (PLR = {plr:.2f} Å)  —  {PLD_TOLERANCE} Å tolerance, winding-loop constrained",
        scene=dict(
            aspectmode="data",
        ),
    )

    return fig


def print_comparison_table(data: dict):
    """Print comparison: Chao vs winding-loop approach."""
    plr = data["plr"]
    pld = data["pld"]
    graph = data["graph"]
    perc_edges = data["perc_edges"]
    perc_regions = data["perc_regions"]
    results = data["results"]
    connections = results.get("connections", [])

    print(f"\n{'='*70}")
    print(f"  PLD Window Comparison for {data['atoms'].get_chemical_formula()}")
    print(f"  PLR = {plr:.4f} Å,  PLD = {pld:.4f} Å")
    print(f"{'='*70}")

    # ── Chao's approach: exact PLR on all connections in perc_regions ──
    chao_count = 0
    chao_radii = []
    for c in connections:
        i_label, j_label = c[0]
        if i_label in perc_regions and j_label in perc_regions:
            if c[4] is not None and c[3] is not None:
                radii = np.asarray(c[4])
                indices = np.where(np.isclose(radii, plr))[0]
                for idx in indices:
                    chao_count += 1
                    chao_radii.append(float(radii[idx]))

    # ── Winding-loop approach ──
    winding_radii = []
    for u, v, key in perc_edges:
        if not graph.has_edge(u, v, key):
            continue
        edge_data = graph.edges[u, v, key]
        wr = edge_data.get("window_radii")
        wr_s = edge_data.get("window_radius")
        if wr is not None:
            winding_radii.extend(float(r) for r in wr)
        elif wr_s is not None:
            winding_radii.append(float(wr_s))

    winding_radii = np.array(winding_radii) if winding_radii else np.array([])

    print(f"\n  Chao (exact PLR, all connections in perc regions):")
    print(f"    Windows: {chao_count}")
    if chao_radii:
        print(f"    Radii: {np.unique(np.round(chao_radii, 4))}")

    print(f"\n  Winding-loop edges (±tolerance):")
    if len(winding_radii) > 0:
        deltas = np.abs(winding_radii - plr)
        for tol in [0.00, 0.05, 0.10, 0.25]:
            n = int(np.sum(deltas <= tol))
            in_band = winding_radii[deltas <= tol]
            unique = np.unique(np.round(in_band, 4)) if len(in_band) > 0 else []
            print(f"    ±{tol:.2f}: {n} windows  radii={unique}")

    print(f"\n  Issues with Chao's approach:")
    if chao_count == 0:
        print(f"    ❌ Finds ZERO windows — np.isclose too strict for this material")
    elif chao_count < int(np.sum(np.abs(winding_radii - plr) <= 0.10)):
        print(f"    ⚠  Misses windows near PLR due to exact-match requirement")
    else:
        print(f"    ✅ Count matches, but uses all connections (not winding-loop constrained)")

    print(f"  Recommended: ±0.10 Å on winding-loop edges")


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_pld_critical_windows.py <path/to/material.cif>")
        print("Example: python scripts/plot_pld_critical_windows.py tests/fixtures/cif/Cu-BTC.cif")
        sys.exit(1)

    cif_path = sys.argv[1]
    print(f"\n{'#'*70}")
    print(f"# PLD Critical Windows — Winding Percolation Analysis")
    print(f"# File: {cif_path}")
    print(f"# Tolerance: ±{PLD_TOLERANCE} Å")
    print(f"{'#'*70}\n")

    data = compute_pld_analysis(cif_path)
    print_comparison_table(data)

    print(f"\nBuilding figure...")
    fig = build_pld_figure(data)
    fig.show()
    print("Done!")

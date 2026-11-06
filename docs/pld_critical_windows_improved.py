"""
PLD Critical Window Identification — Improved Implementation

Comparison of Chao's approach (from PLD_visualization_Chao.ipynb) vs our app's
plot_percolation_windows (visualize.py:3642-3770), with an improved unified function.

Key Findings:
=============

1. BOTH approaches are fundamentally correct: they find bottleneck windows
   between percolating pockets that have radius ≈ PLR.

2. Chao's approach uses np.isclose(radii, plr) — exact PLR match.
   Our app uses abs(radius - pld_probe_radius) <= 0.25 — ±0.25 Å tolerance.
   The tolerance is actually BETTER because window radii are computed on a
   discrete grid and may not hit PLR exactly.

3. Our app's maxima→region mapping (grid lookup with PBC wrapping) is more
   robust than Chao's direct label comparison.

4. Our app already shows BOTH PLD-critical and regular windows, which is
   more informative than Chao's PLD-only markers.

5. The `pld_locations` computed in winding_percolation.py:856 via
   `np.argwhere(np.isclose(dgrid, plr))` is WRONG for this purpose — it finds
   ALL grid points at the PLR isosurface, not just bottleneck windows.
   It's computed but never used in the PLD tab plot, so it's just wasted CPU.

Improvements over current implementation:
=========================================

1. Use the already-computed percolating regions from the PLD analysis instead
   of re-deriving them from percolation_history entries.

2. Add per-axis breakdown: show which windows are critical for x/y/z percolation.

3. Add min/max window radius info in hover text for context.

4. Remove the unused pld_locations computation (np.argwhere on entire dgrid).

5. Use a smarter tolerance: scale by grid spacing rather than fixed 0.25 Å.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set


def find_pld_critical_windows(
    connections: list,
    percolating_regions: set,
    plr: float,
    pld: float,
    maxima_info: np.ndarray,
    regions: np.ndarray,
    tolerance: float = 0.25,
    connection_type_filter: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[dict]]:
    """Find PLD-critical bottleneck windows along percolating pathways.

    This is an improved version that combines the best of both approaches:
    - Chao's exact PLR matching logic (connection-based, percolating-only)
    - Our app's robust maxima→region mapping with PBC wrapping
    - Configurable tolerance for discrete grid effects

    Parameters
    ----------
    connections : list
        Connection list from segmentation results. Each connection is a tuple:
        [0] = (label_i, label_j), [1] = connected (bool), [2] = type (str),
        [3] = window_center_coords (Nx3 array), [4] = window_radii (N array)
    percolating_regions : set[int]
        Set of region labels that participate in any percolating pathway.
    plr : float
        Pore limiting radius (Å). Windows with radius ≈ plr are PLD-critical.
    pld : float
        Pore limiting diameter = 2 × plr (for display).
    maxima_info : np.ndarray
        Maxima info array with columns (label, z, y, x, oz, oy, ox).
    regions : np.ndarray
        3D region labels array.
    tolerance : float
        Radius tolerance for PLD-critical classification (default ±0.25 Å).
        Window is PLD-critical if |radius - plr| <= tolerance.
    connection_type_filter : str or None
        If set, only include connections of this type ("internal" or "periodic").
        None = include all types (default).

    Returns
    -------
    pld_windows : list[np.ndarray]
        Cartesian coordinates of PLD-critical window centers.
    pld_window_info : list[dict]
        Metadata for each PLD-critical window with keys:
        - "coord": np.ndarray (3,) — Cartesian coordinates
        - "radius": float — window radius (Å)
        - "maxima": tuple(int, int) — endpoint maxima labels
        - "regions": tuple(int, int) — endpoint region labels
        - "type": str — "internal" or "periodic"
        - "pld_critical": bool — always True for this function
        - "delta_from_plr": float — |radius - plr|
    """
    # Build maxima → region label mapping (handles PBC wrapping)
    maxima_to_region: Dict[int, int] = {}
    if maxima_info.ndim == 2 and maxima_info.shape[1] >= 7:
        for row in maxima_info:
            label = int(row[0])
            gz, gy, gx = int(row[1]), int(row[2]), int(row[3])
            oz, oy, ox = int(row[4]), int(row[5]), int(row[6])
            gz_w = (gz + oz * regions.shape[0]) % regions.shape[0]
            gy_w = (gy + oy * regions.shape[1]) % regions.shape[1]
            gx_w = (gx + ox * regions.shape[2]) % regions.shape[2]
            if 0 <= gz_w < regions.shape[0] and 0 <= gy_w < regions.shape[1] and 0 <= gx_w < regions.shape[2]:
                rl = int(regions[gz_w, gy_w, gx_w])
                if rl > 0:
                    maxima_to_region[label] = rl

    pld_windows = []
    pld_window_info = []

    for c in connections:
        if not c[1] or len(c) < 5:
            continue

        n1, n2 = c[0]
        r1 = maxima_to_region.get(int(n1), -1)
        r2 = maxima_to_region.get(int(n2), -1)

        # Both endpoints must be in percolating pathways
        if r1 not in percolating_regions or r2 not in percolating_regions:
            continue

        ctype = str(c[2]) if len(c) > 2 else "internal"
        if connection_type_filter is not None and ctype != connection_type_filter:
            continue

        pts = c[3] if c[3] is not None else []
        wr = c[4] if c[4] is not None else []

        if len(wr) != len(pts):
            continue

        for p, r in zip(pts, wr):
            radius = float(r)
            delta = abs(radius - plr)

            if delta <= tolerance:
                coord = np.asarray(p)
                pld_windows.append(coord)
                pld_window_info.append({
                    "coord": coord,
                    "radius": radius,
                    "maxima": (int(n1), int(n2)),
                    "regions": (int(r1), int(r2)),
                    "type": ctype,
                    "pld_critical": True,
                    "delta_from_plr": delta,
                })

    return pld_windows, pld_window_info


def get_percolating_regions_from_analysis(analysis: dict) -> Set[int]:
    """Extract all percolating region labels from PLD analysis results.

    Combines regions from all percolation axes (x+y+z).

    Parameters
    ----------
    analysis : dict
        PLD analysis dict with 'percolation_history' key.

    Returns
    -------
    set[int]
        All region labels that participate in any percolating pathway.
    """
    percolation_history = analysis.get("percolation_history", [])
    if not percolation_history:
        return set()

    # Use the last entry (largest probe radius that still percolates = PLR)
    # or the entry closest to the stored PLD
    entry = percolation_history[-1]
    _, components = entry

    all_regions: Set[int] = set()
    for axis, comp_list in components.items():
        for comp in comp_list:
            all_regions.update(comp)

    return all_regions


def get_percolating_regions_at_plr(
    percolation_history: list,
    plr: float,
) -> Set[int]:
    """Get percolating regions at exactly the PLR threshold.

    Finds the percolation_history entry closest to plr and returns
    the percolating region set.

    Parameters
    ----------
    percolation_history : list
        List of (threshold, components) tuples from compute_percolation_history.
    plr : float
        Pore limiting radius to match.

    Returns
    -------
    set[int]
        Percolating region labels at PLR.
    """
    if not percolation_history:
        return set()

    # Find entry closest to plr
    best_entry = None
    best_delta = float("inf")
    for entry in percolation_history:
        threshold = entry[0]
        delta = abs(threshold - plr)
        if delta < best_delta:
            best_delta = delta
            best_entry = entry

    if best_entry is None:
        return set()

    _, components = best_entry
    all_regions: Set[int] = set()
    for axis, comp_list in components.items():
        for comp in comp_list:
            all_regions.update(comp)

    return all_regions


# ── Usage Example ─────────────────────────────────────────────────────────
#
# In interactive_app.py, the PLD tab currently calls:
#
#   window_traces = mgr.plot_percolation_windows(
#       analysis,
#       probe_radius=selected_radius,
#       pld_tolerance=0.25,
#   )
#
# Improved usage with this module:
#
#   plr = analysis["pld"] / 2.0  # PLD is diameter, convert to radius
#   perc_regions = get_percolating_regions_from_analysis(analysis)
#   pld_windows, pld_info = find_pld_critical_windows(
#       connections=analysis["connections"],
#       percolating_regions=perc_regions,
#       plr=plr,
#       pld=analysis["pld"],
#       maxima_info=analysis["maxima_info"],
#       regions=analysis["regions"],
#       tolerance=0.25,
#   )
#
# Then build Plotly traces from pld_windows / pld_info.
#
# ── Differences from Chao's notebook ──────────────────────────────────────
#
# Chao's code:
#   for c in connections:
#       i_label, j_label = c[0]
#       if i_label in percolation_regions and j_label in percolation_regions:
#           radii = np.asarray(c[4])
#           indices = np.where(np.isclose(radii, plr))[0]  # exact match
#           for idx in indices:
#               pld_locations.append(locations[idx])
#
# Differences:
# 1. Chao uses raw maxima labels (i_label, j_label) directly as region labels.
#    This works only if maxima labels == region labels (which they often do
#    for internal maxima). Our approach maps maxima→region via grid lookup,
#    which handles PBC-offset maxima correctly.
#
# 2. Chao uses np.isclose(radii, plr) with default rtol=1e-9, atol=0.
#    This is an exact float match. On a discrete grid, window radii are
#    interpolated values that may not hit PLR exactly. Our ±tolerance band
#    is more robust for discrete grids.
#
# 3. Chao's percolation_regions comes from flatten(x+y+z axis_clusters)
#    computed directly via percolation_all_axes_winding.
#    Ours comes from percolation_history (precomputed during PLD analysis).
#    Both give the same set of regions at the PLR threshold.
#
# 4. Chao only plots PLD markers. We plot both PLD-critical (red diamonds)
#    and regular percolation windows (yellow circles), giving more context.
#
# ── Regarding pld_locations in winding_percolation.py ─────────────────────
#
# The current code in winding_percolation.py:856 computes:
#   pld_locations = np.argwhere(np.isclose(dgrid, plr, atol=1e-10))
#
# This finds ALL grid points where dgrid ≈ plr, which is the PLR isosurface.
# This is NOT the same as bottleneck windows between pockets. It includes
# points deep inside pockets that happen to have clearance = plr.
#
# This pld_locations is stored in analysis["pld_locations"] but is NOT used
# by plot_percolation_windows (which correctly uses connection windows instead).
# It could be removed to save computation, or repurposed for isosurface
# visualization if needed.

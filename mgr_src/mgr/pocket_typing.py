"""Consolidated pocket-type assignment from segmentation results.

Default policy (May 2026, **shape_refined**):

1. Coarse **probe PSD d₁** bands (``tau_merge`` = 0.5 Å) — meso / micro / interstitial.
2. Within each d₁ band: **max_r** sub-bands only for large ZIF-like groups (≥8 pockets);
   small bands (NU-1000, IRMOF) stay one type (wall/corner/edge share geometry).
3. **Split** max_r sub-peaks only when annulus fingerprints disagree (ZIF ~2.74 vs ~2.79).
4. **Merge** neighboring groups in the d₁ gray zone when fingerprints agree.

Uses PBC-aware void sampling + MIC-wrapped centroids for fingerprints.

Typical call::

    from mgr.pocket_typing import assign_pocket_types_from_segmentation

    out = assign_pocket_types_from_segmentation(segmentation_results, atoms)
    results["regions_cluster"] = out.regions_cluster

``max_r`` typing bands default to a separate Bondi/CRC distance grid
(``typing_radii="bondi_crc_fallback"``); segmentation ``dgrid`` is unchanged
for PSD, windows, and reporting.  Use ``typing_radii="segmentation"`` for the
legacy single-grid behaviour.

``segmentation_results`` is the dict produced by periodic segmentation (``dgrid``,
``regions``, ``regions_pbc``, ``maxima_info``, ``connections``, ``unique_regions``,
optionally ``ase_atoms``). A CIF path is optional (unused when segmentation is
complete).

Benchmark defaults: spacing = 0.5 Å, mask thickness = 0.05 Å, band gap = 0.10 Å.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from skimage.measure import regionprops

from mgr.pocket_framework_env import (
    FP_MERGE_COSINE,
    FP_SMALL_D1_MAX_A,
    FP_SMALL_POCKET_ANGULAR_ONLY,
    FP_SMALL_POCKET_COSINE,
    PocketFingerprint,
    assign_pocket_types_by_max_probe_radius,
    compute_connectivity_fingerprints,
    compute_pocket_fingerprint,
    fingerprint_cosine_similarity,
    max_probe_radius_for_fragment_group,
    peak_cart_for_fragment_group,
)

# Defaults aligned with pocket_peak_minimal_pipeline / benchmarks
DEFAULT_SPACING_A = 0.5
DEFAULT_MASK_THICKNESS_A = 0.05
# Grid-quantized peaks can differ by 0.05 Å (e.g. ZIF 2.74 vs 2.79); gap 0.04 keeps them as separate bands.
DEFAULT_MAX_R_BAND_GAP_A = 0.04
# Merge a 1-pocket max_r band into its neighbor when peak radii are within this (Å)
DEFAULT_SINGLETON_BAND_GAP_A = 0.40
# Within one max_r band: split if peak radii span more than this (Å) — ZIF 2.74 vs 2.79
DEFAULT_MAX_R_SUBSPLIT_GAP_A = 0.045
# Cross-band fingerprint merge: allow larger max_r difference than annulus gate (Å)
DEFAULT_ADJACENT_BAND_MERGE_MAX_R_GAP_A = 0.45
# Peak-first d₁ merge (Å); matches pocket_peak_minimal_pipeline / selective hash
DEFAULT_TAU_MERGE_A = 0.5
DEFAULT_GRAY_EXTRA_A = 0.35
# Below this pocket count per d₁ band, do not max_r-split (wall/corner vs central)
MAX_R_REFINE_WITHIN_D1_MAX_POCKETS = 12
# ZIF-like bands need at least this many pockets before max_r refinement
MAX_R_REFINE_WITHIN_D1_MIN_POCKETS = 8
DEFAULT_PROBE_RADIUS_PSD_A = 1.2
DEFAULT_PSD_MAX_DIAMETER_A = 50.0
DEFAULT_PSD_BIN_WIDTH_A = 0.25
SECOND_PEAK_FRAC = 0.10
WINDOW_TRUST_MIN_A = 1.0
# Lower trust gate for connectivity-direction fingerprint: any real saddle point.
# Watershed region boundaries always sit at dgrid local minima which are small
# (0.3-0.6 A for ZIF-20), so the 1.0 A gate must be relaxed to see connections.
CONN_TRUST_MIN_A = 0.1

METHOD_MAX_PROBE_RADIUS = "max_probe_radius"
METHOD_SHAPE_REFINED = "shape_refined"
METHOD_ENV_CLUSTER = "env_cluster"
DEFAULT_METHOD = METHOD_ENV_CLUSTER

# max_r for typing bands: Bondi/CRC handbook (+ Mendeleev fill), not analysis dgrid
TYPING_RADII_BONDI_CRC = "bondi_crc_fallback"
# Legacy: max_r from the same dgrid as segmentation / PSD / windows
TYPING_RADII_SEGMENTATION = "segmentation"
DEFAULT_TYPING_RADII = TYPING_RADII_BONDI_CRC

# env_cluster: physically motivated parameters (see assign_pocket_types_env_cluster)
ENV_CLUSTER_INITIAL_BAND_GAP_A = 0.15    # initial band split: gap > this (< grid spacing)
ENV_CLUSTER_MERGE_MAX_R_GAP_A = 0.50     # max max_r gap for merge (= 1 grid spacing)
ENV_CLUSTER_MERGE_GAP_THRESHOLD_A = 0.20 # gap boundary between tight and loose cosine
# Tight: for gaps in (initial_band_gap, merge_gap_threshold] — prevents spurious merges
# Loose: for gaps > merge_gap_threshold — allows genuine same-family merges (e.g. ZIF cages)
ENV_CLUSTER_MERGE_COSINE = 0.90          # tight cosine (small-gap candidates)
ENV_CLUSTER_MERGE_COSINE_LARGE_GAP = 0.80  # loose cosine (large-gap candidates)


@dataclass
class PocketTypingResult:
    """Output of :func:`assign_pocket_types_from_segmentation`."""

    ok: bool
    error: str = ""
    method: str = DEFAULT_METHOD
    n_pockets: int = 0
    n_types: int = 0
    rep_to_type: Dict[int, int] = field(default_factory=dict)
    rep_to_band: Dict[int, int] = field(default_factory=dict)
    band_centers_a: List[float] = field(default_factory=list)
    regions_cluster: Optional[np.ndarray] = None
    pockets: List[Dict[str, Any]] = field(default_factory=list)
    fingerprints: Dict[int, PocketFingerprint] = field(default_factory=dict)
    type_summary: List[Dict[str, Any]] = field(default_factory=list)
    pbc_group_count: int = 0
    audit: Dict[str, Any] = field(default_factory=dict)


def psd_bins(
    dgrid: np.ndarray,
    *,
    max_diameter_a: float = DEFAULT_PSD_MAX_DIAMETER_A,
    bin_width_a: float = DEFAULT_PSD_BIN_WIDTH_A,
    min_bins: int = 200,
) -> np.ndarray:
    """Diameter bins for pocketwise probe-occupiable PSDs."""
    dgrid_arr = np.asarray(dgrid, dtype=float)
    dmax_radius = float(np.nanmax(dgrid_arr))
    if not np.isfinite(dmax_radius) or dmax_radius <= 0:
        dmax_radius = max_diameter_a * 0.5
    dmax_diameter = max(max_diameter_a, 2.0 * dmax_radius)
    n_bins = max(min_bins, int(np.ceil(dmax_diameter / bin_width_a)))
    return np.linspace(0.0, dmax_diameter, n_bins + 1)


def _norm_psd(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = y.sum()
    return y / s if s > 0 else y


def _edge_trusted(data: dict) -> bool:
    if data.get("path_weight") == 0:
        return False
    winfo = data.get("window_info")
    if not winfo:
        return False
    # Support both 'max_window' (legacy key) and 'window_radii' (current format from
    # add_window_info_to_rag).  'window_radii' is a 1-D array of dgrid values at saddle
    # points on the connection path; we take the maximum as the effective window radius.
    max_w = winfo.get("max_window")
    if max_w is None:
        radii = winfo.get("window_radii")
        max_w = float(np.max(radii)) if (radii is not None and len(radii) > 0) else 0.0
    return float(max_w) >= WINDOW_TRUST_MIN_A


def _edge_connected(data: dict) -> bool:
    """Any non-trivial connection; used for connectivity-direction fingerprint.

    The CONN_TRUST_MIN_A threshold is much lower than the strict gate in
    _edge_trusted because watershed region-boundary saddle points are always
    at a dgrid local minimum (typically 0.3-0.6 A in ZIF-like materials).
    We still exclude zero-radius connections that may arise from segmentation
    artefacts.
    """
    winfo = data.get("window_info")
    if not winfo:
        return False
    max_w = winfo.get("max_window")
    if max_w is None:
        radii = winfo.get("window_radii")
        max_w = float(np.max(radii)) if (radii is not None and len(radii) > 0) else 0.0
    return float(max_w) >= CONN_TRUST_MIN_A


def _build_periodic_pocket_adjacency(
    connections: Sequence,
    maxima_info: np.ndarray,
    unique_regions: Sequence[int],
    pbc_groups: Sequence[Sequence[int]],
    *,
    conn_min_a: float = CONN_TRUST_MIN_A,
    trusted_min_a: float = WINDOW_TRUST_MIN_A,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Rep-level adjacency from canonical periodic edge map (winding pore graph).

    Uses :func:`collect_periodic_edges_canonical` so all periodic images of a
    connection collapse to one base-node edge with ``window_radius_max``.  This
    matches :func:`make_directed_graph_from_segmentation_results` / PLD winding
    and avoids asymmetric neighbor counts on symmetry-equivalent pockets.
    """
    from collections import defaultdict

    from mgr.winding_percolation import (
        _maxima_pbc_data,
        collect_periodic_edges_canonical,
    )

    maxima_info = np.asarray(maxima_info)
    unique_set = {int(x) for x in unique_regions}
    _, orig_map, _ = _maxima_pbc_data(maxima_info)

    base_to_rep: Dict[int, int] = {}
    for group in pbc_groups:
        if not group or 0 in group:
            continue
        rep = min(int(r) for r in group)
        idxs = [i for i, row in enumerate(maxima_info) if int(row[0]) == rep]
        if not idxs:
            continue
        base = int(orig_map[idxs[0]])
        base_to_rep[base] = rep

    edge_map, _ = collect_periodic_edges_canonical(connections, maxima_info)
    conn_adj: Dict[int, set] = defaultdict(set)
    trusted_adj: Dict[int, set] = defaultdict(set)
    for (bi, bj, _off), data in edge_map.items():
        if bi == bj:
            continue
        wr = float(data.get("window_radius_max", 0.0))
        ri, rj = base_to_rep.get(int(bi)), base_to_rep.get(int(bj))
        if ri is None or rj is None or ri == rj:
            continue
        if wr >= conn_min_a:
            conn_adj[ri].add(rj)
            conn_adj[rj].add(ri)
        if wr >= trusted_min_a:
            trusted_adj[ri].add(rj)
            trusted_adj[rj].add(ri)

    reps = sorted(base_to_rep.values())
    pocket_adj = {r: sorted(trusted_adj.get(r, set())) for r in reps}
    pocket_adj_conn = {r: sorted(conn_adj.get(r, set())) for r in reps}
    return pocket_adj, pocket_adj_conn


def _import_mgr():
    """RAG/PSD helpers from mgr.compute and mgr.rust_fun."""
    import types

    from . import compute as _compute
    from . import rust_fun as _rust_fun

    ns = types.SimpleNamespace()
    ns.rag_from_connections = _compute.rag_from_connections
    ns.add_window_info_to_rag = _compute.add_window_info_to_rag
    ns.add_pbc_to_rag = _compute.add_pbc_to_rag
    ns.add_maxima_to_rag = _compute.add_maxima_to_rag
    ns.compute_psds_pocketwise_rust = _rust_fun.compute_psds_pocketwise_rust
    ns.compute_pocket_psds_for_segmentation = _rust_fun.compute_pocket_psds_for_segmentation
    ns.add_csd_to_rag_with_pbc = _compute.add_csd_to_rag_with_pbc
    ns.add_pocket_psd_to_rag_with_pbc = _compute.add_pocket_psd_to_rag_with_pbc
    ns.add_psd_to_rag_with_pbc = _compute.add_psd_to_rag_with_pbc
    return ns


def _import_get_pbc_groups():
    from mgr.compute import get_pbc_groups

    return get_pbc_groups


def _bonds_from_atoms(atoms) -> Tuple[List, List]:
    from mgr.analyze import update_with_covalent_bonds

    bonded = update_with_covalent_bonds(atoms)
    bonds = bonded.info.get("covalent_bonds", [])
    orders = bonded.info.get("covalent_bond_orders", [])
    return bonds, orders


def resolve_typing_vdw_radii(
    atoms,
    typing_radii: str | np.ndarray,
) -> Optional[np.ndarray]:
    """Per-atom VDW radii (Å) for typing ``max_r``, or ``None`` = use segmentation dgrid."""
    if isinstance(typing_radii, np.ndarray):
        return np.asarray(typing_radii, dtype=float)
    key = str(typing_radii).lower().replace("-", "").replace("_", "")
    if key in {"segmentation", "same", "none", "analysis"}:
        return None
    from mgr.vdw_radii import (
        alvarez_vdw_radii_angstrom,
        bondi_vdw_radii_angstrom,
        mendeleev_vdw_radii_angstrom,
        typing_vdw_radii_angstrom,
    )

    if key in {"bondicrcfallback", "bondicrc", "typing", "default"}:
        return typing_vdw_radii_angstrom(atoms)
    if key in {"bondi", "bondi1964", "crcbondi", "crc1964"}:
        return bondi_vdw_radii_angstrom(atoms)
    if key in {"crc", "mendeleev", "webapp"}:
        return mendeleev_vdw_radii_angstrom(atoms)
    if key in {"alvarez", "alv"}:
        return alvarez_vdw_radii_angstrom(atoms)
    raise ValueError(
        f"unknown typing_radii={typing_radii!r}; use "
        f"{TYPING_RADII_BONDI_CRC!r}, {TYPING_RADII_SEGMENTATION!r}, or a radii array"
    )


def compute_typing_dgrid(
    atoms,
    segmentation: Mapping[str, Any],
    typing_radii: str | np.ndarray = DEFAULT_TYPING_RADII,
) -> Optional[np.ndarray]:
    """Distance grid for typing ``max_r`` only (same spacing as segmentation)."""
    radii = resolve_typing_vdw_radii(atoms, typing_radii)
    if radii is None:
        return None
    from mgr.rust_fun import dgrid_from_atoms_rust

    spacing = float(segmentation.get("spacing", DEFAULT_SPACING_A))
    return dgrid_from_atoms_rust(
        atoms, spacing=spacing, radii=radii,
    ).astype(np.float32)


def build_pockets_from_segmentation(
    segmentation: Mapping[str, Any],
    atoms,
    *,
    probe_radius_psd: float = DEFAULT_PROBE_RADIUS_PSD_A,
    psd_bin_edges: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]], Dict[int, float], Optional[str]]:
    """Build pocket records (id, d1, neighbors) from segmentation + PSD on RAG."""
    mgr = _import_mgr()
    get_pbc_groups = _import_get_pbc_groups()

    dgrid = np.asarray(segmentation["dgrid"], dtype=float)
    regions = np.asarray(segmentation["regions"])
    regions_pbc = np.asarray(segmentation.get("regions_pbc", regions))
    maxima_info = segmentation["maxima_info"]
    all_maxima = segmentation["all_maxima"]
    connections = segmentation.get("connections", [])
    unique_regions = segmentation.get("unique_regions", [])

    if psd_bin_edges is None:
        psd_bin_edges = psd_bins(dgrid)

    pbc_groups_dict = get_pbc_groups(unique_regions, maxima_info)
    pbc_groups = [v for v in pbc_groups_dict.values() if 0 not in v]
    all_maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)]
    maxima = all_maxima[: regions.max()]
    maxima_radii = all_maxima_radii[: regions.max()]
    connections_trimmed = [
        c for c in connections if np.any(np.isin(c[0], unique_regions))
    ]

    rag = mgr.rag_from_connections(connections_trimmed, maxima)
    rag = mgr.add_window_info_to_rag(rag, connections_trimmed)
    rag = mgr.add_pbc_to_rag(rag, pbc_groups)
    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid.shape, atoms)
    props = regionprops(regions)

    pocket_psds = segmentation.get("pocket_psds")
    pocket_psd_error = segmentation.get("pocket_psd_error")
    if pocket_psds is None and pocket_psd_error is None:
        try:
            pocket_psds = mgr.compute_pocket_psds_for_segmentation(
                segmentation,
                atoms,
                num_samples_per_region=1000,
                bins=psd_bin_edges,
                probe_center_min_radius=probe_radius_psd,
            )
        except Exception as exc:
            import logging

            pocket_psd_error = str(exc)
            logging.getLogger(__name__).warning(
                "compute_pocket_psds_for_segmentation failed (%s); pocket typing needs probe PSDs on the RAG.",
                exc,
            )
            pocket_psds = None
    elif pocket_psds is None and pocket_psd_error:
        import logging

        logging.getLogger(__name__).warning(
            "segmentation has pocket_psd_error=%r; skipping pocket PSD attach.",
            pocket_psd_error,
        )

    rag = mgr.add_csd_to_rag_with_pbc(
        rag, props, pbc_groups, psd_bin_edges, atoms, dgrid.shape
    )
    if pocket_psds is not None:
        rag = mgr.add_pocket_psd_to_rag_with_pbc(rag, props, pbc_groups, pocket_psds)
        psd_source = str(rag.graph.get("pocket_psd_source", "pocketwise_rust"))
    else:
        psd_source = "missing_pocket_psd"

    # Full connection list required: trimming to unique_regions drops periodic-image
    # edges that collect_periodic_edges_canonical needs for symmetric base adjacency.
    pocket_adj, pocket_adj_conn = _build_periodic_pocket_adjacency(
        connections,
        maxima_info,
        unique_regions,
        pbc_groups,
    )

    bw = float(psd_bin_edges[1] - psd_bin_edges[0]) if len(psd_bin_edges) > 1 else 0.25
    d1_map: Dict[int, float] = {}
    pockets: List[Dict[str, Any]] = []

    for pi, group in enumerate(pbc_groups):
        rep = min(group)
        nd = rag.nodes[rep]
        x = np.asarray(nd.get("pocket_psd_x_pbc", []), dtype=float)
        y = _norm_psd(np.asarray(nd.get("pocket_psd_y_pbc", []), dtype=float))
        if y.size == 0:
            continue
        i1 = int(np.argmax(y))
        d1 = float(x[i1])
        h1 = float(y[i1])
        d1_map[rep] = d1

        d2, h2 = None, 0.0
        y_work = y.copy()
        y_work[max(0, i1 - 1) : min(len(y_work), i1 + 2)] = 0.0
        if y_work.max() >= SECOND_PEAK_FRAC * h1:
            i2 = int(np.argmax(y_work))
            d2 = float(x[i2])
            h2 = float(y_work[i2])

        nbr_d1 = sorted(
            round(d1_map.get(nb, 0.0), 2) for nb in pocket_adj.get(rep, []) if nb in d1_map
        )
        # Probe-occupiable pocket PSD (Rust) on pocket_psd_x_pbc / pocket_psd_y_pbc.
        psd_x_arr = np.asarray(nd.get("pocket_psd_x_pbc", []), dtype=float)
        psd_y_full = np.asarray(nd.get("pocket_psd_y_pbc", []), dtype=float)
        psd_norm = _norm_psd(psd_y_full)
        pockets.append(
            {
                "id": int(rep),
                "pi": pi,
                "d1_A": d1,
                "d2_A": d2,
                "h1": h1,
                "h2": h2,
                "n_trusted": len(nbr_d1),
                "nbr_d1": nbr_d1,
                "trusted_nbr_reps": sorted(pocket_adj.get(rep, [])),
                "conn_nbr_reps": sorted(pocket_adj_conn.get(rep, [])),
                "psd_x": psd_x_arr,
                "psd_vec": psd_norm,
                "psd_source": psd_source,
            }
        )

    return pockets, pbc_groups, d1_map, pocket_psd_error


def compute_pocket_fingerprints(
    segmentation: Mapping[str, Any],
    atoms,
    pockets: Sequence[Mapping[str, Any]],
    *,
    pbc_groups: Optional[Sequence[Sequence[int]]] = None,
    bonds: Optional[Sequence] = None,
    bond_orders: Optional[Sequence] = None,
    typing_dgrid: Optional[np.ndarray] = None,
) -> Dict[int, PocketFingerprint]:
    """Mouth-annulus fingerprints for all pockets (for typing and diagnostics).

    When ``typing_dgrid`` is set, ``max_probe_radius_a`` is taken from that grid
    (stable Bondi/CRC radii) while void sampling and framework SH still use the
    segmentation ``dgrid`` from the main analysis pipeline.
    """
    regions = np.asarray(segmentation["regions"])
    regions_pbc = np.asarray(segmentation.get("regions_pbc", regions))
    dgrid = np.asarray(segmentation["dgrid"], dtype=float)
    dgrid_max_r = (
        np.asarray(typing_dgrid, dtype=float)
        if typing_dgrid is not None
        else dgrid
    )
    shape = np.asarray(segmentation.get("shape", dgrid.shape))

    if bonds is None:
        bonds, _ = _bonds_from_atoms(atoms)

    if pbc_groups is None:
        get_pbc_groups = _import_get_pbc_groups()
        unique_regions = segmentation.get("unique_regions", [])
        maxima_info = segmentation.get("maxima_info")
        if maxima_info is not None and unique_regions is not None:
            pbc_groups = [
                v for v in get_pbc_groups(unique_regions, maxima_info).values() if 0 not in v
            ]
        else:
            pbc_groups = [[int(p["id"])] for p in pockets]

    rep_to_group = build_rep_to_fragment_group(pbc_groups)
    voxel_vol = float(atoms.get_volume()) / float(np.prod(dgrid.shape))
    props_frag = {p.label: p for p in regionprops(regions)}

    # Compute surface-atom SH fingerprints
    fps: Dict[int, PocketFingerprint] = {}
    pocket_centres: Dict[int, np.ndarray] = {}
    trusted_neighbors: Dict[int, List[int]] = {}
    for p in pockets:
        rep = int(p["id"])
        frag_group = rep_to_group.get(rep, [rep])
        vol = sum(
            float(props_frag[r].area * voxel_vol)
            for r in frag_group
            if r in props_frag
        )
        mr = max_probe_radius_for_fragment_group(regions, dgrid_max_r, frag_group)
        psd_vec = p.get("psd_vec")
        fps[rep] = compute_pocket_fingerprint(
            regions_pbc,
            regions,
            dgrid,
            rep,
            shape,
            atoms,
            bonds,
            r_pocket=float(p["d1_A"]),
            max_probe_radius_a=mr,
            void_volume_a3=vol,
            fragment_group=frag_group,
            void_near_peak=True,
            use_surface_atoms=True,
            psd_vec=psd_vec,
        )
        # Pocket centre (dgrid peak position) for connectivity fingerprint
        pocket_centres[rep] = peak_cart_for_fragment_group(
            regions, dgrid, frag_group, shape, atoms
        )
        # Trusted window-connected neighbours (window > WINDOW_TRUST_MIN_A)
        # Use loose connectivity (conn_nbr_reps) for direction fingerprint;
        # trusted_nbr_reps (strict 1.0 A gate) is for d1-neighbor computation.
        trusted_neighbors[rep] = list(p.get("conn_nbr_reps", p.get("trusted_nbr_reps", [])))

    # Compute connectivity-direction SH fingerprints and attach to existing FPs
    conn_fps = compute_connectivity_fingerprints(pocket_centres, trusted_neighbors, atoms)
    for rep, conn_sh in conn_fps.items():
        if rep in fps:
            fps[rep].conn_sh_power = conn_sh
            fps[rep].n_connections = len(trusted_neighbors.get(rep, []))
    return fps


def build_rep_to_fragment_group(
    pbc_groups: Sequence[Sequence[int]],
) -> Dict[int, List[int]]:
    """PBC representative (min label) → all fragment region labels in that pocket."""
    out: Dict[int, List[int]] = {}
    for group in pbc_groups:
        regs = sorted(int(r) for r in group)
        if not regs:
            continue
        out[regs[0]] = regs
    return out


def merge_singleton_max_r_bands(
    pockets: Sequence[Mapping[str, Any]],
    fps: Mapping[int, PocketFingerprint],
    rep_to_band: Mapping[int, int],
    *,
    gap_tol_a: float = DEFAULT_SINGLETON_BAND_GAP_A,
) -> Dict[int, int]:
    """Reassign isolated max_r bands to the nearest band if within ``gap_tol_a``."""
    band_to_reps: Dict[int, List[int]] = {}
    band_centers: Dict[int, float] = {}
    for p in pockets:
        rep = int(p["id"])
        bi = int(rep_to_band[rep])
        band_to_reps.setdefault(bi, []).append(rep)
        band_centers[bi] = fps[rep].max_probe_radius_a

    merged_band = dict(rep_to_band)
    for bi, reps in band_to_reps.items():
        if len(reps) != 1:
            continue
        r0 = band_centers[bi]
        best_other, best_gap = None, float("inf")
        for bj, rj in band_centers.items():
            if bj == bi:
                continue
            gap = abs(r0 - rj)
            if gap < best_gap:
                best_gap, best_other = gap, bj
        if best_other is not None and best_gap <= gap_tol_a:
            for rep in reps:
                merged_band[rep] = best_other

    # Collapse mutual singleton links (e.g. ZIF-8: two 1-pocket bands within gap_tol).
    all_bands = sorted(set(merged_band.values()) | set(rep_to_band.values()))
    parent = {b: b for b in all_bands}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rep, orig in rep_to_band.items():
        new = merged_band[int(rep)]
        if int(orig) != int(new):
            ra, rb = _find(int(orig)), _find(int(new))
            if ra != rb:
                parent[rb] = ra

    remap: Dict[int, int] = {}
    out: Dict[int, int] = {}
    for rep, bi in merged_band.items():
        root = _find(int(bi))
        if root not in remap:
            remap[root] = len(remap)
        out[int(rep)] = remap[root]
    return out


def _mean_max_r(pids: Sequence[int], fps: Mapping[int, PocketFingerprint]) -> float:
    return float(np.mean([fps[int(p)].max_probe_radius_a for p in pids]))


def _mean_d1(pids: Sequence[int], d1_by_rep: Mapping[int, float]) -> float:
    return float(np.mean([d1_by_rep[int(p)] for p in pids]))


def group_pockets_by_d1_coarse(
    pockets: Sequence[Mapping[str, Any]],
    *,
    tau_merge_a: float = DEFAULT_TAU_MERGE_A,
) -> List[List[int]]:
    """Greedy adjacent merge on probe PSD d₁ (Å); returns rep ids per coarse group."""
    if not pockets:
        return []
    sorted_pockets = sorted(pockets, key=lambda p: float(p["d1_A"]))
    groups: List[List[int]] = []
    cur_d1 = float(sorted_pockets[0]["d1_A"])
    cur: List[int] = [int(sorted_pockets[0]["id"])]
    for p in sorted_pockets[1:]:
        d1 = float(p["d1_A"])
        if d1 - cur_d1 <= tau_merge_a:
            cur.append(int(p["id"]))
        else:
            groups.append(cur)
            cur = [int(p["id"])]
            cur_d1 = d1
    groups.append(cur)
    return groups


def _within_d1_group_use_max_r_refinement(
    pocket_ids: Sequence[int],
    fps: Mapping[int, PocketFingerprint],
) -> bool:
    """True when a d₁ band is large enough for ZIF-like max_r sub-typing."""
    n = len(pocket_ids)
    if n <= 1:
        return False
    max_rs = [fps[int(p)].max_probe_radius_a for p in pocket_ids]
    spread = max(max_rs) - min(max_rs)
    if spread < DEFAULT_MAX_R_SUBSPLIT_GAP_A:
        return False
    if n <= MAX_R_REFINE_WITHIN_D1_MAX_POCKETS:
        return False
    if n >= MAX_R_REFINE_WITHIN_D1_MIN_POCKETS and max(max_rs) < 5.0:
        return True
    return n >= MAX_R_REFINE_WITHIN_D1_MIN_POCKETS and spread >= 0.10


def _refine_pocket_ids_by_max_r(
    pocket_ids: Sequence[int],
    pockets: Sequence[Mapping[str, Any]],
    fps: Mapping[int, PocketFingerprint],
    d1_by_rep: Mapping[int, float],
    *,
    band_gap_a: float,
    singleton_band_gap_a: float,
    subsplit_gap_a: float,
    merge_cosine: float,
    merge_singleton_bands: bool,
) -> List[List[int]]:
    """max_r bands → singleton merge → fingerprint-aware subsplit (subset of pockets)."""
    from collections import defaultdict

    id_set = {int(p) for p in pocket_ids}
    sub_pockets = [p for p in pockets if int(p["id"]) in id_set]
    _, _, rep_to_band = assign_pocket_types_by_max_probe_radius(
        sub_pockets, fps, band_gap_a=band_gap_a,
    )
    if merge_singleton_bands:
        rep_to_band = merge_singleton_max_r_bands(
            sub_pockets, fps, rep_to_band, gap_tol_a=singleton_band_gap_a,
        )

    band_to_pids: Dict[int, List[int]] = defaultdict(list)
    for p in sub_pockets:
        band_to_pids[int(rep_to_band[int(p["id"])])].append(int(p["id"]))

    out: List[List[int]] = []
    for bi in sorted(band_to_pids):
        out.extend(
            split_band_by_max_r_subpeaks(
                band_to_pids[bi],
                fps,
                d1_by_rep,
                subsplit_gap_a=subsplit_gap_a,
                split_cosine=merge_cosine,
            )
        )
    return out


def split_band_by_max_r_subpeaks(
    pocket_ids: Sequence[int],
    fps: Mapping[int, PocketFingerprint],
    d1_by_rep: Mapping[int, float],
    *,
    subsplit_gap_a: float = DEFAULT_MAX_R_SUBSPLIT_GAP_A,
    split_cosine: float = FP_MERGE_COSINE,
) -> List[List[int]]:
    """Split one max_r band when grid peaks differ in radius and annulus fingerprint.

  E.g. ZIF ~2.74 vs ~2.79 Å (split). UiO ~3.78 vs ~3.90 after singleton merge (keep merged
  when fingerprints still match).
    """
    from collections import defaultdict

    if len(pocket_ids) <= 1:
        return [list(pocket_ids)]
    max_rs = [fps[int(pid)].max_probe_radius_a for pid in pocket_ids]
    if max(max_rs) - min(max_rs) <= subsplit_gap_a:
        return [list(pocket_ids)]
    by_peak: Dict[float, List[int]] = defaultdict(list)
    for pid in pocket_ids:
        key = round(fps[int(pid)].max_probe_radius_a, 2)
        by_peak[key].append(int(pid))
    if len(by_peak) <= 1:
        return [list(pocket_ids)]

    peaks = sorted(by_peak.keys())
    rep_a = min(by_peak[peaks[0]], key=lambda p: int(p))
    rep_b = min(by_peak[peaks[-1]], key=lambda p: int(p))
    fa, fb = fps[int(rep_a)], fps[int(rep_b)]
    d1a, d1b = d1_by_rep[int(rep_a)], d1_by_rep[int(rep_b)]
    spread = max(max_rs) - min(max_rs)
    cos_full = fingerprint_cosine_similarity(fa, fb, angular_only=False)
    if cos_full >= split_cosine:
        return [list(pocket_ids)]
    if max(d1a, d1b) <= FP_SMALL_D1_MAX_A:
        cos_ang = fingerprint_cosine_similarity(
            fa, fb, angular_only=FP_SMALL_POCKET_ANGULAR_ONLY,
        )
        if cos_ang >= 0.94:
            return [list(pocket_ids)]
    # ZIF-like mid windows: split only when fingerprints disagree (not wall/corner max_r noise).
    if max(max_rs) < 4.0 and spread >= 0.04 and cos_full < split_cosine:
        return [sorted(v) for v in sorted(by_peak.values(), key=lambda g: _mean_max_r(g, fps))]

    return [list(pocket_ids)]


def fingerprints_similar_adjacent_bands(
    a: PocketFingerprint,
    b: PocketFingerprint,
    *,
    d1_a: float,
    d1_b: float,
    merge_cosine: float = FP_MERGE_COSINE,
    max_probe_gap_a: float = DEFAULT_ADJACENT_BAND_MERGE_MAX_R_GAP_A,
) -> bool:
    """Merge adjacent max_r bands when annulus fingerprints agree (relaxed max_r gate)."""
    if a.max_probe_radius_a > 0 and b.max_probe_radius_a > 0:
        if abs(a.max_probe_radius_a - b.max_probe_radius_a) > max_probe_gap_a:
            return False
    gap_r = abs(a.max_probe_radius_a - b.max_probe_radius_a)
    cos = fingerprint_cosine_similarity(a, b, angular_only=False)
    # Do not merge distinct mid-window grid peaks when fingerprints differ (ZIF ~2.74 vs ~2.79).
    if (
        a.max_probe_radius_a < 4.0
        and b.max_probe_radius_a < 4.0
        and gap_r >= 0.04
        and cos < merge_cosine
    ):
        return False
    thresh = merge_cosine
    # Keep distinct large-cage peaks (e.g. ZIF ~7.31 vs ~7.56 Å) unless extremely similar.
    if a.max_probe_radius_a > 6.0 and b.max_probe_radius_a > 6.0 and gap_r > 0.30:
        thresh = max(thresh, 0.97)
    return cos >= thresh


def merge_adjacent_groups_by_fingerprint(
    groups: List[List[int]],
    fps: Mapping[int, PocketFingerprint],
    d1_by_rep: Mapping[int, float],
    *,
    merge_cosine: float = FP_MERGE_COSINE,
    max_probe_gap_a: float = DEFAULT_ADJACENT_BAND_MERGE_MAX_R_GAP_A,
    tau_merge_a: float = DEFAULT_TAU_MERGE_A,
    gray_extra_a: float = DEFAULT_GRAY_EXTRA_A,
) -> Tuple[List[List[int]], int]:
    """Union-merge neighboring groups (by mean max_r) when d₁ + fingerprints allow."""
    d1_gray_limit = tau_merge_a + gray_extra_a
    if len(groups) <= 1:
        return groups, 0

    order = sorted(range(len(groups)), key=lambda i: _mean_max_r(groups[i], fps))
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merges = 0
    changed = True
    while changed:
        changed = False
        clusters: Dict[int, List[int]] = {}
        for i, grp in enumerate(groups):
            clusters.setdefault(find(i), []).append(i)
        cluster_ids = sorted(
            clusters.keys(),
            key=lambda c: _mean_max_r(
                [p for i in clusters[c] for p in groups[i]], fps,
            ),
        )
        for k in range(len(cluster_ids) - 1):
            ci, cj = cluster_ids[k], cluster_ids[k + 1]
            rep_i = min(
                (p for idx in clusters[ci] for p in groups[idx]),
                key=lambda p: fps[int(p)].max_probe_radius_a,
            )
            rep_j = min(
                (p for idx in clusters[cj] for p in groups[idx]),
                key=lambda p: fps[int(p)].max_probe_radius_a,
            )
            fa, fb = fps[int(rep_i)], fps[int(rep_j)]
            d1_gap = abs(d1_by_rep[int(rep_i)] - d1_by_rep[int(rep_j)])
            if d1_gap > d1_gray_limit:
                continue
            if fingerprints_similar_adjacent_bands(
                fa,
                fb,
                d1_a=d1_by_rep[int(rep_i)],
                d1_b=d1_by_rep[int(rep_j)],
                merge_cosine=merge_cosine,
                max_probe_gap_a=max_probe_gap_a,
            ):
                i0, j0 = clusters[ci][0], clusters[cj][0]
                if find(i0) != find(j0):
                    union(i0, j0)
                    merges += 1
                    changed = True
                    break

    merged: Dict[int, List[int]] = {}
    for i, grp in enumerate(groups):
        merged.setdefault(find(i), []).extend(int(p) for p in grp)
    out = [sorted(set(g)) for g in merged.values()]
    out.sort(key=lambda g: _mean_max_r(g, fps))
    return out, merges


def assign_pocket_types_env_cluster(
    pockets: Sequence[Mapping[str, Any]],
    fps: Mapping[int, PocketFingerprint],
    *,
    initial_band_gap_a: float = ENV_CLUSTER_INITIAL_BAND_GAP_A,
    merge_max_r_gap_a: float = ENV_CLUSTER_MERGE_MAX_R_GAP_A,
    merge_gap_threshold_a: float = ENV_CLUSTER_MERGE_GAP_THRESHOLD_A,
    merge_cosine: float = ENV_CLUSTER_MERGE_COSINE,
    merge_cosine_large_gap: float = ENV_CLUSTER_MERGE_COSINE_LARGE_GAP,
    merge_singleton_bands: bool = True,
    singleton_band_gap_a: float = DEFAULT_SINGLETON_BAND_GAP_A,
) -> Tuple[int, Dict[int, int], Dict[str, Any]]:
    """Pocket typing: max_r initial bands, fingerprint-driven merge only.

    Algorithm
    ---------
    1. Sort pockets by max_probe_radius.
    2. Start a new band whenever the gap to the next pocket exceeds
       *initial_band_gap_a* (grid-noise floor, < grid spacing).
    3. Iteratively merge adjacent bands (by mean max_r) when:
       - |mean_max_r_A − mean_max_r_B| ≤ merge_max_r_gap_a  (physical proximity)
       - mean cosine over all cross-pairs ≥ merge_cosine     (same environment)
    4. Optionally merge 1-pocket max_r bands into the nearest neighbor when
       peak radii are within ``singleton_band_gap_a`` (e.g. ZIF-8 corner vs
       center cages, grid-offset max_r only).
    5. Each remaining band → one pocket type.

    Parameters
    ----------
    initial_band_gap_a : 0.15 Å
        Gap above which two consecutive pockets form separate initial bands.
        Should be less than the grid spacing (0.5 Å) to catch only grid-noise.
    merge_max_r_gap_a : 0.50 Å
        Hard upper bound on the mean max_r gap between two bands for them to
        be merge candidates (= one grid spacing).
    merge_gap_threshold_a : 0.20 Å
        Gap boundary between tight and loose cosine thresholds.
    merge_cosine : 0.90
        Tight cosine threshold for gaps ≤ merge_gap_threshold_a (prevents spurious
        merges of similar-but-different window types, e.g. ZIF-20 2.56 vs 2.74 Å).
    merge_cosine_large_gap : 0.80
        Loose cosine threshold for gaps > merge_gap_threshold_a (allows genuine family
        merges, e.g. ZIF-20 large cages 7.26 vs 7.61 Å, mean cosine ~0.87).
    """
    if not pockets:
        return 0, {}, {"method": METHOD_ENV_CLUSTER, "n_initial_bands": 0, "n_merges": 0}

    sorted_pids = sorted(
        [int(p["id"]) for p in pockets],
        key=lambda pid: fps[pid].max_probe_radius_a,
    )

    # Step 1: initial bands — pockets separated by > initial_band_gap_a get new bands
    bands: List[List[int]] = [[sorted_pids[0]]]
    for pid in sorted_pids[1:]:
        prev_max_r = fps[bands[-1][-1]].max_probe_radius_a
        curr_max_r = fps[pid].max_probe_radius_a
        if curr_max_r - prev_max_r > initial_band_gap_a:
            bands.append([])
        bands[-1].append(pid)
    n_initial = len(bands)

    # Step 2: iterative average-linkage merge of adjacent bands
    n_merges = 0
    changed = True
    while changed:
        changed = False
        bands.sort(key=lambda g: _mean_max_r(g, fps))
        for i in range(len(bands) - 1):
            ga, gb = bands[i], bands[i + 1]
            gap = _mean_max_r(gb, fps) - _mean_max_r(ga, fps)
            if gap > merge_max_r_gap_a:
                continue
            cos_vals = [
                fingerprint_cosine_similarity(fps[a], fps[b], angular_only=False)
                for a in ga
                for b in gb
            ]
            eff_cosine = merge_cosine if gap <= merge_gap_threshold_a else merge_cosine_large_gap
            if float(np.mean(cos_vals)) >= eff_cosine:
                bands[i] = sorted(ga + gb)
                del bands[i + 1]
                n_merges += 1
                changed = True
                break  # restart after each merge

    n_singleton_merges = 0
    if merge_singleton_bands and len(bands) > 1:
        rep_to_band = {int(pid): ti for ti, grp in enumerate(bands) for pid in grp}
        before = dict(rep_to_band)
        rep_to_band = merge_singleton_max_r_bands(
            pockets, fps, rep_to_band, gap_tol_a=singleton_band_gap_a,
        )
        n_singleton_merges = sum(
            1 for rep in rep_to_band if rep_to_band[rep] != before[rep]
        )
        if n_singleton_merges:
            band_ids = sorted(set(rep_to_band.values()))
            remap = {old: new for new, old in enumerate(band_ids)}
            rep_to_type = {
                int(rep): remap[int(rep_to_band[rep])] for rep in rep_to_band
            }
            bands = [
                sorted(rep for rep, bi in rep_to_band.items() if remap[bi] == ti)
                for ti in range(len(band_ids))
            ]
        else:
            rep_to_type = {
                int(pid): ti for ti, grp in enumerate(bands) for pid in grp
            }
    else:
        rep_to_type = {int(pid): ti for ti, grp in enumerate(bands) for pid in grp}

    audit = {
        "method": METHOD_ENV_CLUSTER,
        "n_initial_bands": n_initial,
        "n_merges": n_merges,
        "n_singleton_band_merges": n_singleton_merges,
        "n_types_final": len(bands),
        "initial_band_gap_a": initial_band_gap_a,
        "merge_max_r_gap_a": merge_max_r_gap_a,
        "merge_gap_threshold_a": merge_gap_threshold_a,
        "merge_cosine": merge_cosine,
        "merge_cosine_large_gap": merge_cosine_large_gap,
        "singleton_band_gap_a": singleton_band_gap_a,
    }
    return len(bands), rep_to_type, audit


def assign_pocket_types_shape_refined(
    pockets: Sequence[Mapping[str, Any]],
    fps: Mapping[int, PocketFingerprint],
    *,
    tau_merge_a: float = DEFAULT_TAU_MERGE_A,
    band_gap_a: float = DEFAULT_MAX_R_BAND_GAP_A,
    singleton_band_gap_a: float = DEFAULT_SINGLETON_BAND_GAP_A,
    subsplit_gap_a: float = DEFAULT_MAX_R_SUBSPLIT_GAP_A,
    adjacent_merge_max_r_gap_a: float = DEFAULT_ADJACENT_BAND_MERGE_MAX_R_GAP_A,
    merge_cosine: float = FP_MERGE_COSINE,
    merge_singleton_bands: bool = True,
) -> Tuple[int, Dict[int, int], Dict[str, Any]]:
    """d₁ coarse bands first; max_r refinement only for large ZIF-like groups."""
    d1_by_rep = {int(p["id"]): float(p["d1_A"]) for p in pockets}
    d1_coarse = group_pockets_by_d1_coarse(pockets, tau_merge_a=tau_merge_a)

    working: List[List[int]] = []
    n_max_r_refined_groups = 0
    n_subsplits = 0
    for grp in d1_coarse:
        if _within_d1_group_use_max_r_refinement(grp, fps):
            n_max_r_refined_groups += 1
            subs = _refine_pocket_ids_by_max_r(
                grp,
                pockets,
                fps,
                d1_by_rep,
                band_gap_a=band_gap_a,
                singleton_band_gap_a=singleton_band_gap_a,
                subsplit_gap_a=subsplit_gap_a,
                merge_cosine=merge_cosine,
                merge_singleton_bands=merge_singleton_bands,
            )
            if len(subs) > 1:
                n_subsplits += 1
            working.extend(subs)
        else:
            working.append(list(grp))

    n_before_merge = len(working)
    working, n_fp_merges = merge_adjacent_groups_by_fingerprint(
        working,
        fps,
        d1_by_rep,
        merge_cosine=merge_cosine,
        max_probe_gap_a=adjacent_merge_max_r_gap_a,
        tau_merge_a=tau_merge_a,
    )

    rep_to_type: Dict[int, int] = {}
    for ti, grp in enumerate(working):
        for pid in grp:
            rep_to_type[int(pid)] = ti

    audit = {
        "n_coarse_d1_bands": len(d1_coarse),
        "n_d1_groups_max_r_refined": n_max_r_refined_groups,
        "n_working_before_fp_merge": n_before_merge,
        "n_types_final": len(working),
        "n_max_r_subsplits": n_subsplits,
        "n_adjacent_fp_merges": n_fp_merges,
        "tau_merge_a": tau_merge_a,
        "band_gap_a": band_gap_a,
        "subsplit_gap_a": subsplit_gap_a,
        "adjacent_merge_max_r_gap_a": adjacent_merge_max_r_gap_a,
        "merge_cosine": merge_cosine,
    }
    return len(working), rep_to_type, audit


def _types_from_bands(
    pockets: Sequence[Mapping[str, Any]],
    rep_to_band: Mapping[int, int],
) -> Tuple[int, Dict[int, int]]:
    """Map band ids → contiguous type ids 0..n-1."""
    band_ids = sorted(set(rep_to_band[int(p["id"])] for p in pockets))
    band_to_type = {bi: ti for ti, bi in enumerate(band_ids)}
    rep_to_type = {int(p["id"]): band_to_type[rep_to_band[int(p["id"])]] for p in pockets}
    return len(band_ids), rep_to_type


def build_fragment_label_to_rep(
    pbc_groups: Sequence[Sequence[int]],
) -> Dict[int, int]:
    """Map every fragment region label to its PBC representative (min label in group)."""
    label_to_rep: Dict[int, int] = {}
    for group in pbc_groups:
        regs = sorted(int(r) for r in group)
        if not regs:
            continue
        rep = regs[0]
        for reg in regs:
            label_to_rep[reg] = rep
    return label_to_rep


def build_fragment_label_to_type(
    rep_to_type: Mapping[int, int],
    pbc_groups: Sequence[Sequence[int]],
) -> Dict[int, int]:
    """Map every fragment label → 1-based pocket type id (via PBC rep).

    Types are keyed by the PBC representative (``min(group)``), matching
    ``regions_pbc`` / RAG pocket ids. All periodic copies in the group share
    the same type, analogous to ``clustering_grid_from_regions`` painting each
    RAG node's ``labels[0]`` with the merged cluster id.
    """
    label_to_rep = build_fragment_label_to_rep(pbc_groups)
    label_to_type: Dict[int, int] = {}
    for frag, rep in label_to_rep.items():
        if rep not in rep_to_type:
            continue
        label_to_type[frag] = int(rep_to_type[rep]) + 1
    return label_to_type


def regions_cluster_from_rep_types(
    regions: np.ndarray,
    rep_to_type: Mapping[int, int],
    pbc_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Paint the fragment ``regions`` grid from PBC group → rep → type.

    Follows the same idea as ``mgr.compute.clustering_grid_from_regions``:
    each fragment label in a PBC group is assigned the type of that group's
    representative, not only the representative voxel label.
    """
    regions = np.asarray(regions)
    label_to_type = build_fragment_label_to_type(rep_to_type, pbc_groups)
    regions_cluster = np.zeros_like(regions, dtype=int)
    for frag, tid in label_to_type.items():
        regions_cluster[regions == frag] = tid
    return regions_cluster


def regions_cluster_from_rag(
    regions: np.ndarray,
    rag: Any,
    label_to_type: Mapping[int, int],
) -> np.ndarray:
    """Paint fragment labels via RAG nodes (``labels[0]``), matching legacy clustering."""
    regions = np.asarray(regions)
    regions_cluster = np.zeros_like(regions, dtype=int)
    for node in rag.nodes:
        frag = int(rag.nodes[node]["labels"][0])
        tid = label_to_type.get(frag)
        if tid is not None:
            regions_cluster[regions == frag] = int(tid)
    return regions_cluster


def summarize_types(
    pockets: Sequence[Mapping[str, Any]],
    rep_to_type: Mapping[int, int],
    fps: Mapping[int, PocketFingerprint],
) -> List[Dict[str, Any]]:
    from collections import defaultdict

    by_type: Dict[int, List[int]] = defaultdict(list)
    for p in pockets:
        by_type[int(rep_to_type[int(p["id"])])].append(int(p["id"]))

    rows: List[Dict[str, Any]] = []
    for tid in sorted(by_type):
        pids = by_type[tid]
        max_rs = [fps[pid].max_probe_radius_a for pid in pids if pid in fps]
        vols = [fps[pid].void_volume_a3 for pid in pids if pid in fps]
        d1s = [float(next(p["d1_A"] for p in pockets if int(p["id"]) == pid)) for pid in pids]
        rows.append(
            {
                "type_id": tid,
                "n_pockets": len(pids),
                "max_r_min_a": float(min(max_rs)) if max_rs else 0.0,
                "max_r_max_a": float(max(max_rs)) if max_rs else 0.0,
                "max_r_mean_a": float(np.mean(max_rs)) if max_rs else 0.0,
                "volume_min_a3": float(min(vols)) if vols else 0.0,
                "volume_max_a3": float(max(vols)) if vols else 0.0,
                "d1_min_a": float(min(d1s)) if d1s else 0.0,
                "d1_max_a": float(max(d1s)) if d1s else 0.0,
                "rep_ids": sorted(pids),
            }
        )
    return rows


def assign_pocket_types_from_segmentation(
    segmentation: Mapping[str, Any],
    atoms=None,
    *,
    bonds: Optional[Sequence] = None,
    bond_orders: Optional[Sequence] = None,
    cif_path: Optional[Path | str] = None,
    method: str = DEFAULT_METHOD,
    band_gap_a: float = DEFAULT_MAX_R_BAND_GAP_A,
    merge_singleton_bands: bool = True,
    singleton_band_gap_a: float = DEFAULT_SINGLETON_BAND_GAP_A,
    subsplit_gap_a: float = DEFAULT_MAX_R_SUBSPLIT_GAP_A,
    adjacent_merge_max_r_gap_a: float = DEFAULT_ADJACENT_BAND_MERGE_MAX_R_GAP_A,
    merge_cosine: float = FP_MERGE_COSINE,
    probe_radius_psd: float = DEFAULT_PROBE_RADIUS_PSD_A,
    psd_bin_edges: Optional[np.ndarray] = None,
    env_cluster_initial_band_gap_a: Optional[float] = None,
    typing_radii: str | np.ndarray = DEFAULT_TYPING_RADII,
) -> PocketTypingResult:
    """Assign pocket types from periodic segmentation results.

    Default ``method`` is ``env_cluster`` (max_r initial bands + fingerprint merge).
    Use ``method="shape_refined"`` for the legacy d₁-first approach, or
    ``method="max_probe_radius"`` for bands-only typing.

    Parameters
    ----------
    segmentation
        Dict with at least ``dgrid``, ``regions``, ``maxima_info``, ``all_maxima``,
        ``connections``, ``unique_regions``. Prefer ``regions_pbc`` when present.
        The segmentation ``dgrid`` continues to drive PSD, windows, void sampling,
        and framework fingerprints.
    atoms
        ASE Atoms; defaults to ``segmentation["ase_atoms"]``.
    typing_radii
        VDW table for typing ``max_r`` bands only (default
        ``bondi_crc_fallback``: Bondi/CRC handbook + Mendeleev fill for missing
        symbols).  Pass ``segmentation`` to use the analysis ``dgrid`` for
        ``max_r`` as before, or a per-atom radii array (Å).
    cif_path
        Reserved for upstream Mofography; not required when segmentation is complete.
    """
    _ = cif_path  # optional metadata hook for callers
    try:
        if atoms is None:
            atoms = segmentation.get("ase_atoms")
        if atoms is None:
            return PocketTypingResult(ok=False, error="atoms missing (pass atoms or ase_atoms in segmentation)")

        required = ("dgrid", "regions", "maxima_info", "all_maxima")
        missing = [k for k in required if k not in segmentation]
        if missing:
            return PocketTypingResult(
                ok=False,
                error=f"segmentation missing keys: {', '.join(missing)}",
            )

        pockets, pbc_groups, _d1_map, pocket_psd_error = build_pockets_from_segmentation(
            segmentation,
            atoms,
            probe_radius_psd=probe_radius_psd,
            psd_bin_edges=psd_bin_edges,
        )
        if not pockets:
            err = "no pockets extracted from segmentation"
            if pocket_psd_error:
                err += f" (pocket PSD failed: {pocket_psd_error})"
            return PocketTypingResult(ok=False, error=err, method=method)

        typing_dgrid = compute_typing_dgrid(atoms, segmentation, typing_radii)
        fps = compute_pocket_fingerprints(
            segmentation,
            atoms,
            pockets,
            pbc_groups=pbc_groups,
            bonds=bonds,
            bond_orders=bond_orders,
            typing_dgrid=typing_dgrid,
        )

        used_method = method
        if method == METHOD_ENV_CLUSTER:
            env_kw: Dict[str, float] = {}
            if env_cluster_initial_band_gap_a is not None:
                env_kw["initial_band_gap_a"] = float(env_cluster_initial_band_gap_a)
            n_types, rep_to_type, refine_audit = assign_pocket_types_env_cluster(
                pockets,
                fps,
                merge_singleton_bands=merge_singleton_bands,
                singleton_band_gap_a=singleton_band_gap_a,
                **env_kw,
            )
            audit_extra = refine_audit
        elif method == METHOD_SHAPE_REFINED:
            n_types, rep_to_type, refine_audit = assign_pocket_types_shape_refined(
                pockets,
                fps,
                band_gap_a=band_gap_a,
                singleton_band_gap_a=singleton_band_gap_a,
                subsplit_gap_a=subsplit_gap_a,
                adjacent_merge_max_r_gap_a=adjacent_merge_max_r_gap_a,
                merge_cosine=merge_cosine,
                merge_singleton_bands=merge_singleton_bands,
            )
            audit_extra = refine_audit
        else:
            _, _, rep_to_band = assign_pocket_types_by_max_probe_radius(
                pockets, fps, band_gap_a=band_gap_a,
            )
            n_singleton_merges = 0
            if merge_singleton_bands:
                before = dict(rep_to_band)
                rep_to_band = merge_singleton_max_r_bands(
                    pockets, fps, rep_to_band, gap_tol_a=singleton_band_gap_a,
                )
                n_singleton_merges = sum(
                    1 for rep in rep_to_band if rep_to_band[rep] != before[rep]
                )
            n_types, rep_to_type = _types_from_bands(pockets, rep_to_band)
            audit_extra = {
                "n_singleton_band_merges": n_singleton_merges,
                "n_bands": len(set(rep_to_band.values())),
            }

        regions = np.asarray(segmentation["regions"])
        label_to_rep = build_fragment_label_to_rep(pbc_groups)
        label_to_type = build_fragment_label_to_type(rep_to_type, pbc_groups)
        regions_cluster = regions_cluster_from_rep_types(
            regions, rep_to_type, pbc_groups,
        )
        type_summary = summarize_types(pockets, rep_to_type, fps)

        audit = {
            "method": used_method,
            "band_gap_a": band_gap_a,
            "singleton_band_gap_a": singleton_band_gap_a,
            "probe_radius_psd_a": probe_radius_psd,
            "spacing_a": segmentation.get("spacing", DEFAULT_SPACING_A),
            "mask_thickness_a": segmentation.get("mask_thickness", DEFAULT_MASK_THICKNESS_A),
            "typing_radii": (
                "segmentation"
                if typing_dgrid is None
                else (str(typing_radii) if isinstance(typing_radii, str) else "array")
            ),
            "typing_dgrid_separate": typing_dgrid is not None,
            "fragment_label_to_rep": label_to_rep,
            "fragment_label_to_type": label_to_type,
        }
        audit.update(audit_extra)

        rep_to_band = {int(rep): int(rep_to_type[int(rep)]) for rep in rep_to_type}
        band_centers = sorted(
            {tid: _mean_max_r([p for p in rep_to_type if rep_to_type[p] == tid], fps)
             for tid in set(rep_to_type.values())}.values()
        )

        return PocketTypingResult(
            ok=True,
            method=used_method,
            n_pockets=len(pockets),
            n_types=n_types,
            rep_to_type=dict(rep_to_type),
            rep_to_band=rep_to_band,
            band_centers_a=band_centers,
            regions_cluster=regions_cluster,
            pockets=list(pockets),
            fingerprints=fps,
            type_summary=type_summary,
            pbc_group_count=len(pbc_groups),
            audit=audit,
        )
    except Exception as exc:
        return PocketTypingResult(ok=False, error=str(exc))


def pocket_typing_result_for_app(out: PocketTypingResult) -> Dict[str, Any]:
    """Convert :class:`PocketTypingResult` to the dict shape used by Streamlit/CLI."""
    if not out.ok:
        return {"ok": False, "error": out.error}
    num_clusters = int(out.n_types)
    return {
        "ok": True,
        "regions_cluster": out.regions_cluster,
        "num_clusters": num_clusters,
        "psd_source": "pocket_psd",
        "probe_psd": out.audit.get("probe_radius_psd_a", DEFAULT_PROBE_RADIUS_PSD_A),
        "pbc_group_count": out.pbc_group_count,
        "cluster_diags": [],
        "pocket_typing_method": out.method,
        "pocket_type_summary": out.type_summary,
        "pocket_rep_to_type": out.rep_to_type,
        "pocket_fragment_label_to_type": out.audit.get("fragment_label_to_type", {}),
        "pocket_typing_audit": out.audit,
    }

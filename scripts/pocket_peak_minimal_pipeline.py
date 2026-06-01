#!/usr/bin/env python3
"""Minimal peak-first pocket typing: d1 primary, neighbors tie-break, optional d2."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))
sys.path.insert(0, str(REPO / "python_cli"))

FIXTURE_DIR = REPO / "tests" / "fixtures" / "cif"
RASPA2_DIR = Path.home() / "repos" / "RASPA2" / "structures" / "mofs" / "cif"

CIF_CANDIDATES: Dict[str, List[Path]] = {
    "NU-1000": [FIXTURE_DIR / "NU-1000_S1.cif", REPO / "cifs" / "NU-1000_S1.cif"],
    "UMCM-1": [RASPA2_DIR / "UMCM-1.cif"],
    "CuBTC": [FIXTURE_DIR / "Cu-BTC.cif", RASPA2_DIR / "Cu-BTC.cif"],
    "ZIF-20": [RASPA2_DIR / "ZIF-20.cif"],
    "UiO-66": [RASPA2_DIR / "UIO-66.cif", RASPA2_DIR / "UiO-66.cif"],
    "IRMOF-1": [RASPA2_DIR / "IRMOF-1.cif"],
    "LTA": [REPO / "cifs" / "LTA96.cif"],
}


def resolve_cif(material: str) -> Optional[Path]:
    for p in CIF_CANDIDATES.get(material, []):
        if p.is_file():
            return p
    return None

logging.basicConfig(level=logging.WARNING)

SPACING = 0.5
MASK_THICKNESS = 0.05
BINS = np.linspace(0, 50, 201)
BIN_WIDTH = float(BINS[1] - BINS[0])
WINDOW_TRUST_MIN_A = 1.0
SECOND_PEAK_FRAC = 0.10  # only report d2 if h2 >= this fraction of h1


def _import_mgr():
    import mgr
    return mgr


def _import_cli_funcs():
    try:
        from python_cli.mgrcli.cli import (
            compute_dgrid_rust,
            get_pbc_groups,
            python_do_periodic_segmentation_rust,
        )
    except ImportError:
        from mgrcli.cli import (
            compute_dgrid_rust,
            get_pbc_groups,
            python_do_periodic_segmentation_rust,
        )
    return compute_dgrid_rust, python_do_periodic_segmentation_rust, get_pbc_groups


def _norm(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = y.sum()
    return y / s if s > 0 else y


def _edge_max_window(data: dict) -> float:
    if data.get("path_weight") == 0:
        return 0.0
    winfo = data.get("window_info")
    if not winfo:
        return 0.0
    wr = winfo.get("window_radii")
    if wr is None:
        return 0.0
    arr = np.asarray(wr, dtype=float).ravel()
    return float(np.max(arr)) if arr.size else 0.0


def _edge_trusted(data: dict, min_win: float = WINDOW_TRUST_MIN_A) -> bool:
    if data.get("path_weight") == 0:
        return False
    return _edge_max_window(data) > min_win


def extract_pocket_peaks(cif_path: Path) -> List[Dict[str, Any]]:
    from ase.io import read as ase_read
    from skimage.measure import regionprops

    mgr = _import_mgr()
    compute_dgrid_rust, do_seg_rust, get_pbc_groups = _import_cli_funcs()

    atoms = ase_read(str(cif_path))
    dgrid = compute_dgrid_rust(atoms, SPACING)
    seg = do_seg_rust(
        dgrid, atoms, mask_thickness=MASK_THICKNESS, h=0.2, spacing=SPACING,
    )
    regions = seg["regions"]
    regions_pbc = seg.get("regions_pbc", regions)
    maxima_info = seg["maxima_info"]
    all_maxima = seg["all_maxima"]
    connections = seg.get("connections", [])
    unique_regions = seg.get("unique_regions", [])

    pbc_groups = [v for v in get_pbc_groups(unique_regions, maxima_info).values() if 0 not in v]
    all_maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)]
    maxima = all_maxima[: regions.max()]
    maxima_radii = all_maxima_radii[: regions.max()]
    connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]

    rag = mgr.rag_from_connections(connections_trimmed, maxima)
    rag = mgr.add_window_info_to_rag(rag, connections_trimmed)
    rag = mgr.add_pbc_to_rag(rag, pbc_groups)
    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid.shape, atoms)
    props = regionprops(regions)

    pocket_psds = mgr.compute_psds_pocketwise_rust(
        dgrid, regions_pbc, atoms,
        num_samples_per_region=1000, bins=BINS, return_fig=False,
    )
    rag = mgr.add_csd_to_rag_with_pbc(rag, props, pbc_groups, BINS, atoms, dgrid.shape)
    if pocket_psds is not None:
        rag = mgr.add_pocket_psd_to_rag_with_pbc(rag, props, pbc_groups, pocket_psds)

    region_to_rep: Dict[int, int] = {}
    for group in pbc_groups:
        rep = min(group)
        for reg in group:
            region_to_rep[int(reg)] = rep

    pocket_adj: Dict[int, List[int]] = {min(g): [] for g in pbc_groups}
    for u, v, data in rag.edges(data=True):
        if not _edge_trusted(data):
            continue
        ru, rv = region_to_rep.get(int(u)), region_to_rep.get(int(v))
        if ru is None or rv is None or ru == rv:
            continue
        if rv not in pocket_adj[ru]:
            pocket_adj[ru].append(rv)
        if ru not in pocket_adj[rv]:
            pocket_adj[rv].append(ru)

    pockets: List[Dict[str, Any]] = []
    d1_map: Dict[int, float] = {}

    for pi, group in enumerate(pbc_groups):
        rep = min(group)
        nd = rag.nodes[rep]
        x = np.asarray(nd.get("pocket_psd_x_pbc", BINS), dtype=float)
        y = _norm(np.asarray(nd.get("pocket_psd_y_pbc", []), dtype=float))
        if y.size == 0:
            continue
        i1 = int(np.argmax(y))
        d1 = float(x[i1])
        h1 = float(y[i1])
        d1_map[rep] = d1

        # second tallest local maximum (optional)
        d2, h2 = None, 0.0
        y_work = y.copy()
        y_work[max(0, i1 - 1): min(len(y_work), i1 + 2)] = 0.0
        if y_work.max() >= SECOND_PEAK_FRAC * h1:
            i2 = int(np.argmax(y_work))
            d2 = float(x[i2])
            h2 = float(y_work[i2])

        nbr_d1 = sorted(round(d1_map.get(nb, 0.0), 2) for nb in pocket_adj.get(rep, []) if nb in d1_map)
        pockets.append({
            "id": int(rep),
            "pi": pi,
            "d1_A": d1,
            "d2_A": d2,
            "h1": h1,
            "h2": h2,
            "n_trusted": len(nbr_d1),
            "nbr_d1": nbr_d1,
        })

    return pockets


def tau_merge(bw: float = BIN_WIDTH) -> float:
    """Merge dominant peaks closer than this (Å). 2Δr catches adjacent-bin splits."""
    return max(2.0 * bw, 0.35)


def neighbor_signature_key(nbr_d1: Sequence[float]) -> Tuple[int, ...]:
    """Multiset of neighbor dominant diameters, 0.25 Å bins."""
    return tuple(sorted(int(round(d / BIN_WIDTH)) for d in nbr_d1))


def group_peak_first(
    pockets: List[Dict[str, Any]],
    tau: Optional[float] = None,
    *,
    neighbor_split: bool = False,
) -> List[List[int]]:
    """Return list of groups (indices into pockets)."""
    if not pockets:
        return []
    if tau is None:
        tau = tau_merge()

    order = sorted(range(len(pockets)), key=lambda i: pockets[i]["d1_A"])
    groups: List[List[int]] = []
    cur = [order[0]]
    for k in range(1, len(order)):
        if pockets[order[k]]["d1_A"] - pockets[cur[-1]]["d1_A"] <= tau:
            cur.append(order[k])
        else:
            groups.append(cur)
            cur = [order[k]]
    groups.append(cur)

    if not neighbor_split:
        return groups

    # Optional: split only when d1 tied (same bin) but neighbor signatures differ
    refined: List[List[int]] = []
    for grp in groups:
        if len(grp) <= 1:
            refined.append(grp)
            continue
        d1_spread = max(pockets[i]["d1_A"] for i in grp) - min(pockets[i]["d1_A"] for i in grp)
        if d1_spread < BIN_WIDTH * 0.5:
            buckets: Dict[Tuple[int, ...], List[int]] = {}
            for i in grp:
                key = neighbor_signature_key(pockets[i]["nbr_d1"])
                buckets.setdefault(key, []).append(i)
            if len(buckets) >= 2:
                refined.extend(buckets[k] for k in sorted(buckets))
            else:
                refined.append(grp)
        else:
            refined.append(grp)
    return refined


def label_size_ordered(groups: List[List[int]], pockets: List[Dict[str, Any]]) -> Dict[int, str]:
    """Assign mesopore / micropore / interstitial by decreasing mean d1."""
    names_3 = ["mesopore", "micropore", "interstitial"]
    stats = []
    for gi, grp in enumerate(groups):
        d1s = [pockets[i]["d1_A"] for i in grp]
        stats.append((np.mean(d1s), gi))
    stats.sort(reverse=True)
    out: Dict[int, str] = {}
    for rank, (_mean_d, gi) in enumerate(stats):
        if rank < len(names_3):
            name = names_3[rank]
        else:
            name = f"type_{rank + 1}"
        for i in groups[gi]:
            out[i] = name
    return out


def run_material(
    cif_path: Path,
    material: str,
    *,
    expected: Optional[int] = None,
    neighbor_split: bool = False,
) -> Dict[str, Any]:
    pockets = extract_pocket_peaks(cif_path)
    groups = group_peak_first(pockets, neighbor_split=neighbor_split)
    labels = label_size_ordered(groups, pockets)

    n_types = len(groups)
    match = ""
    if expected is not None:
        match = "OK" if n_types == expected else f"≠ expected {expected}"

    print(f"\n{'=' * 72}")
    print(f"{material}  —  minimal peak-first  (spacing={SPACING}, mask={MASK_THICKNESS})")
    print(f"tau_merge = {tau_merge():.3f} Å   bins Δr = {BIN_WIDTH:.3f} Å")
    print(f"PBC pockets: {len(pockets)}   Types: {n_types}   {match}")
    print(f"{'=' * 72}")

    order_g = sorted(
        range(len(groups)),
        key=lambda g: -np.mean([pockets[i]["d1_A"] for i in groups[g]]),
    )
    for gi in order_g:
        g = groups[gi]
        type_name = labels[g[0]]
        d1s = [pockets[i]["d1_A"] for i in g]
        print(f"\n  [{type_name}]  n={len(g)}  mean d1={np.mean(d1s):.2f} Å  "
              f"range [{min(d1s):.2f}, {max(d1s):.2f}]")
        for i in sorted(g, key=lambda j: pockets[j]["d1_A"]):
            p = pockets[i]
            d2s = f"  d2={p['d2_A']:.2f}" if p["d2_A"] is not None else ""
            print(
                f"    P{p['id']:>3}  d1={p['d1_A']:.2f} Å{d2s}  "
                f"n_trust={p['n_trusted']}  nbr_d1={p['nbr_d1']}"
            )

    band_summary = [
        (labels[g[0]], len(g), float(np.mean([pockets[i]["d1_A"] for i in g])))
        for g in groups
        if g
    ]
    band_summary.sort(key=lambda t: -t[2])
    return {
        "material": material,
        "n_pockets": len(pockets),
        "n_types": n_types,
        "expected": expected,
        "bands": band_summary,
        "groups": groups,
        "pockets": pockets,
    }


EXPECTED_COUNTS = {
    "NU-1000": 3,
    "UMCM-1": 2,
    "CuBTC": 3,
    "ZIF-20": 6,
    "UiO-66": 2,
    "IRMOF-1": 2,
    "LTA": 2,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal peak-first pocket typing")
    ap.add_argument("--material", default="NU-1000")
    ap.add_argument("--cif", type=Path, default=None)
    ap.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Run CuBTC, ZIF-20, LTA, UMCM-1 (and print summary)",
    )
    ap.add_argument("--neighbor-split", action="store_true")
    args = ap.parse_args()

    if args.all_benchmarks:
        names = ["CuBTC", "ZIF-20", "LTA", "UMCM-1"]
        results = []
        for name in names:
            cif = resolve_cif(name)
            if cif is None:
                print(f"{name}: CIF missing")
                continue
            results.append(
                run_material(
                    cif, name,
                    expected=EXPECTED_COUNTS.get(name),
                    neighbor_split=args.neighbor_split,
                )
            )
        print(f"\n{'=' * 72}")
        print("SUMMARY  (d1-only, tau = 2*bin_width)")
        print(f"{'=' * 72}")
        print(f"{'Material':<10} {'Pockets':>8} {'Types':>6} {'Exp':>6} {'Match':>8}  Peak bands (d1 Å)")
        for r in results:
            bands = ", ".join(f"{n}:{cnt}@{d:.2f}" for n, cnt, d in r["bands"])
            exp = r["expected"]
            ok = "OK" if exp is not None and r["n_types"] == exp else ("—" if exp is None else "≠")
            print(f"{r['material']:<10} {r['n_pockets']:>8} {r['n_types']:>6} {exp or '-':>6} {ok:>8}  {bands}")
        return

    cif = args.cif or resolve_cif(args.material)
    if cif is None or not cif.is_file():
        raise SystemExit(f"CIF not found for material {args.material!r}")

    run_material(
        cif, args.material,
        expected=EXPECTED_COUNTS.get(args.material),
        neighbor_split=args.neighbor_split,
    )


if __name__ == "__main__":
    main()

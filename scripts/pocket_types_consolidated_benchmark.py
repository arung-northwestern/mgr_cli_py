#!/usr/bin/env python3
"""Benchmark consolidated max_probe_radius pocket typing (spacing=0.5, mask=0.05)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase.io import read as ase_read

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))
sys.path.insert(0, str(REPO / "scripts"))

from mgr.pocket_typing import (  # noqa: E402
    DEFAULT_MASK_THICKNESS_A,
    DEFAULT_SPACING_A,
    assign_pocket_types_from_segmentation,
)
from pocket_peak_minimal_pipeline import (  # noqa: E402
    CIF_CANDIDATES,
    EXPECTED_COUNTS,
    MASK_THICKNESS,
    SPACING,
    resolve_cif,
)

BENCHMARKS = [
    "LTA",
    "CuBTC",
    "UMCM-1",
    "UiO-66",
    "IRMOF-1",
    "ZIF-20",
    "NU-1000",
]


def _import_cli():
    try:
        from python_cli.mgrcli.cli import (
            compute_dgrid_rust,
            python_do_periodic_segmentation_rust,
        )
    except ImportError:
        from mgrcli.cli import (
            compute_dgrid_rust,
            python_do_periodic_segmentation_rust,
        )
    return compute_dgrid_rust, python_do_periodic_segmentation_rust


def resolve_cif_benchmark(name: str) -> Optional[Path]:
    p = resolve_cif(name)
    if p is not None:
        return p
    for cand in CIF_CANDIDATES.get(name, []):
        if cand.is_file():
            return cand
    return None


def run_one(material: str, cif: Path) -> Dict[str, Any]:
    compute_dgrid_rust, do_seg_rust = _import_cli()
    t0 = time.perf_counter()
    atoms = ase_read(str(cif))
    dgrid = compute_dgrid_rust(atoms, SPACING)
    seg = do_seg_rust(
        dgrid,
        atoms,
        mask_thickness=MASK_THICKNESS,
        h=0.2,
        spacing=SPACING,
    )
    segmentation = {
        "dgrid": dgrid,
        "regions": seg["regions"],
        "regions_pbc": seg.get("regions_pbc", seg["regions"]),
        "maxima_info": seg["maxima_info"],
        "all_maxima": seg["all_maxima"],
        "connections": seg.get("connections", []),
        "unique_regions": seg.get("unique_regions", []),
        "ase_atoms": atoms,
        "spacing": SPACING,
        "mask_thickness": MASK_THICKNESS,
    }

    out = assign_pocket_types_from_segmentation(segmentation, atoms, cif_path=cif)
    elapsed = time.perf_counter() - t0
    exp = EXPECTED_COUNTS.get(material)
    return {
        "material": material,
        "ok": out.ok,
        "error": out.error,
        "n_pockets": out.n_pockets,
        "n_types": out.n_types,
        "expected": exp,
        "elapsed_s": elapsed,
        "type_summary": out.type_summary,
        "audit": out.audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", nargs="*", default=BENCHMARKS)
    args = parser.parse_args()

    print("=" * 92)
    print("Consolidated pocket typing — max probe radius bands")
    print(f"  spacing={DEFAULT_SPACING_A} Å  mask={DEFAULT_MASK_THICKNESS_A} Å")
    print("=" * 92)

    rows: List[Dict[str, Any]] = []
    for mat in args.materials:
        cif = resolve_cif_benchmark(mat)
        if cif is None:
            print(f"\n{mat}: CIF missing — skip")
            continue
        print(f"\n--- {mat} ---")
        r = run_one(mat, cif)
        rows.append(r)
        if not r["ok"]:
            print(f"  FAILED: {r['error']}")
            continue
        exp = r["expected"]
        match = ""
        if exp is not None:
            match = "OK" if r["n_types"] == exp else f"≠ expected {exp}"
        print(
            f"  pockets={r['n_pockets']}  types={r['n_types']}  {match}  "
            f"({r['elapsed_s']:.1f}s)"
        )
        for ts in r["type_summary"]:
            print(
                f"    type {ts['type_id']}: n={ts['n_pockets']:3d}  "
                f"max_r={ts['max_r_min_a']:.2f}-{ts['max_r_max_a']:.2f} Å  "
                f"d1={ts['d1_min_a']:.2f}-{ts['d1_max_a']:.2f} Å"
            )

    print("\n" + "=" * 92)
    print(f"{'Material':<10} {'Pockets':>8} {'Types':>6} {'Exp':>6} {'Match':>8} {'Time':>8}")
    print("-" * 92)
    for r in rows:
        if not r["ok"]:
            print(f"{r['material']:<10} {'—':>8} {'—':>6} {'—':>6} {'FAIL':>8}")
            continue
        exp = r["expected"]
        exp_s = str(exp) if exp is not None else "—"
        match = "OK" if exp is not None and r["n_types"] == exp else ("—" if exp is None else "≠")
        print(
            f"{r['material']:<10} {r['n_pockets']:8d} {r['n_types']:6d} {exp_s:>6} {match:>8} "
            f"{r['elapsed_s']:7.1f}s"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile aromatic ring detection + trace build (per-step breakdown + optional chart)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))

import numpy as np  # noqa: E402


def _timed():
    t0 = time.perf_counter()

    def done() -> float:
        return time.perf_counter() - t0

    return t0, done


def _profile_detection_steps(sgraph, *, use_rust: bool = True) -> dict:
    """Mirror find_aromatic_cycles_in_sgraph with per-step timings."""
    import networkx as nx
    from mgr.analyze import (
        _aromatic_cycles_from_basis,
        _cycle_is_aromatic,
        _expand_aromatic_embeddings_python,
        _expand_aromatic_embeddings_rust,
        find_chemically_unique_cycles,
    )

    default_ring_sizes = [5, 6, 7, 8, 9]
    counts: dict[str, int] = {}

    t0, fin = _timed()
    small_cycles = [
        c
        for c in nx.cycles.cycle_basis(sgraph, 0)
        if len(c) in default_ring_sizes
    ]
    counts["cycle_basis_filtered"] = len(small_cycles)
    t_basis = fin()

    t0, fin = _timed()
    aromatic = [c for c in small_cycles if _cycle_is_aromatic(sgraph, c)]
    counts["aromatic_after_filter"] = len(aromatic)
    t_filter = fin()

    t0, fin = _timed()
    unique_cycles = [list(u) for u in {tuple(sorted(a)) for a in aromatic}]
    counts["unique_cycles"] = len(unique_cycles)
    t_dedupe = fin()

    t0, fin = _timed()
    _idx = find_chemically_unique_cycles(unique_cycles, sgraph)
    chemically_unique = [unique_cycles[i] for i in _idx]
    counts["chemically_unique"] = len(chemically_unique)
    t_chem = fin()

    t_expand_rust = 0.0
    t_expand_py = 0.0
    n_final = len(unique_cycles)

    if len(chemically_unique) == len(unique_cycles):
        counts["expand_skipped"] = 1
    else:
        counts["expand_skipped"] = 0
        if use_rust:
            t0, fin = _timed()
            try:
                expanded = _expand_aromatic_embeddings_rust(sgraph, chemically_unique)
                t_expand_rust = fin()
                n_final = len(expanded)
                counts["expand_backend"] = "rust"
            except Exception as exc:
                counts["expand_rust_error"] = str(exc)
                use_rust = False
        if not use_rust or counts.get("expand_rust_error"):
            t0, fin = _timed()
            expanded = _expand_aromatic_embeddings_python(sgraph, chemically_unique)
            t_expand_py = fin()
            n_final = len(expanded)
            counts["expand_backend"] = "python"

    # Sanity: full pipeline
    t0, fin = _timed()
    from mgr.analyze import find_aromatic_cycles_in_sgraph

    full = find_aromatic_cycles_in_sgraph(sgraph, use_rust=use_rust)
    t_full = fin()
    counts["final_cycles"] = len(full)
    counts["full_matches_steps"] = len(full) == n_final

    steps = {
        "1_cycle_basis": t_basis,
        "2_aromatic_filter": t_filter,
        "3_dedupe": t_dedupe,
        "4_chem_unique": t_chem,
        "5_expand_rust": t_expand_rust,
        "5_expand_python": t_expand_py,
        "total_steps_sum": t_basis + t_filter + t_dedupe + t_chem + t_expand_rust + t_expand_py,
        "find_aromatic_cycles_full": t_full,
    }
    return {"steps": steps, "counts": counts}


def _save_chart(
    steps: dict[str, float],
    counts: dict,
    cif_name: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping chart PNG)")
        return

    # Plot only detection substeps (exclude full pipeline duplicate)
    plot_keys = [
        ("1_cycle_basis", "cycle_basis"),
        ("2_aromatic_filter", "aromatic filter"),
        ("3_dedupe", "dedupe"),
        ("4_chem_unique", "chem unique"),
        ("5_expand_rust", "expand (Rust)"),
        ("5_expand_python", "expand (Python)"),
    ]
    labels, values = [], []
    for key, label in plot_keys:
        v = steps.get(key, 0.0)
        if v > 0:
            labels.append(label)
            values.append(v)

    if not values:
        print("  (no step times to chart)")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(values)))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("seconds")
    ax.set_title(
        f"Aromatic detection — {cif_name}\n"
        f"cycles={counts.get('final_cycles', '?')} "
        f"(basis→aromatic={counts.get('aromatic_after_filter', '?')} "
        f"unique={counts.get('unique_cycles', '?')} "
        f"templates={counts.get('chemically_unique', '?')})"
    )
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}s", va="center", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  chart saved: {out_path}")


def _ascii_bar(label: str, seconds: float, max_s: float, width: int = 40) -> str:
    if max_s <= 0:
        n = 0
    else:
        n = int(round(width * seconds / max_s))
    return f"  {label:22s} |{'█' * n}{'░' * (width - n)}| {seconds:6.3f}s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "cif",
        nargs="?",
        default=str(REPO / "tests/fixtures/cif/NU-1000_S1.cif"),
    )
    p.add_argument(
        "--chart",
        action="store_true",
        help="Save horizontal bar chart PNG (requires matplotlib)",
    )
    p.add_argument(
        "--chart-out",
        type=Path,
        default=None,
        help="PNG path (default: <cif_stem>_aromatic_profile.png in cwd)",
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    p.add_argument(
        "--compare-python-expand",
        action="store_true",
        help="Also time Python VF2 expansion (slow on large MOFs)",
    )
    args = p.parse_args()
    cif = Path(args.cif)
    stem = cif.stem

    from ase.io import read
    from mgr.analyze import find_aromatic_cycles_in_sgraph, sgraph_from_atoms, update_with_covalent_bonds
    from mgr.movies import make_aromatic_rings_trace

    prep: dict[str, float] = {}
    print(f"CIF: {cif}")
    t0, fin = _timed()
    atoms = read(cif)
    prep["read_cif"] = fin()
    print(f"  atoms: {len(atoms)}")

    t0, fin = _timed()
    atoms = update_with_covalent_bonds(atoms)
    prep["covalent_bonds"] = fin()

    t0, fin = _timed()
    sgraph = sgraph_from_atoms(atoms)
    prep["sgraph_from_atoms"] = fin()
    print(
        f"  graph: {sgraph.number_of_nodes()} nodes, "
        f"{sgraph.number_of_edges()} edges"
    )

    print("\n--- detection steps ---")
    prof = _profile_detection_steps(sgraph, use_rust=True)
    steps = prof["steps"]
    counts = prof["counts"]

    if args.compare_python_expand and counts.get("expand_skipped", 0) == 0:
        from mgr.analyze import (
            _expand_aromatic_embeddings_python,
            find_chemically_unique_cycles,
            _aromatic_cycles_from_basis,
        )

        uc = _aromatic_cycles_from_basis(sgraph)
        cu = [uc[i] for i in find_chemically_unique_cycles(uc, sgraph)]
        t0, fin = _timed()
        _expand_aromatic_embeddings_python(sgraph, cu)
        steps["5_expand_python_compare"] = fin()

    step_times = {k: v for k, v in steps.items() if v > 0 and not k.startswith("total") and k != "find_aromatic_cycles_full"}
    max_s = max(step_times.values()) if step_times else 1.0
    for key in sorted(step_times.keys()):
        label = key.split("_", 1)[-1].replace("_", " ")
        print(_ascii_bar(label, step_times[key], max_s))

    print(f"\n  counts: {json.dumps(counts, indent=2)}")
    print(f"  find_aromatic_cycles_in_sgraph (full): {steps['find_aromatic_cycles_full']:.3f}s")
    print(f"  sum(substeps): {steps['total_steps_sum']:.3f}s")

    print("\n--- trace build ---")
    cycles = find_aromatic_cycles_in_sgraph(sgraph)
    t0, fin = _timed()
    traces = make_aromatic_rings_trace(atoms, cycles=cycles)
    t_trace = fin()
    print(f"  make_aromatic_rings_trace: {t_trace:.3f}s")

    circ = traces["circles"]["data"]
    circ_type = traces["circles"].get("type", "?")
    if circ_type == "instanced_ring_outlines":
        n_rings = circ.get("n_rings", 0)
        fill_kb = 0.0
        for key in ("fills_hex", "fills_pent"):
            ft = traces.get(key)
            if ft is not None:
                fill_kb += len(json.dumps(ft["data"])) / 1024
        circ_kb = len(json.dumps(circ)) / 1024
        print(
            f"  render=instanced rings={n_rings} "
            f"outlines={circ_kb:.0f} KB fills={fill_kb:.0f} KB total={circ_kb + fill_kb:.0f} KB"
        )
    else:
        hi = traces["highlights"]["data"]
        pos_circ = len(circ.get("positions", []))
        pos_hi = len(hi.get("positions", []))
        approx_json_kb = (pos_circ + pos_hi + len(hi.get("indices", []))) * 4 / 1024
        print(f"  render=legacy segments={circ.get('n_segments', 0)} ~payload {approx_json_kb:.0f} KB")

    print("\n--- prep (not in detection chart) ---")
    for k, v in prep.items():
        print(f"  {k}: {v:.3f}s")

    summary = {"cif": str(cif), "prep": prep, "detection": prof, "trace_s": t_trace}
    if args.json:
        print(json.dumps(summary, indent=2))

    if args.chart:
        out = args.chart_out or Path(f"{stem}_aromatic_profile.png")
        _save_chart(
            {
                "1_cycle_basis": steps["1_cycle_basis"],
                "2_aromatic_filter": steps["2_aromatic_filter"],
                "3_dedupe": steps["3_dedupe"],
                "4_chem_unique": steps["4_chem_unique"],
                "5_expand_rust": steps["5_expand_rust"],
                "5_expand_python": steps.get("5_expand_python", 0.0),
            },
            counts,
            stem,
            out,
        )


if __name__ == "__main__":
    main()

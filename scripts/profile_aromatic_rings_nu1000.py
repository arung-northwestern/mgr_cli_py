#!/usr/bin/env python3
"""Profile aromatic ring detection + trace build for NU-1000 (or any CIF)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))

import numpy as np  # noqa: E402


def _timed(label: str):
    t0 = time.perf_counter()

    def done():
        dt = time.perf_counter() - t0
        print(f"  {label}: {dt:.3f}s")
        return dt

    return t0, done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "cif",
        nargs="?",
        default=str(REPO / "tests/fixtures/cif/NU-1000_S1.cif"),
    )
    args = p.parse_args()
    cif = Path(args.cif)

    from ase.io import read
    from mgr.analyze import (
        find_aromatic_cycles_in_sgraph,
        find_chemically_unique_cycles,
        sgraph_from_atoms,
        update_with_covalent_bonds,
    )
    from mgr.movies import make_aromatic_rings_trace

    print(f"CIF: {cif}")
    _, fin = _timed("read_cif")
    atoms = read(cif)
    fin()
    print(f"  atoms: {len(atoms)}")

    _, fin = _timed("update_with_covalent_bonds")
    atoms = update_with_covalent_bonds(atoms)
    fin()

    _, fin = _timed("sgraph_from_atoms")
    sgraph = sgraph_from_atoms(atoms)
    fin()
    print(f"  graph: {sgraph.number_of_nodes()} nodes, {sgraph.number_of_edges()} edges")

    _, fin = _timed("find_aromatic_cycles_in_sgraph")
    cycles = find_aromatic_cycles_in_sgraph(sgraph)
    fin()
    n_cycles = len(cycles)
    print(f"  aromatic cycles: {n_cycles}")
    if cycles:
        sizes = [len(c) for c in cycles]
        uniq, counts = np.unique(sizes, return_counts=True)
        print(f"  ring sizes: {dict(zip(uniq.tolist(), counts.tolist()))}")

    _, fin = _timed("make_aromatic_rings_trace")
    traces = make_aromatic_rings_trace(atoms, cycles=cycles)
    fin()

    circ = traces["circles"]["data"]
    hi = traces["highlights"]["data"]
    n_seg = circ.get("n_segments", 0)
    n_verts = hi.get("n_vertices", 0)
    n_tri = hi.get("n_triangles", 0)
    pos_circ = len(circ.get("positions", []))
    pos_hi = len(hi.get("positions", []))
    print(f"  trace circles: segments={n_seg} floats={pos_circ}")
    print(f"  trace highlights: verts={n_verts} tris={n_tri} floats={pos_hi}")
    # Rough JSON payload (positions + indices as text in HTML)
    approx_json_kb = (pos_circ + pos_hi + len(hi.get("indices", []))) * 4 / 1024
    print(f"  ~payload (geometry floats only): {approx_json_kb:.0f} KB")

    # Unique cycles only (skip isomorphism expansion) — compare
    import networkx as nx

    small = [c for c in nx.cycles.cycle_basis(sgraph, 0) if len(c) in (5, 6, 7, 8, 9)]
    bond_types = nx.get_edge_attributes(sgraph, "bond_type")
    aromatic = []
    for cycle in small:
        edges = list(zip(cycle, (cycle[1:] + cycle[:1])))
        if all(bond_types.get(tuple(sorted(e)), "single") == "double" for e in edges):
            aromatic.append(cycle)
    unique = [list(u) for u in set(tuple(sorted(a)) for a in aromatic)]
    uidx = find_chemically_unique_cycles(unique, sgraph)
    chem_unique = [unique[i] for i in uidx]
    print(f"  chemically unique aromatic (no iso expand): {len(chem_unique)}")

    _, fin = _timed("make_aromatic_rings_trace (chem-unique only)")
    make_aromatic_rings_trace(atoms, cycles=chem_unique)
    fin()


if __name__ == "__main__":
    main()

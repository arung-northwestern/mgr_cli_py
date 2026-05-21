#!/usr/bin/env python3
"""Phase 0: profile covalent-bond pipeline buckets (no algorithm changes)."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
MGR_SRC = REPO / "mgr_src"
if str(MGR_SRC) not in sys.path:
    sys.path.insert(0, str(MGR_SRC))

import numpy as np  # noqa: E402

from mgr.analyze import aabb_neighbors_on_atoms, update_with_covalent_bonds  # noqa: E402
from mgr.cif_io import read_cif_to_ase  # noqa: E402

try:
    import mgrcli_rust
except ImportError:
    mgrcli_rust = None  # type: ignore[assignment]

FIXTURE_DIR = REPO / "tests" / "fixtures" / "cif"
RASPA2_MIL101 = Path.home() / "repos" / "RASPA2" / "structures" / "mofs" / "cif" / "MIL-101.cif"

FIXTURE_CASES: List[Tuple[str, Path]] = [
    ("ZIF-8", FIXTURE_DIR / "ZIF-8-RASPA.cif"),
    ("NU-1000", FIXTURE_DIR / "NU-1000_S1.cif"),
    ("MIL-101-primitive", FIXTURE_DIR / "MIL-101-primitive.cif"),
    ("Cu-BTC", FIXTURE_DIR / "Cu-BTC.cif"),
]

WARMUP = 1
REPEATS = 5


def _freud_available() -> bool:
    try:
        import freud  # noqa: F401
        return True
    except ImportError:
        return False


def _rayon_threads() -> Optional[int]:
    if mgrcli_rust is None:
        return None
    try:
        return int(mgrcli_rust.rayon_num_threads())
    except Exception:
        return None


@dataclass(frozen=True)
class BucketSample:
    natoms: int
    npairs: int
    nbonds: int
    n_images: int
    neighbor_backend: str
    neighbor_s: float
    marshal_s: float
    rust_s: float
    unpack_s: float
    e2e_s: float


def _profile_buckets_once(atoms) -> BucketSample:
    if mgrcli_rust is None:
        raise RuntimeError("mgrcli_rust not built")
    tol = 0.2
    backend = "freud" if _freud_available() else "ase"
    t0 = time.perf_counter()
    nlist = aabb_neighbors_on_atoms(atoms)
    neighbor_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    nlist_i = np.array([p[0] for p in nlist], dtype=np.int64)
    nlist_j = np.array([p[1] for p in nlist], dtype=np.int64)
    positions = atoms.get_positions().ravel().tolist()
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    cell_fortran = atoms.get_cell().array.T.flatten(order="F").tolist()
    nlist_i.tolist()
    nlist_j.tolist()
    marshal_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    bonds_flat, bond_orders, image_positions, image_atomic_numbers = mgrcli_rust.covalent_bonds(
        positions, atomic_numbers, cell_fortran, nlist_i.tolist(), nlist_j.tolist(), tol
    )
    rust_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    from ase import Atoms
    from ase.data import chemical_symbols
    bonds = [[bonds_flat[2 * i], bonds_flat[2 * i + 1]] for i in range(len(bonds_flat) // 2)]
    if image_atomic_numbers:
        img_pos_array = np.array(image_positions).reshape(-1, 3)
        img_symbols = [chemical_symbols[z] for z in image_atomic_numbers]
        _ = Atoms(
            symbols=atoms.get_chemical_symbols() + img_symbols,
            positions=np.vstack([atoms.get_positions(), img_pos_array]),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )
    else:
        _ = atoms.copy()
    list(bond_orders)
    unpack_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = update_with_covalent_bonds(atoms)
    e2e_s = time.perf_counter() - t0
    return BucketSample(
        natoms=len(atoms), npairs=len(nlist), nbonds=len(out.info.get("covalent_bonds", [])),
        n_images=max(0, len(out) - len(atoms)), neighbor_backend=backend,
        neighbor_s=neighbor_s, marshal_s=marshal_s, rust_s=rust_s, unpack_s=unpack_s, e2e_s=e2e_s,
    )


def _median_field(samples, name):
    return statistics.median([getattr(s, name) for s in samples])


def _median_profile(atoms, repeats=REPEATS):
    for _ in range(WARMUP):
        _profile_buckets_once(atoms)
    runs = [_profile_buckets_once(atoms) for _ in range(repeats)]
    return BucketSample(
        natoms=runs[-1].natoms, npairs=runs[-1].npairs, nbonds=runs[-1].nbonds,
        n_images=runs[-1].n_images, neighbor_backend=runs[-1].neighbor_backend,
        neighbor_s=_median_field(runs, "neighbor_s"), marshal_s=_median_field(runs, "marshal_s"),
        rust_s=_median_field(runs, "rust_s"), unpack_s=_median_field(runs, "unpack_s"),
        e2e_s=_median_field(runs, "e2e_s"),
    )


def _cases(include_raspa2):
    out = list(FIXTURE_CASES)
    if include_raspa2 and RASPA2_MIL101.is_file():
        out.append(("MIL-101 (RASPA2)", RASPA2_MIL101))
    return out


def run_timing_table(*, include_raspa2=False, repeats=REPEATS):
    if mgrcli_rust is None:
        return "SKIP: mgrcli_rust not importable"
    lines = ["Covalent bonds Phase 0 profile (median seconds)",
             f"  repeats={repeats} warmup={WARMUP} freud={_freud_available()}",
             f"  RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', '(unset)')}",
             f"  rayon_num_threads()={_rayon_threads()}",
             "  Note: covalent_bonds is single-threaded; Rayon does not change rust_s yet.", ""]
    header = f"{'structure':<22} {'natoms':>6} {'npairs':>8} {'nbonds':>7} {'nimg':>5} {'nbr':>4} {'neighbor':>8} {'marshal':>8} {'rust':>8} {'unpack':>8} {'sum':>8} {'e2e':>8} {'m%':>6}"
    lines += [header, "-" * len(header)]
    for label, path in _cases(include_raspa2):
        if not path.is_file():
            lines.append(f"{label:<22} MISSING")
            continue
        s = _median_profile(read_cif_to_ase(path), repeats=repeats)
        sm = s.neighbor_s + s.marshal_s + s.rust_s + s.unpack_s
        mp = 100 * s.marshal_s / sm if sm else 0
        lines.append(f"{label:<22} {s.natoms:6d} {s.npairs:8d} {s.nbonds:7d} {s.n_images:5d} {s.neighbor_backend:>4} {s.neighbor_s:8.4f} {s.marshal_s:8.4f} {s.rust_s:8.4f} {s.unpack_s:8.4f} {sm:8.4f} {s.e2e_s:8.4f} {mp:5.1f}%")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-raspa2-mil101", action="store_true")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--hf-mimic", action="store_true")
    args = p.parse_args()
    if args.hf_mimic:
        os.environ["RAYON_NUM_THREADS"] = "2"
    os.environ.setdefault("MGR_FAST_CIF", "1")
    print(run_timing_table(include_raspa2=args.include_raspa2_mil101, repeats=args.repeats))


if __name__ == "__main__":
    main()

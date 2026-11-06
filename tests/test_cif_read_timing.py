"""Benchmark ASE vs fast Rust CIF read for fixture + RASPA2 structures.

Run (needs built mgrcli_rust):
  pytest tests/test_cif_read_timing.py -v -s
  python tests/test_cif_read_timing.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from ase.io import read as ase_read

try:
    import mgrcli_rust
except ImportError:
    mgrcli_rust = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
MGR_SRC = REPO_ROOT / "mgr_src"
if str(MGR_SRC) not in sys.path:
    sys.path.insert(0, str(MGR_SRC))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "cif"
RASPA2_MIL101 = Path.home() / "repos" / "RASPA2" / "structures" / "mofs" / "cif" / "MIL-101.cif"

FIXTURE_CASES: List[Tuple[str, Path, str]] = [
    ("ZIF-8", FIXTURE_DIR / "ZIF-8-RASPA.cif", "ZIF-8-RASPA.cif"),
    ("CuBTC", FIXTURE_DIR / "Cu-BTC.cif", "Cu-BTC.cif"),
    ("NU-1000", FIXTURE_DIR / "NU-1000_S1.cif", "NU-1000_S1.cif"),
    ("MIL-101-primitive", FIXTURE_DIR / "MIL-101-primitive.cif", "MIL-101-primitive.cif"),
]

WARMUP = 1
REPEATS = 5
# Large RASPA2 supercells: ASE can take minutes; benchmark fast path only.
ASE_SKIP_BYTES = 500_000
# Approximate ASE read (full run not attempted; order ~minutes on 894 KB CIF).
ASE_APPROX_SEC: dict[str, float] = {
    "MIL-101 (RASPA2)": 90.0,
}
LARGE_REPEATS = 1


def _benchmark_paths() -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = [(label, path) for label, path, _ in FIXTURE_CASES]
    if RASPA2_MIL101.is_file():
        out.append(("MIL-101 (RASPA2)", RASPA2_MIL101))
    return out


def _time_ase(path: Path) -> Tuple[float, int]:
    t0 = time.perf_counter()
    atoms = ase_read(str(path))
    elapsed = time.perf_counter() - t0
    return elapsed, len(atoms)


def _time_fast_rust(path: Path, read_cif) -> Tuple[float, int]:
    t0 = time.perf_counter()
    symbols, positions, cell, pbc = read_cif(str(path))
    elapsed = time.perf_counter() - t0
    return elapsed, len(symbols)


def _repeats_for(path: Path) -> int:
    try:
        if path.stat().st_size >= ASE_SKIP_BYTES:
            return LARGE_REPEATS
    except OSError:
        pass
    return REPEATS


def _median_seconds(fn, path: Path) -> Tuple[float, int]:
    n_atoms = 0
    repeats = _repeats_for(path)
    for _ in range(WARMUP):
        _, n_atoms = fn(path)
    samples: List[float] = []
    for _ in range(repeats):
        dt, n_atoms = fn(path)
        samples.append(dt)
    samples.sort()
    return samples[len(samples) // 2], n_atoms


def run_timing_table() -> str:
    if mgrcli_rust is None:
        raise RuntimeError(
            "mgrcli_rust not installed. From repo root: pip install -e . or maturin develop"
        )

    read_cif = mgrcli_rust.read_cif
    rows: List[dict] = []

    for mat, path in _benchmark_paths():
        if not path.is_file():
            rows.append(
                {
                    "mat": mat,
                    "atoms": "-",
                    "time_ase_s": "-",
                    "time_fast_s": "-",
                    "speedup": f"missing: {path}",
                }
            )
            continue

        skip_ase = path.stat().st_size >= ASE_SKIP_BYTES or mat in ASE_APPROX_SEC
        t_fast, n_fast = _median_seconds(lambda p: _time_fast_rust(p, read_cif), path)

        if skip_ase:
            t_ase = ASE_APPROX_SEC.get(mat, float("nan"))
            n_ase = n_fast
            ase_note = "~" if mat in ASE_APPROX_SEC else "?"
        else:
            t_ase, n_ase = _median_seconds(lambda p: _time_ase(p), path)
            ase_note = ""

        speedup = t_ase / t_fast if t_fast > 0 and t_ase == t_ase else float("inf")

        rows.append(
            {
                "mat": mat,
                "atoms": n_ase,
                "time_ase_s": t_ase,
                "time_fast_s": t_fast,
                "speedup": speedup,
                "ase_note": ase_note,
            }
        )

    keys = ("mat", "atoms", "time_ase_s", "time_fast_s", "speedup")
    headers = ("mat", "atoms", "time_ase_s", "time_fast_s", "speedup_ase_over_fast")
    col_w = [max(len(h), max(len(str(r[k])) for r in rows)) for k, h in zip(keys, headers)]

    def fmt_row(cells: Tuple[str, ...]) -> str:
        return " | ".join(c.ljust(col_w[i]) for i, c in enumerate(cells))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in col_w)]
    for r in rows:
        if isinstance(r["time_ase_s"], float) and r["time_ase_s"] == r["time_ase_s"]:
            ase_str = f"{r.get('ase_note', '')}{r['time_ase_s']:.4f}"
            sp = (
                f"{r['speedup']:.0f}x+"
                if r.get("ase_note") == "~"
                else f"{r['speedup']:.2f}x"
            )
            lines.append(
                fmt_row(
                    (
                        str(r["mat"]),
                        str(r["atoms"]),
                        ase_str,
                        f"{r['time_fast_s']:.4f}",
                        sp,
                    )
                )
            )
        else:
            lines.append(
                fmt_row(
                    (
                        str(r["mat"]),
                        str(r["atoms"]),
                        str(r["time_ase_s"]),
                        str(r["time_fast_s"]),
                        str(r["speedup"]),
                    )
                )
            )
    return "\n".join(lines)


def test_cif_read_timing_table() -> None:
    import pytest

    if mgrcli_rust is None:
        pytest.skip("mgrcli_rust not built")
    table = run_timing_table()
    print("\n" + table)
    assert "mat" in table


if __name__ == "__main__":
    os.environ.setdefault("MGR_FAST_CIF", "1")
    print(run_timing_table())

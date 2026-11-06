#!/usr/bin/env python3
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from ase.io import read


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MGR_SRC = REPO_ROOT / "mgr_src"
if str(MGR_SRC) not in sys.path:
    sys.path.insert(0, str(MGR_SRC))

from python_cli.mgrcli.cli import compute_dgrid_rust  # noqa: E402
from mgr.compute import dgrid_from_atoms_cpu  # noqa: E402


def _run_method(name: str, fn):
    t0 = time.perf_counter()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grid = fn()
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return grid, {
        "ok": True,
        "time_s": float(dt),
        "delta_maxrss_mib": float(max(0.0, (rss1 - rss0) / 1024.0)),
        "peak_maxrss_mib": float(rss1 / 1024.0),
        "method": name,
    }


def benchmark_one(path: Path, spacing: float) -> Dict:
    out = {"file": str(path), "name": path.name, "spacing": spacing, "methods": {}}
    try:
        atoms = read(str(path))
    except Exception as exc:
        out["error"] = f"read_failed:{type(exc).__name__}:{exc}"
        return out

    n_atoms = len(atoms)
    nxyz = (atoms.get_cell_lengths_and_angles()[:3] / spacing).astype(int) + 1
    n_grid = int(np.prod(nxyz))
    ops = int(n_atoms * n_grid)
    out.update(
        {"n_atoms": int(n_atoms), "n_grid": int(n_grid), "ops_estimate": int(ops), "grid_shape": [int(x) for x in nxyz]}
    )

    run_python = ops <= 220_000_000
    run_rtree = ops <= 900_000_000

    grids = {}
    if run_python:
        try:
            g, stat = _run_method(
                "python_aabb",
                lambda: dgrid_from_atoms_cpu(atoms, spacing=spacing, block_size=50000, n_neighbors=10).astype(np.float32),
            )
            grids["python_aabb"] = g
            out["methods"]["python_aabb"] = stat
        except Exception as exc:
            out["methods"]["python_aabb"] = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
    else:
        out["methods"]["python_aabb"] = {"ok": False, "skipped": "work_guard"}

    if run_rtree:
        for method, kwargs in (
            ("rust_aabb", {"method": "rust_aabb", "n_neighbors": 10}),
            (
                "rust_aabb_interp",
                {"method": "rust_aabb_interp", "n_neighbors": 10, "coarse_factor": 2},
            ),
        ):
            try:
                g, stat = _run_method(method, lambda kw=kwargs: compute_dgrid_rust(atoms, spacing, **kw))
                grids[method] = g
                out["methods"][method] = stat
            except Exception as exc:
                out["methods"][method] = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
    else:
        out["methods"]["rust_aabb"] = {"ok": False, "skipped": "work_guard"}
        out["methods"]["rust_aabb_interp"] = {"ok": False, "skipped": "work_guard"}

    if "python_aabb" in grids:
        base = grids["python_aabb"]
        for method in ("rust_aabb", "rust_aabb_interp"):
            if method in grids and grids[method].shape == base.shape:
                diff = np.asarray(grids[method] - base, dtype=np.float32)
                out["methods"][method]["mae_vs_python_aabb"] = float(np.mean(np.abs(diff)))
                out["methods"][method]["rmse_vs_python_aabb"] = float(np.sqrt(np.mean(diff * diff)))
                out["methods"][method]["max_abs_vs_python_aabb"] = float(np.max(np.abs(diff)))

    return out


def main():
    spacing = float(os.environ.get("DGRID_BENCH_SPACING", "1.1"))
    raspa_root = Path(os.environ.get("RASPA_CIF_DIR", "cifs"))
    cifs = sorted(raspa_root.rglob("*.cif"))
    results: List[Dict] = []

    print(f"[bench] spacing={spacing} total_cifs={len(cifs)}")
    for i, cif in enumerate(cifs, start=1):
        print(f"[bench] {i}/{len(cifs)} {cif.name}", flush=True)
        results.append(benchmark_one(cif, spacing))

    # Aggregate.
    rows = []
    for item in results:
        for method, stat in item.get("methods", {}).items():
            if stat.get("ok"):
                rows.append(
                    {
                        "method": method,
                        "time_s": stat["time_s"],
                        "peak_mib": stat["peak_maxrss_mib"],
                        "mae": stat.get("mae_vs_python_aabb", math.nan),
                        "max_abs": stat.get("max_abs_vs_python_aabb", math.nan),
                    }
                )

    summary = {
        "spacing": spacing,
        "total_cifs": len(cifs),
        "completed_cases": len(results),
        "method_stats": {},
    }

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            entry = {
                "n": int(len(sub)),
                "median_time_s": float(sub["time_s"].median()),
                "mean_time_s": float(sub["time_s"].mean()),
                "median_peak_mib": float(sub["peak_mib"].median()),
            }
            if method in {"rust_aabb", "rust_aabb_interp"}:
                acc = sub[np.isfinite(sub["mae"])]
                if len(acc) > 0:
                    entry["median_mae_vs_python_aabb"] = float(acc["mae"].median())
                    entry["median_max_abs_vs_python_aabb"] = float(acc["max_abs"].median())
            summary["method_stats"][method] = entry

    out_path = Path("/tmp/dgrid_benchmark_results.json")
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"[bench] wrote {out_path}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

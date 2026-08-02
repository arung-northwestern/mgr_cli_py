#!/usr/bin/env python3
"""Verify ZIF-8 Three.js HTML has window radii and edge filter counts."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_HTML = REPO / "examples/viz_outputs/ZIF-8-RASPA_spacing0.25_threejs.html"


def connection_passes_window_threshold(segment_radii, threshold):
    if threshold is None:
        return True
    if not segment_radii:
        return True
    vals = [float(x) for x in segment_radii if np.isfinite(x)]
    if not vals:
        return True
    return any(v >= threshold for v in vals)


def load_data(html_path: Path) -> dict:
    text = html_path.read_text(encoding="utf-8")
    m = re.search(r"const DATA = (\{.*?\});\s*\n", text, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find DATA JSON in HTML")
    return json.loads(m.group(1))


def trace_by_id(data: dict, trace_id: str) -> dict | None:
    for tr in data.get("static", {}).get("traces", []):
        if tr.get("id") == trace_id:
            return tr
    return None


def count_edges_after_filter(radii: list[list[float]], threshold: float | None) -> int:
    return sum(1 for wr in radii if connection_passes_window_threshold(wr, threshold))


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HTML
    if not path.is_file():
        print(f"Missing {path}; run scripts/gen_zif8_spacing_threejs.py first")
        return 1

    data = load_data(path)
    conn = trace_by_id(data, "connections_internal")
    wins = trace_by_id(data, "windows_internal")
    if conn is None:
        print("FAIL: connections_internal trace missing")
        return 1

    radii = conn.get("data", {}).get("segment_window_radii") or []
    n_seg = conn.get("data", {}).get("n_segments") or 0
    if not radii or n_seg == 0:
        print("FAIL: no internal connection segments / window radii in HTML")
        return 1
    if len(radii) != n_seg:
        print(f"FAIL: radii count {len(radii)} != n_segments {n_seg}")
        return 1

    flat_min = [min(wr) for wr in radii if wr]
    flat_max = [max(wr) for wr in radii if wr]
    print(f"OK: {n_seg} internal edges, segment_window_radii present")
    print(
        f"    window radii along edges: min={min(flat_min):.3f} max={max(flat_max):.3f} Å"
    )

    for thr in (None, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0):
        n = count_edges_after_filter(radii, thr)
        print(f"    threshold={thr}: {n}/{n_seg} edges kept")

    if wins is not None:
        wr = wins.get("data", {}).get("window_radii") or []
        print(f"    windows_internal points: {len(wr)} (window_radii per point)")

    n_none = count_edges_after_filter(radii, None)
    n_06 = count_edges_after_filter(radii, 0.6)
    if n_06 >= n_none:
        print("FAIL: threshold 0.6 did not remove any edges")
        return 1

    print(f"\nManual test: open file://{path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

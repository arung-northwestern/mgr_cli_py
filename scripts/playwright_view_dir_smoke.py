#!/usr/bin/env python3
"""Playwright smoke: view-direction HUD sync + Enter-to-apply (main + percolation)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "playwright_view_dir"
CIF = REPO / "cifs" / "NU-1000_S1.cif"
NPZ = REPO / "hts_output" / "NU-1000_S1" / "results.npz"


def _load_segmentation():
    from ase.io import read

    import mgr

    atoms = read(str(CIF))
    dgrid = np.asarray(np.load(NPZ)["dgrid"], dtype=np.float32)
    results = mgr.do_periodic_segmentation_rust(
        dgrid,
        atoms,
        spacing=0.25,
        mask_thickness=0.05,
        h=0.2,
        threshold_abs=1.5,
        min_distance=1.5,
        extra_padding=5.0,
        return_dgrid=True,
        compute_connections=True,
        watershed_backend="rust_det",
    )
    results["ase_atoms"] = atoms
    return results, atoms


def _build_main_html(results, atoms) -> Path:
    from mgr.threejs_plotting import show_mof_threejs

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "nu1000_main_threejs.html"
    html = show_mof_threejs(
        results,
        options={
            "plot_framework": True,
            "plot_surface": True,
            "plot_connections": True,
            "plot_windows": True,
            "plot_layout": "dark",
        },
    )
    out.write_text(html, encoding="utf-8")
    return out


def _build_percolation_html(analysis, atoms, probe_radius: float) -> Path:
    from mgr.threejs_plotting import show_percolation_threejs

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"nu1000_percolation_r{probe_radius:g}.html"
    html = show_percolation_threejs(
        analysis,
        atoms,
        probe_radius=probe_radius,
        options={"layout": "dark"},
    )
    out.write_text(html, encoding="utf-8")
    return out


def _assert_close_vec(actual: dict, expected: tuple[float, float, float], tol: float = 0.05) -> None:
    ax, ay, az = float(actual["x"]), float(actual["y"]), float(actual["z"])
    ex, ey, ez = expected
    if abs(ax - ex) > tol or abs(ay - ey) > tol or abs(az - ez) > tol:
        raise AssertionError(f"expected {expected}, got ({ax:.3f}, {ay:.3f}, {az:.3f})")


def _playwright_view_dir_case(html_path: Path, label: str) -> None:
    from playwright.sync_api import sync_playwright

    url = html_path.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(url, wait_until="networkidle", timeout=180_000)
        page.wait_for_selector("canvas", timeout=60_000)
        page.wait_for_function("() => typeof window.__mofFigureGetViewDir === 'function'", timeout=60_000)
        page.wait_for_timeout(500)

        page.click("#viewZ")
        page.wait_for_timeout(300)
        inputs = page.evaluate(
            """() => ({
              x: parseFloat(document.getElementById('viewDirX').value),
              y: parseFloat(document.getElementById('viewDirY').value),
              z: parseFloat(document.getElementById('viewDirZ').value),
            })"""
        )
        cam = page.evaluate("() => window.__mofFigureGetViewDir()")
        _assert_close_vec(inputs, (0.0, 0.0, 1.0))
        _assert_close_vec(cam, (0.0, 0.0, 1.0))

        page.fill("#viewDirX", "0")
        page.fill("#viewDirY", "1")
        page.fill("#viewDirZ", "0")
        page.press("#viewDirY", "Enter")
        page.wait_for_timeout(300)
        cam2 = page.evaluate("() => window.__mofFigureGetViewDir()")
        inputs2 = page.evaluate(
            """() => ({
              x: parseFloat(document.getElementById('viewDirX').value),
              y: parseFloat(document.getElementById('viewDirY').value),
              z: parseFloat(document.getElementById('viewDirZ').value),
            })"""
        )
        _assert_close_vec(cam2, (0.0, 1.0, 0.0))
        _assert_close_vec(inputs2, (0.0, 1.0, 0.0))
        browser.close()
    print(f"  OK {label}")


def main() -> int:
    sys.path[:0] = [str(REPO / "mgr_src"), str(REPO / "python_cli")]

    t0 = time.time()
    print("Building NU-1000 scenes...")
    results, atoms = _load_segmentation()
    main_html = _build_main_html(results, atoms)

    from mgrcli.interactive_app import _compute_pld_analysis

    analysis = _compute_pld_analysis(
        results,
        atoms,
        spacing=0.25,
        mask_thickness=0.05,
        h_value=0.2,
        threshold_abs=1.5,
        min_distance=1.5,
        extra_padding=5.0,
    )
    perc_html = _build_percolation_html(analysis, atoms, 3.0)
    print(f"  build time: {time.time() - t0:.1f}s")

    try:
        _playwright_view_dir_case(main_html, "main Three.js")
        _playwright_view_dir_case(perc_html, "percolation Three.js r=3")
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"OK — view dir HUD works. Artifacts: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

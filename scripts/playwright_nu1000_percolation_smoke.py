#!/usr/bin/env python3
"""Playwright smoke: NU-1000 percolation Three.js at probe radii 3 and 4 Å."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "playwright_nu1000_percolation"
CIF = REPO / "cifs" / "NU-1000_S1.cif"
NPZ = REPO / "hts_output" / "NU-1000_S1" / "results.npz"


def _build_analysis():
    from ase.io import read

    import mgr
    from mgrcli.interactive_app import _compute_pld_analysis

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
    return _compute_pld_analysis(
        results,
        atoms,
        spacing=0.25,
        mask_thickness=0.05,
        h_value=0.2,
        threshold_abs=1.5,
        min_distance=1.5,
        extra_padding=5.0,
    ), atoms


def _render_html(analysis, atoms, probe_radius: float, *, filter_graph: bool, window_color: str) -> Path:
    from mgr.threejs_plotting import show_percolation_threejs

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"r{probe_radius:g}_filter{int(filter_graph)}"
    out = OUT / f"nu1000_percolation_{tag}.html"
    html = show_percolation_threejs(
        dict(analysis),
        atoms,
        probe_radius=probe_radius,
        options={
            "layout": "dark",
            "filter_connections_at_probe": filter_graph,
            "plot_windows_marker_color": window_color,
        },
    )
    out.write_text(html, encoding="utf-8")
    return out


def _html_has_traces(html_path: Path) -> dict:
    text = html_path.read_text(encoding="utf-8")
    return {
        "clusters": text.count("percolation_clusters"),
        "windows": text.count('"id": "percolation_windows"'),
        "connections": text.count("connections_internal"),
        "pld": text.count("percolation_windows_pld"),
    }


def _playwright_check(html_path: Path, screenshot_path: Path) -> dict:
    from playwright.sync_api import sync_playwright

    url = html_path.resolve().as_uri()
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.goto(url, wait_until="networkidle", timeout=180_000)
        page.wait_for_selector("canvas", timeout=60_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(screenshot_path), full_page=True)
        canvas_count = page.locator("canvas").count()
        browser.close()

    return {
        "canvas_count": canvas_count,
        "console_errors": console_errors,
        "page_errors": page_errors,
    }


def main() -> int:
    sys.path[:0] = [str(REPO / "mgr_src"), str(REPO / "python_cli")]

    t0 = time.time()
    print("Building NU-1000 PLD analysis...")
    analysis, atoms = _build_analysis()
    history_radii = sorted({float(e[0]) for e in analysis["percolation_history"]})
    print(f"  history steps: {len(history_radii)}  range {history_radii[0]:.2f}-{history_radii[-1]:.2f} Å")
    print(f"  build time: {time.time() - t0:.1f}s")

    cases = []
    for probe in (3.0, 4.0):
        for filter_graph in (False, True):
            cases.append((probe, filter_graph))

    failures = []
    for probe, filter_graph in cases:
        label = f"probe={probe:g} filter={filter_graph}"
        print(f"Rendering {label}...")
        html_path = _render_html(
            analysis,
            atoms,
            probe,
            filter_graph=filter_graph,
            window_color="#00FFAA",
        )
        traces = _html_has_traces(html_path)

        shot = OUT / f"screenshot_{html_path.stem}.png"
        print(f"  Playwright -> {shot.name}")
        pw = _playwright_check(html_path, shot)

        ok = (
            pw["canvas_count"] >= 1
            and not pw["page_errors"]
            and traces["clusters"] >= 1
            and traces["windows"] >= 1
        )

        print(
            f"  traces: clusters={traces['clusters']} windows={traces['windows']} "
            f"connections={traces['connections']} pld={traces['pld']}"
        )
        print(f"  canvas={pw['canvas_count']} page_errors={len(pw['page_errors'])} console_errors={len(pw['console_errors'])}")
        if pw["page_errors"]:
            print("  PAGE ERRORS:", pw["page_errors"][:3])
        if not ok:
            failures.append(label)

    print()
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print("OK — NU-1000 percolation smoke passed at probe 3 and 4 Å (filter on/off).")
    print(f"Artifacts: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

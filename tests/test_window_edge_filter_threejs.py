"""Playwright: min-window filter hides pore-graph edges inside explode groups."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "examples/viz_outputs/ZIF-8-RASPA_spacing0.25_threejs.html"
GEN = REPO / "scripts/gen_zif8_spacing_threejs.py"


def _ensure_html() -> Path:
    if not HTML.is_file():
        subprocess.run([sys.executable, str(GEN)], check=True, cwd=REPO)
    return HTML


def _load_radii(html_path: Path) -> list[list[float]]:
    text = html_path.read_text(encoding="utf-8")
    m = re.search(r"const DATA = (\{.*?\});\s*\n", text, re.DOTALL)
    data = json.loads(m.group(1))
    for tr in data.get("static", {}).get("traces", []):
        if tr.get("id") == "connections_internal":
            return tr.get("data", {}).get("segment_window_radii") or []
    return []


def _expected_edges(radii: list[list[float]], threshold: float | None) -> int:
    if threshold is None:
        return len(radii)
    kept = 0
    for wr in radii:
        vals = [float(x) for x in wr if np.isfinite(x)]
        if vals and min(vals) >= threshold:
            kept += 1
    return kept


@pytest.fixture(scope="module")
def html_path() -> Path:
    return _ensure_html()


def test_window_filter_edge_count_playwright(html_path: Path):
  try:
      from playwright.sync_api import sync_playwright
  except ImportError:
      pytest.skip("playwright not installed")

  radii = _load_radii(html_path)
  assert radii, "missing segment_window_radii in HTML"
  expected_06 = _expected_edges(radii, 0.6)
  expected_none = len(radii)
  assert expected_06 < expected_none, "0.6 Å threshold should drop some edges in static data"

  url = html_path.resolve().as_uri()
  with sync_playwright() as p:
      browser = p.chromium.launch(headless=True)
      page = browser.new_page()
      page.goto(url, wait_until="networkidle", timeout=180_000)
      page.wait_for_function(
          "typeof objectsById !== 'undefined' && objectsById.connections_internal"
      )

      def edge_count() -> int:
          return page.evaluate(
              """() => {
                const obj = objectsById.connections_internal;
                if (!obj || !obj.userData) return -1;
                const hover = obj.userData.activeHover;
                if (hover && hover.length) return hover.length;
                const pos = obj.userData.originalPositions || [];
                return Math.floor(pos.length / 6);
              }"""
          )

      def pore_child_conn_groups() -> int:
          return page.evaluate(
              """() => {
                const grp = objectsById.pore_explode_group;
                if (!grp) return 0;
                let n = 0;
                grp.traverse(function(c) {
                  if (c.name === "connections_internal") n += 1;
                });
                return n;
              }"""
          )

      n0 = edge_count()
      assert n0 == expected_none

      page.check("#hideEdgesBelowWindow")
      page.fill("#windowMinThreshold", "0.6")
      page.dispatch_event("#windowMinThreshold", "input")
      page.wait_for_timeout(300)

      n1 = edge_count()
      assert n1 == expected_06
      assert pore_child_conn_groups() == 1, "orphan connection meshes after filter rebuild"

      page.evaluate("() => { explodeSlider.value = '1.5'; applyExplodeOffsets(); }")
      page.wait_for_timeout(200)
      n2 = edge_count()
      assert n2 == expected_06

      browser.close()

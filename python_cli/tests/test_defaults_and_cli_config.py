"""Tests for shared defaults and CLI config parity with the web app."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgrcli.defaults import (
    SEGMENTATION_DEFAULTS,
    DGRID_DEFAULTS,
    TEXTURAL_DEFAULTS,
    analysis_defaults,
    apply_config_overrides,
    build_segmentation_kwargs,
)
from mgrcli.dgrid_policy import resolve_dgrid_method


class TestSharedDefaults(unittest.TestCase):
  def test_web_defaults_match_streamlit_segmentation(self):
    web = analysis_defaults("web")
    self.assertEqual(web["mask_thickness"], 0.05)
    self.assertEqual(web["watershed_backend"], "rust_det")
    self.assertEqual(web["minimum_window_separation"], 1.0)
    self.assertEqual(web["extra_padding"], 5.0)
    self.assertTrue(web["skip_endchannel_pairs"])
    self.assertEqual(web["vdw_radii_set"], "bondi")
    self.assertTrue(web["auto_dgrid"])
    self.assertEqual(web["sa_method"], "mc")
    self.assertEqual(web["accessibility_method"], "winding")
    self.assertTrue(web["window_fp"])

  def test_legacy_profile_differs(self):
    legacy = analysis_defaults("legacy")
    self.assertEqual(legacy["mask_thickness"], 0.5)
    self.assertEqual(legacy["watershed_backend"], "rust_bucket")
    self.assertEqual(legacy["vdw_radii_set"], "crc")
    self.assertFalse(legacy["auto_dgrid"])

  def test_build_segmentation_kwargs(self):
    cfg = analysis_defaults("web")
    kw = build_segmentation_kwargs(cfg, None, None, spacing=0.5)
    self.assertEqual(kw["mask_thickness"], 0.05)
    self.assertEqual(kw["watershed_backend"], "rust_det")
    self.assertEqual(kw["minimum_window_separation"], 1.0)
    self.assertTrue(kw["skip_endchannel_pairs"])
    self.assertTrue(kw["return_dgrid"])

  def test_apply_config_overrides(self):
    base = analysis_defaults("web")
    ns = type("NS", (), {"spacing": 0.4, "vdw_radii_set": "alvarez", "h": None})()
    merged = apply_config_overrides(base, ns)
    self.assertEqual(merged["spacing"], 0.4)
    self.assertEqual(merged["vdw_radii_set"], "alvarez")
    self.assertEqual(merged["mask_thickness"], 0.05)


class TestDgridPolicy(unittest.TestCase):
  def test_small_structure_stays_rust_exact(self):
    try:
      from ase import Atoms
    except ImportError:
      self.skipTest("ase not installed")
    atoms = Atoms("C", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    method, meta = resolve_dgrid_method(atoms, 0.5, "rust_exact", auto_dgrid=True)
    self.assertEqual(method, "rust_exact")
    self.assertEqual(meta["resolved"], "rust_exact")

  def test_auto_dgrid_off_keeps_configured_method(self):
    try:
      from ase import Atoms
    except ImportError:
      self.skipTest("ase not installed")
    atoms = Atoms("C8", positions=[[i, 0, 0] for i in range(8)], cell=[200, 200, 200], pbc=True)
    method, meta = resolve_dgrid_method(atoms, 0.1, "rust_exact", auto_dgrid=False)
    self.assertEqual(method, "rust_exact")
    self.assertFalse(meta["auto_dgrid"])

  def test_explicit_dgrid_method_not_overridden(self):
    try:
      from ase import Atoms
    except ImportError:
      self.skipTest("ase not installed")
    atoms = Atoms("C8", positions=[[i, 0, 0] for i in range(8)], cell=[200, 200, 200], pbc=True)
    method, _ = resolve_dgrid_method(atoms, 0.1, "rust_aabb", auto_dgrid=True)
    self.assertEqual(method, "rust_aabb")


class TestCLIConfigIntegration(unittest.TestCase):
  test_dir = Path(__file__).parent
  repo_root = test_dir.parent.parent

  def test_generate_config_web_profile(self):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
      config_file = f.name
    try:
      cmd = [
        sys.executable,
        "-m",
        "mgrcli.cli",
        "generate-config",
        "--profile",
        "web",
        "-o",
        config_file,
      ]
      result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
      self.assertEqual(result.returncode, 0, result.stderr)
      with open(config_file) as fh:
        config = json.load(fh)
      self.assertEqual(config["mask_thickness"], SEGMENTATION_DEFAULTS["mask_thickness"])
      self.assertEqual(config["watershed_backend"], SEGMENTATION_DEFAULTS["watershed_backend"])
      self.assertEqual(config["vdw_radii_set"], DGRID_DEFAULTS["vdw_radii_set"])
      self.assertTrue(config["skip_endchannel_pairs"])
      self.assertTrue(config["window_fp"])
      self.assertTrue(config["plot_surface_screen_door"])
      self.assertEqual(config["supercell_nx"], 1)
      self.assertFalse(config["compute_aromatic_rings"])
      self.assertEqual(config["plot_explode"], 0.0)
      self.assertEqual(config["plot_windows_marker_size"], 10)
      self.assertNotIn("plot_framework_style", config)
    finally:
      os.unlink(config_file)

  def test_cli_module_defaults(self):
    from mgrcli.cli import MCLI

    cli = MCLI()
    self.assertEqual(cli.DEFAULTS["mask_thickness"], 0.05)
    self.assertEqual(cli.DEFAULTS["watershed_backend"], "rust_det")
    self.assertEqual(cli.DEFAULTS["vdw_radii_set"], "bondi")

  def test_process_help_lists_new_flags(self):
    cmd = [sys.executable, "-m", "mgrcli.cli", "process", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--vdw-radii-set", result.stdout)
    self.assertIn("--no-auto-dgrid", result.stdout)
    self.assertIn("--sa-method", result.stdout)
    self.assertIn("--percolation-plot", result.stdout)
    self.assertIn("--accessibility-method", result.stdout)
    self.assertIn("--no-window-fp", result.stdout)
    self.assertIn("--aromatic-rings", result.stdout)
    self.assertIn("--supercell", result.stdout)
    self.assertIn("--surface-opacity", result.stdout)
    self.assertIn("--surface-pbc-opacity", result.stdout)
    self.assertIn("--surface-types-opacity", result.stdout)
    self.assertIn("--window-min-radius", result.stdout)
    self.assertIn("--explode", result.stdout)
    self.assertIn("--percolation-history", result.stdout)
    self.assertIn("--cache-dir", result.stdout)
    self.assertIn("--no-cache", result.stdout)
    self.assertIn("--no-screen-door", result.stdout)
    self.assertIn("--filter-graph-at-probe", result.stdout)

  def test_viz_flags_update_config(self):
    from mgrcli.cli import MCLI

    cli = MCLI()
    ns = type(
      "NS",
      (),
      {
        "supercell": [2, 1, 3],
        "aromatic_rings": True,
        "framework_style": "instanced",
        "surface_opacity": 0.4,
        "surface_pbc_opacity": 0.2,
        "surface_types_opacity": 0.5,
        "window_min_radius": 1.25,
        "explode": 1.5,
        "explode_plane": "bc",
        "no_screen_door": True,
        "no_pore_graph": True,
        "filter_graph_at_probe": True,
        "percolation_boundary_width": 3,
        "cache_dir": "cache_out",
        "no_cache": True,
      },
    )()
    merged = cli.update_config_from_args({}, ns)
    self.assertEqual(merged["supercell_nx"], 2)
    self.assertEqual(merged["supercell_ny"], 1)
    self.assertEqual(merged["supercell_nz"], 3)
    self.assertTrue(merged["compute_aromatic_rings"])
    self.assertTrue(merged["plot_framework_show_rings"])
    self.assertEqual(merged["plot_framework_style"], "instanced")
    self.assertTrue(merged["plot_framework_style_from_user"])
    self.assertEqual(merged["plot_surface_opacity"], 0.4)
    self.assertEqual(merged["plot_surface_pbc_opacity"], 0.2)
    self.assertEqual(merged["plot_surface_types_opacity"], 0.5)
    self.assertEqual(merged["plot_window_min_radius"], 1.25)
    self.assertEqual(merged["plot_explode"], 1.5)
    self.assertEqual(merged["plot_explode_plane"], "bc")
    self.assertFalse(merged["plot_surface_screen_door"])
    self.assertFalse(merged["plot_connections"])
    self.assertTrue(merged["percolation_filter_graph_at_probe"])
    self.assertEqual(merged["percolation_boundary_width"], 3)
    self.assertEqual(merged["cache_dir"], "cache_out")
    self.assertFalse(merged["use_result_cache"])
    self.assertEqual(cli._resolve_framework_style(merged, "threejs"), "instanced")
    bare = cli.update_config_from_args({}, type("NS", (), {})())
    self.assertEqual(cli._resolve_framework_style(bare, "threejs"), "instanced")
    self.assertEqual(cli._resolve_framework_style(bare, "plotly"), "scatter")
    self.assertTrue(bare["plot_surface_screen_door"])
    self.assertEqual(bare["plot_windows_marker_color"], "#FFFF00")

  def test_saved_supercell_drives_later_plots(self):
    from mgrcli.cli import MCLI

    cli = MCLI()
    seen = {}

    def fake_prepared(cif, config, announce=True):
      seen["cif"] = cif
      seen["config"] = dict(config)
      seen["announce"] = announce
      return "prepared"

    cli._prepared_atoms = fake_prepared
    atoms = cli._atoms_matching_results(
      Path("structure.cif"),
      {"supercell_nx": 2, "supercell_ny": 2, "supercell_nz": 1, "min_box_thickness": 12.0},
      {"supercell_nx": 1, "supercell_ny": 1, "supercell_nz": 1},
    )
    self.assertEqual(atoms, "prepared")
    self.assertEqual(seen["config"]["supercell_nx"], 2)
    self.assertEqual(seen["config"]["supercell_ny"], 2)
    self.assertEqual(seen["config"]["min_box_thickness"], 12.0)
    self.assertFalse(seen["announce"])

  def test_interactive_help_lists_clean(self):
    cmd = [sys.executable, "-m", "mgrcli.cli", "interactive", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--clean", result.stdout)


if __name__ == "__main__":
  unittest.main()

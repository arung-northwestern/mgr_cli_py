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
    self.assertIn("--auto-dgrid", result.stdout)
    self.assertIn("--no-window-fp", result.stdout)

  def test_interactive_help_lists_clean(self):
    cmd = [sys.executable, "-m", "mgrcli.cli", "interactive", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--clean", result.stdout)


if __name__ == "__main__":
  unittest.main()

"""Tests for Three.js batch HTML export via save_mof_html."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgrcli.plotting import save_mof_html


class TestThreeJsPlotting(unittest.TestCase):
    test_dir = Path(__file__).parent

    def test_save_mof_html_threejs_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scene.html"
            with patch("mgr.threejs_plotting.show_mof_threejs", return_value="<html>threejs</html>") as mock_show:
                save_mof_html(
                    out,
                    {"regions": None, "pocket_types_detected": False},
                    backend="threejs",
                    stylize=True,
                    options={"plot_surface": True},
                    theme="dark",
                )
            self.assertTrue(out.exists())
            self.assertEqual(out.read_text(encoding="utf-8"), "<html>threejs</html>")
            mock_show.assert_called_once()
            call_kwargs = mock_show.call_args.kwargs
            self.assertEqual(call_kwargs.get("options", {}).get("plot_layout"), "dark")

    def test_process_help_lists_threejs_backend(self):
        cmd = [sys.executable, "-m", "mgrcli.cli", "process", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--plot-backend", result.stdout)
        self.assertIn("threejs", result.stdout)

    def test_plot_command_help_lists_threejs(self):
        cmd = [sys.executable, "-m", "mgrcli.cli", "plot", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("threejs", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""CLI pocket typing method resolution and env_cluster wiring."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mgrcli.cli import MCLI
from mgrcli.defaults import analysis_defaults


class TestPocketTypingCLI(unittest.TestCase):
    test_dir = Path(__file__).parent

    def setUp(self):
        self.cli = MCLI()
        self.config = analysis_defaults("web")

    def test_types_flag_resolves_env_cluster(self):
        args = SimpleNamespace(types=True, pocket_typing=None)
        self.assertEqual(self.cli._resolve_pocket_typing_method(self.config, args), "env_cluster")

    def test_pocket_typing_legacy_js_override(self):
        args = SimpleNamespace(types=True, pocket_typing="legacy_js")
        self.assertEqual(self.cli._resolve_pocket_typing_method(self.config, args), "legacy_js")

    def test_pocket_typing_only(self):
        args = SimpleNamespace(types=False, pocket_typing="env_cluster")
        self.assertEqual(self.cli._resolve_pocket_typing_method(self.config, args), "env_cluster")

    def test_no_pocket_typing_by_default(self):
        args = SimpleNamespace(types=False, pocket_typing=None)
        self.assertIsNone(self.cli._resolve_pocket_typing_method(self.config, args))

    def test_process_help_lists_pocket_typing_flags(self):
        cmd = [sys.executable, "-m", "mgrcli.cli", "process", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--pocket-typing", result.stdout)
        self.assertIn("env_cluster", result.stdout)
        self.assertIn("--probe-radius-psd", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""
Tests for MCLI CLI tool
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
import sys
import subprocess

# Add the parent directory to the path so we can import mcli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMCLI(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent
        self.cifs_dir = self.test_dir.parent.parent / "cifs"

        # Check if test CIF files exist
        self.test_cifs = []
        if self.cifs_dir.exists():
            for cif_file in self.cifs_dir.glob("*.cif"):
                if cif_file.is_file():
                    self.test_cifs.append(cif_file)

    def test_generate_config(self):
        """Test configuration generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            # Generate default config
            cmd = [sys.executable, "-m", "mcli.cli", "generate-config", "-o", config_file]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)

            self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

            # Check that config file was created and contains expected keys
            with open(config_file, 'r') as f:
                config = json.load(f)

            expected_keys = [
                'spacing', 'h', 'mask_thickness', 'threshold_abs',
                'min_distance', 'minimum_window_separation', 'wall_thickness',
                'extra_padding', 'min_region_size', 'compute_connections',
                'return_padded_regions', 'return_dgrid', 'disable_tqdm'
            ]

            for key in expected_keys:
                self.assertIn(key, config, f"Missing key: {key}")

        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_generate_config_with_overrides(self):
        """Test configuration generation with parameter overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            # Generate config with custom parameters
            cmd = [
                sys.executable, "-m", "mcli.cli", "generate-config",
                "-o", config_file,
                "--spacing", "0.4",
                "--mask-thickness", "0.8",
                "--threshold-abs", "2.0"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)

            self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

            # Check that custom values were applied
            with open(config_file, 'r') as f:
                config = json.load(f)

            self.assertEqual(config['spacing'], 0.4)
            self.assertEqual(config['mask_thickness'], 0.8)
            self.assertEqual(config['threshold_abs'], 2.0)

        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    @unittest.skipIf(len([]) == 0, "No test CIF files available")
    def test_process_single_cif(self):
        """Test processing a single CIF file."""
        if not self.test_cifs:
            self.skipTest("No test CIF files available")

        cif_file = self.test_cifs[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_results.npz"

            # Process the CIF file
            cmd = [
                sys.executable, "-m", "mcli.cli", "process",
                str(cif_file),
                "-o", temp_dir
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)

            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

            self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

            # Check that output file was created
            self.assertTrue(output_file.exists(), "Output file was not created")

            # Check that output file contains expected data
            import numpy as np
            data = np.load(output_file)

            # Check for required keys
            required_keys = [
                'maxima_pbc', 'regions_pbc', 'maxima_info', 'regions',
                'close_or_inside_maxima', 'close_or_inside_regions',
                'unique_regions', 'input_params', 'timing_results'
            ]

            for key in required_keys:
                self.assertIn(key, data, f"Missing key in output: {key}")

    def test_config_building_interactive(self):
        """Test that config-building command exists (cannot test interactive fully)."""
        cmd = [sys.executable, "-m", "mcli.cli", "config-building", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")

    def test_cli_help(self):
        """Test that CLI help works."""
        cmd = [sys.executable, "-m", "mcli.cli", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)

        self.assertEqual(result.returncode, 0, f"Command failed: {result.stderr}")
        self.assertIn("Mofography CLI Tool", result.stdout)
        self.assertIn("process", result.stdout)
        self.assertIn("generate-config", result.stdout)
        self.assertIn("config-building", result.stdout)


if __name__ == '__main__':
    unittest.main()

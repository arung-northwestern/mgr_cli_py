#!/usr/bin/env python3
"""
MCLI Standalone - CLI Entry Point without numpy dependency

This version is designed for standalone executable builds where minimizing
dependencies is critical. It implements a custom .npz writer that doesn't
require numpy.

Usage for standalone builds:
1. Set USE_STANDALONE = True in the main CLI
2. Or use this standalone version directly
3. Build with tools like PyInstaller: pyinstaller --onefile cli_standalone.py
"""

import argparse
import sys
import os
import json
import struct
import zipfile
import pickle
import array
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the Rust library (will be available when built)
# Note: Import is deferred to avoid immediate failure during CLI help/version commands


class MCLIStandalone:
    """Standalone CLI class for MOF analysis (no numpy dependency)."""

    # Default configuration values
    DEFAULTS = {
        'spacing': 0.5,
        'h': 0.2,
        'mask_thickness': 1.5,
        'threshold_abs': 1.5,
        'min_distance': 1.5,
        'minimum_window_separation': 1.0,
        'compactness': 0,  # Not exposed in CLI
        'wall_thickness': 0.08,
        'extra_padding': 1.0,
        'min_region_size': 0.01,
        'disable_tqdm': False,
        'compute_connections': False,
        'return_dgrid': False,
        'return_padded_regions': False,
    }

    def __init__(self):
        """Initialize the standalone CLI."""
        self._rust_lib = None

    def _get_rust_library(self):
        """Lazily import the Rust library."""
        if self._rust_lib is None:
            try:
                import mcli_rust
                self._rust_lib = mcli_rust
            except ImportError as e:
                print("Error: mcli_rust Rust library not found. Please build the project first.")
                print("Run: pip install -e .")
                print(f"Original error: {e}")
                sys.exit(1)
        return self._rust_lib

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = self.DEFAULTS.copy()

        # Ensure all required keys are present
        for key, default_value in self.DEFAULTS.items():
            if key not in config:
                config[key] = default_value

        return config

    def _save_config(self, config: Dict[str, Any], output_file: Path):
        """Save configuration to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {output_file}")

    def process_single_cif(self, cif_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single CIF file using the Rust library."""
        rust_lib = self._get_rust_library()

        result = rust_lib.do_periodic_segmentation_rust(
            path_to_struct_file=str(cif_file),
            spacing=config['spacing'],
            mask_thickness=config['mask_thickness'],
            h=config['h'],
            min_distance=config['min_distance'],
            threshold_abs=config['threshold_abs'],
            return_padded_regions=config['return_padded_regions'],
            minimum_window_separation=config['minimum_window_separation'],
            wall_thickness=config['wall_thickness'],
            extra_padding=config['extra_padding'],
            compactness=config['compactness'],
            min_region_size=config['min_region_size'],
            disable_tqdm=config['disable_tqdm'],
            compute_connections=config['compute_connections'],
            return_dgrid=config['return_dgrid'],
        )

        # Convert the result to a Python dictionary
        return dict(result)

    def save_results_standalone(self, result: Dict[str, Any], output_file: Path):
        """Save results to a compressed file without numpy dependency."""
        def write_npy_header(f, shape, dtype):
            """Write a minimal .npy header for compatibility."""
            if dtype == 'float64':
                dtype_str = '<f8'
            elif dtype == 'float32':
                dtype_str = '<f4'
            elif dtype == 'int64':
                dtype_str = '<i8'
            elif dtype == 'int32':
                dtype_str = '<i4'
            elif dtype == 'uint32':
                dtype_str = '<u4'
            else:
                dtype_str = '|O'  # Object array for non-numeric data

            header = f"{{'descr': '{dtype_str}', 'fortran_order': False, 'shape': {shape}}}"
            header_padded = header + '\n' * ((16 - (len(header) + 1) % 16) % 16)
            f.write(b'\x93NUMPY\x01\x00')  # Magic + version
            f.write(struct.pack('<H', len(header_padded)))
            f.write(header_padded.encode('ascii'))

        def save_array_to_npy(f, array_data, key):
            """Save array data in .npy format."""
            if hasattr(array_data, 'shape') and hasattr(array_data, 'dtype'):
                # It's a numpy array from Rust
                shape = array_data.shape
                dtype = str(array_data.dtype)
                write_npy_header(f, shape, dtype)
                f.write(array_data.tobytes())
            elif isinstance(array_data, (list, tuple)):
                # Convert list/tuple to bytes
                if all(isinstance(x, (int, float)) for x in array_data):
                    # Numeric list
                    if all(isinstance(x, int) for x in array_data):
                        arr = array.array('i', array_data)
                        write_npy_header(f, (len(array_data),), 'int32')
                    else:
                        arr = array.array('f', array_data)
                        write_npy_header(f, (len(array_data),), 'float64')
                    f.write(arr.tobytes())
                else:
                    # Non-numeric data - save as pickled object
                    write_npy_header(f, (), '|O')
                    pickle.dump(array_data, f)
            else:
                # Other Python objects - save as pickled
                write_npy_header(f, (), '|O')
                pickle.dump(array_data, f)

        # Save as .npz (zip of .npy files)
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for key, value in result.items():
                # Write each array to a temporary .npy file in the zip
                npy_filename = f"{key}.npy"
                with zf.open(npy_filename, 'w') as f:
                    save_array_to_npy(f, value, key)

    def show_banner(self):
        """Display the Mofography CLI banner."""
        banner = r"""
 ███╗   ███╗ ██████╗ ███████╗ ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██╗   ██╗
 ████╗ ████║██╔═══██╗██╔════╝██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║╚██╗ ██╔╝
 ██╔████╔██║██║   ██║█████╗  ██║   ██║██║  ███╗██████╔╝███████║██████╔╝███████║ ╚████╔╝ 
 ██║╚██╔╝██║██║   ██║██╔══╝  ██║   ██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║  ╚██╔╝  
 ██║ ╚═╝ ██║╚██████╔╝██║     ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║   ██║   
 ╚═╝     ╚═╝ ╚═════╝ ╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   

Mofography CLI Tool - High-Performance MOF Analysis
Developed by © Dr. Arun Gopalan
        """
        print(banner)

    def run(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description='Mofography CLI Tool - High-Performance MOF Analysis (Standalone Version)',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  mcli process my_structure.cif
  mcli process *.cif -o ./results/
  mcli generate-config -o config.json
  mcli config-building
  mcli --banner
            """
        )
        
        # Global banner flag
        parser.add_argument(
            '--banner', action='store_true',
            help='Display the Mofography CLI banner'
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Process command
        process_parser = subparsers.add_parser('process', help='Process CIF files')
        process_parser.add_argument('cif_files', nargs='+', help='CIF files to process')
        process_parser.add_argument('-c', '--config', type=Path, help='Configuration file')
        process_parser.add_argument('-o', '--output-dir', type=Path, default=Path('./results'),
                                   help='Output directory')
        process_parser.add_argument('-s', '--spacing', type=float, help='Grid spacing in Å')
        process_parser.add_argument('-H', '--h', type=float, help='Minimum pocket size difference in Å')
        process_parser.add_argument('-mt', '--mask-thickness', type=float, help='Void space mask thickness in Å')
        process_parser.add_argument('-ta', '--threshold-abs', type=float, help='Minimum maxima size threshold in Å')
        process_parser.add_argument('-md', '--min-distance', type=float, help='Minimum distance between maxima in Å')
        process_parser.add_argument('-wt', '--wall-thickness', type=float, help='Wall region percentage (0-1)')
        process_parser.add_argument('-ep', '--extra-padding', type=float, help='Additional grid padding in Å')
        process_parser.add_argument('-mrs', '--min-region-size', type=float, help='Minimum region size percentage')
        process_parser.add_argument('--disable-tqdm', action='store_true', help='Disable progress bars')
        process_parser.add_argument('--compute-connections', action='store_true', help='Compute pore connections')
        process_parser.add_argument('--return-dgrid', action='store_true', help='Return distance grid')
        process_parser.add_argument('--return-padded-regions', action='store_true', help='Return padded regions')

        # Config generation command
        config_parser = subparsers.add_parser('generate-config', help='Generate configuration file')
        config_parser.add_argument('-o', '--output', type=Path, default=Path('config.json'),
                                  help='Output configuration file')
        config_parser.add_argument('-s', '--spacing', type=float, help='Grid spacing in Å')
        config_parser.add_argument('-H', '--h', type=float, help='Minimum pocket size difference in Å')
        config_parser.add_argument('-mt', '--mask-thickness', type=float, help='Void space mask thickness in Å')
        config_parser.add_argument('-ta', '--threshold-abs', type=float, help='Minimum maxima size threshold in Å')
        config_parser.add_argument('-md', '--min-distance', type=float, help='Minimum distance between maxima in Å')
        config_parser.add_argument('-wt', '--wall-thickness', type=float, help='Wall region percentage (0-1)')
        config_parser.add_argument('-ep', '--extra-padding', type=float, help='Additional grid padding in Å')
        config_parser.add_argument('-mrs', '--min-region-size', type=float, help='Minimum region size percentage')
        config_parser.add_argument('--disable-tqdm', action='store_true', help='Disable progress bars')
        config_parser.add_argument('--compute-connections', action='store_true', help='Compute pore connections')
        config_parser.add_argument('--return-dgrid', action='store_true', help='Return distance grid')
        config_parser.add_argument('--return-padded-regions', action='store_true', help='Return padded regions')

        # Interactive config builder
        config_build_parser = subparsers.add_parser('config-building', help='Interactive configuration builder')

        args = parser.parse_args()

        # Handle banner flag
        if args.banner:
            self.show_banner()
            if not args.command:
                return

        if not args.command:
            parser.print_help()
            return

        if args.command == 'process':
            self._handle_process(args)
        elif args.command == 'generate-config':
            self._handle_generate_config(args)
        elif args.command == 'config-building':
            self._handle_config_building()

    def _handle_process(self, args):
        """Handle the process command."""
        # Load base configuration
        config = self._load_config(args.config)

        # Override with command line arguments
        for key in self.DEFAULTS.keys():
            if hasattr(args, key) and getattr(args, key) is not None:
                config[key] = getattr(args, key)

        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(args.cif_files)} CIF file(s)")

        # Process each CIF file
        for cif_file in args.cif_files:
            cif_path = Path(cif_file)
            if not cif_path.exists():
                print(f"Warning: {cif_file} not found, skipping")
                continue

            print(f"Processing {cif_file}...")

            try:
                result = self.process_single_cif(cif_path, config)

                # Save results
                output_file = args.output_dir / f"{cif_path.stem}_results.npz"
                self.save_results_standalone(result, output_file)
                print(f"Results saved to {output_file}")

            except Exception as e:
                print(f"Error processing {cif_file}: {e}")

    def _handle_generate_config(self, args):
        """Handle the generate-config command."""
        config = self.DEFAULTS.copy()

        # Override with command line arguments
        for key in self.DEFAULTS.keys():
            if hasattr(args, key) and getattr(args, key) is not None:
                config[key] = getattr(args, key)

        self._save_config(config, args.output)

    def _handle_config_building(self):
        """Handle the interactive config-building command."""
        print("Interactive configuration builder")
        print("Enter values (press Enter for defaults):")

        config = {}
        for key, default in self.DEFAULTS.items():
            if isinstance(default, bool):
                value = input(f"{key} [{default}]: ").strip()
                config[key] = value.lower() in ('true', '1', 'yes', 'y') if value else default
            elif isinstance(default, (int, float)):
                value = input(f"{key} [{default}]: ").strip()
                try:
                    config[key] = type(default)(value) if value else default
                except ValueError:
                    print(f"Invalid value, using default: {default}")
                    config[key] = default
            else:
                config[key] = default

        output_file = Path('config.json')
        self._save_config(config, output_file)


def main():
    """Main entry point."""
    cli = MCLIStandalone()
    cli.run()


if __name__ == '__main__':
    main()

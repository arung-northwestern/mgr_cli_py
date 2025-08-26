#!/usr/bin/env python3
"""
MCLI - Main CLI Entry Point

Command-line interface for MOF analysis with three main commands:
- process: Process one or more CIF files
- config-building: Interactive configuration builder
- generate-config: Generate configuration files with optional overrides

For standalone builds: Set USE_STANDALONE = True to remove numpy dependency
"""

import argparse
import sys
import os
import json
import struct
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional numpy import - only needed for numpy-based .npz saving
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Import the Rust library (will be available when built)
# Note: Import is deferred to avoid immediate failure during CLI help/version commands


class MCLI:
    """Main CLI class for MOF analysis."""

    # Flag to use standalone (numpy-free) implementation
    USE_STANDALONE = False  # Set to True for standalone builds

    # Default configuration values
    DEFAULTS = {
        'spacing': 0.5,
        'h': 0.2,
        'mask_thickness': 0.5,
        'threshold_abs': 1.5,
        'min_distance': 1.5,
        'minimum_window_separation': 1.0,
        'compactness': 0,  # Not exposed in CLI
        'wall_thickness': 0.08,
        'extra_padding': 2.0,
        'min_region_size': 0.01,
        'disable_tqdm': False,
        'compute_connections': True,
        'return_padded_regions': False,
        'return_dgrid': True,
    }

    def __init__(self):
        """Initialize the CLI."""
        self._rust_lib = None

    def _get_rust_library(self):
        """Lazily import the Rust library."""
        if self._rust_lib is None:
            try:
                import mgr_cli_py
                self._rust_lib = mgr_cli_py
            except ImportError as e:
                print("Error: mgr_cli_py Rust library not found. Please build the project first.")
                print("Run: pip install -e .")
                print(f"Original error: {e}")
                sys.exit(1)
        return self._rust_lib

    def run(self):
        """Run the CLI application."""
        parser = argparse.ArgumentParser(
            description="MCLI - MOF Analysis CLI Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s process structure.cif
  %(prog)s process *.cif -o ./results/
  %(prog)s generate-config -mt 1.0 -ta 2.0 > config.json
  %(prog)s config-building
            """
        )

        # Global options
        parser.add_argument(
            '--version', action='version',
            version=f'%(prog)s {__import__("mcli").__version__}'
        )

        # Create subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Process command
        process_parser = subparsers.add_parser(
            'process',
            help='Process one or more CIF files'
        )
        process_parser.add_argument(
            'cif_files', nargs='*',
            help='CIF files to process'
        )
        process_parser.add_argument(
            '-i', '--input',
            help='Input CIF file (alternative to positional arguments)'
        )
        process_parser.add_argument(
            '-o', '--output-dir', default='.',
            help='Output directory for results (default: current directory)'
        )
        process_parser.add_argument(
            '-c', '--config',
            help='Configuration file (JSON format)'
        )

        # Add all the parameter options with their shorthands
        param_group = process_parser.add_argument_group('analysis parameters')
        param_group.add_argument(
            '-s', '--spacing', type=float,
            help=f'Grid spacing in Å (default: {self.DEFAULTS["spacing"]})'
        )
        param_group.add_argument(
            '-H', '--h', type=float,
            help=f'Minimum difference between pocket sizes in Å (default: {self.DEFAULTS["h"]})'
        )
        param_group.add_argument(
            '-mt', '--mask-thickness', type=float,
            help=f'Mask thickness in Å (default: {self.DEFAULTS["mask_thickness"]})'
        )
        param_group.add_argument(
            '-ta', '--threshold-abs', type=float,
            help=f'Minimum maxima size in Å (default: {self.DEFAULTS["threshold_abs"]})'
        )
        param_group.add_argument(
            '-md', '--min-distance', type=float,
            help=f'Minimum distance between maxima in Å (default: {self.DEFAULTS["min_distance"]})'
        )
        param_group.add_argument(
            '-ms', '--minimum-window-separation', type=float,
            help=f'Minimum separation between windows in Å (default: {self.DEFAULTS["minimum_window_separation"]})'
        )
        param_group.add_argument(
            '-wt', '--wall-thickness', type=float,
            help=f'Wall thickness percentage (default: {self.DEFAULTS["wall_thickness"]})'
        )
        param_group.add_argument(
            '-ep', '--extra-padding', type=float,
            help=f'Extra padding in Å (default: {self.DEFAULTS["extra_padding"]})'
        )
        param_group.add_argument(
            '-mrs', '--min-region-size', type=float,
            help=f'Minimum region size percentage (default: {self.DEFAULTS["min_region_size"]})'
        )

        # Boolean flags
        flag_group = process_parser.add_argument_group('flags')
        flag_group.add_argument(
            '--no-connections', action='store_true',
            help='Disable connection computation'
        )
        flag_group.add_argument(
            '--return-padded-regions', action='store_true',
            help='Return padded regions in output'
        )
        flag_group.add_argument(
            '--return-dgrid', action='store_true',
            help='Return distance grid in output'
        )
        flag_group.add_argument(
            '--disable-tqdm', action='store_true',
            help='Disable progress bars'
        )

        # Generate-config command
        config_parser = subparsers.add_parser(
            'generate-config',
            help='Generate default configuration file'
        )
        config_parser.add_argument(
            '-o', '--output', default='config.json',
            help='Output configuration file (default: config.json)'
        )
        # Add the same parameter options for generate-config
        for action in param_group._group_actions:
            config_parser._add_action(action)

        # Config-building command (interactive)
        subparsers.add_parser(
            'config-building',
            help='Interactive configuration builder'
        )

        # Parse arguments
        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Execute the appropriate command
        if args.command == 'process':
            self.process_command(args)
        elif args.command == 'generate-config':
            self.generate_config_command(args)
        elif args.command == 'config-building':
            self.config_building_command(args)

    def process_command(self, args):
        """Handle the process command."""
        # Load configuration if provided
        config = self.load_config(args.config) if args.config else {}

        # Override config with command line arguments
        config = self.update_config_from_args(config, args)

        # Ensure output directory exists
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each CIF file
        cif_files = []

        # Handle -i flag
        if hasattr(args, 'input') and args.input:
            cif_files.append(Path(args.input))

        # Handle positional arguments
        if args.cif_files:
            for pattern in args.cif_files:
                path = Path(pattern)
                if path.is_file():
                    cif_files.append(path)
                elif '*' in pattern:
                    # Handle glob patterns
                    import glob
                    matches = glob.glob(pattern)
                    cif_files.extend([Path(f) for f in matches])

        if not cif_files:
            print("Error: No CIF files specified. Use -i flag or positional arguments.")
            print("Examples:")
            print("  mcli process structure.cif")
            print("  mcli process -i structure.cif")
            print("  mcli process *.cif")
            sys.exit(1)

        print(f"Found {len(cif_files)} CIF file(s) to process")

        for cif_file in cif_files:
            if not cif_file.exists():
                print(f"Warning: {cif_file} does not exist, skipping")
                continue

            print(f"\nProcessing: {cif_file}")
            try:
                result = self.process_single_cif(cif_file, config)

                # Generate output filename
                output_file = output_dir / f"{cif_file.stem}_results.npz"

                # Save results
                self.save_results(result, output_file)
                print(f"Results saved to: {output_file}")

            except Exception as e:
                print(f"Error processing {cif_file}: {e}")
                continue

    def generate_config_command(self, args):
        """Handle the generate-config command."""
        # Start with defaults
        config = self.DEFAULTS.copy()

        # Override with command line arguments
        config = self.update_config_from_args(config, args)

        # Save to file or stdout
        if args.output == '-':
            json.dump(config, sys.stdout, indent=2)
        else:
            with open(args.output, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to: {args.output}")

    def config_building_command(self, args):
        """Handle the interactive config-building command."""
        print("Interactive Configuration Builder")
        print("=" * 40)

        config = {}

        # Interactive prompts for each parameter
        prompts = {
            'spacing': f'Grid spacing in Å (default: {self.DEFAULTS["spacing"]})',
            'h': f'Minimum difference between pocket sizes in Å (default: {self.DEFAULTS["h"]})',
            'mask_thickness': f'Mask thickness in Å (default: {self.DEFAULTS["mask_thickness"]})',
            'threshold_abs': f'Minimum maxima size in Å (default: {self.DEFAULTS["threshold_abs"]})',
            'min_distance': f'Minimum distance between maxima in Å (default: {self.DEFAULTS["min_distance"]})',
            'minimum_window_separation': f'Minimum separation between windows in Å (default: {self.DEFAULTS["minimum_window_separation"]})',
            'wall_thickness': f'Wall thickness percentage (default: {self.DEFAULTS["wall_thickness"]})',
            'extra_padding': f'Extra padding in Å (default: {self.DEFAULTS["extra_padding"]})',
            'min_region_size': f'Minimum region size percentage (default: {self.DEFAULTS["min_region_size"]})',
        }

        for key, prompt in prompts.items():
            while True:
                try:
                    value = input(f"{prompt}: ").strip()
                    if not value:
                        config[key] = self.DEFAULTS[key]
                        break
                    else:
                        config[key] = float(value)
                        break
                except ValueError:
                    print("Please enter a valid number")

        # Boolean options
        bool_prompts = {
            'compute_connections': 'Compute connections between regions?',
            'return_padded_regions': 'Return padded regions in output?',
            'return_dgrid': 'Return distance grid in output?',
            'disable_tqdm': 'Disable progress bars?',
        }

        for key, prompt in bool_prompts.items():
            response = input(f"{prompt} (y/N): ").strip().lower()
            config[key] = response in ('y', 'yes', 'true', '1')

        # Save configuration
        output_file = input("Save configuration to file (default: config.json): ").strip()
        if not output_file:
            output_file = 'config.json'

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to: {output_file}")

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            return json.load(f)

    def update_config_from_args(self, config: Dict[str, Any], args) -> Dict[str, Any]:
        """Update configuration with command line arguments."""
        # Copy defaults first
        result = self.DEFAULTS.copy()
        result.update(config)

        # Override with command line arguments
        arg_mapping = {
            'spacing': 'spacing',
            'h': 'h',
            'mask_thickness': 'mask_thickness',
            'threshold_abs': 'threshold_abs',
            'min_distance': 'min_distance',
            'minimum_window_separation': 'minimum_window_separation',
            'wall_thickness': 'wall_thickness',
            'extra_padding': 'extra_padding',
            'min_region_size': 'min_region_size',
        }

        for arg_name, config_key in arg_mapping.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                result[config_key] = value

        # Handle boolean flags
        if hasattr(args, 'no_connections') and args.no_connections:
            result['compute_connections'] = False
        if hasattr(args, 'return_padded_regions') and args.return_padded_regions:
            result['return_padded_regions'] = True
        if hasattr(args, 'return_dgrid') and args.return_dgrid:
            result['return_dgrid'] = True
        if hasattr(args, 'disable_tqdm') and args.disable_tqdm:
            result['disable_tqdm'] = True

        return result

    def process_single_cif(self, cif_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single CIF file using the Rust library."""
        # Get the Rust library (will exit if not available)
        rust_lib = self._get_rust_library()

        # Call the Rust function
        result = rust_lib.do_periodic_segmentation_rust(
            path_to_struct_file=str(cif_file),
            spacing=config['spacing'],
            mask_thickness=config['mask_thickness'],
            h=config['h'],
            min_distance=config['min_distance'],
            threshold_abs=config['threshold_abs'],
            return_padded_regions=config['return_padded_regions'],
            minimum_window_separation=config.get('minimum_window_separation'),
            wall_thickness=config['wall_thickness'],
            extra_padding=config['extra_padding'],
            compactness=config['compactness'],
            min_region_size=config['min_region_size'],
            disable_tqdm=config['disable_tqdm'],
            compute_connections=config['compute_connections'],
            return_dgrid=config['return_dgrid'],
        )

        return result

    def save_results_numpy(self, result: Dict[str, Any], output_file: Path):
        """Save results to a compressed numpy file (requires numpy)."""
        # Most data from Rust is already in numpy array format
        # Only need minimal numpy usage for .npz format
        save_dict = {}

        for key, value in result.items():
            # Special handling for connections list which contains tuples
            # that aren't directly numpy-compatible
            if (isinstance(value, list) and len(value) > 0 and
                isinstance(value[0], tuple) and key == 'connections'):
                # Convert connections to a dictionary format that's numpy-friendly
                connections_dict = {
                    'regions': np.array([conn[0] for conn in value]),
                    'is_periodic': np.array([conn[1] for conn in value]),
                    'connection_type': np.array([conn[2] for conn in value]),
                    'centers': np.array([conn[3].flatten() for conn in value]),
                    'radii': np.array([conn[4].flatten() for conn in value])
                }
                save_dict[key] = connections_dict
            else:
                # Rust function already returns PyArray objects for numerical data
                # Python objects (dicts, lists) are handled as-is
                save_dict[key] = value

        np.savez_compressed(output_file, **save_dict)

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
                    import array
                    if all(isinstance(x, int) for x in array_data):
                        arr = array.array('i', array_data)
                        write_npy_header(f, (len(array_data),), 'int32')
                    else:
                        arr = array.array('f', array_data)
                        write_npy_header(f, (len(array_data),), 'float64')
                    f.write(arr.tobytes())
                else:
                    # Non-numeric data - save as pickled object
                    import pickle
                    write_npy_header(f, (), '|O')
                    pickle.dump(array_data, f)
            else:
                # Other Python objects - save as pickled
                import pickle
                write_npy_header(f, (), '|O')
                pickle.dump(array_data, f)

        # Save as .npz (zip of .npy files)
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for key, value in result.items():
                # Write each array to a temporary .npy file in the zip
                npy_filename = f"{key}.npy"
                with zf.open(npy_filename, 'w') as f:
                    save_array_to_npy(f, value, key)

    def save_results(self, result: Dict[str, Any], output_file: Path):
        """Save results using numpy or standalone implementation."""
        if self.USE_STANDALONE or not HAS_NUMPY:
            if not HAS_NUMPY:
                print("Warning: numpy not available, using standalone implementation")
            self.save_results_standalone(result, output_file)
        else:
            self.save_results_numpy(result, output_file)


def main():
    """Main entry point."""
    cli = MCLI()
    cli.run()


if __name__ == '__main__':
    main()

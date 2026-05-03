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
import time
import zipfile
import signal
from pathlib import Path
import subprocess
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import List, Dict, Any, Optional

# Import functions needed for pore diameter and percolation analysis
import sys
from pathlib import Path
# Add mgr_src to path to use reference functions directly
REPO_ROOT = Path(__file__).resolve().parents[2]
MGR_SRC_PATH = REPO_ROOT / "mgr_src"
if str(MGR_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(MGR_SRC_PATH))

def _import_mgr_module():
    """Import mgr lazily to keep CLI startup fast for non-analysis commands."""
    import mgr as _mgr
    return _mgr


class _MgrProxy:
    """Attribute proxy that imports mgr only on first use."""

    _module = None

    def _get_module(self):
        if self._module is None:
            self._module = _import_mgr_module()
        return self._module

    def __getattr__(self, item):
        return getattr(self._get_module(), item)


# Keep existing mgr.* call sites but avoid eager import during CLI startup.
mgr = _MgrProxy()

# Optional numpy import - only needed for numpy-based .npz saving
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

def compute_dgrid_rust(mat_atoms, spacing):
    """Compute distance grid using Rust backend."""
    try:
        # Try to use the reference implementation first
        from mgr.rust_fun import dgrid_from_atoms_rust
        return dgrid_from_atoms_rust(mat_atoms, spacing)
    except ImportError:
        try:
            # Fallback to mgr_visualize
            from mgr_visualize.mgr.rust_fun import dgrid_from_atoms_rust
            return dgrid_from_atoms_rust(mat_atoms, spacing)
        except ImportError:
            # Final fallback to direct mgrcli_rust call
            try:
                import mgrcli_rust
                # Convert ASE atoms to required format
                fpoints = mat_atoms.get_positions()
                cell = mat_atoms.get_cell().array.T  # Transpose to match expected format
                radii = np.array([1.0] * len(fpoints), dtype=np.float64)  # Default radii

                # Get grid dimensions
                nx, ny, nz = int(np.round(cell[0,0] / spacing)), int(np.round(cell[1,1] / spacing)), int(np.round(cell[2,2] / spacing))

                return mgrcli_rust.dgrid_from_atoms(fpoints, cell, radii, nx, ny, nz)
            except ImportError as e:
                print("Error: mgrcli_rust Rust library not found. Please build the project first.")
                print("Run: maturin build --release && pip install --force-reinstall target/wheels/*.whl")
                sys.exit(1)

def python_do_periodic_segmentation_rust(
    dgrid,
    mat_atoms,
    mask_thickness=0.5,
    h=0.2,
    min_distance=1.5,
    threshold_abs=1.5,
    return_padded_regions=False,
    spacing=None,
    minimum_window_separation=None,
    wall_thickness=0.05,
    extra_padding=5,
    compactness=0,
    min_region_size=0.001,
    disable_tqdm=False,
    compute_connections=True,
    return_dgrid=False,
    watershed_backend='rust_bucket',
):
    """Perform periodic segmentation using Rust backend."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Import directly from the reference MGR library
        from mgr.rust_fun import do_periodic_segmentation_rust
    except ImportError:
        try:
            # Fallback to mgr_visualize if mgr_src is not available
            from mgr_visualize.mgr.rust_fun import do_periodic_segmentation_rust
        except ImportError as e:
            print("Error: do_periodic_segmentation_rust function not found. Please ensure MGR library is available.")
            sys.exit(1)

    # Call the reference implementation (mgr.rust_fun)
    result = do_periodic_segmentation_rust(
        dgrid=dgrid,
        mat_atoms=mat_atoms,
        mask_thickness=mask_thickness,
        h=h,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        return_padded_regions=return_padded_regions,
        spacing=spacing,
        minimum_window_separation=minimum_window_separation,
        wall_thickness=wall_thickness,
        extra_padding=extra_padding,
        compactness=compactness,
        min_region_size=min_region_size,
        disable_tqdm=disable_tqdm,
        compute_connections=compute_connections,
        return_dgrid=return_dgrid,
        watershed_backend=watershed_backend,
    )

    return result

# Import the Rust library (will be available when built)
# Note: Import is deferred to avoid immediate failure during CLI help/version commands


def compute_pore_diameter_analysis(dgrid, mat_atoms, spacing=None, segmentation_results=None):
    """Compute pore diameter and percolation analysis.

    This function calculates:
    - Largest Cavity Diameter (LCD)
    - Pore Limiting Diameter (PLD)
    - Distance From Surface Distribution (DFSD)
    - Percolation history (using winding percolation if segmentation results available)

    Parameters:
    - dgrid: 3D distance grid
    - mat_atoms: ASE atoms object
    - spacing: Grid spacing (estimated if None)
    - segmentation_results: Full segmentation results dict (optional, for winding percolation)

    Returns:
    - Dictionary with pore diameter and percolation analysis results
    """

    if spacing is None:
        # Estimate spacing from dgrid and atoms
        spacing = np.min(
            mat_atoms.get_cell_lengths_and_angles()[:3] / (np.array(dgrid.shape) - 1)
        )

    print("Computing pore diameter and percolation analysis...")

    # Compute LCD, PLD, and DFSD using the morphography function
    try:
        lcd, pld, dfsd, _ = mgr.get_pore_dia_mgr(mat_atoms, dgrid=dgrid, kwargs=dict(spacing=spacing))
        print(f"  LCD: {lcd:.3f} Å")
        print(f"  PLD: {pld:.3f} Å")
        print(f"  DFSD: {dfsd:.3f} Å")
    except Exception as e:
        print(f"Warning: Pore diameter calculation failed: {e}")
        lcd, pld, dfsd = 0.0, 0.0, 0.0

    # Compute percolation history using winding percolation if available
    percolation_history = []
    pld_locations = np.array([])

    if segmentation_results is not None:
        # Try winding percolation first (requires maxima_info, connections, regions)
        required_keys = ['maxima_info', 'connections', 'regions']
        if all(key in segmentation_results for key in required_keys):
            try:
                from mgr.winding_percolation import compute_percolation_history_winding, update_results_with_percolation_history_winding

                # Prepare results dict for winding percolation
                results_for_winding = {
                    'dgrid': dgrid,
                    'maxima_info': segmentation_results['maxima_info'],
                    'connections': segmentation_results['connections'],
                    'regions': segmentation_results['regions'],
                    'mask_thickness': segmentation_results.get('kwargs', {}).get('mask_thickness', 1.5),
                }

                # Compute percolation history using winding percolation
                percolation_history, pld_locations, _ = compute_percolation_history_winding(
                    results_for_winding,
                    step=0.05,
                    min_threshold=None,
                    max_threshold=None,
                )
                print(f"  Percolation history points (winding): {len(percolation_history)}")
            except Exception as e:
                print(f"Warning: Winding percolation failed: {e}")
                percolation_history = []
                pld_locations = np.array([])
        else:
            print("Warning: Segmentation results incomplete for winding percolation")
    else:
        print("Info: No segmentation results provided, using fallback percolation analysis")

    # Fallback to original percolation computation if winding failed or not available
    if not percolation_history:
        try:
            percolation_history, pld_locations, _, _, _ = mgr.compute_percolation_history(
                dgrid, mat_atoms
            )
            print(f"  Percolation history points (fallback): {len(percolation_history)}")
        except Exception as e:
            print(f"Warning: Percolation analysis failed: {e}")
            percolation_history = []
            pld_locations = np.array([])

    # Prepare results
    results = {
        'lcd': float(lcd),
        'pld': float(pld),
        'dfsd': float(dfsd),
        'percolation_history': percolation_history,
        'pld_locations': pld_locations,
    }

    return results


def save_pore_diameter_csv(results, output_file):
    """Save pore diameter and percolation analysis results to CSV.

    Parameters:
    - results: Dictionary with pore diameter analysis results
    - output_file: Path to output CSV file
    """
    import csv

    if not results:
        print("No pore diameter results to save.")
        return

    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Metric', 'Value', 'Unit'])

            # Write pore diameter metrics
            writer.writerow(['LCD', f"{results.get('lcd', 'N/A'):.3f}", 'Å'])
            writer.writerow(['PLD', f"{results.get('pld', 'N/A'):.3f}", 'Å'])
            writer.writerow(['DFSD', f"{results.get('dfsd', 'N/A'):.3f}", 'Å'])
            writer.writerow(['Percolation history points', len(results.get('percolation_history', [])), 'count'])

        print(f"Pore diameter analysis saved to: {output_file}")

    except Exception as e:
        print(f"Error saving pore diameter CSV: {e}")


def get_pbc_groups(unmerged_regions, maxima_info):
    """
    Group unmerged regions based on periodic boundary conditions (PBC).

    This function organizes unmerged regions into groups based on their maxima
    indices. Each group is identified by a unique tuple of maxima indices
    (columns 2-4 of `maxima_info`), and contains the corresponding region indices
    (column 1 of `maxima_info`).

    Args:
        unmerged_regions (iterable): A collection of region indices that are not yet merged.
        maxima_info (array-like): A 2D array where each row contains:
            - Column 0: Region index
            - Columns 1-3: Maxima indices (used as grouping keys)
            - Column 4: Offsets in each direction
              You get this as output when you call the 'do_periodic_segmentation' on the distance grid (dgrid)

    Returns:
        dict: A dictionary where keys are tuples of maxima indices (columns 2-4),
            and values are sets of region indices (column 1) belonging to each group.
    """
    import numpy as np
    from collections import defaultdict

    # Create a dictionary where keys are tuples of maxima indices (cols 2-4)
    # and values are lists of corresponding indices (col 1)
    groups = defaultdict(list)

    for row in maxima_info:
        if row[0] in unmerged_regions:
            # Use tuple of columns 2-4 as key (converting to tuple for hashability)
            key = tuple(row[1:4])
            # Add the index (first column) to the corresponding group
            groups[key].append(int(row[0]))

    # Convert the dictionary values to sets if you want unique indices
    groups = {k: set(v) for k, v in groups.items()}
    return groups


def compute_auto_parameters(mat_atoms, spacing):
    """Automatically estimate optimal parameters from structure.

    This function estimates:
    - mask_thickness based on pore limiting radius
    - h_value from percolation history
    - surface_value based on mask_thickness

    Parameters:
    - mat_atoms: ASE atoms object
    - spacing: Grid spacing

    Returns:
    - Dictionary with estimated parameters
    """
    try:
        # Try to import from the reference library first
        from mgr.helper import estimate_h_from_percolation_history
    except ImportError:
        try:
            # Fallback to mgr_visualize
            from mgr_visualize.mgr.helper import estimate_h_from_percolation_history
        except ImportError as e:
            print(f"Warning: Could not import helper functions: {e}")
            return {}

    # Compute distance grid for analysis
    dgrid = compute_dgrid_rust(mat_atoms, spacing)

    # Estimate mask_thickness from pore limiting radius
    pore_limiting_radius = mgr.find_pore_limiting_radius(dgrid)
    mask_thickness = float(np.clip(pore_limiting_radius - 0.25, 0.1, 0.5))

    # Estimate h_value from percolation history
    h_value = float(max(estimate_h_from_percolation_history(dgrid) - 0.1, 0.2))

    # Estimate surface value
    surface_value = float(min(mask_thickness + 0.5, 2.0))

    return {
        'mask_thickness': mask_thickness,
        'h': h_value,
        'surface_value': surface_value,
    }


def compute_psd_analysis(dgrid, mat_atoms, regions_pbc=None, spacing=None,
                        num_samples_total=10000, num_samples_per_region=1000,
                        uniform_max_diameter=50.0, pocket_max_diameter=50.0,
                        uniform_num_bins=200, pocket_num_bins=200,
                        probe_center_min_radius=None):
    """Compute pore size distribution analysis.

    This function calculates:
    - Global PSD (uniform sampling)
    - Pocket-wise PSD (region-based sampling)

    Parameters:
    - dgrid: 3D distance grid
    - mat_atoms: ASE atoms object
    - regions_pbc: Region labels from segmentation (optional)
    - spacing: Grid spacing (estimated if None)
    - num_samples_total: Total samples for uniform PSD
    - num_samples_per_region: Samples per region for pocket PSD
    - uniform_max_diameter: Maximum diameter for uniform PSD
    - pocket_max_diameter: Maximum diameter for pocket PSD
    - uniform_num_bins: Number of bins for uniform PSD
    - pocket_num_bins: Number of bins for pocket PSD
    - probe_center_min_radius: If set and > 0, restrict PSD sample sites to dgrid ≥ this radius (uniform + pockets).

    Returns:
    - Dictionary with PSD analysis results
    """

    if spacing is None:
        # Estimate spacing from dgrid and atoms
        spacing = np.min(
            mat_atoms.get_cell_lengths_and_angles()[:3] / (np.array(dgrid.shape) - 1)
        )

    print("Computing pore size distribution analysis...")

    results = {}

    # Compute uniform PSD
    try:
        bins_uniform = np.linspace(0, float(uniform_max_diameter), int(uniform_num_bins) + 1)
        uniform_result = mgr.compute_psd_uniform_rust(
            dgrid,
            mat_atoms,
            num_samples_total=num_samples_total,
            bins=bins_uniform,
            density=False,
            return_fig=False,
            probe_center_min_radius=probe_center_min_radius,
        )
        # Handle different return formats
        if isinstance(uniform_result, tuple):
            uniform_hist = uniform_result[0]
        else:
            uniform_hist = uniform_result

        # Create uniform DataFrame-like structure
        uniform_data = {
            'bins': bins_uniform,
            'histogram': uniform_hist,
            'bin_centers': 0.5 * (bins_uniform[:-1] + bins_uniform[1:]),
            'num_samples': num_samples_total
        }

        results['psd_uniform'] = uniform_data
        print(f"  Uniform PSD: {num_samples_total} samples, {uniform_num_bins} bins")

    except Exception as e:
        print(f"Warning: Uniform PSD calculation failed: {e}")
        results['psd_uniform'] = None

    # Compute pocket-wise PSD (if regions are available)
    if regions_pbc is not None:
        try:
            bins_pocket = np.linspace(0, float(pocket_max_diameter), int(pocket_num_bins) + 1)
            pocket_result = mgr.compute_psds_pocketwise_rust(
                dgrid,
                regions_pbc,
                mat_atoms,
                num_samples_per_region=num_samples_per_region,
                bins=bins_pocket,
                return_fig=False,
                probe_center_min_radius=probe_center_min_radius,
            )
            # Handle different return formats
            if isinstance(pocket_result, tuple):
                pocket_psds = pocket_result[0]
            else:
                pocket_psds = pocket_result

            # Create pocket-wise data structure
            pocket_data = {
                'bins': bins_pocket,
                'bin_centers': 0.5 * (bins_pocket[:-1] + bins_pocket[1:]),
                'pocket_psds': pocket_psds,
                'num_samples_per_region': num_samples_per_region,
                'num_regions': len(pocket_psds) if pocket_psds is not None else 0
            }

            results['psd_pocket'] = pocket_data
            print(f"  Pocket PSD: {num_samples_per_region} samples/region, {len(pocket_psds) if pocket_psds is not None else 0} regions")

        except Exception as e:
            print(f"Warning: Pocket PSD calculation failed: {e}")
            results['psd_pocket'] = None
    else:
        print("  Pocket PSD: Skipped (no segmentation data available)")
        results['psd_pocket'] = None

    results['spacing'] = float(spacing)
    return results


def save_psd_csv(results, output_file, material_name="material"):
    """Save pore size distribution analysis results to CSV.

    Parameters:
    - results: Dictionary with PSD analysis results
    - output_file: Path to output CSV file
    - material_name: Name of the material for filename
    """
    import csv

    if not results:
        print("No PSD results to save.")
        return

    try:
        # Save uniform PSD CSV
        uniform_data = results.get('psd_uniform')
        if uniform_data is not None:
            uniform_csv_file = output_file.parent / f"{output_file.stem}_psd_uniform.csv"

            with open(uniform_csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['bin_start', 'bin_end', 'bin_center', 'count'])

                bins = uniform_data['bins']
                hist = uniform_data['histogram']

                for i in range(len(bins) - 1):
                    writer.writerow([bins[i], bins[i+1], uniform_data['bin_centers'][i], hist[i]])

            print(f"Uniform PSD saved to: {uniform_csv_file}")

        # Save pocket-wise PSD CSV
        pocket_data = results.get('psd_pocket')
        if pocket_data is not None and pocket_data['pocket_psds'] is not None:
            pocket_csv_file = output_file.parent / f"{output_file.stem}_psd_pockets.csv"

            with open(pocket_csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['region_label', 'bin_start', 'bin_end', 'bin_center', 'count'])

                bins = pocket_data['bins']
                bin_centers = pocket_data['bin_centers']

                pocket_psds = pocket_data['pocket_psds']
                try:
                    # Handle different data structures for pocket_psds
                    if hasattr(pocket_psds, 'items'):  # Dictionary-like
                        for label, region_data in pocket_psds.items():
                            # Extract histogram from region_data dictionary
                            if isinstance(region_data, dict) and 'hist' in region_data:
                                hist = region_data['hist']
                            else:
                                hist = region_data  # Fallback if it's already just the histogram

                            for i in range(len(bins) - 1):
                                writer.writerow([int(label), bins[i], bins[i+1], bin_centers[i], hist[i]])
                    else:  # List-like or array-like
                        for label, hist in enumerate(pocket_psds):
                            for i in range(len(bins) - 1):
                                writer.writerow([int(label), bins[i], bins[i+1], bin_centers[i], hist[i]])
                except Exception as e:
                    print(f"Warning: Error processing pocket PSD data for CSV: {e}")

            print(f"Pocket PSD saved to: {pocket_csv_file}")

    except Exception as e:
        print(f"Error saving PSD CSV: {e}")


def compute_surface_area_analysis(results_dict, mat_atoms, probe_radius=None):
    """
    Compute surface area analysis from results dictionary.

    Parameters:
    - results_dict: Dictionary containing analysis results
    - mat_atoms: ASE atoms object
    - probe_radius: Probe radius for surface area calculation

    Returns:
    - Dictionary with surface area analysis results
    """
    try:
        # Convert results dict to expected format for reference functions
        # Our CLI uses 'dgrid' but reference functions expect 'distance_grid'
        modified_results = results_dict.copy()
        if 'dgrid' in modified_results:
            modified_results['distance_grid'] = modified_results['dgrid']

        from mgr.compute import compute_surface_area_from_results
        return compute_surface_area_from_results(modified_results, mat_atoms, probe_radius)
    except ImportError:
        try:
            # Convert results dict to expected format for reference functions
            modified_results = results_dict.copy()
            if 'dgrid' in modified_results:
                modified_results['distance_grid'] = modified_results['dgrid']

            from mgr_visualize.mgr.compute import compute_surface_area_from_results
            return compute_surface_area_from_results(modified_results, mat_atoms, probe_radius)
        except ImportError:
            print("Warning: Surface area computation functions not available. Please ensure mgr_src is properly installed.")
            return None
    except Exception as e:
        print(f"Warning: Surface area computation failed with error: {e}")
        print("This might be due to inappropriate parameters for this material structure.")
        print("Try using --auto-params to automatically estimate optimal parameters, or adjust:")
        print("  - --mask-thickness (default: 1.5)")
        print("  - --threshold-abs (default: 1.5)")
        print("  - --spacing (default: 0.5)")
        print("  - --h (default: 0.2)")
        return None


def compute_void_volume_analysis(results_dict, mat_atoms, probe_radius=None):
    """
    Compute void volume analysis from results dictionary.

    Parameters:
    - results_dict: Dictionary containing analysis results
    - mat_atoms: ASE atoms object
    - probe_radius: Probe radius for volume calculation

    Returns:
    - Dictionary with void volume analysis results
    """
    try:
        # Convert results dict to expected format for reference functions
        # Our CLI uses 'dgrid' but reference functions expect 'distance_grid'
        modified_results = results_dict.copy()
        if 'dgrid' in modified_results:
            modified_results['distance_grid'] = modified_results['dgrid']

        from mgr.compute import compute_void_volume_from_results
        return compute_void_volume_from_results(modified_results, mat_atoms, probe_radius)
    except ImportError:
        try:
            # Convert results dict to expected format for reference functions
            modified_results = results_dict.copy()
            if 'dgrid' in modified_results:
                modified_results['distance_grid'] = modified_results['dgrid']

            from mgr_visualize.mgr.compute import compute_void_volume_from_results
            return compute_void_volume_from_results(modified_results, mat_atoms, probe_radius)
        except ImportError:
            print("Warning: Void volume computation functions not available. Please ensure mgr_src is properly installed.")
            return None
    except Exception as e:
        print(f"Warning: Void volume computation failed with error: {e}")
        print("This might be due to inappropriate parameters for this material structure.")
        print("Try using --auto-params to automatically estimate optimal parameters, or adjust:")
        print("  - --mask-thickness (default: 1.5)")
        print("  - --threshold-abs (default: 1.5)")
        print("  - --spacing (default: 0.5)")
        print("  - --h (default: 0.2)")
        return None


def save_surface_area_csv(results, output_file, material_name="material"):
    """Save surface area analysis results to CSV.

    Parameters:
    - results: Dictionary with surface area analysis results
    - output_file: Path to output CSV file
    - material_name: Name of the material for filename
    """
    import csv

    if not results:
        print("No surface area results to save.")
        return

    try:
        # Save surface area CSV
        sa_csv_file = output_file.parent / f"{output_file.stem}_surface_area.csv"

        with open(sa_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write global metrics
            writer.writerow(['Type', 'Category', 'Label', 'Surface_Area_m2_g', 'Unit'])

            # Global surface areas
            writer.writerow(['Global', 'Total', 'All', f"{results.get('total_surface_area', 0):.4f}", 'm²/g'])
            writer.writerow(['Global', 'Accessible', 'All', f"{results.get('accessible_surface_area', 0):.4f}", 'm²/g'])
            writer.writerow(['Global', 'Inaccessible', 'All', f"{results.get('total_surface_area', 0) - results.get('accessible_surface_area', 0):.4f}", 'm²/g'])

            # Accessible fraction
            total_sa = results.get('total_surface_area', 0)
            accessible_sa = results.get('accessible_surface_area', 0)
            if total_sa > 0:
                accessible_fraction = accessible_sa / total_sa
                writer.writerow(['Global', 'Fraction', 'Accessible', f"{accessible_fraction:.4f}", 'ratio'])

            # Point probe reference
            point_probe_ref = results.get('point_probe_reference', {})
            point_probe_sa = point_probe_ref.get('surface_area', 0)
            point_probe_fraction = point_probe_ref.get('surface_area_fraction', 0)
            writer.writerow(['Global', 'Point Probe', 'All', f"{point_probe_sa:.4f}", 'm²/g'])
            writer.writerow(['Global', 'Fraction', 'Point Probe', f"{point_probe_fraction:.4f}", 'ratio'])

            # Region-wise surface areas
            by_region = results.get('by_region', {})
            if by_region:
                writer.writerow([])  # Empty row
                writer.writerow(['Region', 'Individual', 'Label', 'Surface_Area_m2_g', 'Unit'])
                for region_id, area in by_region.items():
                    writer.writerow(['Region', 'Individual', f"region_{region_id}", f"{area:.4f}", 'm²/g'])

            # Pocket-wise surface areas
            by_pocket = results.get('by_pocket', {})
            if by_pocket:
                writer.writerow([])  # Empty row
                writer.writerow(['Pocket', 'Complete', 'Label', 'Surface_Area_m2_g', 'Unit'])
                for pocket_id, area in by_pocket.items():
                    writer.writerow(['Pocket', 'Complete', f"{pocket_id}", f"{area:.4f}", 'm²/g'])

            # Percolated cluster surface areas
            by_percolated_cluster = results.get('by_percolated_cluster', {})
            if by_percolated_cluster:
                writer.writerow([])  # Empty row
                writer.writerow(['Cluster', 'Percolated', 'Label', 'Surface_Area_m2_g', 'Unit'])
                for cluster_id, area in by_percolated_cluster.items():
                    writer.writerow(['Cluster', 'Percolated', f"{cluster_id}", f"{area:.4f}", 'm²/g'])

            # Probe radius used
            probe_radius = results.get('probe_radius', 'N/A')
            writer.writerow([])  # Empty row
            writer.writerow(['Parameter', 'Probe_Radius', 'Value', f"{probe_radius}", 'Å'])

        print(f"Surface area analysis saved to: {sa_csv_file}")

    except Exception as e:
        print(f"Error saving surface area CSV: {e}")


def save_void_volume_csv(results, output_file, material_name="material"):
    """Save void volume analysis results to CSV.

    Parameters:
    - results: Dictionary with void volume analysis results
    - output_file: Path to output CSV file
    - material_name: Name of the material for filename
    """
    import csv

    if not results:
        print("No void volume results to save.")
        return

    try:
        # Save void volume CSV
        vol_csv_file = output_file.parent / f"{output_file.stem}_void_volume.csv"

        with open(vol_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write global metrics
            writer.writerow(['Type', 'Category', 'Label', 'Void_Volume_cm3_g', 'Unit'])

            # Global void volumes
            writer.writerow(['Global', 'Total', 'All', f"{results.get('total_void_volume', 0):.4f}", 'cm³/g'])
            writer.writerow(['Global', 'Accessible', 'All', f"{results.get('accessible_void_volume', 0):.4f}", 'cm³/g'])
            writer.writerow(['Global', 'Inaccessible', 'All', f"{results.get('total_void_volume', 0) - results.get('accessible_void_volume', 0):.4f}", 'cm³/g'])

            # Accessible fraction
            total_vol = results.get('total_void_volume', 0)
            accessible_vol = results.get('accessible_void_volume', 0)
            if total_vol > 0:
                accessible_fraction = accessible_vol / total_vol
                writer.writerow(['Global', 'Fraction', 'Accessible', f"{accessible_fraction:.4f}", 'ratio'])

            # Point probe reference
            point_probe_ref = results.get('point_probe_reference', {})
            point_probe_vol = point_probe_ref.get('void_volume', 0)
            point_probe_fraction = point_probe_ref.get('void_volume_fraction', 0)
            writer.writerow(['Global', 'Point Probe', 'All', f"{point_probe_vol:.4f}", 'cm³/g'])
            writer.writerow(['Global', 'Fraction', 'Point Probe', f"{point_probe_fraction:.4f}", 'ratio'])

            # Region-wise void volumes
            by_region = results.get('by_region', {})
            if by_region:
                writer.writerow([])  # Empty row
                writer.writerow(['Region', 'Individual', 'Label', 'Void_Volume_cm3_g', 'Unit'])
                for region_id, volume in by_region.items():
                    writer.writerow(['Region', 'Individual', f"region_{region_id}", f"{volume:.4f}", 'cm³/g'])

            # Pocket-wise void volumes
            by_pocket = results.get('by_pocket', {})
            if by_pocket:
                writer.writerow([])  # Empty row
                writer.writerow(['Pocket', 'Complete', 'Label', 'Void_Volume_cm3_g', 'Unit'])
                for pocket_id, volume in by_pocket.items():
                    writer.writerow(['Pocket', 'Complete', f"{pocket_id}", f"{volume:.4f}", 'cm³/g'])

            # Percolated cluster void volumes
            by_percolated_cluster = results.get('by_percolated_cluster', {})
            if by_percolated_cluster:
                writer.writerow([])  # Empty row
                writer.writerow(['Cluster', 'Percolated', 'Label', 'Void_Volume_cm3_g', 'Unit'])
                for cluster_id, volume in by_percolated_cluster.items():
                    writer.writerow(['Cluster', 'Percolated', f"{cluster_id}", f"{volume:.4f}", 'cm³/g'])

            # Probe radius used
            probe_radius = results.get('probe_radius', 'N/A')
            writer.writerow([])  # Empty row
            writer.writerow(['Parameter', 'Probe_Radius', 'Value', f"{probe_radius}", 'Å'])

        print(f"Void volume analysis saved to: {vol_csv_file}")

    except Exception as e:
        print(f"Error saving void volume CSV: {e}")


class MCLI:
    """Main CLI class for MOF analysis."""

    # Flag to use standalone (numpy-free) implementation
    USE_STANDALONE = False  # Set to True for standalone builds

    PLOT_DEFAULTS = {
        'plot_enable': True,
        'plot_cell': True,
        'plot_framework': True,
        'plot_framework_style': 'mesh',
        'plot_maxima': True,
        'plot_maxima_show_spheres': False,
        'plot_connections': True,
        'plot_surface': True,
        'plot_surface_pbc': True,
        'plot_surface_isovalue': 0.75,
        'plot_surface_opacity': 1.0,
        'plot_surface_pbc_opacity': 1.0,
        'plot_surface_types_opacity': 1.0,
    }

    # Default configuration values
    DEFAULTS = {
        'spacing': 0.5,
        'h': 0.2,
        'mask_thickness': 1.5,
        'threshold_abs': 1.5,
        'min_distance': 1.5,
        'minimum_window_separation': None,
        'compactness': 0,  # Not exposed in CLI
        'wall_thickness': 0.05,
        'extra_padding': 5.0,
        'min_region_size': 0.001,
        'disable_tqdm': False,
        'compute_connections': True,
        'return_padded_regions': False,
        'return_dgrid': True,
        'watershed_backend': 'rust_bucket',
    }
    DEFAULTS.update(PLOT_DEFAULTS)

    def __init__(self):
        """Initialize the CLI."""
        self._rust_lib = None

    def _get_rust_library(self):
        """Lazily import the Rust library."""
        if self._rust_lib is None:
            try:
                import mgrcli_rust
                self._rust_lib = mgrcli_rust
            except ImportError as e:
                print("Error: mgrcli_rust Rust library not found. Please build the project first.")
                print("Run: pip install -e .")
                print(f"Original error: {e}")
                sys.exit(1)
        return self._rust_lib

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
        """Run the CLI application."""
        parser = argparse.ArgumentParser(
            description="Mofography CLI Tool - High-Performance MOF Analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s process structure.cif
  %(prog)s process *.cif -o ./results/
  %(prog)s process structure.cif --types   # Detect pocket types and save RAG pickle
  %(prog)s process structure.cif --rag     # Save RAG pickle only (no clustering)
  %(prog)s generate-config -mt 1.0 -ta 2.0 -o config.json
  %(prog)s config-building
  %(prog)s interactive
  %(prog)s --banner

Notes:
  --types now always saves the region adjacency graph (RAG) as a pickle file in addition to pocket type detection.
  --rag can be used independently to save the RAG after segmentation, even if pocket type detection is not performed.
            """
        )

        # Global options
        parser.add_argument(
            '--version', action='version',
            version=f'%(prog)s {__import__("mgrcli").__version__}'
        )
        parser.add_argument(
            '--banner', action='store_true',
            help='Display the Mofography CLI banner'
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
        param_group.add_argument(
            '-b', '--backend', '--watershed-backend',
            dest='watershed_backend',
            choices=['rust_bucket', 'rust_det', 'rust_det_parallel', 'skimage'],
            default=None,
            help=(
                'Watershed backend for segmentation: rust_bucket (default), rust_det, '
                'rust_det_parallel, or skimage (install scikit-image, e.g. pip install "mgrcli[skimage]").'
            ),
        )
        param_group.add_argument(
            '--js-threshold', type=float, default=0.15,
            help='Jensen-Shannon threshold for pocket type clustering (default: 0.15)'
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
        flag_group.add_argument(
            '--plot-html', action='store_true',
            help='Generate Plotly HTML visualization after processing'
        )
        process_parser.add_argument(
            '--html-dir',
            help='Directory for HTML visualizations (defaults to output directory)'
        )
        flag_group.add_argument(
            '--no-stylize', action='store_true',
            help='Disable stylized framework rendering in plots'
        )
        flag_group.add_argument(
            '--pore-dia', action='store_true',
            help='Compute pore diameter and percolation analysis'
        )
        flag_group.add_argument(
            '--psd', action='store_true',
            help='Compute pore size distribution analysis'
        )
        flag_group.add_argument(
            '--auto-params', action='store_true',
            help='Automatically estimate optimal parameters from structure'
        )
        flag_group.add_argument(
            '--theme', choices=['dark', 'white'], default='dark',
            help='Theme for generated visualizations (default: dark)'
        )
        flag_group.add_argument(
            '--types', action='store_true',
            help='Detect pocket types using agglomerative clustering on cavity size distributions. If set, saves pocket type segmentation and (now) also outputs a RAG pickle file.'
        )
        flag_group.add_argument(
            '--rag', action='store_true',
            help='Save the NetworkX region adjacency graph (RAG) as a pickle file after segmentation. Can be used independently or with --types.'
        )
        flag_group.add_argument(
            '--sa', action='store_true',
            help='Compute surface area analysis with percolation accessibility'
        )
        flag_group.add_argument(
            '--vol', action='store_true',
            help='Compute void volume analysis with percolation accessibility'
        )
        flag_group.add_argument(
            '--psd-accessible', action='store_true',
            help='Compute accessible (percolation-based) PSD at --probe-radius; use with or without --psd. Saves *_psd_accessible_psd_uniform.csv and *_psd_accessible_psd_pockets.csv'
        )

        # Analysis parameters
        analysis_group = process_parser.add_argument_group('analysis options')
        analysis_group.add_argument(
            '--probe-radius', type=float,
            help='Probe radius for surface area and volume calculations (default: helium radius 1.2 Å)'
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

        # Plot command
        plot_parser = subparsers.add_parser(
            'plot',
            help='Generate Plotly visualization from existing results file'
        )
        plot_parser.add_argument(
            'results_files', nargs='+',
            help='Result NPZ file(s) produced by the process command'
        )
        plot_parser.add_argument(
            '-o', '--output-dir', default='.',
            help='Directory to store generated HTML files (default: current directory)'
        )
        plot_parser.add_argument(
            '-c', '--config',
            help='Configuration file to control plotting options'
        )
        plot_parser.add_argument(
            '--no-stylize', action='store_true',
            help='Disable stylized framework rendering in plots'
        )
        plot_parser.add_argument(
            '--cif',
            help='Optional CIF file to use when metadata is missing (only for single input)'
        )

        # Interactive command
        interactive_parser = subparsers.add_parser(
            'interactive',
            help='Launch interactive Streamlit web application'
        )
        interactive_parser.add_argument(
            '--port', type=int,
            help='Port to serve the Streamlit app (default chosen by Streamlit)'
        )
        interactive_parser.add_argument(
            '--no-browser', action='store_true',
            help='Do not automatically open a browser window'
        )
        interactive_parser.add_argument(
            '--clean', action='store_true',
            help='Kill stale interactive app Streamlit processes before launch'
        )

        # Parse arguments
        args = parser.parse_args()

        # Handle banner flag
        if args.banner:
            self.show_banner()
            if not args.command:
                return

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
        elif args.command == 'plot':
            self.plot_command(args)
        elif args.command == 'interactive':
            self.interactive_command(args)

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
            print("  mgrcli process structure.cif")
            print("  mgrcli process -i structure.cif")
            print("  mgrcli process *.cif")
            sys.exit(1)

        print(f"Found {len(cif_files)} CIF file(s) to process")

        # Apply auto-parameters if requested
        if getattr(args, 'auto_params', False):
            print("Estimating optimal parameters...")
            atoms = self._read_atoms(cif_files[0])  # Use first file for parameter estimation
            auto_params = compute_auto_parameters(atoms, config['spacing'])
            if auto_params:
                config.update(auto_params)
                print(f"  Auto-estimated parameters:")
                for key, value in auto_params.items():
                    print(f"    {key}: {value:.3f}")

        plot_options = self._get_plot_options(config)

        for cif_file in cif_files:
            if not cif_file.exists():
                print(f"Warning: {cif_file} does not exist, skipping")
                continue

            print(f"\nProcessing: {cif_file}")
            try:
                result = self.process_single_cif(cif_file, config)
                atoms = self._read_atoms(cif_file)
                # Keep a copy of original atoms without covalent bonds for RAG operations
                # to avoid duplicate atom warnings in MGR library functions
                from ase.io import read
                original_atoms = read(str(cif_file))

                # Generate output filename
                output_file = output_dir / f"{cif_file.stem}_results.npz"

                # Handle pore diameter analysis if requested
                if getattr(args, 'pore_dia', False):
                    pore_dia_results = compute_pore_diameter_analysis(
                        result['dgrid'], atoms, spacing=config['spacing'],
                        segmentation_results=result
                    )
                    if pore_dia_results:
                        # Add to main results
                        result.update(pore_dia_results)

                        # Save CSV file for pore diameter analysis
                        pore_dia_csv_file = output_dir / f"{cif_file.stem}_pore_dia.csv"
                        save_pore_diameter_csv(pore_dia_results, pore_dia_csv_file)

                # Handle PSD analysis if requested
                if getattr(args, 'psd', False):
                    # Get regions data for pocket-wise PSD
                    regions_pbc = result.get('regions_pbc')

                    psd_results = compute_psd_analysis(
                        result['dgrid'], atoms, regions_pbc=regions_pbc, spacing=config['spacing']
                    )
                    if psd_results:
                        # Add to main results
                        result.update(psd_results)

                        # Save CSV files for PSD analysis
                        psd_csv_base_file = output_dir / f"{cif_file.stem}_psd"
                        save_psd_csv(psd_results, psd_csv_base_file, material_name=cif_file.stem)

                # Handle accessible PSD if requested (percolation-based, at probe_radius)
                if getattr(args, 'psd_accessible', False):
                    if result.get('connections') is None:
                        print("Warning: Accessible PSD requires segmentation with connections. Enable compute_connections and re-run.")
                    else:
                        probe_radius = getattr(args, 'probe_radius', None)
                        if probe_radius is None:
                            from ase.data import vdw_radii
                            probe_radius = vdw_radii[2]  # helium radius
                            print(f"  Using default probe radius for accessible PSD: {probe_radius:.3f} Å")
                        else:
                            print(f"  Using probe radius for accessible PSD: {probe_radius:.3f} Å")
                        print("Computing accessible PSD (percolating regions only)...")
                        try:
                            uniform_max_diameter = 50.0
                            uniform_num_bins = 200
                            pocket_max_diameter = 50.0
                            pocket_num_bins = 200
                            num_samples_total = 10000
                            num_samples_per_region = 1000
                            bins_uniform = np.linspace(0, uniform_max_diameter, uniform_num_bins + 1)
                            bins_pocket = np.linspace(0, pocket_max_diameter, pocket_num_bins + 1)

                            acc_uniform = mgr.compute_psd_global_accessible(
                                result,
                                atoms,
                                probe_radius,
                                bins=bins_uniform,
                                samples_per_region=num_samples_per_region,
                                return_fig=False,
                            )
                            uniform_hist = acc_uniform if isinstance(acc_uniform, (list, np.ndarray)) else acc_uniform[0]
                            acc_pocket_psds = mgr.compute_psds_pocketwise_accessible(
                                result,
                                atoms,
                                probe_radius,
                                num_samples_per_region=num_samples_per_region,
                                bins=bins_pocket,
                                return_fig=False,
                            )
                            pocket_psds = acc_pocket_psds if isinstance(acc_pocket_psds, dict) else acc_pocket_psds[0]

                            acc_psd_results = {
                                'psd_uniform': {
                                    'bins': bins_uniform,
                                    'histogram': uniform_hist,
                                    'bin_centers': 0.5 * (bins_uniform[:-1] + bins_uniform[1:]),
                                    'num_samples': num_samples_per_region * (len(pocket_psds) if pocket_psds else 0) or 1,
                                },
                                'psd_pocket': {
                                    'bins': bins_pocket,
                                    'bin_centers': 0.5 * (bins_pocket[:-1] + bins_pocket[1:]),
                                    'pocket_psds': pocket_psds,
                                    'num_samples_per_region': num_samples_per_region,
                                    'num_regions': len(pocket_psds) if pocket_psds else 0,
                                },
                            }
                            psd_accessible_base = output_dir / f"{cif_file.stem}_psd_accessible"
                            save_psd_csv(acc_psd_results, psd_accessible_base, material_name=cif_file.stem)
                            print("  Accessible PSD CSVs saved.")
                        except Exception as e:
                            print(f"Warning: Accessible PSD calculation failed: {e}")

                # Handle surface area analysis if requested
                if getattr(args, 'sa', False):
                    print("Computing surface area analysis...")

                    # Get probe radius from args or use default
                    probe_radius = getattr(args, 'probe_radius', None)
                    if probe_radius is None:
                        from ase.data import vdw_radii
                        probe_radius = vdw_radii[2]  # helium radius
                        print(f"  Using default probe radius: {probe_radius:.3f} Å")
                    else:
                        print(f"  Using probe radius: {probe_radius:.3f} Å")

                    # Compute surface area analysis
                    sa_results = compute_surface_area_analysis(result, atoms, probe_radius)
                    if sa_results:
                        # Add to main results
                        result['surface_area_analysis'] = sa_results

                        # Save CSV file for surface area analysis
                        sa_csv_base_file = output_dir / f"{cif_file.stem}_sa"
                        save_surface_area_csv(sa_results, sa_csv_base_file, material_name=cif_file.stem)

                        # Print summary
                        total_sa = sa_results.get('total_surface_area', 0)
                        accessible_sa = sa_results.get('accessible_surface_area', 0)
                        accessible_fraction = sa_results.get('accessible_fraction', 0)
                        
                        # Add point probe reference
                        point_probe_ref = sa_results.get('point_probe_reference', {})
                        point_probe_sa = point_probe_ref.get('surface_area', 0)
                        point_probe_fraction = point_probe_ref.get('surface_area_fraction', 0)
                        
                        print(f"  Surface Area Results:")
                        print(f"    Total surface area: {total_sa:.4f} m²/g")
                        print(f"    Accessible surface area: {accessible_sa:.4f} m²/g")
                        print(f"    Accessible fraction: {accessible_fraction:.4f}")
                        print(f"    Point probe reference (radius = 0): {point_probe_sa:.4f} m²/g")
                        print(f"    Point probe fraction: {point_probe_fraction:.4f}")
                    else:
                        print("  Surface area analysis failed")

                # Handle void volume analysis if requested
                if getattr(args, 'vol', False):
                    print("Computing void volume analysis...")

                    # Get probe radius from args or use default
                    probe_radius = getattr(args, 'probe_radius', None)
                    if probe_radius is None:
                        from ase.data import vdw_radii
                        probe_radius = vdw_radii[2]  # helium radius
                        print(f"  Using default probe radius: {probe_radius:.3f} Å")
                    else:
                        print(f"  Using probe radius: {probe_radius:.3f} Å")

                    # Compute void volume analysis
                    vol_results = compute_void_volume_analysis(result, atoms, probe_radius)
                    if vol_results:
                        # Add to main results
                        result['void_volume_analysis'] = vol_results

                        # Save CSV file for void volume analysis
                        vol_csv_base_file = output_dir / f"{cif_file.stem}_vol"
                        save_void_volume_csv(vol_results, vol_csv_base_file, material_name=cif_file.stem)

                        # Print summary
                        total_vol = vol_results.get('total_void_volume', 0)
                        accessible_vol = vol_results.get('accessible_void_volume', 0)
                        accessible_fraction = vol_results.get('accessible_fraction', 0)
                        
                        # Add point probe reference
                        point_probe_ref = vol_results.get('point_probe_reference', {})
                        point_probe_vol = point_probe_ref.get('void_volume', 0)
                        point_probe_fraction = point_probe_ref.get('void_volume_fraction', 0)
                        
                        print(f"  Void Volume Results:")
                        print(f"    Total void volume: {total_vol:.4f} cm³/g")
                        print(f"    Accessible void volume: {accessible_vol:.4f} cm³/g")
                        print(f"    Accessible fraction: {accessible_fraction:.4f}")
                        print(f"    Point probe reference (radius = 0): {point_probe_vol:.4f} cm³/g")
                        print(f"    Point probe fraction: {point_probe_fraction:.4f}")
                    else:
                        print("  Void volume analysis failed")

                # Handle RAG output if requested (even without --types)
                if getattr(args, 'rag', False) and not getattr(args, 'types', False):
                    # Only build/save RAG if not already handled by --types
                    import numpy as np
                    import pickle
                    dgrid = result['dgrid']
                    regions = result['regions']
                    maxima_info = result['maxima_info']
                    all_maxima = result['all_maxima']
                    all_maxima_radii = dgrid[tuple(maxima_info[:,1:4].T)]
                    connections = result.get('connections', [])
                    unique_regions_for_pbc = result.get('unique_regions', [])
                    pbc_groups_dict = get_pbc_groups(unique_regions_for_pbc, maxima_info)
                    pbc_groups = [v for k, v in pbc_groups_dict.items()]
                    unique_regions = result.get('unique_regions', [])
                    maxima = all_maxima[:regions.max()]
                    maxima_radii = all_maxima_radii[:regions.max()]
                    connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]
                    if len(maxima) > 0:
                        try:
                            rag = mgr.rag_from_connections(connections_trimmed, maxima)
                            rag = mgr.add_pbc_to_rag(rag, pbc_groups)
                            rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid.shape, original_atoms)
                            rag_pickle_path = output_dir / f"{cif_file.stem}_rag_clustered.pkl"
                            with open(rag_pickle_path, 'wb') as f:
                                pickle.dump(rag, f)
                            print(f"RAG (region adjacency graph) saved to: {rag_pickle_path}")
                        except Exception as e:
                            print(f"Warning: RAG creation failed: {e}")
                    else:
                        print("Warning: Cannot build RAG - no maxima data found")

                # Handle pocket type detection if requested
                if getattr(args, 'types', False):
                    from skimage.measure import regionprops
                    import numpy as np

                    print("Detecting pocket types...")
                    js_threshold = getattr(args, 'js_threshold', 0.2)

                    # Get required data for clustering
                    dgrid = result['dgrid']
                    regions = result['regions']
                    regions_pbc = result.get('regions_pbc', regions)
                    maxima_info = result['maxima_info']
                    all_maxima = result['all_maxima']
                    all_maxima_radii = dgrid[tuple(maxima_info[:,1:4].T)]
                    connections = result.get('connections', [])
                    # Compute proper PBC groups using the reference library function
                    unique_regions_for_pbc = result.get('unique_regions', [])
                    pbc_groups_dict = get_pbc_groups(unique_regions_for_pbc, maxima_info)
                    pbc_groups = [v for k, v in pbc_groups_dict.items()]
                    unique_regions = result.get('unique_regions', [])

                    # Trim maxima using regions.max() as shown in example
                    maxima = all_maxima[:regions.max()]
                    maxima_radii = all_maxima_radii[:regions.max()]
                    connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]

                    if len(maxima) > 0:
                        try:
                            # Rebuild RAG following the example pattern
                            rag = mgr.rag_from_connections(connections_trimmed, maxima)
                            rag = mgr.add_pbc_to_rag(rag, pbc_groups)
                            rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid.shape, original_atoms)

                            # Add PSD and clustering to RAG
                            props = regionprops(regions)
                            bins = np.linspace(0, 30, 50)

                            # Check if any pbc_groups contain label 0 (background)
                            has_zero = any(0 in group for group in pbc_groups)
                            if has_zero:
                                print("Debug: Removing groups with label 0")
                                pbc_groups = [group for group in pbc_groups if 0 not in group]

                            rag = mgr.add_psd_to_rag_with_pbc(rag, props, pbc_groups, bins, original_atoms, dgrid.shape)
                            rag = mgr.add_clustering_to_rag_geometry_js(rag, props, pbc_groups, original_atoms, dgrid.shape, use_pbc=True, js_threshold=js_threshold)

                            # Generate regions cluster
                            regions_cluster = mgr.clustering_grid_from_regions(regions, rag, cluster_key="cluster_geom_pbc")
                        except Exception as e:
                            print(f"Warning: Pocket type detection failed: {e}")
                            regions_cluster = None
                            rag = None

                        if regions_cluster is not None:
                            result['regions_cluster'] = regions_cluster
                            result['pocket_types_detected'] = True
                            result['js_threshold_used'] = js_threshold

                            # Save RAG pickle if --types or --rag is set
                            if getattr(args, 'types', False) or getattr(args, 'rag', False):
                                import pickle
                                rag_pickle_path = output_dir / f"{cif_file.stem}_rag_clustered.pkl"
                                with open(rag_pickle_path, 'wb') as f:
                                    pickle.dump(rag, f)
                                print(f"RAG (region adjacency graph) saved to: {rag_pickle_path}")

                            print(f"Pocket types detected using JS threshold: {js_threshold}")
                            num_clusters = len(np.unique(regions_cluster)) - 1  # -1 for background
                            print(f"Found {num_clusters} different pocket types")
                        else:
                            print("Warning: Cannot perform pocket type detection - no maxima data found or clustering failed")
                            result['pocket_types_detected'] = False


                # Save results
                save_payload = dict(result)
                save_payload['cif_path'] = str(cif_file)
                # Ensure rag_clustered is not present in results
                if 'rag_clustered' in save_payload:
                    del save_payload['rag_clustered']
                self.save_results(save_payload, output_file)
                print(f"Results saved to: {output_file}")

                if getattr(args, 'plot_html', False):
                    from .plotting import prepare_results_for_plot, show_mof, dark_layout, white_layout
                    plot_ready = prepare_results_for_plot(dict(result), atoms)
                    html_dir = Path(getattr(args, 'html_dir', '') or output_dir)
                    html_dir.mkdir(parents=True, exist_ok=True)
                    html_path = html_dir / f"{cif_file.stem}.html"

                    # Apply theme selection
                    theme = getattr(args, 'theme', 'dark')
                    layout = white_layout if theme == 'white' else dark_layout

                    fig = show_mof(plot_ready, stylize=not getattr(args, 'no_stylize', False), options=plot_options)
                    fig.update_layout(**layout)
                    fig.write_html(html_path)
                    print(f"Plot saved to: {html_path} (theme: {theme})")

            except Exception as e:
                print(f"Error processing {cif_file}: {e}")
                continue

    def plot_command(self, args):
        """Generate HTML plots from existing results files."""
        try:
            import numpy as imported_numpy  # noqa: F401  # Ensure numpy is present for npz loading
        except ImportError as exc:  # pragma: no cover
            print("Error: numpy is required to load results files.")
            print(f"Original error: {exc}")
            sys.exit(1)

        global np
        if np is None:
            np = imported_numpy

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from .plotting import prepare_results_for_plot, show_mof

        config_data = self.DEFAULTS.copy()
        if getattr(args, 'config', None):
            loaded = self.load_config(args.config)
            if isinstance(loaded, dict):
                config_data.update(loaded)
        plot_options = self._get_plot_options(config_data)

        for result_path_str in args.results_files:
            result_path = Path(result_path_str)
            if not result_path.exists():
                print(f"Warning: {result_path} does not exist, skipping")
                continue

            print(f"Generating plot for: {result_path}")
            try:
                result_dict = self.load_results_file(result_path)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  Failed to load {result_path}: {exc}")
                continue

            cif_path_str = result_dict.get('cif_path')
            if not cif_path_str:
                if len(args.results_files) == 1 and args.cif:
                    cif_path_str = args.cif
                else:
                    print("  Error: CIF path metadata missing. Re-run process command or supply --cif for single file.")
                    continue

            atoms = self._read_atoms(Path(cif_path_str))
            plot_ready = prepare_results_for_plot(result_dict, atoms)

            html_path = output_dir / f"{result_path.stem}.html"
            fig = show_mof(plot_ready, stylize=not args.no_stylize, options=plot_options)
            fig.write_html(html_path)
            print(f"  Plot saved to: {html_path}")

    def interactive_command(self, args):
        """Launch the Streamlit interactive application."""
        app_path = Path(__file__).resolve().parent / "interactive_app.py"
        if not app_path.exists():
            print("Error: interactive app script not found.")
            sys.exit(1)

        if getattr(args, 'clean', False):
            try:
                current_pid = os.getpid()
                stale_pids = []
                proc = subprocess.run(
                    ["pgrep", "-f", str(app_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0 and proc.stdout:
                    for line in proc.stdout.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pid = int(line)
                        except ValueError:
                            continue
                        if pid != current_pid:
                            stale_pids.append(pid)

                if stale_pids:
                    for pid in stale_pids:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except OSError:
                            pass
                    time.sleep(0.4)
                    for pid in stale_pids:
                        try:
                            os.kill(pid, 0)
                        except OSError:
                            continue
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except OSError:
                            pass
                    print(f"Cleaned stale interactive app processes: {len(stale_pids)}")
                else:
                    print("No stale interactive app processes found.")
            except Exception as e:
                print(f"Warning: failed to clean stale processes: {e}")

        cmd = ["streamlit", "run", str(app_path)]
        if getattr(args, 'port', None):
            cmd.extend(["--server.port", str(args.port)])

        env = os.environ.copy()
        repo_path = str(REPO_ROOT)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{repo_path}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = repo_path

        if getattr(args, 'no_browser', False):
            env["STREAMLIT_SERVER_HEADLESS"] = "true"

        env.setdefault("MCLI_SAMPLE_CIF_DIR", str(REPO_ROOT / "cifs"))
        # Force local behavior for CLI-launched interactive sessions so hosted-demo
        # constraints are never applied by accident due to inherited shell env vars.
        env.setdefault("MGR_FORCE_LOCAL", "1")

        try:
            subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError:  # pragma: no cover
            print("Error: streamlit executable not found. Please install streamlit to use interactive mode.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error: Streamlit exited with error code {e.returncode}")
            print("This is often caused by issues with molecular structure processing:")
            print("  - Duplicate atoms in the structure")
            print("  - Invalid geometry data")
            print("  - Missing or corrupted CIF files")
            print("Try using --verbose flag for more details or check your input structure.")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error launching interactive mode: {e}")
            sys.exit(1)

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
            'plot_surface_isovalue': f'Isovalue for surface plotting (default: {self.DEFAULTS["plot_surface_isovalue"]})',
            'plot_surface_opacity': f'Surface opacity (0-1) (default: {self.DEFAULTS["plot_surface_opacity"]})',
            'plot_surface_pbc_opacity': f'PBC surface opacity (0-1) (default: {self.DEFAULTS["plot_surface_pbc_opacity"]})',
            'plot_surface_types_opacity': f'Pocket types surface opacity (0-1) (default: {self.DEFAULTS["plot_surface_types_opacity"]})',
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
            'plot_enable': 'Enable plotting outputs by default?',
            'plot_cell': 'Include unit cell trace in plots?',
            'plot_framework': 'Include framework in plots?',
            'plot_maxima': 'Include maxima trace in plots?',
            'plot_maxima_show_spheres': 'Show maxima spheres?',
            'plot_connections': 'Include connection graph traces?',
            'plot_surface': 'Include regions surface?',
            'plot_surface_pbc': 'Include regions PBC surface?',
        }

        for key, prompt in bool_prompts.items():
            response = input(f"{prompt} (y/N): ").strip().lower()
            config[key] = response in ('y', 'yes', 'true', '1')

        style_default = self.DEFAULTS['plot_framework_style']
        while True:
            style_input = input(f"Framework plotting style (mesh/scatter) (default: {style_default}): ").strip().lower()
            if not style_input:
                config['plot_framework_style'] = style_default
                break
            if style_input in ('mesh', 'scatter'):
                config['plot_framework_style'] = style_input
                break
            print("Please enter 'mesh' or 'scatter'.")

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

        # Ensure return_dgrid and compute_connections are enabled if sa, vol, or psd_accessible is requested
        if (hasattr(args, 'sa') and args.sa) or (hasattr(args, 'vol') and args.vol) or (hasattr(args, 'psd_accessible') and args.psd_accessible):
            if not result.get('return_dgrid', False):
                result['return_dgrid'] = True
                print("Info: Automatically enabled distance grid (--return-dgrid) for surface area / void volume / accessible PSD analysis")
            if not result.get('compute_connections', False):
                result['compute_connections'] = True
                print("Info: Automatically enabled compute_connections for accessible PSD / percolation analysis")

        # Handle boolean flags
        if hasattr(args, 'no_connections') and args.no_connections:
            result['compute_connections'] = False
        if hasattr(args, 'return_padded_regions') and args.return_padded_regions:
            result['return_padded_regions'] = True
        if hasattr(args, 'return_dgrid') and args.return_dgrid:
            result['return_dgrid'] = True
        if hasattr(args, 'disable_tqdm') and args.disable_tqdm:
            result['disable_tqdm'] = True

        if getattr(args, 'watershed_backend', None) is not None:
            result['watershed_backend'] = args.watershed_backend

        return result

    def process_single_cif(self, cif_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single CIF file using the Python workflow built on the Rust backend."""

        atoms = self._read_atoms(cif_file)
        spacing = config['spacing']

        dgrid = compute_dgrid_rust(atoms, spacing)

        segmentation_kwargs = dict(
            dgrid=dgrid,
            mat_atoms=atoms,
            mask_thickness=config['mask_thickness'],
            h=config['h'],
            min_distance=config['min_distance'],
            threshold_abs=config['threshold_abs'],
            return_padded_regions=config['return_padded_regions'],
            spacing=spacing,
            minimum_window_separation=config.get('minimum_window_separation'),
            wall_thickness=config['wall_thickness'],
            extra_padding=config['extra_padding'],
            compactness=config['compactness'],
            min_region_size=config['min_region_size'],
            disable_tqdm=config['disable_tqdm'],
            compute_connections=config['compute_connections'],
            return_dgrid=config['return_dgrid'],
            watershed_backend=config.get('watershed_backend', 'rust_bucket'),
        )

        total_start = time.perf_counter()
        result = python_do_periodic_segmentation_rust(**segmentation_kwargs)
        total_end = time.perf_counter()

        if config['return_dgrid'] and 'dgrid' not in result:
            result['dgrid'] = dgrid

        timing = result.get('timing_results', {}) or {}
        timing.setdefault('total_time', total_end - total_start)
        result['timing_results'] = timing

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
            elif key == 'rag_clustered':
                # Special handling for NetworkX graphs - preserve the graph object
                # NetworkX graphs can't be directly saved to numpy, so we pickle them
                try:
                    import pickle
                    save_dict[key] = pickle.dumps(value)
                except ImportError:
                    print(f"Warning: Could not save NetworkX graph for {key} - pickle not available")
                    # Fallback: save basic graph information as dict
                    save_dict[key] = {
                        'nodes': list(value.nodes()),
                        'edges': list(value.edges()),
                        'node_attrs': dict(value.nodes(data=True)),
                        'graph_attrs': dict(value.graph)
                    }
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
            # Special handling for NetworkX graphs
            if key == 'rag_clustered':
                try:
                    import networkx as nx
                    if isinstance(array_data, nx.Graph):
                        # NetworkX graph - pickle it for preservation
                        import pickle
                        write_npy_header(f, (), '|O')
                        pickle.dump(array_data, f)
                        return
                except ImportError:
                    pass  # NetworkX not available, fall through to default handling

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
        # Check if NetworkX graphs are present - always use numpy method for proper graph preservation
        has_networkx_graphs = 'rag_clustered' in result
        try:
            import networkx as nx
            has_networkx_graphs = has_networkx_graphs and isinstance(result['rag_clustered'], nx.Graph)
        except ImportError:
            pass

        if has_networkx_graphs and HAS_NUMPY:
            # Always use numpy method for NetworkX graphs to preserve them properly
            self.save_results_numpy(result, output_file)
        elif self.USE_STANDALONE or not HAS_NUMPY:
            if not HAS_NUMPY:
                print("Warning: numpy not available, using standalone implementation")
            self.save_results_standalone(result, output_file)
        else:
            self.save_results_numpy(result, output_file)

    def load_results_file(self, path: Path) -> Dict[str, Any]:
        """Load a previously saved results NPZ file."""
        data: Dict[str, Any] = {}
        with np.load(path, allow_pickle=True) as npz:  # type: ignore[arg-type]
            for key in npz.files:
                value = npz[key]

                if isinstance(value, np.ndarray):
                    if value.dtype == object and value.size == 1:
                        loaded_value = value.item()

                        # Special handling for NetworkX graphs
                        if key == 'rag_clustered':
                            try:
                                import pickle
                                # Check if it's a pickled NetworkX graph (bytes)
                                if isinstance(loaded_value, bytes):
                                    data[key] = pickle.loads(loaded_value)
                                else:
                                    data[key] = loaded_value
                            except ImportError:
                                print(f"Warning: Could not unpickle NetworkX graph for {key} - pickle not available")
                                data[key] = loaded_value
                            except Exception as e:
                                print(f"Warning: Error unpickling {key}: {e}")
                                data[key] = loaded_value
                        else:
                            data[key] = loaded_value
                        continue
                    elif key == 'rag_clustered':
                        # Handle case where rag_clustered is saved as bytes array (dtype |Sxxx)
                        try:
                            loaded_value = value.item() if value.size == 1 else value
                            if isinstance(loaded_value, bytes):
                                import pickle
                                data[key] = pickle.loads(loaded_value)
                            else:
                                data[key] = loaded_value
                        except Exception as e:
                            print(f"Warning: Could not process {key}: {e}")
                            data[key] = value
                        continue

                    if value.shape == ():
                        data[key] = value.item()
                        continue
                elif key == 'rag_clustered':
                    # Handle case where rag_clustered is not a numpy array
                    if isinstance(value, bytes):
                        try:
                            import pickle
                            data[key] = pickle.loads(value)
                        except Exception as e:
                            print(f"Warning: Could not unpickle {key}: {e}")
                            data[key] = value
                    else:
                        data[key] = value
                    continue

                data[key] = value
        return data

    def _get_plot_options(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'plot_enable': config.get('plot_enable', True),
            'plot_cell': config.get('plot_cell', True),
            'plot_framework': config.get('plot_framework', True),
            'plot_framework_style': config.get('plot_framework_style', 'mesh'),
            'plot_maxima': config.get('plot_maxima', True),
            'plot_maxima_show_spheres': config.get('plot_maxima_show_spheres', False),
            'plot_connections': config.get('plot_connections', True),
            'plot_surface': config.get('plot_surface', True),
            'plot_surface_pbc': config.get('plot_surface_pbc', True),
            'plot_surface_isovalue': config.get('plot_surface_isovalue', self.PLOT_DEFAULTS['plot_surface_isovalue']),
            'plot_surface_opacity': config.get('plot_surface_opacity', self.PLOT_DEFAULTS['plot_surface_opacity']),
            'plot_surface_pbc_opacity': config.get('plot_surface_pbc_opacity', self.PLOT_DEFAULTS['plot_surface_pbc_opacity']),
            'plot_surface_types_opacity': config.get('plot_surface_types_opacity', self.PLOT_DEFAULTS['plot_surface_types_opacity']),
        }

    def _read_atoms(self, cif_path: Path):
        """Read an ASE Atoms object from CIF file."""
        try:
            from ase.io import read
        except ImportError as exc:  # pragma: no cover
            print("Error: ASE is required for plotting. Install via pip install ase")
            print(f"Original error: {exc}")
            sys.exit(1)

        try:
            atoms = read(str(cif_path))
            # Compute bonds immediately after loading CIF only if not already done
            REPO_ROOT = Path(__file__).resolve().parents[2]
            MGR_SRC_PATH = REPO_ROOT / "mgr_src"
            if str(MGR_SRC_PATH) not in sys.path:
                sys.path.insert(0, str(MGR_SRC_PATH))

            # Check if covalent bonds have already been computed to avoid duplicate atoms
            if 'covalent_bonds' not in atoms.info:
                try:
                    import mgr as mgr
                    from mgr.analyze import update_with_covalent_bonds
                    atoms = update_with_covalent_bonds(atoms)
                except ImportError:
                    # Fallback to mgr_visualize if mgr_src is not available
                    import mgr_visualize.mgr as mgr
                    from mgr_visualize.mgr.analyze import update_with_covalent_bonds
                    atoms = update_with_covalent_bonds(atoms)

            return atoms
        except Exception as exc:  # pragma: no cover - file errors
            print(f"Error reading CIF file {cif_path}: {exc}")
            sys.exit(1)


def main():
    """Main entry point."""
    cli = MCLI()
    cli.run()


if __name__ == '__main__':
    main()

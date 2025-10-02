#!/usr/bin/env python3
"""
Basic test script for the compute_pore_sizes_for_psd function.
This demonstrates how to use the new watershed-optimized PSD calculation.
"""

import numpy as np
import mgr_rust
import plotly.graph_objects as go
# Import required libraries for atomic structure analysis and pore size distribution computation
from ase.io import read  # For reading atomic structure files
from pathlib import Path  # For file path operations
# Import required libraries for atomic structure analysis and pore size distribution computation
from ase.io import read  # For reading atomic structure files
from pathlib import Path  # For file path operations
import numpy as np  # For numerical operations
import mofography as mgr  # For MOF analysis tools
import plotly.graph_objects as go  # For interactive plotting
import logging
from ase.geometry import cellpar_to_cell
logger = logging.getLogger(__name__)
# Set logging level to INFO to see debug messages
logger.setLevel(logging.INFO)
# # Find all CIF files in the examples directory and randomly select one for analysis
# list_of_cifs = list((Path('..').resolve()/ 'examples' / 'TestCIFs' ).rglob('*.cif'))
# path_to_cif = np.random.choice(np.array(list_of_cifs))  # Convert to numpy array for choice
logging.basicConfig(level=logging.INFO)

import time
start = time.time()

material = 'NU-1000_S1'
path_to_cif = Path(f"../examples/TestCIFs/{material}.cif")
print('The selected CIF is:\n', path_to_cif.name)
import mofography as mgr  # For MOF analysis tools
import plotly.graph_objects as go  # For interactive plotting
#
def test_compute_pore_sizes_pocketwise():
    """Test the compute_pore_sizes_for_psd function with synthetic data."""

    # Create a simple test case
    print("Testing compute_pore_sizes_for_psd_pocketwise function...")

    # Read the atomic structure from the randomly selected CIF file
    mat_atoms = read(path_to_cif)

    # Set grid spacing for distance grid computation (larger spacing = faster but less accurate)
    spacing = 0.25

    # Compute distance grid from atoms - represents distance to nearest atom at each grid point
    dgrid = mgr.dgrid_from_atoms_rust(mat_atoms, spacing=spacing)

    # Perform periodic segmentation to identify distinct pore regions/pockets
    seg_results = mgr.do_periodic_segmentation_rust(dgrid, mat_atoms, compute_connections=True, h=0.2, threshold_abs = 1.5, mask_thickness = 0.5, spacing=spacing, extra_padding=2)
    regions_pbc =seg_results['regions_pbc']  # Extract the periodic region labels
    regions = seg_results['regions']


    # cell matrix for the atoms with cell vectors as columns
    cell = cellpar_to_cell(mat_atoms.cell.cellpar()).T
    print(f"Segmentation complete at {time.time() - start:.2f} seconds")
    print(f"Grid shape: {dgrid.shape}")
    print(f"Cell matrix:\n{cell}")
    print(f"Unique regions: {np.unique(regions_pbc)}")

    try:
        # Call the new function
        result = mgr_rust.compute_pore_sizes_for_psd_pocketwise(
            dgrid,
            regions_pbc,
            cell,
            samples_per_region=1000  # Small number for testing
        )

        sample_points, pore_sizes, region_labels = result

        print(f"✓ PSD pore sizes executed successfully at {time.time() - start:.2f} seconds!")
        print(f"Sample points shape: {sample_points.shape}")
        print(f"Pore sizes shape: {pore_sizes.shape}")
        print(f"Region labels shape: {region_labels.shape}")

        print(f"Sample points (first 5):\n{sample_points[:5]}")
        print(f"Pore sizes (first 5): {pore_sizes[:5]}")
        print(f"Region labels (first 5): {region_labels[:5]}")

        # Basic validation
        assert sample_points.shape[0] == pore_sizes.shape[0] == region_labels.shape[0]
        assert sample_points.shape[1] == 3  # 3D coordinates
        assert pore_sizes.min() >= 0.0  # Pore sizes should be non-negative
        assert region_labels.min() >= 1  # Region labels should be positive (excluding 0)

        print("✓ All basic validations passed!")

        # Create region-wise PSD plot
        print("\nCreating region-wise PSD plots...")
        bins = np.linspace(0, 50, 200)  # histogram bins
        pocket_psds =compute_pocketwise_psd(region_labels, pore_sizes*2, bins=bins)

        # Plot the results
        if pocket_psds:
            print(f"Found {len(pocket_psds)} regions with sufficient samples")
            fig = plot_pocket_psds(pore_sizes*2, region_labels, pocket_psds, bins,
                                 material_name=path_to_cif.name)

            # Print summary statistics
            print("\n" + "="*80)
            print("SUMMARY STATISTICS FOR EACH POCKET:")
            print("="*80)
            for label, psd_data in pocket_psds.items():
                print(f"Pocket {label}:")
                print(f"  Number of samples: {psd_data['num_samples']}")
                print(f"  Mean pore size: {psd_data['mean_pore_size']:.2f} ± {psd_data['std_pore_size']:.2f} Å")
                print(f"  Max probability density: {np.max(psd_data['hist']):.4f}")
                print()

            print(f"Total number of accessible sample points: {len(sample_points)}")
            print(f"Number of unique pockets identified: {len(pocket_psds)}")
            print("="*80)
        else:
            print("No regions with sufficient samples found for plotting")

        return True

    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compute_pore_sizes_uniform():
    """Test the compute_pore_sizes_uniform function with the same data."""

    print("\nTesting compute_pore_sizes_uniform function...")

    try:
        # Read the atomic structure from the CIF file
        start = time.time()
        mat_atoms = read(path_to_cif)


        # Set grid spacing for distance grid computation
        spacing = 0.25

        # Compute distance grid from atoms
        dgrid = mgr.dgrid_from_atoms_rust(mat_atoms, spacing=spacing)

        # cell matrix for the atoms with cell vectors as columns
        cell = cellpar_to_cell(mat_atoms.cell.cellpar()).T

        print(f"Grid shape: {dgrid.shape}")
        print(f"Cell matrix:\n{cell}")

        # Call the uniform sampling function
        result = mgr_rust.compute_pore_sizes_uniform(
            dgrid,
            cell,
            num_samples_total=10000  # Total samples across entire unit cell
        )

        sample_points, pore_sizes = result
        print(f"pore size done at {time.time() - start:.2f} seconds!")
        print(f"✓ Uniform PSD executed successfully at {time.time() - start:.2f} seconds!")
        print(f"Sample points shape: {sample_points.shape}")
        print(f"Pore sizes shape: {pore_sizes.shape}")

        print(f"Sample points (first 5):\n{sample_points[:5]}")
        print(f"Pore sizes (first 5): {pore_sizes[:5]}")

        # Basic validation
        assert sample_points.shape[0] == pore_sizes.shape[0]
        assert sample_points.shape[1] == 3  # 3D coordinates
        assert pore_sizes.min() >= 0.0  # Pore sizes should be non-negative

        print("✓ All basic validations passed!")
    
        bins = np.linspace(0, 50, 200)  # Same bins as region-based test
        histogram_trace = get_pore_size_distribution_trace(pore_sizes*2, normalize=False, cumulative=False, bins=bins)
        dgrid_interpolator = mgr.interpolate_dgrid(dgrid)
        # sample_min = np.min(dgrid_interpolator(sample_points))
        # print(f"Sample minimum: {sample_min:.3f}")
        # Create histogram for uniform sampling
        #

        # hist, bin_edges = np.histogram(pore_sizes * 2, bins=bins, density=False)
        # bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        #
        # Print summary statistics for uniform sampling
        print("UNIFORM SAMPLING STATISTICS:")
        print(f"  Number of samples: {len(pore_sizes)}")
        print(f"  Mean pore size: {np.mean(pore_sizes * 2):.2f} ± {np.std(pore_sizes * 2):.2f} Å")
        # print(f"  Max probability density: {np.max(hist):.4f}")
        fig = go.Figure()
        fig.add_trace(histogram_trace)
        fig.update_layout(
            title=f"Pore Size Distribution: Overall vs Individual Pockets for {material}",
            xaxis_title='Pore Diameter (Å)',
            yaxis_title='Counts',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            width=1000,
            height=600
        )
        fig.show(renderer='chrome')

        return True

    except Exception as e:
        print(f"✗ Error in compute_pore_sizes_uniform: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dgrid_from_atoms():
    """Test the existing dgrid_from_atoms function to ensure library works."""
    print("\nTesting dgrid_from_atoms function...")

    try:
        # Create simple test data
        fpoints = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)  # Atom at center
        cell = np.eye(3, dtype=np.float64) * 10.0  # 10 Å cubic cell
        radii = np.array([1.0], dtype=np.float64)  # Atomic radius

        nx, ny, nz = 5, 5, 5

        result = mgr_rust.dgrid_from_atoms(fpoints, cell, radii, nx, ny, nz)

        print(f"✓ dgrid_from_atoms executed successfully!")
        print(f"Result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")
        print(f"Result range: [{result.min():.3f}, {result.max():.3f}]")

        return True

    except Exception as e:
        print(f"✗ Error in dgrid_from_atoms: {e}")
        import traceback
        traceback.print_exc()
        return False

def compute_psds_from_points_pocketwise(sample_points_labels, pore_sizes, bins=np.linspace(0, 50, 200)):
    """
    Compute pore size distributions for each unique region/pocket instead of the whole unit cell.

    This function:
    1. Identifies unique region labels from the sample points
    2. Groups sample points by their region labels
    3. Computes separate pore size distributions for each region
    4. Returns a dictionary mapping region labels to their PSDs

    Parameters:
    - sample_points_labels (ndarray): Array of region labels for each sample point
    - pore_sizes (ndarray): Array of pore sizes corresponding to each sample point
    - bins (ndarray): Bin edges for histogram computation

    Returns:
    - pocket_psds (dict): Dictionary where keys are region labels and values are dictionaries
                         containing 'hist' (histogram values) and 'bin_centers' (bin centers)
    """
    # Find all unique region labels (excluding background and invalid regions)
    unique_labels = np.unique(sample_points_labels)
    print(f"Unique region labels found: {unique_labels}")
    unique_labels = unique_labels[unique_labels > 0]  # Remove background (0) and any negative labels

    # Initialize dictionary to store PSDs for each pocket/region
    pocket_psds = {}

    print(f"Found {len(unique_labels)} unique regions/pockets")

    # Compute PSD for each unique region
    for label in unique_labels:
        # Find all sample points that belong to this region
        mask = sample_points_labels == label
        region_pore_sizes = pore_sizes[mask]

        # Skip regions with too few sample points for reliable statistics
        if len(region_pore_sizes) < 10:
            print(f"Skipping region {label} - insufficient sample points ({len(region_pore_sizes)})")
            continue

        # Filter out infinite or invalid pore sizes
        region_pore_sizes = region_pore_sizes[np.isfinite(region_pore_sizes)]

        if len(region_pore_sizes) == 0:
            print(f"Skipping region {label} - no valid pore sizes")
            continue

        # Compute histogram for this region's pore sizes
        hist, bin_edges = np.histogram(region_pore_sizes, bins=bins, density=False)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # Store the PSD data for this region
        pocket_psds[label] = {
            'hist': hist,
            'bin_centers': bin_centers,
            'num_samples': len(region_pore_sizes),
            'mean_pore_size': np.mean(region_pore_sizes),
            'std_pore_size': np.std(region_pore_sizes)
        }

        print(f"Region {label}: {len(region_pore_sizes)} samples, "
              f"mean pore size: {np.mean(region_pore_sizes):.2f} Å, "
              f"std: {np.std(region_pore_sizes):.2f} Å")

    return pocket_psds

def get_pore_size_distribution_trace(distances, bins=np.linspace(0, 50, 50), normalize=False, cumulative=False):
    """Create a plotly trace for pore size distribution."""
    import plotly.graph_objects as go
    if normalize:
        distances = distances / np.max(distances)

    # Create histogram data
    hist, bin_edges = np.histogram(distances, bins=bins, density=normalize)

    if cumulative:
        hist = np.cumsum(hist)
    binC = (bin_edges[1:] + bin_edges[:-1])/2
    # Create the histogram trace
    trace = go.Scatter(
        x=binC,
        y=hist,
        mode='lines',
        name='Overall Pore Size Distribution',
        line=dict(width=2)
    )

    return trace

def plot_pocket_psds(pore_sizes, sample_points_labels, pocket_psds, bins, material_name="Test Material"):
    """Create and display plotly figure comparing overall PSD with individual pocket PSDs."""

    # Get the histogram trace for overall PSD
    import plotly.graph_objects as go
    histogram_trace = get_pore_size_distribution_trace(pore_sizes, normalize=False, cumulative=False, bins=bins)

    # Create the figure
    fig = go.Figure()
    fig.add_trace(histogram_trace)

    # Add individual pocket PSDs
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (label, psd_data) in enumerate(pocket_psds.items()):
        # Create trace for this pocket's PSD
        pocket_trace = go.Scatter(
            x=psd_data['bin_centers'],
            y=psd_data['hist'],
            mode='lines',
            name=f'Pocket {label} ({psd_data["num_samples"]} samples)',
            line=dict(width=2, color=colors[i % len(colors)], dash='dot')
        )
        fig.add_trace(pocket_trace)

    # Update plot layout
    fig.update_layout(
        title=f"Pore Size Distribution: Overall vs Individual Pockets for {material_name}",
        xaxis_title='Pore Diameter (Å)',
        yaxis_title='Counts',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.8
        ),
        width=1000,
        height=600
    )

    # Display the plot
    # fig.show(renderer='chrome')
    return fig

def compute_pore_sizes_pocketwise_rust(dgrid, regions_pbc, ase_atoms, samples_per_region=1000):
    import mgr_rust
    from ase.geometry import cellpar_to_cell
    import numpy as np
    cell = cellpar_to_cell(ase_atoms.cell.cellpar()).T
    sample_points, pore_sizes, region_labels = mgr_rust.compute_pore_sizes_for_psd_pocketwise(
                dgrid,
                regions_pbc,
                cell,
                samples_per_region= samples_per_region# Small number for testing
            )
    return sample_points, pore_sizes, region_labels

def compute_psds_pocketwise_rust(dgrid, regions_pbc, ase_atoms, num_samples_per_region=1000, bins=None, return_fig=False, material='Material'):
    # Create region-wise PSD plot
    import numpy as np
    sample_points, pore_sizes, region_labels = compute_pore_sizes_pocketwise_rust(dgrid, regions_pbc, ase_atoms, samples_per_region=num_samples_per_region)
    if bins is None:
        bins = np.linspace(0, 50, 200)  # histogram bins
    print("\nCreating region-wise PSD plots...")
    pocket_psds = compute_psds_from_points_pocketwise(region_labels, pore_sizes*2, bins=bins)
    if return_fig:
        fig = plot_pocket_psds(pore_sizes*2, region_labels, pocket_psds, bins, material_name=material)
        return pocket_psds, fig
    else:
        return pocket_psds

def compute_pore_sizes_uniform_rust(dgrid, ase_atoms, num_samples_total=10000):
    import mgr_rust
    from ase.geometry import cellpar_to_cell
    import numpy as np
    cell = cellpar_to_cell(ase_atoms.cell.cellpar()).T
    sample_points, pore_sizes = mgr_rust.compute_pore_sizes_uniform(dgrid, cell, num_samples_total=num_samples_total)
    return sample_points, pore_sizes

def compute_psd_uniform_rust(dgrid, ase_atoms, num_samples_total=10000, bins=None, density=False, return_fig=False, material='Material'):
    # Create region-wise PSD plot
    import numpy as np
    import plotly.graph_objects as go
    sample_points, pore_sizes = compute_pore_sizes_uniform_rust(dgrid, ase_atoms,num_samples_total= num_samples_total)
    if bins is None:
        bins = np.linspace(0, 50, 200)  # histogram bins
    print("\nCreating region-wise PSD plots...")
    
    psd_hist, _ = np.histogram(pore_sizes * 2, bins=bins, density=density)
    if return_fig:
        histogram_trace = get_pore_size_distribution_trace(pore_sizes*2, normalize=False, cumulative=False, bins=bins)
        dgrid_interpolator = mgr.interpolate_dgrid(dgrid)
        print("UNIFORM SAMPLING STATISTICS:")
        print(f"  Number of samples: {len(pore_sizes)}")
        print(f"  Mean pore size: {np.mean(pore_sizes * 2):.2f} ± {np.std(pore_sizes * 2):.2f} Å")
        # print(f"  Max probability density: {np.max(hist):.4f}")
        fig = go.Figure()
        fig.add_trace(histogram_trace)
        fig.update_layout(
            title=f"Pore Size Distribution with uniform sampling across box {material}",
            xaxis_title='Pore Diameter (Å)',
            yaxis_title='Counts',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            width=1000,
            height=600
        )
        return psd_hist, fig
    else:
        return psd_hist

if __name__ == "__main__":
    print("=" * 50)
    print("PSD Rust Library Test")
    print("=" * 50)

    # Test basic library functionality
    test1_passed = test_dgrid_from_atoms()

    # Test new PSD pocketwise function
    test2_passed = test_compute_pore_sizes_pocketwise()
    # test2_passed = True


    # Test new PSD uniform function
    test3_passed = test_compute_pore_sizes_uniform()


    print("\n" + "=" * 50)
    if test1_passed and test2_passed and test3_passed:
        print("✓ All tests passed! Library is working correctly.")
        print("Check your browser for the interactive PSD plots!")
        print("The plotly figures should have opened in Chrome.")
    else:
        print("✗ Some tests failed. Check the output above.")
    print("=" * 50)

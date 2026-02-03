#!/usr/bin/env python3
"""
Test script to compare CLI vs direct Python usage with identical parameters
"""
import numpy as np
import time
import mgrcli_rust
from pathlib import Path

def test_direct_python_exact():
    """Test calling the Rust function directly with exact same parameters as CLI"""
    print("Testing direct Python call with exact CLI parameters...")

    start_time = time.time()

    result = mgrcli_rust.do_periodic_segmentation_rust(
        path_to_struct_file="cifs/LTA96.cif",
        spacing=0.5,
        mask_thickness=0.5,
        h=0.2,
        min_distance=1.5,
        threshold_abs=1.5,
        return_padded_regions=False,
        minimum_window_separation=1.0,
        wall_thickness=0.08,
        extra_padding=2.0,
        compactness=0,
        min_region_size=0.01,
        disable_tqdm=True,
        compute_connections=True,
        return_dgrid=True,
    )

    end_time = time.time()

    print(".2f")
    return result

def save_results_like_cli(result, output_file):
    """Save results exactly like the CLI does"""
    print(f"Saving results to {output_file}...")

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
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Test 1: Direct call without saving
    print("=== Test 1: Direct call without saving ===")
    result = test_direct_python_exact()

    # Print timing results if available
    if 'timing_results' in result:
        print("\n=== DIRECT CALL TIMING RESULTS ===")
        timing = result['timing_results']
        for key in sorted(timing.keys()):
            print(".4f")
        print("=== END DIRECT CALL TIMING ===")

    # Test 2: Direct call with saving (like CLI)
    print("\n=== Test 2: Direct call with saving (like CLI) ===")
    result2 = test_direct_python_exact()
    save_results_like_cli(result2, Path("test_results.npz"))

    # Print timing results if available
    if 'timing_results' in result2:
        print("\n=== DIRECT CALL WITH SAVING TIMING RESULTS ===")
        timing = result2['timing_results']
        for key in sorted(timing.keys()):
            print(".4f")
        print("=== END DIRECT CALL WITH SAVING TIMING ===")

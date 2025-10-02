#!/usr/bin/env python3
"""
Test script to compare direct Python usage vs CLI
"""
import numpy as np
import time
import mcli

# Import the Rust library directly
import mcli_rust
rust_lib = mcli_rust

def test_direct_python():
    """Test calling the Rust function directly from Python"""
    print("Testing direct Python call...")

    start_time = time.time()

    result = rust_lib.do_periodic_segmentation_rust(
        path_to_struct_file="cifs/LTA96.cif",
        spacing=0.5,
        mask_thickness=0.5,
        h=0.2,
        min_distance=1.5,
        threshold_abs=1.5,
        return_padded_regions=False,
        minimum_window_separation=None,
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

if __name__ == "__main__":
    result = test_direct_python()

    # Print timing results if available
    if 'timing_results' in result:
        print("\n=== PYTHON DIRECT TIMING RESULTS ===")
        timing = result['timing_results']
        for key in sorted(timing.keys()):
            print(".4f")
        print("=== END PYTHON DIRECT TIMING ===")

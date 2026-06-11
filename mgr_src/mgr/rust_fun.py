import numpy as np
import ase
import logging

from .vdw_radii import default_vdw_radii_angstrom

logger = logging.getLogger(__name__)



def _voxel_lengths_for_smoothing(ase_atoms, spacing, grid_shape=None):
    """Per-axis fractional grid step lengths in Å (for anisotropic Gaussian smoothing)."""
    from ase.geometry import cellpar_to_cell

    M = cellpar_to_cell(ase_atoms.cell.cellpar())
    cellpar = ase_atoms.cell.cellpar()
    if grid_shape is not None:
        nx, ny, nz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    else:
        nx, ny, nz = (cellpar[:3] / spacing).astype(int) + 1
    lx = float(np.linalg.norm(M[0]) / max(nx - 1, 1))
    ly = float(np.linalg.norm(M[1]) / max(ny - 1, 1))
    lz = float(np.linalg.norm(M[2]) / max(nz - 1, 1))
    return (lx, ly, lz)

def _cell_matrix_for_rust(ase_atoms):
    """Return C for Rust ``cart = C @ delta_frac`` (matches ASE ``frac @ M``)."""
    from ase.geometry import cellpar_to_cell

    return cellpar_to_cell(ase_atoms.cell.cellpar()).T


def dgrid_from_atoms_rust(ase_atoms, spacing, radii=None):
    """Compute a distance grid from atomic positions using the Rust backend.

    Uses optimized exact MIC (SoA, SIMD on orthogonal cells, Wassenaar 7-image
    fallback on skewed cells).

    This function calculates the distance field (dgrid) for a given atomic structure,
    where each grid point represents the minimum distance to the nearest atom surface,
    accounting for van der Waals radii.

    :param ase_atoms: ASE Atoms object containing the atomic structure
    :type ase_atoms: ase.Atoms
    :param spacing: Grid spacing in Angstroms
    :type spacing: float
    :param radii: Van der Waals radii for each atom. If None, uses the CRC/Bondi
        (1964) VDW table from :func:`~mofography.vdw_radii.default_vdw_radii_angstrom`.
        Pass ``mendeleev_vdw_radii_angstrom(atoms)`` for the legacy Mendeleev table.
    :type radii: numpy.ndarray or None
    :return: 3D distance grid where values represent distances to atom surfaces
    :rtype: numpy.ndarray
    """
    import mgrcli_rust

    logger.info(f"Using {mgrcli_rust.rayon_num_threads()} threads for parallel processing")

    if radii is None:
        radii = default_vdw_radii_angstrom(ase_atoms)

    fpoints = ase_atoms.get_scaled_positions() - 0.5
    cell = _cell_matrix_for_rust(ase_atoms)
    nx, ny, nz = (ase_atoms.cell.cellpar()[:3] / spacing).astype(int) + 1
    dgrid = mgrcli_rust.dgrid_from_atoms(fpoints, cell, radii, int(nx), int(ny), int(nz))
    return dgrid


def dgrid_from_atoms_exp_rust(ase_atoms, spacing, radii=None):
    """Deprecated alias for :func:`dgrid_from_atoms_rust`."""
    return dgrid_from_atoms_rust(ase_atoms, spacing, radii=radii)


def dgrid_from_atoms_aabb_rust(ase_atoms, spacing, radii=None, n_neighbors=10):
    """Compute a distance grid using the Rust AABB/PBC neighbor search backend.

    :param ase_atoms: ASE Atoms object containing the atomic structure
    :type ase_atoms: ase.Atoms
    :param spacing: Grid spacing in Angstroms
    :type spacing: float
    :param radii: Van der Waals radii for each atom. If None, uses the CRC/Bondi
        (1964) VDW table from :func:`~mofography.vdw_radii.default_vdw_radii_angstrom`.
        Pass ``mendeleev_vdw_radii_angstrom(atoms)`` for the legacy Mendeleev table.
    :type radii: numpy.ndarray or None
    :param n_neighbors: Number of nearest neighbors to query from the AABB tree, defaults to 10
    :type n_neighbors: int
    :return: 3D distance grid where values represent distances to atom surfaces
    :rtype: numpy.ndarray
    """
    import mgrcli_rust

    logger.info(f"Using {mgrcli_rust.rayon_num_threads()} threads for parallel processing")

    if radii is None:
        radii = default_vdw_radii_angstrom(ase_atoms)

    fpoints = ase_atoms.get_scaled_positions() - 0.5
    cell = _cell_matrix_for_rust(ase_atoms)
    nx, ny, nz = (ase_atoms.cell.cellpar()[:3] / spacing).astype(int) + 1
    return mgrcli_rust.dgrid_from_atoms_aabb(
        fpoints, cell, int(nx), int(ny), int(nz), radii, int(n_neighbors)
    )


def dgrid_from_atoms_aabb_interp_rust(
    ase_atoms, spacing, coarse_factor=2, radii=None, n_neighbors=10
):
    """Compute a smoothed distance grid using coarse AABB sampling + trilinear interpolation.

    :param ase_atoms: ASE Atoms object containing the atomic structure
    :param spacing: Grid spacing in Angstroms (fine grid)
    :param coarse_factor: Coarsening factor (>1) for the AABB sampling grid; 2 is a mild smoothing
    :param radii: Van der Waals radii for each atom. If None, uses the CRC/Bondi (1964) VDW table from :func:`~mofography.vdw_radii.default_vdw_radii_angstrom`
    :param n_neighbors: Number of nearest neighbors to query from the AABB tree, defaults to 10
    :return: 3D distance grid (float32) with mild smoothing from interpolation
    """
    import mgrcli_rust

    logger.info(f"Using {mgrcli_rust.rayon_num_threads()} threads for parallel processing")

    if radii is None:
        radii = default_vdw_radii_angstrom(ase_atoms)

    fpoints = ase_atoms.get_scaled_positions() - 0.5
    cell = _cell_matrix_for_rust(ase_atoms)
    nx, ny, nz = (ase_atoms.cell.cellpar()[:3] / spacing).astype(int) + 1

    return mgrcli_rust.dgrid_from_atoms_aabb_interp(
        fpoints,
        cell,
        int(nx),
        int(ny),
        int(nz),
        int(coarse_factor),
        radii,
        int(n_neighbors),
    )


def egrid_from_atoms_rust(
    ase_atoms, spacing, forcefield_mixed=None, cutoff=12.5, probe_symbol="He", return_ncells=False
):
    """Compute an energy grid from atomic positions using the Rust backend.

    This function calculates the energy field (egrid) for a given atomic structure,
    where each grid point represents the Lennard-Jones interaction energy with
    the framework atoms, using a probe molecule.

    :param ase_atoms: ASE Atoms object containing the atomic structure
    :type ase_atoms: ase.Atoms
    :param spacing: Grid spacing in Angstroms
    :type spacing: float
    :param forcefield_mixed: Mixed forcefield parameters. If None, computed using Lorentz-Berthelot mixing
    :type forcefield_mixed: numpy.ndarray or None
    :param cutoff: Cutoff distance for energy calculations in Angstroms
    :type cutoff: float
    :param probe_symbol: Chemical symbol of the probe molecule
    :type probe_symbol: str
    :return: 3D energy grid where values represent interaction energies
    :rtype: numpy.ndarray
    """
    import mgrcli_rust
    import mofography as mgr
    from ase.geometry import cellpar_to_cell
    from .compute import get_repetitions_to_meet_cutoff
    logger.info(f"Using {mgrcli_rust.rayon_num_threads()} threads for parallel processing")
    if forcefield_mixed is None:
        forcefield_mixed = mgr.mix_lorentz_berthelot(
            ase_atoms, probe_symbol=probe_symbol
        )
    mat_atoms_repeat, ff_repeat = mgr.repeat_to_meet_cutoff(
        ase_atoms, forcefield_mixed=forcefield_mixed, cutoff=cutoff
    )
    eps, sig = ff_repeat[:, 0], ff_repeat[:, 1]
    frac = mat_atoms_repeat.get_scaled_positions() - 0.5  # centre to [-0.5,0.5]
    cell = cellpar_to_cell(mat_atoms_repeat.cell.cellpar()).T
    # cell = mat_atoms_repeat.get_cell().complete().T        # 3×3 column-vector matrix
    # nx, ny, nz = (mat_atoms_repeat.cell.cellpar()[:3] / spacing).astype(int) + 1
    
    # NOTE: define grid over just one unit cell, exactly like python version, and then repeat exactly
    nx_unit, ny_unit, nz_unit = (ase_atoms.cell.cellpar()[:3]/spacing).astype(int) + 1
    nx_cells, ny_cells, nz_cells = get_repetitions_to_meet_cutoff(ase_atoms, cutoff=cutoff)
    nx, ny, nz = nx_unit * nx_cells, ny_unit * ny_cells, nz_unit * nz_cells # grid points for supercell
    # compute supercell egrid using rust. this is easier for parallelization in rust since we don
    # 't have a cell list
    egrid_super = mgrcli_rust.egrid_from_atoms(frac, cell, eps, sig, nx, ny, nz, cutoff=cutoff)
    egrid = egrid_super[:nx_unit, :ny_unit, :nz_unit] # grab just one unit cell that's all we need

    if return_ncells:
        return egrid, get_repetitions_to_meet_cutoff(ase_atoms, cutoff=cutoff)
    else:
        return egrid


def seeded_watershed_det_rust(image, seeds, mask=None):
    """Deterministic single-threaded seeded watershed wrapper around mgrcli_rust.

    Parameters
    ----------
    image : ndarray, float32
        3D scalar field; larger values flood first. For a distance grid, pass
        the raw ``dgrid`` (do NOT negate).
    seeds : ndarray, uint32
        3D array of markers (0 = background).
    mask : ndarray, uint8 or bool, optional
        3D mask array; voxels with mask == 0 are never labeled.
    """
    import numpy as np
    import mgrcli_rust

    img32 = np.asarray(image, dtype=np.float32)
    seeds32 = np.asarray(seeds, dtype=np.uint32)
    if mask is not None:
        mask8 = np.asarray(mask, dtype=np.uint8)
        return mgrcli_rust.seeded_watershed_det(img32, seeds32, mask8)
    else:
        return mgrcli_rust.seeded_watershed_det(img32, seeds32, None)


def seeded_watershed_det_parallel_rust(image, seeds, mask=None):
    """Deterministic parallel seeded watershed wrapper around mgrcli_rust.

    Parameters
    ----------
    image : ndarray, float32
        3D scalar field; larger values flood first. For a distance grid, pass
        the raw ``dgrid`` (do NOT negate).
    seeds : ndarray, uint32
        3D array of markers (0 = background).
    mask : ndarray, uint8 or bool, optional
        3D mask array; voxels with mask == 0 are never labeled.
    """
    import numpy as np
    import mgrcli_rust

    img32 = np.asarray(image, dtype=np.float32)
    seeds32 = np.asarray(seeds, dtype=np.uint32)
    if mask is not None:
        mask8 = np.asarray(mask, dtype=np.uint8)
        return mgrcli_rust.seeded_watershed_det_parallel(img32, seeds32, mask8)
    else:
        return mgrcli_rust.seeded_watershed_det_parallel(img32, seeds32, None)



def _seeded_watershed_backend(
    padded_grid,
    seeds,
    mask,
    backend,
    *,
    compactness=0,
    watershed_connectivity=3,
):
    """Dispatch seeded watershed to the selected Rust/skimage backend."""
    import mgrcli_rust

    img = padded_grid.astype(np.float32)
    seeds_u32 = seeds.astype(np.uint32)
    mask_u8 = mask.astype(np.uint8)

    if backend == "rust_bucket":
        return mgrcli_rust.seeded_watershed_bucket(img, seeds_u32, mask_u8)
    if backend == "rust_det":
        return mgrcli_rust.seeded_watershed_det(img, seeds_u32, mask_u8)
    if backend == "rust_det_parallel":
        return mgrcli_rust.seeded_watershed_det_parallel(img, seeds_u32, mask_u8)
    if backend == "skimage":
        from skimage.segmentation import watershed as sk_watershed

        return sk_watershed(
            -padded_grid,
            seeds.astype(np.int32),
            mask=mask.astype(bool),
            compactness=compactness,
            connectivity=watershed_connectivity,
        ).astype(np.uint32)
    raise ValueError(
        f"Unknown watershed_backend '{backend}'. Expected 'rust_bucket', 'rust_det', "
        "'rust_det_parallel', or 'skimage'."
    )





def do_periodic_segmentation_rust( dgrid, mat_atoms, mask_thickness=0.05, h=0.2, min_distance=1.5, threshold_abs=1.5, preserve_threshold=3.0, smooth_threshold=6.0, max_smooth_radius=2.0, return_padded_regions=False, spacing=None, minimum_window_separation=None, wall_thickness=0.05, extra_padding=5, compactness=0, min_region_size=0.001, disable_tqdm=False, compute_connections=True, return_dgrid=False, watershed_backend="rust_det", watershed_connectivity=3, skip_endchannel_pairs=True):
    """Perform periodic segmentation on a distance grid to identify pore regions.

    This function implements a comprehensive pore segmentation algorithm that:
    1. Detects local maxima in the distance field
    2. Applies periodic boundary conditions
    3. Performs watershed segmentation
    4. Identifies connections between regions
    5. Returns detailed segmentation results

    :param dgrid: 3D distance grid from dgrid_from_atoms_rust
    :type dgrid: numpy.ndarray
    :param mat_atoms: ASE Atoms object of the material
    :type mat_atoms: ase.Atoms
    :param mask_thickness: Threshold for void space identification
    :type mask_thickness: float
    :param h: Persistence parameter for maxima detection
    :type h: float
    :param min_distance: Minimum distance of separationg required between distinct local maxima
    :type min_distance: float
    :param threshold_abs: Absolute threshold for maxima detection
    :type threshold_abs: float
    :param preserve_threshold: Distance cutoff (Å) below which maxima are untouched during smoothing
    :type preserve_threshold: float
    :param smooth_threshold: Distance threshold (Å) above which maxima are always smoothed
    :type smooth_threshold: float
    :param max_smooth_radius: Maximum radius (Å) used when smoothing maxima
    :type max_smooth_radius: float
    :param return_padded_regions: Whether to return padded region arrays
    :type return_padded_regions: bool
    :param spacing: Grid spacing. If None, estimated from dgrid size and cell lengths. Specify for better accuracy.
    :type spacing: float or None
    :param minimum_window_separation: Minimum separation for window connections
    :type minimum_window_separation: float or None
    :param wall_thickness: Thickness of wall regions to exclude
    :type wall_thickness: float
    :param extra_padding: Additional padding for periodic calculations in angstroms. This over and above the standard_padding + 5 A cushion.
    :type extra_padding: float
    :param compactness: Watershed compactness parameter
    :type compactness: float
    :param min_region_size: Minimum region size as fraction of total volume, regions smaller than this in comparison with the *padded_grid* are removed
    :type min_region_size: float
    :param disable_tqdm: Whether to disable progress bars
    :type disable_tqdm: bool
    :param compute_connections: Whether to compute inter-region connections
    :type compute_connections: bool
    :param return_dgrid: Whether to include dgrid in output
    :type return_dgrid: bool
    :param watershed_backend: Which watershed backend to use: \"rust_det\" (default, deterministic Rust heap), \"rust_bucket\" (parallel bucket flood), \"rust_det_parallel\" (deterministic parallel), or \"skimage\".
    :param skip_endchannel_pairs: If True (default), do not merge tiny wall-touching end-channel slivers into neighbors. Set False only for cosmetic visualization cleanup.
    :type watershed_backend: str
    :param watershed_connectivity: Connectivity for skimage watershed (2=18-NN, 3=26-NN). Only used when watershed_backend="skimage".
    :type watershed_connectivity: int
    :return: Dictionary containing segmentation results including maxima, regions, connections, etc.
    :rtype: dict
    """
    import numpy as np
    import mgrcli_rust

    # from skimage.segmentation import watershed
    from ase.geometry import cellpar_to_cell
    import copy

    # from ase.geometry import complete_cell
    # import sparse
    # from skimage.segmentation import find_boundaries
    # from concurrent.futures import ThreadPoolExecutor
    from time import time

    # from tqdm import tqdm
    import mgrcli_rust

    logger.info(f"Using {mgrcli_rust.rayon_num_threads()} threads for parallel processing")
    logger.info("Watershed backend: %s", watershed_backend)
    start_time = time()
    from mofography import (
        compute_wall_cells,
        move_maximum_to_peak,
        process_wall_maxima,
        is_inside_or_close_to_boundary,
        get_fractional_coordinates,
        interpolate_labels,
        find_touching_region_pairs,
        find_end_channel_pairs,
        merge_small_regions,
        merge_small_regions_pbc,
        find_pore_limiting_radius,
        calculate_padding_thickness,
        detect_periodic_maxima_direct,
        get_pbc_groups,
    )

    # print(f"timer started")
    logger.info(f"timer started")
    # h = 0.11

    # Precompute metric-aware weights if needed by the selected backend (done
    # after cell matrix is available below)
    # note: uncomment these imports when refactoring
    # from .helper import find_touching_region_pairs,compute_wall_cells, move_maximum_to_peak, process_wall_maxima, is_inside_or_close_to_boundary, get_fractional_coordinates, interpolate_labels
    # smooth = false
    if spacing is None:
        logger.warning(
            "warning: no spacing provided. estimating. provide the dgrid spacing, for more accurate calculation!\n"
        )
        spacing = np.min(
            mat_atoms.get_cell_lengths_and_angles()[:3] / (np.array(dgrid.shape) - 1)
        )
        logger.info(f"spacing estimated from dgrid and ase_atoms is {spacing}")

    if minimum_window_separation is None:
        minimum_window_separation = spacing * 3

    logger.info(f"h is {h}")
    plr = find_pore_limiting_radius(dgrid)

    logger.info(
        "time(s): %.2f pore limiting radius is %.2f\n"
        % (np.round(time() - start_time, decimals=2), np.round(plr, decimals=2))
    )
    if (plr - 0.25) < mask_thickness:
        logger.warning(
            "warning: mask thickness is too high, resetting to pore limiting radius - 0.25\n"
        )

    mask_thickness = np.min([plr - 0.25, mask_thickness])
    # min_distance = 1.1
    #     threshold_abs = 1
    # # merged_maxima, all_maxima = detect_periodic_maxima_dgrid(dgrid, ase_atoms=mat_atoms, spacing=spacing, h=h, threshold_abs=threshold_abs, min_distance=min_distance, sigma_smooth=1.0, cell=cell)
    # dt, merged_maxima, all_maxima =detect_periodic_maxima_direct(
    #     dgrid,
    #     mat_atoms,
    #     spacing,
    #     h=h,
    #     threshold_abs=threshold_abs,
    #     min_distance=min_distance,
    #     # smooth=smooth,
    #     maxima_mask=mask_thickness,
    # )

    cell = cellpar_to_cell(mat_atoms.cell.cellpar()).T
    voxel_lengths = _voxel_lengths_for_smoothing(
        mat_atoms, spacing, grid_shape=dgrid.shape
    )
    # merged_maxima, unmerged_maxima, raw_maxima = mgrcli_rust.detect_maxima_h_morph(dgrid, spacing=spacing, h=h, threshold_abs=threshold_abs, min_distance=min_distance, sigma_smooth=1.0, cell=cell)
    merged_maxima, unmerged_maxima, raw_maxima = (
        mgrcli_rust.detect_maxima_selective_smoothing_parallel(
            dgrid,
            preserve_threshold=preserve_threshold,
            smooth_threshold=smooth_threshold,
            max_smooth_radius=max_smooth_radius,
            h=h,
            threshold_abs=threshold_abs,
            min_distance=min_distance,
            cell=cell,
            voxel_lengths=voxel_lengths,
        )
    )

    # Merged (PBC) and unmerged floods share the same backend for parity checks.
    merged_watershed_backend = watershed_backend
    dt = dgrid.copy()
    temp = np.ceil(merged_maxima)
    # compute wall cells
    logger.info(
        "time(s): %.2f computing maxima with pbc\n"
        % np.round(time() - start_time, decimals=2)
    )
    # computed merged maxima nad regions_pbc!
    # padding_thickness=5
    padding_thickness = calculate_padding_thickness(
        dgrid, mat_atoms, spacing
    ) + np.ceil(extra_padding / spacing).astype(int)
    logger.info(f"padding thickness is {padding_thickness}")
    wall_cells = compute_wall_cells(mat_atoms, spacing, wall_thickness)
    wall_maxima_to_remove = process_wall_maxima(
        merged_maxima,
        dgrid,
        padding_thickness=padding_thickness,
        wall_cells=wall_cells,
        ase_atoms=mat_atoms,
    )
    wall_maxima_to_remove = []
    # logger.info("maxima to remove are", wall_maxima_to_remove)
    # optimized_positions = []
    # optimized_values = []
    # original_indices = []
    # for i, pos in enumerate(merged_maxima):
    #     if i not in wall_maxima_to_remove:
    #         # padded_pos = pos + padding_thickness
    #         opt_pos, opt_val = move_maximum_to_peak(pos, dt, padding_thickness)
    #         optimized_positions.append(opt_pos)
    #         optimized_values.append(opt_val)
    #         original_indices.append(i)
    # optimized_positions = np.vstack(optimized_positions)
    # optimized_positions.shape
    # use the rust functions instead
    # * optimize the maxima positions (unused)
    optimized_positions, optimized_values = mgrcli_rust.move_maxima_to_peaks(
        dt.astype(np.float32), merged_maxima.astype(np.uintp)
    )
    optimized_positions = np.asarray(optimized_positions)

    all_removals = set(wall_maxima_to_remove)
    merged_maxima = np.vstack(
        [m for i, m in enumerate(merged_maxima) if i not in all_removals]
    )

    logger.info(
        "time(s): %.2f computing regions with pbc \n"
        % np.round(time() - start_time, decimals=2)
    )
    # compactness = 0
    # compute markers for padded, periodic segmentation
    markers = np.zeros_like(dgrid, dtype=int)
    if len(merged_maxima) > 0:
        maxima_coords = tuple(merged_maxima.T)
        markers[maxima_coords] = np.arange(1, len(merged_maxima) + 1)
    else:
        logger.info("problem: no valid maxima detected after merging.")
    # void mask for segmentation to use the mask_thickness
    void_mask = dgrid > mask_thickness
    padded_grid = np.pad(dt, padding_thickness, mode="wrap")
    padded_markers = np.pad(markers, padding_thickness, mode="wrap")
    padded_void = np.pad(void_mask, padding_thickness, mode="wrap")

    # # run the watershed segmentation using the merged markers.
    # padded_regions = watershed(
    #     -padded_grid,
    #     padded_markers,
    #     mask=padded_void,
    #     compactness=compactness,
    #     connectivity=2,  # 26-connectivity for full 3d
    # )

    # run the watershed segmentation using the merged markers.
    padded_regions = _seeded_watershed_backend(
        padded_grid,
        padded_markers,
        padded_void,
        merged_watershed_backend,
        compactness=compactness,
        watershed_connectivity=watershed_connectivity,
    )

    logger.info(
        "TIME(s): %.2f Looking to remove regions smaller than %f * dgrid.size"
        % (np.round(time() - start_time, decimals=2), min_region_size)
    )
    # clean up the regions_pbc from small regions, remove less than 0.1% of the total size, this is trickier than one would think. We want to find the volume % of just the box, then make our selection to remove, and then pad back up to get the cleaned padded grid.
    padded_regions_box_trimmed = padded_regions[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]
    padded_regions_box_trimmed_cleaned = merge_small_regions(
        padded_regions_box_trimmed,
        min_size=padded_regions_box_trimmed.size * min_region_size,
    )
    padded_regions_cleaned = np.pad(
        padded_regions_box_trimmed_cleaned, padding_thickness, mode="wrap"
    )

    # padded_regions_cleaned = merge_small_regions(padded_regions, min_size=padded_regions.size*min_region_size)

    unique_padded_regions = np.unique(padded_regions)
    unique_padded_regions_cleaned = np.unique(padded_regions_cleaned)
    removed_labels_pbc = unique_padded_regions[
        ~np.isin(unique_padded_regions, unique_padded_regions_cleaned)
    ]
    logger.info(
        "Found that labels are too small within the box even with pbc %s"
        % removed_labels_pbc
    )

    # TODO: is this necesary? isnt' everytihng an original maxima this point.
    # only keep removed labels that correspond to original maxima
    max_label = len(merged_maxima)
    removed_maxima_labels = removed_labels_pbc[
        (removed_labels_pbc >= 1) & (removed_labels_pbc <= max_label)
    ]
    if len(removed_maxima_labels) > 0:
        logger.info(f"Removing these small maxima labels:{removed_maxima_labels}")

    # * remove those rows from merged_maxima nd redo segmentations
    merged_maxima = np.delete(merged_maxima, removed_maxima_labels - 1, axis=0)

    if len(removed_maxima_labels) > 0:
        logger.info(
            "TIME(s): %.2f Redoing segmentation for clean sequential labels in regions pbc"
            % np.round(time() - start_time, decimals=2)
        )
        markers = np.zeros_like(dgrid, dtype=int)
        if len(merged_maxima) > 0:
            maxima_coords = tuple(merged_maxima.T)
            markers[maxima_coords] = np.arange(1, len(merged_maxima) + 1)
        else:
            logger.info("problem: no valid maxima detected after merging.")
        # void mask for segmentation to use the mask_thickness
        void_mask = dgrid > mask_thickness
        padded_grid = np.pad(dt, padding_thickness, mode="wrap")
        padded_markers = np.pad(markers, padding_thickness, mode="wrap")
        padded_void = np.pad(void_mask, padding_thickness, mode="wrap")

        # run the watershed segmentation using the merged markers.
        padded_regions = _seeded_watershed_backend(
            padded_grid,
            padded_markers,
            padded_void,
            merged_watershed_backend,
            compactness=compactness,
            watershed_connectivity=watershed_connectivity,
        )

    # replace regions_pbc with regions_pbc_cleaned
    # padded_regions=padded_regions_cleaned

    # extract central (unpadded) region to obtain regions_pbc.
    regions_pbc = padded_regions[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]
    # print("computed merged maxima nad regions_pbc!!")

    #
    logger.info(
        "time(s): %.2f computing maxima and regions without pbc\n"
        % np.round(time() - start_time, decimals=2)
    )

    # step 2: get unmerged maxima info using valid maxima
    original_shape = np.array(dgrid.shape)
    padded_shape = np.array(padded_grid.shape)

    # Stable periodic seed IDs regardless of watershed backend / merge order.
    merged_maxima = np.asarray(merged_maxima, dtype=int)
    if merged_maxima.size:
        order = np.lexsort(
            (merged_maxima[:, 2], merged_maxima[:, 1], merged_maxima[:, 0])
        )
        merged_maxima = merged_maxima[order]

    new_maxima_info_list = []
    counter = 1
    for m in merged_maxima:
        m_pad = m + padding_thickness
        offset_ranges = [
            range(-((p - 1) // o), (p - 1) // o + 1)
            for p, o in zip(padded_shape, original_shape)
        ]
        for offs in np.array(np.meshgrid(*offset_ranges)).T.reshape(-1, 3):
            candidate = m_pad + offs * original_shape
            if np.all(candidate >= 0) and np.all(candidate < padded_shape):
                candidate_unpadded = candidate - padding_thickness
                offset = candidate_unpadded // original_shape
                image_coord = np.mod(candidate_unpadded, original_shape)
                new_maxima_info_list.append(
                    [
                        counter,
                        image_coord[0],
                        image_coord[1],
                        image_coord[2],
                        offset[0],
                        offset[1],
                        offset[2],
                    ]
                )
                counter += 1

    maxima_info = np.array(new_maxima_info_list, dtype=int)


    # rust version goes here--
    # maxima_info = mgrcli_rust.build_maxima_info(merged_maxima.astype(np.uintp), np.array(dgrid.shape, dtype=np.uintp), int(padding_thickness))
    # maxima_info = np.asarray(maxima_info, dtype=int)

    # print(maxima_info)
    # merged_maxima
    # create unmerged markers and regions using valid maxima
    unmerged_markers = np.zeros_like(padded_grid, dtype=int)
    for info in maxima_info:
        # warn: should we add the offset times shape here?  fixed
        pos = info[1:4] + padding_thickness + info[4:7] * original_shape
        unmerged_markers[tuple(pos)] = info[0]

    # # compute unmerged regions
    #     unmerged_padded_regions = watershed(
    #         -padded_grid,
    #         unmerged_markers,
    #         mask=padded_void,
    #         compactness=compactness,
    #         connectivity=2,
    #     )

    # compute unmerged regions
    unmerged_padded_regions = _seeded_watershed_backend(
        padded_grid,
        unmerged_markers,
        padded_void,
        watershed_backend,
        compactness=compactness,
        watershed_connectivity=watershed_connectivity,
    )

    # grab the central block
    unmerged_regions = unmerged_padded_regions[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]

    logger.info(
        "time(s): %.2f computed regions without pbc\n"
        % np.round(time() - start_time, decimals=2)
    )
    if skip_endchannel_pairs:
        endchannel_pairs = []
    else:
        endchannel_pairs = find_end_channel_pairs(
            regions_pbc, unmerged_regions, size_ratio=0.3
        )
    # so we need to replace only if the removed label is far outside the unit cell
    wall_region_thicknesses = compute_wall_cells(
        mat_atoms, spacing, wall_thickness=wall_thickness
    )
    # let's start by keeping only those maxima that are there in the regions array
    maxima_indices = maxima_info[:, 1:4] + maxima_info[:, 4:7] * np.array(dgrid.shape)
    # get location classifications for all maxima
    inside_close_outside = is_inside_or_close_to_boundary(
        maxima_indices, dgrid.shape, wall_region_thicknesses
    )
    # create a boolean mask for maxima that are either inside or close_outside
    valid_maxima_mask = np.array(
        [loc in ["inside", "close_outside"] for loc in inside_close_outside]
    )
    # find the core maxima ids (region labels), don't forget to add 1 to get label from index
    valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1

    for p in endchannel_pairs:
        # print(f"Trying to replace {p[0]} with {p[1]}")
        i_max = np.where(maxima_info[:, 0] == p[0])[0][0]
        # print('label for i_max', maxima_info[i_max][0])
        j_max = np.where(maxima_info[:, 0] == p[1])[0][0]
        # print('label for j_max', maxima_info[j_max][0])
        # if np.logical_and(np.any(maxima_info[i_max, 4:7]!=0), np.all(maxima_info[j_max, 4:7]==0)):
        if p[0] not in valid_maxima_ids:
            logger.info(
                f"replacing far out {maxima_info[i_max]} with {maxima_info[j_max]}"
            )
            unmerged_regions[unmerged_regions == p[0]] = p[1]

    # print('looking for noisy regions, counts < 0.001 * regions.size')
    # unmerged_regions_cleaned =  merge_small_regions(unmerged_regions, min_size=unmerged_regions.size * 0.001)
    # unique_regions = np.unique(unmerged_regions)
    # unique_regions_cleaned = np.unique(unmerged_regions_cleaned)
    # # find the labels that were removed, if any.
    # removed_labels = unique_regions[~np.isin(unique_regions, unique_regions_cleaned)]
    # # remove the corresponding maxima from maxima_info array
    # maxima_info = np.delete(maxima_info, removed_labels - 1, axis=0)
    # # finally replace unmerged regions with the cleaned version
    # unmerged_regions = unmerged_regions_cleaned
    #
    # set_trace()
    logger.info(f"time(s):', {np.round(time() - start_time, decimals=2)} ,'reindexng\n")

    # note: we need to reindex here,
    # index mapping for the merged_maxima in unmerged parts
    # ensure merged maxima indices run from 1 to n
    wall_region_thicknesses = compute_wall_cells(
        mat_atoms, spacing, wall_thickness=wall_thickness
    )
    # let's start by keeping only those maxima that are there in the regions array
    maxima_indices = maxima_info[:, 1:4] + maxima_info[:, 4:7] * np.array(dgrid.shape)
    # get location classifications for all maxima
    inside_close_outside = is_inside_or_close_to_boundary(
        maxima_indices, dgrid.shape, wall_region_thicknesses
    )
    # create a boolean mask for maxima that are either inside or close_outside
    valid_maxima_mask = np.array(
        [loc in ["inside", "close_outside"] for loc in inside_close_outside]
    )
    # find the core maxima ids (region labels), don't forget to add 1 to get label from index
    valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1
    # print("computed maxima close to the cell")
    valid_maxima_indices = maxima_indices[valid_maxima_mask]
    # reindexing logic
    merged_maxima_indices = {tuple(m): i + 1 for i, m in enumerate(merged_maxima)}
    logger.info("constructing remap dict\n")
    used_indices = set()
    remap_dict = {}

    # note: assign indices to periodic maxima
    for info in maxima_info:
        original_maxima = tuple(info[1:4])
        if np.all(info[4:7] == 0) and original_maxima in merged_maxima_indices:
            # remap_dict[info[0]] = merged_maxima_indices[original_maxima]
            # if info[0] in remap_dict and remap_dict[info[0]] != merged_maxima_indices[original_maxima]:
            target_index = merged_maxima_indices[original_maxima]
            # check if the target index is already used
            if target_index in used_indices:
                logger.info(
                    "conflict: maxima %s maps to already-used index %s"
                    % (info[0], target_index)
                )
            else:
                remap_dict[info[0]] = target_index
                used_indices.add(target_index)

            # remap_dict[info[0]] = merged_maxima_indices[original_maxima]

    # print('stage 1 maxima_pbc', remap_dict)
    unique_unmerged_regions = np.delete(np.unique(unmerged_regions), 0)
    # print(
    #     "all closeby maxima are in unmerged regions",
    #     np.all(np.isin(valid_maxima_ids, unique_unmerged_regions)),
    # )
    # note: assign indices to valid maxima
    # new_index = len(merged_maxima) + 1
    next_index = max(used_indices) + 1 if used_indices else 1
    for vid in valid_maxima_ids:
        if np.sum(unmerged_regions==vid) > 0: # give priority only if the label exists in the box, even if the maxima is closely outside the box for example. you'll see this issue only in finer grid.
            if vid not in remap_dict:
                remap_dict[vid] = next_index
                used_indices.add(next_index)
                next_index += 1
    # print('stage 2 valid_maxima', remap_dict)

    # note: assign indices to remaining unique regions in unmerged_regions
    next_index = max(used_indices) + 1 if used_indices else 1
    for r in unique_unmerged_regions:
        if r not in remap_dict:
            remap_dict[r] = next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 3 unique_regions', remap_dict)

    # note: assign indices to remaining unique regions in unmerged_regions
    unique_unmerged_padded_regions = np.delete(np.unique(unmerged_padded_regions), 0)
    # new_index = len(unique_unmerged_regions) + 1
    next_index = max(used_indices) + 1 if used_indices else 1

    # we need to reindex the padded regions after computing connections
    padding_indices_start_here = next_index
    for r in unique_unmerged_padded_regions:
        if r not in list(remap_dict.keys()):
            remap_dict[r] = next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 4 unique_unmerged_padded_regions', remap_dict)

    # print('remap_dict constructed')
    maxima_info_copy = copy.deepcopy(maxima_info)

    # great! now we have a complet map. time to remap the maxima_info, unmerged_regions, and unmerged_padded_regions
    regions = copy.deepcopy(unmerged_regions)
    regions_padded = copy.deepcopy(unmerged_padded_regions)
    for k, v in remap_dict.items():
        # print(f"replace label {k} with {v}")
        maxima_info_copy[k - 1][0] = v
        regions[unmerged_regions == k] = v
        regions_padded[unmerged_padded_regions == k] = v

    # sort maxima based on first column
    maxima_info_copy = maxima_info_copy[maxima_info_copy[:, 0].argsort()]
    # maxima_info[:10, :], maxima_info_copy[:10, :]
    logger.info(
        "time(s): %.2f.  done with reindexing-1/2\n"
        % np.round(time() - start_time, decimals=2)
    )
    # print('remap_dict is ', remap_dict)

    pbc_groups = [v for k, v in get_pbc_groups(regions, maxima_info_copy).items()]
    len(pbc_groups[0])
    # find touching pairs in the regions padded
    # pairs = find_touching_region_pairs(regions_padded)
    # print(pairs, pairs.shape)
    wall_region_thicknesses = compute_wall_cells(
        mat_atoms, spacing, wall_thickness=wall_thickness
    )
    # let's start by keeping only those maxima that are there in the regions array
    maxima_indices = maxima_info_copy[:, 1:4] + maxima_info_copy[:, 4:7] * np.array(
        dgrid.shape
    )
    # get location classifications for all maxima
    inside_close_outside = is_inside_or_close_to_boundary(
        maxima_indices, dgrid.shape, wall_region_thicknesses
    )
    # create a boolean mask for maxima that are either inside or close_outside
    valid_maxima_mask = np.array(
        [loc in ["inside", "close_outside"] for loc in inside_close_outside]
    )
    # lets find the core maxima ids (region labels), dont' forget to add 1 to get label from index
    valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1
    # print("computed maxima close to the cell")
    # pairs_filtered = [p for p in pairs if np.any(np.isin(p, valid_maxima_ids))]

    # --- [copy everything up to and including maxima_info_copy = maxima_info_copy[maxima_info_copy[:, 0].argsort()] ] ---
    # (Paste the code from the original function up to the first reindexing, just before pbc_groups = ...)
    # --- End of copy ---

    # Only compute connections and further reindexing if requested
    if not compute_connections:
        all_maxima = maxima_info_copy[:, 1:4] + maxima_info_copy[:, 4:7] * np.array(
            dgrid.shape
        )
        logger.info(
            "time(s): %.2f compute_connections is set to False. No connections computed. Calculations Ending! :) \n"
            % np.round(time() - start_time, decimals=2)
        )
        logger.info(
            "TIME(s): %.2f Checking indexing consistency,"
            % np.round(time() - start_time, decimals=2)
        )
        logger.info(
            "regions w pbc: %s"
            % (
                np.unique(regions_pbc, return_counts=False).shape[0] - 1
                == regions_pbc.max()
            )
        )
        logger.info(
            "regions w/o pbc: %s"
            % (
                np.unique(regions, return_counts=False).shape[0] - 1 == regions.max()
            )
        )
        logger.info(
            "regions w/ padding: %s"
            % (
                np.unique(regions_padded, return_counts=False).shape[0] - 1
                == regions_padded.max()
            )
        )
        missing_regions = set(range(0, regions.max())) - set(np.unique(regions))
        missing_regions_pbc = set(range(0, regions_pbc.max())) - set(
            np.unique(regions_pbc)
        )

        if len(missing_regions) > 0:
            logger.warning(f"Potential inconsistency in indexing!:")
            logger.info(f"mising seq indices in regions: {missing_regions}")
            logger.info(f"mising seq indices in regions_pbc: {missing_regions_pbc}")
 
        outputs = dict(
            # dgrid=dgrid,
            all_maxima=all_maxima,
            maxima_pbc=merged_maxima,
            regions_pbc=regions_pbc,
            maxima_info=maxima_info_copy,
            regions=regions,
            pbc_groups=pbc_groups,
            close_or_inside_maxima=valid_maxima_indices,
            close_or_inside_regions=valid_maxima_ids,
            unique_regions=np.delete(np.unique(regions), 0),
            maxima_pbc_dict=merged_maxima_indices,
            maxima_relative_locations=np.array(inside_close_outside),
        )
        outputs["input_params"] = {
            "mask_thickness": mask_thickness,
            "h": h,
            "min_distance": min_distance,
            "threshold_abs": threshold_abs,
            "spacing": spacing,
            "minimum_window_separation": minimum_window_separation,
            "compactness": compactness,
            "wall_thickness": wall_thickness,
            "extra_padding": extra_padding,
            "padding_thickness": padding_thickness,
            "min_region_size": min_region_size,
        }
        if return_padded_regions:
            outputs["regions_padded"] = regions_padded
    else:
        logger.info(
            "time(s): %.2f compute_connections is True--this might take a while! \n"
            % np.round(time() - start_time, decimals=2)
        )

        # def compute_boundary(reg):
        #     return reg, sparse.COO(
        #         find_boundaries(regions_padded == reg, mode="outer").astype(int)
        #     )

        # # find uniques and remove background use the reindexed regions here
        #         regions_to_check = np.delete(np.unique(regions_padded), 0)
        #         with ThreadPoolExecutor() as executor:
        #             results = list(executor.map(compute_boundary, regions_to_check))
        #         outer = dict(results)
        # # don't screw this line up

        # A_unit = complete_cell(mat_atoms.cell).T
        A_unit = cellpar_to_cell(mat_atoms.cell.cellpar()).T
        minimum_window_separation = minimum_window_separation
        # padding_thickness = mgr.calculate_padding_thickness(dgrid, mat_atoms, spacing)
        padded_grid = np.pad(dgrid, padding_thickness, mode="wrap")
        # dinterp = interpolate_labels(dgrid)

        logger.info(
            "TIME(s): %.2f Adjusting for periodic connections\n"
            % np.round(time() - start_time, decimals=2)
        )

        connections_list = mgrcli_rust.window_connections(
            regions_padded.astype(np.uint32),
            padded_grid.astype(np.float32),
            padding_thickness,
            A_unit,
            float(minimum_window_separation),
        )
        connections_list = [
            np.array([c[0], c[1], c[2], c[3], c[4]], dtype="object")
            for c in connections_list
        ]
        # Periodic = edge touches a region outside the unit-cell label set (padding ghosts),
        # not merely a far-outside maximum on an in-cell region.
        unit_cell_region_labels = np.delete(np.unique(regions), 0)
        for c in connections_list:
            if np.any(~np.isin(c[0], unit_cell_region_labels)):
                c[2] = "periodic"
        logger.info(
            "TIME(s): %.2f Done with computing connections\n"
            % np.round(time() - start_time, decimals=2)
        )
        logger.info(
            "TIME(s): %.2f Reindexing part 2/2-- Connected nodes getting priority\n"
            % np.round(time() - start_time, decimals=2)
        )

        # NOTE: if padded regions are in the connections list, we need to prioritize those in the indexing over other regions in regions padded
        # make a remap dictionary
        remap_dict2 = {}
        unique_regions_padded = np.delete(np.unique(regions_padded), 0)
        # here we removed 0 for clarity
        unique_regions = np.delete(np.unique(regions), 0)
        # new_index = len(unique_regions) + 1  # can also do regions.max() + 1
        next_index = padding_indices_start_here
        unique_connected_indices = np.unique(
            np.vstack([conn[0] for conn in connections_list])
        )
        for c in unique_connected_indices:
            # if c not in unique_regions:  # this is node that needs priority
            if c >= padding_indices_start_here:  # this is node that needs priority
                remap_dict2[c] = next_index
                next_index += 1
        for r in unique_regions_padded:
            if np.logical_and(r not in remap_dict2, r >= padding_indices_start_here):
                remap_dict2[r] = next_index
                next_index += 1
        # print('remap dict 2 is', remap_dict2)
        # set_trace()
        regions_padded_copy = copy.deepcopy(regions_padded)
        for k, v in remap_dict2.items():
            # print(f"replace label {k} with {v}")
            regions_padded_copy[regions_padded == k] = v
        connections_list_copy = copy.deepcopy(connections_list)
        for i, c in enumerate(connections_list):
            for j, ci in enumerate(c[0]):
                if ci in remap_dict2:
                    # reindx the connections list
                    connections_list_copy[i][0][j] = remap_dict2[ci]
        maxima_info_copy_copy = copy.deepcopy(maxima_info_copy)
        for k, v in remap_dict2.items():
            # print(f"replace label {k} with {v}")
            maxima_info_copy_copy[k - 1][0] = v
        maxima_info_copy_copy = maxima_info_copy_copy[
            maxima_info_copy_copy[:, 0].argsort()
        ]
        logger.info(
            "TIME(s): %.2f Checking indexing consistency,"
            % np.round(time() - start_time, decimals=2)
        )
        logger.info(
            "regions w pbc: %s"
            % (
                np.unique(regions_pbc, return_counts=False).shape[0] - 1
                == regions_pbc.max()
            )
        )
        logger.info(
            "regions w/o pbc: %s"
            % (
                np.unique(regions, return_counts=False).shape[0] - 1 == regions.max()
            )
        )
        logger.info(
            "regions w/ padding: %s"
            % (
                np.unique(regions_padded, return_counts=False).shape[0] - 1
                == regions_padded.max()
            )
        )
        missing_regions = set(range(0, regions.max())) - set(np.unique(regions))
        missing_regions_pbc = set(range(0, regions_pbc.max())) - set(
            np.unique(regions_pbc)
        )

        if len(missing_regions) > 0:
            logger.warning(f"Potential inconsistency in indexing!:")
            logger.info(f"mising seq indices in regions: {missing_regions}")
            logger.info(f"mising seq indices in regions_pbc: {missing_regions_pbc}")
        # outputs=dict(maxima_pbc=merged_maxima, regions_pbc=regions_pbc, maxima_info=maxima_info_copy_copy, regions=regions,conections=connections_list_copy, pbc_groups=pbc_groups, close_or_inside_maxima=valid_maxima_indices, close_or_inside_regions= valid_maxima_ids, unique_regions = unique_regions)
        all_maxima = maxima_info_copy_copy[:, 1:4] + maxima_info_copy_copy[
            :, 4:7
        ] * np.array(dgrid.shape)
        outputs = dict(
            dgrid=dgrid,
            mat_atoms=mat_atoms,
            maxima_pbc=merged_maxima,
            regions_pbc=regions_pbc,
            maxima_info=maxima_info_copy_copy,
            regions=regions,
            connections=connections_list_copy,
            pbc_groups=pbc_groups,
            close_or_inside_maxima=valid_maxima_indices,
            close_or_inside_regions=valid_maxima_ids,
            unique_regions=unique_regions,
            maxima_pbc_dict=merged_maxima_indices,
            all_maxima=all_maxima,
            maxima_relative_locations=np.array(inside_close_outside),
        )
        # we need to save the input params to the results, for future reference.
        outputs["input_params"] = {
            "watershed_backend": watershed_backend,
            "mask_thickness": mask_thickness,
            "h": h,
            "min_distance": min_distance,
            "threshold_abs": threshold_abs,
            "spacing": spacing,
            "minimum_window_separation": minimum_window_separation,
            "compactness": compactness,
            "wall_thickness": wall_thickness,
            "extra_padding": extra_padding,
            "padding_thickness": padding_thickness,
            "min_region_size": min_region_size,
        }
        if return_padded_regions:
            outputs["regions_padded"] = regions_padded_copy

    if return_dgrid:
        outputs["dgrid"] = dgrid
    logger.info(
        "TIME(s): %.2f Padded segmentation workflow complete :) \n"
        % np.round(time() - start_time, decimals=2)
    )
    # Run a consistency check between maxima_info-based pocket radii and
    # true per-region maxima on the central box. This helps distinguish
    # segmentation issues from missing/under-detected maxima.
    try:
        _check_pocket_maxima_consistency(dgrid, outputs["regions"], outputs["maxima_info"])
    except Exception as exc:
        logger.warning(f"[pocket_maxima_check] consistency check failed with error: {exc}")

    return outputs


def _check_pocket_maxima_consistency(dgrid, regions, maxima_info, tol=0.25):
    """
    Diagnostic check: compare pocket radii from maxima_info with the true
    per-region maxima in dgrid, and log a warning when the gap is large.

    Parameters
    ----------
    dgrid : ndarray
        Distance grid used for segmentation.
    regions : ndarray (int)
        Region labels on the central (unpadded) box.
    maxima_info : ndarray
        Array of shape (n_maxima, 7) with columns
        [label, i, j, k, ox, oy, oz].
    tol : float
        Tolerance (Å). If true_max - pocket_from_max > tol, a warning is logged.
    """
    import numpy as np

    maxima_info = np.asarray(maxima_info)
    regions = np.asarray(regions)

    labels = np.unique(regions)
    labels = labels[labels != 0]

    n_flagged = 0
    for lab in labels:
        rows = maxima_info[maxima_info[:, 0] == lab]
        if rows.size == 0:
            logger.warning(
                f"[pocket_maxima_check] No maxima_info row found for region {lab}"
            )
            continue

        row = rows[0]
        voxel = tuple(row[1:4].astype(int))
        pocket_from_max = float(dgrid[voxel])

        mask = regions == lab
        if not np.any(mask):
            logger.warning(
                f"[pocket_maxima_check] Region {lab} has no voxels in regions array"
            )
            continue

        true_max = float(dgrid[mask].max())
        gap = true_max - pocket_from_max
        if gap > tol:
            n_flagged += 1
            logger.warning(
                "[pocket_maxima_check] Region %d: pocket_radius from maxima_info "
                "(%.3f Å) underestimates true region max (%.3f Å) by %.3f Å; "
                "this suggests a missing or under-detected maximum.",
                lab,
                pocket_from_max,
                true_max,
                gap,
            )
    logger.info(
        "[pocket_maxima_check] Completed maxima vs region-max consistency check: "
        "checked %d regions, %d flagged with gap > %.2f Å",
        int(labels.size),
        int(n_flagged),
        float(tol),
    )


def compute_pore_sizes_uniform_rust(dgrid, ase_atoms, num_samples_total=10000):
    """Compute pore sizes using uniform sampling across the material.

    This function samples points uniformly throughout the material and calculates
    the pore size (distance to nearest atom surface) at each sample point.

    :param dgrid: 3D distance grid from dgrid_from_atoms_rust
    :type dgrid: numpy.ndarray
    :param ase_atoms: ASE Atoms object of the material
    :type ase_atoms: ase.Atoms
    :param num_samples_total: Total number of sample points to generate
    :type num_samples_total: int
    :return: Tuple of (sample_points, pore_sizes) where sample_points are the sampled coordinates and pore_sizes are the corresponding pore diameters
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    import mgrcli_rust
    from ase.geometry import cellpar_to_cell
    import numpy as np

    cell = cellpar_to_cell(ase_atoms.cell.cellpar()).T
    sample_points, pore_sizes = mgrcli_rust.compute_pore_sizes_uniform(
        dgrid, cell, num_samples_total=num_samples_total
    )
    return sample_points, pore_sizes


def compute_psd_uniform_rust(
    dgrid,
    ase_atoms,
    num_samples_total=10000,
    bins=None,
    density=False,
    return_fig=False,
    material="Material",
    probe_center_min_radius=None,
):
    """Compute pore size distribution using uniform sampling.

    This function calculates the pore size distribution by sampling points uniformly
    throughout the material and optionally creates a plotly figure for visualization.

    :param dgrid: 3D distance grid from dgrid_from_atoms_rust
    :type dgrid: numpy.ndarray
    :param ase_atoms: ASE Atoms object of the material
    :type ase_atoms: ase.Atoms
    :param num_samples_total: Total number of sample points to generate
    :type num_samples_total: int
    :param bins: Bin edges for histogram. If None, uses np.linspace(0, 50, 200)
    :type bins: numpy.ndarray or None
    :param density: Whether to normalize the histogram
    :type density: bool
    :param return_fig: Whether to return a plotly figure along with the histogram
    :type return_fig: bool
    :param material: Name of the material for plot title
    :type material: str
    :param probe_center_min_radius: If > 0, only void voxels with dgrid >= this value are sampled.
    :type probe_center_min_radius: float or None
    :return: If return_fig is False, returns histogram values. If True, returns (histogram, figure)
    :rtype: numpy.ndarray or tuple[numpy.ndarray, plotly.graph_objects.Figure]
    """
    # Create region-wise PSD plot
    import numpy as np
    import plotly.graph_objects as go
    from .visualize import get_pore_size_distribution_trace
    from .compute import interpolate_dgrid

    dgrid_eff = _effective_dgrid_uniform_psd_sampling(dgrid, probe_center_min_radius)
    sample_points, pore_sizes = compute_pore_sizes_uniform_rust(
        dgrid_eff, ase_atoms, num_samples_total=num_samples_total
    )
    if bins is None:
        bins = np.linspace(0, 50, 200)  # histogram bins
    print("\nCreating region-wise PSD plots...")

    psd_hist, _ = np.histogram(pore_sizes * 2, bins=bins, density=density)
    if return_fig:
        histogram_trace = get_pore_size_distribution_trace(
            pore_sizes * 2, normalize=False, cumulative=False, bins=bins
        )
        dgrid_interpolator = interpolate_dgrid(dgrid)
        print("UNIFORM SAMPLING STATISTICS:")
        print(f"  Number of samples: {len(pore_sizes)}")
        print(
            f"  Mean pore size: {np.mean(pore_sizes * 2):.2f} ± {np.std(pore_sizes * 2):.2f} Å"
        )
        # print(f"  Max probability density: {np.max(hist):.4f}")
        fig = go.Figure()
        fig.add_trace(histogram_trace)
        mat_title = material
        if probe_center_min_radius is not None and float(probe_center_min_radius) > 0:
            mat_title = (
                f"{material} (probe-center sites ≥ {float(probe_center_min_radius):.2g} Å)"
            )
        fig.update_layout(
            title=f"Pore Size Distribution with uniform sampling across box {mat_title}",
            xaxis_title="Pore Diameter (Å)",
            yaxis_title="Counts",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=1000,
            height=600,
        )
        return psd_hist, fig
    else:
        return psd_hist



def _effective_dgrid_uniform_psd_sampling(dgrid, probe_center_min_radius):
    """Rust uniform PSD collects ``dgrid > 0``; mark sub-threshold void as −1 so it is omitted."""
    import numpy as np

    d = np.asarray(dgrid, dtype=np.float32)
    if probe_center_min_radius is None or float(probe_center_min_radius) <= 0:
        return d
    s = np.float32(float(probe_center_min_radius))
    return np.where(d >= s, d, np.float32(-1.0))


def _effective_regions_pocket_psd_sampling(regions_pbc, dgrid, probe_center_min_radius):
    """Zero region labels wherever clearance would not host the probe center."""
    import numpy as np

    rg = np.asarray(regions_pbc, dtype=np.uint32)
    if probe_center_min_radius is None or float(probe_center_min_radius) <= 0:
        return rg
    d = np.asarray(dgrid, dtype=np.float32)
    s = np.float32(float(probe_center_min_radius))
    valid = (rg > np.uint32(0)) & (d >= s)
    return np.where(valid, rg, np.uint32(0))


def compute_pore_sizes_pocketwise_rust(
    dgrid,
    regions_pbc,
    ase_atoms,
    samples_per_region=1000,
    probe_center_min_radius=None,
):
    """Compute pore sizes for each pocket/region separately.

    This function samples points within each identified pore region and calculates
    the pore size distribution for each region individually.

    :param dgrid: 3D distance grid from dgrid_from_atoms_rust
    :type dgrid: numpy.ndarray
    :param regions_pbc: 3D array of region labels from periodic segmentation
    :type regions_pbc: numpy.ndarray
    :param ase_atoms: ASE Atoms object of the material
    :type ase_atoms: ase.Atoms
    :param samples_per_region: Number of sample points per region label
    :type samples_per_region: int
    :return: Tuple of (sample_points, pore_sizes, region_labels)
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    import mgrcli_rust
    from ase.geometry import cellpar_to_cell
    import numpy as np

    cell = cellpar_to_cell(ase_atoms.cell.cellpar()).T
    dgrid = np.ascontiguousarray(np.asarray(dgrid, dtype=np.float32))
    regions_eff = _effective_regions_pocket_psd_sampling(
        regions_pbc, dgrid, probe_center_min_radius
    )
    sample_points, pore_sizes, region_labels = (
        mgrcli_rust.compute_pore_sizes_for_psd_pocketwise(
            dgrid,
            regions_eff,
            cell,
            samples_per_region=samples_per_region,
        )
    )
    return sample_points, pore_sizes, region_labels


def compute_psds_pocketwise_rust(
    dgrid,
    regions_pbc,
    ase_atoms,
    num_samples_per_region=1000,
    bins=None,
    return_fig=False,
    material="Material",
    probe_center_min_radius=None,
):
    """Compute pore size distributions for each pocket/region.

    This function creates separate pore size distributions for each identified
    pore region and optionally generates a visualization plot.

    :param dgrid: 3D distance grid from dgrid_from_atoms_rust
    :type dgrid: numpy.ndarray
    :param regions_pbc: 3D array of region labels from periodic segmentation
    :type regions_pbc: numpy.ndarray
    :param ase_atoms: ASE Atoms object of the material
    :type ase_atoms: ase.Atoms
    :param num_samples_per_region: Number of sample points per region
    :type num_samples_per_region: int
    :param bins: Bin edges for histogram. If None, uses np.linspace(0, 50, 200)
    :type bins: numpy.ndarray or None
    :param return_fig: Whether to return a plotly figure along with the PSDs
    :type return_fig: bool
    :param material: Name of the material for plot title
    :type material: str
    :return: If return_fig is False, returns pocket PSDs dictionary. If True, returns (pocket_psds, figure)
    :rtype: dict or tuple[dict, plotly.graph_objects.Figure]
    """
    # Create region-wise PSD plot
    import numpy as np
    from .helper import compute_psds_from_points_pocketwise
    from .visualize import plot_pocket_psds

    sample_points, pore_sizes, region_labels = compute_pore_sizes_pocketwise_rust(
        dgrid,
        regions_pbc,
        ase_atoms,
        samples_per_region=num_samples_per_region,
        probe_center_min_radius=probe_center_min_radius,
    )
    if bins is None:
        bins = np.linspace(0, 50, 200)  # histogram bins
    print("\nCreating region-wise PSD plots...")
    pocket_psds = compute_psds_from_points_pocketwise(
        region_labels, pore_sizes * 2, bins=bins
    )
    if return_fig:
        mat_title = material
        if probe_center_min_radius is not None and float(probe_center_min_radius) > 0:
            mat_title = (
                f"{material} (probe-center sites ≥ {float(probe_center_min_radius):.2g} Å)"
            )
        fig = plot_pocket_psds(
            pore_sizes * 2, region_labels, pocket_psds, bins, material_name=mat_title
        )
        return pocket_psds, fig
    else:
        return pocket_psds


def compute_pocket_psds_for_segmentation(
    segmentation,
    atoms,
    *,
    num_samples_per_region=1000,
    bins=None,
    probe_center_min_radius=None,
    probe_radius_psd=None,
):
    """Compute Rust pocket-wise PSDs from a periodic segmentation dict.

    Parameters
    ----------
    segmentation
        Dict with ``dgrid``, ``regions_pbc`` (or ``regions``), from
        :func:`do_periodic_segmentation_rust`.
    atoms
        ASE :class:`~ase.Atoms`.
    probe_center_min_radius, probe_radius_psd
        Minimum clearance (Å) for probe-centre sample sites.
        ``probe_radius_psd`` is an alias (pocket-typing default 1.2 Å).

    Returns
    -------
    dict
        Region id → ``{"hist", "bin_centers"}`` (same as
        :func:`compute_psds_pocketwise_rust` without ``return_fig``).
    """
    import numpy as np

    if probe_center_min_radius is None:
        probe_center_min_radius = probe_radius_psd
    dgrid = np.asarray(segmentation["dgrid"], dtype=np.float32)
    regions_pbc = np.asarray(
        segmentation.get("regions_pbc", segmentation["regions"])
    )
    if bins is None:
        from .pocket_typing import psd_bins

        bins = psd_bins(dgrid)
    return compute_psds_pocketwise_rust(
        dgrid,
        regions_pbc,
        atoms,
        num_samples_per_region=num_samples_per_region,
        bins=bins,
        return_fig=False,
        probe_center_min_radius=probe_center_min_radius,
    )


def compute_accessible_psd_from_results(
    results,
    mat_atoms,
    probe_radius,
    bin_edges,
    samples_per_region=10000,
    accessible_only=True,
    pb_normalization=False,
):
    """
    Compute a PoreBlazer-style accessible PSD by:
    1) identifying percolating pockets at the given probe_radius via winding-number percolation,
    2) sampling pore sizes pocketwise (fixed samples per region),
    3) weighting each pocket's PDF by its voxel fraction to recover a volume-weighted global PSD.

    Parameters
    ----------
    results : dict
        Segmentation results from do_periodic_segmentation_rust (needs regions_pbc, maxima_info, regions, connections, etc.).
        If results does not contain 'dgrid', you must have passed return_dgrid=True when generating it, or supply dgrid separately.
    mat_atoms : ase.Atoms
        ASE Atoms with the simulation cell (used only to build the cell matrix for sampling).
    probe_radius : float
        Probe radius (Å) used as the percolation threshold.
    bin_edges : array-like
        Bin edges for diameters (Å). Use PB-style bins for direct comparison.
    samples_per_region : int
        Fixed number of samples per pocket for the pocketwise sampler.
    accessible_only : bool
        If True, restrict to percolating (accessible) regions; if False, include all regions.

    Returns
    -------
    dict with keys:
        - pdf: global volume-weighted PDF over bin centers
        - cum: cumulative (descending) of pdf
        - psd: finite-difference derivative of cum
        - bin_centers: centers from bin_edges
        - per_region: dict of per-pocket PDFs (unweighted), voxel counts, and weights
        - accessible_region_ids: list of region ids used
    """
    import numpy as np
    import mgrcli_rust
    from ase.geometry import cellpar_to_cell
    from .winding_percolation import percolation_all_axes_winding

    regions_pbc = np.asarray(results["regions_pbc"])
    dgrid = np.asarray(results.get("dgrid", None))
    if dgrid is None:
        raise ValueError("results does not include 'dgrid'; rerun segmentation with return_dgrid=True or supply dgrid.")

    # Determine accessible region ids
    if accessible_only:
        _, _, cluster_metadata = percolation_all_axes_winding(
            results,
            threshold=probe_radius,
            node_radius_attr="pocket_radius",
            edge_radius_attr="window_radius",
        )
        accessible_region_ids = set()
        for meta in cluster_metadata.values():
            region_list = meta.get("regions_full") or meta.get("regions") or []
            accessible_region_ids.update(region_list)
        accessible_region_ids = {int(r) for r in accessible_region_ids if int(r) != 0}
    else:
        accessible_region_ids = set(int(x) for x in np.unique(regions_pbc) if int(x) != 0)

    if not accessible_region_ids:
        return {
            "pdf": np.zeros(len(bin_edges) - 1),
            "cum": np.zeros(len(bin_edges) - 1),
            "psd": np.zeros(len(bin_edges) - 1),
            "bin_centers": 0.5 * (bin_edges[1:] + bin_edges[:-1]),
            "per_region": {},
            "accessible_region_ids": [],
        }

    # Mask regions to accessible only
    mask = np.isin(regions_pbc, list(accessible_region_ids))
    regions_accessible = np.where(mask, regions_pbc, 0)

    # Per-region voxel counts (all voxels have same volume)
    region_ids_present = np.unique(regions_accessible)
    region_ids_present = [int(r) for r in region_ids_present if int(r) != 0]
    voxel_counts = {rid: int((regions_accessible == rid).sum()) for rid in region_ids_present}
    total_voxels = sum(voxel_counts.values())
    if total_voxels == 0:
        return {
            "pdf": np.zeros(len(bin_edges) - 1),
            "cum": np.zeros(len(bin_edges) - 1),
            "psd": np.zeros(len(bin_edges) - 1),
            "bin_centers": 0.5 * (bin_edges[1:] + bin_edges[:-1]),
            "per_region": {},
            "accessible_region_ids": [],
        }

    # Sample pocketwise on the accessible mask
    cell = cellpar_to_cell(mat_atoms.cell.cellpar()).T
    sample_points, pore_sizes, region_labels = mgrcli_rust.compute_pore_sizes_for_psd_pocketwise(
        dgrid,
        regions_accessible,
        cell,
        samples_per_region=int(samples_per_region),
    )

    bin_edges = np.asarray(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    per_region = {}
    pdf_global = np.zeros(len(bin_edges) - 1, dtype=float)

    # Accumulate weighted PDFs
    for rid in region_ids_present:
        mask_r = region_labels == rid
        diam_r = pore_sizes[mask_r] * 2.0
        counts, _ = np.histogram(diam_r, bins=bin_edges)
        total_counts = counts.sum()
        if total_counts > 0:
            pdf_r = counts / total_counts
        else:
            pdf_r = np.zeros_like(pdf_global)
        weight = voxel_counts[rid] / total_voxels
        pdf_global += weight * pdf_r
        per_region[rid] = {
            "pdf": pdf_r,
            "voxel_count": voxel_counts[rid],
            "weight": weight,
        }

    # Cumulative (descending) and PSD derivative
    cum = np.cumsum(pdf_global[::-1])[::-1]
    psd = np.zeros_like(cum)
    if cum.size > 2:
        bin_widths = np.diff(bin_edges)
        mid_width = bin_widths[0] if np.allclose(bin_widths, bin_widths[0]) else None
        if mid_width is not None:
            psd[1:-1] = -(cum[2:] - cum[:-2]) / (2.0 * mid_width)
        else:
            # variable bin widths: use centered differences with local widths
            psd[1:-1] = -(
                (cum[2:] - cum[1:-1]) / bin_widths[1:]
                - (cum[1:-1] - cum[:-2]) / bin_widths[:-1]
            ) / 2.0

    pdf_pb = None
    cum_pb = None
    psd_pb = None
    if pb_normalization:
        # PB-style: normalize cumulative to 1 at the largest bin and take -d(cum)/dD
        cum_pb = np.cumsum(pdf_global[::-1])[::-1]
        if cum_pb.size > 0 and cum_pb[0] > 0:
            cum_pb = cum_pb / cum_pb[0]
        psd_pb = np.zeros_like(cum_pb)
        if cum_pb.size > 2:
            bin_widths = np.diff(bin_edges)
            mid_width = bin_widths[0] if np.allclose(bin_widths, bin_widths[0]) else None
            if mid_width is not None:
                psd_pb[1:-1] = -(cum_pb[2:] - cum_pb[:-2]) / (2.0 * mid_width)
            else:
                psd_pb[1:-1] = -(
                    (cum_pb[2:] - cum_pb[1:-1]) / bin_widths[1:]
                    - (cum_pb[1:-1] - cum_pb[:-2]) / bin_widths[:-1]
                ) / 2.0
        pdf_pb = psd_pb

    return {
        "pdf": pdf_global,
        "cum": cum,
        "psd": psd,
        "bin_centers": bin_centers,
        "per_region": per_region,
        "accessible_region_ids": sorted(region_ids_present),
        "pdf_pb": pdf_pb,
        "cum_pb": cum_pb,
        "psd_pb": psd_pb,
    }


def compute_psd_global_accessible(
    results,
    mat_atoms,
    probe_radius,
    bins=None,
    samples_per_region=10000,
    density=False,
    return_fig=False,
    material="Material",
):
    """Compute global accessible PSD (volume-weighted from percolating regions only).

    Thin wrapper around compute_accessible_psd_from_results that returns the same
    format as compute_psd_uniform_rust for app compatibility: psd_hist or (psd_hist, fig).

    Only void space that percolates at the given probe_radius is included.

    :param results: Segmentation results from do_periodic_segmentation_rust with return_dgrid=True
    :type results: dict
    :param mat_atoms: ASE Atoms with the simulation cell
    :type mat_atoms: ase.Atoms
    :param probe_radius: Probe radius (Å) for percolation threshold
    :type probe_radius: float
    :param bins: Bin edges for histogram. If None, uses np.linspace(0, 50, 200)
    :type bins: numpy.ndarray or None
    :param samples_per_region: Samples per pocket for pocketwise sampler
    :type samples_per_region: int
    :param density: Whether to normalize the histogram (matches compute_psd_uniform_rust)
    :type density: bool
    :param return_fig: Whether to return a plotly figure
    :type return_fig: bool
    :param material: Name of the material for plot title
    :type material: str
    :return: psd_hist or (psd_hist, fig) - same shape as compute_psd_uniform_rust
    :rtype: numpy.ndarray or tuple[numpy.ndarray, plotly.graph_objects.Figure]
    """
    import numpy as np
    import plotly.graph_objects as go

    if bins is None:
        bins = np.linspace(0, 50, 200)
    bin_edges = np.asarray(bins)

    out = compute_accessible_psd_from_results(
        results,
        mat_atoms,
        probe_radius,
        bin_edges=bin_edges,
        samples_per_region=samples_per_region,
        accessible_only=True,
        pb_normalization=False,
    )

    pdf = out["pdf"]
    bin_centers = out["bin_centers"]
    accessible_region_ids = out["accessible_region_ids"]

    # Convert pdf to histogram for consistency with compute_psd_uniform_rust
    # pdf is normalized (sum=1); scale to counts: total_samples = samples_per_region * num_regions
    n_regions = len(accessible_region_ids)
    total_samples = samples_per_region * n_regions if n_regions > 0 else 1
    if density:
        bin_widths = np.diff(bin_edges)
        # density = pdf / bin_width (so integral = 1)
        psd_hist = np.where(bin_widths > 0, pdf / bin_widths, 0.0)
    else:
        psd_hist = pdf * total_samples

    if return_fig:
        print("\nCreating accessible global PSD plot...")
        print(f"  Accessible regions at probe_radius={probe_radius} Å: {n_regions}")
        trace = go.Scatter(
            x=bin_centers,
            y=psd_hist,
            mode="lines",
            name="Accessible PSD",
            line=dict(width=2),
        )
        fig = go.Figure()
        fig.add_trace(trace)
        fig.update_layout(
            title=f"Accessible Pore Size Distribution (probe {probe_radius} Å) - {material}",
            xaxis_title="Pore Diameter (Å)",
            yaxis_title="Counts" if not density else "Density",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=1000,
            height=600,
        )
        return psd_hist, fig
    return psd_hist


def compute_psds_pocketwise_accessible(
    results,
    mat_atoms,
    probe_radius,
    num_samples_per_region=1000,
    bins=None,
    return_fig=False,
    material="Material",
):
    """Compute pocket-wise PSD restricted to accessible (percolating) regions only.

    Returns the same dict structure as compute_psds_pocketwise_rust for app compatibility:
    {region_id: {'hist', 'bin_centers', 'num_samples', 'mean_pore_size', 'std_pore_size'}}

    :param results: Segmentation results from do_periodic_segmentation_rust with return_dgrid=True
    :type results: dict
    :param mat_atoms: ASE Atoms with the simulation cell
    :type mat_atoms: ase.Atoms
    :param probe_radius: Probe radius (Å) for percolation threshold
    :type probe_radius: float
    :param num_samples_per_region: Samples per pocket
    :type num_samples_per_region: int
    :param bins: Bin edges for histogram. If None, uses np.linspace(0, 50, 200)
    :type bins: numpy.ndarray or None
    :param return_fig: Whether to return a plotly figure
    :type return_fig: bool
    :param material: Name of the material for plot title
    :type material: str
    :return: pocket_psds or (pocket_psds, fig) - same shape as compute_psds_pocketwise_rust
    :rtype: dict or tuple[dict, plotly.graph_objects.Figure]
    """
    import numpy as np
    from .visualize import plot_pocket_psds

    if bins is None:
        bins = np.linspace(0, 50, 200)
    bin_edges = np.asarray(bins)

    out = compute_accessible_psd_from_results(
        results,
        mat_atoms,
        probe_radius,
        bin_edges=bin_edges,
        samples_per_region=num_samples_per_region,
        accessible_only=True,
        pb_normalization=False,
    )

    per_region = out["per_region"]
    bin_centers = out["bin_centers"]

    # Convert to same format as compute_psds_pocketwise_rust
    pocket_psds = {}
    for rid, data in per_region.items():
        pdf_r = data["pdf"]
        n_samples = num_samples_per_region
        hist = pdf_r * n_samples
        mean_pore_size = np.sum(pdf_r * bin_centers)
        var = np.sum(pdf_r * (bin_centers - mean_pore_size) ** 2)
        std_pore_size = np.sqrt(var) if var > 0 else 0.0
        pocket_psds[rid] = {
            "hist": hist,
            "bin_centers": bin_centers,
            "num_samples": n_samples,
            "mean_pore_size": mean_pore_size,
            "std_pore_size": std_pore_size,
        }

    if return_fig:
        # Reconstruct pore_sizes and region_labels for plot_pocket_psds
        # (plot_pocket_psds needs them for the overall trace; we use bin_centers + hist)
        if pocket_psds:
            all_bin_centers = list(pocket_psds.values())[0]["bin_centers"]
            # Create synthetic pore_sizes by expanding hist to sample points
            pore_sizes_list = []
            region_labels_list = []
            for rid, psd_data in pocket_psds.items():
                hist = psd_data["hist"]
                bc = psd_data["bin_centers"]
                for i, count in enumerate(hist):
                    n = int(round(count))
                    if n > 0:
                        pore_sizes_list.extend([bc[i]] * n)
                        region_labels_list.extend([rid] * n)
            if pore_sizes_list:
                pore_sizes_arr = np.array(pore_sizes_list)
                region_labels_arr = np.array(region_labels_list)
                fig = plot_pocket_psds(
                    pore_sizes_arr,
                    region_labels_arr,
                    pocket_psds,
                    bins,
                    material_name=material,
                )
                fig.update_layout(
                    title=f"Accessible Pore Size Distribution (probe {probe_radius} Å) - {material}",
                )
                return pocket_psds, fig
        # Empty case: create minimal figure
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.update_layout(
            title=f"Accessible Pore Size Distribution (probe {probe_radius} Å) - {material}",
            xaxis_title="Pore Diameter (Å)",
            yaxis_title="Counts",
            annotations=[
                dict(
                    text="No percolating regions at this probe radius",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
        return pocket_psds, fig

    return pocket_psds

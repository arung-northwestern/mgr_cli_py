"""Local framework environment for pocket typing.

Provides rotation-invariant annulus fingerprints (radial + spherical-harmonic power
spectrum) around the pocket maximum, and max-probe-radius clustering primitives used
by ``assign_pocket_types_env_cluster`` in ``pocket_typing.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from ase.geometry import complete_cell
try:
    from scipy.special import sph_harm as _scipy_sph_harm
except ImportError:
    _scipy_sph_harm = None

# Defaults (Å) — legacy fixed cut; fingerprints use adaptive shell instead
R_CUT_FRAMEWORK_A = 6.0
MAX_VOID_SAMPLES = 250
VOID_SHELL_MAXIMUM_A = 3.5

# Adaptive shell: R = clip(alpha * r_pocket + delta, r_min, r_max)
SHELL_ALPHA = 1.1
SHELL_DELTA_A = 2.0
SHELL_R_MIN_A = 2.0
SHELL_R_MAX_A: Optional[float] = None  # no upper cap by default

# Fingerprint (rotation-invariant radial + SH power spectrum; see module note below)
FP_L_MAX = 6
FP_N_RADIAL_BINS = 16
FP_MERGE_COSINE = 0.85
FP_SPLIT_COSINE = 0.85
# ZIF-like small/mid pockets (dominant d1 roughly 4–8 Å ≈ hard 2–3 Å window radii): stricter merge
FP_SMALL_D1_MAX_A = 8.0
FP_SMALL_POCKET_COSINE = 0.96
FP_SMALL_POCKET_ANGULAR_ONLY = True  # ignore radial onion layers; windows = directions
# Annulus: framework atoms near the cavity mouth (Å from centroid), not the whole 8+ Å ball
FP_ANNULUS_INNER_PAD_A = 0.3
FP_ANNULUS_OUTER_PAD_A = 2.0
# Scalar gates (void volume from segmentation; max probe radius at dgrid peak)
FP_MAX_PROBE_GAP_A = 0.35
FP_VOLUME_RATIO_MAX = 1.35

# Surface-atom fingerprint: void voxels close to framework wall → query atoms from there
SURFACE_DGRID_MAX_A = 1.5    # Å — void voxels with dgrid < this are "surface voxels"
SURFACE_ATOM_QUERY_R_A = 3.5  # Å — search radius from each surface voxel to atoms
SURFACE_MAX_SAMPLES = 300     # max surface voxels sampled (keeps cost manageable)

# Selective typing (Å)
DEFAULT_TAU_MERGE_A = 0.5
DEFAULT_GRAY_EXTRA_A = 0.35
DEFAULT_D1_BIN_A = 0.25


def bond_type_from_order(order: float) -> str:
    if order >= 1.75:
        return "double"
    if order >= 1.25:
        return "aromatic"
    return "single"


def void_cart_for_pocket(
    regions_pbc: np.ndarray,
    regions: np.ndarray,
    dgrid: np.ndarray,
    rep: int,
    shape: np.ndarray,
    atoms,
    *,
    prefer_fragment: bool = True,
    max_samples: int = MAX_VOID_SAMPLES,
    shell_near_max_a: float = VOID_SHELL_MAXIMUM_A,
) -> np.ndarray:
    """Cartesian void sample points for pocket ``rep``.

    Uses ``regions == rep`` when that fragment exists and is not huge vs ``regions_pbc``;
    otherwise falls back to ``regions_pbc == rep`` (subsampled).
    """
    from mgr.helper import find_coord_from_indices

    rep = int(rep)
    mask_frag = regions == rep
    mask_pbc = regions_pbc == rep
    if prefer_fragment and np.any(mask_frag):
        n_frag, n_pbc = int(mask_frag.sum()), int(mask_pbc.sum())
        if n_pbc == 0 or n_frag <= max(n_pbc * 0.6, 5000):
            idx = np.argwhere(mask_frag)
        else:
            idx = np.argwhere(mask_frag)
            vals = dgrid[mask_frag]
            peak = idx[int(np.argmax(vals))]
            all_cart = find_coord_from_indices(idx, shape=shape, ase_atoms=atoms)
            peak_cart = find_coord_from_indices(peak.reshape(1, 3), shape=shape, ase_atoms=atoms)[0]
            d = np.linalg.norm(all_cart - peak_cart, axis=1)
            idx = idx[d <= shell_near_max_a]
    else:
        idx = np.argwhere(mask_pbc)

    if idx.size == 0:
        return np.empty((0, 3))
    if len(idx) > max_samples:
        rng = np.random.default_rng(rep)
        idx = idx[rng.choice(len(idx), max_samples, replace=False)]
    return find_coord_from_indices(idx, shape=shape, ase_atoms=atoms)


def framework_atom_indices_near_void(
    atoms,
    void_cart: np.ndarray,
    r_cut: float = R_CUT_FRAMEWORK_A,
) -> np.ndarray:
    """MIC: keep framework atoms with min distance to any void sample <= r_cut."""
    if void_cart.size == 0:
        return np.array([], dtype=int)
    cell = complete_cell(atoms.get_cell()).T
    frac_atoms = atoms.get_scaled_positions()
    frac_void = np.linalg.solve(cell.T, void_cart.T).T
    frac_void = frac_void - np.floor(frac_void)
    dr = frac_void[None, :, :] - frac_atoms[:, None, :]
    dr = dr - np.round(dr)
    dist = np.linalg.norm(np.einsum("mni,ij->mnj", dr, cell), axis=2)
    return np.where(dist.min(axis=1) <= r_cut)[0].astype(int)


def adaptive_shell_radius(
    r_pocket: float,
    *,
    alpha: float = SHELL_ALPHA,
    delta: float = SHELL_DELTA_A,
    r_min: float = SHELL_R_MIN_A,
    r_max: Optional[float] = SHELL_R_MAX_A,
) -> float:
    """Shell extent (Å) from pocket size scale (d1 or dgrid max)."""
    r = float(alpha) * float(r_pocket) + float(delta)
    r = max(r, r_min)
    if r_max is not None:
        r = min(r, float(r_max))
    return r


def pocket_centroid(void_cart: np.ndarray) -> np.ndarray:
    if void_cart.size == 0:
        return np.zeros(3)
    return np.mean(void_cart, axis=0)


def expand_bond_closure(
    atom_idx: np.ndarray,
    bonds: Sequence[Sequence[int]],
    n_hops: int = 1,
) -> np.ndarray:
    """Add bonded neighbors up to ``n_hops`` (cushion beyond distance shell)."""
    s = {int(i) for i in atom_idx}
    for _ in range(n_hops):
        added: set = set()
        for bond in bonds:
            if len(bond) < 2:
                continue
            a, b = int(bond[0]), int(bond[1])
            if a in s and b not in s:
                added.add(b)
            if b in s and a not in s:
                added.add(a)
        if not added:
            break
        s |= added
    return np.array(sorted(s), dtype=int)


def framework_indices_adaptive_shell(
    atoms,
    void_cart: np.ndarray,
    bonds: Sequence[Sequence[int]],
    r_pocket: float,
    *,
    alpha: float = SHELL_ALPHA,
    delta: float = SHELL_DELTA_A,
    r_min: float = SHELL_R_MIN_A,
    r_max: Optional[float] = SHELL_R_MAX_A,
    bond_closure_hops: int = 1,
) -> Tuple[np.ndarray, float]:
    """Framework atom indices within adaptive shell + optional bond closure."""
    r_shell = adaptive_shell_radius(
        r_pocket, alpha=alpha, delta=delta, r_min=r_min, r_max=r_max,
    )
    n_atoms = len(atoms)
    idx = framework_atom_indices_near_void(atoms, void_cart, r_shell)
    idx = idx[(idx >= 0) & (idx < n_atoms)]
    if bond_closure_hops > 0 and len(idx):
        idx = expand_bond_closure(idx, bonds, n_hops=bond_closure_hops)
        idx = idx[(idx >= 0) & (idx < n_atoms)]
    return idx, r_shell


def mic_vectors_from_centroid(
    atoms,
    centroid_cart: np.ndarray,
    atom_indices: np.ndarray,
) -> np.ndarray:
    """(N, 3) Cartesian MIC displacements centroid → atom."""
    if len(atom_indices) == 0:
        return np.empty((0, 3))
    cell = complete_cell(atoms.get_cell()).T
    frac_atoms = atoms.get_scaled_positions()[atom_indices]
    frac_c = np.linalg.solve(cell.T, np.asarray(centroid_cart, float).reshape(1, 3).T).T[0]
    frac_c = frac_c - np.floor(frac_c)
    dr = frac_atoms - frac_c
    dr = dr - np.round(dr)
    return np.einsum("ni,ij->nj", dr, cell)


def _symbol_to_z(sym: str) -> int:
    from ase.data import atomic_numbers
    return int(atomic_numbers.get(sym, 0))


def _sph_harm_value(m: int, l: int, phi_az: np.ndarray, theta_col: np.ndarray) -> np.ndarray:
    """Complex Y_l^m. ``phi_az`` azimuth, ``theta_col`` colatitude (radians)."""
    if _scipy_sph_harm is not None:
        return _scipy_sph_harm(m, l, phi_az, theta_col)
    from scipy.special import sph_harm_y

    return sph_harm_y(l, m, theta_col, phi_az)


def sh_power_spectrum_unit_dirs(
    unit_dirs: np.ndarray,
    weights: np.ndarray,
    l_max: int,
) -> np.ndarray:
    """Rotation-invariant power |a_l|^2 = sum_m |a_lm|^2 for l=0..l_max."""
    if len(unit_dirs) == 0:
        return np.zeros(l_max + 1)
    theta_col = np.arccos(np.clip(unit_dirs[:, 2], -1.0, 1.0))
    phi_az = np.arctan2(unit_dirs[:, 1], unit_dirs[:, 0])
    power = np.zeros(l_max + 1, dtype=float)
    for l in range(l_max + 1):
        acc = 0.0
        for m in range(-l, l + 1):
            y_lm = _sph_harm_value(m, l, phi_az, theta_col)
            a_lm = np.sum(weights * y_lm)
            acc += float(np.abs(a_lm) ** 2)
        power[l] = acc
    return power


def radial_histogram_normalized(
    r: np.ndarray,
    r_shell: float,
    n_bins: int,
) -> np.ndarray:
    hist = np.zeros(n_bins, dtype=float)
    if len(r) == 0 or r_shell <= 0:
        return hist
    rn = np.clip(r / r_shell, 0.0, 1.0 - 1e-9)
    bc = (rn * n_bins).astype(int)
    bc = np.clip(bc, 0, n_bins - 1)
    for b in bc:
        hist[b] += 1.0
    s = hist.sum()
    return hist / s if s > 0 else hist


@dataclass
class PocketFingerprint:
    """Rotation-invariant environment fingerprint around pocket centroid."""

    rep: int
    vector: np.ndarray
    r_pocket: float
    r_shell: float
    n_atoms: int
    element_z: Tuple[int, ...]
    global_radial: np.ndarray
    global_sh_power: np.ndarray
    per_z_radial: Dict[int, np.ndarray]
    per_z_sh_power: Dict[int, np.ndarray]
    max_probe_radius_a: float = 0.0
    void_volume_a3: float = 0.0
    # Connectivity-direction channel: SH power of unit vectors to trusted neighbours
    conn_sh_power: Optional[np.ndarray] = None
    n_connections: int = 0
    # Pocket PSD channel: normalised PSD histogram for size-based similarity
    psd_vec: Optional[np.ndarray] = None


def max_probe_radius_at_peak(
    regions: np.ndarray,
    dgrid: np.ndarray,
    rep: int,
) -> float:
    """Largest probe radius (Å) at the pocket maximum on the unit-cell fragment."""
    mask = regions == int(rep)
    if not np.any(mask):
        return 0.0
    return float(np.max(dgrid[mask]))


def max_probe_radius_for_fragment_group(
    regions: np.ndarray,
    dgrid: np.ndarray,
    fragment_labels: Sequence[int],
) -> float:
    """Largest probe radius (Å) over all periodic fragment labels in one PBC pocket."""
    labels = [int(x) for x in fragment_labels]
    mask = np.isin(regions, labels)
    if not np.any(mask):
        return 0.0
    return float(np.max(dgrid[mask]))


def void_cart_mic_for_fragment_group(
    regions: np.ndarray,
    dgrid: np.ndarray,
    fragment_labels: Sequence[int],
    shape: np.ndarray,
    atoms,
    *,
    max_samples: int = MAX_VOID_SAMPLES,
) -> np.ndarray:
    """Void points for a full PBC pocket (all fragment copies, MIC-wrapped to peak)."""
    from mgr.helper import find_coord_from_indices

    labels = [int(x) for x in fragment_labels]
    mask = np.isin(regions, labels)
    if not np.any(mask):
        return np.empty((0, 3))
    idx = np.argwhere(mask)
    vals = dgrid[mask]
    peak_i = idx[int(np.argmax(vals))]
    all_cart = find_coord_from_indices(idx, shape=shape, ase_atoms=atoms)
    peak_cart = find_coord_from_indices(peak_i.reshape(1, 3), shape=shape, ase_atoms=atoms)[0]

    cell = complete_cell(atoms.get_cell()).T
    frac_peak = np.linalg.solve(cell.T, peak_cart.reshape(1, 3).T).T[0]
    frac_all = np.linalg.solve(cell.T, all_cart.T).T
    dr = frac_all - frac_peak
    dr = dr - np.round(dr)
    wrapped_frac = frac_peak + dr
    all_cart = np.einsum("ni,ij->nj", wrapped_frac, cell)

    if len(all_cart) > max_samples:
        rng = np.random.default_rng(int(min(labels)) if labels else 0)
        sel = rng.choice(len(all_cart), max_samples, replace=False)
        all_cart = all_cart[sel]
    return all_cart


def void_cart_near_dgrid_peak(
    regions: np.ndarray,
    dgrid: np.ndarray,
    rep: int,
    shape: np.ndarray,
    atoms,
    *,
    shell_a: float = VOID_SHELL_MAXIMUM_A,
) -> np.ndarray:
    """Void samples within ``shell_a`` of the local dgrid maximum (fragment)."""
    from mgr.helper import find_coord_from_indices

    mask = regions == int(rep)
    if not np.any(mask):
        return np.empty((0, 3))
    idx = np.argwhere(mask)
    vals = dgrid[mask]
    peak = idx[int(np.argmax(vals))]
    peak_cart = find_coord_from_indices(peak.reshape(1, 3), shape=shape, ase_atoms=atoms)[0]
    all_cart = find_coord_from_indices(idx, shape=shape, ase_atoms=atoms)
    d = np.linalg.norm(all_cart - peak_cart, axis=1)
    return all_cart[d <= shell_a]


def void_cart_surface_pocket(
    regions_pbc: np.ndarray,
    dgrid: np.ndarray,
    rep: int,
    shape: np.ndarray,
    atoms,
    *,
    surface_dgrid_max: float = SURFACE_DGRID_MAX_A,
    mask_thickness: float = 0.05,
    max_samples: int = SURFACE_MAX_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Surface void voxels of pocket ``rep`` (small dgrid → close to VDW surface).

    Returns
    -------
    surface_voxels_cart : (N, 3) Cartesian, MIC-wrapped around pocket maximum
    peak_cart : (3,) Cartesian position of the dgrid maximum (physical cage centre)
    """
    from mgr.helper import find_coord_from_indices

    rep = int(rep)
    pocket_mask = regions_pbc == rep
    if not np.any(pocket_mask):
        return np.empty((0, 3)), np.zeros(3)

    # Locate the dgrid maximum (physical cage centre)
    idx_all = np.argwhere(pocket_mask)
    vals = dgrid[pocket_mask]
    peak_i = idx_all[int(np.argmax(vals))]
    peak_cart = find_coord_from_indices(
        peak_i.reshape(1, 3), shape=shape, ase_atoms=atoms
    )[0]

    # Surface voxels: in accessible void AND close to VDW surface
    surface_mask = pocket_mask & (dgrid > mask_thickness) & (dgrid < surface_dgrid_max)
    if not np.any(surface_mask):
        # Fallback: all accessible void voxels in this pocket
        surface_mask = pocket_mask & (dgrid > mask_thickness)

    surf_idx = np.argwhere(surface_mask)
    all_cart = find_coord_from_indices(surf_idx, shape=shape, ase_atoms=atoms)

    # MIC-wrap all surface voxels around the peak so they cluster near the cage centre
    cell = complete_cell(atoms.get_cell()).T
    frac_peak = np.linalg.solve(cell.T, peak_cart.reshape(1, 3).T).T[0]
    frac_surf = np.linalg.solve(cell.T, all_cart.T).T
    dr = frac_surf - frac_peak
    dr = dr - np.round(dr)
    all_cart = np.einsum("ni,ij->nj", frac_peak + dr, cell)

    if len(all_cart) > max_samples:
        rng = np.random.default_rng(rep)
        sel = rng.choice(len(all_cart), max_samples, replace=False)
        all_cart = all_cart[sel]
    return all_cart, peak_cart


def peak_cart_for_fragment_group(
    regions: np.ndarray,
    dgrid: np.ndarray,
    fragment_labels: Sequence[int],
    shape: np.ndarray,
    atoms,
) -> np.ndarray:
    """Cartesian position of the dgrid maximum for a PBC pocket group."""
    from mgr.helper import find_coord_from_indices

    labels = [int(x) for x in fragment_labels]
    mask = np.isin(regions, labels)
    if not np.any(mask):
        return np.zeros(3)
    idx = np.argwhere(mask)
    vals = dgrid[mask]
    peak_i = idx[int(np.argmax(vals))]
    return find_coord_from_indices(peak_i.reshape(1, 3), shape=shape, ase_atoms=atoms)[0]


def compute_connectivity_fingerprints(
    pocket_centres: Dict[int, np.ndarray],
    trusted_neighbors: Dict[int, List[int]],
    atoms,
    *,
    l_max: int = FP_L_MAX,
    min_connections: int = 1,
) -> Dict[int, np.ndarray]:
    """SH power spectrum of direction vectors to trusted neighbour pocket centres.

    Returns {rep: power_spectrum_array(l_max+1)}.  Pockets with fewer than
    ``min_connections`` trusted connections get a zero vector.

    The minimum image convention (MIC) is applied so that the nearest periodic
    image of each neighbour is used.  Connections that pass through a window
    smaller than ``WINDOW_TRUST_MIN_A`` (1 Å radius) are already excluded by
    the caller (``build_pockets_from_segmentation`` in pocket_typing.py).
    """
    from ase.geometry import complete_cell as _complete_cell

    cell = _complete_cell(atoms.get_cell()).T   # 3×3, columns are lattice vectors
    result: Dict[int, np.ndarray] = {}
    for rep, center in pocket_centres.items():
        nbrs = trusted_neighbors.get(rep, [])
        dirs: List[np.ndarray] = []
        for nbr in nbrs:
            nbr_center = pocket_centres.get(nbr)
            if nbr_center is None:
                continue
            dr = nbr_center - center
            # MIC: wrap to nearest image
            frac = np.linalg.solve(cell.T, dr)
            frac -= np.round(frac)
            dr_mic = cell @ frac
            dist = float(np.linalg.norm(dr_mic))
            if dist > 1e-6:
                dirs.append(dr_mic / dist)
        n_conn = len(dirs)
        if n_conn < min_connections:
            result[rep] = np.zeros(l_max + 1)
        else:
            dirs_arr = np.array(dirs, dtype=float)
            weights = np.ones(n_conn, dtype=float)
            result[rep] = sh_power_spectrum_unit_dirs(dirs_arr, weights, l_max)
    return result


def compute_pocket_fingerprint(
    regions_pbc: np.ndarray,
    regions: np.ndarray,
    dgrid: np.ndarray,
    rep: int,
    shape: np.ndarray,
    atoms,
    bonds: Sequence[Sequence[int]],
    *,
    r_pocket: float,
    l_max: int = FP_L_MAX,
    n_radial_bins: int = FP_N_RADIAL_BINS,
    alpha: float = SHELL_ALPHA,
    delta: float = SHELL_DELTA_A,
    r_min: float = SHELL_R_MIN_A,
    r_max: Optional[float] = SHELL_R_MAX_A,
    void_near_peak: bool = False,
    use_surface_atoms: bool = False,
    max_probe_radius_a: Optional[float] = None,
    void_volume_a3: float = 0.0,
    use_mouth_annulus: bool = True,
    fragment_group: Optional[Sequence[int]] = None,
    psd_vec: Optional[np.ndarray] = None,
) -> PocketFingerprint:
    """Radial (r/R_shell) + SH angular power per element → L2-normalized vector.

    When ``use_surface_atoms=True``, the annulus is built from all void voxels that
    touch the VDW surface (dgrid < ``SURFACE_DGRID_MAX_A``), not only voxels near the
    dgrid maximum.  This gives a complete view of the cage walls for non-spherical or
    large pockets, and uses the dgrid maximum as the pocket centre rather than the
    centroid of void samples.  The mouth-annulus distance filter is bypassed because
    surface-voxel selection is already wall-targeted."""
    frag_labels = (
        [int(x) for x in fragment_group]
        if fragment_group is not None
        else [int(rep)]
    )
    if max_probe_radius_a is None:
        if len(frag_labels) > 1 or (frag_labels and frag_labels[0] != int(rep)):
            max_probe_radius_a = max_probe_radius_for_fragment_group(
                regions, dgrid, frag_labels,
            )
        else:
            max_probe_radius_a = max_probe_radius_at_peak(regions, dgrid, rep)
    # Shell tracks physical cavity size (peak radius), not only PSD d1
    r_scale = max(float(r_pocket), 2.0 * float(max_probe_radius_a))
    if use_surface_atoms:
        # --- surface-atom branch: atoms lining ALL pocket walls ---
        surf_voxels, centroid = void_cart_surface_pocket(
            regions_pbc, dgrid, rep, shape, atoms,
        )
        # Targeted query radius from surface voxel to atom (~VDW + buffer)
        a_idx = framework_atom_indices_near_void(atoms, surf_voxels, SURFACE_ATOM_QUERY_R_A)
        n_atoms_total = len(atoms)
        a_idx = a_idx[(a_idx >= 0) & (a_idx < n_atoms_total)]
        if len(a_idx) and bonds:
            a_idx = expand_bond_closure(a_idx, bonds, n_hops=1)
            a_idx = a_idx[(a_idx >= 0) & (a_idx < n_atoms_total)]
        # r_shell: normalisation radius = cage size + query reach + margin
        r_shell = float(max_probe_radius_a) + SURFACE_ATOM_QUERY_R_A + 2.0
    elif void_near_peak:
        void_pts = void_cart_near_dgrid_peak(regions, dgrid, rep, shape, atoms)
        if void_pts.size == 0:
            void_pts = void_cart_mic_for_fragment_group(
                regions, dgrid, frag_labels, shape, atoms,
            )
        centroid = pocket_centroid(void_pts)
        a_idx, r_shell = framework_indices_adaptive_shell(
            atoms, void_pts, bonds, r_scale,
            alpha=alpha, delta=delta, r_min=r_min, r_max=r_max,
        )
    else:
        void_pts = void_cart_mic_for_fragment_group(
            regions, dgrid, frag_labels, shape, atoms,
        )
        centroid = pocket_centroid(void_pts)
        a_idx, r_shell = framework_indices_adaptive_shell(
            atoms, void_pts, bonds, r_scale,
            alpha=alpha, delta=delta, r_min=r_min, r_max=r_max,
        )
    sym = [atoms.get_chemical_symbols()[i] for i in a_idx]
    z_by_atom = [_symbol_to_z(s) for s in sym]
    dr = mic_vectors_from_centroid(atoms, centroid, a_idx)
    r = np.linalg.norm(dr, axis=1)
    mask = r > 1e-6
    if not use_surface_atoms and use_mouth_annulus and max_probe_radius_a > 0:
        r_lo = max(0.5, float(max_probe_radius_a) - FP_ANNULUS_INNER_PAD_A)
        r_hi = float(max_probe_radius_a) + FP_ANNULUS_OUTER_PAD_A
        mask = mask & (r >= r_lo) & (r <= r_hi)
    dr = dr[mask]
    r = r[mask]
    z_arr = np.array(z_by_atom, dtype=int)[mask]

    element_z = tuple(sorted(set(int(z) for z in z_arr)))
    per_z_radial: Dict[int, np.ndarray] = {}
    per_z_sh_power: Dict[int, np.ndarray] = {}

    if len(r):
        u = dr / r[:, None]
        global_radial = radial_histogram_normalized(r, r_shell, n_radial_bins)
        global_sh = sh_power_spectrum_unit_dirs(u, np.ones(len(r)), l_max)
    else:
        global_radial = np.zeros(n_radial_bins)
        global_sh = np.zeros(l_max + 1)

    for z in element_z:
        sel = z_arr == z
        ri, dri = r[sel], dr[sel]
        if len(ri) == 0:
            per_z_radial[z] = np.zeros(n_radial_bins)
            per_z_sh_power[z] = np.zeros(l_max + 1)
            continue
        ui = dri / ri[:, None]
        per_z_radial[z] = radial_histogram_normalized(ri, r_shell, n_radial_bins)
        per_z_sh_power[z] = sh_power_spectrum_unit_dirs(ui, np.ones(len(ri)), l_max)

    blocks: List[np.ndarray] = [global_radial, global_sh]
    for z in element_z:
        blocks.extend([per_z_radial[z], per_z_sh_power[z]])
    vec = np.concatenate(blocks).astype(float)
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec /= nrm
    return PocketFingerprint(
        rep=int(rep),
        vector=vec,
        r_pocket=float(r_pocket),
        r_shell=float(r_shell),
        n_atoms=int(len(a_idx)),
        element_z=element_z,
        global_radial=global_radial,
        global_sh_power=global_sh,
        per_z_radial=per_z_radial,
        per_z_sh_power=per_z_sh_power,
        max_probe_radius_a=float(max_probe_radius_a),
        void_volume_a3=float(void_volume_a3),
        psd_vec=np.asarray(psd_vec, dtype=float) if psd_vec is not None else None,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def fingerprint_cosine_similarity(
    a: PocketFingerprint,
    b: PocketFingerprint,
    *,
    angular_only: bool = False,
    use_psd: bool = True,
    use_connectivity: bool = True,
) -> float:
    """Mean cosine over environment + PSD + connectivity channels (rotation-invariant).

    Channels:
      * global_radial, global_sh_power  — atom environment shell
      * per-element radial, sh_power    — element-resolved atom environment
      * psd_vec                         — pocket PSD histogram (size fingerprint)
      * conn_sh_power                   — SH power of directions to trusted neighbours

    ``angular_only`` suppresses radial channels (better for small pockets).
    ``use_psd`` includes the pocket PSD channel when available (default True).
    ``use_connectivity`` includes the connectivity-direction channel when both
    pockets have ≥1 trusted connection (default True).
    """
    parts: List[float] = []
    if not angular_only:
        parts.append(_cosine(a.global_radial, b.global_radial))
    parts.append(_cosine(a.global_sh_power, b.global_sh_power))
    for z in set(a.element_z) & set(b.element_z):
        if not angular_only:
            parts.append(_cosine(a.per_z_radial[z], b.per_z_radial[z]))
        parts.append(_cosine(a.per_z_sh_power[z], b.per_z_sh_power[z]))
    # PSD channel: pocket size fingerprint
    if use_psd and a.psd_vec is not None and b.psd_vec is not None and (
        len(a.psd_vec) > 0 and len(b.psd_vec) > 0
    ):
        parts.append(_cosine(a.psd_vec, b.psd_vec))
    # Connectivity channel: SH power of directions to trusted window-connected neighbours.
    # Weighted 5x because (a) it is topologically distinct from wall chemistry and (b)
    # it is the key distinguishing signal for cage pairs that share identical linker atoms
    # (e.g. ZIF-20 7.26 vs 7.61 A cages: atom-env cosine ~0.97, conn cosine ~0.27).
    # Requiring at least 3 connections avoids spurious low cosines for nearly-isolated
    # pockets that happen to have 1-2 noisy connections.
    if (
        use_connectivity
        and a.conn_sh_power is not None
        and b.conn_sh_power is not None
        and a.n_connections >= 3
        and b.n_connections >= 3
    ):
        conn_cos = _cosine(a.conn_sh_power, b.conn_sh_power)
        for _ in range(5):
            parts.append(conn_cos)
    if not parts:
        return 0.0
    return float(np.mean(parts))



def assign_pocket_types_by_max_probe_radius(
    pockets: Sequence[Dict[str, Any]],
    fps: Dict[int, PocketFingerprint],
    *,
    band_gap_a: float = 0.10,
) -> Tuple[int, Dict[int, int], Dict[int, int]]:
    """Cluster by probe radius at dgrid peak; one type per band (symmetry copies merged).

    Returns (n_types, rep_id -> type_id, rep_id -> band_id).
    """
    ids = sorted({int(p["id"]) for p in pockets}, key=lambda i: fps[i].max_probe_radius_a)
    band_centers: List[float] = []
    pid_band: Dict[int, int] = {}
    for pid in ids:
        r = fps[pid].max_probe_radius_a
        assigned = None
        for bi, c in enumerate(band_centers):
            if abs(r - c) <= band_gap_a:
                assigned = bi
                break
        if assigned is None:
            assigned = len(band_centers)
            band_centers.append(r)
        pid_band[pid] = assigned

    bi_to_type = {bi: ti for ti, bi in enumerate(sorted(set(pid_band.values())))}
    rep_to_type = {pid: bi_to_type[pid_band[pid]] for pid in ids}
    return len(bi_to_type), rep_to_type, pid_band

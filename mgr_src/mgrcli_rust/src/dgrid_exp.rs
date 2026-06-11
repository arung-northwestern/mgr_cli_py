//! Optimized `dgrid_from_atoms`: SoA layout, SIMD (orthogonal), Wassenaar 7-image MIC.

use ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::Py;
use rayon::prelude::*;
use wide::f64x4;

const ORTHO_TOL: f64 = 1e-12;

const OFFSETS_FACE: [[f64; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

#[derive(Clone, Copy)]
struct CellMat {
    c00: f64,
    c01: f64,
    c02: f64,
    c10: f64,
    c11: f64,
    c12: f64,
    c20: f64,
    c21: f64,
    c22: f64,
}

struct AtomsSoA {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    r: Vec<f64>,
}

impl AtomsSoA {
    fn from_views(
        fpoints: &ndarray::ArrayView2<f64>,
        radii: &ndarray::ArrayView1<f64>,
    ) -> Self {
        let n = fpoints.len_of(ndarray::Axis(0));
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for (idx, atom_frac) in fpoints.outer_iter().enumerate() {
            x.push(atom_frac[0]);
            y.push(atom_frac[1]);
            z.push(atom_frac[2]);
            r.push(radii[idx]);
        }
        Self { x, y, z, r }
    }

    fn len(&self) -> usize {
        self.x.len()
    }
}

#[inline]
fn cell_is_orthogonal(cell: &ndarray::ArrayView2<f64>) -> bool {
    cell[[0, 1]].abs() < ORTHO_TOL
        && cell[[0, 2]].abs() < ORTHO_TOL
        && cell[[1, 0]].abs() < ORTHO_TOL
        && cell[[1, 2]].abs() < ORTHO_TOL
        && cell[[2, 0]].abs() < ORTHO_TOL
        && cell[[2, 1]].abs() < ORTHO_TOL
}

#[inline]
fn d_max_sq_from_cell(cell: &ndarray::ArrayView2<f64>) -> f64 {
    let col_len_sq = |col: usize| {
        let c0 = cell[[0, col]];
        let c1 = cell[[1, col]];
        let c2 = cell[[2, col]];
        c0 * c0 + c1 * c1 + c2 * c2
    };
    let min_len_sq = col_len_sq(0).min(col_len_sq(1)).min(col_len_sq(2));
    0.25 * min_len_sq
}

#[inline]
fn frac_to_cart_sq(dx: f64, dy: f64, dz: f64, cell: CellMat) -> f64 {
    let cart_x = cell.c00 * dx + cell.c01 * dy + cell.c02 * dz;
    let cart_y = cell.c10 * dx + cell.c11 * dy + cell.c12 * dz;
    let cart_z = cell.c20 * dx + cell.c21 * dy + cell.c22 * dz;
    cart_x * cart_x + cart_y * cart_y + cart_z * cart_z
}

#[inline]
fn maybe_update_min_dist(min_dist: &mut f64, sq_dist: f64, r_atom: f64) {
    let threshold = *min_dist + r_atom;
    if threshold > 0.0 && sq_dist >= threshold * threshold {
        return;
    }
    let dist = sq_dist.sqrt() - r_atom;
    if dist < *min_dist {
        *min_dist = dist;
    }
}

#[inline]
fn mic_min_dist_skewed(
    raw_dx: f64,
    raw_dy: f64,
    raw_dz: f64,
    r_atom: f64,
    cell: CellMat,
    d_max_sq: f64,
    min_dist: &mut f64,
) {
    let dx = raw_dx - raw_dx.round();
    let dy = raw_dy - raw_dy.round();
    let dz = raw_dz - raw_dz.round();
    let sq_nint = frac_to_cart_sq(dx, dy, dz, cell);

    if sq_nint < d_max_sq {
        maybe_update_min_dist(min_dist, sq_nint, r_atom);
        return;
    }

    maybe_update_min_dist(min_dist, sq_nint, r_atom);
    for off in OFFSETS_FACE {
        let sq = frac_to_cart_sq(dx + off[0], dy + off[1], dz + off[2], cell);
        maybe_update_min_dist(min_dist, sq, r_atom);
    }
}

#[inline]
fn mic_min_dist_orthogonal_simd4(
    x_frac: f64,
    y_frac: f64,
    z_frac: f64,
    atoms: &AtomsSoA,
    base: usize,
    cell: CellMat,
    min_dist: &mut f64,
) {
    let xf = f64x4::splat(x_frac);
    let yf = f64x4::splat(y_frac);
    let zf = f64x4::splat(z_frac);

    let rdx = xf
        - f64x4::new([
            atoms.x[base],
            atoms.x[base + 1],
            atoms.x[base + 2],
            atoms.x[base + 3],
        ]);
    let rdy = yf
        - f64x4::new([
            atoms.y[base],
            atoms.y[base + 1],
            atoms.y[base + 2],
            atoms.y[base + 3],
        ]);
    let rdz = zf
        - f64x4::new([
            atoms.z[base],
            atoms.z[base + 1],
            atoms.z[base + 2],
            atoms.z[base + 3],
        ]);

    let dx = rdx - rdx.round();
    let dy = rdy - rdy.round();
    let dz = rdz - rdz.round();

    let c00 = f64x4::splat(cell.c00);
    let c01 = f64x4::splat(cell.c01);
    let c02 = f64x4::splat(cell.c02);
    let c10 = f64x4::splat(cell.c10);
    let c11 = f64x4::splat(cell.c11);
    let c12 = f64x4::splat(cell.c12);
    let c20 = f64x4::splat(cell.c20);
    let c21 = f64x4::splat(cell.c21);
    let c22 = f64x4::splat(cell.c22);

    let cart_x = c00 * dx + c01 * dy + c02 * dz;
    let cart_y = c10 * dx + c11 * dy + c12 * dz;
    let cart_z = c20 * dx + c21 * dy + c22 * dz;
    let sq = cart_x * cart_x + cart_y * cart_y + cart_z * cart_z;

    let sq_arr = sq.to_array();
    for lane in 0..4 {
        maybe_update_min_dist(min_dist, sq_arr[lane], atoms.r[base + lane]);
    }
}

#[inline]
fn min_dist_at_voxel(
    x_frac: f64,
    y_frac: f64,
    z_frac: f64,
    atoms: &AtomsSoA,
    cell: CellMat,
    is_orthogonal: bool,
    d_max_sq: f64,
) -> f32 {
    let mut min_dist = f64::MAX;
    let n = atoms.len();
    let mut i = 0;

    if is_orthogonal {
        while i + 4 <= n {
            mic_min_dist_orthogonal_simd4(x_frac, y_frac, z_frac, atoms, i, cell, &mut min_dist);
            i += 4;
        }
    }

    while i < n {
        let raw_dx = x_frac - atoms.x[i];
        let raw_dy = y_frac - atoms.y[i];
        let raw_dz = z_frac - atoms.z[i];
        let r_atom = atoms.r[i];

        if is_orthogonal {
            let dx = raw_dx - raw_dx.round();
            let dy = raw_dy - raw_dy.round();
            let dz = raw_dz - raw_dz.round();
            let sq = frac_to_cart_sq(dx, dy, dz, cell);
            maybe_update_min_dist(&mut min_dist, sq, r_atom);
        } else {
            mic_min_dist_skewed(raw_dx, raw_dy, raw_dz, r_atom, cell, d_max_sq, &mut min_dist);
        }
        i += 1;
    }

    min_dist as f32
}

#[pyfunction]
#[pyo3(text_signature = "(fpoints, cell, radii, nx, ny, nz)")]
pub fn dgrid_from_atoms(
    py: Python<'_>,
    fpoints: PyReadonlyArray2<'_, f64>,
    cell: PyReadonlyArray2<'_, f64>,
    radii: PyReadonlyArray1<'_, f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let fpoints = fpoints.as_array();
    let cell_view = cell.as_array();
    let radii = radii.as_array();

    let atoms = AtomsSoA::from_views(&fpoints, &radii);
    let total_points = nx * ny * nz;

    let cell_mat = CellMat {
        c00: cell_view[[0, 0]],
        c01: cell_view[[0, 1]],
        c02: cell_view[[0, 2]],
        c10: cell_view[[1, 0]],
        c11: cell_view[[1, 1]],
        c12: cell_view[[1, 2]],
        c20: cell_view[[2, 0]],
        c21: cell_view[[2, 1]],
        c22: cell_view[[2, 2]],
    };

    let is_orthogonal = cell_is_orthogonal(&cell_view);
    let d_max_sq = if is_orthogonal {
        0.0
    } else {
        d_max_sq_from_cell(&cell_view)
    };

    let buffer: Vec<f32> = (0..total_points)
        .into_par_iter()
        .map(|idx| {
            let i = idx / (ny * nz);
            let j = (idx / nz) % ny;
            let k = idx % nz;

            let x_frac = -0.5 + (i as f64) / ((nx - 1) as f64);
            let y_frac = -0.5 + (j as f64) / ((ny - 1) as f64);
            let z_frac = -0.5 + (k as f64) / ((nz - 1) as f64);

            min_dist_at_voxel(
                x_frac,
                y_frac,
                z_frac,
                &atoms,
                cell_mat,
                is_orthogonal,
                d_max_sq,
            )
        })
        .collect();

    let array = Array3::from_shape_vec((nx, ny, nz), buffer)
        .expect("Shape mismatch while reshaping the output distance grid");

    Ok(PyArray3::from_owned_array(py, array).to_owned())
}

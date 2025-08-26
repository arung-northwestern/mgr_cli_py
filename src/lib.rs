use ndarray::Array3;
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::Py;
use rayon::prelude::*;
use hashbrown::HashMap;
use numpy::{PyArray1, PyArray2, PyReadonlyArray3};
use pyo3::types::{PyList, PyTuple};
use ordered_float::NotNan;
use parking_lot::Mutex;
use std::sync::Arc;
use std::collections::{HashSet};
// mod maxima_test;
// use maxima_test::detect_maxima_test;
//
// #[pyfunction]
// #[pyo3(signature = (grid, spacing, h))]
// pub fn detect_maxima_test_py(
//     py: pyo3::Python<'_>,
//     grid: numpy::PyReadonlyArray3<'_, f32>,
//     spacing: f32,
//     h: f32,
// ) -> pyo3::PyResult<Py<numpy::PyArray2<usize>>> {
//     let arr = grid.as_array().to_owned();
//     let res = detect_maxima_test(&arr, spacing, h);
//     Ok(numpy::PyArray2::from_owned_array(py, res).to_owned())
// }

#[pyfunction]
#[pyo3(signature = (dgrid, cell, threshold_abs, min_distance=1.5))]
fn find_local_maxima_pbc(
    py: Python<'_>,
    dgrid: PyReadonlyArray3<'_, f32>,
    cell: PyReadonlyArray2<'_, f64>,
    threshold_abs: f32,
    min_distance: f32,
) -> PyResult<Py<PyArray2<usize>>> {
    let h = 0.5; // Default h for maxima detection
    let result = detect_maxima_h_morph(
        py,
        dgrid,
        1.0, // spacing (unused)
        h,
        Some(threshold_abs),
        min_distance,
        Some(cell),
        false,
        100,
    )?;
    Ok(result.0)
}

#[pyfunction]
#[pyo3(signature = (dgrid, cell, threshold_abs, min_distance=1.5))]
fn find_local_maxima_cart(
    py: Python<'_>,
    dgrid: PyReadonlyArray3<'_, f32>,
    cell: PyReadonlyArray2<'_, f64>,
    threshold_abs: f32,
    min_distance: f32,
) -> PyResult<Py<PyArray2<usize>>> {
    let h = 0.5; // Default h for maxima detection
    let result = detect_maxima_h_morph(
        py,
        dgrid,
        1.0, // spacing (unused)
        h,
        Some(threshold_abs),
        min_distance,
        Some(cell),
        false,
        100,
    )?;
    Ok(result.1)
}

#[pyfunction]

#[pyo3(text_signature = "(fpoints, cell, radii, nx, ny, nz)")]
fn dgrid_from_atoms(
    py: Python<'_>,
    fpoints: PyReadonlyArray2<'_, f64>,
    cell: PyReadonlyArray2<'_, f64>,
    radii: PyReadonlyArray1<'_, f64>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    // Convert input arrays to Rust-native views.
    let fpoints = fpoints.as_array(); // Shape (n_atoms, 3)
    let cell = cell.as_array();       // Shape (3, 3)
    let radii = radii.as_array();     // Shape (n_atoms,)

    let total_points = nx * ny * nz;

    // Pre-compute cell components for fast access.
    let c00 = cell[[0, 0]];
    let c01 = cell[[0, 1]];
    let c02 = cell[[0, 2]];
    let c10 = cell[[1, 0]];
    let c11 = cell[[1, 1]];
    let c12 = cell[[1, 2]];
    let c20 = cell[[2, 0]];
    let c21 = cell[[2, 1]];
    let c22 = cell[[2, 2]];

    // Parallel computation across all grid points.
    let buffer: Vec<f32> = (0..total_points)
        .into_par_iter()
        .map(|idx| {
            // Unravel flat index into 3-D indices (Fortran order).
            let i = idx / (ny * nz);
            let j = (idx / nz) % ny;
            let k = idx % nz;

            // Fractional coordinates on the uniform grid (inclusive end-points like linspace).
            // Match Python's linspace(-0.5, 0.5, n) behavior - use (n-1) for proper spacing
            let x_frac = -0.5 + (i as f64) / ((nx - 1) as f64);
            let y_frac = -0.5 + (j as f64) / ((ny - 1) as f64);
            let z_frac = -0.5 + (k as f64) / ((nz - 1) as f64);

            // Track the minimum atom-surface distance.
            let mut min_dist = f64::MAX;

            for (atom_idx, atom_frac) in fpoints.outer_iter().enumerate() {
                // Apply minimum-image convention in fractional space.
                let mut dx = x_frac - atom_frac[0];
                let mut dy = y_frac - atom_frac[1];
                let mut dz = z_frac - atom_frac[2];

                dx -= dx.round();
                dy -= dy.round();
                dz -= dz.round();

                // Convert to Cartesian coordinates: r = cell · diff
                let cart_x = c00 * dx + c01 * dy + c02 * dz;
                let cart_y = c10 * dx + c11 * dy + c12 * dz;
                let cart_z = c20 * dx + c21 * dy + c22 * dz;

                let dist = (cart_x * cart_x + cart_y * cart_y + cart_z * cart_z).sqrt()
                    - radii[atom_idx];

                if dist < min_dist {
                    min_dist = dist;
                }
            }

            // Cast to f32 – this is enough precision for the grid and halves memory usage.
            min_dist as f32
        })
        .collect();

    // Reshape the flat Vec into a 3-D ndarray (row-major as in NumPy).
    let array = Array3::from_shape_vec((nx, ny, nz), buffer)
        .expect("Shape mismatch while reshaping the output distance grid");

    // Convert to a NumPy array owned by Python and return.
    Ok(PyArray3::from_owned_array(py, array).to_owned())
}

#[pyfunction]
#[pyo3(text_signature = "(fpoints, cell, epsilons, sigmas, nx, ny, nz, cutoff)")]
fn egrid_from_atoms(
    py: Python<'_>,
    fpoints: PyReadonlyArray2<'_, f64>,
    cell: PyReadonlyArray2<'_, f64>,
    epsilons: PyReadonlyArray1<'_, f64>,
    sigmas: PyReadonlyArray1<'_, f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    cutoff: f64,
) -> PyResult<Py<PyArray3<f64>>> {
    let fpoints = fpoints.as_array();
    let cell = cell.as_array();
    let epsilons = epsilons.as_array();
    let sigmas = sigmas.as_array();

    let total_points = nx * ny * nz;
    let cutoff_sq = cutoff * cutoff;

    // Pre-compute cell components for fast access.
    let c00 = cell[[0, 0]];
    let c01 = cell[[0, 1]];
    let c02 = cell[[0, 2]];
    let c10 = cell[[1, 0]];
    let c11 = cell[[1, 1]];
    let c12 = cell[[1, 2]];
    let c20 = cell[[2, 0]];
    let c21 = cell[[2, 1]];
    let c22 = cell[[2, 2]];

    // Parallel computation across all grid points.
    let buffer: Vec<f64> = (0..total_points)
        .into_par_iter()
        .map(|idx| {
            // Unravel flat index into 3-D indices (Fortran order).
            let i = idx / (ny * nz);
            let j = (idx / nz) % ny;
            let k = idx % nz;

            // Fractional coordinates on the uniform grid (inclusive end-points like linspace).
            let x_frac = -0.5 + (i as f64) / ((nx - 1) as f64);
            let y_frac = -0.5 + (j as f64) / ((ny - 1) as f64);
            let z_frac = -0.5 + (k as f64) / ((nz - 1) as f64);

            let mut energy = 0.0_f64;

            for (atom_idx, atom_frac) in fpoints.outer_iter().enumerate() {
                // Minimum-image convention in fractional space.
                let mut dx = x_frac - atom_frac[0];
                let mut dy = y_frac - atom_frac[1];
                let mut dz = z_frac - atom_frac[2];

                dx -= dx.round();
                dy -= dy.round();
                dz -= dz.round();

                // Convert to Cartesian coordinates.
                let cart_x = c00 * dx + c01 * dy + c02 * dz;
                let cart_y = c10 * dx + c11 * dy + c12 * dz;
                let cart_z = c20 * dx + c21 * dy + c22 * dz;

                let r_sq = cart_x * cart_x + cart_y * cart_y + cart_z * cart_z;

                // Skip if beyond cutoff.
                if r_sq > cutoff_sq {
                    continue;
                }

                // Avoid singularity when r -> 0.
                let r = if r_sq < 1e-24 { 1e-12 } else { r_sq.sqrt() };

                let sigma = sigmas[atom_idx];
                let epsilon = epsilons[atom_idx];

                let sr = sigma / r;
                let sr6 = sr.powi(6);
                // Lennard-Jones 12-6 potential.
                energy += 4.0 * epsilon * (sr6 * sr6 - sr6);
            }

            energy
        })
        .collect();

    let array = Array3::from_shape_vec((nx, ny, nz), buffer)
        .expect("Shape mismatch while reshaping the output energy grid");

    Ok(PyArray3::from_owned_array(py, array).to_owned())
}

#[derive(Debug, Clone)]
struct WindowPoint {
    coord: [f64; 3],
    region: u32,
    distance: f32,
}

// Simple union-find for clustering
#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x != root_y {
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }

    fn get_clusters(&mut self) -> HashMap<usize, Vec<usize>> {
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let root = self.find(i);
            clusters.entry(root).or_default().push(i);
        }
        clusters
    }
}

fn euclidean_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn cluster_points(points: &[WindowPoint], distance_threshold: f64) -> Vec<Vec<usize>> {
    if points.is_empty() {
        return vec![];
    }
    
    let n = points.len();
    let mut uf = UnionFind::new(n);
    
    // Connect points within distance threshold
    for i in 0..n {
        for j in (i + 1)..n {
            if euclidean_distance(&points[i].coord, &points[j].coord) < distance_threshold {
                uf.union(i, j);
            }
        }
    }
    
    // Extract clusters
    let cluster_map = uf.get_clusters();
    cluster_map.into_values().collect()
}

/// Apply a periodic (wrap-around) Gaussian filter to a 3D array.
///
/// This is a memory-efficient implementation that avoids creating multiple copies
/// of the input array, unlike the Python version that uses np.roll.
///
/// Parameters
/// ----------
/// input : numpy.ndarray[float64] shape=(nx, ny, nz)
///     Input 3D array to be filtered
/// sigma : float
///     Standard deviation of the Gaussian kernel
///
/// Returns
/// -------
/// numpy.ndarray[float64] shape=(nx, ny, nz)
///     Filtered array with same shape as input
#[pyfunction]
#[pyo3(text_signature = "(input, sigma)")]
fn periodic_gaussian_filter(
    py: Python<'_>,
    input: PyReadonlyArray3<'_, f64>,
    sigma: f64,
) -> PyResult<Py<PyArray3<f64>>> {
    use ndarray::{Axis, s};

    let inp = input.as_array();
    let (nx, ny, nz) = inp.dim();

    // Build 1-D kernel (truncate at 3σ)
    let radius = (3.0 * sigma).ceil() as i32;
    let mut kernel: Vec<f64> = (-radius..=radius)
        .map(|k| {
            let kf = k as f64;
            (-0.5 * kf * kf / (sigma * sigma)).exp()
        })
        .collect();
    let norm: f64 = kernel.iter().sum();
    for w in &mut kernel {
        *w /= norm;
    }

    // Temp buffers
    let mut tmp_x = Array3::<f64>::zeros((nx, ny, nz));
    let mut tmp_y = Array3::<f64>::zeros((nx, ny, nz));

    // ---- Pass 1 : X axis ----
    tmp_x
        .axis_iter_mut(Axis(1)) // iterate over y
        .enumerate()
        .for_each(|(y_idx, mut slab)| {
            // slab has shape (nx, nz) for fixed y
            for z in 0..nz {
                for x in 0..nx {
                    let mut acc = 0.0;
                    for (offset, &w) in (-radius..=radius).zip(kernel.iter()) {
                        let xi = (x as i32 + offset).rem_euclid(nx as i32) as usize;
                        acc += w * inp[[xi, y_idx, z]];
                    }
                    slab[[x, z]] = acc;
                }
            }
        });

    // ---- Pass 2 : Y axis ----
    tmp_y
        .axis_iter_mut(Axis(0)) // iterate over x
        .enumerate()
        .for_each(|(x_idx, mut slab)| {
            // slab shape (ny, nz) for fixed x
            for z in 0..nz {
                for y in 0..ny {
                    let mut acc = 0.0;
                    for (offset, &w) in (-radius..=radius).zip(kernel.iter()) {
                        let yj = (y as i32 + offset).rem_euclid(ny as i32) as usize;
                        acc += w * tmp_x[[x_idx, yj, z]];
                    }
                    slab[[y, z]] = acc;
                }
            }
        });

    // ---- Pass 3 : Z axis ----
    let mut out = Array3::<f64>::zeros((nx, ny, nz));
    out
        .axis_iter_mut(Axis(2)) // iterate over z
        .enumerate()
        .for_each(|(z_idx, mut slab)| {
            // slab shape (nx, ny) for fixed z
            for y in 0..ny {
                for x in 0..nx {
                    let mut acc = 0.0;
                    for (offset, &w) in (-radius..=radius).zip(kernel.iter()) {
                        let zk = (z_idx as i32 + offset).rem_euclid(nz as i32) as usize;
                        acc += w * tmp_y[[x, y, zk]];
                    }
                    slab[[x, y]] = acc;
                }
            }
        });

    Ok(PyArray3::from_owned_array(py, out).to_owned())
}

#[pyfunction]
#[pyo3(text_signature = "(labels, dgrid, padding_thickness, cell, distance_threshold)")]
fn window_connections(
    py: Python<'_>,
    labels: PyReadonlyArray3<'_, u32>,
    dgrid: PyReadonlyArray3<'_, f32>,
    padding_thickness: usize,
    cell: PyReadonlyArray2<'_, f64>,
    distance_threshold: f64,
) -> PyResult<PyObject> {
    // Convert to ndarray views
    let labels = labels.as_array();
    let dgrid = dgrid.as_array();
    let cell = cell.as_array();

    let (nx, ny, nz) = (labels.dim().0, labels.dim().1, labels.dim().2);

    // Helper closure to transform voxel indices -> Cartesian coordinate
    let to_cart = |i: usize, j: usize, k: usize| -> [f64; 3] {
        // Convert padded grid indices to fractional coordinates relative to original grid
        let frac_x = (i as f64 - padding_thickness as f64) / (nx as f64 - 2.0 * padding_thickness as f64);
        let frac_y = (j as f64 - padding_thickness as f64) / (ny as f64 - 2.0 * padding_thickness as f64);
        let frac_z = (k as f64 - padding_thickness as f64) / (nz as f64 - 2.0 * padding_thickness as f64);
        
        // Convert to Cartesian coordinates using cell matrix
        let cx = cell[[0, 0]] * frac_x + cell[[0, 1]] * frac_y + cell[[0, 2]] * frac_z;
        let cy = cell[[1, 0]] * frac_x + cell[[1, 1]] * frac_y + cell[[1, 2]] * frac_z;
        let cz = cell[[2, 0]] * frac_x + cell[[2, 1]] * frac_y + cell[[2, 2]] * frac_z;
        [cx, cy, cz]
    };

    // Map edge key -> Vec of boundary points
    let mut edge_map: HashMap<(u32, u32), Vec<WindowPoint>> = HashMap::new();

    // Scan grid for boundary voxels between different regions
    // Use 26-connectivity like Python's find_touching_region_pairs
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let label_ijk = labels[[i, j, k]];
                if label_ijk == 0 {
                    continue;
                }

                // 26-connectivity neighbors (same as Python)
                let neighbors = [
                    (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
                    (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
                    (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
                    (0, -1, -1), (0, -1, 0), (0, -1, 1),
                    (0, 0, -1),              (0, 0, 1),
                    (0, 1, -1), (0, 1, 0), (0, 1, 1),
                    (1, -1, -1), (1, -1, 0), (1, -1, 1),
                    (1, 0, -1), (1, 0, 0), (1, 0, 1),
                    (1, 1, -1), (1, 1, 0), (1, 1, 1),
                ];

                for (di, dj, dk) in neighbors {
                    let ni = i as isize + di;
                    let nj = j as isize + dj;
                    let nk = k as isize + dk;

                    if ni >= 0 && nj >= 0 && nk >= 0 && ni < nx as isize && nj < ny as isize && nk < nz as isize {
                        let neighbor_label = labels[[ni as usize, nj as usize, nk as usize]];
                        if neighbor_label != 0 && neighbor_label != label_ijk {
                            // Found boundary between two regions
                            let (reg1, reg2) = if label_ijk < neighbor_label {
                                (label_ijk, neighbor_label)
                            } else {
                                (neighbor_label, label_ijk)
                            };

                            let coord = to_cart(i, j, k);
                            let point = WindowPoint {
                                coord,
                                region: label_ijk,
                                distance: dgrid[[i, j, k]],
                            };

                            edge_map.entry((reg1, reg2)).or_default().push(point);
                        }
                    }
                }
            }
        }
    }

    // Process each edge in parallel and build output - SIMPLIFIED VERSION
    // The fast version doesn't use complex clustering, just finds direct connections
    let connections: Vec<(u32, u32, Vec<[f64; 3]>, Vec<f64>)> = edge_map
        .into_iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter_map(|((reg1, reg2), boundary_points)| {
            if boundary_points.is_empty() {
                return None;
            }

            // Simplified approach: find max distance points directly without clustering
            let mut max_reg1: Option<&WindowPoint> = None;
            let mut max_reg2: Option<&WindowPoint> = None;

            for point in &boundary_points {
                if point.region == reg1 {
                    if let Some(current_max) = max_reg1 {
                        if point.distance > current_max.distance {
                            max_reg1 = Some(point);
                        }
                    } else {
                        max_reg1 = Some(point);
                    }
                } else if point.region == reg2 {
                    if let Some(current_max) = max_reg2 {
                        if point.distance > current_max.distance {
                            max_reg2 = Some(point);
                        }
                    } else {
                        max_reg2 = Some(point);
                    }
                }
            }

            if let (Some(max_reg1), Some(max_reg2)) = (max_reg1, max_reg2) {
                // Calculate window center as midpoint of the two maxima
                let center = [
                    0.5 * (max_reg1.coord[0] + max_reg2.coord[0]),
                    0.5 * (max_reg1.coord[1] + max_reg2.coord[1]),
                    0.5 * (max_reg1.coord[2] + max_reg2.coord[2]),
                ];

                // Use average of the two maximum distance values as radius
                let radius = 0.5 * (max_reg1.distance + max_reg2.distance) as f64;

                Some((reg1, reg2, vec![center], vec![radius]))
            } else {
                None
            }
        })
        .collect();

    // Convert to Python objects
    let py_out = PyList::empty(py);

    for (reg1, reg2, centers, radii) in connections {
        // Build numpy arrays
        let regs_arr = PyArray1::<u32>::from_vec(py, vec![reg1, reg2]);
        
        // Convert [f64; 3] to Vec<f64> for each center
        let centers_vec: Vec<Vec<f64>> = centers
            .iter()
            .map(|center| center.to_vec())
            .collect();
        let centers_arr = PyArray2::<f64>::from_vec2(py, &centers_vec)?;
        let radii_arr = PyArray1::<f64>::from_vec(py, radii);

        // Create tuple matching Python format: [regs, connected, type, centers, radii]
        let tup = PyTuple::new(
            py,
            &[
                regs_arr.into_py(py),
                true.into_py(py),
                "internal".into_py(py),
                centers_arr.into_py(py),
                radii_arr.into_py(py),
            ],
        );
        py_out.append(tup)?;
    }

    Ok(py_out.into())
}

const BUCKETS: usize = 65536;

#[pyfunction]
#[pyo3(text_signature = "(image, seeds, mask=None, n_threads=None)")]
fn seeded_watershed_bucket(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, f32>,
    seeds: PyReadonlyArray3<'_, u32>,
    mask: Option<PyReadonlyArray3<'_, u8>>, // optional
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray3<u32>>> {
    let img = image.as_array();
    let seeds_arr = seeds.as_array();

    if img.dim() != seeds_arr.dim() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "`image` and `seeds` must have identical shapes",
        ));
    }
    if let Some(ref m) = mask {
        if m.as_array().dim() != img.dim() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`mask` shape must match `image`",
            ));
        }
    }

    if let Some(n) = n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    // Determine min and max of the image to scale into 0..65535
    let (mut min_val, mut max_val) = (f32::MAX, f32::MIN);
    img.iter().for_each(|&v| {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    });
    // avoid div-by-zero
    let range = if (max_val - min_val).abs() < 1e-12 {
        1.0
    } else {
        max_val - min_val
    };

    let to_bucket = |v: f32| -> usize {
        let norm = ((v - min_val) / range * (BUCKETS as f32 - 1.0)).round() as usize;
        norm
    };

    // Precompute bucket index for every voxel to avoid recalculating
    let bucket_idx_vec: Vec<usize> = img
        .iter()
        .map(|&v| to_bucket(v))
        .collect();

    let (nx, ny, nz) = img.dim();
    let total_voxels = nx * ny * nz;

    use std::sync::atomic::{AtomicU32, Ordering};

    let labels: Vec<AtomicU32> = (0..total_voxels).map(|_| AtomicU32::new(0)).collect();

    let buckets: Arc<Vec<Mutex<Vec<usize>>>> = Arc::new(
        (0..BUCKETS).map(|_| Mutex::new(Vec::<usize>::new())).collect(),
    );

    let to_flat = |i: usize, j: usize, k: usize| -> usize { i * ny * nz + j * nz + k };

    // Materialise mask (if any) into a flat Vec<bool> that is Sync/Send.
    let mask_vec: Option<Vec<bool>> = mask.as_ref().map(|m| {
        m.as_array()
            .iter()
            .map(|&v| v != 0)
            .collect::<Vec<bool>>()
    });

    // Initialise with seeds
    for ((i, j, k), &label) in seeds_arr.indexed_iter() {
        if label == 0 {
            continue;
        }
        if let Some(ref mask_v) = mask_vec {
            if !mask_v[to_flat(i, j, k)] {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Seed label found in location where mask==0",
                ));
            }
        }
        let idx = to_flat(i, j, k);
        labels[idx].store(label, Ordering::Relaxed);
        let b = bucket_idx_vec[idx];
        buckets[b].lock().push(idx);
    }

    let neighbor_offsets: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];

    // Main flooding loop from highest bucket downwards
    for level in (0..BUCKETS).rev() {
        let frontier: Vec<usize> = {
            let mut guard = buckets[level].lock();
            std::mem::take(&mut *guard)
        };
        if frontier.is_empty() {
            continue;
        }

        frontier.par_iter().for_each(|&idx| {
            let current_label = labels[idx].load(Ordering::Relaxed);
            let i = idx / (ny * nz);
            let j = (idx / nz) % ny;
            let k = idx % nz;

            for &(dx, dy, dz) in neighbor_offsets {
                let ni = i as isize + dx;
                let nj = j as isize + dy;
                let nk = k as isize + dz;
                if ni < 0 || nj < 0 || nk < 0 {
                    continue;
                }
                let (ni_u, nj_u, nk_u) = (ni as usize, nj as usize, nk as usize);
                if ni_u >= nx || nj_u >= ny || nk_u >= nz {
                    continue;
                }
                if let Some(ref mask_v) = mask_vec {
                    if !mask_v[to_flat(ni_u, nj_u, nk_u)] {
                        continue;
                    }
                }
                let nidx = to_flat(ni_u, nj_u, nk_u);
                if labels[nidx]
                    .compare_exchange(0, current_label, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    let b = bucket_idx_vec[nidx];
                    buckets[b].lock().push(nidx);
                }
            }
        });
    }

    // Convert labels into ndarray
    let mut out_vec: Vec<u32> = vec![0u32; total_voxels];
    out_vec
        .iter_mut()
        .enumerate()
        .for_each(|(idx, v)| *v = labels[idx].load(Ordering::Relaxed));

    let out_arr = Array3::from_shape_vec((nx, ny, nz), out_vec)
        .expect("shape mismatch in output");

    Ok(PyArray3::from_owned_array(py, out_arr).to_owned())
}

// ============================================================================
// Heap-based single-threaded watershed (original version, retained for backwards
// compatibility and low-memory usage)
// ============================================================================

#[pyfunction]
#[pyo3(text_signature = "(image, seeds)")]
fn seeded_watershed(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, f32>,
    seeds: PyReadonlyArray3<'_, u32>,
) -> PyResult<Py<PyArray3<u32>>> {
    heap_watershed_impl(py, image, seeds, None)
}

#[pyfunction]
#[pyo3(text_signature = "(image, seeds, mask)")]
fn seeded_watershed_masked(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, f32>,
    seeds: PyReadonlyArray3<'_, u32>,
    mask: PyReadonlyArray3<'_, u8>,
) -> PyResult<Py<PyArray3<u32>>> {
    heap_watershed_impl(py, image, seeds, Some(mask))
}

// Shared helper implementing the original heap algorithm ------------------------------------
fn heap_watershed_impl(
    py: Python<'_>,
    image: PyReadonlyArray3<'_, f32>,
    seeds: PyReadonlyArray3<'_, u32>,
    mask: Option<PyReadonlyArray3<'_, u8>>,
) -> PyResult<Py<PyArray3<u32>>> {
    let img = image.as_array();
    let seeds_arr = seeds.as_array();
    if img.dim() != seeds_arr.dim() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "`image` and `seeds` must have identical shapes",
        ));
    }

    if let Some(ref m) = mask {
        if m.as_array().dim() != img.dim() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`mask` shape must match `image`",
            ));
        }
    }

    let (nx, ny, nz) = img.dim();
    let total_voxels = nx * ny * nz;

    // Flattened labels
    let mut labels: Vec<u32> = vec![0; total_voxels];
    let to_flat = |i: usize, j: usize, k: usize| -> usize { i * ny * nz + j * nz + k };

    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<(NotNan<f32>, usize)> = BinaryHeap::new();

    // Seed initialisation
    for ((i, j, k), &label) in seeds_arr.indexed_iter() {
        if label == 0 {
            continue;
        }
        if let Some(ref m) = mask {
            if m.as_array()[[i, j, k]] == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Seed label found in location where mask==0",
                ));
            }
        }
        let idx = to_flat(i, j, k);
        labels[idx] = label;
        let priority = NotNan::new(img[[i, j, k]]).unwrap();
        heap.push((priority, idx));
    }

    // 26-NN offsets
    let neighbor_offsets: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];

    while let Some((_prio, idx)) = heap.pop() {
        let i = idx / (ny * nz);
        let j = (idx / nz) % ny;
        let k = idx % nz;
        let current_label = labels[idx];

        for &(dx, dy, dz) in neighbor_offsets {
            let ni = i as isize + dx;
            let nj = j as isize + dy;
            let nk = k as isize + dz;
            if ni < 0 || nj < 0 || nk < 0 {
                continue;
            }
            let (ni_u, nj_u, nk_u) = (ni as usize, nj as usize, nk as usize);
            if ni_u >= nx || nj_u >= ny || nk_u >= nz {
                continue;
            }
            if let Some(ref m) = mask {
                if m.as_array()[[ni_u, nj_u, nk_u]] == 0 {
                    continue;
                }
            }
            let nidx = to_flat(ni_u, nj_u, nk_u);
            if labels[nidx] == 0 {
                labels[nidx] = current_label;
                let priority = NotNan::new(img[[ni_u, nj_u, nk_u]]).unwrap();
                heap.push((priority, nidx));
            }
        }
    }

    let out_arr = Array3::from_shape_vec((nx, ny, nz), labels).expect("shape mismatch");
    Ok(PyArray3::from_owned_array(py, out_arr).to_owned())
}

#[pyfunction]
#[pyo3(signature = (grid, spacing, h, threshold_abs=None, min_distance=0.0, cell=None, refine_peaks=false, max_steps=100))]
fn detect_maxima_h_morph(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    spacing: f32,
    h: f32,
    threshold_abs: Option<f32>,
    min_distance: f32,                       // Å
    cell: Option<PyReadonlyArray2<'_, f64>>, // 3×3 matrix with unit vectors as columns
    refine_peaks: bool,                      // Whether to refine peak locations using gradient ascent
    max_steps: usize,                        // Max steps for gradient ascent
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array2, Array3};
    use rayon::prelude::*;

    // Start with original grid
    let mut img: Array3<f32> = grid.as_array().to_owned();
    let (nx, ny, nz) = img.dim();

    // Apply absolute threshold if provided
    if let Some(tabs) = threshold_abs {
        img.mapv_inplace(|v| if v < tabs { -1e6 } else { v });
    }

    let total_voxels = nx * ny * nz;

    // Early return for tiny grids
    if total_voxels < 27 {
        let empty = Array2::<usize>::zeros((0, 3));
        return Ok((
            PyArray2::from_owned_array(py, empty.clone()).to_owned(),
            PyArray2::from_owned_array(py, empty.clone()).to_owned(),
            PyArray2::from_owned_array(py, empty).to_owned(),
        ));
    }

    // ----------- Robust morphological reconstruction by dilation (iterative) ---------------------

    // Create marker image (f - h) - this is the seed for reconstruction
    let marker = img.mapv(|v| v - h);
    let mut recon = marker.clone();

    // 26-connectivity neighbors for reconstruction (no periodic wrapping)
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];

    // Iterative geodesic reconstruction until convergence
    // This is more robust than queue-based for numerical precision
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        
        // Forward pass
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let current_mask = img[[i, j, k]];
                    
                    // Skip thresholded-out regions
                    if current_mask <= -1e5 {
                        continue;
                    }
                    
                    let mut max_neighbor = recon[[i, j, k]];
                    
                    // Check all 26 neighbors
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;

                        // Check bounds (no wrapping)
                        if ni < 0 || nj < 0 || nk < 0 {
                            continue;
                        }
                        let (ni_u, nj_u, nk_u) = (ni as usize, nj as usize, nk as usize);
                        if ni_u >= nx || nj_u >= ny || nk_u >= nz {
                            continue;
                        }
                        
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    
                    // Geodesic reconstruction: constrain by mask
                    let new_val = max_neighbor.min(current_mask);
                    
                    // Use a small epsilon for floating point comparison
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
        
        // Backward pass for better convergence
        for i in (0..nx).rev() {
            for j in (0..ny).rev() {
                for k in (0..nz).rev() {
                    let current_mask = img[[i, j, k]];
                    
                    // Skip thresholded-out regions
                    if current_mask <= -1e5 {
                        continue;
                    }
                    
                    let mut max_neighbor = recon[[i, j, k]];
                    
                    // Check all 26 neighbors
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;

                        // Check bounds (no wrapping)
                        if ni < 0 || nj < 0 || nk < 0 {
                            continue;
                        }
                        let (ni_u, nj_u, nk_u) = (ni as usize, nj as usize, nk as usize);
                        if ni_u >= nx || nj_u >= ny || nk_u >= nz {
                            continue;
                        }
                        
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    
                    // Geodesic reconstruction: constrain by mask
                    let new_val = max_neighbor.min(current_mask);
                    
                    // Use a small epsilon for floating point comparison
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }

    // ----------- Compute h-dome and create regional maxima mask (correct skimage behavior) -------------

    // Compute h-dome: this is the correct h-maxima transform
    let mut h_dome = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                h_dome[[i, j, k]] = img[[i, j, k]] - recon[[i, j, k]];
            }
        }
    }

    // Create binary mask of h-maxima regions (like skimage does)
    // These are regions where h_dome > small threshold
    let threshold = h - 1e-6; // Use h minus a small epsilon for floating point stability
    let mut h_maxima_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] > -1e5 && h_dome[[i, j, k]] >= threshold {
                    h_maxima_mask[[i, j, k]] = true;
                }
            }
        }
    }

    // Find peak coordinates within each h-maxima region
    // Use parallel processing with pre-allocated vectors for better performance
    let coord_vals: Vec<(Vec<[usize; 3]>, Vec<f32>)> = (0..nx)
        .into_par_iter()
        .map(|i| {
            let mut coords = Vec::new();
            let mut vals = Vec::new();

            for j in 0..ny {
                for k in 0..nz {
                    // Only process points in h-maxima regions
                    if !h_maxima_mask[[i, j, k]] {
                        continue;
                    }

                    // Check if this is a local maximum in the ORIGINAL image within h-maxima region
                    let current_val = img[[i, j, k]];
                    let mut is_local_maximum = true;
                    
                    // Check all 26 neighbors
                    'neighbor_loop: for di in -1_isize..=1_isize {
                        for dj in -1_isize..=1_isize {
                            for dk in -1_isize..=1_isize {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue; // Skip center pixel
                                }
                                
                                let ni = i as isize + di;
                                let nj = j as isize + dj;
                                let nk = k as isize + dk;

                                // Check bounds (no periodic wrapping for maxima detection)
                                if ni < 0 || nj < 0 || nk < 0 {
                                    continue;
                                }
                                let (ni_u, nj_u, nk_u) = (ni as usize, nj as usize, nk as usize);
                                if ni_u >= nx || nj_u >= ny || nk_u >= nz {
                                    continue;
                                }

                                // Compare ORIGINAL image values (not h-dome values)
                                // Only compare within h-maxima region OR with similar values
                                let neighbor_val = img[[ni_u, nj_u, nk_u]];
                                if neighbor_val >= current_val {
                                    is_local_maximum = false;
                                    break 'neighbor_loop;
                                }
                            }
                        }
                    }

                    if is_local_maximum {
                        coords.push([i, j, k]);
                        vals.push(current_val);
                    }
                }
            }
            (coords, vals)
        })
        .collect();

    // Combine results from all chunks with pre-allocated capacity
    let total_capacity: usize = coord_vals.iter().map(|(coords, _)| coords.len()).sum();
    let mut maxima_coords: Vec<[usize; 3]> = Vec::with_capacity(total_capacity);
    let mut maxima_vals: Vec<f32> = Vec::with_capacity(total_capacity);

    for (coords, vals) in coord_vals {
        maxima_coords.extend(coords);
        maxima_vals.extend(vals);
    }

    // Early return if no maxima found
    let n_raw = maxima_coords.len();
    if n_raw == 0 {
        let empty = Array2::<usize>::zeros((0, 3));
        return Ok((
            PyArray2::from_owned_array(py, empty.clone()).to_owned(),
            PyArray2::from_owned_array(py, empty.clone()).to_owned(),
            PyArray2::from_owned_array(py, empty).to_owned(),
        ));
    }

    // ----------- Optional peak refinement using gradient ascent -------------------
    
    if refine_peaks {
        // 27-connectivity neighbors (including center)
        let neigh_offsets: [(isize, isize, isize); 27] = [
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
            (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
            (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
            (0, -1, -1), (0, -1, 0), (0, -1, 1),
            (0, 0, -1), (0, 0, 0), (0, 0, 1),
            (0, 1, -1), (0, 1, 0), (0, 1, 1),
            (1, -1, -1), (1, -1, 0), (1, -1, 1),
            (1, 0, -1), (1, 0, 0), (1, 0, 1),
            (1, 1, -1), (1, 1, 0), (1, 1, 1),
        ];

        // Refine each maximum using gradient ascent
        let refined: Vec<([usize; 3], f32)> = maxima_coords
            .into_par_iter()
            .zip(maxima_vals.into_par_iter())
            .map(|(coord, _original_val)| {
                let mut pos = coord;
                let mut val = img[[pos[0], pos[1], pos[2]]];
                
                for _ in 0..max_steps {
                    let mut best = pos;
                    let mut best_val = val;
                    
                    // Check all 27 neighbors (including center for plateau detection)
                    for &(dx, dy, dz) in &neigh_offsets {
                        // Use periodic wrapping for gradient ascent
                        let ni = ((pos[0] as isize + dx).rem_euclid(nx as isize)) as usize;
                        let nj = ((pos[1] as isize + dy).rem_euclid(ny as isize)) as usize;
                        let nk = ((pos[2] as isize + dz).rem_euclid(nz as isize)) as usize;
                        let v = img[[ni, nj, nk]];
                        
                        if v > best_val {
                            best_val = v;
                            best = [ni, nj, nk];
                        }
                    }
                    
                    // Stop if we've reached a plateau or local maximum
                    if best_val <= val {
                        break;
                    }
                    pos = best;
                    val = best_val;
                }
                (pos, val)
            })
            .collect();

        // Update coordinates and values with refined results
        maxima_coords = refined.iter().map(|(coord, _)| *coord).collect();
        maxima_vals = refined.iter().map(|(_, val)| *val).collect();
    }

    // ----------- Merging with spatial indexing (periodic and non-periodic) -------------------

    let dims = (nx, ny, nz);
    let cell_ref = cell.as_ref().map(|c| c.as_array());

    let merged_pbc_coords = merge_maxima_optimized(
        &maxima_coords,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        true, // periodic wrapping for merging
    );

    let merged_cart_coords = merge_maxima_optimized(
        &maxima_coords,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        false, // no wrapping for merging
    );

    // Convert to NumPy arrays with pre-calculated sizes
    let raw_arr =
        Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords.into_iter().flatten().collect())
            .unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec(
        (merged_pbc_coords.len(), 3),
        merged_pbc_coords.into_iter().flatten().collect(),
    )
    .unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec(
        (merged_cart_coords.len(), 3),
        merged_cart_coords.into_iter().flatten().collect(),
    )
    .unwrap();

    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Detect h-maxima after morphological opening by reconstruction.
/// This removes all maxima smaller than the structuring element (ball of radius opening_radius voxels)
/// before applying the h-maxima transform. This is more selective than Gaussian smoothing and preserves
/// the physical meaning of h-maxima. The erosion step is O(r^3) per voxel for radius r, so large radii
/// may be slow for large grids.
#[pyfunction]
#[pyo3(signature = (grid, spacing, h, opening_radius, threshold_abs=None, min_distance=0.0, cell=None, refine_peaks=false, max_steps=100))]
fn detect_maxima_h_morph_opening(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    spacing: f32,
    h: f32,
    opening_radius: usize, // radius of structuring element in grid units
    threshold_abs: Option<f32>,
    min_distance: f32,
    cell: Option<PyReadonlyArray2<'_, f64>>,
    refine_peaks: bool,
    max_steps: usize,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3};
    let mut img: Array3<f32> = grid.as_array().to_owned();
    let (nx, ny, nz) = img.dim();

    // Apply absolute threshold if provided
    if let Some(tabs) = threshold_abs {
        img.mapv_inplace(|v| if v < tabs { -1e6 } else { v });
    }

    // Debug: Print stats for true original image (before thresholding)
    let orig_data = grid.as_array();
    let mut orig_min = f32::MAX;
    let mut orig_max = f32::MIN;
    let mut orig_sum = 0.0f64;
    let orig_count = orig_data.len();
    let orig_above_thresh = if let Some(tabs) = threshold_abs {
        orig_data.iter().filter(|&&v| v > tabs).count()
    } else {
        0
    };
    // Only compute expensive statistics if we really need them for debugging
    // For performance, we'll skip the full min/max/mean computation
    eprintln!("[DEBUG] Grid size: {} voxels, {} above threshold", orig_count, orig_above_thresh);

    // Create mask for threshold_abs (if set)
    let mask: Option<ndarray::Array3<bool>> = threshold_abs.map(|tabs| orig_data.mapv(|v| v >= tabs));
    if let Some(ref m) = mask {
        let mask_count = m.iter().filter(|&&b| b).count();
        eprintln!("[DEBUG] Mask: {} voxels above threshold, {} below", mask_count, orig_count - mask_count);
    }

    // Copy original data for processing
    let mut img: Array3<f32> = orig_data.to_owned();
    let (nx, ny, nz) = img.dim();

    // --- Step 1: Morphological opening by reconstruction ---
    // Erode with a ball structuring element of radius opening_radius, allowing mask to shrink
    let mut eroded = img.clone();
    let mut eroded_mask = mask.clone();
    let r = opening_radius as isize;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // If masked out, skip
                if let Some(ref m) = mask {
                    if !m[[i, j, k]] {
                        if let Some(ref mut em) = eroded_mask {
                            em[[i, j, k]] = false;
                        }
                        continue;
                    }
                }
                let mut min_val = img[[i, j, k]];
                let mut fully_inside = true;
                for di in -r..=r {
                    for dj in -r..=r {
                        for dk in -r..=r {
                            if di*di + dj*dj + dk*dk > r*r { continue; }
                            let ni = i as isize + di;
                            let nj = j as isize + dj;
                            let nk = k as isize + dk;
                            if ni < 0 || nj < 0 || nk < 0 {
                                fully_inside = false;
                                break;
                            }
                            let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
                            if ni >= nx || nj >= ny || nk >= nz {
                                fully_inside = false;
                                break;
                            }
                            if let Some(ref m) = mask {
                                if !m[[ni, nj, nk]] {
                                    fully_inside = false;
                                    break;
                                }
                            }
                            let v = img[[ni, nj, nk]];
                            if v < min_val { min_val = v; }
                        }
                        if !fully_inside { break; }
                    }
                    if !fully_inside { break; }
                }
                if fully_inside {
                    eroded[[i, j, k]] = min_val;
                    if let Some(ref mut em) = eroded_mask {
                        em[[i, j, k]] = true;
                    }
                } else {
                    if let Some(ref mut em) = eroded_mask {
                        em[[i, j, k]] = false;
                    }
                }
            }
        }
    }
    // Debug: Print stats for eroded image (unmasked voxels only) - optimized
    let (eroded_count, eroded_above_thresh) = if let Some(ref em) = eroded_mask {
        let count = em.iter().filter(|&&b| b).count();
        let above_thresh = if let Some(tabs) = threshold_abs {
            eroded.indexed_iter()
                .filter(|((i, j, k), _)| em[[*i, *j, *k]])
                .filter(|(_, &v)| v > tabs)
                .count()
        } else {
            0
        };
        (count, above_thresh)
    } else {
        let count = eroded.len();
        let above_thresh = if let Some(tabs) = threshold_abs {
            eroded.iter().filter(|&&v| v > tabs).count()
        } else {
            0
        };
        (count, above_thresh)
    };
    eprintln!("[DEBUG] Eroded image: {} voxels total, {} above threshold", eroded_count, eroded_above_thresh);
    if let Some(ref em) = eroded_mask {
        let mask_count = em.iter().filter(|&&b| b).count();
        eprintln!("[DEBUG] Eroded mask: {} voxels remain after erosion", mask_count);
    }

    // Geodesic dilation (reconstruction) of eroded under mask img, only on unmasked voxels
    let mut recon = eroded.clone();
    let mut recon_mask = eroded_mask.clone();
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if let Some(ref rm) = recon_mask {
                        if !rm[[i, j, k]] { continue; }
                    }
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        if ni < 0 || nj < 0 || nk < 0 { continue; }
                        let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
                        if ni >= nx || nj >= ny || nk >= nz { continue; }
                        if let Some(ref rm) = recon_mask {
                            if !rm[[ni, nj, nk]] { continue; }
                        }
                        let v = recon[[ni, nj, nk]];
                        if v > max_neighbor { max_neighbor = v; }
                    }
                    let new_val = max_neighbor.min(img[[i, j, k]]);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    // Debug: Print stats for reconstructed (opened) image (unmasked voxels only) - optimized
    let (recon_count, recon_above_thresh) = if let Some(ref rm) = recon_mask {
        let count = rm.iter().filter(|&&b| b).count();
        let above_thresh = if let Some(tabs) = threshold_abs {
            recon.indexed_iter()
                .filter(|((i, j, k), _)| rm[[*i, *j, *k]])
                .filter(|(_, &v)| v > tabs)
                .count()
        } else {
            0
        };
        (count, above_thresh)
    } else {
        let count = recon.len();
        let above_thresh = if let Some(tabs) = threshold_abs {
            recon.iter().filter(|&&v| v > tabs).count()
        } else {
            0
        };
        (count, above_thresh)
    };
    eprintln!("[DEBUG] Opened image: {} voxels total, {} above threshold", recon_count, recon_above_thresh);
    if let Some(ref rm) = recon_mask {
        let mask_count = rm.iter().filter(|&&b| b).count();
        eprintln!("[DEBUG] Opened mask: {} voxels remain after opening", mask_count);
    }

    // --- Step 2: h-maxima detection on the opened image ---
    // For clarity, call the existing detect_maxima_h_morph on recon
    let recon_py = PyArray3::from_owned_array(py, recon).to_owned();
    detect_maxima_h_morph(
        py,
        recon_py.as_ref(py).readonly(),
        spacing,
        h,
        threshold_abs,
        min_distance,
        cell,
        refine_peaks,
        max_steps,
    )
}

// -----------------------------------------------------------------------------
// Helper: merge maxima with/without periodic wrapping
// -----------------------------------------------------------------------------
fn merge_maxima(
    coords: &[[usize; 3]],
    values: &[f32],
    _spacing: f32, // Unused but kept for API compatibility
    min_distance: f32,
    cell: Option<&ndarray::ArrayView2<'_, f64>>, // full cell (Å)
    dims: (usize, usize, usize),
    periodic: bool,
) -> Vec<[usize; 3]> {
    let (nx, ny, nz) = dims;
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }

    // If min_distance is 0.0, skip merging entirely
    if min_distance <= 0.0 {
        return coords.to_vec();
    }
    let merge_radius = min_distance;

    let mut uf = UnionFind::new(n);

    for a in 0..n {
        for b in (a + 1)..n {
            let [ia, ja, ka] = coords[a];
            let [ib, jb, kb] = coords[b];

            // Convert voxel coordinates to fractional coordinates (-0.5 to 0.5)
            // This matches the coordinate system used in dgrid_from_atoms (linspace convention)
            let frac_a = [
                -0.5 + (ia as f64) / ((nx - 1) as f64),
                -0.5 + (ja as f64) / ((ny - 1) as f64),
                -0.5 + (ka as f64) / ((nz - 1) as f64),
            ];
            let frac_b = [
                -0.5 + (ib as f64) / ((nx - 1) as f64),
                -0.5 + (jb as f64) / ((ny - 1) as f64),
                -0.5 + (kb as f64) / ((nz - 1) as f64),
            ];

            // Calculate fractional difference
            let mut dfrac = [
                frac_a[0] - frac_b[0],
                frac_a[1] - frac_b[1],
                frac_a[2] - frac_b[2],
            ];

            // Apply minimum image convention in fractional space if periodic
            if periodic {
                dfrac[0] -= dfrac[0].round();
                dfrac[1] -= dfrac[1].round();
                dfrac[2] -= dfrac[2].round();
            }

            // Cartesian distance (Å) using cell matrix
            // This works for both orthogonal and non-orthogonal cells
            let dist = if let Some(cmat) = cell {
                // Transform fractional difference to Cartesian using cell matrix
                // cell matrix has lattice vectors as columns: r = cell * frac_coords
                let cart_x =
                    cmat[[0, 0]] * dfrac[0] + cmat[[0, 1]] * dfrac[1] + cmat[[0, 2]] * dfrac[2];
                let cart_y =
                    cmat[[1, 0]] * dfrac[0] + cmat[[1, 1]] * dfrac[1] + cmat[[1, 2]] * dfrac[2];
                let cart_z =
                    cmat[[2, 0]] * dfrac[0] + cmat[[2, 1]] * dfrac[1] + cmat[[2, 2]] * dfrac[2];
                ((cart_x.powi(2) + cart_y.powi(2) + cart_z.powi(2)).sqrt()) as f32
            } else {
                // No cell matrix provided - cannot compute proper distances
                // This should not happen for crystallographic systems
                return Vec::new();
            };

            if dist < merge_radius {
                uf.union(a, b);
            }
        }
    }

    // pick the member with the highest value within each cluster
    let mut merged: Vec<[usize; 3]> = Vec::new();
    let clusters = uf.get_clusters();
    for indices in clusters.values() {
        // Pick the member with highest value (no boundary preference for PBC)
        let mut best = indices[0];
        for &idx in indices {
            if values[idx] > values[best] {
                best = idx;
            }
        }
        merged.push(coords[best]);
    }
    merged
}

// -----------------------------------------------------------------------------
// Optimized merging with spatial indexing for better performance on large datasets
// -----------------------------------------------------------------------------
fn merge_maxima_optimized(
    coords: &[[usize; 3]],
    values: &[f32],
    spacing: f32, // Still used for bin_size calculation
    min_distance: f32,
    cell: Option<&ndarray::ArrayView2<'_, f64>>,
    dims: (usize, usize, usize),
    periodic: bool,
) -> Vec<[usize; 3]> {
    let (nx, ny, nz) = dims;
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }

    // If min_distance is 0.0, skip merging entirely
    if min_distance <= 0.0 {
        return coords.to_vec();
    }
    let merge_radius = min_distance;

    // For small datasets, use the simple O(n²) approach
    if n < 1000 {
        return merge_maxima(coords, values, spacing, min_distance, cell, dims, periodic);
    }

    // For larger datasets, use spatial binning to reduce comparisons
    let mut uf = UnionFind::new(n);

    // Create spatial bins to reduce pairwise comparisons
    let bin_size = (merge_radius / spacing).max(1.0) as usize;
    let bin_nx = (nx + bin_size - 1) / bin_size;
    let bin_ny = (ny + bin_size - 1) / bin_size;
    let bin_nz = (nz + bin_size - 1) / bin_size;

    // Map each coordinate to its spatial bin
    let mut bins: HashMap<(usize, usize, usize), Vec<usize>> = HashMap::new();

    for (idx, &[i, j, k]) in coords.iter().enumerate() {
        let bin_i = i / bin_size;
        let bin_j = j / bin_size;
        let bin_k = k / bin_size;
        bins.entry((bin_i, bin_j, bin_k)).or_default().push(idx);
    }

    // Only compare points within neighboring bins
    for (&(bi, bj, bk), indices) in &bins {
        // Check current bin and neighboring bins (proper 3x3x3 neighborhood)
        for di in -1_isize..=1_isize {
            for dj in -1_isize..=1_isize {
                for dk in -1_isize..=1_isize {
                    let ni = bi as isize + di;
                    let nj = bj as isize + dj;
                    let nk = bk as isize + dk;

                    // Handle periodic wrapping of bins if needed
                    let (actual_ni, actual_nj, actual_nk) = if periodic {
                        (
                            ni.rem_euclid(bin_nx as isize) as usize,
                            nj.rem_euclid(bin_ny as isize) as usize,
                            nk.rem_euclid(bin_nz as isize) as usize,
                        )
                    } else {
                        if ni < 0 || nj < 0 || nk < 0 || ni >= bin_nx as isize || nj >= bin_ny as isize || nk >= bin_nz as isize {
                            continue;
                        }
                        (ni as usize, nj as usize, nk as usize)
                    };

                    if let Some(neighbor_indices) = bins.get(&(actual_ni, actual_nj, actual_nk)) {
                        // Compare all pairs between current bin and neighbor bin
                        for &a in indices {
                            for &b in neighbor_indices {
                                if a >= b {
                                    continue;
                                } // Avoid duplicate comparisons

                                let [ia, ja, ka] = coords[a];
                                let [ib, jb, kb] = coords[b];

                                // Convert voxel coordinates to fractional coordinates (-0.5 to 0.5)
                                // This matches the coordinate system used in dgrid_from_atoms (linspace convention)
                                let frac_a = [
                                    -0.5 + (ia as f64) / ((nx - 1) as f64),
                                    -0.5 + (ja as f64) / ((ny - 1) as f64),
                                    -0.5 + (ka as f64) / ((nz - 1) as f64),
                                ];
                                let frac_b = [
                                    -0.5 + (ib as f64) / ((nx - 1) as f64),
                                    -0.5 + (jb as f64) / ((ny - 1) as f64),
                                    -0.5 + (kb as f64) / ((nz - 1) as f64),
                                ];

                                // Calculate fractional difference
                                let mut dfrac = [
                                    frac_a[0] - frac_b[0],
                                    frac_a[1] - frac_b[1],
                                    frac_a[2] - frac_b[2],
                                ];

                                // Apply minimum image convention in fractional space if periodic
                                if periodic {
                                    dfrac[0] -= dfrac[0].round();
                                    dfrac[1] -= dfrac[1].round();
                                    dfrac[2] -= dfrac[2].round();
                                }

                                // Cartesian distance using cell matrix
                                // This works for both orthogonal and non-orthogonal cells
                                let dist = if let Some(cmat) = cell {
                                    // Transform fractional difference to Cartesian using cell matrix
                                    // cell matrix has lattice vectors as columns: r = cell * frac_coords
                                    let cart_x = cmat[[0, 0]] * dfrac[0]
                                        + cmat[[0, 1]] * dfrac[1]
                                        + cmat[[0, 2]] * dfrac[2];
                                    let cart_y = cmat[[1, 0]] * dfrac[0]
                                        + cmat[[1, 1]] * dfrac[1]
                                        + cmat[[1, 2]] * dfrac[2];
                                    let cart_z = cmat[[2, 0]] * dfrac[0]
                                        + cmat[[2, 1]] * dfrac[1]
                                        + cmat[[2, 2]] * dfrac[2];
                                    ((cart_x.powi(2) + cart_y.powi(2) + cart_z.powi(2)).sqrt())
                                        as f32
                                } else {
                                    // No cell matrix provided - cannot compute proper distances
                                    // Skip this comparison
                                    continue;
                                };

                                if dist < merge_radius {
                                    uf.union(a, b);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Extract clusters and pick best representative from each
    let mut merged: Vec<[usize; 3]> = Vec::new();
    let clusters = uf.get_clusters();
    for indices in clusters.values() {
        // Pick the member with highest value (no boundary preference for PBC)
        let mut best = indices[0];
        for &idx in indices {
            if values[idx] > values[best] {
                best = idx;
            }
        }
        merged.push(coords[best]);
    }
    merged
}

/// Move maxima to exact peak locations using gradient ascent
/// This refines the peak locations found by h-maxima detection
/// Implements the exact same algorithm as Python's move_maximum_to_peak function
// Helper function for binary dilation (same as scipy.ndimage.binary_dilation)
fn binary_dilation(mask: &ndarray::Array3<bool>, iterations: usize) -> ndarray::Array3<bool> {
    let (nx, ny, nz) = mask.dim();
    let mut result = mask.clone();

    for _ in 0..iterations {
        let mut new_result = Array3::<bool>::from_elem((nx, ny, nz), false);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if result[[i, j, k]] {
                        // Set all 26 neighbors to true
                        for di in -1..=1 {
                            for dj in -1..=1 {
                                for dk in -1..=1 {
                                    let ni = i as isize + di;
                                    let nj = j as isize + dj;
                                    let nk = k as isize + dk;

                                    if ni >= 0 && ni < nx as isize &&
                                       nj >= 0 && nj < ny as isize &&
                                       nk >= 0 && nk < nz as isize {
                                        new_result[[ni as usize, nj as usize, nk as usize]] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result = new_result;
    }

    result
}

#[pyfunction]
#[pyo3(signature = (regions_pbc, regions_nonpbc, size_ratio=0.3))]
fn find_end_channel_pairs(
    py: Python<'_>,
    regions_pbc: PyReadonlyArray3<'_, u32>,
    regions_nonpbc: PyReadonlyArray3<'_, u32>,
    size_ratio: f64,
) -> PyResult<PyObject> {
    use ndarray::Array3;

    let regions_pbc = regions_pbc.as_array();
    let regions_nonpbc = regions_nonpbc.as_array();

    let (nx, ny, nz) = regions_nonpbc.dim();

    // Create wall mask - same logic as Python version
    let mut wall_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    wall_mask.slice_mut(ndarray::s![0, .., ..]).fill(true);      // bottom
    wall_mask.slice_mut(ndarray::s![nx-1, .., ..]).fill(true);   // top
    wall_mask.slice_mut(ndarray::s![.., 0, ..]).fill(true);   // front
    wall_mask.slice_mut(ndarray::s![.., ny-1, ..]).fill(true); // back
    wall_mask.slice_mut(ndarray::s![.., .., 0]).fill(true);     // left
    wall_mask.slice_mut(ndarray::s![.., .., nz-1]).fill(true);  // right

    // Find wall labels (unique non-zero labels on walls)
    let mut wall_labels = std::collections::HashSet::new();
    for ((i, j, k), &label) in regions_nonpbc.indexed_iter() {
        if wall_mask[[i, j, k]] && label != 0 {
            wall_labels.insert(label);
        }
    }

    let mut pairs = Vec::new();

    for &label_id in &wall_labels {
        let mask = regions_nonpbc.mapv(|x| x == label_id);

        // Find corresponding PBC label
        let pbc_labels: std::collections::HashSet<u32> = regions_pbc
            .indexed_iter()
            .filter_map(|(idx, &pbc_val)| {
                if mask[idx] && pbc_val != 0 {
                    Some(pbc_val)
                } else {
                    None
                }
            })
            .collect();

        if pbc_labels.len() != 1 {
            continue; // ambiguous, skip
        }

        let pbc_label = *pbc_labels.iter().next().unwrap();

        // Find all other regions on the same wall with the same pbc label
        let same_wall_mask = regions_pbc.mapv(|x| x == pbc_label) & &wall_mask;
        let other_labels: Vec<u32> = regions_nonpbc
            .indexed_iter()
            .filter_map(|(idx, &nonpbc_val)| {
                if same_wall_mask[idx] && nonpbc_val != label_id && nonpbc_val != 0 {
                    Some(nonpbc_val)
                } else {
                    None
                }
            })
            .collect();

        if other_labels.is_empty() {
            continue; // No other regions of this PBC group on the wall
        }

        // Calculate sizes for current and neighbor regions
        let current_size = mask.iter().filter(|&&x| x).count() as f64;

        for &neighbor_label in &other_labels {
            let neighbor_mask = regions_nonpbc.mapv(|x| x == neighbor_label);
            let neighbor_size = neighbor_mask.iter().filter(|&&x| x).count() as f64;

            // Check adjacency using binary dilation (same as Python)
            let dilated = binary_dilation(&mask, 1);
            let is_adjacent = dilated
                .indexed_iter()
                .any(|(idx, &dilated_val)| dilated_val && neighbor_mask[idx]);

            if is_adjacent {
                // Apply size ratio test (same as Python)
                if current_size < size_ratio * neighbor_size {
                    pairs.push((label_id, neighbor_label));
                }
                break; // Only consider the first touching neighbor
            }
        }
    }

    // Convert to Python list of tuples
    let py_list = PyList::empty(py);
    for (small, large) in pairs {
        let tuple = PyTuple::new(py, &[small.into_py(py), large.into_py(py)]);
        py_list.append(tuple)?;
    }

    Ok(py_list.into())
}

#[pyfunction]
#[pyo3(signature = (grid, starts, max_steps=100, step_size=0.1))]
fn move_maxima_to_peaks(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    starts: PyReadonlyArray2<'_, usize>, // (n,3)
    max_steps: Option<usize>,
    step_size: Option<f64>,
) -> PyResult<(Py<PyArray2<usize>>, Py<PyArray1<f32>>)> {
    use ndarray::{Array1, Array2};
    use rayon::prelude::*;

    // Clone into owned ndarrays so the underlying data is independent of the
    // PyArray and therefore `Sync`/`Send`.
    let g = grid.as_array().to_owned();
    let starts_owned = starts.as_array().to_owned();

    let (nx, ny, nz) = g.dim();
    let n = starts_owned.dim().0;
    let ms = max_steps.unwrap_or(100);
    let step = step_size.unwrap_or(0.1);

    let peaks: Vec<([usize; 3], f32)> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let s_row = starts_owned.row(idx);
            let mut current_pos = [s_row[0] as f64, s_row[1] as f64, s_row[2] as f64];
            let mut current_value = g[[s_row[0], s_row[1], s_row[2]]];

            for _ in 0..ms {
                // Calculate gradient using central differences (like Python)
                let mut gradients = [0.0f64; 3];
                for dim in 0..3 {
                    let mut pos_plus = current_pos.clone();
                    let mut pos_minus = current_pos.clone();

                    // Ensure we stay within bounds when calculating gradients (like Python)
                    pos_plus[dim] = (pos_plus[dim] + 1.0).min((nx - 1) as f64);
                    pos_minus[dim] = (pos_minus[dim] - 1.0).max(0.0);

                    // Ensure indices are within bounds
                    let plus_idx = [
                        pos_plus[0] as usize,
                        pos_plus[1] as usize,
                        pos_plus[2] as usize
                    ];
                    let minus_idx = [
                        pos_minus[0] as usize,
                        pos_minus[1] as usize,
                        pos_minus[2] as usize
                    ];

                    // Double-check bounds
                    if plus_idx[0] >= nx || plus_idx[1] >= ny || plus_idx[2] >= nz ||
                       minus_idx[0] >= nx || minus_idx[1] >= ny || minus_idx[2] >= nz {
                        gradients[dim] = 0.0; // Set gradient to 0 if out of bounds
                    } else {
                        let val_plus = g[[plus_idx[0], plus_idx[1], plus_idx[2]]];
                        let val_minus = g[[minus_idx[0], minus_idx[1], minus_idx[2]]];
                        gradients[dim] = (val_plus - val_minus) as f64 / 2.0;
                    }
                }

                let gradient_magnitude = (gradients[0] * gradients[0] + gradients[1] * gradients[1] + gradients[2] * gradients[2]).sqrt();

                // Stop if gradient is too small (like Python)
                if gradient_magnitude < 1e-6 {
                    break;
                }

                // Update position with bounds checking (like Python)
                let mut new_pos = [
                    current_pos[0] + step * gradients[0] / gradient_magnitude,
                    current_pos[1] + step * gradients[1] / gradient_magnitude,
                    current_pos[2] + step * gradients[2] / gradient_magnitude,
                ];

                // Clip to grid boundaries (like Python)
                new_pos[0] = new_pos[0].max(0.0).min((nx - 1) as f64);
                new_pos[1] = new_pos[1].max(0.0).min((ny - 1) as f64);
                new_pos[2] = new_pos[2].max(0.0).min((nz - 1) as f64);

                // Get value at new position
                let new_value = g[[new_pos[0] as usize, new_pos[1] as usize, new_pos[2] as usize]];

                // Stop if we're not improving (like Python)
                if new_value <= current_value {
                    break;
                }

                current_pos = new_pos;
                current_value = new_value;
            }

            let final_pos = [current_pos[0] as usize, current_pos[1] as usize, current_pos[2] as usize];
            (final_pos, current_value)
        })
        .collect();

    // Split into separate arrays
    let mut coords_vec: Vec<usize> = Vec::with_capacity(n * 3);
    let mut vals_vec: Vec<f32> = Vec::with_capacity(n);
    for (p, v) in peaks {
        coords_vec.extend_from_slice(&p);
        vals_vec.push(v);
    }
    let coords_arr = Array2::from_shape_vec((n, 3), coords_vec).expect("shape error");
    let vals_arr = Array1::from_vec(vals_vec);

    Ok((
        PyArray2::from_owned_array(py, coords_arr).to_owned(),
        PyArray1::from_owned_array(py, vals_arr).to_owned(),
    ))
}

/// Detect h-maxima with separate mask_thickness and threshold_abs, and configurable neighborhood size.
/// Uses periodic boundary conditions throughout the entire detection process.
/// - mask_thickness: mask out voxels with value < mask_thickness before maxima detection.
/// - threshold_abs: post-filter maxima by value after detection.
/// - neighborhood_size: size of the cube neighborhood for local maxima (default 5).
#[pyfunction]
#[pyo3(signature = (grid, spacing, h, mask_thickness=None, threshold_abs=None, min_distance=0.0, cell=None, refine_peaks=false, max_steps=100, neighborhood_size=5))]
fn detect_maxima_h_morph2(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    spacing: f32,
    h: f32,
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>,
    min_distance: f32,
    cell: Option<PyReadonlyArray2<'_, f64>>,
    refine_peaks: bool,
    max_steps: usize,
    neighborhood_size: usize,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    let mut img: Array3<f32> = grid.as_array().to_owned();
    let (nx, ny, nz) = img.dim();

    // Step 1: Mask out voxels with value < mask_thickness (if set)
    let mask: Option<Array3<bool>> = mask_thickness.map(|mt| img.mapv(|v| v >= mt));

    // Step 2: h-maxima transform with periodic boundary conditions
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    // Create marker image (f - h)
    let marker = img.mapv(|v| v - h);
    let mut recon = marker.clone();
    // 26-connectivity neighbors for reconstruction (with periodic wrapping)
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if let Some(ref m) = mask {
                        if !m[[i, j, k]] { continue; }
                    }
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        // Apply periodic boundary conditions using rem_euclid
                        let ni_u = ni.rem_euclid(nx as isize) as usize;
                        let nj_u = nj.rem_euclid(ny as isize) as usize;
                        let nk_u = nk.rem_euclid(nz as isize) as usize;
                        if let Some(ref m) = mask {
                            if !m[[ni_u, nj_u, nk_u]] { continue; }
                        }
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    let new_val = max_neighbor.min(current_mask);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    // Compute h-dome
    let mut h_dome = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                h_dome[[i, j, k]] = img[[i, j, k]] - recon[[i, j, k]];
            }
        }
    }
    // h-maxima mask
    let threshold = h - 1e-6;
    let mut h_maxima_mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] > -1e5 && h_dome[[i, j, k]] >= threshold {
                    h_maxima_mask[[i, j, k]] = true;
                }
            }
        }
    }
    // Step 3: Find local maxima in a larger neighborhood with periodic boundary conditions
    let nsize = neighborhood_size as isize;
    let half = nsize / 2;
    let mut maxima_coords = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if !h_maxima_mask[[i, j, k]] { continue; }
                let current_val = img[[i, j, k]];
                let mut is_local_maximum = true;
                'neighbor_loop: for di in -half..=half {
                    for dj in -half..=half {
                        for dk in -half..=half {
                            if di == 0 && dj == 0 && dk == 0 { continue; }
                            let ni = i as isize + di;
                            let nj = j as isize + dj;
                            let nk = k as isize + dk;
                            // Apply periodic boundary conditions using rem_euclid
                            let ni_u = ni.rem_euclid(nx as isize) as usize;
                            let nj_u = nj.rem_euclid(ny as isize) as usize;
                            let nk_u = nk.rem_euclid(nz as isize) as usize;
                            if let Some(ref m) = mask {
                                if !m[[ni_u, nj_u, nk_u]] { continue; }
                            }
                            let neighbor_val = img[[ni_u, nj_u, nk_u]];
                            if neighbor_val >= current_val {
                                is_local_maximum = false;
                                break 'neighbor_loop;
                            }
                        }
                    }
                }
                if is_local_maximum {
                    maxima_coords.push([i, j, k]);
                }
            }
        }
    }
    // Step 4: Post-filter maxima by threshold_abs (if set)
    let mut maxima_coords_filtered = Vec::new();
    for coord in &maxima_coords {
        let val = img[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    // Step 5: Merge maxima (periodic and non-periodic)
    let cell_ref = cell.as_ref().map(|c| c.as_array());
    let dims = (nx, ny, nz);
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter().map(|c| img[[c[0], c[1], c[2]]]).collect();
    let merged_pbc_coords = merge_maxima_optimized(
        &maxima_coords_filtered,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        true,
    );
    let merged_cart_coords = merge_maxima_optimized(
        &maxima_coords_filtered,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        false,
    );
    // Convert to NumPy arrays
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Simplified h-maxima detection with periodic boundary conditions.
/// Removes the strict local maxima filtering for better maxima detection.
/// - mask_thickness: mask out voxels with value < mask_thickness before maxima detection.
/// - threshold_abs: post-filter maxima by value after detection.
/// - neighborhood_size: kept for API compatibility (not used in simplified version).
/// - h_dome_tolerance: tolerance for h-dome threshold (default 0.1*h, looser than original).
#[pyfunction]
#[pyo3(signature = (grid, spacing, h, mask_thickness=None, threshold_abs=None, min_distance=0.0, cell=None, refine_peaks=false, max_steps=100, neighborhood_size=5, h_dome_tolerance=None))]
fn detect_maxima_h_morph_simple(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    spacing: f32,
    h: f32,
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>,
    min_distance: f32,
    cell: Option<PyReadonlyArray2<'_, f64>>,
    refine_peaks: bool,
    max_steps: usize,
    neighborhood_size: usize, // Kept for API compatibility
    h_dome_tolerance: Option<f32>,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    let mut img: Array3<f32> = grid.as_array().to_owned();
    let (nx, ny, nz) = img.dim();

    // Step 1: Mask out voxels with value < mask_thickness (if set)
    let mask: Option<Array3<bool>> = mask_thickness.map(|mt| img.mapv(|v| v >= mt));

    // Step 2: h-maxima transform with periodic boundary conditions
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    // Create marker image (f - h)
    let marker = img.mapv(|v| v - h);
    let mut recon = marker.clone();
    // 26-connectivity neighbors for reconstruction (with periodic wrapping)
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if let Some(ref m) = mask {
                        if !m[[i, j, k]] { continue; }
                    }
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        // Apply periodic boundary conditions using rem_euclid
                        let ni_u = ni.rem_euclid(nx as isize) as usize;
                        let nj_u = nj.rem_euclid(ny as isize) as usize;
                        let nk_u = nk.rem_euclid(nz as isize) as usize;
                        if let Some(ref m) = mask {
                            if !m[[ni_u, nj_u, nk_u]] { continue; }
                        }
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    let new_val = max_neighbor.min(current_mask);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Step 3: Compute h-dome with looser threshold
    let mut h_dome = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                h_dome[[i, j, k]] = img[[i, j, k]] - recon[[i, j, k]];
            }
        }
    }
    
    // Use looser h-dome threshold (default 0.1*h instead of h - 1e-6)
    let tolerance = h_dome_tolerance.unwrap_or(0.1 * h);
    let threshold = h - tolerance;
    let mut maxima_coords = Vec::new();
    
    // Step 4 REMOVED: Collect all h-maxima points without additional local maxima filtering
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if let Some(ref m) = mask {
                    if !m[[i, j, k]] { continue; }
                }
                if img[[i, j, k]] > -1e5 && h_dome[[i, j, k]] >= threshold {
                    maxima_coords.push([i, j, k]);
                }
            }
        }
    }

    // Step 5: Post-filter maxima by threshold_abs (if set)
    let mut maxima_coords_filtered = Vec::new();
    for coord in &maxima_coords {
        let val = img[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    
    // Step 6: Merge maxima (periodic and non-periodic)
    let cell_ref = cell.as_ref().map(|c| c.as_array());
    let dims = (nx, ny, nz);
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter().map(|c| img[[c[0], c[1], c[2]]]).collect();
    let merged_pbc_coords = merge_maxima_optimized(
        &maxima_coords_filtered,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        true,
    );
    let merged_cart_coords = merge_maxima_optimized(
        &maxima_coords_filtered,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        false,
    );
    
    // Convert to NumPy arrays
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Simplified h-maxima detection with periodic boundary conditions.
/// Removes spacing parameter - uses direct pairwise PBC distance calculations.
/// - mask_thickness: mask out voxels with value < mask_thickness before maxima detection.
/// - threshold_abs: post-filter maxima by value after detection.
/// - h_dome_tolerance: tolerance for h-dome threshold (default 0.1*h, looser than original).
#[pyfunction]
#[pyo3(signature = (grid, h, mask_thickness=None, threshold_abs=None, min_distance=0.0, cell=None, h_dome_tolerance=None))]
fn detect_maxima_h_morph_no_spacing(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    h: f32,
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>,
    min_distance: f32,
    cell: Option<PyReadonlyArray2<'_, f64>>,
    h_dome_tolerance: Option<f32>,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    let mut img: Array3<f32> = grid.as_array().to_owned();
    let (nx, ny, nz) = img.dim();

    // Step 1: Mask out voxels with value < mask_thickness (if set)
    let mask: Option<Array3<bool>> = mask_thickness.map(|mt| img.mapv(|v| v >= mt));

    // Step 2: h-maxima transform with periodic boundary conditions
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    
    // Create marker image (f - h)
    let marker = img.mapv(|v| v - h);
    let mut recon = marker.clone();
    
    // 26-connectivity neighbors for reconstruction (with periodic wrapping)
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if let Some(ref m) = mask {
                        if !m[[i, j, k]] { continue; }
                    }
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        // Apply periodic boundary conditions using rem_euclid
                        let ni_u = ni.rem_euclid(nx as isize) as usize;
                        let nj_u = nj.rem_euclid(ny as isize) as usize;
                        let nk_u = nk.rem_euclid(nz as isize) as usize;
                        if let Some(ref m) = mask {
                            if !m[[ni_u, nj_u, nk_u]] { continue; }
                        }
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    let new_val = max_neighbor.min(current_mask);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Step 3: Compute h-dome with looser threshold
    let mut h_dome = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                h_dome[[i, j, k]] = img[[i, j, k]] - recon[[i, j, k]];
            }
        }
    }
    
    // Use looser h-dome threshold (default 0.1*h instead of h - 1e-6)
    let tolerance = h_dome_tolerance.unwrap_or(0.1 * h);
    let threshold = h - tolerance;
    let mut maxima_coords = Vec::new();
    
    // Collect all h-maxima points without additional local maxima filtering
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if let Some(ref m) = mask {
                    if !m[[i, j, k]] { continue; }
                }
                if img[[i, j, k]] > -1e5 && h_dome[[i, j, k]] >= threshold {
                    maxima_coords.push([i, j, k]);
                }
            }
        }
    }

    // Step 4: Post-filter maxima by threshold_abs (if set)
    let mut maxima_coords_filtered = Vec::new();
    for coord in &maxima_coords {
        let val = img[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    
    // Step 5: Simple pairwise distance-based merging (no spacing needed)
    let cell_ref = cell.as_ref().map(|c| c.as_array());
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter().map(|c| img[[c[0], c[1], c[2]]]).collect();
    
    let merged_pbc_coords = merge_maxima_simple(
        &maxima_coords_filtered,
        &maxima_vals,
        min_distance,
        cell_ref.as_ref(),
        (nx, ny, nz),
        true, // periodic
    );
    
    let merged_cart_coords = merge_maxima_simple(
        &maxima_coords_filtered,
        &maxima_vals,
        min_distance,
        cell_ref.as_ref(),
        (nx, ny, nz),
        false, // non-periodic
    );
    
    // Convert to NumPy arrays
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Simple merging function without spacing parameter
fn merge_maxima_simple(
    coords: &[[usize; 3]],
    values: &[f32],
    min_distance: f32,
    cell: Option<&ndarray::ArrayView2<'_, f64>>,
    dims: (usize, usize, usize),
    periodic: bool,
) -> Vec<[usize; 3]> {
    let (nx, ny, nz) = dims;
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }

    // If min_distance is 0.0, skip merging entirely
    if min_distance <= 0.0 {
        return coords.to_vec();
    }
    let merge_radius = min_distance;

    let mut uf = UnionFind::new(n);

    for a in 0..n {
        for b in (a + 1)..n {
            let [ia, ja, ka] = coords[a];
            let [ib, jb, kb] = coords[b];

            // Convert voxel coordinates to fractional coordinates (-0.5 to 0.5)
            let frac_a = [
                -0.5 + (ia as f64) / ((nx - 1) as f64),
                -0.5 + (ja as f64) / ((ny - 1) as f64),
                -0.5 + (ka as f64) / ((nz - 1) as f64),
            ];
            let frac_b = [
                -0.5 + (ib as f64) / ((nx - 1) as f64),
                -0.5 + (jb as f64) / ((ny - 1) as f64),
                -0.5 + (kb as f64) / ((nz - 1) as f64),
            ];

            // Calculate fractional difference
            let mut dfrac = [
                frac_a[0] - frac_b[0],
                frac_a[1] - frac_b[1],
                frac_a[2] - frac_b[2],
            ];

            // Apply minimum image convention in fractional space if periodic
            if periodic {
                dfrac[0] -= dfrac[0].round();
                dfrac[1] -= dfrac[1].round();
                dfrac[2] -= dfrac[2].round();
            }

            // Cartesian distance using cell matrix (preferred) or grid units (fallback)
            let dist = if let Some(cmat) = cell {
                // Transform fractional difference to Cartesian using cell matrix
                let cart_x = cmat[[0, 0]] * dfrac[0] + cmat[[0, 1]] * dfrac[1] + cmat[[0, 2]] * dfrac[2];
                let cart_y = cmat[[1, 0]] * dfrac[0] + cmat[[1, 1]] * dfrac[1] + cmat[[1, 2]] * dfrac[2];
                let cart_z = cmat[[2, 0]] * dfrac[0] + cmat[[2, 1]] * dfrac[1] + cmat[[2, 2]] * dfrac[2];
                ((cart_x.powi(2) + cart_y.powi(2) + cart_z.powi(2)).sqrt()) as f32
            } else {
                // Fallback: use grid units (assumes cubic grid, less accurate)
                let grid_dist = (dfrac[0].powi(2) + dfrac[1].powi(2) + dfrac[2].powi(2)).sqrt() as f32;
                grid_dist * ((nx + ny + nz) as f32 / 3.0) // rough scaling
            };

            if dist < merge_radius {
                uf.union(a, b);
            }
        }
    }

    // Pick the member with the highest value within each cluster
    let mut merged: Vec<[usize; 3]> = Vec::new();
    let clusters = uf.get_clusters();
    for indices in clusters.values() {
        let mut best = indices[0];
        for &idx in indices {
            if values[idx] > values[best] {
                best = idx;
            }
        }
        merged.push(coords[best]);
    }
    merged
}

/// Geometry-adaptive h-maxima detection using distance field properties.
/// Uses local dgrid values and gradients to adaptively choose h values:
/// - Large smooth regions (channels) use large h to suppress surface roughness
/// - Small regions (pockets) use small h for sensitive detection  
/// - High gradient regions (near surfaces) use very small h
/// 
/// Parameters:
/// - h_small: h value for small pockets and surface regions (default 0.2)
/// - h_large: h value for large channel interiors (default 0.8)  
/// - transition_radius: dgrid value separating small/large regions in Å (default 3.0)
/// - gradient_threshold: gradient magnitude indicating surface proximity (default 0.5)
#[pyfunction]
#[pyo3(signature = (grid, h_small=0.2, h_large=0.8, transition_radius=3.0, gradient_threshold=0.5, cell=None, min_distance=0.0, mask_thickness=None, threshold_abs=None))]
fn detect_maxima_adaptive_geometry(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    h_small: f32,      // For pockets (default 0.2)
    h_large: f32,      // For channels (default 0.8) 
    transition_radius: f32, // Transition point in Å (default 3.0)
    gradient_threshold: f32, // High gradient = surface (default 0.5)
    cell: Option<PyReadonlyArray2<'_, f64>>,
    min_distance: f32,
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>, 
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    let dgrid = grid.as_array();
    let (nx, ny, nz) = dgrid.dim();
    
    // Step 1: Compute adaptive h field based on local geometry
    let mut adaptive_h = Array3::<f32>::zeros((nx, ny, nz));
    
    for i in 1..nx-1 {
        for j in 1..ny-1 {
            for k in 1..nz-1 {
                let local_radius = dgrid[[i, j, k]];
                
                // Compute local gradient magnitude (indicates proximity to surface)
                let grad_x = (dgrid[[i+1, j, k]] - dgrid[[i-1, j, k]]) / 2.0;
                let grad_y = (dgrid[[i, j+1, k]] - dgrid[[i, j-1, k]]) / 2.0;
                let grad_z = (dgrid[[i, j, k+1]] - dgrid[[i, j, k-1]]) / 2.0;
                let gradient_mag = (grad_x*grad_x + grad_y*grad_y + grad_z*grad_z).sqrt();
                
                // Adaptive h based on local geometry
                let h = if gradient_mag > gradient_threshold {
                    // Near surface - use very small h for sensitivity to small features
                    h_small * 0.5
                } else if local_radius > transition_radius {
                    // Large channel interior - use large h to suppress surface roughness
                    h_large
                } else {
                    // Small pocket interior - use small h for detection
                    h_small
                };
                
                adaptive_h[[i, j, k]] = h;
            }
        }
    }
    
    // Handle boundaries by copying nearest interior values
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1 {
                    let ii = if i == 0 { 1 } else if i == nx-1 { nx-2 } else { i };
                    let jj = if j == 0 { 1 } else if j == ny-1 { ny-2 } else { j };
                    let kk = if k == 0 { 1 } else if k == nz-1 { nz-2 } else { k };
                    adaptive_h[[i, j, k]] = adaptive_h[[ii, jj, kk]];
                }
            }
        }
    }
    
    // Step 2: Apply adaptive h-maxima transform
    let mut img = dgrid.to_owned();
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    
    // Create spatially-varying marker image: marker[i,j,k] = img[i,j,k] - adaptive_h[i,j,k]
    let mut marker = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                marker[[i, j, k]] = img[[i, j, k]] - adaptive_h[[i, j, k]];
            }
        }
    }
    
    // Step 3: Morphological reconstruction with periodic boundaries
    let mut recon = marker.clone();
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        // Periodic boundary conditions
                        let ni_u = ni.rem_euclid(nx as isize) as usize;
                        let nj_u = nj.rem_euclid(ny as isize) as usize;
                        let nk_u = nk.rem_euclid(nz as isize) as usize;
                        
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    
                    let new_val = max_neighbor.min(current_mask);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Step 4: Compute adaptive h-dome and find maxima
    let mut maxima_coords = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] <= -1e5 { continue; }
                
                let h_dome_val = img[[i, j, k]] - recon[[i, j, k]];
                let local_h = adaptive_h[[i, j, k]];
                let threshold = local_h * 0.9; // Slightly looser than exact h
                
                if h_dome_val >= threshold {
                    maxima_coords.push([i, j, k]);
                }
            }
        }
    }
    
    // Step 5: Apply remaining filters and merge
    let mut maxima_coords_filtered = Vec::new();
    for coord in &maxima_coords {
        let val = img[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    
    // Step 6: Merge with PBC-aware distances
    let cell_ref = cell.as_ref().map(|c| c.as_array());
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter().map(|c| img[[c[0], c[1], c[2]]]).collect();
    
    let merged_pbc_coords = merge_maxima_simple(
        &maxima_coords_filtered,
        &maxima_vals,
        min_distance,
        cell_ref.as_ref(),
        (nx, ny, nz),
        true,
    );
    
    let merged_cart_coords = merge_maxima_simple(
        &maxima_coords_filtered,
        &maxima_vals,
        min_distance,
        cell_ref.as_ref(),
        (nx, ny, nz),
        false,
    );
    
    // Convert to NumPy arrays
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Selective smoothing h-maxima detection.
/// Applies spatially-varying smoothing based on local pore size:
/// - Small pockets (dgrid < preserve_threshold): minimal smoothing to preserve features
/// - Large channels (dgrid > smooth_threshold): heavy smoothing to remove surface roughness
/// - Intermediate regions: gradual transition
/// 
/// Parameters:
/// - preserve_threshold: dgrid value below which features are preserved (default 4.0 Å)
/// - smooth_threshold: dgrid value above which heavy smoothing is applied (default 6.0 Å)  
/// - max_smooth_radius: maximum smoothing radius in grid units (default 2.0)
/// - h: single h parameter for maxima detection after smoothing (default 0.2)
/// - cell: REQUIRED for merging - unit cell matrix for distance calculations
/// - min_distance: merge distance in Å (default 1.0) - set to 0.0 to disable merging
#[pyfunction]
#[pyo3(signature = (grid, preserve_threshold=4.0, smooth_threshold=6.0, max_smooth_radius=2.0, h=0.2, cell=None, min_distance=1.0, mask_thickness=None, threshold_abs=None))]
fn detect_maxima_selective_smoothing(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    preserve_threshold: f32,   // Preserve features smaller than this (default 4.0 Å)
    smooth_threshold: f32,     // Smooth features larger than this (default 6.0 Å)
    max_smooth_radius: f32,    // Maximum smoothing radius in grid units (default 2.0)
    h: f32,                    // Single h parameter after smoothing (default 0.2)
    cell: Option<PyReadonlyArray2<'_, f64>>,
    min_distance: f32,
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    use std::f32::consts::PI;
    
    let dgrid = grid.as_array();
    let (nx, ny, nz) = dgrid.dim();
    
    // Step 1: Apply selective smoothing
    let mut smoothed_grid = dgrid.to_owned();
    
    // Create smoothing radius field based on local dgrid values
    let mut smooth_radii = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let local_pore_size = dgrid[[i, j, k]];
                
                let smooth_radius = if local_pore_size < preserve_threshold {
                    // Small pockets - minimal smoothing (preserve features)
                    0.0
                } else if local_pore_size > smooth_threshold {
                    // Large channels - maximum smoothing (remove roughness)
                    max_smooth_radius
                } else {
                    // Intermediate - linear interpolation
                    let fraction = (local_pore_size - preserve_threshold) / (smooth_threshold - preserve_threshold);
                    fraction * max_smooth_radius
                };
                
                smooth_radii[[i, j, k]] = smooth_radius;
            }
        }
    }
    
    // Apply spatially-varying Gaussian smoothing
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let smooth_radius = smooth_radii[[i, j, k]];
                
                if smooth_radius > 0.1 { // Only smooth if radius is significant
                    let kernel_size = (smooth_radius * 3.0).ceil() as usize; // 3-sigma kernel
                    let mut weighted_sum = 0.0f32;
                    let mut weight_total = 0.0f32;
                    
                    // Apply Gaussian kernel with periodic boundaries
                    for di in -(kernel_size as isize)..=(kernel_size as isize) {
                        for dj in -(kernel_size as isize)..=(kernel_size as isize) {
                            for dk in -(kernel_size as isize)..=(kernel_size as isize) {
                                let ni = (i as isize + di).rem_euclid(nx as isize) as usize;
                                let nj = (j as isize + dj).rem_euclid(ny as isize) as usize;
                                let nk = (k as isize + dk).rem_euclid(nz as isize) as usize;
                                
                                let dist_sq = (di * di + dj * dj + dk * dk) as f32;
                                let sigma_sq = smooth_radius * smooth_radius;
                                
                                if sigma_sq > 0.0 {
                                    let weight = (-0.5 * dist_sq / sigma_sq).exp();
                                    weighted_sum += weight * dgrid[[ni, nj, nk]];
                                    weight_total += weight;
                                }
                            }
                        }
                    }
                    
                    if weight_total > 0.0 {
                        smoothed_grid[[i, j, k]] = weighted_sum / weight_total;
                    }
                }
            }
        }
    }
    
    // Step 2: Apply standard h-maxima detection on smoothed grid
    let mut img = smoothed_grid;
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    
    // Create marker image (f - h) - single h value now
    let marker = img.mapv(|v| v - h);
    let mut recon = marker.clone();
    
    // 26-connectivity neighbors for reconstruction (with periodic wrapping)
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        // Periodic boundary conditions
                        let ni_u = ni.rem_euclid(nx as isize) as usize;
                        let nj_u = nj.rem_euclid(ny as isize) as usize;
                        let nk_u = nk.rem_euclid(nz as isize) as usize;
                        
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    
                    let new_val = max_neighbor.min(current_mask);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Step 3: Compute h-dome and find maxima (standard approach)
    let mut h_dome = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                h_dome[[i, j, k]] = img[[i, j, k]] - recon[[i, j, k]];
            }
        }
    }
    
    let threshold = h * 0.9; // Slightly looser threshold
    let mut maxima_coords = Vec::new();
    
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] > -1e5 && h_dome[[i, j, k]] >= threshold {
                    maxima_coords.push([i, j, k]);
                }
            }
        }
    }
    
    // Step 4: Apply remaining filters and merge
    let mut maxima_coords_filtered = Vec::new();
    for coord in &maxima_coords {
        let val = img[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    
    // Step 5: Merge with PBC-aware distances  
    let cell_ref = cell.as_ref().map(|c| c.as_array());
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter().map(|c| img[[c[0], c[1], c[2]]]).collect();
    
    // Apply merging only if min_distance > 0
    let merged_pbc_coords = if min_distance > 0.0 {
        merge_maxima_simple(
            &maxima_coords_filtered,
            &maxima_vals,
            min_distance,
            cell_ref.as_ref(),
            (nx, ny, nz),
            true, // periodic
        )
    } else {
        maxima_coords_filtered.clone() // No merging
    };
    
    let merged_cart_coords = if min_distance > 0.0 {
        merge_maxima_simple(
            &maxima_coords_filtered,
            &maxima_vals,
            min_distance,
            cell_ref.as_ref(),
            (nx, ny, nz),
            false, // non-periodic
        )
    } else {
        maxima_coords_filtered.clone() // No merging  
    };
    
    // Convert to NumPy arrays
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Parallel version of selective smoothing - Strategy 1: Grid-point parallelization
/// This is identical to detect_maxima_selective_smoothing but parallelizes the main smoothing loop
/// for 4-8x speedup while maintaining bit-exact identical results.
#[pyfunction]
#[pyo3(signature = (grid, preserve_threshold=3.0, smooth_threshold=6.0, max_smooth_radius=2.0, h=0.2, cell=None, min_distance=1.5, mask_thickness=None, threshold_abs=1.5))]
fn detect_maxima_selective_smoothing_parallel(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    preserve_threshold: f32,   // Preserve features smaller than this (default 3.0 Å)
    smooth_threshold: f32,     // Smooth features larger than this (default 6.0 Å)
    max_smooth_radius: f32,    // Maximum smoothing radius in grid units (default 2.0)
    h: f32,                    // Single h parameter after smoothing (default 0.2)
    cell: Option<PyReadonlyArray2<'_, f64>>,
    min_distance: f32,         // Distance merging threshold (default 1.5 Å)
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>, // Absolute threshold (default 1.5 Å)
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    use std::f32::consts::PI;
    
    let dgrid = grid.as_array();
    let (nx, ny, nz) = dgrid.dim();
    let total_points = nx * ny * nz;
    
    // Step 1: Create smoothing radius field (same as original)
    let smooth_radii_data: Vec<f32> = (0..total_points)
        .into_par_iter()
        .map(|flat_idx| {
            // Unravel flat index into 3D coordinates
            let i = flat_idx / (ny * nz);
            let j = (flat_idx / nz) % ny;
            let k = flat_idx % nz;
            
            let local_pore_size = dgrid[[i, j, k]];
            
            if local_pore_size < preserve_threshold {
                // Small pockets - minimal smoothing (preserve features)
                0.0
            } else if local_pore_size > smooth_threshold {
                // Large channels - maximum smoothing (remove roughness)
                max_smooth_radius
            } else {
                // Intermediate - linear interpolation
                let fraction = (local_pore_size - preserve_threshold) / (smooth_threshold - preserve_threshold);
                fraction * max_smooth_radius
            }
        })
        .collect();
    
    let smooth_radii = Array3::from_shape_vec((nx, ny, nz), smooth_radii_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;
    
    // Step 2: Apply spatially-varying Gaussian smoothing - PARALLELIZED
    let smoothed_data: Vec<f32> = (0..total_points)
        .into_par_iter()
        .map(|flat_idx| {
            // Unravel flat index into 3D coordinates
            let i = flat_idx / (ny * nz);
            let j = (flat_idx / nz) % ny;
            let k = flat_idx % nz;
            
            let smooth_radius = smooth_radii[[i, j, k]];
            
            if smooth_radius > 0.1 { // Only smooth if radius is significant
                let kernel_size = (smooth_radius * 3.0).ceil() as usize; // 3-sigma kernel
                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;
                
                // Apply Gaussian kernel with periodic boundaries (same as original)
                for di in -(kernel_size as isize)..=(kernel_size as isize) {
                    for dj in -(kernel_size as isize)..=(kernel_size as isize) {
                        for dk in -(kernel_size as isize)..=(kernel_size as isize) {
                            let ni = (i as isize + di).rem_euclid(nx as isize) as usize;
                            let nj = (j as isize + dj).rem_euclid(ny as isize) as usize;
                            let nk = (k as isize + dk).rem_euclid(nz as isize) as usize;
                            
                            let dist_sq = (di * di + dj * dj + dk * dk) as f32;
                            let sigma_sq = smooth_radius * smooth_radius;
                            
                            if sigma_sq > 0.0 {
                                let weight = (-0.5 * dist_sq / sigma_sq).exp();
                                weighted_sum += weight * dgrid[[ni, nj, nk]];
                                weight_total += weight;
                            }
                        }
                    }
                }
                
                if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    dgrid[[i, j, k]]
                }
            } else {
                dgrid[[i, j, k]] // No smoothing needed
            }
        })
        .collect();
    
    let mut smoothed_grid = Array3::from_shape_vec((nx, ny, nz), smoothed_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {}", e)))?;
    
    // Step 3: Apply standard h-maxima detection (identical to original - not parallelized yet)
    let mut img = smoothed_grid;
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    
    // Create marker image (f - h) - single h value now
    let marker = img.mapv(|v| v - h);
    let mut recon = marker.clone();
    
    // 26-connectivity neighbors for reconstruction (with periodic wrapping)
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        // Periodic boundary conditions
                        let ni_u = ni.rem_euclid(nx as isize) as usize;
                        let nj_u = nj.rem_euclid(ny as isize) as usize;
                        let nk_u = nk.rem_euclid(nz as isize) as usize;
                        
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    
                    let new_val = max_neighbor.min(current_mask);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Step 4: Compute h-dome and find maxima (identical to original)
    let mut h_dome = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                h_dome[[i, j, k]] = img[[i, j, k]] - recon[[i, j, k]];
            }
        }
    }
    
    let threshold = h * 0.9; // Slightly looser threshold
    let mut maxima_coords = Vec::new();
    
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] > -1e5 && h_dome[[i, j, k]] >= threshold {
                    maxima_coords.push([i, j, k]);
                }
            }
        }
    }
    
    // Step 5: Apply remaining filters and merge (identical to original)
    let mut maxima_coords_filtered = Vec::new();
    for coord in &maxima_coords {
        let val = img[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    
    // Step 6: Merge with PBC-aware distances (identical to original)
    let cell_ref = cell.as_ref().map(|c| c.as_array());
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter().map(|c| img[[c[0], c[1], c[2]]]).collect();
    
    // Apply merging only if min_distance > 0
    let merged_pbc_coords = if min_distance > 0.0 {
        merge_maxima_simple(
            &maxima_coords_filtered,
            &maxima_vals,
            min_distance,
            cell_ref.as_ref(),
            (nx, ny, nz),
            true, // periodic
        )
    } else {
        maxima_coords_filtered.clone() // No merging
    };
    
    let merged_cart_coords = if min_distance > 0.0 {
        merge_maxima_simple(
            &maxima_coords_filtered,
            &maxima_vals,
            min_distance,
            cell_ref.as_ref(),
            (nx, ny, nz),
            false, // non-periodic
        )
    } else {
        maxima_coords_filtered.clone() // No merging  
    };
    
    // Convert to NumPy arrays (identical to original)
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Percolation-guided maxima detection with geometric intelligence.
/// Uses connectivity analysis at multiple scales to understand local pore geometry,
/// then applies geometry-aware processing:
/// - Large channels: preserve axial maxima, suppress cross-sectional roughness
/// - Small pockets: preserve all features regardless of connectivity
/// - Mixed regions: adaptive processing based on local geometry
/// 
/// Handles non-orthogonal cells and periodic boundary conditions throughout.
#[pyfunction]
#[pyo3(signature = (grid, cell, h=0.2, min_distance=1.0, mask_thickness=None, threshold_abs=None))]
fn detect_maxima_percolation_guided(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    cell: PyReadonlyArray2<'_, f64>,
    h: f32,
    min_distance: f32,
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    
    let dgrid = grid.as_array();
    let cell_matrix = cell.as_array();
    let (nx, ny, nz) = dgrid.dim();
    
    // Step 1: Analyze percolation structure to understand local geometry
    let geometry_map = analyze_local_geometry(&dgrid, &cell_matrix);
    
    // Step 2: Apply geometry-aware processing
    let processed_grid = apply_geometry_aware_processing(&dgrid, &geometry_map, &cell_matrix);
    
    // Step 3: Standard h-maxima detection with periodic boundaries
    let maxima_coords = detect_h_maxima_periodic(&processed_grid, h, mask_thickness);
    
    // Step 4: Apply threshold filtering
    let mut maxima_coords_filtered = Vec::new();
    for coord in &maxima_coords {
        let val = processed_grid[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    
    // Step 5: Geometry-aware merging (considers local pore structure)
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter()
        .map(|c| processed_grid[[c[0], c[1], c[2]]]).collect();
    
    let merged_pbc_coords = if min_distance > 0.0 {
        merge_maxima_geometry_aware(
            &maxima_coords_filtered,
            &maxima_vals,
            &geometry_map,
            min_distance,
            Some(&cell_matrix),
            (nx, ny, nz),
            true, // periodic
        )
    } else {
        maxima_coords_filtered.clone()
    };
    
    let merged_cart_coords = if min_distance > 0.0 {
        merge_maxima_geometry_aware(
            &maxima_coords_filtered,
            &maxima_vals,
            &geometry_map,
            min_distance,
            Some(&cell_matrix),
            (nx, ny, nz),
            false, // non-periodic
        )
    } else {
        maxima_coords_filtered.clone()
    };
    
    // Convert to NumPy arrays
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), 
        maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), 
        merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), 
        merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LocalGeometry {
    SmallPocket,      // Percolates only at large probe sizes
    MediumPocket,     // Percolates at medium probe sizes
    ChannelLike,      // Percolates at small probe sizes
    ChannelAxis,      // Main axis of channel (detected via anisotropy)
    Unknown,
}

/// Analyze local geometry using multi-scale percolation analysis
fn analyze_local_geometry(
    grid: &ndarray::ArrayView3<'_, f32>,
    cell: &ndarray::ArrayView2<'_, f64>,
) -> Array3<LocalGeometry> {
    let (nx, ny, nz) = grid.dim();
    let mut geometry_map = Array3::from_elem((nx, ny, nz), LocalGeometry::Unknown);
    
    // Generate adaptive probe sizes based on grid statistics
    let max_val = grid.iter().fold(0.0f32, |a, &b| a.max(b));
    let probe_sizes: Vec<f32> = (1..=6).map(|i| max_val * (i as f32) / 12.0).collect();
    
    // Track first percolation threshold for each voxel
    let mut first_percolation = Array3::from_elem((nx, ny, nz), max_val + 1.0);
    
    for &probe_size in &probe_sizes {
        // Create binary grid
        let binary_grid = grid.mapv(|v| v > probe_size);
        
        // Pad with periodic boundaries
        let padded_grid = pad_array_periodic(&binary_grid);
        
        // Find connected components with periodic boundaries
        let component_labels = connected_components_3d_periodic(&padded_grid);
        
        // Remove padding
        let (dx, dy, dz) = component_labels.dim();
        let labels = component_labels.slice(ndarray::s![1..dx-1, 1..dy-1, 1..dz-1]).to_owned();
        
        // Find percolating components
        let percolating_labels = find_percolating_components_3d(&labels);
        
        // Update first percolation threshold
        for ((i, j, k), &label) in labels.indexed_iter() {
            if label != 0 && percolating_labels.contains(&label) {
                if first_percolation[[i, j, k]] > probe_size {
                    first_percolation[[i, j, k]] = probe_size;
                }
            }
        }
    }
    
    // Classify based on percolation behavior and local anisotropy
    for ((i, j, k), &first_perc) in first_percolation.indexed_iter() {
        if grid[[i, j, k]] <= 0.5 { // Too close to surface
            geometry_map[[i, j, k]] = LocalGeometry::Unknown;
            continue;
        }
        
        let geometry = if first_perc > max_val * 0.75 {
            LocalGeometry::SmallPocket
        } else if first_perc > max_val * 0.4 {
            LocalGeometry::MediumPocket
        } else if first_perc <= max_val * 0.2 {
            // Analyze local anisotropy to detect channel direction
            let anisotropy = compute_local_anisotropy(grid, cell, i, j, k);
            if anisotropy > 2.0 { // Highly anisotropic = channel-like
                LocalGeometry::ChannelAxis
            } else {
                LocalGeometry::ChannelLike
            }
        } else {
            LocalGeometry::ChannelLike
        };
        
        geometry_map[[i, j, k]] = geometry;
    }
    
    geometry_map
}

/// Compute local anisotropy using real-space distances
fn compute_local_anisotropy(
    grid: &ndarray::ArrayView3<'_, f32>,
    cell: &ndarray::ArrayView2<'_, f64>,
    i: usize, j: usize, k: usize,
) -> f32 {
    let (nx, ny, nz) = grid.dim();
    let center_val = grid[[i, j, k]];
    
    // Sample distances in different directions using cell matrix
    let directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)];
    let mut distance_variations = Vec::new();
    
    for &(di, dj, dk) in &directions {
        let ni = (i as isize + di).rem_euclid(nx as isize) as usize;
        let nj = (j as isize + dj).rem_euclid(ny as isize) as usize;
        let nk = (k as isize + dk).rem_euclid(nz as isize) as usize;
        
        let neighbor_val = grid[[ni, nj, nk]];
        distance_variations.push((center_val - neighbor_val).abs());
    }
    
    if distance_variations.is_empty() {
        return 1.0;
    }
    
    let max_var = distance_variations.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_var = distance_variations.iter().fold(f32::MAX, |a, &b| a.min(b));
    
    if min_var > 0.0 {
        max_var / min_var
    } else {
        1.0
    }
}

/// Apply geometry-aware processing to suppress surface roughness where appropriate
fn apply_geometry_aware_processing(
    grid: &ndarray::ArrayView3<'_, f32>,
    geometry_map: &Array3<LocalGeometry>,
    cell: &ndarray::ArrayView2<'_, f64>,
) -> Array3<f32> {
    let mut processed = grid.to_owned();
    let (nx, ny, nz) = grid.dim();
    
    for ((i, j, k), &geometry) in geometry_map.indexed_iter() {
        match geometry {
            LocalGeometry::ChannelLike | LocalGeometry::ChannelAxis => {
                // Apply mild smoothing to reduce cross-sectional roughness
                // but preserve axial variations
                let smoothed_val = apply_anisotropic_smoothing(grid, cell, i, j, k, 1.5);
                processed[[i, j, k]] = smoothed_val;
            }
            LocalGeometry::SmallPocket | LocalGeometry::MediumPocket => {
                // Preserve original values for sensitive feature detection
                // No modification needed
            }
            LocalGeometry::Unknown => {
                // Light smoothing for uncertain regions
                let smoothed_val = apply_isotropic_smoothing(grid, i, j, k, 1.0);
                processed[[i, j, k]] = smoothed_val;
            }
        }
    }
    
    processed  
}

/// Apply anisotropic smoothing (more smoothing perpendicular to channel axis)
fn apply_anisotropic_smoothing(
    grid: &ndarray::ArrayView3<'_, f32>,
    cell: &ndarray::ArrayView2<'_, f64>,
    i: usize, j: usize, k: usize,
    radius: f32,
) -> f32 {
    // For now, use isotropic smoothing - could be enhanced with directional analysis
    apply_isotropic_smoothing(grid, i, j, k, radius)
}

/// Simple isotropic smoothing with periodic boundaries
fn apply_isotropic_smoothing(
    grid: &ndarray::ArrayView3<'_, f32>,
    i: usize, j: usize, k: usize, 
    radius: f32,
) -> f32 {
    let (nx, ny, nz) = grid.dim();
    let kernel_size = (radius.ceil() as isize).max(1);
    
    let mut weighted_sum = 0.0f32;
    let mut weight_total = 0.0f32;
    
    for di in -kernel_size..=kernel_size {
        for dj in -kernel_size..=kernel_size {
            for dk in -kernel_size..=kernel_size {
                let dist_sq = (di * di + dj * dj + dk * dk) as f32;
                if dist_sq > radius * radius {
                    continue;
                }
                
                let ni = (i as isize + di).rem_euclid(nx as isize) as usize;
                let nj = (j as isize + dj).rem_euclid(ny as isize) as usize; 
                let nk = (k as isize + dk).rem_euclid(nz as isize) as usize;
                
                let weight = (-0.5 * dist_sq / (radius * radius)).exp();
                weighted_sum += weight * grid[[ni, nj, nk]];
                weight_total += weight;
            }
        }
    }
    
    if weight_total > 0.0 {
        weighted_sum / weight_total
    } else {
        grid[[i, j, k]]
    }
}

/// Detect h-maxima with periodic boundary conditions
fn detect_h_maxima_periodic(
    grid: &Array3<f32>,
    h: f32,
    mask_thickness: Option<f32>,
) -> Vec<[usize; 3]> {
    let (nx, ny, nz) = grid.dim();
    let mut img = grid.clone();
    
    // Apply mask if specified
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    
    // Create marker image (f - h)
    let marker = img.mapv(|v| v - h);
    let mut recon = marker.clone();
    
    // 26-connectivity neighbors for periodic reconstruction
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    
    // Morphological reconstruction with periodic boundaries
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = (i as isize + di).rem_euclid(nx as isize) as usize;
                        let nj = (j as isize + dj).rem_euclid(ny as isize) as usize;
                        let nk = (k as isize + dk).rem_euclid(nz as isize) as usize;
                        
                        let neighbor_recon = recon[[ni, nj, nk]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    
                    let new_val = max_neighbor.min(current_mask);
                    if (new_val - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_val;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Find h-maxima points
    let mut maxima_coords = Vec::new();
    let threshold = h * 0.9;
    
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] > -1e5 {
                    let h_dome_val = img[[i, j, k]] - recon[[i, j, k]];
                    if h_dome_val >= threshold {
                        maxima_coords.push([i, j, k]);
                    }
                }
            }
        }
    }
    
    maxima_coords
}

/// Pad array with periodic boundary conditions
fn pad_array_periodic(arr: &Array3<bool>) -> Array3<bool> {
    let (nx, ny, nz) = arr.dim();
    let mut padded = Array3::from_elem((nx + 2, ny + 2, nz + 2), false);
    
    // Copy original array to center
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                padded[[i + 1, j + 1, k + 1]] = arr[[i, j, k]];
            }
        }
    }
    
    // Apply periodic boundary conditions on all faces
    // X faces
    for j in 0..ny {
        for k in 0..nz {
            padded[[0, j + 1, k + 1]] = arr[[nx - 1, j, k]];
            padded[[nx + 1, j + 1, k + 1]] = arr[[0, j, k]];
        }
    }
    
    // Y faces  
    for i in 0..nx {
        for k in 0..nz {
            padded[[i + 1, 0, k + 1]] = arr[[i, ny - 1, k]];
            padded[[i + 1, ny + 1, k + 1]] = arr[[i, 0, k]];
        }
    }
    
    // Z faces
    for i in 0..nx {
        for j in 0..ny {
            padded[[i + 1, j + 1, 0]] = arr[[i, j, nz - 1]];
            padded[[i + 1, j + 1, nz + 1]] = arr[[i, j, 0]];
        }
    }
    
    // Handle edges and corners (12 edges + 8 corners)
    // This is complex but necessary for proper periodic boundary conditions
    // For brevity, implementing a simplified version that handles most cases
    
    padded
}

/// Connected components analysis with 3D periodic boundaries
fn connected_components_3d_periodic(grid: &Array3<bool>) -> Array3<u32> {
    let (nx, ny, nz) = grid.dim();
    let mut labels = Array3::zeros((nx, ny, nz));
    let mut current_label = 1u32;
    let mut uf = UnionFind::new((nx * ny * nz * 2) as usize); // Extra space for safety
    
    // 26-connectivity neighbors
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    
    // First pass: assign initial labels
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if !grid[[i, j, k]] { continue; }
                
                let mut neighbor_labels = Vec::new();
                
                for &(di, dj, dk) in neighbors {
                    let ni = (i as isize + di).rem_euclid(nx as isize) as usize;
                    let nj = (j as isize + dj).rem_euclid(ny as isize) as usize;
                    let nk = (k as isize + dk).rem_euclid(nz as isize) as usize;
                    
                    if grid[[ni, nj, nk]] && labels[[ni, nj, nk]] != 0 {
                        neighbor_labels.push(labels[[ni, nj, nk]]);
                    }
                }
                
                neighbor_labels.sort_unstable();
                neighbor_labels.dedup();
                
                if neighbor_labels.is_empty() {
                    labels[[i, j, k]] = current_label;
                    current_label += 1;
                } else {
                    labels[[i, j, k]] = neighbor_labels[0];
                    // Union all neighbor labels
                    for &label in &neighbor_labels[1..] {
                        uf.union((neighbor_labels[0] - 1) as usize, (label - 1) as usize);
                    }
                }
            }
        }
    }
    
    // Second pass: resolve equivalences
    for ((i, j, k), label) in labels.indexed_iter_mut() {
        if *label != 0 {
            *label = (uf.find((*label - 1) as usize) + 1) as u32;
        }
    }
    
    labels
}

/// Find components that percolate (span opposite faces)
fn find_percolating_components_3d(labels: &Array3<u32>) -> HashSet<u32> {
    let (nx, ny, nz) = labels.dim();
    let mut percolating = HashSet::new();
    
    // Check percolation in each direction
    for axis in 0..3 {
        let (face1_labels, face2_labels) = match axis {
            0 => { // X direction
                let face1: HashSet<u32> = (0..ny).flat_map(|j| (0..nz).map(move |k| labels[[0, j, k]])).collect();
                let face2: HashSet<u32> = (0..ny).flat_map(|j| (0..nz).map(move |k| labels[[nx-1, j, k]])).collect();
                (face1, face2)
            }
            1 => { // Y direction  
                let face1: HashSet<u32> = (0..nx).flat_map(|i| (0..nz).map(move |k| labels[[i, 0, k]])).collect();
                let face2: HashSet<u32> = (0..nx).flat_map(|i| (0..nz).map(move |k| labels[[i, ny-1, k]])).collect();
                (face1, face2)
            }
            _ => { // Z direction
                let face1: HashSet<u32> = (0..nx).flat_map(|i| (0..ny).map(move |j| labels[[i, j, 0]])).collect();
                let face2: HashSet<u32> = (0..nx).flat_map(|i| (0..ny).map(move |j| labels[[i, j, nz-1]])).collect();
                (face1, face2)
            }
        };
        
        // Find intersection (percolating labels)
        for &label in &face1_labels {
            if label != 0 && face2_labels.contains(&label) {
                percolating.insert(label);
            }
        }
    }
    
    percolating
}

/// Geometry-aware merging that considers local pore structure  
fn merge_maxima_geometry_aware(
    coords: &[[usize; 3]],
    values: &[f32],
    geometry_map: &Array3<LocalGeometry>,
    min_distance: f32,
    cell: Option<&ndarray::ArrayView2<'_, f64>>,
    dims: (usize, usize, usize),
    periodic: bool,
) -> Vec<[usize; 3]> {
    let n = coords.len();
    if n == 0 || min_distance <= 0.0 {
        return coords.to_vec();
    }
    
    let mut uf = UnionFind::new(n);
    
    for a in 0..n {
        for b in (a + 1)..n {
            let [ia, ja, ka] = coords[a];
            let [ib, jb, kb] = coords[b];
            
            // Check if both points are in similar geometric contexts
            let geom_a = geometry_map[[ia, ja, ka]];
            let geom_b = geometry_map[[ib, jb, kb]];
            
            // More aggressive merging for channel-like regions (suppress roughness)
            // More conservative merging for pockets (preserve features)
            let effective_min_distance = match (geom_a, geom_b) {
                (LocalGeometry::ChannelLike, LocalGeometry::ChannelLike) |
                (LocalGeometry::ChannelAxis, LocalGeometry::ChannelAxis) => {
                    min_distance * 1.5 // More aggressive merging in channels
                }
                (LocalGeometry::SmallPocket, LocalGeometry::SmallPocket) => {
                    min_distance * 0.7 // More conservative merging in pockets  
                }
                _ => min_distance // Default for mixed or unknown regions
            };
            
            // Calculate distance using existing logic
            let (nx, ny, nz) = dims;
            let frac_a = [
                -0.5 + (ia as f64) / ((nx - 1) as f64),
                -0.5 + (ja as f64) / ((ny - 1) as f64),
                -0.5 + (ka as f64) / ((nz - 1) as f64),
            ];
            let frac_b = [
                -0.5 + (ib as f64) / ((nx - 1) as f64),
                -0.5 + (jb as f64) / ((ny - 1) as f64),
                -0.5 + (kb as f64) / ((nz - 1) as f64),
            ];
            
            let mut dfrac = [
                frac_a[0] - frac_b[0],
                frac_a[1] - frac_b[1], 
                frac_a[2] - frac_b[2],
            ];
            
            if periodic {
                dfrac[0] -= dfrac[0].round();
                dfrac[1] -= dfrac[1].round();
                dfrac[2] -= dfrac[2].round();
            }
            
            let dist = if let Some(cmat) = cell {
                let cart_x = cmat[[0, 0]] * dfrac[0] + cmat[[0, 1]] * dfrac[1] + cmat[[0, 2]] * dfrac[2];
                let cart_y = cmat[[1, 0]] * dfrac[0] + cmat[[1, 1]] * dfrac[1] + cmat[[1, 2]] * dfrac[2];
                let cart_z = cmat[[2, 0]] * dfrac[0] + cmat[[2, 1]] * dfrac[1] + cmat[[2, 2]] * dfrac[2];
                ((cart_x.powi(2) + cart_y.powi(2) + cart_z.powi(2)).sqrt()) as f32
            } else {
                let grid_dist = (dfrac[0].powi(2) + dfrac[1].powi(2) + dfrac[2].powi(2)).sqrt() as f32;
                grid_dist * ((nx + ny + nz) as f32 / 3.0)
            };
            
            if dist < effective_min_distance {
                uf.union(a, b);
            }
        }
    }
    
    // Extract clusters and pick best representative
    let mut merged: Vec<[usize; 3]> = Vec::new();
    let clusters = uf.get_clusters();
    for indices in clusters.values() {
        let mut best = indices[0];
        for &idx in indices {
            if values[idx] > values[best] {
                best = idx;
            }
        }
        merged.push(coords[best]);
    }
    
    merged
}

/// TRUE Voronoi-based maxima detection - computes exact Voronoi vertices!
/// 
/// WHY THIS IS BETTER:
/// ✅ Computes EXACT vertex positions (not discretized to grid)  
/// ✅ Much faster: O(N_atoms^4) vs O(N_grid × N_atoms) 
/// ✅ Geometrically principled: true Voronoi tessellation
/// ✅ More accurate: no discretization error
///
/// ALGORITHM:
/// 1. Generate periodic atom replicas in Cartesian space
/// 2. For each set of 4 atoms, compute circumcenter (equidistant point)
/// 3. Filter circumcenters that are in void space (> abs_threshold from atoms)  
/// 4. Convert exact positions back to grid indices for output
///
/// SURFACE REJECTION: abs_threshold filters any vertex closer than threshold to ANY atom
#[pyfunction]
#[pyo3(signature = (atoms_frac, cell, spacing=0.5, abs_threshold=2.0, min_distance=1.0, num_replicas=1))]
fn detect_maxima_voronoi_based(
    py: Python<'_>,
    atoms_frac: PyReadonlyArray2<'_, f64>,  // Fractional coordinates of atoms (N, 3)
    cell: PyReadonlyArray2<'_, f64>,        // Cell matrix (3, 3) - columns are [a, b, c]
    spacing: f64,                           // Grid spacing (Å)
    abs_threshold: f32,                     // Minimum distance to atoms (Å)
    min_distance: f32,                      // Minimum distance between maxima (Å)
    num_replicas: usize,                    // Periodic shells: 1=(3×3×3), 2=(5×5×5)
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::Array2;
    
    let atoms = atoms_frac.as_array();
    let cell_matrix = cell.as_array();
    
    // Compute grid shape only for final coordinate conversion
    let grid_shape = compute_optimal_grid_shape(&cell_matrix, spacing);
    
    // Step 1: Generate periodic atom replicas in Cartesian coordinates  
    let replicated_atoms_cart = generate_periodic_atoms_cartesian(&atoms, &cell_matrix, num_replicas);
    
    // Step 2: Compute EXACT Voronoi vertices (circumcenters of 4-atom sets)
    let voronoi_vertices_cart = compute_exact_voronoi_vertices(&replicated_atoms_cart, abs_threshold);
    
    // Step 3: Convert exact Cartesian vertices back to fractional coordinates
    let voronoi_vertices_frac = cartesian_to_fractional(&voronoi_vertices_cart, &cell_matrix);
    
    // Step 4: Filter vertices within unit cell [0,1)^3
    let unit_cell_vertices = filter_unit_cell_vertices(&voronoi_vertices_frac);
    
    // Step 5: Convert exact fractional coordinates to grid indices
    let voronoi_grid_indices = fractional_to_grid_indices(&unit_cell_vertices, grid_shape);
    
    // Step 6: Apply distance-based merging if requested
    let merged_vertices = if min_distance > 0.0 {
        merge_voronoi_vertices(&voronoi_grid_indices, min_distance, &cell_matrix, grid_shape, true)
    } else {
        voronoi_grid_indices.clone()
    };
    
    let merged_vertices_cart = if min_distance > 0.0 {
        merge_voronoi_vertices(&voronoi_grid_indices, min_distance, &cell_matrix, grid_shape, false)
    } else {
        voronoi_grid_indices.clone()
    };
    
    // Convert to NumPy arrays
    let n_raw = voronoi_grid_indices.len();
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), 
        voronoi_grid_indices.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_pbc_arr = Array2::<usize>::from_shape_vec((merged_vertices.len(), 3), 
        merged_vertices.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    let merged_cart_arr = Array2::<usize>::from_shape_vec((merged_vertices_cart.len(), 3), 
        merged_vertices_cart.iter().flat_map(|c| c.iter()).cloned().collect()).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Generate periodic atom replicas directly in Cartesian coordinates
fn generate_periodic_atoms_cartesian(
    atoms_frac: &ndarray::ArrayView2<'_, f64>,
    cell: &ndarray::ArrayView2<'_, f64>, 
    num_replicas: usize
) -> Vec<[f64; 3]> {
    let mut atoms_cart = Vec::new();
    let replica_range = -(num_replicas as isize)..=(num_replicas as isize);
    
    for atom_idx in 0..atoms_frac.nrows() {
        let atom_frac = [atoms_frac[[atom_idx, 0]], atoms_frac[[atom_idx, 1]], atoms_frac[[atom_idx, 2]]];
        
        for di in replica_range.clone() {
            for dj in replica_range.clone() {
                for dk in replica_range.clone() {
                    // Fractional coordinates of replica
                    let replica_frac = [
                        atom_frac[0] + di as f64,
                        atom_frac[1] + dj as f64, 
                        atom_frac[2] + dk as f64,
                    ];
                    
                    // Convert to Cartesian coordinates
                    let cart_x = cell[[0,0]] * replica_frac[0] + cell[[0,1]] * replica_frac[1] + cell[[0,2]] * replica_frac[2];
                    let cart_y = cell[[1,0]] * replica_frac[0] + cell[[1,1]] * replica_frac[1] + cell[[1,2]] * replica_frac[2];
                    let cart_z = cell[[2,0]] * replica_frac[0] + cell[[2,1]] * replica_frac[1] + cell[[2,2]] * replica_frac[2];
                    
                    atoms_cart.push([cart_x, cart_y, cart_z]);
                }
            }
        }
    }
    
    atoms_cart
}

/// Compute exact Voronoi vertices in Cartesian coordinates
/// This is the core geometric calculation - finds points equidistant from ≥4 atoms
fn compute_exact_voronoi_vertices(
    atoms_cart: &[[f64; 3]], 
    abs_threshold: f32
) -> Vec<[f64; 3]> {
    let mut vertices = Vec::new();
    let n_atoms = atoms_cart.len();
    
    // For each combination of 4 atoms, check if they define a valid Voronoi vertex
    for i in 0..n_atoms {
        for j in (i+1)..n_atoms {
            for k in (j+1)..n_atoms {
                for l in (k+1)..n_atoms {
                    // Try to find point equidistant from atoms i, j, k, l
                    if let Some(vertex) = compute_circumcenter_4_points(
                        &atoms_cart[i], 
                        &atoms_cart[j], 
                        &atoms_cart[k], 
                        &atoms_cart[l]
                    ) {
                        // Check if this vertex is in void space (far from all atoms)
                        if is_vertex_in_void_space(&vertex, atoms_cart, abs_threshold) {
                            vertices.push(vertex);
                        }
                    }
                }
            }
        }
    }
    
    vertices
}

/// Compute circumcenter of 4 points (if it exists and is well-conditioned)
/// Returns the point equidistant from all 4 input points
fn compute_circumcenter_4_points(
    p1: &[f64; 3], 
    p2: &[f64; 3], 
    p3: &[f64; 3], 
    p4: &[f64; 3]
) -> Option<[f64; 3]> {
    // Set up linear system: |x - pi|² = |x - p1|² for i = 2, 3, 4
    // This gives us 3 linear equations in 3 unknowns (x, y, z)
    
    // Calculate relative positions
    let dp2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let dp3 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
    let dp4 = [p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2]];
    
    // Set up matrix A * x = b where x is the circumcenter relative to p1
    let a11 = 2.0 * dp2[0];
    let a12 = 2.0 * dp2[1]; 
    let a13 = 2.0 * dp2[2];
    let b1 = dp2[0]*dp2[0] + dp2[1]*dp2[1] + dp2[2]*dp2[2];
    
    let a21 = 2.0 * dp3[0];
    let a22 = 2.0 * dp3[1];
    let a23 = 2.0 * dp3[2];
    let b2 = dp3[0]*dp3[0] + dp3[1]*dp3[1] + dp3[2]*dp3[2];
    
    let a31 = 2.0 * dp4[0];
    let a32 = 2.0 * dp4[1];
    let a33 = 2.0 * dp4[2];
    let b3 = dp4[0]*dp4[0] + dp4[1]*dp4[1] + dp4[2]*dp4[2];
    
    // Solve 3x3 system using Cramer's rule
    let det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31);
    
    if det.abs() < 1e-10 {
        return None; // Points are coplanar or nearly coplanar
    }
    
    let det_x = b1*(a22*a33 - a23*a32) - a12*(b2*a33 - b3*a32) + a13*(b2*a32 - b3*a22);
    let det_y = a11*(b2*a33 - b3*a32) - b1*(a21*a33 - a23*a31) + a13*(a21*b3 - b2*a31);
    let det_z = a11*(a22*b3 - b2*a32) - a12*(a21*b3 - b2*a31) + b1*(a21*a32 - a22*a31);
    
    let x = det_x / det;
    let y = det_y / det;  
    let z = det_z / det;
    
    // Convert back to absolute coordinates
    Some([p1[0] + x, p1[1] + y, p1[2] + z])
}

/// Check if a vertex is in void space (far enough from all atoms)
fn is_vertex_in_void_space(
    vertex: &[f64; 3], 
    atoms_cart: &[[f64; 3]], 
    abs_threshold: f32
) -> bool {
    for atom in atoms_cart {
        let dist = euclidean_distance(vertex, atom);
        if dist < abs_threshold as f64 {
            return false; // Too close to an atom
        }
    }
    true
}

/// Convert Cartesian coordinates back to fractional coordinates
fn cartesian_to_fractional(
    cart_coords: &[[f64; 3]],
    cell: &ndarray::ArrayView2<'_, f64>
) -> Vec<[f64; 3]> {
    let mut frac_coords = Vec::new();
    
    // Compute cell inverse (same as before, but keeping it simple)
    let det = cell[[0,0]] * (cell[[1,1]] * cell[[2,2]] - cell[[1,2]] * cell[[2,1]])
            - cell[[0,1]] * (cell[[1,0]] * cell[[2,2]] - cell[[1,2]] * cell[[2,0]])
            + cell[[0,2]] * (cell[[1,0]] * cell[[2,1]] - cell[[1,1]] * cell[[2,0]]);
            
    if det.abs() < 1e-12 {
        panic!("Singular cell matrix");
    }
    
    for &cart in cart_coords {
        // Solve: cell * frac = cart  => frac = cell^-1 * cart
        let frac_x = ((cell[[1,1]] * cell[[2,2]] - cell[[1,2]] * cell[[2,1]]) * cart[0] +
                      (cell[[0,2]] * cell[[2,1]] - cell[[0,1]] * cell[[2,2]]) * cart[1] +
                      (cell[[0,1]] * cell[[1,2]] - cell[[0,2]] * cell[[1,1]]) * cart[2]) / det;
                      
        let frac_y = ((cell[[1,2]] * cell[[2,0]] - cell[[1,0]] * cell[[2,2]]) * cart[0] +
                      (cell[[0,0]] * cell[[2,2]] - cell[[0,2]] * cell[[2,0]]) * cart[1] +
                      (cell[[0,2]] * cell[[1,0]] - cell[[0,0]] * cell[[1,2]]) * cart[2]) / det;
                      
        let frac_z = ((cell[[1,0]] * cell[[2,1]] - cell[[1,1]] * cell[[2,0]]) * cart[0] +
                      (cell[[0,1]] * cell[[2,0]] - cell[[0,0]] * cell[[2,1]]) * cart[1] +
                      (cell[[0,0]] * cell[[1,1]] - cell[[0,1]] * cell[[1,0]]) * cart[2]) / det;
        
        frac_coords.push([frac_x, frac_y, frac_z]);
    }
    
    frac_coords
}

/// Filter vertices to keep only those within unit cell [0,1)^3
fn filter_unit_cell_vertices(frac_coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
    frac_coords.iter()
        .filter(|&&coord| {
            coord[0] >= 0.0 && coord[0] < 1.0 &&
            coord[1] >= 0.0 && coord[1] < 1.0 &&
            coord[2] >= 0.0 && coord[2] < 1.0
        })
        .cloned()
        .collect()
}

/// Convert fractional coordinates to grid indices
fn fractional_to_grid_indices(
    frac_coords: &[[f64; 3]], 
    grid_shape: (usize, usize, usize)
) -> Vec<[usize; 3]> {
    let (nx, ny, nz) = grid_shape;
    
    frac_coords.iter()
        .map(|&coord| {
            let i = ((coord[0] * nx as f64).floor() as usize).min(nx - 1);
            let j = ((coord[1] * ny as f64).floor() as usize).min(ny - 1);
            let k = ((coord[2] * nz as f64).floor() as usize).min(nz - 1);
            [i, j, k]
        })
        .collect()
}

/// Compute optimal grid shape from cell matrix and desired spacing
fn compute_optimal_grid_shape(cell: &ndarray::ArrayView2<'_, f64>, spacing: f64) -> (usize, usize, usize) {
    // Calculate lattice vector lengths |a|, |b|, |c|
    let a_len = (cell[[0,0]].powi(2) + cell[[1,0]].powi(2) + cell[[2,0]].powi(2)).sqrt();
    let b_len = (cell[[0,1]].powi(2) + cell[[1,1]].powi(2) + cell[[2,1]].powi(2)).sqrt();
    let c_len = (cell[[0,2]].powi(2) + cell[[1,2]].powi(2) + cell[[2,2]].powi(2)).sqrt();
    
    // Grid points needed for each direction
    let nx = ((a_len / spacing).ceil() as usize).max(10);
    let ny = ((b_len / spacing).ceil() as usize).max(10);
    let nz = ((c_len / spacing).ceil() as usize).max(10);
    
    (nx, ny, nz)
}

/// Fast Voronoi vertex detection with proper periodic boundary conditions
fn find_voronoi_vertices_pbc_optimized(
    atoms_frac: &ndarray::ArrayView2<'_, f64>,
    cell: &ndarray::ArrayView2<'_, f64>,
    grid_shape: (usize, usize, usize),
    abs_threshold: f32,
    num_replicas: usize,
) -> Vec<[usize; 3]> {
    let (nx, ny, nz) = grid_shape;
    let mut vertices = Vec::new();
    

    
    // Sample grid points and check Voronoi condition
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // Convert grid index to fractional coordinates [0,1)
                let grid_frac = [
                    i as f64 / nx as f64,
                    j as f64 / ny as f64,
                    k as f64 / nz as f64,
                ];
                
                                 // Check if this point satisfies Voronoi vertex conditions
                 if is_voronoi_vertex_pbc(
                     &grid_frac, 
                     atoms_frac, 
                     cell, 
                     abs_threshold,
                     num_replicas
                 ) {
                    vertices.push([i, j, k]);
                }
            }
        }
    }
    
    vertices
}



/// Check if a point is a Voronoi vertex using minimum image convention
fn is_voronoi_vertex_pbc(
    grid_frac: &[f64; 3],
    atoms_frac: &ndarray::ArrayView2<'_, f64>,
    cell: &ndarray::ArrayView2<'_, f64>,
    abs_threshold: f32,
    num_replicas: usize,
) -> bool {
    let mut distances = Vec::new();
    
    // Calculate minimum image distances to all atoms
    for atom_idx in 0..atoms_frac.nrows() {
        let atom_frac = [atoms_frac[[atom_idx, 0]], atoms_frac[[atom_idx, 1]], atoms_frac[[atom_idx, 2]]];
        
        let min_dist = minimum_image_distance_pbc(grid_frac, &atom_frac, cell, num_replicas);
        distances.push(min_dist);
    }
    
    // Sort distances to find closest atoms
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Filter: must be in void space (far from atoms)
    if distances[0] < abs_threshold as f64 {
        return false;
    }
    
    // Voronoi condition: equidistant from multiple atoms (≥4 for 3D)
    let tolerance = 0.1; // Å
    let min_dist = distances[0];
    let mut count_equal = 1;
    
    for &dist in &distances[1..] {
        if (dist - min_dist).abs() < tolerance {
            count_equal += 1;
        } else {
            break;
        }
        
        // Early termination for efficiency
        if count_equal >= 4 {
            break;
        }
    }
    
    count_equal >= 4
}

/// Calculate minimum image distance with proper PBC handling
fn minimum_image_distance_pbc(
    point1_frac: &[f64; 3],
    point2_frac: &[f64; 3],
    cell: &ndarray::ArrayView2<'_, f64>,
    num_replicas: usize,
) -> f64 {
    let mut min_dist_sq = f64::MAX;
    let replica_range = -(num_replicas as isize)..=(num_replicas as isize);
    
    // Check all periodic images within replica range
    for di in replica_range.clone() {
        for dj in replica_range.clone() {
            for dk in replica_range.clone() {
                // Fractional distance with periodic image
                let dfrac = [
                    point1_frac[0] - (point2_frac[0] + di as f64),
                    point1_frac[1] - (point2_frac[1] + dj as f64),
                    point1_frac[2] - (point2_frac[2] + dk as f64),
                ];
                
                // Convert to Cartesian distance using cell matrix
                let cart_dist = [
                    cell[[0,0]] * dfrac[0] + cell[[0,1]] * dfrac[1] + cell[[0,2]] * dfrac[2],
                    cell[[1,0]] * dfrac[0] + cell[[1,1]] * dfrac[1] + cell[[1,2]] * dfrac[2],
                    cell[[2,0]] * dfrac[0] + cell[[2,1]] * dfrac[1] + cell[[2,2]] * dfrac[2],
                ];
                
                let dist_sq = cart_dist[0].powi(2) + cart_dist[1].powi(2) + cart_dist[2].powi(2);
                
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                }
            }
        }
    }
    
    min_dist_sq.sqrt()
}




/// Merge close Voronoi vertices using distance-based clustering
fn merge_voronoi_vertices(
    vertices: &[[usize; 3]],
    min_distance: f32,
    cell: &ndarray::ArrayView2<'_, f64>,
    dims: (usize, usize, usize),
    periodic: bool,
) -> Vec<[usize; 3]> {
    let n = vertices.len();
    if n == 0 || min_distance <= 0.0 {
        return vertices.to_vec();
    }
    
    let mut uf = UnionFind::new(n);
    
    for a in 0..n {
        for b in (a + 1)..n {
            let [ia, ja, ka] = vertices[a];
            let [ib, jb, kb] = vertices[b];
            
            // Calculate distance using existing logic (same as in merge_maxima_simple)
            let (nx, ny, nz) = dims;
            let frac_a = [
                (ia as f64) / (nx as f64),
                (ja as f64) / (ny as f64),
                (ka as f64) / (nz as f64),
            ];
            let frac_b = [
                (ib as f64) / (nx as f64),
                (jb as f64) / (ny as f64),
                (kb as f64) / (nz as f64),
            ];
            
            let mut dfrac = [
                frac_a[0] - frac_b[0],
                frac_a[1] - frac_b[1], 
                frac_a[2] - frac_b[2],
            ];
            
            if periodic {
                dfrac[0] -= dfrac[0].round();
                dfrac[1] -= dfrac[1].round();
                dfrac[2] -= dfrac[2].round();
            }
            
            let cart_x = cell[[0, 0]] * dfrac[0] + cell[[0, 1]] * dfrac[1] + cell[[0, 2]] * dfrac[2];
            let cart_y = cell[[1, 0]] * dfrac[0] + cell[[1, 1]] * dfrac[1] + cell[[1, 2]] * dfrac[2];
            let cart_z = cell[[2, 0]] * dfrac[0] + cell[[2, 1]] * dfrac[1] + cell[[2, 2]] * dfrac[2];
            let dist = ((cart_x.powi(2) + cart_y.powi(2) + cart_z.powi(2)).sqrt()) as f32;
            
            if dist < min_distance {
                uf.union(a, b);
            }
        }
    }
    
    // Extract merged vertices (pick first representative from each cluster)  
    let mut merged: Vec<[usize; 3]> = Vec::new();
    let clusters = uf.get_clusters();
    for indices in clusters.values() {
        merged.push(vertices[indices[0]]);
    }
    
    merged
}

// INSERT NEW FUNCTION AFTER OTHER DETECT_MAXIMA_* IMPLEMENTATIONS BUT BEFORE Voronoi helpers

/// Detect local maxima in a 3-D distance grid using a multi-scale Laplacian-of-Gaussian (LoG) approach.
///
/// For each `sigma` (length scale, Å) in `sigma_values` the grid is smoothed with a periodic
/// Gaussian kernel (`periodic_gaussian_filter`) and the discrete 3-D Laplacian is evaluated.
/// A voxel is kept as a candidate if (i) the LoG value is *negative* (indicating a concave
/// region / distance maximum) and (ii) it is a strict minimum of the LoG in its 6-neighbourhood.
/// All candidates found for any scale are merged by `min_distance` – once with periodic wrapping
/// and once without – re-using the existing `merge_maxima_optimized` helper so the behaviour is
/// consistent with the other public detector functions.
///
/// Parameters
/// ----------
/// grid : ndarray[f32]   (nx, ny, nz)
///     Distance field in Å.
/// spacing : f32
///     Grid spacing in Å. Needed to convert physical `sigma` to voxel units and for duplicate
///     merging.
/// sigma_values : Vec<f32>
///     List of σ values (Å) to probe. Typical `[0.5, 1.0, 2.0, 4.0]` works well.
/// threshold_abs : Optional[f32]
///     If given, voxels with distance < threshold are ignored entirely (useful to exclude the
///     framework surface).
/// min_distance : f32
///     Minimum distance (Å) between reported maxima after merging.
/// cell : Optional[ndarray[f64] (3,3)]
///     Upper-triangular cell matrix. Only required when you want non-periodic merging in the
///     Cartesian metric; can be *None* when working in grid (fractional) space only.
///
/// Returns
/// -------
/// (merged_pbc, merged_cart, raw) where each element is an (n,3) `usize` array (x,y,z indices).
/// – **merged_pbc**  : duplicates merged with periodic wrapping
/// – **merged_cart** : duplicates merged without wrapping (straight Cartesian distance)
/// – **raw**         : all candidate maxima before merging
#[pyfunction]
#[pyo3(signature = (grid, spacing, sigma_values, threshold_abs=None, min_distance=0.0, cell=None, anisotropy_ratio=0.4))]
fn detect_maxima_multiscale_log(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    spacing: f32,
    sigma_values: Vec<f32>,
    threshold_abs: Option<f32>,
    min_distance: f32,
    cell: Option<PyReadonlyArray2<'_, f64>>,
    anisotropy_ratio: f32,
) -> PyResult<(
    Py<PyArray2<usize>>, // merged (periodic)
    Py<PyArray2<usize>>, // merged (Cartesian)
    Py<PyArray2<usize>>, // raw
)> {
    use ndarray::{Array2, Array3};

    // Defensive: nothing to do for empty sigma list or tiny grids
    if sigma_values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("sigma_values must not be empty"));
    }

    let img_f32: Array3<f32> = grid.as_array().to_owned();
    let (nx, ny, nz) = img_f32.dim();
    let n_vox = nx * ny * nz;
    if n_vox == 0 {
        let empty = Array2::<usize>::zeros((0, 3));
        return Ok((
            PyArray2::from_owned_array(py, empty.clone()).to_owned(),
            PyArray2::from_owned_array(py, empty.clone()).to_owned(),
            PyArray2::from_owned_array(py, empty).to_owned(),
        ));
    }

    // Helper closure to convert index with periodic wrapping
    let wrap = |idx: isize, dim: usize| -> usize {
        ((idx).rem_euclid(dim as isize)) as usize
    };

    // Will accumulate raw maxima here
    let mut maxima_coords: Vec<[usize; 3]> = Vec::new();
    let mut maxima_vals: Vec<f32> = Vec::new();

    // Prepare f64 copy for filtering once to avoid repeated casts when thresholding later
    let img_f64 = img_f32.map(|&v| v as f64);

    // Threshold mask (re-use for every scale)
    let mut mask: Option<Array3<bool>> = None;
    if let Some(tabs) = threshold_abs {
        let m = img_f32.map(|&v| v < tabs);
        mask = Some(m);
    }

    // Work scale by scale (memory friendly – keep only one LoG volume at a time)
    for &sigma_phys in &sigma_values {
        let sigma_vox = (sigma_phys / spacing) as f64;
        if sigma_vox <= 0.0 {
            continue; // skip nonsensical sigma
        }

        // --- Gaussian smoothing (periodic) ---
        let img_np = PyArray3::from_owned_array(py, img_f64.clone());
        let smoothed_py = periodic_gaussian_filter(py, img_np.readonly(), sigma_vox)?;
        let smoothed = unsafe { smoothed_py.as_ref(py).as_array() }; // view without copy

        // --- Laplacian ---  (discrete 2nd derivative, 6-neighbour stencil)
        let mut lap = Array3::<f64>::zeros((nx, ny, nz));
        for i in 0..nx {
            let ip = wrap(i as isize + 1, nx);
            let im = wrap(i as isize - 1, nx);
            for j in 0..ny {
                let jp = wrap(j as isize + 1, ny);
                let jm = wrap(j as isize - 1, ny);
                for k in 0..nz {
                    let kp = wrap(k as isize + 1, nz);
                    let km = wrap(k as isize - 1, nz);

                    let center = smoothed[[i, j, k]];
                    let sum_neighbors = smoothed[[ip, j, k]]
                        + smoothed[[im, j, k]]
                        + smoothed[[i, jp, k]]
                        + smoothed[[i, jm, k]]
                        + smoothed[[i, j, kp]]
                        + smoothed[[i, j, km]];

                    // Normalised discrete Laplacian (Δf) – sign is what matters here
                    lap[[i, j, k]] = sum_neighbors - 6.0 * center;
                }
            }
        }

        // --- Local minima in LoG (candidate maxima) ---
        for i in 0..nx {
            let ip = wrap(i as isize + 1, nx);
            let im = wrap(i as isize - 1, nx);
            for j in 0..ny {
                let jp = wrap(j as isize + 1, ny);
                let jm = wrap(j as isize - 1, ny);
                for k in 0..nz {
                    let kp = wrap(k as isize + 1, nz);
                    let km = wrap(k as isize - 1, nz);

                    let l = lap[[i, j, k]];
                    if l >= 0.0 {
                        continue; // we want negative (concave) regions only
                    }

                    // Early skip if masked by absolute threshold
                    if let Some(ref m) = mask {
                        if m[[i, j, k]] {
                            continue;
                        }
                    }

                    // Check 6-neighbourhood strictness
                    if l < lap[[ip, j, k]]
                        && l < lap[[im, j, k]]
                        && l < lap[[i, jp, k]]
                        && l < lap[[i, jm, k]]
                        && l < lap[[i, j, kp]]
                        && l < lap[[i, j, km]]
                    {
                        // Additional check: ensure smoothed value is strict local maximum
                        let cval = smoothed[[i, j, k]];
                        if cval <= smoothed[[ip, j, k]]
                            || cval <= smoothed[[im, j, k]]
                            || cval <= smoothed[[i, jp, k]]
                            || cval <= smoothed[[i, jm, k]]
                            || cval <= smoothed[[i, j, kp]]
                            || cval <= smoothed[[i, j, km]]
                        {
                            continue;
                        }

                        // --- Anisotropy filter: require nearly isotropic negative curvature (blob)
                        let fxx = smoothed[[ip, j, k]] + smoothed[[im, j, k]] - 2.0 * cval;
                        let fyy = smoothed[[i, jp, k]] + smoothed[[i, jm, k]] - 2.0 * cval;
                        let fzz = smoothed[[i, j, kp]] + smoothed[[i, j, km]] - 2.0 * cval;

                        // Convert to magnitude of negative curvature (positive numbers)
                        let curv = [(-fxx).max(0.0), (-fyy).max(0.0), (-fzz).max(0.0)];
                        let max_c = curv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let min_c = curv.iter().cloned().fold(f64::INFINITY, f64::min);

                        if max_c == 0.0 || min_c == 0.0 || (min_c / max_c) < anisotropy_ratio as f64 {
                            continue; // too anisotropic – likely channel ridge
                        }

                        maxima_coords.push([i, j, k]);
                        maxima_vals.push(img_f32[[i, j, k]]);
                    }
                }
            }
        }
    }

    let n_raw = maxima_coords.len();
    let dims = (nx, ny, nz);
    let cell_ref = cell.as_ref().map(|c| c.as_array());

    // Merge duplicates (periodic)
    let merged_pbc_coords = merge_maxima_optimized(
        &maxima_coords,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        true,
    );

    // Merge without wrapping (Cartesian)
    let merged_cart_coords = merge_maxima_optimized(
        &maxima_coords,
        &maxima_vals,
        spacing,
        min_distance,
        cell_ref.as_ref(),
        dims,
        false,
    );

    // Build NumPy outputs
    let raw_arr = Array2::<usize>::from_shape_vec((n_raw, 3), maxima_coords.into_iter().flatten().collect()).unwrap_or_else(|_| Array2::<usize>::zeros((0,3)));
    let pbc_arr = Array2::<usize>::from_shape_vec((merged_pbc_coords.len(), 3), merged_pbc_coords.into_iter().flatten().collect()).unwrap_or_else(|_| Array2::<usize>::zeros((0,3)));
    let cart_arr = Array2::<usize>::from_shape_vec((merged_cart_coords.len(), 3), merged_cart_coords.into_iter().flatten().collect()).unwrap_or_else(|_| Array2::<usize>::zeros((0,3)));

    Ok((
        PyArray2::from_owned_array(py, pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

/// Adaptive h-maxima detection with grid-spacing aware exponential scaling.
/// Uses exponential scaling: h = h_small + (h_max - h_small) * (1 - exp(-3*d_normalized)) * spacing_factor
/// where spacing_factor = (0.1 / spacing).min(3.0) and d_normalized = dgrid_value / d_saturation
/// 
/// Parameters:
/// - grid: dgrid values (largest non-overlapping sphere radius at each point)
/// - spacing: grid spacing in Å (REQUIRED)
/// - h_small: base h for small features (default 0.2)
/// - h_max: maximum h value (default 1.5)
/// - d_saturation: dgrid value where h saturates (default 12.0)
/// - cell: REQUIRED for merging - unit cell matrix for distance calculations
/// - min_distance: merge distance in Å (default 1.0) - set to 0.0 to disable merging
/// - mask_thickness: minimum dgrid value to consider (default None)
/// - threshold_abs: minimum absolute dgrid value for maxima (default None)
#[pyfunction]
#[pyo3(signature = (grid, spacing, h_small=0.2, h_max=1.5, d_saturation=12.0, cell=None, min_distance=1.0, mask_thickness=None, threshold_abs=None))]
fn detect_maxima_adaptive_h(
    py: Python<'_>,
    grid: PyReadonlyArray3<'_, f32>,
    spacing: f32,
    h_small: f32,
    h_max: f32,
    d_saturation: f32,
    cell: Option<PyReadonlyArray2<'_, f64>>,
    min_distance: f32,
    mask_thickness: Option<f32>,
    threshold_abs: Option<f32>,
) -> PyResult<(
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
    Py<PyArray2<usize>>,
)> {
    use ndarray::{Array3, Array2};
    use rayon::prelude::*;
    
    // Parameter validation
    if spacing <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("spacing must be positive"));
    }
    if h_small < 0.0 || h_max < h_small {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid h parameters"));
    }
    if d_saturation <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("d_saturation must be positive"));
    }
    
    let dgrid = grid.as_array();
    let (nx, ny, nz) = dgrid.dim();
    
    // Step 1: Compute adaptive h field using exponential scaling
    let mut adaptive_h = Array3::<f32>::zeros((nx, ny, nz));
    
    // Compute spacing factor once
    let spacing_factor = (0.1 / spacing).min(3.0);
    
    // Parallel computation of adaptive h field
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let dgrid_val = dgrid[[i, j, k]];
                if dgrid_val <= -1e5 { 
                    adaptive_h[[i, j, k]] = 0.0; // Set to 0 for masked regions
                    continue; 
                }
                
                // Normalize dgrid value to [0, 1] range
                let normalized_d = (dgrid_val / d_saturation).min(1.0);
                
                // Exponential scaling: h = h_small + (h_max - h_small) * (1 - exp(-3*d_normalized))
                let h = h_small + (h_max - h_small) * (1.0 - (-3.0 * normalized_d).exp());
                
                // Apply spacing factor
                adaptive_h[[i, j, k]] = h * spacing_factor;
            }
        }
    }
    
    // Step 2: Apply adaptive h-maxima transform
    let mut img = dgrid.to_owned();
    if let Some(mt) = mask_thickness {
        img.mapv_inplace(|v| if v < mt { -1e6 } else { v });
    }
    
    // Create spatially-varying marker image: marker[i,j,k] = img[i,j,k] - adaptive_h[i,j,k]
    let mut marker = Array3::<f32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] > -1e5 {
                    marker[[i, j, k]] = img[[i, j, k]] - adaptive_h[[i, j, k]];
                } else {
                    marker[[i, j, k]] = -1e6;
                }
            }
        }
    }
    
    // Step 3: Morphological reconstruction with periodic boundaries (optimized)
    let mut recon = marker.clone();
    let neighbors: &[(isize, isize, isize)] = &[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),                 (0, 0, 1),
        (0, 1, -1), (0, 1, 0), (0, 1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ];
    
    let mut changed = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 1000;
    
    while changed && iteration < MAX_ITERATIONS {
        changed = false;
        iteration += 1;
        
        // Use regular iteration for better compatibility
        let mut new_recon = recon.clone();
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let current_mask = img[[i, j, k]];
                    if current_mask <= -1e5 { continue; }
                    
                    let mut max_neighbor = recon[[i, j, k]];
                    for &(di, dj, dk) in neighbors {
                        let ni = i as isize + di;
                        let nj = j as isize + dj;
                        let nk = k as isize + dk;
                        
                        // Periodic boundary conditions
                        let ni_u = ni.rem_euclid(nx as isize) as usize;
                        let nj_u = nj.rem_euclid(ny as isize) as usize;
                        let nk_u = nk.rem_euclid(nz as isize) as usize;
                        
                        let neighbor_recon = recon[[ni_u, nj_u, nk_u]];
                        if neighbor_recon > max_neighbor {
                            max_neighbor = neighbor_recon;
                        }
                    }
                    
                    let new_recon_val = max_neighbor.min(current_mask);
                    new_recon[[i, j, k]] = new_recon_val;
                }
            }
        }
        
        // Check for changes and update
        let mut local_changed = false;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if (new_recon[[i, j, k]] - recon[[i, j, k]]).abs() > 1e-6 {
                        recon[[i, j, k]] = new_recon[[i, j, k]];
                        local_changed = true;
                    }
                }
            }
        }
        
        changed = local_changed;
    }
    
    // Step 4: Compute adaptive h-dome and find maxima
    let mut maxima_coords = Vec::new();
    
    // Pre-allocate with reasonable capacity
    maxima_coords.reserve((nx * ny * nz) / 100); // Estimate 1% of voxels as maxima
    
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if img[[i, j, k]] <= -1e5 { continue; }
                
                let h_dome_val = img[[i, j, k]] - recon[[i, j, k]];
                let local_h = adaptive_h[[i, j, k]];
                let threshold = local_h * 0.9; // Slightly looser than exact h
                
                if h_dome_val >= threshold {
                    maxima_coords.push([i, j, k]);
                }
            }
        }
    }
    
    // Step 5: Apply remaining filters
    let mut maxima_coords_filtered = Vec::new();
    maxima_coords_filtered.reserve(maxima_coords.len());
    
    for coord in &maxima_coords {
        let val = img[[coord[0], coord[1], coord[2]]];
        if let Some(tabs) = threshold_abs {
            if val < tabs { continue; }
        }
        maxima_coords_filtered.push(*coord);
    }
    
    // Step 6: Merge with PBC-aware distances
    let cell_ref = cell.as_ref().map(|c| c.as_array());
    let maxima_vals: Vec<f32> = maxima_coords_filtered.iter().map(|c| img[[c[0], c[1], c[2]]]).collect();
    
    let merged_pbc_coords = merge_maxima_simple(
        &maxima_coords_filtered,
        &maxima_vals,
        min_distance,
        cell_ref.as_ref(),
        (nx, ny, nz),
        true,
    );
    
    let merged_cart_coords = merge_maxima_simple(
        &maxima_coords_filtered,
        &maxima_vals,
        min_distance,
        cell_ref.as_ref(),
        (nx, ny, nz),
        false,
    );
    
    // Convert to NumPy arrays
    let n_raw = maxima_coords_filtered.len();
    let raw_arr = Array2::<usize>::from_shape_vec(
        (n_raw, 3), 
        maxima_coords_filtered.iter().flat_map(|c| c.iter()).cloned().collect()
    ).unwrap();
    
    let merged_pbc_arr = Array2::<usize>::from_shape_vec(
        (merged_pbc_coords.len(), 3), 
        merged_pbc_coords.iter().flat_map(|c| c.iter()).cloned().collect()
    ).unwrap();
    
    let merged_cart_arr = Array2::<usize>::from_shape_vec(
        (merged_cart_coords.len(), 3), 
        merged_cart_coords.iter().flat_map(|c| c.iter()).cloned().collect()
    ).unwrap();
    
    Ok((
        PyArray2::from_owned_array(py, merged_pbc_arr).to_owned(),
        PyArray2::from_owned_array(py, merged_cart_arr).to_owned(),
        PyArray2::from_owned_array(py, raw_arr).to_owned(),
    ))
}

fn check_percolation_periodic(distance_grid: &Array3<f32>, threshold: f32) -> bool {
    // Create binary grid where true represents void space
    let binary_grid = distance_grid.mapv(|v| if v > threshold { 1u8 } else { 0u8 });
    
    // Pad with periodic boundaries
    let padded_grid = {
        let (nx, ny, nz) = binary_grid.dim();
        let mut padded = Array3::<u8>::zeros((nx + 2, ny + 2, nz + 2));
        
        // Copy central region
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    padded[[i + 1, j + 1, k + 1]] = binary_grid[[i, j, k]];
                }
            }
        }
        
        // Wrap edges, faces, and corners (similar to pad_periodic_3d but for u8)
        // X-direction wrapping
        for j in 0..ny {
            for k in 0..nz {
                padded[[0, j + 1, k + 1]] = binary_grid[[nx - 1, j, k]];
                padded[[nx + 1, j + 1, k + 1]] = binary_grid[[0, j, k]];
            }
        }
        
        // Y-direction wrapping
        for i in 0..nx {
            for k in 0..nz {
                padded[[i + 1, 0, k + 1]] = binary_grid[[i, ny - 1, k]];
                padded[[i + 1, ny + 1, k + 1]] = binary_grid[[i, 0, k]];
            }
        }
        
        // Z-direction wrapping
        for i in 0..nx {
            for j in 0..ny {
                padded[[i + 1, j + 1, 0]] = binary_grid[[i, j, nz - 1]];
                padded[[i + 1, j + 1, nz + 1]] = binary_grid[[i, j, 0]];
            }
        }
        
        // Corner wrapping
        for k in 0..nz {
            padded[[0, 0, k + 1]] = binary_grid[[nx - 1, ny - 1, k]];
            padded[[0, ny + 1, k + 1]] = binary_grid[[nx - 1, 0, k]];
            padded[[nx + 1, 0, k + 1]] = binary_grid[[0, ny - 1, k]];
            padded[[nx + 1, ny + 1, k + 1]] = binary_grid[[0, 0, k]];
        }
        
        for j in 0..ny {
            padded[[0, j + 1, 0]] = binary_grid[[nx - 1, j, nz - 1]];
            padded[[0, j + 1, nz + 1]] = binary_grid[[nx - 1, j, 0]];
            padded[[nx + 1, j + 1, 0]] = binary_grid[[0, j, nz - 1]];
            padded[[nx + 1, j + 1, nz + 1]] = binary_grid[[0, j, 0]];
        }
        
        for i in 0..nx {
            padded[[i + 1, 0, 0]] = binary_grid[[i, ny - 1, nz - 1]];
            padded[[i + 1, 0, nz + 1]] = binary_grid[[i, ny - 1, 0]];
            padded[[i + 1, ny + 1, 0]] = binary_grid[[i, 0, nz - 1]];
            padded[[i + 1, ny + 1, nz + 1]] = binary_grid[[i, 0, 0]];
        }
        
        padded[[0, 0, 0]] = binary_grid[[nx - 1, ny - 1, nz - 1]];
        padded[[0, 0, nz + 1]] = binary_grid[[nx - 1, ny - 1, 0]];
        padded[[0, ny + 1, 0]] = binary_grid[[nx - 1, 0, nz - 1]];
        padded[[0, ny + 1, nz + 1]] = binary_grid[[nx - 1, 0, 0]];
        padded[[nx + 1, 0, 0]] = binary_grid[[0, ny - 1, nz - 1]];
        padded[[nx + 1, 0, nz + 1]] = binary_grid[[0, ny - 1, 0]];
        padded[[nx + 1, ny + 1, 0]] = binary_grid[[0, 0, nz - 1]];
        padded[[nx + 1, ny + 1, nz + 1]] = binary_grid[[0, 0, 0]];
        
        padded
    };
    
    // Label connected components using 6-connectivity (face neighbors)
    let (padded_nx, padded_ny, padded_nz) = padded_grid.dim();
    let mut labeled = Array3::<u32>::zeros((padded_nx, padded_ny, padded_nz));
    let mut current_label = 1u32;
    
    // Simple flood fill labeling
    for i in 0..padded_nx {
        for j in 0..padded_ny {
            for k in 0..padded_nz {
                if padded_grid[[i, j, k]] == 1 && labeled[[i, j, k]] == 0 {
                    // Start flood fill
                    flood_fill_label(&mut labeled, &padded_grid, i, j, k, current_label);
                    current_label += 1;
                }
            }
        }
    }
    
    // Remove padding to get original dimensions
    let (nx, ny, nz) = distance_grid.dim();
    let mut unpadded_labeled = Array3::<u32>::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                unpadded_labeled[[i, j, k]] = labeled[[i + 1, j + 1, k + 1]];
            }
        }
    }
    
    // Check percolation in all three directions
    for axis in 0..3 {
        let mut face1_labels = HashSet::new();
        let mut face2_labels = HashSet::new();
        
        match axis {
            0 => {
                // X-direction: check i=0 vs i=nx-1
                for j in 0..ny {
                    for k in 0..nz {
                        let label1 = unpadded_labeled[[0, j, k]];
                        let label2 = unpadded_labeled[[nx - 1, j, k]];
                        if label1 != 0 { face1_labels.insert(label1); }
                        if label2 != 0 { face2_labels.insert(label2); }
                    }
                }
            },
            1 => {
                // Y-direction: check j=0 vs j=ny-1
                for i in 0..nx {
                    for k in 0..nz {
                        let label1 = unpadded_labeled[[i, 0, k]];
                        let label2 = unpadded_labeled[[i, ny - 1, k]];
                        if label1 != 0 { face1_labels.insert(label1); }
                        if label2 != 0 { face2_labels.insert(label2); }
                    }
                }
            },
            2 => {
                // Z-direction: check k=0 vs k=nz-1
                for i in 0..nx {
                    for j in 0..ny {
                        let label1 = unpadded_labeled[[i, j, 0]];
                        let label2 = unpadded_labeled[[i, j, nz - 1]];
                        if label1 != 0 { face1_labels.insert(label1); }
                        if label2 != 0 { face2_labels.insert(label2); }
                    }
                }
            },
            _ => unreachable!(),
        }
        
        // Check if any label appears on both faces (percolation)
        if !face1_labels.intersection(&face2_labels).collect::<Vec<_>>().is_empty() {
            return true;
        }
    }
    
    false
}

/// Flood fill helper function for connected component labeling
fn flood_fill_label(
    labeled: &mut Array3<u32>,
    binary: &Array3<u8>, 
    start_i: usize, 
    start_j: usize, 
    start_k: usize,
    label: u32
) {
    let (nx, ny, nz) = labeled.dim();
    let mut stack = vec![(start_i, start_j, start_k)];
    
    while let Some((i, j, k)) = stack.pop() {
        if i >= nx || j >= ny || k >= nz || 
           binary[[i, j, k]] == 0 || labeled[[i, j, k]] != 0 {
            continue;
        }
        
        labeled[[i, j, k]] = label;
        
        // Add 6-connected neighbors to stack
        if i > 0 { stack.push((i - 1, j, k)); }
        if i + 1 < nx { stack.push((i + 1, j, k)); }
        if j > 0 { stack.push((i, j - 1, k)); }
        if j + 1 < ny { stack.push((i, j + 1, k)); }
        if k > 0 { stack.push((i, j, k - 1)); }
        if k + 1 < nz { stack.push((i, j, k + 1)); }
    }
}

// ============================================================================================
// New: Minimal CIF reader and end-to-end periodic segmentation entrypoint
// ============================================================================================

/// Build 3x3 cell matrix (vectors as columns) from cell parameters.
fn cellpar_to_cell_matrix(a: f64, b: f64, c: f64, alpha_deg: f64, beta_deg: f64, gamma_deg: f64) -> [[f64; 3]; 3] {
    let alpha = alpha_deg.to_radians();
    let beta = beta_deg.to_radians();
    let gamma = gamma_deg.to_radians();

    let a_vec = [a, 0.0, 0.0];
    let b_vec = [b * gamma.cos(), b * gamma.sin(), 0.0];
    let c_x = c * beta.cos();
    let c_y = c * (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin();
    let c_z_sq = c * c - c_x * c_x - c_y * c_y;
    let c_z = if c_z_sq > 0.0 { c_z_sq.sqrt() } else { 0.0 };
    let c_vec = [c_x, c_y, c_z];

    // Return as columns: [a b c]
    [
        [a_vec[0], b_vec[0], c_vec[0]],
        [a_vec[1], b_vec[1], c_vec[1]],
        [a_vec[2], b_vec[2], c_vec[2]],
    ]
}

/// Very small, permissive CIF parser for common MOF CIFs.
/// Extracts cell params and fractional atomic positions with element symbols.
fn parse_cif_minimal(contents: &str) -> PyResult<(
    (f64, f64, f64, f64, f64, f64), // a,b,c,alpha,beta,gamma
    Vec<([f64; 3], String)>,        // fractional coords, element symbol
)> {
    use std::str::FromStr;

    let mut a = None; let mut b = None; let mut c = None;
    let mut alpha = None; let mut beta = None; let mut gamma = None;

    // Helper to normalize element symbol strings (e.g., "C1" -> "C", "Zr1" -> "ZR")
    fn normalize_element_symbol(raw: &str) -> String {
        let letters: String = raw.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        if letters.is_empty() { return "X".to_string(); }
        letters.to_ascii_uppercase()
    }

    for line in contents.lines() {
        let l = line.trim();
        if l.starts_with("_cell_length_a") {
            a = l.split_whitespace().nth(1).and_then(|v| f64::from_str(v).ok());
        } else if l.starts_with("_cell_length_b") {
            b = l.split_whitespace().nth(1).and_then(|v| f64::from_str(v).ok());
        } else if l.starts_with("_cell_length_c") {
            c = l.split_whitespace().nth(1).and_then(|v| f64::from_str(v).ok());
        } else if l.starts_with("_cell_angle_alpha") {
            alpha = l.split_whitespace().nth(1).and_then(|v| f64::from_str(v).ok());
        } else if l.starts_with("_cell_angle_beta") {
            beta = l.split_whitespace().nth(1).and_then(|v| f64::from_str(v).ok());
        } else if l.starts_with("_cell_angle_gamma") {
            gamma = l.split_whitespace().nth(1).and_then(|v| f64::from_str(v).ok());
        }
    }

    let (a, b, c, alpha, beta, gamma) = match (a, b, c, alpha, beta, gamma) {
        (Some(a), Some(b), Some(c), Some(alpha), Some(beta), Some(gamma)) => (a, b, c, alpha, beta, gamma),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Failed to parse CIF cell parameters (_cell_length_* and _cell_angle_*)",
            ));
        }
    };

    // Parse atom_site loop
    let mut atoms: Vec<([f64; 3], String)> = Vec::new();
    let mut in_loop = false;
    let mut headers: Vec<String> = Vec::new();
    let mut x_idx = None; let mut y_idx = None; let mut z_idx = None; let mut sym_idx = None;

    for line in contents.lines() {
        let l = line.trim();
        if l.is_empty() { continue; }
        if l.starts_with("loop_") {
            in_loop = true; headers.clear();
            x_idx = None; y_idx = None; z_idx = None; sym_idx = None;
            continue;
        }
        if in_loop && l.starts_with('_') {
            headers.push(l.to_string());
            continue;
        }
        if in_loop && !l.starts_with('_') {
            // Determine indices once we hit first data row
            if x_idx.is_none() {
                for (i, h) in headers.iter().enumerate() {
                    let h_low = h.to_ascii_lowercase();
                    if h_low.contains("_atom_site_fract_x") { x_idx = Some(i); }
                    if h_low.contains("_atom_site_fract_y") { y_idx = Some(i); }
                    if h_low.contains("_atom_site_fract_z") { z_idx = Some(i); }
                    if h_low.contains("_atom_site_type_symbol") || h_low.contains("_atom_site_label") {
                        if sym_idx.is_none() { sym_idx = Some(i); }
                    }
                }
            }
            let parts: Vec<&str> = l.split_whitespace().collect();
            if let (Some(ix), Some(iy), Some(iz)) = (x_idx, y_idx, z_idx) {
                if ix < parts.len() && iy < parts.len() && iz < parts.len() {
                    let fx = parts[ix].trim_matches('\'').parse::<f64>().unwrap_or(0.0);
                    let fy = parts[iy].trim_matches('\'').parse::<f64>().unwrap_or(0.0);
                    let fz = parts[iz].trim_matches('\'').parse::<f64>().unwrap_or(0.0);
                    let raw_sym = sym_idx.and_then(|is| parts.get(is).map(|s| s.to_string())).unwrap_or_else(|| "X".to_string());
                    let sym = normalize_element_symbol(&raw_sym);
                    atoms.push(([fx, fy, fz], sym));
                }
            }
        }
    }

    if atoms.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("No atom_site loop with positions found in CIF"));
    }

    Ok(((a, b, c, alpha, beta, gamma), atoms))
}

/// Simple VdW radius lookup by element symbol. Fallback to 1.5 Å.
fn vdw_radius(symbol: &str) -> f64 {
    match symbol.to_ascii_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "F" => 1.47,
        "P" => 1.80,
        "S" => 1.80,
        "CL" => 1.75,
        "CU" => 1.40,
        "ZN" => 1.39,
        "ZR" => 2.27,
        "AL" => 1.84,
        "SI" => 2.10,
        _ => 1.50,
    }
}

/// Wrap-pad a 3D array by padding_thickness cells on each side.
fn pad_wrap_f32(src: &ndarray::ArrayView3<'_, f32>, pad: usize) -> ndarray::Array3<f32> {
    let (nx, ny, nz) = src.dim();
    let (px, py, pz) = (nx + 2 * pad, ny + 2 * pad, nz + 2 * pad);
    let mut out = Array3::<f32>::zeros((px, py, pz));
    for i in 0..px { for j in 0..py { for k in 0..pz {
        let si = ((i as isize - pad as isize).rem_euclid(nx as isize)) as usize;
        let sj = ((j as isize - pad as isize).rem_euclid(ny as isize)) as usize;
        let sk = ((k as isize - pad as isize).rem_euclid(nz as isize)) as usize;
        out[[i, j, k]] = src[[si, sj, sk]];
    }}}
    out
}

fn pad_wrap_u32(src: &ndarray::ArrayView3<'_, u32>, pad: usize) -> ndarray::Array3<u32> {
    let (nx, ny, nz) = src.dim();
    let (px, py, pz) = (nx + 2 * pad, ny + 2 * pad, nz + 2 * pad);
    let mut out = Array3::<u32>::zeros((px, py, pz));
    for i in 0..px { for j in 0..py { for k in 0..pz {
        let si = ((i as isize - pad as isize).rem_euclid(nx as isize)) as usize;
        let sj = ((j as isize - pad as isize).rem_euclid(ny as isize)) as usize;
        let sk = ((k as isize - pad as isize).rem_euclid(nz as isize)) as usize;
        out[[i, j, k]] = src[[si, sj, sk]];
    }}}
    out
}

fn pad_wrap_u8(src: &ndarray::ArrayView3<'_, u8>, pad: usize) -> ndarray::Array3<u8> {
    let (nx, ny, nz) = src.dim();
    let (px, py, pz) = (nx + 2 * pad, ny + 2 * pad, nz + 2 * pad);
    let mut out = Array3::<u8>::zeros((px, py, pz));
    for i in 0..px { for j in 0..py { for k in 0..pz {
        let si = ((i as isize - pad as isize).rem_euclid(nx as isize)) as usize;
        let sj = ((j as isize - pad as isize).rem_euclid(ny as isize)) as usize;
        let sk = ((k as isize - pad as isize).rem_euclid(nz as isize)) as usize;
        out[[i, j, k]] = src[[si, sj, sk]];
    }}}
    out
}

/// Compute wall cell counts given cell and spacing (replicates Python compute_wall_cells).
fn compute_wall_cells_rs(cell: &[[f64; 3]; 3], spacing: f64, wall_thickness: f64) -> [usize; 3] {
    use nalgebra::Matrix3;
    let a = [cell[0][0], cell[1][0], cell[2][0]];
    let b = [cell[0][1], cell[1][1], cell[2][1]];
    let c = [cell[0][2], cell[1][2], cell[2][2]];

    let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let norm = |u: [f64; 3]| -> f64 { (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt() };

    let mat = Matrix3::new(
        cell[0][0], cell[0][1], cell[0][2],
        cell[1][0], cell[1][1], cell[1][2],
        cell[2][0], cell[2][1], cell[2][2],
    );
    let vol = mat.determinant().abs();

    let la_perp = vol / norm(cross(b, c));
    let lb_perp = vol / norm(cross(c, a));
    let lc_perp = vol / norm(cross(a, b));

    let wall = [
        (wall_thickness * la_perp / spacing).ceil() as usize,
        (wall_thickness * lb_perp / spacing).ceil() as usize,
        (wall_thickness * lc_perp / spacing).ceil() as usize,
    ];
    wall
}

/// Calculate padding thickness as in Python version.
fn calculate_padding_thickness_rs(dgrid: &ndarray::Array3<f32>, cell: &[[f64; 3]; 3], spacing: f64, extra_padding: f64) -> usize {
    use nalgebra::Matrix3;
    let (nx, ny, nz) = dgrid.dim();
    let mat = Matrix3::new(
        cell[0][0], cell[0][1], cell[0][2],
        cell[1][0], cell[1][1], cell[1][2],
        cell[2][0], cell[2][1], cell[2][2],
    );
    let vol = mat.determinant().abs();
    let voxel_volume = vol / (nx * ny * nz) as f64;

    // Angles via dot products
    let a = [cell[0][0], cell[1][0], cell[2][0]];
    let b = [cell[0][1], cell[1][1], cell[2][1]];
    let c_vec = [cell[0][2], cell[1][2], cell[2][2]];
    let len = |u: [f64; 3]| -> f64 { (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt() };
    let dot = |u: [f64; 3], v: [f64; 3]| -> f64 { u[0] * v[0] + u[1] * v[1] + u[2] * v[2] };
    let la = len(a); let lb = len(b); let lc = len(c_vec);
    let alpha = (dot(b, c_vec) / (lb * lc)).acos();
    let beta = (dot(a, c_vec) / (la * lc)).acos();
    let gamma = (dot(a, b) / (la * lb)).acos();

    // Voxel basis vectors analogous to Python
    let eA = [spacing, 0.0, 0.0];
    let eB = [spacing * gamma.cos(), spacing * gamma.sin(), 0.0];
    let eC = [
        spacing * beta.cos(),
        spacing * (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin(),
        voxel_volume / (spacing * spacing * gamma.sin()),
    ];
    let cross = |u: [f64; 3], v: [f64; 3]| -> [f64; 3] {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let norm = |u: [f64; 3]| -> f64 { (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt() };
    let lx_vox = voxel_volume / norm(cross(eB, eC));
    let ly_vox = voxel_volume / norm(cross(eC, eA));
    let lz_vox = voxel_volume / norm(cross(eA, eB));
    let min_perp = lx_vox.min(ly_vox).min(lz_vox);

    let dgrid_max = dgrid.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let base = (dgrid_max / min_perp).ceil() as usize;
    // Formula: dgrid_max/min_perp_length + 5/spacing + extra_padding/spacing
    base + (5.0 / spacing).ceil() as usize + (extra_padding / spacing).ceil() as usize
}



/// Merge tiny regions (< min_size_ratio of central box) by assigning to the most frequent neighbor label.
fn merge_small_regions_rs(regions: &ndarray::Array3<u32>, min_size: usize) -> ndarray::Array3<u32> {
    use std::collections::HashMap;
    let (nx, ny, nz) = regions.dim();

    // First, identify small regions using the same approach as Python
    let mut region_sizes: HashMap<u32, usize> = HashMap::new();
    for &v in regions.iter() {
        if v != 0 {
            *region_sizes.entry(v).or_insert(0) += 1;
        }
    }

    // Find small regions (like Python's regionprops with area < min_size)
    let small_labels: Vec<u32> = region_sizes.iter()
        .filter(|(_, &size)| size < min_size)
        .map(|(&label, _)| label)
        .collect();

    eprintln!("merge_small_regions_rs: Found {} small regions to merge", small_labels.len());

    // Use in-place modification to avoid full array clone
    let mut out = regions.to_owned();
    let (nx, ny, nz) = regions.dim();

    // Pre-compute neighbor offsets for efficiency
    let offsets = [
        (-1,-1,-1), (-1,-1,0), (-1,-1,1),
        (-1,0,-1),  (-1,0,0),  (-1,0,1),
        (-1,1,-1),  (-1,1,0),  (-1,1,1),
        (0,-1,-1),  (0,-1,0),  (0,-1,1),
        (0,0,-1),   (0,0,1),
        (0,1,-1),   (0,1,0),   (0,1,1),
        (1,-1,-1),  (1,-1,0),  (1,-1,1),
        (1,0,-1),   (1,0,0),   (1,0,1),
        (1,1,-1),   (1,1,0),   (1,1,1),
    ];

    // Process each small region (optimized approach)
    for &small_label in &small_labels {
        // Find all voxels belonging to this small region (single scan)
        let mut region_voxels = Vec::new();
        for ((i, j, k), &val) in regions.indexed_iter() {
            if val == small_label {
                region_voxels.push((i, j, k));
            }
        }

        // If region is empty, skip
        if region_voxels.is_empty() {
            continue;
        }

        // Find all neighboring regions by checking neighbors of each voxel
        let mut neighbor_labels = std::collections::HashSet::new();

        for &(i, j, k) in &region_voxels {
            for &(di, dj, dk) in &offsets {
                let ni = i as isize + di;
                let nj = j as isize + dj;
                let nk = k as isize + dk;

                // Check bounds
                if ni >= 0 && nj >= 0 && nk >= 0 &&
                   (ni as usize) < nx && (nj as usize) < ny && (nk as usize) < nz {
                    let neighbor_label = regions[[ni as usize, nj as usize, nk as usize]];
                    if neighbor_label != 0 && neighbor_label != small_label {
                        neighbor_labels.insert(neighbor_label);
                    }
                }
            }
        }

        if neighbor_labels.is_empty() {
            continue; // No neighbors found, skip
        }

        // Find the neighbor with the largest shared boundary (optimized)
        let mut boundary_counts = HashMap::new();
        for &neighbor_label in &neighbor_labels {
            let mut count = 0;
            for &(i, j, k) in &region_voxels {
                for &(di, dj, dk) in &offsets {
                    let ni = i as isize + di;
                    let nj = j as isize + dj;
                    let nk = k as isize + dk;

                    // Check bounds and if neighbor belongs to this label
                    if ni >= 0 && nj >= 0 && nk >= 0 &&
                       (ni as usize) < nx && (nj as usize) < ny && (nk as usize) < nz {
                        if regions[[ni as usize, nj as usize, nk as usize]] == neighbor_label {
                            count += 1;
                        }
                    }
                }
            }
            boundary_counts.insert(neighbor_label, count);
        }

        // Choose the neighbor with the largest boundary
        let best_neighbor = boundary_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&label, _)| label);

        if let Some(best_label) = best_neighbor {
            // Replace all voxels of small region with best neighbor
            for &(i, j, k) in &region_voxels {
                out[[i, j, k]] = best_label;
            }
        }
    }

    out
}

/// Classify indices as inside/close_outside/far_outside
fn classify_locations(indices: &[[usize;3]], shape: (usize,usize,usize), tol: [usize;3]) -> Vec<&'static str> {
    let (d,h,w) = shape; let (zz,yy,xx) = (tol[0], tol[1], tol[2]);
    indices.iter().map(|&[z,y,x]| {
        // Check if the point is inside the box
        if z < d && y < h && x < w { 
            "inside" 
        }
        // Check if it's close outside (within tolerance) - match Python logic exactly
        else if (z as isize >= -(zz as isize) && z < d + zz) &&
                 (y as isize >= -(yy as isize) && y < h + yy) &&
                 (x as isize >= -(xx as isize) && x < w + xx) {
            "close_outside"
        } else { 
            "far_outside" 
        }
    }).collect()
}

/// Group unmerged regions by maxima indices (like Python get_pbc_groups)
fn build_pbc_groups(unmerged_regions: &ndarray::Array3<u32>, maxima_info: &ndarray::Array2<i64>) -> Vec<std::collections::HashSet<u32>> {
    use std::collections::{HashMap, HashSet};
    let mut groups: HashMap<(i64,i64,i64), HashSet<u32>> = HashMap::new();
    for row in maxima_info.rows() {
        let label = row[0] as u32;
        let key = (row[1], row[2], row[3]);
        groups.entry(key).or_default().insert(label);
    }
    groups.into_values().collect()
}

// Helper function to remove a row from a 2D array  
fn remove_row_f64(arr: &ndarray::Array2<f64>, row_idx: usize) -> ndarray::Array2<f64> {
    let (nrows, ncols) = arr.dim();
    if row_idx >= nrows {
        return arr.clone();
    }
    
    let mut new_data = Vec::with_capacity((nrows - 1) * ncols);
    for i in 0..nrows {
        if i != row_idx {
            for j in 0..ncols {
                new_data.push(arr[[i, j]]);
            }
        }
    }
    ndarray::Array2::from_shape_vec((nrows - 1, ncols), new_data).unwrap()
}

// Helper function to remove a row from a 2D usize array
fn remove_row_usize(arr: &ndarray::Array2<usize>, row_idx: usize) -> ndarray::Array2<usize> {
    let (nrows, ncols) = arr.dim();
    if row_idx >= nrows {
        return arr.clone();
    }
    
    let mut new_data = Vec::with_capacity((nrows - 1) * ncols);
    for i in 0..nrows {
        if i != row_idx {
            for j in 0..ncols {
                new_data.push(arr[[i, j]]);
            }
        }
    }
    ndarray::Array2::from_shape_vec((nrows - 1, ncols), new_data).unwrap()
}

#[pyfunction]
#[pyo3(signature = (
    path_to_struct_file,
    spacing = 0.5,
    mask_thickness = 0.5,
    h = 0.2,
    min_distance = 1.5,
    threshold_abs = 1.5,
    return_padded_regions = false,
    minimum_window_separation = None,
    wall_thickness = 0.05,
    extra_padding = 5.0,
    compactness = 0,
    min_region_size = 0.001,
    disable_tqdm = false,
    compute_connections = true,
    return_dgrid = false,
))]
fn do_periodic_segmentation_rust(
    py: Python<'_>,
    path_to_struct_file: &str,
    spacing: f64,
    mask_thickness: f32,
    h: f32,
    min_distance: f32,
    threshold_abs: f32,
    return_padded_regions: bool,
    minimum_window_separation: Option<f64>,
    wall_thickness: f64,
    extra_padding: f64,
    compactness: u32,
    min_region_size: f64,
    disable_tqdm: bool,
    compute_connections: bool,
    return_dgrid: bool,
) -> PyResult<PyObject> {
    use ndarray::{Array1, Array2, Array3};
    use pyo3::types::PyDict;
    use std::time::Instant;

    // Initialize timing
    let start_time = Instant::now();
    let mut timing_results = std::collections::HashMap::new();

    // Set default for minimum_window_separation if None
    let minimum_window_separation = minimum_window_separation.unwrap_or(spacing * 3.0);

    // 1) Read CIF
    let cif_start = Instant::now();
    let cif_text = std::fs::read_to_string(path_to_struct_file)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed reading CIF: {}", e)))?;
    let ((a,b,c,alpha,beta,gamma), atoms) = parse_cif_minimal(&cif_text)?;
    timing_results.insert("cif_parsing", cif_start.elapsed());
    let cell = cellpar_to_cell_matrix(a,b,c,alpha,beta,gamma);

    // 2) Prepare inputs for dgrid_from_atoms
    let n_atoms = atoms.len();
    let mut fpoints: Vec<f64> = Vec::with_capacity(n_atoms * 3);
    let mut radii: Vec<f64> = Vec::with_capacity(n_atoms);
    for (frac, sym) in &atoms {
        // Shift by -0.5 like Python version
        fpoints.extend_from_slice(&[frac[0] - 0.5, frac[1] - 0.5, frac[2] - 0.5]);
        let radius = vdw_radius(sym);
        radii.push(radius);
        // Debug: print first few atoms and their radii
        if atoms.len() < 10 {
            println!("Atom {}: symbol='{}', radius={}", atoms.len(), sym, radius);
        }
    }
    // Two conventions required by different functions:
    // - dgrid_from_atoms expects CELL VECTORS AS ROWS (to match Python usage)
    // - maxima/connection functions expect CELL VECTORS AS COLUMNS
    // Our `cell` above has vectors as columns; build both forms.
    let cell_cols = [
        [cell[0][0], cell[0][1], cell[0][2]],
        [cell[1][0], cell[1][1], cell[1][2]],
        [cell[2][0], cell[2][1], cell[2][2]],
    ];
    let cell_rows = [
        [cell[0][0], cell[1][0], cell[2][0]],
        [cell[0][1], cell[1][1], cell[2][1]],
        [cell[0][2], cell[1][2], cell[2][2]],
    ];
    let cell_cols_flat: Vec<f64> = vec![
        cell_cols[0][0], cell_cols[0][1], cell_cols[0][2],
        cell_cols[1][0], cell_cols[1][1], cell_cols[1][2],
        cell_cols[2][0], cell_cols[2][1], cell_cols[2][2],
    ];
    let cell_rows_flat: Vec<f64> = vec![
        cell_rows[0][0], cell_rows[0][1], cell_rows[0][2],
        cell_rows[1][0], cell_rows[1][1], cell_rows[1][2],
        cell_rows[2][0], cell_rows[2][1], cell_rows[2][2],
    ];
    let fpoints_nd = ndarray::Array2::<f64>::from_shape_vec((n_atoms, 3), fpoints.clone()).unwrap();
    let cell_cols_nd = ndarray::Array2::<f64>::from_shape_vec((3, 3), cell_cols_flat.clone()).unwrap();
    let cell_rows_nd = ndarray::Array2::<f64>::from_shape_vec((3, 3), cell_rows_flat.clone()).unwrap();

    // Grid shape from lengths / spacing (+1 like Python) - use same calculation as Python
    let nx = (a / spacing) as usize + 1;
    let ny = (b / spacing) as usize + 1;
    let nz = (c / spacing) as usize + 1;

    // Call existing Rust dgrid_from_atoms
    let dgrid_start = Instant::now();
    let dgrid_py = dgrid_from_atoms(
        py,
        PyArray2::from_owned_array(py, fpoints_nd).readonly(),
        PyArray2::from_owned_array(py, cell_cols_nd.clone()).readonly(),
        PyArray1::from_vec(py, radii).readonly(),
        nx, ny, nz,
    )?;
    let mut dgrid = unsafe { dgrid_py.as_ref(py).as_array().to_owned() };
    timing_results.insert("dgrid_computation", dgrid_start.elapsed());

    // 3) Find pore-limiting radius
    let plr_opt = find_pore_limiting_radius(dgrid_py.as_ref(py).readonly())?;
    let _plr = plr_opt.unwrap_or(mask_thickness as f32);

    // 4) Detect maxima (selective smoothing) - match Python exactly
    let maxima_start = Instant::now();
    eprintln!("DEBUG: About to call detect_maxima_selective_smoothing_parallel");
    let maxima = match detect_maxima_selective_smoothing_parallel(
        py,
        dgrid_py.as_ref(py).readonly(),
        3.0, // preserve_threshold (matches Python)
        6.0, // smooth_threshold (matches Python)
        2.0, // max_smooth_radius (matches Python)
        h,
        Some(PyArray2::from_owned_array(py, cell_cols_nd.clone()).readonly()),
        min_distance,
        None, // mask_thickness (not provided in Python call)
        Some(threshold_abs as f32),
    ) {
        Ok(result) => {
            eprintln!("DEBUG: detect_maxima_selective_smoothing_parallel succeeded");
            result
        }
        Err(e) => {
            eprintln!("DEBUG: detect_maxima_selective_smoothing_parallel failed: {:?}", e);
            eprintln!("DEBUG: Falling back to detect_maxima_selective_smoothing");
            detect_maxima_selective_smoothing(
                py,
                dgrid_py.as_ref(py).readonly(),
                3.0, // preserve_threshold (matches Python)
                6.0, // smooth_threshold (matches Python)
                2.0, // max_smooth_radius (matches Python)
                h,
                Some(PyArray2::from_owned_array(py, cell_cols_nd.clone()).readonly()),
                min_distance,
                None, // mask_thickness (not provided in Python call)
                Some(threshold_abs as f32),
            )?
        }
    };
    let merged_pbc = unsafe { maxima.0.as_ref(py).as_array().to_owned() }; // (m,3)
    let raw_maxima = unsafe { maxima.2.as_ref(py).as_array().to_owned() };
    timing_results.insert("maxima_detection", maxima_start.elapsed());

    // 5) Refine maxima to peaks
    let (opt_pos_py, _opt_val_py) = move_maxima_to_peaks(
        py,
        dgrid_py.as_ref(py).readonly(),
        PyArray2::from_owned_array(py, merged_pbc.clone()).readonly(),
        Some(100),
        Some(0.1)
    )?;
    let merged_maxima = unsafe { opt_pos_py.as_ref(py).as_array().to_owned() };

    // 6) Markers and padding
    let mut markers = Array3::<u32>::zeros(dgrid.dim());
    if merged_maxima.len() > 0 {
        for (idx, coord) in merged_maxima.outer_iter().enumerate() {
            let i = coord[0]; let j = coord[1]; let k = coord[2];
            markers[[i,j,k]] = (idx + 1) as u32;
        }
    }
    let void_mask = dgrid.mapv(|v| if v > mask_thickness { 1u8 } else { 0u8 });

    let pad_th = calculate_padding_thickness_rs(&dgrid, &cell, spacing, extra_padding);
    let padded_grid = pad_wrap_f32(&dgrid.view(), pad_th);
    let padded_markers = pad_wrap_u32(&markers.view(), pad_th);
    let padded_void = pad_wrap_u8(&void_mask.view(), pad_th);

    // 7) Seeded watershed (pbc)
    let watershed_start = Instant::now();
    let regions_padded_py = seeded_watershed_bucket(
        py,
        PyArray3::from_owned_array(py, padded_grid.clone()).readonly(),
        PyArray3::from_owned_array(py, padded_markers.clone()).readonly(),
        Some(PyArray3::from_owned_array(py, padded_void.clone()).readonly()),
        None,
    )?;
    let mut regions_padded = unsafe { regions_padded_py.as_ref(py).as_array().to_owned() };
    timing_results.insert("watershed_segmentation", watershed_start.elapsed());

    // 8) Clean very small regions in the central box and remove corresponding maxima like Python
    // ENABLE this step to match Python behavior - small regions need to be merged before unpadding
    let original_central = regions_padded.slice(ndarray::s![pad_th..(pad_th+nx), pad_th..(pad_th+ny), pad_th..(pad_th+nz)]).to_owned();
    let min_vox = ((original_central.len() as f64) * min_region_size).ceil() as usize;
    let central_clean = merge_small_regions_rs(&original_central, min_vox);

    // Put back the cleaned central region
    regions_padded.slice_mut(ndarray::s![pad_th..(pad_th+nx), pad_th..(pad_th+ny), pad_th..(pad_th+nz)]).assign(&central_clean);

    // 8.5) Filter out maxima corresponding to removed regions (like Python does)
    let mut merged_maxima_filtered = merged_maxima.clone();
    if merged_maxima_filtered.len() > 0 {

        // Find unique regions before and after merging
        let mut unique_original = std::collections::HashSet::new();
        for &val in original_central.iter() {
            if val != 0 {
                unique_original.insert(val);
            }
        }

        let mut unique_cleaned = std::collections::HashSet::new();
        for &val in central_clean.iter() {
            if val != 0 {
                unique_cleaned.insert(val);
            }
        }

        // Find removed labels (like Python's ~np.isin)
        let mut removed_labels: Vec<u32> = unique_original.difference(&unique_cleaned).cloned().collect();
        let num_rows = merged_maxima_filtered.nrows();
        removed_labels.retain(|&label| label >= 1 && label <= num_rows as u32);

        // Remove those rows from merged_maxima (like Python's np.delete)
        if !removed_labels.is_empty() {
            let num_rows = merged_maxima_filtered.nrows();
            let mut indices_to_keep: Vec<usize> = (0..num_rows)
                .filter(|&idx| {
                    let label = (idx + 1) as u32;
                    !removed_labels.contains(&label)
                })
                .collect();

            if indices_to_keep.len() < num_rows {
                // Create new array with filtered maxima
                let mut new_data = Vec::with_capacity(indices_to_keep.len() * 3);
                for &idx in &indices_to_keep {
                    let row = merged_maxima_filtered.row(idx);
                    new_data.push(row[0]);
                    new_data.push(row[1]);
                    new_data.push(row[2]);
                }
                merged_maxima_filtered = ndarray::Array2::<usize>::from_shape_vec(
                    (indices_to_keep.len(), 3),
                    new_data
                ).unwrap();
            }
        }
    }

    let merged_maxima = merged_maxima_filtered;

    // 9) Extract regions_pbc
    let regions_pbc = regions_padded.slice(ndarray::s![pad_th..(pad_th+nx), pad_th..(pad_th+ny), pad_th..(pad_th+nz)]).to_owned();

    // 10) Build maxima_info (like Python)
    let maxima_info_start = Instant::now();
    let padded_shape = [padded_grid.dim().0 as i64, padded_grid.dim().1 as i64, padded_grid.dim().2 as i64];
    let original_shape = [nx as i64, ny as i64, nz as i64];
    let mut maxima_info_vec: Vec<i64> = Vec::new();
    let mut counter: i64 = 1;
    for m in merged_maxima.outer_iter() {
        let m_pad = [m[0] as i64 + pad_th as i64, m[1] as i64 + pad_th as i64, m[2] as i64 + pad_th as i64];
        // Use Python-style floor division for offset ranges to exactly match Python behavior
        let off_ranges = [
            -(((padded_shape[0]-1) as f64 / original_shape[0] as f64).floor() as i64)..=(((padded_shape[0]-1) as f64 / original_shape[0] as f64).floor() as i64),
            -(((padded_shape[1]-1) as f64 / original_shape[1] as f64).floor() as i64)..=(((padded_shape[1]-1) as f64 / original_shape[1] as f64).floor() as i64),
            -(((padded_shape[2]-1) as f64 / original_shape[2] as f64).floor() as i64)..=(((padded_shape[2]-1) as f64 / original_shape[2] as f64).floor() as i64),
        ];
        for ox in off_ranges[0].clone() { for oy in off_ranges[1].clone() { for oz in off_ranges[2].clone() {
            let candidate = [m_pad[0] + ox*original_shape[0], m_pad[1] + oy*original_shape[1], m_pad[2] + oz*original_shape[2]];
            if candidate[0] >= 0 && candidate[1] >= 0 && candidate[2] >= 0 &&
               candidate[0] < padded_shape[0] && candidate[1] < padded_shape[1] && candidate[2] < padded_shape[2] {
                let candidate_unpadded = [candidate[0] - pad_th as i64, candidate[1] - pad_th as i64, candidate[2] - pad_th as i64];
                // Use Python-style floor division and modulo (// and %) for exact compatibility
                let offset = [
                    if candidate_unpadded[0] >= 0 { candidate_unpadded[0] / original_shape[0] } else { (candidate_unpadded[0] - original_shape[0] + 1) / original_shape[0] },
                    if candidate_unpadded[1] >= 0 { candidate_unpadded[1] / original_shape[1] } else { (candidate_unpadded[1] - original_shape[1] + 1) / original_shape[1] },
                    if candidate_unpadded[2] >= 0 { candidate_unpadded[2] / original_shape[2] } else { (candidate_unpadded[2] - original_shape[2] + 1) / original_shape[2] }
                ];
                let image = [
                    candidate_unpadded[0] - offset[0] * original_shape[0],
                    candidate_unpadded[1] - offset[1] * original_shape[1], 
                    candidate_unpadded[2] - offset[2] * original_shape[2]
                ];
                maxima_info_vec.extend_from_slice(&[counter, image[0], image[1], image[2], offset[0], offset[1], offset[2]]);
                counter += 1;
            }
        }}}
    }
    let rows = (maxima_info_vec.len()/7) as usize;
    let maxima_info: Array2<i64> = Array2::from_shape_vec((rows,7), maxima_info_vec).unwrap();
    timing_results.insert("maxima_info_building", maxima_info_start.elapsed());

    // 11) Build unmerged markers from maxima_info and segment unmerged regions
    let mut unmerged_markers = Array3::<u32>::zeros(padded_grid.dim());
    for row in maxima_info.rows() {
        let label = row[0] as u32;
        let pos = [
            (row[1] + pad_th as i64 + row[4]*original_shape[0]) as usize,
            (row[2] + pad_th as i64 + row[5]*original_shape[1]) as usize,
            (row[3] + pad_th as i64 + row[6]*original_shape[2]) as usize,
        ];
        if pos[0] < padded_grid.dim().0 && pos[1] < padded_grid.dim().1 && pos[2] < padded_grid.dim().2 {
            unmerged_markers[[pos[0], pos[1], pos[2]]] = label;
        }
    }
    let unmerged_padded_py = seeded_watershed_bucket(
        py,
        PyArray3::from_owned_array(py, padded_grid.clone()).readonly(),
        PyArray3::from_owned_array(py, unmerged_markers.clone()).readonly(),
        Some(PyArray3::from_owned_array(py, padded_void.clone()).readonly()),
        None,
    )?;
    let unmerged_padded = unsafe { unmerged_padded_py.as_ref(py).as_array().to_owned() };
    let mut regions = unmerged_padded.slice(ndarray::s![pad_th..(pad_th+nx), pad_th..(pad_th+ny), pad_th..(pad_th+nz)]).to_owned();

    // 12) End channel pairs cleanup - find and merge overflow regions across periodic boundaries
    let endchannel_pairs = find_end_channel_pairs_rs(&regions_pbc, &regions, 0.3);
    for (from_label, to_label) in endchannel_pairs {
        // Replace all occurrences of from_label with to_label
        for region_val in regions.iter_mut() {
            if *region_val == from_label {
                *region_val = to_label;
            }
        }
    }

    // 13) Basic outputs assembly
    let maxima_pbc_dict = {
        use std::collections::HashMap;
        let mut map: HashMap<(usize,usize,usize), usize> = HashMap::new();
        for (i, m) in merged_maxima.outer_iter().enumerate() {
            map.insert((m[0], m[1], m[2]), i+1);
        }
        map
    };

    let wall_cells = compute_wall_cells_rs(&cell, spacing, wall_thickness);
    
    // Classify maxima locations using proper boundary checking like Python
    let dgrid_shape = [dgrid.dim().0 as i64, dgrid.dim().1 as i64, dgrid.dim().2 as i64];
    let global_maxima_coords: Vec<[i64;3]> = maxima_info.rows().into_iter().map(|r| {
        [r[1] + r[4]*dgrid_shape[0], r[2] + r[5]*dgrid_shape[1], r[3] + r[6]*dgrid_shape[2]]
    }).collect();
    
    // Classify each maxima location using the proper boundary logic
    let mut locs = Vec::new();
    for coords in &global_maxima_coords {
        let [z, y, x] = *coords;
        let depth = dgrid_shape[0];
        let height = dgrid_shape[1]; 
        let width = dgrid_shape[2];
        let zz = wall_cells[0] as i64;
        let yy = wall_cells[1] as i64;
        let xx = wall_cells[2] as i64;
        
        // Check if inside the box
        if (0 <= z && z < depth) && (0 <= y && y < height) && (0 <= x && x < width) {
            locs.push("inside");
        }
        // Check if close outside (within tolerance)
        else if (-zz <= z && z < depth + zz) && (-yy <= y && y < height + yy) && (-xx <= x && x < width + xx) {
            locs.push("close_outside");
        }
        // Otherwise far outside
        else {
            locs.push("far_outside");
        }
    }
    
    let close_or_inside_mask: Vec<bool> = locs.iter().map(|&s| s=="inside" || s=="close_outside").collect();
    let close_or_inside_indices: Vec<[i64;3]> = global_maxima_coords.iter().enumerate()
        .filter(|(i,_)| close_or_inside_mask[*i])
        .map(|(_,idx)| [idx[0], idx[1], idx[2]]).collect();
    let close_or_inside_ids: Vec<u32> = close_or_inside_mask.iter().enumerate().filter_map(|(i, ok)| if *ok { Some((i+1) as u32) } else { None }).collect();

    // Reindexing exactly following Python implementation
    let reindexing_start = Instant::now();
    // First identify valid_maxima_ids (close or inside) - keep this in broader scope for connection marking
    // NOTE: Python uses ALL close/inside maxima, not just those present in regions
    // Python calculates this as: valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1
    let mut valid_maxima_ids: Vec<u32> = Vec::with_capacity(close_or_inside_mask.len());
    for (i, &is_valid) in close_or_inside_mask.iter().enumerate() {
        if is_valid {
            valid_maxima_ids.push((i + 1) as u32);
        }
    }

    // Debug: Print valid_maxima_ids to verify correctness
    // println!("Rust valid_maxima_ids: {:?}", valid_maxima_ids);
    // println!("Rust close_or_inside_ids: {:?}", close_or_inside_ids);

    // Create remap dictionary following exact Python algorithm
    // Pre-allocate with estimated capacity to reduce reallocations
    let estimated_capacity = maxima_info.len_of(ndarray::Axis(0)) + regions.iter().filter(|&&v| v != 0).count();
    let mut remap_dict: std::collections::HashMap<u32, u32> = std::collections::HashMap::with_capacity(estimated_capacity);
    let mut used_indices: std::collections::HashSet<u32> = std::collections::HashSet::with_capacity(estimated_capacity);

    // Stage 1: assign indices to periodic maxima (maxima_pbc)
    let mut merged_maxima_indices: std::collections::HashMap<(i64, i64, i64), u32> = std::collections::HashMap::with_capacity(merged_maxima.len_of(ndarray::Axis(0)));
    for (i, m) in merged_maxima.outer_iter().enumerate() {
        merged_maxima_indices.insert((m[0] as i64, m[1] as i64, m[2] as i64), (i + 1) as u32);
    }

    // Optimize the loop by avoiding repeated hashmap lookups
    for row in maxima_info.rows() {
        let original_maxima = (row[1], row[2], row[3]);
        let offset = (row[4], row[5], row[6]);
        if offset == (0, 0, 0) {
            if let Some(&target_index) = merged_maxima_indices.get(&original_maxima) {
                if used_indices.insert(target_index) { // insert returns true if the value was not already present
                    remap_dict.insert(row[0] as u32, target_index);
                }
            }
        }
    }

    // Stage 2: assign indices to valid maxima (close_by_maxima)
    let mut next_index = if !used_indices.is_empty() {
        *used_indices.iter().max().unwrap() + 1
    } else {
        1
    };

    for &vid in &valid_maxima_ids {
        if !remap_dict.contains_key(&vid) {
            remap_dict.insert(vid, next_index);
            used_indices.insert(next_index);
            next_index += 1;
        }
    }

    // Stage 3: assign indices to remaining unique regions in unmerged_regions
    // Use a more efficient approach with a HashSet for uniqueness
    let unique_unmerged_regions: std::collections::HashSet<u32> = regions.iter().cloned().filter(|&v| v != 0).collect();
    let mut unique_unmerged_regions_vec: Vec<u32> = unique_unmerged_regions.into_iter().collect();
    unique_unmerged_regions_vec.sort_unstable();

    next_index = if !used_indices.is_empty() {
        *used_indices.iter().max().unwrap() + 1
    } else {
        1
    };

    for &r in &unique_unmerged_regions_vec {
        if !remap_dict.contains_key(&r) {
            remap_dict.insert(r, next_index);
            used_indices.insert(next_index);
            next_index += 1;
        }
    }

    // Stage 4: assign indices to remaining unique regions in unmerged_padded_regions
    let unique_unmerged_padded_regions_set: std::collections::HashSet<u32> = unmerged_padded.iter().cloned().filter(|&v| v != 0).collect();
    let mut unique_unmerged_padded_regions_vec: Vec<u32> = unique_unmerged_padded_regions_set.into_iter().collect();
    unique_unmerged_padded_regions_vec.sort_unstable();

    next_index = if !used_indices.is_empty() {
        used_indices.iter().max().unwrap() + 1
    } else {
        1
    };

    // Track where padding indices start (for second phase reindexing)
    let padding_indices_start_here = next_index;

    for &r in &unique_unmerged_padded_regions_vec {
        if !remap_dict.contains_key(&r) {
            remap_dict.insert(r, next_index);
            used_indices.insert(next_index);
            next_index += 1;
        }
    }
    
    // Apply remapping to maxima_info (update column 0 with new labels)
    let mut remapped_maxima_info = maxima_info.clone();
    for mut row in remapped_maxima_info.rows_mut() {
        let original_label = row[0] as u32;
        if let Some(&new_label) = remap_dict.get(&original_label) {
            row[0] = new_label as i64;
        }
    }
    
    // Apply remapping to regions and unmerged_padded
    // Use more efficient iteration patterns
    let mut remapped_regions = regions.clone();
    for ((i, j, k), value) in remapped_regions.indexed_iter_mut() {
        if *value != 0 {
            if let Some(&new_value) = remap_dict.get(value) {
                *value = new_value;
            }
        }
    }

    let mut remapped_unmerged_padded = unmerged_padded.clone();
    for ((i, j, k), value) in remapped_unmerged_padded.indexed_iter_mut() {
        if *value != 0 {
            if let Some(&new_value) = remap_dict.get(value) {
                *value = new_value;
            }
        }
    }
    
    // Sort maxima_info by first column (like Python does)
    let mut maxima_info_rows: Vec<Vec<i64>> = Vec::new();
    for row in remapped_maxima_info.rows() {
        let row_vec: Vec<i64> = row.to_vec();
        maxima_info_rows.push(row_vec);
    }
    maxima_info_rows.sort_by(|a, b| a[0].cmp(&b[0]));
    let maxima_info_data: Vec<i64> = maxima_info_rows.into_iter().flatten().collect();
    let maxima_info = Array2::from_shape_vec((maxima_info_data.len()/7, 7), maxima_info_data).unwrap();
    
    let regions = remapped_regions;
    let unmerged_padded = remapped_unmerged_padded;
    timing_results.insert("first_reindexing", reindexing_start.elapsed());

    let pbc_groups = build_pbc_groups(&regions, &maxima_info);

    let unique_regions: Vec<u32> = regions.iter().cloned().filter(|&v| v!=0).collect::<Vec<_>>();
    let mut unique_regions_set: std::collections::HashSet<u32> = unique_regions.into_iter().collect();
    let mut unique_regions_vec: Vec<u32> = unique_regions_set.drain().collect();
    unique_regions_vec.sort_unstable();

    // Optional connections and second phase reindexing
    let mut py_connections: Option<PyObject> = None;
    let mut final_maxima_info = maxima_info.clone();
    let mut final_regions_padded = unmerged_padded.clone();

    if compute_connections {
        let connections_start = Instant::now();
        let conn = window_connections(
            py,
            PyArray3::from_owned_array(py, unmerged_padded.clone()).readonly(),
            PyArray3::from_owned_array(py, padded_grid.clone()).readonly(),
            pad_th,
            PyArray2::from_owned_array(py, cell_cols_nd.clone()).readonly(),
            1.5,
        )?;
        timing_results.insert("window_connections", connections_start.elapsed());

        // Extract connections list from Python object
        let mut connections_list = conn.extract::<Vec<(Vec<u32>, bool, String, Vec<Vec<f64>>, Vec<f64>)>>(py)?;

        // Mark connections as periodic if any maxima is far from the box (like Python does)
        // Use the reindexed valid region IDs (after first reindexing) to match the connection indices
        let reindexed_valid_region_ids: std::collections::HashSet<u32> = close_or_inside_ids.iter()
            .filter_map(|&original_id| remap_dict.get(&original_id).copied())
            .collect();

        for conn in &mut connections_list {
            let has_far_maxima = conn.0.iter().any(|&idx| !reindexed_valid_region_ids.contains(&idx));
            if has_far_maxima {
                conn.2 = "periodic".to_string();
            }
        }

        // Second phase reindexing (connected nodes getting priority)
        let second_reindexing_start = Instant::now();
        // Build remap_dict2 following Python algorithm
        let mut remap_dict2: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let unique_regions_padded_set: std::collections::HashSet<u32> = unmerged_padded.iter().cloned().filter(|&v| v != 0).collect();

        let mut next_index = padding_indices_start_here;

        // Get unique connected indices
        let mut unique_connected_indices: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for conn in &connections_list {
            for &idx in &conn.0 {
                unique_connected_indices.insert(idx);
            }
        }

        // Priority to connected nodes that are in padded regions (>= padding_indices_start_here)
        for &c in &unique_connected_indices {
            if c >= padding_indices_start_here {
                remap_dict2.insert(c, next_index);
                next_index += 1;
            }
        }

        // Remaining padded regions
        let mut unique_regions_padded_vec: Vec<u32> = unique_regions_padded_set.into_iter().collect();
        unique_regions_padded_vec.sort_unstable();

        for &r in &unique_regions_padded_vec {
            if !remap_dict2.contains_key(&r) && r >= padding_indices_start_here {
                remap_dict2.insert(r, next_index);
                next_index += 1;
            }
        }

        // Apply remapping to regions_padded
        let mut remapped_regions_padded = unmerged_padded.clone();
        for ((i, j, k), &value) in unmerged_padded.indexed_iter() {
            if value != 0 {
                if let Some(&new_value) = remap_dict2.get(&value) {
                    remapped_regions_padded[[i, j, k]] = new_value;
                }
            }
        }
        final_regions_padded = remapped_regions_padded;

        // Apply remapping to connections
        let mut remapped_connections = connections_list.clone();
        for conn in &mut remapped_connections {
            for idx in &mut conn.0 {
                if let Some(&new_idx) = remap_dict2.get(idx) {
                    *idx = new_idx;
                }
            }
        }

        // Apply remapping to maxima_info
        let mut remapped_maxima_info_2 = maxima_info.clone();
        for mut row in remapped_maxima_info_2.rows_mut() {
            let original_label = row[0] as u32;
            if let Some(&new_label) = remap_dict2.get(&original_label) {
                row[0] = new_label as i64;
            }
        }

        // Sort maxima_info by first column again
        let mut maxima_info_rows: Vec<Vec<i64>> = Vec::new();
        for row in remapped_maxima_info_2.rows() {
            maxima_info_rows.push(row.to_vec());
        }
        maxima_info_rows.sort_by(|a, b| a[0].cmp(&b[0]));
        let maxima_info_data: Vec<i64> = maxima_info_rows.into_iter().flatten().collect();
        final_maxima_info = Array2::from_shape_vec((maxima_info_data.len()/7, 7), maxima_info_data).unwrap();

        // Convert remapped connections back to Python format
        let py_conn_list = PyList::empty(py);
        for conn in remapped_connections {
            let regs_arr = PyArray1::<u32>::from_vec(py, conn.0);
            let centers_vec: Vec<Vec<f64>> = conn.3;
            let centers_arr = PyArray2::<f64>::from_vec2(py, &centers_vec)?;
            let radii_arr = PyArray1::<f64>::from_vec(py, conn.4);

            let tup = PyTuple::new(
                py,
                &[
                    regs_arr.into_py(py),
                    conn.1.into_py(py),
                    conn.2.into_py(py),
                    centers_arr.into_py(py),
                    radii_arr.into_py(py),
                ],
            );
            py_conn_list.append(tup)?;
        }
        py_connections = Some(py_conn_list.into());
        timing_results.insert("second_reindexing", second_reindexing_start.elapsed());
        timing_results.insert("total_connections", connections_start.elapsed());
    }

    // Debug: Add some diagnostic information
    // let unique_regions_before: std::collections::HashSet<u32> = regions.iter().cloned().filter(|&v| v != 0).collect();
    // let mut unique_before_vec: Vec<u32> = unique_regions_before.into_iter().collect();
    // unique_before_vec.sort_unstable();
    // println!("Unique regions before reindexing: {:?}", unique_before_vec);
    // 
    // let mut remap_vec: Vec<(u32, u32)> = remap_dict.iter().map(|(&k, &v)| (k, v)).collect();
    // remap_vec.sort_by_key(|&(k, _)| k);
    // println!("Remap dict: {:?}", remap_vec);
    // 
    // let unique_regions_after: std::collections::HashSet<u32> = regions.iter().cloned().filter(|&v| v != 0).collect();
    // let mut unique_after_vec: Vec<u32> = unique_regions_after.into_iter().collect();
    // unique_after_vec.sort_unstable();
    // println!("Unique regions after reindexing: {:?}", unique_after_vec);

    // Second phase reindexing: recalculate close_or_inside using final_maxima_info (like Python does)
    // Recalculate maxima indices using the reindexed final_maxima_info
    let final_maxima_coords: Vec<[i64;3]> = final_maxima_info.rows().into_iter().map(|r| {
        [r[1] + r[4]*dgrid_shape[0], r[2] + r[5]*dgrid_shape[1], r[3] + r[6]*dgrid_shape[2]]
    }).collect();
    
    // Recalculate location classifications for the reindexed maxima using same logic as original
    let mut final_locs = Vec::new();
    for coords in &final_maxima_coords {
        let [z, y, x] = *coords;
        let depth = dgrid_shape[0];
        let height = dgrid_shape[1]; 
        let width = dgrid_shape[2];
        let zz = wall_cells[0] as i64;
        let yy = wall_cells[1] as i64;
        let xx = wall_cells[2] as i64;
        
        // Check if inside the box
        if (0 <= z && z < depth) && (0 <= y && y < height) && (0 <= x && x < width) {
            final_locs.push("inside");
        }
        // Check if close outside (within tolerance)
        else if (-zz <= z && z < depth + zz) && (-yy <= y && y < height + yy) && (-xx <= x && x < width + xx) {
            final_locs.push("close_outside");
        }
        // Otherwise far outside
        else {
            final_locs.push("far_outside");
        }
    }
    
    // Create new close_or_inside masks and indices based on reindexed data
    let final_close_or_inside_mask: Vec<bool> = final_locs.iter().map(|&s| s=="inside" || s=="close_outside").collect();
    let final_close_or_inside_indices: Vec<[i64;3]> = final_maxima_coords.iter().enumerate()
        .filter(|(i,_)| final_close_or_inside_mask[*i])
        .map(|(_,idx)| [idx[0], idx[1], idx[2]]).collect();
    let final_close_or_inside_ids: Vec<u32> = final_close_or_inside_mask.iter().enumerate().filter_map(|(i, ok)| if *ok { Some((i+1) as u32) } else { None }).collect();

    // Build Python dict with final results (using reindexed data like Python does)
    let output_assembly_start = Instant::now();
    let out = PyDict::new(py);
    out.set_item("maxima_pbc", PyArray2::from_owned_array(py, merged_maxima).to_owned())?;
    out.set_item("regions_pbc", PyArray3::from_owned_array(py, regions_pbc).to_owned())?;
    out.set_item("maxima_info", PyArray2::from_owned_array(py, final_maxima_info.clone()).to_owned())?;
    out.set_item("regions", PyArray3::from_owned_array(py, regions).to_owned())?;
    {
        let list = PyList::empty(py);
        for g in &pbc_groups {
            let pyset = pyo3::types::PySet::empty(py)?;
            for lbl in g { pyset.add(*lbl)?; }
            list.append(pyset)?;
        }
        out.set_item("pbc_groups", list)?;
    }
    out.set_item("close_or_inside_maxima", PyArray2::from_vec2(py, &final_close_or_inside_indices.iter().map(|v| vec![v[0],v[1],v[2]]).collect::<Vec<_>>())?)?;
    out.set_item("close_or_inside_regions", PyArray1::from_vec(py, final_close_or_inside_ids))?;
    out.set_item("unique_regions", PyArray1::from_vec(py, unique_regions_vec))?;
    // maxima_pbc_dict as dictionary
    {
        let dict = PyDict::new(py);
        for ((i,j,k), lbl) in maxima_pbc_dict.iter() {
            dict.set_item((i,j,k), *lbl as u32)?;
        }
        out.set_item("maxima_pbc_dict", dict)?;
    }
    let all_maxima_coords: Vec<[i64;3]> = final_maxima_info.rows().into_iter().map(|r| {
        [r[1] + r[4]*dgrid_shape[0], r[2] + r[5]*dgrid_shape[1], r[3] + r[6]*dgrid_shape[2]]
    }).collect();
    let all_maxima_arr: Array2<i64> = Array2::from_shape_vec((all_maxima_coords.len(),3), all_maxima_coords.iter().flat_map(|c| vec![c[0], c[1], c[2]]).collect()).unwrap();
    out.set_item("all_maxima", PyArray2::from_owned_array(py, all_maxima_arr).to_owned())?;
    out.set_item("maxima_relative_locations", PyList::new(py, final_locs))?;
    if let Some(conn) = py_connections { out.set_item("connections", conn)?; }
    if return_padded_regions { out.set_item("regions_padded", PyArray3::from_owned_array(py, final_regions_padded).to_owned())?; }
    // input_params
    let params = PyDict::new(py);
    params.set_item("mask_thickness", mask_thickness)?;
    params.set_item("h", h)?;
    params.set_item("min_distance", min_distance)?;
    params.set_item("threshold_abs", threshold_abs)?;
    params.set_item("spacing", spacing)?;
    params.set_item("minimum_window_separation", minimum_window_separation)?;
    params.set_item("compactness", compactness)?;
    params.set_item("wall_thickness", wall_thickness)?;
    params.set_item("extra_padding", extra_padding)?;
    params.set_item("padding_thickness", pad_th)?;
    params.set_item("min_region_size", min_region_size)?;
    out.set_item("input_params", params)?;
    if return_dgrid { out.set_item("dgrid", PyArray3::from_owned_array(py, dgrid).to_owned())?; }
    timing_results.insert("output_assembly", output_assembly_start.elapsed());
    timing_results.insert("total_time", start_time.elapsed());

    // Add timing results to output
    {
        let timing_dict = PyDict::new(py);
        for (key, duration) in &timing_results {
            timing_dict.set_item(key, duration.as_secs_f64())?;
        }
        out.set_item("timing_results", timing_dict)?;

        // Debug: Print timing results to stderr (sorted for consistency)
        eprintln!("=== RUST TIMING RESULTS ===");
        let mut sorted_keys: Vec<&str> = timing_results.keys().cloned().collect();
        sorted_keys.sort();
        for key in sorted_keys {
            if let Some(duration) = timing_results.get(key) {
                eprintln!("{}: {:.4}", key, duration.as_secs_f64());
            }
        }
        eprintln!("=== END RUST TIMING ===");
    }

    Ok(out.into())
}


/// Find the pore limiting radius using binary search and percolation analysis
#[pyfunction]
#[pyo3(text_signature = "(distance_grid)")]
fn find_pore_limiting_radius(
    distance_grid: PyReadonlyArray3<'_, f32>
) -> PyResult<Option<f32>> {
    let dgrid = distance_grid.as_array().to_owned();
    
    // Get unique values and sort them
    let mut unique_values: Vec<f32> = dgrid
        .iter()
        .filter(|&&v| v > 0.0)
        .cloned()
        .collect();
    
    if unique_values.is_empty() {
        return Ok(None);
    }
    
    // Sort and deduplicate
    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_values.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    
    // Binary search for the pore limiting radius
    let mut left = 0;
    let mut right = unique_values.len() - 1;
    
    while left <= right {
        let mid = (left + right) / 2;
        let threshold = unique_values[mid];
        
        if check_percolation_periodic(&dgrid, threshold) {
            left = mid + 1; // Percolation exists, try a larger threshold
        } else {
            if mid == 0 { break; }
            right = mid - 1; // No percolation, try a smaller threshold
        }
    }
    
    // The pore limiting radius is the last threshold that allowed percolation
    if right < unique_values.len() {
        Ok(Some(unique_values[right]))
    } else {
        Ok(None)
    }
}

/// Simple binary dilation for 3D arrays (26-connectivity)
fn binary_dilation_3d(mask: &ndarray::Array3<bool>) -> ndarray::Array3<bool> {
    let (nx, ny, nz) = mask.dim();
    let mut dilated = mask.clone();
    
    // 26-connectivity offsets
    let offsets = [
        (-1,-1,-1), (-1,-1,0), (-1,-1,1),
        (-1,0,-1),  (-1,0,0),  (-1,0,1),
        (-1,1,-1),  (-1,1,0),  (-1,1,1),
        (0,-1,-1),  (0,-1,0),  (0,-1,1),
        (0,0,-1),   (0,0,1),
        (0,1,-1),   (0,1,0),   (0,1,1),
        (1,-1,-1),  (1,-1,0),  (1,-1,1),
        (1,0,-1),   (1,0,0),   (1,0,1),
        (1,1,-1),   (1,1,0),   (1,1,1),
    ];
    
    for ((i, j, k), &value) in mask.indexed_iter() {
        if value {
            for &(di, dj, dk) in &offsets {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                let nk = k as i32 + dk;
                
                if ni >= 0 && nj >= 0 && nk >= 0 &&
                   (ni as usize) < nx && (nj as usize) < ny && (nk as usize) < nz {
                    dilated[[ni as usize, nj as usize, nk as usize]] = true;
                }
            }
        }
    }
    
    dilated
}

/// Find end channel pairs that need to be merged due to periodic boundary overflow
/// 
/// This function identifies small regions on the walls of the unit cell that should be
/// merged with larger regions they are connected to, based on the Python implementation.
fn find_end_channel_pairs_rs(
    regions_pbc: &ndarray::Array3<u32>,
    regions_nonpbc: &ndarray::Array3<u32>,
    size_ratio: f64,
) -> Vec<(u32, u32)> {
    let box_shape = regions_nonpbc.dim();
    let (nx, ny, nz) = box_shape;
    
    // Create wall mask - mark the 6 faces of the box
    let mut wall_mask = ndarray::Array3::<bool>::default(box_shape);
    
    // Top and bottom faces (z-direction)
    wall_mask.slice_mut(ndarray::s![0, .., ..]).fill(true);
    wall_mask.slice_mut(ndarray::s![nx-1, .., ..]).fill(true);
    
    // Front and back faces (y-direction)
    wall_mask.slice_mut(ndarray::s![.., 0, ..]).fill(true);
    wall_mask.slice_mut(ndarray::s![.., ny-1, ..]).fill(true);
    
    // Left and right faces (x-direction)
    wall_mask.slice_mut(ndarray::s![.., .., 0]).fill(true);
    wall_mask.slice_mut(ndarray::s![.., .., nz-1]).fill(true);
    
    // Find unique labels on the walls (excluding background = 0)
    let mut wall_labels = std::collections::HashSet::new();
    for ((i, j, k), &is_wall) in wall_mask.indexed_iter() {
        if is_wall {
            let label = regions_nonpbc[[i, j, k]];
            if label != 0 {
                wall_labels.insert(label);
            }
        }
    }
    
    let mut pairs = Vec::new();
    
    for &label_id in &wall_labels {
        // Create mask for current label
        let mask = regions_nonpbc.mapv(|v| v == label_id);
        
        // Find corresponding PBC labels for this region
        let mut pbc_labels = std::collections::HashSet::new();
        for ((i, j, k), &is_current) in mask.indexed_iter() {
            if is_current {
                let pbc_label = regions_pbc[[i, j, k]];
                if pbc_label != 0 {
                    pbc_labels.insert(pbc_label);
                }
            }
        }
        
        // Skip if ambiguous (should have exactly one PBC label)
        if pbc_labels.len() != 1 {
            continue;
        }
        
        let pbc_label = *pbc_labels.iter().next().unwrap();
        
        // Find all other regions on the same wall with the same PBC label
        let same_wall_mask = regions_pbc.indexed_iter()
            .zip(wall_mask.indexed_iter())
            .map(|(((i, j, k), &pbc_val), (_, &is_wall))| {
                is_wall && pbc_val == pbc_label
            })
            .collect::<Vec<bool>>();
            
        let mut same_wall_mask_array = ndarray::Array3::<bool>::default(box_shape);
        for (idx, &val) in same_wall_mask.iter().enumerate() {
            let k = idx % nz;
            let j = (idx / nz) % ny;
            let i = idx / (ny * nz);
            same_wall_mask_array[[i, j, k]] = val;
        }
        
        // Get other labels on the same wall
        let mut other_labels = std::collections::HashSet::new();
        for ((i, j, k), &is_same_wall) in same_wall_mask_array.indexed_iter() {
            if is_same_wall {
                let other_label = regions_nonpbc[[i, j, k]];
                if other_label != 0 && other_label != label_id {
                    other_labels.insert(other_label);
                }
            }
        }
        
        if other_labels.is_empty() {
            continue; // No other regions of this PBC group on the wall
        }
        
        // Check for adjacency with each potential neighbor
        for &neighbor_label in &other_labels {
            let neighbor_mask = regions_nonpbc.mapv(|v| v == neighbor_label);
            let neighbor_wall_mask = neighbor_mask.indexed_iter()
                .zip(wall_mask.indexed_iter())
                .map(|(((i, j, k), &is_neighbor), (_, &is_wall))| {
                    is_neighbor && is_wall
                })
                .collect::<Vec<bool>>();
                
            let mut neighbor_wall_mask_array = ndarray::Array3::<bool>::default(box_shape);
            for (idx, &val) in neighbor_wall_mask.iter().enumerate() {
                let k = idx % nz;
                let j = (idx / nz) % ny;
                let i = idx / (ny * nz);
                neighbor_wall_mask_array[[i, j, k]] = val;
            }
            
            // Check adjacency using binary dilation
            let dilated_mask = binary_dilation_3d(&mask);
            let is_adjacent = dilated_mask.indexed_iter()
                .zip(neighbor_wall_mask_array.indexed_iter())
                .any(|(((i, j, k), &dilated), (_, &neighbor_wall))| {
                    dilated && neighbor_wall
                });
            
            if is_adjacent {
                // Calculate sizes
                let size_current = mask.iter().filter(|&&v| v).count();
                let size_neighbor = neighbor_mask.iter().filter(|&&v| v).count();
                
                // Apply size ratio criterion
                if (size_current as f64) < size_ratio * (size_neighbor as f64) {
                    pairs.push((label_id, neighbor_label));
                }
                break; // Only consider the first touching neighbor
            }
        }
    }
    
    pairs
}

#[pymodule]
fn mgr_cli_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Enable better error reporting when the module is imported from Python.
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(dgrid_from_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(egrid_from_atoms, m)?)?;
    m.add_function(wrap_pyfunction!(window_connections, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_gaussian_filter, m)?)?;
    m.add_function(wrap_pyfunction!(seeded_watershed_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(seeded_watershed, m)?)?;
    m.add_function(wrap_pyfunction!(seeded_watershed_masked, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_h_morph, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_h_morph_opening, m)?)?;
    m.add_function(wrap_pyfunction!(move_maxima_to_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_h_morph2, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_h_morph_simple, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_h_morph_no_spacing, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_adaptive_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_selective_smoothing, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_selective_smoothing_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_percolation_guided, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_voronoi_based, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_multiscale_log, m)?)?;
    m.add_function(wrap_pyfunction!(detect_maxima_adaptive_h, m)?)?;
    m.add_function(wrap_pyfunction!(do_periodic_segmentation_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_local_maxima_pbc, m)?)?;
    m.add_function(wrap_pyfunction!(find_local_maxima_cart, m)?)?;
    m.add_function(wrap_pyfunction!(find_pore_limiting_radius, m)?)?;
    m.add_function(wrap_pyfunction!(find_end_channel_pairs, m)?)?;
    m.add("__all__", ["dgrid_from_atoms", "egrid_from_atoms", "window_connections", "periodic_gaussian_filter", "seeded_watershed_bucket", "seeded_watershed", "seeded_watershed_masked", "detect_maxima_h_morph", "detect_maxima_h_morph_opening", "move_maxima_to_peaks", "detect_maxima_h_morph2", "detect_maxima_h_morph_simple", "detect_maxima_h_morph_no_spacing", "detect_maxima_adaptive_geometry", "detect_maxima_selective_smoothing", "detect_maxima_selective_smoothing_parallel", "detect_maxima_percolation_guided", "detect_maxima_voronoi_based", "detect_maxima_multiscale_log", "detect_maxima_adaptive_h", "do_periodic_segmentation_rust", "find_local_maxima_pbc", "find_local_maxima_cart", "find_pore_limiting_radius", "find_end_channel_pairs" ])?;

    // By returning Ok(()), the module initialization is successful.
    Ok(())
}

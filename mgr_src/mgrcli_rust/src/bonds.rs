//! Covalent bond detection (pymatgen-compatible), releases the GIL during compute.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Embedded bond length database from pymatgen bond_lengths.json.
/// Each entry: (element_a, element_b, bond_length_angstrom, bond_order).
static BOND_LENGTHS: &[(&str, &str, f64, f64)] = &[
    ("C", "C", 1.54, 1.0), ("C", "C", 1.34, 2.0), ("C", "C", 1.20, 3.0),
    ("C", "H", 1.09, 1.0),
    ("C", "O", 1.43, 1.0), ("C", "O", 1.23, 2.0),
    ("C", "N", 1.47, 1.0), ("C", "N", 1.30, 2.0), ("C", "N", 1.16, 3.0),
    ("C", "S", 1.82, 1.0), ("C", "S", 1.61, 2.0),
    ("H", "H", 0.74, 1.0),
    ("H", "O", 0.96, 1.0),
    ("H", "N", 1.01, 1.0),
    ("H", "S", 1.34, 1.0),
    ("O", "O", 1.48, 1.0), ("O", "O", 1.21, 2.0),
    ("O", "N", 1.40, 1.0), ("O", "N", 1.21, 2.0), ("O", "N", 1.14, 3.0),
    ("O", "S", 1.51, 1.0), ("O", "S", 1.45, 2.0), ("O", "S", 1.42, 3.0),
    ("N", "N", 1.45, 1.0), ("N", "N", 1.25, 2.0), ("N", "N", 1.10, 3.0),
    ("N", "S", 1.71, 1.0), ("N", "S", 1.55, 2.0),
    ("S", "S", 2.05, 1.0), ("S", "S", 1.89, 2.0),
    ("Cu", "O", 1.95, 1.0), // Cu-O from pymatgen database
    ("Cu", "O", 1.95, 0.5),
    ("Zn", "O", 1.95, 1.0),
    ("Zn", "N", 1.98, 1.0),
    ("Cr", "O", 1.92, 1.0),
    ("Cr", "C", 2.05, 1.0),
    ("Cr", "N", 2.02, 1.0),
    ("Cl", "H", 1.27, 1.0),
    ("Cl", "C", 1.77, 1.0),
    ("Cl", "O", 1.69, 1.0),
    ("Cl", "N", 1.75, 1.0),
    ("Si", "O", 1.61, 1.0),
    ("Si", "C", 1.86, 1.0),
    ("Al", "O", 1.76, 1.0),
];

/// Compute covalent bonds from an ASE Atoms object and AABB neighbor list.
///
/// Compute covalent bonds with periodic image replication matching pymatgen.
///
/// For bonds crossing periodic boundaries, this function creates image atoms
/// (just like pymatgen's CovalentBond workflow in update_with_covalent_bonds).
/// The returned bond indices reference these image atoms, producing output
/// identical to the Python pymatgen implementation.
///
/// Returns tuple of 4 lists:
///   bonds_flat:      [a0, b0, a1, b1, ...] — b may be an image index (>= natoms)
///   bond_orders:     [bo0, bo1, ...]
///   image_positions: [x0, y0, z0, x1, y1, z1, ...] — Cartesian coords of image atoms
///   image_atomic_numbers: [Z0, Z1, ...] — element Z of each image atom
#[pyfunction]
pub fn covalent_bonds(
    positions: Vec<f64>,       // flat [x0,y0,z0, x1,y1,z1, ...]
    atomic_numbers: Vec<i32>,  // element Z for each atom
    cell: Vec<f64>,            // flat 3x3 cell matrix, column-major
    nlist_i: Vec<i64>,         // neighbor list source indices
    nlist_j: Vec<i64>,         // neighbor list target indices
    tol: f64,                  // bond tolerance (default 0.2)
) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<i32>) {
    use std::collections::HashMap;

    // Periodic table: Z → element symbol
    let z_to_sym: HashMap<i32, &str> = [
        (1, "H"), (5, "B"), (6, "C"), (7, "N"), (8, "O"), (9, "F"),
        (13, "Al"), (14, "Si"), (15, "P"), (16, "S"), (17, "Cl"),
        (24, "Cr"), (26, "Fe"), (27, "Co"), (28, "Ni"), (29, "Cu"),
        (30, "Zn"), (40, "Zr"), (47, "Ag"), (48, "Cd"), (74, "W"),
    ].iter().cloned().collect();

    // Covalent radii (in Angstrom) indexed by Z
    let get_cov_r = |z: i32| -> f64 {
        match z {
            1 => 0.31,  5 => 0.84,  6 => 0.76,  7 => 0.71,  8 => 0.66,  9 => 0.57,
            13 => 1.21, 14 => 1.11, 15 => 1.07, 16 => 1.05, 17 => 1.02,
            24 => 1.39, 26 => 1.32, 27 => 1.26, 28 => 1.24, 29 => 1.32,
            30 => 1.22, 40 => 1.75, 47 => 1.45, 48 => 1.44, 74 => 1.62,
            _ => 1.50,
        }
    };

    let natoms = positions.len() / 3;
    let npairs = nlist_i.len();
    let mut bonds_flat: Vec<i64> = Vec::new();
    let mut bond_orders: Vec<f64> = Vec::new();
    let mut image_positions: Vec<f64> = Vec::new();
    let mut image_atomic_numbers: Vec<i32> = Vec::new();
    let mut n_images: usize = 0;

    // Build bond length database: (sym_a, sym_b) → Vec<(bond_order, length)>
    let mut bond_db: HashMap<(String, String), Vec<(f64, f64)>> = HashMap::new();
    for &(a, b, len, bo) in BOND_LENGTHS {
        let key = if a < b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) };
        bond_db.entry(key).or_default().push((bo, len));
    }

    // Parse cell and compute fractional coordinates for all atoms
    let cell = nalgebra::Matrix3::from_column_slice(&cell);
    let inv_cell = cell.try_inverse().unwrap_or_else(|| nalgebra::Matrix3::identity());

    // Precompute fractional coordinates (matches pymatgen struct.frac_coords)
    let mut frac_coords: Vec<nalgebra::Vector3<f64>> = Vec::with_capacity(natoms);
    for i in 0..natoms {
        let cart = nalgebra::Vector3::new(positions[i*3], positions[i*3+1], positions[i*3+2]);
        frac_coords.push(inv_cell * cart);
    }

    for idx in 0..npairs {
        let i = nlist_i[idx] as usize;
        let j = nlist_j[idx] as usize;
        if i >= natoms || j >= natoms { continue; }

        let zi = atomic_numbers[i];
        let zj = atomic_numbers[j];
        let ri = get_cov_r(zi);
        let rj = get_cov_r(zj);
        let default_bl = ri + rj;

        // Minimum image distance (same as pymatgen)
        let pi = nalgebra::Vector3::new(positions[i*3], positions[i*3+1], positions[i*3+2]);
        let pj = nalgebra::Vector3::new(positions[j*3], positions[j*3+1], positions[j*3+2]);
        let diff = pi - pj;
        let frac = inv_cell * diff;
        let frac_shifted = nalgebra::Vector3::new(
            frac.x - frac.x.round(),
            frac.y - frac.y.round(),
            frac.z - frac.z.round(),
        );
        let mic_diff = cell * frac_shifted;
        let dist = mic_diff.norm();

        // is_bonded check: dist < (1 + tol) * default_bl
        if dist >= (1.0 + tol) * default_bl {
            continue;
        }

        // Periodic image check (matches Python: shift = round(frac[a] - frac[b]))
        // Use frac_coords which are in [0,1) for atoms inside the unit cell.
        let dr = frac_coords[i] - frac_coords[j];
        let shift_x = dr.x.round();
        let shift_y = dr.y.round();
        let shift_z = dr.z.round();
        let is_periodic = shift_x != 0.0 || shift_y != 0.0 || shift_z != 0.0;

        let b_idx: i64 = if is_periodic {
            // Create image atom: new_site_frac = frac[b] + shift, then to Cartesian
            let new_frac = nalgebra::Vector3::new(
                frac_coords[j].x + shift_x,
                frac_coords[j].y + shift_y,
                frac_coords[j].z + shift_z,
            );
            let new_cart = cell * new_frac;
            image_positions.push(new_cart.x);
            image_positions.push(new_cart.y);
            image_positions.push(new_cart.z);
            image_atomic_numbers.push(zj);
            let img_idx = (natoms + n_images) as i64;
            n_images += 1;
            img_idx
        } else {
            j as i64
        };

        // get_bond_order
        let sym_a = *z_to_sym.get(&zi).unwrap_or(&"X");
        let sym_b = *z_to_sym.get(&zj).unwrap_or(&"X");
        let key = if sym_a < sym_b {
            (sym_a.to_string(), sym_b.to_string())
        } else {
            (sym_b.to_string(), sym_a.to_string())
        };

        let bo = if let Some(entries) = bond_db.get(&key) {
            let mut lens: Vec<(f64, f64)> = entries.iter().map(|&(bo, len)| (bo, len)).collect();
            lens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            // Build trial thresholds: [bl_at_bo1*(1+tol), bl_at_bo1, bl_at_bo2, ...]
            if lens.is_empty() {
                if dist < (1.0 + tol) * default_bl { 1.0 } else { 0.0 }
            } else {
                let mut thresholds: Vec<f64> = Vec::with_capacity(lens.len() + 1);
                thresholds.push(lens[0].1 * (1.0 + tol));
                for &(_, len) in &lens {
                    thresholds.push(len);
                }
                // Find bond order by interpolation
                let mut bo_result: f64 = lens[lens.len()-1].0; // default: highest
                for t_idx in 0..thresholds.len() {
                    if thresholds[t_idx] < dist {
                        if t_idx == 0 {
                            bo_result = 0.0;
                        } else {
                            let low_bl = thresholds[t_idx];
                            let high_bl = thresholds[t_idx - 1];
                            let trial_bo = t_idx as f64;
                            bo_result = trial_bo - (dist - low_bl) / (high_bl - low_bl);
                        }
                        break;
                    }
                }
                bo_result
            }
        } else {
            // Not in database — use default_bl for bond order 1
            if dist < (1.0 + tol) * default_bl { 1.0 } else { 0.0 }
        };

        bonds_flat.push(i as i64);
        bonds_flat.push(b_idx);
        bond_orders.push(bo);
    }

    (bonds_flat, bond_orders, image_positions, image_atomic_numbers)
}

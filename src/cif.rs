//! MOF/RASPA-style CIF reader (no ASE). Output matches ASE Atoms conventions.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fs;
use std::path::Path;

/// Build 3×3 cell matrix (rows = lattice vectors). Matches ``ase.geometry.cell.cellpar_to_cell`` defaults.
fn cellpar_to_cell(a: f64, b: f64, c: f64, alpha_deg: f64, beta_deg: f64, gamma_deg: f64) -> [[f64; 3]; 3] {
    let eps = 2.0 * f64::EPSILON * 90.0;
    let mut cos_alpha = (alpha_deg.to_radians()).cos();
    let mut cos_beta = (beta_deg.to_radians()).cos();
    if (alpha_deg.abs() - 90.0).abs() < eps {
        cos_alpha = 0.0;
    }
    if (beta_deg.abs() - 90.0).abs() < eps {
        cos_beta = 0.0;
    }
    let (cos_gamma, sin_gamma) = if (gamma_deg - 90.0).abs() < eps {
        (0.0, 1.0)
    } else if (gamma_deg + 90.0).abs() < eps {
        (0.0, -1.0)
    } else {
        let g = gamma_deg.to_radians();
        (g.cos(), g.sin())
    };
    let va = [a, 0.0, 0.0];
    let vb = [b * cos_gamma, b * sin_gamma, 0.0];
    let cx = cos_beta;
    let cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    let cz_sqr = (1.0 - cx * cx - cy * cy).max(0.0);
    let cz = cz_sqr.sqrt();
    let vc = [c * cx, c * cy, c * cz];
    [va, vb, vc]
}

fn normalize_symbol(raw: &str) -> String {
    let mut sym = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphabetic() {
            if ch.is_ascii_uppercase() || sym.is_empty() {
                sym.push(ch);
            } else if ch.is_ascii_lowercase() {
                sym.push(ch);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if sym.is_empty() {
        sym.push('X');
    }
    sym
}

fn wrap_frac(x: f64) -> f64 {
    let mut v = x % 1.0;
    if v < 0.0 {
        v += 1.0;
    }
    v
}

fn wrap_frac3(f: [f64; 3]) -> [f64; 3] {
    [wrap_frac(f[0]), wrap_frac(f[1]), wrap_frac(f[2])]
}

fn frac_to_cart(frac: [f64; 3], cell: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        frac[0] * cell[0][0] + frac[1] * cell[1][0] + frac[2] * cell[2][0],
        frac[0] * cell[0][1] + frac[1] * cell[1][1] + frac[2] * cell[2][1],
        frac[0] * cell[0][2] + frac[1] * cell[1][2] + frac[2] * cell[2][2],
    ]
}

fn parse_tag_value(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if !line.starts_with('_') {
        return None;
    }
    let mut parts = line.split_whitespace();
    let tag = parts.next()?;
    let value = parts.next()?;
    Some((tag, value))
}

struct AtomLoop {
    symbol_col: usize,
    fract_cols: Option<[usize; 3]>,
    cart_cols: Option<[usize; 3]>,
}

fn parse_cif_content(content: &str) -> Result<(Vec<String>, Vec<[f64; 3]>, [[f64; 3]; 3]), String> {
    let mut a = 1.0;
    let mut b = 1.0;
    let mut c = 1.0;
    let mut alpha = 90.0;
    let mut beta = 90.0;
    let mut gamma = 90.0;

    let lines: Vec<&str> = content.lines().collect();
    let mut atom_loop: Option<AtomLoop> = None;
    let mut loop_headers: Vec<String> = Vec::new();
    let mut symbols: Vec<String> = Vec::new();
    let mut frac_positions: Vec<[f64; 3]> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }
        if let Some((tag, value)) = parse_tag_value(line) {
            match tag {
                "_cell_length_a" => a = value.parse::<f64>().map_err(|e| e.to_string())?,
                "_cell_length_b" => b = value.parse::<f64>().map_err(|e| e.to_string())?,
                "_cell_length_c" => c = value.parse::<f64>().map_err(|e| e.to_string())?,
                "_cell_angle_alpha" => alpha = value.parse::<f64>().map_err(|e| e.to_string())?,
                "_cell_angle_beta" => beta = value.parse::<f64>().map_err(|e| e.to_string())?,
                "_cell_angle_gamma" => gamma = value.parse::<f64>().map_err(|e| e.to_string())?,
                _ => {}
            }
        }
        if line == "loop_" {
            loop_headers.clear();
            i += 1;
            while i < lines.len() {
                let hline = lines[i].trim();
                if hline.is_empty() {
                    i += 1;
                    continue;
                }
                if !hline.starts_with('_') {
                    break;
                }
                loop_headers.push(hline.to_string());
                i += 1;
            }
            let has_fract = loop_headers.iter().any(|h| h.contains("_atom_site_fract_"));
            let has_cart = loop_headers.iter().any(|h| h.contains("_atom_site_Cartn_"));
            if has_fract || has_cart {
                let mut symbol_col = None;
                let mut fract_cols: Option<[usize; 3]> = None;
                let mut cart_cols: Option<[usize; 3]> = None;
                for (idx, h) in loop_headers.iter().enumerate() {
                    if h.contains("_atom_site_type_symbol") {
                        symbol_col = Some(idx);
                    } else if h.ends_with("_atom_site_label") && symbol_col.is_none() {
                        symbol_col = Some(idx);
                    }
                    if h.ends_with("_atom_site_fract_x") {
                        fract_cols = Some([idx, 0, 0]);
                    }
                    if h.ends_with("_atom_site_fract_y") {
                        fract_cols.get_or_insert([0, 0, 0])[1] = idx;
                    }
                    if h.ends_with("_atom_site_fract_z") {
                        fract_cols.get_or_insert([0, 0, 0])[2] = idx;
                    }
                    if h.ends_with("_atom_site_Cartn_x") {
                        cart_cols = Some([idx, 0, 0]);
                    }
                    if h.ends_with("_atom_site_Cartn_y") {
                        cart_cols.get_or_insert([0, 0, 0])[1] = idx;
                    }
                    if h.ends_with("_atom_site_Cartn_z") {
                        cart_cols.get_or_insert([0, 0, 0])[2] = idx;
                    }
                }
                if let Some(sym_i) = symbol_col {
                    if fract_cols.is_some() || cart_cols.is_some() {
                        atom_loop = Some(AtomLoop {
                            symbol_col: sym_i,
                            fract_cols,
                            cart_cols,
                        });
                        while i < lines.len() {
                            let dline = lines[i].trim();
                            if dline.is_empty()
                                || dline.starts_with("loop_")
                                || dline.starts_with('_')
                                || dline.starts_with("data_")
                            {
                                break;
                            }
                            let cols: Vec<&str> = dline.split_whitespace().collect();
                            if cols.len() < loop_headers.len() {
                                break;
                            }
                            let sym = normalize_symbol(cols[sym_i]);
                            symbols.push(sym.to_string());
                            if let Some(fc) = fract_cols {
                                let fx: f64 = cols[fc[0]].parse::<f64>().map_err(|e| e.to_string())?;
                                let fy: f64 = cols[fc[1]].parse::<f64>().map_err(|e| e.to_string())?;
                                let fz: f64 = cols[fc[2]].parse::<f64>().map_err(|e| e.to_string())?;
                                frac_positions.push(wrap_frac3([fx, fy, fz]));
                            } else if let Some(cc) = cart_cols {
                                let cx: f64 = cols[cc[0]].parse::<f64>().map_err(|e| e.to_string())?;
                                let cy: f64 = cols[cc[1]].parse::<f64>().map_err(|e| e.to_string())?;
                                let cz: f64 = cols[cc[2]].parse::<f64>().map_err(|e| e.to_string())?;
                                frac_positions.push([cx, cy, cz]);
                            }
                            i += 1;
                        }
                        continue;
                    }
                }
            }
        }
        i += 1;
    }

    if symbols.is_empty() {
        return Err("no atom_site loop found".into());
    }

    let cell = cellpar_to_cell(a, b, c, alpha, beta, gamma);
    let use_cart = atom_loop.as_ref().map(|l| l.cart_cols.is_some()).unwrap_or(false);
    let positions: Vec<[f64; 3]> = if use_cart {
        frac_positions
    } else {
        frac_positions
            .iter()
            .map(|f| frac_to_cart(*f, &cell))
            .collect()
    };

    Ok((symbols, positions, cell))
}

/// Read CIF → (symbols, positions[N,3], cell[3,3], pbc[3]) for ASE construction.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn read_cif(py: Python<'_>, path: String) -> PyResult<PyObject> {
    let content = fs::read_to_string(Path::new(&path))
        .map_err(|e| PyIOError::new_err(format!("read_cif: {e}")))?;
    let (symbols, positions, cell) =
        parse_cif_content(&content).map_err(|e| PyValueError::new_err(e))?;

    let n = symbols.len();
    let mut pos_flat = Vec::with_capacity(n * 3);
    for p in &positions {
        pos_flat.extend_from_slice(p);
    }
    let pos_array = Array2::from_shape_vec((n, 3), pos_flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    let cell_array = Array2::from_shape_vec(
        (3, 3),
        vec![
            cell[0][0], cell[0][1], cell[0][2],
            cell[1][0], cell[1][1], cell[1][2],
            cell[2][0], cell[2][1], cell[2][2],
        ],
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?
    .into_pyarray(py);
    let pbc = Array1::from(vec![true, true, true]).into_pyarray(py);
    let sym_list = PyList::new(py, &symbols);

    Ok((sym_list, pos_array, cell_array, pbc).into_py(py))
}

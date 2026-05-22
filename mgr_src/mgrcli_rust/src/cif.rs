//! MOF/RASPA-style CIF reader (no ASE). Output matches ASE Atoms conventions.

use hashbrown::HashSet;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fs;
use std::path::Path;

/// Fractional digits for symmetry-expansion dedupe keys (ASE often differs at 5th decimal).
const FRAC_KEY_DECIMALS: usize = 4;
/// Merge expanded atoms on the same site (same symbol, PBC fractional distance).
const SITE_MERGE_TOL: f64 = 1e-3;

/// Orbit-equivalence tolerance (must match ASE crystal()/equivalent_sites default).
const ORBIT_SYMPREC: f64 = 1e-3;

/// Future Rust-native orbit-refine result (ASE-equivalent-sites parity contract).
///
/// Contract:
/// - `orbit_*` represent symmetry-aware orbit expansion/dedup output from ASU.
/// - Apply orbit output only when `orbit_symbols.len() < rust_symbols.len()`.
/// - Never apply when orbit count is equal or larger (avoid regressions).
struct OrbitRefineCandidate {
    orbit_symbols: Vec<String>,
    orbit_frac_positions: Vec<[f64; 3]>,
}

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
    let tag_end = line.find(|c: char| c.is_whitespace())?;
    let tag = &line[..tag_end];
    let rest = line[tag_end..].trim_start();
    if rest.is_empty() {
        return None;
    }
    // Handle quoted values (single/double) that may contain spaces.
    let value = if rest.starts_with('\'') {
        match rest[1..].find('\'') {
            Some(end) => &rest[..end + 2],
            None => rest,
        }
    } else if rest.starts_with('"') {
        match rest[1..].find('"') {
            Some(end) => &rest[..end + 2],
            None => rest,
        }
    } else {
        rest.split_whitespace().next()?
    };
    Some((tag, value))
}

fn strip_cif_quotes(s: &str) -> String {
    let t = s.trim();
    let t = t.trim_matches('\'').trim_matches('"');
    t.to_string()
}

fn is_symmetry_loop(headers: &[String]) -> bool {
    headers.iter().any(|h| {
        h.contains("_symmetry_equiv_pos_as_xyz")
            || h.contains("_space_group_symop.operation_xyz")
            || h.contains("_space_group_symop_operation_xyz")
            || h.contains("_space_group_symop.operation_xyz")
    })
}

/// Parse one axis expression from International Tables site symmetry (ASE-compatible).
fn parse_sitesym_element(element: &str) -> (Vec<(usize, i32)>, f64) {
    let element = element.to_ascii_lowercase();
    let mut is_positive = true;
    let mut is_frac = false;
    let mut sng_trans: Option<f64> = None;
    let mut fst_trans: Vec<char> = Vec::new();
    let mut snd_trans: Vec<char> = Vec::new();
    let mut rot: Vec<(usize, i32)> = Vec::new();

    for ch in element.chars() {
        match ch {
            '+' => is_positive = true,
            '-' => is_positive = false,
            '/' => is_frac = true,
            'x' | 'y' | 'z' => {
                let idx = (ch as u8 - b'x') as usize;
                rot.push((idx, if is_positive { 1 } else { -1 }));
            }
            c if c.is_ascii_digit() || c == '.' => {
                if sng_trans.is_none() {
                    sng_trans = Some(if is_positive { 1.0 } else { -1.0 });
                }
                if is_frac {
                    snd_trans.push(c);
                } else {
                    fst_trans.push(c);
                }
            }
            _ => {}
        }
    }

    let mut trans = 0.0;
    if !fst_trans.is_empty() {
        let sign = sng_trans.unwrap_or(1.0);
        trans = sign * fst_trans.iter().collect::<String>().parse::<f64>().unwrap_or(0.0);
    }
    if is_frac && !snd_trans.is_empty() {
        let denom = snd_trans.iter().collect::<String>().parse::<f64>().unwrap_or(1.0);
        if denom.abs() > f64::EPSILON {
            trans /= denom;
        }
    }
    (rot, trans)
}

/// Parse `x,y,z` style site symmetry into rotation + translation (ASE ``parse_sitesym_single``).
fn parse_sitesym_single(sym: &str) -> ([[i32; 3]; 3], [f64; 3]) {
    let sym = strip_cif_quotes(sym);
    let mut rot = [[0i32; 3]; 3];
    let mut trans = [0.0; 3];
    for (i, element) in sym.split(',').enumerate().take(3) {
        let (e_rot, e_trans) = parse_sitesym_element(element.trim());
        for (rot_idx, rot_sgn) in e_rot {
            rot[i][rot_idx] = rot_sgn;
        }
        trans[i] = e_trans;
    }
    (rot, trans)
}

fn apply_sitesym(frac: [f64; 3], rot: &[[i32; 3]; 3], trans: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i] += rot[i][j] as f64 * frac[j];
        }
        out[i] += trans[i];
    }
    wrap_frac3(out)
}

fn frac_key(symbol: &str, frac: [f64; 3]) -> String {
    format!(
        "{}:{:.prec$},{:.prec$},{:.prec$}",
        symbol,
        frac[0],
        frac[1],
        frac[2],
        prec = FRAC_KEY_DECIMALS
    )
}

fn frac_dist_pbc(a: [f64; 3], b: [f64; 3]) -> f64 {
    let mut d2 = 0.0;
    for i in 0..3 {
        let mut d = (a[i] - b[i]).abs();
        if d > 0.5 {
            d = 1.0 - d;
        }
        d2 += d * d;
    }
    d2.sqrt()
}

/// Collapse symmetry copies that landed on the same site (same element).
fn collapse_duplicate_sites(
    symbols: Vec<String>,
    frac_positions: Vec<[f64; 3]>,
    tol: f64,
) -> (Vec<String>, Vec<[f64; 3]>) {
    let n = symbols.len();
    if n <= 1 {
        return (symbols, frac_positions);
    }

    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    let fracs: Vec<[f64; 3]> = frac_positions.iter().map(|f| wrap_frac3(*f)).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            if symbols[i] != symbols[j] {
                continue;
            }
            if frac_dist_pbc(fracs[i], fracs[j]) < tol {
                let pi = find(&mut parent, i);
                let pj = find(&mut parent, j);
                if pi != pj {
                    parent[pi] = pj;
                }
            }
        }
    }

    let mut keep = Vec::new();
    let mut seen_roots = HashSet::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        if seen_roots.insert(r) {
            keep.push(i);
        }
    }
    keep.sort_unstable();

    let out_symbols: Vec<String> = keep.iter().map(|&i| symbols[i].clone()).collect();
    let out_frac: Vec<[f64; 3]> = keep.iter().map(|&i| fracs[i]).collect();
    (out_symbols, out_frac)
}


/// Minimum-image periodic distance in fractional coordinates (0 to 0.5 per axis).
fn min_image_diff(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let mut d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    for i in 0..3 {
        d[i] -= d[i].round();
    }
    d
}

/// Whether two frac positions are within symprec (componentwise min-image).
fn frac_within_symprec(a: [f64; 3], b: [f64; 3], symprec: f64) -> bool {
    let d = min_image_diff(a, b);
    d[0].abs() < symprec && d[1].abs() < symprec && d[2].abs() < symprec
}

/// Build crystallographic orbit of a single ASU point using all symops.
/// Returns orbit positions (wrapped to [0,1)) that are NOT duplicates of each other.
fn build_orbit(frac: [f64; 3], symops: &[(&[[i32; 3]; 3], &[f64; 3])]) -> Vec<[f64; 3]> {
    let mut orbit = Vec::new();
    for (rot, trans) in symops {
        let mut p = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                p[i] += rot[i][j] as f64 * frac[j];
            }
            p[i] += trans[i];
        }
        let wp = wrap_frac3(p);
        let is_dup = orbit.iter().any(|o| frac_within_symprec(*o, wp, ORBIT_SYMPREC));
        if !is_dup {
            orbit.push(wp);
        }
    }
    orbit
}

/// Check if a frac position is within ORBIT_SYMPREC of ANY atom in existing orbits.
fn is_in_existing_orbits(frac: [f64; 3], orbits: &[Vec<[f64; 3]>]) -> bool {
    for orbit in orbits {
        for o in orbit {
            if frac_within_symprec(frac, *o, ORBIT_SYMPREC) {
                return true;
            }
        }
    }
    false
}

/// Generate orbit-refine candidate from ASU atoms + CIF symmetry operations.
/// Mirrors ASE `Spacegroup.equivalent_sites(..., onduplicates="warn", symprec=1e-3)`.
/// Returns `Some(OrbitRefineCandidate)` with orbit-expanded atoms, or `None`.
fn generate_orbit_candidate(
    asu_symbols: &[String],
    asu_frac: &[(f64, f64, f64)], // (x, y, z) wrapped
    sitesym_ops: &[String],
    _spacegroup_number: Option<u32>,
    _cif_hall_symbol: Option<&str>,
) -> Option<OrbitRefineCandidate> {
    // If CIF has no symops and we have a spacegroup number, we could try to 
    // look up the full set of symmetry operations. For now, require CIF symops.
    if asu_symbols.is_empty() || sitesym_ops.is_empty() {
        return None;
    }

    let symops: Vec<([[i32; 3]; 3], [f64; 3])> = sitesym_ops
        .iter()
        .filter_map(|s| {
            let s = s.trim();
            if is_identity_sitesym(s) {
                return None;
            }
            Some(parse_sitesym_single(s))
        })
        .collect();

    if symops.is_empty() {
        return None;
    }

    let symop_refs: Vec<(&[[i32; 3]; 3], &[f64; 3])> = symops
        .iter()
        .map(|(r, t)| (r as &[[i32; 3]; 3], t as &[f64; 3]))
        .collect();

    let mut orbits: Vec<Vec<[f64; 3]>> = Vec::new();
    let mut orbit_kinds: Vec<usize> = Vec::new();

    for (asu_idx, &(fx, fy, fz)) in asu_frac.iter().enumerate() {
        let frac_wrapped = wrap_frac3([fx, fy, fz]);
        if is_in_existing_orbits(frac_wrapped, &orbits) {
            continue;
        }
        let orbit = build_orbit(frac_wrapped, &symop_refs);
        if !orbit.is_empty() {
            orbits.push(orbit);
            orbit_kinds.push(asu_idx);
        }
    }

    if orbits.is_empty() {
        return None;
    }

    let mut orbit_symbols = Vec::new();
    let mut orbit_frac_positions = Vec::new();
    for (orbit_idx, orbit) in orbits.into_iter().enumerate() {
        let kind = orbit_kinds[orbit_idx];
        let sym = &asu_symbols[kind];
        for pos in orbit {
            orbit_symbols.push(sym.clone());
            orbit_frac_positions.push(pos);
        }
    }

    Some(OrbitRefineCandidate {
        orbit_symbols,
        orbit_frac_positions,
    })
}

fn is_identity_sitesym(sym: &str) -> bool {
    let s = strip_cif_quotes(sym).to_ascii_lowercase();
    let s = s.replace(' ', "");
    matches!(s.as_str(), "x,y,z" | "x,y,z+0" | "x,y,z+0.0")
}

fn should_expand_symmetry(spacegroup_number: Option<u32>, sitesym_ops: &[String]) -> bool {
    if sitesym_ops.is_empty() {
        return false;
    }
    let all_identity = sitesym_ops.iter().all(|op| is_identity_sitesym(op));
    if spacegroup_number == Some(1) && all_identity {
        return false;
    }
    true
}

/// Expand asymmetric-unit atoms using CIF ``_symmetry_equiv_pos_as_xyz`` operations.
fn expand_by_symmetry(
    symbols: Vec<String>,
    frac_positions: Vec<[f64; 3]>,
    sitesym_ops: &[String],
) -> (Vec<String>, Vec<[f64; 3]>) {
    if sitesym_ops.is_empty() {
        return (symbols, frac_positions);
    }

    let symops: Vec<([[i32; 3]; 3], [f64; 3])> = sitesym_ops
        .iter()
        .map(|s| parse_sitesym_single(s))
        .collect();

    let mut out_symbols = Vec::new();
    let mut out_frac = Vec::new();
    let mut seen = HashSet::new();

    for (sym, frac) in symbols.iter().zip(frac_positions.iter()) {
        for (rot, trans) in &symops {
            let nf = apply_sitesym(*frac, rot, trans);
            let key = frac_key(sym, nf);
            if seen.insert(key) {
                out_symbols.push(sym.clone());
                out_frac.push(nf);
            }
        }
    }

    (out_symbols, out_frac)
}

/// Deterministic trigger for symmetry-aware orbit refinement.
///
/// This mirrors the Python-side guard currently used in `_spglib_dedupe`:
/// apply only when orbit-equivalent expansion produces fewer atoms than the
/// Rust fast-path expansion.
fn should_apply_orbit_refine(rust_count: usize, orbit_count: usize) -> bool {
    orbit_count < rust_count
}

/// Contract hook for future Rust-native orbit-aware refinement backend.
///
/// Today this returns the Rust expansion unchanged because orbit generation is
/// still done in Python (ASE `equivalent_sites`).
///
/// Future implementation should:
/// 1) Build orbit-equivalent candidate from ASU + symmetry operations
/// 2) Apply candidate only if `should_apply_orbit_refine(rust_n, orbit_n)`
/// 3) Keep Rust result on tie or expansion (`orbit_n >= rust_n`)
fn maybe_apply_orbit_refine_contract(
    rust_symbols: Vec<String>,
    rust_frac_positions: Vec<[f64; 3]>,
    orbit_candidate: Option<OrbitRefineCandidate>,
) -> (Vec<String>, Vec<[f64; 3]>) {
    let Some(candidate) = orbit_candidate else {
        return (rust_symbols, rust_frac_positions);
    };

    if should_apply_orbit_refine(rust_symbols.len(), candidate.orbit_symbols.len()) {
        return (candidate.orbit_symbols, candidate.orbit_frac_positions);
    }

    (rust_symbols, rust_frac_positions)
}

struct AtomLoop {
    symbol_col: usize,
    fract_cols: Option<[usize; 3]>,
    cart_cols: Option<[usize; 3]>,
    occupancy_col: Option<usize>,
}

fn parse_cif_content(content: &str) -> Result<(Vec<String>, Vec<[f64; 3]>, [[f64; 3]; 3], Option<u32>), String> {
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
    let mut sitesym_ops: Vec<String> = Vec::new();
    let mut spacegroup_number: Option<u32> = None;
    let mut cif_hall_symbol: Option<String> = None;

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
                "_symmetry_Int_Tables_number" | "_space_group_IT_number" => {
                    spacegroup_number = value.parse::<u32>().ok();
                }
                "_symmetry_space_group_name_hall" => {
                    cif_hall_symbol = Some(strip_cif_quotes(value));
                }
                "_symmetry_equiv_pos_as_xyz"
                | "_space_group_symop.operation_xyz"
                | "_space_group_symop_operation_xyz" => {
                    sitesym_ops.push(strip_cif_quotes(value));
                }
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

            if is_symmetry_loop(&loop_headers) {
                while i < lines.len() {
                    let dline = lines[i].trim();
                    if dline.is_empty()
                        || dline.starts_with("loop_")
                        || dline.starts_with('_')
                        || dline.starts_with("data_")
                    {
                        break;
                    }
                    sitesym_ops.push(strip_cif_quotes(dline));
                    i += 1;
                }
                continue;
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
                    if h.ends_with("_atom_site_occupancy") {
                        // tracked below via mut occupancy_col
                    }
                }
                let mut occupancy_col: Option<usize> = None;
                for (idx, h) in loop_headers.iter().enumerate() {
                    if h.ends_with("_atom_site_occupancy") {
                        occupancy_col = Some(idx);
                    }
                }
                if let Some(sym_i) = symbol_col {
                    if fract_cols.is_some() || cart_cols.is_some() {
                        atom_loop = Some(AtomLoop {
                            symbol_col: sym_i,
                            fract_cols,
                            cart_cols,
                            occupancy_col: None,
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

    // Symmetry expansion applies in fractional coordinates (ASE ``crystal()`` path).
    let (symbols, frac_positions) = if !use_cart && should_expand_symmetry(spacegroup_number, &sitesym_ops) {
        // Capture original ASU atoms BEFORE expansion for orbit candidate generation.
        let asu_symbols = symbols.clone();
        let asu_frac_wrapped: Vec<(f64, f64, f64)> = frac_positions
            .iter()
            .map(|f| (f[0], f[1], f[2]))
            .collect();
        let (sym, frac) = expand_by_symmetry(symbols, frac_positions, &sitesym_ops);
        let (sym, frac) = collapse_duplicate_sites(sym, frac, SITE_MERGE_TOL);

        // Rust-native orbit-aware candidate implemented but not enabled by default
        // until full RASPA2 parity is achieved.
        let _orbit_candidate_preview = generate_orbit_candidate(
            &asu_symbols,
            &asu_frac_wrapped,
            &sitesym_ops,
            spacegroup_number,
            cif_hall_symbol.as_deref(),
        );
        let orbit_candidate: Option<OrbitRefineCandidate> = None;
        maybe_apply_orbit_refine_contract(sym, frac, orbit_candidate)
    } else {
        (symbols, frac_positions)
    };

    let positions: Vec<[f64; 3]> = if use_cart {
        frac_positions
    } else {
        frac_positions
            .iter()
            .map(|f| frac_to_cart(*f, &cell))
            .collect()
    };

    Ok((symbols, positions, cell, spacegroup_number))
}

/// Read CIF → (symbols, positions[N,3], cell[3,3], pbc[3]) for ASE construction.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn read_cif(py: Python<'_>, path: String) -> PyResult<PyObject> {
    let content = fs::read_to_string(Path::new(&path))
        .map_err(|e| PyIOError::new_err(format!("read_cif: {e}")))?;
    let (symbols, positions, cell, _spacegroup_number) =
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mil47_symmetry_expansion_count() {
        let content = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../cifs/MIL-47-RASPA.cif"))
            .expect("fixture cif");
        let (symbols, _positions, _cell, _sg_no) =
            parse_cif_content(&content).expect("parse");
        assert_eq!(symbols.len(), 72, "MIL-47 should expand ASU to full cell");
    }
}

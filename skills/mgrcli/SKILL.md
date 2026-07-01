---
name: mgrcli
description: >-
  Run Mofography MOF analysis via the mgrcli CLI (batch process, textural
  properties, Three.js HTML, interactive Streamlit). Use when analyzing CIF
  files, generating configs, comparing CLI vs web defaults, or running
  segmentation / SA / vol / PSD / pore diameter workflows.
---

# mgrcli — MOF analysis CLI

Repository: `arung-northwestern/mgr_cli_py`. Batch CLI + `mgrcli interactive` (Streamlit). Core numerics: `mgrcli_rust` (PyO3) + `mgr_src/mgr`.

**Cursor:** symlink or copy this file to `.cursor/skills/mgrcli/SKILL.md` for auto-discovery.

## Install (local)

Run from the **repository root** after clone (paths like `config/web_analysis_template.json` and `cifs/` are relative to root).

```bash
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py

# Rust toolchain (required once) — maturin compiles mgrcli_rust during pip install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e .

# After editing Rust: mgr_src/mgrcli_rust/
pip install --force-reinstall -e .

mgrcli --help
mgrcli process cifs/ZIF-8-RASPA.cif -o ./out/ --return-dgrid
```

**Requirements:** Python **3.10+** recommended (`requires-python >=3.8` in `pyproject.toml`). Rust **1.70+**. ~2 GB RAM for compilation.

**Alternative:** `./install/install.sh` (uv-based; same editable install).

**Sample structures:** `cifs/` (`ZIF-8-RASPA.cif`, `P1-Cu-BTC.cif`, `MIL-47-RASPA.cif`, etc.).

**Verify env:** `which mgrcli` should point inside `.venv/bin/` after activate. No need to `cd python_cli/` — the console script is on PATH.

## Commands

| Command | Purpose |
|---------|---------|
| `mgrcli process <cif>… -o OUT` | Segment + optional textural props + optional HTML |
| `mgrcli generate-config -o config.json` | Emit defaults (`--profile web` or `legacy`) |
| `mgrcli plot results.npz -o OUT` | HTML from saved NPZ (no re-segmentation) |
| `mgrcli interactive [--clean]` | Streamlit web app (`--port`, `--no-browser`) |
| `mgrcli config-building` | Interactive config wizard |

## Quick analysis (web-aligned defaults)

Use the checked-in template or generate one:

```bash
mgrcli generate-config --profile web -o /tmp/web.json   # optional
mgrcli process cifs/ZIF-8-RASPA.cif \
  -o ./out \
  -c config/web_analysis_template.json \
  --pore-dia --psd --psd-accessible --sa --vol \
  --types \
  --plot-html --plot-backend threejs
```

**Full textural + plots (batch):**

```bash
mgrcli process cifs/ZIF-8-RASPA.cif -o out/ \
  -c config/web_analysis_template.json \
  --pore-dia \
  --percolation-plot \
  --psd --psd-accessible \
  --sa --sa-plot-html \
  --vol --vol-plot-html \
  --probe-radius 1.2
```

`--sa`, `--vol`, and `--psd-accessible` **auto-enable** `return_dgrid` and `compute_connections` when missing (web template already has both `true`).

## Defaults (web profile)

Source of truth: `python_cli/mgrcli/defaults.py` and `config/web_analysis_template.json`.

### Segmentation

| Key / flag | Default (web) |
|------------|---------------|
| `spacing` / `-s` | 0.5 Å |
| `h` / `-H` | 0.2 Å |
| `mask_thickness` / `-mt` | 0.05 Å |
| `threshold_abs` / `-ta` | 1.5 Å |
| `min_distance` / `-md` | 1.5 Å |
| `minimum_window_separation` / `-ms` | 1.0 Å |
| `watershed_backend` / `-b` | `rust_det` |
| `skip_endchannel_pairs` | `true` |
| `compute_connections` | `true` |
| `return_dgrid` | `true` |
| `vdw_radii_set` | `bondi` |
| `dgrid_method` | `rust_exact` |
| `auto_dgrid` | **`true`** — auto-switch large cells to `rust_aabb`; disable with `--no-auto-dgrid` |
| `window_fp` | `true` (disable with `--no-window-fp`) |

### Textural properties

| Key / flag | Default (web) |
|------------|---------------|
| `probe_radius` / `--probe-radius` | 1.2 Å (helium) |
| `probe_radius_psd` / `--probe-radius-psd` | 1.2 Å |
| `accessibility_method` / `--accessibility-method` | `winding` (`lattice` + `--lattice-variant v2`) |
| `sa_method` / `--sa-method` | `mc` (`mesh` optional) |
| `vol_method` | `voxel` |
| `mc_n_samples` | 20000 |
| `mc_shell_thickness` | 0.5 Å |
| `percolation_plot_backend` | `threejs` |
| `pocket_typing` | `none` (`--types` → `env_cluster`) |

### Plotting (batch CLI)

| Key / flag | Default |
|------------|---------|
| `plot_backend` / `--plot-backend` | **`plotly`** (Streamlit UI defaults to Three.js; override with `--plot-backend threejs` or config) |
| `plot_layout` / `--theme` | `dark` |

**Legacy profile** (`generate-config --profile legacy`): `rust_bucket`, mask 0.5, `crc` radii, `auto_dgrid` off — regression tests only.

## Key flags

### Textural analysis

| Flag | Output / behavior |
|------|-------------------|
| `--pore-dia` | `{stem}_pore_dia.csv` via `get_pore_dia_mgr_winding` + percolation history |
| `--percolation-plot` | `{stem}_percolation.html` |
| `--percolation-plot-backend` | `threejs` (config default) or `plotly` |
| `--psd` | `{stem}_psd_psd_uniform.csv`, `{stem}_psd_psd_pockets.csv` |
| `--psd-accessible` | `{stem}_psd_accessible_psd_uniform.csv`, `{stem}_psd_accessible_psd_pockets.csv` |
| `--sa` | `{stem}_sa_surface_area.csv` (MC + winding by default) |
| `--sa-method` | `mc` (default) or `mesh` |
| `--sa-plot-html` | `{stem}_surface_area_plot.html` (Plotly bars) |
| `--vol` | `{stem}_vol_void_volume.csv` (voxel + winding) |
| `--vol-plot-html` | `{stem}_void_volume_plot.html` (Plotly bars) |

### Pocket typing & visualization

| Flag | Behavior |
|------|----------|
| `--types` | `env_cluster` pocket typing → `regions_cluster` in NPZ |
| `--pocket-typing legacy_js` | RAG + Jensen–Shannon clustering (`--js-threshold`, default 0.15) |
| `--rag` | Save `{stem}_rag_clustered.pkl` (also saved when using `--types`) |
| `--plot-html` | `{stem}.html` after process |
| `--plot-backend threejs` | Override config `plot_backend` (default batch: plotly) |
| `--auto-params` | Estimate segmentation parameters from structure |

## Outputs

| File | When |
|------|------|
| `{stem}_results.npz` | Always (segmentation + optional `dgrid`, `regions_cluster`, embedded analyses) |
| CSV / HTML siblings | Per flags above |
| `{stem}.html` | `--plot-html` |
| `{stem}_rag_clustered.pkl` | `--rag` or `--types` |

Load NPZ: `numpy.load(path, allow_pickle=True)`; `connections` may be dict-encoded.

Re-plot without re-segmenting:

```bash
mgrcli plot out/{stem}_results.npz -c config/web_analysis_template.json \
  --plot-backend threejs -o out
```

## CLI vs web app

Shared library: `mgr_src/mgr/compute.py`, `mgr/winding_percolation.py`. Defaults: `python_cli/mgrcli/defaults.py`.

| Property | Web app | CLI batch | Match? |
|----------|---------|-----------|--------|
| Segmentation | `rust_det`, mask 0.05, bondi | `-c config/web_analysis_template.json` or `--profile web` | **Yes** |
| Pore LCD/PLD/DFSD | `get_pore_dia_mgr_winding` | `--pore-dia` | **Yes** |
| Percolation plot | Streamlit tab | `--percolation-plot` | **Yes** |
| SA | MC + winding (UI selectable mesh) | `--sa` / `--sa-method` | **Yes** |
| Vol | voxel + winding | `--vol` | **Yes** |
| Lattice accessibility | UI option | `--accessibility-method lattice` | **Yes** |
| PSD total + pocket | Rust uniform + pocketwise | `--psd` | **Yes** |
| PSD accessible | winding percolation | `--psd-accessible` | **Yes** |
| SA/vol breakdown CSV + bar HTML | Streamlit | `--sa`/`--vol` + `--sa-plot-html`/`--vol-plot-html` | **Yes** |
| Pocket typing | `env_cluster` | `--types` | **Yes** |

## Three.js full-scene config

Override `plot_backend` in JSON (template defaults to `plotly`):

```json
{
  "plot_backend": "threejs",
  "plot_framework": true,
  "plot_framework_show_rings": true,
  "plot_cell": true,
  "plot_surface": true,
  "plot_surface_pbc": true,
  "plot_connections": true,
  "plot_windows": true,
  "plot_maxima": true,
  "plot_show_periodic": true
}
```

```bash
mgrcli process file.cif -o out -c config.json --plot-html --types
```

## Pitfalls (agents)

1. **CWD:** run from repo root so `-c config/...` and `cifs/...` resolve.
2. **Large cells:** `auto_dgrid` is on; use `--no-auto-dgrid` only if you need to force `rust_exact`, or set `--dgrid-method rust_aabb` explicitly.
3. **Textural flags:** `--psd-accessible` / `--sa` / `--vol` need connections (auto-enabled).
4. **Stale Streamlit:** `mgrcli interactive --clean` kills old processes.
5. **Rust edits:** change `mgr_src/mgrcli_rust/` (built via `pyproject.toml` `manifest-path`), then `pip install --force-reinstall -e .`. Root `src/` is not the maturin build path.
6. **Config vs flags:** CLI flags override JSON config keys.
7. **HF Spaces deploy:** see repo `AGENTS.md` / `CLAUDE.md` — no `git push` to Hugging Face without explicit user consent.

## Reference paths

| Path | Role |
|------|------|
| `python_cli/mgrcli/cli.py` | CLI entry, flags, orchestration |
| `python_cli/mgrcli/defaults.py` | Web / legacy default dicts |
| `python_cli/mgrcli/textural_analysis.py` | Shared textural helpers |
| `python_cli/mgrcli/interactive_app.py` | Streamlit UI |
| `config/web_analysis_template.json` | Web-aligned analysis config |
| `mgr_src/mgrcli_rust/` | Rust extension sources (maturin build) |
| `cifs/` | Sample CIF structures |
| `docs/plan_cli_web_parity.md` | Parity roadmap |
| `docs/accessibility_and_surface_area.org` | Winding / lattice / mesh / MC APIs |

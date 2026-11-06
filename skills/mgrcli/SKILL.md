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

## Install (local, no Conda)

Not on PyPI. **Conda is not required.** Python **3.10–3.13** (tested on 3.13). Rust **1.70+** via rustup (`pymatgen` is pinned `<2026`).

### One command (macOS / Linux)

```bash
# public repo only — raw.githubusercontent.com does not use git credentials
curl -fsSL https://raw.githubusercontent.com/arung-northwestern/mgr_cli_py/master/install/install.sh | bash
# private repo, from a clone: ./install/install.sh
# then: source "$HOME/.mgrcli/venv/bin/activate"
```

### pip or uv from GitHub

```bash
pip install "git+https://github.com/arung-northwestern/mgr_cli_py.git"
uv pip install "git+https://github.com/arung-northwestern/mgr_cli_py.git"
uv tool install "git+https://github.com/arung-northwestern/mgr_cli_py.git"
```

Private checkout: `gh auth login` and `gh auth setup-git`. Do not embed a token.

### Development install

```bash
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                   # builds mgrcli_rust; mgr stays the live checkout
mgrcli --version
mgrcli process cifs/ZIF-8-RASPA.cif -o ./out/ --return-dgrid
mgrcli interactive --clean         # Streamlit; --clean kills a stale session
```

Windows: `install\install.bat` or the PowerShell steps in the README. After Rust edits: `pip install --force-reinstall --no-deps -e .`

### Interactive mode notes

- `mgrcli interactive` runs the installed `mgrcli/interactive_app.py` (in a checkout that file is `python_cli/mgrcli/interactive_app.py`).
- Sets local mode (`MGR_FORCE_LOCAL=1`) — no HF Spaces limits on your machine.
- HTS tab in the web UI is **local/Docker only**; from terminal use `mgrcli batch` instead.
- Sample structures: `cifs/` (`ZIF-8-RASPA.cif`, etc.).

## Commands

| Command | Purpose |
|---------|---------|
| `mgrcli process <cif>… -o OUT` | Segment + optional textural props + optional HTML |
| `mgrcli batch <cif>… -o OUT` | **Parallel** `process` via Dask (`--workers`, `--input-dir`, `--tar`) |
| `mgrcli generate-config -o config.json` | Emit defaults (`--profile web` or `legacy`) |
| `mgrcli plot results.npz -o OUT` | HTML from saved NPZ (no re-segmentation) |
| `mgrcli interactive [--clean]` | Streamlit web app |
| `mgrcli config-building` | Interactive config wizard |

Serial multi-file: `mgrcli process cifs/*.cif …` or `xargs`. Use `batch` when Dask parallelism helps.

## Quick analysis (web-aligned)

```bash
mgrcli process cifs/ZIF-8-RASPA.cif \
  -o ./out \
  -c config/web_analysis_template.json \
  --pore-dia --percolation-plot \
  --psd --psd-accessible --psd-plot-html \
  --sa --vol --sa-plot-html --vol-plot-html \
  --types \
  --plot-html --plot-backend threejs
```

**Parallel batch:**

```bash
mgrcli batch --input-dir cifs/ -o out/ \
  -c config/web_analysis_template.json \
  --workers 4 \
  --pore-dia --psd --psd-accessible --sa --vol
```

`--sa`, `--vol`, `--psd-accessible` auto-enable `return_dgrid` and `compute_connections`.

## Probe radii (per-analysis overrides)

| Flag / config key | Used for |
|-------------------|----------|
| `--probe-radius` / `probe_radius` | Global fallback (default 1.2 Å He) |
| `--sa-probe-radius` | `--sa` |
| `--vol-probe-radius` | `--vol` |
| `--psd-probe-radius` | Occupiable `--psd` sample-site clearance σ |
| `--psd-accessible-probe-radius` | `--psd-accessible` percolation probe |
| `--percolation-probe-radius` | `--percolation-plot` |
| `--probe-radius-psd` | `env_cluster` pocket typing only (not occupiable PSD) |

Example — SA at 1.2 Å, accessible PSD at 1.5 Å:

```bash
mgrcli process MOF.cif -o out/ -c config/web_analysis_template.json \
  --sa --sa-probe-radius 1.2 \
  --psd-accessible --psd-accessible-probe-radius 1.5
```

## PSD sampling (`--psd` / config JSON)

| Flag / config | Default |
|---------------|---------|
| `--psd-num-samples-total` / `psd_num_samples_total` | 10000 |
| `--psd-num-samples-per-region` / `psd_num_samples_per_region` | 1000 |
| `--psd-uniform-num-bins` / `psd_uniform_num_bins` | 200 |
| `--psd-pocket-num-bins` / `psd_pocket_num_bins` | 200 |
| `--psd-uniform-max-diameter` | 50.0 Å |
| `--psd-pocket-max-diameter` | 50.0 Å |
| `--psd-uniform-log-bins` | off |
| `--psd-plot-html` | `{stem}_psd_plot.html` (and accessible variant) |

## Segmentation defaults (web profile)

| Key / flag | Default |
|------------|---------|
| `spacing` / `-s` | 0.5 Å |
| `mask_thickness` / `-mt` | 0.05 Å |
| `watershed_backend` / `-b` | `rust_det` |
| `vdw_radii_set` | `bondi` |
| `dgrid_method` | `rust_exact` |
| `auto_dgrid` | **on**; `--no-auto-dgrid` to disable |
| `min_box_thickness` / `--min-box-thickness` | off (optional supercell convenience) |
| `supercell_nx/ny/nz` / `--supercell NX NY NZ` | 1 1 1 |
| `window_fp` | on; `--no-window-fp` to disable |
| `plot_backend` | `plotly` (web UI opens on Three.js; `--plot-backend threejs`) |
| `plot_surface_screen_door` | on; `--no-screen-door` |
| `compute_aromatic_rings` / `--aromatic-rings` | off (experimental) |
| `plot_explode` / `--explode` | 0 (Three.js HTML; `--explode-plane ab\|bc\|ca`) |
| `plot_window_min_radius` / `--window-min-radius` | off; hides small windows and, by default, edges |
| `cache_dir` / `--cache-dir` | unset; `--no-cache` forces recompute |

## Textural flags & outputs

| Flag | Output |
|------|--------|
| `--pore-dia` | `{stem}_pore_dia.csv` and `{stem}_percolation_history.json` |
| `--percolation-history` | `{stem}_percolation_history.json` (also with `--pore-dia` / `--percolation-plot`) |
| `--percolation-plot` | `{stem}_percolation.html` (PLD-tab snapshot) |
| `--psd` | `{stem}_psd_psd_uniform.csv`, `{stem}_psd_psd_pockets.csv` |
| `--psd-accessible` | `{stem}_psd_accessible_psd_*.csv` |
| `--psd-plot-html` | `{stem}_psd_plot.html` |
| `--sa` | `{stem}_sa_surface_area.csv` |
| `--sa-plot-html` | `{stem}_surface_area_plot.html` |
| `--vol` | `{stem}_vol_void_volume.csv` |
| `--vol-plot-html` | `{stem}_void_volume_plot.html` |
| `--types` | `env_cluster` → `regions_cluster` in NPZ |
| `--plot-html` | `{stem}.html` |

Always: `{stem}_results.npz`.

## CLI vs web app

| Feature | CLI | Web | Match? |
|---------|-----|-----|--------|
| Segmentation web profile | `-c config/web_analysis_template.json` | UI | **Yes** |
| HTS parallel batch | `mgrcli batch` (Dask) | HTS tab (local only) | **Yes** |
| Per-analysis probe radii | `--sa-probe-radius`, etc. | single global in HTS | **CLI richer** |
| PSD sampling + plot | flags + `--psd-plot-html` | PSD tab / HTS | **Yes** |
| Min box thickness | `--min-box-thickness` | HTS checkbox | **Yes** (convenience) |
| Crystallographic supercell | `--supercell` | sidebar repeats | **Yes** |
| Aromatic rings | `--aromatic-rings` | experimental checkbox, off | **Yes** |
| Opacities, windows, explode | flags above | visualization HUD | **Yes** (explode is the initial Three.js value) |
| Percolation HTML + history JSON | `--percolation-plot`, `--percolation-history` | PLD tab downloads | **Yes** |
| Segmentation cache | `--cache-dir` / `--no-cache` | in-memory cache / Force recompute | **Yes** (CLI cache is on disk) |
| Plot engine default | `plotly` unless `--plot-backend threejs` | Three.js selected | CLI stays Plotly until the flag |
| Lattice accessibility | `--accessibility-method lattice` | not in single-material UI | CLI only |
| Movies | not in this release | not a batch output | leave `movies.py` playback alone |

## Pitfalls

1. **CWD:** repo root for `config/` and `cifs/` paths.
2. **Rust edits:** `mgr_src/mgrcli_rust/`, then `pip install --force-reinstall --no-deps -e .`.
3. **Batch:** needs `dask[distributed]` (declared in `pyproject.toml`).

## Reference paths

| Path | Role |
|------|------|
| `python_cli/mgrcli/cli.py` | CLI entry, `process` / `batch` |
| `python_cli/mgrcli/batch.py` | Dask parallel runner |
| `python_cli/mgrcli/defaults.py` | Web / legacy defaults |
| `config/web_analysis_template.json` | Web-aligned config |
| `docs/plan_cli_web_parity.md` | Parity notes |

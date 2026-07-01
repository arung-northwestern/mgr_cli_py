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

Run from the **repository root** after clone.

```bash
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e .

mgrcli --help
mgrcli process cifs/ZIF-8-RASPA.cif -o ./out/ --return-dgrid
```

**Requirements:** Python **3.10+** recommended (`requires-python >=3.8`). Rust **1.70+**. Sample CIFs: `cifs/`.

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
| `window_fp` | on; `--no-window-fp` to disable |
| `plot_backend` | `plotly` (override `--plot-backend threejs`) |

## Textural flags & outputs

| Flag | Output |
|------|--------|
| `--pore-dia` | `{stem}_pore_dia.csv` |
| `--percolation-plot` | `{stem}_percolation.html` |
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
| Lattice accessibility | `--accessibility-method lattice` | not in single-material UI | CLI only |

## Pitfalls

1. **CWD:** repo root for `config/` and `cifs/` paths.
2. **Rust edits:** `mgr_src/mgrcli_rust/`, then `pip install --force-reinstall -e .`.
3. **Batch:** needs `dask[distributed]` (in `pyproject.toml`); not on HF Spaces.
4. **HF push:** see `AGENTS.md` — no push without explicit user consent.

## Reference paths

| Path | Role |
|------|------|
| `python_cli/mgrcli/cli.py` | CLI entry, `process` / `batch` |
| `python_cli/mgrcli/batch.py` | Dask parallel runner |
| `python_cli/mgrcli/defaults.py` | Web / legacy defaults |
| `config/web_analysis_template.json` | Web-aligned config |
| `docs/plan_cli_web_parity.md` | Parity notes |

---
name: mgrcli
description: >-
  Run Mofography MOF analysis via the mgrcli CLI (batch process, textural
  properties, Three.js HTML, interactive Streamlit). Use when analyzing CIF
  files, generating configs, comparing CLI vs web defaults, or running
  segmentation / SA / vol / PSD / pore diameter workflows.
---

# mgrcli — MOF analysis CLI

Repository: `arung-northwestern/mgr_cli_py`. Batch CLI + `mgrcli interactive` (Streamlit). Core numerics: `mgrcli_rust` + `mgr_src/mgr`.

**Cursor:** symlink or copy this file to `.cursor/skills/mgrcli/SKILL.md` for auto-discovery.

## Install (local)

```bash
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e .
# After Rust changes: pip install --force-reinstall -e .
mgrcli --help
```

Requires Python 3.10+ and Rust 1.70+ for the extension build.

## Commands

| Command | Purpose |
|---------|---------|
| `mgrcli process <cif>… -o OUT` | Segment + optional textural props + optional HTML |
| `mgrcli generate-config -o config.json` | Emit defaults (`--profile web` or `legacy`) |
| `mgrcli plot results.npz -o OUT` | HTML from saved NPZ |
| `mgrcli interactive [--clean]` | Streamlit web app |
| `mgrcli config-building` | Interactive config wizard |

Working directory for module invocations: `python_cli/` or any env where `mgrcli` is on PATH.

## Quick analysis template

Copy and edit `config/web_analysis_template.json` (repo root), then:

```bash
mgrcli process structure.cif \
  -o ./out \
  -c config/web_analysis_template.json \
  --pore-dia --psd --psd-accessible --sa --vol \
  --types \
  --plot-html --plot-backend threejs
```

**Full textural recipe (batch):**

```bash
mgrcli process MOF.cif -o out/ \
  -c config/web_analysis_template.json \
  --return-dgrid \
  --pore-dia \
  --psd \
  --psd-accessible \
  --sa \
  --vol \
  --probe-radius 1.2
```

`--sa` / `--vol` / `--psd-accessible` auto-enable `return_dgrid` and `compute_connections` if missing.

## Key flags

### Segmentation

| Flag / config | Default (web profile) |
|---------------|----------------------|
| `--spacing` / `-s` | 0.5 Å |
| `--mask-thickness` | 0.05 Å |
| `-b` / `--watershed-backend` | `rust_det` |
| `--vdw-radii-set` | `bondi` |
| `--dgrid-method` | `rust_exact` |
| `--auto-dgrid-method` | off; opt-in AABB for large cells |
| `--no-window-fp` | window FP on by default for SA/vol |

### Textural properties

| Flag | Output |
|------|--------|
| `--pore-dia` | `*_pore_dia.csv` (LCD, PLD, DFSD, percolation history count) |
| `--psd` | `*_psd_psd_uniform.csv`, `*_psd_psd_pockets.csv` (total / occupiable) |
| `--psd-accessible` | `*_psd_accessible_*.csv` (percolation-accessible at probe) |
| `--sa` | `*_sa_surface_area.csv` (total, accessible, by region/pocket/cluster) |
| `--vol` | `*_vol_void_volume.csv` (same breakdown shape) |
| `--probe-radius` | He default 1.2 Å if unset; used for SA/vol/accessible PSD |
| `--probe-radius-psd` | Pocket typing PSD probe (default 1.2 Å) |

### Pocket typing & viz

| Flag | Behavior |
|------|----------|
| `--types` | `env_cluster` pocket typing → `regions_cluster` in NPZ |
| `--pocket-typing legacy_js` | Old RAG+JS path |
| `--plot-html` | Standalone HTML after process |
| `--plot-backend threejs` | Three.js (default batch: plotly) |
| `--rag` | Save `*_rag_clustered.pkl` |

## Outputs

- `{stem}_results.npz` — segmentation arrays, optional `dgrid`, `regions_cluster`, analyses
- CSV siblings per flag above
- `{stem}.html` when `--plot-html`

Load NPZ in Python: `numpy.load(path, allow_pickle=True)`; connections may be dict-encoded.

## Textural properties: CLI vs web app

Shared library: `mgr_src/mgr/compute.py`, `mgr/winding_percolation.py`. Defaults in `python_cli/mgrcli/defaults.py`.

| Property | Web app | CLI | Same algorithm / defaults? |
|----------|---------|-----|----------------------------|
| Segmentation | `rust_det`, mask 0.05, bondi, `skip_endchannel_pairs` | Web profile matches | **Yes** (use `-c` web template or `generate-config --profile web`) |
| Pore LCD/PLD/DFSD | `get_pore_dia_mgr_winding` | `--pore-dia` uses `get_pore_dia_mgr` + winding **history** | **Partial** — LCD/PLD path differs; history is winding |
| Percolation history | `compute_percolation_history_winding` | Same when segmentation complete | **Yes** |
| Percolation cluster **plot** | Plotly / Three.js interactive tab | Not exported in batch | **No** — CSV only via `--pore-dia` |
| SA accessible | `compute_surface_area_mesh_winding`, `window_fp=True` | `--sa` → same function, `window_fp` from config | **Yes** (mesh only) |
| SA Monte Carlo | Available in library, not main UI default | Not exposed (`--sa-method` missing) | **No** |
| Vol accessible | `compute_void_volume_from_results_winding` | `--vol` → same | **Yes** |
| Lattice accessibility | Library only (`*_lattice` functions) | Not exposed | **No** |
| PSD total (global + pocket) | Rust uniform + pocketwise | `--psd` | **Yes** |
| PSD accessible | `compute_psd_global_accessible`, pocketwise accessible | `--psd-accessible` | **Yes** (needs connections) |
| SA/vol breakdown CSV | fragment / pocket / cluster columns | `--sa` / `--vol` CSV | **Yes** |
| SA/vol breakdown **bar HTML** | Streamlit plots | Not in CLI | **No** |
| Pocket typing | `env_cluster` | `--types` | **Yes** |

For Zeo++-style lattice comparisons, call `mgr.compute_surface_area_mc_lattice` etc. from Python on saved NPZ — not yet CLI flags.

## Three.js full-scene config

Enable all traces in config JSON:

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

Then: `mgrcli process file.cif -o out -c config.json --plot-html --types`

Re-plot without re-segmenting: `mgrcli plot out/file_results.npz -c config.json --plot-backend threejs -o out`

## Pitfalls

1. **Large cells**: use `--auto-dgrid-method` or `--dgrid-method rust_aabb` to avoid OOM on `rust_exact`.
2. **`--psd-accessible` / `--sa` / `--vol`**: need `compute_connections: true` (auto-enabled).
3. **Stale Streamlit**: `mgrcli interactive --clean`.
4. **Rust rebuild**: after editing `src/`, run `pip install --force-reinstall -e .`.
5. **Config JSON**: use `generate-config --profile web`; merge overrides with CLI flags (flags win).

## Reference paths

| Path | Role |
|------|------|
| `python_cli/mgrcli/cli.py` | CLI entry, flags, orchestration |
| `python_cli/mgrcli/defaults.py` | Shared web/legacy defaults |
| `python_cli/mgrcli/interactive_app.py` | Streamlit UI |
| `config/web_analysis_template.json` | Copy-paste analysis config |
| `docs/plan_cli_web_parity.md` | Parity roadmap |
| `docs/accessibility_and_surface_area.org` | Winding/lattice/mesh/MC APIs |

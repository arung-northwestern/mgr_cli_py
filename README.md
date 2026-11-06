# Mofography CLI Tool

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)

A high-performance command-line interface for analyzing Metal-Organic Frameworks (MOFs) using advanced segmentation algorithms. This repository ships a **terminal CLI** (`mgrcli`) and an **interactive Streamlit web app** (`mgrcli interactive`), powered by a **Rust extension** (`mgrcli_rust`, PyO3) compiled from the workspace root and Python packages under `python_cli/`.

## Table of Contents
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Verify](#verify)
  - [How the Rust backend is built](#how-the-rust-backend-is-built)
  - [Troubleshooting](#troubleshooting)
- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Textural properties (CLI vs web app)](#textural-properties-cli-vs-web-app)
- [AI agents](#ai-agents)
- [Output Format](#output-format)
- [Examples](#examples)
- [Workflow Diagram](#workflow-diagram)
- [Reading Results in Python](#reading-results-in-python)
- [Testing](#testing)
- [Development](#development)
- [Performance](#performance)
- [License](#license)
- [Contributing](#contributing)
- [Citation](#citation)
- [Next Steps](#next-steps)

## Features

* **High-Performance Analysis**: Rust-powered segmentation algorithms for fast MOF analysis
* **Interactive Browser Mode**: Streamlit-based web interface for visual analysis
* **Batch Processing**: Process multiple CIF files in parallel
* **Flexible Configuration**: Extensive parameter control with sensible defaults
* **Compressed Output**: Efficient NPZ format for large datasets
* **Cross-Platform**: Works on Windows, Linux, and macOS
* **Rich Output**: Multiple analysis outputs for comprehensive characterization
* **Smart Defaults**: Pre-configured parameters for common use cases

## Why Choose CLI Over Python Package?

* **🚀 Quick Use**: Analyze materials instantly without opening Python files or IPython sessions
* **⚡ Fast Processing**: Optimized for rapid analysis of one or two materials without overhead
* **🤖 AI Agent Friendly**: Perfect for AI assistants and agents without requiring MCP server context
* **🎯 Distraction-Free**: For TUI lovers who prefer focused, command-line workflows
* **🔄 Automation Ready**: Easy integration into scripts, pipelines, and automated workflows
* **💾 Resource Efficient**: Lower memory footprint compared to full Python environment

## Relationship to Mofography Python Package

This CLI tool is **separate** from the main Mofography Python package but maintains compatibility with the same core algorithms and reference functions. It's kept up-to-date with the latest computational methods while offering the advantages of a dedicated command-line interface and interactive web mode.

## Architecture

### Backend technologies
* **`mgrcli_rust` (Rust)**: Native extension module (PyO3) with the heavy numerical work: distance grids, watershed, maxima detection, window connections, and related kernels. The CLI and the Streamlit app both import this module.
* **Python (`python_cli/mgrcli/`)**: `argparse` CLI, Streamlit UI (`interactive_app.py`), plotting glue, orchestration, and saving results (for example `.npz`).
* **PyO3 + NumPy**: Rust functions receive and return NumPy arrays where appropriate.

### Which `Cargo.toml` is used
The Rust crate lives in **`mgr_src/mgrcli_rust/`** (`Cargo.toml` + `src/`). `pyproject.toml` points maturin at it, and `pip install` compiles it into `mgrcli_rust`.

### Where Python vs Rust is used
* **Python**: CLI, Streamlit, configuration, file I/O, Plotly/pyvista workflows, Dask/HTS drivers.
* **Rust (`mgrcli_rust`)**: Core segmentation algorithms, numerical kernels, parallelism (Rayon), and tight array work.

> **Note**: This is completely standalone from the main mofography Python library, although the core segmentation algorithms (distance grids, watershed, maxima detection, and window connections) use the same Rust backend functions. However, all bookkeeping operations (CIF file reading, atomic radii assignment, periodic boundary condition handling, and reindexing) are implemented entirely in Rust here, which may produce slightly different results compared to the Python version.

<a id="installation"></a>

# Installation

mgrcli is **not on PyPI**. Install from this Git repository. `pip install` compiles the Rust extension `mgrcli_rust` in release mode with [maturin](https://www.maturin.rs/). The first compile is often 5–15 minutes. There are no prebuilt wheels: a local release build matches your CPU.

<a id="one-command"></a>

## One command (macOS / Linux)

This installs from the `master` branch:

```bash
curl -fsSL https://raw.githubusercontent.com/arung-northwestern/mgr_cli_py/master/install/install.sh | bash
```

Pin a branch or tag:

```bash
curl -fsSL https://raw.githubusercontent.com/arung-northwestern/mgr_cli_py/master/install/install.sh | MGRCLI_REF=master bash
```

The script detects the OS, installs [rustup](https://rustup.rs/) when `cargo` is missing (no sudo), creates a virtualenv at `$MGRCLI_HOME` (default `$HOME/.mgrcli`; `uv` if it is on `PATH`, otherwise `python -m venv`), installs this repo, and checks `mgrcli --version`. Re-running it is safe.

From a clone, the same script installs **that** checkout and does not switch branches:

```bash
./install/install.sh
```

<a id="pip"></a>

## pip

```bash
pip install "git+https://github.com/arung-northwestern/mgr_cli_py.git"
```

<a id="uv"></a>

## uv

```bash
uv pip install "git+https://github.com/arung-northwestern/mgr_cli_py.git"
uv tool install "git+https://github.com/arung-northwestern/mgr_cli_py.git"
```

`uv tool install` puts the `mgrcli` command on your PATH without activating a project venv.

<a id="from-source"></a>

## From source (development)

```bash
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

`pip install -e .` is the development install: it compiles `mgrcli_rust` and points `mgrcli` at this checkout. It also creates `python_cli/mgr` as a symlink to `mgr_src/mgr` so `import mgr` follows the live tree. Do not edit or commit `python_cli/mgr`; edit `mgr_src/mgr`. After you change Rust (`mgr_src/mgrcli_rust/`), rebuild with:

```bash
pip install --force-reinstall --no-deps -e .
```

A non-editable install of the same tree is `pip install .`. That builds a wheel, installs `mgrcli`, `mgr`, and `mgrcli_rust` into the environment, and does not need the checkout on `PYTHONPATH`.

<a id="windows"></a>

## Windows

```powershell
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py
.\install\install.bat
```

The batch file creates `%USERPROFILE%\.mgrcli\venv` (or `%MGRCLI_HOME%\venv`), installs this checkout, and runs `mgrcli --version`. It does not request administrator rights.

Manual steps, if you prefer PowerShell without the batch file:

```powershell
winget install Rustlang.Rustup
# open a new terminal so cargo is on PATH
py -3 -m venv $env:USERPROFILE\.mgrcli\venv
& $env:USERPROFILE\.mgrcli\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
mgrcli --version
```

If PowerShell refuses to activate the venv, allow local scripts for your user: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

<a id="prerequisites"></a>

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **OS** | macOS, Linux, or Windows. The curl script is macOS/Linux. Windows uses `install\install.bat` or the PowerShell steps above. |
| **Python** | **3.10–3.13** is the practical range. `pyproject.toml` still says `requires-python >= 3.8`, but current `pymatgen` and `freud-analysis` wheels do not support 3.8 or 3.9. **Tested:** CPython 3.13 on macOS arm64 with this install flow. 3.11 and 3.12 are the usual sweet spot if 3.13 wheels for a dependency are missing. |
| **pymatgen** | Pinned `pymatgen>=2023.1,<2026` so a 2026 release cannot change the CLI’s Python support or API expectations without an explicit bump. |
| **RDKit** | Declared as `rdkit` (PyPI). `import mgr` loads `mgr.analyze`, which imports RDKit. |
| **Rust** | [rustup](https://rustup.rs/), stable toolchain, 1.70 or newer (`rustc`, `cargo` on `PATH`). The installer runs rustup when `cargo` is missing. Set `MGRCLI_SKIP_RUST=1` to refuse that and print the manual step instead. |
| **Compiler** | Xcode Command Line Tools on macOS, or a C compiler / `build-essential` on Linux. The extension is compiled on your machine. |
| **Git** | Required for clone and for `pip install git+https://...`. |

**CuPy** is not a dependency. A few `mgr` helpers import it only when you call a GPU entry point. Install a CuPy build that matches your CUDA stack if you use those.



<a id="verify"></a>

## Verify

```bash
mgrcli --version
mgrcli process cifs/ZIF-8-RASPA.cif -o ./out_test
python -c "import mgrcli, mgr, mgrcli_rust; print(mgrcli.__version__)"
```

`cifs/ZIF-8-RASPA.cif` ships in the git checkout. A non-editable install does not bundle sample CIFs; pass a path to a CIF you have. `mgrcli interactive` launches Streamlit on the installed `interactive_app.py` (next to the `mgrcli` package) and does not need the repository root on `PYTHONPATH`.

<a id="how-the-rust-backend-is-built"></a>

## How the Rust backend is built

| Approach | When to use it | What compiles Rust? |
|----------|----------------|---------------------|
| **`./install/install.sh`** or the curl one-liner | First install on macOS/Linux | pip / uv invokes maturin, which runs `cargo` in **release** |
| **`pip install "git+https://github.com/arung-northwestern/mgr_cli_py.git"`** | Install without a long-lived clone | same |
| **`pip install -e .`** | Day-to-day development in a clone | same, and Python edits are picked up from the checkout |
| **`pip install .`** | Non-editable install of a clone | same; `mgrcli`, `mgr`, and `mgrcli_rust` are copied into the environment |
| **`cargo build --release` alone** | Check that the crate compiles | Does **not** install `mgrcli_rust` into Python |

You do not need a separate “compile Rust, then pip” sequence. `rustc` and `cargo` must be on `PATH` before `pip install`.

<a id="troubleshooting"></a>

## Troubleshooting

**`mgrcli` is not found**

The command is installed inside the virtualenv, not as a system package.

```bash
source "$HOME/.mgrcli/venv/bin/activate"   # or your MGRCLI_HOME
mgrcli --version
python -m mgrcli.cli --version
```

**Rust compilation failed**

```bash
rustup update stable
rustc --version    # 1.70 or newer
cargo --version
```

From a checkout, after a failed build:

```bash
pip uninstall -y mgrcli
rm -rf build/ target/
pip install -e .
```

On macOS, if the compiler is missing: `xcode-select --install`.

**`No module named mgrcli_rust` or `No module named mgr`**

The extension or the `mgr` package did not land in the environment. Reinstall from the repository root (the directory that contains `pyproject.toml`):

```bash
pip install --force-reinstall --no-deps .
python -c "import mgrcli, mgr, mgrcli_rust"
```

**`pip` cannot find a version of pymatgen or freud-analysis**

Use Python 3.10, 3.11, 3.12, or 3.13. Python 3.8 and 3.9 are below the wheels those libraries publish. The pin `pymatgen<2026` is intentional.

**Windows: `install.bat` cannot find Python or cargo**

Install Python from python.org (enable “Add python.exe to PATH”) or use `py -3`, install rustup, open a **new** terminal, and run `install\install.bat` again.

**Interactive app**

```bash
mgrcli interactive --clean
```

`--clean` stops a leftover Streamlit process for this app. The app file is the installed `mgrcli/interactive_app.py`.

## Quick Start

Once installed, MCLI is ready to analyze your CIF files:

### Full analysis (template config)

Copy [`config/web_analysis_template.json`](config/web_analysis_template.json) or generate web-aligned defaults:

```bash
mgrcli generate-config --profile web -o my_config.json
```

Run segmentation plus textural properties (web-aligned defaults):

```bash
mgrcli process structure.cif \
  -o ./out \
  -c config/web_analysis_template.json \
  --pore-dia --percolation-plot \
  --psd --psd-accessible --sa --vol \
  --sa-plot-html --vol-plot-html \
  --probe-radius 1.2 \
  --types \
  --plot-html --plot-backend threejs
```

**Default textural methods** (web app + `generate-config --profile web`):

| Property | Method |
|----------|--------|
| Pore diameter | IMGR winding (`get_pore_dia_mgr_winding`) |
| Surface area | Monte Carlo shell + winding accessibility |
| Void volume | Voxel integration + winding accessibility |
| Distance grid | `rust_exact`, auto-switch to `rust_aabb` on large cells |

### Basic Usage

```bash
# Show the banner
mgrcli --banner

# Process a single CIF file with default settings
mgrcli process my_structure.cif

# Process multiple CIF files
mgrcli process *.cif -o ./results/

# Process with custom parameters
mgrcli process my_mof.cif --spacing 0.3 -H 0.5 --mask-thickness 0.8

# Launch interactive web interface
mgrcli interactive

# Get help and see all options
mgrcli --help
mgrcli process --help
```

### Expected Output

Each command generates:
- **`.npz` file**: Compressed results with all analysis data
- **Console output**: Progress indicators and timing information
- **Compatible format**: Can be loaded with `numpy.load()`

### Next Steps

- **[Configuration](#configuration)**: Customize analysis parameters
- **[Examples](#examples)**: See advanced usage patterns
- **[Reading Results](#reading-results-in-python)**: How to work with output files

## Commands Overview

MCLI provides several commands for different tasks:

```bash
mgrcli process          # Analyze CIF files with comprehensive segmentation and analysis
mgrcli batch            # Parallel process many CIFs (Dask; same outputs as process)
mgrcli generate-config  # Create configuration files with custom parameters
mgrcli config-building  # Interactive configuration builder with guided prompts
mgrcli plot            # Generate Plotly visualizations from existing results
mgrcli interactive     # Launch interactive Streamlit web application
```

## Configuration

### Default template

[`config/web_analysis_template.json`](config/web_analysis_template.json) — web-aligned segmentation defaults (bondi VDW, `rust_det`, mask 0.05 Å, winding accessibility settings). See [`config/README.md`](config/README.md).

### Generate config

```bash
mgrcli generate-config --profile web -o config.json   # Streamlit-aligned (default)
mgrcli generate-config --profile legacy -o legacy.json  # old batch defaults
```

### Customize parameters

```bash
mgrcli generate-config \
    --profile web \
    --spacing 0.4 \
    --mask-thickness 0.05 \
    -o custom_config.json
```

### Interactive builder

```bash
mgrcli config-building
```

### Analysis parameters (web profile)

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `spacing` | `-s` | 0.5 | Grid spacing (Å) |
| `mask-thickness` | `-mt` | 0.05 | Void mask thickness (Å) |
| `watershed-backend` | `-b` | `rust_det` | Watershed backend |
| `vdw-radii-set` | | `bondi` | VDW table for distance grid |
| `minimum-window-separation` | `-ms` | 1.0 | Window separation (Å) |
| `dgrid-method` | | `rust_exact` | Distance grid method |
| `auto_dgrid` | `--no-auto-dgrid` | **on** | Auto-switch to `rust_aabb` for large cells |
| `sa_method` | `--sa-method` | `mc` | Surface area: Monte Carlo (or `mesh`) |
| `accessibility_method` | `--accessibility-method` | `winding` | Winding percolation (or `lattice`) |
| `probe-radius` | | 1.2 | Global default probe Å (fallback for all analyses) |
| `sa_probe_radius` | `--sa-probe-radius` | 1.2 | SA probe (overrides global) |
| `vol_probe_radius` | `--vol-probe-radius` | 1.2 | Void volume probe |
| `psd_probe_radius` | `--psd-probe-radius` | 1.2 | Occupiable PSD sample-site clearance σ |
| `psd_accessible_probe_radius` | `--psd-accessible-probe-radius` | 1.2 | Accessible PSD percolation probe |
| `percolation_probe_radius` | `--percolation-probe-radius` | 1.2 | Percolation cluster plot probe |
| `probe_radius_psd` | `--probe-radius-psd` | 1.2 | env_cluster pocket typing PSD only |
| `psd_num_samples_total` | `--psd-num-samples-total` | 10000 | Uniform PSD samples |
| `psd_num_samples_per_region` | `--psd-num-samples-per-region` | 1000 | Pocket-wise PSD samples / region |
| `psd_uniform_num_bins` | `--psd-uniform-num-bins` | 200 | Uniform PSD histogram bins |
| `psd_pocket_num_bins` | `--psd-pocket-num-bins` | 200 | Pocket PSD histogram bins |
| `psd_uniform_max_diameter` | `--psd-uniform-max-diameter` | 50.0 | Uniform PSD max diameter (Å) |
| `psd_pocket_max_diameter` | `--psd-pocket-max-diameter` | 50.0 | Pocket PSD max diameter (Å) |
| `psd_uniform_log_bins` | `--psd-uniform-log-bins` | off | Log-spaced uniform PSD bins |
| `min_box_thickness` | `--min-box-thickness` | off | Optional supercell convenience (Å) |
| `window_fp` | `--no-window-fp` | on | Window fingerprint rescue (SA/vol winding) |

Legacy profile differences: `mask_thickness=0.5`, `rust_bucket`, `crc` radii.

### Analysis flags

| Flag | Description |
|------|-------------|
| `--pore-dia` | LCD, PLD, DFSD + winding percolation history (CSV) |
| `--percolation-plot` | Percolation cluster visualization HTML (default backend: threejs) |
| `--percolation-plot-backend` | `threejs` or `plotly` |
| `--psd` | Global + pocketwise total PSD (CSV); uses `--psd-probe-radius` for occupiable clearance |
| `--psd-accessible` | Accessible PSD at `--psd-accessible-probe-radius` (CSV) |
| `--psd-plot-html` | Uniform PSD Plotly line chart HTML |
| `--sa` | Accessible surface area (default: MC + winding; CSV breakdown) |
| `--sa-method` | `mc` (default) or `mesh` |
| `--sa-plot-html` | Surface area breakdown bar chart HTML |
| `--vol` | Accessible void volume (voxel + winding; CSV breakdown) |
| `--vol-plot-html` | Void volume breakdown bar chart HTML |
| `--accessibility-method` | `winding` (default) or `lattice` |
| `--no-auto-dgrid` | Disable automatic rust_exact → rust_aabb switching |
| `--types` | Pocket typing (`env_cluster`) |
| `--pocket-typing {env_cluster,legacy_js,none}` | Explicit typing method |
| `--plot-html` | Standalone HTML after process |
| `--plot-backend {plotly,threejs}` | Visualization engine (default: plotly) |
| `--rag` | Save RAG pickle |
| `--return-dgrid` | Include distance grid in NPZ |
| `--no-connections` | Disable connection graph |
| `--min-box-thickness` | Optional: repeat cell so shortest axis ≥ value (Å) |

### Structure, plots, and PLD downloads

These match the Streamlit sidebar and the PLD tab. Every flag is also a key in `mgrcli generate-config` JSON (snake_case dest, except `--supercell NX NY NZ` → `supercell_nx`, `supercell_ny`, `supercell_nz`).

| Flag | Config key | Default | What it does |
|------|------------|---------|--------------|
| `--supercell NX NY NZ` | `supercell_nx/ny/nz` | 1 1 1 | Crystallographic repeats before analysis |
| `--aromatic-rings` | `compute_aromatic_rings`, `plot_framework_show_rings` | off | Experimental ring detection and highlights |
| `--framework-style` | `plot_framework_style` | scatter (Plotly); instanced when `--plot-backend threejs` and the key is unset | `scatter`, `mesh`, or `instanced` |
| `--framework-quality` | `plot_framework_mesh_quality`, `plot_quality` | low / medium | Three.js preset; Plotly surface step uses `plot_quality` |
| `--atom-scale` | `plot_framework_mesh_scale` | 0.65 | Atom sphere scale |
| `--bond-radius` | `plot_framework_bond_radius` | 0.15 | Bond cylinder radius (Å) |
| `--projection` | `plot_projection` | perspective | `perspective` or `orthographic` |
| `--surface-isovalue` | `plot_surface_isovalue` | 1.0 | Surface isovalue (Å) |
| `--surface-opacity` | `plot_surface_opacity` | 1.0 | Pore surface opacity |
| `--surface-pbc-opacity` | `plot_surface_pbc_opacity` | 1.0 | Periodic surface opacity |
| `--surface-types-opacity` | `plot_surface_types_opacity` | 1.0 | Pocket-type surface opacity |
| `--no-screen-door` | `plot_surface_screen_door` | on | Screen-door transparency (web default on) |
| `--no-pore-graph` | `plot_connections` | shown | Hide pore-graph edges |
| `--no-windows` | `plot_windows` | shown | Hide window markers |
| `--window-marker-color` | `plot_windows_marker_color` | `#FFFF00` | Window marker color |
| `--window-marker-size` | `plot_windows_marker_size` | 10 | Window marker size |
| `--window-min-radius` | `plot_window_min_radius` | off | Hide windows below this radius (Å) |
| `--no-hide-edges-below-window` | `plot_hide_edges_below_window` | on | Keep edges when the window filter would drop them |
| `--explode` | `plot_explode` | 0 | Initial Three.js exploded-view offset (Å) |
| `--explode-plane` | `plot_explode_plane` | `ab` | `ab`, `bc`, or `ca` |
| `--no-surface` / `--no-surface-pbc` | `plot_surface` / `plot_surface_pbc` | shown | Hide surface meshes |
| `--no-cell` / `--no-framework` / `--no-maxima` | `plot_cell` / `plot_framework` / `plot_maxima` | shown | Hide those traces |
| `--maxima-spheres` | `plot_maxima_show_spheres` | off | Spherical maxima (Plotly) |
| `--show-periodic` | `plot_show_periodic` | off | Periodic images |
| `--percolation-history` | (output flag) | with `--pore-dia` or `--percolation-plot` | `{stem}_percolation_history.json` |
| `--filter-graph-at-probe` | `percolation_filter_graph_at_probe` | off | Drop percolation-plot edges below the probe |
| `--percolation-isovalue` | `percolation_cluster_isovalue` | surface isovalue | Cluster mesh isovalue (Å) |
| `--percolation-boundary-color` | `percolation_boundary_color` | `#000000` | Plotly boundary color |
| `--percolation-boundary-width` | `percolation_boundary_width` | 6 | Plotly boundary width |
| `--percolation-window-marker-color` | `percolation_window_marker_color` | window color | PLD-tab window color |
| `--percolation-window-marker-size` | `percolation_window_marker_size` | window size | PLD-tab window size |
| `--cache-dir DIR` | `cache_dir` | unset | Reuse a matching segmentation NPZ |
| `--no-cache` | `use_result_cache` | cache used when `cache_dir` is set | Recompute (web “Force recompute”) |

`--pore-dia` still writes `{stem}_pore_dia.csv`. `--percolation-plot` writes `{stem}_percolation.html` (the PLD tab’s HTML snapshot). The history JSON is written whenever `--pore-dia`, `--percolation-plot`, or `--percolation-history` is set and a history was computed.

Example (headless, scriptable):

```bash
mgrcli process structure.cif -o ./out \
  --supercell 2 2 1 \
  --plot-html --plot-backend threejs --framework-style instanced \
  --surface-opacity 0.6 --window-min-radius 1.0 --explode 1.5 --explode-plane ab \
  --pore-dia --percolation-plot --percolation-history \
  --filter-graph-at-probe --cache-dir ./cache
```

Movies are not part of the CLI yet.

### CLI vs web app

| Feature | CLI | Web app |
|---------|-----|---------|
| Segmentation defaults | `generate-config --profile web` | Sidebar defaults |
| Crystallographic supercell | `--supercell NX NY NZ` | a/b/c repeat inputs |
| Min box thickness | `--min-box-thickness` | Optional checkbox |
| Aromatic rings | `--aromatic-rings` (off) | “Show aromatic rings” (off, experimental) |
| Framework style | scatter; instanced with `--plot-backend threejs` | Plotly scatter; Three.js instanced |
| Plot engine | `--plot-backend` default `plotly` | Three.js selected by default in the UI |
| Surface / PBC / pocket-type opacity | `--surface-opacity` and siblings (1.0) | Opacity sliders (1.0) |
| Screen-door transparency | on; `--no-screen-door` | Checkbox on |
| Pore graph | on; `--no-pore-graph` | “Show pore graph” |
| Windows + edge filter | `--window-marker-*`, `--window-min-radius` | Window color/size and min-window filter |
| Exploded view | `--explode`, `--explode-plane` (Three.js HTML) | HUD slider |
| PLD HTML snapshot | `--percolation-plot` → `{stem}_percolation.html` | Download percolation HTML |
| Percolation history JSON | `--percolation-history` or `--pore-dia` | Download percolation history JSON |
| Result cache | `--cache-dir` / `--no-cache` | In-memory cache and Force recompute |
| PSD, SA, vol, pocket types | flags in the table above | Matching tabs |
| Batch | `mgrcli batch` (Dask) | HTS tab on a local install |
| Trajectory movies | not in this release | not in the batch CLI |

### Batch (parallel)

Same flags as `process`, plus:

```bash
mgrcli batch cifs/*.cif -o out/ -c config/web_analysis_template.json \
  --workers 4 --pore-dia --psd --sa --vol --psd-plot-html
mgrcli batch --input-dir ./structures -o out/ --workers 8 --tar
```

Uses Dask `LocalCluster` (local installs only; serial fallback: `mgrcli process *.cif` or `xargs`).

## Textural properties (CLI vs web app)

Both use `mgr_src` and shared defaults in `python_cli/mgrcli/defaults.py` (`sa_method: mc`, winding accessibility, auto dgrid).

| Analysis | CLI | Web app | Parity |
|----------|-----|---------|--------|
| Segmentation defaults | `generate-config --profile web` | UI defaults | **Yes** |
| Pore LCD/PLD/DFSD | `--pore-dia` (`get_pore_dia_mgr_winding`) | PLD tab | **Yes** |
| Percolation history | `--pore-dia` / auto before SA/vol | PLD tab | **Yes** |
| Percolation cluster plot | `--percolation-plot` (threejs default) | Interactive tab | **Yes** |
| SA (MC + winding) | `--sa` (default) | SA/Vol tab, MC default | **Yes** |
| SA mesh (faster) | `--sa-method mesh` | SA/Vol tab radio | **Yes** |
| Vol (voxel + winding) | `--vol` | SA/Vol tab | **Yes** |
| Lattice accessibility | `--accessibility-method lattice` | not in UI | CLI only |
| SA/vol breakdown bar HTML | `--sa-plot-html` / `--vol-plot-html` | Streamlit charts | **Yes** |
| Auto dgrid | on by default (`auto_dgrid: true`) | UI auto | **Yes** (`--no-auto-dgrid` to disable) |

For lattice or MC comparisons, use `mgr.compute_surface_area_mc_lattice` / `compute_surface_area_mesh_lattice` on saved NPZ in Python.

## AI agents

Cursor skill: [`skills/mgrcli/SKILL.md`](skills/mgrcli/SKILL.md) (symlink to `.cursor/skills/mgrcli/` for auto-discovery).

**Minimal agent recipe:**

```bash
pip install -e .   # from repo root
mgrcli process cifs/NU-1000_S1.cif -o out/ \
  -c config/web_analysis_template.json \
  --pore-dia --psd --psd-accessible --sa --vol --types
```

### Input / output

| Option | Description |
|--------|-------------|
| `-i, --input` | Input CIF file |
| `-o, --output-dir` | Output directory |
| `-c, --config` | JSON config (see `config/web_analysis_template.json`) |
| `--html-dir` | HTML output directory (default: output dir) |
| `--theme` | `dark` or `white` for plots |
| `--no-stylize` | Disable stylized framework in plots |
| `--disable-tqdm` | Disable progress bars |
| `--auto-params` | Estimate parameters from structure |

### Plot command

```bash
mgrcli plot results.npz -o out/ -c config.json --plot-backend threejs
```

| Option | Description |
|--------|-------------|
| `-o, --output-dir` | HTML output directory |
| `-c, --config` | Plot options from config |
| `--plot-backend` | `plotly` or `threejs` |
| `--cif` | CIF if missing from NPZ metadata |

### Interactive command

```bash
mgrcli interactive --clean --port 8501
```

| Option | Description |
|--------|-------------|
| `--port` | Streamlit port |
| `--no-browser` | Do not open browser |
| `--clean` | Kill stale Streamlit for this app |

### Global options

| Option | Description |
|--------|-------------|
| `--banner` | Display banner |
| `--version` | Version info |

## Output Format

Results saved as compressed NumPy files (`.npz`) containing:

- **RAG Pickle Output:**
  - If you use `--types` or `--rag`, a NetworkX region adjacency graph (RAG) is also saved as a pickle file (`*_rag_clustered.pkl`).
  - This file can be loaded with `pickle` and is compatible with NetworkX for further graph analysis or visualization.


* **`maxima_pbc`**: Periodic boundary condition maxima coordinates
* **`regions_pbc`**: Periodic boundary condition region segmentation
* **`maxima_info`**: Detailed maxima information with PBC offsets
* **`regions`**: Non-periodic region segmentation
* **`pbc_groups`**: Periodic boundary condition groups
* **`close_or_inside_maxima`**: Close or inside maxima coordinates
* **`unique_regions`**: Unique region identifiers
* **`timing_results`**: Performance timing for each step
* **`input_params`**: Parameters used for analysis

## Examples

### Basic Processing
```bash
# Single file with defaults
mgrcli process ZIF-8.cif

# Multiple files with custom output directory
mgrcli process *.cif -o ./mof_analysis/
```

### Advanced Analysis
```
# High-resolution analysis with distance grid
mgrcli process my_mof.cif \
    --spacing 0.25 \
    --mask-thickness 0.7 \
    --return-dgrid \
    --return-padded-regions

# Detect pocket types (env_cluster)
mgrcli process my_mof.cif --types
# Save RAG pickle only (no clustering)
mgrcli process my_mof.cif --rag
# The RAG will be saved as my_mof_rag_clustered.pkl in the output directory.
```


### Batch Processing with Config
```bash
# Create optimized config for high-throughput
mgrcli generate-config \
    --spacing 0.4 \
    -H 0.3 \
    --min-distance 2.0 > hts_config.json

# Process entire dataset
mgrcli process dataset/*.cif \
    -c hts_config.json \
    -o ./hts_results/
```

### Research Workflow
```bash
# 1. Explore with defaults
mgrcli process initial_structure.cif

# 2. Optimize parameters interactively
mgrcli config-building > optimized_config.json

# 3. Process with optimized settings
mgrcli process all_structures/*.cif \
    -c optimized_config.json \
    -o ./research_results/
```

**Key:**
* **Rust Backend**: High-performance numerical computations
* **Python Interface**: User interface and data serialization
* **Input/Output**: File operations and data persistence

## Reading Results in Python

```python
import numpy as np

# Load the compressed results
results = np.load('ZIF-8_results.npz', allow_pickle=True)

# Access key data
maxima_coords = results['maxima_pbc']  # Shape: (n_maxima, 3)
regions = results['regions_pbc']       # Shape: (nx, ny, nz)
timing = results['timing_results']     # Dictionary with timing info

# Basic analysis
print(f"Found {len(maxima_coords)} pore maxima")
print(f"Grid shape: {regions.shape}")
print(f"Total analysis time: {sum(timing.values()):.2f}s")

# Show unique regions
unique_regions = results['unique_regions']
print(f"Number of unique regions: {len(unique_regions)}")

# Access detailed timing information
print("\nTiming breakdown:")
for step, duration in timing.items():
    print(f"  {step}: {duration:.3f}s")

# Access input parameters
input_params = results['input_params']
print(f"\nGrid spacing used: {input_params['spacing']} Å")
```

## Testing

### Run Tests

```bash
# Run Python unit tests (from repository root)
cd python_cli && python -m pytest tests/ && cd ..
```

### Test Data

Test CIF files are available in the `cifs/` directory:
* `ZIF-8-RASPA.cif`: Single material test case
* `P1-Cu-BTC.cif`: Additional test material
* `LTA.cif`: Additional test material

## Development

### Project structure
```
mgr_cli_py/                    # repository root — run pip/maturin from here
├── pyproject.toml             # Python package metadata; maturin build backend
├── python_cli/
│   ├── mgrcli/
│   │   ├── cli.py             # CLI entry and commands
│   │   ├── interactive_app.py  # Streamlit app (mgrcli interactive)
│   │   └── ...
│   └── tests/
├── mgr_src/
│   ├── mgr/                   # Python library code used by the CLI
│   └── mgrcli_rust/           # Rust crate (PyO3 extension, built by maturin)
├── cifs/                      # Sample CIF inputs
└── requirements.txt           # Mirrors pyproject.toml (optional `pip install -r` workflows)
```

### Building a wheel for distribution

From the repository root, `pip install .` (or `pip wheel . -w dist/`) builds a release wheel that contains `mgrcli`, `mgr`, and `mgrcli_rust`. Install that wheel with pip on the same OS and CPU architecture. `maturin build` by itself does not stage `mgr_src/mgr` into the wheel; use pip so `packaging/mgrcli_pep517.py` runs.

Cross-platform wheels for the Rust extension require **per-target builds** (or CI such as cibuildwheel). This project does not publish them.


## Performance

### Why Rust?
* **Zero-cost abstractions**: High-level code that compiles to optimal machine code
* **Parallel processing**: Automatic parallelization for multi-core systems
* **Memory safety**: Compile-time guarantees prevent crashes and security issues
* **Low-level control**: Direct hardware optimization when needed

### Benchmarks
* **Distance grid computation**: ~3x faster than pure Python implementations
* **Watershed segmentation**: ~5x faster with parallel processing
* **Maxima detection**: ~10x faster with optimized algorithms
* **Memory usage**: ~50% less than equivalent Python implementations

## License

Licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use MCLI in your research, please cite:
(not a real citation, yet!)
```bibtex
@software{mgrcli2024,
  title={{MCLI: MOF Analysis CLI Tool}},
  author={Gopalan, Arun},
  year={2024},
  url={https://github.com/arung-northwestern/mgr_cli_py}
}
```

## Development Notes & Recent Fixes

### Recent Bug Fixes

#### 1. Array Bounds Error (FIXED)
**Issue**: `collapse_axis: Index 16 must be less than axis length 16 for array with shape [16, 3]`
**Root Cause**: Used `.len()` instead of `.nrows()` on 2D ndarray in Rust code
- `.len()` returns total elements (48 for [16,3] array)
- `.nrows()` returns number of rows (16 for [16,3] array)
**Fix**: Lines 5771, 5777 in `mgr_src/mgrcli_rust/src/lib.rs` - replaced `.len()` with `.nrows()`
**Lesson**: Always use `.nrows()` for 2D array bounds checking in ndarray

#### 2. Numpy Inhomogeneous Shape Error (FIXED)
**Issue**: `setting an array element with a sequence. The requested array has an inhomogeneous shape`
**Root Cause**: `connections` list contained tuples with mixed data types that numpy couldn't serialize
**Fix**: Added special handling in `save_results_numpy()` method to convert connections to numpy-friendly dictionary format
**Lesson**: Complex data structures need preprocessing before numpy serialization

### CLI Default Configuration

All CLI parameter defaults are defined in `python_cli/mgrcli/cli.py` lines 41-56:

```python
DEFAULTS = {
    'spacing': 0.5,                    # Grid spacing in Å
    'h': 0.2,                          # Min pocket size difference in Å
    'mask_thickness': 0.5,             # Mask thickness in Å
    'threshold_abs': 1.5,              # Min maxima size in Å
    'min_distance': 1.5,               # Min distance between maxima in Å
    'minimum_window_separation': 1.0,  # Min window separation in Å
    'wall_thickness': 0.08,            # Wall thickness percentage
    'extra_padding': 1.0,              # Extra padding in Å
    'min_region_size': 0.01,           # Min region size percentage
    'disable_tqdm': False,             # Progress bar flag
    'compute_connections': True,       # Connection computation flag
    'return_padded_regions': False,    # Return padded regions flag
    'return_dgrid': False,             # Return distance grid flag
}
```

**To change CLI defaults**: Simply modify the values in this `DEFAULTS` dictionary.

### Key files for development

- **`mgr_src/mgrcli_rust/src/lib.rs`**: Rust implementation compiled into **`mgrcli_rust`**
- **`python_cli/mgrcli/cli.py`**: CLI entry and commands (imports `mgrcli_rust`)
- **`python_cli/mgrcli/interactive_app.py`**: Streamlit UI (same backend)
- **`pyproject.toml`**: Python dependencies and **maturin** build backend
- **`mgr_src/mgrcli_rust/Cargo.toml`**: Crate consumed by maturin for `pip install`

### Build and development tips

```bash
# Rebuild after Rust changes (from repository root)
pip install --force-reinstall -e .
```

### Future Development Considerations

- Consider moving `DEFAULTS` to external config file for easier customization
- Add comprehensive input validation before Rust function calls
- Profile watershed segmentation performance for optimization opportunities
- Add async processing for large batch operations
- Consider error recovery for malformed CIF files

---

Happy analyzing!

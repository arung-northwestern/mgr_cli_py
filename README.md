# Mofography CLI Tool

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A high-performance command-line interface for analyzing Metal-Organic Frameworks (MOFs) using advanced segmentation algorithms. This repository ships a **terminal CLI** (`mgrcli`) and an **interactive Streamlit web app** (`mgrcli interactive`), powered by a **Rust extension** (`mgrcli_rust`, PyO3) compiled from the workspace root and Python packages under `python_cli/`.

## Table of Contents
- [Installation Guide](#-installation-guide)
  - [Prerequisites and install](#prerequisites-and-install)
  - [How the Rust backend is built](#how-the-rust-backend-is-built)
  - [Troubleshooting](#-troubleshooting)
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
* **🐳 Docker path**: For a browser-only experience without a local Python/Rust toolchain, use the Docker-based Streamlit image (see [Docker (Local)](#docker-local)); public registry images will be documented when published.

## Relationship to Mofography Python Package

This CLI tool is **separate** from the main Mofography Python package but maintains compatibility with the same core algorithms and reference functions. It's kept up-to-date with the latest computational methods while offering the advantages of a dedicated command-line interface and interactive web mode.

## Architecture

### Backend technologies
* **`mgrcli_rust` (Rust)**: Native extension module (PyO3) with the heavy numerical work: distance grids, watershed, maxima detection, window connections, and related kernels. The **CLI and Streamlit app import this module**; they do not call a separate “mgrui” or second Rust product.
* **Python (`python_cli/mgrcli/`)**: `argparse` CLI, Streamlit UI (`interactive_app.py`), plotting glue, orchestration, and saving results (for example `.npz`).
* **PyO3 + NumPy**: Rust functions receive and return NumPy arrays where appropriate.

### Which `Cargo.toml` is used
**The repository root** `Cargo.toml` (next to `pyproject.toml`), with Rust sources in **`src/`**, is what **maturin** / `pip install` compile into `mgrcli_rust`. A duplicate crate under `mgr_src/mgrcli_rust/` is not used for that install path; ignore it unless you are working on that subtree explicitly.

### Where Python vs Rust is used
* **Python**: CLI, Streamlit, configuration, file I/O, Plotly/pyvista workflows, Dask/HTS drivers.
* **Rust (`mgrcli_rust`)**: Core segmentation algorithms, numerical kernels, parallelism (Rayon), and tight array work.

> **Note**: This is completely standalone from the main mofography Python library, although the core segmentation algorithms (distance grids, watershed, maxima detection, and window connections) use the same Rust backend functions. However, all bookkeeping operations (CIF file reading, atomic radii assignment, periodic boundary condition handling, and reindexing) are implemented entirely in Rust here, which may produce slightly different results compared to the Python version.

# 🚀 Installation Guide

Install from source into a Python environment. **`pip install -e .` pulls in all Python dependencies** declared in `pyproject.toml` (including the **`mgr_src`** stack: pymatgen, freud, networkx, trimesh, scikit-learn, tqdm, rich, and the same web/data helpers as the [mofography](https://github.com/arung-northwestern/mofography) README’s pip list where this repo imports them). It also **builds `mgrcli_rust` automatically** via [maturin](https://www.maturin.rs/). You only need a working **Rust toolchain** on the machine so that compile step can run.

**GPU helpers:** A few functions in `mgr_src` (for example `dgrid_from_atoms_gpu`, Poisson spectral GPU paths) import **CuPy** when called. CuPy is **not** declared in `pyproject.toml`; install a CuPy wheel matching your CUDA stack only if you use those entry points.

For users who do not want a local Python/Rust setup, the **Docker** workflow runs the Streamlit app in a container; **public image names and pull commands will be added to this README when the image is published.**

## Prerequisites and install

### Prerequisites

* **Python 3.10+** recommended (3.8+ minimum): [Download Python](https://www.python.org/downloads/)
* **Rust & Cargo** (1.70+): required so `mgrcli_rust` can compile (PyO3 / maturin)
* **Git**: to clone the repository

### Local install (quick path)

From a fresh clone — one editable install builds the Rust extension and Python packages together:

```bash
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e .                   # compiles mgrcli_rust + installs mgrcli + mgr_src
```

**After changing Rust code** (`src/`, root `Cargo.toml`), reinstall the extension:

```bash
pip install --force-reinstall -e .
```

**Verify:**

```bash
mgrcli --help
mgrcli process cifs/P1-IRMOF-1.cif -o ./out/ --return-dgrid
mgrcli interactive --clean         # kill stale Streamlit, then open web UI
```

> **Note:** `pip install -e .` is the normal path — maturin runs `cargo` for you. You do **not** need a separate manual `cargo build` unless you are debugging the Rust crate in isolation.

**Interactive app:** `mgrcli interactive` runs Streamlit on `python_cli/mgrcli/interactive_app.py`, sets `PYTHONPATH` to the repo root, and forces local mode (`MGR_FORCE_LOCAL=1`). Use **`--clean`** if a previous session left Streamlit running on the same app file.

See also: [CLI ↔ web parity plan](docs/plan_cli_web_parity.md) for aligning batch CLI defaults with the Streamlit app.

### Step 1: Install Rust (one-time)

**Windows:**
```bash
# Option 1: winget (recommended)
winget install Rustlang.Rustup

# Option 2: Manual download from https://rustup.rs/
```

**Linux:**
```bash
# Ubuntu/Debian
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Or package manager
sudo apt install cargo rustc  # Ubuntu
sudo dnf install cargo rustc  # Fedora
```

**macOS:**
```bash
# Homebrew (recommended)
brew install rust

# Or manual
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Verify Rust installation:**
```bash
rustc --version  # Should show version 1.70 or higher
cargo --version  # Should show cargo version
```

### Step 2: Clone the repository

```bash
git clone https://github.com/arung-northwestern/mgr_cli_py.git
cd mgr_cli_py
```

Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Step 3: Install the package (Rust builds here)

From the **repository root** (the directory that contains `Cargo.toml`, `pyproject.toml`, and `python_cli/`):

```bash
pip install --upgrade pip
pip install -e .
```

Equivalent with **uv**:

```bash
uv pip install -e .
```

**Non-editable install from a wheel** (for example on a machine where you copy artifacts instead of the full repo):

```bash
pip install maturin
maturin build --release
pip install target/wheels/mgrcli-*.whl
```

Adjust the glob if your shell does not expand it (install the single `.whl` file that maturin prints).

### Step 4: Verify

```bash
mgrcli --help
mgrcli process cifs/ZIF-8-RASPA.cif -o ./out/
mgrcli interactive
```

**What `pip install -e .` does:** resolves Python dependencies from `pyproject.toml`, runs **maturin**, which compiles the **`mgrcli_rust`** crate from the **root** `Cargo.toml` / `src/`, and wires the **`mgrcli`** console script to `python_cli/mgrcli/cli.py`. The CLI and Streamlit app call into **`import mgrcli_rust`** for the hot paths.

**Interactive app:** `mgrcli interactive` runs Streamlit on `python_cli/mgrcli/interactive_app.py` and sets `PYTHONPATH` to the repo root so `mgr_src` resolves.

### Expected first-time build

- Roughly 5–15 minutes for a release-quality Rust compile, depending on CPU.
- **`mgrcli`** is available inside the active environment.

## How the Rust backend is built

| Approach | When to use it | What compiles Rust? |
|----------|----------------|---------------------|
| **`pip install -e .`** | Normal development | **Automatic** — PEP 517 invokes **maturin**, which runs `cargo` and installs the extension plus Python sources. |
| **`maturin develop --release`** | Same as editable, but you invoke maturin yourself | maturin |
| **`maturin build --release`** then **`pip install target/wheels/*.whl`** | CI, air-gapped installs, or shipping a wheel without the source tree | maturin (you install the built wheel afterward) |
| **`cargo build --release` alone** | Sanity-check the Rust toolchain | **Does not** install `mgrcli_rust` into Python; use one of the rows above for a usable Python module. |

You do **not** need a separate manual “compile Rust, then pip” sequence for day-to-day work: **`pip install -e .` already includes the Rust build** as long as `rustc` / `cargo` are on your `PATH`.

---

## 🔧 **Troubleshooting**

### Common Issues

**"mgrcli command not found"**
```bash
# Check if installation succeeded
pip list | grep mgrcli

# Try using Python module directly
python -m mgrcli.cli --help
```

**"Rust compilation failed"**
```bash
# Update Rust toolchain
rustup update

# From the repository root: clear build cache and retry
pip uninstall mgrcli
rm -rf build/ target/
pip install -e .
```

**"No module named 'mgrcli_rust'"**
```bash
# The Rust extension didn't build properly
# Check Rust installation and retry
rustc --version
cargo --version
```

### Getting Help

```bash
# Get detailed help
mgrcli --help

# See all available commands
mgrcli process --help
mgrcli generate-config --help

# Check version and installation
mgrcli --version
```

## Docker (Local)

**CLI** = use the terminal with a pip/uv install (`mgrcli`). **GUI only** (no terminal) = run the Docker image via Docker Desktop; the image serves the Streamlit app. This section covers the Docker (GUI) path.

### Build + Run Streamlit
```bash
./scripts/docker_build_run.sh
```
You will be prompted to choose an output folder. The app runs at `http://localhost:8501`.

### Docker Desktop (GUI-only) Run Steps
1. Open Docker Desktop → Images
2. Find `mgrcli:local` → Click Run
3. In Optional settings:
   - Ports: set Host `8501` → Container `8501`
   - Volumes: mount a host folder to `/results`
4. Click Run, then open `http://localhost:8501`

For non-terminal users, see `docs/GUI_ONLY_QUICKSTART.md`.

### Prebuilt image (when published)

Public pull/run instructions for a registry image (Docker Hub or similar) will be added here once the image is published. Until then, build locally with `./scripts/docker_build_run.sh` or `docker build` as above.

### CLI (terminal users)

From the **repository root**, an editable install compiles `mgrcli_rust` and pulls in all runtime dependencies declared in `pyproject.toml`:

```bash
# Optional: uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .

# Or plain pip
pip install -e .
```

### Environment Overrides
```bash
# Image name
IMAGE_NAME=mgrcli:local ./scripts/mgrbox --help

# Output folder
MGR_RESULTS_DIR="$HOME/MGR_Results" ./scripts/mgrbox process *.cif -o /results

# Optional: override Rayon thread count
MGR_RAYON_THREADS=4 ./scripts/mgrbox process *.cif -o /results
```

### Cross-Platform Build (Windows, Linux amd64, ARM Mac)

An image built with `docker build -t mgrcli:local .` on an ARM Mac is **linux/arm64** and will not run on Windows or Intel Linux (they need **linux/amd64**). The Rust backend is compiled inside the image for the target platform, so you must use a **multi-platform** build.

**Use the buildx script** (not plain `docker build`) so the image runs on all targets:

```bash
./scripts/docker_buildx.sh
```

- By default this builds for `linux/amd64,linux/arm64`. Only the **first** platform is loaded into Docker (amd64), so the same image runs on Windows and Intel Linux. On ARM Mac it will run via emulation; for native ARM Mac use a single-platform build (below).
- **Build for one platform only** (faster, no registry):
  - Windows / Intel Linux: `PLATFORMS=linux/amd64 ./scripts/docker_buildx.sh`
  - ARM Mac: `PLATFORMS=linux/arm64 ./scripts/docker_buildx.sh`
- **Export to tar files** (no Docker Hub): build both platforms and save as tarballs for sharing (e.g. Google Drive). Filenames are friendly: **mgrcli-windows_linux_64.tar** (Windows + Intel/AMD Linux) and **mgrcli-mac_arm64.tar** (ARM Mac).
  ```bash
  EXPORT_TAR=1 ./scripts/docker_buildx.sh
  # Produces mgrcli-windows_linux_64.tar and mgrcli-mac_arm64.tar
  # Colabs: see docs/DOCKER_DESKTOP_SHARED_IMAGE.md for load + Docker Desktop settings (ports, volume, Dask).
  ```

### GUI Launcher Behavior (Spec)
- Check Docker Desktop is installed/running
- Prompt user to select output folder (default per OS)
- Pull latest image on launch (auto-update)
- Start container with bind mount to output folder
- Open browser to `http://localhost:8501`

### Hugging Face Spaces (demo deployment)

**Private testing** (default for agent deploys): `arung-northwestern/mofography-web-demo-private-testing`

**Production** (public demo): `arung-northwestern/mofography-web-demo` — deploy only after explicit approval.

From repo root, bundle and push to private testing:

```bash
./scripts/push_hf_space.sh --confirm-push -m "Deploy: MC SA default + textural parity"
```

This runs `deploy_to_hf_space.sh`, syncs into a local Space clone, commits, and pushes to `main` (triggers rebuild). Without `--confirm-push`, only the bundle is prepared.

Manual alternative:

```bash
./scripts/deploy_to_hf_space.sh
# push build_space/ to your Space — see spaces/README.md
```

HTS tab shows a message only on Spaces; full HTS is for local Docker/pip.

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
| `probe-radius` | | 1.2 | He probe for SA/vol/accessible PSD (Å) |
| `window_fp` | `--no-window-fp` | on | Window fingerprint rescue (SA/vol winding) |

Legacy profile differences: `mask_thickness=0.5`, `rust_bucket`, `crc` radii. See [parity plan](docs/plan_cli_web_parity.md).

### Analysis flags

| Flag | Description |
|------|-------------|
| `--pore-dia` | LCD, PLD, DFSD + winding percolation history (CSV) |
| `--percolation-plot` | Percolation cluster visualization HTML (default backend: threejs) |
| `--percolation-plot-backend` | `threejs` or `plotly` |
| `--psd` | Global + pocketwise total PSD (CSV) |
| `--psd-accessible` | Accessible PSD at `--probe-radius` (CSV) |
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

For lattice or MC comparisons, use `mgr.compute_surface_area_mc_lattice` / `compute_surface_area_mesh_lattice` on saved NPZ in Python — see `docs/accessibility_and_surface_area.org`.

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
├── Cargo.toml                 # Rust crate mgrcli_rust (PyO3 extension)
├── src/lib.rs                 # Rust implementation compiled into mgrcli_rust
├── pyproject.toml             # Python package metadata; maturin build backend
├── setup.py                   # Legacy setuptools-rust layout (optional)
├── python_cli/
│   ├── mgrcli/
│   │   ├── cli.py             # CLI entry and commands
│   │   ├── interactive_app.py  # Streamlit app (mgrcli interactive)
│   │   └── ...
│   └── tests/
├── mgr_src/                   # Python reference / library code used by the CLI
├── cifs/                      # Sample CIF inputs
└── requirements.txt           # Mirrors pyproject.toml (optional `pip install -r` workflows)
```
A duplicate Rust tree under `mgr_src/mgrcli_rust/` is not used for **`pip install`**; the authoritative crate is **root `Cargo.toml` + `src/`**.

### Building a wheel for distribution

```bash
pip install maturin
maturin build --release
# Wheel under target/wheels/ — install with pip on compatible platforms
pip install target/wheels/mgrcli-*.whl
```

Cross-platform wheels for the Rust extension typically require **per-target builds** (or CI such as cibuildwheel); maturin documents supported targets.

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

Licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
  author={MOF Analysis Team},
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
**Fix**: Lines 5771, 5777 in `src/lib.rs` - replaced `.len()` with `.nrows()`
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

- **`src/lib.rs`**: Rust implementation compiled into **`mgrcli_rust`**
- **`python_cli/mgrcli/cli.py`**: CLI entry and commands (imports `mgrcli_rust`)
- **`python_cli/mgrcli/interactive_app.py`**: Streamlit UI (same backend)
- **`pyproject.toml`**: Python dependencies and **maturin** build backend
- **Root `Cargo.toml`**: Crate consumed by maturin for `pip install`
- **`setup.py`**: Legacy setuptools-rust path (optional; normal installs use maturin)

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

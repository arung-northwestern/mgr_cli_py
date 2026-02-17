# Mofography CLI Tool

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A high-performance command-line interface for analyzing Metal-Organic Frameworks (MOFs) using advanced segmentation algorithms. This is a separate, comparable version of the Mofography library that offers both CLI and interactive Streamlit modes, built with Rust for performance and Python for usability.

## Table of Contents
- [Installation Guide](#-installation-guide)
  - [Choose Your Installation Path](#choose-your-installation-path)
  - [Path 1: Developer Installation](#-path-1-developer-installation)
  - [Path 2: Standalone Executable](#-path-2-standalone-executable)
  - [Troubleshooting](#-troubleshooting)
- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Examples](#examples)
- [Workflow Diagram](#workflow-diagram)
- [Reading Results in Python](#reading-results-in-python)
- [Standalone Builds](#standalone-builds)
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
* **📦 Zero Dependencies**: Standalone executables available without Python installation

## Relationship to Mofography Python Package

This CLI tool is **separate** from the main Mofography Python package but maintains compatibility with the same core algorithms and reference functions. It's kept up-to-date with the latest computational methods while offering the advantages of a dedicated command-line interface and interactive web mode.

## Architecture

### Backend Technologies
* **Rust Core**: High-performance segmentation algorithms (distance grids, watershed, maxima detection, window connections)
* **Python Interface**: User-friendly CLI built with argparse
* **PyO3 Bridge**: Seamless Rust-Python interoperability
* **Numpy Arrays**: Efficient data structures for scientific computing

### Where Python vs Rust is Used
* **Python**: CLI interface, configuration management, file I/O, result serialization
* **Rust**: Core algorithms, numerical computations, memory management, parallel processing

> **Note**: This is completely standalone from the main mofography Python library, although the core segmentation algorithms (distance grids, watershed, maxima detection, and window connections) use the same Rust backend functions. However, all bookkeeping operations (CIF file reading, atomic radii assignment, periodic boundary condition handling, and reindexing) are implemented entirely in Rust here, which may produce slightly different results compared to the Python version.

# 🚀 Installation Guide

## Choose Your Installation Path

### 📋 **Quick Decision Guide**

| Your Situation | Recommended Path | Requirements | Installation Time |
|----------------|------------------|--------------|-------------------|
| **Have Python + Rust** | [Path 1: Source Installation](#-path-1-developer-installation) | Python 3.8+, Rust, Git | ~5-10 minutes |
| **Want standalone .exe** | [Path 2: Standalone Executable](#-path-2-standalone-executable) | None! | ~2 minutes |
| **New to programming** | [Path 2: Standalone Executable](#-path-2-standalone-executable) | None! | ~2 minutes |

---

## 🛠️ **Path 1: Developer Installation** (Recommended if you have Python & Rust)

Perfect for developers, researchers, or anyone comfortable with Python development.

### Prerequisites

* **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
* **Rust & Cargo**: Required for building the high-performance core
* **Git**: For cloning the repository

### Step 1: Install Rust (One-time setup)

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

### Step 2: Install MCLI

```bash
# Clone the repository (replace with actual repository URL)
git clone https://github.com/your-username/mgr_standalone.git
cd mgr_standalone/mgrcli_rust

# Install Python dependencies (numpy optional)
pip install -r requirements.txt

# Build and install (compiles Rust core + Python bindings)
pip install -e .
```

> **Note**: Replace `https://github.com/your-username/mgr_standalone.git` with the actual repository URL from GitHub, GitLab, or your git hosting service.

### Step 3: Verify Installation

```bash
# Test the installation
mgrcli --help

# Process your first CIF file
mgrcli process your_file.cif
```

**What happens during installation:**
1. **Rust Compilation**: High-performance segmentation algorithms are compiled
2. **Python Bindings**: PyO3 creates seamless Python↔Rust interface
3. **CLI Setup**: `mgrcli` command becomes available globally

### 🎯 **Expected Results**
- Installation time: ~5-10 minutes
- CLI command: `mgrcli` (available globally)
- Full functionality: All features available
- Dependencies: Python + compiled Rust extension

---

## ⚡ **Path 2: Standalone Executable** (No Python/Rust Required!)

Perfect for end users, deployment, or when you just want to analyze CIF files.

### Step 1: Get the Executable

**Option A: Download Pre-built (Coming Soon)**
```bash
# Future: Direct download links for Windows/Linux/macOS
# No installation required - just download and run!
```

**Option B: Build from Source (if you have Python + Rust)**
```bash
# If you have Python + Rust, you can build standalone executables:

# Install PyInstaller
pip install pyinstaller

# Clone and enter the repository (replace with actual repository URL)
git clone https://github.com/your-username/mgr_standalone.git
cd mgr_standalone/mgrcli_rust

# Build standalone executable (28MB, numpy-free)
pyinstaller --onefile --name mgrcli \
    --hidden-import=mgrcli_rust \
    python_cli/mgrcli/cli_standalone.py

# Find your executable in dist/mgrcli_standalone
ls -lh dist/mgrcli_standalone  # Should show ~28MB
```

> **Note**: Replace `https://github.com/your-username/mgr_standalone.git` with the actual repository URL.

### Step 2: Run the Executable

```bash
# Make executable (Linux/macOS)
chmod +x dist/mgrcli_standalone

# Test it works
./dist/mgrcli_standalone --help

# Process your CIF files
./dist/mgrcli_standalone process your_structure.cif
./dist/mgrcli_standalone process *.cif -o ./results/
```

### 🎯 **Expected Results**
- Installation time: ~2 minutes (or 0 if downloading)
- Executable size: ~28MB (numpy-free) or ~47MB (with numpy)
- Dependencies: **None required!**
- Platform support: Windows, Linux, macOS
- Single file: Everything bundled into one executable

### 📊 **Standalone vs Developer Installation**

| Feature | Standalone Executable | Developer Installation |
|---------|----------------------|------------------------|
| **Python/Rust Required** | ❌ No | ✅ Yes |
| **Installation Time** | ~2 min | ~10 min |
| **File Size** | 28MB (numpy-free) | Full source + dependencies |
| **Dependencies** | None | Python + Rust toolchain |
| **Update Method** | Download new executable | `git pull` + rebuild |
| **Customization** | Limited | Full source access |
| **Performance** | Same | Same |

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

# Clear build cache and retry
pip uninstall mgrcli
rm -rf build/
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

### Prebuilt Image (When Published)
```bash
docker pull <your-org>/mgrcli:latest
docker run --rm -p 8501:8501 <your-org>/mgrcli:latest
```
Open `http://localhost:8501`.

### CLI (Terminal Users)
If you prefer the CLI, install it from the repo using `uv` (builds the Rust backend and installs the Python CLI):
```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root
uv pip install -e .
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
To deploy the app as a public demo on [Hugging Face Spaces](https://huggingface.co/spaces) (HTS tab shows a message only; full HTS is for local Docker/pip):

```bash
./scripts/deploy_to_hf_space.sh
```

Then push the generated **`build_space/`** directory to a new Space (Docker). See **`spaces/README.md`** for step-by-step instructions.

## Quick Start

Once installed, MCLI is ready to analyze your CIF files:

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
- **Compatible format**: Can be loaded with `numpy.load()` or the standalone reader

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

### Generate Default Config
```bash
mgrcli generate-config -o config.json
```

### Customize Parameters
```bash
mgrcli generate-config \
    --spacing 0.3 \
    -H 0.5 \
    --mask-thickness 0.8 \
    --threshold-abs 2.0 \
    -o custom_config.json
```

### Interactive Builder
```bash
mgrcli config-building
# Follow prompts to create custom configuration
```

### Analysis Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `spacing` | `-s` | 0.5 | Grid spacing in Ångstroms |
| `h` | `-H` | 0.2 | Minimum difference between pocket sizes in Å |
| `mask-thickness` | `-mt` | 0.5 | Void space mask thickness in Å |
| `threshold-abs` | `-ta` | 1.5 | Minimum maxima size in Å |
| `min-distance` | `-md` | 1.5 | Minimum distance between maxima in Å |
| `minimum-window-separation` | `-ms` | 1.0 | Minimum separation between windows in Å |
| `wall-thickness` | `-wt` | 0.08 | Wall thickness percentage (0-1) |
| `extra-padding` | `-ep` | 1.0 | Additional grid padding in Å |
| `min-region-size` | `-mrs` | 0.01 | Minimum region size percentage |
| `js-threshold` | | 0.15 | Jensen-Shannon threshold for pocket type clustering |

### Analysis Flags

| Flag | Description |
|------|-------------|
| `--no-connections` | Disable connection computation |
| `--return-padded-regions` | Include padded regions in output |
| `--return-dgrid` | Include distance grid in output |
| `--disable-tqdm` | Disable progress bars |
| `--auto-params` | Automatically estimate optimal parameters from structure |
| `--pore-dia` | Compute pore diameter and percolation analysis |
| `--psd` | Compute pore size distribution analysis |
| `--types` | Detect pocket types using agglomerative clustering on cavity size distributions |

### Visualization Options

| Flag | Description |
|------|-------------|
| `--plot-html` | Generate Plotly HTML visualization after processing |
| `--html-dir` | Directory for HTML visualizations (defaults to output directory) |
| `--no-stylize` | Disable stylized framework rendering in plots |
| `--theme` | Theme for generated visualizations (dark/white, default: dark) |

### Input/Output Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input CIF file (alternative to positional arguments) |
| `-o, --output-dir` | Output directory for results (default: current directory) |
| `-c, --config` | Configuration file (JSON format) |

### Plot Command Options

```bash
mgrcli plot results.npz [options]
```

| Option | Description |
|--------|-------------|
| `-o, --output-dir` | Directory to store generated HTML files |
| `-c, --config` | Configuration file to control plotting options |
| `--no-stylize` | Disable stylized framework rendering in plots |
| `--cif` | Optional CIF file to use when metadata is missing |

### Interactive Command Options

```bash
mgrcli interactive [options]
```

| Option | Description |
|--------|-------------|
| `--port` | Port to serve the Streamlit app (default chosen by Streamlit) |
| `--no-browser` | Do not automatically open a browser window |

### Global Options

```bash
mgrcli --banner
```

| Option | Description |
|--------|-------------|
| `--banner` | Display the Mofography CLI banner with developer information |
| `--version` | Show version information |

**Note**: To learn more about the developer, check out our cool banner with `mgrcli --banner`!

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

# Detect pocket types and save RAG pickle
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
# Run the simple test suite
python test_cli.py

# Run Python unit tests
cd python_cli
python -m pytest tests/
```

### Test Data

Test CIF files are available in the `cifs/` directory:
* `ZIF-8-RASPA.cif`: Single material test case
* `P1-Cu-BTC.cif`: Additional test material
* `LTA.cif`: Additional test material

## Development

### Project Structure
```
mgrcli_rust/
├── src/lib.rs              # Rust core algorithms
├── python_cli/
│   ├── mgrcli/
│   │   ├── __init__.py     # Package initialization
│   │   └── cli.py          # CLI implementation
│   └── tests/              # Python unit tests
├── cifs/                   # Test CIF files
├── Cargo.toml             # Rust dependencies
├── setup.py               # Python package setup
└── README.md              # This file
```

### Building for Distribution

```bash
# Build wheels for multiple platforms
pip install build
python setup.py bdist_wheel

# Cross-platform builds (advanced)
pip install cibuildwheel
cibuildwheel --platform linux --arch x86_64
```

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
  url={https://github.com/your-repo/mgrcli}
}
```

## Roadmap

## Standalone Builds

For users who want to create single-executable distributions or minimize dependencies, MCLI supports a **numpy-free mode** that eliminates the numpy dependency while maintaining full functionality.

### Why Remove NumPy?

- **Size Reduction**: NumPy adds ~18MB (38% reduction) to executable size
- **Deployment Simplicity**: Fewer dependencies to manage
- **Cross-Platform Consistency**: More reliable builds across different systems

### Standalone Options

#### Option 1: Automatic Fallback (Recommended)
The main CLI automatically detects if NumPy is available and falls back to the standalone implementation:

```python
# In python_cli/mgrcli/cli.py
if not HAS_NUMPY:
    print("Warning: numpy not available, using standalone implementation")
```

#### Option 2: Force Standalone Mode
Set the flag in the main CLI for guaranteed numpy-free operation:

```python
# In python_cli/mgrcli/cli.py
USE_STANDALONE = True  # Forces numpy-free implementation
```

#### Option 3: Dedicated Standalone Version
Use the completely separate standalone implementation:

```bash
# Use the standalone version directly
python python_cli/mgrcli/cli_standalone.py --help

# Or create an alias
alias mgrcli='python python_cli/mgrcli/cli_standalone.py'
```

### Building Standalone Executables

#### With PyInstaller (Recommended & Tested)

```bash
# Install PyInstaller
pip install pyinstaller

# Build numpy-free executable (28MB)
pyinstaller --onefile --name mgrcli \
    --hidden-import=mgrcli_rust \
    python_cli/mgrcli/cli_standalone.py

# Build numpy-enabled executable (47MB)
pyinstaller --onefile --name mgrcli \
    --hidden-import=mgrcli_rust \
    python_cli/mgrcli/cli_numpy.py

# Test both executables
./dist/mgrcli_standalone --help     # 28MB - numpy-free
./dist/mgrcli_numpy --help          # 47MB - full numpy support
```

### Performance Comparison (Tested Results)

| Implementation | Dependencies | Executable Size | .npz Compatibility | Performance | Linux Tested |
|----------------|--------------|-----------------|-------------------|-------------|--------------|
| **NumPy-based** | numpy, rust | **47MB** | ✅ Full | Fastest | ✅ Works |
| **Standalone** | rust only | **28MB** | ✅ Compatible* | Near identical | ✅ Works |
| **Improvement** | **-1 dependency** | **-38%** | ✅ Same | ✅ Same | ✅ Same |

*Standalone implementation creates .npz files that are fully compatible with NumPy's `np.load()` function.

### PyInstaller Compatibility

**✅ Linux Support**: Both versions tested and working on Linux
**✅ Cross-Platform**: PyInstaller executables work on Windows, Linux, and macOS
**✅ Single File**: `--onefile` flag creates self-contained executables
**✅ Dependencies**: Automatically bundles all required libraries

### NumPy Dependency Status

**Current Status**: NumPy is **optional** in `requirements.txt`:
```txt
# numpy>=1.20.0  # Optional - only needed for numpy-based .npz saving
setuptools-rust
```

**Impact**: Minimal usage - only for `.npz` file saving, since Rust already returns numpy arrays for numerical data.

### Analysis & Visualization
* [ ] **Plotting Capabilities**: Built-in visualization for pore networks, segmentation results, and property distributions
* [ ] **Textural Properties**: Calculate pore diameters, pore size distributions, surface areas, volumes, and percolation analysis
* [ ] **NetworkX Integration**: Export pore graphs in NetworkX-compatible format for machine learning applications
* [ ] **Advanced Metrics**: Compute BET surface area, pore volume distributions, and structural descriptors

### Deployment & Distribution
* [ ] **Standalone Executables**: Create single-binary executables for Windows, Linux, and macOS (no Rust/Python installation required)
* [ ] **Web Application**: Build a web-based interface using this Rust backend for non-programmers
* [ ] **Docker Images**: Containerized deployment for easy setup and reproducibility

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

### Key Files for Future Development

- **`src/lib.rs`**: Main Rust backend - all core algorithms and data processing
- **`python_cli/mgrcli/cli.py`**: Python CLI interface - argument parsing and result handling
- **`setup.py`**: Build configuration - setuptools-rust for Rust/Python interop
- **`Cargo.toml`**: Rust dependencies and build settings
- **`python_cli/mgrcli/__init__.py`**: Package initialization

### Build & Development Tips

```bash
# Rebuild after Rust changes
pip install --force-reinstall -e .

# Debug array bounds issues
# Use .nrows() not .len() for 2D ndarray bounds

# Debug numpy serialization
# Check for mixed data types in complex structures
```

### Future Development Considerations

- Consider moving `DEFAULTS` to external config file for easier customization
- Add comprehensive input validation before Rust function calls
- Profile watershed segmentation performance for optimization opportunities
- Add async processing for large batch operations
- Consider error recovery for malformed CIF files

---

Happy analyzing!

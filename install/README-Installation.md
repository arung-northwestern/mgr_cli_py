# MGRCLI Installation Guide

This directory contains installation scripts for MGRCLI that create a virtual environment and install the CLI (and optional local GUI launcher). **For CLI:** use terminal after pip/uv install. **For GUI only (no terminal):** use the Docker image and Docker Desktop instead (see main README Docker section).

## Quick Start

### For Linux/macOS Users:
```bash
# Download and extract the repository
cd mgr_cli_py
./install/install.sh
```

### For Windows Users:
```batch
REM Download and extract the repository
cd mgr_cli_py
install\install.bat
```

## What the Installation Scripts Do

Both scripts perform the following steps:

1. **Install UV Package Manager** - A fast Python package manager (if not already installed)
2. **Create Virtual Environment** - Isolated environment for MCLI dependencies
3. **Build with Optimizations** - Compiles the Rust backend with maximum performance optimizations
4. **Create Launch Scripts** - Simple command-line launchers for both CLI and GUI
5. **Create Desktop Shortcuts** - Easy access from your desktop (Windows & Linux)

## After Installation

### Using MGRCLI

**Command Line Interface:**
```bash
# Linux/macOS
./mgrcli --help
./mgrcli process cifs/ZIF-8-RASPA.cif

# Windows
mgrcli.bat --help
mgrcli.bat process cifs\ZIF-8-RASPA.cif
```

**Web Interface:**
```bash
# Linux/macOS
./mgrcli-gui

# Windows
mgrcli-gui.bat
```

### Desktop Shortcuts

- **Windows**: Desktop shortcuts created automatically as "MGRCLI CLI" and "MGRCLI GUI"
- **Linux**: Application menu entries created (if supported)
- **macOS**: You can manually create aliases or use the launch scripts

## Manual Installation (Advanced)

If you prefer to install manually or the scripts don't work:

```bash
# Install UV
pip install uv

# Create environment
uv venv mcli_env
source mcli_env/bin/activate  # Linux/macOS
# or
mcli_env\Scripts\activate.bat  # Windows

# Install with optimizations
export SETUPTOOLS_RUST_RELEASE=1
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=true -C codegen-units=1"
uv pip install -e . --force-reinstall --no-cache-dir
```

## Troubleshooting

### "UV command not found"
```bash
pip install uv
```

### "Rust compilation failed"
Make sure you have Rust installed:
```bash
# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Windows
winget install Rustlang.Rustup
```

### "Permission denied" (Linux/macOS)
```bash
chmod +x install/install.sh
./install/install.sh
```

### "PowerShell execution policy" (Windows)
Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Performance Notes

The installation scripts build MCLI with release optimizations that provide:
- **~14x faster performance** than debug builds
- **Native CPU optimization** for your specific processor
- **Link-time optimization** for maximum speed

## System Requirements

- **Python 3.8+** (required for UV)
- **Rust 1.70+** (for compiling the backend)
- **2GB+ RAM** (for compilation)
- **500MB+ disk space** (for installation)

## Support

- **Documentation**: https://github.com/arung-northwestern/mgr_cli_py
- **Issues**: https://github.com/arung-northwestern/mgr_cli_py/issues
- **Sample Files**: Check the `cifs/` directory for test structures

## Future: UVX Support

Once MCLI is published to PyPI, you'll be able to use it even more easily:

```bash
# No installation required!
uvx run mgr-cli-py process structure.cif
uvx run mgr-cli-py interactive
```

This will provide true zero-installation usage for non-technical users.
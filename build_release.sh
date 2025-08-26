#!/bin/bash
# MGR CLI PY - Release Build Script
# Builds with maximum optimizations for production performance
# This addresses the critical performance issue where debug builds were 14x slower

set -e

echo "🔧 Building MGR CLI PY with RELEASE optimizations..."
echo "⚠️  This is REQUIRED for production performance!"
echo "   (Debug builds are ~14x slower due to lack of compiler optimizations)"
echo ""

# Set environment variables for release build with maximum optimizations
export SETUPTOOLS_RUST_RELEASE=1
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=true -C codegen-units=1"

echo "Environment variables set:"
echo "  SETUPTOOLS_RUST_RELEASE=1"
echo "  RUSTFLAGS=$RUSTFLAGS"
echo ""

# Clean previous build
echo "🧹 Cleaning previous build..."
cargo clean

# Build and install with release optimizations
echo "🏗️  Building with release optimizations..."
pip install -e . --force-reinstall --no-cache-dir

echo ""
echo "✅ Build completed successfully!"
echo ""
echo "📊 Performance expectations:"
echo "   - Total time: ~0.5s (vs 7.08s debug)"
echo "   - Dgrid computation: ~0.12s (vs 2.30s debug)"
echo "   - Overall speedup: ~14x faster"
echo ""
echo "🧪 Test with:"
echo "   python -m mcli.cli process cifs/LTA96.cif --disable-tqdm"
echo ""
echo "📚 See MCP_MEMORY_LAYER_MGR_CLI_PY.md for full documentation"

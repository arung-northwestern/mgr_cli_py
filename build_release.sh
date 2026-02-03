#!/bin/bash
# MGR CLI PY - Release Build Script
# Builds with maximum optimizations for production performance using maturin
# This addresses the critical performance issue where debug builds were 14x slower

set -e

echo "🔧 Building MGR CLI PY with RELEASE optimizations..."
echo "⚠️  This is REQUIRED for production performance!"
echo "   (Debug builds are ~14x slower due to lack of compiler optimizations)"
echo ""

# Set environment variables for release build with maximum optimizations
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"

echo "Environment variables set:"
echo "  RUSTFLAGS=$RUSTFLAGS"
echo ""

# Clean previous build
echo "🧹 Cleaning previous build..."
cargo clean

# Build and install with release optimizations using maturin
echo "🏗️  Building and installing with release optimizations..."
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
echo "   mgrcli process cifs/LTA96.cif --disable-tqdm"
echo ""
echo "📦 To build for distribution:"
echo "   maturin build --release --strip"
echo ""

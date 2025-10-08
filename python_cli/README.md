# MCLI - MOF Analysis CLI Tool

A command-line interface for analyzing Metal-Organic Frameworks (MOFs) using advanced segmentation algorithms.

## Features

- **Process CIF files**: Analyze MOF structures from CIF files
- **Batch processing**: Process multiple CIF files in parallel
- **Configurable parameters**: Fine-tune analysis parameters for different use cases
- **Interactive configuration**: Build configurations interactively
- **Cross-platform**: Works on Windows, Linux, and macOS
- **Compressed output**: Results saved as efficient NPZ files

## Installation

### From Source

```bash
# Build the Rust library first
cd ..
pip install -e .
```

## Usage

### Basic Commands

```bash
# Process a single CIF file
mcli process structure.cif

# Process multiple CIF files
mcli process *.cif

# Specify output directory
mcli process structure.cif -o ./results/

# Use custom configuration
mcli process structure.cif -c config.json
```

### Configuration

```bash
# Generate default configuration
mcli generate-config > config.json

# Generate configuration with custom parameters
mcli generate-config -mt 1.0 -ta 2.0 > config.json

# Interactive configuration builder
mcli config-building
```

### Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| spacing | -s | 0.5 | Grid spacing in Å |
| h | --h | 0.2 | Minimum difference between pocket sizes in Å |
| mask-thickness | -mt | 0.5 | Mask thickness in Å |
| threshold-abs | -ta | 1.5 | Minimum maxima size in Å |
| min-distance | -md | 1.5 | Minimum distance between maxima in Å |
| minimum-window-separation | -ms | 1.0 | Minimum separation between windows in Å |
| wall-thickness | -wt | 0.08 | Wall thickness percentage |
| extra-padding | -ep | 1.0 | Extra padding in Å |
| min-region-size | -mrs | 0.01 | Minimum region size percentage |

### Flags

- `--no-connections`: Disable connection computation
- `--return-padded-regions`: Return padded regions in output
- `--return-dgrid`: Return distance grid in output
- `--disable-tqdm`: Disable progress bars

## Output

Results are saved as compressed NumPy files (`.npz`) containing:

- **RAG Pickle Output:**
  - If you use `--types` or `--rag`, a NetworkX region adjacency graph (RAG) is also saved as a pickle file (`*_rag_clustered.pkl`).
  - This file can be loaded with `pickle` and is compatible with NetworkX for further graph analysis or visualization.


- `maxima_pbc`: Periodic boundary condition maxima
- `regions_pbc`: Periodic boundary condition regions
- `maxima_info`: Detailed maxima information
- `regions`: Non-periodic regions
- `pbc_groups`: Periodic boundary condition groups
- `close_or_inside_maxima`: Close or inside maxima coordinates
- `close_or_inside_regions`: Close or inside region IDs
- `unique_regions`: Unique region IDs
- `maxima_pbc_dict`: Maxima dictionary
- `all_maxima`: All maxima coordinates
- `maxima_relative_locations`: Relative location classifications
- `input_params`: Input parameters used
- `timing_results`: Performance timing information
- `connections`: Region connections (if enabled)
- `regions_padded`: Padded regions (if enabled)
- `dgrid`: Distance grid (if enabled)

## Examples

### Process with custom parameters

```bash
mcli process ZIF-8.cif \
    --spacing 0.4 \
    --mask-thickness 0.8 \
    --threshold-abs 2.0 \
    --return-dgrid

# Detect pocket types and save RAG pickle
mcli process ZIF-8.cif --types
# Save RAG pickle only (no clustering)
mcli process ZIF-8.cif --rag
# The RAG will be saved as ZIF-8_rag_clustered.pkl in the output directory.
```

### Batch processing with configuration

```bash
# Create configuration
mcli generate-config \
    --spacing 0.3 \
    --h 0.3 \
    --mask-thickness 0.7 > analysis_config.json

# Process all CIF files in directory
mcli process *.cif \
    -c analysis_config.json \
    -o ./batch_results/
```

### Interactive configuration

```bash
mcli config-building
# Follow the prompts to create a custom configuration
```

## Requirements

- Python 3.8+
- NumPy
- Rust (for building from source)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

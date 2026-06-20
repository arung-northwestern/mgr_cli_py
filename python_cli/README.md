# MCLI Python CLI (`python_cli/`)

Command-line and Streamlit entry points for MOF analysis.

**Full install and usage:** see the [repository root README](../README.md).

## Quick start

```bash
# From repo root (after pip install -e .)
mgrcli process structure.cif -o ./results/
mgrcli interactive --clean
```

## Layout

| Path | Role |
|------|------|
| `mgrcli/cli.py` | `mgrcli process`, `plot`, `generate-config`, `interactive` |
| `mgrcli/interactive_app.py` | Streamlit web UI |
| `mgrcli/plotting.py` | Plotly / Three.js HTML helpers |

## CLI vs web defaults

Batch CLI defaults (`cli.py` → `DEFAULTS`) differ from the Streamlit app on mask thickness, watershed backend, and VDW radii. Parity plan: [docs/plan_cli_web_parity.md](../docs/plan_cli_web_parity.md).

## Tests

```bash
cd python_cli && python -m pytest tests/
```

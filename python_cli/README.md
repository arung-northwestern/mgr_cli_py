# MCLI Python CLI (`python_cli/`)

Command-line and Streamlit entry points for MOF analysis.

**Full install and usage:** see the [repository root README](../README.md).

**AI agents:** [`skills/mgrcli/SKILL.md`](../skills/mgrcli/SKILL.md)  
**Config template:** [`config/web_analysis_template.json`](../config/web_analysis_template.json)

## Quick start

```bash
# From repo root (after pip install -e .)
mgrcli process structure.cif -o ./results/ -c config/web_analysis_template.json
mgrcli process structure.cif -o ./results/ --pore-dia --psd --sa --vol
mgrcli interactive --clean
```

## Layout

| Path | Role |
|------|------|
| `mgrcli/cli.py` | `mgrcli process`, `plot`, `generate-config`, `interactive` |
| `mgrcli/interactive_app.py` | Streamlit web UI |
| `mgrcli/plotting.py` | Plotly / Three.js HTML helpers |

## CLI vs web defaults

Web-aligned batch defaults: `mgrcli generate-config --profile web` or `config/web_analysis_template.json`. Textural property parity: [README § Textural properties](../README.md#textural-properties-cli-vs-web-app). Roadmap: [docs/plan_cli_web_parity.md](../docs/plan_cli_web_parity.md).

## Tests

```bash
cd python_cli && python -m pytest tests/
```

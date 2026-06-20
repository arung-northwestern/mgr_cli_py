# mgrcli configuration templates

## `web_analysis_template.json`

Web-aligned defaults for batch analysis (segmentation + optional textural properties).

**Generate fresh defaults:**

```bash
mgrcli generate-config --profile web -o my_config.json
```

**Full characterization example:**

```bash
mgrcli process structure.cif \
  -o ./out \
  -c config/web_analysis_template.json \
  --pore-dia --psd --psd-accessible --sa --vol \
  --probe-radius 1.2
```

**Legacy batch profile** (pre–web parity): `mgrcli generate-config --profile legacy -o legacy_config.json`

See [README](../README.md#textural-properties-cli-vs-web-app) for which analyses match the Streamlit app.

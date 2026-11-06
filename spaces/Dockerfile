# Hugging Face Spaces: same image as local Docker but MGR_DEPLOYMENT=spaces (HTS tab
# shows message only) and app listens on port 7860. Use this Dockerfile at repo root
# when deploying to HF Spaces (see scripts/deploy_to_hf_space.sh).

# Cache-busting: change this timestamp to force a full Docker rebuild on HF Spaces.
# Last updated: 2026-06-20 (drop Docker HEALTHCHECK — HF uses app_port probe)
ARG BUILD_DATE=2026-11-07a

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

WORKDIR /app

COPY . /app

# Build mgr_src/mgrcli_rust (not repo-root src/lib.rs — aromatic expand lives there only).
RUN pip install --no-cache-dir uv \
    && uv pip install --system maturin \
    && maturin build --release --manifest-path mgr_src/mgrcli_rust/Cargo.toml

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    PYTHONPATH=/app/python_cli \
    MGR_RESULTS_DIR=/results \
    MGR_DASK_DASHBOARD_PORT=8787 \
    MGR_DASK_DASHBOARD_HOST_PORT=8787 \
    MGR_DEPLOYMENT=spaces \
    MGRCLI_BUILD=13

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.docker.lock /app/requirements.docker.lock

RUN pip install --no-cache-dir uv \
    && uv pip install --system -r /app/requirements.docker.lock

COPY --from=builder /app/mgr_src/mgrcli_rust/target/wheels/*.whl /app/target/wheels/
COPY python_cli /app/python_cli
COPY mgr_src /app/mgr_src
COPY cifs /app/cifs
COPY scripts/docker_entrypoint.sh /app/scripts/docker_entrypoint.sh
COPY spaces/.streamlit/config.toml /app/.streamlit/config.toml

RUN uv pip install --system /app/target/wheels/*.whl \
    && chmod +x /app/scripts/docker_entrypoint.sh

# HF Spaces probes app_port itself; a Docker HEALTHCHECK can leave the Hub UI stuck on
# "Starting" even when Streamlit is up and serving on 0.0.0.0:7860.
EXPOSE 7860

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["streamlit", "run", "python_cli/mgrcli/interactive_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]

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

RUN pip install --no-cache-dir uv \
    && uv pip install --system maturin \
    && maturin build --release

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    PYTHONPATH=/app/python_cli

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.docker.lock /app/requirements.docker.lock

RUN pip install --no-cache-dir uv \
    && uv pip install --system -r /app/requirements.docker.lock

COPY --from=builder /app/target/wheels/*.whl /app/target/wheels/
COPY . /app

RUN uv pip install --system /app/target/wheels/*.whl \
    && chmod +x /app/scripts/docker_entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["streamlit", "run", "python_cli/mgrcli/interactive_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

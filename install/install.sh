#!/usr/bin/env bash
# Install mgrcli (Python CLI + Rust extension) on macOS or Linux.
#
# One-liner (public repo, or a private repo after `gh auth setup-git`):
#   curl -fsSL https://raw.githubusercontent.com/arung-northwestern/mgr_cli_py/master/install/install.sh | bash
#
# From a clone (uses this checkout; does not switch branches):
#   ./install/install.sh
#
# Environment:
#   MGRCLI_HOME          install prefix (default: $HOME/.mgrcli)
#   MGRCLI_REF           branch, tag, or commit when cloning (default: master)
#   MGRCLI_REPO          use this checkout instead of cloning
#   MGRCLI_REPO_URL      git URL (default: the GitHub repository)
#   MGRCLI_PYTHON        python executable to use for the venv
#   MGRCLI_SKIP_RUST=1   do not install rustup when cargo is missing
#   MGRCLI_SKIP_SAMPLE=1 skip the ZIF-8 sample run (version and imports still run)
#
# No sudo. Safe to re-run.

set -euo pipefail

MGRCLI_HOME="${MGRCLI_HOME:-${HOME}/.mgrcli}"
MGRCLI_REF="${MGRCLI_REF:-master}"
MGRCLI_REPO_URL="${MGRCLI_REPO_URL:-https://github.com/arung-northwestern/mgr_cli_py.git}"
VENV="${MGRCLI_HOME}/venv"
MIN_PY_MINOR=10

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "==> $*" >&2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is not on PATH. $2"
}

python_ok() {
  local bin="$1"
  "$bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

pick_python() {
  if [[ -n "${MGRCLI_PYTHON:-}" ]]; then
    [[ -x "${MGRCLI_PYTHON}" || -n "$(command -v "${MGRCLI_PYTHON}" || true)" ]] || die "MGRCLI_PYTHON=${MGRCLI_PYTHON} is not executable."
    python_ok "${MGRCLI_PYTHON}" || die "MGRCLI_PYTHON=${MGRCLI_PYTHON} is older than Python 3.${MIN_PY_MINOR}."
    echo "${MGRCLI_PYTHON}"
    return
  fi
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_ok "${candidate}"; then
      command -v "${candidate}"
      return
    fi
  done
  die "Python 3.${MIN_PY_MINOR}+ is required (3.11–3.13 recommended). Install it, or set MGRCLI_PYTHON."
}

ensure_rust() {
  if [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
  fi
  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    info "Rust already installed: $(rustc --version)"
    return
  fi
  if [[ "${MGRCLI_SKIP_RUST:-}" == "1" ]]; then
    die "cargo is not on PATH. Install Rust from https://rustup.rs and re-run, or unset MGRCLI_SKIP_RUST."
  fi
  need_cmd curl "It is required to download rustup."
  info "cargo is not on PATH. Installing rustup into ${HOME}/.cargo (no sudo)."
  info "Abort next time with MGRCLI_SKIP_RUST=1 if you would rather install Rust yourself."
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
  command -v cargo >/dev/null 2>&1 || die "rustup finished but cargo is still not on PATH. Source ${HOME}/.cargo/env and re-run."
  info "Installed $(rustc --version)"
}

is_checkout() {
  local dir="$1"
  [[ -f "${dir}/pyproject.toml" && -f "${dir}/mgr_src/mgrcli_rust/Cargo.toml" && -f "${dir}/python_cli/mgrcli/cli.py" ]]
}

resolve_repo() {
  if [[ -n "${MGRCLI_REPO:-}" ]]; then
    local given
    given="$(cd "${MGRCLI_REPO}" && pwd)"
    is_checkout "${given}" || die "MGRCLI_REPO=${given} is not an mgrcli checkout (missing pyproject.toml or mgr_src/mgrcli_rust)."
    echo "${given}"
    return
  fi

  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    local script_dir candidate
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    candidate="$(cd "${script_dir}/.." && pwd)"
    if is_checkout "${candidate}"; then
      info "Using the checkout that contains this script: ${candidate}"
      echo "${candidate}"
      return
    fi
  fi

  local dest="${MGRCLI_HOME}/src"
  need_cmd git "It is required to clone ${MGRCLI_REPO_URL}."
  mkdir -p "${MGRCLI_HOME}"
  if [[ -d "${dest}/.git" ]]; then
    info "Updating existing clone ${dest} to ${MGRCLI_REF}"
    git -C "${dest}" fetch --tags origin
    git -C "${dest}" checkout "${MGRCLI_REF}"
    if git -C "${dest}" show-ref --verify --quiet "refs/remotes/origin/${MGRCLI_REF}"; then
      git -C "${dest}" merge --ff-only "origin/${MGRCLI_REF}"
    fi
  else
    info "Cloning ${MGRCLI_REPO_URL} (${MGRCLI_REF}) into ${dest}"
    if ! git clone --depth 1 --branch "${MGRCLI_REF}" "${MGRCLI_REPO_URL}" "${dest}"; then
      echo "error: git clone failed." >&2
      echo "If this repository is still private, authenticate and re-run:" >&2
      echo "  gh auth login && gh auth setup-git" >&2
      echo "Do not paste a token into the install command or into this script." >&2
      exit 1
    fi
  fi
  is_checkout "${dest}" || die "Clone at ${dest} does not look like mgrcli."
  echo "${dest}"
}

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
case "${OS_NAME}" in
  Darwin|Linux) ;;
  *)
    die "Unsupported OS '${OS_NAME}'. On Windows, clone the repo and run install\\install.bat from PowerShell or cmd."
    ;;
esac
info "Detected ${OS_NAME} ${ARCH_NAME}"

ensure_rust
PYBIN="$(pick_python)"
info "Using Python: ${PYBIN} ($("${PYBIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

REPO="$(resolve_repo)"
info "Installing from ${REPO}"

mkdir -p "${MGRCLI_HOME}"
if [[ -x "${VENV}/bin/python" ]]; then
  info "Reusing virtualenv ${VENV}"
elif command -v uv >/dev/null 2>&1; then
  info "Creating virtualenv with uv at ${VENV}"
  uv venv --seed --python "${PYBIN}" "${VENV}"
else
  info "uv not found; creating virtualenv with ${PYBIN} -m venv at ${VENV}"
  "${PYBIN}" -m venv "${VENV}"
fi
[[ -x "${VENV}/bin/python" ]] || die "Virtualenv has no ${VENV}/bin/python."
if ! "${VENV}/bin/python" -m pip --version >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${VENV}/bin/python" pip
  else
    die "The virtualenv has no pip. Recreate it or install pip into ${VENV}."
  fi
fi

info "Installing mgrcli into the virtualenv (compiles mgrcli_rust; often 5–15 minutes the first time)"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install "${REPO}"

info "Checking the install"
"${VENV}/bin/mgrcli" --version
"${VENV}/bin/python" -c "import mgrcli, mgr, mgrcli_rust; print('import ok', mgrcli.__version__)"

SAMPLE="${REPO}/cifs/ZIF-8-RASPA.cif"
if [[ "${MGRCLI_SKIP_SAMPLE:-}" == "1" ]]; then
  info "Skipping sample CIF (MGRCLI_SKIP_SAMPLE=1)"
elif [[ -f "${SAMPLE}" ]]; then
  info "Running a short segmentation on cifs/ZIF-8-RASPA.cif"
  mkdir -p "${MGRCLI_HOME}/verify_out"
  "${VENV}/bin/mgrcli" process "${SAMPLE}" -o "${MGRCLI_HOME}/verify_out"
else
  info "No sample CIF at ${SAMPLE}; skipped process check"
fi

echo
echo "mgrcli is installed."
echo "Activate it with:"
echo "  source \"${VENV}/bin/activate\""
echo "Or call it directly:"
echo "  ${VENV}/bin/mgrcli --version"

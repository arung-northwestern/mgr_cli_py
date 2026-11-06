"""PEP 517 backend that stages ``mgr`` next to ``mgrcli`` for maturin.

Maturin copies Python packages from a single ``python-source`` directory.
``mgrcli`` lives in ``python_cli/mgrcli`` and ``mgr`` lives in ``mgr_src/mgr``.
A non-editable wheel only contains ``mgr`` if that tree is copied under
``python_cli/mgr`` before maturin runs. The copy is deleted afterwards.

Editable installs keep a symlink ``python_cli/mgr`` -> ``../mgr_src/mgr`` so
maturin’s ``.pth`` (which points at ``python_cli``) imports the live sources.
Maturin 1.8–1.11 does not follow a directory symlink when building a wheel,
so wheel builds use a real copy.

Maturin 1.12 and newer abort when ``python-source`` is set and
``python_cli/mgrcli_rust/`` does not exist. The Rust extension must stay a
top-level module (``import mgrcli_rust``), so the build dependency is pinned
below 1.12 in ``pyproject.toml``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


def _project_root() -> Path:
    return Path.cwd().resolve()


def _staged_mgr(root: Path) -> Path:
    return root / "python_cli" / "mgr"


def _source_mgr(root: Path) -> Path:
    src = root / "mgr_src" / "mgr"
    init = src / "__init__.py"
    if not init.is_file():
        raise RuntimeError(
            f"Cannot stage the mgr package: {init} is missing. "
            "Run this build from the mgr_cli_py repository root."
        )
    return src


def _remove_staged(dest: Path) -> None:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)


def stage_mgr_copy(root: Path | None = None) -> Path:
    """Copy ``mgr_src/mgr`` to ``python_cli/mgr`` so a wheel can include it."""
    root = root or _project_root()
    src = _source_mgr(root)
    dest = _staged_mgr(root)
    _remove_staged(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.so", "*.pyd"),
    )
    return dest


def stage_mgr_symlink(root: Path | None = None) -> Path:
    """Point ``python_cli/mgr`` at the live ``mgr_src/mgr`` tree."""
    root = root or _project_root()
    _source_mgr(root)
    dest = _staged_mgr(root)
    _remove_staged(dest)
    dest.symlink_to(Path("..") / "mgr_src" / "mgr")
    return dest


def _delegate(name: str):
    import maturin

    fn = getattr(maturin, name, None)
    if fn is None:
        raise RuntimeError(
            f"Installed maturin has no {name}(). "
            "Install maturin>=1.8,<1.12 (see pyproject.toml)."
        )
    return fn


def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    root = _project_root()
    stage_mgr_copy(root)
    try:
        return _delegate("build_wheel")(wheel_directory, config_settings, metadata_directory)
    finally:
        _remove_staged(_staged_mgr(root))


def build_editable(
    wheel_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    # Leave the symlink in place: the editable .pth adds python_cli to sys.path.
    stage_mgr_symlink(_project_root())
    return _delegate("build_editable")(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory: str, config_settings: Optional[Mapping[str, Any]] = None) -> str:
    return _delegate("build_sdist")(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings: Optional[Mapping[str, Any]] = None):
    return _delegate("get_requires_for_build_wheel")(config_settings)


def get_requires_for_build_editable(config_settings: Optional[Mapping[str, Any]] = None):
    return _delegate("get_requires_for_build_editable")(config_settings)


def get_requires_for_build_sdist(config_settings: Optional[Mapping[str, Any]] = None):
    return _delegate("get_requires_for_build_sdist")(config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: Optional[Mapping[str, Any]] = None
) -> str:
    stage_mgr_copy(_project_root())
    return _delegate("prepare_metadata_for_build_wheel")(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: Optional[Mapping[str, Any]] = None
) -> str:
    stage_mgr_symlink(_project_root())
    return _delegate("prepare_metadata_for_build_editable")(metadata_directory, config_settings)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "--stage-copy":
        print(stage_mgr_copy())
    elif command == "--stage-symlink":
        print(stage_mgr_symlink())
    elif command == "--unstage":
        _remove_staged(_staged_mgr(_project_root()))
        print("removed")
    else:
        print(
            "usage: python packaging/mgrcli_pep517.py "
            "--stage-copy | --stage-symlink | --unstage",
            file=sys.stderr,
        )
        sys.exit(2)

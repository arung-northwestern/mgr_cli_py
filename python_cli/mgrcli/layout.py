"""Locate the source checkout versus an installed copy of mgrcli.

Non-editable wheels install ``mgrcli``, ``mgr``, and ``mgrcli_rust`` as
top-level modules. A git checkout keeps those trees in ``python_cli/mgrcli``
and ``mgr_src/mgr``. Callers use that difference instead of assuming
``Path(__file__).parents[2]`` is the repository root (that guess is the
repository in a checkout and a directory above ``site-packages`` once the
package is installed).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _package_file() -> Path:
    return Path(__file__).resolve()


def source_checkout_root() -> Path | None:
    """Return the git checkout root when this module is the source tree."""
    here = _package_file()
    if here.parent.name != "mgrcli" or here.parents[1].name != "python_cli":
        return None
    repo = here.parents[2]
    # Docker/Spaces images copy python_cli/ and mgr_src/ but not pyproject.toml.
    if (repo / "mgr_src" / "mgr" / "__init__.py").is_file():
        return repo
    return None


def ensure_import_paths() -> None:
    """Put the checkout's ``mgr`` package on ``sys.path`` when developing in-tree.

    Installed environments already expose ``mgr``. Adding ``site-packages``
    parents, or a path computed as ``parents[2]`` of an installed file, would
    point at the wrong tree.
    """
    here = _package_file()
    if here.parent.name != "mgrcli":
        return
    pkg_parent = str(here.parents[1])
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    repo = source_checkout_root()
    if repo is None:
        return
    mgr_src = str(repo / "mgr_src")
    if mgr_src not in sys.path:
        sys.path.insert(0, mgr_src)

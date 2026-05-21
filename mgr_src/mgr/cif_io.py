"""Fast CIF → ASE Atoms (Rust parser). Falls back to ase.io.read on failure."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)

# On Spaces, large CIFs that fail the Rust parser should not fall back to ASE.
SPACES_LARGE_CIF_WARN_BYTES = 500 * 1024


class LargeCifSlowAseError(ValueError):
    """Rust CIF parse failed on a large file where ASE fallback is impractical on Spaces."""


def _is_spaces_deployment() -> bool:
    if os.environ.get("MGR_FORCE_LOCAL") == "1":
        return False
    if (os.environ.get("MGR_DEPLOYMENT") or "").strip().lower() == "spaces":
        return True
    return bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))


def _spaces_large_cif_slow_ase_message(path: Path) -> str:
    size_kb = path.stat().st_size / 1024
    return (
        f"Fast CIF parser failed for this upload ({size_kb:.0f} KB, >500 KB). "
        "ASE fallback would be too slow on the hosted demo. "
        "Reformat the CIF (e.g. primitive cell, fewer atoms per loop) and re-upload."
    )


def _use_fast_cif() -> bool:
    if os.environ.get("MGR_FORCE_LOCAL") == "1":
        return os.environ.get("MGR_FAST_CIF", "0") == "1"
    deployment = (os.environ.get("MGR_DEPLOYMENT") or "").strip().lower()
    if deployment == "spaces":
        return os.environ.get("MGR_FAST_CIF", "1") != "0"
    return os.environ.get("MGR_FAST_CIF", "0") == "1"


def read_cif_to_ase(
    path: Union[str, Path],
    *,
    on_large_cif_slow_ase_warning: Optional[Callable[[str], None]] = None,
):
    """Return ``ase.Atoms`` from CIF path using Rust fast path when enabled."""
    from ase import Atoms

    path = Path(path)
    if _use_fast_cif():
        try:
            import mgrcli_rust

            symbols, positions, cell, pbc = mgrcli_rust.read_cif(str(path))
            return Atoms(
                symbols=list(symbols),
                positions=positions,
                cell=cell,
                pbc=[bool(pbc[i]) for i in range(3)],
            )
        except Exception as exc:
            logger.warning(
                "Fast CIF read failed for %s: %s — falling back to ASE", path, exc
            )
            if (
                _is_spaces_deployment()
                and path.is_file()
                and path.stat().st_size > SPACES_LARGE_CIF_WARN_BYTES
            ):
                msg = _spaces_large_cif_slow_ase_message(path)
                if on_large_cif_slow_ase_warning is not None:
                    on_large_cif_slow_ase_warning(msg)
                raise LargeCifSlowAseError(msg) from exc

    from ase.io import read as ase_read

    return ase_read(str(path))

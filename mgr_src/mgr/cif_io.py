"""Fast CIF → ASE Atoms (Rust parser). Falls back to ase.io.read on failure."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def _use_fast_cif() -> bool:
    if os.environ.get("MGR_FORCE_LOCAL") == "1":
        return os.environ.get("MGR_FAST_CIF", "0") == "1"
    deployment = (os.environ.get("MGR_DEPLOYMENT") or "").strip().lower()
    if deployment == "spaces":
        return os.environ.get("MGR_FAST_CIF", "1") != "0"
    return os.environ.get("MGR_FAST_CIF", "0") == "1"


def read_cif_to_ase(path: Union[str, Path]):
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

    from ase.io import read as ase_read

    return ase_read(str(path))

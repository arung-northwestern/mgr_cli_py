"""Fast CIF → ASE Atoms (Rust parser). Falls back to ase.io.read on failure.

Phase 2: Wyckoff-aware deduplication via ``Spacegroup.equivalent_sites``.
Always attempted after the Rust parse; internal guards make this a no-op
(returns None) for CIFs where Rust is already correct.  The only cost is
a single ASE ``parse_cif`` call (~5–15 ms for typical MOF CIFs).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# On Spaces, large CIFs that fail the Rust parser should not fall back to ASE.
SPACES_LARGE_CIF_WARN_BYTES = 500 * 1024

# ASE Spacegroup equivalent_sites symprec (matches ASE crystal() default).
_SYMPREC = 0.001


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
    # Default: Rust parser everywhere (Spaces and local). Set MGR_FAST_CIF=0 for ASE-only.
    return os.environ.get("MGR_FAST_CIF", "1") != "0"


def _spglib_dedupe(atoms):
    """Re-expand from CIF asymmetric unit using ASE ``Spacegroup.equivalent_sites``.

    Returns deduped ``Atoms`` if ``equivalent_sites`` produces *fewer* atoms
    than Rust (merging Wyckoff duplicates), or ``None`` otherwise.

    This handles files where Rust's CIF sym-op expansion produces extra
    Wyckoff-equivalent atoms that ASE merges via its ``crystal()`` path
    (e.g. COF-202, PCN-60/61/68).
    """
    from ase.io.cif import parse_cif
    from ase.spacegroup import Spacegroup

    cif_path = atoms.info.get("_cif_path")
    if cif_path is None:
        return None

    cif_path = Path(cif_path)
    if not cif_path.is_file():
        return None

    try:
        with open(cif_path) as f:
            blocks = list(parse_cif(f, reader="ase"))
        block = blocks[0]
    except Exception:
        return None

    fx = block.get("_atom_site_fract_x")
    fy = block.get("_atom_site_fract_y")
    fz = block.get("_atom_site_fract_z")
    sg_no = block.get("_symmetry_int_tables_number")

    if fx is None or fy is None or fz is None or sg_no is None or sg_no == 1:
        return None

    try:
        asu_pos = np.column_stack([fx, fy, fz])
        sg_no_int = int(sg_no) if isinstance(sg_no, (int, float)) else None
        if sg_no_int is None or sg_no_int <= 1:
            return None
        sg = Spacegroup(sg_no_int)
        sites, kinds = sg.equivalent_sites(
            asu_pos, onduplicates="warn", symprec=_SYMPREC
        )
    except Exception:
        return None

    equiv_n = len(sites)
    rust_n = len(atoms)

    if equiv_n == rust_n:
        return None  # Same count — no dedupe needed.

    if equiv_n > rust_n:
        # equivalent_sites produced MORE atoms than Rust — the CIF ASU
        # has symmetry-equivalent entries that orbit expansion grows further.
        # Do NOT override Rust's (correct) result.
        return None

    # equiv_n < rust_n → equivalent_sites merged duplicates that Rust missed.
    from ase import Atoms

    orig_symbols = list(block.get("_atom_site_type_symbol", []))
    symbols = [
        orig_symbols[k] if k < len(orig_symbols) else "X" for k in kinds
    ]
    deduped = Atoms(
        symbols=symbols,
        scaled_positions=sites,
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
        info=dict(atoms.info),
    )
    # Copy over any extra arrays (occupancy, tags, etc.).
    for name in atoms.arrays:
        if deduped.has(name):
            continue
        arr = atoms.get_array(name)
        deduped.new_array(
            name, [arr[i] for i in kinds], dtype=arr.dtype, shape=arr.shape[1:]
        )

    logger.info(
        "spglib dedupe: %s rust=%d → equiv=%d",
        cif_path.name, rust_n, equiv_n,
    )
    return deduped


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
            atoms = Atoms(
                symbols=list(symbols),
                positions=positions,
                cell=cell,
                pbc=[bool(pbc[i]) for i in range(3)],
            )
            atoms.info["_cif_path"] = str(path)

            # Phase 2: spglib Wyckoff dedup — always attempt when Rust path
            # succeeds.  _spglib_dedupe has internal guards (no _cif_path,
            # SG ≤ 1, equiv ≥ rust → returns None) so the overhead for
            # already-correct CIFs is ~5–15 ms (a single ASE parse_cif call).
            deduped = _spglib_dedupe(atoms)
            if deduped is not None:
                return deduped

            return atoms
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

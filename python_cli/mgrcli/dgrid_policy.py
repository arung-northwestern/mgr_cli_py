"""Distance-grid method selection (shared by CLI and Streamlit)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

DGRID_FLOAT_BYTES = 4
LOCAL_DGRID_RAM_BUDGET_BYTES = int(16 * 1024**3)
DGRID_OPS_AUTO_AABB = 1_000_000_000


def _dgrid_shape_from_cell_lengths(cell_lengths_abc, spacing: float) -> Tuple[int, int, int]:
    spacing = max(float(spacing), 1e-9)
    lengths = np.asarray(cell_lengths_abc[:3], dtype=float)
    shape = (lengths / spacing).astype(int) + 1
    return int(shape[0]), int(shape[1]), int(shape[2])


def estimate_rust_dgrid_memory_bytes(
    n_atoms: int,
    spacing: float,
    cell_lengths_abc,
) -> Tuple[int, int, Tuple[int, int, int]]:
    nx, ny, nz = _dgrid_shape_from_cell_lengths(cell_lengths_abc, spacing)
    n_grid = nx * ny * nz
    n_atoms = max(int(n_atoms), 0)
    output_bytes = n_grid * DGRID_FLOAT_BYTES
    input_bytes = n_atoms * 32
    estimated = int((output_bytes + input_bytes) * 2)
    return estimated, n_grid, (nx, ny, nz)


def _ops_auto_aabb_threshold() -> int:
    raw = os.environ.get("MGR_DGRID_OPS_AUTO_AABB", "").strip()
    if raw:
        try:
            return int(float(raw))
        except ValueError:
            pass
    return DGRID_OPS_AUTO_AABB


def should_auto_aabb_dgrid(
    atoms,
    spacing: float,
    *,
    ram_budget_bytes: Optional[int] = None,
    ops_threshold: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    budget = LOCAL_DGRID_RAM_BUDGET_BYTES if ram_budget_bytes is None else int(ram_budget_bytes)
    ops_thr = _ops_auto_aabb_threshold() if ops_threshold is None else int(ops_threshold)
    lengths = tuple(float(x) for x in atoms.cell.cellpar()[:3])
    memory_b, n_grid, shape = estimate_rust_dgrid_memory_bytes(len(atoms), spacing, lengths)
    ops = n_grid * len(atoms)
    if memory_b > budget:
        use_aabb, reason = True, "memory"
    elif ops > ops_thr:
        use_aabb, reason = True, "work"
    else:
        use_aabb, reason = False, None
    return use_aabb, {
        "n_atoms": len(atoms),
        "n_grid": n_grid,
        "grid_shape": shape,
        "estimated_bytes": memory_b,
        "budget_bytes": budget,
        "ops": ops,
        "ops_threshold": ops_thr,
        "reason": reason,
    }


def resolve_dgrid_method(
    atoms,
    spacing: float,
    configured_method: str,
    *,
    auto_dgrid: bool = True,
    ram_budget_bytes: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Pick dgrid method; auto-switch rust_exact → rust_aabb when budget exceeded."""
    meta: Dict[str, Any] = {"configured": configured_method, "auto_dgrid": bool(auto_dgrid)}
    if not auto_dgrid or configured_method != "rust_exact":
        meta["resolved"] = configured_method
        return configured_method, meta
    use_aabb, aabb_meta = should_auto_aabb_dgrid(
        atoms, spacing, ram_budget_bytes=ram_budget_bytes
    )
    meta.update(aabb_meta)
    if use_aabb:
        meta["resolved"] = "rust_aabb"
        return "rust_aabb", meta
    meta["resolved"] = "rust_exact"
    return "rust_exact", meta

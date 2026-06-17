"""Method-specific keys and helpers for percolation history and pore diameters on results dicts."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PERCOLATION_HISTORY_WINDING = "percolation_history_winding"
PERCOLATION_HISTORY_LATTICE = "percolation_history_lattice"
PERCOLATION_HISTORY_LEGACY = "percolation_history"
PERCOLATION_LATTICE_VARIANT = "percolation_lattice_variant"

PORE_DIA_WINDING = "pore_dia_winding"
PORE_DIA_LATTICE = "pore_dia_lattice"

HistoryEntry = Tuple[float, Dict[Any, List[List[int]]]]


def get_percolation_history(
    results: Dict[str, object],
    method: str = "winding",
    *,
    lattice_variant: Optional[str] = None,
) -> List[HistoryEntry]:
    """Return stored percolation history for *method*, with legacy fallback."""
    method = method.lower()
    if method == "winding":
        key = PERCOLATION_HISTORY_WINDING
    elif method == "lattice":
        key = PERCOLATION_HISTORY_LATTICE
    else:
        raise ValueError(f"Unknown percolation method: {method!r}")

    if key in results:
        return list(results[key])  # type: ignore[arg-type]

    legacy = results.get(PERCOLATION_HISTORY_LEGACY)
    if legacy is not None:
        warnings.warn(
            f"Reading legacy {PERCOLATION_HISTORY_LEGACY!r}; prefer {key!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(legacy)  # type: ignore[arg-type]

    return []


def has_percolation_history(results: Dict[str, object], method: str) -> bool:
    method = method.lower()
    key = (
        PERCOLATION_HISTORY_WINDING
        if method == "winding"
        else PERCOLATION_HISTORY_LATTICE
    )
    if key in results:
        return True
    if PERCOLATION_HISTORY_LEGACY in results and method == "winding":
        return True
    return False


def store_percolation_history(
    results: Dict[str, object],
    method: str,
    history: List[HistoryEntry],
    *,
    lattice_variant: Optional[str] = None,
    pld_locations=None,
) -> Dict[str, object]:
    """Write history under a method-specific key without removing the other method."""
    method = method.lower()
    if method == "winding":
        results[PERCOLATION_HISTORY_WINDING] = history
    elif method == "lattice":
        results[PERCOLATION_HISTORY_LATTICE] = history
        if lattice_variant is not None:
            results[PERCOLATION_LATTICE_VARIANT] = lattice_variant
    else:
        raise ValueError(f"Unknown percolation method: {method!r}")

    if pld_locations is not None:
        suffix = "winding" if method == "winding" else "lattice"
        results[f"pld_locations_{suffix}"] = pld_locations

    return results


def store_pore_dia(
    results: Optional[Dict[str, object]],
    method: str,
    lcd: float,
    pld: float,
    dfsd: float,
    *,
    lattice_variant: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Store pore diameters under ``pore_dia_winding`` or ``pore_dia_lattice``."""
    if results is None:
        return
    method = method.lower()
    payload: Dict[str, Any] = {
        "lcd": float(lcd),
        "pld": float(pld),
        "dfsd": float(dfsd),
        "method": method,
    }
    if lattice_variant is not None:
        payload["variant"] = lattice_variant
    if extra:
        payload.update(extra)
    if method == "winding":
        results[PORE_DIA_WINDING] = payload
    elif method == "lattice":
        results[PORE_DIA_LATTICE] = payload
    else:
        raise ValueError(f"Unknown pore-dia method: {method!r}")


def get_pore_dia(
    results: Dict[str, object],
    method: str = "winding",
) -> Optional[Dict[str, Any]]:
    method = method.lower()
    key = PORE_DIA_WINDING if method == "winding" else PORE_DIA_LATTICE
    val = results.get(key)
    return dict(val) if isinstance(val, dict) else None

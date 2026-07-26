"""Plotly vs Three.js trace/figure backend checks for mgr plotting."""
from __future__ import annotations

from typing import Any, Iterator

MOFOGRAPHY_BACKEND_KEY = "_mofography_backend"
THREEJS_BACKEND = "threejs"
PLOTLY_BACKEND = "plotly"

_NATIVE_THREEJS_TRACE_TYPES = frozenset(
    {
        "mesh",
        "points",
        "line_segments",
        "unit_cell_lines",
        "instanced_spheres",
        "instanced_cylinders",
    }
)


def tag_threejs_trace(trace: dict[str, Any]) -> dict[str, Any]:
    trace[MOFOGRAPHY_BACKEND_KEY] = THREEJS_BACKEND
    return trace


def is_threejs_trace(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get(MOFOGRAPHY_BACKEND_KEY) == THREEJS_BACKEND:
        return True
    return (
        "type" in obj
        and "data" in obj
        and "id" in obj
        and obj.get("type") in _NATIVE_THREEJS_TRACE_TYPES
    )


def is_plotly_trace(obj: Any) -> bool:
    try:
        from plotly.basedatatypes import BaseTraceType

        return isinstance(obj, BaseTraceType)
    except Exception:
        return hasattr(obj, "to_plotly_json") and not is_threejs_trace(obj)


def is_threejs_figure(fig: Any) -> bool:
    from .movies import MovieFigure

    fig_cls = type(fig)
    if fig_cls.__name__ in ("Figure3JS", "Figure3js"):
        return True
    return isinstance(fig, MovieFigure)


def is_plotly_figure(fig: Any) -> bool:
    try:
        import plotly.graph_objects as go

        return isinstance(fig, go.Figure)
    except Exception:
        return False


def iter_leaf_traces(traces: Any) -> Iterator[Any]:
    if traces is None:
        return
    if isinstance(traces, dict):
        if is_threejs_trace(traces) or is_plotly_trace(traces):
            yield traces
            return
        for key, val in traces.items():
            if key == "payload":
                continue
            yield from iter_leaf_traces(val)
        return
    if isinstance(traces, (list, tuple)):
        for item in traces:
            yield from iter_leaf_traces(item)
        return
    yield traces


def ensure_threejs_trace(obj: Any, *, context: str = "add_trace") -> None:
    if is_threejs_trace(obj):
        return
    if is_plotly_trace(obj):
        raise TypeError(
            f"{context}: received a Plotly trace but this figure is Three.js. "
            "Use mgr.plot_* for Plotly, or mgr.plot_*_3js / Figure3js for Three.js."
        )
    raise TypeError(
        f"{context}: expected a mgr Three.js trace dict "
        f"(keys id, type, data, style), got {type(obj).__name__}."
    )


def ensure_plotly_trace(obj: Any, *, context: str = "add_trace") -> None:
    if is_plotly_trace(obj):
        return
    if is_threejs_trace(obj):
        raise TypeError(
            f"{context}: received a Three.js trace but this figure is Plotly. "
            "Build a Figure3js scene or use plotly.graph_objects traces only."
        )
    raise TypeError(
        f"{context}: expected a Plotly trace (plotly.graph_objects.*), "
        f"got {type(obj).__name__}."
    )


def ensure_threejs_figure(fig: Any, *, context: str = "add_traces") -> None:
    if is_threejs_figure(fig):
        return
    if is_plotly_figure(fig):
        raise TypeError(
            f"{context}: expected Figure3js / Figure3D, got plotly.graph_objects.Figure. "
            "Use plot_*_3js traces only on Three.js figures."
        )
    raise TypeError(
        f"{context}: expected Figure3js or Figure3D (Three.js), got {type(fig).__name__}."
    )


def ensure_plotly_figure(fig: Any, *, context: str = "add_traces") -> None:
    if is_plotly_figure(fig):
        return
    if is_threejs_figure(fig):
        raise TypeError(
            f"{context}: expected plotly.graph_objects.Figure, got a Three.js figure. "
            "Use Figure3js.add_traces() for Three.js trace bundles."
        )
    raise TypeError(
        f"{context}: expected plotly.graph_objects.Figure, got {type(fig).__name__}."
    )


def validate_threejs_traces(traces: Any, *, context: str = "add_traces") -> None:
    for leaf in iter_leaf_traces(traces):
        ensure_threejs_trace(leaf, context=context)


def validate_plotly_traces(traces: Any, *, context: str = "add_traces") -> None:
    for leaf in iter_leaf_traces(traces):
        ensure_plotly_trace(leaf, context=context)

"""Streamlit application for interactive MOF analysis."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import hashlib
import html
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO, StringIO

import plotly.express as px

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import plotly.graph_objects as go
import pandas as pd

import sys
from pathlib import Path
# Add mgr_src to path to use reference functions directly
REPO_ROOT = Path(__file__).resolve().parents[2]
MGR_SRC_PATH = REPO_ROOT / "mgr_src"
if str(MGR_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(MGR_SRC_PATH))

def _import_mgr_module():
    import mgr as _mgr
    return _mgr


class _MgrProxy:
    _module = None

    def _get_module(self):
        if self._module is None:
            self._module = _import_mgr_module()
        return self._module

    def __getattr__(self, item):
        return getattr(self._get_module(), item)


mgr = _MgrProxy()


def _read_atoms_from_cif(cif_path: Path):
    from ase.io import read as _read

    return _read(str(cif_path))


def _step_status(step: int, total: int, label: str, color: str = "#7FDBFF", bg: str = "#102a43"):
    """Render a numbered step status bar for the main plotting pipeline."""
    return (
        f"<div style='padding:6px;background:{bg};color:{color};text-align:center;"
        f"font-family:monospace;font-size:13px;border-radius:4px;'>"
        f"<b>Step {step}/{total}</b> — {label}</div>"
    )


# Watershed backend labels (selectbox) and descriptions (shown under the selection).
WATERSHED_BACKEND_LABELS: Dict[str, str] = {
    "rust_det": "rust_det — heap watershed (Rust)",
    "rust_bucket": "rust_bucket — parallel buckets (Rust)",
    "rust_det_parallel": "rust_det_parallel — deterministic parallel (Rust)",
    "skimage": "skimage — scikit-image watershed",
}
WATERSHED_BACKEND_DETAILS: Dict[str, str] = {
    "rust_det": (
        "**rust_det** — Heap watershed (Rust). Deterministic; medium speed; single-threaded. "
        "Skimage-like tie handling for many structures."
    ),
    "rust_bucket": (
        "**rust_bucket** — Parallel buckets (Rust). Fastest and most memory-efficient. "
        "*Not* bit-deterministic under parallel execution: thread concurrency can change tie-breaks (*)."
    ),
    "rust_det_parallel": (
        "**rust_det_parallel** — Deterministic parallel (Rust). Deterministic; typically faster than "
        "**rust_det** on large grids."
    ),
    "skimage": (
        "**skimage** — scikit-image watershed. Deterministic; single-threaded. Speed broadly comparable "
        "to **rust_det**. Requires the `scikit-image` package."
    ),
}
WATERSHED_BACKEND_ORDER: Tuple[str, ...] = (
    "rust_bucket",
    "rust_det",
    "rust_det_parallel",
    "skimage",
)

SPACES_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
SPACES_MIN_SPACING = 0.25
SPACES_MAX_EXTRA_PADDING = 10.0
SPACES_MAX_MIN_BOX_THICKNESS = 12.0
SPACES_MIN_SECONDS_BETWEEN_RUNS = 15.0
SPACES_MAX_ATOMS = 5000
_SPACES_COMPUTE_LOCK = Lock()


def _is_spaces_deployment() -> bool:
    # Local CLI runs can inherit shell env accidentally; allow an explicit local override.
    if os.environ.get("MGR_FORCE_LOCAL") == "1":
        return False
    deployment = (os.environ.get("MGR_DEPLOYMENT") or "").strip().lower()
    if deployment:
        return deployment == "spaces"
    # Fallback for native Spaces envs even when MGR_DEPLOYMENT is not set.
    return bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))


def _spaces_allow_compute_now() -> bool:
    """Enforce a per-session cooldown for expensive compute actions on hosted demo."""
    if not _is_spaces_deployment():
        return True
    now = time.time()
    last_run = float(st.session_state.get("spaces_last_compute_at", 0.0))
    dt = now - last_run
    if dt < SPACES_MIN_SECONDS_BETWEEN_RUNS:
        wait_for = int(SPACES_MIN_SECONDS_BETWEEN_RUNS - dt) + 1
        st.error(f"Server busy (cooldown). Estimated wait: ~{wait_for}s.")
        st.caption("This public demo limits request rate so everyone gets a chance to run analyses.")
        return False
    st.session_state["spaces_last_compute_at"] = now
    return True


def _viz_settings_title_suffix(
    *,
    spacing: float,
    h_value: float,
    threshold_abs: float,
    mask_thickness: float,
) -> str:
    """Suffix for titles / HTML downloads: basic segmentation settings (underscore-separated)."""
    return (
        f"spacing_{spacing:g}_h_{h_value:g}_threshAbs_{threshold_abs:g}_mask_{mask_thickness:g}"
    )


def _percolation_axes_for_display(components: Any) -> Dict[str, Any]:
    """Normalize winding percolation axis keys (0–2 or x/y/z) for JSON / UI."""
    out: Dict[str, Any] = {"a-direction": [], "b-direction": [], "c-direction": []}
    if not isinstance(components, dict):
        return out
    letters = ("a-direction", "b-direction", "c-direction")
    for axis_key, comps in components.items():
        idx = None
        if isinstance(axis_key, int) and axis_key in (0, 1, 2):
            idx = axis_key
        elif isinstance(axis_key, str):
            lk = axis_key.lower()
            if lk in ("x", "a"):
                idx = 0
            elif lk in ("y", "b"):
                idx = 1
            elif lk in ("z", "c"):
                idx = 2
        if idx is None:
            continue
        if isinstance(comps, (list, tuple)):
            out[letters[idx]] = list(comps)
        else:
            out[letters[idx]] = [comps] if comps is not None else []
    return out


def _safe_uploaded_cif_path(uploaded_file: Any) -> Path:
    """Persist uploaded CIF to a temp location with a sanitized filename."""
    raw_name = Path(getattr(uploaded_file, "name", "upload.cif")).name
    safe_stem = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in Path(raw_name).stem)
    safe_stem = safe_stem[:80] or "upload"
    fd, tmp_name = tempfile.mkstemp(prefix=f"mgr_{safe_stem}_", suffix=".cif")
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.write_bytes(uploaded_file.getbuffer())
    return tmp_path


def _watershed_backend_widgets(widget_key_suffix: str) -> str:
    """Selectbox plus help text for the chosen backend. Suffix keeps Streamlit widget keys unique."""
    sel_key = f"watershed_backend_select_{widget_key_suffix}"
    choice = st.selectbox(
        "Watershed backend",
        options=list(WATERSHED_BACKEND_ORDER),
        index=1,
        format_func=lambda k: WATERSHED_BACKEND_LABELS[k],
        key=sel_key,
        help="Seeded watershed implementation used on the padded periodic distance grid.",
    )
    st.markdown(WATERSHED_BACKEND_DETAILS[choice])
    if choice == "skimage":
        try:
            import skimage.segmentation  # noqa: F401
        except ImportError:
            st.warning(
                "scikit-image is not installed. Install with: `pip install scikit-image` "
                'or `pip install "mgrcli[skimage]"`.'
            )
    return choice


def results_data_to_npz(results: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    serializable: Dict[str, Any] = {}
    for key, value in results.items():
        if key == "ase_atoms":
            continue
        if isinstance(value, np.ndarray):
            serializable[key] = value
        elif isinstance(value, (int, float, bool, str)):
            serializable[key] = value
    if not serializable:
        return b""
    np.savez(buffer, **serializable)
    buffer.seek(0)
    return buffer.getvalue()


def make_uniform_psd_dataframe(bins: np.ndarray, histogram: np.ndarray) -> pd.DataFrame:
    bin_starts = bins[:-1]
    bin_ends = bins[1:]
    bin_centers = 0.5 * (bin_starts + bin_ends)
    return pd.DataFrame(
        {
            "bin_start": bin_starts,
            "bin_end": bin_ends,
            "bin_center": bin_centers,
            "count": histogram,
        }
    )


def make_pocket_psd_dataframe(bins: np.ndarray, pocket_psds: Dict[Any, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if bins is None or len(bins) < 2:
        return pd.DataFrame(rows)

    bin_edges = np.asarray(bins, dtype=float)
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]
    default_centers = 0.5 * (bin_starts + bin_ends)

    for label, psd_data in pocket_psds.items():
        data_dict = psd_data if isinstance(psd_data, dict) else {"hist": np.asarray(psd_data)}
        hist = np.asarray(data_dict.get("hist", []), dtype=float)
        if hist.size == 0:
            continue

        centers = np.asarray(data_dict.get("bin_centers", default_centers), dtype=float)
        if centers.size != hist.size:
            centers = default_centers[: hist.size]

        for start, end, center, count in zip(bin_starts[: hist.size], bin_ends[: hist.size], centers, hist):
            rows.append(
                {
                    "region_label": int(label),
                    "bin_start (Å)": float(start),
                    "bin_end (Å)": float(end),
                    "bin_center (Å)": float(center),
                    "count": float(count),
                }
            )

    return pd.DataFrame(rows)

_CLI_FUNCS = None
_PLOTTING_SYMBOLS = None
_MESH_PREWARM_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class _StreamingProgress:
    """Thread-safe progress tracker for compute operations."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.phase: str = "Starting..."
        self.start_time: float = time.time()
        self.done: bool = False
        self.result: Any = None
        self.error: Optional[str] = None
        self._lock = Lock()

    def add_line(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            if len(self.lines) > 100:
                self.lines = self.lines[-80:]

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def latest_lines(self, count: int = 5) -> List[str]:
        with self._lock:
            return list(self.lines[-count:])


_PHASE_PATTERNS: List[Tuple[str, str]] = [
    ("Maxima detection", "computing maxima with pbc"),
    ("Pore limiting radius", "pore limiting radius is"),
    ("Watershed segmentation", "computing regions with pbc"),
    ("Regions computed", "computed regions without pbc"),
    ("Computing connections", "compute_connections is True"),
    ("Processing connections", "reindexng"),
    ("Padded segmentation complete", "padded segmentation workflow complete"),
    ("Maxima detection (direct)", "detect_periodic_maxima_direct"),
    ("Cleaning up regions", "looking to remove regions"),
    ("Reindexing", "done with reindexing"),
]


def _detect_phase_from_log(line: str) -> Optional[str]:
    line_lower = line.lower()
    for phase_label, pattern in _PHASE_PATTERNS:
        if pattern in line_lower:
            return phase_label
    return None


class _BufferingLogHandler(logging.Handler):
    """Capture log records emitted during a compute action."""

    def __init__(self, streaming_progress: Optional[_StreamingProgress] = None) -> None:
        super().__init__()
        self.lines: List[str] = []
        self._progress = streaming_progress

    def emit(self, record: logging.LogRecord) -> None:
        formatted = self.format(record)
        self.lines.append(formatted)
        if self._progress is not None:
            self._progress.add_line(formatted)
            detected_phase = _detect_phase_from_log(formatted)
            if detected_phase:
                self._progress.set_phase(detected_phase)


def _load_cli_funcs():
    global _CLI_FUNCS
    if _CLI_FUNCS is not None:
        return _CLI_FUNCS

    if __package__ is None or __package__ == "":  # pragma: no cover - streamlit entry
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from python_cli.mgrcli.cli import (
            compute_dgrid_rust as _compute_dgrid_rust,
            python_do_periodic_segmentation_rust as _python_do_periodic_segmentation_rust,
            get_pbc_groups as _get_pbc_groups,
        )
    else:  # pragma: no cover - package import path
        from .cli import (
            compute_dgrid_rust as _compute_dgrid_rust,
            python_do_periodic_segmentation_rust as _python_do_periodic_segmentation_rust,
            get_pbc_groups as _get_pbc_groups,
        )
    _CLI_FUNCS = (_compute_dgrid_rust, _python_do_periodic_segmentation_rust, _get_pbc_groups)
    return _CLI_FUNCS


def _load_plotting_symbols():
    global _PLOTTING_SYMBOLS
    if _PLOTTING_SYMBOLS is not None:
        return _PLOTTING_SYMBOLS

    if __package__ is None or __package__ == "":  # pragma: no cover - streamlit entry
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from python_cli.mgrcli.plotting import (
            show_mof as _show_mof,
            STYLIZE_LIGHTING as _STYLIZE_LIGHTING,
            STYLIZE_LIGHTPOSITION as _STYLIZE_LIGHTPOSITION,
            dark_layout as _dark_layout,
            white_layout as _white_layout,
            prepare_results_for_plot as _prepare_results_for_plot,
            prewarm_framework_cache as _prewarm_framework_cache,
            clear_framework_cache as _clear_framework_cache,
        )
    else:  # pragma: no cover - package import path
        from .plotting import (
            show_mof as _show_mof,
            STYLIZE_LIGHTING as _STYLIZE_LIGHTING,
            STYLIZE_LIGHTPOSITION as _STYLIZE_LIGHTPOSITION,
            dark_layout as _dark_layout,
            white_layout as _white_layout,
            prepare_results_for_plot as _prepare_results_for_plot,
            prewarm_framework_cache as _prewarm_framework_cache,
            clear_framework_cache as _clear_framework_cache,
        )
    _PLOTTING_SYMBOLS = (
        _show_mof,
        _STYLIZE_LIGHTING,
        _STYLIZE_LIGHTPOSITION,
        _dark_layout,
        _white_layout,
        _prepare_results_for_plot,
        _prewarm_framework_cache,
        _clear_framework_cache,
    )
    return _PLOTTING_SYMBOLS


def compute_dgrid_rust(*args, **kwargs):
    func, _, _ = _load_cli_funcs()
    return func(*args, **kwargs)


def python_do_periodic_segmentation_rust(*args, **kwargs):
    _, func, _ = _load_cli_funcs()
    return func(*args, **kwargs)


def get_pbc_groups(*args, **kwargs):
    _, _, func = _load_cli_funcs()
    return func(*args, **kwargs)


def show_mof(*args, **kwargs):
    func, _, _, _, _, _, _, _ = _load_plotting_symbols()
    return func(*args, **kwargs)


def prepare_results_for_plot(*args, **kwargs):
    _, _, _, _, _, func, _, _ = _load_plotting_symbols()
    return func(*args, **kwargs)


def prewarm_framework_cache(*args, **kwargs):
    _, _, _, _, _, _, func, _ = _load_plotting_symbols()
    return func(*args, **kwargs)


def clear_framework_cache(*args, **kwargs):
    _, _, _, _, _, _, _, func = _load_plotting_symbols()
    return func(*args, **kwargs)


STYLIZE_LIGHTING = None
STYLIZE_LIGHTPOSITION = None
dark_layout = None
white_layout = None


def _queue_mesh_prewarm(results_data: Optional[Dict[str, Any]], cache_identity: str, stylize: bool, show_rings: bool = True) -> None:
    if not results_data or "ase_atoms" not in results_data:
        return
    jobs = st.session_state.setdefault("mesh_prewarm_jobs", {})
    job_key = f"{cache_identity}::stylize={int(bool(stylize))}::rings={int(bool(show_rings))}"
    existing = jobs.get(job_key)
    if existing is not None and not existing.done():
        return
    if existing is not None and existing.done():
        return
    # Copy atoms so plotting prewarm can run safely in the background.
    atoms_copy = results_data["ase_atoms"].copy()
    jobs[job_key] = _MESH_PREWARM_EXECUTOR.submit(
        prewarm_framework_cache,
        atoms_copy,
        bool(stylize),
        bool(show_rings),
    )


def _append_runtime_logs(lines: List[str]) -> None:
    logs = st.session_state.setdefault("ui_logs", [])
    logs.extend([line for line in lines if line and line.strip()])
    if len(logs) > 600:
        del logs[: len(logs) - 600]


def _run_with_output_capture(callable_obj, *args, **kwargs):
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    root_logger = logging.getLogger()
    handler = _BufferingLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    prev_level = root_logger.level
    root_logger.addHandler(handler)
    if prev_level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = callable_obj(*args, **kwargs)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(prev_level)
    captured_lines: List[str] = []
    captured_lines.extend(handler.lines)
    captured_lines.extend(stdout_buffer.getvalue().splitlines())
    captured_lines.extend(stderr_buffer.getvalue().splitlines())
    _append_runtime_logs(captured_lines)
    return result


def _start_compute_thread(
    working_atoms,
    spacing: float,
    mask_thickness: float,
    h_value: float,
    min_distance: float,
    threshold_abs: float,
    extra_padding: float,
    watershed_backend: str,
    use_aabb_method: bool,
    block_size: int,
    min_box_thickness_enabled: bool = False,
    min_box_thickness: float = 6.0,
) -> _StreamingProgress:
    progress = _StreamingProgress()

    if use_aabb_method:
        progress.set_phase("Distance grid (AABB)")

    root_logger = logging.getLogger()
    handler = _BufferingLogHandler(streaming_progress=progress)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    prev_level = root_logger.level
    root_logger.addHandler(handler)
    if prev_level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    def _run():
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Handle minimum box thickness
                w_atoms = working_atoms
                if min_box_thickness_enabled:
                    cell_lengths = working_atoms.get_cell_lengths_and_angles()[:3]
                    if min(cell_lengths) < min_box_thickness:
                        repetitions = mgr.get_repetitions_to_meet_cutoff(working_atoms, min_box_thickness)
                        w_atoms = working_atoms * repetitions

                if use_aabb_method:
                    from mgr.compute import dgrid_from_atoms

                    def _aabb_progress(current: int, total: int, label: str):
                        progress.set_phase(f"{label} {current}/{total}")

                    dgrid_dask = dgrid_from_atoms(
                        w_atoms,
                        spacing=spacing,
                        block_size=block_size,
                        progress_callback=_aabb_progress,
                    )
                    dgrid = dgrid_dask.compute().astype(np.float32)
                else:
                    progress.set_phase("Distance grid (Rust)")
                    dgrid = compute_dgrid_rust(w_atoms, spacing)

                _results = _run_segmentation(
                    w_atoms,
                    dgrid,
                    spacing=spacing,
                    mask_thickness=mask_thickness,
                    h_value=h_value,
                    min_distance=min_distance,
                    threshold_abs=threshold_abs,
                    extra_padding=extra_padding,
                    watershed_backend=watershed_backend,
                )
                _results["ase_atoms"] = w_atoms
                _results["dgrid"] = _results.get("dgrid", dgrid)
                progress.result = _results
        except Exception as exc:
            progress.error = str(exc)
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(prev_level)
            captured_lines: List[str] = []
            captured_lines.extend(handler.lines)
            captured_lines.extend(stdout_buffer.getvalue().splitlines())
            captured_lines.extend(stderr_buffer.getvalue().splitlines())
            _append_runtime_logs(captured_lines)
            progress.done = True

    thread = Thread(target=_run, daemon=True)
    thread.start()
    st.session_state["_compute_thread"] = thread
    st.session_state["_compute_progress"] = progress
    return progress


def _render_progress_ui(progress: _StreamingProgress) -> None:
    elapsed = time.time() - progress.start_time
    phase = progress.phase or "Processing..."
    lines = progress.latest_lines(4)

    log_rows = ""
    for line in lines:
        safe_line = html.escape(line)[:140]
        log_rows += f'<div style="color:#7f8c8d;font-size:11px;font-family:monospace;line-height:1.3;margin:1px 0;">{safe_line}</div>'

    border_color = "#2ecc71" if not progress.done else "#1f6feb"
    status_markup = f"""<div style="border:2px solid {border_color};border-radius:8px;background:#0d1117;padding:12px;margin:8px 0;">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
  <div style="width:24px;height:24px;border:3px solid {border_color};border-top-color:transparent;border-radius:50%;animation:spin 1s linear infinite;"></div>
  <div>
    <div style="color:{border_color};font-weight:bold;font-size:15px;">⚡ {phase}</div>
    <div style="color:#8b949e;font-size:12px;">Elapsed: {elapsed:.1f}s</div>
  </div>
</div>
{log_rows}
</div>
<style>@keyframes spin {{ from {{ transform:rotate(0deg); }} to {{ transform:rotate(360deg); }} }}</style>"""

    components.html(status_markup, height=180)


def _render_logs_panel(log_text: str) -> None:
    safe_logs = html.escape(log_text or "(no logs yet)")
    components.html(
        f"""
        <div style="border:1px solid #3a3a3a;border-radius:8px;background:#111;padding:8px;">
          <pre id="mgr-runtime-logs" style="margin:0;max-height:260px;overflow:auto;white-space:pre-wrap;color:#e6e6e6;font-size:12px;line-height:1.35;">{safe_logs}</pre>
          <div style="display:flex;justify-content:flex-end;margin-top:8px;">
            <button
              onclick="navigator.clipboard.writeText(document.getElementById('mgr-runtime-logs').innerText)"
              style="background:#1f6feb;color:white;border:none;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px;"
            >
              Copy logs
            </button>
          </div>
        </div>
        """,
        height=330,
    )


ASCII_LOGO = r"""
███╗   ███╗ ██████╗ ███████╗ ██████╗  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██╗   ██╗
████╗ ████║██╔═══██╗██╔════╝██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║╚██╗ ██╔╝
██╔████╔██║██║   ██║█████╗  ██║   ██║██║  ███╗██████╔╝███████║██████╔╝███████║ ╚████╔╝ 
██║╚██╔╝██║██║   ██║██╔══╝  ██║   ██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║  ╚██╔╝  
██║ ╚═╝ ██║╚██████╔╝██║     ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║   ██║   
╚═╝     ╚═╝ ╚═════╝ ╚═╝      ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   
"""


def _mofography_logo_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Resolve official MOFography logos: (light_bg_logo, dark_bg_logo)."""
    light_name = "mofography_logo_white_background.png"
    dark_name = "mofography_logo_dark_background.png"
    bases: list[Path] = []
    # 1. In-repo docs/ (works in Docker / HF Spaces)
    cli_root = Path(__file__).resolve().parents[2]
    bases.append(cli_root / "docs")
    # 2. External MOFOGRAPHY_REPO env override
    env = (os.environ.get("MOFOGRAPHY_REPO") or "").strip()
    if env:
        bases.append(Path(env).expanduser())
    # 3. Local dev fallback
    bases.append(Path.home() / "repos" / "mofography")
    bases.append(cli_root.parent / "mofography")

    light_p: Optional[Path] = None
    dark_p: Optional[Path] = None
    for base in bases:
        docs = base / "docs"
        lp, dp = docs / light_name, docs / dark_name
        if lp.is_file():
            light_p = lp
        if dp.is_file():
            dark_p = dp
        if light_p and dark_p:
            break
    return light_p, dark_p


def _encode_image(image_path: Path) -> str:
    """Encode image file as base64 string for inline HTML."""
    import base64
    if not image_path or not image_path.is_file():
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_sample_cifs() -> Dict[str, Path]:
    sample_dir = Path(os.environ.get("MCLI_SAMPLE_CIF_DIR", Path(__file__).resolve().parents[2] / "cifs"))
    samples: Dict[str, Path] = {}
    if sample_dir.exists():
        for cif_path in sorted(sample_dir.glob("*.cif")):
            samples[cif_path.name] = cif_path
    return samples





def _run_segmentation(
    mat_atoms,
    dgrid: np.ndarray,
    spacing: float,
    mask_thickness: float,
    h_value: float,
    min_distance: float = 1.5,
    threshold_abs: float = 1.5,
    extra_padding: float = 5.0,
    watershed_backend: str = "rust_det",
) -> Dict[str, np.ndarray]:
    segmentation_kwargs = dict(
        dgrid=dgrid,
        mat_atoms=mat_atoms,
        mask_thickness=mask_thickness,
        h=h_value,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        return_padded_regions=False,
        spacing=spacing,
        minimum_window_separation=1.0,
        wall_thickness=0.05,
        extra_padding=extra_padding,
        compactness=0,
        min_region_size=0.001,
        disable_tqdm=True,
        compute_connections=True,
        return_dgrid=True,
        watershed_backend=watershed_backend,
    )
    return python_do_periodic_segmentation_rust(**segmentation_kwargs)


def _compute_pld_analysis(
    results: Dict[str, Any],
    mat_atoms,
    spacing: float,
    mask_thickness: float,
    h_value: float,
    threshold_abs: float,
    min_distance: float,
    extra_padding: float,
    force_cpu_dgrid: bool = True,
    watershed_backend: str = "rust_det",
) -> Dict[str, Any]:
    # For percolation, use CPU dgrid + fresh segmentation to match reference behavior.
    if force_cpu_dgrid:
        dgrid = mgr.dgrid_from_atoms_cpu(mat_atoms, spacing=spacing).astype(np.float32)
        analysis_results = mgr.do_periodic_segmentation_rust(
            dgrid,
            mat_atoms,
            spacing=spacing,
            mask_thickness=mask_thickness,
            h=h_value,
            threshold_abs=threshold_abs,
            min_distance=min_distance,
            extra_padding=extra_padding,
            return_dgrid=True,
            watershed_backend=watershed_backend,
        )
        prepared = prepare_results_for_plot(analysis_results, mat_atoms)
    else:
        analysis_results = results
        prepared = prepare_results_for_plot(results, mat_atoms)

    dgrid = np.asarray(prepared.get("dgrid"))
    print('Got grid')
    regions = np.asarray(prepared.get("regions"))
    print('Got regions')
    maxima_info = prepared.get("maxima_info")
    if dgrid is None or regions is None or dgrid.size == 0 or regions.size == 0:
        raise ValueError("Segmentation results are missing distance grid or regions.")

    maxima_raw = np.asarray(analysis_results.get("all_maxima", []), dtype=int)

    if maxima_raw.ndim == 1:
        maxima_raw = maxima_raw.reshape(-1, 3) if maxima_raw.size else np.empty((0, 3), dtype=int)
    if maxima_raw.ndim != 2:
        maxima_raw = np.empty((0, 3), dtype=int)

    unique_regions = np.asarray(analysis_results.get("unique_regions", []), dtype=int)

    # if unique_regions.size:
    #     max_label = int(unique_regions.max())
    # elif regions.size:
    #     max_label = int(np.max(regions))
    # else:
    #     max_label = maxima_raw.shape[0]
    #

    # max_label = min(max_label, maxima_raw.shape[0])
    # maxima = maxima_raw[:max_label] if max_label > 0 else np.empty((0, 3), dtype=int)

    # Use winding percolation for pore diameter analysis
    lcd, pld, dfsd, _ = mgr.winding_percolation.get_pore_dia_mgr_winding(
        mat_atoms=mat_atoms,
        dgrid=dgrid,
        kwargs=dict(
            spacing=spacing,
            mask_thickness=mask_thickness,
            h=h_value,
            threshold_abs=threshold_abs,
            min_distance=min_distance,
            extra_padding=extra_padding,
        ),
        results=analysis_results,
    )
    print('got pore dia (winding)')

    # Use winding percolation for percolation history
    percolation_history, pld_locations, final_clusters = mgr.winding_percolation.compute_percolation_history_winding(
        analysis_results,
        step=0.05,
        expand_components=False,
    )
    print('got percolation hist (winding)')

    pld_locations = np.asarray(pld_locations)

    connections_prepared = prepared.get("connections", [])
    connections_filtered= [ conn for conn in connections_prepared if np.all(np.isin(conn[0], unique_regions) )]

    all_maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)]
    maxima = maxima_raw[:regions.max()]
    maxima_radii = all_maxima_radii[:regions.max()]
    rag = mgr.rag_from_connections(connections_filtered, maxima)
    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, regions.shape, mat_atoms)
    rag.graph["regions"] = regions
    rag.graph["ase_atoms"] = mat_atoms

    # Match reference behavior: isovalue tracks probe radius with caps.
    percolation_isovalue = float(
        np.clip(
            np.min([pld / 2.0 - 0.25, 1.75]),
            0.1,
            None,
        )
    )

    return {
        "lcd": float(lcd),
        "pld": float(pld),
        "dfsd": float(dfsd),
        "percolation_history": percolation_history,
        "pld_locations": pld_locations,
        "rag": rag,
        "maxima": maxima,
        "maxima_radii": maxima_radii,
        "dgrid": dgrid,
        "regions": regions,
        "connections": connections_filtered,
        "isovalue": percolation_isovalue,
        "percolation_source": "cpu_dgrid" if force_cpu_dgrid else "segmentation_results",
    }


def _make_plot(results: Dict[str, Any], *, stylize: bool, projection: str, layout: str,
               framework_style: str, show_surface: bool, show_surface_pbc: bool,
               surface_value: float, surface_opacity: float, surface_pbc_opacity: float, surface_types_opacity: float,
               show_connections: bool, show_windows: bool = True, show_maxima_spheres: bool = False) -> go.Figure:
    options = {
        "plot_framework_style": framework_style,
        "plot_surface": show_surface,
        "plot_surface_pbc": show_surface_pbc,
        "plot_surface_isovalue": surface_value,
        "plot_surface_opacity": surface_opacity,
        "plot_surface_pbc_opacity": surface_pbc_opacity,
        "plot_surface_types_opacity": surface_types_opacity,
        "plot_connections": show_connections,
        "plot_windows": show_windows,
        "plot_maxima_show_spheres": show_maxima_spheres,
        "plot_projection": projection,
        "plot_layout": layout,
    }
    return show_mof(results, stylize=stylize, options=options)


def _hts_one_material(
    cif_path: Any,
    out_dir: Any,
    params: Dict[str, Any],
    options: Dict[str, Any],
) -> tuple:
    """
    Run segmentation and optional outputs for one CIF. Designed to run inside a Dask worker.
    Returns (material_name, output_folder_path_or_none, success, error_message_or_none).
    """
    import re
    import pickle as pkl
    from pathlib import Path as P

    out_dir = P(out_dir)
    if isinstance(cif_path, (str, P)):
        cif_path = P(cif_path)
        atoms = _read_atoms_from_cif(cif_path)
        material_name = cif_path.stem or "material"
    else:
        atoms = cif_path
        material_name = getattr(atoms, "info", {}).get("name", "material") or "material"

    if atoms is None:
        return (material_name, None, False, "Failed to load CIF or atoms is None.")

    def sanitize(s: str) -> str:
        return re.sub(r"[^\w\-.]", "_", s)[:120]

    try:
        from mgr.analyze import update_with_covalent_bonds
        atoms = update_with_covalent_bonds(atoms)
    except Exception:
        pass

    try:
        min_box = params.get("min_box_thickness")
        if min_box is not None and min_box > 0:
            cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
            if min(cell_lengths) < min_box:
                repetitions = mgr.get_repetitions_to_meet_cutoff(atoms, min_box)
                atoms = atoms * repetitions
    except Exception:
        pass

    try:
        spacing = float(params.get("spacing", 0.5))
        use_aabb = bool(params.get("use_aabb_method", False))
        block_size = int(params.get("block_size", 50000))

        if use_aabb:
            from mgr.compute import dgrid_from_atoms_cpu
            dgrid = dgrid_from_atoms_cpu(atoms, spacing=spacing, block_size=block_size).astype(np.float32)
        else:
            if __package__ and "mgrcli" in __package__:
                from mgrcli.cli import compute_dgrid_rust
            else:
                import sys
                _root = P(__file__).resolve().parents[2]
                if str(_root) not in sys.path:
                    sys.path.insert(0, str(_root))
                from python_cli.mgrcli.cli import compute_dgrid_rust
            dgrid = compute_dgrid_rust(atoms, spacing)

        mask_thickness = float(params.get("mask_thickness", 1.5))
        h_value = float(params.get("h_value", 0.4))
        min_distance = float(params.get("min_distance", 1.5))
        threshold_abs = float(params.get("threshold_abs", 1.5))
        extra_padding = float(params.get("extra_padding", 5.0))
        watershed_backend_w = str(params.get("watershed_backend", "rust_bucket"))

        if __package__ and "mgrcli" in __package__:
            from mgrcli.cli import python_do_periodic_segmentation_rust
        else:
            import sys
            _root = P(__file__).resolve().parents[2]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from python_cli.mgrcli.cli import python_do_periodic_segmentation_rust

        results = python_do_periodic_segmentation_rust(
            dgrid=dgrid,
            mat_atoms=atoms,
            mask_thickness=mask_thickness,
            h=h_value,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            return_padded_regions=False,
            spacing=spacing,
            minimum_window_separation=1.0,
            wall_thickness=0.05,
            extra_padding=extra_padding,
            compactness=0,
            min_region_size=0.001,
            disable_tqdm=True,
            compute_connections=True,
            return_dgrid=True,
            watershed_backend=watershed_backend_w,
        )
        results["ase_atoms"] = atoms
        results["dgrid"] = results.get("dgrid", dgrid)

        dgrid_r = results.get("dgrid")
        regions = results.get("regions")
        maxima_info = results.get("maxima_info")
        all_maxima = results.get("all_maxima", np.empty((0, 3)))
        if dgrid_r is not None and regions is not None and regions.max() > 0 and len(all_maxima) > 0:
            if maxima_info is not None and hasattr(maxima_info, "shape") and len(maxima_info.shape) == 2 and maxima_info.shape[1] >= 4:
                all_maxima_radii = dgrid_r[tuple(maxima_info[:, 1:4].T)]
            else:
                all_maxima_radii = dgrid_r[tuple(all_maxima.T)] if len(all_maxima) else np.empty(0)
            final_maxima = all_maxima[: int(regions.max())]
            results["maxima"] = final_maxima
            results["maxima_radii"] = all_maxima_radii[: len(final_maxima)]
        else:
            results["maxima"] = np.empty((0, 3), dtype=int)
            results["maxima_radii"] = np.empty(0, dtype=float)

        material_folder = out_dir / sanitize(material_name)
        material_folder.mkdir(parents=True, exist_ok=True)

        if options.get("save_npz", True):
            npz_bytes = results_data_to_npz(results)
            if npz_bytes:
                (material_folder / "results.npz").write_bytes(npz_bytes)

        if options.get("save_rag"):
            if __package__ and "mgrcli" in __package__:
                from mgrcli.cli import get_pbc_groups
            else:
                import sys
                _root = P(__file__).resolve().parents[2]
                if str(_root) not in sys.path:
                    sys.path.insert(0, str(_root))
                from python_cli.mgrcli.cli import get_pbc_groups
            connections = results.get("connections")
            maxima = results.get("maxima")
            maxima_radii = results.get("maxima_radii")
            unique_regions = results.get("unique_regions")
            maxima_info_r = results.get("maxima_info")
            if all(x is not None for x in (connections, maxima, maxima_radii, dgrid_r, unique_regions, maxima_info_r)):
                pbc_groups_dict = get_pbc_groups(unique_regions, maxima_info_r)
                pbc_groups = [v for k, v in pbc_groups_dict.items()]
                rag = mgr.rag_from_connections(connections, maxima)
                rag = mgr.add_pbc_to_rag(rag, pbc_groups)
                rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid_r.shape, atoms)
                with open(material_folder / "rag.pkl", "wb") as f:
                    pkl.dump(rag, f)

        if options.get("detect_pocket_types"):
            try:
                if __package__ and "mgrcli" in __package__:
                    from mgrcli.cli import get_pbc_groups
                else:
                    import sys
                    _root = P(__file__).resolve().parents[2]
                    if str(_root) not in sys.path:
                        sys.path.insert(0, str(_root))
                    from python_cli.mgrcli.cli import get_pbc_groups
                from skimage.measure import regionprops
                unique_regions = results.get("unique_regions")
                maxima_info = results.get("maxima_info")
                connections = results.get("connections")
                maxima = results.get("maxima")
                maxima_radii = results.get("maxima_radii")
                pbc_groups_dict = get_pbc_groups(unique_regions, maxima_info)
                pbc_groups = [v for k, v in pbc_groups_dict.items()]
                connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]
                rag = mgr.rag_from_connections(connections_trimmed, maxima)
                rag = mgr.add_pbc_to_rag(rag, pbc_groups)
                rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid_r.shape, atoms)
                props = regionprops(results["regions"])
                bins = np.linspace(0, 30, 50)
                rag = mgr.add_psd_to_rag_with_pbc(rag, props, pbc_groups, bins, atoms, dgrid_r.shape)
                js_threshold = float(options.get("js_threshold", 0.2))
                rag = mgr.add_clustering_to_rag_geometry_js(rag, props, pbc_groups, atoms, dgrid_r.shape, use_pbc=True, js_threshold=js_threshold)
                results["regions_cluster"] = mgr.clustering_grid_from_regions(results["regions"], rag, cluster_key="cluster_geom_pbc")
                results["pocket_types_detected"] = True
            except Exception:
                pass

        if options.get("save_html"):
            if __package__ and "mgrcli" in __package__:
                from mgrcli.plotting import show_mof
            else:
                import sys
                _root = P(__file__).resolve().parents[2]
                if str(_root) not in sys.path:
                    sys.path.insert(0, str(_root))
                from python_cli.mgrcli.plotting import show_mof
            viz_opts = options.get("viz_options") or {}
            stylize = options.get("stylize", False)
            fig = show_mof(results, stylize=stylize, options=viz_opts)
            fig.write_html(str(material_folder / "plot.html"))

        if options.get("do_pore_dia"):
            if __package__ and "mgrcli" in __package__:
                from mgrcli.cli import compute_pore_diameter_analysis, save_pore_diameter_csv
            else:
                import sys
                _root = P(__file__).resolve().parents[2]
                if str(_root) not in sys.path:
                    sys.path.insert(0, str(_root))
                from python_cli.mgrcli.cli import compute_pore_diameter_analysis, save_pore_diameter_csv
            try:
                pore_dia_results = compute_pore_diameter_analysis(
                    results["dgrid"], atoms, spacing=spacing, segmentation_results=results
                )
                if options.get("save_pore_dia_csv") and pore_dia_results:
                    save_pore_diameter_csv(pore_dia_results, material_folder / f"{material_name}_pore_dia.csv")
            except Exception:
                pass

        if options.get("do_psd"):
            if __package__ and "mgrcli" in __package__:
                from mgrcli.cli import compute_psd_analysis, save_psd_csv
            else:
                import sys
                _root = P(__file__).resolve().parents[2]
                if str(_root) not in sys.path:
                    sys.path.insert(0, str(_root))
                from python_cli.mgrcli.cli import compute_psd_analysis, save_psd_csv
            try:
                regions_pbc = results.get("regions_pbc")
                domain = str(options.get("psd_domain", "Occupiable only"))
                do_occupiable = domain in ("Occupiable only", "Both")
                do_accessible = domain in ("Accessible only", "Both")
                n_total = int(options.get("psd_num_samples_total", 10000))
                n_per_region = int(options.get("psd_num_samples_per_region", 1000))
                unif_bins = int(options.get("psd_uniform_num_bins", 200))
                pock_bins = int(options.get("psd_pocket_num_bins", 200))
                unif_max_d = float(options.get("psd_uniform_max_diameter", 50.0))
                pock_max_d = float(options.get("psd_pocket_max_diameter", 50.0))
                probe_psd = float(options.get("probe_radius_psd", 1.2))
                psd_results = {}
                if do_occupiable:
                    out = compute_psd_analysis(
                        results["dgrid"], atoms, regions_pbc=regions_pbc, spacing=spacing,
                        num_samples_total=n_total, num_samples_per_region=n_per_region,
                        uniform_max_diameter=unif_max_d, pocket_max_diameter=pock_max_d,
                        uniform_num_bins=unif_bins, pocket_num_bins=pock_bins,
                        probe_center_min_radius=probe_psd,
                    )
                    if out:
                        psd_results.update(out)
                if do_accessible and results.get("connections") is not None:
                    try:
                        percol = _compute_pld_analysis(
                            results,
                            atoms,
                            spacing,
                            mask_thickness,
                            h_value,
                            threshold_abs,
                            min_distance,
                            extra_padding,
                            force_cpu_dgrid=True,
                            watershed_backend=watershed_backend_w,
                        )
                        mod = dict(results)
                        mod.update(percol)
                        mod.setdefault("distance_grid", mod.get("dgrid"))
                        bins_unif = np.linspace(0, unif_max_d, unif_bins + 1)
                        bins_pock = np.linspace(0, pock_max_d, pock_bins + 1)
                        acc_uniform = mgr.compute_psd_global_accessible(mod, atoms, probe_radius=probe_psd, bins=bins_unif, samples_per_region=n_total)
                        acc_pocket = mgr.compute_psds_pocketwise_accessible(mod, atoms, probe_radius=probe_psd, num_samples_per_region=n_per_region, bins=bins_pock)
                        bc_unif = 0.5 * (bins_unif[:-1] + bins_unif[1:])
                        bc_pock = 0.5 * (bins_pock[:-1] + bins_pock[1:])
                        if do_occupiable and psd_results:
                            psd_results["psd_accessible_uniform"] = {"bins": bins_unif, "histogram": acc_uniform, "bin_centers": bc_unif}
                            psd_results["psd_accessible_pocket"] = {"bins": bins_pock, "bin_centers": bc_pock, "pocket_psds": acc_pocket}
                        else:
                            psd_results["psd_uniform"] = {"bins": bins_unif, "histogram": acc_uniform, "bin_centers": bc_unif, "num_samples": n_total}
                            pocket_psds_list = [acc_pocket[rid]["hist"] for rid in sorted(acc_pocket)]
                            psd_results["psd_pocket"] = {"bins": bins_pock, "bin_centers": bc_pock, "pocket_psds": pocket_psds_list, "num_regions": len(pocket_psds_list)}
                    except Exception:
                        pass
                if options.get("save_psd_csv") and psd_results:
                    save_psd_csv(psd_results, material_folder / f"{material_name}_psd", material_name=material_name)
                if options.get("save_psd_plot") and psd_results:
                    try:
                        import plotly.graph_objects as go
                        u = psd_results.get("psd_uniform")
                        if u and u.get("histogram") is not None:
                            hist = np.asarray(u["histogram"])
                            bins_arr = np.asarray(u.get("bins", np.linspace(0, 50, len(hist) + 1)))
                            centers = 0.5 * (bins_arr[:-1] + bins_arr[1:]) if len(bins_arr) > 1 else bins_arr[: len(hist)]
                            if len(centers) > len(hist):
                                centers = centers[: len(hist)]
                            fig_psd = go.Figure(data=[go.Scatter(x=centers, y=hist, mode="lines", name="PSD", line=dict(width=2))])
                            fig_psd.update_layout(
                                title=f"Pore Size Distribution with uniform sampling across box - {material_name}",
                                xaxis_title="Pore Diameter (Å)",
                                yaxis_title="Counts",
                                legend=dict(
                                    orientation="h",
                                    yanchor="top",
                                    y=-0.1,
                                    xanchor="center",
                                    x=0.5
                                ),
                                width=1000,
                                height=600,
                            )
                            fig_psd.write_html(str(material_folder / f"{material_name}_psd_plot.html"))
                    except Exception:
                        pass
            except Exception:
                pass

        if options.get("do_sa_vol"):
            if __package__ and "mgrcli" in __package__:
                from mgrcli.cli import save_surface_area_csv, save_void_volume_csv
            else:
                import sys
                _root = P(__file__).resolve().parents[2]
                if str(_root) not in sys.path:
                    sys.path.insert(0, str(_root))
                from python_cli.mgrcli.cli import save_surface_area_csv, save_void_volume_csv
            try:
                percolation_analysis = _compute_pld_analysis(
                    results,
                    atoms,
                    spacing,
                    mask_thickness,
                    h_value,
                    threshold_abs,
                    min_distance,
                    extra_padding,
                    force_cpu_dgrid=True,
                    watershed_backend=watershed_backend_w,
                )
                modified_results = dict(results)
                modified_results.update(percolation_analysis)
                if "distance_grid" not in modified_results:
                    modified_results["distance_grid"] = modified_results.get("dgrid")
                probe = float(options.get("probe_radius_sa_vol", 1.2))
                from mgr.compute import compute_surface_area_from_results, compute_void_volume_from_results
                sa_results = compute_surface_area_from_results(modified_results, atoms, probe)
                vol_results = compute_void_volume_from_results(modified_results, atoms, probe)
                if options.get("save_sa_vol_csv"):
                    if sa_results:
                        save_surface_area_csv(sa_results, material_folder / material_name)
                    if vol_results:
                        save_void_volume_csv(vol_results, material_folder / material_name, material_name=material_name)
                if options.get("save_sa_plot") and sa_results:
                    try:
                        import plotly.express as px
                        breakdown_data = []
                        for region_id, area in sa_results.get("by_region", {}).items():
                            breakdown_data.append({"Type": "Region (non-PBC)", "Category": "Segmented", "Label": f"region_{region_id}", "Surface Area (m²/g)": area})
                        for pocket_id, area in sa_results.get("by_pocket", {}).items():
                            breakdown_data.append({"Type": "Pocket (PBC)", "Category": "Grouped", "Label": str(pocket_id), "Surface Area (m²/g)": area})
                        for cluster_id, area in sa_results.get("by_percolated_cluster", {}).items():
                            breakdown_data.append({"Type": "Cluster (percolated)", "Category": "Accessible", "Label": str(cluster_id), "Surface Area (m²/g)": area})
                        if not breakdown_data:
                            total_sa = sa_results.get("total_surface_area", 0)
                            acc_sa = sa_results.get("accessible_surface_area", 0)
                            breakdown_data = [
                                {"Type": "Global", "Category": "Total", "Label": "Total", "Surface Area (m²/g)": total_sa},
                                {"Type": "Global", "Category": "Accessible", "Label": "Accessible", "Surface Area (m²/g)": acc_sa},
                                {"Type": "Global", "Category": "Inaccessible", "Label": "Inaccessible", "Surface Area (m²/g)": total_sa - acc_sa},
                            ]
                        sa_df = __import__("pandas").DataFrame(breakdown_data)
                        fig_sa = px.bar(sa_df, x="Label", y="Surface Area (m²/g)", color="Type", barmode="group", title="Surface Area Breakdown by Category")
                        fig_sa.write_html(str(material_folder / f"{material_name}_surface_area_plot.html"))
                    except Exception:
                        pass
                if options.get("save_vol_plot") and vol_results:
                    try:
                        import plotly.express as px
                        breakdown_data = []
                        for region_id, volume in vol_results.get("by_region", {}).items():
                            breakdown_data.append({"Type": "Region (non-PBC)", "Category": "Segmented", "Label": f"region_{region_id}", "Void Volume (cm³/g)": volume})
                        for pocket_id, volume in vol_results.get("by_pocket", {}).items():
                            breakdown_data.append({"Type": "Pocket (PBC)", "Category": "Grouped", "Label": str(pocket_id), "Void Volume (cm³/g)": volume})
                        for cluster_id, volume in vol_results.get("by_percolated_cluster", {}).items():
                            breakdown_data.append({"Type": "Cluster (percolated)", "Category": "Accessible", "Label": str(cluster_id), "Void Volume (cm³/g)": volume})
                        if not breakdown_data:
                            total_vol = vol_results.get("total_void_volume", 0)
                            acc_vol = vol_results.get("accessible_void_volume", 0)
                            breakdown_data = [
                                {"Type": "Global", "Category": "Total", "Label": "Total", "Void Volume (cm³/g)": total_vol},
                                {"Type": "Global", "Category": "Accessible", "Label": "Accessible", "Void Volume (cm³/g)": acc_vol},
                                {"Type": "Global", "Category": "Inaccessible", "Label": "Inaccessible", "Void Volume (cm³/g)": total_vol - acc_vol},
                            ]
                        vol_df = __import__("pandas").DataFrame(breakdown_data)
                        fig_vol = px.bar(vol_df, x="Label", y="Void Volume (cm³/g)", color="Type", barmode="group", title="Void Volume Breakdown by Category")
                        fig_vol.write_html(str(material_folder / f"{material_name}_void_volume_plot.html"))
                    except Exception:
                        pass
            except Exception:
                pass

        return (material_name, str(material_folder), True, None)
    except Exception as e:
        return (material_name, None, False, str(e))


def main() -> None:
    global STYLIZE_LIGHTING, STYLIZE_LIGHTPOSITION, dark_layout, white_layout
    if STYLIZE_LIGHTING is None or STYLIZE_LIGHTPOSITION is None or dark_layout is None or white_layout is None:
        _, STYLIZE_LIGHTING, STYLIZE_LIGHTPOSITION, dark_layout, white_layout, _, _, _ = _load_plotting_symbols()

    st.set_page_config(
        layout="wide",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )
    st.session_state.setdefault("ui_logs", [])
    mode_label = "spaces" if _is_spaces_deployment() else "local"
    st.sidebar.caption(f"Debug: deployment mode = `{mode_label}`")
    show_logs = st.sidebar.toggle("Show runtime logs", value=False, key="show_runtime_logs")
    col_reset, col_logs, col_refresh = st.sidebar.columns(3)
    with col_reset:
        if st.button("Clear/Reset", use_container_width=True, key="reset_app"):
            clear_framework_cache()
            next_gen = int(st.session_state.get("widget_reset_generation", 0)) + 1
            st.session_state.clear()
            st.session_state["widget_reset_generation"] = next_gen
            st.rerun()
    with col_logs:
        if st.button("Clear Logs", use_container_width=True, key="clear_logs"):
            st.session_state["ui_logs"] = []
            st.rerun()
    with col_refresh:
        if st.button("Refresh Logs", use_container_width=True, key="refresh_logs"):
            st.rerun()

    # Detect current Streamlit theme (runtime, not server config)
    # st.get_option returns server config; st._config.get_option returns runtime value
    try:
        _current_theme = st._config.get_option("theme.base") or "dark"
    except Exception:
        try:
            _current_theme = str(st.get_option("theme.base") or "dark").lower()
        except Exception:
            _current_theme = "dark"
    
    _browser_theme = _current_theme.lower() if _current_theme else "dark"
    st.session_state["browser_theme"] = _browser_theme

    # Header: outlined logo (transparent bg + black stroke) works on any theme.
    # Use base64-encoded logo embedded in _logos.py (avoids binary files in Docker/HF).
    _logo_src = None
    try:
        if __package__:
            from ._logos import MOFOGRAPHY_LOGO_OUTLINED as _logo_b64
        else:
            import sys
            _cli_root = Path(__file__).resolve().parent
            if str(_cli_root) not in sys.path:
                sys.path.insert(0, str(_cli_root))
            from _logos import MOFOGRAPHY_LOGO_OUTLINED as _logo_b64
        _logo_src = f"data:image/png;base64,{_logo_b64}"
    except Exception:
        _logo_src = None

    if _logo_src:
        st.markdown(
            f"""
            <div style="
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                padding: 12px;
                box-sizing: border-box;
            ">
                <img src="{_logo_src}" style="
                    width: 100%;
                    max-width: 800px;
                    height: auto;
                    display: block;
                "/>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Responsive ASCII logo: smaller font on narrow screens, horizontal scroll
        st.markdown(
            f"""
            <div style="width:100%;overflow-x:auto;">
                <pre style="
                    font-size: clamp(4px, 1.2vw, 12px);
                    line-height: 1.1;
                    margin: 0 auto;
                    width: fit-content;
                    min-width: 100%;
                    white-space: pre;
                ">{ASCII_LOGO}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
    build_id = os.environ.get("MGRCLI_BUILD", "1")
    app_version = f"{time.strftime('%Y.%m.%d', time.gmtime())}.{build_id}"
    st.markdown(
        (
            "<div style='line-height:1.2;'>"
            f"<span style='font-family:Menlo,Consolas,monospace;font-size:0.82rem;"
            "color:#2ecc71;background:rgba(46,204,113,0.10);padding:0.12rem 0.35rem;border-radius:0.35rem;'>"
            f"{app_version}</span></div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='font-size:0.92rem;opacity:0.9;'>Pore Graph Analysis and Visualization of Nanoporous Materials</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.80rem;opacity:0.85;'><em>(code and research paper coming soon!)</em></div>", unsafe_allow_html=True)
    st.caption("Experimental; bugs and freezes may occur. Thank you for testing 🙏.")
    if show_logs:
        log_text = "\n".join(st.session_state.get("ui_logs", []))
        _render_logs_panel(log_text)

    app_mode = st.radio(
        "Mode",
        ["Single material", "High-Throughput (HTS) (experimental)"],
        index=0,
        horizontal=True,
        key="app_mode",
    )

    if "HTS" in app_mode:
        _render_hts_mode()
        return

    st.markdown("### Single material")
    samples = _get_sample_cifs()

    sample_choice = None
    if samples:
        sample_choice = st.selectbox(
            "Or select a sample CIF from the repository",
            ["<None>"] + list(samples.keys()),
        )
        if sample_choice == "<None>":
            sample_choice = None

    st.markdown("""
    **Note:** If you upload a CIF file, it will be used instead of the selected sample material below.
    """)
    _upload_key = int(st.session_state.get("widget_reset_generation", 0))
    uploaded_file = st.file_uploader(
        "Upload CIF file", type=["cif"], key=f"single_cif_upload_{_upload_key}"
    )

    atoms = None
    cif_path: Optional[Path] = None
    selected_source_key: Optional[str] = None

    # Streamlit reruns on any widget change. Cache expensive CIF->bonds processing
    # in session state and only recompute when the source CIF changes.
    atoms_cache_key = "single_material_atoms_with_bonds"
    atoms_source_key_state = "single_material_atoms_source_key"
    atoms_cif_path_state = "single_material_atoms_cif_path"
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        selected_source_key = (
            f"upload::{getattr(uploaded_file, 'name', 'upload.cif')}::"
            f"{len(uploaded_bytes)}::{hashlib.sha1(uploaded_bytes).hexdigest()}"
        )

        if _is_spaces_deployment():
            uploaded_size = int(getattr(uploaded_file, "size", 0) or len(uploaded_bytes))
            if uploaded_size > SPACES_MAX_UPLOAD_BYTES:
                st.error(
                    f"Uploaded CIF is too large ({uploaded_size / (1024 * 1024):.1f} MB). "
                    f"Please keep uploads below {SPACES_MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB for the public demo."
                )
                return

        cache_hit = st.session_state.get(atoms_source_key_state) == selected_source_key
        if not cache_hit:
            previous_upload_path = st.session_state.get("uploaded_cif_temp_path")
            if previous_upload_path:
                try:
                    Path(previous_upload_path).unlink(missing_ok=True)
                except Exception:
                    pass
            cif_path = _safe_uploaded_cif_path(uploaded_file)
            st.session_state["uploaded_cif_temp_path"] = str(cif_path)
            atoms = _read_atoms_from_cif(cif_path)
            from mgr.analyze import update_with_covalent_bonds

            atoms = update_with_covalent_bonds(atoms)
            st.session_state[atoms_cache_key] = atoms
            st.session_state[atoms_source_key_state] = selected_source_key
            st.session_state[atoms_cif_path_state] = str(cif_path)
        else:
            atoms = st.session_state.get(atoms_cache_key)
            cached_cif_path = st.session_state.get(atoms_cif_path_state)
            if cached_cif_path:
                cif_path = Path(cached_cif_path)
    elif sample_choice:
        cif_path = samples[sample_choice]
        try:
            mtime_ns = cif_path.stat().st_mtime_ns
        except Exception:
            mtime_ns = 0
        selected_source_key = f"sample::{cif_path.resolve()}::{mtime_ns}"
        cache_hit = st.session_state.get(atoms_source_key_state) == selected_source_key
        if not cache_hit:
            atoms = _read_atoms_from_cif(cif_path)
            from mgr.analyze import update_with_covalent_bonds

            atoms = update_with_covalent_bonds(atoms)
            st.session_state[atoms_cache_key] = atoms
            st.session_state[atoms_source_key_state] = selected_source_key
            st.session_state[atoms_cif_path_state] = str(cif_path)
        else:
            atoms = st.session_state.get(atoms_cache_key)

    if atoms is None and st.session_state.get(atoms_source_key_state) == selected_source_key:
        atoms = st.session_state.get(atoms_cache_key)

    if atoms is None:
        st.info("Upload a CIF file or pick a sample to begin.")
        return
    if _is_spaces_deployment() and len(atoms) > SPACES_MAX_ATOMS:
        st.error(
            f"This structure has {len(atoms)} atoms, which is above the public-demo limit "
            f"of {SPACES_MAX_ATOMS} atoms. Please run larger systems locally."
        )
        return

    st.markdown("### 🔧 Workflow Parameters")
    st.info("After changing any parameter or material, please click 'Compute and Visualize' to update the results.")

    spacing = st.number_input(
        "Grid Spacing (Å)",
        value=0.5,
        min_value=0.1,
        step=0.1,
        help="Grid spacing in Angstroms. Smaller values increase resolution and runtime.",
    )
    if _is_spaces_deployment() and spacing < SPACES_MIN_SPACING:
        spacing = SPACES_MIN_SPACING
        st.info(
            f"Grid spacing is clamped to {SPACES_MIN_SPACING:.1f} A on the hosted demo "
            "to keep runtime manageable."
        )

    # Distance grid computation method selection
    use_aabb_method = st.checkbox(
        "Use Python AABB (CPU) method for distance grid",
        value=False,
        help="Use the CPU-based AABB (Axis-Aligned Bounding Box) method with Freud library instead of Rust exact method. "
             "This uses dask for parallelization and may be useful for very large systems or when Rust is unavailable."
    )

    block_size = 50000  # Default block size for AABB method
    if use_aabb_method:
        block_size = st.number_input(
            "AABB Block Size (number of grid points per worker)",
            value=50000,
            min_value=1000,
            max_value=200000,
            step=5000,
            help="Number of grid points processed by each dask worker. Larger values use more memory but may be faster. "
                 "Default is 50000. Adjust based on your system's available RAM."
        )

    # Minimum box thickness option
    min_box_thickness_enabled = st.checkbox(
        "Ensure minimum box thickness", 
        value=False,
        help="Create supercell to ensure minimum box dimensions for accurate analysis. Useful for small unit cells."
    )
    
    min_box_thickness = 6.0  # Default 6Å
    if min_box_thickness_enabled:
        min_box_thickness = st.number_input(
            "Minimum box thickness (Å)",
            value=6.0,
            min_value=1.0,
            max_value=50.0,
            step=0.5,
            help="Minimum dimension for the simulation box. Creates supercell if unit cell is smaller."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        mask_thickness = st.number_input(
            "Mask thickness (Å)",
            value=0.5,
            min_value=0.05,
            help="Points closer than this to the framework are excluded.",
        )
    with col2:
        h_value = st.number_input(
            "h (Å)",
            value=0.2,
            min_value=0.05,
            help="Minimum separation between neighbouring maxima.",
        )
    with col3:
        surface_value = st.number_input(
            "Surface value (Å)",
            value=1.0,
            min_value=0.1,
            help="Isosurface value for visualization.",
        )

    # Additional segmentation parameters
    with st.expander("Advanced Segmentation Parameters", expanded=False):
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        with col_adv1:
            min_distance = st.number_input(
                "Min distance between maxima (Å)",
                value=1.5,
                min_value=0.1,
                step=0.1,
                help="Minimum distance required between distinct local maxima.",
            )
        with col_adv2:
            threshold_abs = st.number_input(
                "Threshold absolute (Å)",
                value=1.5,
                min_value=0.1,
                step=0.1,
                help="Absolute threshold for maxima detection.",
            )
        with col_adv3:
            extra_padding = st.number_input(
                "Extra padding (Å)",
                value=5.0,
                min_value=0.0,
                step=0.5,
                help="Extra padding around the unit cell for periodic boundary conditions.",
            )
        if _is_spaces_deployment() and extra_padding > SPACES_MAX_EXTRA_PADDING:
            extra_padding = SPACES_MAX_EXTRA_PADDING
            st.info(
                f"Extra padding is clamped to {SPACES_MAX_EXTRA_PADDING:.1f} A on the hosted demo."
            )
        watershed_backend = _watershed_backend_widgets("visualization")

    st.markdown("### 🎨 Visualization Options")
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        viz_backend = st.radio(
            "3D Rendering Backend",
            ["Plotly", "Three.js"],
            index=0,
            horizontal=True,
            help="Three.js: faster WebGL, SSAO, dynamic controls. Plotly: standard, offline-capable.",
        )
        percolation_backend = st.radio(
            "Percolation 3D Backend",
            ["Plotly", "Three.js"],
            index=0,
            horizontal=True,
            help="Three.js: faster WebGL percolation cluster rendering.",
        )
        mesh_framework = st.checkbox("Show framework as mesh", value=(viz_backend == "Three.js"))
        if mesh_framework:
            atom_scale = st.slider(
                "Atom scale",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.05,
                help="Scale factor for atom sphere radii.",
            )
            bond_radius = st.slider(
                "Bond radius",
                min_value=0.05,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Radius of bond cylinders.",
            )
        else:
            atom_scale = 0.5
            bond_radius = 0.25
        if viz_backend == "Plotly":
            simple_framework = st.checkbox("Show simple framework scatter", value=True)
            if mesh_framework and simple_framework:
                st.warning("Select either mesh or simple framework.")
        else:
            simple_framework = False
        projection_choice = st.selectbox("Projection", ["perspective", "orthographic"], index=0)
        stylize = st.checkbox("Stylize plot", value=True, disabled=(projection_choice == "orthographic"))
        if projection_choice == "orthographic":
            stylize = False
        # Keep viz background in sync with the current theme.
        # If the user toggled the Streamlit theme (⚙️ menu), re-default accordingly.
        _last_viz_theme = st.session_state.get("viz_background_theme")
        if _last_viz_theme != _browser_theme:
            st.session_state["viz_background"] = "white" if _browser_theme == "light" else "dark"
            st.session_state["viz_background_theme"] = _browser_theme
        layout_choice = st.selectbox(
            "Background",
            ["dark", "white"],
            index=0 if st.session_state.get("viz_background", "dark") == "dark" else 1,
            key="viz_background",
        )
        viz_plot_layout = str(layout_choice or st.session_state.get("viz_background", "dark")).strip().lower()
        if viz_plot_layout not in ("dark", "white"):
            viz_plot_layout = "white" if _browser_theme == "light" else "dark"
        st.caption(
            "Default matches the server theme. If you toggled the app theme via the ⚙️ menu, "
            "select the matching background here for best contrast."
        )
    with col_vis2:
        show_surface = st.checkbox(f"Show pore surface at {surface_value:.2f} Å", value=True)
        show_surface_pbc = st.checkbox("Show periodic pore surface", value=True)
        show_connections = st.checkbox("Show pore graph", value=True)
        show_windows = st.checkbox("Show windows (internal)", value=True)
        show_maxima_spheres = st.checkbox("Show maxima spheres", value=False)
        detect_pocket_types = st.checkbox("Detect Pocket Types", value=False)

        surface_opacity = st.slider("Surface Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        surface_pbc_opacity = st.slider("Periodic Surface Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        surface_types_opacity = st.slider("Pocket Types Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

        js_threshold = 0.2
        if detect_pocket_types:
            js_threshold = st.number_input(
                "Jensen-Shannon Threshold",
                min_value=0.01,
                max_value=1.0,
                value=0.2,
                step=0.01,
                help="Threshold for agglomerative clustering based on Jensen-Shannon distance between cavity size distributions"
            )



    results_key = f"{getattr(cif_path, 'name', 'uploaded')}::{spacing:.3f}::aabb={use_aabb_method}::bs={block_size}"
    compute_clicked = st.button("Compute and Visualize")

    if compute_clicked:
        if not _spaces_allow_compute_now():
            return
        lock_acquired = True
        if _is_spaces_deployment():
            lock_acquired = _SPACES_COMPUTE_LOCK.acquire(blocking=False)
            if not lock_acquired:
                estimated = int(st.session_state.get("spaces_last_compute_duration_sec", 45))
                estimated = max(10, min(180, estimated))
                st.error(f"Server busy. Estimated wait: ~{estimated}s.")
                st.caption("Another user is currently running a compute job on this demo instance.")
                return
            st.session_state["_compute_spaces_lock"] = True
        st.session_state["_compute_started_at"] = time.time()
        _min_box = min_box_thickness
        if _is_spaces_deployment() and _min_box > SPACES_MAX_MIN_BOX_THICKNESS:
            _min_box = SPACES_MAX_MIN_BOX_THICKNESS
            st.info(
                f"Minimum box thickness is clamped to {SPACES_MAX_MIN_BOX_THICKNESS:.1f} A on the hosted demo."
            )
        _start_compute_thread(
            atoms,
            spacing=spacing,
            mask_thickness=mask_thickness,
            h_value=h_value,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            extra_padding=extra_padding,
            watershed_backend=watershed_backend,
            use_aabb_method=use_aabb_method,
            block_size=block_size,
            min_box_thickness_enabled=min_box_thickness_enabled,
            min_box_thickness=_min_box,
        )
        st.rerun()

    if st.session_state.get("_compute_progress") is not None:
        progress = st.session_state["_compute_progress"]
        if not progress.done:
            _render_progress_ui(progress)
            time.sleep(0.6)
            st.rerun()

        if progress.error:
            st.error(f"Compute failed: {progress.error}")
        else:
            results = progress.result
            working_atoms = results.get("ase_atoms", atoms)
            if min_box_thickness_enabled:
                try:
                    min_current_length = min(atoms.get_cell_lengths_and_angles()[:3])
                    if min_current_length < min_box_thickness:
                        repetitions = mgr.get_repetitions_to_meet_cutoff(atoms, min_box_thickness)
                        if hasattr(repetitions, "__len__") and len(repetitions) >= 3:
                            rep_str = f"{repetitions[0]}x{repetitions[1]}x{repetitions[2]}"
                        else:
                            rep_str = f"{repetitions}x{repetitions}x{repetitions}"
                        st.info(f"Creating {rep_str} supercell to meet minimum box thickness of {min_box_thickness:.1f} A")
                except Exception:
                    pass

            try:
                dgrid_r = results.get("dgrid")
                regions = results.get("regions")
                maxima_info = results.get("maxima_info")
                all_maxima = results.get("all_maxima")
                if dgrid_r is not None and regions is not None and maxima_info is not None and all_maxima is not None and regions.max() > 0:
                    all_maxima_radii = dgrid_r[tuple(maxima_info[:, 1:4].T)]
                    final_maxima = all_maxima[:regions.max()]
                    final_maxima_radii = all_maxima_radii[:regions.max()]
                    results["maxima"] = final_maxima
                    results["maxima_radii"] = final_maxima_radii
                else:
                    results["maxima"] = np.empty((0, 3), dtype=int)
                    results["maxima_radii"] = np.empty(0, dtype=float)
            except Exception as e:
                st.warning(f"Could not derive final maxima/radii: {e}. RAG download may fail.")

            if detect_pocket_types:
                from skimage.measure import regionprops
                import plotly.express as px

                dgrid_r = results["dgrid"]
                regions = results["regions"]
                regions_pbc = results["regions_pbc"]
                maxima_info = results["maxima_info"]
                all_maxima = results["all_maxima"]
                all_maxima_radii = dgrid_r[tuple(maxima_info[:, 1:4].T)]
                connections = results["connections"]
                unique_regions = results["unique_regions"]
                mat_atoms_r = results["ase_atoms"]

                pbc_groups_dict = get_pbc_groups(unique_regions, maxima_info)
                pbc_groups = [v for k, v in pbc_groups_dict.items()]

                maxima_r = all_maxima[:regions.max()]
                maxima_radii_r = all_maxima_radii[:regions.max()]
                connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]

                try:
                    if len(maxima_r) > 0:
                        rag = mgr.rag_from_connections(connections_trimmed, maxima_r)
                        rag = mgr.add_pbc_to_rag(rag, pbc_groups)
                        rag = mgr.add_maxima_to_rag(rag, maxima_r, maxima_radii_r, dgrid_r.shape, mat_atoms_r)

                        props = regionprops(regions)
                        bins = np.linspace(0, 30, 50)
                        rag = mgr.add_psd_to_rag_with_pbc(rag, props, pbc_groups, bins, mat_atoms_r, dgrid_r.shape)
                        rag = mgr.add_clustering_to_rag_geometry_js(rag, props, pbc_groups, mat_atoms_r, dgrid_r.shape, use_pbc=True, js_threshold=js_threshold)

                        regions_cluster = mgr.clustering_grid_from_regions(regions, rag, cluster_key="cluster_geom_pbc")
                        results["regions_cluster"] = regions_cluster
                        results["rag_clustered"] = rag
                        results["pocket_types_detected"] = True
                        results["js_threshold_used"] = js_threshold

                        num_clusters = len(np.unique(regions_cluster)) - 1
                        st.info(f"Found {num_clusters} different pocket types using JS threshold: {js_threshold}")
                    else:
                        results["pocket_types_detected"] = False
                except Exception as e:
                    st.warning(f"Pocket type detection failed: {e}")
                    results["pocket_types_detected"] = False

            st.session_state["latest_results"] = results
            st.session_state["latest_metadata"] = {
                "spacing": float(spacing),
                "mask_thickness": float(mask_thickness),
                "surface_value": float(surface_value),
                "h_value": float(h_value),
                "threshold_abs": float(threshold_abs),
                "watershed_backend": str(watershed_backend),
            }
            st.session_state["analysis_key"] = results_key
            pld_cache = st.session_state.setdefault("pld_cache", {})
            pld_cache.pop(results_key, None)

        elapsed = time.time() - progress.start_time
        st.session_state["_compute_log"] = {
            "lines": list(progress.lines),
            "elapsed": elapsed,
            "final_phase": progress.phase,
        }

        if _is_spaces_deployment() and st.session_state.get("_compute_spaces_lock"):
            try:
                _SPACES_COMPUTE_LOCK.release()
            except RuntimeError:
                pass
            st.session_state["spaces_last_compute_duration_sec"] = max(
                1.0, time.time() - st.session_state.get("_compute_started_at", time.time())
            )
        del st.session_state["_compute_progress"]
        del st.session_state["_compute_thread"]
        st.session_state.pop("_compute_spaces_lock", None)
        st.session_state.pop("_compute_started_at", None)

    results_data = st.session_state.get("latest_results")
    metadata = st.session_state.get("latest_metadata", {})
    analysis_key = st.session_state.get("analysis_key")

    if results_data is not None and st.session_state.get("_compute_log"):
        log = st.session_state["_compute_log"]
        with st.expander(
            f"Compute log ({log['elapsed']:.1f}s) — {log['final_phase']}",
            expanded=False,
        ):
            st.caption(
                "Captured logger output and phase transitions during the last compute run."
                " Scroll to review timing and diagnostics."
            )
            st.code("\n".join(log["lines"]) if log["lines"] else "(no log output)", language=None)
        del st.session_state["_compute_log"]

    if results_data is not None and not mesh_framework:
        framework_identity = str(
            st.session_state.get("single_material_atoms_source_key")
            or getattr(cif_path, "name", "framework")
        )
        _queue_mesh_prewarm(
            results_data,
            framework_identity,
            stylize=bool(stylize),
            show_rings=True,
        )

    visualization_tab, percolation_tab, psd_tab, sa_vol_tab = st.tabs([
        "Visualization", "PLD & Percolation", "Pore Size Distribution", "Surface Areas & Void Volumes",
    ])

    with visualization_tab:
        if results_data is None:
            st.info("Run the computation to visualize the structure.")
        else:
            material_name = getattr(cif_path, 'stem', 'Material')
            st.markdown(f"## Material: **{material_name}**")
            framework_style = "scatter" if simple_framework and not mesh_framework else "mesh"
            main_plot_status = st.empty()

            plot_log: List[str] = []
            _plot_start = time.time()

            def _update_plot_status(step, total, label):
                elapsed = time.time() - _plot_start
                plot_log.append(f"[{elapsed:.2f}s] Step {step}/{total} — {label}")
                main_plot_status.markdown(
                    _step_status(step, total, label),
                    unsafe_allow_html=True,
                )

            viz_opts = dict(
                plot_framework=(simple_framework or mesh_framework),
                plot_framework_style=framework_style,
                plot_framework_show_rings=True,
                plot_framework_mesh_scale=float(atom_scale),
                plot_framework_bond_radius=float(bond_radius),
                plot_surface=show_surface,
                plot_surface_pbc=show_surface_pbc,
                plot_surface_isovalue=surface_value,
                plot_surface_opacity=surface_opacity,
                plot_surface_pbc_opacity=surface_pbc_opacity,
                plot_surface_types_opacity=surface_types_opacity,
                plot_connections=show_connections,
                plot_windows=show_windows,
                plot_maxima_show_spheres=show_maxima_spheres,
                plot_projection=projection_choice,
                plot_layout=viz_plot_layout,
            )

            md_plot = metadata if isinstance(metadata, dict) else {}
            _settings_slug = _viz_settings_title_suffix(
                spacing=float(md_plot.get("spacing", spacing)),
                h_value=float(md_plot.get("h_value", h_value)),
                threshold_abs=float(md_plot.get("threshold_abs", threshold_abs)),
                mask_thickness=float(md_plot.get("mask_thickness", mask_thickness)),
            )

            if viz_backend == "Three.js":
                from mgr.threejs_plotting import show_mof_threejs
                html_str = show_mof_threejs(
                    results_data,
                    stylize=stylize,
                    options=viz_opts,
                    on_step=_update_plot_status,
                )
                if plot_log:
                    _plot_total = time.time() - _plot_start
                    with st.expander(
                        f"Plotting log ({_plot_total:.1f}s, {len(plot_log)} stages)",
                        expanded=False,
                    ):
                        st.caption("Plot trace rendering stages and cumulative timing.")
                        st.code("\n".join(plot_log), language=None)
                components.html(
                    html_str,
                    height=1000,
                )
                st.info(
                    "Three.js viewer: drag to rotate, scroll to zoom. "
                    "Use the HUD controls for opacity, lighting, and axis views."
                )
            else:
                fig = show_mof(
                    results_data,
                    stylize=stylize,
                    options=viz_opts,
                    on_step=_update_plot_status,
                )
                # Apply selected layout explicitly (includes scene.bgcolor; template alone can miss 3D canvas).
                fig.update_layout(**dict(white_layout if viz_plot_layout == "white" else dark_layout))
                _main_title = f"{material_name} | {_settings_slug}"
                if viz_plot_layout == "white":
                    fig.update_layout(
                        title=dict(
                            text=_main_title,
                            font=dict(color="#1f1f1f"),
                            y=0.95,
                        ),
                        font=dict(color="#1f1f1f"),
                        legend_font=dict(color="#1f1f1f"),
                    )
                else:
                    fig.update_layout(
                        title=dict(
                            text=_main_title,
                            font=dict(color="#ffffff"),
                            y=0.95,
                        ),
                        font=dict(color="#ffffff"),
                        legend_font=dict(color="#ffffff"),
                    )

                _plotly_cfg = {
                    'responsive': True,
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'doubleClick': 'reset',
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{material_name}_plot',
                    },
                }

                if plot_log:
                    _plot_total = time.time() - _plot_start
                    with st.expander(
                        f"Plotting log ({_plot_total:.1f}s, {len(plot_log)} stages)",
                        expanded=False,
                    ):
                        st.caption(
                            "Plot trace rendering stages and cumulative timing."
                        )
                        st.code("\n".join(plot_log), language=None)

                st.plotly_chart(fig, use_container_width=True, config=_plotly_cfg)
                # Inject CSS to ensure touch gestures reach Plotly inside Streamlit iframes.
                st.markdown(
                    """
                    <style>
                    .stPlotlyChart,
                    .stPlotlyChart > div,
                    .stPlotlyChart iframe,
                    .js-plotly-plot,
                    .js-plotly-plot .plotly,
                    .js-plotly-plot .plotly .scene {
                        touch-action: none !important;
                        -webkit-touch-callout: none !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.info(
                    "Hover over the plot to reveal the toolbar (top-right corner). "
                    "The camera icon downloads a screenshot of the current rotated/zoomed view."
                )

            # Camera preservation + screenshot fix + mobile viewport fix.
            # Uses localStorage keyed by material name so each structure keeps its own view.
            _cam_key = html.escape(material_name or "default")
            components.html(
                f"""
                <script>
                (function() {{
                    // Allow pinch-zoom on mobile by making the page scalable.
                    try {{
                        var parentDoc = window.parent.document;
                        var meta = parentDoc.querySelector('meta[name="viewport"]');
                        if (meta) {{
                            meta.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes');
                        }} else {{
                            meta = parentDoc.createElement('meta');
                            meta.setAttribute('name', 'viewport');
                            meta.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes');
                            parentDoc.head.appendChild(meta);
                        }}
                    }} catch(e) {{}}

                    var CAM_KEY = 'mofography_camera_{_cam_key}';
                    var POLL_MS = 200;
                    var MAX_TRIES = 50;
                    var tries = 0;

                    function getPlotlyWindows() {{
                        var wins = [];
                        var iframes = document.querySelectorAll('iframe');
                        for (var i = 0; i < iframes.length; i++) {{
                            try {{
                                var cw = iframes[i].contentWindow;
                                if (cw && cw.Plotly) {{
                                    wins.push(cw);
                                }}
                            }} catch(e) {{}}
                        }}
                        if (window.Plotly) {{
                            wins.push(window);
                        }}
                        return wins;
                    }}

                    function findPlotlyDivs(win) {{
                        if (!win || !win.document) return [];
                        return win.document.querySelectorAll('.js-plotly-plot');
                    }}

                    function saveCamera(plotDiv) {{
                        try {{
                            var scene = plotDiv._fullLayout.scene;
                            if (scene && scene.camera) {{
                                localStorage.setItem(CAM_KEY, JSON.stringify(scene.camera));
                            }}
                        }} catch(e) {{}}
                    }}

                    function restoreCamera(plotDiv, win) {{
                        try {{
                            var saved = localStorage.getItem(CAM_KEY);
                            if (!saved) return;
                            var cam = JSON.parse(saved);
                            if (win.Plotly && win.Plotly.relayout) {{
                                win.Plotly.relayout(plotDiv, {{'scene.camera': cam}});
                            }}
                        }} catch(e) {{}}
                    }}

                    function patchDownloadImage(win) {{
                        if (!win.Plotly || win.Plotly._mofDownloadPatched) return;
                        win.Plotly._mofDownloadPatched = true;
                        var orig = win.Plotly.downloadImage;
                        win.Plotly.downloadImage = function(gd, opts) {{
                            try {{
                                var scene = gd._fullLayout.scene;
                                if (scene && scene.camera && win.Plotly.relayout) {{
                                    win.Plotly.relayout(gd, {{'scene.camera': scene.camera}}).then(function() {{
                                        orig(gd, opts);
                                    }}).catch(function() {{
                                        orig(gd, opts);
                                    }});
                                }} else {{
                                    orig(gd, opts);
                                }}
                            }} catch(e) {{
                                orig(gd, opts);
                            }}
                        }};
                    }}

                    function attachListeners() {{
                        var wins = getPlotlyWindows();
                        var attached = false;
                        for (var w = 0; w < wins.length; w++) {{
                            patchDownloadImage(wins[w]);
                            var divs = findPlotlyDivs(wins[w]);
                            for (var d = 0; d < divs.length; d++) {{
                                var div = divs[d];
                                if (div._mofCameraAttached) continue;
                                div._mofCameraAttached = true;
                                attached = true;

                                restoreCamera(div, wins[w]);

                                div.on('plotly_relayout', function() {{
                                    saveCamera(div);
                                }});
                            }}
                        }}
                        return attached;
                    }}

                    var timer = setInterval(function() {{
                        if (attachListeners() || tries > MAX_TRIES) {{
                            clearInterval(timer);
                        }}
                        tries++;
                    }}, POLL_MS);
                }})();
                </script>
                """,
                height=0,
            )

            # --- Download buttons in a row ---
            _html_stem = f"{material_name}_{_settings_slug}".replace(" ", "_").replace("/", "_")
            col_html, col_npz, col_rag = st.columns(3)
            if viz_backend == "Three.js":
                # Three.js: write the already-generated HTML string
                html_path = Path("interactive_plot.html")
                with open(html_path, "w") as fh:
                    fh.write(html_str)
                with open(html_path, "rb") as file_handle:
                    with col_html:
                        st.download_button(
                            label="Download interactive HTML",
                            data=file_handle,
                            file_name=f"{_html_stem}.html",
                            mime="text/html",
                        )
            else:
                # Plotly
                html_path = Path("interactive_plot.html")
                fig.write_html(html_path)
                with open(html_path, "rb") as file_handle:
                    with col_html:
                        st.download_button(
                            label="Download interactive HTML",
                            data=file_handle,
                            file_name=f"{_html_stem}.html",
                            mime="text/html",
                        )
            if results_data:
                npz_bytes = results_data_to_npz(results_data)
                if npz_bytes:
                    with col_npz:
                        st.download_button(
                            label="Download results NPZ",
                            data=npz_bytes,
                            file_name=f"{getattr(cif_path, 'stem', 'analysis')}_results.npz",
                            mime="application/octet-stream",
                        )

                # --- Download RAG (Pickle) Button ---
                import pickle
                rag_obj = None
                try:
                    # Gather required fields for CLI-style RAG (no clustering, no PSD)
                    connections = results_data.get("connections")
                    maxima = results_data.get("maxima")
                    maxima_radii = results_data.get("maxima_radii")
                    dgrid = results_data.get("dgrid")
                    regions = results_data.get("regions")
                    mat_atoms = results_data.get("ase_atoms")
                    unique_regions = results_data.get("unique_regions")
                    maxima_info = results_data.get("maxima_info")
                    required_fields = [
                        ("connections", connections),
                        ("maxima", maxima),
                        ("maxima_radii", maxima_radii),
                        ("dgrid", dgrid),
                        ("regions", regions),
                        ("ase_atoms", mat_atoms),
                        ("unique_regions", unique_regions),
                        ("maxima_info", maxima_info),
                    ]
                    missing = [name for name, val in required_fields if val is None]
                    if missing:
                        raise ValueError(f"Missing fields: {missing}")
                    pbc_groups_dict = get_pbc_groups(unique_regions, maxima_info)
                    pbc_groups = [v for k, v in pbc_groups_dict.items()]
                    # Build RAG (steps 1-3, no clustering, no PSD)
                    rag = mgr.rag_from_connections(connections, maxima)
                    rag = mgr.add_pbc_to_rag(rag, pbc_groups)
                    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid.shape, mat_atoms)
                    rag_obj = rag
                except Exception as e:
                    st.warning(f"Could not reconstruct RAG for download: {e}")
                    rag_obj = None

                if rag_obj is not None:
                    rag_bytes = pickle.dumps(rag_obj)
                    with col_rag:
                        st.download_button(
                            label="Download RAG (.pkl)",
                            data=rag_bytes,
                            file_name=f"{getattr(cif_path, 'stem', 'analysis')}_rag.pkl",
                            mime="application/octet-stream",
                        )
                else:
                    with col_rag:
                        st.info("RAG is not available for download. Ensure segmentation is complete.")

            # --- Tips/help below plot and buttons ---
            st.markdown("""
            ### Tips
            - Toggle traces via the legend to focus on specific features.
            - Use simple framework for a faster plot, or turn it off if not needed.
            - Performance mode speeds up surface generation on large unit cells.
            - Orthographic projection pairs best with stylize disabled.
            """)
            if _is_spaces_deployment():
                st.markdown(
                    f"""
### Hosted demo (Hugging Face Spaces)
- **Cooldown:** allow about **{int(SPACES_MIN_SECONDS_BETWEEN_RUNS)} seconds** between compute runs; the public Space shares CPU with other visitors.
- **Uploads:** CIFs up to **{SPACES_MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB**; structures larger than **{SPACES_MAX_ATOMS:,}** atoms are not supported here.
- **Parameters:** grid spacing is at least **{SPACES_MIN_SPACING:g} Å** and some padding / box settings are capped compared to running locally.
- For the full feature set and best performance, run **`mgrcli interactive`** on your machine.
"""
                )

            main_plot_status.markdown("<div style='padding:6px;background:#06310c;color:#6FE37F;text-align:center;font-family:monospace;'>✅ Main visualization complete — all traces rendered</div>", unsafe_allow_html=True)

    with percolation_tab:
        if results_data is None:
            st.info("Run the computation to enable PLD and percolation analysis.")
        else:
            material_name = getattr(cif_path, 'stem', 'Material')
            st.markdown(f"## Material: **{material_name}**")
            pld_cache = st.session_state.setdefault("pld_cache", {})
            effective_key = analysis_key or results_key
            compute_pld = st.button("Compute PLD & percolation metrics", key="compute_pld")
            if compute_pld:
                with st.spinner("Computing PLD and percolation history..."):
                    try:
                        pld_cache[effective_key] = _compute_pld_analysis(
                            results_data,
                            atoms,
                            spacing=float(metadata.get("spacing", spacing)),
                            mask_thickness=float(metadata.get("mask_thickness", mask_thickness)),
                            h_value=float(h_value),
                            threshold_abs=float(threshold_abs),
                            min_distance=float(min_distance),
                            extra_padding=float(extra_padding),
                            watershed_backend=str(
                                metadata.get("watershed_backend", "rust_bucket")
                            ),
                        )
                    except Exception as exc:
                        st.error(f"Percolation analysis failed: {exc}")

            analysis = pld_cache.get(effective_key)
            if not analysis:
                st.info("Click the button above to derive PLD, DFSD, and percolation visuals.")
            else:
                metrics_cols = st.columns(3)
                metrics_cols[0].metric("LCD (Å)", f"{analysis['lcd']:.2f}")
                metrics_cols[1].metric("PLD (Å)", f"{analysis['pld']:.2f}")
                metrics_cols[2].metric("DFSD (Å)", f"{analysis['dfsd']:.2f}")

                # Layout selection for percolation plots
                # Re-default when the detected theme changes so the plot matches the app theme.
                _last_perc_theme = st.session_state.get("percolation_layout_theme")
                if _last_perc_theme != _browser_theme:
                    st.session_state["percolation_layout"] = "white" if _browser_theme == "light" else "dark"
                    st.session_state["percolation_layout_theme"] = _browser_theme
                _perc_default = st.session_state.get("percolation_layout", "dark")
                percolation_layout_choice = st.selectbox(
                    "Background",
                    ["dark", "white"],
                    index=0 if _perc_default == "dark" else 1,
                    key="percolation_layout",
                )
                history = analysis["percolation_history"]
                if not history:
                    st.warning("No percolating clusters detected for the computed grid.")
                else:
                    history_rows = []
                    for radius, percolation_components in history:
                        axis_map = {0: "Along a", 1: "Along b", 2: "Along c", "x": "Along a", "y": "Along b", "z": "Along c"}
                        axis_counts = {"Along a": 0, "Along b": 0, "Along c": 0}
                        for axis_key, comps in percolation_components.items():
                            label = axis_map.get(axis_key)
                            if label is not None:
                                axis_counts[label] = len(comps or [])
                        history_rows.append(
                            {
                                "Probe radius / threshold (Å)": round(float(radius), 3),
                                "Percolated clusters (a)": axis_counts["Along a"],
                                "Percolated clusters (b)": axis_counts["Along b"],
                                "Percolated clusters (c)": axis_counts["Along c"],
                            }
                        )
                    st.markdown(
                        "Percolation history: probe radius (threshold) and **non-unique** percolated clusters "
                        "along each lattice direction."
                    )
                    st.dataframe(history_rows, use_container_width=True)

                    radius_options = sorted({float(entry[0]) for entry in history})
                    min_radius = min(radius_options)
                    max_radius = max(radius_options)
                    prev_radius = st.session_state.get("percolation_radius", max_radius)
                    if "percolation_radius_value" not in st.session_state:
                        st.session_state["percolation_radius_value"] = float(prev_radius)
                    # Clamp to available range when material changes.
                    if st.session_state["percolation_radius_value"] < min_radius:
                        st.session_state["percolation_radius_value"] = float(min_radius)
                    if st.session_state["percolation_radius_value"] > max_radius:
                        st.session_state["percolation_radius_value"] = float(max_radius)

                    desired_radius = float(st.session_state["percolation_radius_value"])
                    selected_radius = min(radius_options, key=lambda r: abs(r - desired_radius))
                    st.session_state["percolation_radius"] = selected_radius
                    st.session_state["percolation_radius_value"] = float(selected_radius)
                    st.session_state["percolation_radius_slider"] = float(selected_radius)
                    st.session_state["percolation_radius_input"] = float(selected_radius)

                    def _sync_radius_from_slider() -> None:
                        st.session_state["percolation_radius_value"] = st.session_state["percolation_radius_slider"]

                    def _sync_radius_from_input() -> None:
                        st.session_state["percolation_radius_value"] = st.session_state["percolation_radius_input"]

                    coarse_col, exact_col = st.columns(2)
                    with coarse_col:
                        st.slider(
                            "Probe radius (Å) coarse",
                            min_value=float(min_radius),
                            max_value=float(max_radius),
                            value=float(st.session_state["percolation_radius_value"]),
                            step=0.5,
                            format="%.2f",
                            key="percolation_radius_slider",
                            on_change=_sync_radius_from_slider,
                        )
                    with exact_col:
                        st.number_input(
                            "Probe radius (Å) exact",
                            min_value=float(min_radius),
                            max_value=float(max_radius),
                            value=float(st.session_state["percolation_radius_value"]),
                            step=0.05,
                            format="%.2f",
                            key="percolation_radius_input",
                            on_change=_sync_radius_from_input,
                        )

                    if abs(selected_radius - desired_radius) > 1e-6:
                        st.caption(f"Using nearest available radius: {selected_radius:.2f} Å")

                    boundary_col, width_col = st.columns(2)
                    with boundary_col:
                        boundary_line_color = st.color_picker(
                            "Boundary color",
                            value="#000000",
                            key="percolation_boundary_color",
                        )
                    with width_col:
                        boundary_line_width = st.slider(
                            "Boundary width",
                            min_value=1,
                            max_value=6,
                            value=6,
                            step=1,
                            key="percolation_boundary_width",
                        )

                    st.markdown(
                        f"Selected probe radius: **{selected_radius:.2f} Å** (diameter {2*selected_radius:.2f} Å)"
                    )
                    st.caption(
                        "Cluster pore surfaces use the same **Surface value (Å)** as the Visualization section "
                        "(main pane pore isosurface). The controls here only pick the percolation probe radius / "
                        "which cluster set to draw."
                    )

                    plot_status = st.empty()

                    if percolation_backend == "Three.js":
                        plot_status.markdown(
                            _step_status(1, 1, "Building Three.js percolation scene"),
                            unsafe_allow_html=True,
                        )
                        from mgr.threejs_plotting import show_percolation_threejs
                        mesh_isovalue = max(float(surface_value), 0.05)
                        analysis_copy = dict(analysis)
                        analysis_copy["isovalue"] = mesh_isovalue
                        html_str = show_percolation_threejs(
                            analysis_copy,
                            atoms,
                            probe_radius=selected_radius,
                            options=dict(layout=percolation_layout_choice),
                        )
                        components.html(
                            html_str,
                            height=1000,
                        )
                        plot_status.markdown("<div style='padding:6px;background:#06310c;color:#6FE37F;text-align:center;font-family:monospace;'>✅ Percolation plot complete (Three.js)</div>", unsafe_allow_html=True)
                    else:
                        plot_status.markdown(
                            _step_status(1, 3, "Building percolation traces"),
                            unsafe_allow_html=True,
                        )

                        try:
                            # Ensure plotly express is accessible for color scale
                            import plotly.express as px
                            # Match main visualization isosurface (same physical meaning as pore surface in first tab).
                            mesh_isovalue = max(float(surface_value), 0.05)
                            plot_cache = st.session_state.setdefault("percolation_plot_cache", {})
                            plot_key = (
                                analysis_key,
                                float(selected_radius),
                                float(mesh_isovalue),
                                boundary_line_color,
                                int(boundary_line_width),
                            )
                            if plot_key in plot_cache:
                                cluster_traces = plot_cache[plot_key]
                            else:
                                cluster_traces = mgr.plot_percolated_cluster_meshes(
                                    analysis["percolation_history"],
                                    analysis["dgrid"],
                                    analysis["regions"],
                                    atoms,
                                    probe_radius=selected_radius,
                                    isovalue=mesh_isovalue,
                                    stylize=True,
                                    opacity=1.0,
                                    cscale=px.colors.qualitative.Pastel,
                                    boundary_line_width=boundary_line_width,
                                    boundary_line_color=boundary_line_color,
                                )
                                plot_cache[plot_key] = cluster_traces
                        except Exception as exc:
                            st.error(str(exc))
                            cluster_traces = []

                        plot_status.markdown(
                            _step_status(2, 3, "Assembling traces (cell, graph, maxima, PLD)"),
                            unsafe_allow_html=True,
                        )
                        plot_traces = []
                        plot_traces.append(mgr.plot_cell(atoms))
                        rag_trace = mgr.plot_rag(analysis["rag"], name="Pore Graph")
                        if isinstance(rag_trace, (list, tuple)):
                            plot_traces.extend(rag_trace)
                        else:
                            plot_traces.append(rag_trace)

                        if analysis["maxima"].size:
                            maxima_trace = mgr.plot_maxima(
                                analysis["dgrid"],
                                analysis["maxima"],
                                atoms,
                                show_spheres=False,
                                sphere_opacity=0.85,
                            )
                            if isinstance(maxima_trace, tuple):
                                plot_traces.extend(maxima_trace)
                            else:
                                plot_traces.append(maxima_trace)

                        if analysis["pld_locations"].size:
                            pld_traces = mgr.plot_pld_locations(
                                analysis["dgrid"],
                                analysis["pld_locations"],
                                atoms,
                                show_spheres=False,
                                sphere_opacity=0.85,
                                marker_size=10,
                                marker_color="limegreen",
                            )
                            if isinstance(pld_traces, tuple):
                                plot_traces.extend(pld_traces)
                            else:
                                plot_traces.append(pld_traces)

                        plot_traces.extend(cluster_traces)

                        plot_status.markdown(
                            _step_status(3, 3, "Rendering percolation figure"),
                            unsafe_allow_html=True,
                        )
                        percolation_fig = go.Figure(data=plot_traces)
                        # Use the selected layout for percolation plot
                        layout = white_layout if percolation_layout_choice == "white" else dark_layout
                        percolation_fig.update_layout(**layout)
                        md_p = st.session_state.get("latest_metadata") or {}
                        if not isinstance(md_p, dict):
                            md_p = {}
                        _p_slug = _viz_settings_title_suffix(
                            spacing=float(md_p.get("spacing", 0.5)),
                            h_value=float(md_p.get("h_value", 0.4)),
                            threshold_abs=float(md_p.get("threshold_abs", 1.5)),
                            mask_thickness=float(md_p.get("mask_thickness", 1.5)),
                        )
                        _perc_title = f"Percolation probe r={selected_radius:.2f}Å | {material_name} | {_p_slug}"
                        if percolation_layout_choice == "white":
                            percolation_fig.update_layout(
                                title=dict(
                                    text=_perc_title,
                                    font=dict(color="#1f1f1f"),
                                    y=0.95,
                                ),
                                font=dict(color="#1f1f1f"),
                                legend_font=dict(color="#1f1f1f"),
                            )
                        else:
                            percolation_fig.update_layout(
                                title=dict(
                                    text=_perc_title,
                                    font=dict(color="#ffffff"),
                                    y=0.95,
                                ),
                                font=dict(color="#ffffff"),
                                legend_font=dict(color="#ffffff"),
                            )

                        for trace in percolation_fig.data:
                            if isinstance(trace, go.Mesh3d):
                                trace.update(
                                    lighting=STYLIZE_LIGHTING,
                                    lightposition=STYLIZE_LIGHTPOSITION,
                                    showscale=False,
                                )

                        _percol_cfg = {
                            'responsive': True,
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'displaylogo': False,
                            'doubleClick': 'reset',
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'{material_name}_percolation',
                            },
                        }
                        st.plotly_chart(percolation_fig, use_container_width=True, config=_percol_cfg)

                        plot_status.markdown("<div style='padding:6px;background:#06310c;color:#6FE37F;text-align:center;font-family:monospace;'>✅ Percolation plot complete</div>", unsafe_allow_html=True)

                    selected_components = next(
                        (
                            entry[1]
                            for entry in history
                            if np.isclose(float(entry[0]), float(selected_radius), rtol=0.0, atol=1e-5)
                        ),
                        {},
                    )
                    with st.expander("Percolation components at selected radius", expanded=False):
                        st.json(_percolation_axes_for_display(selected_components))

    with psd_tab:
        if results_data is None:
            st.info("Run the computation to enable pore size distribution analysis.")
        else:
            psd_cache = st.session_state.setdefault("psd_cache", {})
            effective_key = analysis_key or results_key
            regions_pbc = results_data.get("regions_pbc")
            has_connections = results_data.get("connections") is not None

            with st.form("psd_settings"):
                psd_domain = st.radio(
                    "Domain",
                    options=["Occupiable only", "Accessible only", "Both"],
                    index=0,
                    help="Occupiable (probe center): every pore region subject to the clearance rule. Accessible (probe center): only regions that percolate at σ, same clearance rule.",
                )
                probe_radius_psd = st.number_input(
                    "Probe radius (Å)",
                    value=1.2,
                    min_value=0.1,
                    max_value=10.0,
                    step=0.05,
                    format="%.2f",
                    help="Radius σ of the probe sphere. PSD sample sites are restricted to grid locations where clearance to the framework surface is ≥ σ.",
                )

                st.subheader("Global PSD (uniform sampling)")
                num_samples_total = st.number_input(
                    "Total samples",
                    value=10000,
                    min_value=1000,
                    step=1000,
                    help="Number of uniformly sampled points across the unit cell.",
                )
                uniform_num_bins = st.number_input(
                    "Number of bins",
                    value=200,
                    min_value=10,
                    step=10,
                )
                uniform_max_diameter = st.number_input(
                    "Maximum pore diameter (Å)",
                    value=50.0,
                    min_value=5.0,
                    step=5.0,
                )
                global_psd_log_bins = st.checkbox(
                    "Log-spaced bins for global PSD",
                    value=False,
                    help="Uses log-spaced histogram edges for the global plot (often easier to read at small diameters).",
                )

                st.subheader("Pocket-wise PSD")
                num_samples_per_region = st.number_input(
                    "Samples per region",
                    value=1000,
                    min_value=100,
                    step=100,
                    help="Number of samples drawn within each pore region.",
                )
                pocket_num_bins = st.number_input(
                    "Number of bins (pocket-wise)",
                    value=200,
                    min_value=10,
                    step=10,
                )
                pocket_max_diameter = st.number_input(
                    "Maximum pore diameter (Å) for pockets",
                    value=50.0,
                    min_value=5.0,
                    step=5.0,
                )

                compute_psd = st.form_submit_button("Compute PSDs")

            st.info(
                "**Distance grid:** geometric clearance to the cavity surface after assigning radii to framework atoms "
                "(Euclidean distance transform on the pore side of that surface). "
                "**Probe-centered** means we only draw histogram samples from void voxels where clearance **≥ probe radius σ**. "
                "**Occupiable (probe center):** PSD over all segmented pore regions under that rule. "
                "**Accessible (probe center):** same rule, but only regions that **percolate at σ**."
            )

            need_occupiable = psd_domain in ("Occupiable only", "Both")
            need_accessible = psd_domain in ("Accessible only", "Both")

            key_occupiable = (
                effective_key,
                "occupiable",
                bool(global_psd_log_bins),
                round(float(probe_radius_psd), 3),
                int(num_samples_total),
                int(uniform_num_bins),
                float(uniform_max_diameter),
                int(num_samples_per_region),
                int(pocket_num_bins),
                float(pocket_max_diameter),
            )
            key_accessible = (
                effective_key,
                "accessible",
                bool(global_psd_log_bins),
                round(float(probe_radius_psd), 3),
                int(num_samples_total),
                int(uniform_num_bins),
                float(uniform_max_diameter),
                int(num_samples_per_region),
                int(pocket_num_bins),
                float(pocket_max_diameter),
            )

            if need_accessible and not has_connections:
                st.warning(
                    "Accessible PSD requires segmentation with connections (percolation graph). "
                    "Re-run the main computation to enable accessible PSD."
                )

            if compute_psd:
                if isinstance(metadata, dict) and metadata.get("spacing") is not None:
                    spacing_ref = float(metadata["spacing"])
                else:
                    spacing_ref = float(spacing)
                d_hi = float(uniform_max_diameter)
                if global_psd_log_bins:
                    d_lo = float(
                        np.clip(
                            2.0 * spacing_ref,
                            0.05,
                            min(2.0, 0.45 * d_hi),
                        )
                    )
                    bins_uniform = np.geomspace(
                        max(d_lo, 1e-6),
                        max(d_hi, d_lo + 1e-6),
                        int(uniform_num_bins) + 1,
                    )
                else:
                    bins_uniform = np.linspace(0.0, d_hi, int(uniform_num_bins) + 1)
                bins_pocket = np.linspace(0, float(pocket_max_diameter), int(pocket_num_bins) + 1)
                material_name = getattr(cif_path, "stem", "Material")

                if need_occupiable:
                    with st.spinner("Computing occupiable PSDs..."):
                        uniform_hist, uniform_fig = mgr.compute_psd_uniform_rust(
                            results_data["dgrid"],
                            results_data["ase_atoms"],
                            num_samples_total=int(num_samples_total),
                            bins=bins_uniform,
                            density=False,
                            return_fig=True,
                            material=material_name,
                            probe_center_min_radius=float(probe_radius_psd),
                        )
                        uniform_df = make_uniform_psd_dataframe(bins_uniform, uniform_hist)
                        pocket_results = None
                        if regions_pbc is not None:
                            pocket_psds, pocket_fig = mgr.compute_psds_pocketwise_rust(
                                results_data["dgrid"],
                                regions_pbc,
                                results_data["ase_atoms"],
                                num_samples_per_region=int(num_samples_per_region),
                                bins=bins_pocket,
                                return_fig=True,
                                material=material_name,
                                probe_center_min_radius=float(probe_radius_psd),
                            )
                            pocket_df = make_pocket_psd_dataframe(bins_pocket, pocket_psds)
                            pocket_results = {
                                "psds": pocket_psds,
                                "fig": pocket_fig,
                                "bins": bins_pocket,
                                "df": pocket_df,
                            }
                        else:
                            st.warning("Pocket-wise PSD requires segmented regions; skipping pocket computation.")
                        occ_entry = {
                            "uniform": {
                                "hist": uniform_hist,
                                "fig": uniform_fig,
                                "bins": bins_uniform,
                                "df": uniform_df,
                                "log_bins": bool(global_psd_log_bins),
                            },
                            "pocket": pocket_results,
                        }
                        psd_cache[str(key_occupiable)] = occ_entry

                if need_accessible and has_connections:
                    with st.spinner("Computing accessible PSDs..."):
                        try:
                            acc_uniform_hist, acc_uniform_fig = mgr.compute_psd_global_accessible(
                                results_data,
                                results_data["ase_atoms"],
                                probe_radius=float(probe_radius_psd),
                                bins=bins_uniform,
                                samples_per_region=int(num_samples_per_region),
                                return_fig=True,
                                material=material_name,
                                enforce_probe_center_occupiable=True,
                            )
                            acc_uniform_df = make_uniform_psd_dataframe(bins_uniform, acc_uniform_hist)
                            acc_pocket_psds, acc_pocket_fig = mgr.compute_psds_pocketwise_accessible(
                                results_data,
                                results_data["ase_atoms"],
                                probe_radius=float(probe_radius_psd),
                                num_samples_per_region=int(num_samples_per_region),
                                bins=bins_pocket,
                                return_fig=True,
                                material=material_name,
                                enforce_probe_center_occupiable=True,
                            )
                            acc_pocket_df = make_pocket_psd_dataframe(bins_pocket, acc_pocket_psds)
                            acc_entry = {
                                "uniform": {
                                    "hist": acc_uniform_hist,
                                    "fig": acc_uniform_fig,
                                    "bins": bins_uniform,
                                    "df": acc_uniform_df,
                                    "log_bins": bool(global_psd_log_bins),
                                },
                                "pocket": {
                                    "psds": acc_pocket_psds,
                                    "fig": acc_pocket_fig,
                                    "bins": bins_pocket,
                                    "df": acc_pocket_df,
                                },
                            }
                            psd_cache[str(key_accessible)] = acc_entry
                        except Exception as e:
                            st.error(f"Accessible PSD failed: {e}")

                st.success("PSDs computed.")

            def _render_psd_section(entry, section_title: str, file_suffix: str):
                if not entry:
                    return
                st.markdown(f"### {section_title}")
                u = entry.get("uniform")
                if u:
                    st.markdown("**Global PSD**")
                    if u.get("log_bins"):
                        _psd_fig = go.Figure(u["fig"])
                        _psd_fig.update_xaxes(
                            type="log",
                            title="Pore diameter (Å, log scale)",
                            exponentformat="power",
                        )
                        st.plotly_chart(_psd_fig, use_container_width=True, config={'responsive': True, 'scrollZoom': True})
                    else:
                        st.plotly_chart(u["fig"], use_container_width=True, config={'responsive': True, 'scrollZoom': True})
                    st.download_button(
                        label="Download global PSD (CSV)",
                        data=u["df"].to_csv(index=False).encode("utf-8"),
                        file_name=f"{getattr(cif_path, 'stem', 'analysis')}_psd_uniform{file_suffix}.csv",
                        mime="text/csv",
                        key=f"psd_dl_uniform_{file_suffix}_{section_title[:20]}",
                    )
                    st.dataframe(u["df"], use_container_width=True)
                p = entry.get("pocket")
                if p:
                    st.markdown("**Pocket-wise PSD**")
                    st.plotly_chart(p["fig"], use_container_width=True, config={'responsive': True, 'scrollZoom': True})
                    st.download_button(
                        label="Download pocket-wise PSDs (CSV)",
                        data=p["df"].to_csv(index=False).encode("utf-8"),
                        file_name=f"{getattr(cif_path, 'stem', 'analysis')}_psd_pockets{file_suffix}.csv",
                        mime="text/csv",
                        key=f"psd_dl_pocket_{file_suffix}_{section_title[:20]}",
                    )
                    st.dataframe(p["df"], use_container_width=True)

            if need_occupiable:
                occ_entry = psd_cache.get(str(key_occupiable))
                _render_psd_section(
                    occ_entry,
                    f"Occupiable (probe center), σ = {probe_radius_psd:.2f} Å",
                    "",
                )

            if need_accessible and has_connections:
                acc_entry = psd_cache.get(str(key_accessible))
                _render_psd_section(
                    acc_entry,
                    f"Accessible (probe center), σ = {probe_radius_psd:.2f} Å",
                    "_accessible",
                )

    with sa_vol_tab:
        if results_data is None:
            st.info("Run the computation to enable surface area and void volume analysis.")
        else:
            material_name = getattr(cif_path, 'stem', 'Material')
            st.markdown(f"## Material: **{material_name}**")
            st.subheader("Surface Areas & Void Volumes")

            # Add probe radius input
            probe_radius = st.number_input(
                "Probe Radius (Å)",
                value=1.2,  # Default helium radius
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                help="Probe radius for surface area and volume calculations. Default is helium radius (1.2 Å)."
            )

            # Clear cached SA/V results when probe radius changes
            current_probe_key = f"probe_{probe_radius:.3f}"
            if st.session_state.get("last_probe_key") != current_probe_key:
                st.session_state.pop("sa_results", None)
                st.session_state.pop("vol_results", None)
                st.session_state["last_probe_key"] = current_probe_key

            st.markdown("### Surface Area & Void Volume Analysis")
            if st.button("Compute Surface Area & Void Volume", key="compute_sa_vol"):
                try:
                    from mgr.compute import compute_surface_area_from_results, compute_void_volume_from_results

                    # First, ensure we have percolation analysis data
                    analysis_key = st.session_state.get("analysis_key")
                    sa_vol_cache_key = analysis_key or f"{getattr(cif_path, 'name', 'uploaded')}::{float(metadata.get('spacing', spacing)):.3f}"

                    pld_cache = st.session_state.setdefault("pld_cache", {})
                    percolation_analysis = pld_cache.get(sa_vol_cache_key)

                    if percolation_analysis is None:
                        with st.spinner("Computing required percolation analysis first..."):
                            percolation_analysis = _compute_pld_analysis(
                                results_data,
                                atoms,
                                spacing=float(metadata.get("spacing", spacing)),
                                mask_thickness=float(metadata.get("mask_thickness", mask_thickness)),
                                h_value=float(h_value),
                                threshold_abs=float(threshold_abs),
                                min_distance=float(min_distance),
                                extra_padding=float(extra_padding),
                                watershed_backend=str(
                                    metadata.get("watershed_backend", "rust_bucket")
                                ),
                            )
                            pld_cache[sa_vol_cache_key] = percolation_analysis

                    # Prepare complete results dictionary like CLI does
                    modified_results = results_data.copy()
                    if 'dgrid' in modified_results:
                        modified_results['distance_grid'] = modified_results['dgrid']

                    # Add percolation analysis data to results
                    modified_results.update(percolation_analysis)

                    atoms = results_data.get('ase_atoms')
                    if atoms is None:
                        st.error("Atoms object not found in results. Please re-run the computation.")
                        sa_results = None
                        vol_results = None
                    else:
                        with st.spinner("Computing surface area analysis..."):
                            print(f"Computing surface area with probe radius: {probe_radius:.3f} Å")
                            print(f"Results keys available: {list(modified_results.keys())}")
                            sa_results = compute_surface_area_from_results(modified_results, atoms, probe_radius)
                            print(f"Surface area results: {sa_results}")
                        with st.spinner("Computing void volume analysis..."):
                            print(f"Computing void volume with probe radius: {probe_radius:.3f} Å")
                            vol_results = compute_void_volume_from_results(modified_results, atoms, probe_radius)
                            print(f"Void volume results: {vol_results}")

                    if sa_results:
                        st.session_state["sa_results"] = sa_results
                        st.success("Surface area analysis completed!")
                    else:
                        st.error("Surface area analysis failed. Please ensure mgr_src is properly installed.")
                    if vol_results:
                        st.session_state["vol_results"] = vol_results
                        st.success("Void volume analysis completed!")
                    else:
                        st.error("Void volume analysis failed. Please ensure mgr_src is properly installed.")
                except ImportError:
                    st.error("Surface area or void volume computation functions not available. Please ensure mgr_src is properly installed.")
                except Exception as e:
                    st.error(f"Error computing surface area or void volume: {e}")
                    st.info("This might be due to inappropriate parameters for this material structure. Try adjusting:")
                    st.info("- Grid spacing (smaller values like 0.3-0.4 Å may help)")
                    st.info("- Mask thickness (default: 1.5 Å)")
                    st.info("- Threshold absolute (default: 1.5 Å)")
                    st.info("- Or use 'Auto set parameters' option")

            # Display results if available
            sa_results = st.session_state.get("sa_results")
            vol_results = st.session_state.get("vol_results")

            if sa_results:
                st.markdown("### Surface Area Results")

                # Global metrics
                total_sa = sa_results.get('total_surface_area', 0)
                accessible_sa = sa_results.get('accessible_surface_area', 0)
                inaccessible_sa = total_sa - accessible_sa
                accessible_fraction = sa_results.get('accessible_fraction', 0)
                
                # Point probe reference
                point_probe_ref = sa_results.get('point_probe_reference', {})
                point_probe_sa = point_probe_ref.get('surface_area', 0)
                point_probe_fraction = point_probe_ref.get('surface_area_fraction', 0)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Surface Area", f"{total_sa:.4f} m²/g")
                col2.metric("Accessible Surface Area", f"{accessible_sa:.4f} m²/g")
                col3.metric("Inaccessible Surface Area", f"{inaccessible_sa:.4f} m²/g")
                col4.metric("Accessible Fraction", f"{accessible_fraction:.4f}")
                col5.metric("Point Probe (r=0)", f"{point_probe_sa:.4f} m²/g", help=f"Surface area accessible to point probe (radius = 0 Å)\nFraction of point probe: {point_probe_fraction:.4f}")

                # Create breakdown data
                breakdown_data = []

                # Region-wise breakdown (non-PBC)
                by_region = sa_results.get('by_region', {})
                if by_region:
                    for region_id, area in by_region.items():
                        breakdown_data.append({
                            'Type': 'Region (non-PBC)',
                            'Category': 'Segmented',
                            'Label': f'region_{region_id}',
                            'Surface Area (m²/g)': area
                        })

                # Pocket-wise breakdown (PBC groups)
                by_pocket = sa_results.get('by_pocket', {})
                if by_pocket:
                    for pocket_id, area in by_pocket.items():
                        breakdown_data.append({
                            'Type': 'Pocket (PBC)',
                            'Category': 'Grouped',
                            'Label': str(pocket_id),
                            'Surface Area (m²/g)': area
                        })

                # Percolated cluster breakdown
                by_percolated_cluster = sa_results.get('by_percolated_cluster', {})
                if by_percolated_cluster:
                    for cluster_id, area in by_percolated_cluster.items():
                        breakdown_data.append({
                            'Type': 'Cluster (percolated)',
                            'Category': 'Accessible',
                            'Label': str(cluster_id),
                            'Surface Area (m²/g)': area
                        })

                if breakdown_data:
                    st.markdown("#### Detailed Breakdown")
                    sa_df = pd.DataFrame(breakdown_data)

                    # Create bar chart
                    import plotly.express as px
                    fig_sa = px.bar(sa_df, x='Label', y='Surface Area (m²/g)',
                                  color='Type', barmode='group',
                                  title="Surface Area Breakdown by Category")
                    st.plotly_chart(fig_sa, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

                    # Show data table
                    st.dataframe(sa_df, use_container_width=True)

                    # Download CSV
                    csv_sa = sa_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download surface area results (CSV)",
                        data=csv_sa,
                        file_name=f"{getattr(cif_path, 'stem', 'analysis')}_surface_area.csv",
                        mime="text/csv",
                    )

            if vol_results:
                st.markdown("### Void Volume Results")

                # Global metrics
                total_vol = vol_results.get('total_void_volume', 0)
                accessible_vol = vol_results.get('accessible_void_volume', 0)
                inaccessible_vol = total_vol - accessible_vol
                accessible_fraction = vol_results.get('accessible_fraction', 0)
                
                # Point probe reference
                point_probe_ref = vol_results.get('point_probe_reference', {})
                point_probe_vol = point_probe_ref.get('void_volume', 0)
                point_probe_fraction = point_probe_ref.get('void_volume_fraction', 0)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Void Volume", f"{total_vol:.4f} cm³/g")
                col2.metric("Accessible Void Volume", f"{accessible_vol:.4f} cm³/g")
                col3.metric("Inaccessible Void Volume", f"{inaccessible_vol:.4f} cm³/g")
                col4.metric("Accessible Fraction", f"{accessible_fraction:.4f}")
                col5.metric("Point Probe (r=0)", f"{point_probe_vol:.4f} cm³/g", help=f"Void volume accessible to point probe (radius = 0 Å)\nFraction of point probe: {point_probe_fraction:.4f}")

                # Create breakdown data
                breakdown_data = []

                # Region-wise breakdown (non-PBC)
                by_region = vol_results.get('by_region', {})
                if by_region:
                    for region_id, volume in by_region.items():
                        breakdown_data.append({
                            'Type': 'Region (non-PBC)',
                            'Category': 'Segmented',
                            'Label': f'region_{region_id}',
                            'Void Volume (cm³/g)': volume
                        })

                # Pocket-wise breakdown (PBC groups)
                by_pocket = vol_results.get('by_pocket', {})
                if by_pocket:
                    for pocket_id, volume in by_pocket.items():
                        breakdown_data.append({
                            'Type': 'Pocket (PBC)',
                            'Category': 'Grouped',
                            'Label': str(pocket_id),
                            'Void Volume (cm³/g)': volume
                        })

                # Percolated cluster breakdown
                by_percolated_cluster = vol_results.get('by_percolated_cluster', {})
                if by_percolated_cluster:
                    for cluster_id, volume in by_percolated_cluster.items():
                        breakdown_data.append({
                            'Type': 'Cluster (percolated)',
                            'Category': 'Accessible',
                            'Label': str(cluster_id),
                            'Void Volume (cm³/g)': volume
                        })

                if breakdown_data:
                    st.markdown("#### Detailed Breakdown")
                    vol_df = pd.DataFrame(breakdown_data)

                    # Create bar chart
                    import plotly.express as px
                    fig_vol = px.bar(vol_df, x='Label', y='Void Volume (cm³/g)',
                                   color='Type', barmode='group',
                                   title="Void Volume Breakdown by Category")
                    st.plotly_chart(fig_vol, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

                    # Show data table
                    st.dataframe(vol_df, use_container_width=True)

                    # Download CSV
                    csv_vol = vol_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download void volume results (CSV)",
                        data=csv_vol,
                        file_name=f"{getattr(cif_path, 'stem', 'analysis')}_void_volume.csv",
                        mime="text/csv",
                    )


def _render_hts_mode() -> None:
    """Render the full High-Throughput mode UI (CIF source, options, run, results)."""
    st.markdown("## High-Throughput Characterization")

    # Restrict HTS to local installs (Docker / pip); show message only on hosted demos (e.g. Hugging Face Spaces)
    if os.environ.get("MGR_DEPLOYMENT") == "spaces":
        st.info(
            "**High-Throughput Characterization is only available on local installs.** "
            "Use the **Docker Desktop** image (Docker Hub) or **pip install** (PyPI) when we're public for full HTS and batch runs. "
            "This keeps the demo lightweight; run the app locally for full features."
        )
        return

    st.caption("Process many CIFs in parallel. Options below mirror single-material mode.")

    samples = _get_sample_cifs()
    use_default_set = st.checkbox(
        "Test HTS using included default set of materials",
        value=False,
        key="hts_use_default_set",
        help="Use the built-in sample CIFs for exploring the app. Leave unchecked to use your own folder or uploads.",
    )
    n_default = len(samples) if samples else 0
    if use_default_set and n_default:
        st.caption(f"Default test set: {n_default} materials.")
    hts_folder = ""
    hts_uploads = []
    if not use_default_set:
        hts_folder = st.text_input(
            "Path to folder containing CIFs",
            value="",
            help="For Docker, use a mounted path. Leave empty to use uploads below.",
            key="hts_folder",
        )
        hts_uploads = st.file_uploader(
            "Or upload CIF files",
            type=["cif"],
            accept_multiple_files=True,
            key="hts_upload",
        )

    col_left, col_right = st.columns(2)
    with col_left:
        hts_select_all = st.checkbox(
            "Select all (all calculations, both PSD types, all outputs & images)",
            value=False,
            key="hts_select_all",
            help="Enable all optional analyses, outputs, and image exports. You can still tweak individual options below.",
        )
        hts_global_probe = st.number_input(
            "Global probe radius (Å)",
            value=1.2,
            min_value=0.1,
            step=0.05,
            key="hts_global_probe",
            help="Default probe radius for PSD (accessible) and SA/Vol. Override individually below if needed.",
        )
        st.markdown("#### Dgrid options")
        hts_spacing = st.number_input("Grid spacing (Å)", value=0.5, min_value=0.1, step=0.1, key="hts_spacing")
        hts_use_aabb = st.checkbox("Use Python AABB (CPU) method for distance grid", value=False, key="hts_aabb")
        hts_block_size = 50000
        if hts_use_aabb:
            hts_block_size = st.number_input("AABB block size (grid points per worker)", value=50000, min_value=1000, max_value=200000, step=5000, key="hts_bs")

        st.markdown("#### Segmentation options")
        hts_mask = st.number_input("Mask thickness (Å)", value=0.5, min_value=0.05, key="hts_mask")
        hts_h = st.number_input("h (Å)", value=0.2, min_value=0.05, key="hts_h")
        hts_surface_value = st.number_input("Surface value (Å)", value=1.0, min_value=0.1, key="hts_surface_value")
        hts_min_dist, hts_threshold, hts_extra_pad = 1.5, 1.5, 5.0
        hts_watershed_backend = "rust_bucket"
        with st.expander("Advanced segmentation", expanded=False):
            hts_min_dist = st.number_input("Min distance (Å)", value=1.5, key="hts_md")
            hts_threshold = st.number_input("Threshold abs (Å)", value=1.5, key="hts_thr")
            hts_extra_pad = st.number_input("Extra padding (Å)", value=5.0, key="hts_epad")
            hts_watershed_backend = _watershed_backend_widgets("hts")
        hts_min_box_enabled = st.checkbox("Ensure minimum box thickness", value=False, key="hts_minbox")
        hts_min_box = 6.0
        if hts_min_box_enabled:
            hts_min_box = st.number_input("Minimum box thickness (Å)", value=6.0, min_value=1.0, max_value=50.0, step=0.5, key="hts_minbox_val")

        with st.expander("Visualization options (for HTML plot)", expanded=False):
            hts_plot_framework = st.checkbox("Show framework", value=True, key="hts_plot_fw")
            hts_framework_mesh = st.checkbox("Show framework as mesh", value=False, key="hts_fw_mesh")
            hts_framework_scatter = st.checkbox("Show simple framework scatter", value=True, key="hts_fw_scatter")
            hts_show_rings = st.checkbox("Show rings in framework", value=True, key="hts_show_rings")
            hts_plot_cell = st.checkbox("Show cell", value=True, key="hts_plot_cell")
            hts_plot_maxima = st.checkbox("Show maxima", value=True, key="hts_plot_maxima")
            hts_projection = st.selectbox("Projection", ["perspective", "orthographic"], index=0, key="hts_proj")
            hts_stylize = st.checkbox("Stylize plot", value=True, disabled=(hts_projection == "orthographic"), key="hts_stylize")
            hts_layout = st.selectbox("Background", ["dark", "white"], index=0, key="hts_layout")
            hts_show_surface = st.checkbox("Show pore surface", value=True, key="hts_show_surf")
            hts_show_surface_pbc = st.checkbox("Show periodic pore surface", value=True, key="hts_show_surf_pbc")
            hts_show_connections = st.checkbox("Show pore graph", value=True, key="hts_show_conn")
            hts_show_maxima_spheres = st.checkbox("Show maxima spheres", value=False, key="hts_show_max_sph")
            hts_plot_pocket_types = st.checkbox("Plot pocket types on surface", value=False, key="hts_plot_pocket_types")
            hts_surface_types_opacity = st.slider("Pocket types surface opacity", 0.0, 1.0, 1.0, 0.1, key="hts_surf_types_opacity")
            hts_surface_opacity = st.slider("Surface opacity", 0.0, 1.0, 1.0, 0.1, key="hts_surf_opacity")
            hts_surface_pbc_opacity = st.slider("Periodic surface opacity", 0.0, 1.0, 1.0, 0.1, key="hts_surf_pbc_opacity")

        with st.expander("Optional analyses", expanded=False):
            hts_do_pore_dia = st.checkbox("Compute pore diameter and percolation", value=False, key="hts_pore_dia")
            hts_save_pore_dia_csv = st.checkbox("Save pore diameter CSV", value=False, key="hts_save_pore_dia")
            hts_detect_pocket_types = st.checkbox("Detect pocket types", value=False, key="hts_detect_pocket_types")
            if hts_detect_pocket_types or hts_plot_pocket_types:
                hts_js_threshold = st.slider("Jensen–Shannon threshold for pocket clustering", 0.05, 0.5, 0.2, 0.01, key="hts_js_threshold")
            else:
                hts_js_threshold = 0.2
            hts_do_psd = st.checkbox("Compute PSD", value=False, key="hts_psd")
            hts_psd_domain = st.radio("PSD domain", ["Occupiable only", "Accessible only", "Both"], index=0, key="hts_psd_domain")
            if hts_psd_domain in ("Accessible only", "Both"):
                hts_probe_psd = st.number_input("PSD probe radius (Å)", value=float(hts_global_probe), min_value=0.1, step=0.05, key="hts_probe_psd", help="Overrides global probe for accessible PSD.")
            else:
                hts_probe_psd = float(hts_global_probe)
            hts_save_psd_csv = st.checkbox("Save PSD CSVs", value=False, key="hts_save_psd")
            if hts_do_psd:
                hts_psd_samples_total = st.number_input("PSD total samples", value=10000, min_value=1000, step=1000, key="hts_psd_n_total")
                hts_psd_uniform_bins = st.number_input("PSD uniform bins", value=200, min_value=10, step=10, key="hts_psd_unif_bins")
                hts_psd_uniform_max_d = st.number_input("PSD uniform max diameter (Å)", value=50.0, min_value=5.0, step=5.0, key="hts_psd_unif_max")
                hts_psd_per_region = st.number_input("PSD samples per region", value=1000, min_value=100, step=100, key="hts_psd_per_reg")
                hts_psd_pocket_bins = st.number_input("PSD pocket bins", value=200, min_value=10, step=10, key="hts_psd_pock_bins")
                hts_psd_pocket_max_d = st.number_input("PSD pocket max diameter (Å)", value=50.0, min_value=5.0, step=5.0, key="hts_psd_pock_max")
            else:
                hts_psd_samples_total, hts_psd_uniform_bins = 10000, 200
                hts_psd_uniform_max_d, hts_psd_per_region = 50.0, 1000
                hts_psd_pocket_bins, hts_psd_pocket_max_d = 200, 50.0
            hts_do_sa_vol = st.checkbox("Compute surface area and void volume", value=False, key="hts_sa_vol")
            hts_save_sa_vol_csv = st.checkbox("Save SA/Vol CSVs", value=False, key="hts_save_sa_vol")
            hts_probe_sa_vol = st.number_input("SA/Vol probe radius (Å)", value=float(hts_global_probe), min_value=0.1, step=0.1, key="hts_probe_sa_vol", help="Overrides global probe for SA/Vol.")

        st.markdown("#### Output options")
        hts_save_npz = st.checkbox("Save results NPZ", value=True, key="hts_npz")
        hts_save_rag = st.checkbox("Save pore graph (RAG .pkl)", value=False, key="hts_rag")
        hts_save_html = st.checkbox("Generate interactive HTML plot", value=False, key="hts_html")
        with st.expander("Images to output", expanded=False):
            hts_save_psd_plot = st.checkbox("Save PSD plot (HTML)", value=False, key="hts_out_psd_plot")
            hts_save_sa_plot = st.checkbox("Save surface area breakdown plot (HTML)", value=False, key="hts_out_sa_plot")
            hts_save_vol_plot = st.checkbox("Save void volume breakdown plot (HTML)", value=False, key="hts_out_vol_plot")
        default_results_root = os.environ.get("MGR_RESULTS_DIR", ".")
        default_hts_out = str(Path(default_results_root) / "hts_output")
        hts_out_dir = st.text_input(
            "Output directory",
            value=default_hts_out,
            key="hts_out",
            help="Directory where HTS results will be written. In Docker, this defaults to the mounted results volume.",
        )
        hts_tar = st.checkbox("Compress output to single tar when done", value=False, key="hts_tar")

    with col_right:
        st.markdown(
            """
            <style>
            div[data-testid="column"]:nth-of-type(2) .stButton > button {
                background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
                color: white;
                font-weight: 700;
                font-size: 1.1rem;
                padding: 0.6rem 1.2rem;
                border: none;
                border-radius: 8px;
                width: 100%;
            }
            div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
                background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        hts_run = st.button("⚡ Run HTS", key="hts_run")
        hts_results_key = "hts_run_results"
        hts_status = st.empty()

        def _collect_hts_cifs():
            if use_default_set and samples:
                return list(samples.values())
            cif_paths = []
            folder_path = str(hts_folder).strip() if hts_folder else ""
            if folder_path and Path(folder_path).exists():
                for p in Path(folder_path).rglob("*.cif"):
                    cif_paths.append(p)
            if not cif_paths and hts_uploads:
                import tempfile
                tmp = tempfile.mkdtemp(prefix="hts_cifs_")
                for u in hts_uploads:
                    path = Path(tmp) / (u.name or "upload.cif")
                    path.write_bytes(u.getvalue())
                    cif_paths.append(path)
            return cif_paths

        if hts_run:
            cif_list = _collect_hts_cifs()
            if not cif_list:
                hts_status.warning("Enable default set, provide a folder path with CIFs, or upload CIF files.")
            else:
                try:
                    from dask.distributed import Client, LocalCluster
                    from dask import delayed
                except ImportError:
                    hts_status.error("Install dask[distributed] to use High-Throughput (e.g. pip install 'dask[distributed]').")
                else:
                    hts_status.info(f"Starting HTS for {len(cif_list)} CIFs…")
                    out_dir = Path(hts_out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    framework_style = "scatter" if hts_framework_scatter and not hts_framework_mesh else "mesh"
                    viz_options = {
                        "plot_framework": hts_plot_framework,
                        "plot_framework_style": framework_style,
                        "plot_framework_show_rings": hts_show_rings,
                        "plot_cell": hts_plot_cell,
                        "plot_maxima": hts_plot_maxima,
                        "plot_surface": hts_show_surface,
                        "plot_surface_pbc": hts_show_surface_pbc,
                        "plot_surface_isovalue": hts_surface_value,
                        "plot_surface_opacity": hts_surface_opacity,
                        "plot_surface_pbc_opacity": hts_surface_pbc_opacity,
                        "plot_surface_types_opacity": hts_surface_types_opacity,
                        "plot_connections": hts_show_connections,
                        "plot_windows": True,
                        "plot_maxima_show_spheres": hts_show_maxima_spheres,
                        "plot_projection": hts_projection,
                        "plot_layout": hts_layout,
                    }
                    params = {
                        "spacing": hts_spacing,
                        "use_aabb_method": hts_use_aabb,
                        "block_size": hts_block_size,
                        "mask_thickness": hts_mask,
                        "h_value": hts_h,
                        "surface_value": hts_surface_value,
                        "min_distance": hts_min_dist,
                        "threshold_abs": hts_threshold,
                        "extra_padding": hts_extra_pad,
                        "min_box_thickness": float(hts_min_box) if hts_min_box_enabled else None,
                        "watershed_backend": str(hts_watershed_backend),
                    }
                    _all = hts_select_all
                    options = {
                        "save_npz": hts_save_npz,
                        "save_rag": hts_save_rag or _all,
                        "save_html": hts_save_html or _all,
                        "viz_options": viz_options,
                        "stylize": hts_stylize and hts_projection != "orthographic",
                        "do_pore_dia": hts_do_pore_dia or _all,
                        "save_pore_dia_csv": hts_save_pore_dia_csv or _all,
                        "do_psd": hts_do_psd or _all,
                        "save_psd_csv": hts_save_psd_csv or _all,
                        "probe_radius_psd": float(hts_probe_psd),
                        "do_sa_vol": hts_do_sa_vol or _all,
                        "save_sa_vol_csv": hts_save_sa_vol_csv or _all,
                        "probe_radius_sa_vol": float(hts_probe_sa_vol),
                        "psd_num_samples_total": int(hts_psd_samples_total),
                        "psd_num_samples_per_region": int(hts_psd_per_region),
                        "psd_uniform_num_bins": int(hts_psd_uniform_bins),
                        "psd_pocket_num_bins": int(hts_psd_pocket_bins),
                        "psd_uniform_max_diameter": float(hts_psd_uniform_max_d),
                        "psd_pocket_max_diameter": float(hts_psd_pocket_max_d),
                        "psd_domain": "Both" if _all else hts_psd_domain,
                        "detect_pocket_types": hts_detect_pocket_types or hts_plot_pocket_types or _all,
                        "plot_pocket_types": hts_plot_pocket_types or _all,
                        "js_threshold": float(hts_js_threshold),
                        "save_psd_plot": hts_save_psd_plot or _all,
                        "save_sa_plot": hts_save_sa_plot or _all,
                        "save_vol_plot": hts_save_vol_plot or _all,
                    }
                    # Configure Dask dashboard to be reachable from Docker
                    dash_port = int(os.environ.get("MGR_DASK_DASHBOARD_PORT", "8787"))
                    dashboard_address = f"0.0.0.0:{dash_port}"
                    cluster = LocalCluster(n_workers=4, threads_per_worker=1, dashboard_address=dashboard_address)
                    client = Client(cluster)
                    dashboard_url = getattr(cluster, "dashboard_link", None) or getattr(client, "dashboard_link", None)
                    # Prefer host-facing URL (for Docker): default 8787 so mapping 8787:8787 works without env
                    host_dash_port = os.environ.get("MGR_DASK_DASHBOARD_HOST_PORT") or str(dash_port)
                    dashboard_url = f"http://localhost:{host_dash_port}"
                    if dashboard_url:
                        hts_status.markdown(f"**HTS running.** Monitor workers: [Open Dask Dashboard]({dashboard_url})")
                    log_file = out_dir / "hts_run.log"
                    hts_logger = logging.getLogger("mgrcli.hts")
                    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                    file_handler.setLevel(logging.INFO)
                    hts_logger.addHandler(file_handler)
                    hts_logger.setLevel(logging.INFO)
                    try:
                        import time
                        t0 = time.time()
                        hts_logger.info("HTS started with %d materials (output: %s)", len(cif_list), str(out_dir))
                        delayed_list = [delayed(_hts_one_material)(str(p), str(out_dir), params, options) for p in cif_list]
                        futures = client.compute(delayed_list)
                        total = len(futures)
                        progress_placeholder = st.empty()
                        with st.spinner("HTS running… This may take several minutes."):
                            last_done = -1
                            no_progress_sec = 0
                            while True:
                                done = sum(1 for f in futures if (hasattr(f, "done") and f.done()) or (getattr(f, "status", None) in ("finished", "error", "cancelled", "lost")))
                                if done == last_done:
                                    no_progress_sec += 1
                                    if no_progress_sec >= 600:
                                        hts_logger.warning("No progress for 10 min; stopping wait.")
                                        break
                                else:
                                    no_progress_sec = 0
                                    last_done = done
                                progress_placeholder.caption(f"Progress: {done}/{total} materials finished")
                                if done >= total:
                                    break
                                time.sleep(1)
                            results_list = []
                            result_timeout = 120
                            for i, f in enumerate(futures):
                                try:
                                    r = f.result(timeout=result_timeout)
                                    if r is None or not (isinstance(r, tuple) and len(r) == 4):
                                        results_list.append(("?", None, False, "No result (worker returned None)"))
                                    else:
                                        results_list.append(r)
                                except Exception as e:
                                    results_list.append(("?", None, False, str(e)))
                            elapsed = time.time() - t0
                            n_ok = sum(1 for r in results_list if isinstance(r, tuple) and len(r) >= 3 and r[2])
                            n_err = total - n_ok
                            n_no_result = sum(1 for r in results_list if isinstance(r, tuple) and len(r) >= 2 and r[1] is None and not (r[2] if len(r) > 2 else False))
                            progress_placeholder.caption(
                                f"**Summary:** Completed: {n_ok}, Errored: {n_err} ({n_no_result} no result), Total: {total} · Time: {elapsed:.1f}s"
                            )
                            st.session_state[hts_results_key] = results_list
                            hts_status.success(f"HTS finished: {n_ok} completed, {n_err} errored ({n_no_result} no result), total {elapsed:.1f}s.")
                            hts_logger.info(
                                "HTS finished: completed=%d errored=%d no_result=%d total=%d time=%.1fs",
                                n_ok, n_err, n_no_result, total, elapsed,
                            )
                            for r in results_list:
                                if isinstance(r, tuple) and len(r) >= 4 and not r[2] and r[3]:
                                    hts_logger.warning("Material %s failed: %s", r[0], r[3])
                    finally:
                        hts_logger.removeHandler(file_handler)
                        file_handler.close()
                        client.close()
                        cluster.close()

        if hts_results_key in st.session_state:
            rows = st.session_state[hts_results_key]
            def _row(r):
                if r is None or not (isinstance(r, tuple) and len(r) >= 4):
                    return {"Material": "?", "Status": "Failed", "Output path": "—", "Error": "No result"}
                return {"Material": r[0], "Status": "OK" if r[2] else "Failed", "Output path": r[1] or "—", "Error": (r[3] or "")}
            df = pd.DataFrame([_row(r) for r in rows])
            st.dataframe(df, use_container_width=True)
            if hts_tar and rows and any((r[2] if isinstance(r, tuple) and len(r) >= 3 else False) for r in rows):
                import tarfile
                import io
                out_dir = Path(hts_out_dir)
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    for r in rows:
                        if isinstance(r, tuple) and len(r) >= 2 and r[2] and r[1]:
                            tar.add(r[1], arcname=Path(r[1]).name)
                buf.seek(0)
                st.download_button("Download results.tar.gz", data=buf.getvalue(), file_name="hts_results.tar.gz", mime="application/gzip", key="hts_tar_dl")


if __name__ == "__main__":
    main()

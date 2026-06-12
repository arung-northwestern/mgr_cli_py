"""Persistent queue state for the Spaces compute worker.

This module lives outside the Streamlit app's file-watcher scope
(python_cli/) so that module-level globals survive Streamlit hot-reloads
and devMode reruns.

All state is process-scoped: it persists across st.rerun() calls within
a single Python process, but is lost on container/process restart.
"""
from __future__ import annotations

import os
import queue
import threading
from typing import Any, Dict

JOB_QUEUE: "queue.PriorityQueue" = queue.PriorityQueue()
JOB_REGISTRY: Dict[str, dict] = {}
JOB_LOCK = threading.Lock()
RESULT_CACHE: Dict[str, Any] = {}
COMPUTE_TIMEOUT_SEC: int = int(os.environ.get("MGR_COMPUTE_TIMEOUT_SEC", "240"))
QUEUE_WORKER_STARTED: bool = False


def clear_result_cache() -> int:
    """Drop all cached segmentation results (process-global, survives st.rerun)."""
    n = len(RESULT_CACHE)
    RESULT_CACHE.clear()
    return n

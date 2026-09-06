"""Connection edge filter: keep edge unless every window on that pair is below threshold."""
from __future__ import annotations

import numpy as np

from mgr.helper import connection_passes_window_threshold


def test_keep_edge_when_any_window_above_threshold():
    assert connection_passes_window_threshold([0.2, 1.5], 0.6)
    assert connection_passes_window_threshold([2.0, 3.5], 3.0)
    assert not connection_passes_window_threshold([1.2], 3.0)
    assert not connection_passes_window_threshold([0.2, 0.5], 0.6)


def test_keep_edge_when_no_window_radii():
    assert connection_passes_window_threshold([], 0.6)
    assert connection_passes_window_threshold([np.nan], 0.6)


def test_numpy_window_radii_array():
    assert connection_passes_window_threshold(np.array([0.2, 1.5]), 0.6)
    assert not connection_passes_window_threshold(np.array([0.2, 0.5]), 0.6)

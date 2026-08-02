"""Connection edge filter: keep edge unless every window on that pair is below threshold."""
from __future__ import annotations

import numpy as np


def connection_passes_window_threshold(segment_radii, threshold):
    if threshold is None:
        return True
    if not segment_radii:
        return True
    vals = [float(x) for x in segment_radii if np.isfinite(x)]
    if not vals:
        return True
    return any(v >= threshold for v in vals)


def test_keep_edge_when_any_window_above_threshold():
    assert connection_passes_window_threshold([0.2, 1.5], 0.6)
    assert connection_passes_window_threshold([2.0, 3.5], 3.0)
    assert not connection_passes_window_threshold([1.2], 3.0)
    assert not connection_passes_window_threshold([0.2, 0.5], 0.6)


def test_keep_edge_when_no_window_radii():
    assert connection_passes_window_threshold([], 0.6)
    assert connection_passes_window_threshold([np.nan], 0.6)

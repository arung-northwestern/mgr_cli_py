"""Tests for PBC-aware regions_cluster painting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))

from mgr.pocket_typing import (
    build_fragment_label_to_type,
    regions_cluster_from_rep_types,
)


def test_all_pbc_copies_share_type():
    """Periodic copies (fragment labels 1 and 2) must both get the rep's type."""
    regions = np.zeros((12, 12, 12), dtype=int)
    regions[0:4, 0:4, 0:4] = 1
    regions[6:10, 6:10, 6:10] = 2
    rep_to_type = {1: 0, 3: 1}
    pbc_groups = [[1, 2], [3]]

    label_to_type = build_fragment_label_to_type(rep_to_type, pbc_groups)
    assert label_to_type[1] == 1
    assert label_to_type[2] == 1
    assert label_to_type[3] == 2

    regions_cluster = regions_cluster_from_rep_types(regions, rep_to_type, pbc_groups)
    assert np.all(regions_cluster[regions == 1] == 1)
    assert np.all(regions_cluster[regions == 2] == 1)
    assert np.all(regions_cluster[regions == 3] == 2)
    assert np.all(regions_cluster[regions == 0] == 0)


def test_rep_only_paint_would_miss_copy():
    """Old bug: painting only rep label leaves periodic copy uncolored."""
    regions = np.zeros((8, 8, 8), dtype=int)
    regions[0:3] = 1
    regions[5:8] = 2
    rep_to_type = {1: 0}
    pbc_groups = [[1, 2]]

    wrong = np.zeros_like(regions)
    wrong[regions == 1] = 1
    assert np.any(regions == 2) and np.all(wrong[regions == 2] == 0)

    fixed = regions_cluster_from_rep_types(regions, rep_to_type, pbc_groups)
    assert np.all(fixed[regions == 2] == 1)

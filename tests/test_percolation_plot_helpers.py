"""Percolation plotting helpers: stable colors and threshold selection."""
from __future__ import annotations

import pytest

from mgr.helper import (
    filter_connections_at_probe_radius,
    percolation_axis_dir_str,
    select_percolation_history_entry,
    sorted_percolation_component_items,
)


def test_sorted_components_stable_under_axis_order_shuffle():
    components_a = {
        "z": [[3, 1], [2]],
        "x": [[2, 1]],
        "y": [[3]],
    }
    components_b = {
        "x": [[2, 1]],
        "y": [[3]],
        "z": [[3, 1], [2]],
    }
    keys_a = [item[0] for item in sorted_percolation_component_items(components_a)]
    keys_b = [item[0] for item in sorted_percolation_component_items(components_b)]
    assert keys_a == keys_b == [(1, 2), (1, 3), (2,), (3,)]


def test_percolation_axis_dir_str_accepts_winding_and_lattice_keys():
    assert percolation_axis_dir_str(["x", "z"]) == "xz"
    assert percolation_axis_dir_str([0, 2]) == "xz"
    assert percolation_axis_dir_str(["y"]) == "y"


def test_select_percolation_history_entry_exact_before_ceiling():
    history = [(1.0, {}), (1.5, {}), (2.0, {})]
    assert select_percolation_history_entry(history, 1.5)[0] == 1.5
    assert select_percolation_history_entry(history, 1.55)[0] == 2.0


def test_select_percolation_history_entry_raises_when_above_max():
    history = [(1.0, {}), (1.5, {})]
    with pytest.raises(ValueError):
        select_percolation_history_entry(history, 3.0)


def test_filter_connections_at_probe_radius_nodes_and_windows():
    connections = [
        ((1, 2), True, "internal", [], [0.4, 1.2]),
        ((2, 3), True, "internal", [], [1.0]),
        ((1, 3), True, "internal", [], [0.8]),
    ]
    maxima_radii = [1.5, 1.0, 0.5]
    kept = filter_connections_at_probe_radius(connections, 0.9, maxima_radii)
    assert len(kept) == 1
    assert kept[0][0] == (1, 2)


def test_percolation_history_to_json_roundtrip():
    import json

    from mgrcli.textural_analysis import percolation_history_to_json

    analysis = {
        "lcd": 12.5,
        "pld": 8.0,
        "dfsd": 6.25,
        "percolation_history": [
            (1.0, {"x": [[1, 2]], "y": [[3]], "z": []}),
            (1.5, {0: [[4]], 1: [], 2: [[5, 6]]}),
        ],
    }
    doc = json.loads(percolation_history_to_json(analysis))
    assert doc["lcd_A"] == 12.5
    assert doc["pld_A"] == 8.0
    assert len(doc["history"]) == 2
    assert doc["history"][0]["probe_radius_A"] == 1.0
    assert doc["history"][0]["unique_clusters"] == 2
    assert doc["history"][0]["axis_components"]["a"] == [[1, 2]]
    assert doc["history"][1]["axis_component_counts"]["c"] == 1

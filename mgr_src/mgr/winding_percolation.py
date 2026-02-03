"""
Periodic winding-number percolation helpers for directed pore graphs.

This module targets the deduplicated NetworkX ``MultiDiGraph`` produced by the
PBC-aware segmentation pipeline described in ``winding_percolation_gpt5.1.md``:

* Nodes correspond to unique ``maxima_pbc`` representatives and carry at least
  ``pocket_radius`` (Å) and fractional coordinates ``frac_pos``.
* Edges are stored only once with the canonical ordering ``i <= j`` and provide
  ``relative_offset`` – an integer tuple telling how much node ``j`` must be
  shifted (in box lengths) to reach the connected periodic image from ``i``.
  Self-loops with non-zero offsets represent pores directly connected to their
  own periodic images and therefore can encode valid winding loops.
* The transpose edge (``j -> i`` with ``-relative_offset``) is intentionally
  absent to match CGD/topology file conventions, so any traversal routine must
  treat edges as *undirected* while respecting the stored orientation when
  accumulating winding numbers.

The functions below keep the BFS/DFS logic entirely in graph space – no voxel
replication is required.  They accumulate integer offsets per node, detect when
back-edges close non-contractible cycles, and report per-axis percolation along
with the actual winding clusters.  When testing percolation along one axis we
implicitly allow periodic connectivity in the other two axes, ensuring that
clusters such as the large NU-1000 corner pocket (which has multiple XY images
but a single Z loop) are reported as one physical cluster instead of duplicate
self-loops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import mofography as mgr

Offset = Tuple[int, int, int]


@dataclass(frozen=True)
class CanonicalEdgeID:
    """Reference to an edge independent of traversal direction."""

    u: int
    v: int
    key: int

    @classmethod
    def from_nodes(cls, a: int, b: int, key: int) -> "CanonicalEdgeID":
        if a <= b:
            return cls(a, b, key)
        return cls(b, a, key)


def _normalize_offset(value: Sequence[int]) -> Offset:
    """Return an integer tuple regardless of numpy scalar types."""
    return tuple(int(x) for x in value)


def _iter_neighbor_edges(
    G: nx.MultiDiGraph, node: int
) -> Iterator[Tuple[int, int, Offset, Offset, bool]]:
    """
    Yield (neighbor, key, canonical_offset, delta, forward) for edges touching ``node``.

    ``forward=True`` means ``node`` matches the stored direction (edge data is
    ``node -> neighbor``).  ``forward=False`` means we are traversing the edge
    in reverse (``neighbor -> node`` in the stored graph) and therefore the
    offset must be negated.
    """

    # Outgoing edges: orientation matches storage.
    for nbr, keyed in G.succ[node].items():
        for key, attrs in keyed.items():
            rel = _normalize_offset(attrs.get("relative_offset", (0, 0, 0)))
            yield nbr, key, rel, rel, True

    # Incoming edges: orientation reversed.
    for nbr, keyed in G.pred[node].items():
        for key, attrs in keyed.items():
            rel = _normalize_offset(attrs.get("relative_offset", (0, 0, 0)))
            if nbr == node:
                # Self-loops already handled by the outgoing iteration above.
                continue
            delta = tuple(-x for x in rel)
            yield nbr, key, rel, delta, False


def _tree_path(u: int, v: int, parent: Dict[int, Optional[int]]) -> List[int]:
    """Return the list of nodes along the unique tree path connecting u and v."""
    path_u = []
    cur = u
    while cur is not None:
        path_u.append(cur)
        cur = parent[cur]

    path_v = []
    cur = v
    while cur is not None:
        path_v.append(cur)
        cur = parent[cur]

    ancestors_u = set(path_u)
    lca = next(node for node in path_v if node in ancestors_u)

    pu = path_u[: path_u.index(lca) + 1]
    pv = path_v[: path_v.index(lca) + 1]

    return pu + pv[::-1][1:]


def percolation_winding_multidigraph(
    G: nx.MultiDiGraph,
) -> Tuple[Dict[str, bool], List[Dict[str, object]], Dict[int, Offset]]:
    """
    Evaluate percolation using winding numbers on a deduplicated MultiDiGraph.

    Returns
    -------
    percolates : dict
        Keys ``'x'``, ``'y'``, ``'z'`` map to booleans indicating whether at
        least one non-contractible loop exists along that axis.
    loops : list of dict
        Each loop dictionary contains ``winding`` (Offset), ``nodes`` (list of
        node IDs along the tree path that closes the loop), ``edge`` (canonical
        ``(u, v, key)``), ``edge_offset`` (stored offset on that edge) and
        ``delta`` (the traversal offset that triggered detection).
    acc_offsets : dict
        Maps node ID to its accumulated integer offset relative to the BFS root
        of its connected component.
    """

    percolates = {"x": False, "y": False, "z": False}
    loops: List[Dict[str, object]] = []
    acc_offsets: Dict[int, Offset] = {}
    parent: Dict[int, Optional[int]] = {}
    visited = set()

    undirected = nx.Graph(G.to_undirected(as_view=True))

    for comp_nodes in nx.connected_components(undirected):
        comp_nodes = set(comp_nodes)
        root = next(iter(comp_nodes))
        if root in visited:
            continue
        visited.add(root)
        parent[root] = None
        acc_offsets[root] = (0, 0, 0)
        queue = deque([root])

        while queue:
            u = queue.popleft()
            u_acc = np.array(acc_offsets[u], dtype=int)

            for nbr, key, canonical_offset, delta, forward in _iter_neighbor_edges(G, u):
                ced = CanonicalEdgeID.from_nodes(u, nbr, key)

                # Ensure traversal stays within the current component
                if nbr not in comp_nodes:
                    continue

                implied = tuple((u_acc + np.array(delta, dtype=int)).tolist())

                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = u
                    acc_offsets[nbr] = implied
                    queue.append(nbr)
                    continue

                current = acc_offsets[nbr]
                if implied == current:
                    continue

                winding_array = np.array(implied, dtype=int) - np.array(current, dtype=int)
                winding = tuple(int(x) for x in winding_array)
                if winding == (0, 0, 0):
                    continue

                cycle_nodes = _tree_path(u, nbr, parent)
                loop_info = {
                    "winding": winding,
                    "nodes": cycle_nodes,
                    "edge": (ced.u, ced.v, ced.key),
                    "edge_offset": canonical_offset,
                    "delta": delta,
                }
                loops.append(loop_info)
                if winding[0] != 0:
                    percolates["x"] = True
                if winding[1] != 0:
                    percolates["y"] = True
                if winding[2] != 0:
                    percolates["z"] = True

    unique_loops = []
    seen = set()
    for loop in loops:
        key = (loop["winding"], tuple(loop["nodes"]), loop["edge"])
        if key in seen:
            continue
        seen.add(key)
        unique_loops.append(loop)

    return percolates, unique_loops, acc_offsets


def detect_percolation_with_threshold(
    G: nx.MultiDiGraph,
    threshold: float,
    node_radius_attr: str = "pocket_radius",
    edge_radius_attr: str = "window_radius",
) -> Tuple[Dict[str, bool], List[Dict[str, object]], Dict[int, Offset], nx.MultiDiGraph]:
    """
    Convenience wrapper that filters nodes/edges below ``threshold`` before
    running :func:`percolation_winding_multidigraph`.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Pore graph following the deduplicated conventions.
    threshold : float
        Minimum radius (Å) for both nodes and edges to participate.
    node_radius_attr, edge_radius_attr : str
        Attribute names storing radii on nodes/edges respectively.
    """

    eligible_nodes = {
        n
        for n, data in G.nodes(data=True)
        if float(data.get(node_radius_attr, 0.0)) >= threshold
    }

    if not eligible_nodes:
        return {"x": False, "y": False, "z": False}, [], {}, nx.MultiDiGraph()

    H = nx.MultiDiGraph()
    for n in eligible_nodes:
        H.add_node(n, **G.nodes[n])

    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in eligible_nodes or v not in eligible_nodes:
            continue
        edge_radii = data.get("window_radii")
        if edge_radii:
            max_radius = max(float(val) for val in edge_radii)
        else:
            max_radius = float(data.get(edge_radius_attr, 0.0))
        if max_radius < threshold:
            continue
        H.add_edge(u, v, key=key, **data)

    if H.number_of_edges() == 0:
        return {"x": False, "y": False, "z": False}, [], {}, H

    percolates, loops, acc_offsets = percolation_winding_multidigraph(H)
    return percolates, loops, acc_offsets, H


def _maxima_pbc_data(maxima_info: np.ndarray):
    zero_mask = np.all(maxima_info[:, 4:7] == 0, axis=1)
    zero_rows = maxima_info[zero_mask]
    coords = zero_rows[:, 1:4].astype(int)
    maxima_pbc_dict = {tuple(coord): int(row[0]) for coord, row in zip(coords, zero_rows)}
    orig_map = np.array(
        [maxima_pbc_dict[tuple(map(int, row[1:4]))] for row in maxima_info], dtype=int
    )
    return maxima_pbc_dict, orig_map, zero_rows

def collect_periodic_edges(connections, maxima_info):
    import numpy as np

    label_to_idx = {int(row[0]): idx for idx, row in enumerate(maxima_info)}
    maxima_pbc = np.vstack([m[1:4] for m in maxima_info if np.all(m[4:7] == 0)])
    maxima_pbc_dict = {tuple(m): i + 1 for i, m in enumerate(maxima_pbc)}

    is_asymmetric = np.all(maxima_info[:, 4:7] == 0, axis=1)
    asymmetric_labels = set(int(maxima_info[i, 0]) for i in np.where(is_asymmetric)[0])
    origina_maxima_mapping = np.array(
        [maxima_pbc_dict[tuple(m[1:4])] for m in maxima_info], dtype=int
    )

    edge_map: dict[
        tuple[int, int, tuple[int, int, int]], dict[str, float | int | list[float]]
    ] = {}
    edges_debug = []

    for c in connections:
        if not c[1]:
            continue
        i_label, j_label = c[0]
        if not (i_label in asymmetric_labels or j_label in asymmetric_labels):
            continue

        idx_i = label_to_idx.get(i_label)
        idx_j = label_to_idx.get(j_label)
        if idx_i is None or idx_j is None:
            continue

        base_i = int(origina_maxima_mapping[idx_i])
        base_j = int(origina_maxima_mapping[idx_j])
        offset_i = maxima_info[idx_i, 4:7].astype(int)
        offset_j = maxima_info[idx_j, 4:7].astype(int)
        # Use the largest window observed for this connection so threshold filtering
        # later uses the true maximum rather than an average.
        window_radius = float(np.max(c[4]))

        start_base = base_i
        end_base = base_j
        start_offset = offset_i
        end_offset = offset_j
        if base_j < base_i:
            start_base, end_base = base_j, base_i
            start_offset, end_offset = offset_j, offset_i

        relative_offset = tuple(int(x) for x in (end_offset - start_offset))
        if start_base == end_base and relative_offset == (0, 0, 0):
            continue

        key = (start_base, end_base, relative_offset)
        entry = edge_map.setdefault(
            key,
            {
                "window_radii": [],
                "window_radius_min": window_radius,
                "window_radius_max": window_radius,
                "multiplicity": 0,
            },
        )
        entry["window_radii"].append(window_radius)
        entry["window_radius_min"] = min(entry["window_radius_min"], window_radius)
        entry["window_radius_max"] = max(entry["window_radius_max"], window_radius)
        entry["multiplicity"] = int(entry["multiplicity"]) + 1
        edges_debug.append(
            [
                i_label,
                j_label,
                start_base,
                end_base,
                *start_offset,
                *end_offset,
                *relative_offset,
                window_radius,
            ]
        )

    return edge_map, np.array(edges_debug) if edges_debug else None

# def collect_periodic_edges(connections, maxima_info):
#     import numpy as np

#     label_to_idx = {int(row[0]): idx for idx, row in enumerate(maxima_info)}
#     maxima_pbc = np.vstack([m[1:4] for m in maxima_info if np.all(m[4:7] == 0)])
#     maxima_pbc_dict = {tuple(m): i + 1 for i, m in enumerate(maxima_pbc)}

#     is_asymmetric = np.all(maxima_info[:, 4:7] == 0, axis=1)
#     asymmetric_labels = set(int(maxima_info[i, 0]) for i in np.where(is_asymmetric)[0])
#     origina_maxima_mapping = np.array(
#         [maxima_pbc_dict[tuple(m[1:4])] for m in maxima_info], dtype=int
#     )

#     edge_map: dict[tuple[int, int, tuple[int, int, int]], float] = {}
#     edges_debug = []

#     for c in connections:
#         if not c[1]:
#             continue
#         i_label, j_label = c[0]
#         if not (i_label in asymmetric_labels or j_label in asymmetric_labels):
#             continue

#         idx_i = label_to_idx[i_label]
#         idx_j = label_to_idx[j_label]
#         base_i = int(origina_maxima_mapping[idx_i])
#         base_j = int(origina_maxima_mapping[idx_j])
#         offset_i = maxima_info[idx_i, 4:7].astype(int)
#         offset_j = maxima_info[idx_j, 4:7].astype(int)
#         window_radius = float(np.mean(c[4]))

#         start_base = base_i
#         end_base = base_j
#         start_offset = offset_i
#         end_offset = offset_j
#         if base_j < base_i:
#             start_base, end_base = base_j, base_i
#             start_offset, end_offset = offset_j, offset_i

#         relative_offset = tuple(int(x) for x in (end_offset - start_offset))
#         if start_base == end_base and relative_offset == (0, 0, 0):
#             continue

#         key = (start_base, end_base, relative_offset)
#         if key not in edge_map:
#             edge_map[key] = window_radius
#             edges_debug.append(
#                 [
#                     i_label,
#                     j_label,
#                     start_base,
#                     end_base,
#                     *start_offset,
#                     *end_offset,
#                     *relative_offset,
#                     window_radius,
#                 ]
#             )
#         else:
#             edge_map[key] = min(edge_map[key], window_radius)

    return edge_map, np.array(edges_debug) if edges_debug else None
# def make_directed_graph_from_segmentation_results(results, dgrid=None):
#     """
#     Convenience helper to build a MultiDiGraph directly from the deduplicated
#     edge map returned by *make_edges_set_from_connections*.
#     """

#     import networkx as nx
#     import numpy as np
#     maxima_info = results["maxima_info"]
#     connections = results["connections"]
#     edge_map, edges_debug = collect_periodic_edges(connections, maxima_info)
#     if dgrid is None:
#         dgrid = results['dgrid']
#     dgrid_shape = dgrid.shape
#     frac_coords = (maxima_info[:, 1:4] / dgrid_shape) + maxima_info[:, 4:7]
#     G = nx.MultiDiGraph()
#     for (i, j, rel_offset), window_radius in edge_map.items():
#         G.add_edge(i, j, relative_offset=rel_offset, window_radius=window_radius)

#     for node in G.nodes():
#         G.nodes[node]["frac_pos"] = np.round(frac_coords[node - 1], 2)
#         voxel_idx = tuple(maxima_info[node - 1, 1:4].astype(int))
#         G.nodes[node]["pocket_radius"] = float(dgrid[voxel_idx])

#     return G

def make_directed_graph_from_segmentation_results(results, dgrid=None):
    """
    Convenience helper to build a MultiDiGraph directly from the deduplicated
    edge map returned by *make_edges_set_from_connections*.
    """

    import networkx as nx
    import numpy as np
    maxima_info = results["maxima_info"]
    connections = results["connections"]
    edge_map, edges_debug = collect_periodic_edges(connections, maxima_info)
    if dgrid is None:
        dgrid = results['dgrid']
    dgrid_shape = dgrid.shape
    frac_coords = (maxima_info[:, 1:4] / dgrid_shape) + maxima_info[:, 4:7]
    G = nx.MultiDiGraph()
    for (i, j, rel_offset), data in edge_map.items():
        window_radii = tuple(float(r) for r in data.get("window_radii", ()))
        if window_radii:
            window_radius_max = max(window_radii)
            window_radius_min = min(window_radii)
        else:
            window_radius_max = float(data.get("window_radius_max", 0.0))
            window_radius_min = float(data.get("window_radius_min", window_radius_max))
        G.add_edge(
            i,
            j,
            relative_offset=rel_offset,
            window_radius=window_radius_max,
            window_radius_max=window_radius_max,
            window_radius_min=window_radius_min,
            window_radii=window_radii,
            multiplicity=int(data.get("multiplicity", 1)),
        )

    for node in G.nodes():
        G.nodes[node]["frac_pos"] = np.round(frac_coords[node - 1], 2)
        voxel_idx = tuple(maxima_info[node - 1, 1:4].astype(int))
        G.nodes[node]["pocket_radius"] = float(dgrid[voxel_idx])

    return G


def build_base_region_labels(maxima_info: np.ndarray, region_labels: set, maxima_pbc_dict: Dict[Tuple[int, int, int], int]):
    base_region_labels: Dict[int, set] = {}
    for row in maxima_info:
        label = int(row[0])
        if label not in region_labels:
            continue
        base_idx = tuple(map(int, row[1:4]))
        base_label = maxima_pbc_dict.get(base_idx)
        if base_label is None:
            continue
        base_region_labels.setdefault(base_label, set()).add(label)
    return base_region_labels


def build_base_region_labels_from_results(results: Dict[str, object]) -> Dict[int, set]:
    maxima_info = np.asarray(results["maxima_info"])
    maxima_pbc_dict, _, _ = _maxima_pbc_data(maxima_info)
    regions = np.asarray(results["regions"])
    region_labels = set(int(x) for x in np.unique(regions) if int(x) != 0)
    return build_base_region_labels(maxima_info, region_labels, maxima_pbc_dict)


def _prepare_graph_and_labels(
    results: Dict[str, object],
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
) -> Tuple[nx.MultiDiGraph, Dict[int, set]]:
    if graph is None:
        graph = make_directed_graph_from_segmentation_results(results)
    if base_region_labels is None:
        base_region_labels = build_base_region_labels_from_results(results)
    return graph, base_region_labels


def _axes_from_winding(winding: Tuple[int, int, int]) -> List[str]:
    axes = []
    if winding[0] != 0:
        axes.append("x")
    if winding[1] != 0:
        axes.append("y")
    if winding[2] != 0:
        axes.append("z")
    return axes


def annotate_clusters(loop_list: List[Dict[str, object]]) -> Dict[Tuple[frozenset, Tuple[int, int, int]], int]:
    cluster_ids: Dict[Tuple[frozenset, Tuple[int, int, int]], int] = {}
    next_id = 1
    for loop in loop_list:
        key = (
            frozenset(loop["nodes"]),
            tuple(abs(int(w)) for w in loop["winding"]),
        )
        if key not in cluster_ids:
            cluster_ids[key] = next_id
            next_id += 1
        loop["cluster_id"] = cluster_ids[key]
    return cluster_ids


def percolation_all_axes_winding(
    results: Dict[str, object],
    threshold: float,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    node_radius_attr: str = "pocket_radius",
    edge_radius_attr: str = "window_radius",
    include_dangling: bool = False,
    expand_components: bool = True,
) -> Tuple[
    Dict[str, List[List[int]]],
    Dict[int, List[int]],
    Dict[int, Dict[str, object]],
]:
    graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
    percolates, loops, acc_offsets, filtered_graph = detect_percolation_with_threshold(
        graph, threshold, node_radius_attr=node_radius_attr, edge_radius_attr=edge_radius_attr
    )
    annotate_clusters(loops)
    graph_undirected = nx.Graph(filtered_graph.to_undirected(as_view=True))
    component_lookup: Dict[int, set] = {}
    if expand_components:
        for comp in nx.connected_components(graph_undirected):
            comp_set = set(comp)
            for node in comp_set:
                component_lookup[node] = comp_set

    axis_clusters: Dict[str, List[List[int]]] = {"x": [], "y": [], "z": []}
    cluster_details: Dict[int, List[int]] = {}
    cluster_axes: Dict[int, set] = {}
    cluster_region_sets: Dict[int, set] = {}
    cluster_node_sets: Dict[int, set] = {}

    for loop in loops:
        cid = loop["cluster_id"]
        cluster_axes.setdefault(cid, set()).update(_axes_from_winding(tuple(loop["winding"])))
        region_set = cluster_region_sets.setdefault(cid, set())
        for base in loop["nodes"]:
            region_set.update(base_region_labels.get(base, ()))
        cluster_node_sets.setdefault(cid, set()).update(loop["nodes"])

    cluster_metadata: Dict[int, Dict[str, object]] = {}
    for cid in cluster_region_sets:
        if expand_components:
            comp_nodes: set = set()
            for node in cluster_node_sets.get(cid, ()):
                comp_nodes.update(component_lookup.get(node, ()))
            comp_nodes = {n for n in comp_nodes if n in filtered_graph}
        else:
            comp_nodes = set(cluster_node_sets.get(cid, ()))
        if not comp_nodes:
            continue

        if expand_components:
            comp_subgraph = graph_undirected.subgraph(comp_nodes)
            subgraph_no_loops = comp_subgraph.copy()
            subgraph_no_loops.remove_edges_from(nx.selfloop_edges(subgraph_no_loops))
            try:
                core_nodes = set(nx.k_core(subgraph_no_loops, k=2).nodes())
            except nx.NetworkXError:
                core_nodes = set(subgraph_no_loops.nodes())
            if not core_nodes:
                core_nodes = comp_nodes
            full_nodes = comp_nodes
            dangling_nodes = full_nodes - core_nodes
        else:
            core_nodes = comp_nodes
            full_nodes = comp_nodes
            dangling_nodes = set()

        edge_set: set = set()
        for u, v, key in filtered_graph.edges(keys=True):
            if u in core_nodes and v in core_nodes:
                if u <= v:
                    edge_set.add((u, v, key))
                else:
                    edge_set.add((v, u, key))
        edge_set_full: set = set()
        for u, v, key in filtered_graph.edges(keys=True):
            if u in full_nodes and v in full_nodes:
                if u <= v:
                    edge_set_full.add((u, v, key))
                else:
                    edge_set_full.add((v, u, key))

        pruned_region_set = set()
        for node in core_nodes:
            pruned_region_set.update(base_region_labels.get(node, ()))
        full_region_set = set()
        for node in full_nodes:
            full_region_set.update(base_region_labels.get(node, ()))

        region_list = sorted(pruned_region_set)
        region_list_full = sorted(full_region_set)
        dangling_nodes_sorted = sorted(dangling_nodes)
        if include_dangling:
            nodes_out = sorted(full_nodes)
            edges_out = sorted(edge_set_full)
            regions_out = region_list_full
        else:
            nodes_out = sorted(core_nodes)
            edges_out = sorted(edge_set)
            regions_out = region_list
        cluster_details[cid] = region_list
        metadata: Dict[str, object] = {
            "regions": regions_out,
            "nodes": nodes_out,
            "edges": edges_out,
            "axes": sorted(cluster_axes.get(cid, ())),
            "regions_full": region_list_full,
            "nodes_full": sorted(full_nodes),
            "edges_full": sorted(edge_set_full),
            "dangling_nodes": dangling_nodes_sorted,
        }
        cluster_metadata[cid] = metadata
        for axis in cluster_axes.get(cid, ()):
            axis_clusters[axis].append(regions_out)

    return axis_clusters, cluster_details, cluster_metadata




def check_percolation_winding(
    results: Dict[str, object],
    threshold: float,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    include_dangling: bool = False,
) -> bool:
    axis_clusters, _, _ = percolation_all_axes_winding(
        results,
        threshold,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
    )
    return any(axis_clusters[axis] for axis in axis_clusters)


def _gather_threshold_candidates(graph: nx.MultiDiGraph) -> List[float]:
    node_vals = {float(data.get("pocket_radius", 0.0)) for _, data in graph.nodes(data=True)}
    edge_vals: set[float] = set()
    for _, _, data in graph.edges(data=True):
        radii = data.get("window_radii")
        if radii:
            edge_vals.update(float(val) for val in radii)
        else:
            edge_vals.add(float(data.get("window_radius", 0.0)))
    return sorted({val for val in node_vals.union(edge_vals) if val > 0})


def _default_mask_thickness(results: Dict[str, object]) -> float:
    if "mask_thickness" in results:
        return float(results["mask_thickness"])
    kwargs = results.get("kwargs", {})
    if isinstance(kwargs, dict) and "mask_thickness" in kwargs:
        return float(kwargs["mask_thickness"])
    return 0.0


def find_pore_limiting_radius_winding(
    results: Dict[str, object],
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    include_dangling: bool = False,
) -> Optional[float]:
    graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
    candidates = _gather_threshold_candidates(graph)
    if not candidates:
        return None

    min_default = _default_mask_thickness(results)
    if min_threshold is None:
        min_threshold = min_default
    min_threshold = max(min_threshold, min_default)
    if max_threshold is None:
        max_threshold = candidates[-1]

    filtered = [val for val in candidates if min_threshold <= val <= max_threshold]
    if not filtered:
        return None

    left, right = 0, len(filtered) - 1
    last_good = None
    while left <= right:
        mid = (left + right) // 2
        threshold = filtered[mid]
        axis_clusters, _, _ = percolation_all_axes_winding(
            results,
            threshold,
            graph=graph,
            base_region_labels=base_region_labels,
            include_dangling=include_dangling,
        )
        if any(axis_clusters[axis] for axis in axis_clusters):
            last_good = threshold
            left = mid + 1
        else:
            right = mid - 1
    final_radius = last_good
    if last_good is not None:
        (
            _,
            _,
            cluster_metadata,
        ) = percolation_all_axes_winding(
            results,
            last_good,
            graph=graph,
            base_region_labels=base_region_labels,
            include_dangling=include_dangling,
        )
        percolating_clusters = {
            cid: data
            for cid, data in cluster_metadata.items()
            if data.get("axes")
        }
        results["plr_clusters_winding"] = percolating_clusters
        results["plr_threshold_candidate_winding"] = last_good
        results["plr_radius_winding"] = last_good
    return final_radius


# def compute_percolation_history_from_results_winding(
#     results: Dict[str, object],
#     thresholds: Optional[Sequence[float]] = None,
#     step: float = 0.05,
#     min_threshold: Optional[float] = None,
#     max_threshold: Optional[float] = None,
#     graph: Optional[nx.MultiDiGraph] = None,
#     base_region_labels: Optional[Dict[int, set]] = None,
# ) -> List[Tuple[float, Dict[str, List[List[int]]]]]:
#     graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
#     if thresholds is None:
#         min_default = _default_mask_thickness(results)
#         if min_threshold is None:
#             min_threshold = min_default
#         min_threshold = max(min_threshold, min_default)
#         if max_threshold is None:
#             plr = find_pore_limiting_radius_winding(
#                 results,
#                 min_threshold=min_threshold,
#                 max_threshold=None,
#                 graph=graph,
#                 base_region_labels=base_region_labels,
#             )
#             if plr is None:
#                 return []
#             max_threshold = plr
#         thresholds = np.arange(min_threshold, max_threshold + 1e-9, step)

#     history: List[Tuple[float, Dict[str, List[List[int]]]]] = []
#     for threshold in thresholds:
#         axis_clusters, _ = percolation_all_axes_winding(
#             results, float(threshold), graph=graph, base_region_labels=base_region_labels
#         )
#         if any(axis_clusters[axis] for axis in axis_clusters):
#             history.append((float(threshold), axis_clusters))
#     return history

def update_results_with_percolation_history_winding(
    results: Dict[str, object],
    thresholds: Optional[Sequence[float]] = None,
    step: float = 0.05,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None,
    graph: Optional[nx.MultiDiGraph] = None,
    base_region_labels: Optional[Dict[int, set]] = None,
    include_dangling: bool = False,
    expand_components: bool = True,
) -> Dict[str, object]:
    graph, base_region_labels = _prepare_graph_and_labels(results, graph, base_region_labels)
    min_default = _default_mask_thickness(results)
    if min_threshold is None:
        min_threshold = min_default
    min_threshold = max(min_threshold, min_default)
    plr = find_pore_limiting_radius_winding(
        results,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        graph=graph,
        base_region_labels=base_region_labels,
        include_dangling=include_dangling,
    )
    if plr is None:
        results["percolation_history"] = []
        results["pld_locations"] = np.empty((0, 3), dtype=int)
        results.setdefault("plr_clusters_winding", {})
        return results

    if thresholds is None:
        if max_threshold is None:
            max_threshold = plr
        thresholds = np.arange(min_threshold, max_threshold + 1e-9, step)

    history: List[Tuple[float, Dict[str, List[List[int]]]]] = []
    for threshold in thresholds:
        axis_clusters, _, _ = percolation_all_axes_winding(
            results,
            float(threshold),
            graph=graph,
            base_region_labels=base_region_labels,
            include_dangling=include_dangling,
            expand_components=expand_components,
        )
        if any(axis_clusters[axis] for axis in axis_clusters):
            history.append((float(threshold), axis_clusters))
    results["percolation_history"] = history
    if results.get("dgrid") is not None:
        dgrid = np.asarray(results["dgrid"])
        pld_locations = np.argwhere(np.isclose(dgrid, plr, atol=1e-10))
    else:
        pld_locations = np.empty((0, 3), dtype=int)
    results["pld_locations"] = pld_locations
    return results



def compute_percolation_history_winding(
    results: Dict[str, object],
    include_dangling: bool = True,
    expand_components: bool = True,
    **kwargs,
) -> Tuple[List[Tuple[float, Dict[str, List[List[int]]]]], np.ndarray, Dict[int, Dict[str, object]]]:
    """Wrapper similar to ``compute_percolation_history2`` but also returns cluster metadata."""
    results = update_results_with_percolation_history_winding(
        results,
        include_dangling=include_dangling,
        expand_components=expand_components,
        **kwargs,
    )
    percolation_history = results["percolation_history"]
    pld_locations = results["pld_locations"]
    final_clusters = results.get("plr_clusters_winding", {})
    return percolation_history, pld_locations, final_clusters


def get_pore_dia_mgr_winding(
    mat_atoms=None,
    dgrid: Optional[np.ndarray] = None,
    kwargs: Optional[Dict[str, object]] = None,
    results: Optional[Dict[str, object]] = None,
) -> Tuple[float, float, float, np.ndarray]:
    if kwargs is None:
        kwargs = {}

    spacing = kwargs.get("spacing", 0.5)
    if results is None:
        if dgrid is None:
            if mat_atoms is None:
                raise ValueError("mat_atoms or results must be provided.")
            dgrid = mgr.dgrid_from_atoms_cpu(mat_atoms, spacing=spacing, block_size=50000)
        results = mgr.do_periodic_segmentation_rust(
            dgrid,
            mat_atoms,
            mask_thickness=kwargs.get("mask_thickness", 1.5),
            h=kwargs.get("h", 0.2),
            spacing=spacing,
            return_dgrid=True,
            threshold_abs=kwargs.get("threshold_abs", 1.5),
            min_distance=kwargs.get("min_distance", 1.5),
            return_padded_regions=True,
            extra_padding=kwargs.get("extra_padding", 2),
        )
    else:
        dgrid = np.asarray(results["dgrid"])

    graph = make_directed_graph_from_segmentation_results(results)
    base_region_labels = build_base_region_labels_from_results(results)

    lcd = float(dgrid.max() * 2)
    plr = find_pore_limiting_radius_winding(results, graph=graph, base_region_labels=base_region_labels)
    if plr is None:
        return lcd, 0.0, 0.0, dgrid

    pld = float(plr * 2)
    axis_clusters, _, _ = percolation_all_axes_winding(
        results, plr, graph=graph, base_region_labels=base_region_labels
    )
    percolating_labels = set()
    for groups in axis_clusters.values():
        for labels in groups:
            percolating_labels.update(labels)

    dfsd = 0.0
    if percolating_labels:
        regions = np.asarray(results["regions"])
        mask = np.isin(regions, list(percolating_labels))
        if mask.any():
            dfsd = float(dgrid[mask].max() * 2)

    return lcd, pld, dfsd, dgrid


CORNER_LOOP_GRID = np.array(
    [
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0],
    ],
    dtype=int,
)


STRIPED_WRAP_GRID = np.array(
    [
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
        [0, 1, 1, 1, 0,  0, 1, 0, 0],
        [0, 0, 0, 0, 0,  0, 1, 0, 0],
        [1, 1, 1, 1, 0,  0, 1, 1, 1],
        [0, 0, 0, 1, 0,  0, 0, 0, 0],
        [1, 1, 0, 1, 0,  0, 1, 1, 1],
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
        [0, 1, 0, 1, 0,  0, 1, 0, 0],
    ],
    dtype=int,
)


def make_periodic_graph_from_binary_grid(
    grid: np.ndarray,
    allow_diagonals: bool = False,
    node_radius: float = 1.0,
    edge_radius: float = 1.0,
) -> nx.MultiDiGraph:
    """
    Convert a binary 2D lattice into the directed periodic graph expected by the
    winding-number percolation routines.

    Parameters
    ----------
    grid : np.ndarray
        2D array where ``1`` (or truthy) entries represent pore voxels.  The
        lattice is treated as periodic along both axes.
    allow_diagonals : bool
        Include diagonal neighbors (8-connectivity) when ``True``.  Otherwise,
        only orthogonal neighbors (4-connectivity) are considered.
    node_radius, edge_radius : float
        Values stored on each node/edge so that downstream helpers expecting
        radii attributes continue to work.
    """

    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("Binary grid must be a 2D array.")

    rows, cols = grid.shape
    node_ids: Dict[Tuple[int, int], int] = {}
    G = nx.MultiDiGraph()

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 1:
                continue
            node_id = len(node_ids)
            node_ids[(r, c)] = node_id
            G.add_node(
                node_id,
                frac_pos=(r / rows, c / cols, 0.0),
                pocket_radius=node_radius,
            )

    base_dirs = [(0, 1), (1, 0)]
    if allow_diagonals:
        base_dirs.extend([(1, 1), (1, -1)])

    for (r, c), node_id in node_ids.items():
        for dr, dc in base_dirs:
            raw_r = r + dr
            raw_c = c + dc
            wrap_r = -1 if raw_r < 0 else (1 if raw_r >= rows else 0)
            wrap_c = -1 if raw_c < 0 else (1 if raw_c >= cols else 0)
            nbr_r = raw_r % rows
            nbr_c = raw_c % cols
            nbr_id = node_ids.get((nbr_r, nbr_c))
            if nbr_id is None:
                continue

            rel = (wrap_r, wrap_c, 0)
            u, v = node_id, nbr_id
            if u <= v:
                final_rel = rel
            else:
                u, v = v, u
                final_rel = tuple(-x for x in rel)
            G.add_edge(u, v, relative_offset=final_rel, window_radius=edge_radius)

    return G


def percolation_winding_on_binary_grid(
    grid: np.ndarray,
    allow_diagonals: bool = False,
    verbose: bool = False,
) -> Tuple[Dict[str, bool], List[Dict[str, object]], Dict[int, Offset]]:
    """
    Apply :func:`percolation_winding_multidigraph` to a binary periodic lattice.

    Parameters
    ----------
    grid : np.ndarray
        Binary 2D lattice treated as periodic in x/y.
    allow_diagonals : bool
        When ``True`` includes diagonal neighbors as valid connections.
    verbose : bool
        Print a human-readable summary when enabled.
    """

    G = make_periodic_graph_from_binary_grid(grid, allow_diagonals=allow_diagonals)
    percolates, loops, acc_offsets = percolation_winding_multidigraph(G)

    if verbose:
        print("Binary grid (1 = pore, 0 = empty):")
        print(np.asarray(grid, dtype=int))
        if loops:
            print("Detected winding loops:")
            for loop in loops:
                print(f"  winding={loop['winding']} nodes={loop['nodes']} edge={loop['edge']}")
        else:
            print("Detected winding loops: none (all cycles are contractible).")
        print("Per-axis percolation flags:", percolates)

    return percolates, loops, acc_offsets



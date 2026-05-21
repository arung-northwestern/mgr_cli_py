//! Expand aromatic ring templates to all subgraph embeddings (VF2), GIL released.

use hashbrown::HashSet;
use petgraph::algo::isomorphism::subgraph_isomorphisms_iter;
use petgraph::graph::{NodeIndex, UnGraph};
use pyo3::prelude::*;

fn bond_type_matches(a: &str, b: &str) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    if a == "aromatic" || b == "aromatic" {
        return true;
    }
    a == b
}

/// Undirected host graph; node weight = atom index, edge = bond_type (matches NetworkX).
fn build_host_graph(
    n_nodes: usize,
    edge_i: &[u32],
    edge_j: &[u32],
    edge_bond_types: &[String],
) -> (UnGraph<u32, String>, Vec<NodeIndex>) {
    let mut g: UnGraph<u32, String> = UnGraph::with_capacity(n_nodes, edge_i.len());
    let mut idx: Vec<NodeIndex> = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        idx.push(g.add_node(i as u32));
    }
    for k in 0..edge_i.len() {
        let a = edge_i[k] as usize;
        let b = edge_j[k] as usize;
        if a >= n_nodes || b >= n_nodes || a == b {
            continue;
        }
        let bt = edge_bond_types
            .get(k)
            .cloned()
            .unwrap_or_else(|| "single".to_string());
        g.add_edge(idx[a], idx[b], bt);
    }
    (g, idx)
}

fn build_pattern(
    template: &[u32],
    host: &UnGraph<u32, String>,
    host_indices: &[NodeIndex],
) -> UnGraph<u32, String> {
    let mut p: UnGraph<u32, String> = UnGraph::with_capacity(template.len(), template.len());
    let mut atom_to_pat: hashbrown::HashMap<u32, NodeIndex> = hashbrown::HashMap::new();
    for &atom in template {
        atom_to_pat.insert(atom, p.add_node(atom));
    }
    let n = template.len();
    if n >= 2 {
        for i in 0..n {
            for j in (i + 1)..n {
                let a = template[i] as usize;
                let b = template[j] as usize;
                let ia = host_indices[a];
                let ib = host_indices[b];
                if let Some(eid) = host.find_edge(ia, ib) {
                    let bt = host[eid].clone();
                    let pi = atom_to_pat[&template[i]];
                    let pj = atom_to_pat[&template[j]];
                    p.add_edge(pi, pj, bt);
                }
            }
        }
    }
    p
}

/// Expand chemically unique templates to all isomorphic embeddings in the host graph.
#[pyfunction]
#[pyo3(signature = (
    n_nodes,
    edge_i,
    edge_j,
    node_elements,
    edge_bond_types,
    template_cycles,
))]
pub fn expand_aromatic_cycle_embeddings(
    n_nodes: usize,
    edge_i: Vec<u32>,
    edge_j: Vec<u32>,
    node_elements: Vec<String>,
    edge_bond_types: Vec<String>,
    template_cycles: Vec<Vec<u32>>,
) -> PyResult<Vec<Vec<u32>>> {
    if n_nodes == 0 || template_cycles.is_empty() {
        return Ok(Vec::new());
    }

    let (host, host_indices) = build_host_graph(n_nodes, &edge_i, &edge_j, &edge_bond_types);
    let elements: Vec<String> = node_elements.iter().map(|s| s.to_string()).collect();

    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut out: Vec<Vec<u32>> = Vec::new();

    for template in template_cycles {
        if template.len() < 3 {
            continue;
        }
        let pattern = build_pattern(&template, &host, &host_indices);
        let n_pat = pattern.node_count();

        let mut node_match = |pat_atom: &u32, host_atom: &u32| -> bool {
            let p = *pat_atom as usize;
            let h = *host_atom as usize;
            p < elements.len() && h < elements.len() && elements[p] == elements[h]
        };
        let mut edge_match = |pe: &String, he: &String| bond_type_matches(he, pe);

        let pattern_ref = &pattern;
        let host_ref = &host;
        let mappings: Vec<Vec<usize>> = subgraph_isomorphisms_iter(
            &pattern_ref,
            &host_ref,
            &mut node_match,
            &mut edge_match,
        )
        .map(|iter| iter.collect())
        .unwrap_or_default();

        for mapping in mappings {
            if mapping.len() != n_pat {
                continue;
            }
            let cycle: Vec<u32> = mapping
                .iter()
                .map(|&host_ni| host[NodeIndex::new(host_ni)])
                .collect();
            let mut key = cycle.clone();
            key.sort_unstable();
            if seen.insert(key) {
                out.push(cycle);
            }
        }
    }

    out.sort_by_cached_key(|c| {
        let mut k = c.clone();
        k.sort_unstable();
        k
    });
    Ok(out)
}

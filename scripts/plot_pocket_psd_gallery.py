#!/usr/bin/env python3
"""Build per-material HTML gallery of pocket PSDs for visual typing decisions.

Per pocket:
  - central PSD (probe-accessible, pocket-wise)
  - first-shell neighbor PSDs (only edges with max window radius > 1.0 Å)
  - combined = (1 - w_shell) * central + w_shell * mean(neighbor central PSDs)
  - n_trusted_neighbors

Outputs interactive HTML under tmp_pocket_analysis/psd_gallery/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mgr_src"))
sys.path.insert(0, str(REPO / "python_cli"))

logging.basicConfig(level=logging.WARNING)

FIXTURE_DIR = REPO / "tests" / "fixtures" / "cif"
RASPA2_DIR = Path.home() / "repos" / "RASPA2" / "structures" / "mofs" / "cif"

MATERIALS: Dict[str, List[Path]] = {
    "NU-1000": [FIXTURE_DIR / "NU-1000_S1.cif"],
    "UMCM-1": [RASPA2_DIR / "UMCM-1.cif"],
    "CuBTC": [FIXTURE_DIR / "Cu-BTC.cif", RASPA2_DIR / "Cu-BTC.cif"],
    "ZIF-20": [RASPA2_DIR / "ZIF-20.cif"],
    "UiO-66": [RASPA2_DIR / "UIO-66.cif"],
    "IRMOF-1": [RASPA2_DIR / "IRMOF-1.cif"],
    "LTA": [REPO / "cifs" / "LTA96.cif"],
}

SPACING = 0.5
MASK_THICKNESS = 0.05
BINS = np.linspace(0, 50, 201)
BW = float(BINS[1] - BINS[0])
WINDOW_TRUST_MIN_A = 1.0
SHELL_WEIGHT = 0.25  # neighbor shell down-weighted vs central


def resolve_cif(name: str) -> Optional[Path]:
    for p in MATERIALS.get(name, []):
        if p.is_file():
            return p
    return None


def _import_mgr():
    import mgr
    return mgr


def _import_cli_funcs():
    try:
        from python_cli.mgrcli.cli import (
            compute_dgrid_rust,
            get_pbc_groups,
            python_do_periodic_segmentation_rust,
        )
    except ImportError:
        from mgrcli.cli import (
            compute_dgrid_rust,
            get_pbc_groups,
            python_do_periodic_segmentation_rust,
        )
    return compute_dgrid_rust, python_do_periodic_segmentation_rust, get_pbc_groups


def _norm_psd(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = y.sum()
    if s <= 0:
        return y
    return y / s


def _peak_diameter(x: np.ndarray, y: np.ndarray) -> float:
    """Dominant bin on pocket PSD x-axis (probe-accessible diameter, Å)."""
    if len(y) == 0:
        return 0.0
    return float(x[int(np.argmax(y))])


def _edge_max_window(data: dict) -> float:
    pw = data.get("path_weight")
    if pw == 0:
        return 0.0
    winfo = data.get("window_info")
    if not winfo:
        return 0.0
    wr = winfo.get("window_radii")
    if wr is None:
        return 0.0
    arr = np.asarray(wr, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def _edge_trusted(data: dict, min_win: float = WINDOW_TRUST_MIN_A) -> bool:
    """Direct connection with at least one window wider than min_win (Å)."""
    if data.get("path_weight") == 0:
        return False
    return _edge_max_window(data) > min_win


def extract_material(name: str, cif_path: Path) -> Dict[str, Any]:
    from ase.io import read as ase_read
    from skimage.measure import regionprops

    mgr = _import_mgr()
    compute_dgrid_rust, do_seg_rust, get_pbc_groups = _import_cli_funcs()

    atoms = ase_read(str(cif_path))
    dgrid = compute_dgrid_rust(atoms, SPACING)
    seg = do_seg_rust(
        dgrid, atoms, mask_thickness=MASK_THICKNESS, h=0.2, spacing=SPACING,
    )
    regions = seg["regions"]
    regions_pbc = seg.get("regions_pbc", regions)
    maxima_info = seg["maxima_info"]
    all_maxima = seg["all_maxima"]
    connections = seg.get("connections", [])
    unique_regions = seg.get("unique_regions", [])

    pbc_groups_dict = get_pbc_groups(unique_regions, maxima_info)
    pbc_groups = [v for v in pbc_groups_dict.values() if 0 not in v]

    all_maxima_radii = dgrid[tuple(maxima_info[:, 1:4].T)]
    maxima = all_maxima[: regions.max()]
    maxima_radii = all_maxima_radii[: regions.max()]
    connections_trimmed = [c for c in connections if np.any(np.isin(c[0], unique_regions))]

    rag = mgr.rag_from_connections(connections_trimmed, maxima)
    rag = mgr.add_window_info_to_rag(rag, connections_trimmed)
    rag = mgr.add_pbc_to_rag(rag, pbc_groups)
    rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, dgrid.shape, atoms)
    props = regionprops(regions)

    pocket_psds = None
    try:
        pocket_psds = mgr.compute_psds_pocketwise_rust(
            dgrid, regions_pbc, atoms,
            num_samples_per_region=1000, bins=BINS, return_fig=False,
        )
    except Exception as exc:
        return {"name": name, "error": f"pocket_psd failed: {exc}"}

    rag = mgr.add_psd_to_rag_with_pbc(
        rag, props, pbc_groups, BINS, atoms, dgrid.shape, pocket_psds=pocket_psds,
    )

    region_to_rep: Dict[int, int] = {}
    rep_to_pi: Dict[int, int] = {}
    for pi, group in enumerate(pbc_groups):
        rep = min(group)
        rep_to_pi[rep] = pi
        for reg in group:
            region_to_rep[int(reg)] = rep

    # pocket rep -> normalized central PSD
    central: Dict[int, np.ndarray] = {}
    peak_map: Dict[int, float] = {}
    for pi, group in enumerate(pbc_groups):
        node_id = min(group)
        nd = rag.nodes[node_id]
        x = np.asarray(nd.get("psd_x_pbc", BINS), dtype=float)
        y = _norm_psd(np.asarray(nd.get("psd_y_pbc", []), dtype=float))
        if y.size == 0:
            continue
        central[node_id] = y
        peak_map[node_id] = _peak_diameter(x, y)

    # trusted pocket-pocket adjacency + per-edge max window
    pocket_adj: Dict[int, List[Tuple[int, float]]] = {min(g): [] for g in pbc_groups}
    for u, v, data in rag.edges(data=True):
        if not _edge_trusted(data):
            continue
        wmax = _edge_max_window(data)
        ru = region_to_rep.get(int(u))
        rv = region_to_rep.get(int(v))
        if ru is None or rv is None or ru == rv:
            continue
        pu, pv = ru, rv
        if pv not in dict(pocket_adj.get(pu, [])):
            pocket_adj.setdefault(pu, []).append((pv, wmax))
        if pu not in dict(pocket_adj.get(pv, [])):
            pocket_adj.setdefault(pv, []).append((pu, wmax))

    x_axis = np.asarray(
        rag.nodes[min(pbc_groups[0])].get("psd_x_pbc", BINS), dtype=float,
    )

    pockets_out: List[Dict[str, Any]] = []
    combined_mat: List[np.ndarray] = []

    for pi, group in enumerate(pbc_groups):
        rep = min(group)
        y0 = central.get(rep)
        if y0 is None:
            continue
        nbr_reps = [nb for nb, _w in pocket_adj.get(rep, []) if nb in central]
        n_trusted = len(nbr_reps)
        if nbr_reps:
            y_shell = np.mean([central[nb] for nb in nbr_reps], axis=0)
        else:
            y_shell = np.zeros_like(y0)
        y_comb = (1.0 - SHELL_WEIGHT) * y0 + SHELL_WEIGHT * y_shell
        y_comb = _norm_psd(y_comb)
        combined_mat.append(y_comb)

        nbr_peaks = [round(peak_map[nb], 3) for nb in nbr_reps[:12]]
        pk_d = peak_map[rep]
        pockets_out.append({
            "id": int(rep),
            "pi": pi,
            "peak_d_A": round(pk_d, 4),
            "peak_r_A": round(0.5 * pk_d, 4),
            "n_trusted_neighbors": n_trusted,
            "neighbor_peaks_A": nbr_peaks,
            "x": x_axis.tolist(),
            "central": y0.tolist(),
            "shell_mean": y_shell.tolist(),
            "combined": y_comb.tolist(),
        })

    # pairwise JS on combined PSDs
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import jensenshannon, squareform

    n = len(combined_mat)
    js = np.zeros((n, n), dtype=float)
    labels = [f"P{p['id']} d={p['peak_d_A']:.2f}" for p in pockets_out]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(jensenshannon(combined_mat[i], combined_mat[j], base=2.0))
            js[i, j] = js[j, i] = d

    dendro = None
    if n >= 2:
        condensed = squareform(js, checks=False)
        z = linkage(condensed, method="single")
        # scipy dendrogram coords for plotly
        dn = dendrogram(z, labels=labels, no_plot=True)
        dendro = {
            "icoord": dn["icoord"],
            "dcoord": dn["dcoord"],
            "leaves": dn["leaves"],
            "ivl": dn["ivl"],
        }

    return {
        "name": name,
        "cif": str(cif_path),
        "n_pockets": len(pockets_out),
        "spacing_A": SPACING,
        "mask_A": MASK_THICKNESS,
        "window_trust_min_A": WINDOW_TRUST_MIN_A,
        "shell_weight": SHELL_WEIGHT,
        "pockets": pockets_out,
        "js_matrix": js.tolist(),
        "labels": labels,
        "dendrogram": dendro,
    }


def _html_page(materials: List[Dict[str, Any]]) -> str:
    payload = json.dumps(materials)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Pocket PSD gallery</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e8eaed; }}
    header {{ padding: 12px 20px; background: #1a1d24; border-bottom: 1px solid #333; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 20px; background: #14171c; }}
    nav button {{ padding: 8px 14px; border: 1px solid #444; background: #22262e; color: #eee; border-radius: 6px; cursor: pointer; }}
    nav button.active {{ background: #3d5afe; border-color: #3d5afe; }}
    main {{ padding: 12px 16px 40px; }}
    .meta {{ color: #9aa0a6; font-size: 13px; margin-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    @media (min-width: 1100px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
    .panel {{ background: #1a1d24; border-radius: 8px; padding: 8px; min-height: 380px; }}
    .panel h3 {{ margin: 4px 8px 0; font-size: 14px; font-weight: 600; }}
    .full {{ grid-column: 1 / -1; }}
    #detail {{ margin-top: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1 style="margin:0;font-size:20px">Pocket PSD gallery</h1>
    <p style="margin:6px 0 0;font-size:13px;color:#9aa0a6">
      Central PSD (solid) · shell-mean of trusted neighbors (dotted, w={SHELL_WEIGHT}) · combined (bold).
      Trusted edge = path_weight 1 and max window &gt; {WINDOW_TRUST_MIN_A} Å.
    </p>
  </header>
  <nav id="tabs"></nav>
  <main>
    <div class="meta" id="meta"></div>
    <div class="grid">
      <div class="panel full"><h3>All pockets — central PSD (normalized)</h3><div id="plot-central"></div></div>
      <div class="panel full"><h3>All pockets — combined PSD (central + down-weighted shell)</h3><div id="plot-combined"></div></div>
      <div class="panel"><h3>Pairwise JS distance (combined PSD)</h3><div id="plot-heatmap"></div></div>
      <div class="panel"><h3>Single-linkage dendrogram (JS)</h3><div id="plot-dendro"></div></div>
      <div class="panel full"><h3>Selected pocket detail</h3>
        <label class="meta">Pocket <select id="pocket-select"></select></label>
        <div id="detail-meta" class="meta"></div><div id="plot-detail"></div></div>
    </div>
  </main>
  <script>
    const DATA = {payload};
    let cur = 0;

    function el(id) {{ return document.getElementById(id); }}

    function buildTabs() {{
      const nav = el('tabs');
      nav.innerHTML = '';
      DATA.forEach((m, i) => {{
        const b = document.createElement('button');
        b.textContent = m.error ? m.name + ' (err)' : m.name;
        b.onclick = () => {{ cur = i; render(); }};
        if (i === cur) b.className = 'active';
        b.dataset.i = i;
        nav.appendChild(b);
      }});
    }}

    function tracesOverlay(pockets, key, opacity=0.45) {{
      return pockets.map(p => ({{
        x: p.x, y: p[key], mode: 'lines',         name: `P${{p.id}} d=${{p.peak_d_A}} n=${{p.n_trusted_neighbors}}`,
        opacity, line: {{ width: 1.2 }},
        hovertemplate: `Pocket ${{p.id}}<br>peak d=${{p.peak_d_A}} Å (r=${{p.peak_r_A}})<br>trusted nbrs=${{p.n_trusted_neighbors}}<extra></extra>`,
      }}));
    }}

    function render() {{
      buildTabs();
      document.querySelectorAll('nav button').forEach((b,i) => b.classList.toggle('active', i===cur));
      const m = DATA[cur];
      if (m.error) {{
        el('meta').textContent = m.name + ': ' + m.error;
        return;
      }}
      el('meta').textContent = `${{m.name}} · ${{m.n_pockets}} pockets · ${{m.cif}} · spacing=${{m.spacing_A}} mask=${{m.mask_A}}`;

      const layoutBase = {{
        paper_bgcolor: '#1a1d24', plot_bgcolor: '#1a1d24',
        font: {{ color: '#e8eaed' }},
        margin: {{ l: 50, r: 20, t: 10, b: 45 }},
        xaxis: {{ title: 'Pore diameter (Å)', gridcolor: '#333' }},
        yaxis: {{ title: 'Probability', gridcolor: '#333' }},
        showlegend: m.n_pockets <= 24,
        legend: {{ font: {{ size: 9 }} }},
      }};

      Plotly.newPlot('plot-central', tracesOverlay(m.pockets, 'central'), layoutBase, {{responsive:true}});

      Plotly.newPlot('plot-combined', tracesOverlay(m.pockets, 'combined', 0.5), layoutBase, {{responsive:true}});

      Plotly.newPlot('plot-heatmap', [{{
        z: m.js_matrix, x: m.labels, y: m.labels, type: 'heatmap',
        colorscale: 'Viridis', colorbar: {{ title: 'JS' }},
      }}], {{
        ...layoutBase, xaxis: {{ ...layoutBase.xaxis, tickangle: -45, tickfont: {{size:9}} }},
        yaxis: {{ ...layoutBase.yaxis, tickfont: {{size:9}} }},
      }}, {{responsive:true}});

      if (m.dendrogram) {{
        const shapes = [];
        const d = m.dendrogram;
        for (let k = 0; k < d.icoord.length; k++) {{
          const xs = d.icoord[k].map(v => v / 10);
          const ys = d.dcoord[k];
          for (let j = 0; j < xs.length-1; j++) {{
            shapes.push({{ type:'line', x0:xs[j], x1:xs[j+1], y0:ys[j], y1:ys[j+1], line:{{color:'#8ab4f8',width:1}} }});
          }}
        }}
        Plotly.newPlot('plot-dendro', [], {{
          ...layoutBase, shapes,
          xaxis: {{ ...layoutBase.xaxis, title: 'Pocket', tickmode: 'array', tickvals: d.leaves.map((_,i)=>i), ticktext: d.ivl, tickangle: -45 }},
          yaxis: {{ ...layoutBase.yaxis, title: 'JS linkage' }},
        }}, {{responsive:true}});
      }} else {{
        el('plot-dendro').innerHTML = '<p style="padding:12px">Need ≥2 pockets for dendrogram.</p>';
      }}

      const sel = el('pocket-select');
      sel.innerHTML = '';
      m.pockets.forEach((p, idx) => {{
        const o = document.createElement('option');
        o.value = idx; o.textContent = `P${{p.id}} d=${{p.peak_d_A}} n=${{p.n_trusted_neighbors}}`;
        sel.appendChild(o);
      }});
      sel.onchange = () => renderDetail(m, m.pockets[Number(sel.value)]);
      renderDetail(m, m.pockets[0]);
    }}

    function renderDetail(m, p) {{
      el('detail-meta').textContent = `Pocket ${{p.id}} · peak d=${{p.peak_d_A}} Å (r=${{p.peak_r_A}}) · trusted neighbors=${{p.n_trusted_neighbors}} · nbr peak d=[${{(p.neighbor_peaks_A||[]).join(', ')}}] Å`;
      const t = [
        {{ x:p.x, y:p.central, name:'central', line:{{color:'#81c995',width:3}} }},
        {{ x:p.x, y:p.shell_mean, name:`shell mean (w=${{m.shell_weight}})`, line:{{color:'#f28b82',width:2,dash:'dot'}} }},
        {{ x:p.x, y:p.combined, name:'combined', line:{{color:'#8ab4f8',width:3}} }},
      ];
      Plotly.newPlot('plot-detail', t, {{
        paper_bgcolor:'#1a1d24', plot_bgcolor:'#1a1d24', font:{{color:'#e8eaed'}},
        margin:{{l:50,r:20,t:10,b:45}},
        xaxis:{{title:'Pore diameter (Å)', gridcolor:'#333'}},
        yaxis:{{title:'Probability', gridcolor:'#333'}},
      }}, {{responsive:true}});
    }}

    render();
  </script>
</body>
</html>
"""


def main() -> None:
    out_dir = REPO / "tmp_pocket_analysis" / "psd_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    materials: List[Dict[str, Any]] = []
    for name in MATERIALS:
        cif = resolve_cif(name)
        if cif is None:
            materials.append({"name": name, "error": "CIF not found"})
            print(f"{name}: MISSING CIF")
            continue
        print(f"{name}: extracting from {cif} ...")
        try:
            rec = extract_material(name, cif)
        except Exception as exc:
            rec = {"name": name, "error": str(exc)}
        materials.append(rec)
        if rec.get("error"):
            print(f"  ERROR: {rec['error']}")
        else:
            print(f"  {rec['n_pockets']} pockets, JS matrix {len(rec['js_matrix'])}x{len(rec['js_matrix'])}")

    json_path = out_dir / "pocket_psd_data.json"
    json_path.write_text(json.dumps(materials, indent=2))

    html_path = out_dir / "index.html"
    html_path.write_text(_html_page(materials))

    print(f"\nWrote {html_path}")
    print(f"     {json_path}")
    print("Open index.html in a browser.")


if __name__ == "__main__":
    main()

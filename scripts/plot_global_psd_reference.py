#!/usr/bin/env python3
"""
Plot global uniform PSD (same path as interactive app: rust Fun + np.histogram on diameter).

Usage (from repo root, conda env with mofography):

  conda run --no-capture-output -n mgr_env python scripts/plot_global_psd_reference.py \\
      --cif /path/to/file.cif --out abavij_psd_ref

Writes <out>.html (Plotly, matches app style: counts vs bin centers).
Writes <out>.png if matplotlib is available.

Example:
  python scripts/plot_global_psd_reference.py \\
      --cif ../mofography/examples/TestCIFs/ABAVIJ_clean.cif \\
      --out artifacts/abavij_global_psd
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cif", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output path without suffix")
    p.add_argument("--spacing", type=float, nargs="+", default=[0.25, 0.6])
    p.add_argument("--samples", type=int, default=25_000)
    p.add_argument("--bins", type=int, default=200)
    p.add_argument("--dmax", type=float, default=50.0)
    args = p.parse_args()

    import numpy as np
    import plotly.graph_objects as go
    from ase.io import read
    import mofography as mf

    atoms = read(str(args.cif))
    stems = "_".join(f"{s:g}".replace(".", "p") for s in args.spacing)
    title = (
        f"Global PSD (uniform) reference — {args.cif.name} — spacing={args.spacing} Å "
        f"bins=[0,{args.dmax}] n={args.bins}"
    )

    fig = go.Figure()
    for spacing in args.spacing:
        dgrid = mf.dgrid_from_atoms_rust(atoms, float(spacing))
        bin_edges = np.linspace(0.0, float(args.dmax), int(args.bins) + 1)
        hist = mf.compute_psd_uniform_rust(
            dgrid,
            atoms,
            num_samples_total=int(args.samples),
            bins=bin_edges,
            density=False,
            return_fig=False,
        )
        hist = np.asarray(hist, dtype=float)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=hist,
                mode="lines",
                name=f"spacing {spacing:g} Å (grid {dgrid.shape})",
                line=dict(width=2),
            )
        )
        f0 = float(hist[0]) / max(float(hist.sum()), 1.0)
        print(
            f"spacing={spacing}: first_bin_counts={int(hist[0])} "
            f"fraction={100*f0:.2f}% sum={int(hist.sum())}"
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Pore diameter (Å)",
        yaxis_title="Counts",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1000,
        height=600,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    html_path = args.out.with_suffix(".html")
    fig.write_html(str(html_path))
    print(f"Wrote {html_path.resolve()}")

    try:
        import matplotlib.pyplot as plt

        fig_m, ax = plt.subplots(figsize=(10, 5))
        for tr in fig.data:
            ax.plot(tr.x, tr.y, label=tr.name, linewidth=2)
        ax.set_xlabel("Pore diameter (Å)")
        ax.set_ylabel("Counts")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        png_path = args.out.with_suffix(".png")
        fig_m.tight_layout()
        fig_m.savefig(png_path, dpi=150)
        plt.close(fig_m)
        print(f"Wrote {png_path.resolve()}")
    except Exception as e:
        print(f"Skipping PNG (matplotlib save failed): {e}")


if __name__ == "__main__":
    main()

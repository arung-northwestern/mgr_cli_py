#!/usr/bin/env python3
"""Seed memvid memory project-mgr_cli_py with aromatic-rings handoff knowledge."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEMVID_PATH = Path.home() / ".memvid" / "project-mgr_cli_py.mv2"


def main() -> None:
    from memvid_sdk import create, use

    MEMVID_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MEMVID_PATH.exists():
        create(str(MEMVID_PATH), enable_vec=True, enable_lex=True)

    m = use("basic", str(MEMVID_PATH))

    chunks = [
        (
            "mgr_cli_py aromatic goal",
            "Show ALL aromatic ring VF2 embeddings at distinct atom sites. NOT chem-unique-only. "
            "Visual parity OK. Phase 1-2: cache, bond tolerance, Rust VF2.",
        ),
        (
            "mgr_cli_py aromatic pipeline",
            "mgr_src/mgr/analyze.py: cycle_basis 5-9; _edge_is_aromatic; dedupe; "
            "find_chemically_unique_cycles; VF2 expand. Short-circuit if "
            "len(chem_unique)==len(unique_cycles). Legacy: find_aromatic_cycles_in_sgraph_legacy.",
        ),
        (
            "mgr_cli_py aromatic cache",
            "compute_aromatic_cycles -> ase_atoms.info aromatic_cycles + aromatic_cycles_key. "
            "interactive_app.py after bonds; movies make_aromatic_rings_trace; "
            "threejs add_aromatic_rings(cycles=...).",
        ),
        (
            "mgr_cli_py aromatic rust",
            "mgr_src/mgrcli_rust/src/aromatic.rs expand_aromatic_cycle_embeddings. "
            "CRITICAL: pattern edges = ALL host bonds among template atoms (basis order != bond walk). "
            "UnGraph. maturin develop --release. MGR_AROMATIC_RUST=1.",
        ),
        (
            "mgr_cli_py aromatic MIL-101 bottleneck",
            "find_chemically_unique_cycles ~99% of detection (~2.6s). Rust expand ~0.02s. "
            "168 unique, 1 template, 168 final. Profile: scripts/profile_aromatic_rings_nu1000.py --chart.",
        ),
        (
            "mgr_cli_py aromatic next",
            "Priority: optimize find_chemically_unique_cycles. Then instanced Three.js rings, Plotly mesh.",
        ),
    ]
    for title, text in chunks:
        m.put(title=title, text=text, tags=["mgr_cli_py", "aromatic"], search_text=text)

    for rel in (
        "docs/handoff_aromatic_rings.md",
        "docs/plan_aromatic_rings_detection.md",
    ):
        path = REPO / rel
        if path.is_file():
            body = path.read_text()
            m.put(text=body, title=path.stem, label="md", tags=["mgr_cli_py"], search_text=body)

    m.add_memory_cards(
        [
            {
                "entity": "mgr_cli_py_aromatic",
                "slot": "bottleneck",
                "value": "find_chemically_unique_cycles dominates MIL-101",
                "kind": "Fact",
            },
            {
                "entity": "mgr_cli_py_aromatic",
                "slot": "rust_critical",
                "value": "pattern edges all template atom pairs bonded in host",
                "kind": "Fact",
            },
        ]
    )
    m.doctor(quiet=True)
    print(f"Seeded {MEMVID_PATH}")


if __name__ == "__main__":
    main()

"""Parallel batch processing for mgrcli (Dask local cluster)."""

from __future__ import annotations

import logging
import tarfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("mgrcli.batch")


def collect_cif_paths(
    *,
    positional: Optional[Sequence[str]] = None,
    input_file: Optional[str] = None,
    input_dir: Optional[str] = None,
) -> List[Path]:
    """Collect CIF paths from positional args, -i, and/or --input-dir."""
    import glob

    cif_files: List[Path] = []
    if input_file:
        cif_files.append(Path(input_file))
    if positional:
        for pattern in positional:
            path = Path(pattern)
            if path.is_file():
                cif_files.append(path)
            elif "*" in pattern:
                cif_files.extend(Path(f) for f in glob.glob(pattern))
    if input_dir:
        root = Path(input_dir)
        if root.is_dir():
            cif_files.extend(sorted(root.rglob("*.cif")))
    # de-dupe preserving order
    seen: set[str] = set()
    unique: List[Path] = []
    for p in cif_files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def batch_process_one(
    cif_path: str,
    output_dir: str,
    config: Dict[str, Any],
    flags: Dict[str, Any],
) -> Tuple[str, bool, Optional[str]]:
    """Worker entry: process one CIF with the same outputs as ``mgrcli process``."""
    from mgrcli.cli import MCLI

    cif = Path(cif_path)
    out = Path(output_dir)
    args = Namespace(**flags)
    args.output_dir = str(out)
    mcli = MCLI()
    try:
        mcli._process_one_cif_file(cif, config, args, out)
        return (cif.stem, True, None)
    except Exception as exc:
        logger.exception("Batch failed for %s", cif)
        return (cif.stem, False, str(exc))


def run_batch(
    cif_files: Sequence[Path],
    output_dir: Path,
    config: Dict[str, Any],
    args,
    *,
    workers: int = 4,
    tar_output: bool = False,
) -> List[Tuple[str, bool, Optional[str]]]:
    """Run ``mgrcli process`` logic in parallel via Dask."""
    if not cif_files:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    flags = {k: v for k, v in vars(args).items() if k != "command"}

    try:
        from dask.distributed import Client, LocalCluster
        from dask import delayed
    except ImportError as exc:
        raise RuntimeError(
            "Install dask[distributed] for mgrcli batch (included in mgrcli dependencies)."
        ) from exc

    cluster = LocalCluster(n_workers=max(1, int(workers)), threads_per_worker=1)
    client = Client(cluster)
    results: List[Tuple[str, bool, Optional[str]]] = []
    try:
        delayed_jobs = [
            delayed(batch_process_one)(str(p), str(output_dir), dict(config), dict(flags))
            for p in cif_files
        ]
        futures = client.compute(delayed_jobs)
        for fut in futures:
            try:
                results.append(fut.result(timeout=3600))
            except Exception as exc:
                results.append(("?", False, str(exc)))
    finally:
        client.close()
        cluster.close()

    if tar_output:
        tar_path = output_dir / "batch_results.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for item in output_dir.iterdir():
                if item.is_file() and item.name != tar_path.name:
                    tar.add(item, arcname=item.name)
                elif item.is_dir():
                    tar.add(item, arcname=item.name)
        print(f"Batch archive written to: {tar_path}")

    return results

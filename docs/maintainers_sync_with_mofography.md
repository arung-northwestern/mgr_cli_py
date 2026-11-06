# Keeping mgr_cli_py in sync with mofography

`mofography` is the source of truth: it is tested and updated continuously. This
repo is the CLI + web-app fork (adds caching, instanced spheres, Spaces policy).
Shared modules should track upstream; local-only behavior should live in small
wrapper modules so shared files can be copied wholesale.

## Why not a git submodule
- Package names differ (`mgr` vs `mofography`, `mgrcli_rust` vs `mgr_rust`).
- Several module bodies diverge (cache, instanced rings, web helpers).
- This fork needs only some functions from each module.

## Plan
0. **Default: manual, AI-assisted sync with hunk review.** Diff each shared module against
   upstream (after the rename), review every hunk, and apply only what is safe. Local code is
   sometimes newer (e.g. `infer_segmentation_spacing`), so never copy files blindly.
1. **Optional: report-only sync script** (`scripts/sync_from_mofography.py`):
   - manifest lists upstream file -> local file (start: `pocket_typing.py`,
     `helper.py`, `cif_io.py`);
   - rewrites imports for the rename;
   - writes a diff report only; it never overwrites files.
2. **Shrink divergence**: move local-only code (cache, instanced rings, Spaces
   policy) into wrapper modules, then sync shared files wholesale.
3. **Real dependency (later)**: if mofography becomes pip-installable from git,
   import it instead of vendoring (needs the package rename resolved).

## Open divergences (check before each sync)
- `WINDOW_TRUST_MIN_A`: upstream 0.5, here 1.0.
- `infer_segmentation_spacing` shape check exists only here.
- `estimate_h` cap: upstream 0.4, here 0.2 (matches the web `h` default).
- Rust crate: this repo builds `mgr_src/mgrcli_rust`; upstream rtree dgrid entry
  points are not exported here yet.

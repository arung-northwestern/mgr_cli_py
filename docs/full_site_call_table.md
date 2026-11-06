# MOFography full-site call table

Inventory for standalone migration. **Tier:** Hot = interactive/CLI default path; Warm = CLI optional / analysis; Cold = HTS / ML / dev only.

**Legend — Action:** `RUST` already in mgrcli_rust · `KEEP` stay in Python v1 · `PORT` move to Rust · `DROP` remove from slim v1 · `DEFER` later · `SUB` subprocess/sidecar only

---

## mgrcli_rust (`src/lib.rs`)

| Symbol | Tier | Action |
|--------|------|--------|
| `dgrid_from_atoms` | Hot | RUST |
| `egrid_from_atoms` | Warm | RUST |
| `periodic_gaussian_filter` | Hot | RUST |
| `seeded_watershed_bucket` | Hot | RUST |
| `seeded_watershed_det` | Hot | RUST |
| `seeded_watershed_det_parallel` | Hot | RUST |
| `seeded_watershed` | Warm | RUST |
| `seeded_watershed_masked` | Warm | RUST |
| `detect_maxima_h_morph` | Hot | RUST |
| `detect_maxima_h_morph_opening` | Warm | RUST |
| `move_maxima_to_peaks` | Hot | RUST |
| `detect_maxima_h_morph2` | Warm | RUST |
| `detect_maxima_h_morph_simple` | Warm | RUST |
| `detect_maxima_h_morph_no_spacing` | Warm | RUST |
| `detect_maxima_adaptive_geometry` | Warm | RUST |
| `detect_maxima_selective_smoothing` | Warm | RUST |
| `detect_maxima_selective_smoothing_parallel` | Warm | RUST |
| `detect_maxima_selective_smoothing_parallel2` | Warm | RUST |
| `detect_maxima_percolation_guided` | Warm | RUST |
| `detect_maxima_voronoi_based` | Warm | RUST |
| `detect_maxima_multiscale_log` | Warm | RUST |
| `detect_maxima_adaptive_h` | Warm | RUST |
| `window_connections` | Hot | RUST |
| `compute_pore_sizes_uniform` | Hot | RUST |
| `compute_pore_sizes_for_psd_pocketwise` | Hot | RUST |
| `find_pore_limiting_radius` | Warm | RUST (grid PLD; not winding) |
| `covalent_bonds` | Hot | RUST |
| `rayon_num_threads` | Cold | RUST |

## mgr.rust_fun

| Symbol | Tier | Action |
|--------|------|--------|
| `dgrid_from_atoms_rust` | Hot | KEEP → wraps RUST |
| `do_periodic_segmentation_rust` | Hot | KEEP → PORT orchestration to Rust |
| `egrid_from_atoms_rust` | Warm | KEEP |
| `compute_psd_uniform_rust` | Hot | KEEP |
| `compute_psds_pocketwise_rust` | Hot | KEEP |
| `compute_pore_sizes_*_rust` | Hot | KEEP |
| `compute_accessible_psd_from_results` | Warm | KEEP |
| `compute_psd_global_accessible` | Warm | KEEP |
| `compute_psds_pocketwise_accessible` | Warm | KEEP |

## mgr.winding_percolation

| Symbol | Tier | Action |
|--------|------|--------|
| `percolation_winding_multidigraph` | Hot | PORT |
| `detect_percolation_with_threshold` | Hot | PORT |
| `collect_periodic_edges` | Hot | PORT |
| `make_directed_graph_from_segmentation_results` | Hot | PORT |
| `find_pore_limiting_radius_winding` | Hot | PORT |
| `percolation_all_axes_winding` | Hot | PORT |
| `compute_percolation_history_winding` | Hot | PORT |
| `get_pore_dia_mgr_winding` | Hot | KEEP (SUB until PORT) |
| `update_results_with_percolation_history_winding` | Hot | KEEP |
| `check_percolation_winding` | Warm | PORT |
| `make_periodic_graph_from_binary_grid` | Cold | DEFER (tests) |

## mgr.analyze (bonds + chemistry)

| Symbol | Tier | Action |
|--------|------|--------|
| `update_with_covalent_bonds` | Hot | KEEP (freud + RUST) |
| `aabb_neighbors_on_atoms` | Hot | KEEP |
| `_update_with_covalent_bonds_python` | Warm | DROP (pymatgen fallback) |
| `voronoi_on_atoms` | Cold | DROP |
| `find_functional_groups` | Cold | DROP |
| `find_cycles_in_sgraph` | Cold | DROP |
| `mesh_for_atoms` / `batched_mesh_for_atoms` | Warm | DEFER (viz) |

## mgr.threejs_plotting

| Symbol | Tier | Action |
|--------|------|--------|
| `show_mof_threejs` | Hot | KEEP (v1 viewer) |
| `show_percolation_threejs` | Warm | KEEP |

## python_cli/mgrcli/cli.py

| Symbol | Tier | Action |
|--------|------|--------|
| `compute_dgrid_rust` | Hot | KEEP |
| `python_do_periodic_segmentation_rust` | Hot | KEEP |
| `compute_pore_diameter_analysis` | Warm | KEEP (legacy percolation fallback) |
| `compute_psd_analysis` | Warm | KEEP |
| `compute_auto_parameters` | Warm | KEEP |
| `compute_surface_area_analysis` | Warm | KEEP |
| `compute_void_volume_analysis` | Warm | KEEP |
| `main` | Warm | KEEP (CLI entry) |

## python_cli/mgrcli/plotting.py

| Symbol | Tier | Action |
|--------|------|--------|
| `show_mof` | Warm | DEFER (Plotly) |
| `prepare_results_for_plot` | Hot | KEEP |
| `plot_framework_as_mesh` | Warm | DEFER |
| `plot_surface_dgrid` | Warm | DEFER |
| `plot_maxima` | Warm | DEFER |
| `prewarm_framework_cache` | Hot | KEEP (HF perf) |
| `clear_framework_cache` | Hot | KEEP |

## python_cli/mgrcli/interactive_app.py (selected)

| Symbol | Tier | Action |
|--------|------|--------|
| `main` | Hot | DROP → Tauri |
| `_start_queue_worker` | Hot | DROP |
| `_start_compute_thread` | Hot | DROP → sidecar job |
| `_run_segmentation` | Hot | SUB |
| `_compute_pld_analysis` | Hot | SUB |
| `_make_plot` | Hot | SUB |
| `_render_spaces_queue_banner` | Hot | DROP |
| `_hts_one_material` | Cold | DROP |
| `results_data_to_npz` | Warm | KEEP |

## mgr.queue_state

| Symbol | Tier | Action |
|--------|------|--------|
| `JOB_QUEUE`, `JOB_REGISTRY`, `QUEUE_WORKER_STARTED` | Hot (Spaces) | DROP on desktop |

---

## mgr.compute — full definition inventory (56 defs)

| Line | Symbol | Tier | Action | Notes |
|------|--------|------|--------|-------|
| 20 | `gpd_aabb` | Warm | KEEP | freud AABB nearest-neighbor dgrid evaluation |
| 70 | `dask_grid_over_atoms` | Cold | DROP | Dask meshgrid — Rust does this natively |
| 104 | `dgrid_from_atoms_cpu_no_aabb` | Cold | DROP | Dask block dgrid without AABB — slow, unused in Rust path |
| 748 | `dgrid_from_atoms_gpu` | Cold | DROP | CuPy GPU dgrid — Rust rayon is faster for our grid sizes |
| 1067 | `regions_from_dgrid` | Warm | DEFER | Python watershed — Rust `seeded_watershed_det` replaces |
| 1333 | `get_region_props_for_regions` | Warm | KEEP | scipy.ndimage regionprops for PSD |
| 1344 | `add_window_info_to_rag` | Hot | KEEP | Adds window connection metadata to RAG graph |
| 1370 | `add_maxima_to_rag` | Hot | KEEP | Attaches maxima positions/radii to RAG |
| 1439 | `add_psd_to_rag` | Hot | KEEP | PSD bins to RAG graph |
| 1489 | `rag_from_connections` | Hot | KEEP | Builds RAG from window_connections output |
| 1531 | `get_rag_from_regions_and_grid` | Warm | KEEP | Convenience: regions → RAG in one call |
| 1623 | `apply_periodic_map` | Warm | KEEP | Periodic boundary mapping helper |
| 1698 | `AABB_on_atoms` | Hot | KEEP | freud AABB query object constructor |
| 1722 | `get_wall_windows3` | Warm | DEFER | Wall window detection variant 3 |
| 1756 | `get_wall_windows2` | Warm | DEFER | Wall window detection variant 2 |
| 1940 | `add_props_to_rag` | Warm | KEEP | Generic property attachment to RAG |
| 1967 | `interpolate_dgrid` | Warm | KEEP | scipy interpolation for smooth dgrid |
| 2902 | `filter_percolated_paths` | Warm | KEEP | Filters paths by probe size |
| 2938 | `get_pld_from_percolated_paths_path_windows` | Warm | KEEP | PLD from legacy percolation paths |
| 2950 | `get_pld_for_rag` | Warm | DEFER | PLD via RAG (legacy, winding replaces) |
| 3346 | `add_centroids_to_rag_with_pbc` | Warm | KEEP | Region centroids with PBC to RAG |
| 3402 | `add_psd_to_rag_with_pbc` | Warm | KEEP | PSD with periodic boundary corrections |
| 3728 | `add_clustering_to_rag` | Cold | DROP | DBSCAN/Agglomerative clustering — not in v1 |
| 4130 | `compare_two_paths` | Cold | DROP | Path comparison for clustering |
| 4147 | `find_isomorphic_path_groups` | Cold | DROP | Isomorphic path grouping |
| 4168 | `calc_path_components` | Cold | DROP | Path component calculation |
| 4206 | `find_paths` | Cold | DROP | Path finding between regions |
| 4580 | `get_diametrically_opposite_points` | Warm | KEEP | Convex hull sampling for PSD paths |
| 4721 | `get_psd_on_rag` | Cold | DROP | Dask-parallel PSD via dijkstra paths — Rust PSD replaces |
| 4813 | `sample_path` | Cold | DEFER | Spline resampling along path |
| 4832 | `get_distances_for_path` | Cold | DEFER | AABB distance along path |
| 5144 | `add_pbc_to_rag` | Warm | KEEP | PBC group metadata on RAG |
| 5162 | `clustering_grid_from_regions` | Cold | DROP | Grid clustering — not in v1 |
| 5185 | `get_repetitions_to_meet_cutoff` | Warm | KEEP | Supercell repetition calculation |
| 5224 | `repeat_to_meet_cutoff` | Warm | KEEP | ASE supercell repetition |
| 5256 | `gpe_aabb2` | Warm | KEEP | Grid-point energy eval variant 2 |
| 5364 | `gpe_aabb` | Warm | KEEP | Grid-point energy eval (freud AABB) |
| 5916 | `get_energy_interpolator` | Warm | DEFER | Energy grid interpolation |
| 6328 | `pad_periodic` | Warm | KEEP | Periodic padding helper |
| 6333 | `check_percolation_periodic` | Warm | DEFER | Legacy percolation check |
| 6363 | `find_pore_limiting_radius` | Warm | KEEP | Grid-based PLD (not winding) |
| 6382 | `compute_pld` | Warm | DEFER | Legacy PLD wrapper |
| 6411 | `get_pore_dia_mgr` | Warm | DEFER | Legacy pore diameter entry |
| 6463 | `compute_percolation_history_from_results` | Warm | KEEP | Percolation history from cached results |
| 6475 | `compute_percolation_history` | Warm | DEFER | Full percolation history sweep |
| 6554 | `setup_poisson_spectral` | Cold | DEFER | Spectral Poisson setup |
| 6591 | `solve_poisson_spectral` | Cold | DEFER | Spectral Poisson solve |
| 6651 | `setup_poisson_spectral_gpu` | Cold | DROP | GPU Poisson — not in v1 |
| 6687 | `solve_poisson_spectral_gpu` | Cold | DROP | GPU Poisson solve — not in v1 |
| 6752 | `compute_void_volume` | Hot | PORT | Void volume — scipy marching_cubes |
| 6792 | `compute_surface_area` | Hot | PORT | Surface area — scipy marching_cubes |
| 6824 | `gpd_aabb_hybrid` | Warm | KEEP | Hybrid AABB + brute-force dgrid |
| 7126 | `get_pbc_groups` | Warm | KEEP | Identify PBC-connected region groups |
| 7499 | `do_periodic_segmentation` | Warm | DEFER | Python segmentation — Rust replaces |
| 8649 | `select_percolation_threshold` | Warm | KEEP | Auto-select percolation threshold from history |
| 8700 | `compute_point_probe_reference` | Warm | KEEP | Point probe reference calculation |

### compute.py summary

| Category | Count | Action |
|----------|-------|--------|
| Dask/GPU dgrid | 3 | DROP |
| Legacy percolation | 5 | DEFER/DROP |
| Clustering/Optuna | 4 | DROP |
| GPU Poisson | 2 | DROP |
| RAG graph ops | 8 | KEEP |
| Surface/void | 2 | PORT (Phase 2) |
| Window detection | 2 | DEFER |
| PSD helpers | 3 | KEEP |
| Path/dijkstra | 4 | DEFER/DROP |
| Energy grid | 3 | KEEP/DEFER |
| Hot-path keep | 13 | KEEP |

---

## mgr.visualize (grouped)

Plotly/PyVista: `plot_connections`, `plot_surface_dgrid`, `mesh_for_atoms`, `plot_percolation_cluster_meshes`, etc.

| Tier | Action |
|------|--------|
| Warm | DEFER — Three.js is v1 default |
| Cold | DROP from slim binary |

---

## End-to-end call chains

### A. Spaces "Compute" button

1. `interactive_app.main` → queue submit  
2. `_start_compute_thread`  
3. `update_with_covalent_bonds`  
4. `compute_dgrid_rust` → `dgrid_from_atoms_rust` → **`mgrcli_rust.dgrid_from_atoms`**  
5. `_run_segmentation` → **`do_periodic_segmentation_rust`** → watershed + maxima + **`window_connections`**  
6. (optional) `_compute_pld_analysis` → **`get_pore_dia_mgr_winding`** + **`compute_percolation_history_winding`**  
7. `show_mof_threejs` or `show_mof` ← `prepare_results_for_plot`

### B. CLI batch

`cli.main` → same 4–6 without Streamlit queue.

### C. HTS mode (drop in v1)

`_hts_one_material` → Dask + repeated segmentation.

---

## Dependency matrix (hot path)

| Package | Role in chain | Slim v1 |
|---------|---------------|---------|
| mgrcli_rust | Core numerics | Yes |
| ase | Structure I/O | Yes |
| numpy | Arrays | Yes |
| freud | Neighbor list for bonds | Yes |
| networkx | Winding percolation | Yes until PORT |
| scipy | Surfaces, PSD peaks | Optional |
| skimage | marching_cubes | Optional |
| dask | Alternate dgrid, HTS | No |
| pymatgen | Bond fallback | No |
| streamlit | UI | No |
| plotly | Plotly viz | Optional |
| pyvista/trimesh | Mesh | No |
| rdkit | Chemistry | No |

---

*See also `docs/standalone_migration.md`.*

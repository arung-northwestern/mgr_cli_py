# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: magic
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PLD and Percolation 🌟
#  In this notebook, we'll see 📚
#  - how to compute the Pore Limiting Diameter (PLD) for a given material 🔍
#  - how to output and interpret the percolation history (by which the PLD is estimated) 📈
#  - how to visualize the connected component of regions (pockets) at any probe size from the percolation history 🎨
#  
#  Prerequisites: How to compute the distance grid. ⚙️
#  
#  In this tutorial, we'll skip over the explanation of the first steps; if anything is unclear, please refer to the basic tutorials. ❓
#  
#  Let's get started! 🚀

# %% [markdown]
# ## Import necessary libraries 📚🔧

# %%
from pathlib import Path
import numpy as np
import os, sys
from ase.io import read
path_to_mgr = Path('../../../22sep2024/mofography').resolve()
sys.path.append(str(path_to_mgr))
import mofography as mgr
import random
import plotly.graph_objects as go


# %% [markdown]
#  ## Compute PLD for a Material 🌟
#  
#  Pore Limiting Diameter (PLD) is the probe size at which all percolation in a material stops. 'Percolation' in this context simply implies that there is at least one path through which 
#  a spherical probe of a given size can pass from one face of the unit cell to the opposite, assuming periodicity in the other two directions. 🔄
#  
#  This section of the code performs the following tasks:
#  - Randomly selects one CIF file from the list in the `TestCIFs` folder and resolves its path. 📂
#  - Reads the selected CIF file to obtain atomic structure data. 🧪
#  - Computes the pore diameters (LCD, PLD, DFSD) and the distance grid using the `get_pore_dia_mgr` function on the `ase.atoms` object. 📏
#

# %%
# cif_files = list(Path('../examples/TestCIFs').glob('*.cif'))
# Choose a random CIF file
# path_to_cif = Path(random.choice(cif_files)).resolve()
path_to_cif = Path('../examples/TestCIFs/super_hMOF_22242.cif').resolve()
print(f"Selected CIF path: {path_to_cif}")
mat_atoms = read(path_to_cif)
lcd, pld, dfsd, distance_grid = mgr.get_pore_dia_mgr(mat_atoms, kwargs=dict(spacing=0.25))
print('lcd ', lcd, 'pld ', pld, 'dfsd ', dfsd)



# %% [markdown]
# if you just want the PLD for some reason, and not the Largest Cavity Diameter (LCD) and the Directional Free Sphere Diameter (DFSD), then you can use the `compute_pld` function on the CIF file

# %%
mgr.compute_pld(path_to_cif, kwargs={'spacing':0.25})

# %% [markdown]
#  ## 🌊 Percolation History 🌊
#  
#  Given a probe size, the probe will not percolate through all the pore windows (constrictions) that are smaller than the probe size. Hence, at each probe size, we can often identify more than one nected component of pockets, since not all the pockets may be accessible from every other pocket. 🕳️🔍 Percolation history refers to the connected components in each direction `0, 1` and `2` existing  a given probe size. Let's compute the pore graph, the percolation history, and then visualize the percolation history at a couple of different probe sizes to get a sense of the process. 📊✨ If you're ure about the pore graph, please refer to the basic tutorial on pore graph construction. 📚
#  
#  Before we proceed, let's take a look at the arguments of the `compute_percolation_history` function. 🔧

# %%
help(mgr.compute_percolation_history)

# %% [markdown]
# As we can see, this function is quite versatile, 
#
# - you can either pass just the `path_to_cif` or the `distance_grid` directly if you have it.
# - if you have the `regions`, `maxima` variables for this material, then you can pass them along directly instead, to save some compute time or to ensure consistency with some other step in your workflow.
# - regardless of the arguments you choose, `compute_percolation_history` returns the `percolation_history`, `pld_locations`, `pld_radii`, `distance_grid`, `regions` and `maxima` variables back, just to ensure consistency and convenient downstream usage
#
# Let's compute the percolation history and pore graph

# %%

percolation_history, pld_locations, _ , regions, maxima = mgr.compute_percolation_history(distance_grid=distance_grid, kwargs=dict(mask_thickness=np.min([pld/2-0.5, 1.5])))
connections = mgr.connections_from_regions_and_dgrid(regions, distance_grid, maxima, mat_atoms)
rag = mgr.rag_from_connections(connections, maxima)
maxima_radii = distance_grid[tuple(maxima.T)]
rag = mgr.add_maxima_to_rag(rag, maxima, maxima_radii, regions.shape, mat_atoms)
rag.graph['regions'] = regions
rag.graph['ase_atoms'] = mat_atoms


# %% [markdown]
# ### Interpret the percolation history 🌊🔍
#
# At each probe size, you can see the connected components detected in each direction. For example, if probe_size is 0, then all the regions should be connected, and there will be only one connected component in each direction. 🔗✨
#
# *PS: The probe size keys in percolation history returned are the probe radii*

# %%
percolation_history[0]

# %% [markdown]
# the last entry in the percolation history, is the last probe size tested before PLD was detected. Hence, this would often show the last active percolation connected component in the material, before all percolation in the material ceased.

# %%
percolation_history[-1]

# %% [markdown]
# So for this material, in the largest probe size for which some percolation exists, the largest connected components (two of them) detected in 'a' direction are [[3, 7, 9, 15, 17], [4, 8, 10, 16, 18]], where the numbers indicate the indices of the connected regions in each component.No connected components are detected in the 'b' or 'c' direction.

# %% [markdown]
# ### Visualize percolation at different probe sizes 🌊🔍
#
# We can make more sense of these lists by visualizing the touching percolated regions at a given probe size. `MOFography` includes functions to visualize these connected components easily. 📊✨
#
# As usual, let's include the unit cell, pore graph and local maxima in our visualization for added clarity. 🏗️🔬 Note that we chose the probe size as the last value in the percolation_history, and chose the isosurface to be farther than the mask_thickness we used in the previous cell (otherwise all regions would be colored as background). 🎨
#

# %%
rag_trace = mgr.plot_rag(rag, name='Pore Graph')
percolated_cluster_meshes = mgr.plot_percolated_cluster_meshes(percolation_history, distance_grid, regions, mat_atoms, isovalue=np.min([pld/2 - 0.25, 1.75]), stylize=True, probe_radius=percolation_history[-1][0])
cell_trace = mgr.plot_cell(mat_atoms)
maxima_trace = mgr.plot_maxima(distance_grid, maxima, mat_atoms, show_spheres=False, sphere_opacity=0.85)
pld_traces = mgr.plot_pld_locations(distance_grid, pld_locations, mat_atoms, show_spheres=False, sphere_opacity=0.85, marker_size=10, marker_color='limegreen')
fig = go.Figure(data=[cell_trace, rag_trace, maxima_trace, pld_traces] + percolated_cluster_meshes, layout=mgr.epic_layout)
fig.show(renderer='png')

# %% [markdown]
# Each of the connected components at the probe size is shown in the plot. Note that as the probe size increased, distinct connected components appeared in the plot. 
#
# If you make the same plot at probe size = 0, all the regions should be connected, resulting in only one connected component in each direction. I'll leave it to you to verify this.

# %% [markdown]
# Note that we showed some PLD locations in the plot. These are the locations of the grid locations where the value exactly matched the PLD. **This list is not inclusive of all the PLD locations**, because the values on the distance grid need not be exacly symmetric subject to grid and unit cell definitions. However, PLD locations will include at least one such location, which is useful. There might also be a cluster of closely located positions on here. If you want to find the exact PLD locations, please look at the pore windows along the path including the PLD, which can be read from the `connections` array. 

# %% [markdown]
# ## Conclusions 🌟
#
# - Pore Limiting Diameter (PLD) is the probe size at which all end-to-end, percolation in the material stops. 🚫
# - PLD tells us whether a spherical probe will pass through the material or not. ⚪️➡️🧱
# - We learned how to compute the PLD using MOFography. 📊🔍
# - Percolation history shows how the connected components of regions change with increasing probe sizes until PLD is detected. 📈🔄
# - We learned how to visualize the connected components from the percolation history using MOFography. 🎨✨

# %%

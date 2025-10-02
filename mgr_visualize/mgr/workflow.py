"""
 Functions for creating 
 ██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗███████╗██╗      ██████╗ ██╗    ██╗
 ██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝██╔════╝██║     ██╔═══██╗██║    ██║
 ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ █████╗  ██║     ██║   ██║██║ █╗ ██║
 ██║███╗██║██║   ██║██╔══██╗██╔═██╗ ██╔══╝  ██║     ██║   ██║██║███╗██║
 ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗██║     ███████╗╚██████╔╝╚███╔███╔╝
  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ 
                                                                       
 
"""


def quick_cluster(ase_atoms,

                  spacing=0.5,
                  mask_thickness=1.5,
                  h=0.2,
                  compactness=0,
                  zoom_ratio=2,
                  wall_thickness=1.0,
                  isoval=1.75,
                  bins=None,
                  block_size=50000, a=1, b=1, c=1, 
                  density=True,
                  js_threshold = 0.15,
                  distance_threshold=0.4, minimum_overlap_fraction=0.5, make_figures=True, renderer='browser', sample_fraction=0.5, cscale=None,return_these_vars=[]):  # distance threshold for agglomerative clustering
    # # # * Pick a random material from the database
    # if chosen is None:
    #     chosen = random.choice(np.arange(len(iza_atoms)))
    # # chosen = 108
    # # chosen = 107
    # print(chosen, df['name'][chosen])
    # mat_atoms = iza_atoms[chosen]
    # from ase.io import read
    # mat_atoms = read('../examples/TestCIFs/NU-1000_S1.cif')
    """ 
    It takes in a material (in the form of an ASE atoms object), and returns a list of objects that you
    can specify in the list_return_strings argument.
    
    :param ase_atoms: ASE atoms object
    :param spacing: the spacing of the distance grid
    :param mask_thickness: The thickness of the mask used to create the distance grid
    :param h: float, optional
    :param compactness: float, optional
    :param zoom_ratio: The distance grid is made finer by zooming in, here we are making it twice as
    fine, defaults to 2 (optional)
    :param wall_thickness: The thickness of the walls of the material
    :param bins: the bins for the cavity size distribution histogram
    :param block_size: the size of the blocks in the dask array, this is the size of the chunks that are
    computed in parallel, defaults to 50000 (optional)
    :param a: number of unit cells in the a direction, defaults to 1 (optional)
    :param b: int, defaults to 1 (optional)
    :param c: number of unit cells in the c direction, defaults to 1 (optional)
    :param distance_threshold: distance threshold for agglomerative clustering
    :param minimum_overlap_fraction: The minimum fraction of overlap between the projections of two regions on opposite walls of the unit cell for them to be
    considered as part of the same pbc group
    :param show_figures: whether to show the figures or not, defaults to True (optional)
    :param renderer: 'browser' or 'png', defaults to browser (optional)
    :param list_return_strings: list of strings, the names of the variables you want to return
    """


    

    import numpy as np 

    #* to find and return variables 
    import inspect

    #* for atomic radii
    from ase.data import covalent_radii

    # * cif manipulation
    import ase
    from ase.build import make_supercell

    # * For manipulating the distance grid efficiently
    import dask.array as da

    #* make grid finer
    from scipy.ndimage import zoom 

    #* measure some properties ofthe regions 
    from skimage.measure import regionprops

    #* Plotting and visualization
    import plotly.graph_objects as go
    # Import necessary libraries for plotting and analysis
    import plotly.express as px 
    from .analyze import update_with_covalent_bonds
    from .compute import (dgrid_from_atoms_cpu, regions_from_dgrid, apply_pbc, 
                           connections_from_regions_and_dgrid, rag_from_connections, 
                           add_pbc_to_rag, add_centroids_to_rag_with_pbc, 
                           add_psd_to_rag_with_pbc, clustering_grid_from_regions, 
                           add_clustering_to_rag_geometry_js, add_maxima_to_rag)
    from .data import mix_lorentz_berthelot
    from .visualize import (plot_surface_dgrid, plot_framework_as_mesh, 
                            plot_cell, plot_maxima, plot_psds, 
                            white_layout, plot_vdw_hists)
    from .helper import zoom_grid

    # Set the colorscale for plotting if not already defined
    if cscale is None:
        # Use Plotly's qualitative colorscale for consistency in figures
        cscale = px.colors.qualitative.Plotly

    # Define bins for the cavity feature histogram if not provided
    if bins is None:
        # Create 50 evenly spaced bins between 0 and 30
        bins = np.linspace(0, 30, 50)

    # Update the atomic structure with covalent bonds information
    mat_updated = update_with_covalent_bonds(ase_atoms)

    # Generate a distance grid from the atomic structure
    dgrid = dgrid_from_atoms_cpu(ase_atoms, spacing=spacing, precompute=True)

    # Extract regions and maxima from the distance grid
    regions, maxima = regions_from_dgrid(dgrid, mask_thickness=mask_thickness, 
                                         compactness=compactness, h=h)

    # Get the radii of the maxima from the distance grid
    maxima_radii = dgrid[tuple(maxima.T)]

    # Apply periodic boundary conditions (PBC) to the regions and maxima
    pbc_groups = apply_pbc(regions, maxima, maxima_radii, ase_atoms, 
                            wall_thickness=wall_thickness, return_conn=True, 
                            minimum_overlap_fraction=minimum_overlap_fraction)  # Returns a list of PBC groups

    # Get the regions with PBC applied, returning a 3D array
    regions_pbc = apply_pbc(regions, maxima, maxima_radii, ase_atoms, 
                             wall_thickness=wall_thickness, return_conn=False, 
                             minimum_overlap_fraction=minimum_overlap_fraction)

    # Create connections from regions and the distance grid
    connections = connections_from_regions_and_dgrid(regions, dgrid, maxima,ase_atoms)

    # Generate a region adjacency graph (RAG) from the connections
    rag = rag_from_connections(connections, maxima)

    # Add maxima information to the RAG
    rag = add_maxima_to_rag(rag, maxima, maxima_radii, regions.shape, ase_atoms)

    # Measure properties of the regions
    props = regionprops(regions)

    # Incorporate PBC information into the RAG
    rag = add_pbc_to_rag(rag, pbc_groups)   

    # Add centroids to the RAG with PBC consideration
    rag = add_centroids_to_rag_with_pbc(rag, props, pbc_groups,ase_atoms, regions.shape)

    # Add cavity size distribution (PSD) information to the RAG with PBC
    rag = add_psd_to_rag_with_pbc(rag, props, pbc_groups, bins=bins, 
                                   ase_atoms=ase_atoms, shape=regions.shape)

    # Add clustering information to the RAG geometry
    rag =  add_clustering_to_rag_geometry_js(rag, props, pbc_groups, ase_atoms, dgrid.shape)

    # Generate clusters from the regions based on the RAG
    regions_cluster =  clustering_grid_from_regions(regions, rag, cluster_key='cluster_geom_pbc')

    if make_figures:
        # vdw_traces = plot_vdw_hists(rag, pbc_groups, cscale=cscale, cluster_key='cluster_geom_pbc')
        vdw_traces =plot_psds(rag, pbc_groups, cscale=cscale, cluster_key='cluster_geom_pbc')
        fig_vdw = go.Figure(data=vdw_traces)
        fig_vdw.update_layout(xaxis_title='Distance to centroid (A)', yaxis_title='Normalized histogram')
        # fig_vdw.show(renderer=renderer)

    if make_figures:


        # * Plot the distance isosurface at 1.75 A colored in different ways
        surface_trace =  plot_surface_dgrid(dgrid,isoval, select='all', ase_atoms=mat_updated,
                                               region_labels=regions, cscale=cscale, stylize=True, name='Raw segmentation', hover_label='Region :')
        surface_trace_pbc =  plot_surface_dgrid(dgrid,isoval, select='all', ase_atoms=mat_updated,
                                                   region_labels=regions_pbc-1, cscale=cscale, stylize=True, name='PBC groups', hover_label='PBC Group :')
        


        surface_trace_cluster =  plot_surface_dgrid(dgrid,isoval, select='all', ase_atoms=mat_updated,
                                                       region_labels=regions_cluster-1, cscale=cscale, stylize=True, name='Cavity types', hover_label='Cavity type :')

        # * plot the framework atoms and bonds as a mesh
        frame_trace =  plot_framework_as_mesh(mat_updated, show_rings=True, stylize=True)
        # surface_trace= plot_surface_dgrid(grid.compute(), 1.75, select=list(pbc_groups[0]),ase_atoms=mat_updated,region_labels=regions, stylize=True)

        # * plot the unit cell as a box
        cell_trace =  plot_cell(mat_updated)

        # * plot the local maxima as points colored by their size (radius)
        maxima_trace =  plot_maxima(maxima=maxima, dgrid=dgrid, ase_atoms=mat_updated)

        #* plot the windows of the material
        # connections =  get_connections_for_rag(rag, region_labels=regions, ase_atoms = mat_atoms, maxima=maxima, dgrid=grid.compute())
        # windows_trace =  plot_windows(connections)
        # * collect the traces into a figure object
        fig_full_visual = go.Figure(data=[surface_trace, surface_trace_pbc,
                                    surface_trace_cluster, *frame_trace, cell_trace, maxima_trace])
        # fig_full_visual = go.Figure(data=[surface_trace, surface_trace_pbc, surface_trace_cluster, *frame_trace, cell_trace, maxima_trace, windows_trace])
        fig_full_visual.update(layout= white_layout)
        fig_full_visual.update_layout(scene=dict(aspectmode='data'))
        # fig_full_visual.show(renderer=renderer)

        
    # # # * compute and plot the energy histograms for each pbc group
    # rag =  add_vdw_hist_to_rag(rag, props, mat_atoms, forcefield_mixed, regions.shape,use_pbc=True,  pbc_groups=pbc_groups, density=True)
    # ehist_traces =  plot_vdw_hists(rag, pbc_groups, cscale=cscale)
    # fig_ehist = go.Figure(data=ehist_traces)
    # fig_ehist.update_layout(xaxis_title='Energy (K)', yaxis_title='Normalized histogram')
    # fig_ehist.show(renderer=renderer)
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    # callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    if len(return_these_vars) > 0:
        print('Some return objects specified, possible options to pass in list_return_strings are:')
        # list_return_objects = [val for name, val in callers_local_vars if name in return_these_vars]
        # list_return_objects = [val for i,val in enumerate(values) if args[i] in return_these_vars]
        list_return_objects = [frame.f_locals[var] for var in return_these_vars]

        return list_return_objects + [[fig_full_visual, fig_vdw]]
    else:
        print('No return objects specified, possible options to pass in list_return_strings are:')
        # print(callers_local_vars)
        print(frame.f_locals.keys())
        if make_figures:
            print('Returning the figures if make_figures=True')
            return [fig_full_visual, fig_vdw]

# def quick_cluster_copy(ase_atoms,

#                   spacing=0.5,
#                   mask_thickness=1.5,
#                   h=0.2,
#                   compactness=0,
#                   zoom_ratio=2,
#                   wall_thickness=1.0,
#                   isoval=1.75,
#                   bins=None,
#                   block_size=50000, a=1, b=1, c=1, 
#                   density=True,
#                   distance_threshold=0.4, minimum_overlap_fraction=0.5, make_figures=True, renderer='browser', use_energy_for_clustering=False, sample_fraction=0.5, cscale=None,return_these_vars=[]):  # distance threshold for agglomerative clustering
#     # # # * Pick a random material from the database
#     # if chosen is None:
#     #     chosen = random.choice(np.arange(len(iza_atoms)))
#     # # chosen = 108
#     # # chosen = 107
#     # print(chosen, df['name'][chosen])
#     # mat_atoms = iza_atoms[chosen]
#     # from ase.io import read
#     # mat_atoms = read('../examples/TestCIFs/NU-1000_S1.cif')
#     """ 
#     It takes in a material (in the form of an ASE atoms object), and returns a list of objects that you
#     can specify in the list_return_strings argument.
    
#     :param ase_atoms: ASE atoms object
#     :param spacing: the spacing of the distance grid
#     :param mask_thickness: The thickness of the mask used to create the distance grid
#     :param h: float, optional
#     :param compactness: float, optional
#     :param zoom_ratio: The distance grid is made finer by zooming in, here we are making it twice as
#     fine, defaults to 2 (optional)
#     :param wall_thickness: The thickness of the walls of the material
#     :param bins: the bins for the cavity size distribution histogram
#     :param block_size: the size of the blocks in the dask array, this is the size of the chunks that are
#     computed in parallel, defaults to 50000 (optional)
#     :param a: number of unit cells in the a direction, defaults to 1 (optional)
#     :param b: int, defaults to 1 (optional)
#     :param c: number of unit cells in the c direction, defaults to 1 (optional)
#     :param distance_threshold: distance threshold for agglomerative clustering
#     :param minimum_overlap_fraction: The minimum fraction of overlap between the projections of two regions on opposite walls of the unit cell for them to be
#     considered as part of the same pbc group
#     :param show_figures: whether to show the figures or not, defaults to True (optional)
#     :param renderer: 'browser' or 'png', defaults to browser (optional)
#     :param list_return_strings: list of strings, the names of the variables you want to return
#     """


    

#     import numpy as np 

#     #* to find and return variables 
#     import inspect

#     #* for atomic radii
#     from ase.data import covalent_radii

#     # * cif manipulation
#     import ase

#     # * For manipulating the distance grid efficiently
#     import dask.array as da

#     #* make grid finer
#     from scipy.ndimage import zoom 

#     #* measure some properties ofthe regions 
#     from skimage.measure import regionprops

#     #* Plotting and visualization
#     import plotly.graph_objects as go

#     #* colorscale for plotting 
#     import plotly.express as px 
#     from .analyze import update_with_covalent_bonds
#     from .compute import dgrid_from_atoms,regions_from_dgrid,apply_pbc, get_rag_from_regions_and_grid,add_pbc_to_rag, add_psd_to_rag, add_psd_to_rag_with_pbc, add_clustering_to_rag_pca, clustering_grid_from_regions,add_clustering_to_rag_energy, add_vdw_hist_to_rag
#     from .data import mix_lorentz_berthelot
#     from .visualize import plot_surface_dgrid, plot_framework_as_mesh, plot_cell, plot_maxima, plot_psds, white_layout, plot_vdw_hists, draw_framework
#     if cscale is None:
#         #* set colorscale for plotting, make things consistent across the two figures
#         cscale = px.colors.qualitative.Alphabet

#     #* bins for the cavity feature histogram
#     if bins is None:
#         bins = np.linspace(0,30,50)

#     mat_updated =  update_with_covalent_bonds(ase_atoms)

#     # * compute the distance grid, locate the local maxima and make a list of their size (radii)
#     radii = covalent_radii[ase_atoms.get_atomic_numbers()]  # assign covalent radii
#     # compute the distance grid (dask array) of given spacing and partition size
#     grid =  dgrid_from_atoms(ase_atoms, radii=radii, spacing=spacing, block_size=block_size)
#     # (optional) make a supercell of the material
#     ase_atoms = ase.build.make_supercell(ase_atoms, [[a, 0, 0], [0, b, 0], [0, 0, c]])
#     grid = da.from_array(np.tile(grid.compute(), [a, b, c]))
#     # (optional) make the distance grid finer by zooming in, here we are aamking it twice as fine
#     grid = da.from_array(zoom(grid.compute(), zoom_ratio, mode='grid-wrap',
#                          order=1, grid_mode=False), chunks=(40, 40, 40))
#     # compute the region segementaion and array indices of the maxima corresponding to each region
#     regions, maxima =  regions_from_dgrid(
#         grid.compute(), mask_thickness=mask_thickness, compactness=compactness, h=h)
#     maxima_radii = grid.compute()[tuple(maxima.T)]  # get teh distance values (size) of the maxima

#     #* get the connectivity of the pockets, each pbc group is a list of regions that are split up by the PBC, that come together to form a complete cavity.
#     # * return_conn = True returns the list of pbc groups, return_conn = False returns the a 3D array of regions where region index is replaced by the pbc group index
#     pbc_groups =  apply_pbc(regions, maxima, maxima_radii, ase_atoms, wall_thickness=wall_thickness,
#                                return_conn=True, minimum_overlap_fraction=minimum_overlap_fraction)  # this returns a list of pbc groups
#     regions_pbc =  apply_pbc(regions, maxima, maxima_radii, ase_atoms, wall_thickness=wall_thickness,
#                                 return_conn=False, minimum_overlap_fraction=minimum_overlap_fraction)  # this returns a 3D array
#     # * construct the region adjacency graph, a.k.a the pore graph form the regions, grid, maxima indices and the ase atoms object
#     rag =  get_rag_from_regions_and_grid(regions, grid.compute(), ase_atoms, maxima)

#     # * in this part we will be adding properties to the region adjacency graph
#     rag =  add_pbc_to_rag(rag, pbc_groups)
#     props = regionprops(regions)  # * get the region properties from skimage.measure.regionprops
#     rag =  add_psd_to_rag(rag, region_props=props,  bins=bins, ase_atoms=ase_atoms,
#                               shape=regions.shape)  # *  add the cavity histogram without pbc to the RAG
#     rag =  add_psd_to_rag_with_pbc(rag=rag, props=props, pbc_groups=pbc_groups, bins=bins,
#                                        ase_atoms=ase_atoms, shape=regions.shape)  # * add the cavity histogram with pbc to the RAG
#     # rag=  add_clustering_to_rag(rag) # * add the clustering labels to the RAG without pbc

#     # * add energy histogram to the RAG
#     # forcefield_mixed =  mix_lorentz_berthelot(mat_atoms)
#     # rag =  add_vdw_hist_to_rag_with_pbc(rag, props, pbc_groups, mat_atoms, forcefield_mixed, regions.shape)
#     if use_energy_for_clustering:
#         # * use the energy criterion for clustering
#         props = regionprops(regions) # * get the  region properties from skimage.measure.regionprops
#         forcefield_mixed = mix_lorentz_berthelot(ase_atoms) # * mix the lorentz-berthelot forcefield using UFF and helium probe
#         # * do the clustering using the sampled point energies with pbc, 
#         # * using the kolmogorov-smirnov two-sided test with 95% confidence interval to tell if the two distributions are same or not
#         rag = add_vdw_hist_to_rag(rag, props, ase_atoms, forcefield_mixed, regions.shape, use_pbc=True, pbc_groups=pbc_groups, sample_fraction=sample_fraction, density=density)
#         rag = add_clustering_to_rag_energy(rag=rag, props=props, pbc_groups=pbc_groups, ase_atoms=ase_atoms, forcefield_mixed =forcefield_mixed, shape=regions.shape, use_pbc=True, sample_fraction=sample_fraction)
#         regions_cluster =  clustering_grid_from_regions(regions, rag, cluster_key='cluster_energy_pbc')
#         if make_figures:
#             vdw_traces = plot_vdw_hists(rag, pbc_groups, cscale=cscale, cluster_key='cluster_energy_pbc')
#             fig_vdw = go.Figure(data=vdw_traces)
#             fig_vdw.update_layout(xaxis_title='Distance to centroid (A)', yaxis_title='Normalized histogram')
#             # fig_vdw.show(renderer=renderer)
#     else:
#         pass
#         # rag, cluster_obj =  add_clustering_to_rag_pca(rag, no_pbc=False, kwargs_cluster=dict(distance_threshold=distance_threshold, kernel='linear'), return_cluster_obj=True, kwargs_pca=dict(
#         #     kernel='cosine'))  # * do clustering of the regions with PCA and default settings (clustering kernel='linear', linkage='single', distance='euclidean', pca kernel ='cosine')

#         # #---------------------------------------------------#
#         # # #* add energy histogram to the RAG
#         # # props = regionprops(regions)
#         # # forcefield_mixed =  mix_lorentz_berthelot(mat_atoms)
#         # # rag =  add_vdw_hist_to_rag(rag, props, mat_atoms, forcefield_mixed, regions.shape, use_pbc=True,pbc_groups=pbc_groups,density=False)

#         # # #* add clustering to rag using energy with pbc
#         # # rag, cluster_obj  = add_clustering_to_rag_pca2(rag, no_pbc=False, kwargs_cluster=dict(distance_threshold=distance_threshold, kernel = 'linear'), return_cluster_obj=True, kwargs_pca=dict(kernel='cosine')) # * do clustering of the regions with PCA and default settings (clustering kernel='linear', linkage='single', distance='euclidean', pca kernel ='cosine')
#         # #-------------------------------------------------#
#         # # * get the 3D array of cluster labels
#         # regions_cluster =  clustering_grid_from_regions(regions, rag)
#         if make_figures:
#             # * plot the cavity histogram for each pbc group
#             csd_traces =  plot_psds(rag, pbc_groups, cscale=cscale)
#             fig_vdw = go.Figure(data=csd_traces)
#             fig_vdw.update_layout(xaxis_title='Distance to centroid (A)', yaxis_title='Normalized histogram')
#             # fig_vdw.show(renderer=renderer)

#     if make_figures:


#         # * Plot the distance isosurface at 1.75 A colored in different ways
#         surface_trace =  plot_surface_dgrid(grid.compute(),isoval, select='all', ase_atoms=mat_updated,
#                                                region_labels=regions, cscale=cscale, stylize=True, name='Raw segmentation', hover_label='Region :')
#         surface_trace_pbc =  plot_surface_dgrid(grid.compute(),isoval, select='all', ase_atoms=mat_updated,
#                                                    region_labels=regions_pbc-1, cscale=cscale, stylize=True, name='PBC groups', hover_label='PBC Group :')
        

#         # ## * mesh 
#         # # * Compute the isosurface using Marching Cubes Lewiner as usual
#         # from skimage.measure import marching_cubes as mc
#         # verts, faces, normals, values = mc(grid.compute(), isoval, step_size=1)

#         # # * create a dictionary of colors for the cluster types
#         # import networkx as nx
#         # unique_clusters = np.unique(list(nx.get_node_attributes(rag, 'cluster').values())).astype(int)
#         # cmap = dict(zip(unique_clusters, cscale[:len(unique_clusters)]))
#         # cmap[-1]=cscale[-1]


#         # # * create an interpolator for the cluster grid
#         # from scipy.interpolate import RegularGridInterpolator as RGI
#         # xx = np.linspace(0,regions_cluster.shape[0] - 1,regions_cluster.shape[0])
#         # yy = np.linspace(0,regions_cluster.shape[1] - 1,regions_cluster.shape[1])
#         # zz = np.linspace(0,regions_cluster.shape[2] - 1,regions_cluster.shape[2])
#         # rl = RGI((xx, yy, zz),regions_cluster-1, method='nearest') # starts from -1 for backgroudn and then 0, 1, 2 etc  for pocket types
        

#         # colours = rl(verts)
#         # #  Find the actual coordinates of the vertices for plotting
#         # from ase.geometry import complete_cell
#         # hovertext = ['Pocket type:' + str(int(c)) for c in colours]
#         # cell = complete_cell(mat_updated.get_cell())
#         # true_verts = np.dot(cell, (verts / regions_cluster.shape).T).T
#         # vertexcolors = [cmap[c] for c in colours]
#         # surface_trace_cluster = go.Mesh3d(x=true_verts[:, 0], y=true_verts[:, 1], z=true_verts[:, 2], vertexcolor=vertexcolors,
#         #                       hoverinfo='all', hovertext=hovertext, flatshading=False, colorscale=cscale, i=faces[:, 0],
#         #                       j=faces[:, 1], k=faces[:, 2], opacity=1, name='Pocket types', showlegend=False)


#         surface_trace_cluster =  plot_surface_dgrid(grid.compute(),isoval, select='all', ase_atoms=mat_updated,
#                                                        region_labels=regions_cluster-1, cscale=cscale, stylize=True, name='Cavity types', hover_label='Cavity type :')

#         # * plot the framework atoms and bonds as a mesh
#         # frame_trace =  plot_framework_as_mesh(mat_updated, show_rings=True, stylize=True)
#         frame_trace =draw_framework(mat_updated, show_aromatics=True, bond_color='gray')
#         # surface_trace= plot_surface_dgrid(grid.compute(), 1.75, select=list(pbc_groups[0]),ase_atoms=mat_updated,region_labels=regions, stylize=True)

#         # * plot the unit cell as a box
#         cell_trace =  plot_cell(mat_updated)

#         # * plot the local maxima as points colored by their size (radius)
#         maxima_trace =  plot_maxima(maxima=maxima, dgrid=grid.compute(), ase_atoms=mat_updated)

#         #* plot the windows of the material
#         # connections =  get_connections_for_rag(rag, region_labels=regions, ase_atoms = mat_atoms, maxima=maxima, dgrid=grid.compute())
#         # windows_trace =  plot_windows(connections)
#         # * collect the traces into a figure object
#         fig_full_visual = go.Figure(data=[surface_trace, surface_trace_pbc,
#                                     surface_trace_cluster, *frame_trace, cell_trace, maxima_trace])
#         # fig_full_visual = go.Figure(data=[surface_trace, surface_trace_pbc, surface_trace_cluster, *frame_trace, cell_trace, maxima_trace, windows_trace])
#         fig_full_visual.update(layout= white_layout)
#         fig_full_visual.update_layout(scene=dict(aspectmode='data'))
#         # fig_full_visual.show(renderer=renderer)

        
#     # # # * compute and plot the energy histograms for each pbc group
#     # rag =  add_vdw_hist_to_rag(rag, props, mat_atoms, forcefield_mixed, regions.shape,use_pbc=True,  pbc_groups=pbc_groups, density=True)
#     # ehist_traces =  plot_vdw_hists(rag, pbc_groups, cscale=cscale)
#     # fig_ehist = go.Figure(data=ehist_traces)
#     # fig_ehist.update_layout(xaxis_title='Energy (K)', yaxis_title='Normalized histogram')
#     # fig_ehist.show(renderer=renderer)
#     frame = inspect.currentframe()
#     args, _, _, values = inspect.getargvalues(frame)
#     # callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     if len(return_these_vars) > 0:
#         print('Some return objects specified, possible options to pass in list_return_strings are:')
#         # list_return_objects = [val for name, val in callers_local_vars if name in return_these_vars]
#         # list_return_objects = [val for i,val in enumerate(values) if args[i] in return_these_vars]
#         list_return_objects = [frame.f_locals[var] for var in return_these_vars]

#         return list_return_objects # + [[fig_full_visual, fig_vdw]]
#     else:
#         print('No return objects specified, possible options to pass in list_return_strings are:')
#         # print(callers_local_vars)
#         print(frame.f_locals.keys())
#         if make_figures:
#             print('Returning the figures if make_figures=True')
#             return [fig_full_visual, fig_vdw]

# """
 
#  Visualization subroutines for mofography
#  ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗     ██╗███████╗███████╗
#  ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║     ██║╚══███╔╝██╔════╝
#  ██║   ██║██║███████╗██║   ██║███████║██║     ██║  ███╔╝ █████╗  
#  ╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║     ██║ ███╔╝  ██╔══╝  
#   ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗██║███████╗███████╗
#    ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚══════╝
                                                                 
 
# """

# TODO draw pore graph
# todo show the region wise properties well ?   



global white_layout
white_layout = dict(template='plotly_white',
                    scene=dict(xaxis=dict(zeroline=False, visible=False), yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
                    margin=dict(t=20, b=20, l=20, r=20), showlegend=True, legend=dict(x=1.1, y=0.75, groupclick='togglegroup'))
global dark_layout
dark_layout = dict(template='plotly_dark',
                   scene=dict(xaxis=dict(zeroline=False, visible=False), yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
                   margin=dict(t=20, b=20, l=20, r=20), showlegend=True, legend=dict(x=0.1, y=1.1, groupclick='togglegroup'))
global epic_layout
epic_layout = dict(template='plotly_dark',
                   scene=dict(xaxis=dict(zeroline=False, visible=False), yaxis_visible=False, zaxis_visible=False),
                   margin=dict(t=20, b=20, l=20, r=20), showlegend=True,
                   legend=dict(x=-0.1, y=1.1, groupclick='togglegroup',font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

global publication_layout
publication_layout = dict(template='plotly_white',
                    scene=dict(xaxis=dict(zeroline=False, visible=True, linecolor='black', linewidth=2, mirror=True), yaxis=dict(zeroline=False, visible=True, linecolor='black', linewidth=2, mirror=True),  aspectmode='data'),
                    margin=dict(t=20, b=20, l=20, r=20), showlegend=True, legend=dict(x=1.1, y=0.75, groupclick='togglegroup'))

global default_mesh_lighting
default_mesh_lighting= dict(
        ambient=0.3,
        diffuse=1.0,
        fresnel=0.2,
        specular=0.05,
        roughness=0.5,
    )

import numpy as np
# def plot_surface_dgrid(dgrid, isovalue, ase_atoms, region_labels=None, select='all', opacity=1.0, cscale=None,
#                        color='blue', stepsize=1, stylize=False, random_seed=0):
#     """
#     It takes a 3D grid, an isovalue, an ASE atoms object, and a few other optional parameters and
#     returns a plotly Mesh3D trace object that can be plotted using the plotly library.
    
#     :param dgrid: the density grid
#     :param isovalue: the value of the isosurface to be plotted
#     :param ase_atoms: ASE atoms object
#     :param region_labels: a 3D array of the same shape as the dgrid. This array contains the region
#     labels for each voxel
#     :param select: the region labels to be plotted. If 'all' is selected, all regions are plotted,
#     defaults to all (optional)
#     :param opacity: opacity of the surface
#     :param cscale: a colorscale for the regions. If None, a random colorscale is generated
#     :param color: color of the surface, defaults to blue (optional)
#     :param stepsize: The step size for the marching cubes algorithm, defaults to 1 (optional)
#     :param stylize: if True, will apply some random style choices, purely personal preference, defaults
#     to False (optional)
#     :param random_seed: int, optional, defaults to 0 (optional)
#     :return: A plotly Mesh3D trace
#     """

#     import numpy as np
#     import copy
#     import plotly.graph_objects as go
#     import ase
#     import plotly.express as px

#     dgrid_copy = copy.deepcopy(dgrid)  # deepcopy of the grid
#     cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T  # get cell matrix
#     # * Exclude the regions that are not selected
#     if select != 'all':
#         dgrid_copy[np.isin(region_labels, select, assume_unique=False, invert=True)] = -10

#     # * Compute the isosurface using Marching Cubes Lewiner
#     from skimage.measure import marching_cubes as mc
#     verts, faces, normals, values = mc(dgrid_copy, isovalue, step_size=stepsize)
#     np.random.seed(random_seed)
#     if np.any(region_labels) != None:  # if region labels are provided

#         # *  Make a colorscale using random colors if no colorscale is provided
#         if cscale is None:
#             cvals = np.linspace(0, 1, (region_labels.max()).astype(int))
#             rand_ints = np.random.randint(low=0, high=256, size=(region_labels.max().astype(int), 3))

#             cscale = [[cvals[i],
#                        "rgb(" + str(rand_ints[i][0]) + ',' + str(rand_ints[i][1]) + ',' + str(
#                            rand_ints[i][2]) + ')'] for i in range((region_labels.max()).astype(int))]
#         if type(cscale) == str:
#             try:
#                 cscale = getattr(px.colors.qualitative, cscale)
#             except:
#                 try:
#                     cscale = getattr(px.colors.sequential, cscale)
#                 except:
#                     pass

#             # cscale = [[cvals[i],
#             #         "rgb(" + str(np.random.randint(0, 256)) + ',' + str(np.random.randint(0, 256)) + ',' + str(
#             #             np.random.randint(0, 256)) + ')'] for i in range((region_labels.max()).astype(int))]

#         #  Find the colour for the vertices
#         from scipy.interpolate import RegularGridInterpolator as RGI
#         xx = np.linspace(0, region_labels.shape[0] - 1, region_labels.shape[0])
#         yy = np.linspace(0, region_labels.shape[1] - 1, region_labels.shape[1])
#         zz = np.linspace(0, region_labels.shape[2] - 1, region_labels.shape[2])
#         rl = RGI((xx, yy, zz), region_labels, method='nearest')
#         colours = rl(verts)

#         #  Create a plotly Mesh3D trace
#         #        ------------------------------------------

#         #  Find the actual coordinates of the vertices for plotting
#         hovertext = ["Region: " + str(int(c)) for c in colours]
#         true_verts = np.dot(cell, (verts / region_labels.shape).T).T
#         data_plot = go.Mesh3d(x=true_verts[:, 0], y=true_verts[:, 1], z=true_verts[:, 2], intensity=colours.astype(int),
#                               hoverinfo='all', hovertext=hovertext, flatshading=False, colorscale=cscale, i=faces[:, 0],
#                               j=faces[:, 1], k=faces[:, 2], opacity=opacity, name='isosurface', showlegend=True)

#     else:  # if no region labels are provided color by a flat color

#         true_verts = np.dot(cell, (verts / dgrid.shape).T).T
#         data_plot = go.Mesh3d(x=true_verts[:, 0], y=true_verts[:, 1], z=true_verts[:, 2],
#                               hoverinfo='all',
#                               i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=opacity, color=color,
#                               name='pore surface', showlegend=True)

#     if stylize:  # some random style choices, purely personal preference.
#         # can be modified after ploting as well. see plotly docs.
#         data_plot.update(lighting=dict(ambient=0.18,
#                                        diffuse=1,
#                                        fresnel=0.1,
#                                        specular=1,
#                                        roughness=0.05,
#                                        facenormalsepsilon=1e-15,
#                                        vertexnormalsepsilon=1e-15),
#                          lightposition=dict(x=10,
#                                             y=10,
#                                             z=30
#                                             ), showscale=False
#                          )

def plot_paths(rag, paths: list, colorscale: str='Pastel', **kwargs) -> list:
    """
    The function `plot_paths` generates 3D scatter plots with paths represented as markers connected by
    lines in different colorscales.
    
    :param rag: The `rag` parameter in the `plot_paths` function seems to represent a networkx graph
    object. It is used to extract node attributes from the graph for plotting paths
    :param paths: The `paths` parameter in the `plot_paths` function is a list of lists. Each inner list
    represents a path in a graph. The function iterates over these paths and creates a 3D scatter plot
    with markers and lines to visualize the paths. Each path is represented by markers connected by
    :type paths: list
    :param colorscale: The parameter `colorscale` in the `plot_paths` function is used to specify the
    color palette that will be used to color the paths in the plot. The default value is set to
    'Pastel', but you can provide any other valid color scale from the `px.colors.qualitative` module,
    defaults to Pastel
    :type colorscale: str (optional)
    :return: The function `plot_paths` returns a list of plotly `Scatter3d` traces representing the
    paths provided in the input list.
    """

    import plotly.graph_objects as go 
    import networkx as nx 
    import random
    import plotly.express as px 
    import numpy as np

    # Get all maxima coordinates at once
    maxima_dict = nx.get_node_attributes(rag, 'maxima')
    colorscale = getattr(px.colors.qualitative, colorscale)

    path_traces = []
    for i, path in enumerate(paths):
        # Get coordinates for this path
        coordinates = np.array([maxima_dict[node] for node in path])
        
        marker_color = random.choice(colorscale)
        
        path_trace = go.Scatter3d(
            x=coordinates[:, 0], 
            y=coordinates[:, 1], 
            z=coordinates[:, 2], 
            mode='markers+lines', 
            marker=dict(size=25, color=marker_color),
            line=dict(width=10, color='yellow'), 
            hovertext=f'P{i}: {" -> ".join(map(str, path))}',
            name=f'Path:{i}',
            **kwargs
        )
        path_traces.append(path_trace)
    
    return path_traces

def plot_surface_dgrid(dgrid, isovalue, ase_atoms, region_labels=None, select='all', opacity=1.0, cscale=None,
                       color='blue', stepsize=1, stylize=False, random_seed=0, hover_label='Region :', **kwargs_mesh):
    """
    It takes a 3D grid, an isovalue, an ASE atoms object, and a few other optional parameters and
    returns a plotly mesh3d object that can be plotted using plotly
    
    Args:
      dgrid: the density grid
      isovalue: the value of the isosurface to be plotted
      ase_atoms: ASE atoms object
      region_labels: a 3D array of the same shape as the density grid. Each voxel in the density grid is
    assigned a label. The labels are used to color the surface.
      select: a list of region labels to plot. If 'all' is selected, all regions are plotted. Defaults
    to all
      opacity: The opacity of the surface.
      cscale: a colorscale for the regions. If None, a random colorscale is generated.
      color: color of the cell. Defaults to blue
      stepsize: The step size for the marching cubes algorithm. Defaults to 1
      stylize: if True, will apply some random style choices, purely personal preference. Defaults to
    False
      random_seed: This is used to generate the random colors for the regions. If you want to reproduce
    the same colors, use the same random seed. Defaults to 0
    
    Returns:
      a plotly mesh3d object.
    """

    import numpy as np
    import copy
    import plotly.graph_objects as go
    import ase
    import plotly.express as px
    import trimesh 

    regions_copy = copy.deepcopy(region_labels)
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T

    # * Compute the isosurface using Marching Cubes Lewiner as usual
    from skimage.measure import marching_cubes as mc
    verts, faces, normals, values = mc(dgrid, isovalue, step_size=stepsize)



    # replace default **kwargs_mesh with user defined ones

    name = kwargs_mesh.pop('name', '{} A isosurface'.format(isovalue))
    showlegend = kwargs_mesh.pop('showlegend', True)

    #* creae trimesh from verts and faces 
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals = normals, process=False)
    # mesh.submesh()

    np.random.seed(random_seed)

    if isinstance(regions_copy, np.ndarray) and regions_copy.ndim == 3:
        # *  Make a colorscale using random colors if no colorscale is provided
        if cscale is None:
            max_region = int(regions_copy.max())
            cvals = np.linspace(0, 1, max_region)
            rand_ints = np.random.randint(low=0, high=256, size=(max_region, 3))
            cscale = [[cvals[i],
                       f"rgba({rand_ints[i][0]},{rand_ints[i][1]},{rand_ints[i][2]},{opacity})"] 
                      for i in range(max_region)]
        elif isinstance(cscale, str):
            try:
                cscale = getattr(px.colors.qualitative, cscale)
            except:
                try:
                    cscale = getattr(px.colors.sequential, cscale)
                except:
                    pass

        from scipy.interpolate import RegularGridInterpolator as RGI
        xx = np.linspace(0, regions_copy.shape[0] - 1, regions_copy.shape[0])
        yy = np.linspace(0, regions_copy.shape[1] - 1, regions_copy.shape[1])
        zz = np.linspace(0, regions_copy.shape[2] - 1, regions_copy.shape[2])
        rl = RGI((xx, yy, zz), regions_copy, method='nearest')

        if select != 'all':
            #* make a submesh using the selected regions
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
            
            # Create a mask for faces to select
            face_mask = np.all(np.isin(rl(verts[faces]), select, assume_unique=False), axis=1)
            
            # Get the indices of the faces to keep
            faces_to_keep = np.where(face_mask)[0]
            
            # Use submesh with the face indices
            submesh = mesh.submesh([faces_to_keep], append=True)
            
            # submesh is now a Trimesh object, so we can directly access its vertices and faces
            verts, faces = submesh.vertices, submesh.faces
        
        #* Generate values to color by using the interpolated region labels
        colours = rl(verts)
        #  Find the actual coordinates of the vertices for plotting
        hovertext = [hover_label + str(int(c)) for c in colours]
        true_verts = np.dot(cell, (verts / region_labels.shape).T).T
        
        # Map colours to the full colorscale range
        colour_indices = colours.astype(int) - 1  # Adjust for 0-based indexing
        
        data_plot = go.Mesh3d(x=true_verts[:, 0], y=true_verts[:, 1], z=true_verts[:, 2], 
                              intensity=colour_indices, intensitymode='vertex',
                              hoverinfo='all', hovertext=hovertext, flatshading=False, 
                              colorscale=cscale, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                              opacity=opacity, name=name, showlegend=showlegend, **kwargs_mesh)
    else:
        true_verts = np.dot(cell, (verts / dgrid.shape).T).T
        data_plot = go.Mesh3d(x=true_verts[:, 0], y=true_verts[:, 1], z=true_verts[:, 2],
                              hoverinfo='all', i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                              opacity=opacity, color=color, name=name, showlegend=showlegend, **kwargs_mesh)

    if stylize:
        data_plot.update(lighting=dict(ambient=0.18, diffuse=1, fresnel=0.1, specular=1,
                                       roughness=0.05, facenormalsepsilon=1e-15, vertexnormalsepsilon=1e-15),
                         lightposition=dict(x=10, y=10, z=30), showscale=False)

    return data_plot

def plot_cell(ase_atoms, color='purple', name='cell'):
    """
    It takes an ASE atoms object and returns a plotly scatter3d object that can be used to plot the unit
    cell of the atoms object
    
    Args:
      ase_atoms: the atoms object from ASE
      color: the color of the cell lines. Defaults to purple
      name: the name of the cell. Defaults to cell
    
    Returns:
      A plotly graph object
    """
    
    def draw_cell(A):
        import numpy as np
        avec = A[:, 0].T
        bvec = A[:, 1].T
        cvec = A[:, 2].T
        origin = np.array([0, 0, 0])
        cell_lines = []
        cell_lines.append(origin)
        cell_lines.append(avec)
        cell_lines.append([None, None, None])
        cell_lines.append(origin)
        cell_lines.append(bvec)
        cell_lines.append([None, None, None])
        cell_lines.append(origin)
        cell_lines.append(cvec)
        cell_lines.append([None, None, None])
        cell_lines.append(avec)
        cell_lines.append(avec + bvec)
        cell_lines.append([None, None, None])
        cell_lines.append(bvec)
        cell_lines.append(avec + bvec)
        cell_lines.append([None, None, None])
        cell_lines.append(bvec)
        cell_lines.append(cvec + bvec)
        cell_lines.append([None, None, None])
        cell_lines.append(cvec)
        cell_lines.append(cvec + bvec)
        cell_lines.append([None, None, None])
        cell_lines.append(cvec)
        cell_lines.append(cvec + avec)
        cell_lines.append([None, None, None])
        cell_lines.append(avec)
        cell_lines.append(cvec + avec)
        cell_lines.append([None, None, None])
        cell_lines.append(avec + bvec)
        cell_lines.append(cvec + avec + bvec)
        cell_lines.append([None, None, None])
        cell_lines.append(avec + cvec)
        cell_lines.append(cvec + avec + bvec)
        cell_lines.append([None, None, None])
        cell_lines.append(cvec + bvec)
        cell_lines.append(cvec + avec + bvec)
        cell_lines.append([None, None, None])
        return np.array(cell_lines)

    import plotly.graph_objects as go
    from ase.io import read
    from ase.geometry import cellpar_to_cell
    A_unit = cellpar_to_cell(ase_atoms.cell.cellpar()).T
    # A_unit = ase_atoms.get_cell().T
    cell_lines = draw_cell(A_unit)
    data_cell = go.Scatter3d(z=cell_lines[:, 2], y=cell_lines[:, 1], x=cell_lines[:, 0], mode='lines',
                             line=dict(color=color, width=8), name=name)
    return data_cell


def get_default_colors():
    """[summary]
    Uses the ASE's jmol colorscheme as the default for plotting the atoms.
    :return: [Normalized rgb values of the jmol colorscheme as included within ASE python module. ]
    :rtype: [(N,3) numpy array of floats all within [0,1]]
    """
    from ase.data import colors
    return colors.jmol_colors


# def draw_framework(ase_atoms, colorscheme=get_default_colors(), opacity=[1.0, 1.0], dia_scale=1):
#     """
#     It takes an ASE atoms object and returns two plotly scatter3d objects, one for the atoms and one for the bonds

#     :param ase_atoms: ASE atoms object
#     :param colorscheme: A list of RGB colors
#     :param opacity: The opacity of the atoms and bonds
#     :param dia_scale: This is a scaling factor for the atom sizes, defaults to 1 (optional)
#     :return: A list of plotly traces.
#     """

#     from ase.io import read
#     from ase import neighborlist
#     # data = read(path_to_cif)
#     A_unit = ase_atoms.get_cell().T
#     coord = ase_atoms.get_positions()
#     names = ase_atoms.get_chemical_symbols()

#     def find_covalent_bonds(molecule):
#         import itertools
#         from pymatgen.core.bonds import CovalentBond
#         number_of_sites = len(molecule.sites)
#         bonds = []
#         for a, b in itertools.combinations(range(number_of_sites), 2):
#             if CovalentBond.is_bonded(molecule.sites[a], molecule.sites[b], default_bl=2.0):
#                 bonds.append([a, b])
#         return bonds

#     import numpy as np
#     from pymatgen.core.structure import Molecule
#     print("1. Creating Molecule ...")
#     # coord = np.dot(A_unit, (coord+[0.5,0.5,0.5]).T).T
#     mol = Molecule(names, coord)
#     print("2. Computing bonds ...")
#     bonds = find_covalent_bonds(mol)
#     # print(bonds)
#     # print(np.hstack(bonds))
#     edges_master = []
#     # bonds_atoms_list = []
#     for b in bonds:
#         edges_master.append(coord[b[0]])
#         edges_master.append(coord[b[1]])
#         edges_master.append([None, None, None])

#         # * Append the atom id's that are in each bond, None is a spacer.
#         # bonds_atoms_list.append(b[0])
#         # bonds_atoms_list.append(b[1])
#         # bonds_atoms_list.append(None)

#     edges_master = np.vstack(edges_master)
#     # print(bonds_atoms_list)
#     print("3. Creating Plotly figure traces")
#     # * Figure out how to plot the framework atoms
#     import plotly.graph_objects as go
#     # atom_colors           = {'C':'teal', 'H':'yellow', 'O':'red', 'N':'blue ', 'Zn':'red', 'Zr': 'pink'} # this
#     # line needs to be updated. scatter_colors        = [atom_colors[t] for t in np.array(names)]

#     # * Create a color scale using ASE cpk colorlist.
#     from ase.data import colors
#     from sklearn.preprocessing import minmax_scale
#     cols = np.round(colorscheme * 255, decimals=0).astype(int)
#     colorlist = ["rgb(" + ",".join(c.astype(str)) + ")" for c in cols]
#     # atomic_numbers        = np.vstack(data.get_atomic_numbers())
#     # atom_colors           =   minmax_scale(np.linspace(0,103, 104), feature_range=(0,1))
#     atom_colors = np.linspace(0, 1, len(cols))
#     cscale = [[atom_colors[i], colorlist[i]] for i in range(len(colorlist))]

#     # * Hover-text for each point
#     hovertext = ['Atom No: ' + str(i) + ', Atom type: ' + names[i] for i in range(len(names))]

#     # * Scale the atom sizes beautifully.
#     # atom_dia              = f1.sigma_array*2-f1.sigma
#     atom_dia = 2 * neighborlist.natural_cutoffs(ase_atoms)
#     #   print(atom_dia)
#     scatter_size = np.round(minmax_scale(atom_dia, feature_range=(12, 30)) * dia_scale)

#     bond_atoms_diameter = np.array(atom_dia)[np.hstack(bonds)]
#     bond_atoms_Z = ase_atoms.get_atomic_numbers()[
#                        np.hstack(bonds)] - 1  # The indices in the color scale start at 0, not 1.
#     bond_scatter_size = np.round(minmax_scale(bond_atoms_diameter, feature_range=(12, 35)))

#     data_scatter = go.Scatter3d(x=coord[:, 0], y=coord[:, 1], z=coord[:, 2], mode='markers', hovertext=hovertext,
#                                 marker=dict(opacity=opacity[0], size=scatter_size,
#                                             color=atom_colors[ase_atoms.get_atomic_numbers() - 1], colorscale=cscale,
#                                             sizemode='diameter'), showlegend=True, name='framework atoms')
#     data_edges = go.Scatter3d(x=edges_master[:, 0], y=edges_master[:, 1], z=edges_master[:, 2],
#                               line=dict(width=15, color='grey'), mode='lines', opacity=opacity[1],
#                               name='framework bonds', showlegend=True)
#     # data_framework          = go.Scatter3d(x=edges_master[:,0] , y=edges_master[:,1], z=edges_master[:,2] ,
#     # line=dict(width=15, color='grey'), mode='markers+lines', hovertext=hovertext, marker=dict(opacity=opacity[0],
#     # size=bond_scatter_size, color=bond_atoms_Z, colorscale=cscale), opacity=opacity[1], name='Framework')
#     return data_scatter, data_edges


def scatter3d(X, name='trace name', mode='markers', marker_color='blue', marker_size=12, marker_line_color='white',
              marker_symbol='circle', marker_line_width=1, line_color='black', line_width=2, hovertext=None, text=None,
              **kwargs):
    """
    It takes a 3D array of points and returns a Plotly scatter3d object

    :param X: the data matrix, with each row being a data point and each column being a feature
    :param name: name of the trace, defaults to trace name here (optional)
    :param mode: 'markers', 'lines', 'lines+markers', 'text', 'none', defaults to markers (optional)
    :param marker_color: color of the points, defaults to blue (optional)
    :param marker_size: size of the marker, defaults to 12 (optional)
    :param marker_line_color: The color of the line around the marker, defaults to white (optional)
    :param marker_symbol: https://plotly.com/python/reference/#scatter-marker-symbol, defaults to circle (optional)
    :param marker_line_width: The width of the lines around the markers, defaults to 1 (optional)
    :param line_color: color of the line connecting the points, defaults to black (optional)
    :param line_width: width of the line connecting the points, defaults to 2 (optional)
    :param hovertext: text to show when hovering over a point
    :param text: text to display when hovering over a point
    :return: A plotly scatter3d object
    """
    if hovertext is None:
        hovertext = ['entry '+ str(i+1) for i in range(len(X))]
    import plotly.graph_objects as go
    return go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode=mode,
                        marker=dict(color=marker_color, size=marker_size, symbol=marker_symbol,
                                    line=dict(color=marker_line_color, width=marker_line_width)), name=name,
                        line=dict(color=line_color, width=line_width), text=None, hovertext=hovertext, **kwargs)


# *  Function to plot the region adjacency graph
def plot_rag(G, plot_with='maxima', **kwargs):
    import numpy as np
    import plotly.graph_objects as go

    # If there are no edges, plot only the nodes
    if G.number_of_edges() == 0:
        node_coords = []
        node_labels = []
        for node in G.nodes():
            if node != 0:  # Skip node 0
                node_coords.append(G.nodes[node][plot_with])
                node_labels.append(f'Node: {G.nodes[node]["labels"][0]}')
        
        if node_coords:
            node_coords = np.array(node_coords)
            return go.Scatter3d(
                x=node_coords[:, 0],
                y=node_coords[:, 1],
                z=node_coords[:, 2],
                mode='markers',
                hovertext=node_labels,
                **kwargs
            )
        else:
            # If there are no nodes except 0, return an empty scatter plot
            return go.Scatter3d(x=[], y=[], z=[], mode='markers', **kwargs)

    # If there are edges, proceed with the original edge plotting
    edge_trace = []
    hovertext = []
    for edge in G.edges():
        if 0 not in edge:
            n1, n2 = edge
            edge_trace.append(G.nodes[n1][plot_with])
            edge_trace.append(G.nodes[n2][plot_with])
            edge_trace.append([None, None, None])
            hovertext.append(f'Node: {G.nodes[n1]["labels"][0]}')
            hovertext.append(f'Node: {G.nodes[n2]["labels"][0]}')
            hovertext.append(None)

    if edge_trace:
        edge_trace = np.vstack(edge_trace)
        return go.Scatter3d(
            x=edge_trace[:, 0],
            y=edge_trace[:, 1],
            z=edge_trace[:, 2],
            mode='lines+markers',
            hovertext=hovertext,
            **kwargs
        )
    else:
        # If all edges involve node 0, plot only the nodes
        return plot_rag(G, plot_with, **kwargs)

def plot_rag_separate_traces(G, plot_with='maxima', **kwargs):
    """Plot region adjacency graph with separate traces for nodes and edges.
    
    Args:
        G: NetworkX graph
        plot_with: Node attribute to use for coordinates ('maxima' or other)
        **kwargs: Additional arguments passed to traces
            node_kwargs: Dict of arguments for node trace
            edge_kwargs: Dict of arguments for edge trace
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Get kwargs specific to nodes and edges
    node_kwargs = kwargs.pop('node_kwargs', {})
    edge_kwargs = kwargs.pop('edge_kwargs', {})
    
    # Get colorscale
    cscale = kwargs.get('cscale', px.colors.sequential.Turbo)
    
    # Create color mapping for nodes
    node_to_color = {node: px.colors.sample_colorscale(cscale, (i)/len(G.nodes))[0] 
                    for i, node in enumerate(G.nodes)}

    # Initialize lists for node trace
    node_coords = []
    node_labels = []
    node_trace_colors = []

    # Collect node data
    for node in G.nodes():
        if node != 0:  # Skip node 0
            node_coords.append(G.nodes[node][plot_with])
            node_labels.append(f'Node: {G.nodes[node]["labels"][0]}')
            node_trace_colors.append(node_to_color[node])

    traces = []
    
    # Create node trace if we have nodes
    if node_coords:
        node_coords = np.array(node_coords)
        node_trace = go.Scatter3d(
            x=node_coords[:, 0],
            y=node_coords[:, 1],
            z=node_coords[:, 2],
            mode='markers',
            hovertext=node_labels,
            marker=dict(
                color=node_trace_colors,
                size=node_kwargs.pop('marker_size', 10),
                symbol=node_kwargs.pop('marker_symbol', 'circle')
            ),
            name='Internal Nodes',
            legendgroup='Pore Graph',
            legendgrouptitle_text='Pore Graph',
            **node_kwargs
        )
        traces.append(node_trace)

    # If there are edges, create edge trace
    if G.number_of_edges() > 0:
        edge_coords = []
        edge_colors = []
        edge_labels = []
        
        for edge in G.edges():
            if 0 not in edge:
                n1, n2 = edge
                edge_coords.append(G.nodes[n1][plot_with])
                edge_coords.append(G.nodes[n2][plot_with])
                edge_coords.append([None, None, None])
                
                edge_colors.extend([node_to_color[n1], node_to_color[n2], None])
                edge_labels.extend([
                    f'Edge: {G.nodes[n1]["labels"][0]} - {G.nodes[n2]["labels"][0]}',
                    f'Edge: {G.nodes[n1]["labels"][0]} - {G.nodes[n2]["labels"][0]}',
                    None
                ])

        if edge_coords:
            edge_coords = np.array(edge_coords)
            edge_trace = go.Scatter3d(
                x=edge_coords[:, 0],
                y=edge_coords[:, 1],
                z=edge_coords[:, 2],
                mode='lines',
                hovertext=edge_labels,
                line=dict(
                    width=edge_kwargs.pop('line_width', 4),
                    color=edge_kwargs.pop('line_color', 'red')
                ),
                name='Internal Connections',
                legendgroup='Pore Graph',
                **edge_kwargs
            )
            traces.append(edge_trace)

    return traces

def plot_pore_graph(G):
    import networkx as nx
    import plotly.graph_objects as go
    import numpy as np
    coords = [n[1]['node_coords'] for n in G.nodes(data=True)]
    coords = np.vstack(coords)
    # print(coords)
    edges_list = []
    for edge in list(G.edges()):
        edges_list.append(G.nodes(data=True)[edge[0]]['node_coords'])
        edges_list.append(G.nodes(data=True)[edge[1]]['node_coords'])
        edges_list.append([None, None, None])

    nodes_trace = scatter3d(np.array(coords), name='nodes', showlegend=True, legendgroup='pore_graph',
                            legendgrouptitle_text='pore graph')
    edges_trace = scatter3d(np.array(edges_list), mode='lines', name='edges', line_width=8, showlegend=True,
                            legendgroup='pore_graph', legendgrouptitle_text='pore graph')

    return nodes_trace, edges_trace


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    # Do something if this file is invoked on its own
    print("You probably shouldn't call this by itself")


# def draw_framework2(ase_atoms, bonds=None, bond_orders=None, show_aromatics=True, show_atom_labels=False,
#                     bond_data=None, bond_color=None, bond_opacity=0.9, colorscheme=get_default_colors(),
#                     atoms_opacity=0.9, dia_scale=1):
#     """
#         It takes an ASE atoms object and returns a list of plotly traces that can be used to plot the
#         framework
        
#         :param ase_atoms: ASE atoms object
#         :param bonds: list of bonds in the form of [[0,1],[1,2],...]
#         :param bond_orders: A list of bond orders for each bond in the bonds list
#         :param show_rings: If True, the function will detect and plot aromatic rings, defaults to True
#         (optional)
#         :param bond_data: A dictionary of bond data. The keys are the names of the bond data, and the values
#         are the bond data. The bond data should be a list of the same length as the number of bonds in the
#         structure
#         :param colorscheme: A dictionary of colors for each element. The default is the ASE CPK color scheme
#         :param atoms_opacity: Opacity of the atoms
#         :param dia_scale: This is a scaling factor for the atom sizes, defaults to 1 (optional)
#         :param default_ring_sizes: list of integers, default = [5,6,7,8]
#         :return: A list of plotly.graph_objects.Scatter3d objects.
#         """

#     # from ase.io import read
#     from ase import neighborlist
#     from .analyze import update_with_covalent_bonds, find_aromatic_cycles_in_sgraph
#     # from ase.data import covalent_radii
#     # data = read(path_to_cif)
#     # A_unit = data.get_cell().T
#     coord = ase_atoms.get_positions()
#     names = ase_atoms.get_chemical_symbols()

#     import numpy as np
#     from pymatgen.core import Molecule
#     from .analyze import sgraph_from_atoms
#     # print("1. Creating Molecule ...")
#     # mol = Molecule(names, coord)

#     # print(bonds_atoms_list)
#     print("Creating Plotly figure traces")
#     # * Figure out how to plot the framework atoms

#     import plotly.graph_objects as go

#     # * Create a color scale using ASE cpk colorlist.
#     from ase.data import colors
#     from sklearn.preprocessing import minmax_scale
#     cols = np.round(colorscheme * 255, decimals=0).astype(int)
#     colorlist = ["rgb(" + ",".join(c.astype(str)) + ")" for c in cols]
#     # atomic_numbers        = np.vstack(data.get_atomic_numbers())
#     # atom_colors           =   minmax_scale(np.linspace(0,103, 104), feature_range=(0,1))
#     atom_colors = np.linspace(0, 1, len(cols))
#     cscale = [[atom_colors[i], colorlist[i]] for i in range(len(colorlist))]

#     # * Hovertext for each point
#     hovertext = ['Atom No: ' + str(i) + ', Atom type: ' + names[i] for i in range(len(names))]

#     # * Scale the atom sizes beautifully.
#     # atom_dia              = f1.sigma_array*2-f1.sigma
#     atom_dia = 2 * neighborlist.natural_cutoffs(ase_atoms)

#     # # * Create a color scale using ASE cpk colorlist.
#     # from ase.data import colors
#     # from sklearn.preprocessing import minmax_scale
#     # colors_in_mof = colorscheme[ase_atoms.get_atomic_numbers()]
#     # cols = np.round(colors_in_mof * 255, decimals=0).astype(np.int)
#     # colorlist = ["rgb(" + ",".join(c.astype(np.str)) + ")" for c in cols]

#     # # * Hovertext for each point
#     # hovertext = ['Atom No: ' + str(i) + ', Atom type: ' + names[i] for i in range(len(names))]

#     # # * Scale the atom sizes beautifully.
#     # atom_dia = 2 * neighborlist.natural_cutoffs(ase_atoms)
#     # print(atom_dia)
#     scatter_size = np.round(minmax_scale(atom_dia, feature_range=(12, 30)) * dia_scale)

#     traces = []
#     print("1. Creating atoms trace\n")

#     trace_scatter = go.Scatter3d(x=coord[:, 0], y=coord[:, 1], z=coord[:, 2], mode='markers', hovertext=hovertext,
#                                  marker=dict(opacity=atoms_opacity, size=scatter_size,
#                                              color=atom_colors[ase_atoms.get_atomic_numbers() - 1],
#                                              sizemode='diameter', colorscale=cscale), name='framework atoms',
#                                  text=names, textfont=dict(color=colorlist, size=20), surfaceaxis=-1)
#     traces.append(trace_scatter)

#     if bonds is None:
#         ase_atoms_updated = update_with_covalent_bonds(ase_atoms)
#         bonds = ase_atoms_updated.info['covalent_bonds']
#         if bond_orders is None:
#             bond_orders = ase_atoms_updated.info['covalent_bond_orders']
#         else:
#             raise Exception("You can't provide bond orders without providing a list of bonds!")
#     else:
#         ase_atoms_updated = ase_atoms

#     print("2. Creating bonds' trace\n")
#     edges_master = []
#     for b in bonds:
#         edges_master.append(coord[b[0]])
#         edges_master.append(coord[b[1]])
#         edges_master.append([None, None, None])

#     if bond_color is None:
#         bond_color = np.repeat(bond_orders, 3)
#     elif type(bond_color) == str:
#         bond_color = bond_color
#     else:
#         bond_color = 'rgba(0,0,0,0.35)'
#     edges_master = np.vstack(edges_master)
#     print(bond_color)
#     try:
#         trace_edges = go.Scatter3d(x=edges_master[:, 0], y=edges_master[:, 1], z=edges_master[:, 2],
#                                    line=dict(width=15, color=bond_color, colorscale='aggrnyl',
#                                              colorbar=dict(thickness=20, len=0.6, title='bond order')),
#                                    opacity=bond_opacity, mode='lines',
#                                    name='framework bonds')
#     except:
#         trace_edges = go.Scatter3d(x=edges_master[:, 0], y=edges_master[:, 1], z=edges_master[:, 2],
#                                    line=dict(width=15, color='rgba(150,150,150,0.35)', colorscale='aggrnyl'),
#                                    opacity=bond_opacity, mode='lines',
#                                    name='framework bonds')
#     traces.append(trace_edges)

#     if show_aromatics:
#         print('\t2-a. Detecting and plotting aromatic rings\n')
#         # show aromatic rings
#         # if bond order data is provided

#         # Detect and show requested rings
#         # bond_types = nx.get_edge_attributes(G, 'bond_type')
#         # Detect aromatic rings
#         G = sgraph_from_atoms(ase_atoms_updated)
#         isomorphic_aromatic_cycles = find_aromatic_cycles_in_sgraph(G)
#         # Highlight and circle all aromatic rings

#         ring_traces = plot_aromatic_rings(ase_atoms=ase_atoms_updated,
#                                           isomorphic_aromatic_cycles=isomorphic_aromatic_cycles)
#         traces = traces + list(ring_traces)
#         # circle_data=[]
#         # from numpy.core.numeric import cross
#         # theta = np.linspace(0, 2*np.pi, 100)
#         # xc = 0.5*np.cos(theta)
#         # yc = 0.5*np.sin(theta)
#         # zc = np.ones(np.shape(xc))
#         # circle_coord = np.array([xc, yc, zc]).T

#         # from scipy.spatial.transform import Rotation as R

#         # # ring circles
#         # for ring in isomorphic_aromatic_cycles:
#         #     ring_coord = coord[ring]
#         #     ring_center = np.mean(ring_coord, axis=0)
#         #     ring_normal = np.cross(ring_coord[3]-ring_coord[2],ring_coord[2]-ring_coord[1])
#         #     ring_normal = ring_normal/np.linalg.norm(ring_normal)
#         #     r = R.align_vectors(ring_normal.reshape(-1,3),np.array([0,0,1]).reshape(-1,3))
#         #     new_circle = r[0].apply(circle_coord)
#         #     new_circle= new_circle + ring_center-np.mean(new_circle, axis=0)
#         #     circle_data.append(new_circle)
#         #     circle_data.append([None, None, None])
#         # circle_data = np.vstack(circle_data)
#         # data_circle = go.Scatter3d(x=circle_data[:,0], y=circle_data[:,1], z=circle_data[:,2], mode='lines', line=dict(color='white',width=2), showlegend=False, name='Aromatic Cloud', legendgroup='aromatic circles')

#         # traces.append(data_circle)

#         # # ring highlight
#         # ring_centers = []
#         # for ring in isomorphic_aromatic_cycles:
#         #     ring_coord = coord[ring]
#         #     ring_centers.append(np.mean(ring_coord, axis=0))
#         #     ring_highlight_trace = go.Mesh3d(x=ring_coord[:,0],y=ring_coord[:,1],z=ring_coord[:,2], color='lightskyblue', opacity=0.1, alphahull=0, name='Benzene ring highlight')
#         #     traces.append(ring_highlight_trace)

#     if bool(bond_data):
#         print('4. Creating traces for additional bond data provided')
#         for key in bond_data.keys():
#             true_edges = edges_master[edges_master != None].reshape(-2, 3)
#             true_edges = 0.5 * (true_edges[0::2] + true_edges[1::2])
#             trace_bond_data = go.Scatter3d(x=true_edges[:, 0], y=true_edges[:, 1], z=true_edges[:, 2], mode='text',
#                                            textfont=dict(color='white', size=9), text=list(map(str, bond_data[key])),
#                                            name=key)
#             traces.append(trace_bond_data)
#     if show_atom_labels:
#         trace_atom_labels = go.Scatter3d(x=coord[:, 0], y=coord[:, 1], z=coord[:, 2], mode='text',
#                                          textfont=dict(color=colorlist, size=20), text=names,
#                                          name='atom labels')
#         traces.append(trace_atom_labels)
#     return traces


def plot_aromatic_rings(ase_atoms, isomorphic_aromatic_cycles=None,
                        kwargs_rings=dict(mode='lines', line=dict(color='white', width=2), showlegend= False,
                                        legendgroup='aromatics', legendgrouptitle_text='Aromatic rings', name='Aromatic rings'),
                        kwargs_highlight=dict(color='skyblue', opacity=0.5, alphahull=-1, showlegend=False,
                        legendgroup='aromatics', legendgrouptitle_text='Aromatic rings', name='Aromatic highlights')):
    """
    It takes in a graph and a list of aromatic cycles, and returns a tuple of plotly traces that can be
    used to plot the aromatic cycles
    
    :param sgraph: the sgraph object
    :param isomorphic_aromatic_cycles: a list of lists of node indices that are aromatic rings. If you
    don't have this, you can set it to None and the function will try to find it for you
    :param highlight_color: The color of the highlighted aromatic rings, defaults to lightskyblue
    (optional)
    :return: A tuple of the data_circle and ring_highlight_traces
    """
    from .analyze import find_aromatic_cycles_in_sgraph, update_with_covalent_bonds, sgraph_from_atoms

    if not ('covalent_bonds' in ase_atoms.info.keys()):
        ase_atoms = update_with_covalent_bonds(ase_atoms)

    sgraph = sgraph_from_atoms(ase_atoms)
    import plotly.graph_objects as go
    import numpy as np

    if isomorphic_aromatic_cycles is None:
        print('isomorphic aromatic cycles is None, trying to calculate')
        try:
            isomorphic_aromatic_cycles = find_aromatic_cycles_in_sgraph(sgraph)
            # print(isomorphic_aromatic_cycles)
        except:
            Exception('Cannot find aromatic cycles from sgraph! Please provide as input.')
    circle_data = []
    from numpy.core.numeric import cross
    theta = np.linspace(0, 2 * np.pi, 100)
    xc = 0.5 * np.cos(theta)
    yc = 0.5 * np.sin(theta)
    zc = np.ones(np.shape(xc))
    circle_coord = np.array([xc, yc, zc]).T

    from scipy.spatial.transform import Rotation as R
    import networkx as nx
    # coord  = nx.get_node_attributes(sgraph, "position")# ring circles
    try:
        coord = np.vstack(list(nx.get_node_attributes(sgraph, "position").values()))
    except:
        raise Exception("Cannot find position data in your sgraph")
    # print(isomorphic_aromatic_cycles)
    if isomorphic_aromatic_cycles is not None:
        if len(isomorphic_aromatic_cycles) >0:
            for ring in isomorphic_aromatic_cycles:
                ring_coord = coord[ring]
                ring_center = np.mean(ring_coord, axis=0)
                ring_normal = np.cross(ring_coord[3] - ring_coord[2], ring_coord[2] - ring_coord[1])
                ring_normal = ring_normal / np.linalg.norm(ring_normal)
                r = R.align_vectors(ring_normal.reshape(-1, 3), np.array([0, 0, 1]).reshape(-1, 3))
                new_circle = r[0].apply(circle_coord)
                new_circle = new_circle + ring_center - np.mean(new_circle, axis=0)
                circle_data.append(new_circle)
                circle_data.append([None, None, None])
            circle_data = np.vstack(circle_data)
            data_circle = go.Scatter3d(x=circle_data[:, 0], y=circle_data[:, 1], z=circle_data[:, 2], **kwargs_rings)

            import pyvista as pv
            ring_coords = [coord[r] for r in isomorphic_aromatic_cycles]
            ring_centers = [np.mean(coord[r], axis=0) for r in isomorphic_aromatic_cycles]
            # ring_normals = [np.cross(co[3]-co[2], co[2]-co[1]) for co in ring_coords]
            # ring_points = [np.vstack((r,c)) for r,c in zip(ring_coords,ring_centers)]
            polys = [pv.Polygon(center=c, radius=np.linalg.norm(r[0] - c) * 0.95, normal=np.cross(r[3] - r[2], r[2] - r[1]),
                                n_sides=len(r)).delaunay_2d() for r, c in zip(ring_coords, ring_centers)]

            # polys = [pv.PolyData(rp).delaunay_2d(inplace=True).subdivide(2, subfilter='butterfly', inplace=True) for rp in ring_points]
            # polygons = [pv.Polygon(center = c ,radius=np.linalg.norm(co[0]-c), normal = n, n_sides = len(co)).triangulate() for  co,c,n in zip(ring_coords, ring_centers, ring_normals)]
            verts = [p.points for p in polys]
            faces = [p.faces.reshape((-1, 4))[:, 1:] for p in polys]
            ring_highlight_traces = [
                go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], **kwargs_highlight) for v, f in
                zip(verts, faces)]

        # aromatic_graphs = [sgraph.subgraph(g) for g in isomorphic_aromatic_cycles]
        # aromatic_vectorize = [np.vectorize({an:i for i,an in enumerate(a.nodes())}.get) for a in aromatic_graphs]
        # aromatic_tri_faces = [v(np.vstack(list(a.edges()))) for a,v in zip(aromatic_graphs,aromatic_vectorize)]
        # from icecream import ic
        # ic.disable()
        # # ring highlight

        # ring_centers = []
        # ring_highlight_traces = []

        # for i,ring in enumerate(isomorphic_aromatic_cycles):
        #     ring_coord = coord[ring]
        #     ring_centers.append(np.mean(ring_coord, axis=0))

        #     faces = np.insert(aromatic_tri_faces[i],2,len(ring_coord), axis=1)
        #     ring_coord = np.vstack((ring_coord , (np.mean(ring_coord, axis=0))))
        #     ic(faces)
        #     ic(ring_coord)
        #     # ring_highlight_trace = go.Mesh3d(x=ring_coord[:,0],y=ring_coord[:,1],z=ring_coord[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2],**kwargs_highlight)
        #     ring_highlight_trace = go.Mesh3d(x=highlight.vertices[:,0],y=highlight.vertices[:,1],z=highlight.vertices[:,2], i=highlight.faces[:,0], j=highlight.faces[:,1], k=highlight.faces[:,2],**kwargs_highlight)
        #     # ring_highlight_trace = go.Surface(x=ring_coord[:,0],y=ring_coord[:,1],z=ring_coord[:,2], **kwargs_highlight)
        #     ring_highlight_traces.append(ring_highlight_trace)

            return tuple([data_circle] + ring_highlight_traces)
    else:
        return tuple([])


def plot_cycle_highlights(sgraph, cycles_to_highlight,
                          kwargs_highlight=dict(color='lightskyblue', opacity=0.3, alphahull=-1,
                                                name='Aromatic ring highlight')):
    """
    It takes in a graph and a list of aromatic cycles, and returns a tuple of plotly traces that can be
    used to plot the aromatic cycles
    
    :param sgraph: the sgraph object
    :param isomorphic_aromatic_cycles: a list of lists of node indices that are aromatic rings. If you
    don't have this, you can set it to None and the function will try to find it for you
    :param highlight_color: The color of the highlighted aromatic rings, defaults to lightskyblue
    (optional)
    :return: A tuple of the data_circle and ring_highlight_traces
    """

    # from .analyze import find_aromatic_cycles_in_sgraph
    import plotly.graph_objects as go
    import numpy as np

    # if cycles_to_hightlight is None:
    #     try:
    #         cycles_to_hightlight = find_aromatic_cycles_in_sgraph(sgraph)
    #     except:
    #         Exception('Cannot find aromatic cycles from sgraph! Please provide as input.')
    # circle_data=[]
    # from numpy.core.numeric import cross
    # theta = np.linspace(0, 2*np.pi, 100)
    # xc = 0.5*np.cos(theta)
    # yc = 0.5*np.sin(theta) 
    # zc = np.ones(np.shape(xc))
    # circle_coord = np.array([xc, yc, zc]).T

    # from scipy.spatial.transform import Rotation as R
    import networkx as nx
    # coord  = nx.get_node_attributes(sgraph, "position")# ring circles

    try:
        coord = np.vstack(list(nx.get_node_attributes(sgraph, "position").values()))
    except:
        raise Exception("Cannot find position data in your sgraph")
    # for ring in cycles_to_hightlight:
    #     ring_coord = coord[ring]
    #     ring_center = np.mean(ring_coord, axis=0)
    #     ring_normal = np.cross(ring_coord[3]-ring_coord[2],ring_coord[2]-ring_coord[1])
    #     ring_normal = ring_normal/np.linalg.norm(ring_normal)
    #     r = R.align_vectors(ring_normal.reshape(-1,3),np.array([0,0,1]).reshape(-1,3))
    #     new_circle = r[0].apply(circle_coord)
    #     new_circle= new_circle + ring_center-np.mean(new_circle, axis=0) 
    #     circle_data.append(new_circle)
    #     circle_data.append([None, None, None])
    # circle_data = np.vstack(circle_data)
    # data_circle = go.Scatter3d(x=circle_data[:,0], y=circle_data[:,1], z=circle_data[:,2], **kwargs_rings)

    import pyvista as pv
    ring_coords = [coord[r] for r in cycles_to_highlight]
    ring_centers = [np.mean(coord[r], axis=0) for r in cycles_to_highlight]
    # ring_normals = [np.cross(co[3]-co[2], co[2]-co[1]) for co in ring_coords]
    ring_points = [np.vstack((r, c)) for r, c in zip(ring_coords, ring_centers)]
    polys = [pv.PolyData(rp).delaunay_2d().subdivide_adaptive() for rp in ring_points]
    # polygons = [pv.Polygon(center = c ,radius=np.linalg.norm(co[0]-c), normal = n, n_sides = len(co)).triangulate() for  co,c,n in zip(ring_coords, ring_centers, ring_normals)]
    verts = [p.points for p in polys]
    faces = [p.faces.reshape((-1, 4))[:, 1:] for p in polys]
    cycle_highlight_traces = [
        go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], **kwargs_highlight) for v, f in
        zip(verts, faces)]

    # aromatic_graphs = [sgraph.subgraph(g) for g in isomorphic_aromatic_cycles]
    # aromatic_vectorize = [np.vectorize({an:i for i,an in enumerate(a.nodes())}.get) for a in aromatic_graphs]
    # aromatic_tri_faces = [v(np.vstack(list(a.edges()))) for a,v in zip(aromatic_graphs,aromatic_vectorize)]
    # from icecream import ic
    # ic.disable()
    # # ring highlight

    # ring_centers = []
    # ring_highlight_traces = []

    # for i,ring in enumerate(isomorphic_aromatic_cycles):
    #     ring_coord = coord[ring]
    #     ring_centers.append(np.mean(ring_coord, axis=0))

    #     faces = np.insert(aromatic_tri_faces[i],2,len(ring_coord), axis=1)
    #     ring_coord = np.vstack((ring_coord , (np.mean(ring_coord, axis=0))))
    #     ic(faces)
    #     ic(ring_coord)
    #     # ring_highlight_trace = go.Mesh3d(x=ring_coord[:,0],y=ring_coord[:,1],z=ring_coord[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2],**kwargs_highlight)
    #     ring_highlight_trace = go.Mesh3d(x=highlight.vertices[:,0],y=highlight.vertices[:,1],z=highlight.vertices[:,2], i=highlight.faces[:,0], j=highlight.faces[:,1], k=highlight.faces[:,2],**kwargs_highlight)
    #     # ring_highlight_trace = go.Surface(x=ring_coord[:,0],y=ring_coord[:,1],z=ring_coord[:,2], **kwargs_highlight)
    #     ring_highlight_traces.append(ring_highlight_trace)

    return tuple(cycle_highlight_traces)


# * Functions for ease
def trimesh_from_pyvista(pv_mesh):
    """
    `trimesh_from_pyvista(pv_mesh)` takes a PyVista mesh and returns a Trimesh mesh
    
    :param pv_mesh: a pyvista mesh object
    :return: A trimesh object.
    """
    import trimesh
    faces_array = pv_mesh.faces.reshape((pv_mesh.n_faces_strict, 4))[:, 1:]
    tmesh = trimesh.Trimesh(pv_mesh.points, faces_array)
    return tmesh


def spheres_for_atoms(ase_atoms, radii=None, scale=0.8, colormap=None, theta_resolution=10, phi_resolution=10):
    """
    It takes an ASE atoms object and returns a list of trimesh objects, each of which is a sphere with
    radius and center corresponding to the atom's covalent radius and position.
    
    :param ase_atoms: an ase atoms object
    :param radii: the radii of the spheres. If None, the covalent radii from ase.data.covalent_radii are
    used
    :param scale: the scale of the sphere radius
    :return: A list of trimesh objects.
    """
    import numpy as np
    import pyvista as pv
    from ase.data import covalent_radii
    from ase.data.colors import cpk_colors
    natoms = ase_atoms.get_global_number_of_atoms()
    if radii is None:
        radii = covalent_radii[ase_atoms.get_atomic_numbers()]
    if colormap is None:
        colormap = cpk_colors
    positions = ase_atoms.get_positions()
    spheres = [pv.Sphere(radius=radii[i] * scale, center=positions[i], theta_resolution=theta_resolution,
                         phi_resolution=phi_resolution) for i in range(natoms)]
    spheres_tmesh = [trimesh_from_pyvista(s) for s in spheres]
    atomic_numbers = ase_atoms.get_atomic_numbers()
    for i, s in enumerate(spheres_tmesh):
        # s.visual =trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
        # s.visual.material.baseColorFactor= colormap[atomic_numbers[i]]
        # s.visual.material.metallicFactor=1.0
        # s.visual.material.roughnesFactor=0.15
        s.visual.vertex_colors = colormap[atomic_numbers[i]]
    return spheres_tmesh


def spheres_for_positions(positions, radii=None, scale=0.8, color=[[255, 255, 255, 1]], theta_resolution=20,

                          phi_resolution=20):

    """
    It takes a list of positions and a list of radii and returns a list of trimesh objects that are
    spheres centered at the positions with the given radii
    
    :param positions: list of lists of floats, the positions of the atoms
    :param radii: list of floats, default None
    :param scale: the size of the spheres
    :param color: a list of colors for each atom. If you want all atoms to be the same color, just put
    the one color in a list
    :param theta_resolution: The number of points around the polar angle, defaults to 20
    (optional)
    :param phi_resolution: The number of points around the azimuthal angle, defaults to 20 (optional)
    :return: A list of trimesh objects.
    """
    import numpy as np
    import pyvista as pv
    from ase.data import covalent_radii
    from ase.data.colors import cpk_colors
    natoms = len(positions)
    # if radii is None:
    #     radii = covalent_radii[ase_atoms.get_atomic_numbers()]
    # if colormap is None:
    #     colormap = cpk_colors
    if radii is None:
        radii = [3.0] * len(positions)
    spheres = [pv.Sphere(radius=radii[i] * scale, center=positions[i], theta_resolution=theta_resolution,
                         phi_resolution=phi_resolution) for i in range(natoms)]
    spheres_tmesh = [trimesh_from_pyvista(s) for s in spheres]
    # atomic_numbers =ase_atoms.get_atomic_numbers()
    if len(color) == 1:
        color = color * len(positions)
    for i, s in enumerate(spheres_tmesh):
        # s.visual =trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
        # s.visual.material.baseColorFactor= colormap[atomic_numbers[i]]
        # s.visual.material.metallicFactor=1.0
        # s.visual.material.roughnesFactor=0.15
        s.visual.vertex_colors = color[i]
    return spheres_tmesh
def plot_snapshot_raspa_pdb(pdb_file, index, natoms, bonds_in_ref=[], trace_name='adsorbates', pbc=[1, 1, 1]):
    """Plots a snapshot of a RASPA PDB file.

    Args:
        pdb_file (str): The path to the PDB file.
        index (int): The index of the snapshot to plot.
        natoms (int): The number of atoms in a single molecule.
        bonds_in_ref (list, optional): A list of bonds to reference. Defaults to an empty list. If the list is empty, 
        it assumes that the bonds are 1-2, 2-3, 3-4, etc.
        trace_name (str, optional): The name of the trace for the plot. Defaults to 'adsorbates'.
        pbc (list, optional): A list indicating periodic boundary conditions. Defaults to [1, 1, 1].

    Returns:
        mol_mesh_trace: The mesh trace for the plotted molecules.
    """
    
    from .data import read_raspa_pdb 
    import numpy as np

    data = read_raspa_pdb(pdb_file)
    coords = data['coords'][index]
    sym = data['symbols'][index]
    sym = [str(c) for c in sym]
    cell = data['cells'][index]
    nmolecules = len(coords) // natoms  # number of CO2 molecules

    # Convert to ASE Atoms object
    from ase import Atoms 
    atoms = Atoms(symbols=sym, positions=coords, cell=cell, pbc=[1, 1, 1])

    # We also need the CO2 bonds for plotting
    # In this case, this is easy, since we know the bonds are 1-2 and 2-3
    if len(bonds_in_ref) == 0:
        bonds = []
        for i in range(nmolecules):
            for a in range(natoms - 1):
                bonds.append([i * natoms + a, i * natoms + a + 1])
    else:
        bonds = []
        for i in range(nmolecules):
            for b in bonds_in_ref:
                bonds.append([i * natoms + b[0], i * natoms + b[1]])

    mol_mesh = mesh_for_atoms(ase_atoms=atoms, bonds=bonds)
    mol_mesh_trace = plot_mesh([mol_mesh], name=trace_name, stylize=True)
    atoms.info['covalent_bonds'] = bonds
    mol_mesh_trace = add_hoverdata_to_atoms_traces(mol_mesh_trace, atoms)
    return mol_mesh_trace

def cylinders_for_bonds(bonds, ase_atoms, radius=0.25, bond_rgba_array=[127, 128, 129, 255], cylinder_resolution=15):
    """
    It takes a list of bonds (each bond is a list of two indices) and an ASE atoms object, and returns a
    list of trimesh objects, each of which is a cylinder centered on the bond with radius 0.25 and
    height equal to half the bond length
    
    :param bonds: a list of pairs of atom indices
    :param ase_atoms: an ASE atoms object
    :return: A list of trimesh objects.
    """
    import numpy as np
    import pyvista as pv
    positions = ase_atoms.get_positions()
    bond_coords = np.array([positions[b] for b in bonds])
    directions = bond_coords[:, 1] - bond_coords[:, 0]
    centers = (bond_coords[:, 0] + bond_coords[:, 1]) / 2
    heights = np.linalg.norm(directions, axis=1)
    cylinders = [
        pv.Cylinder(center=c, direction=d, resolution=cylinder_resolution, radius=radius, height=h).triangulate() for
        c, d, h in zip(centers, directions, heights)]
    cylinders_tmesh = [trimesh_from_pyvista(c) for c in cylinders]
    for i, c in enumerate(cylinders_tmesh):
        # c.visual =trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
        # c.visual.material.baseColorFactor= bond_rgba_array #cpk_colors[atomic_numbers[i]]
        # c.visual.material.metallicFactor=1.0
        # c.visual.material.roughnesFactor=0.15
        c.visual.vertex_colors = bond_rgba_array  # cpk_colors[atomic_numbers[i]]
    return cylinders_tmesh


def mesh_for_atoms(ase_atoms, bonds=None, colormap=None, bond_rgba_array=[127, 128, 129, 255],
                   kwargs_spheres=dict(scale=0.8, radii=None), kwargs_cylinders=dict(radius=0.25), return_scene=False):
    """
    > It takes an ASE atoms object and returns a trimesh object that can be rendered in a Jupyter
    notebook
    
    :param ase_atoms: an ASE atoms object
    :param bonds: a list of bonds, where each bond is a list of two atom indices
    :param colormap: a dictionary of colors for each atomic number
    :param bond_rgba_array: The color of the bonds
    :param kwargs_spheres: dict of arguments for the spheres, see the spheres_for_atom function
    :param kwargs_cylinders: radius of the cylinder, see the cylinders_for_bonds function
    :param return_scene: if True, returns a trimesh.Scene object, which can be rendered with the
    trimesh.Scene.show() method, defaults to False (optional)
    :return: A trimesh.Scene object.
    """

    import trimesh
    import numpy as np
    if colormap is None:
        from ase.data.colors import cpk_colors
        colormap = cpk_colors
    if bonds is None:
        if 'covalent_bonds' in ase_atoms.info.keys():
            bonds = ase_atoms.info['covalent_bonds']
        else:
            from .analyze import update_with_covalent_bonds
            ase_atoms_updated = update_with_covalent_bonds(ase_atoms)
            ase_atoms = ase_atoms_updated
            bonds = ase_atoms.info['covalent_bonds']
            # atoms_graph = mgr.sgraph_from_atoms(ase_atoms_updated)
            # bonds = np.vstack(list(atoms_graph.edges()))

    spheres = spheres_for_atoms(ase_atoms, **kwargs_spheres)
    atomic_numbers = ase_atoms.get_atomic_numbers()
    # for i,s in enumerate(spheres):
    #     # s.visual =trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
    #     # s.visual.material.baseColorFactor= colormap[atomic_numbers[i]]
    #     # s.visual.material.metallicFactor=1.0
    #     # s.visual.material.roughnesFactor=0.15
    #     s.visual.vertex_colors =colormap[atomic_numbers[i]]
    cylinders = cylinders_for_bonds(bonds, ase_atoms, **kwargs_cylinders)

    combined = trimesh.util.concatenate(spheres, cylinders)
    if return_scene:
        for i, s in enumerate(spheres):
            s.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
            s.visual.material.baseColorFactor = colormap[atomic_numbers[i]]
            s.visual.material.metallicFactor = 1.0
            s.visual.material.roughnesFactor = 0.15

        for i, c in enumerate(cylinders):
            c.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
            c.visual.material.baseColorFactor = bond_rgba_array  # cpk_colors[atomic_numbers[i]]
            c.visual.material.metallicFactor = 1.0
            c.visual.material.roughnesFactor = 0.15
        return trimesh.Scene(cylinders + spheres)
    else:
        return combined

def get_isosurface_mesh(data, isoval,ase_atoms, spacing=(1,1,1), stepsize=1 ):
    """
    The function `get_isosurface_mesh` generates a triangular mesh representing an isosurface from input
    data using the marching cubes algorithm.
    
    :param data: The `data` parameter in the `get_isosurface_mesh` function is likely a 3D array
    representing the volumetric data from which the isosurface mesh will be generated. This data could
    be obtained from various sources such as medical imaging, scientific simulations, or any other 3D
    :param isoval: The `isoval` parameter in the `get_isosurface_mesh` function represents the isovalue
    at which the isosurface is extracted from the input data. It is a scalar value that determines the
    surface geometry based on the data values
    :param ase_atoms: The `ase_atoms` parameter in the `get_isosurface_mesh` function is expected to be
    an ASE Atoms object, which represents a collection of atoms and their properties. It typically
    includes information such as atomic positions, unit cell dimensions, and atomic species. This object
    is used to define the
    :param spacing: The `spacing` parameter in the `get_isosurface_mesh` function represents the voxel
    spacing in each dimension for the input `data`. It is a tuple of three values representing the
    spacing along the x, y, and z axes respectively. This parameter determines the resolution of the
    isosurface mesh
    :param stepsize: The `stepsize` parameter in the `get_isosurface_mesh` function you provided
    controls the step size used during the marching cubes algorithm, defaults to 1 (optional)
    :return: The function `get_isosurface_mesh` returns a trimesh object representing the isosurface
    mesh generated from the input data at the specified isovalue.
    """

    import numpy as np
    from skimage.measure import marching_cubes
    from ase.geometry import complete_cell 
    cell = complete_cell(ase_atoms.get_cell()).T
    verts, faces, vertex_normals, _ = marching_cubes(data, isoval,spacing=spacing, step_size=stepsize)
    verts = np.dot(cell, (verts/data.shape).T).T # * transform to cartesian coordinates
    import trimesh 
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=vertex_normals)
    return mesh

def get_isosurface_submeshes(dgrid, isoval, ase_atoms, region_labels,
                            steps=(1, 1, 1), stepsize=1, colorscale=None):
    """Get submeshes of isosurface based on region labels.
    
    Args:
        dgrid: Distance grid
        region_labels: Region labels array
        ase_atoms: ASE atoms object
        isoval: Isovalue for surface
        steps: Tuple of steps in x,y,z (default: (1,1,1))
        stepsize: Step size for interpolation (default: 1)
    
    Returns:
        List of submeshes corresponding to each region
    """
    
    from .helper import get_fractional_coordinates, interpolate_labels
    import numpy as np
    import plotly.express as px
    dist_iso_surf = get_isosurface_mesh(dgrid, isoval, ase_atoms)
    vertices = dist_iso_surf.vertices
    faces = dist_iso_surf.faces
    
    # Find fractional coordinates of vertices
    frac_vertices = get_fractional_coordinates(vertices, ase_atoms)
    
    # Make interpolator for regions with PBC
    rpbc_interp = interpolate_labels(region_labels)
    
    # Assign region labels to vertices
    rl_verts = rpbc_interp(frac_vertices % 1)
    
    # Get faces with same region label for all vertices
    flags_submeshes = np.vstack([
        np.all(rl_verts[faces] == r, axis=1) 
        for r in np.unique(region_labels)
    ])
    
    # Create submeshes for each region (excluding background r=0)
    dist_iso_submeshes = [
        dist_iso_surf.submesh([np.where(f)[0]], append=True) 
        for f in flags_submeshes[1:]
    ]
    # let's add some vertex colors before sendingthis over
    if colorscale is None:
        colorscale = px.colors.qualitative.Pastel
    def parse_rgb(rgb_str):
        # Extract RGB values from 'rgb(r, g, b)' string
        rgb = rgb_str.strip("rgb()").split(",")
        return [int(x) for x in rgb] + [255]
    sampled_colors = px.sample_colorscale(colorscale, np.unique(region_labels)[1:]/region_labels.max())
    mesh_colors = [parse_rgb(s) for s in sampled_colors]
    for i,m in enumerate(dist_iso_submeshes):
        # Check if m is a valid trimesh object with visual attributes
        if hasattr(m, 'visual') and hasattr(m.visual, 'vertex_colors'):
            m.visual.vertex_colors = mesh_colors[i] 
        else:
            print(f"Warning: Submesh {i} is not a valid trimesh object, skipping color assignment")
    return dist_iso_submeshes

def get_isosurface_submeshes(dgrid, isoval, ase_atoms, region_labels,
                            steps=(1, 1, 1), stepsize=1, colorscale=None):
    """Get submeshes of isosurface based on region labels.
    
    Args:
        dgrid: Distance grid
        region_labels: Region labels array
        ase_atoms: ASE atoms object
        isoval: Isovalue for surface
        steps: Tuple of steps in x,y,z (default: (1,1,1))
        stepsize: Step size for interpolation (default: 1)
    
    Returns:
        List of submeshes corresponding to each region
    """
    
    from .helper import get_fractional_coordinates, interpolate_labels
    import numpy as np
    import plotly.express as px
    dist_iso_surf = get_isosurface_mesh(dgrid, isoval, ase_atoms)
    vertices = dist_iso_surf.vertices
    faces = dist_iso_surf.faces
    
    # Find fractional coordinates of vertices
    frac_vertices = get_fractional_coordinates(vertices, ase_atoms)
    
    # Make interpolator for regions with PBC
    rpbc_interp = interpolate_labels(region_labels)
    
    # Assign region labels to vertices
    rl_verts = rpbc_interp(frac_vertices % 1)
    
    # Get faces with same region label for all vertices
    flags_submeshes = np.vstack([
        np.all(rl_verts[faces] == r, axis=1) 
        for r in np.unique(region_labels)
    ])
    
    # Create submeshes for each region (excluding background r=0)
    dist_iso_submeshes = [
        dist_iso_surf.submesh([np.where(f)[0]], append=True) 
        for f in flags_submeshes[1:]
    ]
    # let's add some vertex colors before sendingthis over
    if colorscale is None:
        colorscale = px.colors.qualitative.Pastel
    def parse_rgb(rgb_str):
        # Extract RGB values from 'rgb(r, g, b)' string
        rgb = rgb_str.strip("rgb()").split(",")
        return [int(x) for x in rgb] + [255]
    sampled_colors = px.colors.sample_colorscale(colorscale, np.unique(region_labels)/region_labels.max())
    mesh_colors = [parse_rgb(s) for s in sampled_colors]
    print('mesh colors', mesh_colors)
    for i,m in enumerate(dist_iso_submeshes):
        # Check if m is a valid trimesh object with visual attributes
        if hasattr(m, 'visual') and hasattr(m.visual, 'vertex_colors'):
            m.visual.vertex_colors = mesh_colors[i] 
        else:
            print(f"Warning: Submesh {i} is not a valid trimesh object, skipping color assignment")
    return dist_iso_submeshes
def plot_framework_as_mesh(ase_atoms, show_rings=True, stylize=False, kwargs_frame_mesh=dict(showlegend=True, showscale=False, name='Framework', legendgroup='framework', legendgrouptitle_text='Framework'), kwargs_mesh_for_atoms={}, kwargs_aromatic_rings={}):
    """
    It takes an ASE atoms object and returns a plotly mesh trace of the framework and optionally a trace
    of the aromatic rings.
    
    :param ase_atoms: an ase.Atoms object
    :param show_rings: if True, will plotO the aromatic rings as well, defaults to True (optional)
    :param stylize: if True, will apply some lighting and other effects to the mesh, defaults to False
    (optional)
    :param kwargs_frame_mesh: keyword arguments for mesh_for_atoms
    :param kwargs_highlight: keyword arguments for plot_aromatic_rings
    :return: A tuple of plotly traces.
    """

    from .analyze import update_with_covalent_bonds
    if 'covalent_bonds' not in ase_atoms.info.keys():
        ase_atoms = update_with_covalent_bonds(ase_atoms)
    combined = mesh_for_atoms(ase_atoms, **kwargs_mesh_for_atoms)
    import plotly.graph_objects as go
    verts = combined.vertices
    faces = combined.faces
    colors = ['rgba(' + ','.join(c.astype(str)) + ')' for c in combined.visual.vertex_colors]
    framework_mesh_trace = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], flatshading=False, i=faces[:, 0],
                                     j=faces[:, 1], k=faces[:, 2], vertexcolor=colors, **kwargs_frame_mesh) 
    if stylize:
        framework_mesh_trace.update(lighting=dict(ambient=0.18,
                                                  diffuse=1,
                                                  fresnel=0.1,
                                                  specular=1,
                                                  roughness=0.05,
                                                  facenormalsepsilon=1e-15,
                                                  vertexnormalsepsilon=1e-15),
                                    lightposition=dict(x=10,
                                                       y=10,
                                                       z=30
                                                       ), **kwargs_frame_mesh
                                    )
    framework_mesh_trace = add_hoverdata_to_atoms_traces(framework_mesh_trace, ase_atoms)
    traces = (framework_mesh_trace,)
    if show_rings:
        if not ('covalent_bonds' in ase_atoms.info.keys()):
            ase_atoms = update_with_covalent_bonds(ase_atoms)
        # sgraph = sgraph_from_atoms(ase_atoms=ase_atoms)
        aromatic_rings_trace = plot_aromatic_rings(ase_atoms, **kwargs_aromatic_rings)
        traces = (framework_mesh_trace, *aromatic_rings_trace)
        # group all the traces into the same legend
        for t in traces:
            t['legendgroup'] = traces[0]['legendgroup']
    

    
    # traces = update_framework_mesh_with_hoverdata(traces, ase_atoms,)


    return traces

def add_hoverdata_to_atoms_traces(atoms_trace, ase_atoms, bond_rgba=[127,128,129,255]):
# bond_rgba = [127,128,129,255]

    bonds = ase_atoms.info['covalent_bonds']

    import ase 
    from ase.data import atomic_masses as am  
    import numpy as np 
    atomic_numbers = ase_atoms.get_atomic_numbers() #actual atomic numbers in the material
    chemical_symbols =ase_atoms.get_chemical_symbols()
    atomic_masses =am[atomic_numbers]
    positions = ase_atoms.get_positions()

    vertexcolors = np.array(atoms_trace['vertexcolor'])

    # Find the number of vertices per sphere by dividing the total vertices that belong to spheres
    # by the number of atoms.
    bond_rgba_string = 'rgba('+','.join(np.array(bond_rgba).astype(str))+')'
    vertices_per_sphere = int(len(vertexcolors[vertexcolors!=bond_rgba_string])/ase_atoms.get_number_of_atoms())
    vertex_atom_ids = np.repeat(np.arange(ase_atoms.get_number_of_atoms()), vertices_per_sphere)
    vertex_chemical_symbols = np.repeat(chemical_symbols, vertices_per_sphere)
    vertex_atomic_masses = np.repeat(atomic_masses, vertices_per_sphere)
    vertex_atomic_numbers = np.repeat(atomic_numbers, vertices_per_sphere)
    vertex_positions = np.repeat(np.round(positions, decimals=4), vertices_per_sphere, axis=0)
    vertex_positions = [ ' '.join(v.astype(str)) for v in vertex_positions]

    number_of_bonds = len(bonds)
    vertices_per_bond = int(len(vertexcolors[vertexcolors==bond_rgba_string])/number_of_bonds)
    bond_atom_ids = np.repeat(bonds, vertices_per_bond, axis=0)
    bond_atom_ids = [ 'Bond between atoms '+ ' and '.join(v.astype(str)) for v in bond_atom_ids]


    #make the hovertext list for atoms 
    hovertext_atoms = ['Id:'+ str(i) +'<br>' +'Element:'+s+'<br> Mass:'+str(m)+'<br> Position:'+p for i,s,m,p in zip(vertex_atom_ids, vertex_chemical_symbols, vertex_atomic_masses, vertex_positions)]

    #ake hover text for bonds
    hovertext_bonds = bond_atom_ids

    #combine the two lists
    hovertext = hovertext_atoms + hovertext_bonds


    atoms_trace['hovertext']=hovertext # this isthe frameowk trace
    return atoms_trace

def plot_mesh(mesh, stylize=False, **kwargs_plotly_mesh3d):
    """
    It takes a trimesh object and returns a plotly mesh3d object.
    
    :param mesh: a list of trimesh objects
    :param stylize: If True, the mesh will be rendered with a nice lighting effect, defaults to False
    (optional)
    :return: A plotly mesh object
    """
    
    
    import trimesh
    if len(mesh) > 1:
        mesh = trimesh.util.concatenate(mesh)
    else:
        mesh = mesh[0]
    import plotly.graph_objects as go
    verts = mesh.vertices
    faces = mesh.faces
    colors = ['rgba(' + ','.join(c.astype(str)) + ')' for c in mesh.visual.vertex_colors]

    kwargs_plotly_mesh3d.setdefault('name', 'mesh')
    kwargs_plotly_mesh3d.setdefault('flatshading', False)
    kwargs_plotly_mesh3d.setdefault('vertexcolor', colors)
    kwargs_plotly_mesh3d.setdefault('showlegend', True)
    kwargs_plotly_mesh3d.setdefault('showscale', False)
    if 'color' in kwargs_plotly_mesh3d.keys():
        kwargs_plotly_mesh3d.pop('vertexcolor')
    mesh_trace = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                           **kwargs_plotly_mesh3d)
    if stylize:
        mesh_trace.update(lighting=dict(ambient=0.18,
                                        diffuse=1,
                                        fresnel=0.1,
                                        specular=1,
                                        roughness=0.05,
                                        facenormalsepsilon=1e-15,
                                        vertexnormalsepsilon=1e-15),
                          lightposition=dict(x=10,
                                             y=10,
                                             z=30
                                             ), showscale=False
                          )
    # traces = ( mesh_trace,)

    return mesh_trace


def plot_everything(everything):
    """
    It takes in the output of the do_it_all function and returns a plotly figure
    
    :param everything: a tuple of the following:
    :return: A plotly figure object.
    """
    import numpy as np
    import plotly.graph_objects as go
    from .analyze import update_with_covalent_bonds
    from .helper import find_coord_from_indices
    updated_hkust1, grid, regions, maxima, connections, pgraph = everything
    cage_radii = grid.compute()[tuple(maxima.T)]
    cage_labels = [str(c) + ' \u212b' for c in np.round(cage_radii, decimals=2)]
    # %%
    # * Let's make some window labels for showing on hover as information
    windows = np.vstack([c[3] for c in connections if c[1]])
    window_labels1 = np.array(
        ['-'.join((str(c[0][0]), str(c[0][1]))) for c in connections if c[1] for w in c[3] if c[1]])
    window_labels2 = np.array([c[2] for c in connections if c[1] for w in c[3] if c[1]])
    window_labels3 = np.array([str(c[4]) for c in connections if c[1] for w in c[3] if c[1]])
    window_labels = [w1 + "<br>" + w2 + "<br>" + w3 for w1, w2, w3 in
                     zip(window_labels1, window_labels2, window_labels3)]
    kwargs_frame_mesh = {'kwargs_spheres': dict(phi_resolution=10, theta_resolution=10),
                         'kwargs_cylinders': dict(cylinder_resolution=10)}
    frame_trace = plot_framework_as_mesh(ase_atoms=updated_hkust1, show_rings=True, stylize=True,
                                         kwargs_frame_mesh=kwargs_frame_mesh)
    fig = go.Figure(data=[

        plot_surface_dgrid(grid.compute(), isovalue=2.00, opacity=1, stepsize=1, region_labels=regions, select='all',
                           ase_atoms=updated_hkust1, stylize=True, cscale=None, random_seed=123),
        plot_cell(updated_hkust1),
        scatter3d(find_coord_from_indices(maxima, shape=grid.shape, ase_atoms=updated_hkust1), name='local maxima',
                  marker_color=grid.compute()[tuple(maxima.T)], marker_size=15,
                  hovertext=[str(i + 1) + ' ' + j for i, j in enumerate(cage_labels)]),
        scatter3d(windows, name='windows', marker_color='yellowgreen', marker_size=10, opacity=0.5,
                  hovertext=window_labels),
        *plot_pore_graph(pgraph),
        *frame_trace,
        # *mgr.draw_framework2(mat_updated, bonds=bonds, bond_orders=bond_orders, show_aromatics=False, bond_color='grey', bond_opacity=0.5)
    ], layout=dark_layout)
    # fig.write_html('lta.html')
    return fig

def make_bg_image( type='radial', cmap='plasma_r'):
    """
    It takes a 2D array and a colormap and returns a plotly image object that can be used as a
    background image
    
    :param type: 'radial', 'left_to_right', 'bottom_to_top', defaults to radial (optional)
    :param cmap: The colormap to use, defaults to plasma_r (optional)
    :return: A plotly image object
    """

    import plotly.graph_objects as go 
    import numpy as np
    from PIL import Image

    #* Change range for non square
    import numpy as np
    x= np.linspace(0,1,10)
    y=np.linspace(0,1,10)
    X,Y = np.meshgrid(x,y, indexing='xy')
    F = np.zeros(X.shape)

    if type=='radial':
        #* Radial gradient abount (5,5)
        F = (X-0.5)**2 + (Y-0.5)**2

    # F=np.random.randn(10,10).astype(np.uint8)

    # #* Left to Right gradient
    elif type=='left_to_right':
        F =  X**2

    elif type=='bottom_to_top':    # #* Top to bottom gradient
        F =  Y**2
    
    else:
        Exception('Gradient style not coded make random background')
        F = np.random.random((10,10))
        F = ((F/F.max()) * 255).astype(np.uint8)

    #* Make a background image (we are using matplotlib here for the colormap)
    from matplotlib import cm  
    cmap_ = getattr(cm, cmap)
    bg_image = Image.fromarray(np.uint8(cmap_(F/F.max()) * 255))
    image =go.layout.Image(
                    source= bg_image,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    sizex=1, sizey=1,   
                    xanchor="center",
                    yanchor="middle",
                    sizing="stretch",
                    layer="below")
    return image
# def plot_paths(rag, paths: list, colorscale: str='Pastel', **kwargs) -> list:
#     """
#     Plot the paths in a RAG.

#     Parameters
#     ----------
#     rag : nx.Graph
#         The region adjacency graph.
#     paths : list
#         A list of paths in the RAG.
#     colorscale : str, optional
#         The colorscale to use for the paths, by default 'Pastel'

#     Returns
#     -------
#     list
#         A list of Plotly traces.
#     """
#     import plotly.graph_objects as go 
#     import networkx as nx 
#     import random
#     import plotly.express as px 
#     import numpy as np
#     # coordinates = []
#     # hovertext=[]
#     # marker_color = []
#     path_traces =[]
#     colorscale = getattr(px.colors.qualitative, colorscale)
#     for i,path in enumerate(paths):
#         # coordinates = coordinates+list(nx.get_node_attributes(rag.subgraph(path), 'maxima').values())
#         maxima_dict= nx.get_node_attributes(rag, 'maxima')
#         coordinates= np.vstack(map(maxima_dict.get, path))
#         # coordinates = np.vstack(nx.get_node_attributes(rag.subgraph(path), 'maxima').values())
#         # coordinates.append([None, None, None])
#         # hovertext = hovertext + ['Path: '+str(i)]*len(path)
#         hovertext ='Path: '+str(i)        # hovertext.append(None)
#         # marker_color= marker_color+ [random.choice(colorscale)]*len(path)
#         marker_color= random.choice(colorscale)
#         line_color = 'yellow'
#         # marker_color.append('rgb(0,0,0)')
#         path_trace = go.Scatter3d(x=coordinates[:,0], y=coordinates[:,1], z=coordinates[:,2], mode='markers+lines', marker=dict(size=25, color=marker_color),line=dict(width=10,color=line_color), hovertext=hovertext, name='Path:'+str(i) ,**kwargs)
#         path_traces.append(path_trace)
    
#     return path_traces


def plot_windows(connections, **kwargs):
    """
    It takes a list of connections and returns a plotly scatter trace of the windows.
    
    :param connections: a list of lists, output from the connections from regions function
    :return: A plotly scatter 3D trace of the windows
    """



    import numpy as np 
    import plotly.graph_objects as go



    #* Let's make some window labels for showing on hover as information
    windows = np.vstack([c[3] for c in connections if c[1]])
    window_labels1 = np.array(['-'.join((str(c[0][0]),str(c[0][1]))) for c in connections if c[1] for w in c[3] if c[1]])
    window_labels2 = np.array([c[2] for c in connections if c[1] for w in c[3] if c[1]])
    window_labels3 = np.array([str(c[4]) for c in connections if c[1] for w in c[3] if c[1]])
    window_labels = [w1+"<br>"+w2+"<br>"+w3 for w1,w2,w3 in zip(window_labels1, window_labels2, window_labels3)]



    #set defaults for kwargs
    name = kwargs.get('name', 'windows')
    marker_color = kwargs.get('marker_color', 'yellowgreen')
    marker_size = kwargs.get('marker_size', 10)
    opacity = kwargs.get('opacity', 0.5)
    hovertext = kwargs.get('hovertext', window_labels)
    #remove the entires from kwargs that we have already read in
    kwargs.pop('name', None)
    kwargs.pop('marker_color', None)
    kwargs.pop('marker_size', None)
    kwargs.pop('opacity', None)
    kwargs.pop('hovertext', None)


    #return the scatter trace
    return go.Scatter3d(x=windows[:,0], y=windows[:,1], z=windows[:,2], mode='markers', marker=dict(size=marker_size, color=marker_color, opacity=opacity), hovertext=hovertext, name=name, **kwargs)


# def plot_maxima(dgrid, maxima, ase_atoms, **kwargs):
#     """
#     it takes in a 3d grid of data, a list of indices of local maxima, and an ase atoms object, and
#     returns a plotly scatter3d object with the local maxima plotted as points.
    
#     :param dgrid: the density grid
#     :param maxima: the indices of the local maxima
#     :param ase_atoms: the atoms object from ase
#     :return: a plotly scatter3d object
#     """


#     import numpy as np 
#     import plotly.graph_objects as go

#     from .helper import find_coord_from_indices
#     maxima_coords = find_coord_from_indices(maxima, dgrid.shape,ase_atoms)
#     #set defaults for kwargs
#     cage_radii = dgrid[tuple(maxima.t)]
#     # angstrom = str('\u212b').encode('utf-8', errors='ignore').decode('utf-8')
#     # a = a.encode('ascii', 'ignore')
#     cage_labels = [('{} a').format(c) for c in np.round(cage_radii, decimals =2)]
#     hovertext = [str(i+1)+' '+j for i,j in enumerate(cage_labels)]

#     # read entries from kwargs, with defaults
#     name = kwargs.get('name', 'local maxima')
#     marker_size = kwargs.get('marker_size', 15)
#     opacity = kwargs.get('opacity', 0.85)
#     marker_color = kwargs.get('marker_color', cage_radii)
#     hovertext = kwargs.get('hovertext', hovertext)
#     marker_colorscale = kwargs.get('marker_colorscale', 'viridis')
#     #remove the entries we have already read 
#     kwargs.pop('name', none)
#     kwargs.pop('marker_size', none)
#     kwargs.pop('opacity', none)
#     kwargs.pop('marker_color', none)
#     kwargs.pop('hovertext', none)

#     return go.scatter3d(x=maxima_coords[:,0], y=maxima_coords[:,1], z=maxima_coords[:,2], mode='markers', marker=dict(size=marker_size, color=marker_color, opacity=opacity, colorscale=marker_colorscale), hovertext=hovertext, name=name, **kwargs)

# def plot_maxima(dgrid, maxima, ase_atoms, show_spheres=False, **kwargs):
#     """
#     The function `plot_maxima` generates a 3D scatter plot with markers representing local maxima,
#     optionally showing spheres around the maxima points.
    
#     :param dgrid: The `dgrid` parameter seems to represent a grid of values, possibly related to density
#     or distances
#     :param maxima: The `maxima` parameter in the `plot_maxima` function represents the indices of the
#     local maxima in the density grid. These indices are used to locate the positions of the maxima in 3D
#     space
#     :param ase_atoms: The `ase_atoms` parameter in the `plot_maxima` function seems to represent the
#     atomic structure of a material system. It is likely an input that provides information about the
#     atomic positions and possibly other properties of the atoms in the system. This information is used
#     in conjunction with the `dgrid`
#     :param show_spheres: The `show_spheres` parameter in the `plot_maxima` function is a boolean
#     parameter that determines whether to display spheres around the maxima points in the plot. If
#     `show_spheres` is set to `True`, spheres will be displayed around the maxima points based on the
#     provided coordinates, defaults to False (optional)
#     :return: The function `plot_maxima` returns either a `scatter` plot or a combination of `scatter`
#     plot and `mesh` depending on the value of the `show_spheres` parameter. If `show_spheres` is `True`,
#     it returns both the `scatter` plot and the `mesh` object. If `show_spheres` is `False`, it returns
#     only the `scatter
#     """
    
#     import numpy as np
#     import plotly.graph_objects as go
#     import trimesh
#     from .helper import find_coord_from_indices
#     # from mofography import spheres_for_positions, plot_mesh, find_coord_from_indices
#     maxima_coords = mgr.find_coord_from_indices(maxima, dgrid.shape, ase_atoms)
#     cage_radii = dgrid[tuple(maxima.T)]
#     cage_labels = [("{} A").format(c) for c in np.round(cage_radii, decimals=2)]
#     hovertext = [str(i + 1) + " " + j for i, j in enumerate(cage_labels)]

#     name = kwargs.get("name", "local maxima")
#     marker_size = kwargs.get("marker_size", 15)
#     opacity = kwargs.get("opacity", 0.85)
#     marker_color = kwargs.get("marker_color", cage_radii)
#     hovertext = kwargs.get("hovertext", hovertext)
#     marker_colorscale = kwargs.get("marker_colorscale", "Viridis")
#     sphere_opacity = kwargs.get("sphere_opacity", 0.5)

#     kwargs.pop("name", None)
#     kwargs.pop("marker_size", None)
#     kwargs.pop("opacity", None)
#     kwargs.pop("marker_color", None)
#     kwargs.pop("hovertext", None)
#     kwargs.pop("sphere_opacity",None)

#     scatter = go.Scatter3d(
#         x=maxima_coords[:, 0],
#         y=maxima_coords[:, 1],
#         z=maxima_coords[:, 2],
#         mode="markers",
#         marker=dict(
#             size=marker_size,
#             color=marker_color,
#             opacity=opacity,
#             colorscale=marker_colorscale,
#         ),
#         hovertext=hovertext,
#         name=name,
#         **kwargs,
#     )

#     if show_spheres:
#         spheres =spheres_for_positions(maxima_coords, cage_radii, color=[255,255,255, sphere_opacity])
#         # Concatenate all spheres into one trimesh object
#         spheres = trimesh.util.concatenate(spheres)
#         mesh = plot_mesh([spheres], stylize=True)
#         return scatter, mesh
#     else:
#         return scatter

def plot_maxima(
    dgrid, maxima, ase_atoms, show_spheres=False, sphere_opacity=0.5, **kwargs
):
    """
    The function `plot_maxima` generates a 3D scatter plot with markers representing local maxima, and
    optionally displays spheres around the maxima with interpolated colors. The maxima indices can be outside the box as well. 
    this function should still work fine. it will find the corresponding coordinates inside the box to get the cage radii.

    :param dgrid: The `dgrid` parameter is likely a grid of values representing some property, such as
    electron density or energy, at different points in space. This grid is used to identify local
    maxima, which are points where the value is higher than at neighboring points. The function
    `plot_maxima` takes
    :param maxima: The `plot_maxima` function you provided seems to be for plotting local maxima in a 3D
    plot using Plotly. The `maxima` parameter you mentioned is likely a NumPy array containing the
    coordinates of the local maxima points in the grid
    :param ase_atoms: The `ase_atoms` parameter in the `plot_maxima` function is used to provide atomic
    positions and information about the structure. It is used to find the coordinates of the maxima in
    the grid relative to the atomic positions provided by `ase_atoms`. This information is essential for
    visualizing the local
    :param show_spheres: The `show_spheres` parameter in the `plot_maxima` function is a boolean flag
    that determines whether to display spheres at the maximum coordinates. If set to `True`, spheres
    will be shown at the maximum coordinates based on the provided data. If set to `False`, only markers
    will be, defaults to False (optional)
    :param sphere_opacity: The `sphere_opacity` parameter in the `plot_maxima` function controls the
    opacity of the spheres that are displayed in the plot. It determines how transparent or opaque the
    spheres will appear in the visualization. The value of `sphere_opacity` ranges from 0 (completely
    transparent) to 1
    :return: The function `plot_maxima` returns either a Scatter3d plot or a combination of Scatter3d
    plot and mesh plot, depending on the value of the `show_spheres` parameter. If `show_spheres` is
    `True`, it returns both the Scatter3d plot and the mesh plot of spheres representing the maxima. If
    `show_spheres` is `False`, it returns
    """
    import numpy as np
    import plotly.graph_objects as go
    import trimesh
    from matplotlib.colors import to_rgba
    from .helper import find_coord_from_indices
    maxima_coords = find_coord_from_indices(maxima, dgrid.shape, ase_atoms)
    # np.mod(maxima_indices, dgrid.shape)
    cage_radii = dgrid[tuple((np.mod(maxima, dgrid.shape)).astype(int).T)]

    cage_labels = [("{} A").format(c)
                   for c in np.round(cage_radii, decimals=2)]

    hovertext = [
        str(i + 1) + "  [" + " ".join(map(str, maxima[i])) + "]  " + j
        for i, j in enumerate(cage_labels)
    ]

    name = kwargs.get("name", "local maxima")
    marker_size = kwargs.get("marker_size", 15)
    opacity = kwargs.get("opacity", 0.85)
    marker_color = kwargs.get(
        "marker_color", np.array(range(1, len(maxima) + 1)))
    hovertext = kwargs.get("hovertext", hovertext)
    marker_colorscale = kwargs.get("marker_colorscale", "Viridis")

    kwargs.pop("name", None)
    kwargs.pop("marker_size", None)
    kwargs.pop("opacity", None)
    kwargs.pop("marker_color", None)
    kwargs.pop("hovertext", None)

    scatter = go.Scatter3d(
        x=maxima_coords[:, 0],
        y=maxima_coords[:, 1],
        z=maxima_coords[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=marker_color,
            opacity=opacity,
            colorscale=marker_colorscale,
            line=dict(width=2, color="black"),
        ),
        hovertext=hovertext,
        name=name,
        **kwargs,
    )

    if show_spheres:
        colorscale = scatter.marker.colorscale
        data_values = np.array(scatter.marker.color, dtype=float)

        # Only normalize if using a sequential colorscale
        if isinstance(colorscale[0][1], str) and colorscale[0][1].startswith("rgb"):
            # For qualitative colorscales, just use the colors directly
            def parse_rgb(rgb_str):
                # Extract RGB values from 'rgb(r, g, b)' string
                rgb = rgb_str.strip("rgb()").split(",")
                return [int(x) for x in rgb] + [int(sphere_opacity * 255)]

            # If marker_color is sequential (1,2,3...), use it to index into colorscale
            if np.issubdtype(data_values.dtype, np.number):
                indices = (
                    (data_values - data_values.min())
                    / (data_values.max() - data_values.min())
                    * (len(colorscale) - 1)
                ).astype(int)
                rgba_colors_with_opacity = [
                    parse_rgb(colorscale[i][1]) for i in indices
                ]
            else:
                # If marker_color is not numeric, cycle through the colorscale
                rgba_colors_with_opacity = [
                    parse_rgb(color[1]) for color in colorscale]
                rgba_colors_with_opacity = [
                    rgba_colors_with_opacity[i % len(rgba_colors_with_opacity)]
                    for i in range(len(maxima))
                ]
        else:
            # Original normalization for sequential colorscales
            normalized_values = (data_values - data_values.min()) / (
                data_values.max() - data_values.min()
            )

            def interpolate_color(colorscale, value):
                for i in range(len(colorscale) - 1):
                    if colorscale[i][0] <= value <= colorscale[i + 1][0]:
                        t = (value - colorscale[i][0]) / (
                            colorscale[i + 1][0] - colorscale[i][0]
                        )
                        color1 = np.array(to_rgba(colorscale[i][1]))
                        color2 = np.array(to_rgba(colorscale[i + 1][1]))
                        return (1 - t) * color1 + t * color2
                return np.array(to_rgba(colorscale[-1][1]))

            rgba_colors = [
                interpolate_color(colorscale, val) for val in normalized_values
            ]
            rgba_colors_with_opacity = [
                [
                    int(c[0] * 255),
                    int(c[1] * 255),
                    int(c[2] * 255),
                    int(sphere_opacity * 255),
                ]
                for c in rgba_colors
            ]

        spheres =spheres_for_positions(
            maxima_coords, cage_radii, theta_resolution=30, phi_resolution=30
        )
        for i, s in enumerate(spheres):
            s.visual.vertex_colors = np.array(rgba_colors_with_opacity[i])

        mesh = plot_mesh(
            spheres, stylize=True, opacity=sphere_opacity, name="maxima spheres"
        )
        return scatter, mesh
    else:
        return scatter

# def plot_maxima(dgrid, maxima, ase_atoms, show_spheres=False, sphere_opacity=0.5, **kwargs):
#     """
#     The function `plot_maxima` generates a 3D scatter plot with markers representing local maxima, and
#     optionally displays spheres around the maxima with interpolated colors.
#
#     :param dgrid: The `dgrid` parameter is likely a grid of values representing some property, such as
#     electron density or energy, at different points in space. This grid is used to identify local
#     maxima, which are points where the value is higher than at neighboring points. The function
#     `plot_maxima` takes
#     :param maxima: The `plot_maxima` function you provided seems to be for plotting local maxima in a 3D
#     plot using Plotly. The `maxima` parameter you mentioned is likely a NumPy array containing the
#     coordinates of the local maxima points in the grid
#     :param ase_atoms: The `ase_atoms` parameter in the `plot_maxima` function is used to provide atomic
#     positions and information about the structure. It is used to find the coordinates of the maxima in
#     the grid relative to the atomic positions provided by `ase_atoms`. This information is essential for
#     visualizing the local
#     :param show_spheres: The `show_spheres` parameter in the `plot_maxima` function is a boolean flag
#     that determines whether to display spheres at the maximum coordinates. If set to `True`, spheres
#     will be shown at the maximum coordinates based on the provided data. If set to `False`, only markers
#     will be, defaults to False (optional)
#     :param sphere_opacity: The `sphere_opacity` parameter in the `plot_maxima` function controls the
#     opacity of the spheres that are displayed in the plot. It determines how transparent or opaque the
#     spheres will appear in the visualization. The value of `sphere_opacity` ranges from 0 (completely
#     transparent) to 1
#     :return: The function `plot_maxima` returns either a Scatter3d plot or a combination of Scatter3d
#     plot and mesh plot, depending on the value of the `show_spheres` parameter. If `show_spheres` is
#     `True`, it returns both the Scatter3d plot and the mesh plot of spheres representing the maxima. If
#     `show_spheres` is `False`, it returns
#     """
#
#     import numpy as np
#     import plotly.graph_objects as go
#     import trimesh
#     from matplotlib.colors import to_rgba
#     from .helper import find_coord_from_indices
#     # from mofography import spheres_for_positions, plot_mesh, find_coord_from_indices
#     maxima_coords = find_coord_from_indices(maxima, dgrid.shape, ase_atoms)
#     cage_radii = dgrid[tuple(maxima.T)]
#     cage_labels = [("{} A").format(c) for c in np.round(cage_radii, decimals=2)]
#     hovertext = [str(i + 1) + " " + j for i, j in enumerate(cage_labels)]
#
#     name = kwargs.get("name", "local maxima")
#     marker_size = kwargs.get("marker_size", 15)
#     opacity = kwargs.get("opacity", 0.85)
#     marker_color = kwargs.get("marker_color", cage_radii)
#     hovertext = kwargs.get("hovertext", hovertext)
#     marker_colorscale = kwargs.get("marker_colorscale", "Viridis")
#
#     kwargs.pop("name", None)
#     kwargs.pop("marker_size", None)
#     kwargs.pop("opacity", None)
#     kwargs.pop("marker_color", None)
#     kwargs.pop("hovertext", None)
#
#     scatter = go.Scatter3d(
#         x=maxima_coords[:, 0],
#         y=maxima_coords[:, 1],
#         z=maxima_coords[:, 2],
#         mode="markers",
#         marker=dict(
#             size=marker_size,
#             color=marker_color,
#             opacity=opacity,
#             colorscale=marker_colorscale,
#         ),
#         hovertext=hovertext,
#         name=name,
#         **kwargs,
#     )
#     data_values = np.array(scatter.marker.color, dtype=float)
#
#     # Normalize data values to range [0, 1]
#     normalized_values = (data_values - data_values.min()) / (data_values.max() - data_values.min())
#
#     # Define the colorscale
#     colorscale =scatter.marker.colorscale
#
#
#     # Function to interpolate colors
#     def interpolate_color(colorscale, value):
#         for i in range(len(colorscale) - 1):
#             if colorscale[i][0] <= value <= colorscale[i + 1][0]:
#                 t = (value - colorscale[i][0]) / (colorscale[i + 1][0] - colorscale[i][0])
#                 color1 = np.array(to_rgba(colorscale[i][1]))
#                 color2 = np.array(to_rgba(colorscale[i + 1][1]))
#                 return (1 - t) * color1 + t * color2
#         return np.array(to_rgba(colorscale[-1][1]))
#
#     # Interpolate colors for each normalized value
#     rgba_colors = [interpolate_color(colorscale, val) for val in normalized_values]
#
#     # Convert colors to RGBA format with specified opacity
#     rgba_colors_with_opacity = [[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), int(sphere_opacity*255)] for c in rgba_colors]
#     # rgba_colors_with_opacity = [[int(c[0] ), int(c[1] ), int(c[2] ), opacity] for c in rgba_colors]
#
#     # print(rgba_colors_with_opacity)
#
#     if show_spheres:
#         spheres =spheres_for_positions(maxima_coords, cage_radii, theta_resolution=30, phi_resolution=30)
#         # print([s.visual.vertex_colors for s in spheres])
#         for i, s in enumerate(spheres):
#             # print(rgba_colors_with_opacity[i])
#             s.visual.vertex_colors = np.array(rgba_colors_with_opacity[i]) 
#             # s.visual.vertex_colors =  trimesh.visual.random_color()
#         # print([s.visual.vertex_colors for s in spheres])
#         # Concatenate all spheres into one trimesh object
#         # spheres = trimesh.util.concatenate(spheres)
#         mesh = plot_mesh(spheres, stylize=True, opacity=sphere_opacity, name='maxima spheres')
#         return scatter, mesh
#     else:
#         return scatter

def plot_psds(rag, pbc_groups, cscale=None, no_pbc=False, cluster_key='cluster'):
    """
    > This function takes in a RAG and a list of PBC groups and plots the PSD for each PBC group. 
    
    Let's see how it works:
    
    :param rag: the region adjacency graph
    :param pbc_groups: a list of sets of nodes that are connected by periodic boundary conditions
    :return: A plotly figure object
    """

    import plotly.graph_objects as go
    import plotly.express as px 
    import networkx as nx 
    import numpy as np
    if cscale is None:
        cscale = px.colors.qualitative.Alphabet
    unique_clusters = np.unique(list(nx.get_node_attributes(rag,cluster_key).values())).astype(float).astype(int)

    from plotly.express.colors import sample_colorscale
    
    from scipy.interpolate import interp1d
    mm = interp1d([0,np.max(unique_clusters)], [0,1])
    cmap = dict(zip(unique_clusters, sample_colorscale(cscale, mm(unique_clusters))))
    
    #  cmap = dict(zip(unique_clusters, cscale[:len(unique_clusters)]))
    traces = []
    if no_pbc:
        for node in list(rag.nodes().keys()): 
            # node = np.min(list(g))
            traces.append(go.Scatter(x=rag.nodes[node]['psd_x'], y=rag.nodes[node]['psd_y'], mode='markers+lines', line_color = cmap.get(int(float(rag.nodes[node][cluster_key]))), hovertext = 'Cluster: '+ rag.nodes[node][cluster_key]+'</br>'+'Region: {}'.format(node) , name='Region: ' +str(rag.nodes[node]['labels'][0])))
    else:
        for i,g in enumerate(pbc_groups): 
            node = np.min(list(g))
            traces.append(go.Scatter(x=rag.nodes[node]['psd_x_pbc'], y=rag.nodes[node]['psd_y_pbc'], mode='markers+lines', line_color = cmap.get(int(float(rag.nodes[node][cluster_key]))), hovertext = 'Cluster: '+ rag.nodes[node][cluster_key]+'</br>'+'Regions: {}'.format(','.join(map(str, list(g)))) , name= 'PBC group: {}'.format(i) ))
            # traces.append(go.Bar(x=rag.nodes[node]['psd_x_pbc'], y=rag.nodes[node]['psd_y_pbc'], marker_color=cmap.get(rag.nodes[node]['cluster']), marker_opacity=0.5, name='PBC group: {}'.format(i+1) ,hovertext = 'Cluster: '+ rag.nodes[node]['cluster']+'</br>'+'Regions: {}'.format(','.join(map(str, list(g))))) )
    
    return traces
def plot_vdw_hists(rag, pbc_groups, cscale=None, no_pbc=False, cluster_key='cluster'):


    import plotly.graph_objects as go
    import plotly.express as px 
    import networkx as nx 
    import numpy as np
    if cscale is None:
        cscale = px.colors.qualitative.Alphabet
    unique_clusters = np.unique(list(nx.get_node_attributes(rag,cluster_key).values()))

    cmap = dict(zip(unique_clusters, cscale[:len(unique_clusters)]))
    traces = []
    if no_pbc:
        for node in list(rag.nodes().keys()): 
            # node = np.min(list(g))
            traces.append(go.Scatter(x=rag.nodes[node]['vdw_hist_x'], y=rag.nodes[node]['vdw_hist_y'], mode='markers+lines', line_color = cmap.get(rag.nodes[node][cluster_key]), hovertext = 'Cluster: '+ rag.nodes[node][cluster_key]+'</br>'+'Region: {}'.format(node) , name='Region: '+ str(rag.nodes[node]['labels'][0])))
    else:
        # * pick one group from each cluster type
        # types_list = [rag.nodes[list(g)[0]]['cluster'] for g in pbc_groups]
        # ctypes, cindices = np.unique(types_list,return_inverse=True)
        # # for c in ctypes:
        # types_indices = [np.where([t==c for t in types_list])[0] for c in ctypes]
        # # types_counts = [len(t) for t in types_indices]
        # type_selection = [np.random.choice(ti ,1, replace=False)  for i,ti in enumerate(types_indices)]
        # # print(types_list, ctypes, cindices, types_indices, type_selection)

         # for i,ts in enumerate(type_selection):
        #     g = pbc_groups[ts[0]] 
        #     node = np.min(list(g))           
        #     traces.append(go.Bar(x=rag.nodes[node]['vdw_hist_x_pbc'], y=rag.nodes[node]['vdw_hist_y_pbc'], marker_color=cmap.get(rag.nodes[node]['cluster']), marker_opacity=1.0, name='Cluster (cavity type): '+ctypes[i] ,hovertext = 'Cluster: '+ rag.nodes[node]['cluster']+'</br>'+'Regions: {}'.format(','.join(map(str, list(g))))) )
        for i,g in enumerate(pbc_groups):
            node = np.min(list(g)) # pick any node from the group, they all have the same cluster index. here we are picking the first one
            traces.append(go.Bar(x=rag.nodes[node]['vdw_hist_x_pbc'], y=rag.nodes[node]['vdw_hist_y_pbc'], marker_color=cmap.get(rag.nodes[node][cluster_key]), marker_opacity=1.0, name='PBC group: {}'.format(i) ,hovertext = 'Cluster: '+ rag.nodes[node][cluster_key]+'</br>'+'Regions: {}'.format(','.join(map(str, list(g))))) )
    return traces

def plot_subset_of_atoms(ase_atoms, subset, **kwargs):

    """
    > This function takes an ASE atoms object and a list of atom ids and returns a plotly mesh trace of
    the atoms in the list
    
    :param ase_atoms: an ase atoms object
    :param subset: a list of atom ids that you want to plot
    :return: A plotly mesh3D trace that can be added to a plotly figure
    """

    from .analyze import sgraph_from_atoms, update_with_covalent_bonds, get_subset_atoms


    # construct structure graph and subgraph
    import networkx as nx
    import numpy as np
    mat_graph = sgraph_from_atoms(ase_atoms)
    sg = mat_graph.subgraph(subset)
    sg = nx.convert_node_labels_to_integers(sg, first_label=0)
    subatoms =get_subset_atoms(ase_atoms, subset=subset) 
    subatoms.info['covalent_bonds'] = np.vstack(list(sg.edges()))
    subatoms.info['covalent_bond_orders'] = np.ones((len(subatoms.info['covalent_bonds']),), dtype=float)  # assuming all bonds are single bonds
    subatoms.info['atom ids'] = list(nx.get_node_attributes(sg, 'element').keys())


    # plotting arguments
    stylize = kwargs.get('stylize', True)
    show_rings = kwargs.get('show_rings', True)
    # trace_name = kwargs.get('name', 'subset')
    name = kwargs.get('name', 'subset')
    kwargs_frame_mesh = kwargs.get('kwargs_frame_mesh', dict(showlegend=True, showscale=False, name=name, legendgroup='subset', legendgrouptitle_text='Subset'))
    kwargs.pop('stylize', None)
    kwargs.pop('show_rings', None)
    # kwargs.pop('name', None)
    spheres= spheres_for_atoms(subatoms)
    cylinders= cylinders_for_bonds(subatoms.info['covalent_bonds'], subatoms, show_rings=show_rings, stylize=stylize, **kwargs)
    atoms_trace = plot_mesh(spheres+cylinders, stylize=stylize)
    # atoms_trace = plot_framework_as_mesh(subatoms, show_rings=show_rings, stylize =stylize, **kwargs)    
    return atoms_trace

def draw_framework(ase_atoms, bond_orders=None, show_aromatics=True, show_atom_labels=False,
                    bond_data=None, bond_color=None, bond_colorscale='Cividis', bond_opacity=0.9, colorscheme=get_default_colors(),
                    atoms_opacity=0.9, dia_scale=1):
    """
    It takes an ASE atoms object and returns a list of plotly traces that can be used to plot the
    framework
    
    :param ase_atoms: ASE atoms object
    :param bond_orders: A list of bond orders for each bond. If None, then the bond orders are
    calculated using covalent radii
    :param show_aromatics: If True, it will detect aromatic rings and plot them, defaults to True
    (optional)
    :param show_atom_labels: If True, shows the atom labels, defaults to False (optional)
    :param bond_data: A dictionary of bond data. The keys are the names of the data and the values are
    the data itself
    :param bond_color: This is the color of the bonds. If you provide a string, it will be used as a key
    to look for a list of bond colors in the ase_atoms.info dict. If you provide a list, it will be used
    as the bond colors. If you provide a single
    :param bond_colorscale: The colorscale for the bonds, defaults to Cividis (optional)
    :param bond_opacity: opacity of the bonds
    :param colorscheme: A list of colors to be used for coloring the atoms
    :param atoms_opacity: opacity of the atoms
    :param dia_scale: This is a scaling factor for the atom sizes, defaults to 1 (optional)
    :return: A list of plotly.graph_objects.Scatter3d objects.
    """


    # from ase.io import read
    import mofography as mgr
    from ase import neighborlist
    from  .analyze import update_with_covalent_bonds
    # from ase.data import covalent_radii
    # data = read(path_to_cif)
    # A_unit = data.get_cell().T
    coord = ase_atoms.get_positions()
    names = ase_atoms.get_chemical_symbols()
    import numpy as np
    # from pymatgen.core import Molecule
    # from mofography import sgraph_from_atoms
    from .visualize import plot_aromatic_rings
    # print("1. Creating Molecule ...")
    # mol = Molecule(names, coord)

    # print(bonds_atoms_list)
    print("Creating Plotly figure traces")
    # * Figure out how to plot the framework atoms

    import plotly.graph_objects as go

    # * Create a color scale using ASE cpk colorlist.
    from ase.data import colors
    from sklearn.preprocessing import minmax_scale
    cols = np.round(colorscheme * 255, decimals=0).astype(int)
    colorlist = ["rgb(" + ",".join(c.astype(str)) + ")" for c in cols]
    # atomic_numbers        = np.vstack(data.get_atomic_numbers())
    # atom_colors           =   minmax_scale(np.linspace(0,103, 104), feature_range=(0,1))
    atom_colors = np.linspace(0, 1, len(cols))
    cscale = [[atom_colors[i], colorlist[i]] for i in range(len(colorlist))]

    # * Hovertext for each point
    hovertext = ['Atom No: ' + str(i) + ', Atom type: ' + names[i] for i in range(len(names))]

    # * Scale the atom sizes beautifully.
    # atom_dia              = f1.sigma_array*2-f1.sigma
    atom_dia = 2 * neighborlist.natural_cutoffs(ase_atoms)


    scatter_size = np.round(minmax_scale(atom_dia, feature_range=(12, 30)) * dia_scale)

    traces = []
    print("1. Creating atoms trace\n")

    trace_scatter = go.Scatter3d(x=coord[:, 0], y=coord[:, 1], z=coord[:, 2], mode='markers', hovertext=hovertext, marker=dict(opacity=atoms_opacity, size=scatter_size, color=atom_colors[ase_atoms.get_atomic_numbers() - 1], sizemode='diameter', colorscale=cscale), name='Atoms', text=names, textfont=dict(color=colorlist, size=20), surfaceaxis=-1, legendgroup='framework', legendgrouptitle_text = 'Framework',)
    traces.append(trace_scatter)

    if 'covalent_bonds' not in ase_atoms.info.keys():
    # if bonds is None:
        print("No bonds found in the ASE atoms object. Creating bonds using covalent radii ...")
        ase_atoms_updated = update_with_covalent_bonds(ase_atoms)
        bonds = ase_atoms_updated.info['covalent_bonds']
        if bond_orders is None:
            bond_orders = ase_atoms_updated.info['covalent_bond_orders']
        else:
            raise Exception("You can't provide bond orders without including 'covalent_bonds' in ase_atom.info dict!")
    else:
        print("Bonds found in the ASE atoms object. Using them ...")
        ase_atoms_updated = ase_atoms
        bonds = ase_atoms_updated.info['covalent_bonds']

    print("2. Creating bonds' trace\n")
    edges_master = []

    for b in bonds:
        edges_master.append(coord[b[0]])
        edges_master.append(coord[b[1]])
        edges_master.append([None, None, None])

    # print(bonds)
    if bond_color is None:
        bond_color = 'rgba(0,0,0,0.35)'
    elif type(bond_color) == str:
        print("Using bond color: ", bond_color)
        if bond_color in ase_atoms.info.keys():
            # key_name = bond_color
            # bond_color = []
            # for i,b in enumerate(bonds):
                # bond_color.append(ase_atoms_updated.info[key_name][i])
                # bond_color.append(ase_atoms_updated.info[key_name][i])
                # bond_color.append(None)
            bond_color = np.repeat(ase_atoms_updated.info[bond_color], 3)
        else:
            pass
    edges_master = np.vstack(edges_master)
    # print(bond_color)
    # try:
    trace_edges = go.Scatter3d(x=edges_master[:, 0], y=edges_master[:, 1], z=edges_master[:, 2], line=dict(width=15, color=bond_color, colorscale=bond_colorscale, showscale=False, colorbar=dict(thickness=20, len=0.6, title='bond order', )), opacity=bond_opacity, mode='lines', name='Bonds', legendgroup='framework', )
    # except:
    #     trace_edges = go.Scatter3d(x=edges_master[:, 0], y=edges_master[:, 1], z=edges_master[:, 2],
    #                                line=dict(width=15, color='rgba(150,150,150,0.35)', colorscale='aggrnyl'),
    #                                opacity=bond_opacity, mode='lines',
                                #    name='framework bonds')
    traces.append(trace_edges)

    if show_aromatics:
        print('\t2-a. Detecting and plotting aromatic rings\n')

        # G = sgraph_from_atoms(ase_atoms_updated)

        ring_traces = plot_aromatic_rings(ase_atoms=ase_atoms_updated)
        
        # Assign the same legendgroup to aromatic rings as atoms and bonds
        for ring_trace in ring_traces:
            ring_trace['legendgroup'] = 'framework'
            ring_trace['legendgrouptitle_text'] = 'Framework'

        traces = traces + list(ring_traces)
    if bool(bond_data):
        print('4. Creating traces for additional bond data provided')
        # for key in bond_data.keys():
        true_edges = edges_master[edges_master != None].reshape(-2, 3)
        true_edges = 0.5 * (true_edges[0::2] + true_edges[1::2])
        number_of_bonds=  len(true_edges)
        bond_hoverdata = ["<br>".join(['{}:{}'.format(k,bond_data[k][i]) for k in bond_data.keys()]) for i in range(number_of_bonds)]
        bond_data_opacity = 1E-5*np.ones((number_of_bonds,))
        trace_bond_data = go.Scatter3d(x=true_edges[:, 0], y=true_edges[:, 1], z=true_edges[:, 2], mode='markers',
                                        marker_opacity=0.0004, hovertext=bond_hoverdata,
                                        name='Bond data on hover',showlegend=False)
        traces.append(trace_bond_data)
    if show_atom_labels:
        trace_atom_labels = go.Scatter3d(x=coord[:, 0], y=coord[:, 1], z=coord[:, 2], mode='text',
                                         textfont=dict(color=colorlist, size=20), text=names,
                                         name='atom labels')
        traces.append(trace_atom_labels)
    return traces


def plot_molecule_at_best_optuna_trials(rag, molecule_atoms, molecule_name, **kwargs_plot_mesh):
    '''This function plots a molecule using the best Optuna trials.
    
    Parameters
    ----------
    rag
        a RAG (Region Adjacency Graph) object
    molecule_atoms
        an ASE Atoms object representing the molecule to be plotted
    molecule_name
        The name of the molecule being plotted.
    
    Returns
    -------
        a list of traces, where each trace is a plot of a molecule at its best Optuna trial. The plot is
    generated using the provided molecule's atomic positions and the van der Waals radii of its atoms.
    The function uses the `spheres_for_positions` and `plot_mesh` functions from the `visualize` module.
    
    '''

    import numpy as np
    from ase.data import vdw_radii
    # from .visualize import spheres_for_positions, plot_mesh
    mat_atoms = rag.graph['ase_atoms']
    molecule_atoms.set_cell(mat_atoms.get_cell())
    def plot_molecule(cart_coords, **kwargs_plot_mesh):
        molecule_atoms.set_positions(cart_coords)
        atoms_mesh = mesh_for_atoms(molecule_atoms, kwargs_spheres=dict(radii=vdw_radii[molecule_atoms.get_atomic_numbers()]))
        mesh_trace = plot_mesh([atoms_mesh], **kwargs_plot_mesh)
        return mesh_trace
    traces = [plot_molecule(rag.nodes[i]['best_trial_{}'.format(molecule_name)].user_attrs['cart_coords'],
                            **kwargs_plot_mesh) for i in rag.nodes if 'best_trial_{}'.format(molecule_name) in rag.nodes[i].keys()  ]
    return traces

def make_pore_size_distribution_fig_from_rag(rag):
    """
    The function `make_pore_size_distribution_fig_from_rag` generates a plotly figure displaying pore
    size distribution and cumulative distribution from a given RAG.
    
    :param rag: The function `make_pore_size_distribution_fig_from_rag` takes a RAG (Region Adjacency
    Graph) object as input and creates a plotly figure showing the Pore Size Distribution (PDF) and
    Cumulative Distribution Function (CDF) based on the data stored in the RAG object
    :return: The function `make_pore_size_distribution_fig_from_rag` returns a Plotly figure object that
    contains a plot of the Pore Size Distribution (PDF) and Cumulative Distribution Function (CDF) based
    on the input Reservoir Analog Graph (RAG) data.
    """

    for key in ['pore_size_dist', 'pore_size_cdf', 'pore_size_bins']:
        if key not in rag.graph:
            raise ValueError('The pore size distribution has not been computed for this RAG. Please run the get_psd_on_rag function first.')
    import plotly.graph_objects as go
    # extract the PDF, CDF, and bin edges from the rag
    pdf = rag.graph['pore_size_dist']
    cdf = rag.graph['pore_size_cdf']
    bin_edges = rag.graph['pore_size_bins']

    # create a figure
    fig = go.Figure()

    # add a trace for the PDF
    fig.add_trace(go.Scatter(
        x=bin_edges[:-1],  # use the left edge of each bin as the x-coordinate
        y=pdf,
        mode='lines',
        name='Pore Size Distribution',
        line=dict(color='blue')
    ))

    # add a trace for the CDF
    fig.add_trace(go.Scatter(
        x=bin_edges[:-1],
        y=cdf,
        mode='lines',
        name='Cumulative Distribution',
        line=dict(dash='dot', color='red'),
        yaxis='y2'
    ))

    # set the x-axis title
    fig.update_xaxes(title_text='Pore Dia. (Å)')

    # set the y-axis title
    fig.update_layout(
        yaxis=dict(
            title='Pore Size Distribution',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Cumulative Distribution',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        )
    )

    return fig
# %%
def plot_pld_locations(dgrid, pld_locations, ase_atoms, show_spheres=False, sphere_opacity=0.5, **kwargs):
    """
    Generate a 3D scatter plot with markers representing PLD locations, and
    optionally display spheres around the PLD locations.
    
    Parameters:
    -----------
    dgrid : numpy.ndarray
        The distance grid of the structure.
    pld_locations : numpy.ndarray
        Array of grid indices where PLD occurs.
    ase_atoms : ase.Atoms
        The ASE Atoms object representing the structure.
    show_spheres : bool, optional
        Whether to display spheres at the PLD locations. Default is False.
    sphere_opacity : float, optional
        Opacity of the spheres. Default is 0.5.
    **kwargs : dict
        Additional keyword arguments to customize the plot.

    Returns:
    --------
    plotly.graph_objs.Scatter3d or tuple
        Scatter3d plot, or tuple of (Scatter3d, Mesh3d) if show_spheres is True.
    """

    import numpy as np
    import plotly.graph_objects as go
    import trimesh
    from matplotlib.colors import to_rgba
    from .helper import find_coord_from_indices

    pld_coords = find_coord_from_indices(pld_locations, dgrid.shape, ase_atoms)
    pld_radii = dgrid[tuple(pld_locations.T)]
    pld_labels = [("{:.2f} A").format(r) for r in pld_radii]  # Diameter = 2 * radius
    hovertext = [f"PLD {i+1}: {label}" for i, label in enumerate(pld_labels)]

    name = kwargs.get("name", "PLD locations")
    marker_size = kwargs.get("marker_size", 15)
    opacity = kwargs.get("opacity", 0.85)
    marker_color = kwargs.get("marker_color", "red")
    marker_symbol = kwargs.get("marker_symbol", "diamond")

    scatter = go.Scatter3d(
        x=pld_coords[:, 0],
        y=pld_coords[:, 1],
        z=pld_coords[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=marker_color,
            opacity=opacity,
            symbol=marker_symbol,
        ),
        hovertext=hovertext,
        name=name,
        **kwargs,
    )

    if show_spheres:
        # from .helper import spheres_for_positions, plot_mesh
        spheres = spheres_for_positions(pld_coords, pld_radii, theta_resolution=30, phi_resolution=30)
        for s in spheres:
            s.visual.vertex_colors = np.array([255, 0, 0, int(sphere_opacity*255)])  # Red color
        mesh = plot_mesh(spheres, stylize=True, opacity=sphere_opacity, name='PLD spheres')
        return scatter, mesh
    else:
        return scatter
def plot_percolated_cluster_meshes(percolation_history, distance_grid, regions, ase_atoms, probe_radius=None, isovalue=1.6, **kwargs):
    """
    Generate mesh traces for percolated clusters at a given probe radius.

    Parameters:
    -----------
    percolation_history : list
        The percolation history as returned by compute_percolation_history.
    distance_grid : numpy.ndarray
        The distance grid of the structure.
    regions : numpy.ndarray
        The segmented regions grid.
    ase_atoms : ase.Atoms
        The ASE Atoms object representing the structure.
    probe_radius : float, optional
        The probe radius to visualize. If None, use the last entry in percolation_history.
    isovalue : float, optional
        The isovalue for the surface plot. Default is 0.5.
    **kwargs : dict
        Additional keyword arguments to pass to plot_surface_dgrid.

    Returns:
    --------
    list
        A list of plotly.graph_objects.Mesh3d objects representing the percolated clusters.
    """
    if probe_radius is None:
        entry = percolation_history[-1]
    else:
        entry = next((e for e in percolation_history if e[0] >= probe_radius), None)
        if entry is None:
            raise ValueError(f"No percolation data found for probe radius {probe_radius}")

    threshold, components = entry

    # Collect unique percolating components and their directions
    unique_components = {}
    for axis, comp_list in components.items():
        for comp in comp_list:
            comp_key = tuple(sorted(comp))
            if comp_key in unique_components:
                unique_components[comp_key].append(axis)
            else:
                unique_components[comp_key] = [axis]

    mesh_traces = []

    # Generate mesh for each unique component
    for i, (comp, axes) in enumerate(unique_components.items(), 1):
        # Get direction labels
        dir_labels = ['x' if 0 in axes else '', 'y' if 1 in axes else '', 'z' if 2 in axes else '']
        dir_str = ''.join(dir_labels)
        
        # Generate the mesh
        mesh = plot_surface_dgrid(
            distance_grid, 
            isovalue, 
            ase_atoms, 
            region_labels=regions,
            select=comp,
            name=f'Component {i}_{dir_str}',
            **kwargs
        )
        mesh.legendgroup = 'percolated'
        mesh.legendgrouptitle = dict(text='Percolated Clusters')
        mesh_traces.append(mesh)

    return mesh_traces
def plot_electrostatic_volume(potential_V, cell, atoms):
    """
    Create a Volume trace for the electrostatic potential with meaningful isosurfaces.
    
    :param potential_V: 3D array of potential values in volts
    :param cell: Cell matrix
    :param atoms: ASE Atoms object
    :return: Volume trace
    """
    import plotly.graph_objects as go
    import numpy as np
    from collections import defaultdict
    
    nx, ny, nz = potential_V.shape
    
    # Create a meshgrid in fractional coordinates
    x, y, z = np.meshgrid(np.linspace(0, 1, nx),
                          np.linspace(0, 1, ny),
                          np.linspace(0, 1, nz),
                          indexing='ij')
    
    # Transform fractional coordinates to Cartesian
    cart_coords = np.dot(np.column_stack((x.flatten(), y.flatten(), z.flatten())), cell)
    x_cart, y_cart, z_cart = cart_coords[:, 0].reshape(x.shape), cart_coords[:, 1].reshape(y.shape), cart_coords[:, 2].reshape(z.shape)

    # Get unique elements in the system
    elements = set(atoms.get_chemical_symbols())

    # Define groups of elements
    metal_group = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'}
    electronegative_group = {'O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'P', 'As', 'Sb'}

    # Find present metals and electronegative elements
    present_metals = metal_group.intersection(elements)
    present_electronegative = electronegative_group.intersection(elements)

    # Collect potentials for different element groups
    potentials_dict = defaultdict(list)

    for element in elements:
        if element in present_metals or element in present_electronegative or element == 'C':
            element_potentials = get_potential_near_atoms(atoms, potential_V, potential_V.shape, element=element, radius=1.0)
            potentials_dict[element].extend(element_potentials)

    # Calculate isosurface values
    if present_metals:
        positive_iso = max(np.percentile(potentials, 75) for element, potentials in potentials_dict.items() if element in present_metals and len(potentials) > 0)
    else:
        positive_iso = np.percentile(potential_V, 95)

    if present_electronegative:
        negative_iso = min(np.percentile(potentials, 25) for element, potentials in potentials_dict.items() if element in present_electronegative and len(potentials) > 0)
    else:
        negative_iso = np.percentile(potential_V, 5)

    # Add more isosurfaces
    num_isosurfaces = 9
    isosurfaces = np.linspace(negative_iso, positive_iso, num_isosurfaces)

    # Create the volume trace
    volume_trace = go.Volume(
        x=x_cart.flatten(),
        y=y_cart.flatten(),
        z=z_cart.flatten(),
        value=potential_V.flatten(),
        isomin=negative_iso,
        isomax=positive_iso,
        opacity=0.1,
        surface_count=num_isosurfaces,
        colorscale='RdBu',
        reversescale=True,
        colorbar=dict(title="Potential (V)"),
        hoverinfo='all',
        hoverlabel=dict(bgcolor='white'),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Electrostatic potential Poisson',
        showlegend=True,
    )

    print(f"Isosurface values: {isosurfaces}")
    print(f"Positive isosurface threshold: {positive_iso:.2f}")
    print(f"Negative isosurface threshold: {negative_iso:.2f}")

    return volume_trace
def get_potential_near_atoms(atoms, potential, grid_shape, element='Zn', radius=1.0):
    """
    Get potential values near atoms of a specific element.
    
    :param atoms: ASE Atoms object
    :param potential: 3D array of potential values
    :param grid_shape: Shape of the potential grid
    :param element: Element symbol to look for
    :param radius: Radius around the atom to check (in Angstroms)
    :return: List of potential values near the specified atoms
    """
    import numpy as np

    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    
    potentials_near_atoms = []
    
    print(f"Searching for {element} atoms...")
    
    for pos, sym in zip(positions, symbols):
        if sym == element:
            print(f"Found {element} atom at position {pos}")
            # Convert position to grid indices
            grid_pos = pos * grid_shape
            
            # Define a small box around the atom
            box_size = int(radius / (cell.cellpar()[:3] / grid_shape).max())
            x, y, z = np.round(grid_pos).astype(int)
            
            # Get potential values in the box
            x_indices = (max(0, x - box_size), min(grid_shape[0], x + box_size + 1))
            y_indices = (max(0, y - box_size), min(grid_shape[1], y + box_size + 1))
            z_indices = (max(0, z - box_size), min(grid_shape[2], z + box_size + 1))
            
            potentials = potential[
                x_indices[0]:x_indices[1],
                y_indices[0]:y_indices[1],
                z_indices[0]:z_indices[1]
            ]
            
            potentials_near_atoms.extend(potentials.flatten())
    
    print(f"Found {len(potentials_near_atoms)} potential values near {element} atoms")
    
    return potentials_near_atoms
def plot_electrostatic_isosurfaces(potential_V, atoms, isovals=None, num_isosurfaces=10, cscale='viridis', step=1, base_opacity=0.3):
    """
    Create a single Plotly mesh trace for the electrostatic potential isosurfaces with correct scaling and hover info.
    
    :param potential_V: 3D array of potential values in volts
    :param cell: Cell matrix
    :param atoms: ASE Atoms object
    :param step: Step size for marching cubes algorithm. Higher values result in coarser surfaces.
    :param base_opacity: Base opacity for the surfaces. Adjust this to make surfaces more or less transparent.
    :return: A single combined Plotly Mesh3d trace
    """
    import numpy as np
    from collections import defaultdict
    from skimage import measure
    from .helper import find_coord_from_indices
    from ase.geometry import complete_cell
    import plotly.graph_objects as go
    # define the cell needed for plotting 
    cell = complete_cell(atoms.cell).T
    # Get unique elements in the system
    elements = set(atoms.get_chemical_symbols())

    # Define groups of elements
    metal_group = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'}
    electronegative_group = {'O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'P', 'As', 'Sb'}

    # Find present metals and electronegative elements
    present_metals = metal_group.intersection(elements)
    present_electronegative = electronegative_group.intersection(elements)

    # Collect potentials for different element groups
    potentials_dict = defaultdict(list)

    for element in elements:
        if element in present_metals or element in present_electronegative or element == 'C':
            element_potentials = get_potential_near_atoms(atoms, potential_V, potential_V.shape, element=element, radius=1.0)
            potentials_dict[element].extend(element_potentials)

    # Calculate isosurface values
    if present_metals and any(potentials for element, potentials in potentials_dict.items() if element in present_metals):
        positive_iso = max(np.percentile(potentials, 75) for element, potentials in potentials_dict.items() if element in present_metals and len(potentials) > 0)
    else:
        positive_iso = np.percentile(potential_V, 95)

    if present_electronegative and any(potentials for element, potentials in potentials_dict.items() if element in present_electronegative):
        negative_iso = min(np.percentile(potentials, 25) for element, potentials in potentials_dict.items() if element in present_electronegative and len(potentials) > 0)
    else:
        negative_iso = np.percentile(potential_V, 5)

    # Add more isosurfaces
    # num_isosurfaces = 10
    if isovals is None:
        iso_values = np.linspace(negative_iso, positive_iso, num_isosurfaces)
    else:
        iso_values = isovals

    print(f"Isosurface values: {iso_values}")

    # Get the cell vectors from the ASE Atoms object
    cell_vectors = cell
    print(f"Cell vectors: {cell_vectors}")
    # Create mesh traces
    all_vertices = []
    all_faces = []
    all_intensities = []
    vertex_offset = 0

    for iso_value in iso_values:
        verts, faces, _, _ = measure.marching_cubes(potential_V, iso_value, step_size=step)
        # print(f"verts: {verts}")
        # print(f"faces: {faces}")
        cartesian_vertices = find_coord_from_indices(verts, potential_V.shape, atoms)       
        # Scale vertices to the correct cartesian coordinates
        # Correct scaling for non-orthorhombic cells
        # Calculate the scaling factors based on the actual dimensions of the cell
        # scaling_factors = np.array(potential_V.shape) / np.array([cell_vectors[0, 0], cell_vectors[1, 1], cell_vectors[2, 2]])
        # verts_scaled = np.dot(verts / scaling_factors, cell_vectors)
        
        all_vertices.append(cartesian_vertices)
        # all_vertices.append(verts_scaled)
        all_faces.append(faces + vertex_offset)
        all_intensities.extend([iso_value] * len(verts))
        vertex_offset += len(verts)

    all_vertices = np.vstack(all_vertices)
    all_faces = np.vstack(all_faces)

    combined_mesh = go.Mesh3d(
        x=all_vertices[:, 0],
        y=all_vertices[:, 1],
        z=all_vertices[:, 2],
        i=all_faces[:, 0],
        j=all_faces[:, 1],
        k=all_faces[:, 2],
        colorscale=cscale,
        intensity=all_intensities,
        intensitymode='vertex',
        opacity=base_opacity,
        name='Electrostatic Potential',
        showlegend=True,
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>Potential: %{intensity:.2f} V<extra></extra>'
    )

    return [combined_mesh]

# def plot_connections(connections, maxima, shape, ase_atoms, plot_periodic=True, **kwargs):
#     """Plot connections graph including periodic boundary connections.
    
#     Args:
#         connections: List of connection arrays containing [node_indices, exists, type, coords, window_size]
#         maxima: Array of maxima positions in grid coordinates
#         shape: Shape of the grid (nx, ny, nz)
#         ase_atoms: ASE atoms object containing cell information
#         plot_periodic: Whether to plot periodic connections and nodes (default: True)
#         **kwargs: Additional arguments passed to go.Scatter3d
#     """
#     import numpy as np
#     import plotly.graph_objects as go
#     import plotly.express as px
#     from ase.geometry import complete_cell
    
#     def get_nearest_image_to_window(pos, window_pos, cell):
#         """Find nearest periodic image of pos relative to window_pos"""
#         frac_pos = np.linalg.solve(cell.T, pos)
#         frac_window = np.linalg.solve(cell.T, window_pos)
#         diff = frac_pos - frac_window
#         diff = diff - np.round(diff)
#         nearest_frac = frac_window + diff
#         return np.dot(cell.T, nearest_frac)

#     # Get unique nodes from connections (both entries)
#     nodes = set()
#     for conn in connections:
#         nodes.update(conn[0])  # conn[0] contains both node indices
#     nodes = sorted(list(nodes))
#     print(nodes)
    
#     # Get cell matrix with vectors as columns
#     cell = complete_cell(ase_atoms.get_cell()).T
    
#     # Calculate node positions in real space
#     node_positions = {}
#     for i, node in enumerate(nodes):
#         # Convert grid coordinates to fractional coordinates
#         frac_coords = maxima[node-1] / np.array(shape)
#         # Convert fractional to real space coordinates
#         real_coords = np.dot(cell, frac_coords.T).T
#         node_positions[node] = real_coords
    
#     # Modified color assignment
#     cscale = kwargs.get('cscale', px.colors.sequential.Turbo)
#     #get the line color from kwargs, defaults to red
#     line_color = kwargs.get('line_color', 'red')
#     # If sequential colorscale is provided as string, get the actual colorscale
#     if isinstance(cscale, str):
#         try:
#             cscale = getattr(px.colors.sequential, cscale)
#         except:
#             try:
#                 cscale = getattr(px.colors.qualitative, cscale)
#             except:
#                 pass
    
#     # Create color mapping
#     if isinstance(cscale, list):
#         node_to_color = {node: px.colors.sample_colorscale(cscale, (i)/len(nodes))[0] 
#                         for i, node in enumerate(nodes)}
#     else:
#         node_to_color = {node: px.colors.sample_colorscale(cscale, (i)/len(nodes))[0] 
#                         for i, node in enumerate(nodes)}
    
#     node_colors = [node_to_color[node] for node in nodes]
#     print(node_to_color)
#     # Create node positions and hover text for internal nodes
#     node_x, node_y, node_z = [], [], []
#     node_hover = []
    
#     for node in nodes:
#         pos = node_positions[node]
#         node_x.append(pos[0])
#         node_y.append(pos[1])
#         node_z.append(pos[2])
#         hover_text = f"Index: {node}<br>Type: Internal<br>Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) Å"
#         node_hover.append(hover_text)

#     # Create internal edges from connections
#     edge_x, edge_y, edge_z = [], [], []
#     edge_hover = []
    
#     for conn in connections:
#         if conn[1] and conn[2] == 'internal':  # if connection exists and is internal
#             n1, n2 = conn[0]
#             pos1 = node_positions[n1]
#             pos2 = node_positions[n2]
            
#             edge_x.extend([pos1[0], pos2[0], None])
#             edge_y.extend([pos1[1], pos2[1], None])
#             edge_z.extend([pos1[2], pos2[2], None])
            
#             if conn[3] is not None:
#                 for window_pos, window_size in zip(conn[3], conn[4]):
#                     hover_text = (f"Internal edge: {n1}-{n2}<br>"
#                                 f"Window position: ({window_pos[0]:.2f}, {window_pos[1]:.2f}, {window_pos[2]:.2f}) Å<br>"
#                                 f"Window size: {window_size:.2f} Å")
#                     edge_hover.extend([hover_text, hover_text, None])
#             else:
#                 hover_text = f"Internal edge: {n1}-{n2}"
#                 edge_hover.extend([hover_text, hover_text, None])

#     # Create traces for internal structure
#     traces = [
#         # Nodes trace
#         go.Scatter3d(
#             x=node_x, y=node_y, z=node_z,
#             mode='markers',
#             marker=dict(
#                 size=kwargs.get('marker_size', 10),
#                 color=node_colors,
#                 line=dict(width=2, color=node_colors)
#             ),
#             hovertext=node_hover,
#             hoverinfo='text',
#             name='Internal Maxima',
#             legendgroup='Pore Graph',
#             legendgrouptitle_text='Pore Graph'
#         ),
#         # Edges trace
#         go.Scatter3d(
#             x=edge_x, y=edge_y, z=edge_z,
#             mode='lines',
#             line=dict(width=4, color=line_color),
#             hovertext=edge_hover,
#             hoverinfo='text',
#             name='Internal Connections',
#             legendgroup='Pore Graph'
#         )
#     ]

#     if not plot_periodic:
#         return traces

#     # Process periodic connections
#     periodic_edges = []
#     periodic_nodes = []
#     periodic_edge_hover = []
#     periodic_node_hover = []

#     for conn in connections:
#         if conn[1] and conn[2] == 'periodic':  # if connection exists and is periodic
#             n1, n2 = conn[0]
#             pos1 = node_positions[n1]
#             pos2 = node_positions[n2]
            
#             for window_pos, window_size in zip(conn[3], conn[4]):
#                 pos2_image = get_nearest_image_to_window(pos2, window_pos, cell)
                
#                 periodic_edges.extend([pos1, pos2_image, [None, None, None]])
                
#                 hover_text = (f"Periodic edge: {n1}-{n2}<br>"
#                             f"Window position: ({window_pos[0]:.2f}, {window_pos[1]:.2f}, {window_pos[2]:.2f}) Å<br>"
#                             f"Window size: {window_size:.2f} Å")
#                 periodic_edge_hover.extend([hover_text, hover_text, None])
                
#                 periodic_nodes.append(pos2_image)
                
#                 node_hover_text = (f"Index: {n2}<br>"
#                                  f"Type: Periodic<br>"
#                                  f"Position: ({pos2_image[0]:.2f}, {pos2_image[1]:.2f}, {pos2_image[2]:.2f}) Å")
#                 periodic_node_hover.append(node_hover_text)

#     if periodic_edges:
#         # Add periodic edges trace
#         edge_array = np.vstack(periodic_edges)
#         traces.append(go.Scatter3d(
#             x=edge_array[:, 0],
#             y=edge_array[:, 1],
#             z=edge_array[:, 2],
#             mode='lines',
#             line=dict(
#                 dash='dot',
#                 width=kwargs.get('line_width', 4),
#                 color=line_color
#             ),
#             hovertext=periodic_edge_hover,
#             hoverinfo='text',
#             name='Periodic Connections',
#             legendgroup='Pore Graph'
#         ))

#         # Add periodic nodes trace
#         periodic_nodes = np.vstack(periodic_nodes)
#         maxima_indices = [conn[0][1] for conn in connections if conn[1] and conn[2] == 'periodic']
#         periodic_colors = [node_to_color[idx] for idx in maxima_indices]
        
#         traces.append(go.Scatter3d(
#             x=periodic_nodes[:, 0],
#             y=periodic_nodes[:, 1],
#             z=periodic_nodes[:, 2],
#             mode='markers',
#             marker=dict(
#                 symbol='circle-open',
#                 size=kwargs.get('marker_size', 10),
#                 color=periodic_colors,
#                 line=dict(width=4, color=periodic_colors),
#                 opacity=1,
#             ),
#             hovertext=periodic_node_hover,
#             hoverinfo='text',
#             name='Periodic Maxima',
#             legendgroup='Pore Graph'
#         ))

#     return traces

def plot_connections(connections, maxima, shape, ase_atoms, plot_periodic=True, **kwargs):
    """Plot connections graph including periodic boundary connections.
    
    Args:
        connections: List of connection arrays containing [node_indices, exists, type, coords, window_size]
        maxima: Array of maxima positions in grid coordinates
        shape: Shape of the grid (nx, ny, nz)
        ase_atoms: ASE atoms object containing cell information
        plot_periodic: Whether to plot periodic connections and nodes (default: True)
        **kwargs: Additional arguments passed to go.Scatter3d
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from ase.geometry import complete_cell
    
    def get_nearest_image_to_window(pos, window_pos, cell):
        """Find nearest periodic image of pos relative to window_pos"""
        # Convert both positions to fractional coordinates
        # cell has vectors as columns, so we need to solve with cell instead of cell.T
        frac_pos = np.linalg.solve(cell, pos)
        frac_window = np.linalg.solve(cell, window_pos)
        
        # Calculate difference and apply PBC
        diff = frac_pos - frac_window
        diff = diff - np.round(diff)
        
        # Get the nearest image in fractional coords
        nearest_frac = frac_window + diff
        
        # Convert back to Cartesian - use direct dot product since cell vectors are columns
        return np.dot(cell, nearest_frac)

    # Get unique nodes from connections (both entries)
    nodes = set()
    for conn in connections:
        nodes.update(conn[0])  # conn[0] contains both node indices
    nodes = sorted(list(nodes))
    print(nodes)
    
    # Get cell matrix with vectors as columns
    cell = complete_cell(ase_atoms.get_cell()).T
    
    # Calculate node positions in real space
    node_positions = {}
    for i, node in enumerate(nodes):
        # Convert grid coordinates to fractional coordinates
        frac_coords = maxima[node-1] / np.array(shape)
        # Convert fractional to real space coordinates
        real_coords = np.dot(cell, frac_coords.T).T
        node_positions[node] = real_coords
    
    # Modified color assignment
    cscale = kwargs.get('cscale', px.colors.sequential.Turbo)
    #get the line color from kwargs, defaults to red
    line_color = kwargs.get('line_color', 'red')
    # If sequential colorscale is provided as string, get the actual colorscale
    if isinstance(cscale, str):
        try:
            cscale = getattr(px.colors.sequential, cscale)
        except:
            try:
                cscale = getattr(px.colors.qualitative, cscale)
            except:
                pass
    
    # Create color mapping
    if isinstance(cscale, list):
        node_to_color = {node: px.colors.sample_colorscale(cscale, (i)/len(nodes))[0] 
                        for i, node in enumerate(nodes)}
    else:
        node_to_color = {node: px.colors.sample_colorscale(cscale, (i)/len(nodes))[0] 
                        for i, node in enumerate(nodes)}
    
    node_colors = [node_to_color[node] for node in nodes]
    print(node_to_color)
    # Create node positions and hover text for internal nodes
    node_x, node_y, node_z = [], [], []
    node_hover = []
    
    for node in nodes:
        pos = node_positions[node]
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_z.append(pos[2])
        hover_text = f"Index: {node}<br>Type: Internal<br>Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) Å"
        node_hover.append(hover_text)

    # Create internal edges from connections
    edge_x, edge_y, edge_z = [], [], []
    edge_hover = []
    
    for conn in connections:
        if conn[1] and conn[2] == 'internal':  # if connection exists and is internal
            n1, n2 = conn[0]
            pos1 = node_positions[n1]
            pos2 = node_positions[n2]
            
            edge_x.extend([pos1[0], pos2[0], None])
            edge_y.extend([pos1[1], pos2[1], None])
            edge_z.extend([pos1[2], pos2[2], None])
            
            if conn[3] is not None:
                for window_pos, window_size in zip(conn[3], conn[4]):
                    hover_text = (f"Internal edge: {n1}-{n2}<br>"
                                f"Window position: ({window_pos[0]:.2f}, {window_pos[1]:.2f}, {window_pos[2]:.2f}) Å<br>"
                                f"Window size: {window_size:.2f} Å")
                    edge_hover.extend([hover_text, hover_text, None])
            else:
                hover_text = f"Internal edge: {n1}-{n2}"
                edge_hover.extend([hover_text, hover_text, None])

    # Create traces for internal structure
    traces = [
        # Nodes trace
        go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=kwargs.get('marker_size', 10),
                color=node_colors,
                line=dict(width=2, color=node_colors)
            ),
            hovertext=node_hover,
            hoverinfo='text',
            name='Internal Maxima',
            legendgroup='Pore Graph',
            legendgrouptitle_text='Pore Graph'
        ),
        # Edges trace
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=4, color=line_color),
            hovertext=edge_hover,
            hoverinfo='text',
            name='Internal Connections',
            legendgroup='Pore Graph'
        )
    ]

    if not plot_periodic:
        return traces

    # Process periodic connections
    periodic_edges = []
    periodic_nodes = []
    periodic_edge_hover = []
    periodic_node_hover = []

    for conn in connections:
        if conn[1] and conn[2] == 'periodic':  # if connection exists and is periodic
            n1, n2 = conn[0]
            pos1 = node_positions[n1]
            pos2 = node_positions[n2]
            
            for window_pos, window_size in zip(conn[3], conn[4]):
                pos2_image = get_nearest_image_to_window(pos2, window_pos, cell)
                
                periodic_edges.extend([pos1, pos2_image, [None, None, None]])
                
                hover_text = (f"Periodic edge: {n1}-{n2}<br>"
                            f"Window position: ({window_pos[0]:.2f}, {window_pos[1]:.2f}, {window_pos[2]:.2f}) Å<br>"
                            f"Window size: {window_size:.2f} Å")
                periodic_edge_hover.extend([hover_text, hover_text, None])
                
                periodic_nodes.append(pos2_image)
                
                node_hover_text = (f"Index: {n2}<br>"
                                 f"Type: Periodic<br>"
                                 f"Position: ({pos2_image[0]:.2f}, {pos2_image[1]:.2f}, {pos2_image[2]:.2f}) Å")
                periodic_node_hover.append(node_hover_text)

    if periodic_edges:
        # Add periodic edges trace
        edge_array = np.vstack(periodic_edges)
        traces.append(go.Scatter3d(
            x=edge_array[:, 0],
            y=edge_array[:, 1],
            z=edge_array[:, 2],
            mode='lines',
            line=dict(
                dash='dot',
                width=kwargs.get('line_width', 4),
                color=line_color
            ),
            hovertext=periodic_edge_hover,
            hoverinfo='text',
            name='Periodic Connections',
            legendgroup='Pore Graph'
        ))

        # Add periodic nodes trace
        periodic_nodes = np.vstack(periodic_nodes)
        maxima_indices = [conn[0][1] for conn in connections if conn[1] and conn[2] == 'periodic']
        periodic_colors = [node_to_color[idx] for idx in maxima_indices]
        
        traces.append(go.Scatter3d(
            x=periodic_nodes[:, 0],
            y=periodic_nodes[:, 1],
            z=periodic_nodes[:, 2],
            mode='markers',
            marker=dict(
                symbol='circle-open',
                size=kwargs.get('marker_size', 10),
                color=periodic_colors,
                line=dict(width=4, color=periodic_colors),
                opacity=1,
            ),
            hovertext=periodic_node_hover,
            hoverinfo='text',
            name='Periodic Maxima',
            legendgroup='Pore Graph'
        ))

    return traces


def make_fig_orthographic(fig):
    for trace in fig.data:
        if trace.type == "mesh3d":
            trace.update(lighting=default_mesh_lighting)
            trace.update(lightposition=dict(x=None, y=None, z=None))
    fig.update_layout(scene_camera_projection_type="orthographic")
    return fig
def make_fig_orthographic_all_subplots(fig):
    # Update lighting for each mesh
    for subplot in fig.data:
        if subplot.type == "mesh3d":
            subplot.update(lighting=default_mesh_lighting)
            subplot.update(lightposition=dict(x=None, y=None, z=None))

    # Loop through all layout keys to find all 3D scenes
    for scene_key in fig.layout:
        if scene_key.startswith("scene"):
            fig.layout[scene_key].update(camera=dict(projection=dict(type="orthographic")))

    return fig

def plot_connections_as_it_is( connections, maxima, shape, ase_atoms, plot_periodic=True, plot_disconnected=False, **kwargs):
    """Plot connections graph including periodic boundary connections.

    Args:
        connections: List of connection arrays containing [node_indices, exists, type, coords, window_size]
        maxima: Array of maxima positions in grid coordinates
        shape: Shape of the grid (nx, ny, nz)
        ase_atoms: ASE atoms object containing cell information
        plot_periodic: Whether to plot periodic connections and nodes (default: True)
        **kwargs: Additional arguments passed to go.Scatter3d
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from ase.geometry import complete_cell

    # def get_nearest_image_to_window(pos, window_pos, cell):
    #     """Find nearest periodic image of pos relative to window_pos"""
    #     # Convert both positions to fractional coordinates
    #     # cell has vectors as columns, so we need to solve with cell instead of cell.T
    #     frac_pos = np.linalg.solve(cell, pos)
    #     frac_window = np.linalg.solve(cell, window_pos)
    #
    #     # Calculate difference and apply PBC
    #     diff = frac_pos - frac_window
    #     diff = diff - np.round(diff)
    #
    #     # Get the nearest image in fractional coords
    #     nearest_frac = frac_window + diff
    #
    #     # Convert back to Cartesian - use direct dot product since cell vectors are columns
    #     return np.dot(cell, nearest_frac)

    # Get unique nodes from connections (both entries)
    nodes = set()
    for conn in connections:
        nodes.update(conn[0])  # conn[0] contains both node indices
    nodes = sorted(list(nodes))
    print('these are all the nodes from connections', nodes)

    # Get cell matrix with vectors as columns
    cell = complete_cell(ase_atoms.get_cell()).T

    # Calculate node positions in real space
    node_positions = {}
    for i, node in enumerate(nodes):
        # Convert grid coordinates to fractional coordinates
        frac_coords = maxima[node - 1] / np.array(shape)
        # Convert fractional to real space coordinates
        real_coords = np.dot(cell, frac_coords.T).T
        node_positions[node] = real_coords

    # Modified color assignment
    cscale = kwargs.get("cscale", px.colors.sequential.Turbo)
    # get the line color from kwargs, defaults to red
    line_color = kwargs.get("line_color", "red")
    # If sequential colorscale is provided as string, get the actual colorscale
    if isinstance(cscale, str):
        try:
            cscale = getattr(px.colors.sequential, cscale)
        except:
            try:
                cscale = getattr(px.colors.qualitative, cscale)
            except:
                pass

    # Create color mapping
    if isinstance(cscale, list):
        # Create color mapping based on all maxima
        node_to_color = {
            node: px.colors.sample_colorscale(cscale, (node - 1) / (len(maxima) - 1))[0]
            for node in range(1, len(maxima) + 1)
        }
        # node_to_color = { node: px.colors.sample_colorscale(cscale, (i) / len(nodes))[0]
        #     for i, node in enumerate(nodes)
        # }
    else:
        # Create color mapping based on all maxima
        node_to_color = {
            node: px.colors.sample_colorscale(cscale, (node - 1) / (len(maxima) - 1))[0]
            for node in range(1, len(maxima) + 1)
        }

        # node_to_color = {
        #
        #     node: px.colors.sample_colorscale(cscale, (i) / len(nodes))[0]
        #     for i, node in enumerate(nodes)
        # }
        #
    node_colors = [node_to_color[node] for node in nodes]
    print('node to color: ', node_to_color)
    # Create node positions and hover text for internal nodes
    node_x, node_y, node_z = [], [], []
    node_hover = []
    # nodes that appear in periodic and internal connections
    # nodes that apear in periodic connections
    periodic_connected_nodes = set()

    for conn in connections:
        if (
            conn[1] and conn[2] == "periodic"
        ):  # Check if connection exists and is periodic
            # Add both nodes in the connection
            periodic_connected_nodes.update(conn[0])
    # print("periodic connected nodes are ", periodic_connected_nodes)
    # Extract indices of nodes in internal connections
    internal_connected_nodes = set()

    for conn in connections:
        if (
            conn[1] and conn[2] == "internal"
        ):  # Check if connection exists and is internal
            # Add both nodes in the connection
            internal_connected_nodes.update(conn[0])

    # print("internal connected nodes are ", internal_connected_nodes)
    # Find nodes that only appear in periodic connections
    exclusive_periodic_nodes = periodic_connected_nodes - internal_connected_nodes
    # print("Nodes exclusive to periodic connections:", exclusive_periodic_nodes)


    # find nodes that are not inside the box, but also are not connected internally or perioically
    # print('58 status: ', 58 in internal_connected_nodes, 58 in periodic_connected_nodes)
    # these are plotting the internal maxima
    for node in nodes:
        if node not in exclusive_periodic_nodes:
            pos = node_positions[node]
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_z.append(pos[2])
            hover_text = f"Index: {node}<br>Type: Internal<br>Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) Å<br>Indices: [{maxima[node - 1][0]},{maxima[node - 1][1]}, {maxima[node - 1][2]}]"
            node_hover.append(hover_text)

    # Create internal edges from connections
    edge_x, edge_y, edge_z = [], [], []
    edge_hover = []

    for conn in connections:
        if conn[1] and conn[2] == "internal":  # if connection exists and is internal
            n1, n2 = conn[0]
            pos1 = node_positions[n1]
            pos2 = node_positions[n2]

            edge_x.extend([pos1[0], pos2[0], None])
            edge_y.extend([pos1[1], pos2[1], None])
            edge_z.extend([pos1[2], pos2[2], None])

            if conn[3] is not None:
                for window_pos, window_size in zip(conn[3], conn[4]):
                    hover_text = (
                        f"Internal edge: {n1}-{n2}<br>"
                        f"Window position: ({window_pos[0]:.2f}, {window_pos[1]:.2f}, {window_pos[2]:.2f}) Å<br>"
                        f"Window size: {window_size:.2f} Å"
                    )
                    edge_hover.extend([hover_text, hover_text, None])
            else:
                hover_text = f"Internal edge: {n1}-{n2}"
                edge_hover.extend([hover_text, hover_text, None])

    # Create traces for internal structure
    traces = [
        # Nodes trace
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(
                size=kwargs.get("marker_size", 10),
                color=node_colors,
                line=dict(width=2, color=node_colors),
            ),
            hovertext=node_hover,
            hoverinfo="text",
            name="Internal Maxima",
            legendgroup="Pore Graph",
            legendgrouptitle_text="Pore Graph",
        ),
        # Edges trace
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=4, color=line_color),
            hovertext=edge_hover,
            hoverinfo="text",
            name="Internal Connections",
            legendgroup="Pore Graph",
        ),
    ]

    if not plot_periodic:
        return traces

    # Process periodic connections
    periodic_edges = []
    periodic_nodes = []
    periodic_edge_hover = []
    periodic_node_hover = []

    for conn in connections:
        if conn[1] and conn[2] == "periodic":  # if connection exists and is periodic
            n1, n2 = conn[0]
            pos1 = node_positions[n1]
            pos2 = node_positions[n2]

            for window_pos, window_size in zip(conn[3], conn[4]):
                # pos2_image = get_nearest_image_to_window(
                    # pos2, window_pos, cell)
                pos2_image = pos2

                periodic_edges.extend([pos1, pos2_image, [None, None, None]])

                hover_text = (
                    f"Periodic edge: {n1}-{n2}<br>"
                    f"Window position: ({window_pos[0]:.2f}, {window_pos[1]:.2f}, {window_pos[2]:.2f}) Å<br>"
                    f"Window size: {window_size:.2f} Å"
                )
                periodic_edge_hover.extend([hover_text, hover_text, None])

                periodic_nodes.append(pos2_image)

                node_hover_text = (
                    f"Index: {n2}<br>"
                    f"Type: Periodic<br>"
                    f"Position: ({pos2_image[0]:.2f}, {pos2_image[1]:.2f}, {pos2_image[2]:.2f}) Å"
                )
                periodic_node_hover.append(node_hover_text)
    if periodic_edges:
        # Create a "dotted" effect by plotting individual points along the edges
        dotted_x, dotted_y, dotted_z = [], [], []
        for i in range(0, len(periodic_edges) - 1, 3):  # Process each edge
            if periodic_edges[i] is not None and periodic_edges[i + 1] is not None:
                start = periodic_edges[i]
                end = periodic_edges[i + 1]
                num_points = kwargs.get("edge_resolution", 20)  # Default: 100 points
                t = np.linspace(0, 1, num_points)
                dotted_x.extend((1 - t) * start[0] + t * end[0])
                dotted_y.extend((1 - t) * start[1] + t * end[1])
                dotted_z.extend((1 - t) * start[2] + t * end[2])
                # Add None for breaks between edges
                dotted_x.append(None)
                dotted_y.append(None)
                dotted_z.append(None)

        # Add periodic edges as individual points
        traces.append(
            go.Scatter3d(
                x=dotted_x,
                y=dotted_y,
                z=dotted_z,
                mode="markers",
                marker=dict(
                    size=kwargs.get("dot_size", 0.5),  # Adjust dot size
                    color=line_color,
                    opacity=1,
                ),
                hoverinfo="skip",
                name="Periodic Connections (Dotted)",
                legendgroup="Pore Graph",
            )
        )
# if periodic_edges:
    #     # Add periodic edges trace
    #     edge_array = np.vstack(periodic_edges)
    #     traces.append(
    #         go.Scatter3d(
    #             x=edge_array[:, 0],
    #             y=edge_array[:, 1],
    #             z=edge_array[:, 2],
    #             mode="lines",
    #             line=dict(
    #                 dash="dot", width=kwargs.get("line_width", 4), color=line_color
    #             ),
    #             hovertext=periodic_edge_hover,
    #             hoverinfo="text",
    #             name="Periodic Connections",
    #             legendgroup="Pore Graph",
    #             # connectgaps=True,
    #         )
    #     )all_maxima
    #
        # Add periodic nodes trace
        periodic_nodes = np.vstack(periodic_nodes)
        maxima_indices = [
            conn[0][1] for conn in connections if conn[1] and conn[2] == "periodic"
        ]
        periodic_colors = [node_to_color[idx] for idx in maxima_indices]

        traces.append(
            go.Scatter3d(
                x=periodic_nodes[:, 0],
                y=periodic_nodes[:, 1],
                z=periodic_nodes[:, 2],
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=kwargs.get("marker_size", 10),
                    color=periodic_colors,
                    line=dict(width=4, color=periodic_colors),
                    opacity=1,
                ),
                hovertext=periodic_node_hover,
                hoverinfo="text",
                name="Periodic Maxima",
                legendgroup="Pore Graph",
            )
        )

    if plot_disconnected:
        maxima_coords = np.dot(cell, (maxima/shape).T).T
        # Identify disconnected nodes
        all_nodes = set(range(1, len(maxima) + 1))  # All nodes from maxima
        connected_nodes = set(nodes)  # Nodes from connections
        disconnected_nodes = all_nodes - connected_nodes

        # Create positions and colors for disconnected nodes
        disconnected_x, disconnected_y, disconnected_z = [], [], []
        disconnected_colors = []
        disconnected_hover = []

        for node in disconnected_nodes:
            # pos = maxima[node - 1]  # Convert 1-indexed to 0-indexed
            pos =maxima_coords[node - 1]
            disconnected_x.append(pos[0])
            disconnected_y.append(pos[1])
            disconnected_z.append(pos[2])
            disconnected_colors.append(node)  # Use node index for consistent coloring
            disconnected_hover.append(f"Index: {node}<br>Type:Not listed in connections")

        # Add a trace for disconnected nodes
        traces.append(
            go.Scatter3d(
                x=disconnected_x,
                y=disconnected_y,
                z=disconnected_z,
                mode="markers",
                marker=dict(
                    size=kwargs.get("marker_size", 10),
                    color=disconnected_colors,
                    colorscale=cscale,
                    opacity=0.3,  # Reduced opacity
                    line=dict(width=1, color="gray"),
                ),
                hovertext=disconnected_hover,
                hoverinfo="text",
                name="Disconnected Maxima",
                legendgroup="Pore Graph",
            )
        )

    return traces

def plot_dgrid_as_volume(
        dgrid,
        mat_atoms,
        original_shape=None,           # (ox, oy, oz) of the un-padded grid
        num_isosurfaces=5, cscale='viridis', opacity=0.1):
    """
    Visualise a distance grid (optionally padded in ‘wrap’ mode).

    Parameters
    ----------
    dgrid : np.ndarray
        3-D array holding the (possibly padded) distance grid.
    mat_atoms : ASE Atoms object
        Provides the unit-cell vectors.
    original_shape : tuple[int,int,int] or None
        Shape of the un-padded grid.  If None, the grid is assumed NOT padded.
    num_isosurfaces : int
        How many iso-value surfaces to draw.
    """
    import numpy as np
    import plotly.graph_objects as go
    from ase.geometry import complete_cell, cellpar_to_cell

    nx, ny, nz = dgrid.shape

    # ---- 1.  Fractional coordinate axes -----------------------------------
    if original_shape is None:
        # old behaviour: grid fills the unit cell exactly
        x_frac = np.linspace(0, 1, nx)
        y_frac = np.linspace(0, 1, ny)
        z_frac = np.linspace(0, 1, nz)
    else:
        ox, oy, oz = original_shape
        pad_x = (nx - ox) // 2   # # layers added on each face
        pad_y = (ny - oy) // 2
        pad_z = (nz - oz) // 2

        # index-based formulation keeps spacing identical to the original grid
        x_frac = (np.arange(nx) - pad_x) / ox      # −p/ox … 1+p/ox
        y_frac = (np.arange(ny) - pad_y) / oy
        z_frac = (np.arange(nz) - pad_z) / oz

    x, y, z = np.meshgrid(x_frac, y_frac, z_frac, indexing='ij')

    # ---- 2.  Convert to Cartesian -----------------------------------------
    cell = complete_cell(cellpar_to_cell(mat_atoms.cell.cellpar())).T
    cart_coords = np.dot(np.column_stack((x.ravel(), y.ravel(), z.ravel())), cell)
    x_cart, y_cart, z_cart = (c.reshape(x.shape) for c in np.split(cart_coords, 3, axis=1))

    # ---- 3.  Plotly Volume trace ------------------------------------------
    volume_trace = go.Volume(
        x=x_cart.flatten(),
        y=y_cart.flatten(),
        z=z_cart.flatten(),
        value=dgrid.flatten(),
        isomin=0,
        isomax=dgrid.max(),
        opacity=opacity,
        surface_count=num_isosurfaces,
        colorscale=cscale,
        reversescale=True,
        colorbar=dict(title="Distance (Å)"),
        hoverinfo='all',
        hoverlabel=dict(bgcolor='white'),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Distance Grid',
        showlegend=True,
    )
    return volume_trace

def get_pore_size_distribution_trace(distances, bins=np.linspace(0, 50, 50), normalize=False, cumulative=False):
    """Create a plotly trace for pore size distribution."""
    import plotly.graph_objects as go
    if normalize:
        distances = distances / np.max(distances)

    # Create histogram data
    hist, bin_edges = np.histogram(distances, bins=bins, density=normalize)

    if cumulative:
        hist = np.cumsum(hist)
    binC = (bin_edges[1:] + bin_edges[:-1])/2
    # Create the histogram trace
    trace = go.Scatter(
        x=binC,
        y=hist,
        mode='lines',
        name='Overall Pore Size Distribution',
        line=dict(width=2)
    )

    return trace

def plot_pocket_psds(pore_sizes, sample_points_labels, pocket_psds, bins, material_name="Test Material"):
    """Create and display plotly figure comparing overall PSD with individual pocket PSDs."""

    # Get the histogram trace for overall PSD
    import plotly.graph_objects as go
    histogram_trace = get_pore_size_distribution_trace(pore_sizes, normalize=False, cumulative=False, bins=bins)

    # Create the figure
    fig = go.Figure()
    fig.add_trace(histogram_trace)

    # Add individual pocket PSDs
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (label, psd_data) in enumerate(pocket_psds.items()):
        # Create trace for this pocket's PSD
        pocket_trace = go.Scatter(
            x=psd_data['bin_centers'],
            y=psd_data['hist'],
            mode='lines',
            name=f'Pocket {label} ({psd_data["num_samples"]} samples)',
            line=dict(width=2, color=colors[i % len(colors)], dash='dot')
        )
        fig.add_trace(pocket_trace)

    # Update plot layout
    fig.update_layout(
        title=f"Pore Size Distribution: Overall vs Individual Pockets for {material_name}",
        xaxis_title='Pore Diameter (Å)',
        yaxis_title='Counts',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.8
        ),
        width=1000,
        height=600
    )

    # Display the plot
    # fig.show(renderer='chrome')
    return fig



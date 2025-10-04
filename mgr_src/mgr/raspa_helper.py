def make_raspa_input_for_cif(folder_path, raspa_settings_dict, other_files_to_copy, templates_dir, template_name):
    
    
    import jinja2 as jj 
    import shutil
    import os 
    ej = jj.Environment(loader = jj.FileSystemLoader(templates_dir))
    tlt = ej.get_template(template_name)
    simulation_input_string = tlt.render(**raspa_settings_dict)
    with open(os.path.join(os.path.abspath(folder_path), 'simulation.input'), 'w') as f: 
        f.write(simulation_input_string)
    for f in other_files_to_copy:
        shutil.copy(os.path.abspath(f), os.path.abspath(folder_path))

def get_isotherm_df_for_raspa_folder(path_to_folder):
    """
    It takes a list of paths to folders containing raspa output files and returns a dictionary with the
    isotherm data
    
    :param path_to_folder: the path to the folder containing the raspa output files
    :return: A dictionary with the isotherm data
    """

    import numpy as np
    import os
    import glob
    import pandas as pd

    current_path = os.getcwd()
    os.chdir(path_to_folder)
    files = glob.glob('**/*.data', recursive=True)
    data = np.zeros((len(files), 8))
    for i in range(len(files)):
            file = files[i]
            # print(file)
            data[i,0] = float(file.split('/')[0]) # Pressure
            lines = open(file).readlines()
            if np.sum(['Average loading excess [molecules/unit cell]' in ll for ll in lines])>0:
                # * Read the average loading and isosteric heats
                pfl = np.where(['Partial fugacity' in ll for ll in lines])[0][0]
                mucl =  np.where(['Average loading excess [molecules/unit cell]' in ll for ll in lines])[0][0]
                ccpgl =  np.where(['Average loading excess [cm^3 (STP)/gr framework]' in ll for ll in lines])[0][0]
                ugl=  np.where(['Ug should be subtracted' in ll for ll in lines])[0][0]
                data[i, 1] = float(lines[pfl].split()[2])
                data[i, 2] = float(lines[mucl].split()[5])
                data[i, 3] = float(lines[mucl].split()[7])
                data[i, 4] = float(lines[ccpgl].split()[6])
                data[i, 5] = float(lines[ccpgl].split()[8])
                data[i, 6] = float(lines[ugl-1].split()[0])
                data[i, 7] = float(lines[ugl-1].split()[2])
            else:
                data[i,:] = np.NaN*np.ones(np.size(data[i,:]))
    data = data[np.argsort(data[:, 0])]
    df = pd.DataFrame(data,
                      columns=['Pressure', 'Partial fugacity', 'Average loading excess [molecules/unit cell]', 'Error',
                               'Average loading excess [cm^3 (STP)/gr framework]', 'Error', 'Isos. Heat (kJ/mol)',
                               'Error'])
    # np.savetxt(filename,out,fmt='%.4f',delimiter='\t\t')
    os.chdir(current_path)
    return df
def get_isotherms_dict_for_multiple_raspa_folders(list_of_folder_paths):
    '''
    This function takes a list of paths to folders containing raspa output files and returns a dictionary with the isotherm data
    :param list_of_folder_paths: list of paths to the folders containing the raspa output files
    :return: a dictionary with the isotherm data
    '''
    import numpy as np
    from tqdm import tqdm
    isotherms_dict = {}
    for folder_path in tqdm(list_of_folder_paths):
        df = get_isotherm_df_for_raspa_folder(folder_path)
        isotherms_dict[folder_path.split('/')[-1]] = df
    return isotherms_dict



#TODO: make a function to do clustered regions grid for a give output folder

# TODO: make a function to decompose the isotherm into component isotherms

# TODO: make a function to do the isotherm decomposition for a given output folder that wraps around all the above

def get_isos_fit_for_raspa_folder(path_to_folder, T, order):
    """* Description
    :type path_to_folder: string
    :param path_to_folder: Path to the RASPA folder with all the output_*.data files
    :type T: float
    :param T: Temperature
    :type order: integer
    :param order:Order of the Legendre polynomials to be used in regression
    :raises:
    :rtype: Number in box array, RASPA P.E array [K], Fitted P.E array[K], number in mol/kg, Isosteric heat (J/mol)
    """
    """
        """
    #* ###  #####  #######  #####  ####### ####### ######  ###  #####     #     # #######    #    #######
    #*  #  #     # #     # #     #    #    #       #     #  #  #     #    #     # #         # #      #
    #*  #  #       #     # #          #    #       #     #  #  #          #     # #        #   #     #
    #*  #   #####  #     #  #####     #    #####   ######   #  #          ####### #####   #     #    #
    #*  #        # #     #       #    #    #       #   #    #  #          #     # #       #######    #
    #*  #  #     # #     # #     #    #    #       #    #   #  #     #    #     # #       #     #    #
    #* ###  #####  #######  #####     #    ####### #     # ###  #####     #     # ####### #     #    #
    """
        """
    # * Python code to read the RASPA output and calulate the isosteric heat of adsorption
    # * by regressing <U> vs <N>
    # * This code uses Legendre polynomials to do the same
    # * Takes the path to the folder as the input
    # * Written and tested by Arun Gopalan, Snurr Group, Northwestern University @ 2019
    import numpy as np
    import glob
    import pandas as pd
    import scipy as sp
    import os
    import pandas as pd
    cwd = os.getcwd()
    # self.path= path_to_folder #'/projects/p20832/ALKANES_ZIFs/ZIF-4/flexible/hybrid/pentane_ArunGopalan2018ZIF-4_TraPPE_ZIF-4_mod_FLEX_EMPTY_10NVE_UNCHARGED_LOG_303'
    # self.T = T
    os.chdir(path_to_folder)
    datafiles = glob.glob('**/*.data', recursive=True)
    # * Read  all the required energies and the number of molecules in the box
    total_energy = []
    guest_guest_energy = []
    guest_host_energy = []
    number = []
    number_molpg = []
    for d in datafiles:
        filename = d
        f1 = open(filename).readlines()
        for i in range(len(f1)):
            if "Total energy:\n" in f1[i]:
                total_energy.append(float(f1[i+8].split()[1]))
            # if "Average Adsorbate-Adsorbate energy:\n" in f1[i]:
            #     guest_guest_energy.append(float(f1[i+8].split()[1]))
            # if "Average Host-Adsorbate energy:\n" in f1[i]:
            #     guest_host_energy.append(float(f1[i+8].split()[1]))
            # ! Reads the previous line which is the average
            if "Average loading absolute [molecules/unit cell]" in f1[i]:
                # print(f1[i-5:i+5])
                number.append(float(f1[i-1].split()[3]))
            if "Average loading absolute [mol/kg framework]" in f1[i]:
                number_molpg.append(float(f1[i].split()[5]))
    total_energy = np.array(total_energy)  # in kelvin per box
    number = np.array(number)        # per box
    # sort the energy and loading
    total_energy = total_energy[np.argsort(number)]
    number = np.sort(number)
    number_molpg = np.array(number_molpg)
    number_molpg = np.sort(number_molpg)
    R = 8.314  # J/(K. mol)

    # * Do the Legendre fit and derivative
    fit_coeff = np.polynomial.legendre.legfit(
        number, total_energy, order, full=False)
    energy_fit = np.polynomial.legendre.legval(number, fit_coeff)
    der_coeff = np.polynomial.legendre.legder(fit_coeff, m=1, scl=1)
    dUdN = np.polynomial.legendre.legval(number, der_coeff)
    Isos = R*(T - dUdN)  # in J/mol
    output = np.column_stack((number, total_energy, energy_fit, number_molpg, Isos))
    df = pd.DataFrame(output, columns=['Number', 'U_total', 'U_total_fit', 'Number_molpg', 'Isos'])
    os.chdir(cwd)
    return df

def get_isos_fit_for_multiple_raspa_folders(list_of_folder_paths, T, order):
    """
    It takes a list of folder paths, and a temperature, and returns a dictionary of the
    isosteric heat fits for each folder
    
    :param list_of_folder_paths: a list of strings, each string being the path to a folder
    containing RASPA output files
    :param T: temperature in K
    :param order: the order of the polynomial fit
    :return: A dictionary of the form {folder_path: isos_fit_for_raspa_folder(folder_path,
    T, order)}
    """
    from tqdm import tqdm
    isos_heats_dict = {}
    for folder in tqdm(list_of_folder_paths):
        isos_heats_dict[folder] = get_isos_fit_for_raspa_folder(folder, T, order)
    return isos_heats_dict



# def get_atoms_list_for_pdb(pdb_file, index_list=None):#, natoms, bonds_in_ref=None, trace_name='adsorbates', pbc=[1, 1, 1]):
#     """
#     It reads a RASPA pdb file and returns a list of ASE atoms objects
    
#     :param pdb_file: the name of the pdb file
#     :param index_list: a list of indices of the configurations you want to extract from the pdb file. If
#     you want all of them, just leave it as None
#     :return: A list of atoms objects.
#     """

#     from ase import Atoms
#     import numpy as np
#     from .data import read_raspa_pdb
#     from rich.progress import track
#     config_data = read_raspa_pdb(pdb_file)
#     atoms_list = []
#     if index_list is None:
#         index_list = np.arange(len(config_data['coords']))
#     for index in track(index_list, description='Reading Configs'):
#         coords = config_data['coords'][index]
#         sym = config_data['symbols'][index]
#         sym = [str(c) for c in sym]
#         cell = config_data['cells'][index]
#         # nmolecules = len(coords) // natoms  # numbre of co2 molecules
#         # convert to ase atoms object
#         atoms = Atoms(symbols=sym, positions=coords, cell=cell, pbc=[1, 1, 1])
#         atoms_list.append(atoms)
#     return atoms_list
def get_atoms_list_for_pdb(pdb_file, index_list=None, symbol_map=None):
    """
    It reads a RASPA pdb file and returns a list of ASE atoms objects
    
    :param pdb_file: the name of the pdb file
    :param index_list: a list of indices of the configurations you want to extract from the pdb file. If
    you want all of them, just leave it as None
    :param symbol_map: Optional dictionary mapping custom symbols to standard element symbols.
    :return: A list of atoms objects.
    """

    from ase import Atoms
    import numpy as np
    from .data import read_raspa_pdb
    from rich.progress import track

    config_data = read_raspa_pdb(pdb_file, symbol_map=symbol_map)
    atoms_list = []
    
    if index_list is None:
        index_list = np.arange(len(config_data['coords']))
    
    for index in track(index_list, description='Reading Configs'):
        coords = config_data['coords'][index]
        sym = config_data['symbols'][index]
        
        # Check if coords is empty or not
        if coords.size == 0:
            print(f"Warning: No coordinates found for index {index}. Skipping this configuration.")
            continue
        
        # Ensure symbols are in the correct format
        sym = [str(c) for c in sym]
        
        # Check if cell dimensions are available
        if index < len(config_data['cells']):
            cell = config_data['cells'][index]
        else:
            print(f"Warning: No cell dimensions found for index {index}. Skipping this configuration.")
            continue
        
        # Convert to ASE Atoms object
        try:
            atoms = Atoms(symbols=sym, positions=coords, cell=cell, pbc=[1, 1, 1])
            atoms_list.append(atoms)
        except Exception as e:
            print(f"Error creating Atoms object for index {index}: {e}")
    
    return atoms_list
def apply_region_map_to_raspa_pdb_wrapped(pdb_file, regions_cluster, unit_cell, natoms_per_molecule=1, index_list=None, symbol_map=None):

    """
    Applies a region map to a RASPA PDB file, wrapping coordinates to a single unit cell.

    :param pdb_file: the pdb file you want to analyze
    :param regions_cluster: the region map for a single unit cell
    :param unit_cell: the unit cell matrix or ASE Atoms object containing the unit cell information
    :param natoms_per_molecule: number of atoms per molecule. This is currently only implemented for 1,
    defaults to 1 (optional)
    :param index_list: list of indices of atoms to use in the pdb file. If None, all atoms are used
    :return: A dictionary with the number of atoms in each cluster type and the total number of atoms.
    """
    import numpy as np
    from .helper import interpolate_labels
    from ase.geometry import wrap_positions
    from ase import Atoms

    atoms_list = get_atoms_list_for_pdb(pdb_file, index_list=index_list, symbol_map=symbol_map)
    unique_cluster_types = np.unique(regions_cluster)
    # remove background 0
    unique_cluster_types = unique_cluster_types[unique_cluster_types!=0]

    # * make a dictionary of cluster types to store their counts and means
    cluster_type_counts = {str(c-1):[] for c in unique_cluster_types}
    cluster_type_counts['total']= []

    # this is the interpolator function for a single unit cell
    region_interp = interpolate_labels(regions_cluster)

    if natoms_per_molecule != 1:
        print('ERROR: natoms_per_molecule is not 1, currently only implemented for 1')

    # Get the unit cell
    if isinstance(unit_cell, Atoms):
        cell = unit_cell.get_cell()
    else:
        cell = unit_cell

    # loop over the snapshots in the pdb
    for snap in atoms_list:
        # Get the positions from the snapshot
        positions = snap.get_positions()
        
        # Wrap the positions to the unit cell
        wrapped_positions = wrap_positions(positions, cell)
        
        # Convert to fractional coordinates
        scaled_positions = cell.scaled_positions(wrapped_positions)

        # shift any scaled positions if they are out of bounds
        scaled_positions = scaled_positions%1
        # Apply the region map
        cluster_indices = region_interp(scaled_positions)
        
        number_total = len(cluster_indices)
        for c in unique_cluster_types:
            cluster_type_counts[str(c-1)].append(len(np.where(cluster_indices == c)[0]))
        cluster_type_counts['total'].append(number_total)

    for c in unique_cluster_types:
        cluster_type_counts[str(c-1)+'_mean'] = np.mean(cluster_type_counts[str(c-1)])
        cluster_type_counts[str(c-1)+'_std'] = np.std(cluster_type_counts[str(c-1)])
    cluster_type_counts['total_mean'] = np.mean(cluster_type_counts['total'])

    return cluster_type_counts

# TODO: Implement for natoms > 1
def apply_region_map_to_raspa_pdb(pdb_file, regions_cluster_super, natoms_per_molecule=1, index_list=None):
    """
    It takes a pdb file and a region map and returns a dictionary of the number of atoms in each region
    
    :param pdb_file: the pdb file you want to analyze
    :param regions_cluster_super: the supercell regions map
    :param natoms_per_molecule: number of atoms per molecule. This is currently only implemented for 1,
    defaults to 1 (optional)
    :param index_list: list of indices of atoms to use in the pdb file. If None, all atoms are used
    :return: A dictionary with the number of atoms in each cluster type and the total number of atoms.
    """

    import numpy as np
    from .helper import interpolate_labels
    atoms_list = get_atoms_list_for_pdb(pdb_file, index_list=index_list)
    unique_cluster_types = np.unique(regions_cluster_super)
    # remove background 0
    unique_cluster_types = unique_cluster_types[unique_cluster_types!=0]

    # * make a dictionary of cluster types to store their counts and means
    cluster_type_counts = {str(c-1):[] for c in unique_cluster_types}
    cluster_type_counts['total']= []

    # this is the interpolator function
    region_interp_super = interpolate_labels(regions_cluster_super)
    if natoms_per_molecule !=1:
        print('ERROR: natoms_per_molecule is not 1, currently only implemented for 1')
    # loop over the snapshots in the pdb
    for snap in atoms_list:
        cluster_indices = region_interp_super(snap.get_scaled_positions())
        number_total = len(cluster_indices)
        for c in unique_cluster_types:
            cluster_type_counts[str(c-1)].append(len(np.where(cluster_indices == c)[0]))
        cluster_type_counts['total'].append(number_total)
    for c in unique_cluster_types:
        cluster_type_counts[str(c-1)+'_mean'] = np.mean(cluster_type_counts[str(c-1)])
        cluster_type_counts[str(c-1)+'_std'] = np.std(cluster_type_counts[str(c-1)])
    cluster_type_counts['total_mean'] = np.mean(cluster_type_counts['total'])

    return cluster_type_counts


def block_pockets_at_random_and_write(file_path, rag, ase_atoms, random_seed=123):
    """
    > This function takes a graph, ase atoms object, and a file path, and writes a file with the
    fractional coordinates of the centroids of the pockets in the graph, and the radius of the pockets
    
    :param file_path: the path to the file you want to write the blocking information to
    :param rag: the networkx graph object
    :param ase_atoms: ASE atoms object
    :param random_seed: the random seed to use for the random number generator, defaults to 123
    (optional)
    :return: The blocking_fractions and _ are being returned.
    """

    from .helper import get_fractional_coordinates
    import numpy as np
    import networkx as nx
    cluster_type_dict = nx.get_node_attributes(rag, 'cluster')
    centroid_dict = nx.get_node_attributes(rag, 'centroid')
    size_dict = nx.get_node_attributes(rag, 'maxima_radii')
    np.random.seed(random_seed)
    unique_pocket_types = np.unique(list(cluster_type_dict.values())).astype(int)
    nunique = len(unique_pocket_types)
    blocking_fractions = np.random.random(nunique) # what percentage of each cavity type to block
    blocking_file = open(file_path, 'w')

    # print(unique_pocket_types)
    for i,c in enumerate(unique_pocket_types):
        typec = [k for k,v in cluster_type_dict.items() if v == str(int(c))]
        pockets_to_block = np.random.choice(typec, np.max([int(blocking_fractions[i]*len(typec)),1]), replace=False)
        # print(typec, int(blocking_fractions[i]*len(typec)), pockets_to_block)
        pockets_to_block_centroids = np.vstack([centroid_dict[p] for p in pockets_to_block])
        pockets_to_block_sizes = np.vstack([size_dict[p] for p in pockets_to_block])
        scaled_centroids = get_fractional_coordinates(pockets_to_block_centroids, ase_atoms)
        scaled_centroids = np.round(scaled_centroids,2)
        for i,s in enumerate(scaled_centroids):
            blocking_file.write(str(s[0])+'\t'+str(s[1])+'\t'+str(s[2])+'\t'+str(np.round(pockets_to_block_sizes[i][0],2))+'\n')
    blocking_file.close()
    return blocking_fractions, _


def block_specific_pockets_and_write(file_path, rag, ase_atoms, pocket_ids_to_block):
    """
    Takes specific pocket IDs and writes their blocking information to a file.
    
    :param file_path: the path to the file you want to write the blocking information to
    :param rag: the networkx graph object
    :param ase_atoms: ASE atoms object
    :param pocket_ids_to_block: list of pocket IDs to block
    """
    from .helper import get_fractional_coordinates
    import numpy as np
    import networkx as nx
    
    centroid_dict = nx.get_node_attributes(rag, 'centroid')
    size_dict = nx.get_node_attributes(rag, 'maxima_radii')
    
    # Get centroids and sizes for specified pockets
    pockets_to_block_centroids = np.vstack([centroid_dict[p] for p in pocket_ids_to_block])
    pockets_to_block_sizes = np.vstack([size_dict[p] for p in pocket_ids_to_block])
    
    # Convert to fractional coordinates
    scaled_centroids = get_fractional_coordinates(pockets_to_block_centroids, ase_atoms)
    scaled_centroids = np.round(scaled_centroids, 2)
    
    # Write to file
    with open(file_path, 'w') as blocking_file:
        for i, s in enumerate(scaled_centroids):
            blocking_file.write(f"{s[0]}\t{s[1]}\t{s[2]}\t{np.round(pockets_to_block_sizes[i][0], 2)}\n")

#
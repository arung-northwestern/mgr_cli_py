# """
#  Functions for analyzing the chemical structure in mofography
#   █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗
#  ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝
#  ███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  
#  ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  
#  ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗
#  ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝
                                                         
 
# """

from rdkit import Chem
from typing import Iterable, Dict, List


def find_functional_groups(ase_atoms, functional_groups):
    from rdkit import Chem
    from networkx.algorithms import isomorphism
    import networkx as nx

    # Get bonds and bond orders without replication
    bonds, bond_orders = get_bonds_and_orders_without_replication(ase_atoms)

    # Store bonds and bond orders in the atoms object's info dictionary
    ase_atoms.info['covalent_bonds'] = bonds
    ase_atoms.info['covalent_bond_orders'] = bond_orders

    # Convert the supercell to a NetworkX graph
    G = sgraph_from_atoms(ase_atoms)

    # Convert functional groups to NetworkX graphs
    functional_group_graphs = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()}

    # Custom node match function
    def node_match(node1, node2):
        return node1['atomic_number'] == node2['atomic_number']

    # Custom edge match function
    def edge_match(edge1, edge2):
        bond_type1 = edge1['bond_type'].lower()
        bond_type2 = edge2['bond_type']
        if bond_type1 == 'aromatic':
            bond_type1 = 'double'
        return bond_type1 == bond_type2

    # Initialize matches with None for all functional groups
    matches = {name: None for name in functional_groups.keys()}

    # Search for functional groups in the molecule graph
    for name, fg_mol in functional_group_graphs.items():
        fg_G = nx.Graph()
        for atom in fg_mol.GetAtoms():
            fg_G.add_node(atom.GetIdx(),
                          atomic_number=atom.GetAtomicNum(),
                          formal_charge=atom.GetFormalCharge(),
                          chirality=atom.GetChiralTag())
        for bond in fg_mol.GetBonds():
            bond_type = bond.GetBondType().name.lower()
            if bond_type == 'aromatic':
                bond_type = 'double'
            fg_G.add_edge(bond.GetBeginAtomIdx(),
                          bond.GetEndAtomIdx(),
                          bond_type=bond_type)
        GM = isomorphism.GraphMatcher(G, fg_G, node_match=node_match, edge_match=edge_match)
        if GM.subgraph_is_isomorphic():
            matches[name] = list(GM.subgraph_isomorphisms_iter())

    return matches
# def find_functional_groups(ase_atoms, functional_groups):
#     from rdkit import Chem
#     from networkx.algorithms import isomorphism
#     import networkx as nx
#     # import mofography as mgr 

#     # Convert the supercell to a NetworkX graph
#     ase_atoms_updated = update_with_covalent_bonds(ase_atoms)
#     G = sgraph_from_atoms(ase_atoms_updated)

#     # Convert functional groups to NetworkX graphs
#     functional_group_graphs = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()}

#     # Custom node match function
#     def node_match(node1, node2):
#         return node1['atomic_number'] == node2['atomic_number']

#     # Custom edge match function
#     def edge_match(edge1, edge2):
#         bond_type1 = edge1['bond_type'].lower()
#         bond_type2 = edge2['bond_type']
#         if bond_type1 == 'aromatic':
#             bond_type1 = 'double'
#         return bond_type1 == bond_type2

#     # Initialize matches with None for all functional groups
#     matches = {name: None for name in functional_groups.keys()}

#     # Search for functional groups in the molecule graph
#     for name, fg_mol in functional_group_graphs.items():
#         fg_G = nx.Graph()
#         for atom in fg_mol.GetAtoms():
#             fg_G.add_node(atom.GetIdx(),
#                           atomic_number=atom.GetAtomicNum(),
#                           formal_charge=atom.GetFormalCharge(),
#                           chirality=atom.GetChiralTag())
#         for bond in fg_mol.GetBonds():
#             bond_type = bond.GetBondType().name.lower()
#             if bond_type == 'aromatic':
#                 bond_type = 'double'
#             fg_G.add_edge(bond.GetBeginAtomIdx(),
#                           bond.GetEndAtomIdx(),
#                           bond_type=bond_type)
#         GM = isomorphism.GraphMatcher(G, fg_G, node_match=node_match, edge_match=edge_match)
#         if GM.subgraph_is_isomorphic():
#             matches[name] = list(GM.subgraph_isomorphisms_iter())

#     return matches


# def find_functional_groups(ase_atoms, functional_groups):

#     # Convert the supercell to an RDKit molecule
#     from io import StringIO
#     import networkx as nx
#     from rdkit import Chem
#     from ase import Atoms
#     from ase.io import write, read
#     from networkx.algorithms import isomorphism
#     pdb_str = StringIO()
#     write(pdb_str, ase_atoms, format='proteindatabank')
#     mol = Chem.MolFromPDBBlock(pdb_str.getvalue(), sanitize=False, removeHs=False)
    
#     mol.UpdatePropertyCache(strict=False)
#     Chem.GetSymmSSSR(mol)

#     # Convert the RDKit molecule to a NetworkX graph
#     G = nx.Graph()
#     for atom in mol.GetAtoms():
#         G.add_node(atom.GetIdx(),
#                    atomic_num=atom.GetAtomicNum(),
#                    formal_charge=atom.GetFormalCharge(),
#                    chirality=atom.GetChiralTag())
#     for bond in mol.GetBonds():
#         G.add_edge(bond.GetBeginAtomIdx(),
#                    bond.GetEndAtomIdx(),
#                    bond_type=bond.GetBondType())

#     # Convert functional groups to NetworkX graphs
#     functional_group_graphs = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()}

#     # Custom node match function
#     def node_match(node1, node2):
#         return node1['atomic_num'] == node2['atomic_num']

#     # Custom edge match function
#     def edge_match(edge1, edge2):
#         return edge1['bond_type'] == edge2['bond_type']

#     # Search for functional groups in the molecule graph
#     matches = {}
#     for name, fg_mol in functional_group_graphs.items():
#         fg_G = nx.Graph()
#         for atom in fg_mol.GetAtoms():
#             fg_G.add_node(atom.GetIdx(),
#                           atomic_num=atom.GetAtomicNum(),
#                           formal_charge=atom.GetFormalCharge(),
#                           chirality=atom.GetChiralTag())
#         for bond in fg_mol.GetBonds():
#             fg_G.add_edge(bond.GetBeginAtomIdx(),
#                           bond.GetEndAtomIdx(),
#                           bond_type=bond.GetBondType())
#         GM = isomorphism.GraphMatcher(G, fg_G, node_match=node_match, edge_match=edge_match)
#         if GM.subgraph_is_isomorphic():
#             matches[name] = list(GM.subgraph_isomorphisms_iter())

#     return matches




def detect_smiles_in_mol(smiles_molecule, smiles_pattern):
    """
    It takes a SMILES string for a molecule and a SMILES string for a pattern, and returns a list of
    dictionaries that map the nodes of the pattern to the nodes of the molecule
    
    :param smiles_molecule: The SMILES string of the molecule you want to search in
    :param smiles_pattern: The SMILES pattern you want to search for
    :return: A list of dictionaries. Each dictionary contains the mapping of the nodes in the pattern to
    the nodes in the molecule.
    """
    import pysmiles as ps
    from networkx.algorithms.isomorphism import ISMAGS, numerical_edge_match, categorical_node_match

    pattern = ps.read_smiles(smiles_pattern)
    molecule = ps.read_smiles(smiles_molecule)
    isomo = ISMAGS(molecule, pattern, node_match =categorical_node_match('element', default='none'), edge_match=numerical_edge_match('order', default='100') )
    return list(isomo.subgraph_isomorphisms_iter())

def find_smiles_in_mofid(mofid, pattern_smiles):
    """
    It takes a MOFid string and a list of SMILES strings, and returns a list of the number of times each
    SMILES string is found in the MOFid string
    
    :param mofid: the MOFid string
    :param pattern_smiles: a list of SMILES strings that you want to search for in the MOFid
    :return: A list of lists of the patterns found in the MOFid.
    """
    import numpy as np
    mofid_parts = mofid.split('.')
    
    return [detect_smiles_in_mol(mofid_parts[i].split(' ')[0], pattern_smiles) if 'MOFid' in mofid_parts[i] else detect_smiles_in_mol(mofid_parts[i], pattern_smiles) for i in range(1, len(mofid_parts)-2)]


# def find_covalent_bonds_nl(data, nlist):
#     """
#     It takes the data and the neighbor list and returns the bonds and bond orders. 
    
#     The function is a bit long, but it's not too complicated. 
    
#     The first thing we do is import the necessary modules. 
    
#     Then we create a molecule object from the data. 
    
#     We then create a list of bonds and bond orders. 
    
#     We then create a list of covalent radii. 
    
#     We then loop over the neighbor list. 
    
#     We then check if the bond is covalent. 
    
#     If it is, we add it to the list of bonds and bond orders. 
    
#     We then return the bonds and bond orders. 
    
#     Let's test it out.
    
#     :param data: the atoms object
#     :param nlist: a list of nearest neighbors, which is a list of tuples. Each tuple is a pair of
#     indices of atoms
#     :return: the bonds and bond orders.
#     """
#     import itertools
#     from pymatgen.core.bonds import CovalentBond
#     from pymatgen.io.ase import AseAtomsAdaptor 
#     from ase.data import covalent_radii
#     import numpy as np

#     molecule = AseAtomsAdaptor.get_molecule(data)
#     number_of_sites = len(molecule.sites)
#     bonds = []
#     bond_order=[]
#     radii = covalent_radii[data.get_atomic_numbers()]
#     # frac = molecule.frac_coords
#     for (a, b) in nlist:
#         # First we check if the nearest image b from a is in the unit cell
#         # if not we will add that site, similar to vesta
        
        
        
#         cov1 = radii[a]
#         cov2 = radii[b]
#         bond_test = CovalentBond(molecule.sites[a], molecule.sites[b])
        
        
#         if CovalentBond.is_bonded(molecule.sites[a], molecule.sites[b], default_bl=cov1+cov2, tol=0.1):
#             bonds.append([a, b])
#             bond_order.append(bond_test.get_bond_order(default_bl=cov1+cov2, tol=0.1))
#             # print(bonds, bond_order)
#     return np.array(bonds), np.array(bond_order)

# def check_bonds_in_nlist(ase_atoms, nlist):
#     """
#     > The function takes an ASE atoms object and a list of nearest neighbors and returns an updated ASE
#     atoms object with the nearest neighbors added to the unit cell and the bonds and bond orders
    
#     :param ase_atoms: the atoms object from ASE
#     :param nlist: a list of pairs of atoms that are nearest neighbors
#     :return: the updated ase_atoms object, the bonds array, and the bond_order array.
#     """
#     from pymatgen.core.bonds import CovalentBond
#     from pymatgen.io.ase import AseAtomsAdaptor 
#     from ase.data import covalent_radii
#     import numpy as np

#     struct = AseAtomsAdaptor.get_structure(ase_atoms)
#     bonds = []
#     bond_order=[]
#     radii = covalent_radii[ase_atoms.get_atomic_numbers()]
#     frac = struct.frac_coords
#     for (a, b) in nlist:
#         # First we check if the nearest image b from a is in the unit cell
#         # if not we will add that site to the structure even if it is across 
#         # periodic boundaries, similar to how VESTA does it
#         cov1 = radii[a]
#         cov2 = radii[b]

            
#         if CovalentBond.is_bonded(struct.sites[a], struct.sites[b], default_bl=cov1+cov2, tol=0.1):
#             bond_test = CovalentBond(struct.sites[a], struct.sites[b])
#             dr = frac[a] - frac[b]
#             shift = np.round(dr)
#             if np.any(shift !=0):
#                 new_site_frac =frac[b] + shift
#                 new_site_coord =np.dot( ase_atoms.get_cell().T, new_site_frac).T
#                 struct.append(species=struct[b].species, coords=new_site_coord, coords_are_cartesian=True, properties = {'image of': b})
#                 b = struct.num_sites - 1 # the last one, the newly added site
#             bonds.append([a, b])
#             bond_order.append(bond_test.get_bond_order(default_bl=cov1+cov2, tol=0.1))
        
#     ase_atoms_updated = AseAtomsAdaptor.get_atoms(struct)
            
#     return ase_atoms_updated, np.array(bonds), np.array(bond_order)

def voronoi_on_atoms(ase_atoms, points=None):
    """
    It takes an ASE atoms object and returns a freud Voronoi object
    
    :param ase_atoms: the atoms object from ASE
    :return: The Voronoi object
    """
    from freud.box import Box 
    from freud.locality import Voronoi
    import numpy as np
    import ase
    if points is None:
        points = ase_atoms.get_positions()  
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T # TODO: should there be a T here? cell vectors should be columns
    box = Box.from_matrix(cell, dimensions =3)

    from freud.locality import Voronoi
    # voro = Voronoi(box=box, buffer=(ase_atoms.get_cell())/2)
    voro = Voronoi()
    voro.compute((box, points))
    return voro


def aabb_neighbors_on_atoms(ase_atoms, r_mult: float = 1.2):
    """Return periodic neighbor pairs using freud AABBQuery (fast cell list).

    Much faster than ``voronoi_on_atoms`` — uses an axis-aligned bounding
    box tree instead of full Voronoi tessellation.  On NU-1000 (558 atoms),
    AABBQuery is ~22× faster than Voronoi (0.001s vs 0.022s).

    Parameters
    ----------
    ase_atoms:
        ASE Atoms object.
    r_mult:
        Multiplier on max covalent radius sum for the query radius.
        Default 1.2 is tight enough to capture all bonded pairs while
        minimising false-positive candidates.

    Returns
    -------
    nlist : list[tuple[int, int]]
        List of (i, j) atom-index pairs within bonding distance.
    """
    from freud.box import Box
    from freud.locality import AABBQuery
    from ase.data import covalent_radii
    import numpy as np
    import ase

    points = ase_atoms.get_positions()
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T
    box = Box.from_matrix(cell, dimensions=3)

    # Query radius: sum of largest two covalent radii, with safety margin.
    # Clamp to half the minimum box dimension — freud AABBQuery rejects
    # r_max larger than the box (common for small cells with heavy atoms).
    radii = covalent_radii[ase_atoms.get_atomic_numbers()]
    max_bond_dist = float(np.max(radii) * 2 * r_mult)
    box_extents = box.Lx, box.Ly, box.Lz
    max_allowed = float(min(box_extents)) * 0.49  # slightly under box half
    if max_bond_dist > max_allowed:
        max_bond_dist = max_allowed

    aq = AABBQuery(box, points)
    query_result = aq.query(points, dict(r_max=max_bond_dist, exclude_ii=True))
    nl = query_result.toNeighborList()
    return [(int(nl[i, 0]), int(nl[i, 1])) for i in range(len(nl))]


def update_with_covalent_bonds(ase_atoms, nlist=None):
    """
    Compute covalent bonds using Rust (mgrcli_rust.covalent_bonds) — 100x+ faster than pymatgen.
    
    For bonds crossing periodic boundaries, creates image atoms (copies of atoms across
    periodic boundaries) so that bond vectors don't appear to wrap across the unit cell.
    The returned Atoms object includes these image atoms, and bond indices reference them.
    
    :param ase_atoms: ASE Atoms object
    :param nlist: optional list of (i, j) neighbor pairs. If None, uses aabb_neighbors_on_atoms.
    :return: ASE Atoms object with 'covalent_bonds' and 'covalent_bond_orders' in info.
    """

    import numpy as np
    from ase import Atoms

    # Build neighbor list if not provided
    if nlist is None:
        nlist = aabb_neighbors_on_atoms(ase_atoms)
        tol = 0.2
    else:
        tol = 0.1

    if len(nlist) == 0:
        ase_atoms_updated = ase_atoms.copy()
        ase_atoms_updated.info['covalent_bonds'] = []
        ase_atoms_updated.info['covalent_bond_orders'] = []
        return ase_atoms_updated

    # Convert nlist to separate i, j arrays
    nlist_i = np.array([p[0] for p in nlist], dtype=np.int64)
    nlist_j = np.array([p[1] for p in nlist], dtype=np.int64)

    # Prepare inputs for Rust function
    positions = ase_atoms.get_positions().ravel().tolist()
    atomic_numbers = ase_atoms.get_atomic_numbers().tolist()
    # Rust expects column-major (Fortran order) of cell.T
    cell_fortran = ase_atoms.get_cell().array.T.flatten(order='F').tolist()

    # Call Rust covalent_bonds (100x+ faster than pymatgen)
    try:
        import mgrcli_rust
        bonds_flat, bond_orders, image_positions, image_atomic_numbers = \
            mgrcli_rust.covalent_bonds(positions, atomic_numbers, cell_fortran,
                                       nlist_i.tolist(), nlist_j.tolist(), tol)
    except (ImportError, AttributeError):
        # Fallback to pure Python if Rust extension not built
        return _update_with_covalent_bonds_python(ase_atoms, nlist, tol)

    # Build bonds list from flat array
    bonds = [[bonds_flat[2*i], bonds_flat[2*i+1]] for i in range(len(bonds_flat)//2)]

    # If there are image atoms, build a new Atoms object with original + image atoms
    n_images = len(image_atomic_numbers)
    if n_images > 0:
        # Convert image_positions (flat list) to (n_images, 3) array
        img_pos_array = np.array(image_positions).reshape(-1, 3)
        img_z_array = np.array(image_atomic_numbers)

        from ase.data import chemical_symbols
        img_symbols = [chemical_symbols[z] for z in img_z_array]

        # Build combined atoms: original + image atoms
        orig_positions = ase_atoms.get_positions()
        orig_symbols = ase_atoms.get_chemical_symbols()
        all_positions = np.vstack([orig_positions, img_pos_array])
        all_symbols = orig_symbols + img_symbols

        ase_atoms_updated = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            cell=ase_atoms.get_cell(),
            pbc=ase_atoms.get_pbc(),
        )
    else:
        ase_atoms_updated = ase_atoms.copy()

    ase_atoms_updated.info['covalent_bonds'] = bonds
    ase_atoms_updated.info['covalent_bond_orders'] = list(bond_orders)

    return ase_atoms_updated


def _update_with_covalent_bonds_python(ase_atoms, nlist_pairs, tol=0.2):
    """Fallback pure-Python implementation using pymatgen (original behavior)."""
    from pymatgen.core.bonds import CovalentBond
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.data import covalent_radii
    import numpy as np

    struct = AseAtomsAdaptor.get_structure(ase_atoms)
    bonds = []
    bond_order = []
    radii = covalent_radii[ase_atoms.get_atomic_numbers()]
    frac = struct.frac_coords

    for (a, b) in nlist_pairs:
        cov1 = radii[a]
        cov2 = radii[b]
        if CovalentBond.is_bonded(struct.sites[a], struct.sites[b], default_bl=cov1 + cov2, tol=tol):
            bond_test = CovalentBond(struct.sites[a], struct.sites[b])
            dr = frac[a] - frac[b]
            shift = np.round(dr)
            if np.any(shift != 0):
                new_site_frac = frac[b] + shift
                new_site_coord = np.dot(ase_atoms.get_cell().T, new_site_frac).T
                struct.append(species=struct[b].species, coords=new_site_coord,
                              coords_are_cartesian=True, properties={'image of': b})
                b = struct.num_sites - 1
            bonds.append([a, b])
            bond_order.append(bond_test.get_bond_order(default_bl=cov1 + cov2, tol=tol))

    ase_atoms_updated = AseAtomsAdaptor.get_atoms(struct)
    ase_atoms_updated.info['covalent_bonds'] = bonds
    ase_atoms_updated.info['covalent_bond_orders'] = bond_order
    return ase_atoms_updated


def get_bonds_and_orders_without_replication(ase_atoms, nlist=None):
    from pymatgen.core.bonds import CovalentBond
    from pymatgen.io.ase import AseAtomsAdaptor 
    from ase.data import covalent_radii
    import numpy as np

    struct = AseAtomsAdaptor.get_structure(ase_atoms)
    bonds = []
    bond_order=[]
    radii = covalent_radii[ase_atoms.get_atomic_numbers()]
    frac = struct.frac_coords
    if nlist is None:
        nlist = aabb_neighbors_on_atoms(ase_atoms)
    for (a, b) in nlist:
        cov1 = radii[a]
        cov2 = radii[b]
        if CovalentBond.is_bonded(struct.sites[a], struct.sites[b], default_bl=cov1+cov2, tol=0.2):
            bond_test = CovalentBond(struct.sites[a], struct.sites[b])
            bonds.append([a, b])
            bond_order.append(bond_test.get_bond_order(default_bl=cov1+cov2, tol=0.2))

    return bonds, bond_order
def find_cycles_in_sgraph(sgraph, default_ring_sizes=[5,6,7,8,9]):
    """
    > The function takes a graph and returns a list of cycles in the graph. 
    
    The function is written in Python and uses the [NetworkX](https://networkx.github.io/) library. 
    
    The function takes two arguments: 
    
    1. `sgraph`: The graph to be analyzed. 
    2. `default_ring_sizes`: A list of integers that specify the sizes of rings to be considered. 
    
    The function returns a list of cycles in the graph. 
    
    The function first finds all the cycles in the graph using the `cycle_basis` function from NetworkX.
    
    
    The function then filters the cycles based on the `default_ring_sizes` argument. 
    
    The function then finds the bond types of the edges in the cycles. 
    
    The function then finds the unique cycles in the graph using the `find_unique_cycles` function
    
    :param sgraph: The graph that you want to find cycles in
    :param default_ring_sizes: The ring sizes you want to look for
    :return: A list of lists of nodes that are in the same ring.
    """


    import networkx as nx 
    import numpy as np 
    

    small_cycles = [cycle for cycle in nx.cycles.cycle_basis(sgraph, 0) if len(cycle) in default_ring_sizes]
    
    
    bond_types = nx.get_edge_attributes(sgraph,'bond_type')
    cycles_edges = [list(zip(cycle,(cycle[1:]+cycle[:1]))) for cycle in small_cycles]
    cycle_bond_types = [[bond_types[tuple(sorted(edge))] for edge in cycle_edges] for cycle_edges in cycles_edges]
    # selected_cycles = [np.all(np.array(cycle_bonds) == 'double') for cycle_bonds in cycle_bond_types]
    chosen_cycles = np.array(small_cycles)#[selected_cycles]

    # Find unique rings using SMILES
    import pysmiles as ps 
    cycle_graphs = [sgraph.subgraph(cycle) for cycle in chosen_cycles]
    unique_cycles = set(tuple(sorted(a)) for a in chosen_cycles)
    unique_cycles = [list(u) for u in unique_cycles]
    unique_chosen_cycles = find_chemically_unique_cycles(unique_cycles, sgraph)
    # unique_cycle_indices = find_unique_cycles(chosen_cycles, sgraph)
    # unique_cycle_indices = np.unique([ps.write_smiles(cycle) for cycle in cycle_graphs], return_index=True)[1]
    # unique_chosen_cycles = np.array(cycle_graphs)[unique_cycle_indices]

    from itertools import combinations 
    import networkx.algorithms.isomorphism as isomo
    isomorphic_cycles = list([])
    for cycle in unique_chosen_cycles:
        graph = sgraph.subgraph(cycle)
        iso1 = isomo.GraphMatcher(sgraph, graph, node_match=isomo.categorical_node_match('element',default='None'), edge_match=isomo.categorical_edge_match('bond_type',default='None'))
        isomorphic_cycles = isomorphic_cycles + [list(subgraph.keys()) for subgraph in iso1.subgraph_isomorphisms_iter()]
    isomorphic_cycles = [list(b) for b in set(tuple(sorted(c)) for c in isomorphic_cycles)]
    return isomorphic_cycles
    # return unique_cycles

def find_aromatic_cycles_in_sgraph(sgraph, default_ring_sizes=[5,6,7,8,9]):
    """
    > Find all cycles in the graph, then find all cycles that are composed of double bonds, then find
    all unique cycles, then find all isomorphic cycles. 
    
    The function returns a list of lists, where each list contains the indices of the atoms in the
    aromatic cycle. 
    
    Let's try it out on our example molecule:
    
    :param sgraph: The graph object that you want to find aromatic cycles in
    :param default_ring_sizes: The ring sizes to consider
    :return: A list of lists of nodes that are part of aromatic cycles.
    """

    import networkx as nx 
    import numpy as np 
    
    from icecream import ic 
    ic.disable()
    small_cycles = [cycle for cycle in nx.cycles.cycle_basis(sgraph, 0) if len(cycle) in default_ring_sizes]
    
    
    bond_types = nx.get_edge_attributes(sgraph,'bond_type')
    cycles_edges = [list(zip(cycle,(cycle[1:]+cycle[:1]))) for cycle in small_cycles]
    cycle_bond_types = [[bond_types.get(tuple(sorted(edge)), 'single') for edge in cycle_edges] for cycle_edges in cycles_edges]
    selected_cycles = [np.all(np.array(cycle_bonds) == 'double') for cycle_bonds in cycle_bond_types]
    aromatic_cycles = [c for c, s in zip(small_cycles, selected_cycles) if s]
    ic(aromatic_cycles)

    cycle_graphs = [sgraph.subgraph(cycle) for cycle in aromatic_cycles]
    unique_cycles = set(tuple(sorted(a)) for a in aromatic_cycles)
    unique_cycles = [list(u) for u in unique_cycles]
    _unique_indices = find_chemically_unique_cycles(unique_cycles, sgraph)
    chemically_unique_cycles = [unique_cycles[i] for i in _unique_indices]
    # unique_cycle_indices = find_unique_cycles(aromatic_cycles, sgraph) #np.unique([ps.write_smiles(cycle) for cycle in cycle_graphs], return_index=True)[1]
    # unique_aromatic_cycles = np.array(cycle_graphs)[unique_cycle_indices]
    ic(chemically_unique_cycles)
    # print(chemically_unique_cycles)
    # print(len(chemically_unique_cycles))
    # from itertools import combinations 
    import networkx.algorithms.isomorphism as isomo
    isomorphic_aromatic_cycles = list([])
    isomorphic_aromatic_cycles = []# list([])
    
    for cycle in chemically_unique_cycles:
        graph = sgraph.subgraph(cycle)
        iso1 = isomo.GraphMatcher(sgraph, graph, node_match=isomo.categorical_node_match('element',default='None'), edge_match=isomo.categorical_edge_match('bond_type',default='None'))
        isomorphic_aromatic_cycles = isomorphic_aromatic_cycles + [list(subgraph.keys()) for subgraph in iso1.subgraph_isomorphisms_iter()]
        # ic(list(iso1.subgraph_isomorphisms_iter()))
        # isomorphic_aromatic_cycles.append(list(iso1.subgraph_isomorphisms_iter()))#[subgraph.keys() for subgraph in iso1.subgraph_isomorphisms_iter()])
    isomorphic_aromatic_cycles = [list(b) for b in set(tuple(sorted(c)) for c in isomorphic_aromatic_cycles)]
    
    return isomorphic_aromatic_cycles
    # return unique_cycles

def sgraph_from_atoms(ase_atoms,node_props_dict=None, edge_props_dict=None):
    """
    It takes an ase atoms object and returns a networkx graph object with the nodes and edges
    corresponding to the atoms and bonds in the atoms object. 
    
    The function also takes two optional arguments: 
    
    1. node_props_dict: A dictionary of node properties. The keys are the names of the properties and
    the values are the values of the properties. The values should be a list of the same length as the
    number of atoms in the atoms object. 
    
    2. edge_props_dict: A dictionary of edge properties. The keys are the names of the properties and
    the values are the values of the properties. The values should be a list of the same length as the
    number of bonds in the atoms object. 
    
    If the node_props_dict or edge_props_dict are not provided, the function will use the default set of
    properties. 
    
    The default set of node properties are:
    
    :param ase_atoms: ASE atoms object
    :param node_props_dict: A dictionary of node properties. The keys are the property names and the
    values are the property values. The values should be a list of length equal to the number of atoms
    in the structure
    :param edge_props_dict: A dictionary of edge properties. The keys are the property names and the
    values are the property values. The values should be a list of length equal to the number of edges
    in the graph
    :return: A networkx graph object.
    """
     
    import networkx as nx 
    import pandas as pd
    import numpy as np
    # from pymatgen.io.ase import AseAtomsAdaptor 
    # # struct = AseAtomsAdaptor.get_structure(ase_atoms)
    # from pymatgen.analysis.graphs import StructureGraph 
    # from pymatgen.analysis.local_env import CrystalNN

    # if use_crystalNN is False:
    sgraph = nx.Graph()
    # struct = AseAtomsAdaptor.get_structure(ase_atoms)
    # temp = StructureGraph.with_empty_graph(struct)
    # sgraph.te
    if 'covalent_bonds' not in ase_atoms.info.keys():
        Exception("Bond list not in the ase_atoms.info dictionary.\n Please call 'update_atoms_with_covalent_bonds(ase_atoms)' first. \n Or provide your own bond info under keys 'covalent_bonds' under ase_atoms.info.")
    else:
        if node_props_dict is None:
            print('No node props provided, using default set: [position, scaled_position, element and atomic_number]')
            node_props_dict = dict(position = ase_atoms.get_positions(), element = ase_atoms.get_chemical_symbols(), scaled_position = ase_atoms.get_scaled_positions(), atomic_number = ase_atoms.get_atomic_numbers())
        bonds = ase_atoms.info['covalent_bonds']
        # print(node_props_dict)
        sgraph.add_nodes_from([tuple((i, {key:value[i] for key, value in node_props_dict.items()})) for i in range(ase_atoms.get_global_number_of_atoms())])
        if edge_props_dict is None:
            try:
                print('No edge properties provided. Use the default set: [bond_order, bond_type]')
                orders_raw = ase_atoms.info['covalent_bond_orders']
                orders_arr = np.asarray(orders_raw, dtype=float).reshape(-1)
                nb = len(bonds)
                # Bonds list can diverge from bond_orders after replication / merges — mismatched
                # lengths crash edge creation with numpy IndexError when indexing orders[i].
                if orders_arr.size != nb:
                    if orders_arr.size < nb:
                        pad = np.ones(nb - orders_arr.size, dtype=float)
                        orders_arr = np.concatenate([orders_arr, pad])
                    else:
                        orders_arr = orders_arr[:nb]
                edge_props_dict = dict(
                    bond_order=orders_arr.tolist(),
                    bond_type=get_bond_types(orders_arr),
                )
            except:
                print("'covalent_bond_orders' not in ase_atoms.info. Hence no edge_properties added for now.")
                edge_props_dict = {}
        sgraph.add_edges_from([tuple((bonds[i][0], bonds[i][1], {key:value[i] for key,value in edge_props_dict.items()})) for i in range(len(bonds))])
    # else :
    #     print('Making a pymatgen StructureGraph.graph using CrystalNN')
    #     cnn = CrystalNN(weighted_cn=True, distance_cutoffs=(0.3,0.8), search_cutoff=5)
    #     sgraph = StructureGraph.with_local_env_strategy(struct, strategy=cnn).graph
    #     if node_props_dict is None:
    #             print('No node props provided, using default set: [position, scaled_position, element and atomic_number]')
    #             node_props_dict = dict(position = ase_atoms.get_positions(), element = ase_atoms.get_chemical_symbols(), scaled_position = ase_atoms.get_scaled_positions(), atomic_number = ase_atoms.get_atomic_numbers())
    #             bonds = ase_atoms.info['covalent_bonds']
    #     node_props_dict = {i:{key:value[i] for key, value in node_props_dict.items()} for i in range(ase_atoms.get_global_number_of_atoms())}
    #     nx.set_node_attributes(sgraph,node_props_dict)
    #     if edge_props_dict is None:
    #             try:
    #                 print('No edge properties provided. Use the default set: [bond_order, bond_type]')
    #                 edge_props_dict = dict(bond_order = ase_atoms.info['covalent_bond_orders'], bond_type=get_bond_types(ase_atoms.info['covalent_bond_orders']))
    #             except:
    #                 print("'covalent_bond_orders' not in ase_atoms.info. Hence no edge_properties added for now.")
    #                 edge_props_dict = {}
    #     edge_props_dict = {edge:{key:value[i] for key, value in edge_props_dict.items()} for i, edge in enumerate(list(sgraph.edges()))}
    #     nx.set_edge_attributes(sgraph, edge_props_dict)
    return sgraph

def get_bond_types(bond_orders):
    """
    It takes a list of bond orders and returns a list of bond types
    
    :param bond_orders: a list of bond orders, e.g. [1.0, 2.0, 1.0, 3.0]
    :return: A pandas series with the bond types
    """
    import pandas as pd
    return  pd.cut(bond_orders, bins=[0,1, 2, 3], labels=['single', 'double', 'triple'])


def find_chemically_unique_cycles(list_of_cycles, graph):
    """
    > The function takes a list of cycles and a graph as input, and returns the indices of the unique
    cycles in the list, after matching the 'element' of the nodes and the 'bond_order' of the edges
    
    The function is a bit long, so let's break it down. 
    
    First, we import the necessary libraries.
    
    :param list_of_cycles: a list of cycles, each cycle is a list of nodes
    :param graph: the graph object
    :return: The indices of the unique cycles in the list of cycles.
    """
    import networkx as nx 
    import networkx.algorithms.isomorphism as isomo
    sub_graphs = [graph.subgraph(cycle) for cycle in list_of_cycles]
    import numpy as np
    import itertools
    from icecream import ic 
    ic.disable() 
    # [(i.nodes(data='element'), j.nodes(data='element')) for (i,j) in list(itertools.combinations(sub_graphs, r=2))]
    # from networkx.algorithms import isomorphism as isomo 
    em = isomo.categorical_edge_match('bond_type', default='single')
    nm = isomo.categorical_node_match('element',default='None')
    iso_check = [isomo.GraphMatcher(*g,node_match=nm, edge_match=em).is_isomorphic() for g in itertools.combinations(sub_graphs, r= 2)]
    ic(iso_check)
    if len(iso_check) >0:
        to_delete = np.array(list(itertools.combinations(range(len(sub_graphs)),2)))[iso_check][:,1]
        return np.delete(list(range(len(sub_graphs))), to_delete.astype(int))
    else:
        raise Exception(" No cycles of given lengths found!")
            
def find_rings_in_ase_atoms(mat_atoms, cycle_lengths=[5, 6, 7, 8], make_unique=True):
    """
    The function `find_rings_in_ase_atoms` takes an ASE atoms object, identifies cycles in the
    corresponding graph, filters cycles based on length, and optionally returns unique cycles.
    
    :param mat_atoms: The `mat_atoms` parameter is a representation of a material system in the form of
    atoms. It likely contains information about the atomic positions, types of atoms, and possibly other
    properties of the material. The function `find_rings_in_ase_atoms` takes this material system
    represented by atoms and performs
    :param cycle_lengths: The `cycle_lengths` parameter is a list of integers that specifies the lengths
    of cycles to be considered when filtering the cycles found in the graph. In the provided function
    `find_rings_in_ase_atoms`, the cycles found in the graph are filtered based on their length, and
    only cycles with lengths
    :param make_unique: The `make_unique` parameter in the `find_rings_in_ase_atoms` function determines
    whether to return only unique cycles or all cycles that meet the specified criteria, defaults to
    True (optional)
    :return: The function `find_rings_in_ase_atoms` returns either unique cycles or all cycles based on
    the input parameters. If `make_unique` is set to True, it returns chemically unique cycles of
    specified lengths (5, 6, 7, 8). If `make_unique` is set to False, it returns all cycles of specified
    lengths without filtering for uniqueness.
    """
    import networkx as nx


    # Convert ase atoms to sgraph
    bonds, bond_orders = get_bonds_and_orders_without_replication(mat_atoms)
    mat_atoms.info['covalent_bonds'] = bonds
    mat_atoms.info['covalent_bond_orders'] = bond_orders
    mat_graph =sgraph_from_atoms(mat_atoms)

    # Find all cycles in the graph
    list_of_cycles = list(nx.cycle_basis(mat_graph))

    # Filter cycles based on their length
    small_cycles = [cycle for cycle in list_of_cycles if len(cycle) in cycle_lengths]

    if make_unique:
        # Find unique cycles
        unique_cycle_indices =find_chemically_unique_cycles(small_cycles, mat_graph)
        unique_cycles = [small_cycles[i] for i in unique_cycle_indices]
        return unique_cycles
    else:
        return small_cycles      
def find_pattern_in_sgraph(sgraph, pattern):
    """
    It takes a graph and a pattern graph and returns a list of dictionaries that map the nodes of the
    pattern graph to the nodes of the original graph
    
    :param sgraph: the graph we want to search for the pattern in
    :param pattern: The pattern graph to be searched for
    :return: A list of dictionaries. Each dictionary has the node labels of the pattern as keys and the
    node labels of the sgraph as values.
    """


    import networkx as nx 
    import networkx.algorithms.isomorphism as isomo
    import numpy as np

    em = isomo.categorical_edge_match('bond_type', 'single')
    nm = isomo.categorical_node_match('element','None')
    iso_parts = isomo.GraphMatcher(sgraph, pattern, node_match = nm , edge_match= em)
    return list(iso_parts.subgraph_isomorphisms_iter())


def get_subset_atoms(ase_atoms, subset):
    """
    It takes an ASE atoms object and a list of atom ids, and returns a new ASE atoms object containing
    only the atoms in the list
    
    :param ase_atoms: the atoms object of the material
    :param subset: a list of atom ids
    :return: The atoms object with the subset of atoms.
    """
    
    # from .analyze import sgraph_from_atoms, update_with_covalent_bonds

    import networkx as nx
    mat_graph = sgraph_from_atoms(ase_atoms)
    sg = mat_graph.subgraph(subset)
    cell = ase_atoms.get_cell()
    from ase import Atoms
    #make atoms object from the subgraph
    atoms = Atoms(symbols=list(nx.get_node_attributes(sg, 'element').values()), positions = list(nx.get_node_attributes(sg, 'position').values()), cell=cell)
    atoms_ = update_with_covalent_bonds(atoms)
    atoms_.info['atom ids'] = list(nx.get_node_attributes(sg, 'element').keys())
    return atoms_


def get_atoms_from_smiles(smiles, cell=None):
    """
    This function takes a SMILES string as input and returns an ASE Atoms object with the atomic
    positions and information about covalent bonds and bond orders.
    
    :param smiles: A string representing the SMILES notation of a molecule
    :param cell: The cell parameter is an optional argument that specifies the size of the simulation
    cell in Angstroms. If it is not provided, a default cell size of 10x10x10 Angstroms is used
    :return: an ASE Atoms object that represents the molecule specified by the SMILES string input. The
    Atoms object contains information about the atomic positions, atomic symbols, and covalent bonds in
    the molecule. Additionally, the Atoms object contains information about the SMILES string used to
    generate the molecule and the cell size used to embed the molecule.
    """

    from rdkit.Chem import AllChem
    from rdkit import Chem
    from ase import Atoms
    import numpy as np
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    out= Chem.MolToXYZBlock(mol).split('\n')
    # print('\n'.join(out))
    # edit = Chem.RWMol(mol)
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atomic_symbols = [Chem.GetPeriodicTable().GetElementSymbol(atomic_number) for atomic_number in atomic_numbers]
    if cell is None:
        cell = np.array([10, 10, 10])
    atoms = Atoms(symbols=atomic_symbols, positions=np.vstack(positions)+cell/2)
    atoms.info['smiles'] = smiles
    atoms.info['name'] = smiles
    # if there is at least one bond in the molecule, add the covalent bonds and bond orders to the atoms
    if mol.GetNumBonds() > 0:
        atoms.info['covalent_bonds'] = np.vstack([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()])
        atoms.info['covalent_bond_orders'] = [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]
    else:
        atoms.info['covalent_bonds'] = []
        atoms.info['covalent_bond_orders'] = []
    return atoms


def find_unique_sgraphs(sgraphs):

    import networkx as nx 
    import networkx.algorithms.isomorphism as isomo
    import numpy as np
    import itertools
    from icecream import ic 
    ic.disable() 
    # [(i.nodes(data='element'), j.nodes(data='element')) for (i,j) in list(itertools.combinations(sub_graphs, r=2))]
    # from networkx.algorithms import isomorphism as isomo 
    em = isomo.categorical_edge_match('bond_type', default='single')
    nm = isomo.categorical_node_match('element',default='None')
    iso_check = [isomo.GraphMatcher(*g,node_match=nm, edge_match=em).is_isomorphic() for g in itertools.combinations(sgraphs, r= 2)]
    ic(iso_check)
    if len(iso_check) >0:
        to_delete = np.array(list(itertools.combinations(range(len(sgraphs)),2)))[iso_check][:,1]
        return np.delete(list(range(len(sgraphs))), to_delete.astype(int))
    else:
        raise Exception(" No cycles of given lengths found!")

# --------------------------------------------------------------------------- #
def ase_to_rdkit(atoms) -> Chem.Mol:

    """ASE → RDKit with bond deduplication (no sanitisation needed)."""
    rw = Chem.RWMol()

    # atoms
    for at in atoms:
        rw.AddAtom(Chem.Atom(int(at.number)))          # cast NumPy → int

    # bonds (canonical order, unique)
    seen = set()
    for i, j in atoms.info["covalent_bonds"]:
        i, j = int(i), int(j)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        rw.AddBond(i, j, Chem.BondType.SINGLE)
        seen.add(key)

    return rw.GetMol()
# --------------------------------------------------------------------------- #

def rings_by_size_rdkit(
    atoms,
    ring_lengths: Iterable[int],
) -> Dict[int, List[List[int]]]:

    """Return {ring_size: [rings]} using RDKit’s C++ ring perception."""
    wanted = set(int(x) for x in ring_lengths)
    out: Dict[int, List[List[int]]] = {k: [] for k in wanted}

    mol = ase_to_rdkit(atoms)

    # GetSymmSSSR() triggers ring perception and yields each SSSR ring
    for ring in Chem.GetSymmSSSR(mol):      # ring is an IntVect of atom idxs
        size = len(ring)
        if size in wanted:
            out[size].append(list(ring))

    for k in out:                           # deterministic ordering (optional)
        out[k].sort()
    return out

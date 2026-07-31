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


def _ase_neighbor_pairs_on_atoms(ase_atoms, r_mult: float = 1.2):
    """Periodic neighbor candidacy via ASE (no freud)."""
    import numpy as np
    from ase.neighborlist import natural_cutoffs, neighbor_list

    cutoffs = np.asarray(natural_cutoffs(ase_atoms, mult=r_mult), dtype=float)
    i_idx, j_idx = neighbor_list("ij", ase_atoms, cutoffs, self_interaction=False)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i, j in zip(i_idx, j_idx):
        a, b = int(i), int(j)
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _aabb_neighbors_freud(ase_atoms, r_mult: float = 1.2):
    from freud.box import Box
    from freud.locality import AABBQuery
    from ase.data import covalent_radii
    import numpy as np
    import ase

    points = ase_atoms.get_positions()
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T
    box = Box.from_matrix(cell, dimensions=3)

    radii = covalent_radii[ase_atoms.get_atomic_numbers()]
    max_bond_dist = float(np.max(radii) * 2 * r_mult)
    box_extents = box.Lx, box.Ly, box.Lz
    max_allowed = float(min(box_extents)) * 0.49
    if max_bond_dist > max_allowed:
        max_bond_dist = max_allowed

    aq = AABBQuery(box, points)
    query_result = aq.query(points, dict(r_max=max_bond_dist, exclude_ii=True))
    nl = query_result.toNeighborList()
    return [(int(nl[i, 0]), int(nl[i, 1])) for i in range(len(nl))]


def aabb_neighbors_on_atoms(ase_atoms, r_mult: float = 1.2):
    """Periodic neighbor pairs for covalent-bond candidacy (freud AABB or ASE fallback)."""
    try:
        return _aabb_neighbors_freud(ase_atoms, r_mult=r_mult)
    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "freud not installed; using ASE neighbor_list for covalent-bond neighbor candidacy"
        )
        return _ase_neighbor_pairs_on_atoms(ase_atoms, r_mult=r_mult)


def _bond_neighbor_method() -> str:
    import os

    return os.environ.get("MOFOGRAPHY_BOND_NEIGHBORS", "aabb").strip().lower()


def _bond_timeout_seconds(timeout) -> float:
    import os

    if timeout is not None:
        return max(0.0, float(timeout))
    raw = os.environ.get("MGR_COVALENT_BONDS_TIMEOUT_SEC", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _bond_skip_atom_threshold(skip_if_atoms_above) -> int:
    import os

    if skip_if_atoms_above is not None:
        return max(0, int(skip_if_atoms_above))
    raw = os.environ.get("MGR_COVALENT_BONDS_SKIP_ATOMS", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 0


def _neighbor_pairs_for_bonds(ase_atoms, nlist=None):
    if nlist is not None:
        return list(nlist), 0.1
    method = _bond_neighbor_method()
    if method == "voronoi":
        voro = voronoi_on_atoms(ase_atoms)
        return voro.nlist, 0.2
    return aabb_neighbors_on_atoms(ase_atoms), 0.2


def _atoms_without_bonds(ase_atoms, *, reason: str):
    import logging

    out = ase_atoms.copy()
    out.info["covalent_bonds"] = []
    out.info["covalent_bond_orders"] = []
    out.info["covalent_bonds_skipped"] = True
    out.info["covalent_bonds_skip_reason"] = reason
    logging.getLogger("mofography.analyze").warning(
        "Covalent bonds skipped (%s) for %d atoms", reason, len(ase_atoms)
    )
    return out


def update_with_covalent_bonds(
    ase_atoms,
    nlist=None,
    *,
    timeout=None,
    skip_if_atoms_above=None,
    neighbor_method=None,
):
    """
    Compute covalent bonds using Rust (mgr_rust.covalent_bonds) — 100x+ faster than pymatgen.

    For bonds crossing periodic boundaries, creates image atoms (copies of atoms across
    periodic boundaries) so that bond vectors don't appear to wrap across the unit cell.
    The returned Atoms object includes these image atoms, and bond indices reference them.

    :param ase_atoms: ASE Atoms object
    :param nlist: optional list of (i, j) neighbor pairs. If None, uses aabb_neighbors_on_atoms (env MOFOGRAPHY_BOND_NEIGHBORS=voronoi for legacy).
    :return: ASE Atoms object with 'covalent_bonds' and 'covalent_bond_orders' in info.
    """
    import concurrent.futures
    import logging
    import os

    skip_thr = _bond_skip_atom_threshold(skip_if_atoms_above)
    if skip_thr > 0 and len(ase_atoms) > skip_thr:
        return _atoms_without_bonds(ase_atoms, reason=f">{skip_thr} atoms")

    tlim = _bond_timeout_seconds(timeout)
    if neighbor_method is not None:
        os.environ["MOFOGRAPHY_BOND_NEIGHBORS"] = str(neighbor_method).strip().lower()

    if tlim <= 0:
        return _update_with_covalent_bonds_impl(ase_atoms, nlist=nlist)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_update_with_covalent_bonds_impl, ase_atoms, nlist=nlist)
        try:
            return fut.result(timeout=tlim)
        except concurrent.futures.TimeoutError:
            return _atoms_without_bonds(ase_atoms, reason=f"timeout after {tlim:.0f}s")


def _update_with_covalent_bonds_impl(ase_atoms, nlist=None):
    import logging
    import time

    import numpy as np

    from ase import Atoms

    _log = logging.getLogger("mofography.analyze")
    _t0 = time.perf_counter()
    _log.info("Covalent bonds: starting (%d atoms)", len(ase_atoms))

    # Build neighbor list if not provided
    if nlist is None:
        nlist, tol = _neighbor_pairs_for_bonds(ase_atoms, nlist=None)
    else:
        tol = 0.1

    if len(nlist) == 0:
        ase_atoms_updated = ase_atoms.copy()
        ase_atoms_updated.info['covalent_bonds'] = []
        ase_atoms_updated.info['covalent_bond_orders'] = []
        _log.info("Covalent bonds: done in %.3fs (0 bonds)", time.perf_counter() - _t0)
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
        import mgr_rust
        bonds_flat, bond_orders, image_positions, image_atomic_numbers =             mgr_rust.covalent_bonds(positions, atomic_numbers, cell_fortran,
                                    nlist_i.tolist(), nlist_j.tolist(), tol)
    except (ImportError, AttributeError):
        # Fallback to pure Python if Rust extension not built
        return _update_with_covalent_bonds_python(ase_atoms, nlist, tol)

    # Build bonds list from flat array
    bonds = [[bonds_flat[2 * i], bonds_flat[2 * i + 1]] for i in range(len(bonds_flat) // 2)]

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

    _log.info(
        "Covalent bonds: done in %.3fs (%d pairs, %d bonds, %d images)",
        time.perf_counter() - _t0,
        len(nlist),
        len(bonds),
        n_images,
    )
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
        voro = voronoi_on_atoms(ase_atoms)
        nlist = voro.nlist
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

def _aromatic_env_float(name: str, default: str) -> float:
    import os

    return float(os.environ.get(name, default))


def _aromatic_env_int(name: str, default: str) -> int:
    import os

    return int(os.environ.get(name, default))


AROMATIC_BOND_ORDER_MIN = _aromatic_env_float("MGR_AROMATIC_ORDER_MIN", "1.5")
AROMATIC_BOND_ORDER_TOL = _aromatic_env_float("MGR_AROMATIC_ORDER_TOL", "0.2")
AROMATIC_MAX_WEAK_EDGES = _aromatic_env_int("MGR_AROMATIC_MAX_WEAK_EDGES", "1")
AROMATIC_CYCLE_MEAN_MIN = _aromatic_env_float("MGR_AROMATIC_CYCLE_MEAN_MIN", "1.35")
AROMATIC_EDGE_ORDER_FLOOR = _aromatic_env_float("MGR_AROMATIC_EDGE_ORDER_FLOOR", "0.9")


def _use_rust_aromatic_expansion() -> bool:
    import os

    return os.environ.get("MGR_AROMATIC_RUST", "1") != "0"


def _edge_data(sgraph, u: int, v: int):
    if sgraph.has_edge(u, v):
        return sgraph.get_edge_data(u, v) or {}
    return sgraph.get_edge_data(v, u) or {}


def _edge_is_aromatic(edge_data: dict) -> bool:
    bt = str(edge_data.get("bond_type", "single")).lower()
    if bt in ("double", "aromatic"):
        return True
    try:
        bo = float(edge_data.get("bond_order", 1.0))
    except (TypeError, ValueError):
        return False
    return bo >= AROMATIC_BOND_ORDER_MIN - AROMATIC_BOND_ORDER_TOL


def _cycle_is_aromatic(sgraph, cycle: list) -> bool:
    n = len(cycle)
    if n < 3:
        return False

    aromatic_edges = 0
    bond_orders: list[float] = []
    for i in range(n):
        u, v = cycle[i], cycle[(i + 1) % n]
        edge = _edge_data(sgraph, u, v)
        try:
            bo = float(edge.get("bond_order", 1.0))
        except (TypeError, ValueError):
            bo = 1.0
        bond_orders.append(bo)
        if _edge_is_aromatic(edge):
            aromatic_edges += 1

    if aromatic_edges == n:
        return True

    weak_edges = n - aromatic_edges
    if weak_edges > AROMATIC_MAX_WEAK_EDGES:
        return False
    if not bond_orders:
        return False
    if min(bond_orders) < AROMATIC_EDGE_ORDER_FLOOR:
        return False
    return (sum(bond_orders) / float(n)) >= AROMATIC_CYCLE_MEAN_MIN


def _aromatic_cycles_from_basis(sgraph, default_ring_sizes=None) -> list[list[int]]:
    import networkx as nx

    if default_ring_sizes is None:
        default_ring_sizes = [5, 6, 7, 8, 9]
    small_cycles = [
        cycle
        for cycle in nx.cycles.cycle_basis(sgraph, 0)
        if len(cycle) in default_ring_sizes
    ]
    aromatic = [c for c in small_cycles if _cycle_is_aromatic(sgraph, c)]
    return [list(u) for u in {tuple(sorted(a)) for a in aromatic}]


def _expand_aromatic_embeddings_python(sgraph, chemically_unique_cycles: list) -> list[list[int]]:
    import networkx.algorithms.isomorphism as isomo

    out: list[list[int]] = []
    for cycle in chemically_unique_cycles:
        graph = sgraph.subgraph(cycle)
        iso1 = isomo.GraphMatcher(
            sgraph,
            graph,
            node_match=isomo.categorical_node_match("element", default="None"),
            edge_match=isomo.categorical_edge_match("bond_type", default="None"),
        )
        out.extend(list(subgraph.keys()) for subgraph in iso1.subgraph_isomorphisms_iter())
    return [list(b) for b in {tuple(sorted(c)) for c in out}]


def _sgraph_to_rust_arrays(sgraph):
    n = sgraph.number_of_nodes()
    elements = [str(sgraph.nodes[i].get("element", "C")) for i in range(n)]
    edge_i, edge_j, bond_types = [], [], []
    for u, v, data in sgraph.edges(data=True):
        edge_i.append(int(u))
        edge_j.append(int(v))
        bond_types.append(str(data.get("bond_type", "single")))
    return n, edge_i, edge_j, elements, bond_types


def _expand_aromatic_embeddings_rust(sgraph, chemically_unique_cycles: list) -> list[list[int]]:
    import mgrcli_rust

    n, ei, ej, elems, bts = _sgraph_to_rust_arrays(sgraph)
    templates = [[int(x) for x in cycle] for cycle in chemically_unique_cycles]
    return mgrcli_rust.expand_aromatic_cycle_embeddings(n, ei, ej, elems, bts, templates)


def find_aromatic_cycles_in_sgraph_legacy(sgraph, default_ring_sizes=None):
    """Original strict-double NetworkX implementation (parity tests)."""
    import networkx as nx
    import numpy as np

    if default_ring_sizes is None:
        default_ring_sizes = [5, 6, 7, 8, 9]
    small_cycles = [
        c for c in nx.cycles.cycle_basis(sgraph, 0) if len(c) in default_ring_sizes
    ]
    bond_types = nx.get_edge_attributes(sgraph, "bond_type")
    cycles_edges = [
        list(zip(cycle, (cycle[1:] + cycle[:1]))) for cycle in small_cycles
    ]
    cycle_bond_types = [
        [bond_types.get(tuple(sorted(edge)), "single") for edge in cycle_edges]
        for cycle_edges in cycles_edges
    ]
    selected = [np.all(np.array(bt) == "double") for bt in cycle_bond_types]
    aromatic_cycles = [c for c, s in zip(small_cycles, selected) if s]
    unique_cycles = [list(u) for u in {tuple(sorted(a)) for a in aromatic_cycles}]
    _idx = find_chemically_unique_cycles(unique_cycles, sgraph)
    chemically_unique_cycles = [unique_cycles[i] for i in _idx]
    return _expand_aromatic_embeddings_python(sgraph, chemically_unique_cycles)


def find_aromatic_cycles_in_sgraph(
    sgraph,
    default_ring_sizes=None,
    *,
    expand_isomorphisms: bool = True,
    use_rust: bool | None = None,
):
    """
    Aromatic rings as atom-index lists (all distinct embeddings unless expand_isomorphisms=False).

    Bond-order tolerance helps unrelaxed CIFs (MGR_AROMATIC_ORDER_MIN/TOL).
    """
    import logging
    import time

    _log = logging.getLogger("mgr.analyze")
    _t0 = time.perf_counter()

    unique_cycles = _aromatic_cycles_from_basis(sgraph, default_ring_sizes)
    _log.info(
        "Aromatic cycles: basis+filter %.3fs (%d unique)",
        time.perf_counter() - _t0,
        len(unique_cycles),
    )
    if not unique_cycles:
        return []

    _MAX_NODES_FOR_EXPANSION = 10_000
    if sgraph.number_of_nodes() > _MAX_NODES_FOR_EXPANSION:
        _log.info(
            "Aromatic cycles: skipping expansion for large graph "
            "(%d nodes, %d unique rings); total %.3fs",
            sgraph.number_of_nodes(), len(unique_cycles),
            time.perf_counter() - _t0,
        )
        return [list(c) for c in unique_cycles]

    _t1 = time.perf_counter()
    _idx = find_chemically_unique_cycles(unique_cycles, sgraph)
    chemically_unique_cycles = [unique_cycles[i] for i in _idx]
    _log.info(
        "Aromatic cycles: chem-unique %.3fs (%d templates)",
        time.perf_counter() - _t1,
        len(chemically_unique_cycles),
    )

    if not expand_isomorphisms or len(chemically_unique_cycles) == len(unique_cycles):
        _log.info(
            "Aromatic cycles: total %.3fs (%d rings, expansion skipped)",
            time.perf_counter() - _t0,
            len(unique_cycles),
        )
        return [list(c) for c in unique_cycles]

    _t2 = time.perf_counter()
    if use_rust is None:
        use_rust = _use_rust_aromatic_expansion()
    backend = "python"
    if use_rust:
        try:
            out = _expand_aromatic_embeddings_rust(sgraph, chemically_unique_cycles)
            backend = "rust"
        except Exception as exc:
            _log.warning(
                "Aromatic cycles: Rust expand unavailable (%s), using Python",
                exc,
            )
            out = _expand_aromatic_embeddings_python(sgraph, chemically_unique_cycles)
    else:
        out = _expand_aromatic_embeddings_python(sgraph, chemically_unique_cycles)
    _log.info(
        "Aromatic cycles: expand (%s) %.3fs (%d rings); total %.3fs",
        backend,
        time.perf_counter() - _t2,
        len(out),
        time.perf_counter() - _t0,
    )
    return out


def _aromatic_cache_key(ase_atoms, **opts) -> str:
    import hashlib

    bonds = ase_atoms.info.get("covalent_bonds") or []
    orders = ase_atoms.info.get("covalent_bond_orders") or []
    payload = repr(
        (len(ase_atoms), tuple(map(tuple, bonds)), tuple(orders), tuple(sorted(opts.items())))
    ).encode()
    return hashlib.sha1(payload).hexdigest()

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
                edge_props_dict = dict(
                    bond_order=ase_atoms.info["covalent_bond_orders"],
                    bond_type=get_bond_types(
                        ase_atoms.info["covalent_bond_orders"],
                        bonds=bonds,
                        ase_atoms=ase_atoms,
                    ),
                )
            except:
                print("'covalent_bond_orders' not in ase_atoms.info. Hence no edge_properties added for now.")
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

# ---------------------------------------------------------------------------
# Aromatic ring detection — bond-length tolerances
# ---------------------------------------------------------------------------
# Continuous bond orders from pymatgen (``CovalentBond.get_bond_order``, tol=0.2
# in :func:`update_with_covalent_bonds`) are binned into single / double / triple.
# Crystallographic aromatic and fused-ring contacts often fall near bin edges and
# are misclassified. When bond order is ambiguous, we fall back to minimum-image
# (MIC) distances for C/N pairs on the aromatic search graph.
#
# Literature reference distances (Å; room-temperature organic / MOF linkers):
#
#   Contact              Typical range    Example / source
#   -------------------  ---------------  ---------------------------------
#   Aromatic C–C         1.38 – 1.42      Benzene 1.397 (Allen et al., Acta
#                                         Cryst. B 2002, CSD statistics)
#   Kekulé C–C in ring   1.44 – 1.47      Alternating single in benzene ~1.46
#   Fused aromatic C–C   1.47 – 1.58      Naphthalene fusion ~1.42–1.48;
#                                         porphyrin/benzene fusion ~1.55–1.56
#                                         (NU-1000 TBAPy linkers)
#   Aromatic C–N         1.30 – 1.37      Pyridine ~1.34; imidazole ~1.33–1.37
#   C≡N (nitrile)        1.15 – 1.16      True triple; keep below triple cutoff
#
# Implementation thresholds (Å, MIC) — upper bounds for “aromatic framework”:
_AROMATIC_CC_MAX_DIST = 1.62
# ^ Above Kekulé singles and fused benzene/pyrrole links; below typical C(sp3)–C.
_AROMATIC_CN_MAX_DIST = 1.50
# ^ Above aromatic C–N; below most saturated C–N singles.
_AROMATIC_CN_TRIPLE_MAX_DIST = 1.42
# ^ Reclassify pymatgen “triple” (bond order > 2) to aromatic double when the
#   contact is still in the aromatic C–N range (ZIF-20 imidazole–benzene fusion
#   ~1.30 Å was a common failure mode before this cutoff).
#
# See also: ``docs/aromatic_ring_detection.qmd`` (plotting / visualization).
# ---------------------------------------------------------------------------


def _mic_bond_distance(a, b, positions, cell):
    """Minimum-image distance between two atom indices."""
    import numpy as np
    from ase.geometry import get_distances

    pos = np.asarray(positions, dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    if bool(np.any(cell_arr)):
        return float(
            np.asarray(get_distances(pos[a], pos[b], cell=cell_arr, pbc=True)[1]).ravel()[0]
        )
    return float(np.linalg.norm(pos[a] - pos[b]))


def _distance_is_aromatic_framework(a, b, positions, cell, symbols):
    """True when a C/N pair is short enough to be aromatic framework (distance fallback).

    Uses module thresholds ``_AROMATIC_CC_MAX_DIST`` and ``_AROMATIC_CN_MAX_DIST``
    (see literature table in source comments above ``_AROMATIC_CC_MAX_DIST``).
    """
    sa, sb = symbols[int(a)], symbols[int(b)]
    if sa not in "CN" or sb not in "CN":
        return False
    dist = _mic_bond_distance(int(a), int(b), positions, cell)
    if sa == "C" and sb == "C":
        return dist <= _AROMATIC_CC_MAX_DIST
    if {sa, sb} == {"C", "N"}:
        return dist <= _AROMATIC_CN_MAX_DIST
    return False


def _is_aromatic_framework_edge(
    a,
    b,
    bond_types,
    positions,
    cell,
    symbols,
    *,
    graph=None,
    aug_edges=None,
):
    """Whether an edge counts as aromatic ring framework (type or distance)."""
    edge = tuple(sorted((int(a), int(b))))
    if aug_edges and edge in aug_edges:
        return True
    if graph is not None and graph.has_edge(a, b) and graph.edges[a, b].get("mic_closure"):
        return True
    bt = bond_types.get(edge)
    if bt in ("double", "aromatic"):
        return True
    sa, sb = symbols[int(a)], symbols[int(b)]
    if sa not in "CN" or sb not in "CN":
        return False
    dist = _mic_bond_distance(int(a), int(b), positions, cell)
    if bt == "triple" and {sa, sb} == {"C", "N"} and dist <= _AROMATIC_CN_TRIPLE_MAX_DIST:
        return True
    return _distance_is_aromatic_framework(int(a), int(b), positions, cell, symbols)


def _cycle_has_chord(graph, cycle):
    """True if non-adjacent cycle nodes share an edge (not a chordless cycle)."""
    import networkx as nx

    n = len(cycle)
    if n < 3:
        return False
    idx = {int(v): i for i, v in enumerate(cycle)}
    g = graph if isinstance(graph, nx.Graph) else nx.Graph(graph)
    for i, u in enumerate(cycle):
        u = int(u)
        for v in g.neighbors(u):
            v = int(v)
            if v not in idx:
                continue
            j = idx[v]
            if j == (i - 1) % n or j == (i + 1) % n:
                continue
            return True
    return False


# Dense C/N subgraphs (duplicate PBC sites) rarely host aromatic rings > 6 members.
_DENSE_CN_EDGE_RATIO = 2.5
_DENSE_CN_MAX_RING_SIZE = 6


def _ring_sizes_for_component(n_nodes, n_edges, default_ring_sizes):
    """Ring sizes to search in a biconnected component (skip hopeless sizes)."""
    sizes = sorted({int(s) for s in default_ring_sizes if 3 <= int(s) <= n_nodes})
    if not sizes:
        return []
    if n_nodes > 0 and (n_edges / float(n_nodes)) > _DENSE_CN_EDGE_RATIO:
        sizes = [s for s in sizes if s <= _DENSE_CN_MAX_RING_SIZE]
    return sizes


def _chordless_cycles_bounded(graph, max_size, *, min_size=3):
    """Enumerate chordless cycles with min_size <= length <= max_size.

    Uses biconnected-component decomposition so MOF linker subgraphs stay small.
    Compatible with NetworkX releases that lack ``chordless_cycles``.
    """
    import networkx as nx

    if max_size is None or max_size < 3 or min_size > max_size:
        return
    min_size = max(3, int(min_size))
    max_size = int(max_size)
    g = nx.Graph(graph)
    seen: set[tuple[int, ...]] = set()
    for nodes in nx.biconnected_components(g):
        if len(nodes) < min_size:
            continue
        sub = g.subgraph(nodes)
        for start in sorted(sub.nodes()):
            start = int(start)
            stack = [(start, [start], {start})]
            while stack:
                node, path, visited = stack.pop()
                plen = len(path)
                if plen > max_size:
                    continue
                for nbr in sub.neighbors(node):
                    nbr = int(nbr)
                    if nbr == start:
                        if min_size <= plen <= max_size and not _cycle_has_chord(sub, path):
                            key = tuple(sorted(path))
                            if key not in seen:
                                seen.add(key)
                                yield list(path)
                        continue
                    if nbr in visited or nbr < start or plen >= max_size:
                        continue
                    stack.append((nbr, path + [nbr], visited | {nbr}))


def _iter_chordless_cycles(graph, default_ring_sizes):
    """Yield chordless cycles whose lengths are in ``default_ring_sizes``."""
    import networkx as nx

    allowed = set(int(s) for s in default_ring_sizes)
    if not allowed:
        return
    g = nx.Graph(graph)
    try:
        from networkx.algorithms.cycles import chordless_cycles as nx_chordless
    except ImportError:
        nx_chordless = None

    for nodes in nx.biconnected_components(g):
        if len(nodes) < 3:
            continue
        sub = g.subgraph(nodes)
        nn = len(nodes)
        ne = sub.number_of_edges()
        sizes = _ring_sizes_for_component(nn, ne, allowed)
        if not sizes:
            continue
        if nx_chordless is not None:
            max_s = max(sizes)
            for cycle in nx_chordless(sub, length_bound=max_s):
                n = len(cycle)
                if n in allowed and n in sizes:
                    yield cycle
            continue
        for size in sizes:
            yield from _chordless_cycles_bounded(sub, max_size=size, min_size=size)


def get_bond_types(bond_orders, *, bonds=None, ase_atoms=None):
    """
    Map continuous bond orders to single / double / triple labels.

    With ``bonds`` and ``ase_atoms``, applies distance-based aromatic tolerances
    when pymatgen bond orders sit on bin edges:

    * **single → double** if MIC distance ≤ aromatic C–C / C–N limits
    * **triple → double** if C–N distance ≤ ``_AROMATIC_CN_TRIPLE_MAX_DIST``
      (short aromatic C–N misclassified as triple)

    Bins: ``(0, 1]`` single, ``(1, 2]`` double, ``(2, 3]`` triple — same convention
    as Plotly framework coloring via :func:`sgraph_from_atoms`.

    :param bond_orders: a list of bond orders, e.g. [1.0, 2.0, 1.0, 3.0]
    :param bonds: optional ``covalent_bonds`` list; with ``ase_atoms``, enables
        distance-based promotion of borderline singles to double (aromatic
        framework bonds that pymatgen classifies as ~1.0 order).
    :param ase_atoms: ASE atoms for MIC distances when applying aromatic tolerance.
    :return: A pandas Categorical with bond types
    """
    import pandas as pd

    types = list(pd.cut(bond_orders, bins=[0, 1, 2, 3], labels=["single", "double", "triple"]))
    if bonds is not None and ase_atoms is not None:
        pos = ase_atoms.get_positions()
        cell = ase_atoms.get_cell()
        sym = ase_atoms.get_chemical_symbols()
        for i, bond in enumerate(bonds):
            a, b = int(bond[0]), int(bond[1])
            if types[i] == "single" and _distance_is_aromatic_framework(a, b, pos, cell, sym):
                types[i] = "double"
                continue
            if types[i] == "triple":
                sa, sb = sym[a], sym[b]
                if {sa, sb} == {"C", "N"}:
                    dist = _mic_bond_distance(a, b, pos, cell)
                    if dist <= _AROMATIC_CN_TRIPLE_MAX_DIST:
                        types[i] = "double"
    return pd.Categorical(types, categories=["single", "double", "triple"])


def find_chemically_unique_cycles(list_of_cycles, graph):
    """
    Return indices of chemically non-isomorphic cycles (element + bond_type).

    Incremental representatives per ring size (avoids O(n^2) pairwise isomorphism).
    """
    import networkx.algorithms.isomorphism as isomo
    from collections import defaultdict

    n = len(list_of_cycles)
    if n <= 1:
        return list(range(n))

    em = isomo.categorical_edge_match("bond_type", default="single")
    nm = isomo.categorical_node_match("element", default="None")

    by_size: dict[int, list[int]] = defaultdict(list)
    for idx, cycle in enumerate(list_of_cycles):
        by_size[len(cycle)].append(idx)

    keep: list[int] = []
    for indices in by_size.values():
        reps: list = []
        for idx in indices:
            sg = graph.subgraph(list_of_cycles[idx])
            if any(
                isomo.GraphMatcher(sg, rep_sg, node_match=nm, edge_match=em).is_isomorphic()
                for _, rep_sg in reps
            ):
                continue
            reps.append((idx, sg))
            keep.append(idx)
    return sorted(keep)
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

def _sgraph_to_rust_arrays(sgraph):
    """Convert networkx structure graph to flat arrays for Rust."""
    import networkx as nx
    n = sgraph.number_of_nodes()
    edge_i, edge_j, elements, bond_types = [], [], [], []
    for node in sgraph.nodes():
        elements.append(str(sgraph.nodes[node].get('specie', 'C')))
    for u, v, data in sgraph.edges(data=True):
        edge_i.append(int(u))
        edge_j.append(int(v))
        bond_types.append(str(data.get('bond_type', 'single')))
    return n, edge_i, edge_j, elements, bond_types


def _expand_aromatic_embeddings_rust(sgraph, chemically_unique_cycles: list) -> list:
    """Use Rust petgraph subgraph isomorphism to expand aromatic embeddings."""
    import mgr_rust
    n, ei, ej, elems, bts = _sgraph_to_rust_arrays(sgraph)
    templates = [[int(x) for x in cycle] for cycle in chemically_unique_cycles]
    return mgr_rust.expand_aromatic_cycle_embeddings(n, ei, ej, elems, bts, templates)


def canonicalize_duplicate_site_atoms(ase_atoms, *, frac_digits: int = 3):
    """Merge atoms with the same element and wrapped fractional coordinates.

    RASPA / symmetry-expanded CIFs often contain multiple atom records per
    crystallographic site. Those duplicates fragment the bond graph and cause
    aromatic cycle detection to miss rings (e.g. ZIF-8 imidazolates).

    Returns ``(merged_atoms, merged_to_original)`` where ``merged_atoms`` has
    remapped ``covalent_bonds`` / ``covalent_bond_orders`` and
    ``merged_to_original[i]`` is the source atom index in ``ase_atoms``.
    """
    import numpy as np
    from ase import Atoms

    n = len(ase_atoms)
    if n == 0:
        return ase_atoms.copy(), []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    frac = ase_atoms.get_scaled_positions(wrap=True)
    znums = ase_atoms.get_atomic_numbers()
    rounded = np.round(frac, frac_digits)
    for i in range(n):
        for j in range(i + 1, n):
            if znums[i] != znums[j]:
                continue
            if np.all(rounded[i] == rounded[j]):
                union(i, j)

    rep_by_root: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        if root not in rep_by_root or i < rep_by_root[root]:
            rep_by_root[root] = i
    keep = sorted(rep_by_root.values())
    if len(keep) == n:
        return ase_atoms.copy(), keep

    old_to_new = {old: new for new, old in enumerate(keep)}
    merged = ase_atoms[keep].copy()
    bonds_in = ase_atoms.info.get("covalent_bonds") or []
    orders_in = ase_atoms.info.get("covalent_bond_orders") or []
    bonds_out: list[list[int]] = []
    orders_out: list[float] = []
    seen: set[tuple[int, int]] = set()
    for idx, (a, b) in enumerate(bonds_in):
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            continue
        a2 = old_to_new[rep_by_root[ra]]
        b2 = old_to_new[rep_by_root[rb]]
        edge = (min(a2, b2), max(a2, b2))
        if edge in seen:
            continue
        seen.add(edge)
        bonds_out.append([edge[0], edge[1]])
        if idx < len(orders_in):
            orders_out.append(float(orders_in[idx]))
        else:
            orders_out.append(1.0)
    merged.info["covalent_bonds"] = bonds_out
    merged.info["covalent_bond_orders"] = orders_out
    return merged, keep


# Max MIC covalent bond length (Å) for local aromatic cycle detection.
_LOCAL_AROMATIC_MAX_MIC_BOND = 2.5
# Max unwrapped ring span (Å) — imidazole/benzene are ~2–3 Å across.
_LOCAL_AROMATIC_MAX_UNWRAPPED_SPAN = 4.5


def filter_sgraph_to_local_bonds(
    sgraph,
    positions,
    cell,
    *,
    max_mic_distance: float = _LOCAL_AROMATIC_MAX_MIC_BOND,
):
    """Drop graph edges whose minimum-image distance exceeds ``max_mic_distance``."""
    import copy

    import numpy as np
    from ase.geometry import get_distances

    local = copy.copy(sgraph)
    pos = np.asarray(positions, dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    pbc = bool(np.any(cell_arr))
    to_remove = []
    for u, v in sgraph.edges():
        if pbc:
            mic = float(np.asarray(get_distances(pos[u], pos[v], cell=cell_arr, pbc=True)[1]).ravel()[0])
        else:
            mic = float(np.linalg.norm(pos[u] - pos[v]))
        if mic > max_mic_distance:
            to_remove.append((u, v))
    local.remove_edges_from(to_remove)
    return local


def unwrap_aromatic_cycle_positions(cycle, positions, cell):
    """Return Cartesian positions for a ring, unwrapping across periodic boundaries."""
    import numpy as np
    from ase.geometry import find_mic

    pos = np.asarray(positions, dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    pbc = bool(np.any(cell_arr))
    out = [pos[cycle[0]].copy()]
    if not pbc:
        return pos[np.asarray(cycle, dtype=int)]

    for idx in range(1, len(cycle)):
        a, b = cycle[idx - 1], cycle[idx]
        vec = find_mic(pos[b] - pos[a], cell_arr, pbc=True)[0]
        out.append(out[-1] + vec)
    return np.asarray(out, dtype=float)


def is_local_aromatic_cycle(
    cycle,
    positions,
    cell,
    *,
    max_mic_bond: float = _LOCAL_AROMATIC_MAX_MIC_BOND,
    max_unwrapped_span: float = _LOCAL_AROMATIC_MAX_UNWRAPPED_SPAN,
):
    """True when consecutive ring bonds are short and the unwrapped ring is compact."""
    import numpy as np
    from ase.geometry import get_distances, find_mic

    if len(cycle) < 3:
        return False
    pos = np.asarray(positions, dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    pbc = bool(np.any(cell_arr))

    unwrapped = unwrap_aromatic_cycle_positions(cycle, pos, cell_arr)
    span = float(
        max(
            np.linalg.norm(unwrapped[i] - unwrapped[j])
            for i in range(len(cycle))
            for j in range(i + 1, len(cycle))
        )
    )
    if span > max_unwrapped_span:
        return False

    for i in range(len(cycle)):
        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
        if pbc:
            bond_len = float(np.asarray(get_distances(pos[a], pos[b], cell=cell_arr, pbc=True)[1]).ravel()[0])
        else:
            bond_len = float(np.linalg.norm(pos[a] - pos[b]))
        if bond_len > max_mic_bond:
            return False
        if pbc:
            step = float(np.linalg.norm(find_mic(pos[b] - pos[a], cell_arr, pbc=True)[0]))
            if step > max_mic_bond:
                return False
    return True


def filter_local_aromatic_cycles(
    cycles,
    positions,
    cell,
    *,
    max_mic_bond: float = _LOCAL_AROMATIC_MAX_MIC_BOND,
    max_unwrapped_span: float = _LOCAL_AROMATIC_MAX_UNWRAPPED_SPAN,
):
    """Keep only aromatic cycles that are local (not PBC-spanning artifacts)."""
    kept = []
    for cycle in cycles:
        if is_local_aromatic_cycle(
            cycle,
            positions,
            cell,
            max_mic_bond=max_mic_bond,
            max_unwrapped_span=max_unwrapped_span,
        ):
            kept.append(cycle)
    return kept


# Max MIC length (Å) for optional aromatic closure edges (site-image bond gaps).
_AROMATIC_MIC_CLOSURE_MAX = 1.5
# Max augmented (MIC-only) edges allowed per detected ring.
_AROMATIC_MAX_CLOSURE_EDGES = 1


def _build_aromatic_search_graph(ase_atoms, *, mic_closure_max: float = _AROMATIC_MIC_CLOSURE_MAX):
    """C/N subgraph with optional MIC closure edges for missing aromatic bonds."""
    import numpy as np
    from ase.geometry import get_distances
    from ase.neighborlist import neighbor_list

    sym = ase_atoms.get_chemical_symbols()
    pos = ase_atoms.get_positions()
    cell = ase_atoms.get_cell()
    pbc = bool(np.any(ase_atoms.get_pbc()))
    cn_nodes = [i for i, s in enumerate(sym) if s in "CN"]

    base = sgraph_from_atoms(ase_atoms)
    bond_types = __import__("networkx").get_edge_attributes(base, "bond_type")

    import networkx as nx

    graph = base.subgraph(cn_nodes).copy()
    aug_edges: set[tuple[int, int]] = set()

    if mic_closure_max > 0 and cn_nodes:
        if pbc:
            i_idx, j_idx, dists = neighbor_list(
                "ijd", ase_atoms, cutoff=float(mic_closure_max), self_interaction=False,
            )
            pairs = zip(i_idx.tolist(), j_idx.tolist(), dists.tolist())
        else:
            pairs = []
            for a in cn_nodes:
                for b in cn_nodes:
                    if b <= a:
                        continue
                    d = float(np.linalg.norm(pos[a] - pos[b]))
                    if d <= mic_closure_max:
                        pairs.append((a, b, d))

        for a, b, _dist in pairs:
            if sym[a] not in "CN" or sym[b] not in "CN":
                continue
            if graph.has_edge(a, b):
                continue
            if pbc:
                mic = float(
                    np.asarray(get_distances(pos[a], pos[b], cell=cell, pbc=True)[1]).ravel()[0]
                )
                if mic > mic_closure_max:
                    continue
            graph.add_edge(a, b, bond_type="double", mic_closure=True)
            aug_edges.add(tuple(sorted((int(a), int(b)))))

    return graph, bond_types, aug_edges


def _is_valid_aromatic_cycle(
    cycle,
    graph,
    bond_types,
    aug_edges,
    positions,
    cell,
    symbols,
    *,
    max_closure_edges: int = _AROMATIC_MAX_CLOSURE_EDGES,
):
    """Aromatic ring candidate: double bonds + <=N MIC closures + compact local geometry."""
    n = len(cycle)
    if n < 3:
        return False
    closure = 0
    for i in range(n):
        a, b = int(cycle[i]), int(cycle[(i + 1) % n])
        edge = tuple(sorted((a, b)))
        if not graph.has_edge(a, b):
            return False
        if edge in aug_edges or graph.edges[a, b].get("mic_closure"):
            closure += 1
            continue
        if not _is_aromatic_framework_edge(
            a, b, bond_types, positions, cell, symbols,
            graph=graph, aug_edges=aug_edges,
        ):
            return False
    if closure > max_closure_edges:
        return False
    return is_local_aromatic_cycle(cycle, positions, cell)


def find_aromatic_cycles_in_unit_cell(
    ase_atoms,
    default_ring_sizes=None,
    *,
    mic_closure_max: float = _AROMATIC_MIC_CLOSURE_MAX,
    max_closure_edges: int = _AROMATIC_MAX_CLOSURE_EDGES,
):
    """Detect local aromatic rings on every atom in the unit cell.

  General detector for small aromatic rings (default sizes 5–9: imidazole,
  benzene, pyridine, and related MOF linkers). Uses covalent bond orders from
  ``ase_atoms.info`` (call :func:`update_with_covalent_bonds` first).

  Bond graph edges marked aromatic/double define the ring framework. When
  periodic-image duplicate sites leave a single missing short C–C/C–N contact,
  at most one MIC closure edge (≤ ``mic_closure_max`` Å) may complete the
  cycle. Rings must be compact after PBC unwrapping (not cell-spanning).

  **Bond-length tolerances:** continuous bond orders from pymatgen are binned
  into single/double/triple; borderline contacts use MIC distance cutoffs for
  aromatic C–C and C–N (see module comments on ``_AROMATIC_*_MAX_DIST`` in
  this file and ``docs/aromatic_ring_detection.qmd``). Used by Plotly
  (:func:`~mofography.visualize.plot_aromatic_rings`) and Three.js
  (:func:`~mofography.movies.make_aromatic_rings_trace`) ring highlights.

  Inputs: atomic positions, elements, cell/PBC, and ``covalent_bonds`` /
  ``covalent_bond_orders`` in ``ase_atoms.info``.
    """
    try:
        from networkx.algorithms.cycles import chordless_cycles as _chordless_cycles
    except ImportError:
        _chordless_cycles = None  # noqa: F841 — fallback via _iter_chordless_cycles

    if default_ring_sizes is None:
        default_ring_sizes = [5, 6, 7, 8, 9]

    sym = ase_atoms.get_chemical_symbols()
    pos = ase_atoms.get_positions()
    cell = ase_atoms.get_cell()
    graph, bond_types, aug_edges = _build_aromatic_search_graph(
        ase_atoms, mic_closure_max=mic_closure_max,
    )

    kept: set[tuple[int, ...]] = set()
    allowed_sizes = set(default_ring_sizes)
    for cycle in _iter_chordless_cycles(graph, default_ring_sizes):
        n = len(cycle)
        if n not in allowed_sizes:
            continue
        if not _is_valid_aromatic_cycle(
            cycle, graph, bond_types, aug_edges, pos, cell, sym,
            max_closure_edges=max_closure_edges,
        ):
            continue
        kept.add(tuple(sorted(int(i) for i in cycle)))
    return [list(c) for c in sorted(kept)]


def find_imidazole_rings_in_unit_cell(ase_atoms):
    """Deprecated alias — use :func:`find_aromatic_cycles_in_unit_cell`."""
    return find_aromatic_cycles_in_unit_cell(ase_atoms, default_ring_sizes=[5])


def expand_aromatic_cycles_to_unit_cell(ase_atoms, canonical_cycles):
    """Deprecated — use :func:`find_aromatic_cycles_in_unit_cell`."""
    del canonical_cycles
    return find_aromatic_cycles_in_unit_cell(ase_atoms)


def cache_aromatic_cycles_if_needed(ase_atoms, *, show_rings: bool) -> None:
    """Compute aromatic cycles once and cache on ``ase_atoms.info`` (Plotly + Three.js)."""
    if not show_rings or ase_atoms is None:
        return
    compute_aromatic_cycles(ase_atoms)


def compute_aromatic_cycles(
    ase_atoms,
    *,
    force_recompute: bool = False,
    expand_isomorphisms: bool = True,
    default_ring_sizes=None,
    merge_duplicate_sites: bool = True,
    expand_to_unit_cell: bool = True,
):
    """Detect aromatic cycles and cache the result in ``ase_atoms.info``.

    1. Ensures covalent bonds are present (calls
       :func:`update_with_covalent_bonds` if needed).
    2. Returns the cached ``ase_atoms.info["aromatic_cycles"]`` when
       ``force_recompute=False`` and the cache key matches.
    3. Otherwise detects rings (unit-cell chordless search by default), stores the
       result in ``ase_atoms.info["aromatic_cycles"]``, and returns it.

    When ``merge_duplicate_sites`` is True (default), atoms that share the
    same element and wrapped fractional coordinates are merged before graph
    construction. Cycle atom indices are mapped back to the input structure.

    When ``expand_to_unit_cell`` is True (default), rings are found on the
    full structure with :func:`find_aromatic_cycles_in_unit_cell` (sizes
    ``default_ring_sizes``, default 5–9) so every distinct linker in the cell
    is highlighted. When False, uses graph cycle detection on a merged or
    full structure (representative atoms only for merged sites).

    Used by :func:`~mgr.visualize.plot_aromatic_rings` (Plotly) and
    :func:`~mgr.movies.make_aromatic_rings_trace` (Three.js) when a
    pre-computed cycle list is not supplied.
    """
    if "covalent_bonds" not in ase_atoms.info:
        ase_atoms = update_with_covalent_bonds(ase_atoms)

    cache_opts = {
        "expand_isomorphisms": expand_isomorphisms,
        "default_ring_sizes": default_ring_sizes,
        "merge_duplicate_sites": merge_duplicate_sites,
        "expand_to_unit_cell": expand_to_unit_cell,
    }
    key = _aromatic_cache_key(ase_atoms, **cache_opts)
    if (
        not force_recompute
        and ase_atoms.info.get("aromatic_cycles_key") == key
        and ase_atoms.info.get("aromatic_cycles") is not None
    ):
        return ase_atoms.info["aromatic_cycles"]

    kw = {} if default_ring_sizes is None else {"default_ring_sizes": default_ring_sizes}
    ring_sizes = default_ring_sizes if default_ring_sizes is not None else [5, 6, 7, 8, 9]

    if expand_to_unit_cell:
        cycles = find_aromatic_cycles_in_unit_cell(
            ase_atoms,
            default_ring_sizes=ring_sizes,
        )
    elif merge_duplicate_sites:
        merged_atoms, merged_to_original = canonicalize_duplicate_site_atoms(ase_atoms)
        sgraph = sgraph_from_atoms(merged_atoms)
        sgraph = filter_sgraph_to_local_bonds(
            sgraph,
            merged_atoms.get_positions(),
            merged_atoms.get_cell(),
        )
        merged_cycles = find_aromatic_cycles_in_sgraph(sgraph, **kw)
        merged_cycles = filter_local_aromatic_cycles(
            merged_cycles,
            merged_atoms.get_positions(),
            merged_atoms.get_cell(),
        )
        cycles = [
            [merged_to_original[i] for i in cycle]
            for cycle in merged_cycles
        ]
    else:
        sgraph = sgraph_from_atoms(ase_atoms)
        sgraph = filter_sgraph_to_local_bonds(
            sgraph,
            ase_atoms.get_positions(),
            ase_atoms.get_cell(),
        )
        cycles = find_aromatic_cycles_in_sgraph(sgraph, **kw)
        cycles = filter_local_aromatic_cycles(
            cycles,
            ase_atoms.get_positions(),
            ase_atoms.get_cell(),
        )
    ase_atoms.info["aromatic_cycles"] = cycles
    ase_atoms.info["aromatic_cycles_key"] = key
    return cycles

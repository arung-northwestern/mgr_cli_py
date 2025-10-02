# """
#  Compute functions for mofography
#   ██████╗ ██████╗ ███╗   ███╗██████╗ ██╗   ██╗████████╗███████╗
#  ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║   ██║╚══██╔══╝██╔════╝
#  ██║     ██║   ██║██╔████╔██║██████╔╝██║   ██║   ██║   █████╗
#  ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║   ██║   ██║   ██╔══╝
#  ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ╚██████╔╝   ██║   ███████╗
#   ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝      ╚═════╝    ╚═╝   ╚══════╝


# """
# TODO Identify the different unique regions
# TODO Fix the issue with  the border regions getting cut off

import numpy as np 
import ase
import logging
logger = logging.getLogger(__name__)

def gpd_aabb(grid_points, ase_atoms, radii, n_neighbors=10, probe_radius=0.0):
    """This subroutine computes the shortest distance to the framework surface for any grid point using the Axis Aligned
    Bounding Box method  as implemented by the Glotzer group from Univ. of Michigan.

    :param grid_points: (N,3) array of the cartesian coordinates of the grid points
    :type grid_points: numpy array of floats
    :param ase_atoms: ase atoms object containing the framework information
    :type ase_atoms: ase.atoms obj
    :param n_neighbors: number of neighbors of each grid point checked for the shortest dist to surface, defaults to 10
    :type n_neighbors: int, optional
    :param probe_radius: radius of the distance grid calculation probe, defaults to 0.0
    :type probe_radius: float, optional
    :return: (N,1) array of the shortest distance corresponding to the N grid points.
    :rtype: numpy array of floats
    """
    from freud.locality import AABBQuery
    from freud.box import Box
    import ase

    # from ase.data import vdw_radii
    from icecream import ic
    import pandas as pd
    import numpy as np

    framework_atom_positions = ase_atoms.get_positions()
    cell = ase.geometry.complete_cell(
        ase_atoms.get_cell()
    ).T  # this needs to be an upper triangular matrix, essential column vectors, not row vectors

    box = Box.from_matrix(cell, dimensions=3)  # Define the box for Freud
    ab = AABBQuery(box=box, points=framework_atom_positions)  # Compute the AABB Tree
    # Perform the nearest neighbor query with n-neighbors (10 by default)
    ab_r = ab.query(
        grid_points,
        query_args=dict(mode="nearest", num_neighbors=n_neighbors, exclude_ii=True),
    )
    ab_r_nl = ab_r.toNeighborList(
        sort_by_distance=True
    )  # convert query to neighborlist
    nls = ab_r_nl.point_indices.reshape(
        -1, n_neighbors
    )  # get the indices of the neighbors for each grid pt. to lookup the radii later
    dists = ab_r_nl.distances.reshape(
        -1, n_neighbors
    )  # get the distances to the neighbors of each grid pt.
    # print(ab, ab_r, ab_r_nl, nls, [radii[nl] for nl in nls])#, np.min(dists - [radii[nl] for nl in nls],axis=1))
    # return the minimum (shortest distance to the surface) in each row
    return np.min(dists - [radii[nl] for nl in nls], axis=1)


def dask_grid_over_atoms(ase_atoms, spacing=0.1, chunksize=50000):
    """Generate a dask array of grid points over an ase atoms object with a specified grid spacing

    :param ase_atoms: ASE atoms object
    :type ase_atoms: ase.atoms obj
    :param spacing: grid fineness in angstroms, defaults to 0.1
    :type spacing: float, optional
    :param chunksize: number of grid points in each block of the dask array, defaults to 50000
    :type chunksize: int, optional
    :return: dask array of size (N_grid_points, 3) with each block of size 'chunksize'.
    :rtype: dask.array
    """
    import ase
    import dask.array as da
    import numpy as np

    # number of grid points in each direction
    [nx, ny, nz] = (ase_atoms.get_cell_lengths_and_angles()[0:3] / spacing).astype(
        int
    ) + 1
    gpoints = (
        da.stack(
            da.meshgrid(
                np.linspace(0, 1, nx),
                np.linspace(0, 1, ny),
                np.linspace(0, 1, nz),
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 3)
    ).rechunk(chunksize, 3)
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T  # cell matrix
    return da.dot(cell, gpoints.T).T, (nx, ny, nz)  # return the actual coordinates

def dgrid_from_atoms_cpu_no_aabb(ase_atoms, radii=None, spacing=0.2, block_size=20000):
    """
    The function `dgrid_from_atoms_no_aabb` calculates a distance grid based on the cpu but without using 
    an AABB cell list to accelerate the calculation. Although substantially slower, this function might provide
    better smoothness of the distance grid, especially matrials with small pores. 

    :param ase_atoms: The `ase_atoms` parameter in the `dgrid_from_atoms_no_aabb` function is expected to be
    an ASE Atoms object, which represents a collection of atoms and their properties. This object
    typically contains information such as atomic positions, unit cell parameters, atomic symbols, and
    other properties of the atoms in
    :param radii: The `radii` parameter in the `dgrid_from_atoms_no_aabb` function represents the van der
    Waals radii of the atoms in the atomic structure provided as input. These radii are used in the
    calculation of the distance grid. If the `radii` parameter is not provided by the
    :param spacing: The `spacing` parameter in the `dgrid_from_atoms_no_aabb` function represents the
    distance between grid points in the generated grid. It determines the resolution of the grid where
    the distances from atoms to grid points are calculated. A smaller spacing value will result in a
    higher resolution grid with more grid points,
    :param block_size: The `block_size` parameter in the `dgrid_from_atoms_no_aabb` function determines the
    size of each block used for processing the data. In this function, it is used for chunking the data
    and processing it in parallel on the CPU. A larger `block_size` can be more efficient but too big can lead to memory issues, defaults
    to 10000 (optional)
    :return: The function `dgrid_from_atoms_no_aabb` returns a 3D numpy array representing a distance grid
    computed based on the input atomic structure provided in the form of ASE atoms. The distance grid is
    calculated by computing the minimum distance between each point on the grid and the atoms in the
    structure, taking into account the atomic radii.
    """

    from ase.io import read
    import numpy as np
    import dask.array as da
    # import cupy as cp
    import pandas as pd
    from tqdm import tqdm
    import ase

    # data    = read(path_to_cif)
    fpoints = (ase_atoms.get_scaled_positions() - 0.5)
    cell =  (ase.geometry.complete_cell(ase_atoms.get_cell())).T
    # from ase.data import vdw_radii
    # radius_series = pd.Series({'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
    #                         'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66,
    #                         'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
    #                         'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
    #                         'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
    #                         'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    #                         'Mn': 1.50, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24,
    #                         'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
    #                         'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    #                         'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
    #                         'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
    #                         'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    #                         'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
    #                         'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
    #                         'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
    #                         'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
    #                         'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
    #                         'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
    #                         'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
    #                         'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
    #                         'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
    #                         'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21,
    #                         'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96,
    #                         'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69})
    # radii = cp.asarray(radius_series[data.get_chemical_symbols()].values).reshape(-1, 1)
    if radii is None:
        vdw_radii = pd.Series(
            {
                "H": 110.00000000000001,
                "He": 140.0,
                "Li": 182.0,
                "Be": 153.0,
                "B": 192.0,
                "C": 170.0,
                "N": 155.0,
                "O": 152.0,
                "F": 147.0,
                "Ne": 154.0,
                "Na": 227.0,
                "Mg": 173.0,
                "Al": 184.0,
                "Si": 210.0,
                "P": 180.0,
                "S": 180.0,
                "Cl": 175.0,
                "Ar": 188.0,
                "K": 275.0,
                "Ca": 231.0,
                "Sc": 215.0,
                "Ti": 211.0,
                "V": 206.99999999999997,
                "Cr": 206.0,
                "Mn": 204.99999999999997,
                "Fe": 204.0,
                "Co": 200.0,
                "Ni": 197.0,
                "Cu": 196.0,
                "Zn": 200.99999999999997,
                "Ga": 187.0,
                "Ge": 211.0,
                "As": 185.0,
                "Se": 190.0,
                "Br": 185.0,
                "Kr": 202.0,
                "Rb": 303.0,
                "Sr": 249.00000000000003,
                "Y": 231.99999999999997,
                "Zr": 223.0,
                "Nb": 218.00000000000003,
                "Mo": 217.0,
                "Tc": 216.0,
                "Ru": 213.0,
                "Rh": 210.0,
                "Pd": 210.0,
                "Ag": 211.0,
                "Cd": 218.00000000000003,
                "In": 193.0,
                "Sn": 217.0,
                "Sb": 206.0,
                "Te": 206.0,
                "I": 198.0,
                "Xe": 216.0,
                "Cs": 343.0,
                "Ba": 268.0,
                "La": 243.00000000000003,
                "Ce": 242.0,
                "Pr": 240.0,
                "Nd": 239.0,
                "Pm": 238.0,
                "Sm": 236.0,
                "Eu": 235.0,
                "Gd": 234.0,
                "Tb": 233.0,
                "Dy": 231.0,
                "Ho": 229.99999999999997,
                "Er": 229.0,
                "Tm": 227.0,
                "Yb": 225.99999999999997,
                "Lu": 224.00000000000003,
                "Hf": 223.0,
                "Ta": 222.00000000000003,
                "W": 218.00000000000003,
                "Re": 216.0,
                "Os": 216.0,
                "Ir": 213.0,
                "Pt": 213.0,
                "Au": 214.0,
                "Hg": 223.0,
                "Tl": 196.0,
                "Pb": 202.0,
                "Bi": 206.99999999999997,
                "Po": 197.0,
                "At": 202.0,
                "Rn": 220.00000000000003,
                "Fr": 348.0,
                "Ra": 283.0,
                "Ac": 247.00000000000003,
                "Th": 245.00000000000003,
                "Pa": 243.00000000000003,
                "U": 241.0,
                "Np": 239.0,
                "Pu": 243.00000000000003,
                "Am": 244.0,
                "Cm": 245.00000000000003,
                "Bk": 244.0,
                "Cf": 245.00000000000003,
                "Es": 245.00000000000003,
                "Fm": 245.00000000000003,
                "Md": 246.0,
                "No": 246.0,
                "Lr": 246.0,
                "Rf": np.nan,
                "Db": np.nan,
                "Sg": np.nan,
                "Bh": np.nan,
                "Hs": np.nan,
                "Mt": np.nan,
                "Ds": np.nan,
                "Rg": np.nan,
                "Cn": np.nan,
                "Nh": np.nan,
                "Fl": np.nan,
                "Mc": np.nan,
                "Lv": np.nan,
                "Ts": np.nan,
                "Og": np.nan,
            }
        )
        radii = (
            vdw_radii[ase_atoms.get_chemical_symbols()].values / 100.0
        ).reshape(-1, 1)

    # * the vdw radii mendeleev is in picometers
    # radii = cp.asarray(vdw_radii[ase_atoms.get_atomic_numbers()].values/100).reshape(-1, 1)
    [nx, ny, nz] = (ase_atoms.cell.cellpar()[0:3] / spacing).astype(
        int
    ) + 1
    #     print([nx,ny,nz])
    gpoints = (
        da.stack(
            da.meshgrid(
                np.linspace(-0.5, 0.5, nx),
                np.linspace(-0.5, 0.5, ny),
                np.linspace(-0.5, 0.5, nz),
                indexing="ij",
            ),
            -1,
        )
        .reshape(-1, 3)
        .rechunk(block_size, 3)
    )

    #     def gpd(points, fpoints=fpoints, cell=cell, radii=radii):
    #         points = cp.asarray(points)
    #         return cp.min(cp.linalg.norm(cp.dot(cell, (cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0)-cp.around(cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0))).reshape(-1,3).T).T, axis=1).reshape(fpoints.shape[0],-1)-radii, axis=0)

    def gpd_np(points, fpoints=fpoints, cell=cell, radii=radii):
        # points = cp.asarray(points)
        # fpoints = cp.asarray(fpoints)
        # radii = cp.asarray(radii)
        diff = np.expand_dims(points, axis=1) - np.expand_dims(fpoints, axis=0)
        diff = (diff - np.around(diff)).reshape(-1, 3)
        diff = np.dot(cell, diff.T).T
        diff = np.linalg.norm(diff, axis=1).reshape(-1, fpoints.shape[0]) - radii.T
        return np.min(diff, axis=1)

    dgrid = gpoints.map_blocks(gpd_np, chunks=(block_size, ), dtype="float32", drop_axis=1)
    return dgrid.compute().reshape(nx, ny, nz)
  
def dgrid_from_atoms(
    ase_atoms,
    radii=None,
    spacing=0.25,
    block_size=50000,
    n_neighbors=10,
    probe_radius=0,
):
    """
    The function `dgrid_from_atoms` generates a distance grid over atoms using Dask arrays and computes
    the grid based on atomic radii and specified parameters.

    :param ase_atoms: The `ase_atoms` parameter in the `dgrid_from_atoms` function is expected to be an
    ASE Atoms object, which represents a collection of atoms and their properties. This object typically
    contains information about the atomic positions, chemical symbols, and unit cell of a molecular
    system. You can create an
    :param radii: The `radii` parameter in the `dgrid_from_atoms` function is used to specify the atomic
    radii for the atoms in the input `ase_atoms`. If the `radii` parameter is not provided by the user,
    a default set of van der Waals radii for different elements
    :param spacing: The `spacing` parameter in the `dgrid_from_atoms` function represents the distance
    between grid points in the mesh grid that is created over the atoms. It determines how finely the
    space is divided for computing the distance grid. A smaller spacing value will result in a denser
    grid with more grid points
    :param block_size: The `block_size` parameter in the `dgrid_from_atoms` function is used to specify
    the size of each block when processing the grid points. This parameter determines how many grid
    points are processed at a time, which can impact memory usage and computational efficiency.
    Adjusting the `block_size` can, defaults to 50000 (optional)
    :param n_neighbors: The `n_neighbors` parameter in the `dgrid_from_atoms` function represents the
    number of nearest neighbors to consider when computing the distance grid for each point in the grid.
    This parameter determines how many neighboring atoms will influence the calculation of distances in
    the grid. Increasing the `n_neighbors` value will, defaults to 10 (optional)
    :param probe_radius: The `probe_radius` parameter in the `dgrid_from_atoms` function represents the
    radius of the probe used in the calculation. This parameter is used in the computation of the
    distance grid over atoms. The probe radius is typically used in molecular modeling and simulations
    to mimic the size of a solvent molecule or, defaults to 0 (optional)
    :return: The function `dgrid_from_atoms` returns a Dask array representing the distance grid
    computed over the input ASE atoms. The distance grid is computed based on the provided parameters
    such as radii, spacing, block size, number of neighbors, and probe radius. The function utilizes
    Dask for parallel computation and returns the distance grid reshaped to the correct shape specified
    by the input ASE atoms.
    """

    import numpy as np
    import dask.array as da
    from rich.progress import track
    from tqdm import tqdm

    gpoints, shape = dask_grid_over_atoms(
        ase_atoms, spacing=spacing, chunksize=block_size
    )  # mesh grid
    dgrid = []
    import pandas as pd

    if radii is None:
        vdw_radii = pd.Series(
            {
                "H": 110.00000000000001,
                "He": 140.0,
                "Li": 182.0,
                "Be": 153.0,
                "B": 192.0,
                "C": 170.0,
                "N": 155.0,
                "O": 152.0,
                "F": 147.0,
                "Ne": 154.0,
                "Na": 227.0,
                "Mg": 173.0,
                "Al": 184.0,
                "Si": 210.0,
                "P": 180.0,
                "S": 180.0,
                "Cl": 175.0,
                "Ar": 188.0,
                "K": 275.0,
                "Ca": 231.0,
                "Sc": 215.0,
                "Ti": 211.0,
                "V": 206.99999999999997,
                "Cr": 206.0,
                "Mn": 204.99999999999997,
                "Fe": 204.0,
                "Co": 200.0,
                "Ni": 197.0,
                "Cu": 196.0,
                "Zn": 200.99999999999997,
                "Ga": 187.0,
                "Ge": 211.0,
                "As": 185.0,
                "Se": 190.0,
                "Br": 185.0,
                "Kr": 202.0,
                "Rb": 303.0,
                "Sr": 249.00000000000003,
                "Y": 231.99999999999997,
                "Zr": 223.0,
                "Nb": 218.00000000000003,
                "Mo": 217.0,
                "Tc": 216.0,
                "Ru": 213.0,
                "Rh": 210.0,
                "Pd": 210.0,
                "Ag": 211.0,
              "Cd": 218.00000000000003,
                "In": 193.0,
                "Sn": 217.0,
                "Sb": 206.0,
                "Te": 206.0,
                "I": 198.0,
                "Xe": 216.0,
                "Cs": 343.0,
                "Ba": 268.0,
                "La": 243.00000000000003,
                "Ce": 242.0,
                "Pr": 240.0,
                "Nd": 239.0,
                "Pm": 238.0,
                "Sm": 236.0,
                "Eu": 235.0,
                "Gd": 234.0,
                "Tb": 233.0,
                "Dy": 231.0,
                "Ho": 229.99999999999997,
                "Er": 229.0,
                "Tm": 227.0,
                "Yb": 225.99999999999997,
                "Lu": 224.00000000000003,
                "Hf": 223.0,
                "Ta": 222.00000000000003,
                "W": 218.00000000000003,
                "Re": 216.0,
                "Os": 216.0,
                "Ir": 213.0,
                "Pt": 213.0,
                "Au": 214.0,
                "Hg": 223.0,
                "Tl": 196.0,
                "Pb": 202.0,
                "Bi": 206.99999999999997,
                "Po": 197.0,
                "At": 202.0,
                "Rn": 220.00000000000003,
                "Fr": 348.0,
                "Ra": 283.0,
                "Ac": 247.00000000000003,
                "Th": 245.00000000000003,
                "Pa": 243.00000000000003,
                "U": 241.0,
                "Np": 239.0,
                "Pu": 243.00000000000003,
                "Am": 244.0,
                "Cm": 245.00000000000003,
                "Bk": 244.0,
                "Cf": 245.00000000000003,
                "Es": 245.00000000000003,
                "Fm": 245.00000000000003,
                "Md": 246.0,
                "No": 246.0,
                "Lr": 246.0,
                "Rf": np.nan,
                "Db": np.nan,
                "Sg": np.nan,
                "Bh": np.nan,
                "Hs": np.nan,
                "Mt": np.nan,
                "Ds": np.nan,
                "Rg": np.nan,
                "Cn": np.nan,
                "Nh": np.nan,
                "Fl": np.nan,
                "Mc": np.nan,
                "Lv": np.nan,
                "Ts": np.nan,
                "Og": np.nan,
            }
        )
        radii = vdw_radii[ase_atoms.get_chemical_symbols()].values / 100.0

    # for block in track(gpoints.blocks, description='Computing distance grid', total=gpoints.npartitions):
    for block in tqdm(
        gpoints.blocks, desc="Computing distance grid", total=gpoints.npartitions
    ):
        dgrid.append(
            gpd_aabb(
                block.compute(),
                ase_atoms,
                radii,
                n_neighbors=n_neighbors,
                probe_radius=probe_radius,
            )
        )  # compute grid
    return (
        da.hstack(dgrid).rechunk(chunks=block_size).reshape(shape)
    )  # return in correct shape


def dgrid_from_atoms_cpu(
    ase_atoms,
    radii=None,
    spacing=0.25,
    block_size=50000,
    n_neighbors=10,
    probe_radius=0,
    precompute=True,
):
    """
    This function generates a distance grid over atoms using Dask arrays and pandas, with an option to
    precompute the grid.

    :param ase_atoms: The `ase_atoms` parameter in the `dgrid_from_atoms_cpu` function is expected to be
    an ASE Atoms object, which represents a collection of atoms and their positions in space. This
    object typically contains information about the atomic species, positions, and unit cell of a
    molecular system
    :param radii: The `radii` parameter in the `dgrid_from_atoms_cpu` function is used to specify the
    atomic radii for the atoms in the input `ase_atoms` object. If the `radii` parameter is not provided
    by the user, a default set of van der Waals radii
    :param spacing: The `spacing` parameter in the `dgrid_from_atoms_cpu` function represents the
    distance between grid points in the mesh grid that is created over the atoms. It determines how
    finely the grid is sampled in space. A smaller spacing value will result in a denser grid with more
    grid points, while
    :param block_size: The `block_size` parameter in the `dgrid_from_atoms_cpu` function is used to
    specify the size of each block when processing the grid points. It determines how the grid points
    are divided into smaller blocks for computation. A larger block size can lead to better performance
    in terms of memory usage and, defaults to 50000 (optional)
    :param n_neighbors: The `n_neighbors` parameter in the `dgrid_from_atoms_cpu` function represents
    the number of nearest neighbors to consider when computing the distance grid. It is used in the
    calculation of the distance grid based on the provided atoms and radii. Increasing the `n_neighbors`
    value will result in a, defaults to 10 (optional)
    :param probe_radius: The `probe_radius` parameter in the `dgrid_from_atoms_cpu` function represents
    the radius of the probe used for computing the distance grid. It is used in conjunction with the
    atomic radii to determine the distance between atoms and the grid points. The probe radius helps in
    defining the spatial constraints for, defaults to 0 (optional)
    :param precompute: The `precompute` parameter in the `dgrid_from_atoms_cpu` function determines
    whether the computation of the distance grid should be immediately executed and the final result
    returned (`precompute=True`), or if a lazy Dask array representing the computation should be
    returned without executing it (`precompute=False`), defaults to True (optional)
    :return: The function `dgrid_from_atoms_cpu` returns the computed distance grid. If `precompute` is
    set to True, it returns the computed distance grid as a Dask array. If `precompute` is set to False,
    it returns the Dask array without computing it, allowing for further computations to be performed
    later.
    """

    import numpy as np
    import dask.array as da
    from rich.progress import track
    from tqdm import tqdm

    # import mofography as mgr
    gpoints, shape = dask_grid_over_atoms(
        ase_atoms, spacing=spacing, chunksize=block_size
    )  # mesh grid
    dgrid = []
    import pandas as pd

    if radii is None:
        vdw_radii = pd.Series(
            {
                "H": 110.00000000000001,
                "He": 140.0,
                "Li": 182.0,
                "Be": 153.0,
                "B": 192.0,
                "C": 170.0,
                "N": 155.0,
                "O": 152.0,
                "F": 147.0,
                "Ne": 154.0,
                "Na": 227.0,
                "Mg": 173.0,
                "Al": 184.0,
                "Si": 210.0,
                "P": 180.0,
                "S": 180.0,
                "Cl": 175.0,
                "Ar": 188.0,
                "K": 275.0,
                "Ca": 231.0,
                "Sc": 215.0,
                "Ti": 211.0,
                "V": 206.99999999999997,
                "Cr": 206.0,
                "Mn": 204.99999999999997,
                "Fe": 204.0,
                "Co": 200.0,
                "Ni": 197.0,
                "Cu": 196.0,
                "Zn": 200.99999999999997,
                "Ga": 187.0,
                "Ge": 211.0,
                "As": 185.0,
                "Se": 190.0,
                "Br": 185.0,
                "Kr": 202.0,
                "Rb": 303.0,
                "Sr": 249.00000000000003,
                "Y": 231.99999999999997,
                "Zr": 223.0,
                "Nb": 218.00000000000003,
                "Mo": 217.0,
                "Tc": 216.0,
                "Ru": 213.0,
                "Rh": 210.0,
                "Pd": 210.0,
                "Ag": 211.0,
                "Cd": 218.00000000000003,
                "In": 193.0,
                "Sn": 217.0,
                "Sb": 206.0,
                "Te": 206.0,
                "I": 198.0,
                "Xe": 216.0,
                "Cs": 343.0,
                "Ba": 268.0,
                "La": 243.00000000000003,
                "Ce": 242.0,
                "Pr": 240.0,
                "Nd": 239.0,
                "Pm": 238.0,
                "Sm": 236.0,
                "Eu": 235.0,
                "Gd": 234.0,
                "Tb": 233.0,
                "Dy": 231.0,
                "Ho": 229.99999999999997,
                "Er": 229.0,
                "Tm": 227.0,
                "Yb": 225.99999999999997,
                "Lu": 224.00000000000003,
                "Hf": 223.0,
                "Ta": 222.00000000000003,
                "W": 218.00000000000003,
                "Re": 216.0,
                "Os": 216.0,
                "Ir": 213.0,
                "Pt": 213.0,
                "Au": 214.0,
                "Hg": 223.0,
                "Tl": 196.0,
                "Pb": 202.0,
                "Bi": 206.99999999999997,
                "Po": 197.0,
                "At": 202.0,
                "Rn": 220.00000000000003,
                "Fr": 348.0,
                "Ra": 283.0,
                "Ac": 247.00000000000003,
                "Th": 245.00000000000003,
                "Pa": 243.00000000000003,
                "U": 241.0,
                "Np": 239.0,
                "Pu": 243.00000000000003,
                "Am": 244.0,
                "Cm": 245.00000000000003,
                "Bk": 244.0,
                "Cf": 245.00000000000003,
                "Es": 245.00000000000003,
                "Fm": 245.00000000000003,
                "Md": 246.0,
                "No": 246.0,
                "Lr": 246.0,
                "Rf": np.nan,
                "Db": np.nan,
                "Sg": np.nan,
                "Bh": np.nan,
                "Hs": np.nan,
                "Mt": np.nan,
                "Ds": np.nan,
                "Rg": np.nan,
                "Cn": np.nan,
                "Nh": np.nan,
                "Fl": np.nan,
                "Mc": np.nan,
                "Lv": np.nan,
                "Ts": np.nan,
                "Og": np.nan,
            }
        )
        radii = vdw_radii[ase_atoms.get_chemical_symbols()].values / 100.0

    # for block in track(gpoints.blocks, description='Computing distance grid', total=gpoints.npartitions):
    dgrid = gpoints.map_blocks(
        gpd_aabb,
        ase_atoms,
        radii,
        n_neighbors=n_neighbors,
        probe_radius=probe_radius,
        dtype=float,
        drop_axis=1,
    ).reshape(shape)
    #     for block in tqdm(gpoints.blocks, desc='Computing distance grid', total=gpoints.npartitions):
    #         dgrid.append(gpd_aabb(block, ase_atoms, radii, n_neighbors=n_neighbors,probe_radius=probe_radius )) #compute grid
    #     return da.hstack(dgrid).rechunk(chunks=block_size).reshape(shape) #return in correct shape
    if precompute:
        return dgrid.compute()
    else:
        return dgrid


#
def dgrid_from_atoms_gpu(ase_atoms, radii=None, spacing=0.2, block_size=10000):
    """
    The function `dgrid_from_atoms_gpu` calculates a distance grid based on atomic positions using GPU
    acceleration.

    :param ase_atoms: The `ase_atoms` parameter in the `dgrid_from_atoms_gpu` function is expected to be
    an ASE Atoms object, which represents a collection of atoms and their properties. This object
    typically contains information such as atomic positions, unit cell parameters, atomic symbols, and
    other properties of the atoms in
    :param radii: The `radii` parameter in the `dgrid_from_atoms_gpu` function represents the van der
    Waals radii of the atoms in the atomic structure provided as input. These radii are used in the
    calculation of the distance grid. If the `radii` parameter is not provided by the
    :param spacing: The `spacing` parameter in the `dgrid_from_atoms_gpu` function represents the
    distance between grid points in the generated grid. It determines the resolution of the grid where
    the distances from atoms to grid points are calculated. A smaller spacing value will result in a
    higher resolution grid with more grid points,
    :param block_size: The `block_size` parameter in the `dgrid_from_atoms_gpu` function determines the
    size of each block used for processing the data. In this function, it is used for chunking the data
    and processing it in parallel on the GPU. A larger `block_size` can lead to more efficient, defaults
    to 10000 (optional)
    :return: The function `dgrid_from_atoms_gpu` returns a 3D numpy array representing a distance grid
    computed based on the input atomic structure provided in the form of ASE atoms. The distance grid is
    calculated by computing the minimum distance between each point on the grid and the atoms in the
    structure, taking into account the atomic radii. The grid is computed using GPU-accelerated
    calculations for efficiency.
    """

    from ase.io import read
    import numpy as np
    import dask.array as da
    import cupy as cp
    import pandas as pd
    from tqdm import tqdm
    import ase

    # data    = read(path_to_cif)
    fpoints = cp.asarray(ase_atoms.get_scaled_positions() - 0.5)
    cell = cp.asarray(ase.geometry.complete_cell(ase_atoms.get_cell())).T
    # from ase.data import vdw_radii
    # radius_series = pd.Series({'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
    #                         'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66,
    #                         'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
    #                         'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
    #                         'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
    #                         'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
    #                         'Mn': 1.50, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24,
    #                         'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
    #                         'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    #                         'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
    #                         'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
    #                         'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    #                         'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
    #                         'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
    #                         'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
    #                         'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
    #                         'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
    #                         'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
    #                         'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
    #                         'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
    #                         'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
    #                         'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21,
    #                         'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96,
    #                         'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69})
    # radii = cp.asarray(radius_series[data.get_chemical_symbols()].values).reshape(-1, 1)
    if radii is None:
        vdw_radii = pd.Series(
            {
                "H": 110.00000000000001,
                "He": 140.0,
                "Li": 182.0,
                "Be": 153.0,
                "B": 192.0,
                "C": 170.0,
                "N": 155.0,
                "O": 152.0,
                "F": 147.0,
                "Ne": 154.0,
                "Na": 227.0,
                "Mg": 173.0,
                "Al": 184.0,
                "Si": 210.0,
                "P": 180.0,
                "S": 180.0,
                "Cl": 175.0,
                "Ar": 188.0,
                "K": 275.0,
                "Ca": 231.0,
                "Sc": 215.0,
                "Ti": 211.0,
                "V": 206.99999999999997,
                "Cr": 206.0,
                "Mn": 204.99999999999997,
                "Fe": 204.0,
                "Co": 200.0,
                "Ni": 197.0,
                "Cu": 196.0,
                "Zn": 200.99999999999997,
                "Ga": 187.0,
                "Ge": 211.0,
                "As": 185.0,
                "Se": 190.0,
                "Br": 185.0,
                "Kr": 202.0,
                "Rb": 303.0,
                "Sr": 249.00000000000003,
                "Y": 231.99999999999997,
                "Zr": 223.0,
                "Nb": 218.00000000000003,
                "Mo": 217.0,
                "Tc": 216.0,
                "Ru": 213.0,
                "Rh": 210.0,
                "Pd": 210.0,
                "Ag": 211.0,
                "Cd": 218.00000000000003,
                "In": 193.0,
                "Sn": 217.0,
                "Sb": 206.0,
                "Te": 206.0,
                "I": 198.0,
                "Xe": 216.0,
                "Cs": 343.0,
                "Ba": 268.0,
                "La": 243.00000000000003,
                "Ce": 242.0,
                "Pr": 240.0,
                "Nd": 239.0,
                "Pm": 238.0,
                "Sm": 236.0,
                "Eu": 235.0,
                "Gd": 234.0,
                "Tb": 233.0,
                "Dy": 231.0,
                "Ho": 229.99999999999997,
                "Er": 229.0,
                "Tm": 227.0,
                "Yb": 225.99999999999997,
                "Lu": 224.00000000000003,
                "Hf": 223.0,
                "Ta": 222.00000000000003,
                "W": 218.00000000000003,
                "Re": 216.0,
                "Os": 216.0,
                "Ir": 213.0,
                "Pt": 213.0,
                "Au": 214.0,
                "Hg": 223.0,
                "Tl": 196.0,
                "Pb": 202.0,
                "Bi": 206.99999999999997,
                "Po": 197.0,
                "At": 202.0,
                "Rn": 220.00000000000003,
                "Fr": 348.0,
                "Ra": 283.0,
                "Ac": 247.00000000000003,
                "Th": 245.00000000000003,
                "Pa": 243.00000000000003,
                "U": 241.0,
                "Np": 239.0,
                "Pu": 243.00000000000003,
                "Am": 244.0,
                "Cm": 245.00000000000003,
                "Bk": 244.0,
                "Cf": 245.00000000000003,
                "Es": 245.00000000000003,
                "Fm": 245.00000000000003,
                "Md": 246.0,
                "No": 246.0,
                "Lr": 246.0,
                "Rf": np.nan,
                "Db": np.nan,
                "Sg": np.nan,
                "Bh": np.nan,
                "Hs": np.nan,
                "Mt": np.nan,
                "Ds": np.nan,
                "Rg": np.nan,
                "Cn": np.nan,
                "Nh": np.nan,
                "Fl": np.nan,
                "Mc": np.nan,
                "Lv": np.nan,
                "Ts": np.nan,
                "Og": np.nan,
            }
        )
        radii = cp.asarray(
            vdw_radii[ase_atoms.get_chemical_symbols()].values / 100.0
        ).reshape(-1, 1)

    # * the vdw radii mendeleev is in picometers
    # radii = cp.asarray(vdw_radii[ase_atoms.get_atomic_numbers()].values/100).reshape(-1, 1)
    [nx, ny, nz] = (ase_atoms.get_cell_lengths_and_angles()[0:3] / spacing).astype(
        int
    ) + 1
    #     print([nx,ny,nz])
    gpoints = (
        da.stack(
            da.meshgrid(
                np.linspace(-0.5, 0.5, nx),
                np.linspace(-0.5, 0.5, ny),
                np.linspace(-0.5, 0.5, nz),
                indexing="ij",
            ),
            -1,
        )
        .reshape(-1, 3)
        .rechunk(block_size, 3)
    )

    #     def gpd(points, fpoints=fpoints, cell=cell, radii=radii):
    #         points = cp.asarray(points)
    #         return cp.min(cp.linalg.norm(cp.dot(cell, (cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0)-cp.around(cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0))).reshape(-1,3).T).T, axis=1).reshape(fpoints.shape[0],-1)-radii, axis=0)

    def gpd(points, fpoints=fpoints, cell=cell, radii=radii):
        points = cp.asarray(points)
        fpoints = cp.asarray(fpoints)
        radii = cp.asarray(radii)
        diff = cp.expand_dims(points, axis=1) - cp.expand_dims(fpoints, axis=0)
        diff = (diff - cp.around(diff)).reshape(-1, 3)
        diff = cp.dot(cell, diff.T).T
        diff = cp.linalg.norm(diff, axis=1).reshape(-1, fpoints.shape[0]) - radii.T
        return cp.min(diff, axis=1)

    gpoints_da = gpoints.map_blocks(gpd, chunks=(block_size, 1), dtype="float32")

    distance_grid = []
    from rich.progress import track

    for block in track(
        gpoints_da.blocks,
        total=gpoints_da.npartitions,
        description="Computing distance grid",
    ):
        # print("Working on block {} \n".format(i))
        distance_grid.append(block.compute())

    #  Free the gpu memeory before returning the output
    del cell, fpoints, radii
    cp._default_memory_pool.free_all_blocks()
    cp._default_pinned_memory_pool.free_all_blocks()
    return np.hstack(distance_grid).reshape(nx, ny, nz)


# def regions_from_dgrid(dgrid, mask_thickness=1, **kwargs):
#     """Computes the region segmentaion from the distance grid

#     :param dgrid: 3D distance grid
#     :type dgrid: numpy.array
#     # :param peak_min:minimum height of the peak in angstrom, defaults to 2.0
#     # :type peak_min: float, optional, defaults to 2.0 angstroms
#     :param relative_threshold: Minimum intensity of peaks, calculated as max(image) * relative_threshold, defaults to 0.1
#     :type relative_threshold: float,
#     :param mask_thickness:thickness of the framework cover. Points within this distance from the framework surface are not segmented, defaults to 1
#     :type mask_thickness: float, optional
#     :return:region labels, same shape as input grid. 0 refers to the back ground
#     :rtype: 3D numpy.array same shape and input grid
#     """
#     from scipy import ndimage as ndi
#     import numpy as np
#     import sparse
#     from .helper import interpolate_labels, find_coord_from_indices
#     from skimage import segmentation
#     from skimage.morphology import extrema
#     from skimage.feature import peak_local_max, corner_peaks
#     from sklearn.cluster import AgglomerativeClustering
#     # peak_min =2
#     # max_indices = np.vstack(np.where(extrema.h_maxima(dgrid, h=0.5)))
#     # max_indices = np.vstack(np.where(extrema.local_maxima(dgrid)))
#     # max_indices = np.array(peak_local_max(dgrid, **kwargs)).T
#     # max_indices = np.array(peak_local_max(dgrid, **kwargs)).T
#     # Finding the local maxima of the corner response function.

#     # ag = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=np.max(np.array(dgrid.shape))/5, compute_full_tree=True).fit(max_indices)
#     # max_indices= np.vstack([max_indices[ag.labels_==i].mean(axis=0) for i in np.unique(ag.labels_) if i != -1]).astype(int)
#     # print(max_indices)
#     # markers_sparse = sparse.COO(max_indices, 1, shape=dgrid.shape)
#     from icecream import ic
#     probe = mask_thickness
#     import dask.array as da
#     import warnings
#     # reg_labels = segmentation.watershed(-dgrid, ndi.label(markers_sparse.todense())[0], mask=(dgrid>probe).astype(int))

#     # max_indices = np.array(peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1, **kwargs)).T
#     max_indices = np.array(peak_local_max(dgrid,  **kwargs)).T
#     markers_sparse = sparse.COO(max_indices, 1, shape=dgrid.shape)

#     probe = mask_thickness
#     import dask.array as da
#     reg_labels = segmentation.watershed(-dgrid, ndi.label(markers_sparse.todense())[0], mask=(dgrid>probe).astype(int))


#     new_max_indices = np.array(peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1)).T
#     if len(new_max_indices.T) < len(max_indices.T):
#         warnings.warn('Some maxima excluded as they fell inside the mask_thickness')

#     markers_sparse = sparse.COO(new_max_indices, 1, shape=dgrid.shape)

#     probe = mask_thickness
#     import dask.array as da
#     reg_labels = segmentation.watershed(-dgrid, ndi.label(markers_sparse.todense())[0], mask=(dgrid>probe).astype(int))

#     #* Now interpolate the region labels and sort the maxima in the correct order
#     sort_index = np.argsort([reg_labels[tuple(m)] for m in new_max_indices.T])
#     new_max_indices = (new_max_indices.T[sort_index]).T
#     # ir = interpolate_labels(reg_labels)
#     # new_max_indices = (new_max_indices.T[np.argsort(ir(new_max_indices.T/dgrid.shape))]).T

#     # # new_max_indices = []
#     # old_labels = np.unique(reg_labels)  # This is sorted aalreadylready
#     # number_of_regions = len(old_labels)
#     # new_labels = range(number_of_regions)


#     # # * Replace the old-labels with new
#     # for i in range(number_of_regions):
#     #     reg_labels[reg_labels== old_labels[i]] = new_labels[i]
#     return reg_labels, new_max_indices.T
def regions_from_dgrid(dgrid, mask_thickness=1, h=0.2, compactness=0, threshold_abs=0.5):
    """
    The function `regions_from_dgrid` takes a density grid as input and identifies regions based on
    local maxima, with options for adjusting parameters like mask thickness and compactness.

    :param dgrid: The `dgrid` parameter is a 2D array representing the input data grid. It is used in
    the function to perform calculations and identify regions based on the data values in the grid
    :param mask_thickness: The `mask_thickness` parameter in the `regions_from_dgrid` function
    determines the thickness of the mask used in the watershed segmentation algorithm. It is used to
    create a mask where regions with values greater than the `mask_thickness` are considered as markers
    for segmentation. Increasing the `mask_thickness` value, defaults to 1 (optional)
    :param h: The `h` parameter in the `regions_from_dgrid` function is used to specify the height
    threshold for detecting local maxima in the input `dgrid` array. It is used in the
    `extrema.h_maxima` function to find the local maxima points based on the specified height
    :param min_distance: The `min_distance` parameter in the `regions_from_dgrid` function specifies the
    minimum distance between peaks in the peak_local_max function. It is used to find peaks in the input
    data that are separated by at least this distance. Peaks that are closer than `min_distance` to each
    other are, defaults to 2 (optional)
    :param compactness: The `compactness` parameter in the `regions_from_dgrid` function controls how
    uniform the region sizes are in the output. A higher value for `compactness` will result in more
    equally sized regions, while a lower value will allow for more size variation among the regions.
    Adjusting the `, defaults to 0 (optional)
    :return: The function `regions_from_dgrid` returns two values: `reg_labels` and `new_max_indices.T`.
    """

    from scipy import ndimage as ndi
    import numpy as np

    try:
        import sparse
    except ImportError:
        import collections.abc

        # sparse needs the four following aliases to be done manually.
        collections.Iterable = collections.abc.Iterable
        collections.Mapping = collections.abc.Mapping
        collections.MutableSet = collections.abc.MutableSet
        collections.MutableMapping = collections.abc.MutableMapping
        import sparse

    # from .helper import interpolate_labels, find_coord_from_indices
    from skimage import segmentation
    from skimage.morphology import extrema
    from skimage.feature import peak_local_max, corner_peaks
    from sklearn.cluster import AgglomerativeClustering
    # peak_min =2
    # max_indices = np.vstack(np.where(extrema.h_maxima(dgrid, h=0.5)))
    # max_indices = np.vstack(np.where(extrema.local_maxima(dgrid)))
    # max_indices = np.array(peak_local_max(dgrid, **kwargs)).T
    # max_indices = np.array(peak_local_max(dgrid, **kwargs)).T
    # Finding the local maxima of the corner response function.

    # ag = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=np.max(np.array(dgrid.shape))/5, compute_full_tree=True).fit(max_indices)
    # max_indices= np.vstack([max_indices[ag.labels_==i].mean(axis=0) for i in np.unique(ag.labels_) if i != -1]).astype(int)
    # print(max_indices)
    # markers_sparse = sparse.COO(max_indices, 1, shape=dgrid.shape)
    from icecream import ic

    probe = mask_thickness
    import dask.array as da
    import warnings
    # reg_labels = segmentation.watershed(-dgrid, ndi.label(markers_sparse.todense())[0], mask=(dgrid>probe).astype(int))

    # max_indices = np.array(peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1, **kwargs)).T
    # max_indices = np.array(peak_local_max(dgrid,  **kwargs)).T
    max_indices = np.vstack(np.where(extrema.h_maxima(dgrid, h=h)))
    markers_sparse = sparse.COO(max_indices, 1, shape=dgrid.shape)

    probe = mask_thickness
    import dask.array as da

    reg_labels = segmentation.watershed(
        -dgrid,
        ndi.label(markers_sparse.todense())[0],
        mask=(dgrid > probe).astype(int),
        compactness=compactness,
    )

    # print(keep_these)
    # new_max_indices = max_indices[keep_these-1].T

    new_max_indices = np.array(
        peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1, p_norm=2, threshold_abs=threshold_abs)
    ).T
    if len(new_max_indices.T) < len(max_indices.T):
        warnings.warn("Some maxima excluded as they fell inside the mask_thickness")
        # warnings.warn('Some maxima excluded as they the regions were less that 0.001 of the box')

    markers_sparse = sparse.COO(new_max_indices, 1, shape=dgrid.shape)

    probe = mask_thickness
    import dask.array as da

    reg_labels = segmentation.watershed(
        -dgrid,
        ndi.label(markers_sparse.todense())[0],
        mask=(dgrid > probe).astype(int),
        compactness=compactness,
    )

    new_max_indices = np.array(
        peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1, p_norm=2, threshold_abs=threshold_abs)
    ).T
    # ir = interpolate_labels(reg_labels)
    # new_max_indices = (new_max_indices.T[np.argsort(ir(new_max_indices.T/dgrid.shape))]).T

    # # new_max_indices = []
    # old_labels = np.unique(reg_labels)  # This is sorted aalreadylready
    # number_of_regions = len(old_labels)
    # new_labels = range(number_of_regions)

    # # * Replace the old-labels with new
    # for i in range(number_of_regions):
    #     reg_labels[reg_labels== old_labels[i]] = new_labels[i]

    # # *  Lets clean out the regions that are too small
    # props = regionprops(reg_labels)
    # print([(p.label, p.area/dgrid.size) for p in props])
    # len([(p.label, p.area/dgrid.size) for p in props if p.area > dgrid.size/1000])
    # keep_these = [p.label for p in props if p.area > dgrid.size/1000]
    # maxima2 = np.vstack([new_max_indices.T[k-1] for k in keep_these])
    # markers_sparse = sparse.COO(maxima2.T, 1, shape=dgrid.shape)

    # # probe = mask_thickness
    # # import dask.array as da
    # reg_labels = segmentation.watershed(-dgrid, ndi.label(markers_sparse.todense())[0], mask=(dgrid>probe).astype(int), compactness=False)
    # new_max_indices = np.array(peak_local_max(dgrid, labels=reg_labels, num_peaks_per_label=1)).T

    # * Now interpolate the region labels and sort the maxima in the correct order
    sort_index = np.argsort([reg_labels[tuple(m)] for m in new_max_indices.T])
    new_max_indices = (new_max_indices.T[sort_index]).T

    return reg_labels, new_max_indices.T


# def connections_from_regions():

#     """
#     It takes a 3D array of region labels, a list of maxima, an ASE atoms object, and a distance
#     threshold, and returns a list of connections between regions, and the centers of the windows between
#     them

#     :param region_labels: The labels of the regions, as returned by the skimage.measure.label function
#     :param maxima: the maxima of the image
#     :param ase_atoms: ASE atoms object
#     :param wall_windows: If True, then the windows on the wall are also calculated, defaults to True
#     (optional)
#     :param distance_threshold: The distance threshold for the agglomerative clustering
#     :return: A list of connections between regions.
#     """


#     from skimage.segmentation import find_boundaries
#     import numpy as np
#     import itertools
#     import ase
#     # from sklearn.cluster import DBSCAN
#     # from sklearn.preprocessing import StandardScaler
#     from icecream import ic
#     from sklearn.cluster import AgglomerativeClustering
#     from .helper import interpolate_labels, get_fractional_coordinates
#     A_unit = ase.geometry.complete_cell(ase_atoms.get_cell()).T
#     from icecream import ic
#     def check_connected(regs):
#         import numpy as np
#         from sklearn.cluster import AgglomerativeClustering
#         i = regs[0]
#         j = regs[1]
#         ic.disable()
#         ic(regs)

#         check_flag = np.logical_or(np.logical_and(region_labels == i, outer.get(j)), np.logical_and(region_labels == j, outer.get(i)))
#         if np.sum(check_flag) > 0:
#             # print("Edges found between "+str(i)+' and '+ str(j) + ' sum: '+str(np.sum(check_flag)))
#             windices = np.vstack(np.where(check_flag)).T  # * (N,3)
#             window_points = np.dot(A_unit, (windices / region_labels.shape).T).T  # * (N,3)
#             # ic(window_points)
#             # * No clustering
#             # window_centers = np.mean(window_points, axis=0)

#             # * DBScan
#             # X_scaled = StandardScaler().fit_transform(window_points)
#             # db = DBSCAN(eps=0.5, min_samples=10).fit(X_scaled)
#             # window_centers = np.vstack([window_points[db.labels_==i].mean(axis=0) for i in np.unique(db.labels_) if i != -1])

#             # if len(window_points) >1:
#             # * Agglomerative
#             ag = AgglomerativeClustering(n_clusters=None, linkage='single',distance_threshold=minimum_window_separation, compute_full_tree=True).fit(window_points)
#             window_centers = np.vstack([window_points[ag.labels_==i].mean(axis=0) for i in np.unique(ag.labels_) if i != -1])
#         # else:
#             # window_centers = window_points
#             window_fractional = get_fractional_coordinates(window_centers,ase_atoms)
#             window_radii = np.array(dinterp(window_fractional))
#             return np.array([regs, True,'internal', window_centers, window_radii], dtype='object') # returns a list of [[r1,r2], connected or not, array of window_centers]
#         else:
#             return np.array([regs, False,'internal', None, None], dtype='object')

#     dinterp = interpolate_labels(dgrid)
#     # * Find the outer boundary of each individual region
#     regions_to_check = np.delete(np.unique(region_labels),0)
#     ic(regions_to_check)
#     outer = {reg:find_boundaries(region_labels == reg, mode='outer') for reg in regions_to_check}
#     ic(len(outer))
#     # * Check for connectedness between regions
#     check_list = np.vstack(list(itertools.combinations(regions_to_check, 2))) # list of pairs of region labels
#     connections_list = [check_connected(c) for c in check_list] # now this step could cause an overflow
#     # connected_regions = [[c[0], [c[2]] for c in connections_list if c[1] == True] # Make a new list with all the

#     #* Windows on the wall are to be calculated
#     if wall_windows:
#         wwl = get_wall_windows2(regions=region_labels, maxima=maxima, dgrid=dgrid, ase_atoms=ase_atoms, wall_thickness=wall_thickness)
#         if len(wwl)>0:
#             connections_list.append(wwl)
#     return np.vstack(connections_list)
# return np.vstack([c for c in np.vstack(connections_list) if c[1]])# for only the true connections use connections_list[:,[0,2]][np.where(connections_list[:,1]==True)] #np.vstack(check_list[np.where(flag_list[:, 0])]), np.round(np.vstack(flag_list[:, 1][np.where(flag_list[:, 0])]), decimals=2))

# def get_wall_windows(regions, maxima, ase_atoms):
#     """
#     > It finds the windows in the walls of the simulation box, and returns the indices of the regions
#     that are in the window

#     :param regions: the regions array
#     :param maxima: the maxima of the regions
#     :param ase_atoms: ASE atoms object
#     """
#     import numpy as np
#     from .helper import find_coord_from_indices
#     wall_windows = []
#     shape =regions.shape
#     for a in [0,1,2]:

#         llr = regions.take(indices=0 ,axis=a) #* left wall
#         rrr = regions.take(indices=shape[a]-1 ,axis=a)  #* right wall

#         ll = [l for l in np.delete(np.unique(llr),0) if maxima[l-1][a]!=0] #* unique regions in left wall whose maxima are not on the wall
#         rr = [ r for r in np.delete(np.unique(rrr),0) if maxima[r-1][a]!=shape[a]-1]#* unique regions in the right wall whose maxima are not on the wall
#         left_right_map = np.vectorize(dict(zip(llr.flatten(), rrr.flatten())).get)
#         right_left_map = np.vectorize(dict(zip(rrr.flatten(), llr.flatten())).get)
#         ll_2 = [l for l in ll if maxima[left_right_map(l)-1][a]!=shape[a]-1]
#         rr_2 = [r for r in rr if maxima[right_left_map(r)-1][a]!=0]

#         if len(ll_2)>0:
#             #* Mean of the indices of each windowed region in the left nad right wall, only if the opposite wall maxima is not on the wall
#             wls = np.vstack([np.vstack(np.where(llr==l)).mean(axis=1) for l in ll_2 ])
#             #* Add the third index value, in the correct position, 0 for left wall, shape[axis] - 1 for the right wall
#             wls = np.insert(wls,[a], np.zeros((len(wls),1)), axis=1)
#             wls = find_coord_from_indices(wls, shape=shape, ase_atoms=ase_atoms)
#             # * use the map to come up with the correct touching regions
#             ll_p =  left_right_map(ll_2) #[np.unique(rrr[np.where(llr==l)])[0] for l in ll ]
#             left_windows  = [[np.array([r1, r2]), True,'periodic', np.array(wc).reshape(-1,3)] for r1, r2,wc in zip(ll, ll_p, wls)]
#             wall_windows.append(left_windows)

#         if len(rr_2)>0:
#             wrs = np.vstack([np.vstack(np.where(rrr==r)).mean(axis=1) for r in rr_2 ])
#             wrs = np.insert(wrs,[a], (shape[a]-1)*np.ones((len(wrs),1)), axis=1)
#             wrs = find_coord_from_indices(wrs, shape=shape, ase_atoms=ase_atoms)
#             rr_p = right_left_map(rr_2) # [np.unique(llr[np.where(rrr==r)])[0] for r in rr ]
#             right_windows = [[np.array([r1, r2]), True,'periodic', np.array(wc).reshape(-1,3)] for r1, r2,wc in zip(rr, rr_p, wrs)]
#             #* Append the newly found window indices
#             wall_windows.append(right_windows)

#     if len(wall_windows)>0:
#         wall_windows =  np.vstack(wall_windows)
#     return wall_windows


def get_region_props_for_regions(regions):
    """
    It takes a list of regions and returns a list of region properties

    :param regions: The labeled image
    """
    from skimage.measure import regionprops

    return regionprops(regions)


def add_window_info_to_rag(rag, connections):
    """
    It takes a RAG and a list of connections and adds the window information to the RAG as edge
    attributes

    :param rag: the region adjacency graph
    :param connections: list of tuples, each tuple is a connection between two nodes, and contains the
    following information:
    :return: The rag is being returned with the window information added to it.
    """

    import networkx as nx

    window_attribute = {}
    for c in connections:
        if c[1]:
            window_attribute[tuple(c[0])] = {
                "window_type": c[2],
                "window_position": c[3],
                "window_radii": c[4],
            }
        # else:
    nx.set_edge_attributes(rag, window_attribute, "window_info")
    return rag


def add_maxima_to_rag(G, maxima, maxima_radii, shape, ase_atoms):
    """
    > This function takes a graph, a list of maxima, a list of maxima radii, a shape, and an ase_atoms
    object and adds the maxima and maxima radii to the graph.

    Let's look at the function in more detail.

    The first thing we do is import the `find_coord_from_indices` function from the `helper` module.

    Next, we check to make sure that the number of nodes in the graph is equal to the number of maxima.
    If not, we raise a `ValueError`.

    If the number of nodes in the graph is equal to the number of maxima, we loop through the nodes and
    maxima and add the maxima indices, cartesian coordinates and maxima radii to the graph.



    :param G: the graph
    :param maxima: the indices of the maxima (2nd output of regions_from_grid routine)
    :param maxima_radii: the radius of the maxima
    :param shape: the shape of the image
    :param ase_atoms: The atoms object from ASE
    :return: The graph with the maxima added to the nodes
    """

    from .helper import find_coord_from_indices

    if len(G.nodes()) != len(maxima):
        raise ValueError(
            "The number of nodes in the graph and the number of maxima do not match"
        )
    else:
        # regionprops = get_region_props_for_regions(G.nodes())
        for n, m in zip(G.nodes(), maxima):
            G.nodes[n]["maxima"] = find_coord_from_indices(
                maxima[G.nodes[n]["labels"][0] - 1], shape=shape, ase_atoms=ase_atoms
            )  # * add the coordinates of the maxima to the node
            G.nodes[n]["maxima_indices"] = maxima[
                G.nodes[n]["labels"][0] - 1
            ]  # * add the indcies of the maxima
            G.nodes[n]["maxima_radii"] = maxima_radii[
                G.nodes[n]["labels"][0] - 1
            ]  # * add the indcies of the maxima
    return G


# def add_size_dist_to_graph(G, dgrid):

# def zoom_dgrid(grid, zoom_ratio, order=1):

#     """
#     It takes a 2D array and a zoom ratio, and returns a zoomed
#     version of the array

#     :param grid: the grid to be zoomed
#     :param zoom_ratio: the ratio of the new grid to the old grid.
#     For example, if the old grid is 100x100 and the new grid is
#     200x200, then zoom_ratio = 2
#     :param order: The order of the spline interpolation, default
#     is 1. The order has to be in the range 0-5, defaults to 1
#     (optional)
#     :return: The grid_zoomed is being returned.
#     """

#     from scipy.ndimage import zoom
#     grid_zoomed = zoom(grid, zoom_ratio, mode='grid-wrap', order=order, grid_mode=False)
#     return grid_zoomed


def add_psd_to_rag(G, region_props, bins, ase_atoms, shape):
    """
    > This function takes a graph, a list of region properties, a list of bins, an ase atoms object, and
    a shape, and adds the psd to the graph.

    For each region, we calculate the distances between the centroid and the coordinates.

    Then we calculate the histogram of the distances.

    Then we calculate the bin centers.

    Then we add the bin centers and the histogram to the graph.

    Then we return the graph.

    Note that these distance histograms or centroids do not account for periodic boundary conditions.
    To include periodic boundary conditions, use the add_psd_to_rag_with_pbc function.

    :param G: the graph
    :param region_props: a list of dictionaries, each dictionary contains the properties of a region
    :param bins: The number of bins to use for the histogram
    :param ase_atoms: The atoms object from ASE
    :param shape: shape of the image
    :return: The graph with the psd_x and psd_y attributes added to the nodes.
    """

    import numpy as np
    from .helper import find_coord_from_indices

    for region_prop in region_props:
        distances = np.linalg.norm(
            find_coord_from_indices(
                region_prop["centroid"] - region_prop["coords"],
                shape=shape,
                ase_atoms=ase_atoms,
            ),
            axis=1,
        )

        G.nodes[region_prop["label"]]["psd_y"], bin_edges = np.histogram(
            distances,
            bins=bins,
            density=True,
        )
        # bin_edges = np.histogram(region_prop['coords']-region_prop['centroid'], bins=bins, normed=True, density=1)[1]
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        G.nodes[region_prop["label"]]["psd_x"] = bin_centers
    return G


def rag_from_connections(connections, maxima, only_use_internal=True):
    """
    This Python function creates a graph representation of regions of interest based on connections and
    maximum values, with an option to include only internal connections.

    :param connections: Connections is a list of tuples where each tuple contains information about a
    connection. Each tuple has the following format: (node_id, is_connected, connection_type)
    :param maxima: The `maxima` parameter in the `rag_from_connections` function seems to represent a
    list of nodes or regions in a graph. It is used to determine the number of nodes in the graph
    created by the function. Each element in `maxima` likely corresponds to a node in the graph
    :param only_use_internal: The `only_use_internal` parameter in the `rag_from_connections` function
    is a boolean parameter that determines whether only internal connections should be used when
    creating the graph. If `only_use_internal` is set to `True`, only connections with the type
    'internal' will be added to the graph, defaults to True (optional)
    :return: The function `rag_from_connections` is returning a NetworkX graph `G` that is constructed
    based on the input connections, maxima, and the optional parameter only_use_internal. The graph `G`
    is built by adding nodes and edges from the input connections based on certain conditions. The nodes
    of the graph have a 'labels' attribute assigned to them.
    """

    import networkx as nx
    import numpy as np

    G = nx.Graph()
    # number_of_regions = np.max([c[0] for c  in connections])
    number_of_nodes = len(maxima)  # np.max([c[0] for c  in connections])
    # print(number_of_nodes)
    # nx.set_node_attributes(G, values={i+1:i+1 for i in range(number_of_nodes)}, name='labels')
    G.add_nodes_from(range(1, number_of_nodes + 1))
    if only_use_internal:  # range stops one short
        G.add_edges_from(
            [c[0] for c in connections if np.logical_and(c[1], c[2] == "internal")]
        )
    else:
        G.add_edges_from([c[0] for c in connections if c[1]])
    # add labels property to rag
    nx.set_node_attributes(
        G, values={i + 1: [i + 1] for i in range(len(G.nodes))}, name="labels"
    )
    return G


def get_rag_from_regions_and_grid(regions, dgrid, ase_atoms, maxima=None):
    """
    > It takes a grid of regions and a grid of distances and returns a region adjacency graph with the centroid of each
    region as a node property

    :param regions: the regions that we computed from the distance grid
    :param dgrid: the distance grid
    :param ase_atoms: the atoms object from ase
    :return: A region adjacency graph
    """

    import numpy as np

    # * compute a region adjacency graph from skimage
    from skimage import graph
    from .helper import find_coord_from_indices

    gr_mean = graph.rag_mean_color(dgrid, regions)

    from skimage.measure import regionprops

    # * add the centroid property to the rag nodes
    region_props = regionprops(regions)
    for region_prop in region_props:
        # add the coordinates of the centroid to the node
        gr_mean.nodes[region_prop["label"]]["centroid"] = find_coord_from_indices(
            indices=np.array(region_prop["centroid"]),
            ase_atoms=ase_atoms,
            shape=dgrid.shape,
        )
        gr_mean.nodes[region_prop["label"]]["centroid_indices"] = region_prop[
            "centroid"
        ]  # * add the indices of the centroid to the node
    gr_mean.remove_node(0)
    if maxima is not None:
        maxima_radii = dgrid[tuple(maxima.T)]  # * radii of the maxima
        gr_mean = add_maxima_to_rag(
            gr_mean, maxima, maxima_radii, regions.shape, ase_atoms
        )
        for edge in gr_mean.edges():
            gr_mean.edges[edge]["weight"] = np.linalg.norm(
                gr_mean.nodes[edge[0]]["maxima"] - gr_mean.nodes[edge[1]]["maxima"]
            )
    else:
        for edge in gr_mean.edges():
            gr_mean.edges[edge]["weight"] = np.linalg.norm(
                gr_mean.nodes[edge[0]]["centroid"] - gr_mean.nodes[edge[1]]["centroid"]
            )

    return gr_mean


def get_periodic_map(
    regions, maxima, maxima_radii, ase_atoms, return_conn=False, **kwargs
):
    """
    It takes a 3D array of integers representing region indices that is obtained from region segmentation, a list of maxima, and a list of maxima radii, and returns a dictionary
    that maps each region to the nearest maximum.

    :param regions: a 3D array of integers, where each integer is a region index, obtained from regions_from_dgrid function
    :param maxima: the 3D indices of the maxima, obtained from regions_from_dgrid function
    :param maxima_radii: the list of radii of the maxima (in Angstrom)
    :param return_conn: if True, returns the connectivity groups of the regions, defaults to False
    (optional)
    :return: a dictionary with the keys being the starting region indices and the values being the periodic group of
    the regions.

    then to apply this periodic map to regions, use:
        np.vectorize(periodic_map.get)(regions)
    """

    # r1 = apply_pbc(regions, maxima, **kwargs)
    if return_conn:
        return apply_pbc(
            regions,
            maxima,
            maxima_radii,
            ase_atoms=ase_atoms,
            return_conn=return_conn,
            **kwargs,
        )
    else:
        return dict(
            zip(
                regions.flatten(),
                apply_pbc(
                    regions, maxima, maxima_radii, ase_atoms=ase_atoms, **kwargs
                ).flatten(),
            )
        )


def apply_periodic_map(input_array, periodic_map):
    """
    It takes an array of integers and a dictionary that maps integers to integers, and returns an array
    of integers that is the result of applying the dictionary to the input array

    :param input_array: a numpy array of integers
    :param periodic_map: a dictionary that maps the input values to the output values
    :return: the value of the key in the dictionary.
    """
    import numpy as np

    return np.vectorize(periodic_map.get)(input_array)


# if __name__ == "__main__":
#     # Do something if this file is invoked on its own
#     print(canvas())
# def graph_from_connections(connections, maxima, only_use_internal=True):

#     """ Create a basic networkx graph (with no properties annotated) from the array of connections
#     :type connections: np.array of shape (Nregions-Choose-2 ,1) where Nregions is the number of unique regions. Each row has
#     three elements, the region indices [r1, r2], a boolean flag showing whether they are connected, and window coordinate(s) if the regions are connected, and None is they are not.
#     :param connections: numpy array

#     :raises:

#     :rtype: a graph object of the class networkx.Graph
#     """
#     import networkx as nx
#     import numpy as np
#     G = nx.Graph()
#     # number_of_regions = np.max([c[0] for c  in connections])
#     number_of_nodes = len(maxima) #np.max([c[0] for c  in connections])
#     # print(number_of_nodes)
#     G.add_nodes_from(range(1, number_of_nodes+1))
#     if only_use_internal:# range stops one short
#         G.add_edges_from([c[0] for c in connections if np.logical_and(c[1], c[2] =='internal')])
#     else:
#         G.add_edges_from([c[0] for c in connections if c[1]])

#     return G


def add_props_to_graph(
    G, node_props=None, node_prop_names=None, edge_props=None, edge_prop_names=None
):
    """
    The function takes in a graph, and a list of node properties and edge properties, and adds them to the graph as node and
    edge attributes.


    :param G: the graph
    :param node_props: a list of lists, where each list contains the node properties for a given node property
    :param node_prop_names: list of names of node properties
    :param edge_props: a list of lists, where each list contains the edge properties for a given edge property
    :param edge_prop_names: list of strings, names of edge properties
    :return: A graph with the node and edge properties added to it.
    """

    import numpy as np
    import networkx as nx

    if np.any(node_props) != None:
        for index in range(len(node_props)):
            d = {i + 1: k for i, k in enumerate(node_props[index])}
            nx.set_node_attributes(G, d, name=node_prop_names[index])

    if np.any(edge_props) != None:
        for index in range(len(edge_props)):
            d = {i: k for i, k in zip(list(G.edges()), edge_props[index])}
            nx.set_edge_attributes(G, d, name=edge_prop_names[index])

    return G


def AABB_on_atoms(ase_atoms, points=None):
    """
    > This function takes an ASE atoms object and returns a freud AABBQuery object

    :param ase_atoms: ASE atoms object
    :param points: the positions of the atoms
    :return: The AABB tree
    """
    from freud.locality import AABBQuery
    from freud.box import Box
    import ase
    # from ase.data import vdw_radii

    if points is None:
        points = ase_atoms.get_positions()
    cell = ase.geometry.complete_cell(
        ase_atoms.get_cell()
    ).T  # this needs to be an upper triangular matrix, essential column vectors, not row vectors

    box = Box.from_matrix(cell, dimensions=3)  # Define the box for Freud
    ab = AABBQuery(box=box, points=points)  # Compute the AABB Tree
    # Perform the nearest neighbor query with n-neighbors (10 by default)

    return ab
def get_wall_windows3(regions, maxima, dgrid, ase_atoms, wall_thickness):
    """
    This function calculates the wall windows based on the provided regions, maxima, dgrid, and other parameters.
    
    :param regions: The labeled regions from the segmentation.
    :param maxima: The maxima points in the dgrid.
    :param dgrid: The dgrid data.
    :param ase_atoms: The ASE atoms object.
    :param wall_thickness: The thickness of the wall to consider.
    :return: A list of wall windows.
    """
    wall_windows = []
    wregs_l, wregs_r = [], []  # Assuming these are defined somewhere in your code
    window_coords_l, window_coords_r = [], []  # Assuming these are defined somewhere in your code
    window_radii_l, window_radii_r = [], []  # Assuming these are defined somewhere in your code

    # Ensure that window_coords_l and window_coords_r are properly shaped
    window_rows_l = np.vstack(
        [
            [*w, np.array(window_coords_l[i]).flatten(), window_radii_l[i]]  # Flatten to ensure 1D
            for i, w in enumerate(wregs_l)
        ]
    )
    window_rows_r = np.vstack(
        [
            [*w, np.array(window_coords_r[i]).flatten(), window_radii_r[i]]  # Flatten to ensure 1D
            for i, w in enumerate(wregs_r)
        ]
    )

    wall_windows.append(window_rows_l)
    wall_windows.append(window_rows_r)

    return wall_windows
def get_wall_windows2(regions, maxima, dgrid, ase_atoms, wall_thickness=2):
    import numpy as np
    from icecream import ic
    from .helper import find_coord_from_indices, interpolate_labels

    wall_windows = []
    shape = regions.shape
    dinterp = interpolate_labels(dgrid)
    for a in [0, 1, 2]:
        ic.disable()
        ic(a)
        llr = regions.take(indices=0, axis=a)  # * left wall
        rrr = regions.take(indices=shape[a] - 1, axis=a)  # * right wall

        map_list = np.unique(list(zip(llr.flatten(), rrr.flatten())), axis=0)
        not_background = [m for m in map_list if np.all(m != 0)]

        # * there should be some non background mappings in the direction
        if len(not_background) > 0:
            map_list = np.vstack([m for m in map_list if np.all(m != 0)])
            flags = np.vstack(
                [
                    [
                        maxima[k - 1][a] > wall_thickness,
                        maxima[v - 1][a] < (shape[a] - wall_thickness),
                    ]
                    for k, v in map_list
                ]
            )
            flags = np.logical_and(flags[:, 0], flags[:, 1])
            # window_indices = [np.vstack(np.where(llr==m[0])).mean(axis=1) for i,m in enumerate(map_list) if flags[i]]
            wregs_l = [
                (np.array([m[0], m[1]]), True, "periodic")
                for i, m in enumerate(map_list)
                if flags[i]
            ]
            wregs_r = [
                (np.array([m[1], m[0]]), True, "periodic")
                for i, m in enumerate(map_list)
                if flags[i]
            ]
            window_indices = [
                np.vstack(np.where(llr == m[0])).mean(axis=1).T
                for i, m in enumerate(map_list)
                if flags[i]
            ]
            ic(window_indices)
            if np.any(flags):
                window_indices_l = np.insert(
                    np.vstack(window_indices),
                    [a],
                    np.zeros((len(window_indices), 1)),
                    axis=1,
                )
                window_indices_r = np.insert(
                    np.vstack(window_indices),
                    [a],
                    (shape[a] - 1) * np.ones((len(window_indices), 1)),
                    axis=1,
                )
                window_radii_l = np.vstack(dinterp(window_indices_l / regions.shape))
                window_radii_r = np.vstack(dinterp(window_indices_r / regions.shape))
                window_coords_l = find_coord_from_indices(
                    window_indices_l, shape=shape, ase_atoms=ase_atoms
                )
                window_coords_r = find_coord_from_indices(
                    window_indices_r, shape=shape, ase_atoms=ase_atoms
                )
                ic(window_coords_l)
                ic(window_coords_r)
                window_rows_l = [
                    [*w, np.array([window_coords_l[i]]), window_radii_l[i]]
                    for i, w in enumerate(wregs_l)
                ]
                window_rows_r = [
                    [*w, np.array([window_coords_r[i]]), window_radii_r[i]]
                    for i, w in enumerate(wregs_r)
                ]
                wall_windows.extend(window_rows_l)
                wall_windows.extend(window_rows_r)

    if len(wall_windows) > 0:
        wall_windows = np.array(wall_windows, dtype=object)
    return wall_windows


# def apply_pbc(regions, maxima, maxima_radii, wall_thickness=2, return_conn =False):

#     import copy
#     # a = 0
#     import numpy as np
#     from icecream import ic
#     ic.disable()
#     regions_pbc = copy.deepcopy(regions)
#     shape = regions.shape
#     import networkx as nx
#     g1= nx.Graph()
#     g1.add_nodes_from(range(len(np.unique(regions))))

#     for a in [0,1,2]:

#         ic(a)
#         llr = regions_pbc.take(indices=0 ,axis=a) #* left wall
#         rrr = regions_pbc.take(indices=shape[a]-1 ,axis=a)  #* right wall

#         # ic(dict(zip(llr.flatten(), rrr.flatten())))
#         # ic(dict(zip(rrr.flatten(),llr.flatten())))
#         # dlist = [dict(sorted(zip(llr.flatten(), rrr.flatten()))),dict(sorted(zip(rrr.flatten(),llr.flatten())))]

#         #* Check left vs right wall
#         # map_dict = dlist[0]
#         # map_dict.pop(0)
#         # ic(map_dict)


#         map_list = np.unique(list(zip(llr.flatten(), rrr.flatten())),axis=0)
#         map_list = map_list[np.all(map_list!=0,axis=1)]
#         ic(map_list)
#         if len(map_list)> 0:
#             reg_numbers = np.vstack([(k, v) for k,v in map_list])
#             flags = np.vstack([[maxima[k-1][a]<wall_thickness, maxima[v-1][a]>(shape[a]-wall_thickness)] for k,v in map_list])
#             # flags = np.logical_or(flags[:,0],flags[:,1,])
#             # flags = np.logical_or(flllags[:,0],flags[:,1,])
#             flags1 = np.logical_and(flags[:,0], flags[:,1])
#             flags2 = np.logical_xor(flags[:,0], flags[:,1])
#             reg_radii = np.vstack([[maxima_radii[r[0]-1], maxima_radii[r[1]-1]] for r in reg_numbers])
#             flags3 = np.vstack([reg_radii[i][np.where(f)]<reg_radii[i][np.where(np.invert(f))][0] if flags2[i] else False for i,f in enumerate(flags)])
#             # flags3 = np.vstack([ True if flags2[i] else False for i,f,r  in enumerate(list(zip(flags, reg_numbers)))])
#             # reg_radii = [maxima_radii[r[0]-1], maxima_radii[r[1]-1] for r in reg_numbers]

#             ic(flags)
#             flags = [flags1[i] if flags1[i] else flags2[i] if flags3[i] else False for i,f in enumerate(flags)]
#             ic(flags1)
#             ic(flags2)
#             ic(flags3)
#             ic(flags)       # flags = [True if flags1[i] else True if np.logical_and(flags2[i], flags3[i] ) else False for i,f in enumerate(flags)]
#             g1.add_edges_from(reg_numbers[np.where(flags)])
#             # ic(flags)

#             # map_dict = dlist[1]
#             # map_dict.pop(0)
#             # ic(map_dict)
#             # reg_numbers = np.vstack([(k, v) for k,v in map_dict.items()])
#             # flags = np.vstack([[maxima[v-1][a]<wall_thickness, maxima[k-1][a]>(shape[a]-wall_thickness)] for k,v in map_dict.items()])
#             # # flags = np.logical_or(flags[:,0],flags[:,1,])
#             # ic(flags)
#             # flags1 = np.logical_and(flags[:,0], flags[:,1])
#             # flags2 = np.logical_xor(flags[:,0], flags[:,1])
#             # reg_radii = np.vstack([[maxima_radii[r[0]-1], maxima_radii[r[1]-1]] for r in reg_numbers])
#             # flags3 = np.vstack([reg_radii[i][np.where(f)]<reg_radii[i][np.where(np.invert(f))] if flags2[i] else False for i,f in enumerate(flags)])
#             # # flags3 = np.vstack([maxima_radii[r[f]-1] < maxima_radii[r[np.invert(f)]-1] if flags2[i] else False for i,f,r  in enumerate(list(zip(flags, reg_numbers)))])
#             # # reg_radii = [maxima_radii[r[0]-1], maxima_radii[r[1]-1] for r in reg_numbers]

#             # flags = [flags1[i] if flags1[i] else flags2[i] if flags3[i] else False for i,f in enumerate(flags)]
#             # g1.add_edges_from(reg_numbers[np.where(flags)])

#             # ic(flags1)
#             # ic(flags2)
#             # ic(flags3)
#             # ic(flags)
#     conn = nx.connected_components(g1)
#     # ic(list(conn))
#     # ic(list(G.edges()))
#     # ic([c for c in list(conn)])
#     for i,c in enumerate(conn):
#         for r in c:
#             regions_pbc[np.where(regions==r)] = i
#             ic('replace {0} with {1}'.format(r, i))

#     # if reindex:
#     #     # * Re-index the regions so that it runs from zero to N-regions.
#     #     old_labels = np.unique(regions_pbc)  # This is sorted aalreadylready
#     #     number_of_regions = len(old_labels)
#     #     new_labels = range(number_of_regions)

#     #     # * Replace the old-labels with new
#     #     for i in range(number_of_regions):
#     #         regions_pbc[regions_pbc== old_labels[i]] = new_labels[i]
#     if return_conn:
#         return list(nx.connected_components(g1))[1:]
#     else:
#         return regions_pbc


def add_props_to_rag(rag, props):
    for node in rag.nodes:
        node_index = [p["label"] == node for p in props].index(True)
        rag.nodes[node]["props"] = props[node_index]
    return rag


# def do_it_all(ase_atoms,spacing=0.25):
#     """
#     > It takes an ase atoms object of the mof, and returns a list of objects that are useful for analyzing the pore
#     network of the material

#     :param ase_atoms: the atoms object
#     :param spacing: the spacing of the grid
#     :return: updated_atoms, grid,regions, maxima, connections, pgraph
#     """
#     from .helper import find_coord_from_indices
#     from .analyze import update_with_covalent_bonds
#     updated_atoms =update_with_covalent_bonds(ase_atoms)
#     grid = dgrid_from_atoms(ase_atoms=ase_atoms, spacing=spacing)
#     regions, maxima =regions_from_dgrid(grid.compute(), mask_thickness=1.5, exclude_border=False, min_distance=int(2/spacing), threshold_abs=1)
#     connections = connections_from_regions(maxima=maxima, region_labels=regions,ase_atoms=ase_atoms,dgrid=grid.compute(),minimum_window_separation=1,wall_windows=True)
#     pgraph = graph_from_connections(connections, maxima, only_use_internal=True)
#     pgraph = add_props_to_graph(pgraph, node_props = [find_coord_from_indices(maxima, shape=grid.shape, ase_atoms=updated_atoms)], node_prop_names=['node_coords'])
#     return [updated_atoms, grid,regions, maxima, connections, pgraph]


def interpolate_dgrid(dgrid):
    """
    The function `interpolate_dgrid` interpolates a 3D grid using linear interpolation.

    :param dgrid: The `interpolate_dgrid` function you provided seems to be interpolating a 3D grid
    using linear interpolation. To use this function, you need to pass a 3D grid (`dgrid`) as a
    parameter. The grid should be a numpy array representing the values at different points in
    :return: The function `interpolate_dgrid` returns a RegularGridInterpolator object `dl` that is
    created using the input `dgrid` and linear interpolation method.
    """

    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI

    xx = np.linspace(0, 1, dgrid.shape[0])
    yy = np.linspace(0, 1, dgrid.shape[1])
    zz = np.linspace(0, 1, dgrid.shape[2])
    dl = RGI((xx, yy, zz), dgrid, method="linear")

    return dl


def get_percolated_paths_on_rag(
    rag, minimum_overlap_fraction=0.5, wall_cells=6, path_jump_cutoff=None
):
    import networkx as nx

    # import mofography as mgr
    import numpy as np
    from icecream import ic
    import copy
    import pandas as pd
    import more_itertools
    from .helper import get_fractional_coordinates

    # from time import time
    from tqdm import tqdm

    # time0 = time()
    keys_to_check = ["regions", "connections", "ase_atoms"]

    for key in keys_to_check:
        if key not in rag.graph.keys():
            raise ValueError(
                f"'{key}' is not in rag.graph.keys(). Please compute and add it first to the rag under the graph dictionary like this: rag.graph['{key}'] = {key}."
            )
    regions = rag.graph["regions"]
    mat_atoms = rag.graph["ase_atoms"]
    connections = rag.graph["connections"]
    shape = regions.shape
    map_list = []
    paths = []
    path_windows = []
    path_lengths = []
    periodic_connections = [c for c in connections if c[1] and c[2] == "periodic"]

    if path_jump_cutoff is None:
        # path_jump_cutoff = np.max([int(np.cbrt(np.max(regions))), 4])
        ranges = [(0, 10), (10, 40), (40, float("inf"))]
        outputs = [8, 4, 3]

        # Create the dictionary
        path_jump_cutoff = pd.cut(
            [rag.number_of_nodes()],
            bins=[r[0] for r in ranges] + [float("inf")],
            labels=outputs,
        )[0]

    if len(periodic_connections) > 0:
        periodic_windows_exist = True
        periodic_connected_regions = np.array(
            [pc[0] for pc in periodic_connections]
        )  # * just the indices of the connected regions, just needed to make sure that we don't miss any periodic windows between the same regions
        periodic_window_coords = np.vstack([c[3] for c in periodic_connections])
        periodic_window_fractional_coords = get_fractional_coordinates(
            periodic_window_coords, mat_atoms
        )

        # * we need to know which wall these windwos are on, round values close to 0 to 0 and values close to 1 to 1
        # * we will use this as an additional check when getting the correct window diameters
        periodic_window_walls = np.where(
            periodic_window_fractional_coords < 0.03,
            0,
            np.where(
                periodic_window_fractional_coords > 0.97,
                1,
                periodic_window_fractional_coords,
            ),
        )

        # * this bit is for creating a master list of where the periodic windows exist
        dist_thresh = 0.03

        # * let's make an array to store the wall window location
        regions_wall_windows = np.full((regions.max(), 7), np.NaN)
        regions_wall_windows[:, 0] = np.arange(1, regions.max() + 1)

        # * get the periodic window radii
        pws_window_radii = [
            w for c in connections if c[1] and c[2] == "periodic" for w in c[4]
        ]

        # * periodic windows occur in pairs, so let's just take the first entry in the regions list
        # * get region numbers and window coordinates
        pws_regions_coords = [
            [c[0][0], w]
            for c in connections
            if c[1] and c[2] == "periodic"
            for w in c[3]
        ]

        # wall_
        pws_flags = []  # * array to store the wall locatoin of the periodic windows
        # * now we have to find which wall the windwo belongs to
        frac_coords = get_fractional_coordinates(
            np.vstack([p[1] for p in pws_regions_coords]), mat_atoms
        )

        for i in range(3):
            wall_mask = np.logical_or(
                frac_coords[:, i] < dist_thresh, frac_coords[:, i] > 1 - dist_thresh
            )
            wall_indices = np.where(wall_mask)[0]
            wall_directions = np.where(frac_coords[wall_indices, i] < dist_thresh, 0, 1)
            pws_flags.extend(
                zip(wall_indices, [i] * len(wall_indices), wall_directions)
            )  # entries of type (R, a, w) where R is the region number, a is the wall direction 0, 1 or 2, and w is the wall index either 0 or 1
        pws_temp_list = []  # * make a list so that we can eliminate some windows if need be

        # * for example, if the rregional maxima is on the wall, then don't consider windows on that wall in that direction (only).
        # * if there are multiple windows for the same oregion on the same wall, take only the largest one

        # * now make the master array of which region and wall the periodic window exists
        for pwf in pws_flags:
            region_number = pws_regions_coords[pwf[0]][
                0
            ]  # get the region number from the list
            wall_index = (
                2 * pwf[1] + pwf[2] + 1
            )  # get the wall index , a=0:1, a=1:2, b=0:3, b=1:4, c=0:5, c=1:6 , 0 if for the region number
            # * note that a periodic window can be on multiple walls
            # regions_wall_windows[region_number][wall_index] = regions_pws[pwf[0]][1] # put the window coordinate in the right locatoin
            wall_window_radius = pws_window_radii[pwf[0]]
            # * the index will be the region_number - 1

            pws_temp_list.append([region_number, wall_index, wall_window_radius])

        # * find rows with the same reigon_number and wall_index, keep the row with the largest window radius
        pws_temp_list = np.array(pws_temp_list)
        pws_temp_list = pws_temp_list[np.argsort(pws_temp_list[:, 2])][
            ::-1
        ]  # sort by window radius
        pws_temp_, pws_unique_indices = np.unique(
            pws_temp_list[:, [0, 1]], axis=0, return_index=True
        )  # remove duplicates
        pws_temp_list = pws_temp_list[
            pws_unique_indices
        ]  # * now we only keep the largest window radius for a given region number on the same wall

        for ptl in pws_temp_list:
            # print('ptl is ', ptl)
            regions_wall_windows[int(ptl[0]) - 1][int(ptl[1])] = ptl[
                2
            ]  # regions_pws[pwf[0]][1] # put the window coordinate in the right locatoin

        # for c in connections:

    else:
        periodic_windows_exist = False
        periodic_connected_regions = []
        periodic_window_coords = []
        periodic_window_fractional_coords = []
        periodic_window_walls = []
        regions_wall_windows = np.full((regions.max(), 7), np.NaN)
        regions_wall_windows[:, 0] = np.arange(1, regions.max() + 1)
    # * handle the single extending regions
    for a in [0, 1, 2]:
        # print(a)
        ic(a, "Direction")
        # [] # * empty array to store all the paths in direction a
        # window_sizes[a] = [] # * empty array to store all the window sizes along all the paths in direction a

        llr = regions.take(indices=0, axis=a)  # * left wall
        rrr = regions.take(indices=shape[a] - 1, axis=a)  # * right wall
        corresponding_regions = np.unique(
            list(zip(llr.flatten(), rrr.flatten())), axis=0
        )

        # * remove rows that are all zero
        corresponding_regions = corresponding_regions[
            np.all(corresponding_regions != 0, axis=1)
        ]
        ic(corresponding_regions, "Map list for direction {}".format(a))
        # print("For direction {} the corresponding regions are {}".format(a, corresponding_regions))
        map_list.append(
            corresponding_regions
        )  # * add the corresponding regions to the map list for direction a

        # * find rows in which both the values are the same
        # * these are the regions that extend across the unit cell
        same_regions_indices = (
            corresponding_regions[:, 0] == corresponding_regions[:, 1]
        )
        if len(same_regions_indices) > 0:
            same_regions = corresponding_regions[same_regions_indices]

            # * find the size of the periodic windows attached to such regions
            # * but these need to be the periodic window in the correct direction also, so we will use the periodic_window_walls array for additional check

            paths_a = []
            path_windows_a = []
            path_lengths_a = []
            for s in same_regions:
                windows_same_region = [
                    c[4]
                    for i, c in enumerate(periodic_connections)
                    if np.all(c[0] == s)
                    and (
                        periodic_window_walls[i][a] == 0
                        or periodic_window_walls[i][a] == 1
                    )
                ]
                if len(windows_same_region) > 0:
                    path_windows_a.append(np.hstack(windows_same_region))
                    paths_a.append(s)
                    path_lengths_a.append(0)
            # print(paths_a, path_windows_a)

            # * for all the corresponding regions, if both of them are in the same periodic group, and connected in perpendicular direction,
            # * then add an edge between them as with path_weight 0 as long as both the regions are not he same
            # for c in corresponding_regions:
            #     if c[0] != c[1]:
            #         if np.any([set(c).issubset(p) for p in pbc_groups]):
            #             rag_temp.add_edge(c[0], c[1], path_weight=0)
            paths.append(paths_a)
            path_windows.append(path_windows_a)
            path_lengths.append(path_lengths_a)
    # * handle all the other end to end paths after adding pbc connections in perpendicular directions
    for a in [0, 1, 2]:
        print("For direction ", a)
        # get teh mapping list for that direction
        corresponding_regions = map_list[a]
        # print('Looking for paths in direction:', a)
        # print('corresponding regions are ', corresponding_regions)
        rag_temp = copy.deepcopy(rag)  # * make a copy of the rag

        # * add a property tot he edges called path_weight with value 1 for all internal edges
        nx.set_edge_attributes(rag_temp, 1, "path_weight")

        ic(rag_temp.edges(data="window_info"), "Edges in rag_temp initially")
        # map_list = map_list[a]

        # * now we will add all the periodic edges in the other two directions
        for j in [0, 1, 2]:
            if j != a:  # * choose the other two directions
                left_wall = regions.take(indices=0, axis=j)  # * left wall
                right_wall = regions.take(indices=shape[j] - 1, axis=j)  # * right wall
                llf = left_wall.flatten()
                rrf = right_wall.flatten()

                # * now for teh regions mapped in j direction, we need to add periodic connections with path_weight 0
                corresponding_regions_j = map_list[j]

                # * we need the overlap fractions of regions in the correspondence list
                # * we will use this check also before adding the periodic connection
                if len(corresponding_regions_j) > 0:
                    reg_numbers = np.vstack(
                        [(k, v) for k, v in corresponding_regions_j]
                    )

                    # * let's compute and prnt percentage overlap between pairs of mapping regions
                    # * let's filter those connections with less thant 10% overlap
                    sumlist = [np.sum(llf == r[0]) for r in reg_numbers]
                    matching = [
                        np.sum(np.logical_and(llf == r[0], rrf == r[1]))
                        for r in reg_numbers
                    ]
                    fractional_overlap = [m / s for m, s in zip(matching, sumlist)]

                    # * flags where there should be a left to right connection
                    left_right = [
                        f > minimum_overlap_fraction for f in fractional_overlap
                    ]

                    ic(a, "Direction")
                    ic(j, "Secondary direction")

                    # reg_numbers = reg_numbers[np.where([f > minimum_overlap_fraction for f in fractional_overlap])]
                    # print('left-right\n')
                    # print(list(zip(reg_numbers, fractional_overlap)))

                    # * backwards check

                    # #* let's compute and prnt percentage overlap between pairs of mapping regions
                    # #* let's filter those connections with less thant minimum overlap fraction
                    sumlist = [np.sum(rrf == r[1]) for r in reg_numbers]
                    matching = [
                        np.sum(np.logical_and(llf == r[0], rrf == r[1]))
                        for r in reg_numbers
                    ]
                    fractional_overlap = [m / s for m, s in zip(matching, sumlist)]

                    # * flags where there should be a right to left connection
                    right_left = [
                        f > minimum_overlap_fraction for f in fractional_overlap
                    ]

                    # print('right-left\n')
                    # print(list(zip(reg_numbers, fractional_overlap)))
                    # now let's combine the two flags
                    pick_these = np.logical_or(left_right, right_left)

                    reg_numbers = reg_numbers[pick_these]
                    ic(
                        reg_numbers,
                        "Regions to add periodic connections between for secondary direction {}".format(
                            j
                        ),
                    )
                    # print('Secondary direction is ', j)
                    # print('corresponding regions are ', corresponding_regions_j)
                    # print('Overlap fractions are ', fractional_overlap)
                    # print('Regions that meet overlap criteria are ', reg_numbers)
                    # for c in corresponding_regions_j:
                    #     if c[0]!=c[1]:
                    #         print('c is ', c)
                    #         if np.any(np.all(reg_numbers==c, axis=1)):
                    #             print('This c:{} is in reg_numbers:{}'.format(c, reg_numbers))
                    #             # if np.any([set(c).issubset(p) for p in pbc_groups]):
                    #             # print('about to add periodic edge between ', c[0], c[1])
                    #             rag_temp.add_edge(c[0], c[1], path_weight=0)
                    for c in reg_numbers:
                        if c[0] != c[1]:
                            # print('c is ', c)
                            # if np.any(np.all(reg_numbers==c, axis=1)):
                            # print('This c:{} is in reg_numbers:{}'.format(c, reg_numbers))
                            # if np.any([set(c).issubset(p) for p in pbc_groups]):
                            # print('about to add periodic edge between ', c[0], c[1])

                            # if the periodic there is a window there, then path weight should be 1

                            if np.any(
                                [
                                    np.all([c[0], c[1]] == p)
                                    or np.all([c[1], c[0]] == p)
                                    for p in periodic_connected_regions
                                ]
                            ):
                                rag_temp.add_edge(c[0], c[1], path_weight=1)

                            # if there is a maxima at the wall, then the ppath weigth should be 0
                            else:
                                rag_temp.add_edge(c[0], c[1], path_weight=0)
                ic(
                    rag_temp.edges(data="path_weight"),
                    "Edges in rag_temp after direction {}".format(j),
                )
        # print('a is ', a)
        # print('edges are ', rag_temp.edges(data='path_weight'))
        # these are all the paths
        paths_a_ = list(nx.all_pairs_dijkstra_path(rag_temp, weight="path_weight"))
        path_lengths_a_ = list(
            nx.all_pairs_dijkstra_path_length(rag_temp, weight="path_weight")
        )
        print("Dijkstra paths are computed for direction {}".format(a))
        print("Computing simple paths for direction {}".format(a))
        # print(rag_temp.edges(data=True))
        # * what are the paths connecting any region in column 0 in corresponding regions to 'any' region in column 1 in corresponding regions
        paths_a = []
        windows_a = []
        path_lengths_a = []

        # corresponding_non_matching = np.vstack([[s,t] for s,t in corresponding_regions if s!=t])
        # sources = corresponding_non_matching[:,0]
        # targets = corresponding_non_matching[:,1]

        # import dask.bag as db
        # all_paths = db.from_product([sources, targets]).map(lambda x: list(nx.all_simple_paths(rag_temp, source=x[0], target=x[1], cutoff=path_jump_cutoff))).compute()
        # Get the unique nodes from the second column of corresponding_regions
        targets = np.unique(corresponding_regions[:, 1])

        for llr in tqdm(corresponding_regions[:, 0]):
            if llr not in targets:
                all_possible_paths = nx.all_simple_paths(
                    rag_temp, source=llr, target=targets, cutoff=path_jump_cutoff
                )
                # print('All possible paths from {} to targets are {}'.format(llr, all_possible_paths))

                for path in all_possible_paths:
                    rrr = path[-1]  # Get the target node from the path
                    # print(rrr, 'is the target node')
                    if llr != rrr:
                        # Rest of the code remains the same
                        # for llr in corresponding_regions[:,0]:
                        #     for rrr in corresponding_regions[:,1]:

                        #         # * what if we looked for allthe simple paths between the two regions and then looked for the PLDs
                        #         if llr != rrr:

                        #             # all_possible_paths = list(nx.all_simple_paths(rag_temp, source =llr, target = rrr, cutoff=path_jump_cutoff))
                        #             all_possible_paths = nx.all_simple_paths(rag_temp, source =llr, target = rrr, cutoff=path_jump_cutoff)
                        #             print('All possible paths between {} and {} are {}'.format(llr, rrr, all_possible_paths))
                        #             #* append the path windows and path length
                        #             for path in all_possible_paths:
                        paths_a.append(path)
                        # subgraph   = rag_temp.subgraph(path)

                        edges_to_select = list(more_itertools.pairwise(path))
                        subgraph = rag_temp.edge_subgraph(edges_to_select)
                        window_radii = [
                            np.array([np.max(c[4])])
                            for e in subgraph.edges()
                            for c in connections
                            if c[1]
                            and (
                                np.all(c[0] == [e[0], e[1]])
                                or np.all(c[0] == [e[1], e[0]])
                            )
                        ]

                        # * append the path lengths
                        column_index = (
                            a * 2 + 1
                        )  # column 1-6 are the windwo radii columns
                        if not np.isnan(
                            regions_wall_windows[llr - 1][column_index]
                        ) and (rag.nodes[llr]["maxima_indices"][a] > wall_cells):
                            ic(llr, regions_wall_windows[llr - 1][column_index])
                            window_radii.append(
                                regions_wall_windows[llr - 1][column_index]
                            )

                        if not np.isnan(
                            regions_wall_windows[rrr - 1][column_index + 1]
                        ) and (
                            rag.nodes[rrr]["maxima_indices"][a]
                            < regions.shape[a] - wall_cells
                        ):
                            ic(rrr, regions_wall_windows[rrr - 1][column_index + 1])
                            window_radii.append(
                                regions_wall_windows[rrr - 1][column_index + 1]
                            )

                        windows_a.append(window_radii)
                        edge_attrs = nx.get_edge_attributes(subgraph, "path_weight")
                        path_weight_sum = sum(edge_attrs.values())
                        path_lengths_a.append(path_weight_sum)

                    if llr != rrr and rrr in paths_a_[llr - 1][1].keys():
                        path_of_interest = paths_a_[llr - 1][1][rrr]
                        paths_a.append(path_of_interest)
                        edges_to_select = list(
                            more_itertools.pairwise(path_of_interest)
                        )

                        subgraph = rag_temp.edge_subgraph(edges_to_select)

                        # * look throug the connectins an find all the windows along hte way, internal, and periodic in the other two directions
                        window_radii = [
                            np.array([np.max(c[4])])
                            for e in subgraph.edges()
                            for c in connections
                            if c[1]
                            and (
                                np.all(c[0] == [e[0], e[1]])
                                or np.all(c[0] == [e[1], e[0]])
                            )
                        ]
                        # window_radii = [np.max(subgraph.edges[e]['window_info']['window_radii']) for e in subgraph.edges() if subgraph.edges[e]['path_weight']==1]
                        ic(subgraph.edges(data="window_info"), "Edges in subgraph")
                        ic([llr, rrr], "Start and end regions")
                        ic(window_radii, "Window radii")

                        # * also if both regions are exact mapping on the walls, then check for a periodic window also between and add that to the path
                        # if [llr, rrr] in corresponding_regions:

                        # * there also might be left window at llr and a right window at rrr in that
                        # * given direction. We need to pick that correctly using the master window location array
                        # * pick the window only if the region maxima is not on the wall

                        column_index = (
                            a * 2 + 1
                        )  # column 1-6 are the windwo radii columns
                        if not np.isnan(
                            regions_wall_windows[llr - 1][column_index]
                        ) and (rag.nodes[llr]["maxima_indices"][a] > wall_cells):
                            ic(llr, regions_wall_windows[llr - 1][column_index])
                            window_radii.append(
                                regions_wall_windows[llr - 1][column_index]
                            )

                        if not np.isnan(
                            regions_wall_windows[rrr - 1][column_index + 1]
                        ) and (
                            rag.nodes[rrr]["maxima_indices"][a]
                            < regions.shape[a] - wall_cells
                        ):
                            ic(rrr, regions_wall_windows[rrr - 1][column_index + 1])
                            window_radii.append(
                                regions_wall_windows[rrr - 1][column_index + 1]
                            )

                        windows_a.append(
                            window_radii
                        )  # once you have all the window diameters, append it to the paths along a list.
                        path_lengths_a.append(
                            path_lengths_a_[llr - 1][1][rrr]
                        )  # append the path length also

        # print('Paths along direction {} after adding pbc are\n {}'.format(a, paths_a))
        # print('Path lengths along direction {} after adding pbc are\n {}'.format(a, path_lengths_a))
        # print('Window sizes along direction {} after adding pbc are\n {}'.format(a, windows_a))
        ic(paths_a, "Paths along direction {} after adding pbc".format(a))
        ic(path_lengths_a, "Path lengths along direction {} after adding pbc".format(a))
        paths.append(paths_a)
        path_windows.append(windows_a)
        path_lengths.append(path_lengths_a)
    #     #* now we need to find the windows along these paths, just get the window sizes when the path_weight of that edge is 1 else it is a periodic window between same regions

    # * let's do some reshaping
    path_windows_list = [
        np.hstack(pws).tolist() if len(pws) > 0 else []
        for pws_ in path_windows
        for pws in pws_
    ]
    paths_list = [pws for pws_ in paths for pws in pws_]
    path_lengths_list = [pws for pws_ in path_lengths for pws in pws_]

    # paths_ = [p for pw_a in paths for p in pw_a ]
    # path_windows_ = [pw for pw_a in path_windows for pw in pw_a]
    # path_lengths_= [pw for pw_a in  path_lengths for pw in pw_a]

    # # print('Paths before clean up' )
    # # print(show_percolated_paths_as_df(paths_, path_windows_, path_lengths_))

    # # * remove paths with path weight 0 and the nodes are not the same
    paths_final = []
    path_windows_final = []
    path_lengths_final = []

    # for p, pw, pl in zip(paths_, path_windows_, path_lengths_):
    for p, pw, pl in zip(paths_list, path_windows_list, path_lengths_list):
        if pl > 0:
            paths_final.append(p)
            path_windows_final.append(pw)
            path_lengths_final.append(pl)
        else:
            # path should only have 2 nodes and both should be the same
            if (len(p) == 2 and p[0] == p[1]) or len(pw) > 0:
                paths_final.append(p)
                path_windows_final.append(pw)
                path_lengths_final.append(pl)

    return paths_final, path_windows_final, path_lengths_final


# def get_percolated_paths_on_rag(rag, minimum_overlap_fraction=0.5, wall_cells=6, path_jump_cutoff=None):

#     import networkx as nx
#     from .helper import get_fractional_coordinates
#     import numpy as np
#     from icecream import ic
#     import copy
#     import pandas as pd
#     import more_itertools
#     keys_to_check = ['regions', 'connections', 'ase_atoms']

#     for key in keys_to_check:
#         if key not in rag.graph.keys():
#             raise ValueError(f"'{key}' is not in rag.graph.keys(). Please compute and add it first to the rag under the graph dictionary like this: rag.graph['{key}'] = {key}.")
#     regions = rag.graph['regions']
#     mat_atoms = rag.graph['ase_atoms']
#     connections = rag.graph['connections']
#     shape = regions.shape
#     map_list = []
#     paths = []
#     path_windows= []
#     path_lengths = []
#     periodic_connections  = [c for c in connections if c[1] and c[2]=='periodic']

#     if path_jump_cutoff is None:
#         # path_jump_cutoff = np.max([int(np.cbrt(np.max(regions))), 4])
#         ranges = [(0, 20), (20, 60), (60, float('inf'))]
#         outputs = [4, 8, 10]

#         # Create the dictionary
#         path_jump_cutoff = pd.cut([10], bins=[r[0] for r in ranges] + [float('inf')], labels=outputs)[0]


#     if len(periodic_connections) > 0:
#         periodic_windows_exist = True
#         periodic_connected_regions= np.array([pc[0] for pc in periodic_connections]) # * just the indices of the connected regions, just needed to make sure that we don't miss any periodic windows between the same regions
#         periodic_window_coords = np.vstack([c[3] for c in periodic_connections])
#         periodic_window_fractional_coords = get_fractional_coordinates(periodic_window_coords, mat_atoms)

#         # * we need to know which wall these windwos are on, round values close to 0 to 0 and values close to 1 to 1
#         #* we will use this as an additional check when getting the correct window diameters
#         periodic_window_walls  = np.where(periodic_window_fractional_coords < 0.03, 0, np.where(periodic_window_fractional_coords > 0.97, 1, periodic_window_fractional_coords))


#         # * this bit is for creating a master list of where the periodic windows exist
#         dist_thresh = 0.03

#         # * let's make an array to store the wall window location
#         regions_wall_windows = np.full((regions.max(), 7), np.NaN)
#         regions_wall_windows[:, 0] = np.arange(1, regions.max() + 1)


#         #* get the periodic window radii
#         pws_window_radii = [w for c in connections if c[1] and c[2]=='periodic' for w in c[4]]

#         # * periodic windows occur in pairs, so let's just take the first entry in the regions list
#         #* get region numbers and window coordinates
#         pws_regions_coords = ([[c[0][0], w] for c in connections if c[1] and c[2]=='periodic' for w in c[3]])


#         # wall_
#         pws_flags = [] # * array to store the wall locatoin of the periodic windows
#         # * now we have to find which wall the windwo belongs to
#         frac_coords= get_fractional_coordinates(np.vstack([p[1] for p in pws_regions_coords]), mat_atoms)

#         for i in range(3):
#             wall_mask = np.logical_or(frac_coords[:, i] < dist_thresh, frac_coords[:, i] > 1 - dist_thresh)
#             wall_indices = np.where(wall_mask)[0]
#             wall_directions = np.where(frac_coords[wall_indices, i] < dist_thresh, 0, 1)
#             pws_flags.extend(zip(wall_indices, [i] * len(wall_indices), wall_directions)) # entries of type (R, a, w) where R is the region number, a is the wall direction 0, 1 or 2, and w is the wall index either 0 or 1
#         pws_temp_list = [] # * make a list so that we can eliminate some windows if need be

#         #* for example, if the rregional maxima is on the wall, then don't consider windows on that wall in that direction (only).
#         #* if there are multiple windows for the same oregion on the same wall, take only the largest one

#         # * now make the master array of which region and wall the periodic window exists
#         for pwf in pws_flags:
#             region_number = pws_regions_coords[pwf[0]][0] # get the region number from the list
#             wall_index = 2*pwf[1] + pwf[2] + 1 # get the wall index , a=0:1, a=1:2, b=0:3, b=1:4, c=0:5, c=1:6 , 0 if for the region number
#             # * note that a periodic window can be on multiple walls
#             # regions_wall_windows[region_number][wall_index] = regions_pws[pwf[0]][1] # put the window coordinate in the right locatoin
#             wall_window_radius = pws_window_radii[pwf[0]]
#             # * the index will be the region_number - 1

#             pws_temp_list.append([region_number, wall_index, wall_window_radius])

#         #* find rows with the same reigon_number and wall_index, keep the row with the largest window radius
#         pws_temp_list = np.array(pws_temp_list)
#         pws_temp_list = pws_temp_list[np.argsort(pws_temp_list[:,2])][::-1] # sort by window radius
#         pws_temp_, pws_unique_indices = np.unique(pws_temp_list[:,[0,1]], axis=0,return_index=True) # remove duplicates
#         pws_temp_list = pws_temp_list[pws_unique_indices] # * now we only keep the largest window radius for a given region number on the same wall

#         for ptl in pws_temp_list:
#             # print('ptl is ', ptl)
#             regions_wall_windows[int(ptl[0]) -1 ][int(ptl[1])] = ptl[2] # regions_pws[pwf[0]][1] # put the window coordinate in the right locatoin

#         # for c in connections:

#     else:
#         periodic_windows_exist = False
#         periodic_connected_regions = []
#         periodic_window_coords = []
#         periodic_window_fractional_coords = []
#         periodic_window_walls = []
#         regions_wall_windows = np.full((regions.max(),7), np.NaN)
#         regions_wall_windows[:,0] = np.arange(1, regions.max()+1)
#     # * handle the single extending regions
#     for a in [0,1,2]:
#         # print(a)
#         ic(a, 'Direction')
#         # [] # * empty array to store all the paths in direction a
#         # window_sizes[a] = [] # * empty array to store all the window sizes along all the paths in direction a

#         llr = regions.take(indices=0 ,axis=a) #* left wall
#         rrr = regions.take(indices=shape[a]-1 ,axis=a)  #* right wall
#         corresponding_regions = np.unique(list(zip(llr.flatten(), rrr.flatten())),axis=0)


#         # * remove rows that are all zero
#         corresponding_regions = corresponding_regions[np.all(corresponding_regions!=0, axis=1)]
#         ic(corresponding_regions, 'Map list for direction {}'.format(a))
#         # print("For direction {} the corresponding regions are {}".format(a, corresponding_regions))
#         map_list.append(corresponding_regions) # * add the corresponding regions to the map list for direction a

#         # * find rows in which both the values are the same
#         # * these are the regions that extend across the unit cell
#         same_regions_indices = corresponding_regions[:,0]==corresponding_regions[:,1]
#         if len(same_regions_indices) > 0:
#             same_regions = corresponding_regions[same_regions_indices]

#             # * find the size of the periodic windows attached to such regions
#             # * but these need to be the periodic window in the correct direction also, so we will use the periodic_window_walls array for additional check

#             paths_a = []
#             path_windows_a= []
#             path_lengths_a = []
#             for s in same_regions:
#                 windows_same_region = [c[4] for i, c in enumerate(periodic_connections) if np.all(c[0]==s) and (periodic_window_walls[i][a]==0 or periodic_window_walls[i][a]==1)]
#                 if len(windows_same_region) > 0:
#                     path_windows_a.append(np.hstack(windows_same_region))
#                     paths_a.append(s)
#                     path_lengths_a.append(0)
#         # print(paths_a, path_windows_a)


#             #* for all the corresponding regions, if both of them are in the same periodic group, and connected in perpendicular direction,
#             # * then add an edge between them as with path_weight 0 as long as both the regions are not he same
#             # for c in corresponding_regions:
#             #     if c[0] != c[1]:
#             #         if np.any([set(c).issubset(p) for p in pbc_groups]):
#             #             rag_temp.add_edge(c[0], c[1], path_weight=0)
#             paths.append(paths_a)
#             path_windows.append(path_windows_a)
#             path_lengths.append(path_lengths_a)
#     # * handle all the other end to end paths after adding pbc connections in perpendicular directions
#     for a in [0,1,2]:
#         # get teh mapping list for that direction
#         corresponding_regions = map_list[a]
#         # print('Looking for paths in direction:', a)
#         # print('corresponding regions are ', corresponding_regions)
#         rag_temp = copy.deepcopy(rag) #* make a copy of the rag

#         #* add a property tot he edges called path_weight with value 1 for all internal edges
#         nx.set_edge_attributes(rag_temp, 1, 'path_weight')

#         ic(rag_temp.edges(data='window_info'), 'Edges in rag_temp initially')
#         # map_list = map_list[a]

#         # * now we will add all the periodic edges in the other two directions
#         for j in [0,1,2]:
#             if j != a: # * choose the other two directions
#                 left_wall = regions.take(indices=0 ,axis=j) #* left wall
#                 right_wall = regions.take(indices=shape[j]-1 ,axis=j)  #* right wall
#                 llf= left_wall.flatten()
#                 rrf= right_wall.flatten()

#                 #* now for teh regions mapped in j direction, we need to add periodic connections with path_weight 0
#                 corresponding_regions_j = map_list[j]

#                 #* we need the overlap fractions of regions in the correspondence list
#                 #* we will use this check also before adding the periodic connection
#                 if len(corresponding_regions_j) > 0:
#                     reg_numbers = np.vstack([(k, v) for k,v in corresponding_regions_j])


#                     #* let's compute and prnt percentage overlap between pairs of mapping regions
#                     #* let's filter those connections with less thant 10% overlap
#                     sumlist = [np.sum(llf==r[0]) for r in reg_numbers]
#                     matching = [np.sum(np.logical_and(llf==r[0], rrf==r[1])) for r in reg_numbers]
#                     fractional_overlap = [m/s for m,s in zip(matching, sumlist)]

#                     #* flags where there should be a left to right connection
#                     left_right = [f> minimum_overlap_fraction for f in fractional_overlap]

#                     ic(a, 'Direction')
#                     ic(j, 'Secondary direction')

#                     # reg_numbers = reg_numbers[np.where([f > minimum_overlap_fraction for f in fractional_overlap])]
#                     # print('left-right\n')
#                     # print(list(zip(reg_numbers, fractional_overlap)))

#                     # * backwards check

#                     # #* let's compute and prnt percentage overlap between pairs of mapping regions
#                     # #* let's filter those connections with less thant minimum overlap fraction
#                     sumlist = [np.sum(rrf==r[1]) for r in reg_numbers]
#                     matching = [np.sum(np.logical_and(llf==r[0], rrf==r[1])) for r in reg_numbers]
#                     fractional_overlap = [m/s for m,s in zip(matching, sumlist)]

#                     #* flags where there should be a right to left connection
#                     right_left = [f> minimum_overlap_fraction for f in fractional_overlap]

#                     # print('right-left\n')
#                     # print(list(zip(reg_numbers, fractional_overlap)))
#                     # now let's combine the two flags
#                     pick_these = np.logical_or(left_right, right_left)

#                     reg_numbers = reg_numbers[pick_these]
#                     ic(reg_numbers, 'Regions to add periodic connections between for secondary direction {}'.format(j))
#                     # print('Secondary direction is ', j)
#                     # print('corresponding regions are ', corresponding_regions_j)
#                     # print('Overlap fractions are ', fractional_overlap)
#                     # print('Regions that meet overlap criteria are ', reg_numbers)
#                     # for c in corresponding_regions_j:
#                     #     if c[0]!=c[1]:
#                     #         print('c is ', c)
#                     #         if np.any(np.all(reg_numbers==c, axis=1)):
#                     #             print('This c:{} is in reg_numbers:{}'.format(c, reg_numbers))
#                     #             # if np.any([set(c).issubset(p) for p in pbc_groups]):
#                     #             # print('about to add periodic edge between ', c[0], c[1])
#                     #             rag_temp.add_edge(c[0], c[1], path_weight=0)
#                     for c in reg_numbers:
#                         if c[0]!=c[1]:
#                             # print('c is ', c)
#                             # if np.any(np.all(reg_numbers==c, axis=1)):
#                             # print('This c:{} is in reg_numbers:{}'.format(c, reg_numbers))
#                             # if np.any([set(c).issubset(p) for p in pbc_groups]):
#                             # print('about to add periodic edge between ', c[0], c[1])

#                             # if the periodic there is a window there, then path weight should be 1

#                             if np.any([np.all([c[0], c[1]]==p)  or np.all([c[1], c[0]]==p) for p in periodic_connected_regions]):
#                                 rag_temp.add_edge(c[0], c[1], path_weight=1)

#                             # if there is a maxima at the wall, then the ppath weigth should be 0
#                             else:
#                                 rag_temp.add_edge(c[0], c[1], path_weight=0)
#                 ic(rag_temp.edges(data='path_weight'), 'Edges in rag_temp after direction {}'.format(j))
#         # print('a is ', a)
#         # print('edges are ', rag_temp.edges(data='path_weight'))
#         # these are all the paths
#         paths_a_ = list(nx.all_pairs_dijkstra_path(rag_temp, weight='path_weight'))
#         path_lengths_a_ = list(nx.all_pairs_dijkstra_path_length(rag_temp, weight='path_weight'))
#         # print(rag_temp.edges(data=True))
#         #* what are the paths connecting any region in column 0 in corresponding regions to 'any' region in column 1 in corresponding regions
#         paths_a = []
#         windows_a = []
#         path_lengths_a = []

#         # corresponding_non_matching = np.vstack([[s,t] for s,t in corresponding_regions if s!=t])
#         # sources = corresponding_non_matching[:,0]
#         # targets = corresponding_non_matching[:,1]

#         # import dask.bag as db
#         # all_paths = db.from_product([sources, targets]).map(lambda x: list(nx.all_simple_paths(rag_temp, source=x[0], target=x[1], cutoff=path_jump_cutoff))).compute()

#         for llr in corresponding_regions[:,0]:
#             for rrr in corresponding_regions[:,1]:

#                 # * what if we looked for allthe simple paths between the two regions and then looked for the PLDs
#                 if llr != rrr:
#                     all_possible_paths = list(nx.all_simple_paths(rag_temp, source =llr, target = rrr, cutoff=path_jump_cutoff))

#                     #* append the path windows and path length
#                     for path in all_possible_paths:
#                         paths_a.append(path)
#                         # subgraph   = rag_temp.subgraph(path)

#                         edges_to_select = list(more_itertools.pairwise(path))
#                         subgraph = rag_temp.edge_subgraph(edges_to_select)
#                         window_radii = [np.array([np.max(c[4])])  for e in subgraph.edges() for c in connections if c[1] and (np.all(c[0]==[e[0],e[1]]) or np.all(c[0]==[e[1],e[0]])) ]

#                         #* append the path lengths
#                         column_index = a*2+1 # column 1-6 are the windwo radii columns
#                         if not np.isnan(regions_wall_windows[llr-1][column_index]) and (rag.nodes[llr]['maxima_indices'][a]>wall_cells):
#                             ic(llr, regions_wall_windows[llr-1][column_index])
#                             window_radii.append(regions_wall_windows[llr-1][column_index])

#                         if not np.isnan(regions_wall_windows[rrr-1][column_index+1]) and (rag.nodes[rrr]['maxima_indices'][a] < regions.shape[a] - wall_cells):
#                             ic(rrr, regions_wall_windows[rrr-1][column_index+1])
#                             window_radii.append(regions_wall_windows[rrr-1][column_index+1])

#                         windows_a.append(window_radii)
#                         edge_attrs = nx.get_edge_attributes(subgraph, 'path_weight')
#                         path_weight_sum = sum(edge_attrs.values())
#                         path_lengths_a.append(path_weight_sum)


#                 if llr != rrr and rrr in paths_a_[llr-1][1].keys():
#                     path_of_interest = paths_a_[llr-1][1][rrr]
#                     paths_a.append(path_of_interest)
#                     edges_to_select = list(more_itertools.pairwise(path_of_interest))

#                     subgraph = rag_temp.edge_subgraph(edges_to_select)


#                     #* look throug the connectins an find all the windows along hte way, internal, and periodic in the other two directions
#                     window_radii = [np.array([np.max(c[4])])  for e in subgraph.edges() for c in connections if c[1] and (np.all(c[0]==[e[0],e[1]]) or np.all(c[0]==[e[1],e[0]])) ]
#                     # window_radii = [np.max(subgraph.edges[e]['window_info']['window_radii']) for e in subgraph.edges() if subgraph.edges[e]['path_weight']==1]
#                     ic(subgraph.edges(data='window_info'), 'Edges in subgraph')
#                     ic([llr, rrr], 'Start and end regions')
#                     ic(window_radii, 'Window radii')

#                     # * also if both regions are exact mapping on the walls, then check for a periodic window also between and add that to the path
#                     # if [llr, rrr] in corresponding_regions:


#                     # * there also might be left window at llr and a right window at rrr in that
#                     # * given direction. We need to pick that correctly using the master window location array
#                     # * pick the window only if the region maxima is not on the wall

#                     column_index = a*2+1 # column 1-6 are the windwo radii columns
#                     if not np.isnan(regions_wall_windows[llr-1][column_index]) and (rag.nodes[llr]['maxima_indices'][a]>wall_cells):
#                         ic(llr, regions_wall_windows[llr-1][column_index])
#                         window_radii.append(regions_wall_windows[llr-1][column_index])

#                     if not np.isnan(regions_wall_windows[rrr-1][column_index+1]) and (rag.nodes[rrr]['maxima_indices'][a] < regions.shape[a] - wall_cells):
#                         ic(rrr, regions_wall_windows[rrr-1][column_index+1])
#                         window_radii.append(regions_wall_windows[rrr-1][column_index+1])


#                     windows_a.append(window_radii) # once you have all the window diameters, append it to the paths along a list.
#                     path_lengths_a.append(path_lengths_a_[llr-1][1][rrr]) # append the path length also


#         # print('Paths along direction {} after adding pbc are\n {}'.format(a, paths_a))
#         # print('Path lengths along direction {} after adding pbc are\n {}'.format(a, path_lengths_a))
#         # print('Window sizes along direction {} after adding pbc are\n {}'.format(a, windows_a))
#         ic(paths_a, 'Paths along direction {} after adding pbc'.format(a))
#         ic(path_lengths_a, 'Path lengths along direction {} after adding pbc'.format(a))
#         paths.append(paths_a)
#         path_windows.append(windows_a)
#         path_lengths.append(path_lengths_a)
#     #     #* now we need to find the windows along these paths, just get the window sizes when the path_weight of that edge is 1 else it is a periodic window between same regions

#     #* let's do some reshaping
#     path_windows_list = [np.hstack(pws).tolist() if len(pws)>0 else [] for pws_ in path_windows for pws in pws_]
#     paths_list = [pws for pws_ in paths for pws in pws_]
#     path_lengths_list = [pws for pws_ in path_lengths for pws in pws_]


#     # paths_ = [p for pw_a in paths for p in pw_a ]
#     # path_windows_ = [pw for pw_a in path_windows for pw in pw_a]
#     # path_lengths_= [pw for pw_a in  path_lengths for pw in pw_a]

#     # # print('Paths before clean up' )
#     # # print(show_percolated_paths_as_df(paths_, path_windows_, path_lengths_))

#     # # * remove paths with path weight 0 and the nodes are not the same
#     paths_final = []
#     path_windows_final = []
#     path_lengths_final = []

#     # for p, pw, pl in zip(paths_, path_windows_, path_lengths_):
#     for p, pw, pl in zip(paths_list, path_windows_list, path_lengths_list):
#         if pl > 0 :
#             paths_final.append(p)
#             path_windows_final.append(pw)
#             path_lengths_final.append(pl)
#         else:
#             # path should only have 2 nodes and both should be the same
#             if (len(p)==2 and p[0]==p[1]) or len(pw) > 0:
#                 paths_final.append(p)
#                 path_windows_final.append(pw)
#                 path_lengths_final.append(pl)

#     return paths_final, path_windows_final, path_lengths_final


def filter_percolated_paths(paths, path_windows, path_lengths, probe_size):
    """
    The function filters paths based on whether their corresponding path windows meet a certain size
    threshold.

    :param paths: The `paths` parameter in the `filter_percolated_paths` function is typically a list of
    lists, where each inner list represents a path. Each path consists of a series of points or
    pocket indices through which the path traverses
    :param path_windows: The `path_windows` parameter seems to be a list of lists, where each inner list
    contains arrays representing windows of a path. Each array contains the radii of the windows along the path.
    :param path_lengths: The `path_lengths` parameter in the `filter_percolated_paths` function likely
    represents the lengths of the paths, basically the number of edges or jumps in the path.
    :param probe_size: Probe size is a parameter that represents the minimum size required for a path to
    be considered "open". Paths with all windows greater than this probe size are considered open paths
    :return: The function `filter_percolated_paths` returns three lists: `open_paths`,
    `open_path_windows`, and `open_path_lengths`. These lists contain the paths, path windows, and path
    lengths respectively, that meet the condition of having all windows in the path windows greater than
    the `probe_size`.
    """

    import numpy as np

    # open_paths = len([[np.all(np.hstack(pw) > probe_size) if len(pw) > 0 else False for pw in pw_a] for pw_a in path_windows ])
    # open_paths_flags = [np.all(np.hstack(pw) > probe_size) if len(pw) > 0 else False  for pw_a in path_windows for pw in pw_a]
    open_paths_flags = [
        np.all(np.hstack(pw) > probe_size) if len(pw) > 0 else False
        for pw in path_windows
    ]
    # all_paths = [p for pw_a in paths for p in pw_a]
    # all_paths = [p for  in paths]
    open_paths = [p for i, p in enumerate(paths) if open_paths_flags[i]]
    open_path_windows = [pw for i, pw in enumerate(path_windows) if open_paths_flags[i]]
    open_path_lengths = [pl for i, pl in enumerate(path_lengths) if open_paths_flags[i]]
    return open_paths, open_path_windows, open_path_lengths


def get_pld_from_percolated_paths_path_windows(paths, path_windows):
    import numpy as np

    key_window_size_list = [np.min(pw) if len(pw) > 0 else 1e-4 for pw in path_windows]
    plr = np.max(key_window_size_list)
    # if return_pld_path_info:
    #     pld_path_info = ()
    #     return plr**2, paths[np.argmax(key_window_size_list)], path_windows[np.argmax(key_window_size_list)]
    # else:
    return plr * 2


def get_pld_for_rag(rag, return_pld_path_info=False, **kwargs):
    import numpy as np
    from ase.build import make_supercell

    for key in ["dgrid", "regions", "connections", "ase_atoms", "maxima"]:
        if key not in rag.graph.keys():
            raise ValueError(
                f"'{key}' is not in rag.graph.keys(). Please (compute and) add it first to the rag under the graph dictionary like this: rag.graph['{key}'] = {key}. If you're unsure how to calculate these, please refer to the basic tutorials linked on the GitHub README."
            )

    spacing = kwargs.get("spacing", 0.5)
    h = kwargs.get("h", 0.5)
    compactness = kwargs.get("compactness", 0)
    mask_thickness = kwargs.get("mask_thickness", 1.0)
    block_size = kwargs.get("block_size", 50000)
    n_neighbors = kwargs.get("n_neighbors", 4)
    precompute = kwargs.get("precompute", True)
    path_jump_cutoff = kwargs.get(
        "path_jump_cutoff",
    )
    # import mofography as mgr
    from skimage.feature import peak_local_max

    mat_atoms = rag.graph["ase_atoms"]
    nx_cells, ny_cells, nz_cells = get_repetitions_to_meet_cutoff(mat_atoms, cutoff=6)
    mat_atoms = make_supercell(
        mat_atoms, [[nx_cells, 0, 0], [0, ny_cells, 0], [0, 0, nz_cells]]
    )
    # mat_updated = mgr.update_with_covalent_bonds(mat_atoms)
    # ca =mat_updated
    ca = mat_atoms
    print("Number of atoms in the supercell:", len(ca))
    # spacing = 0.5
    # h = 0.5
    # compactness =  0
    # # mask_thickness = rag.graph['mask_thickness']
    # mask_thickness =  1.0
    # h = 0.25
    # spacing =1.0

    # # * make supercell just in case
    # na, nb, nc = mgr.get_repetitions_to_meet_cutoff(ca)
    # print('repetitions: ', na, nb, nc)
    # ca = make_supercell(ca, [[na, 0, 0], [0, nb, 0], [0, 0, nc]])

    # dgrid =dgrid_from_atoms(ca, spacing=spacing, block_size=block_size, n_neighbors=n_neighbors, precompute=precompute)
    # dgrid1 = zoom(dgrid, 2, mode = 'grid-wrap', order=1)
    dgrid1 = rag.graph["dgrid"]
    dgrid1 = np.tile(dgrid1, (nx_cells, ny_cells, nz_cells))
    max_indices = peak_local_max(
        dgrid1, min_distance=5, threshold_abs=2, exclude_border=False
    )
    if nx_cells == 1 and ny_cells == 1 and nz_cells == 1:
        print("No need to make supercell")
        regions = rag.graph["regions"]
        maxima = rag.graph["maxima"]
        maxima_radii = dgrid1[tuple(maxima.T)]
        connections = rag.graph["connections"]
    else:
        regions, maxima = regions_from_dgrid(
            dgrid=dgrid1,
            mask_thickness=mask_thickness,
            h=h,
            compactness=compactness,
            max_indices=max_indices.T,
        )
        # rm= regions_from_dgrid(dgrid=dgrid1, mask_thickness=mask_thickness, h=h, compactness=compactness)
        if len(max_indices) == 0:
            max_indices = None
            regions, maxima = regions_from_dgrid(
                dgrid=dgrid1,
                mask_thickness=mask_thickness,
                h=h,
                compactness=0,
                max_indices=None,
            )

        # regions = rm[0]
        # maxima = rm[1]
        maxima_radii = dgrid1[tuple(maxima.T)]

        # maxima_radii = get_maxima_radii_from_grid(dgrid, maxima)
        # rag_new =(mgr.get_rag_from_regions_and_grid)(
        # regions=regions, dgrid=dgrid1, maxima=maxima, ase_atoms=ca)
        print("-Computing connnections for mask_thickness:{}".format(mask_thickness))
        # * compute the connections using custom function instead of skimage
        # connections =connections_from_dgrid(dgrid1, ca, mask_thickness=mask_thickness, h=h, compactness=compactness, distance_threshold=1.0)
        connections = connections_from_regions_and_dgrid(
            regions,
            dgrid1,
            maxima,
            ase_atoms=ca,
            wall_windows=True,
            distance_threshold=1.0,
            wall_thickness=3 / spacing,
        )

    # * construct new rag from connections
    print("- Constructing pore graph")
    rag_new = rag_from_connections(connections, maxima)
    rag_new = add_window_info_to_rag(rag_new, connections)
    print("pore graph constructed")
    # * add maxima to rag
    rag_new = add_maxima_to_rag(rag_new, maxima, maxima_radii, regions.shape, ca)
    # * compute pbc groups and add it to rag
    pbc_groups = (apply_pbc)(
        regions=regions,
        maxima=maxima,
        maxima_radii=maxima_radii,
        ase_atoms=ca,
        return_conn=True,
    )
    print("pB applied")
    # # * add the pbc info to the rag
    rag_new = (add_pbc_to_rag)(rag=rag_new, pbc_groups=pbc_groups)
    rag_new.graph["mask_thickness"] = mask_thickness
    rag_new.graph["h"] = h
    rag_new.graph["compactness"] = compactness
    rag_new.graph["spacing"] = spacing
    rag_new.graph["n_neighbors"] = 4
    rag_new.graph["distance_threshold"] = 1.0
    rag_new.graph["block_size"] = 50000
    rag_new.graph["pbc_groups"] = pbc_groups
    # * compute percolated paths and path windows
    print("- Computing percolated paths")
    print("path jump cutoff is ", path_jump_cutoff)
    paths, path_windows, path_lengths = get_percolated_paths_on_rag(
        rag, path_jump_cutoff=path_jump_cutoff
    )
    iteration = 1
    while len(paths) == 0:
        print("Couldn't find paths the {} time around".format(iteration))
        print("Reducing mask thickness and trying again")
        mask_thickness = mask_thickness - 0.1
        rm = regions_from_dgrid(
            dgrid=dgrid1,
            mask_thickness=mask_thickness,
            h=h,
            compactness=0,
            max_indices=maxima.T,
        )
        regions = rm[0]
        maxima = rm[1]
        maxima_radii = dgrid1[tuple(maxima.T)]

        # maxima_radii = get_maxima_radii_from_grid(dgrid, maxima)
        # rag_new =(mgr.get_rag_from_regions_and_grid)(
        # regions=regions, dgrid=dgrid1, maxima=maxima, ase_atoms=ca)
        print("-Computing connnections for mask_thickness:{}".format(mask_thickness))
        # * compute the connections using custom function instead of skimage
        # connections =connections_from_dgrid(dgrid1, ca, mask_thickness=mask_thickness, h=h, compactness=compactness, distance_threshold=1.0)
        paths, path_windows, path_lengths = get_percolated_paths_on_rag(
            rag,
            wall_windows=True,
            distance_threshold=1.0,
            wall_thickness=3 / spacing,
            path_jump_cutoff=path_jump_cutoff,
        )
        #  =mgr.get_percolated_paths_on_rag(rag, wall_windows=True, distance_threshold=1.0, wall_thickness=3/spacing)

        print("- Constructing pore graph")
        # * construct new rag from connections
        rag_new = rag_from_connections(connections, maxima)
        rag_new = add_window_info_to_rag(rag_new, connections)
        print("pore graph constructed")
        # * add maxima to rag
        rag_new = add_maxima_to_rag(rag_new, maxima, maxima_radii, regions.shape, ca)
        print("- Computing PBC")
        # * compute pbc groups and add it to rag
        pbc_groups = (apply_pbc)(
            regions=regions,
            maxima=maxima,
            maxima_radii=maxima_radii,
            ase_atoms=ca,
            return_conn=True,
        )

        # # * add the pbc info to the rag
        rag_new = (add_pbc_to_rag)(rag=rag_new, pbc_groups=pbc_groups)
        rag_new.graph["mask_thickness"] = mask_thickness
        rag_new.graph["h"] = h
        rag_new.graph["compactness"] = compactness
        rag_new.graph["spacing"] = spacing
        rag_new.graph["n_neighbors"] = 4
        rag_new.graph["distance_threshold"] = 1.0
        rag_new.graph["block_size"] = 50000
        rag_new.graph["pbc_groups"] = pbc_groups
        # * compute percolated paths and path windows
        ic.disable()
        print("- Computing percolated paths")
        # paths, path_windows, path_lengths = percolated_paths_from_regions_and_connections(regions=regions, connections=connections, pbc_groups =pbc_groups, rag_new =rag_new, mat_atoms=ca, path_jump_cutoff=None)
        paths, path_windows, path_lengths = get_percolated_paths_on_rag(rag)
        iteration = iteration + 1
    pld = get_pld_from_percolated_paths_path_windows(paths, path_windows)
    if return_pld_path_info:
        return pld, paths, path_windows, path_lengths
    else:
        return pld


def connections_from_regions_and_dgrid(
    region_labels,
    dgrid,
    maxima,
    ase_atoms,
    wall_windows=True,
    distance_threshold=1.0,
    wall_thickness=2,
):
    """
    It takes a 3D array of region labels, a list of maxima, an ASE atoms object, and a distance
    threshold, and returns a list of connections between regions, and the centers of the windows between
    them

    :param region_labels: The labels of the regions, as returned by the skimage.measure.label function
    :param maxima: the maxima of the image
    :param ase_atoms: ASE atoms object
    :param wall_windows: If True, then the windows on the wall are also calculated, defaults to True
    (optional)
    :param distance_threshold: The distance threshold for the agglomerative clustering
    :return: A list of connections between regions.
    """

    def check_connected(regs):
        import numpy as np
        from icecream import ic

        ic.disable()
        from sklearn.cluster import AgglomerativeClustering

        i = regs[0]
        j = regs[1]
        ic.disable()
        ic(regs)

        check_flag = np.logical_or(
            np.logical_and(region_labels == i, outer.get(j)),
            np.logical_and(region_labels == j, outer.get(i)),
        )
        if np.sum(check_flag) > 0:
            # print("Edges found between "+str(i)+' and '+ str(j) + ' sum: '+str(np.sum(check_flag)))
            windices = np.vstack(np.where(check_flag)).T  # * (N,3)
            window_points = np.dot(
                A_unit, (windices / region_labels.shape).T
            ).T  # * (N,3)
            # ic(window_points)
            # * No clustering
            # window_centers = np.mean(window_points, axis=0)

            # * DBScan
            # X_scaled = StandardScaler().fit_transform(window_points)
            # db = DBSCAN(eps=0.5, min_samples=10).fit(X_scaled)
            # window_centers = np.vstack([window_points[db.labels_==i].mean(axis=0) for i in np.unique(db.labels_) if i != -1])

            # if len(window_points) >1:
            # * Agglomerative
            ag = AgglomerativeClustering(
                n_clusters=None,
                linkage="single",
                distance_threshold=minimum_window_separation,
                compute_full_tree=True,
            ).fit(window_points)
            # * we are taking the centroid of the clusters here as the window location
            # window_centers = np.vstack([window_points[ag.labels_==i].mean(axis=0) for i in np.unique(ag.labels_) if i != -1])

            # * what if we took the index of the point where the distance value is the largest in the window boundary region, that way we don't have to worry bout
            # * regions at the wall
            d_windices = dgrid[
                tuple(windices.T)
            ]  # distance values at the window indices
            r_windices = region_labels[
                tuple(windices.T)
            ]  # region labels at the window indices
            # window_centers = [(window_points[np.argmax(d_windices[np.logical_and(r_windices[ag.labels_ == uwi]==regs[0],ag.labels_==uwi)])] + window_points[np.argmax(d_windices[np.logical_and(r_windices[ag.labels_==uwi]==regs[1],ag.labels_==uwi)])])/2.0 for uwi in np.unique(ag.labels_) if uwi != -1]

            # print('These are window centers', window_centers)
            window_centers = []
            for uwi in np.unique(ag.labels_):
                # for each cluster that is nota noise, do this
                if uwi != -1:
                    # np.argmax(d_windices[ag.labels_==uwi])

                    uwi_dvals = d_windices[
                        ag.labels_ == uwi
                    ]  # distance values in the cluster with label uwi
                    uwi_points = window_points[
                        ag.labels_ == uwi
                    ]  # points in the cluster with label uwi
                    uwi_r = r_windices[ag.labels_ == uwi]

                    # for points in first region

                    # uwi_r1_indices =np.transpose((uwi_r==regs[0])).nonzero()
                    uwi_r1_indices = (np.where)(uwi_r == regs[0])[0]
                    # print('uwi_r1_indices are ', uwi_r1_indices)
                    # print(uwi_r1_indices.shape)
                    uwi_r1_dvals = uwi_dvals[uwi_r1_indices]
                    uwi_r1_points = uwi_points[uwi_r1_indices]
                    uwi_r1_max_indices = np.argmax(
                        uwi_r1_dvals
                    )  # np.where(uwi_r1_dvals == np.max(uwi_r1_dvals))[0] # indices of the maximum distance values in the cluster with label uwi
                    # print(uwi_r1_max_indices)
                    uwi_r1_max_points = uwi_r1_points[uwi_r1_max_indices]

                    # for points in second region
                    uwi_r2_indices = (np.where)(uwi_r == regs[1])[0]
                    uwi_r2_dvals = uwi_dvals[uwi_r2_indices]
                    uwi_r2_points = uwi_points[uwi_r2_indices]
                    uwi_r2_max_indices = np.where(
                        uwi_r2_dvals == np.max(uwi_r2_dvals)
                    )[
                        0
                    ]  # indices of the maximum distance values in the cluster with label uwi
                    uwi_r2_max_points = uwi_r2_points[uwi_r2_max_indices]

                    # take the centroid of the two maxima of each region close to the window
                    uwi_center = (uwi_r1_max_points + uwi_r2_max_points) / 2

                    # uwi_center = np.mean(uwi_max_points, axis=0) # mean of the points where the distance is at maximum if ther are multiple a.k.a the window is symmetric on each side
                    # add that to the list of windows
                    window_centers.append(uwi_center)

            # window_centers = np.vstack(window_centers)
            # else:
            # window_centers = window_points
            window_centers = np.vstack(window_centers)
            window_fractional = get_fractional_coordinates(window_centers, ase_atoms)
            window_fractional = np.round(window_fractional, decimals=4)
            window_radii = np.array(dinterp(window_fractional))
            return np.array(
                [regs, True, "internal", window_centers, window_radii], dtype="object"
            )  # returns a list of [[r1,r2], connected or not, array of window_centers]
        else:
            return np.array([regs, False, "internal", None, None], dtype="object")

    from skimage.segmentation import find_boundaries
    import numpy as np
    import itertools
    import ase

    # from sklearn.cluster import DBSCAN
    # from sklearn.preprocessing import StandardScaler
    from icecream import ic
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from mofography import interpolate_labels, get_fractional_coordinates
    import dask.bag as db

    A_unit = ase.geometry.complete_cell(ase_atoms.get_cell()).T
    from icecream import ic

    minimum_window_separation = distance_threshold

    dinterp = interpolate_dgrid(dgrid)
    # * Find the outer boundary of each individual region
    regions_to_check = np.delete(np.unique(region_labels), 0)
    ic(regions_to_check)
    # outer = {reg:find_boundaries(region_labels == reg, mode='outer') for reg in regions_to_check}
    outer = {
        reg: find_boundaries(region_labels == reg, mode="outer")
        for reg in regions_to_check
    }
    ic(len(outer))
    # * Check for connectedness between regions
    check_list = np.vstack(
        list(itertools.combinations(regions_to_check, 2))
    )  # list of pairs of region labels
    connections_list = [
        check_connected(c) for c in check_list
    ]  # now this step could cause an overflow
    # connections_list = db.from_sequence(check_list).map(check_connected).compute()

    # connected_regions = [[c[0], [c[2]] for c in connections_list if c[1] == True] # Make a new list with all the

    # * Windows on the wall are to be calculated
    if wall_windows:
        wwl = get_wall_windows2(
            regions=region_labels,
            maxima=maxima,
            dgrid=dgrid,
            ase_atoms=ase_atoms,
            wall_thickness=wall_thickness,
        )
        if len(wwl) > 0:
            connections_list.append(wwl)

    # #* we have one final thing to do
    # # * if both the regions forming an internal window are on the same wall, then we need to set the window center and the radius to
    # # * that on the wall, since centroid

    # for c in connections_list:
    #     if np.logical_and(c[1], c[2]=='internal'):

    return np.vstack(connections_list)


def add_centroids_to_rag_with_pbc(rag, props, pbc_groups, ase_atoms, shape):
    """
    > For each group of nodes that are connected by periodic boundary conditions, find the centroid of
    the group and add it to the graph as a node attribute.

    The function takes in a graph, a list of properties, a list of groups of nodes that are connected by
    periodic boundary conditions, an ASE atoms object, and the shape of the image.

    The function first makes a copy of the graph. Then, it loops through each group of nodes that are
    connected by periodic boundary conditions. For each group, it finds the indices of the properties
    that correspond to the nodes in the group. If there is more than one node in the group, it finds the
    centroid of the group by taking the circular mean of the coordinates of the nodes in the group. If
    there is only one node in the group, it finds the centroid of the node by taking the mean of the
    coordinates of the node. Finally, it adds the centroid of the group

    :param rag: the Region Adjacency Graph object from skimage (skimage.future.graph.rag class)
    :param props: a list of dictionaries, each dictionary contains the label and the coordinates of the
    centroid of the cluster
    :param pbc_groups: a list of lists, where each list contains the labels of the atoms that are
    connected by periodic boundary conditions
    :param ase_atoms: the atoms object from ASE
    :param shape: shape of the image
    :return: The rag is being returned with the centroid_pbc added to the nodes.
    """

    from scipy import stats
    import numpy as np
    import copy
    from .helper import find_coord_from_indices

    G = copy.deepcopy(rag)
    prop_labels = np.array([prop["label"] for prop in props])
    for group in pbc_groups: 
        prop_indices = [np.where(prop_labels == g)[0][0] for g in group]
        # print(group, prop_indices)
        if len(group) > 1:
            coords = np.vstack([props[pi]["coords"] for pi in prop_indices]) / shape
            # print(coords.shape)
            centroid = stats.circmean(coords, axis=0, low=0, high=1)
            for i, g in enumerate(group):
                G.nodes[props[prop_indices[i]]["label"]]["centroid_pbc"] = (
                    find_coord_from_indices(
                        indices=centroid * shape, ase_atoms=ase_atoms, shape=shape
                    )
                )
        else:
            pi = np.where(prop_labels == list(group)[0])[0][0]
            # print(pi)
            coords = props[pi]["coords"] / shape
            centroid = np.mean(coords, axis=0)
            G.nodes[props[pi]["label"]]["centroid_pbc"] = find_coord_from_indices(
                indices=centroid * shape, ase_atoms=ase_atoms, shape=shape
            )
    return G


def add_psd_to_rag_with_pbc(rag, props, pbc_groups, bins, ase_atoms, shape):
    """
    > It takes a rag, a list of properties, a list of pbc groups, a list of bins, an ase atoms object
    and a shape and returns a region adjacency graph with pbc properties added to it.

    Let's see how we can use this function.

    First, we need to get the pbc groups.

    We can do that using the `get_pbc_groups` function and setting return_conn option to True (it is False by default)

    Let's see how we can use it.

    :param rag: the region adjacency graph
    :param props: list of dictionaries containing the properties of each region
    :param pbc_groups: list of lists of labels of regions that are connected by periodic boundary
    conditions
    :param bins: the bins for the histogram
    :param ase_atoms: ASE atoms object
    :param shape: shape of the image
    :return: The graph with the pbc properties added to the nodes.
    """

    def get_pbc_distances(coords, centroid_pbc, ase_atoms):
        """
        > The function takes the coordinates of the grid points in the region, the centroid of the region, and
        the atoms object, and returns the distances of the grid points from the periodic region centroid

        :param coords: the coordinates of the atoms in the cluster
        :param centroid_pbc: the centroid of the cluster in the PBC frame
        :param ase_atoms: ASE atoms object
        :return: The distances between the centroid and the atoms in the cluster.
        """

        from ase.geometry import complete_cell

        cell = complete_cell(ase_atoms.get_cell()).T
        # cell_inv = np.linalg.inv(cell)
        df = coords - centroid_pbc
        df = df - np.round(df)
        distances = np.linalg.norm(np.dot(cell, df.T).T, axis=1)
        return distances

    from scipy import stats
    import numpy as np
    import copy
    from .helper import find_coord_from_indices

    # shape = regions.shape
    G = copy.deepcopy(rag)
    prop_labels = np.array([prop["label"] for prop in props])
    for group in pbc_groups:
        prop_indices = [np.where(prop_labels == g)[0][0] for g in group]
        # print(group, prop_indices)
        if len(group) > 1:
            # * fractional coordinates of all the points present in this group
            coords = np.vstack([props[pi]["coords"] for pi in prop_indices]) / shape
            # print(coords.shape)
            # * pbc centroid of the group in fractional coordinates
            centroid = stats.circmean(coords, axis=0, low=0, high=1)
            # * distances for the group in cartesian units
            distances = get_pbc_distances(coords, centroid, ase_atoms)
            # * histogram of the distances and bin centers
            psd_y_pbc, bine = np.histogram(distances, bins=bins, density=True)
            psd_x_pbc = (bine[1:] + bine[:-1]) / 2
            # * Now update the property of all the nodes in the graph with the pbc psd and cetnroid properties
            for i, g in enumerate(group):
                G.nodes[props[prop_indices[i]]["label"]]["centroid_pbc"] = (
                    find_coord_from_indices(
                        indices=centroid * shape, ase_atoms=ase_atoms, shape=shape
                    )
                )
                G.nodes[props[prop_indices[i]]["label"]]["psd_x_pbc"] = psd_x_pbc
                G.nodes[props[prop_indices[i]]["label"]]["psd_y_pbc"] = psd_y_pbc
        else:
            pi = np.where(prop_labels == list(group)[0])[0][0]
            # print(pi)
            # * get coordinates and centroids
            coords = props[pi]["coords"] / shape
            centroid = np.mean(coords, axis=0)
            # * histogram of the distances and bin centers
            distances = get_pbc_distances(coords, centroid, ase_atoms)
            psd_y_pbc, bine = np.histogram(distances, bins=bins, density=True)
            psd_x_pbc = (bine[1:] + bine[:-1]) / 2

            # * Now update the property of all the nodes in the graph with the pbc psd and cetnroid properties
            G.nodes[props[pi]["label"]]["centroid_pbc"] = find_coord_from_indices(
                indices=centroid * shape, ase_atoms=ase_atoms, shape=shape
            )
            G.nodes[props[pi]["label"]]["psd_x_pbc"] = psd_x_pbc
            G.nodes[props[pi]["label"]]["psd_y_pbc"] = psd_y_pbc

    return G


def add_clustering_to_rag_geometry_js(
    rag, props, pbc_groups, ase_atoms, shape, use_pbc=True, js_threshold=0.15
):
    """
    Adds cluster information to a region adjacency graph (RAG) by calculating the Jensen-Shannon distance between two histograms of distances.

    Args:
        rag (networkx.Graph): the RAG graph
        props (list): list of properties of each region in the RAG
        pbc_groups (list): list of groups of regions with periodic boundary conditions (PBC)
        ase_atoms (ase.Atoms): ASE atoms object
        shape (tuple): shape of the region
        use_pbc (bool): whether to use PBC or not
        js_threshold (float): the threshold for the Jensen-Shannon distance

    Returns:
        networkx.Graph: the RAG with cluster information added under keys 'cluster_geom_pbc' or 'cluster_geom_nopbc' depending on whether 'use_pbc' is True or False
    """

    def get_pbc_distances(coords, centroid_pbc, ase_atoms):
        """
        > The function takes the coordinates of the grid points in the region, the centroid of the region, and
        the atoms object, and returns the distances of the grid points from the periodic region centroid

        :param coords: the coordinates of the atoms in the cluster
        :param centroid_pbc: the centroid of the cluster in the PBC frame
        :param ase_atoms: ASE atoms object
        :return: The distances between the centroid and the atoms in the cluster.
        """

        from ase.geometry import complete_cell

        cell = complete_cell(ase_atoms.get_cell()).T
        # cell_inv = np.linalg.inv(cell)
        df = coords - centroid_pbc
        df = df - np.round(df)
        distances = np.linalg.norm(np.dot(cell, df.T).T, axis=1)
        return distances

    def get_js_dist_geom(i, j):
        """
        Calculates the Jensen-Shannon distance between two histograms of distances
        for a given time point t.

        Args:
            t (int): the index of the time point to calculate the distance for

        Returns:
            float: the Jensen-Shannon distance between the two histograms

        Raises:
            IndexError: if t is out of range for the distances_list array
        """
        d1 = distances_list[i]
        d2 = distances_list[j]
        d1 = d1[np.isfinite(d1)]
        d2 = d2[np.isfinite(d2)]
        _, bins = np.histogram(np.hstack((d1, d2)), bins=20)
        d1_hist, bins = np.histogram(d1, bins=bins)
        d2_hist, bins = np.histogram(d2, bins=bins)
        return jensenshannon(d1_hist, d2_hist)

    from scipy import stats
    import numpy as np
    import copy
    from .helper import find_coord_from_indices
    import itertools
    import networkx as nx
    from scipy.spatial.distance import jensenshannon
    import dask.bag as db
    from dask import delayed
    import dask
    from tqdm import tqdm

    # shape = regions.shape
    G = copy.deepcopy(rag)
    prop_labels = np.array([prop["label"] for prop in props])

    # * get a list of distances
    distances_list = []
    if use_pbc:
        for group in pbc_groups:
            prop_indices = [np.where(prop_labels == g)[0][0] for g in group]
            # print(group, prop_indices)
            if len(group) > 1:
                # * fractional coordinates of all the points present in this group
                coords = np.vstack([props[pi]["coords"] for pi in prop_indices]) / shape
                # print(coords.shape)
                # * pbc centroid of the group in fractional coordinates
                centroid = stats.circmean(coords, axis=0, low=0, high=1)
                # * distances for the group in cartesian units
                distances = get_pbc_distances(coords, centroid, ase_atoms)
                # * histogram of the distances and bin centers

                distances_list.append(distances)
                # psd_y_pbc, bine = np.histogram(distances, bins=bins, density=True)
                # psd_x_pbc = (bine[1:] + bine[:-1])/2
                # # * Now update the property of all the nodes in the graph with the pbc psd and cetnroid properties
                # for i, g in enumerate(group):
                #     G.nodes[props[prop_indices[i]]['label']]['centroid_pbc'] = find_coord_from_indices(indices = centroid*shape, ase_atoms=ase_atoms, shape=shape)
                #     # G.nodes[props[prop_indices[i]]['label']]['psd_x_pbc'] = psd_x_pbc
                #     # G.nodes[props[prop_indices[i]]['label']]['psd_y_pbc'] = psd_y_pbc
                #     G.nodes[props[prop_indices[i]]['label']]['circular_distances'] = distances

            else:
                pi = np.where(prop_labels == list(group)[0])[0][0]
                # print(pi)
                # * get coordinates and centroids
                coords = props[pi]["coords"] / shape
                centroid = np.mean(coords, axis=0)
                # * histogram of the distances and bin centers
                distances = get_pbc_distances(coords, centroid, ase_atoms)
                distances_list.append(distances)
                # psd_y_pbc, bine = np.histogram(distances, bins=bins, density=True)
                # psd_x_pbc = (bine[1:] + bine[:-1])/2

                # # * Now update the property of all the nodes in the graph with the pbc psd and cetnroid properties
                # G.nodes[props[pi]['label']]['centroid_pbc'] = find_coord_from_indices(indices = centroid*shape, ase_atoms=ase_atoms, shape=shape)
                # # # The code is assigning the value of `psd_y_pbc` to the `'psd_y_pbc'` attribute of a node
                # # # in a graph `G`. The node is identified by the label stored in `props[pi]['label']`.
                # # # The code is assigning the value of `psd_x_pbc` to the `'psd_x_pbc'` attribute of a node
                # # # in a graph `G`. The node is identified by the label stored in `props[pi]['label']`.
                # # G.nodes[props[pi]['label']]['psd_x_pbc'] = psd_x_pbc
                # # G.nodes[props[pi]['label']]['psd_y_pbc'] = psd_y_pbc
                # G.nodes[props[pi]['label']]['noncircular_distances'] = distances
    else:
        for node in rag.nodes:
            pi = np.where(prop_labels == node)[0][0]
            coords = props[pi]["coords"] / shape
            centroid = np.mean(coords, axis=0)
            distances = get_pbc_distances(coords, centroid, ase_atoms)
            distances_list.append(distances)

    # * now copute the js graph
    if use_pbc:
        # if pbc_groups is None:
        #     raise ValueError('pbc_groups must be provided if use_pbc=True')

        # * we will be classifying the pbc groups
        # energies_list = [get_energies_for_group2(list(
        #     g), regions, energy_interpolator=energy_interpolator) for g in pbc_groups]

        # * define an empty graph
        js_graph = nx.Graph()
        js_graph.add_nodes_from(range(len(pbc_groups)))

        # distances_list  =
        # energies_list = [get_energies_for_group2(list(g), regions, energy_interpolator=energy_interpolator) for g in pbc_groups]
        # * let's add edges between the pbc groups based on teh JS distance
        iterlist = list(itertools.combinations(range(len(pbc_groups)), 2))

        js_dist = [
            get_js_dist_geom(t[0], t[1])
            for t in tqdm(iterlist, desc="comp. dist. for clustering")
        ]
        # for t in iterlist:
        #     js_dist.append(delayed(get_js_dist_geom)(t[0], t[1]))
        # js_dist = dask.compute(js_dist)
        print(js_dist)
        # js_dist = db.from_sequence(iterlist).map(get_js_dist_geom).compute()
        # * print the the JS distances between regions
        # print(js_dist)
        # * check wher teh JS distance is less than the threshold and add edges from iterlist
        add_these_edges = [
            iterlist[k] for k in range(len(js_dist)) if js_dist[k] < js_threshold
        ]
        # print('adding these edges: ', add_these_edges)
        js_graph.add_edges_from(add_these_edges)
        # for k, (i, j) in enumerate(iterlist):
        #     # js = get_js_dist(i, j)
        #     js = js_dist[k]
        #     print(
        #         f'Jensen-Shannon distance for {i}:{pbc_groups[i]} and {j}:{pbc_groups[j]} is {js}')
        #     if js < js_threshold:
        #         js_graph.add_edge(i, j)
        conn = list(nx.connected_components(js_graph))

        # print('connected components are: ', conn)
        energy_groups = [
            set(itertools.chain.from_iterable(np.array(pbc_groups)[list(c)]))
            for c in conn
        ]

        # * add the cluster info to the rag
        for i in G.nodes():
            G.nodes[i]["cluster_geom_pbc"] = str(
                np.where([i in g for g in energy_groups])[0][0]
            )
    elif use_pbc == False:
        # * we will be working with the actual nodes (regions)
        # energies_list = [get_energies_for_group2(
        #     [node], regions, energy_interpolator=energy_interpolator) for node in rag.nodes()]

        # energies_list = [get_energies_for_group(
        #     [node], props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff, sample_fraction=sample_fraction) for node in rag.nodes()]
        # * define empty graph
        js_graph = nx.Graph()
        js_graph.add_nodes_from(range(len(G.nodes())))

        # energies_list = [get_energies_for_group2(
        #     [node], regions, energy_interpolator=energy_interpolator) for node in rag.nodes()]

        # * let's add edges between the pbc groups based on teh JS distance
        iterlist = list(itertools.combinations(range(len(G.nodes())), 2))

        js_dist = [
            get_js_dist_geom(t[0], t[1])
            for t in tqdm(iterlist, desc="comp. dist. for clustering")
        ]
        add_these_edges = [
            iterlist[k] for k in range(len(iterlist)) if js_dist[k] < js_threshold
        ]
        # print('adding these edges: ', add_these_edges)

        js_graph.add_edges_from(add_these_edges)

        # for i, j in itertools.combinations(len(rag.nodes()), 2):
        #     js = get_js_dist(i,j)
        #     if js < js_threshold:
        #         js_graph.add_edge(i, j)
        conn = list(nx.connected_components(js_graph))
        # print('connected components are: ', conn)
        # * add the clustering info  to the rag
        for i in G.nodes():
            G.nodes[i]["cluster_geom_nopbc"] = str(
                np.where([i in g for g in conn])[0][0]
            )

    return G


def add_clustering_to_rag(rag2, return_cluster_obj=False, no_pbc=False, **kwargs):
    """
    The function `add_clustering_to_rag` performs agglomerative clustering on data attributes of a graph
    and adds clustering information to the graph nodes.

    :param rag2: The `rag2` parameter in the `add_clustering_to_rag` function is expected to be a
    NetworkX graph object representing a region adjacency graph (RAG) with nodes and attributes. This
    graph is used to perform clustering on the data associated with the nodes in the graph
    :param return_cluster_obj: The `return_cluster_obj` parameter in the `add_clustering_to_rag`
    function determines whether the function should return the clustering object along with the modified
    graph (`rag2`) or just the modified graph itself, defaults to False (optional)
    :param no_pbc: The `no_pbc` parameter in the `add_clustering_to_rag` function is a boolean flag that
    specifies whether to use data without periodic boundary conditions (PBC). If `no_pbc` is set to
    `True`, the function will use data without PBC, while if it, defaults to False (optional)
    :return: If `return_cluster_obj` is set to `True`, the function will return a tuple containing the
    updated RAG (`rag2`) and the clustering object (`clustering`). If `return_cluster_obj` is set to
    `False`, the function will return just the updated RAG (`rag2`).
    """

    import numpy as np
    import networkx as nx

    # *  get the data array, while reordering to match the node order
    if no_pbc:
        X = np.vstack(
            list(dict(sorted(nx.get_node_attributes(rag2, "psd_y").items())).values())
        )
    else:
        X = np.vstack(
            list(
                dict(sorted(nx.get_node_attributes(rag2, "psd_y_pbc").items())).values()
            )
        )

    # * do the clustering using agglomerative clustering
    from sklearn.cluster import AgglomerativeClustering

    # get kwargs that are passed, if not set defaults
    metric = kwargs.get("metric", "euclidean")
    linkage = kwargs.get("linkage", "single")
    n_clusters = kwargs.get("n_clusters", None)
    connectivity = kwargs.get("connectivity", None)
    compute_full_tree = kwargs.get("compute_full_tree", True)
    distance_threshold = kwargs.get("distance_threshold", 0.25)
    compute_distances = kwargs.get("compute_distances", False)

    if len(X) > 1:
        # Remove 'affinity' from kwargs if it's present
        affinity = kwargs.pop('affinity', None)
        
        # Use 'metric' instead of 'affinity'
        if affinity and not metric:
            metric = affinity  # Use the old affinity value as metric if provided
        
        clustering_params = {
            'compute_full_tree': compute_full_tree,
            'connectivity': connectivity,
            'compute_distances': compute_distances,
            'linkage': linkage,
            'distance_threshold': distance_threshold,
            'n_clusters': n_clusters,
            'metric': metric
        }

        clustering = AgglomerativeClustering(**clustering_params).fit(X)

        # * Add the clustering info to rag
        for i in rag2.nodes():
            rag2.nodes[i]["cluster"] = str(clustering.labels_[i - 1])
        if return_cluster_obj:
            return (rag2, clustering)
        else:
            return rag2
    elif len(X) == 1:
        # rag2.nodes[1]['cluster'] = '1'
        nx.set_node_attributes(rag2, "1", "cluster")
        print("Only one node in the graph")
        return rag2
    else:
        raise ValueError(
            "No PSD with PBC data in the RAG (pore graph). Call add_psd_to_rag_with_pbc() first."
        )


def add_clustering_to_rag_pca(
    rag2,
    return_cluster_obj=False,
    no_pbc=False,
    use_kpca=True,
    kwargs_cluster={},
    kwargs_pca={},
):
    """
    The function `add_clustering_to_rag_pca` performs clustering on data from a graph and adds the
    clustering information to the graph nodes.

    :param rag2: The function `add_clustering_to_rag_pca` takes several parameters to add clustering
    information to a graph object. Here is a brief explanation of each parameter:
    :param return_cluster_obj: The `return_cluster_obj` parameter in the `add_clustering_to_rag_pca`
    function is a boolean flag that determines whether the function should return the clustering object
    along with the modified RAG (Region Adjacency Graph) or just the modified RAG itself. If
    `return_cluster_obj`, defaults to False (optional)
    :param no_pbc: The `no_pbc` parameter in the `add_clustering_to_rag_pca` function is a boolean flag
    that indicates whether to use data without periodic boundary conditions (PBC). If `no_pbc` is set to
    `True`, the function will use data without PBC; otherwise, defaults to False (optional)
    :param use_kpca: The `use_kpca` parameter in the `add_clustering_to_rag_pca` function is a boolean
    flag that determines whether Kernel PCA (Principal Component Analysis) should be used on the data
    before clustering. If set to `True`, Kernel PCA will be applied to the data array `, defaults to
    True (optional)
    :param kwargs_cluster: The `kwargs_cluster` parameter in the `add_clustering_to_rag_pca` function is
    a dictionary that allows you to pass additional keyword arguments to configure the
    AgglomerativeClustering algorithm used for clustering. These keyword arguments include:
    :param kwargs_pca: The `kwargs_pca` parameter in the `add_clustering_to_rag_pca` function is a
    dictionary that can contain the following optional parameters for KernelPCA:
    :return: The function `add_clustering_to_rag_pca` returns either the updated RAG object with
    clustering information added (`rag2`), or a tuple containing the updated RAG object and the
    clustering object if `return_cluster_obj` is set to True.
    """

    import numpy as np
    import networkx as nx

    # *  get the data array, while reordering to match the node order
    if no_pbc:
        X = np.vstack(
            list(dict(sorted(nx.get_node_attributes(rag2, "psd_y").items())).values())
        )
    else:
        X = np.vstack(
            list(
                dict(sorted(nx.get_node_attributes(rag2, "psd_y_pbc").items())).values()
            )
        )

    if use_kpca:
        # get kwargs that are passed, if not set defaults
        kernel = kwargs_pca.get("kernel", "cosine")

        # do the kernal PCA on X
        from sklearn.decomposition import KernelPCA

        kpc = KernelPCA(**kwargs_pca).fit(X)
        X = kpc.transform(X)

        # * if you cannot solve for eigen values use regular PCA
        # * usually happens when there is only one pocket in the framework
        if X.shape[1] == 0:
            X = kpc.X_fit_

    # * do the clustering using agglomerative clustering
    from sklearn.cluster import AgglomerativeClustering

    # get kwargs that are passed, if not set defaults
    affinity = kwargs_cluster.get("affinity", "euclidean")
    linkage = kwargs_cluster.get("linkage", "single")
    n_clusters = kwargs_cluster.get("n_clusters", None)
    connectivity = kwargs_cluster.get("connectivity", None)
    compute_full_tree = kwargs_cluster.get("compute_full_tree", True)
    distance_threshold = kwargs_cluster.get("distance_threshold", 0.4)
    compute_distances = kwargs_cluster.get("compute_distances", False)

    if len(X) > 1:
        clustering = AgglomerativeClustering(
            compute_full_tree=compute_full_tree,
            connectivity=connectivity,
            compute_distances=compute_distances,
            linkage=linkage,
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            affinity=affinity,
        ).fit(X)

        # * Add the clustering info to rag
        for i in rag2.nodes():
            rag2.nodes[i]["cluster"] = str(clustering.labels_[i - 1])
        if return_cluster_obj:
            return (rag2, clustering)
        else:
            return rag2
    elif len(X) == 1:
        # rag2.nodes[1]['cluster'] = '1'
        nx.set_node_attributes(rag2, "1", "cluster")
        print("Only one node in the graph")
        return rag2
    else:
        raise ValueError(
            "No PSD with PBC data in the RAG (pore graph). Call add_pbc_psd_to_rag() first."
        )


# def add_clustering_to_rag_pca2(rag2,  return_cluster_obj = False, no_pbc =False, use_kpca=True, kwargs_cluster={}, kwargs_pca={}):
#     """
#     > We take the RAG, get the PSD data for each node, and then use agglomerative clustering to cluster
#     the nodes.

#     :param rag2: the RAG object
#     :param distance_threshold: The linkage distance at which to cut the dendrogram to determine the
#     clusters
#     :param return_cluster_obj: if True, returns the clustering object as well as the rag, defaults to
#     False (optional)
#     :return: The rag2 object with the cluster labels added to the nodes.
#     """

#     import numpy as np
#     import networkx as nx
#     from sklearn.preprocessing import MinMaxScaler
#     # *  get the data array, while reordering to match the node order
#     if no_pbc:
#         X = np.vstack(dict(sorted(nx.get_node_attributes(rag2, 'vdw_hist_y').items())).values())
#     else:
#         X = np.vstack(dict(sorted(nx.get_node_attributes(rag2, 'vdw_hist_y_pbc').items())).values())

#     if use_kpca:
#         # get kwargs that are passed, if not set defaults
#         kernel = kwargs_pca.get('kernel', 'cosine')

#         # do the kernal PCA on X
#         from sklearn.decomposition import KernelPCA
#         kpc = KernelPCA(**kwargs_pca).fit(X)
#         X = kpc.transform(X)

#         #* if you cannot solve for eigen values use regular PCA
#         # * usually happens when there is only one pocket in the framework
#         if X.shape[1]==0:
#             X = kpc.X_fit_


#     # * do the clustering using agglomerative clustering
#     from sklearn.cluster import AgglomerativeClustering

#     # get kwargs that are passed, if not set defaults
#     affinity = kwargs_cluster.get('affinity', 'euclidean')
#     linkage = kwargs_cluster.get('linkage', 'single')
#     n_clusters = kwargs_cluster.get('n_clusters', None)
#     connectivity = kwargs_cluster.get('connectivity', None)
#     compute_full_tree = kwargs_cluster.get('compute_full_tree', True)
#     distance_threshold = kwargs_cluster.get('distance_threshold', None)
#     compute_distances = kwargs_cluster.get('compute_distances', False)

#     if len(X) > 1:
#         clustering = AgglomerativeClustering(compute_full_tree=compute_full_tree, connectivity=connectivity, compute_distances=compute_distances, linkage=linkage, distance_threshold = distance_threshold, n_clusters=n_clusters, affinity=affinity).fit(X)

#         #* Add the clustering info to rag
#         for i in rag2.nodes():
#             rag2.nodes[i]['cluster'] = str(clustering.labels_[i-1])
#         if return_cluster_obj:
#             return (rag2, clustering)
#         else:
#             return rag2
#     elif len(X) == 1:
#         # rag2.nodes[1]['cluster'] = '1'
#         nx.set_node_attributes(rag2, '1', 'cluster')
#         print('Only one node in the graph')
#         return rag2
#     else:
#         raise ValueError('No PSD with PBC data in the RAG (pore graph). Call add_pbc_psd_to_rag() first.')


def add_clustering_to_rag_energy(
    rag,
    props,
    ase_atoms,
    forcefield_mixed,
    shape,
    use_pbc=True,
    pbc_groups=None,
    cutoff=12.8,
    sample_fraction=1,
):
    import networkx as nx
    import itertools
    from scipy.stats import ks_2samp
    import numpy as np

    if use_pbc:
        if pbc_groups is None:
            raise ValueError("pbc_groups must be provided if use_pbc=True")
        # * we will classifying the pbc groups
        energies_list = [
            get_energies_for_group(
                list(g),
                props,
                ase_atoms,
                forcefield_mixed,
                shape,
                cutoff=cutoff,
                sample_fraction=sample_fraction,
            )
            for g in pbc_groups
        ]
        # * test the samples using ks2samp
        ks_graph = nx.Graph()
        ks_graph.add_nodes_from(range(len(pbc_groups)))
        for i, j in itertools.combinations(range(len(pbc_groups)), 2):
            print(
                f"ks test for {pbc_groups[i]} and {pbc_groups[j]} is {ks_2samp(energies_list[i], energies_list[j])}"
            )
            ks = ks_2samp(energies_list[i], energies_list[j])
            if ks[1] > 0.05:
                ks_graph.add_edge(i, j)
        conn = list(nx.connected_components(ks_graph))
        # print('connected components are: ', conn)

        energy_groups = [
            set(itertools.chain.from_iterable(np.array(pbc_groups)[list(c)]))
            for c in conn
        ]

        # * add the cluster info to the rag
        for i in rag.nodes():
            rag.nodes[i]["cluster_energy_pbc"] = str(
                np.where([i in g for g in energy_groups])[0][0]
            )
    else:
        # * we will be working with the actual nodes (regions)
        energies_list = [
            get_energies_for_group(
                [node],
                props,
                ase_atoms,
                forcefield_mixed,
                shape,
                cutoff=cutoff,
                sample_fraction=sample_fraction,
            )
            for node in rag.nodes()
        ]
        ks_graph = nx.Graph()
        ks_graph.add_nodes_from(range(len(rag.nodes())))
        for i, j in itertools.combinations(len(rag.nodes()), 2):
            ks = ks_2samp(energies_list[i], energies_list[j])
            if ks[1] > 0.05:
                ks_graph.add_edge(i, j)
        conn = list(nx.connected_components(ks_graph))
        # * add the clustering info  to the rag
        for i in rag.nodes():
            rag.nodes[i]["cluster_energy_nopbc"] = str(
                np.where([i in g for g in conn])[0][0]
            )
    return rag
    # energy_groups = [set(itertools.chain.from_iterable(np.array(pbc_groups)[list(c)])) for c in conn]


# %%
# def add_clustering_to_rag_energy_js(rag, props, ase_atoms, forcefield_mixed, shape, use_pbc=True, pbc_groups=None, cutoff=12.8,sample_fraction=1):


#     import networkx as nx
#     import itertools
#     from scipy.stats import ks_2samp
#     from scipy.spatial.distance import jensenshannon
#     import numpy as np
#     import dask.bag as db
#     def get_js_dist(e1, e2):
#         hist, bins = np.histogram(np.hstack((e1,e2)), bins=20)
#         e1_hist, bins  = np.histogram(e1, bins=bins)
#         e2_hist, bins  = np.histogram(e2, bins=bins)
#         return jensenshannon(e1_hist, e2_hist)

#     if use_pbc:
#         if pbc_groups is None:
#             raise ValueError('pbc_groups must be provided if use_pbc=True')
#         # * we will classifying the pbc groups
#         energies_list = [get_energies_for_group(list(g), props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff,sample_fraction=sample_fraction) for g in pbc_groups]
#         # * test the samples using ks2samp
#         js_graph = nx.Graph()
#         js_graph.add_nodes_from(range(len(pbc_groups)))
#         pairwise_list = list(itertools.combinations(range(len(pbc_groups)), 2))
#         # js_list = db.from_sequence(pairwise_list).map(lambda x: get_js_dist(energies_list[x[0]], energies_list[x[1]])).compute()
#         # for i,j in pairwise_list:
#         #     js = js_list[i]
#         #     if js <  0.15:
#         #         js_graph.add_edge(i,j)
#         for i,j in itertools.combinations(range(len(pbc_groups)), 2):
#             print(f'Jensen-Shannon distance for {pbc_groups[i]} and {pbc_groups[j]} is {get_js_dist(energies_list[i], energies_list[j])}')
#             js = get_js_dist(energies_list[i], energies_list[j])
#             if js <  0.15:
#                 js_graph.add_edge(i,j)
#         conn = list(nx.connected_components(js_graph))
#         print('connected components are: ', conn)

#         energy_groups = [set(itertools.chain.from_iterable(np.array(pbc_groups)[list(c)])) for c in conn]


#         # * add the cluster info to the rag
#         for i in rag.nodes():
#             rag.nodes[i]['cluster_energy_pbc'] = str(np.where([i in g for g in energy_groups])[0][0])
#     else:
#         # * we will be working with the actual nodes (regions)
#         energies_list = [get_energies_for_group([node], props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff,sample_fraction=sample_fraction) for node in rag.nodes()]
#         js_graph = nx.Graph()
#         js_graph.add_nodes_from(range(len(rag.nodes())))
#         for i,j in itertools.combinations(len(rag.nodes()), 2):
#             js = get_js_dist(energies_list[i], energies_list[j])
#             if  js < 0.15:
#                 js_graph.add_edge(i,j)
#         conn = list(nx.connected_components(js_graph))
#         #* add the clustering info  to the rag
#         for i in rag.nodes():
#             rag.nodes[i]['cluster_energy_nopbc'] = str(np.where([i in g for g in conn])[0][0])
#     return rag
# %%
def compare_two_paths(p1, p2, rag):
    """
    It takes two paths and a region adjacency graph, and returns a list of tuples, where each tuple
    contains a node in the first path and a node in the second path

    :param p1: a list of tuples, each tuple is a coordinate in the path
    :param p2: the path to be compared to p1
    :param rag: the region adjacency graph
    """
    import networkx.algorithms.isomorphism as iso

    g1 = rag.subgraph(p1)
    g2 = rag.subgraph(p2)
    nm = iso.categorical_node_match("cluster", default="None")
    return iso.is_isomorphic(g1, g2, node_match=nm)


def find_isomorphic_path_groups(list_of_paths, rag, cluster=True):
    import itertools
    import networkx as nx

    gg = nx.Graph()
    if cluster:
        rag = add_clustering_to_rag(rag)
    for i in range(len(list_of_paths)):
        gg.add_node(i)
    indices = list(itertools.combinations(range(len(list_of_paths)), 2))
    for i, p in enumerate(itertools.combinations(list_of_paths, 2)):
        # /else:
        if compare_two_paths(p[0], p[1], rag):
            # print(p)
            gg.add_edge(indices[i][0], indices[i][1])
    # * get a list tof groups of paths that are isomorphic
    ccgg = list(nx.connected_components(gg))
    grouped_paths = [[list_of_paths[i] for i in cg] for cg in ccgg]
    return grouped_paths


def calc_path_components(rag, path: list):
    """
    Calculate the PCA components of a path in a RAG.

    Parameters
    ----------
    rag : nx.Graph
        The RAG to calculate the PCA components for.
    path : list
        The path to calculate the PCA components for.

    Returns
    -------
    np.ndarray
        The PCA components of the path.
    """
    import numpy as np
    import networkx as nx
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca_cart = PCA(n_components=np.min([len(path) - 1, 3]))
    pca_frac = PCA(n_components=np.min([len(path) - 1, 3]))
    maxima_coordinates = np.vstack(
        nx.get_node_attributes(rag.subgraph(path), "maxima_indices").values()
    )
    # maxima_fractional = mgr.get_fractional_coordinates(maxima_cartesian, ase_atoms)
    maxima_coordinates = StandardScaler().fit_transform(maxima_coordinates)
    pca_cart.fit(maxima_coordinates)
    # pca_frac.fit(maxima_fractional)
    # flag = np.logical_and(pca_cart.explained_variance_ratio_[0] > threshold, pca_frac.explained_variance_ratio_[0] > threshold)
    # flag = pca_cart.explained_variance_ratio_[0] > threshold

    # print(path, pca.explained_variance_ratio_)
    # return (pca_cart.explained_variance_ratio_, pca_frac.explained_variance_ratio_, flag )
    return pca_cart.explained_variance_ratio_


def find_paths(regions, rag2):
    import numpy as np

    # * Now we are going to go face by face, and find all the paths
    # * that have source on starting face and destination on one of the other 5 faces
    import networkx as nx

    all_paths = list(nx.all_pairs_bellman_ford_path(rag2, weight="weight"))
    # all_paths = list(nx.all_pairs_dijkstra_path(rag2, weight='weight'))
    # all_paths = list(nx.all_pairs_dijkstra_path(rag2, weight='weight'))
    # * make a subset of these paths to check later for channels
    check_these_paths = []

    shape = regions.shape
    map_list = []
    for a in [0, 1, 2]:
        llr = regions.take(indices=0, axis=a)  # * left wall
        rrr = regions.take(indices=shape[a] - 1, axis=a)  # * right wall
        map_list.append(np.unique(list(zip(llr.flatten(), rrr.flatten())), axis=0))
    # map_list = np.vstack(map_list)

    # * check in the forward direction
    for a in [0, 1, 2]:  # * along the three faces (directions) of the cube
        for thing in all_paths:  # * loop over the tuples in all_paths
            paths = list(thing[1].items())[1:]  # * all paths starting from source
            for (
                dest,
                path,
            ) in (
                paths
            ):  # * the first entry is the source node, second is a dictionary of paths
                if len(path) > 1:
                    # print(len(path))
                    for node_i, node in list(enumerate(path))[1:]:
                        # * check in the forward direction
                        # source_nodes = np.unique(wall_list[a*2+0])
                        source_nodes = map_list[a][:, 0]
                        # print(node_i, node, path, source_nodes)
                        # * only the starting node should be on the starting face
                        if np.isin(node, source_nodes):
                            break
                        # destination_nodes  =   np.unique([v for i,w in enumerate(wall_list) for v in w  if i!=a*2+0])
                        # destination_nodes= np.unique(wall_list[a*2+1])
                        destination_nodes = map_list[a][:, 1]
                        if np.logical_and(
                            *[
                                np.isin(thing[0], source_nodes),
                                np.isin(node, destination_nodes),
                            ]
                        ):  # * thing = (source, {dest1:path1,dest2:path2,... })
                            check_these_paths.append(
                                path[: node_i + 1]
                            )  # * grab that path up to that node
                            # * add a path only once, once a destination is reached, move on to the next path.
                            break
                # * Add the single node paths to the list of paths to check
                # * where a single node occupies the whole path
                # if (len(path)==1):
                #     source_nodes = np.unique(wall_list[a*2+0])
                #     if np.isin(path[0], source_nodes):
                #         check_these_paths.append(path)

    # * Check in the backward direction
    for a in [0, 1, 2]:  # * along the three faces (directions) of the cube
        for thing in all_paths:  # * loop over the tuples in all_paths
            paths = list(thing[1].items())[1:]  # * all paths starting from source
            for (
                dest,
                path,
            ) in (
                paths
            ):  # * the first entry is the source node, second is a dictionary of paths
                if len(path) > 1:
                    # print(len(path))
                    for node_i, node in list(enumerate(path))[1:]:
                        # * check in the backward direction
                        # source_nodes = np.unique(wall_list[a*2+1])
                        source_nodes = map_list[a][:, 1]
                        # print(node_i,node, path, source_nodes)
                        # * only the starting node should be on the starting face
                        if np.isin(node, source_nodes):
                            break
                        # destination_nodes  =   np.unique([v for i,w in enumerate(wall_list) for v in w  if i!=a*2+1])
                        # destination_nodes= np.unique(wall_list[a*2+0])
                        destination_nodes = map_list[a][:, 0]
                        if np.logical_and(
                            *[
                                np.isin(thing[0], source_nodes),
                                np.isin(node, destination_nodes),
                            ]
                        ):  # * thing = (source, {dest1:path1,dest2:path2,... })
                            check_these_paths.append(
                                path[: node_i + 1]
                            )  # * grab that path up to that node
                            # * add a path only once, once a destination is reached, move on to the next path.
                            break
                            # * Add the single node paths to the list of paths to check

    # region #* Remove paths that non unique, in terms of repetiion indepent of order (assuming that no relevant paths are cyclic)
    frozen_paths = []
    check_these_paths_new = []
    for i, p in enumerate(check_these_paths):
        if frozenset(p) not in frozen_paths:
            frozen_paths.append(frozenset(p))
            check_these_paths_new.append(p)
    checks = [calc_path_components(rag2, p) for p in check_these_paths_new]
    return checks, check_these_paths_new


# def get_connections_for_rag(
#     rag,
#     region_labels,
#     maxima,
#     ase_atoms,
#     dgrid,
#     minimum_window_separation=1.0,
#     wall_windows=True,
#     wall_thickness=2,
# ):
#     def find_windows(regs):
#         import numpy as np
#         from sklearn.cluster import AgglomerativeClustering

#         i = regs[0]
#         j = regs[1]
#         check_flag = np.logical_or(
#             np.logical_and(region_labels == i, outer.get(j)),
#             np.logical_and(region_labels == j, outer.get(i)),
#         )
#         if np.sum(check_flag) > 0:
#             # print("Edges found between "+str(i)+' and '+ str(j) + ' sum: '+str(np.sum(check_flag)))
#             windices = np.vstack(np.where(check_flag)).T  # * (N,3)
#             window_points = np.dot(
#                 A_unit, (windices / region_labels.shape).T
#             ).T  # * (N,3)
#             # ic(window_points)
#             # * No clustering
#             # window_centers = np.mean(window_points, axis=0)

#             # * DBScan
#             # X_scaled = StandardScaler().fit_transform(window_points)
#             # db = DBSCAN(eps=0.5, min_samples=10).fit(X_scaled)
#             # window_centers = np.vstack([window_points[db.labels_==i].mean(axis=0) for i in np.unique(db.labels_) if i != -1])

#             # if len(window_points) >1:
#             # * Agglomerative
#             ag = AgglomerativeClustering(
#                 n_clusters=None,
#                 linkage="single",
#                 distance_threshold=minimum_window_separation,
#                 compute_full_tree=True,
#             ).fit(window_points)
#             window_centers = np.vstack(
#                 [
#                     window_points[ag.labels_ == i].mean(axis=0)
#                     for i in np.unique(ag.labels_)
#                     if i != -1
#                 ]
#             )
#             # else:
#             # window_centers = window_points
#             window_fractional = get_fractional_coordinates(window_centers, ase_atoms)
#             window_radii = np.array(dinterp(window_fractional))
#             return np.array(
#                 [regs, True, "internal", window_centers, window_radii], dtype="object"
#             )  # returns a list of [[r1,r2], connected or not, array of window_centers]
#         else:
#             return np.array([regs, False, "internal", None, None], dtype="object")

#     # * find the internal windows
#     from skimage.segmentation import find_boundaries
#     from .helper import interpolate_labels, get_fractional_coordinates
#     import ase
#     import numpy as np
#     from rich.progress import track

#     connected_regions = np.vstack(list(rag.edges()))
#     list_of_region_indices = np.delete(np.unique(region_labels), 0)
#     outer = {
#         reg: find_boundaries(region_labels == reg, mode="outer")
#         for reg in list_of_region_indices
#     }
#     dinterp = interpolate_labels(dgrid)
#     minimum_window_separation = 1.0
#     A_unit = ase.geometry.complete_cell(ase_atoms.get_cell()).T

#     connections = [
#         find_windows(regs) for regs in track(rag.edges, description="Finding windows:")
#     ]

#     # * Windows on the wall are to be calculated (periodic windows)
#     if wall_windows:
#         wwl = get_wall_windows2(
#             regions=region_labels,
#             maxima=maxima,
#             dgrid=dgrid,
#             ase_atoms=ase_atoms,
#             wall_thickness=wall_thickness,
#         )
#         if len(wwl) > 0:
#             connections = connections + wwl.tolist()

#     return connections

def get_connections_for_rag(
    rag,
    region_labels,
    maxima,
    ase_atoms,
    dgrid,
    minimum_window_separation=1.0,
    wall_windows=True,
    wall_thickness=2,
):
    """Find connections between regions in a region adjacency graph (RAG) and identify window locations.

    This function analyzes a RAG to find connections between adjacent regions and identifies the window 
    locations where regions connect. It uses agglomerative clustering to group boundary points into windows
    and finds the optimal window centers based on distance values.

    Parameters
    ----------
    rag : networkx.Graph
        Region adjacency graph representing connections between regions
    region_labels : numpy.ndarray
        3D array of region labels where each unique value represents a different region
    maxima : numpy.ndarray
        Array containing the maxima points of each region
    ase_atoms : ase.Atoms
        ASE Atoms object containing the structure information
    dgrid : numpy.ndarray
        3D array of distance values on the grid
    minimum_window_separation : float, optional
        Minimum distance between separate windows, by default 1.0
    wall_windows : bool, optional
        Whether to calculate windows on periodic boundaries, by default True
    wall_thickness : int, optional
        Thickness of wall region for finding periodic windows, by default 2

    Returns
    -------
    list
        List of connections, where each connection is an array containing:
        - [region1, region2]: The two connected region indices
        - bool: Whether regions are connected
        - str: Type of connection ("internal" or "wall")
        - numpy.ndarray: Window center coordinates
        - numpy.ndarray: Window radii
    """
    # * find the internal windows
    from skimage.segmentation import find_boundaries
    from .helper import interpolate_labels, get_fractional_coordinates
    import ase
    import numpy as np
    from rich.progress import track

    connected_regions = np.vstack(list(rag.edges()))
    list_of_region_indices = np.delete(np.unique(region_labels), 0)
    outer = {
        reg: find_boundaries(region_labels == reg, mode="outer")
        for reg in list_of_region_indices
    }
    dinterp = interpolate_labels(dgrid)
    A_unit = ase.geometry.complete_cell(ase_atoms.get_cell()).T

    def find_windows(regs):
        """Find windows between two regions.

        Parameters
        ----------
        regs : tuple
            Tuple of two region indices to check for connections

        Returns
        -------
        numpy.ndarray
            Array containing connection information:
            [region indices, connected flag, type, window centers, window radii]
        """
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        i = regs[0]
        j = regs[1]
        check_flag = np.logical_or(
            np.logical_and(region_labels == i, outer.get(j)),
            np.logical_and(region_labels == j, outer.get(i)),
        )
        if np.sum(check_flag) > 0:
            windices = np.vstack(np.where(check_flag)).T  # * (N,3)
            window_points = np.dot(
                A_unit, (windices / region_labels.shape).T
            ).T  # * (N,3)
            
            # * Agglomerative clustering to find window centers
            ag = AgglomerativeClustering(
                n_clusters=None,
                linkage="single",
                distance_threshold=minimum_window_separation,
                compute_full_tree=True,
            ).fit(window_points)
            
            # * Get distance values and region labels at window indices
            d_windices = dgrid[tuple(windices.T)]  # distance values at the window indices
            r_windices = region_labels[tuple(windices.T)]  # region labels at the window indices
            
            window_centers = []
            for uwi in np.unique(ag.labels_):
                # For each cluster that is not noise
                if uwi != -1:
                    # Get values for this cluster
                    uwi_dvals = d_windices[ag.labels_ == uwi]  # distance values in the cluster
                    uwi_points = window_points[ag.labels_ == uwi]  # points in the cluster
                    uwi_r = r_windices[ag.labels_ == uwi]  # region labels in the cluster
                    
                    # For points in first region
                    uwi_r1_indices = np.where(uwi_r == regs[0])[0]
                    uwi_r1_dvals = uwi_dvals[uwi_r1_indices]
                    uwi_r1_points = uwi_points[uwi_r1_indices]
                    
                    # Find the point with maximum distance value in first region
                    if len(uwi_r1_indices) > 0:
                        uwi_r1_max_indices = np.argmax(uwi_r1_dvals)
                        uwi_r1_max_points = uwi_r1_points[uwi_r1_max_indices]
                    else:
                        continue  # Skip if no points from first region in this cluster
                    
                    # For points in second region
                    uwi_r2_indices = np.where(uwi_r == regs[1])[0]
                    uwi_r2_dvals = uwi_dvals[uwi_r2_indices]
                    uwi_r2_points = uwi_points[uwi_r2_indices]
                    
                    # Find the point with maximum distance value in second region
                    if len(uwi_r2_indices) > 0:
                        uwi_r2_max_indices = np.argmax(uwi_r2_dvals)
                        uwi_r2_max_points = uwi_r2_points[uwi_r2_max_indices]
                    else:
                        continue  # Skip if no points from second region in this cluster
                    
                    # Take the centroid of the two maxima of each region close to the window
                    uwi_center = (uwi_r1_max_points + uwi_r2_max_points) / 2
                    window_centers.append(uwi_center)
            
            if window_centers:
                window_centers = np.vstack(window_centers)
                window_fractional = get_fractional_coordinates(window_centers, ase_atoms)
                window_radii = np.array(dinterp(window_fractional))
                return np.array(
                    [regs, True, "internal", window_centers, window_radii], dtype="object"
                )
            else:
                return np.array([regs, False, "internal", None, None], dtype="object")
        else:
            return np.array([regs, False, "internal", None, None], dtype="object")

    connections = [
        find_windows(regs) for regs in track(rag.edges, description="Finding windows:")
    ]

    # * Windows on the wall are to be calculated (periodic windows)
    if wall_windows:
        wwl = get_wall_windows2(
            regions=region_labels,
            maxima=maxima,
            dgrid=dgrid,
            ase_atoms=ase_atoms,
            wall_thickness=wall_thickness,
        )
        if len(wwl) > 0:
            connections = connections + wwl.tolist()

    return connections


def get_diametrically_opposite_points(region_indices, ase_atoms, shape, num_points=8):
    from scipy.spatial import ConvexHull
    import numpy as np

    # get the cell matrix
    cell = ase_atoms.get_cell()

    # convert the region indices to fractional coordinates
    region_frac_coords = region_indices / shape

    # convert the fractional coordinates to actual coordinates
    region_coords = np.dot(region_frac_coords, cell)

    # compute the convex hull of the region
    hull = ConvexHull(region_coords)

    # get the vertices of the convex hull
    vertices = region_coords[hull.vertices]

    # compute the centroid of the vertices
    centroid = np.mean(vertices, axis=0)

    # compute the angles of the vertices with respect to the centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])

    # sort the vertices by angle and keep track of the original indices
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    # select num_points points that are evenly spaced around the circumference
    selected_indices = sorted_indices[
        np.linspace(0, len(sorted_vertices), num_points, endpoint=False).astype(int)
    ]

    return region_indices[selected_indices]


def get_sampling_paths_for_region(
    region_index, connections, dgrid, regions, maxima, shape, ase_atoms
):
    """
    > It finds the paths from the maxima of a region to the windows of that region

    :param region_index: the index of the region you want to find the paths for
    :param connections: a list of tuples, each tuple contains the indices of the connected regions and
    the window coordinates
    :param dgrid: the distance grid
    :param maxima: the indices of the maxima of the region
    :param shape: shape of the grid
    :param ase_atoms: ASE atoms object
    :return: The paths are being returned.
    """

    import numpy as np
    import dijkstra3d
    from .helper import get_fractional_coordinates

    # * maxima of the regiuon
    region_maxima_index = maxima[region_index - 1]

    # * use exp field this works fine.
    field = np.exp(-(dgrid - dgrid.min()))

    # * find the paths
    paths = []
    # * let's find the indices ofthe windows of that region
    connected_regions = np.vstack([c[0] for c in connections if c[1]])
    # print(np.sum([(region_index in c) for c in connected_regions]) > 0)
    if np.sum([(region_index in c) for c in connected_regions]) > 0:
        # print('Going here')
        region_window_coords = np.vstack(
            [c[3] for c in connections if (region_index in c[0]) and c[1]]
        )
        # region_window_coords = np.vstack(connections[np.any(connected_regions==region_index, axis=1)][:,-2])
        region_window_frac_coords = get_fractional_coordinates(
            region_window_coords, ase_atoms
        )
        region_window_indices = (region_window_frac_coords * shape).astype(int)

        # * dijkstra path finding
        for wc in region_window_indices:
            paths.append(
                dijkstra3d.dijkstra(field, region_maxima_index, wc, connectivity=26)
            )
    else:
        # print('Region index {} has no windows'.format(region_index))
        shape = dgrid.shape
        region_maxima_index = maxima[region_index - 1]
        region_indices = np.vstack(np.where(regions == region_index)).T
        # print('Number of points in the region: ', region_indices.shape[0])
        hull_points = get_diametrically_opposite_points(
            region_indices, ase_atoms, shape, num_points=8
        )
        # farthest_points = get_farthest_points_on_random_lines(region_maxima_index, np.where(regions==region_index), shape, ase_atoms, num_points=8, num_directions=100)
        # * dijkstra path finding
        for wc in hull_points:
            paths.append(
                dijkstra3d.dijkstra(field, region_maxima_index, wc, connectivity=26)
            )
    return paths


def mark_points_with_largest_including_dia(
    region_index, regions, connections, dgrid, maxima, ase_atoms, maxima_radii
):
    import ase
    import numpy as np

    shape = regions.shape
    paths = get_sampling_paths_for_region(
        region_index, connections, dgrid, regions, maxima, shape, ase_atoms
    )
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T

    # find the points in the region
    region_points = np.where(regions == region_index)
    # initialize the list of radii with zeros
    radii = np.zeros(len(region_points[0]))

    for path in paths:
        path = path[::-1]  # reverse the path
        for i in range(len(path)):
            center = path[i]
            radius = dgrid[
                path[i][0], path[i][1], path[i][2]
            ]  # size of the sphere is dgrid[path_index]
            # radius = min(radius, maxima_radii[center[0], center[1], center[2]])  # limit the radius to the maxima radius
            # print(region_points, shape, center)
            # calculate their distance to the center in fractional coordinates
            dr_scaled = np.vstack(region_points).T / shape - center / shape
            # convert it back to actual distances
            dist = np.linalg.norm(np.dot(cell, dr_scaled.T).T, axis=1)

            # find the points within the sphere
            possible_neighbors = dist < radius
            # assign the radius to the points within the sphere
            radii[possible_neighbors] = radius

    return radii


def get_psd_on_rag(rag, bins=None):
    for key in ["regions", "connections", "dgrid", "maxima", "ase_atoms"]:
        if key not in rag.graph:
            raise ValueError(
                f"{key} not in rag.graph. Please compute and add {key} to rag.graph first before computing the PSD "
            )
    import dask
    from dask.diagnostics import ProgressBar
    import numpy as np

    # create a delayed version of the mark_points_in_spheres function
    mark_points_with_largest_including_dia_delayed = dask.delayed(
        mark_points_with_largest_including_dia
    )
    regions = rag.graph["regions"]
    # get the list of region indices
    region_indices = np.arange(1, regions.max() + 1)

    # create a list of delayed tasks
    tasks = [
        mark_points_with_largest_including_dia_delayed(
            region_index,
            rag.graph["regions"],
            rag.graph["connections"],
            rag.graph["dgrid"],
            rag.graph["maxima"],
            rag.graph["ase_atoms"],
            rag.graph["dgrid"][tuple(rag.graph["maxima"].T)],
        )
        for region_index in region_indices
    ]
    with ProgressBar():
        # compute the results in parallel
        results = dask.compute(*tasks)

    # combine all the non-zero radii
    non_zero_radii = np.concatenate([radii[radii > 0] for radii in results])
    if bins is None:
        # define the bins
        bins = np.linspace(0, 100, 100)

    # compute the histogram
    hist, bin_edges = np.histogram(non_zero_radii * 2, bins=bins, density=True)

    # compute the PDF and CDF
    pdf = hist
    cdf = np.cumsum(hist)
    rag.graph["pore_size_dist"] = pdf
    rag.graph["pore_size_cdf"] = cdf
    rag.graph["pore_size_bins"] = bin_edges
    return rag


# def get_sampling_paths_for_region(region_index, connections, dgrid, maxima,  shape, ase_atoms):
#     """
#     > It finds the paths from the maxima of a region to the windows of that region

#     :param region_index: the index of the region you want to find the paths for
#     :param connections: a list of tuples, each tuple contains the indices of the connected regions and
#     the window coordinates
#     :param dgrid: the distance grid
#     :param maxima: the indices of the maxima of the region
#     :param shape: shape of the grid
#     :param ase_atoms: ASE atoms object
#     :return: The paths are being returned.
#     """

#     import numpy as np
#     from .helper import get_fractional_coordinates
#     #* maxima of the regiuon
#     region_maxima_index = maxima[region_index-1]

#     #* let's find the indices ofthe windows of that region
#     connected_regions = np.vstack([c[0] for c in connections])
#     region_window_coords = np.vstack(np.array(connections)[np.any(connected_regions==region_index, axis=1)][:,-2])
#     region_window_frac_coords = get_fractional_coordinates(region_window_coords, ase_atoms)
#     region_window_indices = (region_window_frac_coords*shape).astype(int)

#     #* dijkstra path finding
#     import dijkstra3d

#     # * use exp field this works fine.
#     field = np.exp(-(dgrid - dgrid.min()))

#     # * find the paths
#     paths = []
#     for wc in region_window_indices:
#         paths.append(dijkstra3d.dijkstra(field, region_maxima_index, wc, connectivity=26))

#     return paths


def sample_path(path_indices, points_per_path=100):
    """
    It takes a list of points and returns a list of points that are evenly spaced along the path defined
    by the input points using spline interpolation

    :param path_indices: a numpy array of shape (n,3) where n is the number of points in the path
    :param points_per_path: The number of points to sample along the path, defaults to 100 (optional)
    :return: a list of points that are evenly spaced along the path.
    """

    from scipy.interpolate import splprep, splev
    import numpy as np

    tck, u = splprep([path_indices[:, 0], path_indices[:, 1], path_indices[:, 2]], s=0)
    x = np.linspace(0, 1, points_per_path)
    new_points = np.array(splev(x, tck)).T
    return new_points


def get_distances_for_path(path_indices, shape, ase_atoms, n_neighbors=10, radii=None):
    """
    > For each grid point, find the nearest neighbors, and return the minimum distance to the nearest
    neighbor minus the covalent radius of the neighbor

    :param path_indices: the indices of the path
    :param shape: the shape of the grid
    :param ase_atoms: the ase atoms object for the framework
    :param n_neighbors: the number of neighbors to consider for each grid point, defaults to 10
    (optional)
    :param radii: the covalent radii of the atoms in the structure. If not provided, they will be
    calculated from the atomic numbers using ase.data.covalent_radii
    :return: The distances between the atoms and the grid points.
    """

    from .helper import find_coord_from_indices
    import numpy as np
    from ase.data import covalent_radii

    if radii is None:
        radii = covalent_radii[ase_atoms.get_atomic_numbers()]
    ab = AABB_on_atoms(ase_atoms)
    ab_r = ab.query(
        find_coord_from_indices(path_indices, shape, ase_atoms),
        query_args=dict(mode="nearest", num_neighbors=n_neighbors, exclude_ii=True),
    )

    ab_r_nl = ab_r.toNeighborList(
        sort_by_distance=True
    )  # convert query to neighborlist
    nls = ab_r_nl.point_indices.reshape(
        -1, n_neighbors
    )  # get the indices of the neighbors for each grid pt. to lookup the radii later
    dists = ab_r_nl.distances.reshape(
        -1, n_neighbors
    )  # get the distances to the neighbors of each grid pt.
    values = np.min(dists - [radii[nl] for nl in nls], axis=1)
    return values


# def get_psd_for_region(region_index, connections, dgrid, maxima, shape, ase_atoms, n_neighbors=10, radii=None, points_per_path=1000, set_bins=50, return_bins=False):
#     """
#     This function takes in a region index, the connections between regions, the distance grid, the
#     maxima, the shape of the distance grid, the ase atoms object, the number of neighbors to consider,
#     the radii of the atoms, the number of points to sample along the path, the number of bins to use for
#     the histogram, and a boolean to return the bins or not.

#     It returns the histogram values and the bins if return_bins is True, otherwise it returns just the
#     histogram values.

#     :param region_index: The index of the region for which you want to get the psd
#     :param connections: list of tuples of the form (region_index, region_index)
#     :param dgrid: distance grid
#     :param maxima: the maxima of the distance map
#     :param shape: shape of the grid
#     :param ase_atoms: ASE atoms object
#     :param n_neighbors: number of neighbors to consider for each point in the path, defaults to 10
#     (optional)
#     :param radii: list of radii for which you want to calculate the psd. If you want to calculate the
#     psd for a single radius, then you can pass a single value
#     :param points_per_path: number of points to sample along the path, defaults to 1000 (optional)
#     :param set_bins: The number of bins to use for the histogram, defaults to 50 (optional)
#     :param return_bins: If True, returns the bin centers as well as the histogram values, defaults to
#     False (optional)
#     :return: the psd for a given region.
#     """

#     import numpy as np
#     #* get the paths
#     connected_regions = np.vstack([c[0] for c in connections])
#     if np.any(connected_regions == region_index):


#         #* get the paths
#         paths = get_sampling_paths_for_region(region_index, connections, dgrid, maxima, shape, ase_atoms)


#         #* resample the paths to make them finer
#         paths = [sample_path(path, points_per_path=points_per_path) for path in paths]

#         #* get the distances for each path
#         distances = []
#         for path in paths:
#             distances.append(get_distances_for_path(path, dgrid.shape, ase_atoms, n_neighbors=n_neighbors,radii=radii))

#         #* get the psd for all the paths
#         vals, bins = np.histogram(np.concatenate(distances), density=True, bins=set_bins)
#         # binc = (bins[1:]+bins[:-1])/2
#         if return_bins:
#             binc = (bins[1:]+bins[:-1])/2
#             return vals, binc
#         else:
#             return vals
#     else:
#         print('The region {} is not connected to any other region. Hence unable to detect cavity axis for sampling'.format(region_index))
#         return 'NaN', 'NaN'


def apply_pbc(
    regions,
    maxima,
    maxima_radii,
    ase_atoms,
    wall_thickness=1,
    return_conn=False,
    minimum_overlap_fraction=0.5,
):
    """
    It takes a 3D array of regions and returns a 3D array of regions with periodic boundary conditions
    applied

    :param regions: the regions array
    :param maxima: the list of maxima
    :param maxima_radii: the radius of each region
    :param ase_atoms: The atoms object from ASE
    :param wall_thickness: The thickness of the wall in Angstroms, defaults to 1 (optional)
    :param return_conn: if True, returns a list of lists of connected regions, defaults to False
    (optional)
    :param minimum_overlap_fraction: This is the minimum fraction of overlap at the wall region between two regions for
    them to be considered for connected via PBC. Then we check the maxima locations also to make sure, they are not separated by periodic windows.
    :return: a list of connected components. Each connected component is a list of regions a.k.a the PBC group.
    """

    import copy

    # a = 0
    import numpy as np
    from icecream import ic

    ic.disable()
    regions_pbc = copy.deepcopy(regions)
    shape = regions.shape
    import networkx as nx

    g1 = nx.Graph()
    g1.add_nodes_from(range(len(np.unique(regions))))

    # * we need to recompute the wall thickness for non orthorhombic cells
    # * for this we shall use the perpendicualr length , which is the norm of the reciprocal lattice vector in each direction
    from ase import geometry

    # * let's compute the perpendicular length of each voxel
    # * we will use the reciprocal lattice vectors to do this
    # * we need this to determine how many voxels we need to select as the wall of the unit cell in each direction
    la = ase_atoms.get_cell_lengths_and_angles()[0] / regions.shape[0]
    lb = ase_atoms.get_cell_lengths_and_angles()[1] / regions.shape[1]
    lc = ase_atoms.get_cell_lengths_and_angles()[2] / regions.shape[2]
    alpha = ase_atoms.get_cell_lengths_and_angles()[3] * (np.pi / 180.0)
    beta = ase_atoms.get_cell_lengths_and_angles()[4] * (np.pi / 180.0)
    gamma = ase_atoms.get_cell_lengths_and_angles()[5] * (np.pi / 180.0)
    vol = ase_atoms.get_volume() / regions.size
    eA = [la, 0, 0]
    eB = [lb * np.cos(gamma), lb * np.sin(gamma), 0]
    eC = [
        lc * np.cos(beta),
        lc * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        vol / (la * lb * np.sin(gamma)),
    ]

    la_perp = vol / np.linalg.norm(np.cross(eB, eC))
    lb_perp = vol / np.linalg.norm(np.cross(eC, eA))
    lc_perp = vol / np.linalg.norm(np.cross(eA, eB))

    wall_cells = np.ceil(
        [wall_thickness / la_perp, wall_thickness / lb_perp, wall_thickness / lc_perp]
    ).astype(int)

    for a in [0, 1, 2]:
        # ic(a)
        # print('a', a)

        # * Here we are comparing the left wall and the right wall, so to get th ecorrespoding regions,
        # * we have to mirror the right wall
        llr = regions_pbc.take(
            indices=np.arange(0, wall_cells[a]), axis=a
        )  # * left wall
        rrr = regions_pbc.take(
            indices=np.arange(shape[a] - 1, shape[a] - 1 - wall_cells[a], -1), axis=a
        )  # * right wall
        llf = llr.flatten()
        rrf = rrr.flatten()
        # ic(dict(zip(llr.flatten(), rrr.flatten())))
        # ic(dict(zip(rrr.flatten(),llr.flatten())))
        # dlist = [dict(sorted(zip(llr.flatten(), rrr.flatten()))),dict(sorted(zip(rrr.flatten(),llr.flatten())))]

        # * Check left vs right wall
        # map_dict = dlist[0]
        # map_dict.pop(0)
        # ic(map_dict)

        map_list = np.unique(list(zip(llr.flatten(), rrr.flatten())), axis=0)
        map_list = map_list[np.all(map_list != 0, axis=1)]
        ic(map_list)
        # print('map_list', map_list)
        if len(map_list) > 0:
            # * forwards check
            reg_numbers = np.vstack([(k, v) for k, v in map_list])

            # * let's compute and prnt percentage overlap between pairs of mapping regions
            # * let's filter those connections with less thant 10% overlap
            sumlist = [np.sum(llf == r[0]) for r in reg_numbers]
            matching = [
                np.sum(np.logical_and(llf == r[0], rrf == r[1])) for r in reg_numbers
            ]
            fractional_overlap = [m / s for m, s in zip(matching, sumlist)]

            # * flags where there should be a left to right connection
            left_right = [f > minimum_overlap_fraction for f in fractional_overlap]

            # reg_numbers = reg_numbers[np.where([f > minimum_overlap_fraction for f in fractional_overlap])]
            # print('left-right\n')
            # print(list(zip(reg_numbers, fractional_overlap)))

            # * backwards check

            # #* let's compute and prnt percentage overlap between pairs of mapping regions
            # #* let's filter those connections with less thant minimum overlap fraction
            sumlist = [np.sum(rrf == r[1]) for r in reg_numbers]
            matching = [
                np.sum(np.logical_and(llf == r[0], rrf == r[1])) for r in reg_numbers
            ]
            fractional_overlap = [m / s for m, s in zip(matching, sumlist)]

            # * flags where there should be a right to left connection
            right_left = [f > minimum_overlap_fraction for f in fractional_overlap]

            # print('right-left\n')
            # print(list(zip(reg_numbers, fractional_overlap)))
            # now let's combine the two flags
            pick_these = np.logical_or(left_right, right_left)

            reg_numbers = reg_numbers[pick_these]
            # print(pick_these)
            # print()
            # print(reg_numbers, 'reg numbers after picking')
            if len(reg_numbers) > 0:
                # TODO: Here we are asuuming the region indices to be in order, causes error in older skimage versions fix using props['labels]
                flags = np.vstack(
                    [
                        [
                            maxima[r[0] - 1][a] < wall_cells[a],
                            maxima[r[1] - 1][a] > (shape[a] - wall_cells[a]),
                        ]
                        for r in reg_numbers
                    ]
                )
                # flags = np.logical_or(flags[:,0],flags[:,1,])
                # flags = np.logical_or(flllags[:,0],flags[:,1,])
                flags1 = np.logical_and(flags[:, 0], flags[:, 1])
                # print('flags1\n',list(zip(reg_numbers,flags1)))

                flags2 = np.logical_xor(flags[:, 0], flags[:, 1])
                # print('flags2\n',list(zip(reg_numbers,flags2)))
                reg_radii = np.vstack(
                    [
                        [maxima_radii[r[0] - 1], maxima_radii[r[1] - 1]]
                        for r in reg_numbers
                    ]
                )
                flags3 = np.vstack(
                    [
                        reg_radii[i][np.where(f)]
                        < reg_radii[i][np.where(np.invert(f))][0]
                        if flags2[i]
                        else False
                        for i, f in enumerate(flags)
                    ]
                )
                # print('flags3\n',list(zip(reg_numbers,flags3)))
                # flags3 = np.vstack([ True if flags2[i] else False for i,f,r  in enumerate(list(zip(flags, reg_numbers)))])
                # reg_radii = [maxima_radii[r[0]-1], maxima_radii[r[1]-1] for r in reg_numbers]

                flags = [
                    flags1[i] if flags1[i] else flags2[i] if flags3[i] else False
                    for i, f in enumerate(flags)
                ]
                # print('after checks\n',list(zip(reg_numbers,flags)))
                # ic(flags1)
                # ic(flags2)
                # ic(flags3)
                # ic(flags)       # flags = [True if flags1[i] else True if np.logical_and(flags2[i], flags3[i] ) else False for i,f in enumerate(flags)]

                # * filter based on region numbers
                reg_numbers = reg_numbers[np.where(flags)]

                # g1.add_edges_from(reg_numbers[np.where(flags)])
                g1.add_edges_from(reg_numbers)

    conn = nx.connected_components(g1)
    # ic(list(conn))
    # ic(list(G.edges()))
    # ic([c for c in list(conn)])
    for i, c in enumerate(conn):
        for r in c:
            regions_pbc[np.where(regions == r)] = i
            ic("replace {0} with {1}".format(r, i))

    # if reindex:
    #     # * Re-index the regions so that it runs from zero to N-regions.
    #     old_labels = np.unique(regions_pbc)  # This is sorted aalreadylready
    #     number_of_regions = len(old_labels)
    #     new_labels = range(number_of_regions)

    #     # * Replace the old-labels with new
    #     for i in range(number_of_regions):
    #         regions_pbc[regions_pbc== old_labels[i]] = new_labels[i]
    if return_conn:
        return list(nx.connected_components(g1))[1:]
    else:
        return regions_pbc


def add_pbc_to_rag(rag, pbc_groups):
    """
    It takes a RAG and a list of PBC groups, and adds a 'pbc_group' attribute to each node in the RAG

    :param rag: the region adjacency graph
    :param pbc_groups: a list of lists of labels that are in the same periodic boundary condition group
    :return: A rag with the pbc_group attribute added to each node.
    """
    import numpy as np

    for node in rag.nodes:
        pbc_index = np.where(
            [rag.nodes[node]["labels"][0] in list(g) for g in pbc_groups]
        )[0][0]
        rag.nodes[node]["pbc_group"] = pbc_index
    return rag


def clustering_grid_from_regions(regions, rag2, cluster_key="cluster"):
    import numpy as np

    regions_cluster = np.zeros(regions.shape)
    if cluster_key not in rag2.nodes[1].keys():
        ValueError(
            "The cluster_key is not in the rag2 nodes, call add_clustering_to_rag_* function first"
        )
    else:
        for i in range(1, len(rag2.nodes) + 1):
            regions_cluster[np.where(regions == rag2.nodes[i]["labels"][0])] = (
                int(rag2.nodes[i][cluster_key]) + 1
            )
    return regions_cluster


# region  # * Energy grid functions
#######################################################
############################
# * Let's put the energy grid functions in this section
############################


def get_repetitions_to_meet_cutoff(ase_atoms, cutoff=12.5):
    """
    It takes an ASE atoms object and a cutoff distance, and returns the number of unit cells in each
    direction that are needed to ensure that the cutoff distance is satisfied

    :param ase_atoms: ASE atoms object
    :param cutoff: the cutoff distance for the pairwise interactions
    :return: The number of repetitions in each direction to meet the cutoff.
    """

    import numpy as np

    la, lb, lc, alpha, beta, gamma = ase_atoms.get_cell_lengths_and_angles()
    alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
    vol = ase_atoms.get_volume()
    eA = [la, 0, 0]
    eB = [lb * np.cos(gamma), lb * np.sin(gamma), 0]
    eC = [
        lc * np.cos(beta),
        lc * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        vol / (la * lb * np.sin(gamma)),
    ]
    A = [eA, eB, eC]
    A = np.array(A)
    A = A.T
    A_inv = np.linalg.inv(A)
    lx_unit, ly_unit, lz_unit = (
        vol / np.linalg.norm(np.cross(eB, eC)),
        vol / np.linalg.norm(np.cross(eC, eA)),
        vol / np.linalg.norm(np.cross(eA, eB)),
    )
    nx_cells, ny_cells, nz_cells = (
        int(np.ceil(2 * cutoff / lx_unit)),
        int(np.ceil(2 * cutoff / ly_unit)),
        int(np.ceil(2 * cutoff / lz_unit)),
    )
    return nx_cells, ny_cells, nz_cells


def repeat_to_meet_cutoff(ase_atoms, forcefield_mixed, cutoff=12.5):
    """
    It takes an ASE atoms object and a forcefield, and returns a new ASE atoms object and a new
    forcefield that are repeated in such a way that the vdw cutoff is met

    :param ase_atoms: the atoms object from ASE
    :param forcefield_mixed: the forcefield parameters for the mixed system
    :param cutoff: the cutoff distance for the forcefield
    :return: ase_atoms_repeat is a supercell of ase_atoms, forcefield_mixed_repeat is a tiled array of
    forcefield_mixed
    """

    # * let's find the repetitions needed to meet cutoff
    import numpy as np

    repeats = get_repetitions_to_meet_cutoff(ase_atoms, cutoff=cutoff)
    nrepeats = np.prod(repeats)

    # * make the P matrix for the supercell
    P = np.eye(3)
    from ase.build import make_supercell

    nrepeats = np.prod(repeats)
    np.fill_diagonal(P, val=list(repeats))

    # * let's repeat everything
    ase_atoms_repeat = make_supercell(ase_atoms, P)

    forcefield_mixed_repeat = np.tile(forcefield_mixed, (nrepeats, 1))
    return ase_atoms_repeat, forcefield_mixed_repeat


def gpe_aabb2(grid_points, positions, cell, forcefield_mixed, cutoff):
    """
    The function `gpe_aabb2` calculates Lennard-Jones energy contributions for grid points based on
    neighboring particles within a specified cutoff distance using positions instead of neighborlist passed as 
    argument as in `gpe_aabb' subroutine.
    
    :param grid_points: The function `gpe_aabb2` seems to be calculating the Lennard-Jones potential
    energy for a set of grid points based on the positions of particles, a cell, a mixed force field,
    and a cutoff distance
    :param positions: The `positions` parameter in the `gpe_aabb2` function represents the positions of
    particles in a simulation. These positions are used to calculate the interactions between the
    particles and the grid points within a certain cutoff distance. The function performs calculations
    based on the positions of the particles and the grid points
    :param cell: The `cell` parameter in the `gpe_aabb2` function represents the simulation box or cell
    in which the particles are contained. It defines the periodic boundaries and the size of the
    simulation box in three dimensions. The `cell` parameter is used in the AABBQuery to specify the
    simulation box
    :param forcefield_mixed: The function `gpe_aabb2` seems to be calculating the Lennard-Jones
    potential energy for a given set of grid points based on the provided force field parameters. The
    key steps involved in the calculation include querying neighboring points within a specified cutoff
    distance, computing the Lennard-Jones
    :param cutoff: The `cutoff` parameter in the `gpe_aabb2` function represents the maximum distance
    within which to consider neighboring points when performing calculations. It is used to filter out
    points that are beyond this distance from each grid point during the computation
    :return: The function `gpe_aabb2` is returning the calculated Lennard-Jones potential energy values
    `v` for each grid point based on the provided input parameters.
    """

    import numpy as np

    from freud.locality import AABBQuery

    ab_r = AABBQuery(cell, positions).query(
        grid_points, query_args=dict(mode="ball", r_max=cutoff, exclude_ii=True)
    )
    # ab_r=aabb_object.query(grid_points,query_args=dict(mode='ball',r_max=cutoff,exclude_ii=True))
    ab_r_nl = ab_r.toNeighborList(sort_by_distance=True)  # convertquerytoneighborlist
    # nls=ab_r_nl.point_indices.reshape(-1,n_neighbors)#gettheindicesoftheneighborsforeachgridpt.tolookuptheradiilater
    # dists=ab_r_nl.distances.reshape(-1,n_neighbors)#getthedistancestotheneighborsofeachgridpt.
    # print(ab,ab_r,ab_r_nl,nls,[radii[nl]fornlinnls])#,np.min(dists-[radii[nl]fornlinnls],axis=1))
    # returntheminimum(shortestdistancetothesurface)ineachrow

    indices = ab_r_nl.point_indices  # * frame indices within cutoff at each grid point
    distances = ab_r_nl.distances  # * distances to frame indices at each grid point
    segments = ab_r_nl.segments  # * number of frame indices at each grid point
    inter_params = np.take(
        forcefield_mixed, indices, axis=0
    )  # * get the parameters for each frame index at each grid point
    sig = inter_params[
        :, 1
    ]  # * get the sigma values for each frame index at each grid point
    eps = inter_params[
        :, 0
    ]  # * get the epsilon values for each frame index at each grid point
    sd = np.power(np.divide(sig, distances), 6)  # * (sigma/distance)^6
    lj = 4 * eps * (sd ** 2 - sd)  # * Lennard-Jones terms

    ncounts = ab_r_nl.neighbor_counts  # * number of neighbors at each grid point
    grid_point_indices = np.repeat(
        np.arange(len(ncounts)), ncounts
    )  # * repeat the grid point indices for each neighbor

    v = np.bincount(
        grid_point_indices, weights=lj, minlength=len(ncounts)
    )  # * sum the energies for each grid point

    # sig = np.array([forcefield[i][1] for i in indices])
    # eps = np.array([forcefield[i][0] for i in indices])

    # print(len(indices),len(distances),len(segments),len(sig),len(eps)) #, np.take(forcefield, indices))
    # v = np.add.reduceat(
    #     np.multiply(4 * eps, ((sd) ** 2 - (sd))), segments
    # )  # * sum the energy contributions for each grid point
    # E = np.add.reduceat(4*eps*((sig/distances)**12-(sig/distances)**6), segments, axis=0)

    # E = [lj_sum(d, s, e) for d, s, e in zip(distances, sig, eps)]

    # return np.vstack(E).reshape(-1,1)
    return v


# def gpe_aabb(grid_points, aabb_object, forcefield_mixed, cutoff):
#     import numpy as np

#     ab_r = aabb_object.query(
#         grid_points, query_args=dict(mode="ball", r_max=cutoff, exclude_ii=True)
#     )

#     ab_r_nl = ab_r.toNeighborList(sort_by_distance=True)

#     indices = ab_r_nl.point_indices
#     distances = ab_r_nl.distances
#     segments = ab_r_nl.segments
#     inter_params = np.take(forcefield_mixed, indices, axis=0)
#     sig = inter_params[:, 1]
#     eps = inter_params[:, 0]
#     sd = np.power(np.divide(sig, distances), 6)

#     # Initialize v with zeros
#     v = np.zeros(len(grid_points))

#     # Only perform reduction on grid points with neighbors
#     has_neighbors = np.diff(segments, prepend=-1) != 0
#     v[has_neighbors] = np.add.reduceat(
#         np.multiply(4 * eps, ((sd) ** 2 - (sd))), segments[has_neighbors]
#     )

#     return v
def gpe_aabb(grid_points, aabb_object, forcefield_mixed, cutoff):
    """
    The function `gpe_aabb` calculates Lennard-Jones potentials for grid points based on their distances
    to neighboring points within a specified cutoff distance.
    
    :param grid_points: The `grid_points` parameter in the `gpe_aabb` function represents the
    coordinates of points in a grid. These points are used for calculations involving a AABB
    (Axis-Aligned Bounding Box) object, a force field, and a specified cutoff distance. The function
    calculates Lennard-J
    :param aabb_object: The function `gpe_aabb` you provided seems to calculate the Lennard-Jones
    potentials for a set of grid points based on an axis-aligned bounding box (AABB) object, a force
    field, and a cutoff distance
    :param forcefield_mixed: The `forcefield_mixed` parameter in the `gpe_aabb` function seems to
    represent a mixed force field containing interaction parameters for each pair of particles. The
    function calculates Lennard-Jones potentials between grid points and their neighboring particles
    based on these interaction parameters
    :param cutoff: The `cutoff` parameter in the `gpe_aabb` function represents the maximum distance
    within which interactions between grid points and AABB (Axis-Aligned Bounding Box) objects will be
    considered. Any interactions beyond this distance will not be included in the calculations
    :return: The function `gpe_aabb` returns an array `v` containing the Lennard-Jones potentials
    calculated for each grid point based on the provided input parameters `grid_points`, `aabb_object`,
    `forcefield_mixed`, and `cutoff`.
    """


    import numpy as np
    ab_r = aabb_object.query(
        grid_points, query_args=dict(mode="ball", r_max=cutoff, exclude_ii=True)
    )

    ab_r_nl = ab_r.toNeighborList(sort_by_distance=True)  # Convert query to neighbor list
    indices = ab_r_nl.point_indices
    distances = ab_r_nl.distances
    # segments = ab_r_nl.segments
    inter_params = np.take(forcefield_mixed,indices, axis=0)
    sig = inter_params[:, 1]
    eps = inter_params[:, 0]
    # Initialize energies array
    # segments = np.cumsum(ab_r_nl.neighbor_counts)
    # segments = np.insert(segments, 0, 0)
    # print(segments)
    # Calculate the scaled distances (sigma/distance)^6
    ncounts = ab_r_nl.neighbor_counts
    # print('number of entries',  len(distances))
    # print('len of neighbors ', len(ncounts))
    # print('total numbero f neighbors', np.sum(ncounts))
    # print('ncounts',    ab_r_nl.neighbor_counts)
    # print('number of grid points', len(grid_points))

    # compute the corresponding Lennard-Jones terms
    sd = np.power(np.divide(sig,distances), 6)
    lj_potentials = 4 * eps * (sd ** 2 - sd)
    # repeat the grid point indices for each neighbor
    grid_point_indices = np.repeat(np.arange(len(ncounts)), ncounts)

    # Use np.bincount to sum energies for each grid point
    # Weights are the energies, and minlength ensures the result includes all grid points
    v = np.bincount(grid_point_indices, weights=lj_potentials, minlength=len(ncounts))

    # Use np.add.reduceat to sum potentials for each segment
    # v = np.add.reduceat(lj_potentials, segments[:-1])
    # print('Len of energies ', len(v))
    # print('min of energies ', v.min())
    # print('grid point with min energy', grid_points[v.argmin()])
    return v

# def gpe_aabb(grid_points, aabb_object, forcefield_mixed, cutoff):
#     """
#     It takes a list of grid points,an AABB query object from freud, a list of forcefield parameters, and a cutoff
#     distance, and returns the energy of each grid point

#     :param grid_points: the points you want to evaluate the energy at
#     :param aabb_object: aabb object from the aabbquery class from freud
#     :param forcefield_mixed: a numpy array of shape (n,2) where n is the number of atoms in the system.
#     The first column is the epsilon parameter and the second column is the sigma parameter
#     :param cutoff: the cutoff distance for the neighbor search
#     :return: The energy of each grid point.
#     """

#     # def lj_sum(dists, sig, eps):
#     #     return np.sum(4*eps*((sig/dists)**12-(sig/dists)**6))

#     # Performthenearestneighborquerywithn-neighbors(10bydefault)

#     import numpy as np

#     ab_r = aabb_object.query(
#         grid_points, query_args=dict(mode="ball", r_max=cutoff, exclude_ii=True)
#     )

#     ab_r_nl = ab_r.toNeighborList(sort_by_distance=True)  # convertquerytoneighborlist
#     # nls=ab_r_nl.point_indices.reshape(-1,n_neighbors)#gettheindicesoftheneighborsforeachgridpt.tolookuptheradiilater
#     # dists=ab_r_nl.distances.reshape(-1,n_neighbors)#getthedistancestotheneighborsofeachgridpt.
#     # print(ab,ab_r,ab_r_nl,nls,[radii[nl]fornlinnls])#,np.min(dists-[radii[nl]fornlinnls],axis=1))
#     # returntheminimum(shortestdistancetothesurface)ineachrow
#     indices = ab_r_nl.point_indices
#     distances = ab_r_nl.distances
#     segments = ab_r_nl.segments
#     inter_params = np.take(forcefield_mixed, indices, axis=0)
#     sig = inter_params[:, 1]
#     eps = inter_params[:, 0]
#     sd = np.power(np.divide(sig, distances), 6)
#     # sig = np.array([forcefield[i][1] for i in indices])
#     # eps = np.array([forcefield[i][0] for i in indices])

#     # print(len(indices),len(distances),len(segments),len(sig),len(eps)) #, np.take(forcefield, indices))
#     v = np.add.reduceat(np.multiply(4 * eps, ((sd) ** 2 - (sd))), segments)
#     # E = np.add.reduceat(4*eps*((sig/distances)**12-(sig/distances)**6), segments, axis=0)

#     # E = [lj_sum(d, s, e) for d, s, e in zip(distances, sig, eps)]

#     # return np.vstack(E).reshape(-1,1)
#     return v
#

# def aabb_for_atoms(ase_atoms):

#     from freud.locality import AABBQuery
#     from freud.box import Box
#     import ase
#     #fromase.dataimportvdw_radii
#     import pandas as pd
#     import numpy as np

#     framework_atom_positions=ase_atoms.get_positions()
#     cell=ase.geometry.complete_cell(ase_atoms.get_cell()).T#thisneedstobeanuppertriangularmatrix,essentialcolumnvectors,notrowvectors

#     box=Box.from_matrix(cell,dimensions=3)#DefinetheboxforFreud
#     aabb=AABBQuery(box=box,points=framework_atom_positions)#ComputetheAABBTree

#     return aabb


def egrid_from_atoms2(
    ase_atoms,
    spacing=0.5,
    blocksize=20000,
    forcefield_mixed=None,
    cutoff=12.5,
    probe_symbol="He",
    return_ncells=False,
    precompute_aabb=True,
):
    """
    > It takes an ASE atoms object, and returns a dask array of the vdw energy on a grid.

    The function is a bit long, but it's not too complicated.

    The first part of the function is just to get the grid points.

    The second part is to get the forcefield parameters for the atoms.

    The third part is to get the positions of the atoms in the unit cell, and to get the unit cell
    vectors.

    The fourth part is to repeat the unit cell to get the positions of the atoms in the repeated unit
    cell.

    The fifth part is to map the grid point energy function onto the grid points.

    The sixth part is to reshape the grid points into a 3D array.

    The seventh part is to return the grid points.

    The eighth part is to return the number of repetitions of the unit cell if that return_ncells parameter is turned on.

    :param ase_atoms: an ase atoms object
    :param spacing: the spacing of the grid points
    :param blocksize: the size of the chunks that the grid is split into. This is important for memory
    management, defaults to 20000 (optional)
    :param forcefield_mixed: This is a list of the forcefield parameters for each atom in the unit cell
    :param cutoff: the cutoff distance for the Lennard-Jones potential
    :param probe_symbol: the symbol of the probe atom used to calculate the van der Waals radii,
    defaults to He (optional)
    :param return_ncells: if True, returns the number of cells needed to meet the cutoff, defaults to
    False (optional)
    :param precompute: if True, then the AABB is precomputed and passed to the function. If False, then
    the positions and cell are passed to the function, defaults to True (optional)
    :return: The grid2 is a dask array, which is a lazy array. It is not computed until you call
    compute() on it.
    """

    import dask.array as da
    import numpy as np
    from .data import mix_lorentz_berthelot
    from .data import forcefields
    import ase

    # import mofography as mgr
    grid, shape = dask_grid_over_atoms(ase_atoms, spacing=spacing, chunksize=blocksize)

    # ab = aabb_for_atoms(ase_atoms)
    if forcefield_mixed is None:
        forcefield_mixed = mix_lorentz_berthelot(
            ase_atoms, forcefield=forcefields["uff"], probe_symbol=probe_symbol
        )
    # grid.rechunk(chunks=(10000,3))

    # * Now here use the the repeated box to get the framework atom positions
    # * also the grid point coordinates now only span one unit cell, since others are simply copies

    from ase.build import make_supercell

    P = np.eye(3)
    repeats = get_repetitions_to_meet_cutoff(ase_atoms, cutoff=cutoff)
    nrepeats = np.prod(repeats)
    np.fill_diagonal(P, val=list(repeats))

    # * let's repeat everything
    ase_atoms_repeat = make_supercell(ase_atoms, P)
    positions_repeat = ase_atoms_repeat.get_positions()
    cell_repeat = ase.geometry.complete_cell(ase_atoms_repeat.get_cell()).T
    forcefield_mixed_repeat = np.tile(forcefield_mixed, (nrepeats, 1))

    # mix_lorentz_berthelot(ase_atoms_repeat, forcefield = forcefields['uff'], probe_symbol=probe_symbol)
    if precompute_aabb:
        ab_repeat = AABB_on_atoms(ase_atoms_repeat)
        # * map the grid point energy function that uses position and cell, on to the grid points
        grid2 = da.map_blocks(
            gpe_aabb,
            grid,
            ab_repeat,
            forcefield_mixed_repeat,
            cutoff=cutoff,
            dtype=np.float64,
            drop_axis=1,
        )
        grid2 = grid2.reshape(shape)
        grid2 = (
            da.from_array(grid2.compute()).rechunk(chunks=(10, 10, 10)).reshape(shape)
        )

    else:
        # * map the grid point energy function that uses position and cell, on to the grid points
        grid2 = da.map_blocks(
            gpe_aabb2,
            grid,
            positions_repeat,
            cell_repeat,
            forcefield_mixed_repeat,
            cutoff=cutoff,
            dtype=np.float64,
            drop_axis=1,
        )
        grid2 = grid2.reshape(shape)

    if return_ncells:
        return grid2, get_repetitions_to_meet_cutoff(ase_atoms, cutoff=cutoff)
    else:
        return grid2


def get_energies_for_group(
    group,
    props,
    ase_atoms,
    shape,
    forcefield_mixed=None,
    cutoff=12.5,
    return_coords=False,
    block_size=20000,
    sample_fraction=1,
):
    from .helper import find_coord_from_indices
    import numpy as np
    import dask.array as da
    from .data import mix_lorentz_berthelot

    # * assign forcefield_mixed if it is not provided
    if forcefield_mixed is None:
        forcefield_mixed = mix_lorentz_berthelot(ase_atoms)

    # * repeat the ase atoms and forcefield to meet cutoff

    # nrepeats = np.prod(mgr.get_repetitions_to_meet_cutoff(ase_atoms, cutoff))
    ase_atoms_repeat, forcefield_mixed_repeat = repeat_to_meet_cutoff(
        ase_atoms, forcefield_mixed, cutoff
    )
    # import numpy as np

    # * get aabb object on repeated system
    ab_repeat = AABB_on_atoms(ase_atoms_repeat)
    # nrepeats = np.prod(mgr.get_repetitions_to_meet_cutoff(ase))

    # ase_atom_repeat = mgr.get_aseatoms_to_meet_cutoff(ase_atoms, cutoff)
    # forcefield_mixed_repeat = np.tile()
    # * get region indices for the regions in the group
    prop_labels = np.array([prop["label"] for prop in props])
    prop_indices = [np.where(prop_labels == g)[0][0] for g in group]

    # * this group has more than one entry, assuming it is a pbc group
    if len(group) > 1:
        # * grid indices of all the points present in this group
        coords_indices = np.vstack([props[pi]["coords"] for pi in prop_indices])

        # * get the actual grid point coordinates from indices for the pbc_group
        coords = find_coord_from_indices(coords_indices, shape, ase_atoms)

        # * if there are too many points, it might be better to random sample
        if sample_fraction < 1:
            # * sample the coordinates
            ncoords = coords.shape[0]
            ncoords_sample = int(ncoords * sample_fraction)
            print(f"{ncoords_sample} points sampled for energy for pbc group {group}")
            coords = coords[np.random.choice(ncoords, ncoords_sample, replace=False), :]

        # * convert the coords to a dask array and chunk it
        coords = da.from_array(coords, chunks=(block_size, 3))

        # * map the gpe_aabb function on the coords and compute the energies
        energies = coords.map_blocks(
            gpe_aabb,
            ab_repeat,
            forcefield_mixed_repeat,
            cutoff,
            dtype=np.float64,
            drop_axis=1,
        ).compute()

        # * get the grid point energies for the points in the group
        # energies = gpe_aabb(coords, ab_repeat, forcefield_mixed_repeat, cutoff)

    else:
        # * if it is only one entry in the pbc group

        pi = np.where(prop_labels == list(group)[0])[0][0]
        # print(pi)
        # * get coordinates and centroids
        coords_indices = props[pi]["coords"]
        coords = find_coord_from_indices(coords_indices, shape, ase_atoms)
        # centroid = np.mean(coords, axis=0)

        # * convert coords into a  dask array and chunk it
        coords = da.from_array(coords, chunks=(block_size, 3))

        # * if there are too many points it might be better to randomly sample
        if sample_fraction < 1:
            # * sample the coordinates
            ncoords = coords.shape[0]
            ncoords_sample = int(ncoords * sample_fraction)
            print(f"{ncoords_sample} points sampled for energy for pbc group {group}")
            coords = coords[np.random.choice(ncoords, ncoords_sample, replace=False), :]

        # * compute the enrgies in parallel
        energies = coords.map_blocks(
            gpe_aabb,
            ab_repeat,
            forcefield_mixed_repeat,
            cutoff,
            dtype=np.float64,
            drop_axis=1,
        ).compute()
        # energies = gpe_aabb(coords, ab_repeat, forcefield_mixed_repeat, cutoff)

    if return_coords:
        return energies, coords
    else:
        return energies

    # def add_vdw_hist_to_rag_with_pbc(rag, props, pbc_groups, ase_atoms, forcefield_mixed, shape, bins_energy=None , density=True, cutoff=12.5, minimum_npockets_per_type=3, add_energies=False):

    #     import numpy as np
    #     import networkx as nx
    #     if 'cluster' not in rag.nodes[1].keys():
    #         raise ValueError('RAG must have a cluster attribute. Follow examples in docs to add clustering to RAG')
    #     else:
    #         types_list = [rag.nodes[list(g)[0]]['cluster'] for g in pbc_groups]

    #         if bins_energy is None:
    #             bins_energy = np.linspace(-3000, 100, 100)

    #         # list(nx.get_node_attributes(rag2, 'cluster').values())
    #         ctypes, cindices = np.unique(types_list,return_inverse=True)
    #         # for c in ctypes:
    #         types_indices = [np.where([t==c for t in types_list])[0] for c in ctypes]
    #         types_counts = [len(t) for t in types_indices]
    #         type_selection = [np.random.choice(ti , minimum_npockets_per_type, replace=False) if types_counts[i]>minimum_npockets_per_type else ti  for i,ti in enumerate(types_indices)]

    #         import itertools
    #         merged_type_selection = [list(itertools.chain(*[list(pbc_groups[t]) for t in ts])) for ts in type_selection]
    #         energies_list = [get_energies_for_group(mts, props, ase_atoms, forcefield_mixed, shape) for mts in merged_type_selection]
    #         histograms_bins= [np.histogram(e, bins=bins_energy, density=density) for e in energies_list]
    #         ehistograms = [hb[0] for hb in histograms_bins]
    #         ebincenters = [(hb[1][:-1]+hb[1][1:])/2 for hb in histograms_bins]
    #         ehists_data = [ehistograms[cindices[i]] for i,g in enumerate(pbc_groups) for gg in list(g)]
    #         ebins_data =  [ebincenters[cindices[i]] for i,g in enumerate(pbc_groups) for gg in list(g)]

    #         for node in rag.nodes():
    #             rag.nodes[node]['vdw_hist_y_pbc']=ehistograms[cindices[rag.nodes[node]['pbc_group']]]
    #             rag.nodes[node]['vdw_hist_x_pbc']=ebincenters[cindices[rag.nodes[node]['pbc_group']]]
    #         # nx.set_node_attributes(rag, dict(zip(rag.nodes,ehists_data)), 'vdw_hist_y_pbc')
    #         # nx.set_node_attributes(rag, dict(zip(rag.nodes,ebins_data)), 'vdw_hist_x_pbc')
    #         if add_energies:
    #             for node in rag.nodes():
    #                 rag.nodes[node]['energies']=energies_list[cindices[rag.nodes[node]['pbc_group']]]

    #             # energies_nodes = [energies_list[c] for c in cindices]
    #             # nx.set_node_attributes(rag, dict(zip(rag.nodes, energies_nodes)), 'vdw_energies_pbc')

    #     return rag

    # def add_vdw_hist_to_rag(rag, props, ase_atoms, forcefield_mixed, shape, use_pbc=False, pbc_groups=None, bins_energy=None , density=False, cutoff=12.5, add_energies=False, sample_fraction=1):
    #     import numpy as np
    #     import networkx as nx
    #     # if bins_energy is None:
    #         # bins_energy = np.linspace(-3000, 100, 25)
    #     if use_pbc:
    #         if pbc_groups is None:
    #             raise ValueError('pbc_groups must be provided if use_pbc=True')

    #         else:
    #             energies_list = [get_energies_for_group(list(g), props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff, sample_fraction=sample_fraction) for g in pbc_groups]
    #             if bins_energy is None:
    #                 hist, bins_energy = np.histogram(np.hstack(energies_list), bins=30, density=density)
    #             histograms_bins= [np.histogram(e, bins=bins_energy, density=density) for e in energies_list]
    #             ehistograms = [hb[0] for hb in histograms_bins]
    #             ebincenters = [(hb[1][:-1]+hb[1][1:])/2 for hb in histograms_bins]
    #             for i, node in enumerate(rag.nodes()):
    #                 rag.nodes[node]['vdw_hist_y_pbc']=ehistograms[rag.nodes[node]['pbc_group']]
    #                 rag.nodes[node]['vdw_hist_x_pbc']=ebincenters[rag.nodes[node]['pbc_group']]

    #     else:
    #         energies_list = [get_energies_for_group([node], props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff, sample_fraction=sample_fraction) for node in rag.nodes()]
    #         histograms_bins= [np.histogram(e, bins=bins_energy, density=density) for e in energies_list]
    #         ehistograms = [hb[0] for hb in histograms_bins]
    #         ebincenters = [(hb[1][:-1]+hb[1][1:])/2 for hb in histograms_bins]

    #         for i, node in enumerate(rag.nodes()):
    #             rag.nodes[node]['vdw_hist_y']=ehistograms[i]
    #             rag.nodes[node]['vdw_hist_x']=ebincenters[i]
    #     # nx.set_node_attributes(rag, dict(zip(rag.nodes,ehists_data)), 'vdw_hist_y_pbc')
    #     # nx.set_node_attributes(rag, dict(zip(rag.nodes,ebins_data)), 'vdw_hist_x_pbc')
    #     if add_energies:
    #         for i,node in rag.nodes():
    #             rag.nodes[node]['energies']=energies_list[i]

    #             # energies_nodes = [energies_list[c] for c in cindices]
    #             # nx.set_node_attributes(rag, dict(zip(rag.nodes, energies_nodes)), 'vdw_energies_pbc')

    #     return rag

    # def add_vdw_hist_to_rag2(rag, props, ase_atoms, forcefield_mixed, shape, use_pbc=False, pbc_groups=None, bins_energy=None , density=False, cutoff=12.5, add_energies=False):
    #     import numpy as np
    #     import networkx as nx
    #     if bins_energy is None:
    #         bins_energy = np.linspace(-3000, 100, 25)
    #     if use_pbc:
    #         if pbc_groups is None:
    #             raise ValueError('pbc_groups must be provided if use_pbc=True')
    #         else:
    #             energies_list = [get_energies_for_group(list(g), props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff) for g in pbc_groups]
    #             histograms_bins= [np.histogram(e, bins=bins_energy, density=density) for e in energies_list]
    #             ehistograms = [hb[0] for hb in histograms_bins]
    #             ebincenters = [(hb[1][:-1]+hb[1][1:])/2 for hb in histograms_bins]
    #             for i, node in enumerate(rag.nodes()):
    #                 rag.nodes[node]['vdw_hist_y_pbc']=ehistograms[rag.nodes[node]['pbc_group']]
    #                 rag.nodes[node]['vdw_hist_x_pbc']=ebincenters[rag.nodes[node]['pbc_group']]

    #     else:
    #         energies_list = [get_energies_for_group([node], props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff) for node in rag.nodes()]
    #         histograms_bins= [np.histogram(e, bins=bins_energy, density=density) for e in energies_list]
    #         ehistograms = [hb[0] for hb in histograms_bins]
    #         ebincenters = [(hb[1][:-1]+hb[1][1:])/2 for hb in histograms_bins]

    #         for i, node in enumerate(rag.nodes()):
    #             rag.nodes[node]['vdw_hist_y']=ehistograms[i]
    #             rag.nodes[node]['vdw_hist_x']=ebincenters[i]
    #     # nx.set_node_attributes(rag, dict(zip(rag.nodes,ehists_data)), 'vdw_hist_y_pbc')
    #     # nx.set_node_attributes(rag, dict(zip(rag.nodes,ebins_data)), 'vdw_hist_x_pbc')
    #     if add_energies:
    #         for i,node in rag.nodes():
    #             rag.nodes[node]['energies']=energies_list[i]

    #             # energies_nodes = [energies_list[c] for c in cindices]
    #             # nx.set_node_attributes(rag, dict(zip(rag.nodes, energies_nodes)), 'vdw_energies_pbc')

    return rag


# endregion
def get_energies_for_group2(
    group, regions, energy_interpolator=None, return_scaled_coords=False, max_sample_limit=10000, **kwargs_egrid
):  # ase_atoms, forcefield_mixed, shape, cutoff=12.5, return_coords=False, block_size=20000, sample_fraction=1):
    """
    This function calculates the energies for a group of regions using an energy interpolator and
    returns the energies and scaled coordinates if specified.

    :param group: A list of labels for the regions in the group
    :param regions: A 3D array representing a segmentation of a larger 3D volume into regions or
    objects. Each region is assigned a unique integer label
    :param energy_interpolator: A function that interpolates energies for a given set of coordinates. It
    is used to calculate the energies for a group of points in a region
    :param return_scaled_coords: A boolean parameter that determines whether the function should return
    the energies and the scaled coordinates of the points in the group or just the energies. If set to
    True, the function will return a tuple of two arrays: the energies and the scaled coordinates. If
    set to False, the function will only return the, defaults to False (optional)
    :param max_sample_limit: Maximum number of samples to take from the group, defaults to 10000 (optional). Use a larger
    max limit to get a smoother energy histogram, perhaps if you have a fine grid on a large pocket with nuanced surface features. 
    :return: either the energies for all the points in the specified group or the energies and scaled
    coordinates for all the points in the specified group, depending on the value of the
    `return_scaled_coords` parameter.
    """

    import numpy as np
    import random

    from skimage.measure import regionprops

    props = regionprops(regions)
    shape = regions.shape

    # * get region indices for the regions in the group
    prop_labels = np.array([prop["label"] for prop in props])
    prop_indices = [np.where(prop_labels == g)[0][0] for g in group]

    # * create an energy interpolotor if not passed
    if energy_interpolator is None:
        egrid = egrid_from_atoms2(**kwargs_egrid)
        from scipy.interpolate import RegularGridInterpolator as RGI

        xx = np.linspace(0, 1, egrid.shape[0])
        yy = np.linspace(0, 1, egrid.shape[1])
        zz = np.linspace(0, 1, egrid.shape[2])
        energy_interpolator = RGI((xx, yy, zz), egrid, method="slinear")

    # * this group has more than one entry, assuming it is a pbc group
    if len(group) > 1:
        # * grid indices of all the points present in this group

        all_group_indices = np.vstack([props[pi]["coords"] for pi in prop_indices])
        if len(all_group_indices) > max_sample_limit:
            all_group_indices = random.sample(list(all_group_indices), max_sample_limit)
        # * normalize the indices to get the fractional coordinates
        all_group_scaled = np.divide(all_group_indices, shape)

        # #* get the energy for all the points in the group
        # energies = energy_interpolator(all_group_scaled.T)
    else:
        # * if it is only one entry in the pbc group

        pi = np.where(prop_labels == list(group)[0])[0][0]
        # print(pi)
        # * get coordinates and centroids
        coords_indices = props[pi]["coords"]
        all_group_scaled = np.divide(coords_indices, shape)
    # * get the energy for all the points in the group
    energies = energy_interpolator(all_group_scaled)

    if return_scaled_coords:
        return energies, all_group_scaled
    else:
        return energies
def get_energy_interpolator(egrid, neighborhood_size=4):
    """
    Creates a 3D regular grid interpolator for energy values, handling NaN values by replacing them
    with a value higher than their local neighborhood maximum.

    :param egrid: a 3D numpy array representing a grid of energy values with shape (n_x, n_y, n_z)
    :param neighborhood_size: size of neighborhood to check in each direction (default: 4)
    :return: a RegularGridInterpolator object for the 3D energy grid
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI
    import warnings

    # Create a copy to avoid modifying the original array
    egrid_clean = np.copy(egrid)
    
    # Find indices of NaN values
    nan_indices = np.where(np.isnan(egrid_clean))
    
    if len(nan_indices[0]) > 0:
        warnings.warn(f"Found {len(nan_indices[0])} NaN values in energy grid. Replacing with high energy values.")
        
        # Process each NaN value
        for idx in zip(*nan_indices):
            # Define neighborhood bounds (handling edge cases)
            x_min, x_max = max(0, idx[0]-neighborhood_size), min(egrid.shape[0], idx[0]+neighborhood_size+1)
            y_min, y_max = max(0, idx[1]-neighborhood_size), min(egrid.shape[1], idx[1]+neighborhood_size+1)
            z_min, z_max = max(0, idx[2]-neighborhood_size), min(egrid.shape[2], idx[2]+neighborhood_size+1)
            
            # Get neighborhood values (excluding NaNs)
            neighborhood = egrid_clean[x_min:x_max, y_min:y_max, z_min:z_max]
            valid_values = neighborhood[~np.isnan(neighborhood)]
            
            if len(valid_values) > 0:
                # Set to 10% higher than the maximum value in neighborhood
                high_energy = np.max(valid_values) * 1.1
                warnings.warn(f"Replacing NaN at position {idx} with value {high_energy:.2e} "
                            f"(10% higher than local maximum {np.max(valid_values):.2e})")
            else:
                # If no valid neighbors, use a very high value
                high_energy = 1E8
                warnings.warn(f"No valid neighbors found for NaN at position {idx} in {neighborhood_size}x{neighborhood_size}x{neighborhood_size} "
                            f"neighborhood. Using default high value: {high_energy:.2e}")
            
            egrid_clean[idx] = high_energy
    
    xx = np.linspace(0, 1, egrid_clean.shape[0])
    yy = np.linspace(0, 1, egrid_clean.shape[1])
    zz = np.linspace(0, 1, egrid_clean.shape[2])
    
    energy_interpolator = RGI((xx, yy, zz), egrid_clean, method="slinear")
    return energy_interpolator

# def get_energy_interpolator(egrid):
#     """
#     This function creates a 3D regular grid interpolator for energy values given an input energy grid.

#     :param egrid: a 3D numpy array representing a grid of energy values. The shape of the array should
#     be (n_x, n_y, n_z), where n_x, n_y, and n_z are the number of grid points in the x, y, and z
#     directions, respectively
#     :return: a RegularGridInterpolator object that can be used to interpolate values from a 3D energy
#     grid.
#     """

#     import numpy as np
#     from scipy.interpolate import RegularGridInterpolator as RGI

#     xx = np.linspace(0, 1, egrid.shape[0])
#     yy = np.linspace(0, 1, egrid.shape[1])
#     zz = np.linspace(0, 1, egrid.shape[2])
#     energy_interpolator = RGI((xx, yy, zz), egrid, method="slinear")
#     return energy_interpolator


def add_clustering_to_rag_energy_js(
    rag,
    regions,
    energy_interpolator=None,
    use_pbc=True,
    pbc_groups=None,
    js_threshold=0.05,
    **kwargs_egrid,
):
    import networkx as nx
    import itertools
    from scipy.stats import ks_2samp
    from scipy.spatial.distance import jensenshannon
    import numpy as np
    import dask.bag as db
    from tqdm import tqdm

    # * get teh Jensen Shannon distance between the energy distributions of two regions
    def get_js_dist(t):
        i, j = t
        # e1 = get_energies_for_group2(list(pbc_groups[i]), regions, energy_interpolator=energy_interpolator)
        # e2 = get_energies_for_group2(list(pbc_groups[j]), regions, energy_interpolator=energy_interpolator)
        # remove inf and nan values
        e1 = energies_list[i]
        e2 = energies_list[j]
        e1 = e1[np.isfinite(e1)]
        e2 = e2[np.isfinite(e2)]
        hist, bins = np.histogram(np.hstack((e1, e2)), bins=20)
        e1_hist, bins = np.histogram(e1, bins=bins)
        e2_hist, bins = np.histogram(e2, bins=bins)
        return jensenshannon(e1_hist, e2_hist)

    # * make energy interpolator if not passed
    if energy_interpolator is None:
        egrid = egrid_from_atoms2(**kwargs_egrid)
        energy_interpolator = get_energy_interpolator(egrid)

    # * if PBC is to be maintained
    if use_pbc:
        if pbc_groups is None:
            raise ValueError("pbc_groups must be provided if use_pbc=True")

        # * we will be classifying the pbc groups
        # energies_list = [get_energies_for_group2(list(
        #     g), regions, energy_interpolator=energy_interpolator) for g in pbc_groups]

        # * define an empty graph
        js_graph = nx.Graph()
        js_graph.add_nodes_from(range(len(pbc_groups)))

        energies_list = [
            get_energies_for_group2(
                list(g), regions, energy_interpolator=energy_interpolator
            )
            for g in pbc_groups
        ]
        # * let's add edges between the pbc groups based on teh JS distance
        iterlist = list(itertools.combinations(range(len(pbc_groups)), 2))
        import dask.bag as db

        js_dist = db.from_sequence(iterlist).map(get_js_dist).compute()
        # * print the the JS distances between regions
        # print(js_dist)
        # * check wher teh JS distance is less than the threshold and add edges from iterlist
        add_these_edges = [
            iterlist[k] for k in range(len(js_dist)) if js_dist[k] < js_threshold
        ]
        # print('adding these edges: ', add_these_edges)
        js_graph.add_edges_from(add_these_edges)
        # for k, (i, j) in enumerate(iterlist):
        #     # js = get_js_dist(i, j)
        #     js = js_dist[k]
        #     print(
        #         f'Jensen-Shannon distance for {i}:{pbc_groups[i]} and {j}:{pbc_groups[j]} is {js}')
        #     if js < js_threshold:
        #         js_graph.add_edge(i, j)
        conn = list(nx.connected_components(js_graph))

        # print('connected components are: ', conn)
        energy_groups = [
            set(itertools.chain.from_iterable(np.array(pbc_groups)[list(c)]))
            for c in conn
        ]

        # * add the cluster info to the rag
        for i in rag.nodes():
            rag.nodes[i]["cluster_energy_pbc"] = str(
                np.where([i in g for g in energy_groups])[0][0]
            )
    else:
        # * we will be working with the actual nodes (regions)
        # energies_list = [get_energies_for_group2(
        #     [node], regions, energy_interpolator=energy_interpolator) for node in rag.nodes()]

        # energies_list = [get_energies_for_group(
        #     [node], props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff, sample_fraction=sample_fraction) for node in rag.nodes()]
        # * define empty graph
        js_graph = nx.Graph()
        js_graph.add_nodes_from(range(len(rag.nodes())))

        energies_list = [
            get_energies_for_group2(
                [node], regions, energy_interpolator=energy_interpolator
            )
            for node in rag.nodes()
        ]

        # * let's add edges between the pbc groups based on teh JS distance
        iterlist = list(itertools.combinations(range(len(rag.nodes())), 2))
        import dask.bag as db

        js_dist = db.from_sequence(iterlist).map(get_js_dist).compute()
        # * check wher teh JS distance is less than the threshold and add edges from iterlist
        add_these_edges = [
            iterlist[k] for k in range(len(iterlist)) if js_dist[k] < js_threshold
        ]
        # print('adding these edges: ', add_these_edges)
        js_graph.add_edges_from(add_these_edges)

        # for i, j in itertools.combinations(len(rag.nodes()), 2):
        #     js = get_js_dist(i,j)
        #     if js < js_threshold:
        #         js_graph.add_edge(i, j)
        conn = list(nx.connected_components(js_graph))
        # print('connected components are: ', conn)
        # * add the clustering info  to the rag
        for i in rag.nodes():
            rag.nodes[i]["cluster_energy_nopbc"] = str(
                np.where([i in g for g in conn])[0][0]
            )
    return rag


def add_vdw_hist_to_rag(
    rag,
    regions,
    energy_interpolator=None,
    upper_bound=1000,
    use_pbc=False,
    pbc_groups=None,
    bins_energy=None,
    density=False,
    **kwargs_egrid,
):
    import numpy as np
    import networkx as nx

    # if bins_energy is None:
    # bins_energy = np.linspace(-3000, 100, 25)
    if energy_interpolator is None:
        egrid = egrid_from_atoms2(**kwargs_egrid)
        energy_interpolator = get_energy_interpolator(egrid)
    if use_pbc:
        if pbc_groups is None:
            raise ValueError("pbc_groups must be provided if use_pbc=True")

        else:
            energies_list = [
                get_energies_for_group2(
                    list(g), regions, energy_interpolator=energy_interpolator
                )
                for g in pbc_groups
            ]
            if bins_energy is None:
                bins_energy = np.linspace(
                    np.min(np.hstack(energies_list)), upper_bound, 25
                )
            hist, bins_energy = np.histogram(
                np.hstack(energies_list), bins=bins_energy, density=density
            )

            histograms_bins = [
                np.histogram(e, bins=bins_energy, density=density)
                for e in energies_list
            ]
            ehistograms = [hb[0] for hb in histograms_bins]
            ebincenters = [(hb[1][:-1] + hb[1][1:]) / 2 for hb in histograms_bins]

            for i, node in enumerate(rag.nodes()):
                rag.nodes[node]["vdw_hist_y_pbc"] = ehistograms[
                    rag.nodes[node]["pbc_group"]
                ]
                rag.nodes[node]["vdw_hist_x_pbc"] = ebincenters[
                    rag.nodes[node]["pbc_group"]
                ]

    else:
        energies_list = [
            get_energies_for_group2(
                [node], regions, energy_interpolator=energy_interpolator
            )
            for node in rag.nodes()
        ]

        # energies_list = [get_energies_for_group(
        #     [node], props, ase_atoms, forcefield_mixed, shape, cutoff=cutoff, sample_fraction=sample_fraction) for node in rag.nodes()]
        if bins_energy is None:
            bins_energy = np.linspace(np.min(np.hstack(energies_list)), upper_bound, 25)
        histograms_bins = [
            np.histogram(e, bins=bins_energy, density=density) for e in energies_list
        ]
        ehistograms = [hb[0] for hb in histograms_bins]
        ebincenters = [(hb[1][:-1] + hb[1][1:]) / 2 for hb in histograms_bins]

        for i, node in enumerate(rag.nodes()):
            rag.nodes[node]["vdw_hist_y"] = ehistograms[i]
            rag.nodes[node]["vdw_hist_x"] = ebincenters[i]
    # nx.set_node_attributes(rag, dict(zip(rag.nodes,ehists_data)), 'vdw_hist_y_pbc')
    # nx.set_node_attributes(rag, dict(zip(rag.nodes,ebins_data)), 'vdw_hist_x_pbc')
    # if add_energies:
    #     for i, node in rag.nodes():
    #         rag.nodes[node]['energies'] = energies_list[i]

    # energies_nodes = [energies_list[c] for c in cindices]
    # nx.set_node_attributes(rag, dict(zip(rag.nodes, energies_nodes)), 'vdw_energies_pbc')

    return rag


def optimize_molecule_at_node(
    node,
    rag,
    ref_coords,
    energy_interpolator=None,
    n_trials=100,
    n_jobs=-1,
    random_seed=42,
    **kwargs_egrid,
):
    import optuna
    import numpy as np
    from .helper import get_fractional_coordinates
    from scipy.spatial.transform import Rotation as R

    if energy_interpolator is None:
        egrid = egrid_from_atoms2(**kwargs_egrid)
        energy_interpolator = get_energy_interpolator(egrid)

    location_in_mof = rag.nodes[node]["maxima"]
    max_dist = rag.nodes[node]["maxima_radii"]
    mat_atoms = rag.graph["ase_atoms"]

    def get_energy(new_mol_coords):
        new_frac_coords = get_fractional_coordinates(new_mol_coords, mat_atoms)
        # * wrap the coordinates
        new_frac_coords = new_frac_coords - np.floor(new_frac_coords)
        # * get the energy
        energy = energy_interpolator(new_frac_coords)
        energy_mean = np.mean(energy)
        return energy_mean

    def objective(trial):
        dx = trial.suggest_float("dx", -1, 1)
        dy = trial.suggest_float("dy", -1, 1)
        dz = trial.suggest_float("dz", -1, 1)
        alpha = trial.suggest_float("alpha", 0, 90)
        beta = trial.suggest_float("beta", 0, 90)
        gamma = trial.suggest_float("gamma", 0, 90)
        r = R.from_euler("xyz", [alpha, beta, gamma], degrees=True)
        new_cart_coords = (
            r.apply(ref_coords)
            + location_in_mof
            - np.mean(ref_coords, axis=0)
            + np.array([dx, dy, dz]) * max_dist
        )
        trial.set_user_attr("cart_coords", new_cart_coords)
        return get_energy(new_cart_coords)

    study = optuna.create_study(
        study_name="energy_reduce_{}".format(node),
        pruner=optuna.pruners.HyperbandPruner(),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed)
    )  # , pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), sampler=optuna.samplers.TPESampler(seed=123))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_trial


def add_best_optuna_trial_to_rag(
    rag,
    ref_coords,
    energy_interpolator,
    molecule_name,
    cluster_key="cluster_energy_pbc",
    one_per_cluster=True,
    n_trials=50,
    n_jobs=-1,
    random_seed = 42
):
    import dask.bag as db
    import networkx as nx
    import numpy as np

    # * do optimization at only one node per node type
    if one_per_cluster:
        cluster_dict = nx.get_node_attributes(rag, cluster_key)
        keys_list = np.array(list(cluster_dict.keys()))
        _, unique_indices, reconstruct_indices = np.unique(
            list(cluster_dict.values()), return_index=True, return_inverse=True
        )
        nodes_to_test_at = keys_list[unique_indices]
        best_trials = (
            db.from_sequence([node for node in nodes_to_test_at])
            .map(
                optimize_molecule_at_node,
                ref_coords=ref_coords,
                rag=rag,
                energy_interpolator=energy_interpolator,
                n_trials=n_trials,
                n_jobs=n_jobs,
                random_seed = random_seed,
            )
            .compute()
        )
        for i, node in enumerate(nodes_to_test_at):
            rag.nodes[node]["best_trial_{}".format(molecule_name)] = best_trials[i]

    # * do this for all the pockets regardless of teh clustering
    else:
        best_trials = (
            db.from_sequence(
                [node for node in rag.nodes if rag.nodes[node]["maxima_radii"] > 1.5]
            )
            .map(
                optimize_molecule_at_node,
                ref_coords=ref_coords,
                rag=rag,
                energy_interpolator=energy_interpolator,
                n_trials=n_trials,
                n_jobs=n_jobs,
                random_seed=random_seed,
            )
            .compute()
        )
        # * add the best trial property to rag
        for node, trial in zip(rag.nodes, best_trials):
            rag.nodes[node]["best_trial_{}".format(molecule_name)] = trial
    return rag
def pad_periodic(arr):
    import numpy as np
    """Pad array with periodic boundaries."""
    return np.pad(arr, 1, mode='wrap')

def check_percolation_periodic(distance_grid, threshold):
    import numpy as np
    from scipy.ndimage import label, generate_binary_structure
    # Create a binary grid where True represents void space
    binary_grid = distance_grid > threshold
    
    # Pad the binary grid with periodic boundaries
    padded_grid = pad_periodic(binary_grid)
    
    # Create a structure for 3D connectivity
    structure = generate_binary_structure(3, 1)
    
    # Label connected components
    labeled, _ = label(padded_grid, structure=structure)
    
    # Remove padding
    labeled = labeled[1:-1, 1:-1, 1:-1]
    
    # Check percolation in all three directions
    for axis in range(3):
        # Get unique labels on opposite faces
        face1 = set(np.unique(labeled.take(0, axis=axis)))
        face2 = set(np.unique(labeled.take(-1, axis=axis)))
        
        # Check if any label (except 0) appears on both faces
        if len(face1.intersection(face2) - {0}) > 0:
            return True
    
    return False

def find_pore_limiting_radius(distance_grid):
    import numpy as np
    # Sort unique values in the distance grid
    unique_radii = np.sort(np.unique(distance_grid))
    unique_radii = unique_radii[unique_radii > 0]  # Consider only positive radii

    # Binary search for the pore limiting diameter
    left, right = 0, len(unique_radii) - 1
    while left <= right:
        mid = (left + right) // 2
        threshold = unique_radii[mid]
        
        if check_percolation_periodic(distance_grid, threshold):
            left = mid + 1  # Percolation exists, try a larger threshold
        else:
            right = mid - 1  # No percolation, try a smaller threshold
    
    # The pore limiting diameter is the last threshold that allowed percolation
    return unique_radii[right] if right >= 0 else None
def compute_pld(path_to_cif, dgrid=None,kwargs={}):
    """
    Compute the pore limiting diameter (PLD) from a CIF file.

    Parameters
    ----------
    path_to_cif : Path
        The path to the CIF file.
    kwargs : dict, optional
        Additional keyword arguments. Default is an empty dictionary.
        - spacing : float, optional
            The spacing for the distance grid. Default is 0.5.
        - block_size : int, optional
            The block size for processing. Default is 50000.

    Returns
    -------
    float or None
        The pore limiting diameter if found, otherwise None.
    """
    from ase.io import read
    spacing = kwargs.get('spacing', 0.5)
    block_size = kwargs.get('block_size', 50000)
    mat_atoms = read(path_to_cif)
    if dgrid is None:
        distance_grid = dgrid_from_atoms_cpu(mat_atoms, spacing=spacing, block_size=block_size)
    else:
        distance_grid = dgrid
    return find_pore_limiting_radius(distance_grid)*2
def get_pore_dia_mgr(mat_atoms, dgrid= None, kwargs={}):
    """
    Compute the pore diameters (LCD, PLD, DFSD) from a CIF file.

    Parameters:
    -----------
    path_to_cif : Path or str
        The path to the CIF file.
    kwargs : dict, optional
        Additional keyword arguments.
        - spacing : float, optional
            The spacing for the distance grid. Default is 0.5.
        - block_size : int, optional
            The block size for processing. Default is 50000.

    Returns:
    --------
    tuple
        (lcd, pld, dfsd, distance_grid)
    """
    from ase.io import read
    import numpy as np
    from scipy.ndimage import label
    spacing = kwargs.get('spacing', 0.5)
    block_size = kwargs.get('block_size', 50000)
    
    # mat_atoms = read(path_to_cif)
    if dgrid is None:
        distance_grid = dgrid_from_atoms_cpu(mat_atoms, spacing=spacing, block_size=block_size)
    else:
        distance_grid = dgrid
    
    lcd = distance_grid.max() * 2
    pld = find_pore_limiting_radius(distance_grid) * 2
    
    # Compute DFSD
    pld_threshold = pld / 2
    binary_grid = distance_grid > pld_threshold
    labeled, _ = label(pad_periodic(binary_grid))
    labeled = labeled[1:-1, 1:-1, 1:-1]
    
    percolating_labels = set()
    for axis in range(3):
        face1 = set(np.unique(labeled.take(0, axis=axis)))
        face2 = set(np.unique(labeled.take(-1, axis=axis)))
        percolating_labels.update(face1.intersection(face2) - {0})
    
    percolating_mask = np.isin(labeled, list(percolating_labels))
    dfsd = distance_grid[percolating_mask].max() * 2
    
    return lcd, pld, dfsd, distance_grid

def compute_percolation_history(distance_grid=None, regions=None, maxima=None, path_to_cif=None, kwargs={}):
    """
    Compute the percolation history with region and maxima information.

    Parameters:
    -----------
    distance_grid : numpy.ndarray, optional
        The pre-computed distance grid.
    regions : numpy.ndarray, optional
        The pre-computed segmented regions.
    maxima : list, optional
        The pre-computed list of local maxima indices.
    path_to_cif : Path or str, optional
        The path to the CIF file (used if distance_grid is None).
    kwargs : dict, optional
        Additional keyword arguments.
        - spacing : float, optional
            The spacing for the distance grid. Default is 0.5.
        - block_size : int, optional
            The block size for processing. Default is 50000.
        - mask_thickness : float, optional
            The mask thickness for region segmentation. Default is 1.5.
        - step : float, optional
            The step size for probe radius increments. Default is 0.1 Angstrom.

    Returns:
    --------
    tuple
        (percolation_history, pld_locations, distance_grid, regions, maxima)
    """
    from scipy.ndimage import label as scipy_label
    from ase.io import read
    import numpy as np
    # import mofography as mgr
    
    spacing = kwargs.get('spacing', 0.5)
    block_size = kwargs.get('block_size', 50000)
    mask_thickness = kwargs.get('mask_thickness', 1.5)
    step = kwargs.get('step', 0.05)
    
    if distance_grid is None:
        if path_to_cif is None:
            raise ValueError("Either distance_grid or path_to_cif must be provided.")
        _, _, _, distance_grid = get_pore_dia_mgr(path_to_cif, kwargs)
    
    if regions is None or maxima is None:
        regions, maxima = regions_from_dgrid(distance_grid, mask_thickness=mask_thickness)
    
    # Compute PLD
    pld = find_pore_limiting_radius(distance_grid) * 2
    
    # Find PLD locations
    pld_locations = np.argwhere(np.isclose(distance_grid, pld/2, atol=1e-10))
    
    percolation_history = []
    
    for threshold in np.arange(0, pld/2 + step, step):
        binary_grid = distance_grid > threshold
        labeled, _ = scipy_label(pad_periodic(binary_grid))
        labeled = labeled[1:-1, 1:-1, 1:-1]
        
        percolating_components = {0: [], 1: [], 2: []}
        
        for axis in range(3):
            face1 = set(np.unique(labeled.take(0, axis=axis)))
            face2 = set(np.unique(labeled.take(-1, axis=axis)))
            percolating_labels = face1.intersection(face2) - {0}
            
            for label in percolating_labels:
                percolating_mask = labeled == label
                percolating_regions = np.unique(regions[percolating_mask])
                percolating_regions = percolating_regions[percolating_regions != 0]  # Remove background
                percolating_components[axis].append(percolating_regions.tolist())
        
        if any(percolating_components.values()):
            percolation_history.append((threshold, percolating_components))
    
    return percolation_history, pld_locations, distance_grid, regions, maxima

def setup_poisson_spectral(atoms, grid_shape):
    """
    Set up the system for solving the Poisson equation using a spectral method.
    
    :param atoms: ASE Atoms object
    :param grid_shape: Tuple of grid points in each direction
    :return: Charge density on grid, grid shape, and cell matrix
    """
    import numpy as np
    from ase.geometry import complete_cell
    from scipy.constants import epsilon_0, e
    cell = complete_cell(atoms.get_cell()).T  # Transpose to get column vectors
    volume = np.abs(np.linalg.det(cell))
    
    rho = np.zeros(grid_shape)
    positions = atoms.get_scaled_positions()
    charges = np.array(atoms.info['_atom_site_charge'])
    
    # Distribute charges to grid points using trilinear interpolation
    for pos, q in zip(positions, charges):
        index = pos * grid_shape
        index_floor = np.floor(index).astype(int) % grid_shape
        weights = index - index_floor
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    weight = (1-weights[0])**abs(i-1) * weights[0]**i * \
                             (1-weights[1])**abs(j-1) * weights[1]**j * \
                             (1-weights[2])**abs(k-1) * weights[2]**k
                    rho[(index_floor[0]+i)%grid_shape[0], 
                        (index_floor[1]+j)%grid_shape[1], 
                        (index_floor[2]+k)%grid_shape[2]] += q * weight
    
    rho /= volume
    return rho, grid_shape, cell

def solve_poisson_spectral(atoms, grid_spacing=0.2, epsilon_r=1.0):
    """
    Solve the Poisson equation using a spectral method for non-orthogonal, periodic systems.
    
    :param atoms: ASE Atoms object
    :param grid_spacing: Desired grid spacing in Angstroms
    :param epsilon_r: Relative permittivity of the system
    :return: Electrostatic potential (V), grid shape, and cell matrix
    """
    import numpy as np
    from scipy.fft import fftn, ifftn
    from ase.geometry import complete_cell
    from scipy.constants import epsilon_0, e

    cell = complete_cell(atoms.get_cell()).T
    cell_lengths = np.linalg.norm(cell, axis=0)
    grid_shape = np.ceil(cell_lengths / grid_spacing).astype(int)
    
    rho, grid_shape, cell = setup_poisson_spectral(atoms, grid_shape)
    
    # Compute metric tensor G
    G = np.linalg.inv(cell.T @ cell)
    
    # Generate k-vectors
    kx = 2 * np.pi * np.fft.fftfreq(grid_shape[0])
    ky = 2 * np.pi * np.fft.fftfreq(grid_shape[1])
    kz = 2 * np.pi * np.fft.fftfreq(grid_shape[2])
    
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Compute k^T * G * k
    k_dot_G_k = (Kx*G[0,0] + Ky*G[1,0] + Kz*G[2,0])*Kx + \
                (Kx*G[0,1] + Ky*G[1,1] + Kz*G[2,1])*Ky + \
                (Kx*G[0,2] + Ky*G[1,2] + Kz*G[2,2])*Kz
    
    # Avoid division by zero
    k_dot_G_k[0,0,0] = 1.0
    
    # Solve Poisson equation in Fourier space
    rho_k = fftn(rho)
    phi_k = rho_k / (epsilon_r * epsilon_0 * k_dot_G_k)
    phi_k[0,0,0] = 0  # Set average potential to zero
    
    # Transform back to real space
    phi = np.real(ifftn(phi_k))
    
    # Convert to Volts
    phi *= e / (4 * np.pi * epsilon_0 * epsilon_r)
    
    # Print important details about the solved grid
    print(f"Grid shape: {grid_shape}")
    cell_m = cell * 1e-10  # Convert cell dimensions to meters
    volume_m3 = np.abs(np.linalg.det(cell_m))
    print(f"Cell volume: {volume_m3:.2e} m^3")
    print(f"Potential stats (V):")
    print(f"  Min: {np.min(phi):.2e}")
    print(f"  Max: {np.max(phi):.2e}")
    print(f"  Mean: {np.mean(phi):.2e}")
    
    return phi, grid_shape, cell
def setup_poisson_spectral_gpu(atoms, grid_shape):
    """
    Set up the system for solving the Poisson equation using a spectral method.
    
    :param atoms: ASE Atoms object
    :param grid_shape: Tuple of grid points in each direction
    :return: Charge density on grid, grid shape, and cell matrix
    """
    from ase.geometry import complete_cell
    import cupy as cp
    cell = cp.asarray(complete_cell(atoms.get_cell())).T  # Transpose to get column vectors
    volume = cp.abs(cp.linalg.det(cell))
    print(grid_shape)
    grid_shape = tuple(grid_shape.tolist())
    rho = cp.zeros(grid_shape)
    positions = cp.asarray(atoms.get_scaled_positions())
    charges = cp.array(atoms.info['_atom_site_charge'])
    
    # Distribute charges to grid points using trilinear interpolation
    for pos, q in zip(positions, charges):
        index = pos * cp.asarray(grid_shape)
        index_floor = cp.floor(index).astype(int) % cp.asarray(grid_shape)
        weights = index - index_floor
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    weight = (1 - weights[0])**abs(i - 1) * weights[0]**i * \
                             (1 - weights[1])**abs(j - 1) * weights[1]**j * \
                             (1 - weights[2])**abs(k - 1) * weights[2]**k
                    rho[(index_floor[0] + i) % grid_shape[0], 
                        (index_floor[1] + j) % grid_shape[1], 
                        (index_floor[2] + k) % grid_shape[2]] += q * weight
    
    rho /= volume
    return rho, grid_shape, cell
def solve_poisson_spectral_gpu(atoms, grid_spacing=0.2, epsilon_r=1.0):
    """
    Solve the Poisson equation using a spectral method for non-orthogonal, periodic systems.
    
    :param atoms: ASE Atoms object
    :param grid_spacing: Desired grid spacing in Angstroms
    :param epsilon_r: Relative permittivity of the system
    :return: Electrostatic potential (V), grid shape, and cell matrix
    """
    import cupy as cp
    from cupy.fft import fftn, ifftn
    from ase.geometry import complete_cell
    from scipy.constants import epsilon_0, e

    cell = cp.asarray(complete_cell(atoms.get_cell())).T
    cell_lengths = cp.linalg.norm(cell, axis=0)
    grid_shape = cp.ceil(cell_lengths / grid_spacing).astype(int)
    
    # Convert grid_shape to a tuple of integers
    # grid_shape_tuple = tuple(grid_shape.tolist())
    
    rho, grid_shape, cell = setup_poisson_spectral_gpu(atoms, grid_shape)
    print(grid_shape)
    # grid_shape = tuple(grid_shape.tolist())
    
    # Compute metric tensor G
    G = cp.linalg.inv(cell.T @ cell)
    
    # Generate k-vectors
    kx = 2 * cp.pi * cp.fft.fftfreq(grid_shape[0])
    ky = 2 * cp.pi * cp.fft.fftfreq(grid_shape[1])
    kz = 2 * cp.pi * cp.fft.fftfreq(grid_shape[2])
    
    Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
    
    # Compute k^T * G * k
    k_dot_G_k = (Kx * G[0, 0] + Ky * G[1, 0] + Kz * G[2, 0]) * Kx + \
                 (Kx * G[0, 1] + Ky * G[1, 1] + Kz * G[2, 1]) * Ky + \
                 (Kx * G[0, 2] + Ky * G[1, 2] + Kz * G[2, 2]) * Kz
    
    # Avoid division by zero
    k_dot_G_k[0, 0, 0] = 1.0
    
    # Solve Poisson equation in Fourier space
    rho_k = fftn(rho)
    phi_k = rho_k / (epsilon_r * epsilon_0 * k_dot_G_k)
    phi_k[0, 0, 0] = 0  # Set average potential to zero
    
    # Transform back to real space
    phi = cp.real(ifftn(phi_k))
    
    # Convert to Volts
    phi *= e / (4 * cp.pi * epsilon_0 * epsilon_r)
    
    # Print important details about the solved grid
    print(f"Grid shape: {grid_shape}")
    cell_m = cell * 1e-10  # Convert cell dimensions to meters
    volume_m3 = cp.abs(cp.linalg.det(cell_m))
    print(f"Cell volume: {volume_m3:.2e} m^3")
    print(f"Potential stats (V):")
    print(f"  Min: {cp.min(phi):.2e}")
    print(f"  Max: {cp.max(phi):.2e}")
    print(f"  Mean: {cp.mean(phi):.2e}")
    
    return phi, grid_shape, cell
def compute_accessible_volume(dgrid, probe_radius=None, mat_atoms=None):
    """
    Compute the accessible volume based on the distance grid and probe radius.

    Parameters
    ----------
    dgrid : numpy.ndarray
        The distance grid computed from the atomic structure.
    probe_radius : float, optional
        The radius of the probe to use for volume calculation. 
        If None, defaults to the radius of helium (atomic number 2).
    mat_atoms : ase.Atoms
        The atomic structure for which the volume is being calculated.

    Returns
    -------
    dict
        A dictionary containing the void fraction, void volume in cc_cc, 
        and void volume in cc_g.
    """
    import numpy as np
    from ase.data import vdw_radii
    from ase.units import _amu

    if probe_radius is None:
        probe_radius = vdw_radii[2]  # helium atomic number is 2

    void_fraction = np.sum(dgrid > 0) / dgrid.size
    voxel_volume = mat_atoms.get_volume() / dgrid.size  # this is in A^3
    void_volume = np.sum(dgrid > probe_radius) * voxel_volume  # this is in A^3
    void_volume_cc_cc = void_volume / mat_atoms.get_volume()  # this is in A^3/A^3
    mass_g = mat_atoms.get_masses().sum() * _amu * 1000  # this is in g
    void_volume_cc_g = void_volume * 1e-24 / mass_g  # this is in cm^3/g
    
    return {
        'void_fraction': void_fraction,
        'void_volume_cc_cc': void_volume_cc_cc,
        'void_volume_cc_g': void_volume_cc_g
    }

def compute_surface_area(dgrid, probe_radius=None, mat_atoms=None):
    """
    Compute the surface area based on the distance grid and probe radius.

    Parameters
    ----------
    dgrid : numpy.ndarray
        The distance grid computed from the atomic structure.
    probe_radius : float, optional
        The radius of the probe to use for surface area calculation. 
        If None, defaults to the radius of helium (atomic number 2).
    mat_atoms : ase.Atoms
        The atomic structure for which the surface area is being calculated.

    Returns
    -------
    float
        The surface area in m^2/g.
    """
    import numpy as np
    from ase.data import vdw_radii
    from ase.units import _amu

    if probe_radius is None:
        probe_radius = vdw_radii[2]  # helium atomic number is 2
    from .visualize import get_isosurface_mesh
    # import mofography as mgr
    isosurface_mesh = get_isosurface_mesh(dgrid, probe_radius, mat_atoms)
    mass_g = mat_atoms.get_masses().sum() * _amu * 1000  # this is in grams
    sa_m2g = isosurface_mesh.area_faces.sum() * 1e-20 / mass_g
    
    return sa_m2g
def gpd_aabb_hybrid(grid_points, ase_atoms, radii, n_neighbors=10):
    """
    Hybrid approach combining an AABB neighbor search with vectorized distance calculations.
    
    Assumptions:
      - Input grid_points are Cartesian coordinates (as returned by dask_grid_over_atoms).
      - The framework atoms are converted to fractional coordinates and centered into the range [-0.5, 0.5].
      - The grid_points are converted to centered fractional coordinates using the inverse cell.
      - The shortest distance is then computed after applying the minimum image convention.
    
    Parameters:
      grid_points : (n_grid_points, 3) numpy array in Cartesian coordinates.
      ase_atoms   : ASE atoms object.
      radii       : numpy array of atomic van der Waals radii.
      n_neighbors : Number of neighbors to consider for each grid point (default 10).
      
    
    Returns:
      min_distances : numpy array of shape (n_grid_points,) with computed distances.
    """
    import numpy as np
    from freud.locality import AABBQuery
    from freud.box import Box
    from ase.geometry import complete_cell
    from .helper import get_fractional_coordinates

    # Ensure grid_points is a 2D array.
    grid_points = np.atleast_2d(grid_points)
    
    # Get the complete (upper-triangular) cell and its inverse.
    cell = complete_cell(ase_atoms.get_cell()).T
    # inv_cell = np.linalg.inv(cell)
    
    # Convert framework atomic positions:
    #    atom_cart: Cartesian positions (for neighbor search)
    #    atom_frac: Fractional positions centered into the range [-0.5, 0.5]
    atom_cart = ase_atoms.get_positions()
    atom_frac = ase_atoms.get_scaled_positions() - 0.5

    # Since dask_grid_over_atoms returns Cartesian grid points,
    # we convert them back to fractional coordinates and then center them.
    # grid_frac = np.dot(grid_points, inv_cell) - 0.5  # now in [-0.5, 0.5]
    grid_frac = get_fractional_coordinates(grid_points, ase_atoms) - 0.5
    


    # --- AABB neighbor search in Cartesian coordinates ---
    box = Box.from_matrix(cell, dimensions=3)
    ab = AABBQuery(box=box, points=atom_cart)
    ab_r = ab.query(grid_points, query_args=dict(mode="nearest", num_neighbors=n_neighbors, exclude_ii=True))
    nl = ab_r.toNeighborList(sort_by_distance=True)
    
    # Reshape the neighbor indices to (n_grid_points, n_neighbors)
    neighbor_indices = nl.point_indices.reshape(-1, n_neighbors)
    
    # Check whether the query returned valid neighbors.
    # (This check prevents a broadcast issue if a block gets no neighbors.)
    if neighbor_indices.size == 0 or neighbor_indices.shape[0] != grid_points.shape[0]:
        return np.full((grid_points.shape[0],), np.inf)
    
    # Get the neighboring atoms' fractional positions and corresponding radii.
    relevant_atoms = atom_frac[neighbor_indices]      # shape: (n_grid_points, n_neighbors, 3)
    # print('SHape of relevant atoms is ', relevant_atoms.shape)
    relevant_radii = radii[neighbor_indices]            # shape: (n_grid_points, n_neighbors)
    
    # --- Compute distances in fractional space ---
    # Expand grid_frac so the subtraction is done per grid point.
    grid_frac_expanded = np.expand_dims(grid_frac, axis=1)  # shape: (n_grid_points, 1, 3)
    diff = grid_frac_expanded - relevant_atoms             # shape: (n_grid_points, n_neighbors, 3)
    diff = (diff - np.round(diff)).reshape(-1,3) # apply the minimum image convention
    
    # Convert the fractional differences back to Cartesian differences.
    diff= np.dot(cell, diff.T).T
    distances = np.linalg.norm(diff, axis=1).reshape(-1, n_neighbors) - relevant_radii # (n_grid_points, n_neighbors)
    

    # For each grid point, select the smallest distance (then subtract the probe radius).
    min_distances = np.min(distances, axis=1)
    return min_distances


def dgrid_from_atoms_cpu_hybrid(
    ase_atoms,
    radii=None,
    spacing=0.25,
    block_size=50000,
    n_neighbors=10,
    
    precompute=True,
):
    """
    Modified dgrid computation using a hybrid AABB query and explicit distance calculation.
    
    This function generates a Cartesian grid over the structure using dask_grid_over_atoms 
    and then computes the distance grid by applying the hybrid function `gpd_aabb_hybrid`.
    
    Parameters:
      ase_atoms   : ASE atoms object.
      radii       : Optional numpy array of atomic radii (if None, default van der Waals radii are used).
      spacing     : Grid spacing in angstroms.
      block_size  : Number of grid points per dask array chunk.
      n_neighbors : Number of nearest neighbors to consider.
      
      precompute  : If True, compute the array immediately.
    
    Returns:
      dgrid : 3D numpy array of computed distances with shape determined by the grid dimensions.
    """
    import numpy as np
    import dask.array as da
    import pandas as pd
    # import mofography as mgr  # assumes mgr.dask_grid_over_atoms is available

    if radii is None:
        vdw_radii = pd.Series(
            {
                "H": 110.00000000000001,
                "He": 140.0,
                "Li": 182.0,
                "Be": 153.0,
                "B": 192.0,
                "C": 170.0,
                "N": 155.0,
                "O": 152.0,
                "F": 147.0,
                "Ne": 154.0,
                "Na": 227.0,
                "Mg": 173.0,
                "Al": 184.0,
                "Si": 210.0,
                "P": 180.0,
                "S": 180.0,
                "Cl": 175.0,
                "Ar": 188.0,
                "K": 275.0,
                "Ca": 231.0,
                "Sc": 215.0,
                "Ti": 211.0,
                "V": 206.99999999999997,
                "Cr": 206.0,
                "Mn": 204.99999999999997,
                "Fe": 204.0,
                "Co": 200.0,
                "Ni": 197.0,
                "Cu": 196.0,
                "Zn": 200.99999999999997,
                "Ga": 187.0,
                "Ge": 211.0,
                "As": 185.0,
                "Se": 190.0,
                "Br": 185.0,
                "Kr": 202.0,
                "Rb": 303.0,
                "Sr": 249.00000000000003,
                "Y": 231.99999999999997,
                "Zr": 223.0,
                "Nb": 218.00000000000003,
                "Mo": 217.0,
                "Tc": 216.0,
                "Ru": 213.0,
                "Rh": 210.0,
                "Pd": 210.0,
                "Ag": 211.0,
                "Cd": 218.00000000000003,
                "In": 193.0,
                "Sn": 217.0,
                "Sb": 206.0,
                "Te": 206.0,
                "I": 198.0,
                "Xe": 216.0,
                "Cs": 343.0,
                "Ba": 268.0,
                "La": 243.00000000000003,
                "Ce": 242.0,
                "Pr": 240.0,
                "Nd": 239.0,
                "Pm": 238.0,
                "Sm": 236.0,
                "Eu": 235.0,
                "Gd": 234.0,
                "Tb": 233.0,
                "Dy": 231.0,
                "Ho": 229.99999999999997,
                "Er": 229.0,
                "Tm": 227.0,
                "Yb": 225.99999999999997,
                "Lu": 224.00000000000003,
                "Hf": 223.0,
                "Ta": 222.00000000000003,
                "W": 218.00000000000003,
                "Re": 216.0,
                "Os": 216.0,
                "Ir": 213.0,
                "Pt": 213.0,
                "Au": 214.0,
                "Hg": 223.0,
                "Tl": 196.0,
                "Pb": 202.0,
                "Bi": 206.99999999999997,
                "Po": 197.0,
                "At": 202.0,
                "Rn": 220.00000000000003,
                "Fr": 348.0,
                "Ra": 283.0,
                "Ac": 247.00000000000003,
                "Th": 245.00000000000003,
                "Pa": 243.00000000000003,
                "U": 241.0,
                "Np": 239.0,
                "Pu": 243.00000000000003,
                "Am": 244.0,
                "Cm": 245.00000000000003,
                "Bk": 244.0,
                "Cf": 245.00000000000003,
                "Es": 245.00000000000003,
                "Fm": 245.00000000000003,
                "Md": 246.0,
                "No": 246.0,
                "Lr": 246.0,
                "Rf": np.nan,
                "Db": np.nan,
                "Sg": np.nan,
                "Bh": np.nan,
                "Hs": np.nan,
                "Mt": np.nan,
                "Ds": np.nan,
                "Rg": np.nan,
                "Cn": np.nan,
                "Nh": np.nan,
                "Fl": np.nan,
                "Mc": np.nan,
                "Lv": np.nan,
                "Ts": np.nan,
                "Og": np.nan,
            }
        )
        radii = vdw_radii[ase_atoms.get_chemical_symbols()].values / 100.0

    # Get the grid points -- note that dask_grid_over_atoms returns Cartesian coordinates and the grid shape.
    gpoints, shape = dask_grid_over_atoms(ase_atoms, spacing=spacing, chunksize=block_size)
    # Compute the distance grid by applying the hybrid function blockwise.


    dgrid = gpoints.map_blocks(
        gpd_aabb_hybrid,
        ase_atoms,
        radii,
        n_neighbors=n_neighbors,
        
        dtype=float,
        drop_axis=1
    ).reshape(shape)
    
    if precompute:
        return dgrid.compute()
    return dgrid

# new mods for periodic segmentation.
def calculate_padding_thickness(
    dgrid: np.ndarray, ase_atoms: ase.Atoms, spacing: float
) -> int:
    """Calculate required padding thickness for periodic calculations.

    Args:
        ase_atoms: ASE Atoms object containing cell information
        spacing: Grid spacing in Angstroms

    Returns:
        Minimum number of padding cells needed to handle periodicity correctly
    """
    import numpy as np
    # Get the complete cell vectors
    # volume of each voxel
    voxel_volume = ase_atoms.get_volume() / np.prod(dgrid.shape)
    # box lengths and angles
    la, lb, lc, alpha, beta, gamma = ase_atoms.cell.cellpar()
    # convert to radians
    alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
    # these are not unit vectors, they are voxel vectors
    # unlike the cell, the voxel has same lengths in all directions, just the angles are different
    eA = [spacing, 0, 0]
    eB = [spacing * np.cos(gamma), spacing * np.sin(gamma), 0]
    eC = [
        spacing * np.cos(beta),
        spacing * (np.cos(alpha) - np.cos(beta) *
                   np.cos(gamma)) / np.sin(gamma),
        voxel_volume / (spacing * spacing * np.sin(gamma)),
    ]
    # min voxel perpendicual length
    lx_voxel, ly_voxel, lz_voxel = (
        voxel_volume / np.linalg.norm(np.cross(eB, eC)),
        voxel_volume / np.linalg.norm(np.cross(eC, eA)),
        voxel_volume / np.linalg.norm(np.cross(eA, eB)),
    )
    min_perp_length_voxel = np.min([lx_voxel, ly_voxel, lz_voxel])
    # Calculate required padding thickness
    # Add 1 and use ceil to ensure we have enough padding
    # just added some tolerance just to be safe
    padding_thickness = int(np.ceil(dgrid.max() / min_perp_length_voxel)) + np.ceil(5/spacing).astype(int)

    return padding_thickness

def get_pbc_groups(unmerged_regions, maxima_info):
    """
        Group unmerged regions based on periodic boundary conditions (PBC).

        This function organizes unmerged regions into groups based on their maxima
        indices. Each group is identified by a unique tuple of maxima indices 
        (columns 2-4 of `maxima_info`), and contains the corresponding region indices 
        (column 1 of `maxima_info`).

        Args:
            unmerged_regions (iterable): A collection of region indices that are not yet merged.
            maxima_info (array-like): A 2D array where each row contains:
                - Column 0: Region index
                - Columns 1-3: Maxima indices (used as grouping keys)
                - Column 4: Offsets in each direction
                  You get this as output when you call the 'do_periodic_segmentation' on the distance grid (dgrid) 

        Returns:
            dict: A dictionary where keys are tuples of maxima indices (columns 2-4),
                and values are sets of region indices (column 1) belonging to each group.
        """
    import numpy as n
    from collections import defaultdict

    # Assuming your array is called 'arr'
    # Create a dictionary where keys are tuples of maxima indices (cols 2-4)
    # and values are lists of corresponding indices (col 1)
    groups = defaultdict(list)

    for row in maxima_info:
        if row[0] in unmerged_regions:
            # Use tuple of columns 2-4 as key (converting to tuple for hashability)
            key = tuple(row[1:4])
            # Add the index (first column) to the corresponding group
            groups[key].append(int(row[0]))

    # Convert the dictionary values to sets if you want unique indices
    groups = {k: set(v) for k, v in groups.items()}
    return groups

def detect_periodic_maxima_enhanced(
    distance_grid: np.ndarray,
    ase_atoms: ase.Atoms,
    spacing: float,
    maxima_mask: float = 0,
    h: float = None,
    threshold_abs: float = None,
    min_distance: float = 1.0,
    padding_thickness: int = None,
    # smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhanced periodic maxima detection that handles:
    1. Proper periodic influence regions
    2. Detection of missing symmetric maxima
    3. Merging of periodic copies
    4. Special handling of boundary-split maxima

    Parameters match detect_periodic_distance_maxima for consistency
    """
    import numpy as np
    import itertools
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import h_maxima
    from scipy.cluster.hierarchy import linkage, fcluster
    from .helper import estimate_h_from_percolation_history
    from ase.geometry import complete_cell
    # create binary mask from distance grid
    binary_mask = distance_grid > maxima_mask

    # First compute periodic EDT with larger padding
    if padding_thickness is None:
        padding_thickness = calculate_padding_thickness(
            distance_grid, ase_atoms = ase_atoms, spacing =spacing)

    print("padding thickness in maxima detection is ", padding_thickness)

    # padding = int(np.ceil(binary_mask.max()/2/spacing))
    padded_mask = np.pad(binary_mask, padding_thickness, mode="wrap")
    dt_padded = distance_transform_edt(
        padded_mask.astype(np.float32), sampling=spacing)

    # Extract central region and smooth
    dt_central = dt_padded[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]
    # this cannot be fully trusted, so let's not do this just yet.
    # if smooth:
    #     dt_smoothed = mgr.spectral_smooth_3d(
    #         dt_central,
    #         ase_atoms,
    #         grid_spacing=spacing,
    #         sigma=spacing * 3,  # Increased smoothing
    #     )
    # else:
    #     dt_smoothed = dt_central
    #
    # If h is not provided, compute it from percolation history
    if h is None:
        h = estimate_h_from_percolation_history(dt_central)
        print("h guess from percolation history is ", h)
    else:
        print("h is provided and is ", h)

    # Continue with existing maxima detection using computed h
    maxima = h_maxima(dt_central, h)
    maxima_indices = np.array(np.where(maxima)).T
    print("maxima indices", maxima_indices, "for h", h)
    # Apply absolute threshold strictly based on the true grid not edt
    if threshold_abs is not None:
        threshold_mask =distance_grid[tuple(maxima_indices.T)] >= threshold_abs
        maxima_indices = maxima_indices[threshold_mask]
        if len(maxima_indices) == 0:
            return dt_central, maxima_indices

    # Calculate volumes for initial filtering
    voxel_volume = abs(np.linalg.det(ase_atoms.cell)) / \
        np.prod(dt_central.shape)
    box_volume = abs(np.linalg.det(ase_atoms.cell))

    # Check for missing symmetric maxima and boundary-split cases
    cell = complete_cell(ase_atoms.get_cell())
    new_maxima = []
    # Define boundary tolerance (in grid points)
    boundary_tol = max(
        1, int(0.1 * min(dt_central.shape))
    )  # 10% of smallest dimension or at least 1

    for max_idx in maxima_indices:
        # Check if this maximum is near a boundary
        near_boundary = any(
            coord < boundary_tol or coord > (size - boundary_tol)
            for coord, size in zip(max_idx,dt_central.shape)
        )

        max_value =dt_central[tuple(max_idx)]

        # Check periodic images
        for i, j, k in itertools.product([-1, 0, 1], repeat=3):
            if i == j == k == 0:
                continue

            # Get periodic image position
            periodic_pos = max_idx + \
                np.array([i, j, k]) * np.array(dt_central.shape)
            periodic_idx = periodic_pos %dt_central.shape

            # Convert to fractional then cartesian coordinates
            frac_orig = max_idx /dt_central.shape
            frac_periodic = periodic_pos /dt_central.shape

            cart_orig = np.dot(cell, frac_orig)
            cart_periodic = np.dot(cell, frac_periodic)

            periodic_value =dt_central[tuple(periodic_idx)]

            # Check for split maxima (near boundary and nearly identical values)
            if (
                near_boundary and abs(
                    max_value - periodic_value) < 0.1 * max_value
            ):  # 10% tolerance
                continue  # Skip adding as new maximum - will be handled in clustering

            # Check for missing symmetric maxima
            if (
                np.linalg.norm(cart_orig - cart_periodic) < min_distance
                and abs(max_value - periodic_value) < h
                and not maxima[tuple(periodic_idx)]
            ):
                new_maxima.append(periodic_idx)

    # Combine original and new maxima
    all_maxima = (
        np.vstack([maxima_indices, new_maxima]
                  ) if new_maxima else maxima_indices
    )

    print("maxima before merging", all_maxima)
    print("maxima shape before merging", all_maxima.shape)
    # Convert to fractional coordinates for periodic distance comparison
    frac_coords = all_maxima /dt_central.shape
    cart_coords = np.dot(cell, frac_coords.T).T
    values =dt_central[tuple(all_maxima.T)]

    if len(all_maxima) > 1:
        # Custom distance matrix for clustering that considers periodicity
        from scipy.spatial.distance import pdist, squareform

        def periodic_distance(x1, x2):
            dr = x1 - x2
            # Map to [-0.5, 0.5] for each dimension
            dr = dr - np.round(dr)
            # Convert to cartesian
            dr_cart = np.dot(cell, dr)
            return np.linalg.norm(dr_cart)

        distances = pdist(frac_coords, metric=periodic_distance)
        Z = linkage(distances, method="single")
        clusters = fcluster(Z, min_distance / spacing, criterion="distance")

        merged_maxima = []
        for cluster_id in range(1, clusters.max() + 1):
            cluster_mask = clusters == cluster_id
            cluster_values = values[cluster_mask]
            cluster_points = all_maxima[cluster_mask]
            best_idx = np.argmax(cluster_values)
            merged_maxima.append(cluster_points[best_idx])

        merged_maxima = np.array(merged_maxima)
    else:
        merged_maxima = all_maxima

    return dt_central, merged_maxima, all_maxima

def detect_periodic_maxima_direct(

    distance_grid: np.ndarray,
    ase_atoms: ase.Atoms,
    spacing: float,
    maxima_mask: float = 0,
    h: float = None,
    threshold_abs: float = None,
    min_distance: float = 1.0,
    padding_thickness: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Periodic maxima detection using the original distance grid directly.

    Parameters:
    - distance_grid: The input grid to detect maxima.
    - ase_atoms: ASE Atoms object for periodic boundary conditions.
    - spacing: Grid spacing.
    - maxima_mask: Threshold for binary mask creation.
    - h: Height threshold for h-maxima.
    - threshold_abs: Absolute threshold for maxima filtering.
    - min_distance: Minimum distance for merging maxima.
    - padding_thickness: Padding thickness for periodic boundary handling.

    Returns:
    - distance_grid: The original grid.
    - merged_maxima: Final merged maxima indices.
    - all_maxima: All detected maxima indices before merging.
    """
    import numpy as np
    import itertools
    from skimage.morphology import h_maxima
    from scipy.cluster.hierarchy import linkage, fcluster
    from .helper import estimate_h_from_percolation_history, get_perpendicular_voxel_lengths

    from ase.geometry import complete_cell

    # If h is not provided, compute it from percolation history
    if h is None:
        h = estimate_h_from_percolation_history(distance_grid)
        logger.info(f"h guess from percolation history is {h}")

    else:
        logger.info(f"h is provided and is {h}")

    # compute footprint

    # Example voxel lengths
    vx, vy, vz = get_perpendicular_voxel_lengths(ase_atoms, spacing,distance_grid.shape)
    # Desired radius in Å
    radius = 0.5
    # Compute halfwidths
    hw_i = int(np.ceil(radius / vx))
    hw_j = int(np.ceil(radius / vy))
    hw_k = int(np.ceil(radius / vz))

    # # Build index grids
    # i_range = np.arange(-hw_i, hw_i + 1)
    # j_range = np.arange(-hw_j, hw_j + 1)
    # k_range = np.arange(-hw_k, hw_k + 1)
    # I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')
    # # Build index grids
    # distances = np.sqrt((I * vx)**2 + (J * vy)**2 + (K * vz)**2)
    # # Build footprint
    # footprint = distances <= radius
    
    footprint = np.ones((2*hw_i+1, 2*hw_j+1, 2*hw_k+1), dtype=bool)  # Use a simple rectangular footprint

    # Detect maxima using h-maxima
    maxima = h_maxima(distance_grid, h, footprint=footprint)
    # maxima = h_maxima(distance_grid, h)
    maxima_indices = np.array(np.where(maxima)).T
    # print("maxima indices", maxima_indices, "for h", h)

    # Apply absolute threshold
    if threshold_abs is not None:
        threshold_mask = distance_grid[tuple(maxima_indices.T)] >= threshold_abs
        maxima_indices = maxima_indices[threshold_mask]
        if len(maxima_indices) == 0:
            return distance_grid, maxima_indices

    # Check for missing symmetric maxima and boundary-split cases
    cell = complete_cell(ase_atoms.get_cell())
    new_maxima = []
    boundary_tol = max(1, int(0.1 * min(distance_grid.shape)))  # 10% of smallest dimension or at least 1

    for max_idx in maxima_indices:
        near_boundary = any(
            coord < boundary_tol or coord > (size - boundary_tol)
            for coord, size in zip(max_idx, distance_grid.shape)
        )
        max_value = distance_grid[tuple(max_idx)]

        for i, j, k in itertools.product([-1, 0, 1], repeat=3):
            if i == j == k == 0:
                continue

            periodic_pos = max_idx + np.array([i, j, k]) * np.array(distance_grid.shape)
            periodic_idx = periodic_pos % distance_grid.shape

            frac_orig = max_idx / distance_grid.shape
            frac_periodic = periodic_pos / distance_grid.shape

            cart_orig = np.dot(cell, frac_orig)
            cart_periodic = np.dot(cell, frac_periodic)

            periodic_value = distance_grid[tuple(periodic_idx)]

            if (
                near_boundary
                and abs(max_value - periodic_value) < 0.1 * max_value
            ):
                continue

            if (
                np.linalg.norm(cart_orig - cart_periodic) < min_distance
                and abs(max_value - periodic_value) < h
                and not maxima[tuple(periodic_idx)]
            ):
                new_maxima.append(periodic_idx)

    all_maxima = (
        np.vstack([maxima_indices, new_maxima]) if new_maxima else maxima_indices
    )

    frac_coords = all_maxima / distance_grid.shape
    cart_coords = np.dot(cell, frac_coords.T).T
    values = distance_grid[tuple(all_maxima.T)]

    if len(all_maxima) > 1:
        from scipy.spatial.distance import pdist, squareform

        def periodic_distance(x1, x2):
            dr = x1 - x2
            dr = dr - np.round(dr)
            dr_cart = np.dot(cell, dr)
            return np.linalg.norm(dr_cart)

        distances = pdist(frac_coords, metric=periodic_distance)
        Z = linkage(distances, method="single")
        clusters = fcluster(Z, min_distance / spacing, criterion="distance")

        merged_maxima = []
        for cluster_id in range(1, clusters.max() + 1):
            cluster_mask = clusters == cluster_id
            cluster_values = values[cluster_mask]
            cluster_points = all_maxima[cluster_mask]
            best_idx = np.argmax(cluster_values)
            merged_maxima.append(cluster_points[best_idx])

        merged_maxima = np.array(merged_maxima) 
    else:
        merged_maxima = all_maxima
 
    return distance_grid, merged_maxima, all_maxima


def do_periodic_segmentation(dgrid, mat_atoms, mask_thickness=0.5, h=0.11, min_distance=1.1, threshold_abs=1.0, return_padded_regions=False, spacing=None,minimum_window_separation=None, wall_thickness=0.05, extra_padding=5, compactness=0,min_region_size=100, disable_tqdm=False):
    import numpy as np
    from skimage.segmentation import watershed
    import copy
    from ase.geometry import complete_cell
    import sparse
    from skimage.segmentation import find_boundaries
    from concurrent.futures import ThreadPoolExecutor
    from time import time
    from tqdm import tqdm
    start_time = time()
    from .helper import compute_wall_cells, move_maximum_to_peak, process_wall_maxima, is_inside_or_close_to_boundary, get_fractional_coordinates, interpolate_labels, find_touching_region_pairs, find_end_channel_pairs, merge_small_regions, merge_small_regions_pbc
    # print(f"TIMER STARTED")
    logger.info(f"TIMER STARTED")
    # h = 0.11
    # NOTE: uncomment these imports when refactoring
    # from .helper import find_touching_region_pairs,compute_wall_cells, move_maximum_to_peak, process_wall_maxima, is_inside_or_close_to_boundary, get_fractional_coordinates, interpolate_labels
    # smooth = False
    if spacing is None:
        logger.warning('WARNING: No spacing provided. Estimating. Provide the dgrid spacing, for more accurate calculation!\n')
        spacing = np.min(mat_atoms.get_cell_lengths_and_angles()[:3]/ (np.array(dgrid.shape)-1))
        logger.info(f"Spacing estimated from dgrid and ase_atoms is {spacing}")

    if minimum_window_separation is None:
        minimum_window_separation = spacing * 3

    logger.info(f"h is {h}")
    plr = find_pore_limiting_radius(dgrid)

    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Pore limiting radius is {np.round(plr, decimals=2)}\n")
    if (plr-0.25) < mask_thickness:
        logger.warning('WARNING: Mask thickness is too high, resetting to pore limiting radius - 0.25\n')

    mask_thickness = np.min([plr - 0.25, mask_thickness])
    # min_distance = 1.1
#     threshold_abs = 1
# # merged_maxima, all_maxima = detect_periodic_maxima_dgrid(dgrid, ase_atoms=mat_atoms, spacing=spacing, h=h)
    dt, merged_maxima, all_maxima =detect_periodic_maxima_direct(
        dgrid,
        mat_atoms,
        spacing,
        h=h,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
        # smooth=smooth,
        maxima_mask=mask_thickness,
    )

    temp = np.ceil(merged_maxima)
    # compute wall cells
    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Computing maxima with PBC\n")
    #Computed merged maxima nad regions_pbc!
    # padding_thickness=5
    padding_thickness = calculate_padding_thickness(dgrid, mat_atoms, spacing) + np.ceil(extra_padding/spacing).astype(int)
    logger.info(f"padding thickness is {padding_thickness}")
    wall_cells = compute_wall_cells(mat_atoms, spacing, wall_thickness)
    wall_maxima_to_remove = process_wall_maxima(
        merged_maxima,
        dgrid,
        padding_thickness=padding_thickness,
        wall_cells=wall_cells,
        ase_atoms=mat_atoms,
    )
    wall_maxima_to_remove = []
    #logger.info("maxima to remove are", wall_maxima_to_remove)
    optimized_positions = []
    optimized_values = []
    original_indices = []
    for i, pos in enumerate(merged_maxima):
        if i not in wall_maxima_to_remove:
            # padded_pos = pos + padding_thickness
            opt_pos, opt_val = move_maximum_to_peak(pos, dt, padding_thickness)
            optimized_positions.append(opt_pos)
            optimized_values.append(opt_val)
            original_indices.append(i)
    optimized_positions = np.vstack(optimized_positions)
    optimized_positions.shape
    all_removals = set(wall_maxima_to_remove)
    merged_maxima = np.vstack(
        [m for i, m in enumerate(merged_maxima) if i not in all_removals]
    )


    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Computing regions with PBC \n")
    # compactness = 0
    # compute markers for padded, periodic segmentation
    markers = np.zeros_like(dgrid, dtype=int)
    if len(merged_maxima) > 0:
        maxima_coords = tuple(merged_maxima.T)
        markers[maxima_coords] = np.arange(1, len(merged_maxima) + 1)
    else:
        logger.info("PROBLEM: No valid maxima detected after merging.")
    # void mask for segmentation to use the mask_thickness
    void_mask = dgrid > mask_thickness
    padded_grid = np.pad(dt, padding_thickness, mode="wrap")
    padded_markers = np.pad(markers, padding_thickness, mode="wrap")
    padded_void = np.pad(void_mask, padding_thickness, mode="wrap")
    # Run the watershed segmentation using the merged markers.
    padded_regions = watershed(
        -padded_grid,
        padded_markers,
        mask=padded_void,
        compactness=compactness,
        connectivity=2,  # 26-connectivity for full 3D
    )

    # clean up the regions_pbc from small regions, remove less than 0.01% of the total size
    padded_regions_cleaned = merge_small_regions(padded_regions, min_size=min_region_size)
    unique_padded_regions= np.unique(padded_regions)
    unique_padded_regions_cleaned = np.unique(padded_regions_cleaned)
    removed_labels_pbc =unique_padded_regions[~np.isin(unique_padded_regions,unique_padded_regions_cleaned)]
    # Only keep removed labels that correspond to original maxima
    max_label = len(merged_maxima)
    removed_maxima_labels = removed_labels_pbc[(removed_labels_pbc >= 1) & (removed_labels_pbc <= max_label)]
    logger.info(f"Removing these small maxima labels{removed_maxima_labels}")
    # Remove those rows from merged_maxima
    merged_maxima = np.delete(merged_maxima, removed_maxima_labels - 1, axis=0)
    # replace regions_pbc with regions_pbc_cleaned
    padded_regions=padded_regions_cleaned

    # Extract central (unpadded) region to obtain regions_pbc.
    regions_pbc = padded_regions[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]
    # print("Computed merged maxima nad regions_pbc!!")
    logger.info('Removing the small regions, with counts < 0.001 * regions_pbc.size')

    #
    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Computing maxima and regions without PBC\n")

# Step 2: Get unmerged maxima info using valid maxima
    original_shape = np.array(dgrid.shape)
    padded_shape = np.array(padded_grid.shape)
    new_maxima_info_list = []
    counter = 1
    for m in merged_maxima:
            m_pad = m + padding_thickness
            offset_ranges = [
                range(-((p - 1) // o), (p - 1) // o + 1)
                for p, o in zip(padded_shape, original_shape)
            ]
            for offs in np.array(np.meshgrid(*offset_ranges)).T.reshape(-1, 3):
                candidate = m_pad + offs * original_shape
                if np.all(candidate >= 0) and np.all(candidate < padded_shape):
                    candidate_unpadded = candidate - padding_thickness
                    offset = candidate_unpadded // original_shape
                    image_coord = np.mod(candidate_unpadded, original_shape)
                    new_maxima_info_list.append(
                        [
                            counter,
                            image_coord[0],
                            image_coord[1],
                            image_coord[2],
                            offset[0],
                            offset[1],
                            offset[2],
                        ]
                    )
                    counter += 1

    # for m in merged_maxima:
    #     m_pad = m + padding_thickness
    #     for offs in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(
    #     # for offs in np.array(np.meshgrid(*[range(-2 ,3 ) for _ in range(3)])).T.reshape(
    #         -1, 3
    #     ):
    #         candidate = m_pad + offs * original_shape
    #         if np.all(candidate >= 0) and np.all(candidate < padded_shape):
    #             candidate_unpadded = candidate - padding_thickness
    #             offset = candidate_unpadded // original_shape
    #             image_coord = np.mod(candidate_unpadded, original_shape)
    #             new_maxima_info_list.append(
    #                 [
    #                     counter,
    #                     image_coord[0],
    #                     image_coord[1],
    #                     image_coord[2],
    #                     offset[0],
    #                     offset[1],
    #                     offset[2],
    #                 ]
    #             )
    #             counter += 1
    #
    maxima_info = np.array(new_maxima_info_list,dtype=int)
    # print(maxima_info)
    # merged_maxima
# Create unmerged markers and regions using valid maxima
    unmerged_markers = np.zeros_like(padded_grid, dtype=int)
    for info in maxima_info:
        # WARN: should we add the offset times shape here?  fixed
        pos = info[1:4] + padding_thickness + info[4:7] * original_shape
        unmerged_markers[tuple(pos)] = info[0]
# Compute unmerged regions
    unmerged_padded_regions = watershed(
        -padded_grid,
        unmerged_markers,
        mask=padded_void,
        compactness=compactness,
        connectivity=2,
    )
# grab the central block
    unmerged_regions = unmerged_padded_regions[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]

    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Computed regions without PBC\n")
    endchannel_pairs = find_end_channel_pairs(regions_pbc, unmerged_regions, size_ratio=0.3)
    for p in endchannel_pairs:
        # print(f"replacing {p[0]} with {p[1]}")
        unmerged_regions[unmerged_regions == p[0]] = p[1]
    

    # print('Looking for noisy regions, counts < 0.001 * regions.size')
    # unmerged_regions_cleaned =  merge_small_regions(unmerged_regions, min_size=unmerged_regions.size * 0.001)
    # unique_regions = np.unique(unmerged_regions)
    # unique_regions_cleaned = np.unique(unmerged_regions_cleaned)
    # # find the labels that were removed, if any.
    # removed_labels = unique_regions[~np.isin(unique_regions, unique_regions_cleaned)]
    # # remove the corresponding maxima from maxima_info array
    # maxima_info = np.delete(maxima_info, removed_labels - 1, axis=0)
    # # finally replace unmerged regions with the cleaned version
    # unmerged_regions = unmerged_regions_cleaned
    #
    # set_trace()
    logger.info(f"TIME(s):', {np.round(time() - start_time, decimals=2)} ,'Reindexng\n")

# NOTE: We need to reindex here,
# index mapping for the merged_maxima in unmerged parts
# Ensure merged maxima indices run from 1 to n
    wall_region_thicknesses = compute_wall_cells(mat_atoms, spacing, wall_thickness=wall_thickness)
# let's start by keeping only those maxima that are there in the regions array
    maxima_indices = maxima_info[:, 1:4] + \
        maxima_info[:, 4:7] * np.array(dgrid.shape)
# Get location classifications for all maxima
    inside_close_outside = is_inside_or_close_to_boundary(
        maxima_indices, dgrid.shape, wall_region_thicknesses
    )
# Create a boolean mask for maxima that are either inside or close_outside
    valid_maxima_mask = np.array(
        [loc in ["inside", "close_outside"] for loc in inside_close_outside]
    )
# Find the core maxima ids (region labels), don't forget to add 1 to get label from index
    valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1
    # print("Computed maxima close to the cell")
    valid_maxima_indices = maxima_indices[valid_maxima_mask]
# Reindexing logic
    merged_maxima_indices = {tuple(m): i + 1 for i, m in enumerate(merged_maxima)}
    logger.info('constructing remap dict\n')
    used_indices = set()
    remap_dict = {}

    # NOTE: Assign indices to periodic maxima
    for info in maxima_info:
        original_maxima = tuple(info[1:4])
        if np.all(info[4:7] == 0) and original_maxima in merged_maxima_indices:
            # remap_dict[info[0]] = merged_maxima_indices[original_maxima]
            # if info[0] in remap_dict and remap_dict[info[0]] != merged_maxima_indices[original_maxima]:
            target_index = merged_maxima_indices[original_maxima]
            # Check if the target index is already used
            if target_index in used_indices:
                logger.info(f"Conflict: Maxima {info[0]} maps to already-used index {target_index}")
            else:
                remap_dict[info[0]] = target_index
                used_indices.add(target_index)

            # remap_dict[info[0]] = merged_maxima_indices[original_maxima]

    # print('stage 1 maxima_pbc', remap_dict)
    unique_unmerged_regions = np.delete(np.unique(unmerged_regions), 0)
    # print(
    #     "All closeby maxima are in unmerged regions",
    #     np.all(np.isin(valid_maxima_ids, unique_unmerged_regions)),
    # )
    # NOTE: Assign indices to valid maxima
    # new_index = len(merged_maxima) + 1
    next_index = max(used_indices) + 1 if used_indices else 1
    for vid in valid_maxima_ids:
        if vid not in remap_dict:
            remap_dict[vid] =next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 2 valid_maxima', remap_dict)

    # NOTE: Assign indices to remaining unique regions in unmerged_regions
    next_index = max(used_indices) + 1 if used_indices else 1
    for r in unique_unmerged_regions:
        if r not in remap_dict:
            remap_dict[r] =next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 3 unique_regions', remap_dict)

    # NOTE: Assign indices to remaining unique regions in unmerged_regions
    unique_unmerged_padded_regions = np.delete(
        np.unique(unmerged_padded_regions), 0)
    # new_index = len(unique_unmerged_regions) + 1
    next_index = max(used_indices) + 1 if used_indices else 1

    # we need to reindex the padded regions after computing connections
    padding_indices_start_here = next_index
    for r in unique_unmerged_padded_regions:
        if r not in list(remap_dict.keys()):
            remap_dict[r] =next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 4 unique_unmerged_padded_regions', remap_dict)

    # print('remap_dict constructed')
    maxima_info_copy = copy.deepcopy(maxima_info)

# great! now we have a complet map. time to remap the maxima_info, unmerged_regions, and unmerged_padded_regions
    regions = copy.deepcopy(unmerged_regions)
    regions_padded= copy.deepcopy(unmerged_padded_regions)
    for k, v in remap_dict.items():
        # print(f"replace label {k} with {v}")
        maxima_info_copy[k - 1][0] = v
        regions[unmerged_regions == k] = v
        regions_padded[unmerged_padded_regions == k] = v
# sort maxima based on first column
    maxima_info_copy = maxima_info_copy[maxima_info_copy[:, 0].argsort()]
# maxima_info[:10, :], maxima_info_copy[:10, :]
    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)}.  Done with reindexing-1/2\n")
    # print('remap_dict is ', remap_dict)

    pbc_groups = [v for k, v in get_pbc_groups(regions, maxima_info_copy).items()]
    len(pbc_groups[0])
# find touching pairs in the regions padded
    pairs = find_touching_region_pairs(regions_padded)
    # print(pairs, pairs.shape)
    wall_region_thicknesses = compute_wall_cells(mat_atoms, spacing, wall_thickness=wall_thickness)
# let's start by keeping only those maxima that are there in the regions array
    maxima_indices = maxima_info_copy[:, 1:4] + maxima_info_copy[:, 4:7] * np.array(
        dgrid.shape
    )
# Get location classifications for all maxima
    inside_close_outside = is_inside_or_close_to_boundary(
        maxima_indices, dgrid.shape, wall_region_thicknesses
    )
# Create a boolean mask for maxima that are either inside or close_outside
    valid_maxima_mask = np.array(
        [loc in ["inside", "close_outside"] for loc in inside_close_outside]
    )
# lets find the core maxima ids (region labels), dont' forget to add 1 to get label from index
    valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1
    # print("Computed maxima close to the cell")
    pairs_filtered = [p for p in pairs if np.any(np.isin(p, valid_maxima_ids))]

    logger.info(f"TIME(s): { np.round(time() - start_time, decimals=2)} Computing connections--this might take a while! \n")

    def compute_boundary(reg):
        return reg, sparse.COO(
            find_boundaries(regions_padded == reg, mode="outer").astype(int)
        )


# find uniques and remove background use the reindexed regions here
    regions_to_check = np.delete(np.unique(regions_padded), 0)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_boundary, regions_to_check))
    outer = dict(results)
# don't screw this line up
    A_unit = complete_cell(mat_atoms.cell).T
    minimum_window_separation = minimum_window_separation
    # padding_thickness = mgr.calculate_padding_thickness(dgrid, mat_atoms, spacing)
    padded_grid = np.pad(dgrid, padding_thickness, mode="wrap")
    dinterp = interpolate_labels(dgrid)

    def check_connected_sparse(regs):
        import numpy as np
        from icecream import ic
        import sparse

        ic.disable()
        from sklearn.cluster import AgglomerativeClustering

        i = regs[0]
        j = regs[1]
        # print(f"region 1 is {i} and region 2 is {j}")
        ic(regs)
        mask_i = sparse.COO((regions_padded == i).astype(int))
        mask_j = sparse.COO((regions_padded == j).astype(int))
        check_flag_sparse = mask_i * outer.get(j) + mask_j * outer.get(i)
        if check_flag_sparse.sum() > 0:
            # print('for regions ', i, j)
            # print("Points found in shared boundary:", check_flag_sparse.sum())

            # WARN: this is on the padded grid
            windices = np.vstack(np.where(check_flag_sparse)).T  # * (N,3)
            window_points = np.dot(
                A_unit, ((windices - padding_thickness) / dgrid.shape).T
            ).T
            # * Agglomerative clustering instead:
            ag = AgglomerativeClustering(
                n_clusters=None,
                linkage="single",
                distance_threshold=minimum_window_separation,
                compute_full_tree=True,
            ).fit(window_points)
            # * what if we took the index of the point where the distance value is the largest in the window boundary region, that way we don't have to worry bout
            # * regions at the wall
            d_windices = padded_grid[
                tuple(windices.T)
            ]  # distance values at the window indices
            r_windices = regions_padded[
                tuple(windices.T)
            ]  # region labels at the window indices
            # print('these are the window regions indices', r_windices)
            # print('These are window centers', window_centers)
            window_centers = []
            # print('unique cluster labels are ', np.unique(ag.labels_))
            # if len(np.unique(ag.labels_)) > 1:
            #print the sizes of each cluster
                # for uwi in np.unique(ag.labels_):
                    # print('cluster label ', uwi, 'size is ', np.sum(ag.labels_ == uwi))
            # set_trace()
            for uwi in np.unique(ag.labels_):
                # for each cluster that is nota noise, do this
                if uwi != -1:
                    uwi_dvals = d_windices[
                        ag.labels_ == uwi
                    ]  # distance values in the cluster with label uwi
                    uwi_points = window_points[
                        ag.labels_ == uwi
                    ]  # points in the cluster with label uwi
                    uwi_r = r_windices[ag.labels_ == uwi]
                    # for points in first region
                    # uwi_r1_indices =np.transpose((uwi_r==regs[0])).nonzero()
                    uwi_r1_indices = (np.where)(uwi_r == regs[0])[0]
                    # print('uwi_r1_indices are ', uwi_r1_indices)
                    # print(uwi_r1_indices.shape)
                    uwi_r1_dvals = uwi_dvals[uwi_r1_indices]
                    uwi_r1_points = uwi_points[uwi_r1_indices]
                    uwi_r1_max_indices = np.argmax( uwi_r1_dvals)  # np.where(uwi_r1_dvals == np.max(uwi_r1_dvals))[0] # indices of the maximum distance values in the cluster with label uwi
                    # print(uwi_r1_max_indices)
                    uwi_r1_max_points = uwi_r1_points[uwi_r1_max_indices]
                    # for points in second region
                    uwi_r2_indices = (np.where)(uwi_r == regs[1])[0]
                    uwi_r2_dvals = uwi_dvals[uwi_r2_indices]
                    uwi_r2_points = uwi_points[uwi_r2_indices]
                    # uwi_r2_max_indices = np.where( uwi_r2_dvals == np.max(uwi_r2_dvals))[ 0 ]  # indices of the maximum distance values in the cluster with label uwi
                    uwi_r2_max_indices = np.argmax( uwi_r2_dvals)  # np.where(uwi_r1_dvals == np.max(uwi_r1_dvals))[0] # indices of the maximum distance values in the cluster with label uwi
                    uwi_r2_max_points = uwi_r2_points[uwi_r2_max_indices]
                    # print(uwi_r2_max_indices)
                    # take the centroid of the two maxima of each region close to the window
                    uwi_center = (uwi_r1_max_points + uwi_r2_max_points) / 2
                    # uwi_center = np.mean(uwi_max_points, axis=0) # mean of the points where the distance is at maximum if ther are multiple a.k.a the window is symmetric on each side
                    # add that to the list of windows
                    window_centers.append(uwi_center)
            # window_centers = np.vstack(window_centers)
            # else:
            # print(f"Window centers before fractoinal are: {window_centers}")
            # window_centers = window_points
            window_centers = np.vstack(window_centers)
            window_fractional = get_fractional_coordinates(
                window_centers, mat_atoms)
            # this doesn't seem right, aren't these on the padded grid?
            window_fractional = np.mod(window_fractional, 1)
            window_fractional = np.round(window_fractional, decimals=4)
            window_radii = np.array(dinterp(window_fractional))
            return np.array(
                [regs, True, "internal", window_centers, window_radii], dtype="object"
            )  # returns a list of [[r1,r2], connected or not, array of window_centers]
        else:
            return np.array([regs, False, "internal", None, None], dtype="object")


    def process_pair(p):
        return check_connected_sparse(p)

    # logger.info('TIME(s):', np.round(time() - start_time, decimals=2),'Adjusting for periodic connections\n')
    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Adjusting for periodic connections\n")

# * actual connections compute.

    with ThreadPoolExecutor() as executor:
        connections_list = list(
            tqdm(executor.map(process_pair, pairs_filtered), total=len(pairs_filtered), disable=disable_tqdm)
        )
    
# let's mark a connection as periodic if any maxima is far from the box, no reindexing just yet.
    for c in connections_list:
        if np.any(~np.isin(c[0], valid_maxima_ids)):
            # print(f"This connection is {c[0]}")
            c[2] = "periodic"
    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Done with computing connections\n")
    logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Reindexing part 2/2-- Connected nodes getting priority\n")

# NOTE: if padded regions are in the connections list, we need to prioritize those in the indexing over other regions in regions padded
# make a remap dictionary
    remap_dict2 = {}
    unique_regions_padded = np.delete(np.unique(regions_padded), 0)
# here we removed 0 for clarity
    unique_regions = np.delete(np.unique(regions), 0)
    # new_index = len(unique_regions) + 1  # can also do regions.max() + 1
    next_index = padding_indices_start_here
    unique_connected_indices = np.unique( np.vstack([conn[0] for conn in connections_list]))
    for c in unique_connected_indices:
        # if c not in unique_regions:  # this is node that needs priority
        if c >= padding_indices_start_here:  # this is node that needs priority
            remap_dict2[c] =next_index
            next_index+= 1
    for r in unique_regions_padded:
        if np.logical_and(r not in remap_dict2, r >= padding_indices_start_here):
            remap_dict2[r] = next_index
            next_index+= 1
    # print('remap dict 2 is', remap_dict2)
    # set_trace()
    regions_padded_copy = copy.deepcopy(regions_padded)
    for k, v in remap_dict2.items():
        # print(f"replace label {k} with {v}")
        regions_padded_copy[regions_padded == k] = v
    connections_list_copy = copy.deepcopy(connections_list)
    for i, c in enumerate(connections_list):
        for j, ci in enumerate(c[0]):
            if ci in remap_dict2:
                # reindx the connections list
                connections_list_copy[i][0][j] = remap_dict2[ci]
    maxima_info_copy_copy = copy.deepcopy(maxima_info_copy)
    for k, v in remap_dict2.items():
        # print(f"replace label {k} with {v}")
        maxima_info_copy_copy[k - 1][0] = v
    maxima_info_copy_copy = maxima_info_copy_copy[maxima_info_copy_copy[:, 0].argsort(
    )]

    # outputs=dict(maxima_pbc=merged_maxima, regions_pbc=regions_pbc, maxima_info=maxima_info_copy_copy, regions=regions,conections=connections_list_copy, pbc_groups=pbc_groups, close_or_inside_maxima=valid_maxima_indices, close_or_inside_regions= valid_maxima_ids, unique_regions = unique_regions)
    all_maxima = maxima_info_copy_copy[:, 1:4] + maxima_info_copy_copy[:, 4:7] * np.array(dgrid.shape)
    outputs=dict(dgrid=dgrid, maxima_pbc=merged_maxima, regions_pbc=regions_pbc, maxima_info=maxima_info_copy_copy, regions=regions,connections=connections_list_copy, pbc_groups=pbc_groups, close_or_inside_maxima=valid_maxima_indices, close_or_inside_regions= valid_maxima_ids, unique_regions = unique_regions, maxima_pbc_dict=merged_maxima_indices, all_maxima=all_maxima, maxima_relative_locations=np.array(inside_close_outside), )
    # we need to save the input params to the results, for future reference.
    outputs['input_params'] = {
        'mask_thickness': mask_thickness,
        'h': h,
        'min_distance': min_distance,
        'threshold_abs': threshold_abs,
        'spacing': spacing,
        'minimum_window_separation': minimum_window_separation,
        'compactness': compactness,
        'wall_thickness': wall_thickness,
        'extra_padding': extra_padding,
        'padding_thickness': padding_thickness,
        'min_region_size': min_region_size,
    }
    if return_padded_regions:
        outputs['regions_padded'] = regions_padded
        
    return outputs

def do_periodic_segmentation_basic(
    dgrid, mat_atoms, mask_thickness=0.5, h=0.11, min_distance=1.1, threshold_abs=1.0,
    return_padded_regions=False, spacing=None, minimum_window_separation=None,
    wall_thickness=0.05, extra_padding=5, compactness=0, min_region_size=100,
    disable_tqdm=False, compute_connections=True,
    return_dgrid = False, 
):
    import numpy as np
    from skimage.segmentation import watershed
    import copy
    from ase.geometry import complete_cell
    import sparse
    from skimage.segmentation import find_boundaries
    from concurrent.futures import ThreadPoolExecutor
    from time import time
    from tqdm import tqdm
    start_time = time()
    from .helper import compute_wall_cells, move_maximum_to_peak, process_wall_maxima, is_inside_or_close_to_boundary, get_fractional_coordinates, interpolate_labels, find_touching_region_pairs, find_end_channel_pairs, merge_small_regions, merge_small_regions_pbc
    # print(f"timer started")
    logger.info(f"timer started")
    # h = 0.11
    # note: uncomment these imports when refactoring
    # from .helper import find_touching_region_pairs,compute_wall_cells, move_maximum_to_peak, process_wall_maxima, is_inside_or_close_to_boundary, get_fractional_coordinates, interpolate_labels
    # smooth = false
    if spacing is None:
        logger.warning('warning: no spacing provided. estimating. provide the dgrid spacing, for more accurate calculation!\n')
        spacing = np.min(mat_atoms.get_cell_lengths_and_angles()[:3]/ (np.array(dgrid.shape)-1))
        logger.info(f"spacing estimated from dgrid and ase_atoms is {spacing}")

    if minimum_window_separation is None:
        minimum_window_separation = spacing * 3

    logger.info(f"h is {h}")
    plr = find_pore_limiting_radius(dgrid)

    logger.info(f"time(s): {np.round(time() - start_time, decimals=2)} pore limiting radius is {np.round(plr, decimals=2)}\n")
    if (plr-0.25) < mask_thickness:
        logger.warning('warning: mask thickness is too high, resetting to pore limiting radius - 0.25\n')

    mask_thickness = np.min([plr - 0.25, mask_thickness])
    # min_distance = 1.1
#     threshold_abs = 1
# # merged_maxima, all_maxima = detect_periodic_maxima_dgrid(dgrid, ase_atoms=mat_atoms, spacing=spacing, h=h)
    dt, merged_maxima, all_maxima =detect_periodic_maxima_direct(
        dgrid,
        mat_atoms,
        spacing,
        h=h,
        threshold_abs=threshold_abs,
        min_distance=min_distance,
        # smooth=smooth,
        maxima_mask=mask_thickness,
    )

    temp = np.ceil(merged_maxima)
    # compute wall cells
    logger.info(f"time(s): {np.round(time() - start_time, decimals=2)} computing maxima with pbc\n")
    #computed merged maxima nad regions_pbc!
    # padding_thickness=5
    padding_thickness = calculate_padding_thickness(dgrid, mat_atoms, spacing) + np.ceil(extra_padding/spacing).astype(int)
    logger.info(f"padding thickness is {padding_thickness}")
    wall_cells = compute_wall_cells(mat_atoms, spacing, wall_thickness)
    wall_maxima_to_remove = process_wall_maxima(
        merged_maxima,
        dgrid,
        padding_thickness=padding_thickness,
        wall_cells=wall_cells,
        ase_atoms=mat_atoms,
    )
    wall_maxima_to_remove = []
    #logger.info("maxima to remove are", wall_maxima_to_remove)
    optimized_positions = []
    optimized_values = []
    original_indices = []
    for i, pos in enumerate(merged_maxima):
        if i not in wall_maxima_to_remove:
            # padded_pos = pos + padding_thickness
            opt_pos, opt_val = move_maximum_to_peak(pos, dt, padding_thickness)
            optimized_positions.append(opt_pos)
            optimized_values.append(opt_val)
            original_indices.append(i)
    optimized_positions = np.vstack(optimized_positions)
    optimized_positions.shape
    all_removals = set(wall_maxima_to_remove)
    merged_maxima = np.vstack(
        [m for i, m in enumerate(merged_maxima) if i not in all_removals]
    )


    logger.info(f"time(s): {np.round(time() - start_time, decimals=2)} computing regions with pbc \n")
    # compactness = 0
    # compute markers for padded, periodic segmentation
    markers = np.zeros_like(dgrid, dtype=int)
    if len(merged_maxima) > 0:
        maxima_coords = tuple(merged_maxima.T)
        markers[maxima_coords] = np.arange(1, len(merged_maxima) + 1)
    else:
        logger.info("problem: no valid maxima detected after merging.")
    # void mask for segmentation to use the mask_thickness
    void_mask = dgrid > mask_thickness
    padded_grid = np.pad(dt, padding_thickness, mode="wrap")
    padded_markers = np.pad(markers, padding_thickness, mode="wrap")
    padded_void = np.pad(void_mask, padding_thickness, mode="wrap")
    # run the watershed segmentation using the merged markers.
    padded_regions = watershed(
        -padded_grid,
        padded_markers,
        mask=padded_void,
        compactness=compactness,
        connectivity=2,  # 26-connectivity for full 3d
    )

    # clean up the regions_pbc from small regions, remove less than 0.01% of the total size
    padded_regions_cleaned = merge_small_regions(padded_regions, min_size=min_region_size)
    unique_padded_regions= np.unique(padded_regions)
    unique_padded_regions_cleaned = np.unique(padded_regions_cleaned)
    removed_labels_pbc =unique_padded_regions[~np.isin(unique_padded_regions,unique_padded_regions_cleaned)]
    # only keep removed labels that correspond to original maxima
    max_label = len(merged_maxima)
    removed_maxima_labels = removed_labels_pbc[(removed_labels_pbc >= 1) & (removed_labels_pbc <= max_label)]
    logger.info(f"removing these small maxima labels{removed_maxima_labels}")
    # remove those rows from merged_maxima
    merged_maxima = np.delete(merged_maxima, removed_maxima_labels - 1, axis=0)
    # replace regions_pbc with regions_pbc_cleaned
    padded_regions=padded_regions_cleaned

    # extract central (unpadded) region to obtain regions_pbc.
    regions_pbc = padded_regions[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]
    # print("computed merged maxima nad regions_pbc!!")
    logger.info('removing the small regions, with counts < 0.001 * regions_pbc.size')

    #
    logger.info(f"time(s): {np.round(time() - start_time, decimals=2)} computing maxima and regions without pbc\n")

# step 2: get unmerged maxima info using valid maxima
    original_shape = np.array(dgrid.shape)
    padded_shape = np.array(padded_grid.shape)
    new_maxima_info_list = []
    counter = 1
    for m in merged_maxima:
            m_pad = m + padding_thickness
            offset_ranges = [
                range(-((p - 1) // o), (p - 1) // o + 1)
                for p, o in zip(padded_shape, original_shape)
            ]
            for offs in np.array(np.meshgrid(*offset_ranges)).T.reshape(-1, 3):
                candidate = m_pad + offs * original_shape
                if np.all(candidate >= 0) and np.all(candidate < padded_shape):
                    candidate_unpadded = candidate - padding_thickness
                    offset = candidate_unpadded // original_shape
                    image_coord = np.mod(candidate_unpadded, original_shape)
                    new_maxima_info_list.append(
                        [
                            counter,
                            image_coord[0],
                            image_coord[1],
                            image_coord[2],
                            offset[0],
                            offset[1],
                            offset[2],
                        ]
                    )
                    counter += 1

    # for m in merged_maxima:
    #     m_pad = m + padding_thickness
    #     for offs in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).t.reshape(
    #     # for offs in np.array(np.meshgrid(*[range(-2 ,3 ) for _ in range(3)])).t.reshape(
    #         -1, 3
    #     ):
    #         candidate = m_pad + offs * original_shape
    #         if np.all(candidate >= 0) and np.all(candidate < padded_shape):
    #             candidate_unpadded = candidate - padding_thickness
    #             offset = candidate_unpadded // original_shape
    #             image_coord = np.mod(candidate_unpadded, original_shape)
    #             new_maxima_info_list.append(
    #                 [
    #                     counter,
    #                     image_coord[0],
    #                     image_coord[1],
    #                     image_coord[2],
    #                     offset[0],
    #                     offset[1],
    #                     offset[2],
    #                 ]
    #             )
    #             counter += 1
    #
    maxima_info = np.array(new_maxima_info_list,dtype=int)
    # print(maxima_info)
    # merged_maxima
# create unmerged markers and regions using valid maxima
    unmerged_markers = np.zeros_like(padded_grid, dtype=int)
    for info in maxima_info:
        # warn: should we add the offset times shape here?  fixed
        pos = info[1:4] + padding_thickness + info[4:7] * original_shape
        unmerged_markers[tuple(pos)] = info[0]
# compute unmerged regions
    unmerged_padded_regions = watershed(
        -padded_grid,
        unmerged_markers,
        mask=padded_void,
        compactness=compactness,
        connectivity=2,
    )
# grab the central block
    unmerged_regions = unmerged_padded_regions[
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
        padding_thickness:-padding_thickness,
    ]

    logger.info(f"time(s): {np.round(time() - start_time, decimals=2)} computed regions without pbc\n")
    endchannel_pairs = find_end_channel_pairs(regions_pbc, unmerged_regions, size_ratio=0.3)
    for p in endchannel_pairs:
        # print(f"replacing {p[0]} with {p[1]}")
        unmerged_regions[unmerged_regions == p[0]] = p[1]
    

    # print('looking for noisy regions, counts < 0.001 * regions.size')
    # unmerged_regions_cleaned =  merge_small_regions(unmerged_regions, min_size=unmerged_regions.size * 0.001)
    # unique_regions = np.unique(unmerged_regions)
    # unique_regions_cleaned = np.unique(unmerged_regions_cleaned)
    # # find the labels that were removed, if any.
    # removed_labels = unique_regions[~np.isin(unique_regions, unique_regions_cleaned)]
    # # remove the corresponding maxima from maxima_info array
    # maxima_info = np.delete(maxima_info, removed_labels - 1, axis=0)
    # # finally replace unmerged regions with the cleaned version
    # unmerged_regions = unmerged_regions_cleaned
    #
    # set_trace()
    logger.info(f"time(s):', {np.round(time() - start_time, decimals=2)} ,'reindexng\n")

# note: we need to reindex here,
# index mapping for the merged_maxima in unmerged parts
# ensure merged maxima indices run from 1 to n
    wall_region_thicknesses = compute_wall_cells(mat_atoms, spacing, wall_thickness=wall_thickness)
# let's start by keeping only those maxima that are there in the regions array
    maxima_indices = maxima_info[:, 1:4] + \
        maxima_info[:, 4:7] * np.array(dgrid.shape)
# get location classifications for all maxima
    inside_close_outside = is_inside_or_close_to_boundary(
        maxima_indices, dgrid.shape, wall_region_thicknesses
    )
# create a boolean mask for maxima that are either inside or close_outside
    valid_maxima_mask = np.array(
        [loc in ["inside", "close_outside"] for loc in inside_close_outside]
    )
# find the core maxima ids (region labels), don't forget to add 1 to get label from index
    valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1
    # print("computed maxima close to the cell")
    valid_maxima_indices = maxima_indices[valid_maxima_mask]
# reindexing logic
    merged_maxima_indices = {tuple(m): i + 1 for i, m in enumerate(merged_maxima)}
    logger.info('constructing remap dict\n')
    used_indices = set()
    remap_dict = {}

    # note: assign indices to periodic maxima
    for info in maxima_info:
        original_maxima = tuple(info[1:4])
        if np.all(info[4:7] == 0) and original_maxima in merged_maxima_indices:
            # remap_dict[info[0]] = merged_maxima_indices[original_maxima]
            # if info[0] in remap_dict and remap_dict[info[0]] != merged_maxima_indices[original_maxima]:
            target_index = merged_maxima_indices[original_maxima]
            # check if the target index is already used
            if target_index in used_indices:
                logger.info(f"conflict: maxima {info[0]} maps to already-used index {target_index}")
            else:
                remap_dict[info[0]] = target_index
                used_indices.add(target_index)

            # remap_dict[info[0]] = merged_maxima_indices[original_maxima]

    # print('stage 1 maxima_pbc', remap_dict)
    unique_unmerged_regions = np.delete(np.unique(unmerged_regions), 0)
    # print(
    #     "all closeby maxima are in unmerged regions",
    #     np.all(np.isin(valid_maxima_ids, unique_unmerged_regions)),
    # )
    # note: assign indices to valid maxima
    # new_index = len(merged_maxima) + 1
    next_index = max(used_indices) + 1 if used_indices else 1
    for vid in valid_maxima_ids:
        if vid not in remap_dict:
            remap_dict[vid] =next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 2 valid_maxima', remap_dict)

    # note: assign indices to remaining unique regions in unmerged_regions
    next_index = max(used_indices) + 1 if used_indices else 1
    for r in unique_unmerged_regions:
        if r not in remap_dict:
            remap_dict[r] =next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 3 unique_regions', remap_dict)

    # note: assign indices to remaining unique regions in unmerged_regions
    unique_unmerged_padded_regions = np.delete(
        np.unique(unmerged_padded_regions), 0)
    # new_index = len(unique_unmerged_regions) + 1
    next_index = max(used_indices) + 1 if used_indices else 1

    # we need to reindex the padded regions after computing connections
    padding_indices_start_here = next_index
    for r in unique_unmerged_padded_regions:
        if r not in list(remap_dict.keys()):
            remap_dict[r] =next_index
            used_indices.add(next_index)
            next_index += 1
    # print('stage 4 unique_unmerged_padded_regions', remap_dict)

    # print('remap_dict constructed')
    maxima_info_copy = copy.deepcopy(maxima_info)

# great! now we have a complet map. time to remap the maxima_info, unmerged_regions, and unmerged_padded_regions
    regions = copy.deepcopy(unmerged_regions)
    regions_padded= copy.deepcopy(unmerged_padded_regions)
    for k, v in remap_dict.items():
        # print(f"replace label {k} with {v}")
        maxima_info_copy[k - 1][0] = v
        regions[unmerged_regions == k] = v
        regions_padded[unmerged_padded_regions == k] = v
# sort maxima based on first column
    maxima_info_copy = maxima_info_copy[maxima_info_copy[:, 0].argsort()]
# maxima_info[:10, :], maxima_info_copy[:10, :]
    logger.info(f"time(s): {np.round(time() - start_time, decimals=2)}.  done with reindexing-1/2\n")
    # print('remap_dict is ', remap_dict)

    pbc_groups = [v for k, v in get_pbc_groups(regions, maxima_info_copy).items()]
    len(pbc_groups[0])
# find touching pairs in the regions padded
    pairs = find_touching_region_pairs(regions_padded)
    # print(pairs, pairs.shape)
    wall_region_thicknesses = compute_wall_cells(mat_atoms, spacing, wall_thickness=wall_thickness)
# let's start by keeping only those maxima that are there in the regions array
    maxima_indices = maxima_info_copy[:, 1:4] + maxima_info_copy[:, 4:7] * np.array(
        dgrid.shape
    )
# get location classifications for all maxima
    inside_close_outside = is_inside_or_close_to_boundary(
        maxima_indices, dgrid.shape, wall_region_thicknesses
    )
# create a boolean mask for maxima that are either inside or close_outside
    valid_maxima_mask = np.array(
        [loc in ["inside", "close_outside"] for loc in inside_close_outside]
    )
# lets find the core maxima ids (region labels), dont' forget to add 1 to get label from index
    valid_maxima_ids = np.where(valid_maxima_mask)[0] + 1
    # print("computed maxima close to the cell")
    pairs_filtered = [p for p in pairs if np.any(np.isin(p, valid_maxima_ids))]


   # --- [copy everything up to and including maxima_info_copy = maxima_info_copy[maxima_info_copy[:, 0].argsort()] ] ---
    # (Paste the code from the original function up to the first reindexing, just before pbc_groups = ...)
    # --- End of copy ---

    # Only compute connections and further reindexing if requested
    if not compute_connections:

        all_maxima = maxima_info_copy[:, 1:4] + maxima_info_copy[:, 4:7] * np.array(dgrid.shape)
        logger.info(f"time(s): { np.round(time() - start_time, decimals=2)} compute_connections is set to False. No connections computed. Calculations Ending! :) \n")
        outputs = dict(
            # dgrid=dgrid,
            all_maxima = all_maxima,
            maxima_pbc=merged_maxima,
            regions_pbc=regions_pbc,
            maxima_info=maxima_info_copy,
            regions=regions,
            pbc_groups = pbc_groups,
            close_or_inside_maxima=valid_maxima_indices,
            close_or_inside_regions=valid_maxima_ids,
            unique_regions=np.delete(np.unique(regions), 0),
            maxima_pbc_dict=merged_maxima_indices,
            maxima_relative_locations=np.array(inside_close_outside))
        outputs['input_params'] = {
            'mask_thickness': mask_thickness,
            'h': h,
            'min_distance': min_distance,
            'threshold_abs': threshold_abs,
            'spacing': spacing,
            'minimum_window_separation': minimum_window_separation,
            'compactness': compactness,
            'wall_thickness': wall_thickness,
            'extra_padding': extra_padding,
            'padding_thickness': padding_thickness,
            'min_region_size': min_region_size,
        }
        if return_padded_regions:
            outputs['regions_padded'] = regions_padded
    else:
        logger.info(f"time(s): { np.round(time() - start_time, decimals=2)} compute_connections is True--this might take a while! \n")

        def compute_boundary(reg):
            return reg, sparse.COO(
                find_boundaries(regions_padded == reg, mode="outer").astype(int)
            )


# find uniques and remove background use the reindexed regions here
        regions_to_check = np.delete(np.unique(regions_padded), 0)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(compute_boundary, regions_to_check))
        outer = dict(results)
# don't screw this line up
        A_unit = complete_cell(mat_atoms.cell).T
        minimum_window_separation = minimum_window_separation
        # padding_thickness = mgr.calculate_padding_thickness(dgrid, mat_atoms, spacing)
        padded_grid = np.pad(dgrid, padding_thickness, mode="wrap")
        dinterp = interpolate_labels(dgrid)

        def check_connected_sparse(regs):
            import numpy as np
            from icecream import ic
            import sparse

            ic.disable()
            from sklearn.cluster import AgglomerativeClustering

            i = regs[0]
            j = regs[1]
            # print(f"region 1 is {i} and region 2 is {j}")
            ic(regs)
            mask_i = sparse.COO((regions_padded == i).astype(int))
            mask_j = sparse.COO((regions_padded == j).astype(int))
            check_flag_sparse = mask_i * outer.get(j) + mask_j * outer.get(i)
            if check_flag_sparse.sum() > 0:
                # print('for regions ', i, j)
                # print("Points found in shared boundary:", check_flag_sparse.sum())

                # WARN: this is on the padded grid
                windices = np.vstack(np.where(check_flag_sparse)).T  # * (N,3)
                window_points = np.dot(
                    A_unit, ((windices - padding_thickness) / dgrid.shape).T
                ).T
                # * Agglomerative clustering instead:
                ag = AgglomerativeClustering(
                    n_clusters=None,
                    linkage="single",
                    distance_threshold=minimum_window_separation,
                    compute_full_tree=True,
                ).fit(window_points)
                # * what if we took the index of the point where the distance value is the largest in the window boundary region, that way we don't have to worry bout
                # * regions at the wall
                d_windices = padded_grid[
                    tuple(windices.T)
                ]  # distance values at the window indices
                r_windices = regions_padded[
                    tuple(windices.T)
                ]  # region labels at the window indices
                # print('these are the window regions indices', r_windices)
                # print('These are window centers', window_centers)
                window_centers = []
                # print('unique cluster labels are ', np.unique(ag.labels_))
                # if len(np.unique(ag.labels_)) > 1:
                #print the sizes of each cluster
                    # for uwi in np.unique(ag.labels_):
                        # print('cluster label ', uwi, 'size is ', np.sum(ag.labels_ == uwi))
                # set_trace()
                for uwi in np.unique(ag.labels_):
                    # for each cluster that is nota noise, do this
                    if uwi != -1:
                        uwi_dvals = d_windices[
                            ag.labels_ == uwi
                        ]  # distance values in the cluster with label uwi
                        uwi_points = window_points[
                            ag.labels_ == uwi
                        ]  # points in the cluster with label uwi
                        uwi_r = r_windices[ag.labels_ == uwi]
                        # for points in first region
                        # uwi_r1_indices =np.transpose((uwi_r==regs[0])).nonzero()
                        uwi_r1_indices = (np.where)(uwi_r == regs[0])[0]
                        # print('uwi_r1_indices are ', uwi_r1_indices)
                        # print(uwi_r1_indices.shape)
                        uwi_r1_dvals = uwi_dvals[uwi_r1_indices]
                        uwi_r1_points = uwi_points[uwi_r1_indices]
                        uwi_r1_max_indices = np.argmax( uwi_r1_dvals)  # np.where(uwi_r1_dvals == np.max(uwi_r1_dvals))[0] # indices of the maximum distance values in the cluster with label uwi
                        # print(uwi_r1_max_indices)
                        uwi_r1_max_points = uwi_r1_points[uwi_r1_max_indices]
                        # for points in second region
                        uwi_r2_indices = (np.where)(uwi_r == regs[1])[0]
                        uwi_r2_dvals = uwi_dvals[uwi_r2_indices]
                        uwi_r2_points = uwi_points[uwi_r2_indices]
                        # uwi_r2_max_indices = np.where( uwi_r2_dvals == np.max(uwi_r2_dvals))[ 0 ]  # indices of the maximum distance values in the cluster with label uwi
                        uwi_r2_max_indices = np.argmax( uwi_r2_dvals)  # np.where(uwi_r1_dvals == np.max(uwi_r1_dvals))[0] # indices of the maximum distance values in the cluster with label uwi
                        uwi_r2_max_points = uwi_r2_points[uwi_r2_max_indices]
                        # print(uwi_r2_max_indices)
                        # take the centroid of the two maxima of each region close to the window
                        uwi_center = (uwi_r1_max_points + uwi_r2_max_points) / 2
                        # uwi_center = np.mean(uwi_max_points, axis=0) # mean of the points where the distance is at maximum if ther are multiple a.k.a the window is symmetric on each side
                        # add that to the list of windows
                        window_centers.append(uwi_center)
                # window_centers = np.vstack(window_centers)
                # else:
                # print(f"Window centers before fractoinal are: {window_centers}")
                # window_centers = window_points
                window_centers = np.vstack(window_centers)
                window_fractional = get_fractional_coordinates(
                    window_centers, mat_atoms)
                # this doesn't seem right, aren't these on the padded grid?
                window_fractional = np.mod(window_fractional, 1)
                window_fractional = np.round(window_fractional, decimals=4)
                window_radii = np.array(dinterp(window_fractional))
                return np.array(
                    [regs, True, "internal", window_centers, window_radii], dtype="object"
                )  # returns a list of [[r1,r2], connected or not, array of window_centers]
            else:
                return np.array([regs, False, "internal", None, None], dtype="object")


        def process_pair(p):
            return check_connected_sparse(p)

        # logger.info('TIME(s):', np.round(time() - start_time, decimals=2),'Adjusting for periodic connections\n')
        logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Adjusting for periodic connections\n")

# * actual connections compute.

        with ThreadPoolExecutor() as executor:
            connections_list = list(
                tqdm(executor.map(process_pair, pairs_filtered), total=len(pairs_filtered), disable=disable_tqdm)
            )
        
# let's mark a connection as periodic if any maxima is far from the box, no reindexing just yet.
        for c in connections_list:
            if np.any(~np.isin(c[0], valid_maxima_ids)):
                # print(f"This connection is {c[0]}")
                c[2] = "periodic"
        logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Done with computing connections\n")
        logger.info(f"TIME(s): {np.round(time() - start_time, decimals=2)} Reindexing part 2/2-- Connected nodes getting priority\n")

# NOTE: if padded regions are in the connections list, we need to prioritize those in the indexing over other regions in regions padded
# make a remap dictionary
        remap_dict2 = {}
        unique_regions_padded = np.delete(np.unique(regions_padded), 0)
# here we removed 0 for clarity
        unique_regions = np.delete(np.unique(regions), 0)
        # new_index = len(unique_regions) + 1  # can also do regions.max() + 1
        next_index = padding_indices_start_here
        unique_connected_indices = np.unique( np.vstack([conn[0] for conn in connections_list]))
        for c in unique_connected_indices:
            # if c not in unique_regions:  # this is node that needs priority
            if c >= padding_indices_start_here:  # this is node that needs priority
                remap_dict2[c] =next_index
                next_index+= 1
        for r in unique_regions_padded:
            if np.logical_and(r not in remap_dict2, r >= padding_indices_start_here):
                remap_dict2[r] = next_index
                next_index+= 1
        # print('remap dict 2 is', remap_dict2)
        # set_trace()
        regions_padded_copy = copy.deepcopy(regions_padded)
        for k, v in remap_dict2.items():
            # print(f"replace label {k} with {v}")
            regions_padded_copy[regions_padded == k] = v
        connections_list_copy = copy.deepcopy(connections_list)
        for i, c in enumerate(connections_list):
            for j, ci in enumerate(c[0]):
                if ci in remap_dict2:
                    # reindx the connections list
                    connections_list_copy[i][0][j] = remap_dict2[ci]
        maxima_info_copy_copy = copy.deepcopy(maxima_info_copy)
        for k, v in remap_dict2.items():
            # print(f"replace label {k} with {v}")
            maxima_info_copy_copy[k - 1][0] = v
        maxima_info_copy_copy = maxima_info_copy_copy[maxima_info_copy_copy[:, 0].argsort(
        )]

        # outputs=dict(maxima_pbc=merged_maxima, regions_pbc=regions_pbc, maxima_info=maxima_info_copy_copy, regions=regions,conections=connections_list_copy, pbc_groups=pbc_groups, close_or_inside_maxima=valid_maxima_indices, close_or_inside_regions= valid_maxima_ids, unique_regions = unique_regions)
        all_maxima = maxima_info_copy_copy[:, 1:4] + maxima_info_copy_copy[:, 4:7] * np.array(dgrid.shape)
        outputs=dict(dgrid=dgrid, maxima_pbc=merged_maxima, regions_pbc=regions_pbc, maxima_info=maxima_info_copy_copy, regions=regions,connections=connections_list_copy, pbc_groups=pbc_groups, close_or_inside_maxima=valid_maxima_indices, close_or_inside_regions= valid_maxima_ids, unique_regions = unique_regions, maxima_pbc_dict=merged_maxima_indices, all_maxima=all_maxima, maxima_relative_locations=np.array(inside_close_outside), )
        # we need to save the input params to the results, for future reference.
        outputs['input_params'] = {
            'mask_thickness': mask_thickness,
            'h': h,
            'min_distance': min_distance,
            'threshold_abs': threshold_abs,
            'spacing': spacing,
            'minimum_window_separation': minimum_window_separation,
            'compactness': compactness,
            'wall_thickness': wall_thickness,
            'extra_padding': extra_padding,
            'padding_thickness': padding_thickness,
            'min_region_size': min_region_size,
        }
        if return_padded_regions:
            outputs['regions_padded'] = regions_padded_copy
            
    if return_dgrid:
        outputs['dgrid'] = dgrid
    return outputs



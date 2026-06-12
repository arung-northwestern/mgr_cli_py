
# """

# Helper functions for mofography
# ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ 
# ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗
# ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝
# ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗
# ██║  ██║███████╗███████╗██║     ███████╗██║  ██║
# ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
                                                

# """                         
import numpy as np
import logging
logger = logging.getLogger(__name__)

def find_coord_from_indices(indices, shape, ase_atoms):
    """Get the cartesian coordinates from the indces of a grid mapped over an ase atoms object.

    :param indices: indices of interest, can be a list of a (N,3) numpy array, where N is number of points
    :type indices: list or np.array (N,3)
    :param shape: overall shape of the 3D grid
    :type shape: np.shape output, so tuple or array of shape (1,3)
    :param ase_atoms: ASE atoms object corressponding to the grid
    :type ase_atoms: ase.atoms object
    :return: cartesian coordinates of the grid points
    :rtype: numpy arry of size (N,3) where N is the number of grid points
    """
    import ase
    import numpy as np
    cell = ase.geometry.complete_cell(ase_atoms.get_cell()).T
    return np.dot(cell, (indices/shape).T).T


def interpolate_labels(regions):
    """Create a scipy.interpolator.RegularGridInterpolator, using 'nearest' method for the region labels (or any 3D numpy grid) 
    over a unit box ([0,1]) in all axes.

    :param regions: 3D numpy array of region labels
    :type regions: numpy.array 
    :return: a scipy RegularGridInterpolator object that works on the fractional coordinates of atoms
    :rtype: scipy.interpolate.RegularGridInterpolator
    """    
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI
    xx = np.linspace(0,  1,regions.shape[0])
    yy = np.linspace(0,  1,regions.shape[1])
    zz = np.linspace(0,  1,regions.shape[2])
    rl = RGI((xx, yy, zz), regions, method='nearest')

    return rl


def get_fractional_coordinates(points, ase_atoms):
    """Find the fractional coordinates for a bunch of cartesian corrdinates in a box corresponding to an ase.atoms object

    :param points: numpy array of shape (Npoints, 3)
    :type points: numpy.array
    :param ase_atoms: ASE atoms object for the box
    :type ase_atoms: ase.atoms
    :return: numpy array of fractional coordinates of shape (Npoints,3)
    :rtype: numpy.array
    """    
    import numpy as np
    import ase
    cell= ase.geometry.complete_cell(ase_atoms.get_cell()).T 
    cell_inv = np.linalg.inv(cell)
    frac_coords = np.dot(cell_inv, points.T).T
    return frac_coords

def zoom_grid(grid, zoom_factor):
    """
    It takes a 2D array and a zoom factor, and returns a 2D array that is the original array zoomed by
    the zoom factor
    
    :param grid: the grid to be zoomed
    :param zoom_factor: The zoom factor along the axes. If a float, zoom is the same for each axis. If
    an array, zoom should contain one value for each axis
    :return: A dask array
    """


    import numpy as np
    import dask.array as da
    from scipy.ndimage import zoom
    return da.from_array(zoom(grid, zoom_factor, mode = 'grid-wrap', order=1))

def find_psd_peaks(psd, **kwargs):


    from scipy.signal import find_peaks_cwt, cwt, ricker

    # * set default parameters
    height  = kwargs.get('height', 0.2)
    width = kwargs.get('width', 15)

    # print('height: {}, width: {}'.format(height, width))
    # * compute the continuous wavelet transform
    cwt_psd= cwt(psd, wavelet=ricker, widths=[width])[0]

    #* compute peaks
    peaks = find_peaks_cwt(psd, min_snr = height,widths=width)

    #* filter out peaks that are below the height threshold
    valid_peaks = [p for p in peaks if cwt_psd[p]>height ]

    return valid_peaks



def compare_psd_peaks(peak_values1, peak_values2):



    import numpy as np 

    # * sort by length of peaks
    peak_values_array = np.array([peak_values1, peak_values2])
    sorted_indices  = np.argsort([len(peak_values1), len(peak_values2)])
    peak_values_array = peak_values_array[sorted_indices]

    # * assign to right order
    x2 = peak_values_array[0] # shorter one
    x1 = peak_values_array[1] # longer one

    # Find the indices of the closest values in x1 for each value in x2
    indices = np.interp(x2, x1, np.arange(len(x1)))

    # Round the indices to the nearest integer
    indices = np.round(indices).astype(int)

    # Get the closest values in x1 for each value in x2
    closest_values = x1[indices]

    sorted_and_matched = np.array([x2, closest_values])

    reconstructed_list = [[],[]]
    for i in sorted_indices:
        reconstructed_list[i] = sorted_and_matched[i]
    # Print the closest values
    # print(closest_values)
    return reconstructed_list

def convert_to_cygwin_path(windows_path):
    """Convert a Windows path to a Cygwin-style path and escape spaces.

    Args:
        windows_path: The Windows-style path to convert.

    Returns:
        str: The converted Cygwin-style path with escaped spaces.

    Example:
        >>> convert_to_cygwin_path(Path(r"C:\Program Files\MyApp"))
        '/c/Program\\ Files/MyApp'
    """
    import os  # Importing os for path manipulation

    cygwin_path = str(windows_path).replace('\\', '/')
    if cygwin_path[1:3] == ':':
        cygwin_path = '/' + cygwin_path[0].lower() + cygwin_path[2:]

    # Escape spaces with backslashes
    cygwin_path = cygwin_path.replace(' ', '\\ ')
    return cygwin_path


def make_command(path_to_cif, executable_path, flags=""):
    """Create a command string for running a Cygwin executable with a CIF file.

    Args:
        path_to_cif: The path to the CIF file.

        executable_path: The path to the executable.
        flags: Additional flags to include in the command.

    Returns:
        str: The command string to be executed in Cygwin.

    Example:
        >>> command = make_command(Path(r"C:\path\to\file.cif"), Path(r"C:\path\to\executable"), "-ha -res")

        >>> print(command)

        'C:/path/to/executable -ha -res C:/path/to/file.cif'
    """
    import os  # Importing os for path manipulation


    # Convert paths to Cygwin format
    cygwin_executable_path = convert_to_cygwin_path(executable_path)
    cygwin_cif_path = convert_to_cygwin_path(path_to_cif)

    # Print the resolved paths for debugging
    print("Cygwin Executable Path:", cygwin_executable_path)
    print("Cygwin CIF Path:", cygwin_cif_path)

    # Prepare the command for Cygwin terminal without quotes around paths
    command = f"{cygwin_executable_path} {flags} {cygwin_cif_path}"
    print('Command is', command)
    return command
def run_cygwin_from_jupyter(cygwin_command, cygwin_path=r"C:\cygwin64\bin"):
    """Run a Cygwin command from a Jupyter notebook.

    Args:
        cygwin_command: The command to run in Cygwin.
        cygwin_path: The path to the Cygwin installation (default is r"C:\cygwin64\bin").

    Returns:
        None: This function prints the output and errors directly.
    """
    import os  # Importing os for environment manipulation
    import subprocess  # Importing subprocess to run commands

    # Add Cygwin bin to the PATH
    os.environ["PATH"] = f"{cygwin_path};{os.environ['PATH']}"

    # Run the command
    result = subprocess.run(
        [f"{cygwin_path}\\bash.exe", "-c", cygwin_command],
        capture_output=True,
        text=True,
        shell=True
    )

    # Print the output
    print(result.stdout)

    # Print any errors
    if result.stderr:
        print("Errors:", result.stderr)

def run_cygwin_from_jupyter_rt(cygwin_command, cygwin_path=r"C:\cygwin64\bin"):
    """Run a Cygwin command from a Jupyter notebook in real-time.

    Args:
        cygwin_command: The command to run in Cygwin.
        cygwin_path: The path to the Cygwin installation (default is r"C:\cygwin64\bin").

    Returns:
        int: The return code of the process.
    """
    import os  # Importing os for environment manipulation
    import subprocess  # Importing subprocess to run commands
    import sys  # Importing sys for flushing output

    # Add Cygwin bin to the PATH
    os.environ["PATH"] = f"{cygwin_path};{os.environ['PATH']}"

    # Run the command
    process = subprocess.Popen(
        [f"{cygwin_path}\\bash.exe", "-c", cygwin_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()  # Ensure output is displayed immediately

    # Print errors (if any)
    for line in process.stderr:
        print("Error:", line, end='', file=sys.stderr)
        sys.stderr.flush()  # Ensure error output is displayed immediately

    # Wait for the process to complete and get the return code
    return_code = process.wait()

    return return_code

def spectral_smooth_3d(dgrid: np.ndarray, atoms, grid_spacing: float, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth a 3D grid using spectral filtering with proper handling of non-orthogonal cells.
    
    Parameters
    ----------
    dgrid : np.ndarray
        3D density grid
    atoms : ase.Atoms
        ASE Atoms object containing cell information
    grid_spacing : float
        Grid spacing in Angstroms (assumed uniform)
    sigma : float
        Smoothing parameter in Angstroms
        
    Returns
    -------
    np.ndarray
        Smoothed 3D grid
    """
    import numpy as np
    from scipy.fft import fftn, ifftn
    from ase.geometry import complete_cell
    
    # Get cell and grid information
    cell = complete_cell(atoms.get_cell()).T
    
    # Compute metric tensor G (equivalent to reciprocal space metric)
    G = np.linalg.inv(cell.T @ cell)
    
    # Generate k-vectors scaled by grid spacing
    # The 2π/L frequency becomes (2π/L)*(L/N) = 2π/(N*grid_spacing)
    kx = 2 * np.pi * np.fft.fftfreq(dgrid.shape[0], d=grid_spacing)
    ky = 2 * np.pi * np.fft.fftfreq(dgrid.shape[1], d=grid_spacing)
    kz = 2 * np.pi * np.fft.fftfreq(dgrid.shape[2], d=grid_spacing)
    
    # Create meshgrid with proper indexing
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Compute k^T * G * k (proper metric for non-orthogonal cell)
    k_squared = (Kx*G[0,0] + Ky*G[1,0] + Kz*G[2,0])*Kx + \
                (Kx*G[0,1] + Ky*G[1,1] + Kz*G[2,1])*Ky + \
                (Kx*G[0,2] + Ky*G[1,2] + Kz*G[2,2])*Kz
    
    # Create and apply Gaussian filter
    gaussian_filter = np.exp(-k_squared * sigma**2 / 2)
    
    # Transform to Fourier space, apply filter, and transform back
    fft_grid = fftn(dgrid)
    filtered_fft = fft_grid * gaussian_filter
    smoothed = np.real(ifftn(filtered_fft))
    
    return smoothed

# these are new functions for periodic segmentation.


def trim_connections_to_unique_regions(connections, unique_regions):
    """Keep connections with at least one endpoint label in *unique_regions*."""
    import numpy as np

    unique = np.asarray(unique_regions, dtype=int)
    if unique.size == 0:
        return list(connections)
    return [c for c in connections if np.any(np.isin(c[0], unique))]

def compute_wall_cells(ase_atoms, spacing, wall_thickness=0.05):
    import numpy as np
    from ase.geometry import complete_cell

    # get cell information for wall processing
    cell = complete_cell(ase_atoms.get_cell())
    ea, eb, ec = cell[0], cell[1], cell[2]
    vol = abs(np.linalg.det(cell))

    # calculate perpendicular lengths and wall cells
    la_perp = vol / np.linalg.norm(np.cross(eb, ec))
    lb_perp = vol / np.linalg.norm(np.cross(ec, ea))
    lc_perp = vol / np.linalg.norm(np.cross(ea, eb))
    wall_distances = np.array(
        [wall_thickness * la_perp, wall_thickness *
            lb_perp, wall_thickness * lc_perp]
    )
    wall_cells = np.ceil(wall_distances / spacing).astype(int)
    return wall_cells


def check_wall(maxima, ase_atoms, spacing, wall_thickness=0.075):
    import numpy as np
    wall_cells = compute_wall_cells(ase_atoms, spacing, wall_thickness)
    # find shape consistent with how the dgrid is constructed in dask_grid_over_atoms
    # need that +1 at the end other wise the spacing wont be exact.
    shape = (np.array(ase_atoms.get_cell_lengths_and_angles()[:3]) / spacing).astype(
        int
    ) + 1
    # print('shape is ', shape)
    is_wall_flags = []
    for i, pos in enumerate(maxima):
        is_wall = any(
            index_dim < wall_cells[dim] or index_dim > (
                shape[dim] - wall_cells[dim])
            for dim, index_dim in enumerate(pos)
        )
        is_wall_flags.append(is_wall)
    return np.vstack(is_wall_flags).reshape(-1, 1)

# functino to check if a maxim is at the wall of a cell, in multiple directions
def check_wall_per_dim(maxima, ase_atoms, spacing, wall_thickness=0.075):
    import numpy as np
    wall_cells = compute_wall_cells(ase_atoms, spacing, wall_thickness)
    # find shape consistent with how the dgrid is constructed in dask_grid_over_atoms
    # need that +1 at the end other wise the spacing wont be exact.
    shape = (np.array(ase_atoms.get_cell_lengths_and_angles()[:3]) / spacing).astype(
        int
    ) + 1
    # print('shape is ', shape)
    is_wall_flags = []
    for i, pos in enumerate(maxima):
        is_wall = [
            index_dim < wall_cells[dim] or index_dim > (
                shape[dim] - wall_cells[dim])
            for dim, index_dim in enumerate(pos)
        ]
        is_wall_flags.append(is_wall)
    return np.vstack(is_wall_flags).reshape(-1, 3)

# find pairs of touchinhg regions to check for conncections, typically on a segemented grid of regions..
def find_touching_region_pairs(
    segmented_grid: np.ndarray, connectivity=26
) -> np.ndarray:
    """
    Given a 3D grid of segmented regions (integer labels, with 0 as background),
    this function returns an array of unique pairs of region labels that are in contact.
    Works on padded grids without applying periodic boundary conditions.

    Parameters
    ----------
    segmented_grid : np.ndarray
        A 3D numpy array where 0 represents background and positive integers are region labels.
    connectivity : int, optional
        Connectivity to use: 6 (face-adjacent), 18 (face and edge-adjacent), or 26 (fully adjacent).
        Default is 26.

    Returns
    -------
    np.ndarray
        An array of shape (n, 2) where each row is a sorted pair of touching region labels.
    """
    import numpy as np
    # Define offsets based on connectivity
    if connectivity == 6:
        # Only face-adjacent (6-connectivity)
        offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0),
                   (0, 0, -1), (0, -1, 0), (-1, 0, 0)]
    elif connectivity == 18:
        # Face and edge-adjacent (18-connectivity)
        offsets = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == dy == dx == 0) and (dz * dy * dx == 0)
        ]
    else:
        # Full 26-connectivity (default)
        offsets = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == dy == dx == 0)
        ]
    touching_pairs = set()
    # Loop over each neighbor offset
    for dz, dy, dx in offsets:
        # Create shifted arrays with appropriate padding
        if dz > 0:
            z_src, z_dst = slice(0, -1), slice(1, None)
        elif dz < 0:
            z_src, z_dst = slice(1, None), slice(0, -1)
        else:
            z_src, z_dst = slice(None), slice(None)

        if dy > 0:
            y_src, y_dst = slice(0, -1), slice(1, None)
        elif dy < 0:
            y_src, y_dst = slice(1, None), slice(0, -1)
        else:
            y_src, y_dst = slice(None), slice(None)

        if dx > 0:
            x_src, x_dst = slice(0, -1), slice(1, None)
        elif dx < 0:
            x_src, x_dst = slice(1, None), slice(0, -1)
        else:
            x_src, x_dst = slice(None), slice(None)

        # Get the regions from the original grid and the shifted position
        reg1 = segmented_grid[z_src, y_src, x_src]
        reg2 = segmented_grid[z_dst, y_dst, x_dst]

        # Create a mask for valid region pairs (non-zero and different)
        mask = (reg1 != reg2) & (reg1 > 0) & (reg2 > 0)

        if np.any(mask):
            # Get the region labels where they're touching
            labels1 = reg1[mask]
            labels2 = reg2[mask]

            # Add sorted pairs to the set
            for a, b in zip(labels1, labels2):
                touching_pairs.add(tuple(sorted((int(a), int(b)))))
    # Convert set of tuples to numpy array
    return np.array(list(touching_pairs))

def move_maximum_to_peak(
    original_pos,
    padded_grid,
    padding_thickness,
    max_steps=100,
    step_size=0.1,
):
    """
    Move a maximum position to the nearest peak using gradient ascent,
    constrained within the padded grid boundaries.

    Parameters
    ----------
    original_pos : np.ndarray
        Initial position of the maximum in grid coordinates
    padded_grid : np.ndarray
        Padded distance grid
    padding_thickness : int
        Thickness of padding
    max_steps : int, optional
        Maximum number of gradient ascent steps
    step_size : float, optional
        Step size for gradient ascent

    Returns
    -------
    tuple[np.ndarray, float]
        Optimized position and its value
    """
    import numpy as np
    # Initialize current position and value
    current_pos = original_pos.copy()
    current_value = padded_grid[tuple(current_pos.astype(int))]
    # print("starting pos is ", current_pos)
    # print("starting value is ", current_value)
    # Get grid dimensions
    grid_dims = np.array(padded_grid.shape)

    for _ in range(max_steps):
        # Calculate gradient using central differences
        gradients = []
        for dim in range(3):
            pos_plus = current_pos.copy()
            pos_minus = current_pos.copy()

            # Ensure we stay within bounds when calculating gradients
            pos_plus[dim] = min(pos_plus[dim] + 1, grid_dims[dim] - 1)
            pos_minus[dim] = max(pos_minus[dim] - 1, 0)

            val_plus = padded_grid[tuple(pos_plus.astype(int))]
            val_minus = padded_grid[tuple(pos_minus.astype(int))]
            gradients.append((val_plus - val_minus) / 2)

        gradient = np.array(gradients)
        gradient_magnitude = np.linalg.norm(gradient)

        # Stop if gradient is too small
        if gradient_magnitude < 1e-6:
            break

        # Update position with bounds checking
        new_pos = current_pos + step_size * gradient / gradient_magnitude

        # Clip to grid boundaries
        new_pos = np.clip(new_pos, 0, grid_dims - 1)

        # Get value at new position
        new_value = padded_grid[tuple(new_pos.astype(int))]

        # Stop if we're not improving
        if new_value <= current_value:
            break

        current_pos = new_pos
        current_value = new_value
    # print("FInal value is ", current_value)
    # print("FInal pos is ", current_pos)
    return current_pos, current_value

def estimate_h_from_percolation_history(distance_grid):
    """

    Estimate h parameter using PLR and analysis of percolation transition.
    Returns a suitable h value for h-maxima detection.

    """
    import numpy as np
    from scipy.ndimage import label as ndimage_label
    from scipy.ndimage import generate_binary_structure
    from scipy.signal import find_peaks
    from .compute import find_pore_limiting_radius
    # Find PLR using binary search
    plr =find_pore_limiting_radius(distance_grid)
    if plr is None:
        print("No PLR found, using default h")
        return 0.4  # fallback value

    print(f"PLR found: {plr:.2f}")

    # Create thresholds around PLR
    # Reduced number of thresholds
    thresholds = np.linspace(plr * 0.6, plr * 1.2, 12)
    local_constrictions = []

    # Parameters for peak detection
    min_prominence = max(0.2, plr * 0.1)  # Scale prominence with PLR
    min_distance = max(3, int(plr / 0.5))  # Scale distance with PLR

    for threshold in thresholds:
        binary = distance_grid > threshold
        padded = np.pad(binary, 1, mode="wrap")
        labeled, num_features = ndimage_label(
            padded, generate_binary_structure(3, 3))
        labeled = labeled[1:-1, 1:-1, 1:-1]

        # Only process if we have a reasonable number of features
        if num_features > 100:  # Skip if too fragmented
            continue

        # Analyze percolation in all directions
        for axis in range(3):
            face1 = set(np.unique(labeled.take(0, axis=axis)))
            face2 = set(np.unique(labeled.take(-1, axis=axis)))
            common = face1.intersection(face2) - {0}

            for label_id in common:
                component = labeled == label_id
                channel_distances = distance_grid[component]

                # Skip if channel is too small
                if len(channel_distances) < 10:
                    continue

                # Normalize distances for peak detection
                normalized_distances = (channel_distances - channel_distances.min()) / (
                    channel_distances.max() - channel_distances.min()
                )

                peaks, properties = find_peaks(
                    -normalized_distances,
                    prominence=min_prominence,
                    distance=min_distance,
                )

                if len(peaks) > 0:
                    # Filter peaks by relative depth
                    for i, peak_idx in enumerate(peaks):
                        peak_value = channel_distances[peak_idx]
                        prominence = properties["prominences"][i]

                        # Only include significant constrictions
                        if prominence > min_prominence and peak_value < plr:
                            local_constrictions.append(
                                {
                                    "value": peak_value,
                                    "prominence": prominence,
                                    "relative_depth": prominence / peak_value,
                                }
                            )

    if not local_constrictions:
        print("No significant constrictions found")
        h = min(0.2, max(0.1, plr * 0.15))
        return h

    # Convert to array and analyze
    constriction_data = np.array(
        [
            (d["value"], d["prominence"], d["relative_depth"])
            for d in local_constrictions
        ],
        dtype=[("value", float), ("prominence", float),
               ("relative_depth", float)],
    )

    # Filter outliers using relative depth
    q1, q3 = np.percentile(constriction_data["relative_depth"], [25, 75])
    iqr = q3 - q1
    valid_mask = (constriction_data["relative_depth"] >= q1 - 1.5 * iqr) & (
        constriction_data["relative_depth"] <= q3 + 1.5 * iqr
    )
    valid_constrictions = constriction_data[valid_mask]

    if len(valid_constrictions) == 0:
        print("No valid constrictions after filtering")
        h = min(0.2, max(0.1, plr * 0.15))
        return h

    print(f"Found {len(valid_constrictions)} valid constrictions")
    print(
        f"Value range: [{valid_constrictions['value'].min():.2f}, {valid_constrictions['value'].max():.2f}]"
    )

    # Calculate h using relative depth and PLR
    avg_relative_depth = np.mean(valid_constrictions["relative_depth"])
    max_prominence = np.max(valid_constrictions["prominence"])
    min_prominence = np.min(valid_constrictions["prominence"])

    # Adaptive h calculation
    h_from_depth = avg_relative_depth * plr * 0.3
    h_from_prominence = max_prominence * 0.5
    print("h_from_depth is ", h_from_depth)
    print("h from prominence is ", h_from_prominence)
    print("average relative depth is ", avg_relative_depth)
    print("minimum prominence is ", min_prominence)
    # h = min(0.4, max(0.1, min(h_from_depth, h_from_prominence)))
    h = min(0.2, min_prominence)

    print(f"PLR: {plr:.2f}")
    print(f"Average relative depth: {avg_relative_depth:.2f}")
    print(f"Max prominence: {max_prominence:.2f}")
    print(f"Min prominence: {min_prominence:.2f}")
    print(f"Estimated h: {h:.2f}")

    return h

def is_inside_or_close_to_boundary(maximum_indices, box_shape, tolerance):
    """
    Determines if points are inside the box or just outside within tolerance. Typically used for checking the maxima positions, but we can use it with basically any point. To get a healthy tolerance with proper accounting of non orthogonality use the `compute_wall_cells` function to find the number of cells forming the wall region in each direction [a, b, c]

    Parameters:
    -----------
    maximum_indices : list of tuples [(z1, y1, x1), (z2, y2, x2), ...]
        The coordinates of the maxima
    box_shape : tuple (depth, height, width)
        The shape of the original box
    tolerance : tuple (zz, yy, xx)
        The tolerance in each direction

    Returns:
    --------
    locations : list of str
        List of classifications for each point:
        'inside' if the point is inside the box
        'close_outside' if it's just outside within tolerance
        'far_outside' if it's outside and beyond tolerance
    """
    depth, height, width = box_shape
    zz, yy, xx = tolerance

    locations = []

    for z, y, x in maximum_indices:
        # Check if the point is inside the box
        if (0 <= z < depth) and (0 <= y < height) and (0 <= x < width):
            locations.append("inside")

        # Check if it's close outside (within tolerance)
        elif (
            (-zz <= z < depth + zz)
            and (-yy <= y < height + yy)
            and (-xx <= x < width + xx)
        ):
            locations.append("close_outside")

        # Otherwise, it's far outside
        else:
            locations.append("far_outside")

    return locations

def process_wall_maxima(
    all_maxima,
    distance_field,
    padding_thickness,
    wall_cells,
    ase_atoms,
    min_merge_ratio=0.8,
):
    """
    Process wall maxima to identify and merge periodic copies.

    Parameters
    ----------
    all_maxima : np.ndarray
        Array of maxima positions in the original unpadded distance grid
    distance_field : np.ndarray
        Distance field
    padding_thickness : int
        Padding thickness used
    wall_cells : np.ndarray
        Number of cells considered as wall region
    ase_atoms,
        ASE atoms object
    min_merge_ratio : float
        Minimum ratio between smaller and larger maxima values to consider merging (default: 0.8)
    """

    import numpy as np
    from ase.geometry import complete_cell

    # original_shape = np.array(padded_grid.shape) - 2 * padding_thickness
    original_shape = np.array(distance_field.shape)

    # First, move all wall maxima to their nearest peaks
    optimized_positions = []
    optimized_values = []
    wall_indices = []

    # unit cell matrix
    cell = complete_cell(ase_atoms.cell).T

    logger.info("\nDEBUG: Moving maxima to peaks")
    for i, pos in enumerate(all_maxima):
        is_wall = any(
            coord < wall_cells[dim] or coord > (
                original_shape[dim] - wall_cells[dim])
            for dim, coord in enumerate(pos)
        )

        if is_wall:
            opt_pos, opt_val = move_maximum_to_peak(
                pos, distance_field, padding_thickness
            )
            # print( f"Maximum #{i}: Original pos = {pos}, Optimized pos = {opt_pos}, Value = {opt_val}")
            optimized_positions.append(opt_pos)
            optimized_values.append(opt_val)
            wall_indices.append(i)

    maxima_to_remove = []
    # print("\nDEBUG: Checking for nearby maxima using pore size criteria")

    for i, idx1 in enumerate(wall_indices):
        pos1 = optimized_positions[i]
        val1 = optimized_values[i]

        for j, idx2 in enumerate(wall_indices[i + 1:], i + 1):
            pos2 = optimized_positions[j]
            val2 = optimized_values[j]

            # Calculate minimum image distance
            min_dist = get_min_distance(pos1, pos2, cell, original_shape)

            # Calculate the radius ratio (smaller/larger)
            radius_ratio = min(val1, val2) / max(val1, val2)

            # Calculate adaptive merging threshold based on the larger radius
            # Use 2x the larger radius as maximum allowed distance
            max_radius = max(val1, val2)
            merge_distance = 2.0 * max_radius

            # print(f"\nComparing maxima #{idx1} and #{idx2}")
            # print(f"Positions: {pos1} vs {pos2}")
            # print(f"Values (pore radii): {val1:.3f} Å vs {val2:.3f} Å")
            # print(f"Radius ratio: {radius_ratio:.3f}")
            # print(f"Minimum image distance: {min_dist:.2f} Å")
            # print(f"Merge distance threshold: {merge_distance:.2f} Å")
            #
            # Merge criteria:
            # 1. Radii are similar enough (controlled by min_merge_ratio)
            # 2. Distance is less than 2x the larger radius
            if radius_ratio > min_merge_ratio and min_dist < merge_distance:
                # Keep the one with larger value
                if val1 < val2:
                    maxima_to_remove.append(idx1)
                    logger.info(f"Removing maximum #{idx1} (smaller radius)")
                    break
                else:
                    maxima_to_remove.append(idx2)
                    logger.info(f"Removing maximum #{idx2} (smaller radius)")

    return maxima_to_remove
def get_min_distance(pos1, pos2, cell, original_shape):
    """
    Calculate minimum image distance between two points using ASE cell.

    Parameters
    ----------
    pos1, pos2 : np.ndarray
        Positions in grid coordinates
    cell : np.ndarray
        ASE cell matrix
    original_shape : np.ndarray
        Shape of the original grid

    Returns
    -------
    float
        Minimum image distance in Angstroms
    """
    import numpy as np
    # Convert to fractional coordinates
    frac1 = pos1 / original_shape
    frac2 = pos2 / original_shape

    # Get minimum image difference
    dr = frac1 - frac2
    dr = dr - np.round(dr)  # Map to [-0.5, 0.5]

    # Convert to cartesian coordinates
    dr_cart = np.dot(cell, dr.T).T
    return np.linalg.norm(dr_cart)

# don't really need this one, but, let's keep this anyway
def periodic_gaussian_filter(arr, sigma):
    """
    Apply a periodic (wrap-around) Gaussian filter using np.roll.
    
    Parameters:
    - arr: input ndarray (1D, 2D, or 3D)
    - sigma: standard deviation of Gaussian
    
    Returns:
    - Smoothed ndarray with same shape
    """
    import numpy as np
    from itertools import product
    arr = np.asarray(arr)
    ndim = arr.ndim
    radius = int(3 * sigma)  # Gaussian kernel truncation

    # Create grid of all shifts
    shifts = list(product(range(-radius, radius + 1), repeat=ndim))

    # Precompute Gaussian weights
    def gaussian_weight(shift):
        return np.exp(-0.5 * sum((s / sigma) ** 2 for s in shift))

    weights = np.array([gaussian_weight(s) for s in shifts])
    weights /= weights.sum()

    # Apply weighted rolling
    smoothed = np.zeros_like(arr, dtype=np.float64)
    for shift, weight in zip(shifts, weights):
        rolled = arr
        for axis, s in enumerate(shift):
            rolled = np.roll(rolled, shift=s, axis=axis)
        smoothed += weight * rolled

    return smoothed

def find_end_channel_pairs(regions_pbc, regions_nonpbc, size_ratio=0.3):
    import numpy as np
    from scipy.ndimage import binary_dilation
    box_shape = regions_nonpbc.shape
    wall_mask = np.zeros_like(regions_nonpbc, dtype=bool)
    wall_mask[0, :, :] = wall_mask[-1, :, :] = True
    wall_mask[:, 0, :] = wall_mask[:, -1, :] = True
    wall_mask[:, :, 0] = wall_mask[:, :, -1] = True

    wall_labels = np.unique(regions_nonpbc[wall_mask])
    wall_labels = wall_labels[wall_labels != 0]  # Exclude background

    pairs = []

    for label_id in wall_labels:
        mask = (regions_nonpbc == label_id)
        pbc_label = np.unique(regions_pbc[mask])
        pbc_label = pbc_label[pbc_label != 0]
        if len(pbc_label) != 1:
            continue  # ambiguous, skip

        # Find all other regions on the same wall with the same pbc label
        same_wall_mask = (regions_pbc == pbc_label[0]) & wall_mask
        other_labels = np.unique(regions_nonpbc[same_wall_mask])
        other_labels = [l for l in other_labels if l != label_id and l != 0]

        if not other_labels:
            continue  # No other regions of this PBC group on the wall, skip

        for neighbor_label in other_labels:
            neighbor_mask = (regions_nonpbc == neighbor_label) & wall_mask
            # Check adjacency on the wall
            if np.any(binary_dilation(mask) & neighbor_mask):
                size_current = np.sum(mask)
                size_neighbor = np.sum(regions_nonpbc == neighbor_label)
                # print(f"label {label_id} is adjacent to {neighbor_label} with size current {size_current} and size neighbor {size_neighbor}")
                if size_current < size_ratio * size_neighbor:
                    pairs.append((label_id, neighbor_label))
                break  # Only consider the first touching neighbor

    return pairs

def merge_small_regions_pbc(labels, min_size):
    """
    Merge small labeled regions (< min_size voxels) into neighboring larger regions,
    accounting for periodic boundary conditions (PBC).

    Parameters:
        labels (ndarray): 3D numpy array of labeled regions.
        min_size (int): Minimum voxel count; regions smaller than this will be merged.

    Returns:
        ndarray: Modified labels array with small regions merged.
    """

    import numpy as np
    from scipy.ndimage import binary_dilation
    from skimage.measure import regionprops

    labels = labels.copy()
    shape = labels.shape
    props = regionprops(labels)

    # Identify small label IDs
    small_labels = [p.label for p in props if p.area < min_size]

    logger.info(f"Found {len(small_labels)} small regions to merge.")

    # Pad with wrap to handle periodicity
    padded_labels = np.pad(labels, 1, mode='wrap')

    for small_lbl in small_labels:
        # Create small region mask (inside padded array)
        small_mask = (padded_labels[1:-1, 1:-1, 1:-1] == small_lbl)
        small_mask = np.pad(small_mask, 1, mode='wrap')

        # Dilate to find neighbors
        dilated = binary_dilation(small_mask)
        neighbor_mask = dilated & (~small_mask)

        neighbor_labels = np.unique(padded_labels[neighbor_mask])
        neighbor_labels = neighbor_labels[neighbor_labels != small_lbl]
        neighbor_labels = neighbor_labels[neighbor_labels != 0]  # exclude background if 0

        if len(neighbor_labels) == 0:
            continue  # no neighbors found

        # Find neighbor with largest shared boundary
        neighbor_counts = [(lbl, np.sum(padded_labels[neighbor_mask] == lbl)) for lbl in neighbor_labels]
        neighbor_counts.sort(key=lambda x: -x[1])
        largest_neighbor = neighbor_counts[0][0]

        # Reassign small region voxels
        reassignment_mask = (padded_labels == small_lbl)
        padded_labels[reassignment_mask] = largest_neighbor

    # Remove padding
    merged_labels = padded_labels[1:-1, 1:-1, 1:-1]
    return merged_labels

def merge_small_regions(labels, min_size):
    """
    Merge small labeled regions (< min_size voxels) into neighboring larger regions.

    Parameters:
        labels (ndarray): 3D numpy array of labeled regions.
        min_size (int): Minimum voxel count; regions smaller than this will be merged.

    Returns:
        ndarray: Modified labels array with small regions merged.
    """

    import numpy as np
    from scipy.ndimage import binary_dilation
    from skimage.measure import regionprops

    labels = labels.copy()

    props = regionprops(labels)

    # Identify small label IDs
    small_labels = [p.label for p in props if p.area < min_size]

    logger.info(f"Found {len(small_labels)} small regions to merge.")

    for small_lbl in small_labels:
        small_mask = (labels == small_lbl)

        # Dilate to find neighbors
        dilated = binary_dilation(small_mask)
        neighbor_labels = np.unique(labels[dilated & (labels != small_lbl)])
        neighbor_labels = neighbor_labels[neighbor_labels != 0]  # exclude background

        if len(neighbor_labels) == 0:
            # No neighbor found — skip or optionally assign to background
            continue

        # Find the neighbor sharing the largest boundary
        neighbor_counts = [(lbl, np.sum(labels[dilated & (labels == lbl)])) for lbl in neighbor_labels]
        neighbor_counts.sort(key=lambda x: -x[1])
        largest_neighbor = neighbor_counts[0][0]

        # Reassign small region voxels to neighbor
        labels[small_mask] = largest_neighbor

    return labels

def get_perpendicular_voxel_lengths(ase_atoms, spacing, grid_shape):
    import numpy as np
    voxel_volume = ase_atoms.get_volume() / np.prod(grid_shape)
    la, lb, lc, alpha, beta, gamma = ase_atoms.cell.cellpar()
    alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
    eA = [spacing, 0, 0]
    eB = [spacing * np.cos(gamma), spacing * np.sin(gamma), 0]
    eC = [
        spacing * np.cos(beta),
        spacing * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
        voxel_volume / (spacing * spacing * np.sin(gamma)),
    ]
    lx_voxel = voxel_volume / np.linalg.norm(np.cross(eB, eC))
    ly_voxel = voxel_volume / np.linalg.norm(np.cross(eC, eA))
    lz_voxel = voxel_volume / np.linalg.norm(np.cross(eA, eB))
    return lx_voxel, ly_voxel, lz_voxel

def compute_psds_from_points_pocketwise(sample_points_labels, pore_sizes, bins=np.linspace(0, 50, 200)):
    """
    Compute pore size distributions for each unique region/pocket instead of the whole unit cell.

    This function:
    1. Identifies unique region labels from the sample points
    2. Groups sample points by their region labels
    3. Computes separate pore size distributions for each region
    4. Returns a dictionary mapping region labels to their PSDs

    Parameters:
    - sample_points_labels (ndarray): Array of region labels for each sample point
    - pore_sizes (ndarray): Array of pore sizes corresponding to each sample point
    - bins (ndarray): Bin edges for histogram computation

    Returns:
    - pocket_psds (dict): Dictionary where keys are region labels and values are dictionaries
                         containing 'hist' (histogram values) and 'bin_centers' (bin centers)
    """
    # Find all unique region labels (excluding background and invalid regions)
    unique_labels = np.unique(sample_points_labels)
    print(f"Unique region labels found: {unique_labels}")
    unique_labels = unique_labels[unique_labels > 0]  # Remove background (0) and any negative labels

    # Initialize dictionary to store PSDs for each pocket/region
    pocket_psds = {}

    print(f"Found {len(unique_labels)} unique regions/pockets")

    # Compute PSD for each unique region
    for label in unique_labels:
        # Find all sample points that belong to this region
        mask = sample_points_labels == label
        region_pore_sizes = pore_sizes[mask]

        # Skip regions with too few sample points for reliable statistics
        if len(region_pore_sizes) < 10:
            print(f"Skipping region {label} - insufficient sample points ({len(region_pore_sizes)})")
            continue

        # Filter out infinite or invalid pore sizes
        region_pore_sizes = region_pore_sizes[np.isfinite(region_pore_sizes)]

        if len(region_pore_sizes) == 0:
            print(f"Skipping region {label} - no valid pore sizes")
            continue

        # Compute histogram for this region's pore sizes
        hist, bin_edges = np.histogram(region_pore_sizes, bins=bins, density=False)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # Store the PSD data for this region
        pocket_psds[label] = {
            'hist': hist,
            'bin_centers': bin_centers,
            'num_samples': len(region_pore_sizes),
            'mean_pore_size': np.mean(region_pore_sizes),
            'std_pore_size': np.std(region_pore_sizes)
        }

        print(f"Region {label}: {len(region_pore_sizes)} samples, "
              f"mean pore size: {np.mean(region_pore_sizes):.2f} Å, "
              f"std: {np.std(region_pore_sizes):.2f} Å")

    return pocket_psds

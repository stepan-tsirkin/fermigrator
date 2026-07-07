import numpy as np
from .get_band_wavefunction import get_wavefunction_on_kpoints


def get_segments_triangle(energy_grid, shifts, below_EF, gradient=False):
    """Extract Fermi surface segments from one triangulation of the BZ.

    Each triangle defined by `shifts` crosses the Fermi level if exactly 1 or
    exactly 2 of its 3 vertices are below E_F.  The crossing points on the two
    edges that straddle E_F form a line segment; its midpoint and integration
    weight are returned.

    Parameters
    ----------
    energy_grid : ndarray, shape (NK1, NK2)
        Band energies on a uniform k-grid, already shifted so that E_F = 0.
    shifts : list of 3 (int, int) tuples
        Grid-index offsets that define the three vertices of each triangle.
    below_EF : ndarray of bool, shape (NK1, NK2)
        True where energy_grid < 0.
    gradient : bool
        If True, also return the Fermi velocity ∇_k E at each segment midpoint.

    Returns
    -------
    gradient=False : (k_center, segments, weight)
    gradient=True  : (k_center, segments, weight, grad)
        k_center  : ndarray (N, 2), reduced coordinates of segment midpoints
        segments  : ndarray (N, 2, 2), endpoints of each segment
        weight    : ndarray (N,), integration weights ∝ segment length / |∇E|
        grad      : ndarray (2, N), gradient ∇_k E in Cartesian coordinates
    """
    NK1, NK2 = energy_grid.shape

    x = np.linspace(0, 1, NK1 + 1, endpoint=True)
    y = np.linspace(0, 1, NK2 + 1, endpoint=True)

    below_EF_roll = np.array([np.roll(below_EF, (-sh[0], -sh[1]), axis=(0, 1))
                              for sh in shifts])

    one_below_EF = np.where(np.sum(below_EF_roll, axis=0) == 1)
    E_one_below_EF = np.array([energy_grid[(one_below_EF[0] + sh[0]) % NK1,
                                           (one_below_EF[1] + sh[1]) % NK2]
                               for sh in shifts])

    two_below_EF = np.where(np.sum(below_EF_roll, axis=0) == 2)
    E_two_below_EF = np.array([energy_grid[(two_below_EF[0] + sh[0]) % NK1,
                                           (two_below_EF[1] + sh[1]) % NK2]
                               for sh in shifts])
    num_two_below_EF = two_below_EF[0].shape[0]

    # For two_below_EF triangles flip the sign so the algorithm always sees
    # exactly one vertex below zero (the isolated one), keeping the same code path.
    E_triangles = np.concatenate([E_one_below_EF, -E_two_below_EF], axis=1).T
    one_or_two_below_EF = np.concatenate([one_below_EF, two_below_EF], axis=1)
    k_triangles = np.array([np.array([xy[one_or_two_below_EF[i] + sh[i]]
                                      for sh in shifts]).T for xy, i in zip([x, y], range(2))])
    srt = np.argsort(E_triangles, axis=1)

    E_triangles = np.take_along_axis(E_triangles, srt, axis=1)
    k_triangles = np.array([np.take_along_axis(k, srt, axis=1)
                           for k in k_triangles])
    kappa = [(k_triangles[:, :, 0]
              + (-E_triangles[:, 0]) /
              (E_triangles[:, i] - E_triangles[:, 0])[None, :]
              * (k_triangles[:, :, i] - k_triangles[:, :, 0])) for i in (1, 2)]
    k_center = (kappa[0] + kappa[1]) / 2
    k_center = k_center.T
    segments = np.array(kappa).transpose(2, 0, 1)

    weight = 2 * (-E_triangles[:, 0]) / ((E_triangles[:, 1] -
                                          E_triangles[:, 0]) * (E_triangles[:, 2] - E_triangles[:, 0]))
    weight = weight / (2 * NK1 * NK2)
    if not gradient:
        return k_center, weight, None, segments

    def cyclic_sum(arr1, arr2):
        return sum(arr1[:, i] * (arr2[:, (i + 1) % 3] - arr2[:, (i + 2) % 3]) for i in range(3))
    denominator = cyclic_sum(k_triangles[0], k_triangles[1])
    grad = np.array([cyclic_sum(E_triangles, k_triangles[1]),
                     -cyclic_sum(E_triangles, k_triangles[0])]) / denominator[None, :]
    # because we flipped the sign of the energy for two_below_EF, we flip it again
    grad[:, -num_two_below_EF:] *= -1
    return k_center, weight, grad, segments


def get_kpoints_and_weights_FS_2D(energy_grid, reciprocal_lattice_vectors, fermi_level,
                                  gradient=False):
    """Compute Fermi surface k-points and integration weights for one band.

    Tiles the 2D BZ with two complementary triangulations (each covering half
    the unit cell) and calls `get_segments` on both.  The diagonal direction
    of the triangulation is chosen to align with the shorter BZ diagonal,
    which minimises interpolation error for non-orthogonal lattices.

    Parameters
    ----------
    energy_grid : ndarray, shape (NK1, NK2)
        Band energies in eV on a uniform k-grid (reduced coordinates).
    reciprocal_lattice_vectors : ndarray, shape (2, 2)
        Rows are the 2D reciprocal lattice vectors in Cartesian coordinates (Å⁻¹).
    fermi_level : float
        Fermi energy in eV.
    gradient : bool
        If True, also return Fermi velocity ∇_k E for each contour point.

    Returns
    -------
    gradient=False : (kpoints, weights)
    gradient=True  : (kpoints, segments, weights, grad)
        kpoints  : ndarray (N, 2), reduced coordinates
        segments : ndarray (N, 2, 2), segment endpoints
        weights  : ndarray (N,), integration weights
        grad     : ndarray (N, 2), Fermi velocity in Cartesian coordinates (eV·Å)
    """
    energy_grid = energy_grid.copy() - fermi_level

    below_EF = (energy_grid < 0)
    assert reciprocal_lattice_vectors.shape == (2, 2)
    scal_prod = np.dot(
        reciprocal_lattice_vectors[0], reciprocal_lattice_vectors[1])
    # Choose the triangulation diagonal that avoids cutting across the obtuse
    # angle of the BZ parallelogram (minimises triangle aspect ratio).
    if scal_prod >= 0:
        shifts = [[(0, 0), (1, 0), (0, 1)],
                  [(1, 0), (0, 1), (1, 1)]]
    else:
        shifts = [[(0, 0), (1, 0), (1, 1)],
                  [(0, 0), (0, 1), (1, 1)]]

    res_list = [get_segments_triangle(
        energy_grid, shifts=sh, below_EF=below_EF, gradient=gradient) for sh in shifts]

    kpoints = np.vstack([res[0] for res in res_list])
    weights = np.concatenate([res[2] for res in res_list], axis=0)
    segments = np.concatenate([res[1] for res in res_list], axis=0)

    if gradient:
        grad = np.hstack([res[3] for res in res_list])
        grad = np.dot(np.linalg.inv(reciprocal_lattice_vectors), grad).T
    else:
        grad = None
    return kpoints, segments, weights, grad


def get_contours_and_WFs(
    Efermi_list=None,
    Nfermi=101,
    return_dict=True,
    get_wf=True,
    contours_db=None,
    return_empty=False,
    ignore_existing=False,
):
    """Compute Fermi surface contours and Bloch wavefunctions for all bands and Fermi levels.

    For each (band, Fermi level) pair, extracts the 2D Fermi surface contour
    and optionally evaluates the Wannier-interpolated wavefunction U_k at each
    contour k-point.  Results are saved to `contours_db` and optionally returned
    as a dictionary keyed by (band_index, Fermi_level).

    Parameters
    ----------
    Efermi_list : array-like of float, optional
        Fermi energies (eV) to process.  Defaults to `Nfermi` values spanning
        the full energy range of the grid.
    Nfermi : int
        Number of Fermi levels when `Efermi_list` is not provided.
    return_dict : bool
        If True, return a dict {(ib, EF): data}.
    get_wf : bool
        If True, compute and store wavefunctions alongside the contour k-points.
    contours_db : ContourDatabase
        Database used for reading the energy grid and writing results.
    return_empty : bool
        If True, include entries for bands with no Fermi surface at a given EF.
    ignore_existing : bool
        If True, recompute contours even when they already exist in the database.
    """
    energies_grid, rec_lattice = contours_db.get_E_grid()
    dim = rec_lattice.shape[0]
    assert dim in (2, 3), "Only 2D and 3D systems are supported"
    assert rec_lattice.shape == (dim, dim), "Expected square reciprocal lattice matrix"
    assert energies_grid.ndim == dim + 1, f"Expected energy grid with {dim} k-space dimensions plus band dimension"
    system = contours_db.system
    ndim_system = sum(system.periodic)
    assert ndim_system == dim, f"System periodicity {system.periodic} does not match energy grid dimension {dim}"
    if dim == 2:
        get_kpoints_and_weights_FS = get_kpoints_and_weights_FS_2D
    elif dim == 3:
        from .contours3D import get_kpoints_and_weights_FS_3D
        get_kpoints_and_weights_FS = get_kpoints_and_weights_FS_3D
    if Efermi_list is None:
        Efermi_list = np.linspace(
            np.min(energies_grid), np.max(energies_grid), Nfermi)
    contours = {}
    for ib in range(energies_grid.shape[-1]):
        for i, e in enumerate(Efermi_list):
            if contours_db.has_contour(ib, e) and not ignore_existing:
                if return_dict:
                    contours[(ib, e)] = contours_db.get_data(
                        "contour", ib=ib, EF=e)
                continue
            centers, segments, weights, grad = get_kpoints_and_weights_FS(
                energies_grid[..., ib], rec_lattice, e, gradient=True)
            if len(centers) == 0:
                if return_empty:
                    contours[(ib, e)] = {"kpoints": centers, "weights": weights,
                                         "segments": segments,
                                         "grad": grad, "wavefunctions": np.zeros((0, system.num_wann))}
                continue
            dic_loc = {"kpoints": centers, "weights": weights,
                       "grad": grad, "segments": segments}
            if get_wf:
                wavefunctions = get_wavefunction_on_kpoints(
                    system, centers, ib)
                dic_loc["wavefunctions"] = wavefunctions
            if return_dict:
                contours[(ib, e)] = dic_loc
            contours_db.set_data("contour", dic_loc, ib=ib, EF=e)
    if return_dict:
        return contours

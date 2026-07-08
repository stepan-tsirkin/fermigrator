import numpy as np
from .get_band_wavefunction import get_wavefunction_on_kpoints

def get_fermi_surface(
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
            segments, weights, grad = get_kpoints_and_weights_FS_ND(
                energies_grid[..., ib], rec_lattice, e, dim=dim)
            centers = np.mean(segments, axis=1)
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


import numpy as np
# from wannierberri.grid.tetrahedron import weight_tetrahedron

def weights_kappa_2d(k_tetra_srt, E_tetra_srt, n):
    kappa = [(k_tetra_srt[:, :, 0]
              + (-E_tetra_srt[:, 0]) /
              (E_tetra_srt[:, i] - E_tetra_srt[:, 0])[None, :]
              * (k_tetra_srt[:, :, i] - k_tetra_srt[:, :, 0])) for i in (1, 2)]
    k_center = (kappa[0] + kappa[1]) / 2
    k_center = k_center.T
    segments = np.array(kappa).transpose(2, 0, 1)

    weight = 2 * (-E_tetra_srt[:, 0]) / ((E_tetra_srt[:, 1] -
                                          E_tetra_srt[:, 0]) * (E_tetra_srt[:, 2] - E_tetra_srt[:, 0]))
    return segments, weight



def weights_kappa_3d(k_tetra_srt, E_tetra_srt, n):
    e1, e2, e3, e4 = E_tetra_srt[:, 0], E_tetra_srt[:, 1], E_tetra_srt[:, 2], E_tetra_srt[:, 3]
    if n == 1:
        kappa = [(k_tetra_srt[:, :, 0]
                  + -e1 /(E_tetra_srt[:, i] - e1)[None, :]
                      * (k_tetra_srt[:, :, i] - k_tetra_srt[:, :, 0])) 
                            for i in (1, 2, 3)]
        triangles = np.array(kappa).transpose(2, 1, 0)
        weights = 3 * e1 ** 2 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    elif n == 2:
        # case of quadrilateral face, split into two triangles
        kappa = np.array([(k_tetra_srt[:, :, j] + (-E_tetra_srt[:, j]) /
                            (E_tetra_srt[:, i] - E_tetra_srt[:, j])[None, :]
                                * (k_tetra_srt[:, :, i] - k_tetra_srt[:, :, j])) 
                                                    for i in (2, 3) for j in (0, 1)]).transpose(2, 1, 0)
        kappa1 = kappa[:, :, [0, 1, 2]]
        kappa2 = kappa[:, :, [1, 3, 2]]
        w1 = np.linalg.norm(np.cross(kappa1[:, :, 1] - kappa1[:, :, 0], kappa1[:, :, 2] - kappa1[:, :, 0]), axis=1)
        w2 = np.linalg.norm(np.cross(kappa2[:, :, 1] - kappa2[:, :, 0], kappa2[:, :, 2] - kappa2[:, :, 0]), axis=1)
        wsum = w1 + w2
        # k_center = get_center_of_mass_quad(np.array(kappa).transpose(2, 0, 1))
        denom2 = 1. / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
        weights = (
            -2 * e1 * ((e3 - e2) * (e4 - e2)) + (2 * e2 * e4 + e2 ** 2) * (e1 - e3) + (
                e1 * e2 + e2 * e3 + e1 * e3) *
            (e2 - e4)
        ) * denom2
        w1 = w1 * weights / wsum
        w2 = w2 * weights / wsum
        weights = np.concatenate([w1, w2], axis=0)
        triangles = np.concatenate([kappa1, kappa2], axis=0)
    else:
        raise ValueError(f"n must be 1 or 2, got {n}")
    return triangles, weights



def get_faces_tetrahedron_n_below(n, energy_grid, shifts, num_below_EF, dim):
    if n == dim:
        reverse_sign = True
        n_loc = 1
        num_below_EF = dim + 1 - num_below_EF
    else:
        reverse_sign = False
        n_loc = n
    # from now on n can be 1 or 2

    NKi = energy_grid.shape
    assert len(NKi) == dim, f"energy_grid must have {dim} dimensions, got {len(NKi)}"
    xx = [np.linspace(0, 1, nk + 1, endpoint=True) for nk in NKi]
    n_below_EF = np.where(num_below_EF == n_loc)

    # the array has shape (4, N) where N is the number of tetrahedrons with n vertices below E_F
    E_n_below_EF = np.array([energy_grid[
        tuple([ (n_below_EF[i] + sh[i]) % NKi[i] for i in range(len(NKi)) ]) ]
                for sh in shifts])
    assert E_n_below_EF.shape[0] == dim+1
    assert E_n_below_EF.shape[1] == len(n_below_EF[0])
    if reverse_sign:
        E_n_below_EF = -E_n_below_EF

    # ktetra is (dim, N, dim+1)
    k_tetra = np.array([np.array([xyz[n_below_EF[i] + sh[i]]
                                  for sh in shifts]).T for i, xyz in enumerate(xx)])
    # Etetra is (N, dim+1)
    E_tetra = E_n_below_EF.T

    # srt is (N, dim+1) 
    srt = np.argsort(E_tetra, axis=1)
    k_tetra_srt = np.array([np.take_along_axis(k, srt, axis=1) for k in k_tetra])
    E_tetra_srt = np.take_along_axis(E_tetra, srt, axis=1)

    if dim == 3:
        triangles, weights = weights_kappa_3d(k_tetra_srt, E_tetra_srt, n_loc)
        weights = weights / (np.prod(NKi) * 6)
    elif dim == 2:
        triangles, weights = weights_kappa_2d(k_tetra_srt, E_tetra_srt, n_loc)
        weights = weights / (np.prod(NKi) * 2)
    shifts = np.array(shifts, dtype=int) / np.array(NKi)[None, :]
    shifts1 = np.linalg.inv(shifts[:-1] - shifts[-1][None, :])
    E_tetra1 = E_tetra[:, :-1] - E_tetra[:, -1][:, None]
    grad = np.einsum('ij, kj -> ki', shifts1, E_tetra1)
    if reverse_sign:
        grad = -grad
    if n_loc == 2:
        # for the case of quadrilateral faces, we have two triangles per tetrahedron, 
        # which have the same gradient, so we duplicate the gradients to match the number of triangles
        grad = np.concatenate([grad, grad], axis=0)
    return triangles, weights, grad


def get_faces(energy_grid, shifts, below_EF, dim):
    """Extract Fermi surface segments/triangles from one triangulation/tetrahedron of the BZ.

    Each tetrahedron defined by `shifts` crosses the Fermi level if exactly 1, 2 or 3
    of its 4 vertices are below E_F.
    If two vertices are below E_F, the face is a quadrilateral, which we split into two triangles.
    The corners, the center and the integration weight of each triangle are returned,
    as well as the Fermi velocity ∇_k E at each triangle center if `gradient=True`.

    Parameters
    ----------
    energy_grid : ndarray, shape (NK1, NK2, NK3)
        Band energies on a uniform k-grid, already shifted so that E_F = 0.
    shifts : list of 4 (int, int, int) tuples
        Grid-index offsets that define the four vertices of each tetrahedron.
    below_EF : ndarray of bool, shape (NK1, NK2, NK3)
        True where energy_grid < 0.
    dim : int
        Dimensionality of the system (2 or 3).

    Returns
    -------
        weight    : ndarray (N,), integration weights ∝ face area / |∇E|
        segments  : ndarray (N, dim, dim), triangle vertices in reduced coordinates
        grad      : ndarray (3, N), gradient ∇_k E in Cartesian coordinates
        k_center  : ndarray (N, dim), centers of the triangles in reduced coordinates
    """
    # print (f"get_faces_tetrahedron: energy_grid.shape={energy_grid.shape}, shifts={shifts}, below_EF.shape={below_EF.shape}, dim={dim}  ")
    shifts = np.array(shifts, dtype=int)
    below_EF_roll = np.array([np.roll(below_EF, tuple(-sh), axis=tuple(range(dim) ))
                              for sh in shifts])

    num_below_EF = np.sum(below_EF_roll, axis=0)
    result_list = [get_faces_tetrahedron_n_below(n, energy_grid, shifts, num_below_EF, dim=dim)
                     for n in range(1, dim + 1)]
    triangles = np.vstack([result[0] for result in result_list])
    weight = np.concatenate([result[1] for result in result_list], axis=0)
    grad = np.vstack([result[2] for result in result_list])
    return triangles, weight, grad


def get_kpoints_and_weights_FS_ND(energy_grid, reciprocal_lattice_vectors, fermi_level,
                                  dim=3):
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

    Returns
    -------
    gradient=False : (kpoints, weights)
    gradient=True  : (kpoints, segments, weights, grad)
        kpoints  : ndarray (N, 2), reduced coordinates
        segments : ndarray (N, 2, 2), segment endpoints
        weights  : ndarray (N,), integration weights
        grad     : ndarray (N, 2), Fermi velocity in Cartesian coordinates (eV·Å)
    """
    assert dim in (2, 3), f"dim must be 2 or 3, got {dim}"
    energy_grid = energy_grid.copy() - fermi_level

    below_EF = (energy_grid < 0)
    assert reciprocal_lattice_vectors.shape == (dim, dim), f"reciprocal_lattice_vectors must be a {dim}x{dim} array, got shape {reciprocal_lattice_vectors.shape}"

    if dim ==2:
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
    elif dim == 3:
        # The coordinates of the vertices of the tetrahedrons which divide a cube into 6 tetrahedrons
        # first, divide one half of the cube into 3 tetrahedrons
        shifts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1]],
                        [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1]]
                        ], dtype=int)
        # now add the opposite tetrahedrons to cover the other half of the cube
        shifts = np.vstack([shifts, 1 - shifts])

    res_list = [get_faces(energy_grid, shifts=sh, below_EF=below_EF, dim=dim)
                for sh in shifts]
    
    triangles = np.concatenate([res[0] for res in res_list], axis=0)
    weights = np.concatenate([res[1] for res in res_list], axis=0)
    grad = np.vstack([res[2] for res in res_list])
    grad = np.einsum('aj, kj -> ka', np.linalg.inv(reciprocal_lattice_vectors), grad)
    return triangles, weights, grad

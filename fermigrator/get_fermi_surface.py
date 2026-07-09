import numpy as np

# from wannierberri.grid.tetrahedron import weight_tetrahedron


def segments_2d(k_tetra_srt, E_tetra_srt, n):
    """
    Parameters
    ----------
    k_tetra_srt : ndarray, shape (N, 2, 3)
        The k-points of the tetrahedra.
    E_tetra_srt : ndarray, shape (N, 4)
        The energies of the tetrahedra.
    n : int
        The number of states below the Fermi level.

    Returns
    -------
    segments : list of ndarray, shape (N, 2, 3)
        The segment values for each tetrahedron.
    """
    segments = (k_tetra_srt[:, 0, :][:, None, :]
                - E_tetra_srt[:, 0] / (E_tetra_srt[:, 1:] - E_tetra_srt[:, 0]) *
                k_tetra_srt[:, 1:, :] - k_tetra_srt[:, 0, :][:, None, :])

    # weight = 2 * (-E_tetra_srt[:, 0]) / ((E_tetra_srt[:, 1] -
    #                                       E_tetra_srt[:, 0]) * (E_tetra_srt[:, 2] - E_tetra_srt[:, 0]))
    return segments


def triangles_3d(k_tetra_srt, E_tetra_srt, n):
    """
    Parameters
    ----------
    k_tetra_srt : ndarray, shape (N, 4, 3)
        The k-points of the tetrahedra.
    E_tetra_srt : ndarray, shape (N, 4)
        The energies of the tetrahedra.
    n : int
        The number of states below the Fermi level.

    Returns
    -------
    triangles : ndarray, shape (N, 3, 3)
        The triangle values for each tetrahedron.
    """
    e1, e2, e3, e4 = E_tetra_srt[:, 0], E_tetra_srt[:, 1], E_tetra_srt[:, 2], E_tetra_srt[:, 3]
    assert np.all(e1 <= e2) and np.all(e2 <= e3) and np.all(e3 <= e4), "E_tetra_srt must be sorted"
    if n == 1:
        triangles = (k_tetra_srt[:, 0, :][:, None, :] -
                     (e1[:, None] / (E_tetra_srt[:, 1:] - e1[:, None]))[:, :, None]
                     * (k_tetra_srt[:, 1:, :] - k_tetra_srt[:, 0, :][:, None, :]))
        # weights = 3 * e1 ** 2 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    elif n == 2:
        # case of quadrilateral face, split into two triangles
        kappa = np.array([k_tetra_srt[:, j, :] - (E_tetra_srt[:, j] /
                                                  (E_tetra_srt[:, i] - E_tetra_srt[:, j]))[:, None]
                          * (k_tetra_srt[:, i, :] - k_tetra_srt[:, j, :])
                          for i in (2, 3) for j in (0, 1)]).transpose(1, 0, 2)
        basis = kappa[:, 1:, :] - kappa[:, 0, :][:, None, :]
        cross1 = np.cross(basis[:, 0, :], basis[:, 1, :])
        cross2 = np.cross(basis[:, 1, :], basis[:, 2, :])
        # Check if the points of the quadrilateral are ordered clockwise or counterclockwise
        revert = np.where(np.einsum("ij,ij->i", cross1, cross2) < 0)[0]
        kappa[revert, 2, :], kappa[revert, 3, :] = kappa[revert, 3, :], kappa[revert, 2, :]
        basis = kappa[:, 1:, :] - kappa[:, 0, :][:, None, :]
        kappa1 = kappa[:, [0, 1, 2], :]
        kappa2 = kappa[:, [0, 2, 3], :]
        triangles = np.concatenate([kappa1, kappa2], axis=0)
        # assert np.all(abs(np.linalg.det(basis)) < 1e-8), f"The cuadrilateral face is not planar, its  a bug, basis={basis}, det={np.linalg.det(basis)} kappa={kappa}"
        # w1 = np.linalg.norm(np.cross(basis[:, 0, :], basis[:, 1, :]), axis=1)
        # w2 = np.linalg.norm(np.cross(basis[:, 1, :], basis[:, 2, :]), axis=1)
        # wsum = w1 + w2
        # k_center = get_center_of_mass_quad(np.array(kappa).transpose(2, 0, 1))
        # denom2 = 1. / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
        # weights_quad = (
        #     -2 * e1 * (e3 - e2) * (e4 - e2) + (2 * e2 * e4 + e2 ** 2) * (e1 - e3)
        #           + (e1 * e2 + e2 * e3 + e1 * e3) * (e2 - e4)
        # ) * denom2

        # w1 = (w1 / wsum) * weights_quad
        # w2 = (w2 / wsum) * weights_quad
        # weights = np.zeros(len(w1) *2)
        # triangles = np.zeros((len(w1) * 2, 3, 3))
        # weights[0::2] = w1
        # weights[1::2] = w2
        # triangles[0::2, :, :] = kappa1
        # triangles[1::2, :, :] = kappa2
        # weights = np.concatenate([w1, w2], axis=0)
        # triangles = np.concatenate([kappa1, kappa2], axis=0)
    else:
        raise ValueError(f"n must be 1 or 2, got {n}")
    return triangles


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
        tuple([(n_below_EF[i] + sh[i]) % NKi[i] for i in range(len(NKi))])]
        for sh in shifts])
    assert E_n_below_EF.shape[0] == dim + 1
    assert E_n_below_EF.shape[1] == len(n_below_EF[0])
    if reverse_sign:
        E_n_below_EF = -E_n_below_EF

    # ktetra is (N, dim+1, dim)
    k_tetra = np.array([np.array([xyz[n_below_EF[i] + sh[i]]
                                  for sh in shifts]).T for i, xyz in enumerate(xx)]).transpose(1, 2, 0)
    # Etetra is (N, dim+1)
    E_tetra = E_n_below_EF.T

    # srt is (N, dim+1)
    srt = np.argsort(E_tetra, axis=1)
    k_tetra_srt = np.take_along_axis(k_tetra, srt[:, :, None], axis=1)
    E_tetra_srt = np.take_along_axis(E_tetra, srt, axis=1)

    if dim == 3:
        triangles = triangles_3d(k_tetra_srt, E_tetra_srt, n_loc)
    elif dim == 2:
        triangles = segments_2d(k_tetra_srt, E_tetra_srt, n_loc)
    shifts = np.array(shifts, dtype=int) / np.array(NKi)[None, :]
    shifts1 = np.linalg.inv(shifts[1:] - shifts[0][None, :])
    E_tetra1 = E_tetra[:, 1:] - E_tetra[:, 0][:, None]
    gradient_red = np.einsum('ij, kj -> ki', shifts1, E_tetra1)
    if reverse_sign:
        gradient_red = -gradient_red
    if n_loc == 2:
        # for the case of quadrilateral faces, we have two triangles per tetrahedron,
        # which have the same gradient, so we duplicate the gradients to match the number of triangles
        gradient_red = np.concatenate([gradient_red, gradient_red], axis=0)
    n = len(triangles)
    if n > 0:
        basis = triangles[:, 1:, :] - triangles[:, 0, :][:, None, :]
        revert = np.linalg.det(np.concatenate([basis, gradient_red[:, None, :]], axis=1)) < 0
        triangles[revert, -1, :], triangles[revert, -2, :] = triangles[revert, -2, :], triangles[revert, -1, :]
    return triangles, gradient_red


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
        triangles  : ndarray (N, dim, dim), triangle vertices in reduced coordinates
        gradient_abs : ndarray (N,), absolute values of gradients ∇_k E in Cartesian coordinates
        k_center  : ndarray (N, dim), centers of the triangles in reduced coordinates
    """
    # print (f"get_faces_tetrahedron: energy_grid.shape={energy_grid.shape}, shifts={shifts}, below_EF.shape={below_EF.shape}, dim={dim}  ")
    shifts = np.array(shifts, dtype=int)
    below_EF_roll = np.array([np.roll(below_EF, tuple(-sh), axis=tuple(range(dim)))
                              for sh in shifts])

    num_below_EF = np.sum(below_EF_roll, axis=0)
    result_list = [get_faces_tetrahedron_n_below(n, energy_grid, shifts, num_below_EF, dim=dim)
                   for n in range(1, dim + 1)]
    triangles = np.vstack([result[0] for result in result_list])
    gradient_red = np.concatenate([result[1] for result in result_list], axis=0)
    return triangles, gradient_red


# def get_kpoints_and_weights_FS_ND(energy_grid, reciprocal_lattice_vectors, fermi_level,
#                                   dim=3):
#     """Compute Fermi surface k-points and integration weights for one band.

#     Tiles the 2D BZ with two complementary triangulations (each covering half
#     the unit cell) and calls `get_segments` on both.  The diagonal direction
#     of the triangulation is chosen to align with the shorter BZ diagonal,
#     which minimises interpolation error for non-orthogonal lattices.

#     Parameters
#     ----------
#     energy_grid : ndarray, shape (NK1, NK2)
#         Band energies in eV on a uniform k-grid (reduced coordinates).
#     reciprocal_lattice_vectors : ndarray, shape (2, 2)
#         Rows are the 2D reciprocal lattice vectors in Cartesian coordinates (Å⁻¹).
#     fermi_level : float
#         Fermi energy in eV.

#     Returns
#     -------
#         kpoints  : ndarray (N, 2), reduced coordinates
#         triangles : ndarray (N, 2, 2), segment endpoints
#         weights  : ndarray (N,), integration weights
#         grad     : ndarray (N, 2), Fermi velocity in Cartesian coordinates (eV·Å)
#     """
#     assert dim in (2, 3), f"dim must be 2 or 3, got {dim}"
#     energy_grid = energy_grid.copy() - fermi_level

#     below_EF = (energy_grid < 0)
#     assert reciprocal_lattice_vectors.shape == (dim, dim), f"reciprocal_lattice_vectors must be a {dim}x{dim} array, got shape {reciprocal_lattice_vectors.shape}"

#     if dim == 2:
#         scal_prod = np.dot(
#             reciprocal_lattice_vectors[0], reciprocal_lattice_vectors[1])
#         # Choose the triangulation diagonal that avoids cutting across the obtuse
#         # angle of the BZ parallelogram (minimises triangle aspect ratio).
#         if scal_prod >= 0:
#             shifts = [[(0, 0), (1, 0), (0, 1)],
#                       [(1, 0), (0, 1), (1, 1)]]
#         else:
#             shifts = [[(0, 0), (1, 0), (1, 1)],
#                       [(0, 0), (0, 1), (1, 1)]]
#     elif dim == 3:
#         # The coordinates of the vertices of the tetrahedrons which divide a cube into 6 tetrahedrons
#         # first, divide one half of the cube into 3 tetrahedrons
#         shifts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
#                            [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1]],
#                            [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1]]
#                            ], dtype=int)
#         # now add the opposite tetrahedrons to cover the other half of the cube
#         shifts = np.vstack([shifts, 1 - shifts])

#     res_list = [get_faces(energy_grid, shifts=sh, below_EF=below_EF, dim=dim) for sh in shifts]

#     triangles = np.concatenate([res[0] for res in res_list], axis=0)
#     weights = np.concatenate([res[1] for res in res_list], axis=0)
#     grad = np.vstack([res[2] for res in res_list])
#     grad = np.einsum('aj, kj -> ka', np.linalg.inv(reciprocal_lattice_vectors), grad)
#     return triangles, weights, grad


def get_shifts_2D(reciprocal_lattice_vectors):
    scal_prod = np.dot(
        reciprocal_lattice_vectors[0], reciprocal_lattice_vectors[1])
    # Choose the triangulation diagonal that avoids cutting across the obtuse
    # angle of the BZ parallelogram (minimises triangle aspect ratio).
    if scal_prod >= 0:
        return [[(0, 0), (1, 0), (0, 1)],
                [(1, 0), (0, 1), (1, 1)]]
    else:
        return [[(0, 0), (1, 0), (1, 1)],
                [(0, 0), (0, 1), (1, 1)]]


def get_shifts_3D():
    # The coordinates of the vertices of the tetrahedrons which divide a cube into 6 tetrahedrons
    # first, divide one half of the cube into 3 tetrahedrons
    shifts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1]],
                       [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1]]
                       ], dtype=int)
    # now add the opposite tetrahedrons to cover the other half of the cube
    return np.vstack([shifts, 1 - shifts])

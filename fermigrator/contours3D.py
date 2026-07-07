import numpy as np
# from wannierberri.grid.tetrahedron import weight_tetrahedron


def get_faces_tetrafedron_n_below(n, energy_grid, shifts, num_below_EF, gradient=True):
    if n == 3:
        reverse_sign = True
        n_loc = 1
        num_below_EF = 4 - num_below_EF
    else:
        reverse_sign = False
        n_loc = n
    # from now on n can be 1 or 2

    NK1, NK2, NK3 = energy_grid.shape
    x = np.linspace(0, 1, NK1 + 1, endpoint=True)
    y = np.linspace(0, 1, NK2 + 1, endpoint=True)
    z = np.linspace(0, 1, NK3 + 1, endpoint=True)
    n_below_EF = np.where(num_below_EF == n_loc)

    # the array has shape (4, N) where N is the number of tetrahedrons with n vertices below E_F
    E_n_below_EF = np.array([energy_grid[
        (n_below_EF[0] + sh[0]) % NK1,
        (n_below_EF[1] + sh[1]) % NK2,
        (n_below_EF[2] + sh[2]) % NK3]
        for sh in shifts])
    assert E_n_below_EF.shape[0] == 4
    assert E_n_below_EF.shape[1] == len(n_below_EF[0])
    if reverse_sign:
        E_n_below_EF = -E_n_below_EF

    # ktetra is (3, N, 4)
    k_tetra = np.array([np.array([xyz[n_below_EF[i] + sh[i]]
                                  for sh in shifts]).T for xyz, i in zip([x, y, z], range(3))])
    # Etetra is (N, 4)
    E_tetra = E_n_below_EF.T

    srt = np.argsort(E_tetra, axis=1)
    k_tetra_srt = np.array([np.take_along_axis(k, srt, axis=1) for k in k_tetra])
    E_tetra_srt = np.take_along_axis(E_tetra, srt, axis=1)

    e1, e2, e3, e4 = E_tetra_srt[:, 0], E_tetra_srt[:, 1], E_tetra_srt[:, 2], E_tetra_srt[:, 3]

    if n_loc == 1:
        kappa = [(k_tetra_srt[:, :, 0]
                  + (-E_tetra_srt[:, 0]) /
                  (E_tetra_srt[:, i] - E_tetra_srt[:, 0])[None, :]
                  * (k_tetra_srt[:, :, i] - k_tetra_srt[:, :, 0])) for i in (1, 2, 3)]
        k_center = np.mean(kappa, axis=0).T
        c11 = 3 * e1 ** 2 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
        weights = c11
    elif n_loc == 2:
        kappa = [(k_tetra_srt[:, :, j]
                  + (-E_tetra_srt[:, j]) /
                  (E_tetra_srt[:, i] - E_tetra_srt[:, j])[None, :]
                  * (k_tetra_srt[:, :, i] - k_tetra_srt[:, :, j])) for i in (2, 3) for j in (0, 1)]
        k_center = get_center_of_mass_quad(np.array(kappa).transpose(2, 0, 1))
        denom2 = 1. / ((e3 - e1) * (e4 - e1) * (e3 - e2) * (e4 - e2))
        weights = (
            -2 * e1 * ((e3 - e2) * (e4 - e2)) + (2 * e2 * e4 + e2 ** 2) * (e1 - e3) + (
                e1 * e2 + e2 * e3 + e1 * e3) *
            (e2 - e4)
        ) * denom2
    else:
        raise ValueError(f"n must be 1 or 2, got {n_loc}")
    weights = weights / (NK1 * NK2 * NK3 * 6)
    if gradient:
        shifts = np.array(shifts, dtype=int) / np.array([NK1, NK2, NK3])[None, :]
        shifts1 = np.linalg.inv(shifts[:-1] - shifts[-1][None, :])
        E_tetra1 = E_tetra[:, :-1] - E_tetra[:, -1][:, None]
        grad = np.einsum('ij, kj -> ki', shifts1, E_tetra1)
        if reverse_sign:
            grad = -grad
    else:
        grad = None
    return k_center, weights, grad


def get_faces_tetrahedron(energy_grid, shifts, below_EF, gradient=False):
    """Extract Fermi surface segments from one triangulation of the BZ.

    Each tetrahedron defined by `shifts` crosses the Fermi level if exactly 1, 2 or 3
    of its 4 vertices are below E_F.
    If two vertices are below E_F, the face is a quadrilateral, which we split into two triangles.
    The corners, the center and the integration weight of each triangle are returned,
    as well as the Fermi velocity ∇_k E at each triangle center if `gradient=True`.

    its midpoint and integration
    weight are returned.

    Parameters
    ----------
    energy_grid : ndarray, shape (NK1, NK2, NK3)
        Band energies on a uniform k-grid, already shifted so that E_F = 0.
    shifts : list of 4 (int, int, int) tuples
        Grid-index offsets that define the four vertices of each tetrahedron.
    below_EF : ndarray of bool, shape (NK1, NK2, NK3)
        True where energy_grid < 0.
    gradient : bool
        If True, also return the Fermi velocity ∇_k E at each face midpoint.

    Returns
    -------
        weight    : ndarray (N,), integration weights ∝ face area / |∇E|
        grad      : ndarray (3, N), gradient ∇_k E in Cartesian coordinates
    """
    shifts = np.array(shifts, dtype=int)
    below_EF_roll = np.array([np.roll(below_EF, (-sh[0], -sh[1], -sh[2]), axis=(0, 1, 2))
                              for sh in shifts])

    num_below_EF = np.sum(below_EF_roll, axis=0)
    k_center_list = []
    weight_list = []
    grad_list = []
    for n in (1, 2, 3):
        kc, w, grad = get_faces_tetrafedron_n_below(n, energy_grid, shifts, num_below_EF, gradient=gradient)
        k_center_list.append(kc)
        weight_list.append(w)
        grad_list.append(grad)
    k_center = np.vstack(k_center_list)
    weight = np.concatenate(weight_list, axis=0)
    if gradient:
        grad = np.vstack(grad_list)
    return k_center, weight, grad


def get_center_of_mass_quad(k_quad):
    # k_quad is (N, 4, 3)
    triangles_1 = k_quad[:, [0, 1, 2]]
    triangles_2 = k_quad[:, [0, 2, 3]]
    k_center_1 = np.mean(triangles_1, axis=1)
    k_center_2 = np.mean(triangles_2, axis=1)
    area_1 = 0.5 * np.linalg.norm(np.cross(triangles_1[:, 1] - triangles_1[:, 0],
                                           triangles_1[:, 2] - triangles_1[:, 0]), axis=1)
    area_2 = 0.5 * np.linalg.norm(np.cross(triangles_2[:, 1] - triangles_2[:, 0],
                                           triangles_2[:, 2] - triangles_2[:, 0]), axis=1)
    k_center = (area_1[:, None] * k_center_1 + area_2[:, None] * k_center_2) / (area_1 + area_2)[:, None]
    return k_center


def get_kpoints_and_weights_FS_3D(energy_grid, reciprocal_lattice_vectors, fermi_level,
                                  gradient=True):
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
    assert reciprocal_lattice_vectors.shape == (3, 3)
    # The coordinates of the vertices of the tetrahedrons which divide a cube into 6 tetrahedrons

    # first, divide one half of the cube into 3 tetrahedrons
    shifts = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1]],
                       [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1]]
                       ], dtype=int)
    # now add the opposite tetrahedrons to cover the other half of the cube
    shifts = np.vstack([shifts, 1 - shifts])

    res_list = [get_faces_tetrahedron(energy_grid, shifts=sh, below_EF=below_EF, gradient=gradient)
                for sh in shifts]
    kpoints = np.vstack([res[0] for res in res_list])
    weights = np.concatenate([res[1] for res in res_list], axis=0)
    if gradient:
        grad = np.vstack([res[2] for res in res_list])
        # grad = np.dot(np.linalg.inv(reciprocal_lattice_vectors), grad.T).T
        grad = np.einsum('aj, kj -> ka', np.linalg.inv(reciprocal_lattice_vectors), grad)
    else:
        grad = None
    return kpoints, None, weights, grad

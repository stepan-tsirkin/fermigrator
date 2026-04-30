import numpy as np
from .get_band_wavefunction import get_wavefunction_on_kpoints


def get_segments(energy_grid, shifts, below_EF, gradient=False):
    NK1, NK2 = energy_grid.shape

    x = np.linspace(0, 1, NK1+1, endpoint=True)
    y = np.linspace(0, 1, NK2+1, endpoint=True)

    below_EF_roll = np.array([np.roll(below_EF, (-sh[0], -sh[1]), axis=(0, 1))
                              for sh in shifts])
    one_below_EF = np.where(np.sum(below_EF_roll, axis=0) == 1)
    E_one_below_EF = np.array([energy_grid[(one_below_EF[0]+sh[0]) % NK1,
                                           (one_below_EF[1]+sh[1]) % NK2]
                               for sh in shifts])

    two_below_EF = np.where(np.sum(below_EF_roll, axis=0) == 2)
    E_two_below_EF = np.array([energy_grid[(two_below_EF[0]+sh[0]) % NK1,
                                           (two_below_EF[1]+sh[1]) % NK2]
                               for sh in shifts])
    num_two_below_EF = two_below_EF[0].shape[0]

    # take a minus, because we do not care the sign, but want one
    # energy below EF, and the other above EF
    E_triangles = np.concatenate([E_one_below_EF, -E_two_below_EF], axis=1).T
    one_or_two_below_EF = np.concatenate([one_below_EF, two_below_EF], axis=1)
    k_triangles = np.array([np.array([xy[one_or_two_below_EF[i] + sh[i]]
                                      for sh in shifts]).T for xy, i in zip([x, y], range(2))])
    srt = np.argsort(E_triangles, axis=1)

    E_triangles = np.take_along_axis(E_triangles, srt, axis=1)
    k_triangles = np.array([np.take_along_axis(k, srt, axis=1)
                           for k in k_triangles])
    kappa = [(k_triangles[:, :, 0] +
              (-E_triangles[:, 0]) /
              (E_triangles[:, i]-E_triangles[:, 0])[None, :]
              * (k_triangles[:, :, i]-k_triangles[:, :, 0])) for i in (1, 2)]
    k_center = (kappa[0] + kappa[1]) / 2
    k_center = k_center.T
    segments = np.array(kappa).transpose(2, 0, 1)

    weight = 2 * (-E_triangles[:, 0]) / ((E_triangles[:, 1] -
                                          E_triangles[:, 0]) * (E_triangles[:, 2]-E_triangles[:, 0]))
    weight = weight / (2*NK1*NK2)
    if not gradient:
        return k_center, weight

    def cyclic_sum(arr1, arr2):
        return sum(arr1[:, i] * (arr2[:, (i+1) % 3] - arr2[:, (i+2) % 3]) for i in range(3))
    denominator = cyclic_sum(k_triangles[0], k_triangles[1])
    grad = np.array([cyclic_sum(E_triangles, k_triangles[1]),
                     -cyclic_sum(E_triangles, k_triangles[0])]) / denominator[None, :]
    # because we flipped the sign of the energy for two_below_EF, we flip it again
    grad[:, -num_two_below_EF:] *= -1
    return k_center, segments, weight, grad


def get_kpoints_and_weights_FS(energy_grid, reciprocal_lattice_vectors, fermi_level,
                               gradient=False):
    energy_grid = energy_grid.copy()-fermi_level

    below_EF = (energy_grid < 0)
    assert reciprocal_lattice_vectors.shape == (2, 2)
    scal_prod = np.dot(
        reciprocal_lattice_vectors[0], reciprocal_lattice_vectors[1])
    if scal_prod >= 0:
        shifts = [[(0, 0), (1, 0), (0, 1)],
                  [(1, 0), (0, 1), (1, 1)]]
    else:
        shifts = [[(0, 0), (1, 0), (1, 1)],
                  [(0, 0), (0, 1), (1, 1)]]

    res1 = get_segments(
        energy_grid, shifts=shifts[0], below_EF=below_EF, gradient=gradient)
    res2 = get_segments(
        energy_grid, shifts=shifts[1], below_EF=below_EF, gradient=gradient)
    kpoints = np.vstack([res1[0], res2[0]])
    if not gradient:
        weights = np.concatenate([res1[1], res2[1]], axis=0)
        return kpoints, weights
    weights = np.concatenate([res1[2], res2[2]], axis=0)
    segments = np.concatenate([res1[1], res2[1]], axis=0)
    grad = np.hstack((res1[3], res2[3]))
    grad = np.dot(np.linalg.inv(reciprocal_lattice_vectors), grad)
    return kpoints, segments, weights, grad.T


def get_contours_and_WFs(
    Efermi_list=None,
    Nfermi=101,
    return_dict=True,
    get_wf=True,
    contours_db=None,
    return_empty=False,
    ignore_existing=False,
):
    energies_grid, rec_lattice = contours_db.get_E_grid()
    system = contours_db.system
    if Efermi_list is None:
        Efermi_list = np.linspace(
            np.min(energies_grid), np.max(energies_grid), Nfermi)
    contours = {}
    for ib in range(energies_grid.shape[2]):
        for i, e in enumerate(Efermi_list):
            if contours_db.has_contour(ib, e) and not ignore_existing:
                if return_dict:
                    contours[(ib, e)] = contours_db.get_data(
                        "contour", ib=ib, EF=e)
                continue
            centers, segments, weights, grad = get_kpoints_and_weights_FS(
                energies_grid[:, :, ib], rec_lattice, e, gradient=True)
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

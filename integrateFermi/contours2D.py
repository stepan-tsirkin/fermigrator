import os
import numpy as np
from wannierberri.grid import Grid
from wannierberri.calculators.tabulate import TabulatorAll
from wannierberri.run import run
from .get_band_wavefunction import get_wavefunction_on_kpoints


def get_segments(energy_grid, shifts, below_EF, gradient=False):
    NK1, NK2 = energy_grid.shape

    x = np.linspace(0, 1, NK1+1, endpoint=True)
    y = np.linspace(0, 1, NK2+1, endpoint=True)
    
    
    below_EF_roll = np.array([np.roll(below_EF, (-sh[0], -sh[1]), axis=(0,1)) 
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
    k_triangles = np.array([np.array([xy[one_or_two_below_EF[i]+ sh[i]]
                                    for sh in shifts]).T for xy, i in zip([x, y], range(2))])
    srt = np.argsort(E_triangles, axis=1)

    E_triangles = np.take_along_axis(E_triangles, srt, axis=1)
    k_triangles = np.array([np.take_along_axis(k, srt, axis=1) for k in k_triangles])
    kappa = [ (k_triangles[:,:, 0] + 
              (-E_triangles[:,0])/(E_triangles[:,i]-E_triangles[:,0])[None,:]
              * (k_triangles[:,:,i]-k_triangles[:,:,0]) ) for i in (1,2)]
    k_center = (kappa[0] + kappa[1]) / 2
    k_center = k_center.T


    weight = 2 *(-E_triangles[:,0]) /(( E_triangles[:,1]-E_triangles[:,0]) * (E_triangles[:,2]-E_triangles[:,0]))
    weight = weight / (2*NK1*NK2)
    if not gradient:
        return k_center, weight
    print ("k_center.shape=", k_center.shape)
    print ("weight.shape=", weight.shape)
    def cyclic_sum(arr1, arr2):
        return sum(arr1[:,i] * (arr2[:,(i+1)%3] - arr2[:,(i+2)%3]) for i in range(3))
    denominator = cyclic_sum(k_triangles[0], k_triangles[1])
    grad = np.array([cyclic_sum(E_triangles, k_triangles[1]) ,
                     -cyclic_sum(E_triangles, k_triangles[0]) ]) / denominator[None,:]
    # because we flipped the sign of the energy for two_below_EF, we flip it again
    grad[:, -num_two_below_EF:] *= -1
    print ("grad.shape=", grad.shape)
    return k_center, weight, grad



def get_kpoints_and_weights_FS(energy_grid, reciprocal_lattice_vectors, fermi_level,
                               gradient=False):
    energy_grid = energy_grid.copy()-fermi_level

    below_EF = (energy_grid < 0)
    assert reciprocal_lattice_vectors.shape == (2,2)
    scal_prod = np.dot(reciprocal_lattice_vectors[0], reciprocal_lattice_vectors[1])
    if scal_prod>=0:
        shifts = [[(0,0), (1,0), (0,1)], 
                  [(1,0), (0,1), (1,1)]]
    else:
        shifts = [ [(0,0), (1,0), (1,1)],
                     [(0,0), (0,1), (1,1)]]

    res1 = get_segments(energy_grid, shifts=shifts[0], below_EF=below_EF, gradient=gradient)
    res2 = get_segments(energy_grid, shifts=shifts[1], below_EF=below_EF, gradient=gradient)
    kpoints = np.vstack([res1[0], res2[0]])
    weights = np.concatenate([res1[1], res2[1]], axis=0)
    if not gradient:
        return kpoints, weights
    grad = np.hstack((res1[2], res2[2]))
    grad = np.dot(np.linalg.inv(reciprocal_lattice_vectors), grad)
    return kpoints, weights, grad.T



def get_contours_and_WFs(system,
                         Efermi_list=None,
                         Nfermi=101,
                         return_dict=True,
                         grid=None,
                         recalculate_E_if_exists=False,
                         save_dir=None,
                         return_empty=False
                         ):
    assert np.all(system.periodic == (True, True, False)), "system should be 2D periodic"
    energies_grid = None
    rec_lattice = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if not recalculate_E_if_exists:
        if save_dir is not None:
            if os.path.exists(os.path.join(save_dir, "energies_grid.npz")):
                f = np.load(os.path.join(save_dir, "energies_grid.npz"))
                energies_grid = f["energies_grid"]
                rec_lattice = f["rec_lattice"]
                print("loaded energies_grid and rec_lattice from file")
    if energies_grid is None:
        assert grid is not None, "grid should be provided if energies_grid is not loaded from file"
        if isinstance (grid, int):
            grid = Grid(system, NK=(grid, grid, 1))
        elif isinstance(grid, tuple) or isinstance(grid, list) or isinstance(grid, np.ndarray):
            assert len(grid) == 2, "grid should have length 2 for 2D system"
            grid = Grid(system, NK=(tuple(grid) + (1,)))
        elif isinstance(grid, dict):
            grid = Grid(system, **grid)
        else:
            raise ValueError("grid should be either a tuple/list/ndarray of NK, or a dict of parameters for Grid")
        assert grid.dense[2] == 1, "grid should be 2D"
        calculators = {"tab": TabulatorAll(tabulators={})}
        results = run(system, calculators=calculators, grid=grid)
        energies_grid = results.results["tab"].results["Energy"].data.reshape(tuple(grid.dense[:2]) + (-1,))
        rec_lattice = system.recip_lattice[:2, :2]
        if save_dir is not None:
            np.savez(os.path.join(save_dir, "energies_grid.npz"), energies_grid=energies_grid, rec_lattice=rec_lattice)
            print("saved energies_grid and rec_lattice to file")

    if Efermi_list is None:
        Efermi_list = np.linspace(np.min(energies_grid), np.max(energies_grid), Nfermi)

    contours = {}
    for ib in range(energies_grid.shape[2]):
        for i, e in enumerate(Efermi_list):
            kpoints, weights, grad = get_kpoints_and_weights_FS(energies_grid[:,:,ib], rec_lattice, e, gradient=True)
            if len(kpoints) == 0:
                if return_empty:
                    contours[(ib, e)] = {"kpoints": kpoints, "weights": weights, "grad": grad, "wavefunctions": np.zeros((0, system.num_wann))}
                continue
            wavefunctions = get_wavefunction_on_kpoints(system, kpoints, ib)
            dic_loc = {"kpoints": kpoints, "weights": weights, "grad": grad, "wavefunctions": wavefunctions}
            if return_dict:
                contours[(ib, e)] = dic_loc
            if save_dir is not None:
                np.savez(os.path.join(save_dir, f"contour_ib{ib}_EF={np.round(e,5):.5f}.npz"), **dic_loc)
    if return_dict:
        return contours
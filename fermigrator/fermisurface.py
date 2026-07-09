

import numpy as np
from propcache import cached_property

from fermigrator.get_fermi_surface import get_faces, get_shifts_2D, get_shifts_3D
from wannierberri.utility import cached_einsum


class FermiSurface:

    def __init__(self,
                 energy,
                 recip_lattice,
                 triangles,
                 gradient_abs,
                 wavefunctions_center=None,
                 iband=None
                 ):
        self.energy = energy
        self.recip_lattice = recip_lattice
        assert self.recip_volume > 0, f"reciprocal lattice volume must be positive, got {self.recip_volume}"
        self.triangles = triangles
        self.gradient_abs = gradient_abs
        self.wavefunctions_center = wavefunctions_center
        self.iband = iband

    @classmethod
    def from_file(cls, filename):
        data = dict(np.load(filename))
        import os
        for part in os.path.basename(filename).strip(".npz").split("_"):
            if part.startswith("EF"):
                data["energy"] = float(part.split("=")[-1])
                break
        return cls(**data)

    @cached_property
    def recip_volume(self):
        return np.linalg.det(self.recip_lattice)

    def as_dict(self):
        keys = ["energy", "recip_lattice", "triangles", "gradient_abs", "wavefunctions_center", "iband"]
        dic = {key: getattr(self, key) for key in keys if getattr(self, key) is not None}
        return dic

    def to_npz(self, filename):
        np.savez(filename, **self.as_dict())

    @property
    def dim(self):
        return self.recip_lattice.shape[0]

    @cached_property
    def centers_red(self):
        return np.mean(self.triangles, axis=(1))

    def get_wavefunction(self, iband):
        if self.wavefunctions is None:
            raise ValueError("Wavefunctions are not available for this Fermi surface.")
        return self.wavefunctions[:, :, iband]

    @property
    def k_centers(self):
        return np.mean(self.triangles, axis=1)

    @property
    def is_empty(self):
        return self.num_triangles == 0

    @property
    def num_triangles(self):
        return len(self.triangles)

    @property
    def triangles_cart(self):
        return np.einsum('ia, kji -> kja', self.recip_lattice, self.triangles)

    @property
    def basis_vectors_cart(self):
        return self.triangles_cart[:, 1:, :] - self.triangles_cart[:, 0:1, :]

    @property
    def perpendicular(self):
        if self.dim == 2:
            perp = np.cross([0, 0, 1], self.basis_vectors_cart[:, 0, :])
        elif self.dim == 3:
            perp = np.cross(self.basis_vectors_cart[:, 0, :], self.basis_vectors_cart[:, 1, :])
        return perp / np.linalg.norm(perp, axis=1)[:, None]

    @property
    def gradient_cart(self):
        return self.gradient_abs[:, None] * self.perpendicular

    @cached_property
    def weights(self):
        if self.dim == 2:
            area = np.linalg.norm(self.basis_vectors_cart[:, 0, :], axis=1)
        elif self.dim == 3:
            area = np.linalg.norm(np.cross(self.basis_vectors_cart[:, 0, :],
                                           self.basis_vectors_cart[:, 1, :]), axis=1) / 2
        return (area / self.gradient_abs) / (self.recip_volume)

    @classmethod
    def from_grid(cls, energy_grid, reciprocal_lattice_vectors, fermi_level, iband=None,
                  get_wf=False, system=None):
        if iband is not None:
            energy_grid = energy_grid[..., iband]
        dim = energy_grid.ndim
        energy_grid = energy_grid.copy() - fermi_level
        assert reciprocal_lattice_vectors.shape == (dim, dim), f"reciprocal_lattice_vectors must be a {dim}x{dim} array, got shape {reciprocal_lattice_vectors.shape}"
        below_EF = (energy_grid < 0)

        if dim == 2:
            shifts = get_shifts_2D(reciprocal_lattice_vectors)
        elif dim == 3:
            shifts = get_shifts_3D()

        res_list = [get_faces(energy_grid, shifts=sh, below_EF=below_EF, dim=dim) for sh in shifts]
        triangles = np.concatenate([res[0] for res in res_list], axis=0)
        gradient_red = np.concatenate([res[1] for res in res_list], axis=0)
        gradient_cart = np.einsum('aj, kj -> ka', np.linalg.inv(reciprocal_lattice_vectors), gradient_red)
        gradient_abs = np.linalg.norm(gradient_cart, axis=1)
        if get_wf:
            if system is None:
                raise ValueError("system must be provided if get_wf is True")
            kpoints = np.mean(triangles, axis=1)
            wavefunctions_center = get_wavefunction_on_kpoints(system, kpoints, iband)
        else:
            wavefunctions_center = None

        return cls(
            energy=fermi_level,
            recip_lattice=reciprocal_lattice_vectors,
            triangles=triangles,
            gradient_abs=gradient_abs,
            iband=iband,
            wavefunctions_center=wavefunctions_center
        )

    def plot(self, ax=None, show=True, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            if self.dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
            else:
                fig, ax = plt.subplots()
        if self.dim == 2:
            for tri in self.triangles_cart:
                ax.fill(tri[:, 0], tri[:, 1], **kwargs)
            ax.set_xlabel("kx")
            ax.set_ylabel("ky")
        elif self.dim == 3:
            if not hasattr(ax, "add_collection3d"):
                raise TypeError("For a 3D Fermi surface, ax must be a 3D matplotlib axis (projection='3d').")
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(self.triangles_cart, **kwargs)
            ax.add_collection3d(poly)
            ax.set_xlabel("kx")
            ax.set_ylabel("ky")
            ax.set_zlabel("kz")
        if show:
            plt.show()

    def get_wavefunctions_at_centers(self, system):
        if self.wavefunctions_center is None:
            self.wavefunctions_center = get_wavefunction_on_kpoints(system, self.kpoints, self.iband)
        return self.wavefunctions_center


def get_wavefunction_on_kpoints(system, kpoints, ibands, batch_size=20):
    print(f"Evaluating wavefunctions at {len(kpoints)} k-points for band(s) {ibands}...")
    if len(kpoints) == 0:
        return np.empty((0, system.num_wann), dtype=np.complex128)
    iRvec = system.rvec.iRvec
    H_R = system.get_R_mat("Ham")
    result = []
    for i in range(0, len(kpoints), batch_size):
        k = kpoints[i:i + batch_size]
        phase = np.exp(1j * (k @ iRvec.T))
        Hk = cached_einsum("kR,Rij->kij", phase, H_R)
        evals, evecs = np.linalg.eigh(Hk)
        result.append(evecs[:, :, ibands])
    return np.array(np.concatenate(result, axis=0))

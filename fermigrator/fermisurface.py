

import numpy as np
from propcache import cached_property

from fermigrator.get_fermi_surface import get_faces, get_shifts_2D, get_shifts_3D


class FermiSurface:

    def __init__(self,
                 energy,
                 recip_lattice,
                 triangles,
                 gradient_abs
                 ):
        self.energy = energy
        self.recip_lattice = recip_lattice
        assert self.recip_volume > 0, f"reciprocal lattice volume must be positive, got {self.recip_volume}"
        self.triangles = triangles
        self.gradient_abs = gradient_abs

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
        return {
            "energy": self.energy,
            "recip_lattice": self.recip_lattice,
            "triangles": self.triangles,
            "gradient_abs": self.gradient_abs,
        }

    def to_npz(self, filename):
        np.savez(filename, **self.as_dict())

    @property
    def dim(self):
        return self.recip_lattice.shape[0]

    @property
    def triangles_cart(self):
        return self.triangles @ self.recip_lattice

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
    def normal(self):
        if self.dim == 2:
            return np.array([0, 0, 1])
        elif self.dim == 3:
            return np.cross(self.triangles_cart[:, 1, :] - self.triangles_cart[:, 0, :],
                            self.triangles_cart[:, 2, :] - self.triangles_cart[:, 0, :])

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
    def from_grid(cls, energy_grid, reciprocal_lattice_vectors, fermi_level):
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
        return cls(
            energy=fermi_level,
            recip_lattice=reciprocal_lattice_vectors,
            triangles=triangles,
            gradient_abs=gradient_abs
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
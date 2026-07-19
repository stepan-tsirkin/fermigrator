

import numpy as np
from propcache import cached_property

from fermigrator.get_fermi_surface import get_faces, get_shifts_2D, get_shifts_3D
from .utility import cached_einsum, clear_cached
from .trajectory import TrajectoryFinder
from .brillouin import Brillouin
# Note: There are three types of coordinates used:
# * Cartesian "cart"
# * Reduced "reduced" (in the basis of the reciprocal lattice vectors)
# * local "local" (in the basis of the triangle vectors, with the third coordinate being perpendicular to the triangle plane)
# * "abs" means the absolute value in cartesian coordinates
# Below I need to make sure that the correct coordinates are used.

from scipy.constants import physical_constants
elementary_charge = physical_constants["elementary charge"][0]
hbar = physical_constants["Planck constant over 2 pi"][0]
coef_Btau = elementary_charge**2 / hbar**2 * 1e-12 * 1e-20  # [ps]*[T] * e^2 / hbar^2 * Ang^2
print(f"coef_Btau = {coef_Btau:.3e}")


class FermiSurface:

    keys = ["energy", "recip_lattice", "triangles_reduced",
            "gradient_abs", "wavefunctions_center", "iband",
            "grid_size", "triangle_neighbours", "is_connected"]
    triangle_side_order = [[0, 1], [0, 2], [1, 2]]  # the order of the triangle sides, used to find neighbours

    def __init__(self,
                 energy,
                 recip_lattice,
                 triangles_reduced,
                 gradient_abs,
                 wavefunctions_center=None,
                 iband=None,
                 grid_size=None,
                 triangle_neighbours=None,
                 is_connected=False
                 ):
        self.energy = energy
        self.recip_lattice = recip_lattice
        assert self.recip_volume > 0, f"reciprocal lattice volume must be positive, got {self.recip_volume}"
        self.triangles_reduced = triangles_reduced
        self.gradient_abs = gradient_abs
        self.wavefunctions_center = wavefunctions_center
        self.iband = iband
        self.grid_size = grid_size
        self.triangle_neighbours = triangle_neighbours
        self.is_connected = is_connected

    @classmethod
    def from_npz(cls, filename):
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

    @cached_property
    def cell_volume(self):
        return (2 * np.pi)**self.dim / self.recip_volume

    @cached_property
    def brillouin(self):
        return Brillouin(self.recip_lattice)

    def to_1bz(self):
        """map the kpoints to the first Brillouin zone"""
        shifts = self.brillouin.get_shifts(self.triangles_centers_reduced)
        self.triangles_reduced += shifts[:, None, :]

    def as_dict(self):
        dic = {key: getattr(self, key) for key in self.keys if getattr(self, key) is not None}
        return dic

    def to_npz(self, filename):
        np.savez(filename, **self.as_dict())

    @property
    def dim(self):
        return self.recip_lattice.shape[0]

    def get_wavefunction(self, iband):
        if self.wavefunctions is None:
            raise ValueError("Wavefunctions are not available for this Fermi surface.")
        return self.wavefunctions[:, :, iband]

    @cached_property
    def triangles_centers_reduced(self):
        return np.mean(self.triangles_reduced, axis=1)

    @property
    def is_empty(self):
        return self.num_triangles == 0

    @property
    def num_triangles(self):
        return len(self.triangles_reduced)

    @cached_property
    def triangles_cart(self):
        return self.triangles_reduced @ self.recip_lattice

    @cached_property
    def basis_vectors_cart(self):
        return self.triangles_cart[:, 1:, :] - self.triangles_cart[:, 0:1, :]

    @cached_property
    def basis_vectors_reduced(self):
        return self.triangles_reduced[:, 1:, :] - self.triangles_reduced[:, 0:1, :]

    @cached_property
    def perpendicular(self):
        """unit vector perpendicular to the triangle plane"""
        if self.dim == 2:
            perp = np.cross([0, 0, 1], self.basis_vectors_cart[:, 0, :])
        elif self.dim == 3:
            perp = np.cross(self.basis_vectors_cart[:, 0, :], self.basis_vectors_cart[:, 1, :])
        return perp / np.linalg.norm(perp, axis=1)[:, None]

    @cached_property
    def basis_vectors_cart_3(self):
        """supplement basis vectors with the unit vector perpendicular to the triangle plane"""
        return np.concatenate([self.basis_vectors_cart, self.perpendicular[:, None, :]], axis=1)

    @cached_property
    def basis_vectors_reduced_3(self):
        """supplement basis vectors with the unit vector perpendicular to the triangle plane"""
        return np.concatenate([self.basis_vectors_reduced, self.perpendicular[:, None, :]], axis=1)

    @cached_property
    def basis_vectors_cart_3_inv(self):
        return np.linalg.inv(self.basis_vectors_cart_3)

    @cached_property
    def basis_vectors_reduced_3_inv(self):
        return np.linalg.inv(self.basis_vectors_reduced_3)

    def project_to_triangle(self, point_reduced, triangle_index):
        """return the local coordinates"""
        return (point_reduced - self.triangles_reduced[triangle_index, 0, :]) @ self.basis_vectors_reduced_3_inv[triangle_index, :, :]

    def unproject_from_triangle(self, point_local, triangle_index):
        point_local_3 = np.concatenate([point_local, [0]])
        return self.triangles_reduced[triangle_index, 0, :] + \
            point_local_3 @ self.basis_vectors_reduced_3[triangle_index, :, :]

    def get_lorentz_force_local(self, B_cart):
        """
        find [V x B] in the basis of the triangle,
        V = |V| * [v1 x v2] / |v1 x v2|, where v1 and v2 are the basis vectors of the triangle
        therefore using bac-cab rule we can write
        [V x B] = |V| * (v1 (v2 . B) - v2 (v1 . B)) / (2*area)

        """
        basis = self.basis_vectors_cart
        v_dot_B = basis @ B_cart
        return (self.gradient_abs / (2 * self.triangle_areas))[:, None] * np.array([v_dot_B[:, 1], -v_dot_B[:, 0]]).T

    def get_magnetoconductivity_batch(self, B_dir_cart, Btau_list, num_samples, num_batches=10):
        if self.num_triangles / num_samples < 3:
            num_samples = self.num_triangles
            num_batches = 1
        conductivity = np.zeros((num_batches, len(Btau_list), self.dim, self.dim))
        for i in range(num_batches):
            conductivity[i] = self.get_magnetoconductivity(B_dir_cart, Btau_list, num_samples // num_batches)
        return np.mean(conductivity, axis=0), np.std(conductivity, axis=0)

    def get_magnetoconductivity(self, B_dir_cart, Btau_list, num_samples, exp_factor=7,
                                spin_factor=2  # added to compare with ShengNan's paper
                                ):
        # we want to evaluate the integral
        # vbae = \int_-\infty^0 dt e^{t/tau} u(k(t)) t/tau
        # by changing variable to t' = - t * e * B / hbar^2 * (Ang^2 * e) we get
        # vbar = \int_0^\infty dt'/Btau_loc e^{-t'/Btau_loc} u(k(t'))
        # where Btau_loc = tau [ps]* B[T] * 1e-12 * e^2 / hbar^2 *  1e-20  = tau [ps]* B[T] * coef_Btau
        Btau_min = 1e-15
        assert np.all(Btau_list >= 0), f"Btau_list must be non-negative, got {Btau_list}"
        select_Btau_nonzero = Btau_list > Btau_min
        Btau_list_loc = Btau_list[select_Btau_nonzero] * coef_Btau

        B_dir_cart = B_dir_cart / np.linalg.norm(B_dir_cart)
        lorentz_force_local = self.get_lorentz_force_local(B_dir_cart)
        trajectory_finder = TrajectoryFinder(
            triangles_reduced=self.triangles_reduced,
            basis_vectors_reduced=self.basis_vectors_reduced,
            basis_vectors_reduced_3_inv=self.basis_vectors_reduced_3_inv,
            lorentz_force_local=lorentz_force_local,
            triangle_neighbours=self.triangle_neighbours,
            triangles_centers_reduced=self.triangles_centers_reduced
        )

        weight_sum = 0
        conductivity = np.zeros((len(Btau_list), self.dim, self.dim))
        time_max = max(Btau_list_loc) * exp_factor
        if num_samples >= self.num_triangles:
            selected_triangles = np.arange(self.num_triangles)
        else:
            selected_triangles = np.random.choice(self.num_triangles, size=num_samples, replace=False)
        cyclic_count = 0
        for i, triangle_index in enumerate(selected_triangles):
            weight = self.weights[triangle_index]
            vn = self.gradient_cart[triangle_index]
            weight_sum += weight
            kpoints, times, triangle_indices, cyclic = trajectory_finder.get_trajectory(triangle_index, time_max)
            times = np.array(times)
            vt = self.gradient_cart[triangle_indices]
            vbar = np.zeros((len(Btau_list), self.dim))
            # remember that trajectory is a broken line, and along the segment, the velocity is constant, so we can integrate analytically
            # vbar = sum_i=0^N-1 \int_{t_i}^{t_{i+1}} dt' / Btau_loc e^{-t'/Btau_loc} v_i =
            #      = sum_i=0^N-1 v_i * (e^{-t_i/Btau_loc} - e^{-t_{i+1}/Btau_loc}) =
            exp_factor = np.exp(-times[None, :] / Btau_list_loc[:, None])
            vbar[:] = vn[None, :]
            vbar[select_Btau_nonzero] -= vn[None, :] * exp_factor[:, 1][:, None]  # initialize with the value at t=0
            exp_factor_diff = exp_factor[:, 1:-1] - exp_factor[:, 2:]
            v_bar_period = exp_factor_diff[:, :] @ vt[1:-1]
            if cyclic:
                # print(f"Trajectory is cyclic, period = {times[-1]-times[1]:.3e}, number of steps = {len(times)}")
                period = times[-1] - times[1]
                factor_period = np.exp(-period / Btau_list_loc)
                factor_period = 1 / (1 - factor_period)  # sum of geometric series
                v_bar_period[:] *= factor_period[:, None]
                cyclic_count += 1
            # else:
                # print(f"Trajectory is not cyclic, time = {times[-1]:.3e}, number of steps = {len(times)}")
            vbar[select_Btau_nonzero] += v_bar_period

            conductivity += weight * vn[None, :, None] * vbar[:, None, :]
            if i % 1000 == 0:
                print(f"Processed {i} triangles out of {len(selected_triangles)}...")
        print(f"Processed {len(selected_triangles)} triangles, of which {cyclic_count} were cyclic.")
        conductivity *= sum(self.weights) / weight_sum
        from wannierberri.factors import factor_ohmic, TAU_UNIT
        conductivity *= factor_ohmic * 1e-12 / TAU_UNIT / self.cell_volume
        return conductivity * spin_factor

    @property
    def gradient_cart(self):
        return self.gradient_abs[:, None] * self.perpendicular

    @cached_property
    def triangle_areas(self):
        if self.dim == 2:
            area = np.linalg.norm(self.basis_vectors_cart[:, 0, :], axis=1)
        elif self.dim == 3:
            area = np.linalg.norm(np.cross(self.basis_vectors_cart[:, 0, :],
                                           self.basis_vectors_cart[:, 1, :]), axis=1) / 2
        return area

    @cached_property
    def weights(self):
        return (self.triangle_areas / self.gradient_abs) / (self.recip_volume)

    @classmethod
    def from_grid(cls, energy_grid, reciprocal_lattice, fermi_level, iband=None,
                  get_wf=False, system=None,
                  set_triangle_neighbours=False):
        if iband is not None:
            energy_grid = energy_grid[..., iband]
        dim = energy_grid.ndim
        energy_grid = energy_grid.copy() - fermi_level
        assert reciprocal_lattice.shape == (dim, dim), f"reciprocal_lattice_vectors must be a {dim}x{dim} array, got shape {reciprocal_lattice.shape}"
        below_EF = (energy_grid < 0)
        grid_size = np.array(energy_grid.shape)

        if dim == 2:
            shifts = get_shifts_2D(reciprocal_lattice)
        elif dim == 3:
            shifts = get_shifts_3D()

        res_list = [get_faces(energy_grid, shifts=sh, below_EF=below_EF, dim=dim) for sh in shifts]
        triangles_reduced = np.concatenate([res[0] for res in res_list], axis=0)
        gradient_reduced = np.concatenate([res[1] for res in res_list], axis=0)
        gradient_cart = np.einsum('aj, kj -> ka', np.linalg.inv(reciprocal_lattice), gradient_reduced)
        gradient_abs = np.linalg.norm(gradient_cart, axis=1)

        if get_wf:
            if system is None:
                raise ValueError("system must be provided if get_wf is True")
            kpoints = np.mean(triangles_reduced, axis=1)
            wavefunctions_center = get_wavefunction_on_kpoints(system, kpoints, iband)
        else:
            wavefunctions_center = None

        obj = cls(
            energy=fermi_level,
            recip_lattice=reciprocal_lattice,
            triangles_reduced=triangles_reduced,
            gradient_abs=gradient_abs,
            iband=iband,
            wavefunctions_center=wavefunctions_center,
            grid_size=grid_size,
        )
        if set_triangle_neighbours:
            obj.set_triangle_neighbours()
        return obj

    def set_triangle_neighbours(self):
        if self.triangle_neighbours is None:
            max_per_cube = 24 if self.dim == 3 else 2
            triangles_centers_int = np.floor(self.triangles_centers_reduced * self.grid_size[None, :]).astype(int)  # % self.grid_size[None, :]
            triangles_in_cubes = -np.ones(tuple(self.grid_size) + (max_per_cube,), dtype=int)
            for i, center in enumerate(triangles_centers_int):
                cube_index = tuple(center)
                for j in range(max_per_cube):
                    if triangles_in_cubes[cube_index][j] == -1:
                        triangles_in_cubes[cube_index][j] = i
                        break
                else:
                    raise RuntimeError(f"Cube {cube_index} already has {max_per_cube} triangles, cannot add triangle {i}."
                                       f"already has triangles {triangles_in_cubes[cube_index]}. ")
            neighhbour_cubes = iterate_pm(self.dim)
            print(f"Finding neighbours for {self.num_triangles} triangles in {len(neighhbour_cubes)} cubes...")
            neighbours = []
            for i, center in enumerate(triangles_centers_int):
                candidates = np.concatenate([triangles_in_cubes[tuple((center + j) % self.grid_size)]
                                            for j in neighhbour_cubes])
                candidates = np.array(sorted(set(candidates) - {-1, i}))
                neighbours_side = []
                diff = self.triangles_reduced[i][None, :, None, :] - self.triangles_reduced[candidates][:, None, :, :]
                diff = diff - np.round(diff)  # (Ncand, dim, dim, dim) : (icand, vertex_i, vertex_cand, dim)
                is_close = np.all(np.isclose(diff, 0), axis=-1)  # (Ncand, dim, dim) : (icand, vertex_i, vertex_cand)
                for side in self.triangle_side_order:
                    shared_vertices = np.sum(is_close[:, side, :], axis=(1, 2))  # (Ncand,) : (icand)
                    neighbours_local = candidates[shared_vertices > 1]
                    if len(neighbours_local) != 1:
                        raise ValueError(f"Triangle {i} at center {center} has {len(neighbours_local)} neighbours on side {side}, expected 1. Neighbours found: {neighbours_local}")
                    neighbours_side.append(neighbours_local[0])
                neighbours.append(neighbours_side)
                if i % 10000 == 0:
                    print(f"Processed {i} triangles out of {self.num_triangles}...")
            self.triangle_neighbours = np.array(neighbours)
        return self.triangle_neighbours

    def plot_pyplot(self, ax=None, show=True, limits=None, **kwargs):
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
        if limits is not None:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            if self.dim == 3:
                ax.set_zlim(limits[2])
        if show:
            plt.show()

    def get_wavefunctions_at_centers(self, system):
        if self.wavefunctions_center is None:
            self.wavefunctions_center = get_wavefunction_on_kpoints(system, self.kpoints, self.iband)
        return self.wavefunctions_center

    def select_pocket_ids(self, triangle_start=0):
        """Select a pocket of the Fermi surface starting from triangle ik_start"""
        selected = set([triangle_start])
        previous_group = [triangle_start]
        while len(previous_group) > 0:
            new_group = set.union(*[set(neighbours) for neighbours in self.triangle_neighbours[previous_group]]) - selected
            selected.update(new_group)
            previous_group = list(new_group)
        return list(sorted(selected))

    def get_mesh(self):
        """Return the mesh of the Fermi surface as a tuple of vertices and faces"""
        triangles = self.triangles_cart
        vertices = triangles.reshape(-1, 3)
        tol = 1e-8
        vertices_round = np.round(vertices / tol).astype(np.int64)

        _, unique_idx, inverse = np.unique(
            vertices_round,
            axis=0,
            return_inverse=True,
            return_index=True
        )

        unique_vertices = vertices[unique_idx]
        faces = inverse.reshape(-1, 3)
        return unique_vertices, faces

    def plot_plotly(self):
        unique_vertices, faces = self.get_mesh()

        import plotly.graph_objects as go

        fig = go.Figure(
            go.Mesh3d(
                x=unique_vertices[:, 0],
                y=unique_vertices[:, 1],
                z=unique_vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                flatshading=False,
                color="blue",
                opacity=0.9,
                lighting=dict(
                    ambient=0.25,
                    diffuse=0.9,
                    specular=1.0,
                    roughness=0.15,
                    fresnel=0.3,
                ),

                lightposition=dict(
                    x=20,
                    y=20,
                    z=20,
                ),
            )
        )

        fig.update_layout(scene_aspectmode="data")
        fig.show()

    def plot_pyvista(self):
        import pyvista as pv

        vertices, faces = self.get_mesh()

        # Convert (N,3) triangle indices into PyVista format
        pv_faces = np.hstack([
            np.full((faces.shape[0], 1), 3),
            faces
        ]).astype(np.int64).ravel()

        mesh = pv.PolyData(vertices, pv_faces)

        plotter = pv.Plotter()

        mesh_FS = mesh.smooth(
            n_iter=20,
            relaxation_factor=0.1,
            feature_smoothing=False,
        )

        vertices_BZ, faces_BZ, edges_BZ = self.brillouin.boundary
        print(f"Brillouin zone has {len(vertices_BZ)} vertices and {len(faces_BZ)} faces.")
        L = np.max(np.ptp(vertices_BZ, axis=0))
        radius = 0.02 * L

        lines = np.hstack([
            np.full((len(edges_BZ), 1), 2),
            edges_BZ
        ]).ravel()
        print(f"lines = {lines}")

        edge_mesh = pv.PolyData(
            vertices_BZ,
            lines=lines
        )

        plotter.add_mesh(
            edge_mesh.tube(radius=0.01 * L),
            color="black",
        )

        plotter.add_mesh(
            mesh_FS,
            color="royalblue",
            smooth_shading=True,
            specular=0.9,
            specular_power=40,
        )

        balls = pv.PolyData(vertices_BZ)
        balls = balls.glyph(
            geom=pv.Sphere(radius=radius),
            scale=False,
            orient=False,
        )

        plotter.add_mesh(
            balls,
            color="black",
        )

        plotter.show()

    def to_pockets(self):
        """Split the Fermi surface into pockets"""
        pockets = []
        while not self.is_empty:
            pocket_ids = self.select_pocket_ids(triangle_start=0)
            pocket_id_inv = {id: i for i, id in enumerate(pocket_ids)}
            neighbours = np.array([[pocket_id_inv[i] for i in neighbours] for neighbours in self.triangle_neighbours[pocket_ids]])
            pocket = FermiSurface(
                energy=self.energy,
                recip_lattice=self.recip_lattice,
                triangles_reduced=self.triangles_reduced[pocket_ids],
                gradient_abs=self.gradient_abs[pocket_ids],
                wavefunctions_center=self.wavefunctions_center[pocket_ids] if self.wavefunctions_center is not None else None,
                iband=self.iband,
                grid_size=self.grid_size,
                triangle_neighbours=neighbours
            )
            pockets.append(pocket)

            # remove the pocket from the Fermi surface
            unselected_bool = np.ones(self.num_triangles, dtype=bool)
            unselected_bool[pocket_ids] = False
            unselected = np.where(unselected_bool)[0]
            unselected_inv = {id: i for i, id in enumerate(unselected)}
            self.triangles_reduced = self.triangles_reduced[unselected]
            self.gradient_abs = self.gradient_abs[unselected]
            if self.wavefunctions_center is not None:
                self.wavefunctions_center = self.wavefunctions_center[unselected]
            if self.triangle_neighbours is not None:
                self.triangle_neighbours = np.array([[unselected_inv[i] for i in neighbours] for neighbours in self.triangle_neighbours[unselected]])
            clear_cached(self, ["triangles_centers_reduced", "basis_vectors_cart", "basis_vectors_reduced",
                                "perpendicular", "basis_vectors_cart_3", "basis_vectors_reduced_3",
                                "basis_vectors_cart_3_inv", "basis_vectors_reduced_3_inv", "triangle_areas", "weights", "gradient_cart"])
            print(f"Found pocket with {len(pocket_ids)} triangles, {self.num_triangles} triangles remaining.")
        return pockets

    def connect(self):
        """to be applied to a connected pocket"""
        checked = set([0])
        previous_group = [0]
        while len(previous_group) > 0:
            new_group = set()
            for triangle in previous_group:
                for neighbour in self.triangle_neighbours[triangle]:
                    if neighbour not in checked:
                        shift = np.mean(self.triangles_reduced[neighbour] - self.triangles_reduced[triangle], axis=0)
                        self.triangles_reduced[neighbour] -= np.round(shift)
                        new_group.add(neighbour)
            previous_group = list(new_group)
            checked.update(new_group)

    def get_slices(self, axis_cart=[0, 0, 1], dk=0.1, k_list=None):
        """get slices of the Fermi surface along a given axis"""
        slices_dict = {}
        axis_cart = np.array(axis_cart)
        axis_cart = axis_cart / np.linalg.norm(axis_cart)
        triangles_corners_proj = self.triangles_cart @ axis_cart
        if k_list is None:
            kz_min = np.min(triangles_corners_proj)
            kz_max = np.max(triangles_corners_proj)
            nk = int(np.ceil((kz_max - kz_min) / dk))
            dk = (kz_max - kz_min) / nk
            kz = np.linspace(kz_min + dk / 2, kz_max - dk / 2, nk, endpoint=True)
        else:
            kz = np.array(k_list)
        for k in kz:
            slices = self.get_segments_kz(k, triangles_corners_proj)
            lines = self.get_all_connected_lines(*slices)
            if len(lines) > 0:
                slices_dict[k] = lines
        dk = kz[1] - kz[0] if len(kz) > 1 else None
        return slices_dict, dk

    def get_segments_kz(self, kz, triangles_corners_proj):
        """
        get the segments of crossing k=kz plane with the triangles of the Fermi surface.

        Parameters
        ----------
        kz : float
            The value of k along the axis_cart direction.
        triangles_corners_proj : np.ndarray
            The projected coordinates of the triangle corners along the axis_cart direction.

        Returns
        -------
        triangle_ids : np.ndarray((N,), dtype=int)
            The indices of the triangles that cross the k=kz plane.
        sides : np.ndarray((N, 2), dtype=int)
            The indices of the sides of the triangles that cross the k=kz plane. (0, 1, 2) corresponds to the sides of the triangle.
        segments : np.ndarray((N, 2, 3), dtype=float)
            The coordinates of the segments that cross the k=kz plane. Each segment is defined by two points in 3D space. (reduced coordinates)
        """
        above = triangles_corners_proj > kz
        below = triangles_corners_proj < kz
        above_any = np.any(above, axis=1)  # should it be >= ?
        below_any = np.any(below, axis=1)  # should it be <= ?
        triangle_ids = np.where(above_any & below_any)[0]
        N = len(triangle_ids)
        segments_reduced = np.zeros((N, 2, 3))
        sides = np.zeros((N, 2), dtype=int)
        for i, id in enumerate(triangle_ids):
            proj = triangles_corners_proj[id]
            above_loc = above[id]
            below_loc = below[id]
            segments_loc = []
            sides_loc = []
            for iside, side in enumerate(self.triangle_side_order):
                if (above_loc[side[0]] and below_loc[side[1]]) or (above_loc[side[1]] and below_loc[side[0]]):
                    sides_loc.append(iside)
                    a = (proj[side[0]] - kz) / (proj[side[0]] - proj[side[1]])
                    point = self.triangles_reduced[id, side[0], :] * (1 - a) + self.triangles_reduced[id, side[1], :] * a
                    segments_loc.append(point)
                if len(segments_loc) == 2:
                    break
            if len(segments_loc) != 2:
                raise ValueError(f"Triangle {id} does not have 2 intersection points with the plane k={kz}. Found {len(segments_loc)} points.")
            segments_reduced[i, :, :] = segments_loc
            sides[i, :] = sides_loc
        return triangle_ids, sides, segments_reduced

    def get_all_connected_lines(self, triangles_ids, sides, segments, tol=1e-6):
        lines = []
        while len(triangles_ids) > 0:
            line, (triangles_ids, sides, segments) = self.get_connected_line(triangles_ids, sides, segments, tol=tol)
            lines.append(line)
        return lines

    def get_connected_line(self, triangles_ids, sides, segments, tol=1e-6, plane_normal_cart=np.zeros(3)):
        triangles_ids_inv = {id: i for i, id in enumerate(triangles_ids)}
        used = np.zeros(len(triangles_ids), dtype=bool)
        line = [segments[0, 0, :], segments[0, 1, :]]
        line_triangle_ids = [triangles_ids[0]]
        used[0] = True
        # go forward

        def propagate(direction):
            """direction = 1 for forward, 0 for backward"""
            while True and not np.all(used):
                last_point = line[-direction]
                last_triangle_id = line_triangle_ids[-direction]
                last_sides = sides[triangles_ids_inv[last_triangle_id]]
                next_triangle_id = self.triangle_neighbours[last_triangle_id, last_sides[direction]]
                if next_triangle_id not in triangles_ids:
                    break
                else:
                    next_triangle_id_loc = triangles_ids_inv[next_triangle_id]
                    if used[next_triangle_id_loc]:
                        break
                    else:
                        next_segment = segments[next_triangle_id_loc]
                        diff = next_segment - last_point[None, :]
                        g = np.round(np.mean(diff, axis=0))
                        if np.dot(plane_normal_cart, g @ self.recip_lattice) > tol:
                            break
                        diff -= g[None, :]

                        dist_next_point = np.linalg.norm(diff, axis=1)

                        if dist_next_point[1 - direction] < tol:
                            pass
                        elif dist_next_point[direction] < tol:
                            segments[next_triangle_id_loc, :, :] = segments[next_triangle_id_loc, [1, 0], :]
                            sides[next_triangle_id_loc, :] = sides[next_triangle_id_loc, [1, 0]]
                        else:
                            break
                        next_point = segments[next_triangle_id_loc, direction, :] - g
                        if direction == 1:
                            line.append(next_point)
                            line_triangle_ids.append(next_triangle_id)
                        else:
                            line.insert(0, next_point)
                            line_triangle_ids.insert(0, next_triangle_id)
                        used[next_triangle_id_loc] = True
        propagate(direction=1)
        propagate(direction=0)
        unused = np.where(~used)[0]
        return (np.array(line), np.array(line_triangle_ids)), (triangles_ids[unused], sides[unused], segments[unused])


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


def iterate_pm(dim):
    if dim == 0:
        return [[]]
    res = []
    for i in [-1, 0, 1]:
        for j in iterate_pm(dim - 1):
            res.append([i] + j)
    return res

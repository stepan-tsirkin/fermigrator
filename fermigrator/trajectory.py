

from numba import float64, int64
from numba.experimental import jitclass
import numpy as np

spec = [
    ('triangles_reduced', float64[:, :, :]),
    ('basis_vectors_reduced', float64[:, :, :]),
    ('basis_vectors_reduced_3_inv', float64[:, :, :]),
    ('lorentz_force_local', float64[:, :]),
    ('triangle_neighbours', int64[:, :]),
    ('triangles_centers_reduced', float64[:, :]),
]


@jitclass(spec)
class TrajectoryFinder:

    def __init__(self, triangles_reduced, basis_vectors_reduced, basis_vectors_reduced_3_inv,
                 lorentz_force_local, triangle_neighbours, triangles_centers_reduced):
        self.triangles_reduced = triangles_reduced
        self.basis_vectors_reduced = basis_vectors_reduced
        self.basis_vectors_reduced_3_inv = basis_vectors_reduced_3_inv
        self.lorentz_force_local = lorentz_force_local
        self.triangle_neighbours = triangle_neighbours
        self.triangles_centers_reduced = triangles_centers_reduced

    def get_trajectory(self, triangle_index, time_max, kpoint_start_reduced=None,
                       end_kpoint_reduced=None, end_triangle_index=None):
        if kpoint_start_reduced is None:
            kpoint_start_reduced = self.triangles_centers_reduced[triangle_index]
        kpoint_reduced_list = [kpoint_start_reduced.copy()]
        time_list = [0]
        triangle_index_list = [triangle_index]
        if end_triangle_index is None:
            end_triangle_index = triangle_index
        kpoint_reduced = kpoint_start_reduced
        cyclic = False
        check_cyclic = False
        while time_list[-1] < time_max:
            kpoint_reduced, dt, triangle_index = self.trajectory_step(  # returned to the
                kpoint_reduced=kpoint_reduced,
                triangle_index=triangle_index,
            )
            time_list.append(time_list[-1] + dt)
            triangle_index_list.append(triangle_index)
            kpoint_reduced_list.append(kpoint_reduced)
            if not check_cyclic and end_kpoint_reduced is None:
                end_kpoint_reduced = kpoint_reduced_list[1]
            if check_cyclic and triangle_index_list[-1] == end_triangle_index:  # returned to the starting triangle
                diff = kpoint_reduced - end_kpoint_reduced
                diff -= np.round(diff)  # account for periodicity
                diff = np.linalg.norm(diff)
                if diff < 1e-5:
                    cyclic = True
                    break
            check_cyclic = True
        return kpoint_reduced_list, time_list, triangle_index_list, cyclic

    def trajectory_step(self,
                        kpoint_reduced,
                        triangle_index,
                        ):
        eps = 1e-10
        side_target = np.array([0, 0, 1], dtype=np.float64)
        side_direction = np.array([[0, -1], [-1, 0], [1, 1]], dtype=np.float64)
        triangle_reduced = self.triangles_reduced[triangle_index]
        basis_vectors_reduced = self.basis_vectors_reduced[triangle_index]
        basis_vectors_reduced_3_inv = self.basis_vectors_reduced_3_inv[triangle_index]
        lorentz_force_local = self.lorentz_force_local[triangle_index]
        kpoint_local = np.dot(kpoint_reduced - triangle_reduced[0, :], basis_vectors_reduced_3_inv[:, :2])
        side_vel = np.dot(side_direction, lorentz_force_local)
        side_k0 = side_target - np.dot(side_direction, kpoint_local)
        dt = np.zeros(3, dtype=np.float64)
        for i in range(3):
            if side_vel[i] <= eps:
                dt[i] = np.inf
            else:
                dt[i] = side_k0[i] / side_vel[i]
        iside = np.argmin(dt)
        dt = dt[iside]
        k_final_local = kpoint_local + lorentz_force_local * dt
        k_final_reduced = triangle_reduced[0, :] + np.dot(k_final_local, basis_vectors_reduced)
        triangle_index = self.triangle_neighbours[triangle_index, iside]
        g_shift = np.round(self.triangles_centers_reduced[triangle_index] - kpoint_reduced)
        return k_final_reduced + g_shift, dt, triangle_index

from numba import njit
import numpy as np


@njit
def get_trajectory(triangle_index, time_max, kpoint_start_reduced,
                   triangles_reduced, basis_vectors_reduced, basis_vectors_reduced_3_inv,
                   lorentz_force_local, triangle_neighbours, triangles_centers_reduced
                   ):
    kpoint_reduced_list = [kpoint_start_reduced.copy()]
    time_list = [0]
    triangle_index_list = [triangle_index]
    kpoint_reduced = kpoint_start_reduced
    while time_list[-1] < time_max:
        kpoint_reduced, dt, triangle_index = trajectory_step(
            kpoint_reduced=kpoint_reduced, triangle_index=triangle_index,
            triangle_reduced=triangles_reduced[triangle_index],
            basis_vectors_reduced=basis_vectors_reduced[triangle_index],
            basis_vectors_reduced_3_inv=basis_vectors_reduced_3_inv[triangle_index],
            lorentz_force_local=lorentz_force_local[triangle_index],
            triangle_neighbours=triangle_neighbours,
            triangles_centers_reduced=triangles_centers_reduced
        )
        time_list.append(time_list[-1] + dt)
        triangle_index_list.append(triangle_index)
        kpoint_reduced_list.append(kpoint_reduced)
    return kpoint_reduced_list, time_list, triangle_index_list


@njit
def trajectory_step(kpoint_reduced,
                    triangle_index,
                    triangle_reduced,
                    basis_vectors_reduced,
                    basis_vectors_reduced_3_inv,
                    lorentz_force_local,
                    triangle_neighbours,
                    triangles_centers_reduced):
    eps = 1e-10
    side_target = np.array([0, 0, 1], dtype=np.float64)
    side_direction = np.array([[0, -1], [-1, 0], [1, 1]], dtype=np.float64)
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
    triangle_index = triangle_neighbours[triangle_index, iside]
    g_shift = np.round(triangles_centers_reduced[triangle_index] - kpoint_reduced)
    return k_final_reduced + g_shift, dt, triangle_index

import pyvista as pv
import numpy as np
# from fermigrator.trajectory import TrajectoryFinder
from fermigrator.fermisurface import FermiSurface

fs = FermiSurface.from_npz("contours/contour_EF=+8.01636_ib=0005.npz")
fs.to_1bz()


# print("separating to pockets")
# pockets = fs.to_pockets()
# print(f"{len(pockets)=}")
# for p in pockets:
#     p.to_1bz()

axis_cart = np.array([1, 1, 0])
axis_cart = axis_cart / np.linalg.norm(axis_cart)

slices_dict, dk = fs.get_slices(axis_cart=axis_cart,
                                dk=0.1)
# k_list = [1.07])
# print(f"{len(slices_dict)=}, {dk=}")

plotter = fs.plot_BZ()
fs.plot_FS(plotter=plotter, opacity=0.98)


# axis_cart = np.array([0, 0, 1])
# lorentz_force_local = fs.get_lorentz_force_local(axis_cart)
# trajectory_finder = TrajectoryFinder(
#                     triangles_reduced=fs.triangles_reduced,
#                     basis_vectors_reduced=fs.basis_vectors_reduced,
#                     basis_vectors_reduced_3_inv=fs.basis_vectors_reduced_3_inv,
#                     lorentz_force_local=lorentz_force_local,
#                     triangle_neighbours=fs.triangle_neighbours,
#                     triangles_centers_reduced=fs.triangles_centers_reduced
#                     )

# istart = np.random.randint(0, len(fs.triangles_reduced))
# print (f"starting from triangle {istart}, center = {fs.triangles_centers_reduced[istart]}")
# kpoint_reduced_list, time_list, triangle_index_list, cyclic = trajectory_finder.get_trajectory(istart, 10)


# print (f"time_list = {time_list}")

# print (f"trajectory has {len(kpoint_reduced_list)} points, cyclic = {cyclic}")

# exit()
# plotter = pv.Plotter()


def add_line(line, color="red", radius=0.005, opacity=1.0):
    kpoints_reduced = line[0]
    if len(kpoints_reduced) < 2:
        return
    kpoints = kpoints_reduced @ fs.recip_lattice
    # print (f"adding line with {len(kpoints)=}")
    # print (f"line = \n{kpoints}")

    line = pv.lines_from_points(kpoints, close=False)

    plotter.add_mesh(
        line.tube(radius=radius * fs.L),
        color=color,
        smooth_shading=True,
        opacity=opacity,
    )

# add_line((kpoint_reduced_list,), color="red")


arrow = pv.Arrow(
    start=axis_cart * -2,
    direction=axis_cart * 5,
    shaft_radius=0.005 * fs.L,
    tip_radius=0.02 * fs.L,
    tip_length=0.2,
    scale="auto"
)

plotter.add_mesh(arrow)

for k in sorted(slices_dict.keys()):
    lines, lines_continuations = slices_dict[k]
    # print (f"{k=}, {len(slice)=}")
    for line, continuation in zip(lines, lines_continuations):
        print(f"{k=}, {len(line[0])=}, {len(continuation[0])=}")
        add_line(line, color="red", opacity=1)
        add_line(continuation, color="green", opacity=1)


plotter.show()

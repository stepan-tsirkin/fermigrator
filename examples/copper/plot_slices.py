import numpy as np
from fermigrator.fermisurface import FermiSurface
from fermigrator.brillouin import Brillouin
from matplotlib import pyplot as plt

fs = FermiSurface.from_npz(f"contours/contour_EF=+8.01636_ib=0005.npz")
fs.to_1bz()
fs.plot_pyvista()

print(fs.brillouin.boundary)

exit()
print("separating to pockets")
pockets = fs.to_pockets()
print(f"{len(pockets)=}")
for p in pockets:
    p.to_1bz()
print("pockets connected")
for p in pockets:
    p.plot_pyvista()

exit()
pocket = pockets[0]

slices_dict, dk = pocket.get_slices(axis_cart=np.array([0, 0, 1]), dk=0.2)
print(f"{len(slices_dict)=}, {dk=}")


kmin = np.min([(l[0] @ fs.recip_lattice)[:, :].min(axis=0) for v in slices_dict.values() for l in v], axis=0) - 1e-8
kmax = np.max([(l[0] @ fs.recip_lattice)[:, :].max(axis=0) for v in slices_dict.values() for l in v], axis=0) + 1e-8
kmiddle = 0.5 * (kmin + kmax)
print(f"{kmin=}, {kmax=}, {kmiddle=}")

for k in sorted(slices_dict.keys()):
    slice = slices_dict[k]
    # print (f"{k=}, {len(slice)=}")
    for line in slice:
        # print (f"{line=}")
        kpoints_reduced = line[0]
        kpoints = kpoints_reduced @ fs.recip_lattice
        kz = kpoints[:, 2]
        assert np.std(kz) < 1e-8, f"kz={kz}, std={np.std(kz)}"
        kz = np.mean(kz)
        # print(f"{kz=}")
        a = (kz - kmin[2]) / (kmax[2] - kmin[2])
        plt.plot(kpoints[:, 0], kpoints[:, 1], color=(a, 0, 1 - a), label=f"{kz=}",
                 linestyle='-' if kz < kmiddle[2] else '--', linewidth=3)

k_range = max(kmax[0] - kmin[0], kmax[1] - kmin[1])
plt.xlim(kmiddle[0] - 0.5 * k_range, kmiddle[0] + 0.5 * k_range)
plt.ylim(kmiddle[1] - 0.5 * k_range, kmiddle[1] + 0.5 * k_range)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()

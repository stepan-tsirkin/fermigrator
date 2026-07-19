from matplotlib import pyplot as plt
import numpy as np
from fermigrator.fermisurface import FermiSurface


NK = 50

try:
    fs = FermiSurface.from_npz(f"fermi_surface-N={NK}.npz")
except FileNotFoundError:
    kx = np.linspace(0, 1, NK) * 2 * np.pi
    ky = np.linspace(0, 1, NK) * 2 * np.pi
    kz = np.linspace(0, 1, NK) * 2 * np.pi

    energy = 2.5 - np.cos(2 * kx[:, None, None]) + np.cos(ky[None, :, None]) + np.cos(kz[None, None, :])

    rec_lattice = np.eye(3) * 2 * np.pi
    fs = FermiSurface.from_grid(energy, reciprocal_lattice=rec_lattice, fermi_level=0.0, set_triangle_neighbours=True)
    fs.to_npz(f"fermi_surface-N={NK}.npz")
# fs.plot()
pockets = fs.to_pockets()
for p in pockets:
    p.connect()
# pockets[0].plot()
# pockets[1].plot()

pocket = pockets[0]

slices_dict, dk = pocket.get_slices(axis_cart=np.array([0, 0, 1]), dk=0.05)


kmin = min(l[0][:, 2].min() for v in slices_dict.values() for l in v) - 1e-8
kmax = max(l[0][:, 2].max() for v in slices_dict.values() for l in v) + 1e-8
kmiddle = 0.5 * (kmin + kmax)
print(f"{kmin=}, {kmax=}, {kmiddle=}")

for k in sorted(slices_dict.keys()):
    slice = slices_dict[k]
    print(f"{k=}, {len(slice)=}")
    for line in slice:
        # print (f"{line=}")
        kpoints = line[0]
        kz = kpoints[:, 2]
        assert np.std(kz) < 1e-8
        kz = np.mean(kz)
        # print(f"{kz=}")
        a = (kz - kmin) / (kmax - kmin)
        plt.plot(kpoints[:, 0], kpoints[:, 1], color=(a, 0, 1 - a), label=f"{kz=}",
                 linestyle='-' if kz < kmiddle else '--', linewidth=3)
plt.legend()
plt.show()

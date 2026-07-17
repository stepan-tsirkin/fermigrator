import numpy as np
from fermigrator.fermisurface import FermiSurface
NK = 20
kx = np.linspace(0, 1, NK) * 2 * np.pi
ky = np.linspace(0, 1, NK) * 2 * np.pi
kz = np.linspace(0, 1, NK) * 2 * np.pi

energy = 2.5 - np.cos(2 * kx[:, None, None]) + np.cos(ky[None, :, None]) + np.cos(kz[None, None, :])

rec_lattice = np.eye(3) * 2 * np.pi
fs = FermiSurface.from_grid(energy, reciprocal_lattice=rec_lattice, fermi_level=0.0, set_triangle_neighbours=True)
# fs.plot()
pockets = fs.to_pockets()
for p in pockets:
    p.connect()
pockets[0].plot()
pockets[1].plot()

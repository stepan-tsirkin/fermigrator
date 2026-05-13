from fermigrator.scattering_matrix import ScatteringMatrix, get_chk
from fermigrator.database import ContourDatabase
from ase.units import Hartree

import numpy as np

elements = np.load("elements.npz")
print(elements.files)
for k in ["V_direct", "V_correction", "V_total", "h_wann", "u_mat", "M", "wannier_centres"]:
    print(f"{k}: {elements[k].shape}")
for k in ["fermi", "cell", "nspin", "wannier_centres"]:
    print(f"{k}: {elements[k]}")

# in the archive, spin is a slow index, but in the system it is a fast index
reorder = [0, 2, 1, 3]
N_supercell = 36

Vkk = elements["V_total"]
nk = Vkk.shape[0]
Vkk = Vkk.reshape(nk, nk, 2, 2, 2, 2).transpose(0, 1, 3, 2, 5, 4) / Hartree * N_supercell
db = ContourDatabase.read("contours")
system = db.system
print(f"System has {system.num_wann} wannier functions with centers at {system.wannier_centers_red} in reduced coordinates and real lattice vectors {system.real_lattice}")
chk = get_chk("wannier/graphene.chk")
scatter = ScatteringMatrix.from_Vkk(Vkkmn_wan=Vkk,
                                    mp_grid=chk.mp_grid,
                                    kpt_red=chk.kpt_red,
                                    real_lattice=chk.real_lattice,
                                    wannier_centers_red=system.wannier_centers_red,
                                    )
iR0=scatter.rvec.iR0
print(f"iR0: {iR0}\n R0: {scatter.rvec.iRvec[iR0]}")
print(f"scatter iRvec shape: {scatter.rvec.iRvec.shape}")
print(f"scatter for R=R'=0: (real part)\n{scatter.Vrrab[iR0, iR0].real}")
print(f"scatter for R=R'=0: (imaginary part)\n{scatter.Vrrab[iR0, iR0].imag}")

scatter.get_Vkk_on_contours_all(contours_db=db)

scatter.get_multipole_on_contours_all(contours_db=db)

scatter.to_npz("scatter_RR.npz")

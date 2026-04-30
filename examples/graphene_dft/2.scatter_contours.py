from integrateFermi.scattering_matrix import ScatteringMatrix, get_chk
from integrateFermi.database import ContourDatabase
from ase.units import Hartree

import numpy as np

elements = np.load("elements.npz")
print (elements.files)
for k in ["V_direct", "V_correction", "V_total", "h_wann", "u_mat", "M", "wannier_centres"]:
    print (f"{k}: {elements[k].shape}")
for k in ["fermi", "cell", "nspin", "wannier_centres"]:
    print (f"{k}: {elements[k]}")

# in the archive, spin is a slow index, but in the system it is a fast index
reorder = [0,2,1,3]
Vkk = elements["V_total"][:,:,reorder,:][:,:,:,reorder] * Hartree

db = ContourDatabase.read("contours")
system = db.system
print (f"System has {system.num_wann} wannier functions with centers at {system.wannier_centers_red} in reduced coordinates and real lattice vectors {system.real_lattice}")
chk = get_chk("wannier/graphene.chk")
scatter = ScatteringMatrix.from_Vkk(Vkkmn_wan=Vkk, 
                                    mp_grid=chk.mp_grid,
                                    kpt_red=chk.kpt_red,
                                    real_lattice=chk.real_lattice,
                                    wannier_centers_red=system.wannier_centers_red,
)

scatter.get_Vkk_on_contours_all(contours_db=db)

scatter.get_multipole_on_contours_all(contours_db=db)

scatter.to_npz("scatter_RR.npz")

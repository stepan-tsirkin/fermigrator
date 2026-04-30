from integrateFermi.scattering_matrix import ScatteringMatrix, get_chk
from integrateFermi.database import ContourDatabase
from wannierberri.w90files import WannierData
from ase.units import Hartree

import numpy as np

elements = np.load("elements.npz")
wandata = WannierData.from_w90_files(seedname="wannier/graphene", files=["chk", "eig"])

kpt_red = wandata.chk.kpt_red
# in the archive, spin is a slow index, but in the system it is a fast index
reorder = [0,2,1,3]
Vkk = elements["V_total"][:,:,reorder,:][:,:,:,reorder] * Hartree

wannier_centers_red = wandata.wannier_centers_cart @ np.linalg.inv(wandata.chk.real_lattice)

wannier_centers_red_spin = np.zeros( (4,3))
wannier_centers_red_spin[0::2] = wannier_centers_red
wannier_centers_red_spin[1::2] = wannier_centers_red

scatter = ScatteringMatrix.from_Vkk(Vkkmn_wan=Vkk, 
                                    mp_grid=wandata.chk.mp_grid,
                                    kpt_red=wandata.chk.kpt_red,
                                    real_lattice=wandata.chk.real_lattice,
                                    wannier_centers_red=wannier_centers_red_spin,
)
Vkk_new = scatter.get_on_kpoints(kpt_red_left=kpt_red, kpt_red_right=kpt_red)
print(f"Wannier centers in reduced coordinates: {wannier_centers_red_spin}")

i=0
j=1
diff = np.abs(Vkk_new[i,j] - Vkk[i,j])
print (f"Vkk_new on kpoints: {Vkk_new[i,j]}")
print (f"Vkk on mp grid: {Vkk[i,j]}")
print (f"Max difference between Vkk on kpoints and Vkk on mp grid is {diff.max():.2e} eV")  
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

diff = np.abs(Vkk_new - Vkk)
diffmax = diff.max()
assert diffmax < 1e-6, f"Max difference between Vkk on kpoints and Vkk on mp grid is {diffmax:.2e} eV"  
print (f"Max difference between Vkk on kpoints and Vkk on mp grid is {diffmax:.2e} eV")  



kpt_red_left = np.array([[0,0,0.]])

nk = 40
kpt_red_right = np.array([[kx,ky,0] for kx in np.linspace(0,1,nk) for ky in np.linspace(0,1,nk)])

kpt_cart_left = kpt_red_left @ wandata.chk.recip_lattice
kpt_cart_right = kpt_red_right @ wandata.chk.recip_lattice

Vkk_new = scatter.get_on_kpoints(kpt_red_left=kpt_red_left, kpt_red_right=kpt_red_right)

from matplotlib import pyplot as plt
fig, ax = plt.subplots(4,4, figsize=(12,12))
for i in range(4):
    for j in range(4):
        ax[i,j].scatter(kpt_cart_right[:,0], kpt_cart_right[:,1], c=Vkk_new[0,:, i,j].real, s=5)
        ax[i,j].set_title(f"Vkk[{i},{j}]")
        ax[i,j].scatter(kpt_cart_left[:,0], kpt_cart_left[:,1], c='red', s=50, marker='x')
        # colorbar
        plt.colorbar(ax[i,j].collections[0], ax=ax[i,j])

        
plt.tight_layout()
plt.savefig("Vkk.png")
                       
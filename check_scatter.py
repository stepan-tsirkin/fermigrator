import glob
import os
import numpy as np
from integrateFermi.scattering import ScatteringMatrix
from wannierberri.w90files import Wannier90data, CheckPoint
from wannierberri.system import System_w90, System_R
from integrateFermi.contours2D import get_contours_and_WFs


import ray
# ray.init()  # uncomment for parallel execution

path = "wannierberri"
seed = "wse2"
seed_w90data = f"{path}/{seed}"
path_system = f"{path}/system"
Vkkmn = np.load(f"{path}/elements.npz")["V"]

path_scatter = f"{path}/scattering"
os.makedirs(path_scatter, exist_ok=True)


try:
    system = System_R.from_npz(f"{path_system}")
    chk = CheckPoint.from_npz(f"{seed_w90data}.chk.npz")
except FileNotFoundError as e:
    print(
        f"System file not found at {path_system}: {e}. Creating system from w90data and saving to file.")
    w90data = Wannier90data.from_npz(seed_w90data,
                                     files=["chk", "eig", "symmetrizer"])
    system = System_w90(w90data=w90data, periodic=(True, True, False))
    system.symmetrize2(w90data.symmetrizer)
    system.positions = [[1/3, 2/3, -0.7], [1/3, 2/3, 0.7], [0, 0, 0]]
    system.atom_labels = [1, 1, 2]
    system.save_npz(f"{path_system}")
    chk = w90data.chk



# path, bands = system.get_bandstructure()
# bands.plot_path_fat(
#     path, save_file=f"{path_scatter}/bands.png", show_fig=False, close_fig=True)


# get_contours_and_WFs(system=system,
#                      grid=50,
#                      recalculate_E_if_exists=False,
#                      save_dir=path_scatter,
#                      Efermi_list=[2.4],
#                      )


file_scatter_matrix = f"{path_scatter}/scatter_RR.npz"
try:
    scatter = ScatteringMatrix.from_npz(file_scatter_matrix)
except FileNotFoundError as e:
    print(
        f"Scattering matrix file not found at {file_scatter_matrix}: {e}. Creating scattering matrix from Vkkmn and saving to file.")
    scatter = ScatteringMatrix(
        center_red=[0, 0, 0],
        gauge="wannier",
        Vkkmn=Vkkmn)
    scatter.set_RR(chk)
    scatter.to_npz(file_scatter_matrix)

lamb, W = scatter.multipole_decomposition_RR()
srt = np.argsort(abs(lamb))


nR, nW = W.shape[1], W.shape[2]
W = W[srt].reshape(W.shape[0], -1)
lamb = lamb[srt]


from matplotlib import pyplot as plt
plt.plot(lamb, "o")
# plt.yscale("log")
plt.xlabel("multipole index")
plt.ylabel("multipole strength")
# plt.ylim(-1, 10)
plt.title("Multipole decomposition of scattering matrix")
plt.savefig(f"{path_scatter}/multipole_decomposition.png")
plt.close()
for i in range(len(lamb)):
    W_loc = abs(W[i].reshape(nR, nW))**2
    plt.imshow(np.log(W_loc), origin="lower")
    plt.colorbar()
    plt.title(f"Multipole {i} with strength {lamb[-i]:.2e}")
    plt.savefig(f"{path_scatter}/multipole_{i}.png")
    # logscale of the colorbar
    plt.close()



for f1 in glob.glob(f"{path_scatter}/contour*.npz"):
    for f2 in glob.glob(f"{path_scatter}/contour*.npz"):
        scatter.get_on_contours(f1, f2, save=True, path=path_scatter)

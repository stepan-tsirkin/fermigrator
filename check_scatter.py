import numpy as np
from integrateFermi.scattering import ScatteringMatrix
from wannierberri.w90files import CheckPoint

seed = "wannierberri/wse2"
chk = CheckPoint.from_npz(seed + ".chk.npz")
Vkkmn = np.load("wannierberri/elements.npz")["V"]
scatter = ScatteringMatrix(
    center_red=[0,0,0],
    gauge="wannier",
    Vkkmn=Vkkmn)
scatter.set_RR(chk)
scatter.to_npz(seed + ".scatter_RR.npz")
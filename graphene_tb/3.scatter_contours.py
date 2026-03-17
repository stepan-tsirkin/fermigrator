from integrateFermi.scattering import ScatteringMatrix

scatter = ScatteringMatrix.from_npz("scatter_RR.npz")

scatter.get_Vkk_on_contours_all(path="contours")
from integrateFermi.scattering import ScatteringMatrix
from wannierberri import System_R
system = System_R.from_npz(f"system")    
scatter = ScatteringMatrix(rvec=system.rvec, num_wann=system.num_wann)
# scatter.set_VRR(1.0, irvec1=[0,0,0], irvec2=[0,0,0], ab=[0,0])
scatter.set_VRR(1.0, irvec1=[0,0,0], irvec2=[0,0,0], ab=[1,1])

scatter.get_Vkk_on_contours_all(path="contours")

scatter.get_multipole_on_contours_all(path="contours")

scatter.to_npz("scatter_RR.npz")

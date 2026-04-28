from integrateFermi.scattering_matrix import ScatteringMatrix
from integrateFermi.database import ContourDatabase



db = ContourDatabase.read("contours")
scatter = ScatteringMatrix(rvec=db.system.rvec, num_wann=db.system.num_wann)
# scatter.set_VRR(1.0, irvec1=[0,0,0], irvec2=[0,0,0], ab=[0,0])
scatter.set_VRR(1.0, irvec1=[0,0,0], irvec2=[0,0,0], ab=[1,1])

scatter.get_Vkk_on_contours_all(contours_db=db)

scatter.get_multipole_on_contours_all(contours_db=db)

scatter.to_npz("scatter_RR.npz")

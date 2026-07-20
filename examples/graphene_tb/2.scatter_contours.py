from fermigrator.scattering_matrix import ScatteringMatrix, get_chk
from fermigrator.database import ContourDatabase

V0 = 1
db = ContourDatabase.read("contours")
system = db.system

scatter = ScatteringMatrix(rvec=system.rvec, num_wann=system.num_wann)
scatter.set_VRR(V0, irvec1=[0, 0, 0], irvec2=[0, 0, 0], ab=[0, 0])
scatter.set_VRR(-V0, irvec1=[0, 0, 0], irvec2=[0, 0, 0], ab=[1, 1])


scatter.get_Vkk_on_contours_all(contours_db=db)

scatter.get_multipole_on_contours_all(contours_db=db)

scatter.to_npz("scatter_RR.npz")

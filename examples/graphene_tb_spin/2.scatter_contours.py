from fermigrator.scattering_matrix import ScatteringMatrix, get_chk
from fermigrator.database import ContourDatabase

V0=1
alpha = 0.5
db = ContourDatabase.read("contours")
system = db.system

scatter = ScatteringMatrix(rvec=system.rvec, num_wann=system.num_wann, nspin=2)
scatter.set_VRR(V0, irvec1=[0,0,0], irvec2=[0,0,0], a=0, b=0, s1=0, s2=0)
scatter.set_VRR(-V0, irvec1=[0,0,0], irvec2=[0,0,0], a=1, b=1, s1=1, s2=1)

for irvec2 in [[0,0,0], [-1, 0, 0], [0,-1,0]]:
    scatter.set_VRR(alpha, irvec1=[0,0,0], irvec2=irvec2, a=0, b=1, s1=0, s2=1)
    scatter.set_VRR(alpha, irvec1=[0,0,0], irvec2=irvec2, a=0, b=1, s1=1, s2=0)


# scatter.get_Vkk_on_contours_all(contours_db=db)

scatter.get_multipole_on_contours_all(contours_db=db)

scatter.to_npz("scatter_RR.npz")

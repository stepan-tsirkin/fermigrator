import numpy as np
from integrateFermi.contours2D import get_contours_and_WFs
from wannierberri.models import Haldane_ptb
from wannierberri.system import System_PythTB
import ray
ray.init()
system = System_PythTB(Haldane_ptb(delta=0, phi=0, hop1=1, hop2=0))
system.save_npz("system")
get_contours_and_WFs(system=system,
                     grid=200,
                     save_dir="contours",
                     get_wf=False,
                     Efermi_list=np.linspace(-1, 1, 21),
                     )

contour_data = np.load("contours/contour_ib1_EF=0.40000.npz")


segments = contour_data["segments"]
gradients = contour_data["grad"]

Q = np.array([1/3, -1/3+0.1])

from integrateFermi.qpi import intersections, shift_contour_mod1
print ("segments shape", segments.shape)
contour1, indices1 = shift_contour_mod1(segments, np.array([0, 0.]))
contour2, indices2 = shift_contour_mod1(segments, -Q)

intersect = intersections(contour1, contour2)

from matplotlib import pyplot as plt
plt.figure()
for segment in contour1:
    plt.plot(segment[:, 0], segment[:, 1], 'b-')


k_list = []


for i, j, x, y in intersect:
    print ("intersection at", x, y, "between original segments", indices1[i], "and", indices2[j])
    grad1 = gradients[indices1[i]]
    grad2 = gradients[indices2[j]]
    print ("grad1", grad1, "grad2", grad2)
    cross_product = grad1[0]*grad2[1] - grad1[1]*grad2[0]
    k_list.append((x, y, (x+Q[0])%1, (y+Q[1])%1, 1./abs(cross_product)))
    plt.plot([x], [y], 'go')
    plt.plot([x+Q[0]], [y+Q[1]], 'ro')
    plt.arrow(x, y, Q[0], Q[1], head_width=0.03, head_length=0.01, fc='k', ec='k')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()


print ("k_list")
# convert array into text table
print ("x1 y1 x2 y2 weight")
for k in k_list:
    print (f"{k[0]:.5f} {k[1]:.5f} {k[2]:.5f} {k[3]:.5f} {k[4]:.5e}")

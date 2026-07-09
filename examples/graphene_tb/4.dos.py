from matplotlib import pyplot as plt
from wannierberri import Grid, run
from wannierberri.calculators.static import DOS, CumDOS
from fermigrator.database import ContourDatabase
from fermigrator.linewidth import getDOS
import numpy as np
dos_dict = {}

contour_db = ContourDatabase("contours")
for Efermi in contour_db.get_all_Efermi():
    dos_dict[Efermi] = getDOS(contour_db, Efermi)

x = []
y = []

for i, Efermi in enumerate(sorted(dos_dict.keys())):
    x.append(float(Efermi))
    y.append(dos_dict[Efermi])


y=np.array(y)
system = contour_db.system


Efermi_list = np.linspace(min(x), max(x), 1000)
calculators = {"dos": DOS(Efermi=Efermi_list, tetra=True),
               "cumdos": CumDOS(Efermi=Efermi_list, tetra=True)}
grid = Grid(system, NK=(90, 90, 1), NKFFT=(10, 10, 1))
# results = run(system=system, grid=grid, calculators=calculators)
result_dos = np.load("result-dos_iter-0000.npz")

plt.scatter(x, y, label="DOS from contours", color="red")
plt.plot(Efermi_list, result_dos["data"],
         label="DOS from WB with tetrahedron method", color="blue")
plt.legend()
plt.xlabel("Efermi")
plt.ylabel("DOS")
# plt.ylim(0, 2)
plt.grid()
plt.savefig("DOS_vs_Efermi.png")
plt.close()

from matplotlib import pyplot as plt
import numpy as np
from fermigrator.database import ContourDatabase
from fermigrator.skew import get_AHC_multipole, get_SHC_multipole, get_all_multipole_vtau

EF0 = np.load("efermi.npz")["EF"]
V0 = 3

contour_db = ContourDatabase.read("contours")

Efermi = contour_db.get_all_Efermi_float()

get_all_multipole_vtau(contour_db, gamma_0=1e-8)

print(f"Available Efermi values in the database: {Efermi}")

AHC_list = []
SHC_list = []

for EF in Efermi:
    AHC_list.append(get_AHC_multipole(contour_db, EF))
    SHC_list.append(get_SHC_multipole(contour_db, EF))

AHC_list = np.array(AHC_list)

for a in range(2):
    for b in range(2):
        plt.plot(Efermi, AHC_list[:, a, b], label=f"AHC[{a},{b}]")
plt.xlabel("Fermi Energy (eV)")
plt.ylabel("Anomalous Hall Conductivity (e^2/hbar)")
plt.title("AHC vs Fermi Energy")
plt.axvline(EF0, color="red", linestyle="--", label=f"EF={EF0:.2f} eV")
plt.legend()
plt.savefig("AHC_vs_EF.png")
plt.close()

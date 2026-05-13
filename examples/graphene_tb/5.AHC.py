import numpy as np
from fermigrator.database import ContourDatabase
from fermigrator.skew import get_Wkk_Efermi, get_AHC

EF0 = np.load("efermi.npz")["EF"]
V0=3

contour_db = ContourDatabase.read("contours")

Efermi = contour_db.get_all_Efermi_float()


print (f"Available Efermi values in the database: {Efermi}")

AHC_list = []

for EF in Efermi:
    # get_Wkk_Efermi(contour_db, EF)
    AHC = get_AHC(contour_db, EF)
    AHC_list.append(AHC)

AHC_list = np.array(AHC_list)

from matplotlib import pyplot as plt
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
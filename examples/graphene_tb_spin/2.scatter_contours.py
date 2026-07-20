from matplotlib import pyplot as plt
from fermigrator.linewidth import get_linewidth_multipole_Efermi as get_linewidth, getDOS
import numpy as np
import matplotlib.pyplot as plt
from fermigrator.scattering_matrix import ScatteringMatrix
from fermigrator.database import ContourDatabase
from fermigrator.skew import get_AHC_multipole, get_SHC_multipole, get_all_multipole_vtau

EF0 = np.load("efermi.npz")["EF"]

V0 = 1
V0z = 0
V1 = 0
V2 = 0
alpha = 0.5
beta = 0.


t = 1
contour_db = ContourDatabase.read("contours")
system = contour_db.system

pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
pauli_0 = np.eye(2)

wannier_centers_red = system.wannier_centers_red

scatter = ScatteringMatrix(rvec=system.rvec, num_wann=system.num_wann, nspin=2)

scatter.set_VRR(V0 * pauli_0, irvec1=[0, 0, 0], irvec2=[0, 0, 0], a=0, b=0)
scatter.set_VRR(V0z * pauli_z, irvec1=[0, 0, 0], irvec2=[0, 0, 0], a=0, b=0)

print(f"real lattice vectors:\n{system.real_lattice}")
print(f"wannier_centers_cart:\n{system.wannier_centers_cart}")

for irvec2 in [[0, 0, 0], [-1, 0, 0], [0, -1, 0]]:
    R = np.array(irvec2)
    scatter.set_VRR(V1 * pauli_0, irvec1=irvec2, irvec2=irvec2, a=1, b=1)
    scatter.set_VRR(V2 * pauli_0, irvec1=[0, 0, 0], irvec2=irvec2, a=0, b=1)

    # [d x s ]
    delta = (R + wannier_centers_red[1] - wannier_centers_red[0]) @ system.real_lattice
    print(f"Setting scatter VRR for irvec2={irvec2}, delta={delta}")
    hop = 1j * alpha * (delta[0] * pauli_y - delta[1] * pauli_x)
    scatter.set_VRR(hop, irvec1=[0, 0, 0], irvec2=irvec2, a=0, b=1, add_Herm_conj=True)
    # scatter.set_VRR(1j*beta*pauli_z, irvec1=[0,0,0], irvec2=irvec2, a=0, b=1, add_Herm_conj=True)
    # scatter.set_VRR(hop.T.conj(), irvec1=irvec2, irvec2=[0,0,0], a=1, b=0)

# scatter.get_Vkk_on_contours_all(contours_db=contour_db)

scatter.get_multipole_on_contours_all(contours_db=contour_db)


scatter.to_npz("scatter_RR.npz")


linewidth_dict = {}
for Efermi in contour_db.get_all_Efermi():
    linewidth_dict[Efermi] = get_linewidth(contour_db, Efermi)

x = []
y = []
dos = []

fig, axes = plt.subplots(1, 1, figsize=(6, 6), layout="tight")
for i, Efermi in enumerate(sorted(linewidth_dict.keys())):
    # if abs(float(Efermi)) >0.95:
    #     continue
    for ib, lw in linewidth_dict[Efermi].items():
        if np.mean(lw) < 100000:
            x.append(float(Efermi))
            y.append(np.mean(lw))
    dos.append(getDOS(contour_db, Efermi))
x = np.array(x) - EF0
srt = np.argsort(x)
x = x[srt]
y = np.array(y)[srt]
dos = np.array(dos)[srt]

dx = x[1:] - x[:-1]
dos_sum = sum((dos[:-1] + dos[1:]) / 2 * dx)

print(f"Integral of DOS over Efermi: {dos_sum:.2f} states/unit cell")

axes.plot(x, y, "o", label="Linewidth")


axes.plot(x, (dos * (V0 + 2 * x * V2 / t)**2) / 4, "x", label=f"$\mathrm{{DOS}}\\times(V_0 + 2(E-E_F)V_2/t)^2/4$, V0={V0} eV")


axes.set_xlabel(r"$E-E_F$ (eV)")
axes.set_ylabel("Average linewidth")
axes.grid()
axes.set_title("Linewidth vs Efermi")
axes.legend()
plt.savefig(f"linewidths_vs_Efermi.png")
plt.close()

Efermi = contour_db.get_all_Efermi_float()

get_all_multipole_vtau(contour_db, gamma_0=1e-8)

# print (f"Available Efermi values in the database: {Efermi}")

AHC_list = []
SHC_list = []

for EF in Efermi:
    AHC_list.append(get_AHC_multipole(contour_db, EF))
    SHC_list.append(get_SHC_multipole(contour_db, EF))

AHC_list = np.array(AHC_list)
SHC_list = np.array(SHC_list) * alpha

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


for a in range(2):
    for b in range(2):
        plt.plot(Efermi, SHC_list[:, a, b], label=f"SHC[{a},{b}]")
plt.xlabel("Fermi Energy (eV)")
plt.ylabel("Spin Hall Conductivity")
plt.title(f"SHC[ab_z] vs Fermi Energy")
plt.axvline(EF0, color="red", linestyle="--")
plt.legend()
plt.savefig("SHC_vs_EF.png")
plt.close()

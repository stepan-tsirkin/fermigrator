from fermigrator.scattering_matrix import ScatteringMatrix, get_chk
from fermigrator.database import ContourDatabase
from ase.units import Hartree
from matplotlib import pyplot as plt
from fermigrator.skew import get_AHC_multipole, get_SHC_multipole, get_all_multipole_vtau


import numpy as np
EF0 = np.load("efermi.npz")["EF"]

print (f"Hartree in eV: {Hartree}")
import numpy as np

elements = np.load("elements.npz")
print(elements.files)
for k in ["V_direct", "V_correction", "V_total", "h_wann", "u_mat", "M", "wannier_centres"]:
    print(f"{k}: {elements[k].shape}")
for k in ["fermi", "cell", "nspin", "wannier_centres"]:
    print(f"{k}: {elements[k]}")

# in the archive, spin is a slow index, so after reshape 
# Vkk has shape (nk, nk, nspin, num_wann, nspin, num_wann)
# We want to reshape it to (nk, nk, num_wann, num_wann, nspin, nspin) 
# in arxiv it is in Hartree
N_supercell = 36

Vkk = elements["V_total"]
nk = Vkk.shape[0]
Vkk = Vkk.reshape(nk, nk, 2, 2, 2, 2).transpose(0, 1, 3, 5, 2, 4) * Hartree * N_supercell
contour_db = ContourDatabase.read("contours")
system = contour_db.system
print(f"System has {system.num_wann} wannier functions with centers at {system.wannier_centers_red} in reduced coordinates and real lattice vectors {system.real_lattice}")
chk = get_chk("wannier/graphene.chk")
scatter = ScatteringMatrix.from_Vkk(Vkkmn_wan=Vkk,
                                    mp_grid=chk.mp_grid,
                                    kpt_red=chk.kpt_red,
                                    real_lattice=chk.real_lattice,
                                    wannier_centers_red=system.wannier_centers_red,
                                    )
iR0=scatter.rvec.iR0
print(f"iR0: {iR0}\n R0: {scatter.rvec.iRvec[iR0]}")
print(f"scatter iRvec shape: {scatter.rvec.iRvec.shape}")
print(f"scatter for R=R'=0: (real part)\n{scatter.Vrrab[iR0, iR0].real}")
print(f"scatter for R=R'=0: (imaginary part)\n{scatter.Vrrab[iR0, iR0].imag}")

# scatter.get_Vkk_on_contours_all(contours_db=db)

scatter.get_multipole_on_contours_all(contours_db=contour_db)

scatter.to_npz("scatter_RR.npz")

from fermigrator.linewidth import get_linewidth_multipole_Efermi as get_linewidth, getDOS

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
    dos.append( getDOS(contour_db, Efermi) )
x = np.array(x)-EF0
srt = np.argsort(x)
x = x[srt]
y = np.array(y)[srt]
dos = np.array(dos)[srt]

dx = x[1:] - x[:-1]
dos_sum = sum((dos[:-1] + dos[1:]) / 2 * dx)

print (f"Integral of DOS over Efermi: {dos_sum:.2f} states/unit cell")

axes.plot(x, y, "-", label="Linewidth")
axes.plot(-x, y, "--", label="Linewidth (E-Efermi flipped)")

V0 = 13.5
V2 = 0.5
t=1

# axes.plot(x, (dos*(V0+ 2*x*V2/t)**2)/4, "x", label=f"$\mathrm{{DOS}}\\times(V_0 + 2(E-E_F)V_2/t)^2/4$, V0={V0} eV") 



axes.set_xlabel(r"$E-E_F$ (eV)")
axes.set_ylabel("Average linewidth (arbitrary units)")
axes.grid()
axes.set_title("Linewidth vs Efermi")
axes.legend()
plt.savefig(f"linewidths_vs_Efermi.png")
plt.close()

Efermi = contour_db.get_all_Efermi_float()

get_all_multipole_vtau(contour_db, gamma_0 = 1e-8)

# print (f"Available Efermi values in the database: {Efermi}")

AHC_list = []
SHC_list = []

for EF in Efermi:
    AHC_list.append(get_AHC_multipole(contour_db, EF))
    SHC_list.append(get_SHC_multipole(contour_db, EF))

AHC_list = np.array(AHC_list)
SHC_list = np.array(SHC_list)

from matplotlib import pyplot as plt
for a in range(2):
    for b in range(2):
        plt.plot(Efermi, AHC_list[:, a, b], label=f"AHC[{'xyz'[a]},{'xyz'[b]}]")
plt.xlabel("Fermi Energy (eV)")
plt.ylabel("Anomalous Hall Conductivity (arbitrary units)")
plt.title("AHC vs Fermi Energy")
plt.axvline(EF0, color="red", linestyle="--", label=f"EF={EF0:.2f} eV")
plt.legend()
plt.savefig("AHC_vs_EF.png")
plt.close()


mult = 1./np.max(np.abs(SHC_list))
for a in range(2):
    for b in range(2):
        key ='xyz'[a] + 'xyz'[b]
        plt.plot(Efermi, SHC_list[:, a, b] * mult, label=r"$\sigma^{SH}_{" + key + r"z}$")
plt.xlabel("Fermi Energy (eV)")
plt.ylabel("Spin Hall Conductivity (arbitrary units)")
plt.title(f"SHC[ab_z] vs Fermi Energy")
plt.axvline(EF0, color="red", linestyle="--")
plt.legend()
plt.savefig("SHC_vs_EF.png")
plt.close()


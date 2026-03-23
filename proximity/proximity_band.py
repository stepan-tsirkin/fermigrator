"""Compare reference proximity_band.py folding with supercell.py."""

import numpy as np
import sys
from qe_defect_sternheimer.model.supercell import build_supercell_hamiltonian
from qe_defect_sternheimer import DefectSystem

# ── Load data ──────────────────────────────────────────────────────
data = np.load("elements.npz")
V = data["V_direct"] + data["V_correction"]
E_dis = data["E_dis"]
u_mat = data["u_mat"]


defect = DefectSystem(outdir="data", prefix="defect")
defect._load_band_structure()

# ── supercell.py (3D, matching notebook usage) ─────────────────────
H_sc, R_sc_arr, subcells = build_supercell_hamiltonian(
    V_kk=V,
    E_dis=E_dis,
    u_mat=u_mat,
    grid_shape=(24, 24, 1),
    M=np.diag([4, 4, 1]),
    nspin=2,
    fermi=defect.fermi,
)

# ── Compare band structure at path k-points ────────────────────────
from ase.units import Hartree, Bohr

nk_path = 120
K = np.array([-1 / 3, 1 / 3, 0])
M_pt = np.array([-0.5, 0, 0])
cell = defect.atoms.cell
path = cell.bandpath(
    "MKG",
    npoints=nk_path,
    special_points={
        "G": 0.004 * np.array([0, 0, 0]) + 0.996 * K,
        "M": 0.008 * M_pt + 0.992 * K,
        "K": K,
    },
)

sz = np.zeros((4, 4, 4, 4, 2, 2, 2, 2), dtype=np.complex128)
for i in range(4):
    for j in range(4):
        for s, spin in enumerate([1, -1]):
            for n in range(2):
                sz[i, j, i, j, s, n, s, n] = spin
sz = sz.transpose((0, 1, 4, 5, 2, 3, 6, 7)).reshape((64, 64))

# supercell.py: eigenvalues
phase_sc = np.exp(2j * np.pi * path.kpts @ R_sc_arr.T)
H_k_sc = np.einsum("kR,Rab->kab", phase_sc, H_sc, optimize=True)
E_sc, U_sc = np.linalg.eigh(H_k_sc)
E_sc *= Hartree
S_sc = np.einsum("kan,ab,kbn->kn", U_sc.conj(), sz, U_sc, optimize=True)

# ── DFT reference band structure ───────────────────────────────────
E_dft = Hartree * (defect.eigenvalues - defect.fermi)

# ── Plot ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "nature", "no-latex"])

x, x_ticks, x_tick_labels = path.get_linear_kpoint_axis()

plt.figure(figsize=(2.5, 3))
for i in range(30, 34):
    plt.scatter(
        1e3 * (x - x_ticks[1]),
        1e3 * E_sc[:, i] - 1.1,
        s=6,
        c=S_sc[:, i].real,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
    )
plt.plot(
    1e3 * (x - x_ticks[1]),
    1e3 * Hartree * (defect.eigenvalues - defect.fermi) - 3.35,
    c="k",
    lw=1,
    label="Full DFT",
)
plt.ylim(-6, 6)
plt.xlim(-1, 1)
plt.ylabel("Energy (meV)")
plt.xlabel(r"k [$10^{-3} \AA^{-1}$]")
plt.axhline(0, c="k", ls="--", lw=0.5)
plt.axvline(0, c="k", ls="--", lw=0.5)
plt.savefig("proximity_band.pdf", dpi=300)

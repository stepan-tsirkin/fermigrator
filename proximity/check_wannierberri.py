"""Check to_wannierberri: band structure + spin via WannierBerri internals."""

import numpy as np
import sys
import wannierberri
from wannierberri.calculators.tabulate import Energy, Spin, TabulatorAll
from wannierberri.grid import Path
from ase.units import Hartree, Bohr
from qe_defect_sternheimer.model.supercell import (
    build_supercell_hamiltonian,
    to_wannierberri,
)
from qe_defect_sternheimer import DefectSystem, PristineSystem

# ── Load data ──────────────────────────────────────────────────────
data = np.load("elements.npz")
V = data["V_direct"] + data["V_correction"]
E_dis = data["E_dis"]
u_mat = data["u_mat"]

pristine = PristineSystem(outdir="data", prefix="pristine", store_wavefunctions=True)
defect = DefectSystem(outdir="data", prefix="defect")
defect._load_band_structure()

M_mat = np.diag([4, 4, 1])
H_sc, R_sc_arr, subcells = build_supercell_hamiltonian(
    V_kk=V,
    E_dis=E_dis,
    u_mat=u_mat,
    grid_shape=(24, 24, 1),
    M=M_mat,
    nspin=2,
    fermi=defect.fermi,
)

# ── Build WannierBerri system (primitive cell in Bohr) ─────────────
cell_sc_ang = np.array(defect.atoms.cell)
cell_prim = np.linalg.inv(M_mat.astype(float)) @ cell_sc_ang
wannier_centres = pristine.atoms.get_scaled_positions() @ cell_prim

system = to_wannierberri(
    H_sc,
    R_sc_arr,
    subcells,
    cell=cell_prim,
    M=M_mat,
    nspin=2,
    periodic=(True, True, False),
    wannier_centres=wannier_centres,
)

# ── K-path (supercell fractional coords) ───────────────────────────
K = np.array([-1 / 3, 1 / 3, 0])
M_pt = np.array([-0.5, 0, 0])
G_near_K = 0.004 * np.array([0, 0, 0]) + 0.996 * K
M_near_K = 0.008 * M_pt + 0.992 * K

wb_path = Path(
    k_list=np.array(
        defect.atoms.cell.bandpath(
            "MKG",
            npoints=120,
            special_points={"G": G_near_K, "M": M_near_K, "K": K},
        ).kpts
    ),
    labels={0: "M'", 59: "K", 119: "G'"},
    system=system,
)

# ── Run WannierBerri calculators ───────────────────────────────────
calculators = {
    "tabulate": TabulatorAll(
        tabulators={
            "Energy": Energy(degen_thresh=0, degen_Kramers=False),
            "Spin": Spin(degen_thresh=0, degen_Kramers=False),
        },
        mode="path",
    )
}

result = wannierberri.run(
    system=system,
    grid=wb_path,
    calculators=calculators,
    parallel=False,
)

tab_result = result.results["tabulate"]
E_wb = tab_result.results["Energy"].data * Hartree  # [nk, nbands]
S_wb = tab_result.results["Spin"].data  # [nk, nbands, 3]
Sz_wb = S_wb[:, :, 2].real  # z-component

# ── Direct FT reference ───────────────────────────────────────────
cell = defect.atoms.cell
path = cell.bandpath(
    "MKG",
    npoints=120,
    special_points={"G": G_near_K, "M": M_near_K, "K": K},
)
phase = np.exp(2j * np.pi * path.kpts @ R_sc_arr.T)
H_k = np.einsum("kR,Rab->kab", phase, H_sc, optimize=True)

norb_sc = H_sc.shape[1]
nsc = len(subcells)
nwann = norb_sc // (nsc * 2)
sz_flat = np.zeros((norb_sc, norb_sc), dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]])
block = np.kron(sigma_z, np.eye(nwann))
for s in range(nsc):
    sl = slice(s * 2 * nwann, (s + 1) * 2 * nwann)
    sz_flat[sl, sl] = block

E_ref = np.zeros((120, norb_sc))
Sz_ref = np.zeros((120, norb_sc))
for k in range(120):
    evals, evecs = np.linalg.eigh(H_k[k])
    E_ref[k] = evals * Hartree
    Sz_ref[k] = np.einsum("an,ab,bn->n", evecs.conj(), sz_flat, evecs).real

print(f"max |E_ref - E_wb| = {np.max(np.abs(E_ref - E_wb)):.3e} eV")
print(f"max |Sz_ref - Sz_wb| = {np.max(np.abs(Sz_ref - Sz_wb)):.3e}")

# ── Plot ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

x, x_ticks, _ = path.get_linear_kpoint_axis()
x_K = x_ticks[1]
x_shifted = 1e3 * (x - x_K)

fig, axes = plt.subplots(1, 2, figsize=(5, 4), sharey=True)

for ax, E, Sz, title in [
    (axes[0], E_ref, Sz_ref, "Direct FT"),
    (axes[1], E_wb, Sz_wb, "WannierBerri"),
]:
    for i in range(norb_sc):
        ax.scatter(
            x_shifted, 1e3 * E[:, i], s=2, c=Sz[:, i], cmap="RdBu", vmin=-1, vmax=1
        )
    ax.set_title(title)
    ax.set_ylim(-6, 6)
    ax.set_xlim(-1, 1)
    ax.axhline(0, c="k", ls="--", lw=0.5)
    ax.axvline(0, c="k", ls="--", lw=0.5)

axes[0].set_ylabel("Energy (meV)")
for ax in axes:
    ax.set_xlabel(r"k [$10^{-3}$ a.u.]")

fig.tight_layout()
fig.savefig("check_wannierberri.png", dpi=150)
print("Plot saved to check_wannierberri.png")

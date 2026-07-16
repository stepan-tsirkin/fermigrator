from matplotlib import pyplot as plt
from gpaw import GPAW
from wannierberri.grid import Path
from wannierberri import System_R, WannierData

seed = "Cu"
try:
    system = System_R.from_npz("system")
except FileNotFoundError as e:
    print(f"System_R not found, creating a new one...: {e}")
    wandata = WannierData.from_npz(f"{seed}", files=["eig", "symmetrizer", "chk"])
    system = System_R.from_wannierdata(wandata=wandata)
    system.to_npz("system")


kpoints = {
    'G': [0.0, 0.0, 0.0],
    'X': [0.5, 0.0, 0.5],
    'W': [0.5, 0.25, 0.75],
    'L': [0.5, 0.5, 0.5],
    'K': [0.375, 0.375, 0.75],
}

path_labels = "WLGXWKG"

path = Path.from_nodes(real_lattice=system.real_lattice,
                       nodes=[kpoints[label] for label in path_labels],
                       labels=list(path_labels),
                       dk=0.01
                       )


calc_bands = GPAW(f"{seed}-bands.gpw",)
bs_dft = calc_bands.band_structure()

fig, ax = plt.subplots(figsize=(8, 6))
bs_dft.plot(show=False, emax=40.0, ax=ax, label="DFT")

path, bands_wannier = system.get_bandstructure(dk=0.05, path=path, return_path=True)

bands_wannier.plot_path_fat(path=path,
                            label="wannierised sp3",
                            # linecolor="orange",
                            axes=ax,
                            close_fig=False,
                            show_fig=False,
                            kwargs_line=dict(linestyle='-', lw=0.5),
                            )
# plt.show()
ax.set_ylim(-10, 30)
plt.savefig(f"{seed}-bands.png", dpi=300)

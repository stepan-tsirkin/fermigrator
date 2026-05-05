
from fermigrator.scattering import (get_contour_files_Efermi,
                                       get_linewidth_multipole_Efermi,
                                       get_all_Fermi_levels)
from matplotlib import pyplot as plt
from wannierberri import System_R
import numpy as np

path = "contours"
system = System_R.from_npz("system")

recip_lattice = system.recip_lattice[:2, :2]

linewidth_dict = {}
for Efermi in get_all_Fermi_levels(path):
    linewidth_dict[Efermi] = get_linewidth_multipole_Efermi(path, Efermi)

nfermi = len(linewidth_dict)
ncols = 4
nrows = nfermi // ncols
if nfermi % ncols != 0:
    nrows += 1
fig, axes = plt.subplots(nrows, ncols, figsize=(
    6*ncols, 6*nrows), layout="tight")
for i, Efermi in enumerate(sorted(linewidth_dict.keys())):
    ax = axes[i//ncols, i % ncols]
    for ib, lw in linewidth_dict[Efermi].items():
        contour = get_contour_files_Efermi(path, Efermi, ib=ib)[0]
        kpoints = np.load(contour)["kpoints"]
        kpoints_cart = kpoints @ recip_lattice
        sc = ax.scatter(
            kpoints_cart[:, 0], kpoints_cart[:, 1], c=lw, cmap="viridis", s=20)
        # make colorbar smaller and to the right of the plot and colorscale from min to max of lw
        vmin, vmax = np.min(lw), np.max(lw)
        sc.set_clim(vmin, vmax)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Linewidth", rotation=270, labelpad=15)
        # fig.colorbar(sc, ax=ax)
    ax.set_title(f"Efermi={Efermi:.2f}, tau={vmin:.2e}+-{(vmax-vmin)/2:.2e}")
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 7)
    # set aspect ratio to 1
    ax.set_aspect("equal")

# remove empty subplots
for j in range(i+1, nrows*ncols):
    fig.delaxes(axes[j//ncols, j % ncols])

plt.savefig("linewidths.png")
plt.close()
x = []
y = []
for i, Efermi in enumerate(sorted(linewidth_dict.keys())):
    for ib, lw in linewidth_dict[Efermi].items():
        if np.mean(lw) < 100000:
            x.append(Efermi)
            y.append(np.mean(lw))
plt.plot(x, y, "o-")
plt.xlabel("Fermi energy")
plt.ylabel("Average linewidth")
plt.savefig("linewidth_vs_Efermi.png")
plt.close()

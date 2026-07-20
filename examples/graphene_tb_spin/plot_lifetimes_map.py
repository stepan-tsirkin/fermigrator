
from collections import defaultdict

from fermigrator.database import ContourDatabase
import numpy as np
import matplotlib.pyplot as plt

contour_db = ContourDatabase.read("contours")


linewidth_dict = defaultdict(lambda: {})
for Efermi in contour_db.get_all_Efermi_float():
    rnd = Efermi / 0.09
    if abs(rnd - round(rnd)) > 1e-3:
        continue
    for ib in contour_db.get_all_bands(Efermi):
        linewidth_dict[Efermi][ib] = contour_db.get_data("linewidth-multipole", EF=Efermi, ib=ib)["linewidth"]

nfermi = len(linewidth_dict)
ncols = 2
nrows = nfermi

print(f"Plotting linewidths for {nfermi} Efermi values in {nrows} rows and {ncols} columns")
fig, axes = plt.subplots(nrows, ncols, figsize=(
    6 * ncols, 6 * nrows), layout="tight")
_, recip_lattice = contour_db.get_E_grid()
axes_cnt = 0


for i, Efermi in enumerate(sorted(linewidth_dict.keys())):
    for ib, lw in linewidth_dict[Efermi].items():
        contour = contour_db.get_data("contour", ib=ib, EF=Efermi)
        kpoints = contour["kpoints"]
        kpoints_cart = kpoints @ recip_lattice
        for s in range(lw.shape[1]):
            ax = axes[axes_cnt // ncols, axes_cnt % ncols]
            axes_cnt += 1
            lws = lw[:, s]
            sc = ax.scatter(
                kpoints_cart[:, 0], kpoints_cart[:, 1], c=lws, cmap="viridis", s=20)
            # make colorbar smaller and to the right of the plot and colorscale from min to max of lw
            vmin, vmax = np.min(lws), np.max(lws)
            sc.set_clim(vmin, vmax)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Linewidth", rotation=270, labelpad=15)
            # fig.colorbar(sc, ax=ax)
            ax.set_title(
                f"Ef={float(Efermi):.2f}, s={s} t={vmin:.2e}+-{(vmax - vmin) / 2:.2e}")
            ax.set_xlim(-1, 10)
            ax.set_ylim(-3, 8)
            # set aspect ratio to 1
            ax.set_aspect("equal")

# # remove empty subplots
# for j in range(axes_cnt, nrows*ncols):
#     fig.delaxes(axes[j//ncols, j % ncols])

plt.savefig(f"linewidths.png")
plt.close()

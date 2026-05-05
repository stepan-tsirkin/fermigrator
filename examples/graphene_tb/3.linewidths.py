import numpy as np
from fermigrator.database import ContourDatabase
from fermigrator.linewidth import getDOS
from matplotlib import pyplot as plt

EF0 = np.load("efermi.npz")["EF"]
V0=3

contour_db = ContourDatabase.read("contours")

method = "multipole"  # "direct" or "multipole"
# methd = "direct"  # "direct" or "multipole"

for method in ["direct",]:# "multipole"]:
    if method == "direct":
        from fermigrator.linewidth import get_linewidth_Efermi as get_linewidths
    elif method == "multipole":
        from fermigrator.linewidth import get_linewidth_multipole_Efermi as get_linewidths

    linewidth_dict = {}
    for Efermi in contour_db.get_all_Efermi():
        linewidth_dict[Efermi] = get_linewidths(contour_db, Efermi)

    nfermi = len(linewidth_dict)
    ncols = 4
    nrows = nfermi // ncols
    if nfermi % ncols != 0:
        nrows += 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        6*ncols, 6*nrows), layout="tight")
    _, recip_lattice = contour_db.get_E_grid()
    for i, Efermi in enumerate(sorted(linewidth_dict.keys())):
        ax = axes[i//ncols, i % ncols]
        for ib, lw in linewidth_dict[Efermi].items():
            contour = contour_db.get_data("contour", ib=ib, EF=Efermi)
            kpoints = contour["kpoints"]
            kpoints_cart = kpoints @ recip_lattice
            sc = ax.scatter(
                kpoints_cart[:, 0], kpoints_cart[:, 1], c=lw, cmap="viridis", s=20)
            # make colorbar smaller and to the right of the plot and colorscale from min to max of lw
            vmin, vmax = np.min(lw), np.max(lw)
            sc.set_clim(vmin, vmax)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Linewidth", rotation=270, labelpad=15)
            # fig.colorbar(sc, ax=ax)
        ax.set_title(
            f"Efermi={float(Efermi):.2f}, tau={vmin:.2e}+-{(vmax-vmin)/2:.2e}")
        ax.set_xlim(0, 10)
        ax.set_ylim(-3, 7)
        # set aspect ratio to 1
        ax.set_aspect("equal")

    # remove empty subplots
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j//ncols, j % ncols])

    plt.savefig(f"linewidths-{method}.png")
    plt.close()

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

    axes.plot(x, y, "o", label="Linewidth")
    axes.plot(x, (dos*V0**2)/4, "x", label=f"DOS*V0^2/4, V0={V0} eV") 
    axes.set_xlabel(r"$E-E_F$ (eV)")
    axes.set_ylabel("Average linewidth")
    axes.grid()
    axes.set_title("Linewidth vs Efermi")
    axes.legend()
    plt.savefig(f"linewidths_vs_Efermi-{method}.png")
    plt.close()


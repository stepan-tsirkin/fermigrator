import numpy as np
from matplotlib import pyplot as plt
fname = "magnetoconductivity_EF=-0.20000_Bdir=0.00-0.00-1.00.npz"


data = np.load(fname)
Btau_list = data['Btau_list']
conductivity_tot = data['conductivity']
errorbar_tot = data['errorbar']
B_dir_cart = data['B_dir_cart']
B_dir_str = data['B_dir_str']
EF = data['EF']


def plot_tensor(tensor, error, fname, label):
    figure, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    for i, a in enumerate("xyz"):
        for j, b in enumerate("xyz"):
            axes[i, j].fill_between(Btau_list, tensor[:, i, j] - error[:, i, j], tensor[:, i, j] + error[:, i, j], alpha=0.2)
            axes[i, j].plot(Btau_list, tensor[:, i, j])
            axes[i, j].set_xlabel('B*tau')
            axes[i, j].set_ylabel(f'{label}_{{{a}{b}}}')
            axes[i, j].grid()
    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=300)
    plt.close()


plot_tensor(conductivity_tot, errorbar_tot, f"magnetoconductivity_EF={EF:.5f}_Bdir={B_dir_str}", "sigma")
rho = np.linalg.inv(conductivity_tot)
plot_tensor(rho, errorbar_tot, f"magnetoresistivity_EF={EF:+.5f}_Bdir={B_dir_str}", "rho")

rho_diag = rho[:, [0, 1, 2], [0, 1, 2]]
MR = (rho - rho[0]) / rho[0]
for i, a in enumerate("xyz"):
    plt.plot(Btau_list, MR[:, i, i], label=f"MR_{a}{a}")
plt.xlabel("B*tau")
plt.ylabel("Magnetoresistance")
plt.title(f"Magnetoresistance, EF={EF:.2f} eV, B_dir={B_dir_str}")
plt.legend()
plt.grid()
plt.savefig(f"MR_EF={EF:+.5f}_Bdir={B_dir_str}.png", dpi=300)

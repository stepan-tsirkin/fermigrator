from matplotlib import pyplot as plt
import numpy as np

styles = {0: 'k-', 0.2: 'r-', -0.2: 'b--'}
styles_scatter = {0: 'k', 0.2: 'r', -0.2: 'b'}


figure, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10, 5))

for EF in 0, 0.2, -0.2:
    fname = f"magnetoconductivity_EF={EF:+.5f}_Bdir=0.00-0.00-1.00.npz"
    data = np.load(fname)
    Btau_list = data['Btau_list']
    conductivity_tot = data['conductivity']
    errorbar_tot = data['errorbar']
    B_dir_cart = data['B_dir_cart']
    B_dir_str = data['B_dir_str']
    # EF = data['EF']
    rho = np.linalg.inv(conductivity_tot)
    rhoxx = rho[:, 0, 0]
    MR = (rhoxx - rhoxx[0]) / rhoxx[0]
    select_fit = Btau_list > 1
    rhoxy = rho[:, 0, 1]

    from scipy.optimize import curve_fit

    def f(x, a, b):
        return a * x**b

    ab, cov = curve_fit(f, Btau_list[select_fit], MR[select_fit], p0=[0.002, 2])
    print(f"coef = {ab[0]} power = {ab[1]} for EF={EF}")

    axes[0].scatter(Btau_list, MR, c=styles_scatter[EF], label=f"EF={EF:+.2f} eV", )
    axes[0].plot(Btau_list, f(Btau_list, ab[0], ab[1]), styles[EF], alpha=0.5, label=f"fit: MR ~ (B*tau)^{ab[1]:.3f}, a={ab[0]:.5f}")
    # plt.plot(Btau_list, np.exp(fit[1]) * Btau_list**fit[0], styles[EF], alpha=0.5, label=f"fit: MR ~ (B*tau)^{fit[0]:.2f}")
    axes[1].plot(Btau_list, rhoxy, styles[EF], alpha=0.5, label=f"EF={EF:+.2f} eV")
# plt.loglog()
axes[0].set_xlabel("B*tau")
axes[0].set_ylabel("Magnetoresistance")
axes[0].set_title(f"Magnetoresistance, EF={EF:.2f} eV, B_dir={B_dir_str}")
axes[0].legend()
axes[0].grid()
axes[1].set_xlabel("B*tau")
axes[1].set_ylabel("rho_xy")
axes[1].set_title(f"Hall resistivity, EF={EF:.2f} eV, B_dir={B_dir_str}")
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.savefig(f"MR_Bdir={B_dir_str}.png", dpi=300)
plt.close()

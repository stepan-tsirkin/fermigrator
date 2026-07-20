from matplotlib import pyplot as plt
import numpy as np


figure, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10, 5))
axes[0], axes[1] = axes[1], axes[0]

EF = +8.01636


def get_rhoxx(fname):
    data = np.load(fname)
    Btau_list = data['Btau_list']
    conductivity_tot = data['conductivity']
    # EF = data['EF']
    rho = np.linalg.inv(conductivity_tot)
    rhoxx = rho[:, 0, 0]
    return Btau_list, rhoxx


color_list = ['orange', 'red', 'black', 'purple']
for theta_deg in [0, 18, 30, 45]:
    fname = f"magnetoconductivity_EF={EF:+.5f}_theta={theta_deg:d}_50000.npz"
    Btau_list, rhoxx = get_rhoxx(fname)
    # EF = data['EF']
    color = color_list.pop(0)
    axes[0].plot(Btau_list, rhoxx, label=f"theta={theta_deg:d} deg", color=color)

    # axes[1].plot(Btau_list, rhoxy, styles[EF], alpha=0.5, label=f"EF={EF:+.2f} eV")
# plt.loglog()


Btau = 40

theta_list = []
rho_list = []
theta_list_prec = []
rho_list_prec = []
for theta_deg in range(0, 91):
    try:
        fname = f"magnetoconductivity_EF={EF:+.5f}_theta={theta_deg:d}.npz"
        Btau_list, rhoxx = get_rhoxx(fname)
        i = np.argmin(np.abs(Btau_list - Btau))
        theta_list.append(theta_deg)
        rho_list.append(rhoxx[i])  # Take the value of rhoxx at the closest Btau for each theta
    except FileNotFoundError:
        print(f"File not found for theta={theta_deg} degrees. Skipping.")
        continue
    try:
        fname = f"magnetoconductivity_EF={EF:+.5f}_theta={theta_deg:d}_50000.npz"
        Btau_list, rhoxx = get_rhoxx(fname)
        i = np.argmin(np.abs(Btau_list - Btau))
        theta_list_prec.append(theta_deg)
        rho_list_prec.append(rhoxx[i])  # Take the value of rhoxx at the closest Btau for each theta
    except FileNotFoundError:
        print(f"File not found for theta={theta_deg} degrees. Skipping.")
        continue


pos = axes[1].get_position()
figure.delaxes(axes[1])

ax = figure.add_axes(pos, projection='polar')


# Convert degrees to radians
# ax.scatter(np.deg2rad(theta_list), rho_list)
# ax.plot(np.deg2rad(theta_list), rho_list, label=f"EF={EF:+.2f} eV, B*tau={Btau:.1f}", color='blue')
ax.plot(np.deg2rad(theta_list_prec), rho_list_prec, label=f"EF={EF:+.2f} eV, B*tau={Btau:.1f}", color='cyan', linewidth=2)
ax.plot(np.pi / 2 - np.deg2rad(theta_list_prec), rho_list_prec, label=f"EF={EF:+.2f} eV, B*tau={Btau:.1f} - reversed", color='blue', linewidth=2, ls='--')

# 0° at the top
ax.set_theta_zero_location('N')

# Angles increase clockwise
ax.set_theta_direction(-1)
ax.set_thetalim(0, np.pi / 2)
ax.legend()


axes[0].set_xlabel("B*tau")
axes[0].set_ylabel(r"$\rho_{xx}$ ($\Omega$ m)")
axes[0].legend()
axes[0].grid()
# axes[1].set_xlabel("B*tau")
# axes[1].set_ylabel(r"$\rho_{xy}$ ($\Omega$ m)")
# axes[1].set_title(f"Hall resistivity, EF={EF:.2f} eV, B_dir={B_dir_str}")
# axes[1].grid()
plt.tight_layout()
plt.savefig("MR.png", dpi=300)
plt.close()

from fermigrator.fermisurface import FermiSurface
import numpy as np

for EF in 0, 0.2, -0.2:
    Btau_list = np.linspace(0.0, 20, 100)
    B_dir_cart = np.array([0, 0, 1])

    B_dir_str = f"{B_dir_cart[0]:.2f}-{B_dir_cart[1]:.2f}-{B_dir_cart[2]:.2f}"

    conductivity_tot = np.zeros((len(Btau_list), 3, 3))
    errorbar_tot_sq = np.zeros((len(Btau_list), 3, 3))
    for ib in 0, 1:
        fs = FermiSurface.from_file(f"contours/contour_EF={EF:+.5f}_ib={ib:04d}.npz")
        conductivity, errorbar = fs.get_magnetoconductivity_batch(B_dir_cart=B_dir_cart,
                                                                  Btau_list=Btau_list,
                                                                  num_samples=5000)
        conductivity_tot += conductivity
        errorbar_tot_sq += errorbar**2
    errorbar_tot = np.sqrt(errorbar_tot_sq)

    np.savez(f"magnetoconductivity_EF={EF:+.5f}_Bdir={B_dir_str}.npz",
             Btau_list=Btau_list,
             conductivity=conductivity_tot,
             errorbar=errorbar_tot,
             B_dir_cart=B_dir_cart,
             B_dir_str=B_dir_str,
             EF=EF)

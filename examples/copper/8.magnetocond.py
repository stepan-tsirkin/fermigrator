from fermigrator.fermisurface import FermiSurface
import numpy as np


for EF in 8.01636, :
    for theta_deg in range(0, 91):
        theta = np.deg2rad(theta_deg)
        Btau_list = np.linspace(0.0, 40, 100)
        B_dir_cart = np.array([0, np.sin(theta), np.cos(theta)])

        B_dir_str = f"theta={theta_deg:d}"

        conductivity_tot = np.zeros((len(Btau_list), 3, 3))
        errorbar_tot_sq = np.zeros((len(Btau_list), 3, 3))
        for ib in range(11):
            fs = FermiSurface.from_file(f"contours/contour_EF={EF:+.5f}_ib={ib:04d}.npz")
            if fs.is_empty:
                print(f"Skipping empty Fermi surface for EF={EF}, ib={ib}")
                continue
            print(f"Fermi surface of band {ib} has {fs.num_triangles} ")
            conductivity, errorbar = fs.get_magnetoconductivity_batch(B_dir_cart=B_dir_cart,
                                                                      Btau_list=Btau_list,
                                                                      num_samples=5000)
            conductivity_tot += conductivity
            errorbar_tot_sq += errorbar**2
        errorbar_tot = np.sqrt(errorbar_tot_sq)

        np.savez(f"magnetoconductivity_EF={EF:+.5f}_{B_dir_str}.npz",
                 Btau_list=Btau_list,
                 conductivity=conductivity_tot,
                 errorbar=errorbar_tot,
                 B_dir_cart=B_dir_cart,
                 B_dir_str=B_dir_str,
                 EF=EF)

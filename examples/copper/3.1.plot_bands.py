from gpaw import GPAW
seed = "Cu"
calc_bands = GPAW(f"{seed}-bands.gpw",)
bs_dft = calc_bands.band_structure()
bs_dft.plot(show=True, emax=40.0)

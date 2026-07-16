from gpaw import GPAW
seed = "Cu"

calc_bands = GPAW(f"{seed}-scf.gpw",).fixed_density(
    nbands=24,
    symmetry='off',
    kpts={'path': 'WLGXWKG', 'npoints': 120},
    convergence={'bands': 20},
    txt=f"{seed}-bands.txt")
calc_bands.write(f"{seed}-bands.gpw", mode="all")

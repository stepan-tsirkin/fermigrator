from gpaw import GPAW
from gpaw.mpi import world
from irrep.spacegroup import SpaceGroup
seed = 'Cu'
calc_scf = GPAW(f'{seed}-scf.gpw', txt=None)
sg = SpaceGroup.from_gpaw(calc_scf)
if world.rank == 0:
    sg.show()
irred_kpt = sg.get_irreducible_kpoints_grid((8, 8, 8))
calc_nscf_irred = calc_scf.fixed_density(
    kpts=irred_kpt,
    symmetry={'symmorphic': False},
    nbands=24,
    convergence={'bands': 20},
    txt=f'{seed}-nscf-irred.txt')
calc_nscf_irred.write(f'{seed}-nscf.gpw', mode='all')

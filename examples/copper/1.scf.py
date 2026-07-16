
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, MixerSum
seed = "Cu"
a = 3.615
lattice = a * (np.ones((3, 3)) - np.eye(3)) / 2
positions = np.array([[0, 0, 0]])
atoms = Atoms("Cu", cell=lattice, pbc=[1, 1, 1], scaled_positions=positions
              )

calc = GPAW(
    mode=PW(500),
    xc="PBE",
    symmetry={'symmorphic': False},
    kpts={"size": [8, 8, 8], "gamma": True},
    convergence={"density": 1e-6},
    mixer=MixerSum(0.25, 8, 100),
    txt=f"{seed}-scf.txt"
)

atoms.calc = calc
atoms.get_potential_energy()
calc.write(f"{seed}-scf.gpw", mode="all")

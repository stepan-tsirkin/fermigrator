from matplotlib import pyplot as plt
from wannierberri.calculators.static import DOS, CumDOS
import numpy as np
from wannierberri import System_R, run, Grid

from wannierberri.models import Haldane_ptb
import ray
ray.init()

def graphene_ptb(delta=0, hop1=-1.0):
    """same as :func:`~wannierberri.models.Haldane_tbm`, but uses `PythTB <http://www.physics.rutgers.edu/pythtb/>`__

    Notes
    -----
    PythTB should be installed to use this (`pip install pythtb`)
    """
    import pythtb
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1. / 3., 1. / 3.], [2. / 3., 2. / 3.]]
    lattice = pythtb.Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    my_model = pythtb.TBModel(lattice)

    my_model.set_onsite([delta, -delta])
    my_model.set_hop(hop1, 0, 1, [0, 0])
    my_model.set_hop(hop1, 1, 0, [1, 0])
    my_model.set_hop(hop1, 1, 0, [0, 1])


system = System_R.from_pythtb(Haldane_ptb(delta=0, phi=0, hop1=1, hop2=0))


nspin=1
if nspin == 2:
    system.double_spin()
system.set_pointgroup(["C6z", "Inversion", "TimeReversal"])

system.to_npz("graphene-system")


EF = np.linspace(-10, 10, 4001)
grid = Grid(system, NKFFT=(18, 18, 1), NKdiv=(10, 10, 1))
calculators = {"dos": DOS(Efermi=EF, tetra=True),
               "cumdos": CumDOS(Efermi=EF, tetra=True)}
result = run(system=system, calculators=calculators, grid=grid)
cumdos = result.results["cumdos"].data

EF0 = EF[np.argmin(np.abs(cumdos - nspin))]

np.savez("efermi.npz", EF=EF0, cumdos=cumdos)

plt.plot(EF, cumdos)
plt.title(f"Cumulative DOS, EF = {EF0:.3f} eV")
plt.axvline(EF0, color="red", linestyle="--", label=f"EF={EF0:.2f} eV")
plt.xlabel("Energy (eV)")
plt.ylabel("Cumulative DOS")
plt.legend()
plt.savefig("cumdos.png")
plt.close()

path, bands = system.get_bandstructure(dk=0.01)
bands.plot_path_fat(path=path, save_file="bands.png")

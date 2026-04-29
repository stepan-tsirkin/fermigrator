import numpy as np
from wannierberri.w90files import WannierData
from wannierberri import System_R, run, Grid

wandata = WannierData.from_w90_files(seedname="wannier/graphene", files=["chk", "eig"])
system = System_R.from_wannierdata(wandata, periodic=[True, True, False])
system.double_spin()
system.set_pointgroup(["C6z", "Inversion", "TimeReversal"])

system.to_npz("graphene-system")


from wannierberri.calculators.static import DOS, CumDOS
EF = np.linspace(-5, -3, 2001)
grid = Grid(system, NKFFT=(18,18,1), NKdiv=(10,10,1))
calculators = {"dos": DOS(Efermi=EF, tetra=True), "cumdos": CumDOS(Efermi=EF, tetra=True)}
result = run(system=system, calculators=calculators, grid=grid)
from matplotlib import pyplot as plt
cumdos = result.results["cumdos"].data

EF0 = EF[np.argmin(np.abs(cumdos - 2))]


plt.plot(EF, cumdos)
plt.title(f"Cumulative DOS, EF = {EF0:.3f} eV")
plt.axvline(EF0, color="red", linestyle="--", label=f"EF={EF0:.2f} eV")
plt.xlabel("Energy (eV)")
plt.ylabel("Cumulative DOS")
plt.legend()
plt.savefig("cumdos.png")


path, bands = system.get_bandstructure(dk=0.01)
bands.plot_path_fat(path=path, save_file="bands.png")


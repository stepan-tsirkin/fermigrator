from fermigrator.database import ContourDatabase
import numpy as np
from fermigrator.contours2D import get_contours_and_WFs
import wannierberri as wb
import ray
# ray.init()

try:
    db = ContourDatabase.read("contours")
except FileNotFoundError:
    db = ContourDatabase("contours")
    from wannierberri.system import System_R
    system = System_R.from_npz("graphene-system")
    db.set_system(system)


# path, bands = db.system.get_bandstructure()
# bands.plot_path_fat(path=path, save_file="bands.png")
grid = wb.Grid(db.system, NK=(300, 300, 1), NKFFT=(10, 10, 1))
db.evaluate_E_grid(grid=grid, ignore_existing=True)
EF0 = np.load("efermi.npz")["EF"]
db.set_fermi_surfaces(
    Efermi_list=np.linspace(EF0 - 3, EF0 + 3, 61),
    ignore_existing=False
)

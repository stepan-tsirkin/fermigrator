from fermigrator.database import ContourDatabase
import wannierberri as wb

try:
    db = ContourDatabase.read("contours")
except FileNotFoundError as e:
    print(f"ContourDatabase not found, creating a new one...: {e}")
    db = ContourDatabase("contours")
    from wannierberri.system import System_R
    system = System_R.from_npz("system")
    db.set_system(system)
    grid = wb.Grid(db.system, NKdiv=4, NKFFT=10)
    import ray
    ray.init()
    db.evaluate_E_grid(grid=grid, ignore_existing=True, dim=3)


print((f"real lattice vectors: {db.system.real_lattice}"))
print((f"reciprocal lattice vectors: {db.system.recip_lattice}"))

energies_grid, rec_lattice = db.get_E_grid()
print(f"Energies_Grid shape: {energies_grid.shape}, Rec. lattice: {rec_lattice}")
# path, bands = db.system.get_bandstructure()
# bands.plot_path_fat(path=path, save_file="bands.png")
db.set_fermi_surfaces(Efermi_list=[8.01636],
                      ignore_existing=True,
                      set_triangle_neighbours=True,
                      )

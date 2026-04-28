import numpy as np
from integrateFermi.contours2D import get_contours_and_WFs
# import ray
# ray.init()
from packaging import version
NEW_PYTHTB_VERSION = version.parse("2.0.0")


def Haldane_ptb(delta=0.2, hop1=-1.0, hop2=0.15, phi=np.pi / 2):
    """same as :func:`~wannierberri.models.Haldane_tbm`, but uses `PythTB <http://www.physics.rutgers.edu/pythtb/>`__

    Notes
    -----
    PythTB should be installed to use this (`pip install pythtb`)
    """
    import pythtb
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1. / 3., 1. / 3.], [2. / 3., 2. / 3.]]
    if version.parse(pythtb.__version__) < NEW_PYTHTB_VERSION:
        my_model = pythtb.tb_model(2, 2, lat, orb)
    else:
        lattice = pythtb.Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
        my_model = pythtb.TBModel(lattice)

    t2 = hop2 * np.exp(1.j * phi)
    t2c = t2.conjugate()

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(hop1, 0, 1, [0, 0])
    my_model.set_hop(hop1, 1, 0, [1, 0])
    my_model.set_hop(hop1, 1, 0, [0, 1])
    my_model.set_hop(t2, 0, 0, [1, 0])
    my_model.set_hop(t2, 1, 1, [1, -1])
    my_model.set_hop(t2, 1, 1, [0, 1])
    my_model.set_hop(t2c, 1, 1, [1, 0])
    my_model.set_hop(t2c, 0, 0, [1, -1])
    my_model.set_hop(t2c, 0, 0, [0, 1])

    return my_model



from integrateFermi.database import ContourDatabase
try:
    db = ContourDatabase.read("contours")
except FileNotFoundError:
    db = ContourDatabase("contours")
    from wannierberri.system import System_PythTB
    system = System_PythTB(Haldane_ptb(delta=0, phi=0., hop1=1., hop2=0.))
    db.set_system(system)


path, bands = db.system.get_bandstructure()
bands.plot_path_fat(path=path, save_file="bands.png")

db.evaluate_E_grid(grid=100, ignore_existing=True)

get_contours_and_WFs(contours_db=db,
                     Efermi_list=np.linspace(-1, 1, 41),
                     ignore_existing=False
                     )
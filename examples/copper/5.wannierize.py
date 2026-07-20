from wannierberri.w90files import WannierData
from wannierberri.symmetry.projections import Projection, ProjectionsSet
import numpy as np
import pickle
seed = "Cu"
wandata = WannierData.from_npz(f"{seed}", files=["mmn", "eig", "symmetrizer", "chk"])
bandstructure = pickle.load(open(f"{seed}-bandstructure.pkl", "rb"))

sg = bandstructure.spacegroup


positions = sg.positions

proj_d = Projection(
    position_num=positions,
    orbital='d',
    spacegroup=sg,
    rotate_basis=True
)

proj_s = Projection(
    position_num=[[1 / 2, 0, 0]],
    orbital='s',
    spacegroup=sg,
    rotate_basis=True
)

proj_set = ProjectionsSet(projections=[proj_d, proj_s])
wandata.set_projections(projections=proj_set, bandstructure=bandstructure)
wandata.symmetrizer.to_npz(f"{seed}.sawf.npz")
wandata.amn.to_npz(f"{seed}.amn.npz")

wandata.wannierise(
    froz_min=-10,
    froz_max=20,
    outer_min=-10,
    outer_max=np.inf,
    num_iter=100,
    conv_tol=1e-10,
    print_progress_every=20,
    sitesym=True,
    localise=True,
    savechk=False,
)
wandata.chk.to_npz(f"{seed}.chk.npz")

import pickle
from wannierberri.w90files import WannierData
from gpaw import GPAW
seed = "Cu"

calc_scf = GPAW(f'{seed}-nscf.gpw', txt=None)
calc_nscf_irred = GPAW(f'{seed}-nscf.gpw', txt=None)
wandata, bandstructure = WannierData.from_gpaw(
    calculator=calc_nscf_irred,
    spin_channel=0,
    irreducible=True,
    files=["mmn", "eig", "symmetrizer"],
    unitary_params=dict(error_threshold=0.1,
                        warning_threshold=0.01,
                        nbands_upper_skip=8),
    return_bandstructure=True
)
pickle.dump(bandstructure, open(f"{seed}-bandstructure.pkl", "wb"))

wandata.to_npz(f"{seed}")

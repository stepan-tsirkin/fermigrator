from collections.abc import Iterable
from wannierberri.calculators.tabulate import Tabulator
from wannierberri.result import KBandResult
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.grid import Path
import numpy as np

class NeverTransform:

    def __init__(self):
        pass

    def __call__(self, res):
        raise RuntimeError("This transform should never be called")
    
nevertransform = NeverTransform()

class TabulatorWaveFunction(Tabulator):

    comment = "Tabulator for band wavefunction"

    def __init__(self, ibands, **kwargs):
        self.iband = ibands

    def __call__(self, data_k):
        U = data_k.UU_K[:,:, self.iband]
        return KBandResult(U, transformTR=nevertransform, transformInv=nevertransform)
        

def get_wavefunction_on_kpoints(system, kpoints, ibands, **kwargs_run):
    assert isinstance(ibands, int) or (isinstance(ibands, Iterable) and all(isinstance(i, int) for i in ibands)), \
        "ibands should be an int or an iterable of ints"
    tabulator = TabulatorWaveFunction(ibands=ibands)
    kpoints = np.array(kpoints)
    if kpoints.ndim == 1:
        kpoints = kpoints[None, :]
    if kpoints.shape[1] ==2:
        kpoints = np.concatenate([kpoints, np.zeros((kpoints.shape[0], 1))], axis=1)
    if kpoints.shape[1] != 3:       
        raise ValueError("kpoints should have shape (N, 3), (N,2) or (3,) or (2,)")
    path = Path(recip_lattice=system.recip_lattice, k_list=kpoints)
    res = evaluate_k_path(system=system, path=path, 
                             tabulators={"wavefunction": tabulator},
                             **kwargs_run
            )
    data = res.results["wavefunction"].data
    if isinstance(ibands, Iterable):
        assert data.ndim == 3 and data.shape[2] == len(ibands)
        return data.swapaxes(1,2)
    else:
        assert data.ndim == 2
        return data
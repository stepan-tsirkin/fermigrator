from collections.abc import Iterable
from wannierberri.calculators.tabulate import Tabulator
from wannierberri.result import KBandResult
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.grid import Path
import numpy as np


class NeverTransform:
    """Sentinel that prevents WannierBerri from symmetrising the wavefunction.

    WannierBerri applies time-reversal / inversion transforms when unfolding
    k-points.  For the raw U_k matrix we want no such mixing, so we pass this
    object as the transform and have it raise if accidentally invoked.
    """

    def __init__(self):
        pass

    def __call__(self, res):
        raise RuntimeError("This transform should never be called")

    def __eq__(self, other):
        return isinstance(other, NeverTransform)


nevertransform = NeverTransform()


class TabulatorWaveFunction(Tabulator):
    """WannierBerri tabulator that extracts U_K[:, :, iband] at each k-point.

    U_K is the unitary rotation from the Wannier basis to the Bloch eigenstates.
    Selecting a single band index gives the Wannier-gauge wavefunction column
    needed for matrix-element evaluations on the Fermi surface.
    """

    comment = "Tabulator for band wavefunction"

    def __init__(self, ibands, **kwargs):
        self.iband = ibands

    def __call__(self, data_k):
        U = data_k.UU_K[:, :, self.iband]
        if isinstance(self.iband, Iterable):
            assert U.ndim == 3 and U.shape[2] == len(self.iband)
            U =  U.swapaxes(1, 2)
        else:
            assert U.ndim == 2
        return KBandResult(U, transformTR=nevertransform, transformInv=nevertransform)


def get_wavefunction_on_kpoints(system, kpoints, ibands, **kwargs_run):
    """Evaluate Wannier-interpolated Bloch wavefunctions at arbitrary k-points.

    Uses WannierBerri's path evaluator (Fourier interpolation via Wannier
    functions) to obtain U_K at k-points that are not on the regular FFT grid.

    Parameters
    ----------
    system : wannierberri.System_R
    kpoints : array-like, shape (N, 2) or (N, 3) or (2,) or (3,)
        k-points in reduced coordinates.  2D inputs are zero-padded to 3D.
    ibands : int or iterable of int
        Band index/indices to extract.

    Returns
    -------
    ndarray
        If ibands is int  : shape (N, num_wann)   — U_K[:, :, iband]
        If ibands is list : shape (N, len(ibands), num_wann)
    """
    assert isinstance(ibands, int) or (isinstance(ibands, Iterable) and all(isinstance(i, int) for i in ibands)), \
        "ibands should be an int or an iterable of ints"
    tabulator = TabulatorWaveFunction(ibands=ibands)
    kpoints = np.array(kpoints)
    if kpoints.ndim == 1:
        kpoints = kpoints[None, :]
    if kpoints.shape[1] == 2:
        kpoints = np.concatenate(
            [kpoints, np.zeros((kpoints.shape[0], 1))], axis=1)
    if kpoints.shape[1] != 3:
        raise ValueError(
            "kpoints should have shape (N, 3), (N,2) or (3,) or (2,)")
    path = Path(recip_lattice=system.recip_lattice, k_list=kpoints)
    res = evaluate_k_path(system=system, path=path,
                          tabulators={"wavefunction": tabulator},
                          **kwargs_run
                          )
    return res.results["wavefunction"].data

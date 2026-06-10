import numpy as np
from wannierberri.utility import cached_einsum
from scipy.constants import hbar

def get_Wkk_Efermi(contours_db, EF):
    """Computes the skew scattering matrix Wkk

    W_{nk,mk'} = sum_{rq} Im[V_{nk,mk'} V_{mk',rq} V_{rq,nk}] w_k * w_{k'} w_q

    where 
    - n, m, r are band indices
    - k, k', q are k-point indices
    - V_{nk,mk'} is the scattering matrix element between states (n,k) and (m,k')
    - w_k is the weight of the k-point in the contour integration for the Fermi energy EF

    Parameters
    ----------
    contours_db : ContoursDB
        The database containing the contours and the scattering matrices
    EF : float
        The Fermi energy at which to compute the skew scattering matrix 
    """
    files_contour = contours_db.get_files_Efermi("contour", EF)
    contour_dict = {contours_db.split_filename(f)["ib"]: f for f in files_contour}
    skew_dict = {}
    ib_list = sorted(contour_dict.keys())
    for n in ib_list:
        contour1 = np.load(contour_dict[n])
        wn = contour1["weights"]
        for m in ib_list:
            contour2 = np.load(contour_dict[m])
            wm = contour2["weights"]
            Vkk_12 = contours_db.get_data(
                typ="Vkk", ib1=n, ib2=m, EF=EF, none_if_missing=False)["Vkk"]
            Wkk = 0
            for r in ib_list:
                contour3 = np.load(contour_dict[r])
                wq = contour3["weights"]
                Vkk_23 = contours_db.get_data(typ="Vkk", ib1=m, ib2=r, EF=EF, none_if_missing=False)["Vkk"]
                Vkk_31 = contours_db.get_data(typ="Vkk", ib1=r, ib2=n, EF=EF, none_if_missing=False)["Vkk"]
                Wkk += cached_einsum('k,kp,p,pq,q,qk->kp', wn, Vkk_12, wm, Vkk_23, wq, Vkk_31).imag
            contours_db.set_data("Wkk", dict(Wkk=Wkk), ib1=n, ib2=m, EF=EF)
    return skew_dict






# Let's figure out the factor for AHC. 
# Internally, we calculate sigma_{ab} = Sum_{k,k',k''} Im[ V_{nk,mk'} V_{mk',rq} V_{rq,nk} ] 
#     w_{k} w_{k'} w_q * (v_{nk}^a  v_{mk'}^b / gamma_{nk} / gamma_{mk'})
#   w_k are in units eV^{-1} Angstrom^{-2} (from the contour integration)
#   v_{mk} are in units eV* Angstrom
#   V_{nk,mk'} are in units of eV
# So, the sum is in units of eV^3 * 1/eV^3 * (eV*Angstrom)^2 * 1/eV^2 = Angstrom^2



def get_AHC(contours_db, EF, gamma_0 = 1e-8):
    """Computes the anomalous Hall conductivity (AHC) from the skew scattering matrix Wkk

    AHC = (e^2 / hbar) * sum_{nk,mk'} W_{nk,mk'} * (v_{nk} x v_{mk'} * tau_{nk} * tau_{mk'})

    where 
    - n, m are band indices
    - k, k' are k-point indices
    - W_{nk,mk'} is the skew scattering matrix element between states (n,k) and (m,k')
    - v_{nk} is the group velocity of state (n,k)

    Parameters
    ----------
    contours_db : ContoursDB
        The database containing the contours and the skew scattering matrices
    EF : float
        The Fermi energy at which to compute the AHC 
    """
    files_contour = contours_db.get_files_Efermi("contour", EF)
    contour_dict = {contours_db.split_filename(f)["ib"]: f for f in files_contour}
    ahc = 0
    ib_list = sorted(contour_dict.keys())
    for n in ib_list:
        contour1 = np.load(contour_dict[n])
        vn = contour1["grad"]
        gamma_n = contours_db.get_data(typ="linewidth", ib=n, EF=EF, none_if_missing=False)["linewidth"]
        gamma_n = np.maximum(gamma_n, gamma_0)
        vtau_n = vn/ gamma_n[:, None]
        for m in ib_list:
            contour2 = np.load(contour_dict[m])
            vm = contour2["grad"]
            gamma_m = contours_db.get_data(typ="linewidth", ib=m, EF=EF, none_if_missing=False)["linewidth"]
            gamma_m = np.maximum(gamma_m, gamma_0)
            vtau_m = vm/ gamma_m[:, None]
            Wkk = contours_db.get_data(typ="Wkk", ib1=n, ib2=m, EF=EF, none_if_missing=False)["Wkk"]
            ahc += cached_einsum('ka,kp,pb->ab', vtau_n, Wkk, vtau_m).real
    return ahc * factor_ahc






def get_multipole_vtau(contours_db, EF, ib, gamma_0 = 1e-8):
    # print (f"Calculating multipole vertex on contour for Efermi={EF} and band index ib={ib} with gamma_0={gamma_0}")
    pauli_z = np.array([[1, 0], [0, -1]])
    contour = contours_db.get_data("contour", ib=ib, EF=EF, none_if_missing=False)
    weight = contour["weights"]
    multipole_eigen = contours_db.get_data(typ="multipole-eigen", ib=ib, EF=EF)
    mult_e = multipole_eigen["eigenvalues"]
    mult_W = multipole_eigen["eigenvectors"]
    nspin = mult_W.shape[2]
    assert nspin == 2, f"Expected 2 spin channels in multipole eigenvectors, but got {nspin}"
    grad = contour["grad"]
    gamma = contours_db.get_data(typ="linewidth-multipole", ib=ib, EF=EF, none_if_missing=False)["linewidth"]
    gamma = np.maximum(gamma, gamma_0)
    vtau = grad[:,None, :] / gamma[:,  :, None]
    vertex_velocity = cached_einsum('lks, k, mks , m, ksa-> lma', 
                                    mult_W.conj(), weight, mult_W, mult_e, vtau)
    vertex_spin_velocity = cached_einsum('lks, k, mks , m, ss, ksa-> lma', 
                                         mult_W.conj(), weight, mult_W, mult_e, pauli_z, vtau)
    contours_db.set_data("multipole-vtau", dict(vertex_velocity=vertex_velocity, vertex_spin_velocity=vertex_spin_velocity), ib=ib, EF=EF)
    return vertex_velocity, vertex_spin_velocity


def get_all_multipole_vtau(contours_db, EF=None, gamma_0 = 1e-8):
    if EF is None:
        EFs = contours_db.get_all_Efermi_float()
        for EF in EFs:
            get_all_multipole_vtau(contours_db, EF=EF, gamma_0=gamma_0)
        return
    iblist = contours_db.get_all_bands(EF)
    # print (f"Calculating multipole vertex on contours for Efermi={EF} and all bands {iblist} with gamma_0={gamma_0}")
    vertex_velocity_sum = 0
    vertex_spin_velocity_sum = 0
    for ib in iblist:
        vle, spinvel = get_multipole_vtau(contours_db, EF, ib, gamma_0=gamma_0)
        vertex_velocity_sum += vle
        vertex_spin_velocity_sum += spinvel
    contours_db.set_data("multipole-vtau-sum", dict(vertex_velocity=vertex_velocity_sum, vertex_spin_velocity=vertex_spin_velocity_sum), EF=EF)
    return vertex_velocity_sum, vertex_spin_velocity_sum

def get_AHC_multipole(contours_db, EF, gamma_0 = 1e-8):
    vertex_v = contours_db.get_data("multipole-vtau-sum", EF=EF)
    vtau = vertex_v["vertex_velocity"]
    # vertex_sv = vertex_v["vertex_spin_velocity"]
    vertex = contours_db.get_data("multipole-vertex-sum", EF=EF)["vertex"]
    ahc = cached_einsum('lma, mnb, nl->ab', vtau, vtau, vertex).imag
    return ahc

def get_SHC_multipole(contours_db, EF, gamma_0 = 1e-8):
    vertex_v = contours_db.get_data("multipole-vtau-sum", EF=EF)
    vtau = vertex_v["vertex_velocity"]
    vertex_sv = vertex_v["vertex_spin_velocity"]
    vertex = contours_db.get_data("multipole-vertex-sum", EF=EF)["vertex"]
    # print (f"{EF=}, {gamma_0=}, vertex shape={vertex.shape}, vtau shape={vtau.shape}, vertex_sv shape={vertex_sv.shape}")
    shc = cached_einsum('lma, mnb, nl->ab', vertex_sv, vtau, vertex).imag
    return shc
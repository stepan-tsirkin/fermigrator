import numpy as np
from wannierberri.utility import cached_einsum


def get_linewidth_Efermi(contours_db, EF):
    """Compute quasiparticle linewidths on the Fermi surface via Fermi's golden rule.

    For each k-point on the Fermi surface of band `ib`, evaluates:

        Γ(k) = Σ_{ib2, k'} |V_{ib,ib2}(k, k')|² w(k')

    where the sum over k' runs over all Fermi surface points of all bands at
    the same Fermi level and w(k') are the contour integration weights
    (proportional to the segment length / |∇E|, approximating the delta function
    δ(E_k' - E_F)).  The pre-computed Vkk files (band matrix elements of the
    scattering matrix on the contour) are read from `contours_db`.

    Parameters
    ----------
    contours_db : ContourDatabase
    EF : str or float
        Fermi energy label as stored in the database.

    Returns
    -------
    dict {band_index: ndarray (N_k,)}
        Linewidth Γ(k) in the same units as the scattering matrix (eV if
        Vkk was stored in eV).
    """
    files_contour = contours_db.get_files_Efermi("contour", EF)
    contour_dict = {contours_db.split_filename(f)["ib"]: f for f in files_contour}
    linewidth_dict = {}
    for ib, file_contour in contour_dict.items():
        # contour1 = np.load(file_contour)
        linewidth_dict[ib] = 0
        for ib2 in contour_dict.keys():
            print(
                f"Calculating linewidth for Efermi={EF} using contours for ib1={ib} and ib2={ib2}")
            w = np.load(contour_dict[ib2])["weights"]
            Vkk = contours_db.get_data(
                typ="Vkk", ib1=ib, ib2=ib2, EF=EF, none_if_missing=False)["Vkk"]
            Vkk_conj = contours_db.get_data(
                typ="Vkk", ib1=ib2, ib2=ib, EF=EF, none_if_missing=False)["Vkk"]
            assert np.allclose(
                Vkk, Vkk_conj.conj().transpose(1, 0, 3, 2)), f"Vkk file for ib1={ib}, ib2={ib2} and Efermi={EF} is not the conjugate transpose of the Vkk file for ib1={ib2}, ib2={ib} and Efermi={EF}, skipping this pair"
            linewidth = cached_einsum('kqst,q,qkts->ks', Vkk, w, Vkk_conj).real
            print(f"{Vkk.shape=}, {w.shape=}, {Vkk_conj.shape=}, linewidth shape={linewidth.shape}")
            print(
                f"Linewidth for ib1={ib}, ib2={ib2} and Efermi={EF} has min {linewidth.min()} and max {linewidth.max()}")
            # contours_db.set_data("linewidth", dict(linewidth=linewidth, kpoints=contour1["kpoints"], weights=contour1["weights"]),
            #                      ib1=ib, ib2=ib2, EF=EF)

            linewidth_dict[ib] += np.real(linewidth)
        contours_db.set_data("linewidth", dict(linewidth=linewidth_dict[ib]), ib=ib, EF=EF)
    return linewidth_dict


def getDOS(contours_db, EF):
    """Compute the density of states at the Fermi level.

    Sums the contour integration weights over all k-points and bands:

        DOS(E_F) = Σ_{ib, k} w_{ib}(k)

    where each weight w(k) ∝ segment_length / |∇_k E| approximates the
    contribution of that k-point to the delta function δ(E_k - E_F).

    Parameters
    ----------
    contours_db : ContourDatabase
    EF : str or float

    Returns
    -------
    float
        Total DOS in states / eV / unit-cell.
    """
    files_contour = contours_db.get_files_Efermi("contour", EF)
    dos = 0
    for file_contour in files_contour:
        contour = np.load(file_contour)
        dos += np.sum(contour["weights"])
    return dos

def getOhmic(contours_db, EF):

    files_contour = contours_db.get_files_Efermi("contour", EF)
    ohmic = 0
    for file_contour in files_contour:
        contour = np.load(file_contour)
        w = contour["weights"]
        vn = contour["grad"]
        ohmic += np.einsum('k,ka,kb->ab', w, vn, vn).real
    from wannierberri.factors import factor_ohmic
    ohmic *= factor_ohmic
    return ohmic


def get_linewidth_multipole_Efermi(contours_db, EF):
    """Compute linewidths using the multipole decomposition of the scattering vertex.

    Equivalent to `get_linewidth_Efermi` but avoids loading the full (N_k × N_k)
    Vkk matrix.  The multipole decomposition expresses the scattering vertex as:

        V(k, k') = Σ_l λ_l W_l*(k) W_l(k')

    where W_l(k) = Σ_{Ra} φ_l[R,a] e^{-ik·R} u_a(k) and φ_l are the
    eigenvectors of V^{RR'} (orthonormal in R-space, NOT on the Fermi surface).

    Fermi's golden rule then gives:

        Γ(k) = Σ_{k'} |V(k,k')|² w(k')
             = Σ_{l,m} λ_l λ_m W_l*(k) W_m(k) [Σ_{k'} W_l(k') W_m*(k') w(k')]
             = Σ_{l,m} λ_l W_l*(k) W_m(k) vertex[l,m]

    `projector[k,l,m] = W_l*(k) W_m(k) λ_m` and
    `vertex[l,m] = Σ_{k'} W_l*(k') W_m(k') λ_m w(k')` are pre-computed by
    `ScatteringMatrix.get_multipole_on_contour`.

    Parameters
    ----------
    contours_db : ContourDatabase
    EF : str or float

    Returns
    -------
    dict {band_index: ndarray (N_k,)}
    """
    linewidths_dict = {}
    vertex = contours_db.get_data("multipole-vertex-sum", EF=EF)["vertex"]
    for ib in contours_db.get_all_bands(EF):
        multipole_eigen = contours_db.get_data(typ="multipole-eigen", ib=ib, EF=EF)
        mult_e = multipole_eigen["eigenvalues"]
        mult_W = multipole_eigen["eigenvectors"]
        linewidths_dict[ib] = cached_einsum(' mks, lks, l, lm-> ks',
                                            mult_W.conj(), mult_W, mult_e, vertex).real
        contours_db.set_data("linewidth-multipole", dict(linewidth=linewidths_dict[ib]), ib=ib, EF=EF)
    return linewidths_dict

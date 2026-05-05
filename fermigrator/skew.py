import numpy as np
from wannierberri.utility import cached_einsum

def get_skew_Efermi(contours_db, EF):
    """Compute quasiparticle skew scattering on the Fermi surface via Fermi's golden rule.

    For each k-point on the Fermi surface of band `ib`, evaluates:

        Γ(ib1, k) = Σ_{ib2, ib3, k',k''} V_{ib1,ib2}(k, k') V_{ib2,ib3}(k', k'') V_{ib3,ib1}(k'', k) w(k') w(k'')

    where the sum over k' and k'' runs over all Fermi surface points of all bands at the same Fermi level and w(k') are the contour integration weights
    (proportional to the segment length / |∇E|, approximating the delta function δ(E_k' - E_F)).  The pre-computed Vkk files (band matrix elements of the scattering matrix on the contour) are read from `contours_db`.    

    
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
    skew_dict = {}
    for ib, file_contour in contour_dict.items():
        contour1 = np.load(file_contour)
        skew_dict[ib] = np.zeros(
            contour1["kpoints"].shape[0], dtype=float)
        for ib2 in contour_dict.keys():
            w2 = np.load(contour_dict[ib2])["weights"]
            Vkk12 = contours_db.get_data(
                typ="Vkk", ib1=ib, ib2=ib2, EF=EF, none_if_missing=False)["Vkk"]
            for ib3 in contour_dict.keys():
                print(f"Calculating skew for Efermi={EF} using contours for ib1={ib}, ib2={ib2} and ib3={ib3}")
                w3 = np.load(contour_dict[ib3])["weights"]
                Vkk_23 = contours_db.get_data(
                    typ="Vkk", ib1=ib2, ib2=ib3, EF=EF, none_if_missing=False)["Vkk"]
                Vkk_31 = contours_db.get_data(
                    typ="Vkk", ib1=ib3, ib2=ib, EF=EF, none_if_missing=False)["Vkk"]
                skew = cached_einsum('kq,q,qp,p,pk->k', Vkk12, w2, Vkk_23, w3, Vkk_31).real
                print(f"Skew for ib1={ib}, ib2={ib2} and Efermi={EF} has min {skew.min()} and max {skew.max()}")
                contours_db.set_data("skew", dict(skew=skew, kpoints=contour1["kpoints"], weights=contour1["weights"]),
                                 ib1=ib, ib2=ib2, ib3=ib3, EF=EF)
                skew_dict[ib] += skew
    return skew_dict


def get_skew_multipole_Efermi(contours_db, EF):
    """
    same as get_skew_Efermi but using the multipole expansion of the scattering matrix
    """
    
    skew_dict = {}
    files_contour = contours_db.get_files_Efermi("contour", EF)
    files_vertices = contours_db.get_files_Efermi("multipole-vertex", EF)
    for file_contour in files_contour:
        ib = contours_db.split_filename(file_contour)["ib"]
        contour = np.load(file_contour)
        skew_dict[ib] = np.zeros(
            contour["kpoints"].shape[0], dtype=float)
        vertex_file = contours_db.get_data(
            typ="multipole-vertex", ib=ib, EF=EF)
        projector = vertex_file["projector"]
        for vertex2_file in files_vertices:
            jb = contours_db.split_filename(vertex2_file)["ib"]
            vertex2 = np.load(vertex2_file)["vertex"]
            print(f"{vertex2.shape=}, {projector.shape=}")
            for vertex3_file in files_vertices:
                lb = contours_db.split_filename(vertex3_file)["ib"]
                vertex3 = np.load(vertex3_file)["vertex"]
                print(f"{vertex3.shape=}, {projector.shape=}")
                skew = cached_einsum('klm, mn, nl -> k', projector, vertex2, vertex3)
                skew_dict[ib] += np.real(skew)
            contours_db.set_data("skew-multipole", dict(linewidth=skew, kpoints=contour["kpoints"], weights=contour["weights"]),
                                 ib=ib, ib2=jb, ib3=lb, EF=EF)
    return skew_dict

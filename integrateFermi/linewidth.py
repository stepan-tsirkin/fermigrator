import numpy as np
from wannierberri.utility import cached_einsum


def get_linewidth_Efermi(contours_db, EF):
    files_contour = contours_db.get_files_Efermi("contour", EF)
    contour_dict = {contours_db.split_filename(
        f)["ib"]: f for f in files_contour}
    linewidth_dict = {}
    for ib, file_contour in contour_dict.items():
        contour1 = np.load(file_contour)
        linewidth_dict[ib] = np.zeros(
            contour1["kpoints"].shape[0], dtype=float)
        for ib2 in contour_dict.keys():
            print(
                f"Calculating linewidth for Efermi={EF} using contours for ib1={ib} and ib2={ib2}")
            w = np.load(contour_dict[ib2])["weights"]
            Vkk = contours_db.get_data(
                typ="Vkk", ib1=ib, ib2=ib2, EF=EF, none_if_missing=False)["Vkk"]
            Vkk_conj = contours_db.get_data(
                typ="Vkk", ib1=ib2, ib2=ib, EF=EF, none_if_missing=False)["Vkk"]
            assert np.allclose(
                Vkk, Vkk_conj.conj().T), f"Vkk file for ib1={ib}, ib2={ib2} and Efermi={EF} is not the conjugate transpose of the Vkk file for ib1={ib2}, ib2={ib} and Efermi={EF}, skipping this pair"
            # linewidth = cached_einsum('kq, q, qk -> k', file_Vkk["Vkk"], contour2["weights"], file_Vkk_conj["Vkk"])
            linewidth = np.einsum('kq,q,qk->k', Vkk, w, Vkk_conj).real
            print(
                f"Linewidth for ib1={ib}, ib2={ib2} and Efermi={EF} has min {linewidth.min()} and max {linewidth.max()}")
            contours_db.set_data("linewidth", dict(linewidth=linewidth, kpoints=contour1["kpoints"], weights=contour1["weights"]),
                                 ib1=ib, ib2=ib2, EF=EF)

            linewidth_dict[ib] += np.real(linewidth)
    return linewidth_dict


def get_linewidth_multipole_Efermi(contours_db, EF):
    linewidths_dict = {}
    files_contour = contours_db.get_files_Efermi("contour", EF)
    files_vertices = contours_db.get_files_Efermi("multipole-vertex", EF)
    for file_contour in files_contour:
        ib = contours_db.split_filename(file_contour)["ib"]
        contour = np.load(file_contour)
        linewidths_dict[ib] = np.zeros(
            contour["kpoints"].shape[0], dtype=float)
        vertex_file = contours_db.get_data(
            typ="multipole-vertex", ib=ib, EF=EF)
        projector = vertex_file["projector"]
        for vertex2_file in files_vertices:
            jb = contours_db.split_filename(vertex2_file)["ib"]
            vertex2 = np.load(vertex2_file)["vertex"]
            print(f"{vertex2.shape=}, {projector.shape=}")
            linewidths = cached_einsum('klm, ml -> k', projector, vertex2)
            linewidths_dict[ib] += np.real(linewidths)
            contours_db.set_data("linewidth-multipole", dict(linewidth=linewidths, kpoints=contour["kpoints"], weights=contour["weights"]),
                                 ib=ib, ib2=jb, EF=EF)
    return linewidths_dict

import numpy as np
from wannierberri.utility import cached_einsum
from wannierberri.fourier.rvectors import Rvectors


class ScatteringMatrix:

    """
    Class for the scattering matrix

    """

    def __init__(self, rvec, Vrrab=None, num_wann=None,
                 ):
        self.rvec = rvec
        if Vrrab is not None:
            self.Vrrab = Vrrab
            assert Vrrab.shape == (rvec.nRvec, rvec.nRvec, self.num_wann,
                                   self.num_wann), f"Vrrab should have shape (nRvec, nRvec, num_wann, num_wann), but got {Vrrab.shape}"
        elif num_wann is not None:
            self.Vrrab = np.zeros(
                (rvec.nRvec, rvec.nRvec, num_wann, num_wann), dtype=complex)
        else:
            raise ValueError("Either Vrrab or num_wann should be provided")

    @classmethod
    def from_Vkk(cls, Vkkmn, gauge="wannier",
                 center_red=None,
                 wannier_centers_red=None,
                 real_lattice=None,
                 chk=None, selected_bands=None):
        if isinstance(Vkkmn, str):
            Vkkmn = np.load(Vkkmn)["V"]
        if isinstance(chk, str):
            from wannierberri.w90files.chk import CheckPoint
            if chk.endswith(".chk"):
                chk = CheckPoint.from_w90_file(chk[:-4])
            elif chk.endswith(".npz"):
                chk = CheckPoint.from_npz(chk)
            else:
                chk = CheckPoint.from_w90_file(chk+".npz")
        assert gauge in [
            "wannier", "bloch"], f"Gauge must be either 'wannier' or 'bloch', but got {gauge}"
        assert Vkkmn.ndim == 4, f"Vkkmn must be a 4D array with shape (NK, NK, NB, NB), but got shape {Vkkmn.shape}"
        assert Vkkmn.shape[0] == Vkkmn.shape[
            1], f"The first two dimensions of Vkkmn must be the same, but got {Vkkmn.shape[0]} and {Vkkmn.shape[1]}"
        assert Vkkmn.shape[2] == Vkkmn.shape[
            3], f"The last two dimensions of Vkkmn must be the same, but got {Vkkmn.shape[2]} and {Vkkmn.shape[3]}"
        num_kpts = Vkkmn.shape[0]
        if gauge == "bloch":
            assert chk is not None, "Checkpoint file must be provided to transform the scattering matrix to the Wannier gauge"
            if selected_bands is not None:
                Vkkmn = Vkkmn[:, :, selected_bands, :][:, :, :, selected_bands]
            num_bands = Vkkmn.shape[2]
            assert chk.num_kpts == num_kpts, f"Number of k-points in the scattering matrix ({num_kpts}) does not match the number of k-points in the checkpoint ({chk.num_kpts})"
            assert chk.num_bands == num_bands, f"Number of bands in the scattering matrix ({num_bands}) does not match the number of bands in the checkpoint ({chk.num_bands})"
            assert chk.v_matrix is not None, "The checkpoint does not contain the v_matrix, which is needed to transform the scattering matrix to the Wannier gauge"
            Vkkab_w = np.zeros(
                (num_kpts, num_kpts, num_bands, chk.num_wann), dtype=complex)
            for ik in range(num_kpts):
                Vkkab_w[ik, :, :, :] = cached_einsum(
                    'kij,jb->kib', Vkkmn[:, ik, :, :], chk.v_matrix[ik])
            del Vkkmn
            Vkkab_wan = np.zeros(
                (num_kpts, chk.num_kpts, chk.num_wann, chk.num_wann), dtype=complex)
            for ik in range(num_kpts):
                Vkkab_wan[:, ik, :, :] = cached_einsum(
                    'kib,ia->kab', Vkkab_w[:, ik, :, :], chk.v_matrix[ik, :].conj())
            del Vkkab_w
        num_wann = Vkkab_wan.shape[2]
        if wannier_centers_red is None:
            if chk is None:
                wannier_centers_red = np.zeros(
                    (num_kpts, num_wann, 3), dtype=float)
            else:
                wannier_centers_red = chk.wannier_centers_cart @ np.linalg.inv(
                    chk.real_lattice)
        if center_red is None:
            center_red = np.zeros(3, dtype=float)
        if real_lattice is None:
            real_lattice = chk.real_lattice

        rvectors = Rvectors(
            lattice=chk.real_lattice,
            shifts_left_red=[center_red],
            shifts_right_red=wannier_centers_red,
        )
        rvectors.set_Rvec(mp_grid=chk.mp_grid)
        rvectors.set_fft_q_to_R(kpt_red=chk.kpt_red)
        Vabcrr = rvectors.qq_to_RR(Vkkab_wan[:, :, :, :, None])
        Vrrab = Vabcrr[:, :, 0, :, :].transpose(2, 3, 0, 1)
        return cls(rvec=rvectors, Vrrab=Vrrab)

    def set_VRR(self, Vrrab, irvec1, irvec2, ab=None):
        assert self.Vrrab is not None, "Vrrab is not set, please initialize the scattering matrix with Vrrab or num_wann first"
        ir1 = self.rvec.iR(irvec1)
        ir2 = self.rvec.iR(irvec2)
        if ab is None:
            self.Vrrab[ir1, ir2] = Vrrab
        else:
            self.Vrrab[ir1, ir2, ab[0], ab[1]] = Vrrab

    def as_dict(self):
        dict = {}
        for key in ["center_red",  "Vrrab",
                    "_multipole_eigenvalues", "_multipole_eigenvectors"]:
            if hasattr(self, key):
                if getattr(self, key) is not None:
                    dict[key] = getattr(self, key)
        if hasattr(self, "rvec") and self.rvec is not None:
            dict["iRvec"] = self.rvec.iRvec
            dict["wannier_centers_red"] = self.rvec.shifts_right_red
            dict["center_red"] = self.rvec.shifts_left_red[0]
            dict["real_lattice"] = self.rvec.lattice
        return dict

    def to_npz(self, filename):
        print(
            f"saving scattering matrix to {filename} with keys {list(self.as_dict().keys())}")
        np.savez(filename, **self.as_dict())

    @classmethod
    def from_dict(cls, dict):
        self = cls.__new__(cls)
        for key in ["Vrrab", "_multipole_eigenvalues", "_multipole_eigenvectors"]:
            if key in dict:
                setattr(self, key, dict[key])
                if key == "Vrrab":
                    num_wann = dict[key].shape[2]
                else:
                    num_wann = 0
        if "iRvec" in dict:
            self.rvec = Rvectors(
                iRvec=dict["iRvec"],
                lattice=dict["real_lattice"],
                shifts_left_red=[
                    dict.get("center_red", np.zeros(3, dtype=float))],
                shifts_right_red=dict.get(
                    "wannier_centers_red", np.zeros((num_wann, 3), dtype=float)),
            )
        return self

    @classmethod
    def from_npz(cls, filename):
        dict = np.load(filename)
        return cls.from_dict(dict)

    @property
    def num_wann(self):
        return self.Vrrab.shape[2]

    def get_on_kpoints(self,
                       kpt_red_left,
                       kpt_red_right,
                       u_left=None,
                       u_right=None,
                       ):
        rvectors = self.rvec.iRvec
        exp_left = np.exp(
            2j * np.pi * cached_einsum('Ri,kj->Rk', rvectors, kpt_red_left))
        exp_right = np.exp(-2j * np.pi *
                           cached_einsum('Ri,kj->Rk', rvectors, kpt_red_right))
        Vkkab = cached_einsum('Rk, Rrab, rq->kqab',
                              exp_left, self.Vrrab, exp_right)
        assert (u_left is None) == (
            u_right is None), "u_left and u_right must be both None or both not None"
        if u_left is not None:
            # TODO - is we combine with the previous - will it be faster?
            return cached_einsum('ka,kqab,qb->kq', u_left.conj(), Vkkab, u_right)
        else:
            return Vkkab

    def get_Vkk_on_contours(self, file1, file2,
                            contours_db=None,):
        with np.load(file1) as f1:
            kpoints1 = f1["kpoints"]
            wavefunctions1 = f1["wavefunctions"]
        with np.load(file2) as f2:
            kpoints2 = f2["kpoints"]
            wavefunctions2 = f2["wavefunctions"]

        V_on_contours = self.get_on_kpoints(kpt_red_left=kpoints1, kpt_red_right=kpoints2,
                                            u_left=wavefunctions1, u_right=wavefunctions2)
        if contours_db is not None:
            ib1 = contours_db.split_filename(file1)["ib"]
            ib2 = contours_db.split_filename(file2)["ib"]
            EF1 = contours_db.split_filename(file1)["EF"]
            EF2 = contours_db.split_filename(file2)["EF"]
            assert EF1 == EF2, f"Fermi levels in the two contour files do not match: {EF1} and {EF2} diff is {EF1-EF2}"
            contours_db.set_data("Vkk", dict(Vkk=V_on_contours),
                                 ib1=ib1, ib2=ib2, EF=EF1)
        return V_on_contours

    def get_Vkk_on_contours_all(self, contours_db, Efermi_list=None):
        if Efermi_list is None:
            Efermi_list = contours_db.get_all_Efermi()
        print(
            f"Calculating scattering matrix on contours for Efermi_list={Efermi_list}")
        for Efermi in Efermi_list:
            file_list = contours_db.get_files_Efermi("contour", Efermi)
            for f1 in file_list:
                for f2 in file_list:
                    print(
                        f"Calculating scattering matrix on contours for Efermi={Efermi} using files {f1} and {f2}")
                    self.get_Vkk_on_contours(f1, f2, contours_db=contours_db)

    @property
    def multipole_eigenvalues(self):
        if not hasattr(self, "_multipole_eigenvalues"):
            self.multipole_decomposition_RR()
        return self._multipole_eigenvalues

    @property
    def multipole_eigenvectors(self):
        if not hasattr(self, "_multipole_eigenvectors"):
            self.multipole_decomposition_RR()
        return self._multipole_eigenvectors

    def multipole_decomposition_RR(self, select_threshold=-1):
        assert self.Vrrab is not None, "Vrrab is not set, please set it first using set_RR"
        Vrarb = self.Vrrab.transpose(0, 2, 1, 3).reshape(
            self.rvec.nRvec*self.num_wann, self.rvec.nRvec*self.num_wann)
        e, v = np.linalg.eigh(Vrarb)
        srt = np.argsort(-abs(e))
        e = e[srt]
        if e[0] < 1e-15:
            print("Warning: the largest eigenvalue is smaller than 1e-15, which may indicate that the scattering matrix is not properly set or that the system is very weakly scattering")
            return np.zeros(0), np.zeros((0, self.rvec.nRvec, self.num_wann))
        nselect = max(np.where(e/e[0] > select_threshold)[0]+1)
        e = e[:nselect]
        v = v[:, srt[:nselect]]
        self._multipole_eigenvalues = e
        self._multipole_eigenvectors = v.T.reshape(
            nselect, self.rvec.nRvec, self.num_wann)
        return self._multipole_eigenvalues, self._multipole_eigenvectors

    def get_multipole_on_contour(self, file,
                                 contours_db=None):
        with np.load(file) as f:
            kpoints = f["kpoints"]
            wavefunctions = f["wavefunctions"]
            weight = f["weights"]

        v = self.multipole_eigenvectors
        e = self.multipole_eigenvalues

        exp = np.exp(-2j * np.pi * cached_einsum('Ri,kj->Rk',
                     self.rvec.iRvec, kpoints))
        W = cached_einsum('lRa, Rk, ka -> lk', v, exp, wavefunctions)
        vertex = cached_einsum('lk, k, mk , m-> lm', W.conj(), weight, W, e)
        projector = cached_einsum('lk, mk, m -> klm', W.conj(), W, e)
        if contours_db is not None:
            ib = contours_db.split_filename(file)["ib"]
            EF = contours_db.split_filename(file)["EF"]
            contours_db.set_data("multipole-vertex", dict(vertex=vertex, projector=projector, kpoints=kpoints),
                                 ib=ib, EF=EF)
        return vertex, projector

    def get_multipole_on_contours_all(self, contours_db, Efermi_list=None):
        if Efermi_list is None:
            Efermi_list = contours_db.get_all_Efermi()
        print(
            f"Calculating multipole vertex on contours for Efermi_list={Efermi_list}")
        for Efermi in Efermi_list:
            file_list = contours_db.get_files_Efermi("contour", Efermi)
            for f in file_list:
                print(
                    f"Calculating multipole vertex on contour for Efermi={Efermi} using file {f}")
                self.get_multipole_on_contour(f, contours_db=contours_db)


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
                Vkk, Vkk_conj), f"Vkk file for ib1={ib}, ib2={ib2} and Efermi={EF} is not the conjugate transpose of the Vkk file for ib1={ib2}, ib2={ib} and Efermi={EF}, skipping this pair"
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

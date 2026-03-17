import glob
import os
import numpy as np
from wannierberri.utility import cached_einsum
from wannierberri.fourier.rvectors import Rvectors


# from .rvectors import RvectorsRR


class ScatteringMatrix:

    """
    Class for the scattering matrix

    """

    def __init__(self, rvec, Vrrab=None, num_wann=None,
                 ):
        self.rvec = rvec
        if Vrrab is not None:
            self.Vrrab = Vrrab
            self.num_wann = Vrrab.shape[2]
            assert Vrrab.shape == (rvec.nRvec, rvec.nRvec, self.num_wann, self.num_wann), f"Vrrab should have shape (nRvec, nRvec, num_wann, num_wann), but got {Vrrab.shape}"
        elif num_wann is not None:
            self.Vrrab = np.zeros((rvec.nRvec, rvec.nRvec, num_wann, num_wann), dtype=complex)
            self.num_wann = num_wann
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
            Vkkab_w = np.zeros((num_kpts, num_kpts, num_bands, chk.num_wann), dtype=complex)
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
                wannier_centers_red = np.zeros((num_kpts, num_wann, 3), dtype=float)
            else:
                wannier_centers_red = chk.wannier_centers_cart @ np.linalg.inv(chk.real_lattice)
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
        for key in ["Vrrab","_multipole_eigenvalues", "_multipole_eigenvectors"]:
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
                shifts_left_red=[dict.get("center_red", np.zeros(3, dtype=float))],
                shifts_right_red=dict.get("wannier_centers_red", np.zeros((num_wann, 3), dtype=float)),
            )
        return self

    @classmethod
    def from_npz(cls, filename):
        dict = np.load(filename)
        return cls.from_dict(dict)


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

    def get_on_contours(self, file1, file2,
                        save=True,
                        path=None):
        with np.load(file1) as f1:
            kpoints1 = f1["kpoints"]
            wavefunctions1 = f1["wavefunctions"]

        with np.load(file2) as f2:
            kpoints2 = f2["kpoints"]
            wavefunctions2 = f2["wavefunctions"]
        V_on_contours = self.get_on_kpoints(kpt_red_left=kpoints1, kpt_red_right=kpoints2,
                                            u_left=wavefunctions1, u_right=wavefunctions2)
        if save:
            path1, f1 = os.path.split(file1)
            path2, f2 = os.path.split(file2)
            if path is not None:
                path1 = path
                if path2 != path1:
                    Warning(
                        f"Warning: the two files are in different directories ({path1} and {path2}), but the output will be saved to {path1}")
            np.savez(
                f"{path1}/Vkk_{f1[8:-4]}_{f2[8:-4]}.npz", Vkk=V_on_contours)
        return V_on_contours
    
    def get_on_contours_all(self, path, Efermi_list=None):
        file_list = glob.glob(os.path.join(path, "contour_ib*_EF=*.npz"))
        Efermi_list_files = [float(f.split("_EF=")[1].split(".npz")[0]) for f in file_list]
        if Efermi_list is not None:
            select_files = np.zeros(len(file_list), dtype=bool)
            for Ef in Efermi_list:
                idx = np.where(np.isclose(Efermi_list_files, Ef, atol=1e-5))[0]
                if np.abs(Efermi_list_files[idx] - Ef) > 1e-5:
                    Warning(f"Warning: no file found for Efermi={Ef}, closest file has Efermi={Efermi_list_files[idx]}, which is different by {Efermi_list_files[idx] - Ef}")
                else:
                    print(f"Selected file {file_list[idx]} for Efermi={Ef}")
                select_files[idx] = True
            file_list = np.array(file_list)[select_files]
            Efermi_list_files = np.array(Efermi_list_files)[select_files]
        for f1, ef1 in zip(file_list, Efermi_list_files):
            for f2, ef2 in zip(file_list, Efermi_list_files):
                if abs(ef1 - ef2) < 1e-5:
                    print (f"Calculating scattering matrix on contours for Efermi={ef1} using files {f1} and {f2}")
                    self.get_on_contours(f1, f2, save=True, path=path)

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
        Vrarb = self.Vrrab.transpose(0, 2, 1, 3).reshape(self.rvec.nRvec*self.num_wann, self.rvec.nRvec*self.num_wann)
        e,v = np.linalg.eigh(Vrarb)
        srt = np.argsort(-abs(e))
        e=e[srt]
        if e[0] <1e-15:
            print("Warning: the largest eigenvalue is smaller than 1e-15, which may indicate that the scattering matrix is not properly set or that the system is very weakly scattering")
            return np.zeros(0), np.zeros((0, self.rvec.nRvec, self.num_wann))
        nselect = max(np.where(e/e[0]>select_threshold)[0]+1)
        e = e[:nselect]
        v = v[:, srt[:nselect]]
        self._multipole_eigenvalues = e
        self._multipole_eigenvectors = v.T.reshape(nselect, self.rvec.nRvec, self.num_wann)
        return self._multipole_eigenvalues, self._multipole_eigenvectors
    

    def get_multipole_on_contours(self, file,
                        save=True,
                        path=None):
        with np.load(file) as f:
            kpoints = f["kpoints"]
            wavefunctions = f["wavefunctions"]
            weight = f["weights"]
        
        v = self.multipole_eigenvectors
        
        exp = np.exp(-2j * np.pi * cached_einsum('Ri,kj->Rk', self.rvec.iRvec, kpoints))
        W = cached_einsum('lRa, Rk, ka -> lk', v, exp, wavefunctions)
        vertex = cached_einsum('lk, k, mk -> lm', W.conj(), weight, W)
        projector = cached_einsum('lk, mk -> lm', W.conj(), W)
        if save:
            path1, f1 = os.path.split(file)
            if path is None:
                path = path1
            np.savez(
                f"{path1}/multipole_vertex_{f1[8:-4]}.npz", vertex=vertex, projector=projector, kpoints=kpoints)
        return vertex, projector
            

    def get_linewidth_multipole(self,path, Efermi):
        linewidths = {}
        kpoints = {}
        file_dict = {}
        for f in glob.glob(os.path.join(path, "multipole_vertex_*.npz")):
            Ef = float(os.path.basename(f).split("_")[2])
            print(f"Found file {f} with Efermi={Ef}")
            if Ef == Efermi:
                ib = int(os.path.basename(f).split("_")[3][:-4])
                if ib in file_dict:
                    Warning(f"Warning: multiple files found for ib={ib} and Efermi={Efermi}, using the first one found: {file_dict[ib]}")
                else:
                    file_dict[ib] = np.load(f)
        for ib, f in file_dict.items():
            lw = np.zeros(f["kpoints"].shape[0], dtype=float)
            projector = f["projector"]
            for f2 in file_dict.values():
                vertex = f2["vertex"]
                lw += np.real(np.diag(cached_einsum('klm, ml, l -> k', vertex, projector, self.multipole_eigenvalues)))
            linewidths[ib] = lw
            kpoints[ib] = f["kpoints"]
        return linewidths, kpoints
    
    def get_linewidth_direct(self, path, Efermi):
        linewidths = {}
        file_dict = {}
        kpoints = {}
        weights = {}
        for f in glob.glob(os.path.join(path, "Vkk_*.npz")):
            Ef1 = float(os.path.basename(f).split("_")[2][3:10])
            Ef2 = float(os.path.basename(f).split("_")[4][3:10])
            print(f"Found file {f} with Efermi1={Ef1} and Efermi2={Ef2}")
            if Ef1 == Efermi and Ef2 == Efermi:
                ib1 = int(os.path.basename(f).split("_")[1][2:])
                ib2 = int(os.path.basename(f).split("_")[3][2:])
                print (f"Selected file {f} for ib1={ib1}, ib2={ib2} and Efermi={Efermi}")
                if (ib1, ib2) in file_dict:
                    Warning(f"Warning: multiple files found for ib1={ib1}, ib2={ib2} and Efermi={Efermi}, using the first one found: {file_dict[(ib1, ib2)]}")
                else:
                    file_dict[(ib1, ib2)] = np.load(f)
                if ib1 not in kpoints:
                    file_kp = np.load(os.path.join(path, f"contour_ib{ib1}_EF={Efermi:.5f}.npz"))
                    kpoints[ib1] = file_kp["kpoints"]
                    weights[ib1] = file_kp["weights"]
                if ib2 not in kpoints:
                    file_kp = np.load(os.path.join(path, f"contour_ib{ib2}_EF={Efermi:.5f}.npz"))
                    kpoints[ib2] = file_kp["kpoints"]
                    weights[ib2] = file_kp["weights"]
        linewidths = {}
        for ib1 in kpoints.keys():
            lw = np.zeros(kpoints[ib1].shape[0], dtype=float)
            for ib2 in kpoints.keys():
                if (ib1, ib2) in file_dict:
                    Vkk = file_dict[(ib1, ib2)]["Vkk"]
                    x = np.real(cached_einsum('kq, q, qk -> k', Vkk, weights[ib2], Vkk.conj()))
                    print (f"{Vkk.shape=}, {weights[ib2].shape=},{lw.shape=}, {x.shape=}")
                    lw += x
            linewidths[ib1] = lw
        return linewidths, kpoints    

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
            assert Vrrab.shape == (rvec.nRvec, rvec.nRvec, self.num_wann, self.num_wann), f"Vrrab should have shape (nRvec, nRvec, num_wann, num_wann), but got {Vrrab.shape}"
        elif num_wann is not None:
            self.Vrrab = np.zeros((rvec.nRvec, rvec.nRvec, num_wann, num_wann), dtype=complex)
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
        exp_right = np.exp( -2j * np.pi *
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
    
    def get_Vkk_on_contours_all(self, path, Efermi_list=None):
        if Efermi_list is None:
            Efermi_list = get_all_Fermi_levels(path)
        for Efermi in Efermi_list:
            file_list = get_contour_files_Efermi(path, Efermi)
            for f1 in file_list:
                for f2 in file_list:
                    print (f"Calculating scattering matrix on contours for Efermi={Efermi} using files {f1} and {f2}")
                    self.get_Vkk_on_contours(f1, f2, save=True, path=path)

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
    

    def get_multipole_on_contour(self, file,
                        save=True,
                        path=None):
        with np.load(file) as f:
            kpoints = f["kpoints"]
            wavefunctions = f["wavefunctions"]
            weight = f["weights"]
        
        v = self.multipole_eigenvectors
        l = self.multipole_eigenvalues
        
        exp = np.exp(-2j * np.pi * cached_einsum('Ri,kj->Rk', self.rvec.iRvec, kpoints))
        W = cached_einsum('lRa, Rk, ka -> lk', v, exp, wavefunctions)
        vertex = cached_einsum('lk, k, mk , m-> lm', W.conj(), weight, W, l)
        projector = cached_einsum('lk, mk, m -> klm', W.conj(), W, l)
        if save:
            path1, f1 = os.path.split(file)
            if path is None:
                path = path1
            np.savez(
                f"{path1}/multipole_vertex_{f1[8:-4]}.npz", vertex=vertex, projector=projector, kpoints=kpoints)
        return vertex, projector
    
    def get_multipole_on_contours_all(self, path, Efermi_list=None):
        if Efermi_list is None:
            Efermi_list = get_all_Fermi_levels(path)
        for Efermi in Efermi_list:
            file_list = get_contour_files_Efermi(path, Efermi)
            for f in file_list:
                print (f"Calculating multipole vertex on contour for Efermi={Efermi} using file {f}")
                self.get_multipole_on_contour(f, save=True, path=path)
            

 
      


def get_EF_from_filename(filename):
    return float(os.path.basename(filename).split("_EF=")[1].split(".npz")[0])

def get_ib_from_filename(filename):
    return int(os.path.basename(filename).split("_ib")[1].split("_")[0])

def get_all_Fermi_levels(path):
    file_list = glob.glob(os.path.join(path, "contour_ib*_EF=*.npz"))
    Efermi_list_files = set([get_EF_from_filename(f) for f in file_list])
    return sorted(Efermi_list_files)

def get_contour_files_Efermi(path, Efermi, ib=None):
    if ib is None:
        file_list = glob.glob(os.path.join(path, "contour_ib*_EF=*.npz"))
        Efermi_list_files = [get_EF_from_filename(f) for f in file_list]
        select = np.isclose(Efermi_list_files, Efermi, atol=1e-5)   
        return np.array(file_list)[select]
    else:
        file_list = get_contour_files_Efermi(path, Efermi, ib=None)
        ib_list = [get_ib_from_filename(f) for f in file_list]
        select = np.array(ib_list) == ib
        res = np.array(file_list)[select]
        assert len(res) <= 1, f"Multiple files found for ib={ib} and Efermi={Efermi}, but expected at most one: {res}"
        return res



def get_ib_EF_from_Vkk_filename(filename):
    base = os.path.basename(filename).split(".npz")[0]
    ib1 = int(base.split("_")[1][2:])
    ib2 = int(base.split("_")[3][2:])
    Efermi1 = float(base.split("_")[2][3:])
    Efermi2 = float(base.split("_")[4][3:])
    return ib1, ib2, Efermi1, Efermi2

def get_Vkk_files_Efermi(path, Efermi):
    file_list_0 = glob.glob(os.path.join(path, "Vkk_ib*_EF=*_ib*_EF=*.npz".format(Efermi, Efermi)))
    file_dict={}
    for f in file_list_0:
        ib1, ib2, Efermi1, Efermi2 = get_ib_EF_from_Vkk_filename(f)
        if np.allclose([Efermi1, Efermi2], Efermi, atol=1e-5):
            file_dict[(ib1, ib2)] = f
    return file_dict

def get_Vkk_file(ib1, ib2, Efermi, path):
    file_list = get_Vkk_files_Efermi(path, Efermi)
    for f in file_list:
        ib1_f, ib2_f, Efermi1_f, Efermi2_f = get_ib_EF_from_Vkk_filename(f)
        if ib1_f == ib1 and ib2_f == ib2:
            return f
        

def get_vertices_files_Efermi(path, Efermi):
    file_list_0 = glob.glob(os.path.join(path, "multipole_vertex_ib*_EF=*.npz".format(Efermi)))
    file_dict={}
    for f in file_list_0:
        base = os.path.basename(f).split(".npz")[0]
        ib = int(base.split("_")[2][2:])
        Efermi_f = float(base.split("_")[3][3:])
        if np.isclose(Efermi_f, Efermi, atol=1e-5):
            file_dict[ib] = f
    return file_dict

def get_vertix_file(ib, Efermi, path):
    file_list = get_vertices_files_Efermi(path, Efermi)
    return file_list[ib]

def get_linewidth_Efermi(path, Efermi):
    files_contour = get_contour_files_Efermi(path, Efermi)
    files_Vkk_dict = get_Vkk_files_Efermi(path, Efermi)
    contour_dict = {get_ib_from_filename(f): f for f in files_contour}
    linewidth_dict = {}
    for ib, file_contour in contour_dict.items():
        contour1 = np.load(file_contour)
        linewidth_dict[ib] = np.zeros(contour1["kpoints"].shape[0], dtype=float)
        for ib2 in contour_dict.keys():
            contour2 = np.load(contour_dict[ib2])
            if (ib, ib2) not in files_Vkk_dict:
                Warning(f"Warning: no Vkk file found for ib1={ib}, ib2={ib2} and Efermi={Efermi}, skipping this pair")
                continue
            else:
                file_Vkk = np.load(files_Vkk_dict[(ib, ib2)])
            if (ib2, ib) not in files_Vkk_dict:
                Warning(f"Warning: no Vkk file found for ib1={ib2}, ib2={ib} and Efermi={Efermi}, skipping this pair")
                continue
            else:
                file_Vkk_conj = np.load(files_Vkk_dict[(ib2, ib)])
            assert np.allclose(file_Vkk["Vkk"], file_Vkk_conj["Vkk"].conj().T), f"Warning: Vkk file for ib1={ib}, ib2={ib2} and Efermi={Efermi} is not the conjugate transpose of the Vkk file for ib1={ib2}, ib2={ib} and Efermi={Efermi}, skipping this pair"
            linewidth = cached_einsum('kq, q, qk -> k', file_Vkk["Vkk"], contour2["weights"], file_Vkk_conj["Vkk"])
            assert np.all(abs(linewidth.imag) < 1e-5), f"Warning: linewidth has significant imaginary part for ib1={ib}, ib2={ib2} and Efermi={Efermi}, but expected to be real, skipping this pair"
            np.savez(os.path.join(path, f"linewidth_ib{ib}_ib{ib2}_EF={Efermi:.5f}.npz"), 
                    linewidth=linewidth, kpoints=contour1["kpoints"], weights=contour1["weights"])
            
            linewidth_dict[ib] += np.real(linewidth)
    return linewidth_dict


def get_linewidth_multipole_Efermi(path, Efermi):
    linewidths_dict = {}
    files_contour = get_contour_files_Efermi(path, Efermi)
    files_vertices_dict = get_vertices_files_Efermi(path, Efermi)
    for file_contour in files_contour:
        ib = get_ib_from_filename(file_contour)
        contour = np.load(file_contour)
        linewidths_dict[ib] = np.zeros(contour["kpoints"].shape[0], dtype=float)
        vertex_file = get_vertix_file(ib, Efermi, path)
        projector = np.load(vertex_file)["projector"]
        for jb, vertex2 in files_vertices_dict.items():
            vertex = np.load(vertex2)["vertex"]
            print (f"{vertex.shape=}, {projector.shape=}")
            linewidths = cached_einsum('klm, ml -> k', projector, vertex)
            linewidths_dict[ib] += np.real(linewidths)
            np.savez(os.path.join(path, f"linewidth_multipole_ib{ib}_ib{jb}_EF={Efermi:.5f}.npz"),
                    linewidth=linewidths, kpoints=contour["kpoints"], weights=contour["weights"])
    return linewidths_dict
            
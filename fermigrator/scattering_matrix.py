import numpy as np
from wannierberri.utility import cached_einsum
from .rvectors import Rvectors2


class ScatteringMatrix:
    """Electron scattering matrix stored in the Wannier R-space representation.

    Internally stores V_ab^{R1 R2}, the double Fourier transform of the
    k-space matrix element V_{mn}(k, k'):

        V_ab^{R1 R2} = (1/NK²) Σ_{k,k'} e^{-ik·R1} V_{ab}(k,k') e^{ik'·R2}

    Hermiticity V_ab^{R1 R2} = (V_ba^{R2 R1})* is enforced on construction.
    """

    def __init__(self, rvec, Vrrab=None, num_wann=None, nspin=1, multipole_threshold=1e-8
                 ):
        self.rvec = Rvectors2.from_Rvectors(rvec)
        if Vrrab is not None:
            if Vrrab.ndim == 4:
                nspin = 1
                assert Vrrab.shape == (rvec.nRvec, rvec.nRvec, self.num_wann,
                                   self.num_wann), f"Vrrab should have shape (nRvec, nRvec, num_wann, num_wann), but got {Vrrab.shape}"
                num_wann = Vrrab.shape[2]// nspin
                Vrrab = Vrrab.reshape(self.rvec.nRvec, self.rvec.nRvec, num_wann, nspin, num_wann, nspin).transpose(0, 1, 2, 4, 3, 5)
            elif Vrrab.ndim == 6:
                nspin = Vrrab.shape[4]
                assert Vrrab.shape == (rvec.nRvec, rvec.nRvec, num_wann, num_wann, nspin, nspin), f"Vrrab should have shape (nRvec, nRvec, num_wann, num_wann, nspin, nspin), but got {Vrrab.shape}"
            self.Vrrab = Vrrab
            diff = Vrrab - Vrrab.transpose(1, 0, 3, 2, 5, 4).conj()
            max_diff = np.max(np.abs(diff))
            assert max_diff < 1e-10, f"Vrrab is not Hermitian, max difference is {max_diff}, which may indicate that the transformation from Vkkmn to Vrrab is not correct or that the input Vkkmn is not correct"
        elif num_wann is not None:
            self.Vrrab = np.zeros(
                (rvec.nRvec, rvec.nRvec, num_wann, num_wann, nspin, nspin), dtype=complex)
        else:
            raise ValueError("Either Vrrab or num_wann should be provided")
        self.multipole_threshold = multipole_threshold

    @property
    def num_wann(self):
        return self.Vrrab.shape[2]
    
    @property
    def nspin(self):
        return self.Vrrab.shape[4]

    @classmethod
    def from_Vkk(cls, Vkkmn_wan,
                 center_red=None,
                 wannier_centers_red=None,
                 real_lattice=None,
                 rvectors=None,
                 mp_grid=None,
                 kpt_red=None):
        """Construct from a k-space scattering matrix in the Wannier gauge.

        Parameters
        ----------
        Vkkmn_wan : ndarray, shape (NK, NK, NW, NW, NSPIN, NSPIN)
            Matrix elements V_{mn}(k, k') in eV, Wannier gauge, on a uniform k-grid.
        center_red : array (3,), optional
            Reduced coordinates of the scattering centre (default: origin).
        wannier_centers_red : array (NW, 3)
            Wannier function centres in reduced coordinates.
        real_lattice : array (3, 3)
            Real-space lattice vectors in Å.
        rvectors : Rvectors, optional
            Pre-built Rvectors object; overrides the lattice/grid parameters.
        mp_grid : array (3,)
            Monkhorst–Pack grid dimensions used in the Wannier90 calculation.
        kpt_red : array (NK, 3)
            k-points in reduced coordinates matching the rows of Vkkmn_wan.
        """
        assert Vkkmn_wan.ndim == 6, f"Vkkmn_wan must be a 6D array with shape (NK, NK, NW, NSPIN, NW, NSPIN), but got shape {Vkkmn_wan.shape}"
        assert Vkkmn_wan.shape[0] == Vkkmn_wan.shape[
            1], f"The first two dimensions of Vkkmn_wan must be the same, but got {Vkkmn_wan.shape[0]} and {Vkkmn_wan.shape[1]}"
        assert Vkkmn_wan.shape[2] == Vkkmn_wan.shape[
            3], f"The num_wann dimensions of Vkkmn_wan must be the same, but got {Vkkmn_wan.shape[2]} and {Vkkmn_wan.shape[3]}"
        assert Vkkmn_wan.shape[4] == Vkkmn_wan.shape[
            5], f"The num_spin dimensions of Vkkmn_wan must be the same, but got {Vkkmn_wan.shape[4]} and {Vkkmn_wan.shape[5]}"
        assert Vkkmn_wan.shape[4] in [1, 2], f"The num_spin dimension of Vkkmn_wan must be 1 or 2, but got {Vkkmn_wan.shape[4]}"

        if center_red is None:
            center_red = np.zeros(3, dtype=float)
        if rvectors is None:
            rvectors = Rvectors2(
                lattice=real_lattice,
                shifts_left_red=[center_red],
                shifts_right_red=wannier_centers_red,
            )
            rvectors.set_Rvec(mp_grid=mp_grid)
            rvectors.set_fft_q_to_R(kpt_red=kpt_red)

        Vabcrr = rvectors.qq_to_RR(Vkkmn_wan[:, :, :, :, None])
        Vrrab = Vabcrr[:, :, 0, :, :].transpose(2, 3, 0, 1)
        return cls(rvec=rvectors, Vrrab=Vrrab)

    def set_VRR(self, Vrrab, irvec1, irvec2, 
                a=slice(None), b=slice(None), s1=slice(None), s2=slice(None), 
                add=True,
                add_Herm_conj=True):
        assert self.Vrrab is not None, "Vrrab is not set, please initialize the scattering matrix with Vrrab or num_wann first"
        ir1 = self.rvec.iR(irvec1)
        ir2 = self.rvec.iR(irvec2)
        if add:
            self.Vrrab[ir1, ir2, a, b, s1, s2] += Vrrab
        else:
            self.Vrrab[ir1, ir2, a, b, s1, s2] = Vrrab
        if add_Herm_conj and not ( ir1==ir2 and a==b and s1==s2):
            Vrrab = np.asarray(Vrrab)
            V_hc = Vrrab.conj().transpose(*np.arange(Vrrab.ndim)[::-1])            
            self.Vrrab[ir2, ir1, b, a, s2, s1] += V_hc
 
    def as_dict(self):
        dict = {}
        for key in ["center_red", "Vrrab", "multipole_threshold",
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
            self.rvec = Rvectors2(
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
        """Evaluate V_{mn}(k, k') for sets of k and k' points.

        Computes the inverse Fourier transform:
            V_{ab}(k, k') = Σ_{R,R'} e^{ik·R} V_ab^{RR'} e^{-ik'·R'}

        and optionally contracts with wavefunctions to give band matrix elements:
            V_{mn}(k, k') = Σ_{ab} u_{ma}^*(k) V_{ab}(k,k') u_{nb}(k')

        Parameters
        ----------
        kpt_red_left, kpt_red_right : ndarray, shape (N, 3) and (M, 3)
            k-points in reduced coordinates.
        u_left, u_right : ndarray, shape (N, NW) and (M, NW), optional
            Wavefunction columns U_K[:, iband] for band projection.

        Returns
        -------
        ndarray, shape (N, M) if wavefunctions given, else (N, M, NW, NW)
        """
        Vkk = self.rvec.RR_to_kk(self.Vrrab, kpt_red_left, kpt_red_right)
        assert (u_left is None) == (
            u_right is None), "u_left and u_right must be both None or both not None"
        if u_left is not None:
            return cached_einsum('ka,kqabst,qb->kqst',
                                 u_left.conj(), Vkk, u_right)
        else:
            return Vkk

    def get_Vkk_on_contours(self, file1, file2, contours_db=None):
        """Evaluate band matrix elements V_{m n}(k, k') between two contour files.

        Loads k-points and wavefunctions from `file1` (left) and `file2` (right),
        calls `get_on_kpoints`, and optionally saves the result to `contours_db`
        as a ``Vkk`` entry keyed by (ib1, ib2, EF).

        Parameters
        ----------
        file1, file2 : str
            Paths to contour .npz files (must contain ``kpoints`` and ``wavefunctions``).
        contours_db : ContourDatabase, optional
            If provided, the result is saved and the band/EF labels are read from
            the filenames.
        """
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
            assert EF1 == EF2, f"Fermi levels in the two contour files do not match: {EF1} and {EF2} diff is {float(EF1) - float(EF2)}"
            contours_db.set_data("Vkk", dict(Vkk=V_on_contours),
                                 ib1=ib1, ib2=ib2, EF=EF1)
        return V_on_contours

    def get_Vkk_on_contours_all(self, contours_db, Efermi_list=None):
        """Compute and save V_{mn}(k,k') for all band pairs at each Fermi level."""
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
            self.multipole_decomposition_RR(select_threshold=self.multipole_threshold)
        return self._multipole_eigenvectors

    def multipole_decomposition_RR(self, select_threshold=1e-8):
        """Decompose V_abst^{RR'} into eigenmodes (multipoles).

        Reshapes V_abst^{R1 R2} as a square matrix V_{(R1,a,s),(R2,b,t)}, then
        diagonalises: V = Σ_l λ_l |φ_l><φ_l|.  Modes are sorted by |λ_l|
        and those with |λ_l|/|λ_0| > select_threshold are kept.

        Results are cached as _multipole_eigenvalues and _multipole_eigenvectors.

        Parameters
        ----------
        select_threshold : float
            Relative cutoff; -1 keeps all modes.
        """
        assert self.Vrrab is not None, "Vrrab is not set, please set it first using set_RR"
        nR = self.rvec.nRvec
        nw = self.num_wann
        ns = self.nspin
        size = nR * nw * ns
        nonzeros = np.where(np.abs(self.Vrrab) > 1e-15)
        for ir1, ir2, a, b, s1, s2 in zip(*nonzeros):
            v = self.Vrrab[ir1, ir2, a, b, s1, s2]
            delta_red = self.rvec.iRvec[ir1] - self.rvec.iRvec[ir2] + self.rvec.shifts_left_red[a] - self.rvec.shifts_right_red[b]
            
            delta_cart = delta_red @ self.rvec.lattice
            print (f"Vrrab[{ir1}, {ir2}, {a}, {b}, {s1}, {s2}] = {v} with delta_red = {delta_red}, delta_cart = {delta_cart}, R1={self.rvec.iRvec[ir1]}, R2={self.rvec.iRvec[ir2]} and wannier centers {self.rvec.shifts_right_red[a]}, {self.rvec.shifts_right_red[b]} and spins {s1}, {s2}")
        Vrasrbt = self.Vrrab.transpose(0, 2, 4, 1, 3, 5).reshape(size, size)
        # print (f"Performing multipole decomposition of Vrasrbt with \n {Vrasrbt}")
        
        assert np.allclose(Vrasrbt, Vrasrbt.conj().T), "Vrrab is not Hermitian, cannot perform multipole decomposition"
        e, v = np.linalg.eigh(Vrasrbt)
        srt = np.argsort(-abs(e))
        e = e[srt]
        if abs(e[0]) < 1e-15:
            print("Warning: the largest eigenvalue is smaller than 1e-15, which may indicate that the scattering matrix is not properly set or that the system is very weakly scattering")
            self._multipole_eigenvalues = np.zeros(0, dtype=float)
            self._multipole_eigenvectors = np.zeros((0, nR, nw, ns), dtype=complex)
        else:
            nselect = max(np.where(abs(e)/abs(e[0]) > select_threshold)[0] + 1)
            e = e[:nselect]
            v = v[:, srt[:nselect]]
            self._multipole_eigenvalues = e
            self._multipole_eigenvectors = v.T.reshape(
                nselect, self.rvec.nRvec, self.num_wann, self.nspin)
            
        Vrasrbt_recon = cached_einsum('xl, yl, l ->  xy', v, v.conj(), e)
        assert np.allclose(Vrasrbt_recon, Vrasrbt), f"Multipole decomposition reconstruction does not match original Vrasrbt, max difference is {np.max(np.abs(Vrasrbt_recon - Vrasrbt))}"
        v = self._multipole_eigenvectors
        e = self._multipole_eigenvalues
        Vrrab_recon = cached_einsum('lRas, lrbt, l -> Rrabst', v, v.conj(), e)
        assert np.allclose(Vrrab_recon, self.Vrrab), f"Multipole decomposition reconstruction does not match original Vrrab, max difference is {np.max(np.abs(Vrrab_recon - self.Vrrab))}"
        return self._multipole_eigenvalues, self._multipole_eigenvectors

    def get_multipole_on_contour(self, file, contours_db=None):
        """Project multipole eigenmodes onto a Fermi surface contour.

        For each eigenmode φ_l with eigenvalue λ_l, evaluates the overlap
        with the Bloch wavefunctions on the contour:

            W_l(k) = Σ_{R,a} φ_l^{Ra} e^{ik·R} u_a(k)

        and computes two quantities saved under ``multipole-vertex``:

        - ``vertex[l, m]``    = Σ_k W_l*(k) w(k) W_m(k) λ_m
          (weighted overlap used in linewidth summations over k')
        - ``projector[k,l,m]``= W_l*(k) W_m(k) λ_m
          (k-resolved factor used when computing Γ(k) for a fixed k)

        Parameters
        ----------
        file : str
            Path to a contour .npz file (must contain kpoints,set_torque_operators_R wavefunctions, weights).
        contours_db : ContourDatabase, optional
            If provided, results are saved under the ``multipole-vertex`` type.
        """
        with np.load(file) as f:
            kpoints = f["kpoints"]
            wavefunctions = f["wavefunctions"]
            weight = f["weights"]
            

        v = self.multipole_eigenvectors
        e = self.multipole_eigenvalues
        
        exp = np.exp(-2j * np.pi * self.rvec.iRvec[:, :2] @ kpoints.T)
        W = cached_einsum('lRas, Rk, ka -> lks', v, exp, wavefunctions.conj())
        vertex = cached_einsum('lks, k, mks , m-> lm', W.conj(), weight, W, e)
        
        if contours_db is not None:
            ib = contours_db.split_filename(file)["ib"]
            EF = contours_db.split_filename(file)["EF"]
            contours_db.set_data("multipole-vertex", dict(vertex=vertex), ib=ib, EF=EF)
            contours_db.set_data("multipole-eigen", dict(eigenvectors=W, eigenvalues=e), ib=ib, EF=EF)
        return vertex, W, e
            


    def get_multipole_on_contours_all(self, contours_db, Efermi_list=None):
        """Compute and save multipole vertex/projector for all contours at each Fermi level."""
        if Efermi_list is None:
            Efermi_list = contours_db.get_all_Efermi()
        # print(f"Calculating multipole vertex on contours for Efermi_list={Efermi_list}")
        for Efermi in Efermi_list:
            file_list = contours_db.get_files_Efermi("contour", Efermi)
            vertex = 0
            for f in file_list:
                # print(f"Calculating multipole vertex on contour for Efermi={Efermi} using file {f}")
                ver, _, __ = self.get_multipole_on_contour(f, contours_db=contours_db)
                vertex += ver
            contours_db.set_data("multipole-vertex-sum", dict(vertex=vertex), EF=Efermi)


def bloch2wann(Vkkmn,
               chk,
               nspin=1,
               selected_bands=None):
    """
    Transform the scattering matrix from the Bloch gauge to the Wannier gauge

    Parameters
    ----------
    Vkkmn : array_like
        The scattering matrix in the Bloch gauge, with shape (NK, NK, NB, NB) (nspin=1)
        or (nspin, NK, NK, NB, 2, NB, 2) (nspin=2)
    chk : CheckPoint or str
        The checkpoint file (from wannier90 or wannierberri)
    nspin : int, optional
        The number of spins, 1 or 2  (nspin=2 assumes that chk is spinless, but the scattering matrix
        depends on spin)

    """
    kpt_red = chk.kpt_red
    num_kpts = kpt_red.shape[0]
    assert chk is not None, "Checkpoint file must be provided to transform the scattering matrix to the Wannier gauge"
    if nspin == 1:
        Vkkmn = Vkkmn[:, :, :, None, :, None]
    if selected_bands is not None:
        Vkkmn = Vkkmn[:, :, selected_bands, :,
                      :, :][:, :, :, :, selected_bands, :]
    num_bands = Vkkmn.shape[2]
    assert chk.num_kpts == num_kpts, f"Number of k-points in the scattering matrix ({num_kpts}) does not match the number of k-points in the checkpoint ({chk.num_kpts})"
    assert chk.num_bands == num_bands, f"Number of bands in the scattering matrix ({num_bands}) does not match the number of bands in the checkpoint ({chk.num_bands})"
    assert chk.v_matrix is not None, "The checkpoint does not contain the v_matrix, which is needed to transform the scattering matrix to the Wannier gauge"
    Vkkab_w = np.zeros(
        (num_kpts, num_kpts, num_bands, nspin, chk.num_wann, nspin), dtype=complex)
    for ik in range(num_kpts):
        Vkkab_w[:, ik, :, :, :] = cached_einsum(
            'kisjt,jb->kisbt', Vkkmn[:, ik], chk.v_matrix[ik])
    Vkkab_wan = np.zeros(
        (num_kpts, chk.num_kpts, chk.num_wann, nspin, chk.num_wann, nspin), dtype=complex)
    for ik in range(num_kpts):
        Vkkab_wan[ik, :, :, :] = cached_einsum(
            'kisbt,ia->kasbt', Vkkab_w[ik, :, :, :], chk.v_matrix[ik, :].conj())
    return Vkkab_wan


def get_chk(chk):
    """Load a Wannier90 checkpoint as a WannierBerri CheckPoint object.

    Accepts a CheckPoint instance (returned as-is), a path ending in ``.chk``
    (loaded via wannier90 binary format), or a path ending in ``.npz``
    (loaded from a previously serialised checkpoint).
    """
    if chk is None:
        return None
    from wannierberri.w90files.chk import CheckPoint
    if isinstance(chk, CheckPoint):
        return chk
    if isinstance(chk, str):
        if chk.endswith(".chk"):
            return CheckPoint.from_w90_file(chk[:-4])
        elif chk.endswith(".npz"):
            return CheckPoint.from_npz(chk)
    raise ValueError(
        f"chk should be either a CheckPoint object or a string ending with .chk or .npz, but got  type {type(chk)} and value {chk}")

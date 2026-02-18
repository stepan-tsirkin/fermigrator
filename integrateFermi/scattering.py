import numpy as np
from wannierberri.utility import cached_einsum


# from .rvectors import RvectorsRR
from wannierberri.fourier.rvectors import Rvectors

class ScatteringMatrix:

    """
    Class for the scattering matrix

    """

    def __init__(self, Vkkmn=None, center_red=None, gauge="wannier"):
        assert gauge in ["wannier", "bloch"], f"Gauge must be either 'wannier' or 'bloch', but got {gauge}"
        self.gauge = gauge
        if center_red is None:
            center_red = [0, 0, 0]
        self.center_red = np.array(center_red, dtype=float)
        if Vkkmn is not None:
            assert Vkkmn.ndim == 4, f"Vkkmn must be a 4D array with shape (NK, NK, NB, NB), but got shape {Vkkmn.shape}"
            assert Vkkmn.shape[0] == Vkkmn.shape[1], f"The first two dimensions of Vkkmn must be the same, but got {Vkkmn.shape[0]} and {Vkkmn.shape[1]}"
            assert Vkkmn.shape[2] == Vkkmn.shape[3], f"The last two dimensions of Vkkmn must be the same, but got {Vkkmn.shape[2]} and {Vkkmn.shape[3]}"
            self.num_kpts = Vkkmn.shape[0]
            self.Vkkmn = Vkkmn
            if gauge == "wannier":
                self.num_wann = self.Vkkmn.shape[2]
                self.num_bands = None
            elif gauge == "bloch":
                self.num_bands = self.Vkkmn.shape[2]
                self.num_wann = None
        else:
            self.Vkkmn = None
            self.num_kpts = None
            self.num_bands = None
            self.num_wann = None
    
    def select_bands(self, selected_bands):
        self.Vkkmn = self.Vkkmn[:, :, selected_bands, :][:, :, :, selected_bands]

    def as_dict(self):
        dict = {}
        for key in ["center_red", "gauge", "num_kpts", "num_bands", "num_wann", "Vkkmn", "Vabrr",
                    "real_lattice"]:
            if hasattr(self, key):
                if getattr(self, key) is not None:
                    dict[key] = getattr(self, key)
        if hasattr(self, "rvec") and self.rvec is not None:
            dict["iRvec"] = self.rvec.iRvec
            dict["wannier_center_red"] = self.rvec.shifts_left_red
        return dict
    
    def to_npz(self, filename):
        print (f"saving scattering matrix to {filename} with keys {list(self.as_dict().keys())}")
        np.savez(filename, **self.as_dict())

    def from_dict(self, dict):
        for key in ["center_red", "gauge", "num_kpts", "num_bands", "num_wann", "Vkkmn", "Vabrr",
                    "real_lattice"]:
            if key in dict:
                setattr(self, key, dict[key])
        if "iRvec" in dict:
            if "wannier_centers_red" in dict:
                wannier_centers_red = dict["wannier_centers_red"]
            else:                
                wannier_centers_red = np.zeros((self.num_wann, 3), dtype=float)
            self.rvec = Rvectors(
                iRvec=dict["iRvec"],
                lattice=self.real_lattice,
                shifts_left_red=[self.center_red],
                shifts_right_red=wannier_centers_red,
                
            )
    
    def from_npz(self, filename):
        dict = np.load(filename)
        self.from_dict(dict)
            
    
    def to_wannier_gauge(self, chk):
        assert self.gauge == "bloch", f"it is already in {self.gauge} gauge"
        assert chk.num_kpts == self.num_kpts, f"Number of k-points in the scattering matrix ({self.num_kpts}) does not match the number of k-points in the checkpoint ({chk.num_kpts})"
        assert chk.num_bands == self.num_bands, f"Number of bands in the scattering matrix ({self.num_bands}) does not match the number of bands in the checkpoint ({chk.num_bands})"
        assert chk.v_matrix is not None, "The checkpoint does not contain the v_matrix, which is needed to transform the scattering matrix to the Wannier gauge"
        
        Vkkab_w = np.zeros((self.num_kpts, self.num_kpts, self.num_bands, self.num_wann), dtype=complex)    
        for ik in range(self.num_kpts):
            Vkkab_w[ik,:,:,:] = cached_einsum('kij,jb->kib', self.Vkkmn[:,ik,:,:], chk.v_matrix[ik,:])
        Vkkab_wan = np.zeros((self.num_kpts, self.num_kpts, self.num_wann, self.num_wann), dtype=complex)
        for ik in range(self.num_kpts):
            Vkkab_wan[:,ik,:,:] = cached_einsum('kib,ia->kab', Vkkab_w[:,ik,:,:], chk.v_matrix[ik,:].conj())
        self.Vkkmn = Vkkab_wan
        self.gauge = "wannier"
        self.num_wann = chk.num_wann
        self.num_bands = None

    def set_RR(self, chk, forget_kk = True):
        """
        Transform the scattering matrix from k-space to real space (the data need to be wannierised)

        Parameters
        ----------
        scatter : `~wannierberri.w90files.scatter.ScatteringMatrix`
            the scattering matrix in k-space (in the basis of the Bloch functions)
        Returns
        -------
        Vrrmn : array((nR, nR, nb, nb), dtype=complex)
            the scattering matrix in real space (in the basis of the Wannier functions)
        Rvectors : "~wannierverri.system.Rvectors" object
        """
        wannier_centers_red = chk.wannier_centers_cart @ np.linalg.inv(chk.real_lattice)

        rvectors = Rvectors(
            lattice=chk.real_lattice,
            shifts_left_red= [self.center_red],
            shifts_right_red=wannier_centers_red,
        )
        rvectors.set_Rvec(mp_grid=chk.mp_grid)
        rvectors.set_fft_q_to_R(kpt_red=chk.kpt_red)

        self.rvec = rvectors
        self.Vabrr = rvectors.qq_to_RR(self.Vkkmn[:,:,:,:,None])[:,:,:,:,0]
        
        if forget_kk:
            self.Vkkmn = None
            self.num_kpts = None
        


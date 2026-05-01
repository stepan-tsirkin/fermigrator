import numpy as np
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.fourier.fft import execute_fft
from wannierberri.utility import cached_einsum


class Rvectors2(Rvectors):

    def qq_to_RR(self, AA_qq):
        assert self.fft_q2R_set, "FFT_q_to_R is not set, please set it first using set_fft_q_to_R"
        shapeA = AA_qq.shape[2:]  # remember the shapes after q
        AA_qq_mp = np.zeros(tuple(self.mp_grid) * 2 + shapeA, dtype=complex)
        for i, k1 in enumerate(self.kpt_mp_grid):
            for j, k2 in enumerate(self.kpt_mp_grid):
                AA_qq_mp[k1 + k2] = AA_qq[i, j]
        # direct FFT is defined as A_R = sum_q A_q exp(-i q R),
        # inverse FFT is defined as A_q = 1/N_R sum_R A_R exp(i q R)
        Nq = np.prod(self.mp_grid)
        AA_qq_mp = execute_fft(AA_qq_mp, axes=(
            0, 1, 2), fftlib=self.fftlib_q2R, destroy=False, inverse=False) / Nq
        AA_qq_mp = execute_fft(AA_qq_mp, axes=(
            3, 4, 5), fftlib=self.fftlib_q2R, destroy=False, inverse=True)
        # A_RR' = 1/N_q^2 sum_{q,q'} A_qq' exp(i q R) exp(- i q' R')
        # return AA_qq_mp.reshape( (Nq, Nq) + shapeA )
        return self.remap_XX_from_grid_to_list_RR(AA_qq_mp)

    def RR_to_kk(self, AA_RR, kpt_red_left, kpt_red_right):
        """
        Computes the inverse Fourier transform:
        V_{ab}(k, k') = Σ_{R,R'} e^{ik·R} V_ab^{RR'} e^{-ik'·R'}
        """
        dim_k_left = kpt_red_left.shape[1]
        dim_k_right = kpt_red_right.shape[1]
        # print (f"RR_to_kk: AA_RR shape {AA_RR.shape}, kpt_red_left shape {kpt_red_left.shape}, kpt_red_right shape {kpt_red_right.shape}")
        exp_left = np.exp(2j * np.pi * self.iRvec[:, :dim_k_left] @ kpt_red_left.T)
        exp_right = np.exp(-2j * np.pi * self.iRvec[:, :dim_k_right] @ kpt_red_right.T)
        return cached_einsum('Rk, Rr..., rq->kq...', exp_left, AA_RR, exp_right)

    def remap_XX_from_grid_to_list_RR(self, XX_RR_grid):
        # TODO : optimize this as for _R
        """
        remap the matrix from the double grid to the double list of R-vectors,
        taking into account the wannier centers (the "left" are used as the central, and the "right" are used asboth left and right, if you understand what I mean)

        Parameters
        ----------
        XX_RR_grid : np.ndarray(shape=(mp_grid[0], mp_grid[1], mp_grid[2], mp_grid[0], mp_grid[1], mp_grid[2], num_wann_r, num_wann_l, num_wann_r, ...))
            The matrix in the grid representation.

        Returns
        -------
        XX_R_new : np.ndarray(shape=(num_wann_r, num_wann_l, num_wann_r, nRvec, nRvec, ...))
            The matrix in the list of R-vectors representation.
        """
        XX_RR_sum_grid = XX_RR_grid.sum(axis=(0, 1, 2, 3, 4, 5))
        num_wann_r = XX_RR_grid.shape[6]
        num_wann_l = XX_RR_grid.shape[8]
        print(
            f"remapping {XX_RR_grid.shape} num_wann_r={num_wann_r}, num_wann_l={num_wann_l}")
        nl = self.nshifts_left
        nr = self.nshifts_right
        assert (nr == 1) or (
            XX_RR_grid.shape[6] == nr), f"XX_RR_grid {XX_RR_grid.shape} should have {nr} WFs"
        assert (nr == 1) or (
            XX_RR_grid.shape[7] == nr), f"XX_RR_grid {XX_RR_grid.shape} should have {nr} WFs"
        assert (nl == 1) or (
            XX_RR_grid.shape[8] == nl), f"XX_RR_grid {XX_RR_grid.shape} should have {nl} right shifts"

        shape_new = XX_RR_grid.shape[6:9] + \
            (self.nRvec,) * 2 + XX_RR_grid.shape[9:]
        print(f"shape_new {shape_new}")
        XX_RR_new = np.zeros(shape_new, dtype=XX_RR_grid.dtype)
        for a in range(num_wann_r):
            ia = 0 if self.nshifts_right == 1 else a
            for b in range(num_wann_r):
                ib = 0 if self.nshifts_right == 1 else b
                for c in range(num_wann_l):
                    ic = 0 if self.nshifts_left == 1 else c
                    ishift1 = self.shift_index[ic, ia]
                    ishift2 = self.shift_index[ic, ib]
                    # print(f"a,b,c = {a},{b},{c} : {ishift1}, {ishift2}")
                    for iRi1, iRm1, nd1 in zip(self.iRvec_index_list[ishift1],
                                               self.iRvec_mod_list[ishift1],
                                               self.Ndegen_list[ishift1]):
                        for iRi2, iRm2, nd2 in zip(self.iRvec_index_list[ishift2],
                                                   self.iRvec_mod_list[ishift2],
                                                   self.Ndegen_list[ishift2]):
                            XX_RR_new[a, b, c, iRi1, iRi2] += XX_RR_grid[tuple(
                                iRm1) + tuple(iRm2) + (a, b, c)] / (nd1 * nd2)
        XX_R_sum_new = XX_RR_new.sum(axis=(3, 4))
        assert np.allclose(
            XX_R_sum_new, XX_RR_sum_grid), f"XX_R_sum_R_new {XX_R_sum_new} != XX_R_sum_T_tmp {XX_RR_sum_grid}"
        return XX_RR_new

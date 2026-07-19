import numpy as np

class Brillouin:

    def __init__(self, recip_lattice, ws_search_max=6):
        self.recip_lattice = np.array(recip_lattice)
        dim = self.recip_lattice.shape[0]
        self.metric = self.recip_lattice @ self.recip_lattice.T
        from wannierberri.utility import iterate_nd
        g_vectors_i = []
        g_vectors_proj = []
        one = 1+1e-8  # 
        for l in range(1, ws_search_max+1):
            done = True
            for ijk in iterate_nd([l]*dim, pm=True):
                if np.all(np.abs(ijk) < l):
                    continue
                take=True
                for g_i, g_proj in zip(g_vectors_i, g_vectors_proj):
                    if np.linalg.norm(np.cross(ijk, g_i)) < 1e-6:
                        # print (f"ijk={ijk} is parallel to g_i={g_i}, skipping")
                        take=False
                        break
                    if abs(ijk @ g_proj) > one:
                        # print (f"ijk={ijk} projected onto {g_i=} is {ijk @ g_proj}, skipping")
                        take=False
                        break
                if take:
                    # print (f"ijk={ijk} is a new Brillouin zone vector {ijk @ self.recip_lattice=}")
                    g_vectors_i.append(ijk)
                    g_sq12 = ijk @ self.metric @ ijk
                    # print (f"{gq12=}, {ijk=}, {ijk @ self.metric=}")
                    g_vectors_proj.append(ijk @ self.metric/g_sq12)
                    done = False
            if done:
                break
        else:
            raise RuntimeError(f"Brillouin zone search did not converge after {ws_search_max} iterations. Increase ws_search_max.")

        self.g_vectors_i = np.array(g_vectors_i)
        self.g_vectors_c = g_vectors_i @ self.recip_lattice
        self.g_vectors_proj = np.array(g_vectors_proj)
        self_proj = self.g_vectors_i @ self.g_vectors_proj.T
        diag = range(len(self.g_vectors_i))
        self_proj[diag, diag] = 0
        # print (f"self_proj = \n{self_proj}")
        exclude = np.any(np.abs(self_proj)> one, axis=1)
        self.g_vectors_i = self.g_vectors_i[~exclude]
        self.g_vectors_c = self.g_vectors_c[~exclude]
        self.g_vectors_proj = self.g_vectors_proj[~exclude]
        

    def get_shifts(self, kpoints_reduced, max_iter=10):
        """get the shifts to apply to the kpoints to place them inside the first Brillouin zone"""
        kpoints_reduced = np.array(kpoints_reduced)
        kpoints_shifted = kpoints_reduced.copy()
        for _ in range(max_iter):
            done = True
            for g_i, g_proj in zip(self.g_vectors_i, self.g_vectors_proj):
                proj = kpoints_shifted @ g_proj
                shift = np.round(proj)[:, None] * g_i[None, :]
                if np.any(shift != 0):
                    done = False
                kpoints_shifted -= shift
            if done:
                break
        else:
            raise RuntimeError(f"Bringing kpoints inside the first Brillouin zone failed to converge after {max_iter} iterations. ")
        return kpoints_shifted - kpoints_reduced

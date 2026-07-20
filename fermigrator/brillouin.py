from functools import cached_property

import numpy as np


class Brillouin:

    def __init__(self, recip_lattice, ws_search_max=6):
        self.recip_lattice = np.array(recip_lattice)
        dim = self.recip_lattice.shape[0]
        self.metric = self.recip_lattice @ self.recip_lattice.T
        from wannierberri.utility import iterate_nd
        g_vectors_i = []
        g_vectors_proj_red = []
        one = 1 + 1e-8  #
        for shell in range(1, ws_search_max + 1):
            done = True
            for ijk in iterate_nd([shell] * dim, pm=True):
                if np.all(np.abs(ijk) < shell):
                    continue
                take = True
                for g_i, g_proj in zip(g_vectors_i, g_vectors_proj_red):
                    if np.linalg.norm(np.cross(ijk, g_i)) < 1e-6:
                        # print (f"ijk={ijk} is parallel to g_i={g_i}, skipping")
                        take = False
                        break
                    if abs(ijk @ g_proj) > one:
                        # print (f"ijk={ijk} projected onto {g_i=} is {ijk @ g_proj}, skipping")
                        take = False
                        break
                if take:
                    # print (f"ijk={ijk} is a new Brillouin zone vector {ijk @ self.recip_lattice=}")
                    g_vectors_i.append(ijk)
                    g_sq12 = ijk @ self.metric @ ijk
                    # print (f"{gq12=}, {ijk=}, {ijk @ self.metric=}")
                    g_vectors_proj_red.append(ijk @ self.metric / g_sq12)
                    done = False
            if done:
                break
        else:
            raise RuntimeError(f"Brillouin zone search did not converge after {ws_search_max} iterations. Increase ws_search_max.")

        self.g_vectors_i = np.array(g_vectors_i)
        self.g_vectors_c = g_vectors_i @ self.recip_lattice
        self.g_vectors_proj_red = np.array(g_vectors_proj_red)
        self_proj = self.g_vectors_i @ self.g_vectors_proj_red.T
        diag = range(len(self.g_vectors_i))
        self_proj[diag, diag] = 0
        # print (f"self_proj = \n{self_proj}")
        exclude = np.any(np.abs(self_proj) > one, axis=1)
        self.g_vectors_i = self.g_vectors_i[~exclude]
        self.g_vectors_c = self.g_vectors_c[~exclude]
        self.g_vectors_proj_red = self.g_vectors_proj_red[~exclude]

    def get_shifts(self, kpoints_reduced, max_iter=10):
        """get the shifts to apply to the kpoints to place them inside the first Brillouin zone"""
        kpoints_reduced = np.array(kpoints_reduced)
        kpoints_shifted = kpoints_reduced.copy()
        for _ in range(max_iter):
            done = True
            for g_i, g_proj in zip(self.g_vectors_i, self.g_vectors_proj_red):
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

    @cached_property
    def boundary(self):
        """get the boundary of the first Brillouin zone as a list of planes (normal, offset)"""
        from scipy.spatial import HalfspaceIntersection
        G = np.vstack([self.g_vectors_c, -self.g_vectors_c])

        halfspaces = np.empty((len(G), 4))

        halfspaces[:, :3] = G
        halfspaces[:, 3] = -0.5 * np.sum(G * G, axis=1)

        hs = HalfspaceIntersection(
            halfspaces,
            interior_point=np.zeros(3)
        )

        vertices = hs.intersections
        # print (f"vertices = {vertices}")
        faces = []
        G_proj = G / (np.linalg.norm(G, axis=1)**2)[:, None]
        # print (f"G_proj = {G_proj}")
        for g_cart, g_proj in zip(G, G_proj):
            proj = vertices @ g_proj
            # print (f"g_cart = {g_cart}")
            # print ("proj = ", proj)
            mask = abs(proj - 0.5) < 1e-8
            vert_ind = np.where(mask)[0]
            srt = order_vertices_face(vertices[vert_ind], g_cart)
            vert_ind = vert_ind[srt]
            faces.append(vert_ind)
        edges = faces_to_edges(faces)

        return vertices, faces, edges


def order_vertices_face(vertices, face_normal):
    """order the vertices of a face in counter-clockwise order"""
    # project the vertices onto the plane defined by the face normal
    face_normal = face_normal / np.linalg.norm(face_normal)
    # find a vector that is not parallel to the face normal
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    assert np.all(np.abs(vertices_centered @ face_normal) < 1e-8), f"vertices are not coplanar with the face normal {face_normal=}, {vertices=}"
    vertices_centered /= np.linalg.norm(vertices_centered, axis=1)[:, None]
    # find a vector that is not parallel to the face normal
    v0 = vertices_centered[0]
    # find angle between v0 and the other vertices in the plane defined by the face normal
    angles = [np.arctan2(v0 @ v, np.linalg.det([face_normal, v0, v]))
              for v in vertices_centered]
    return np.argsort(angles)


def faces_to_edges(faces):
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]

            # store edge in canonical order
            edge = tuple(sorted((a, b)))
            edges.add(edge)
    return np.array(list(edges), dtype=int)

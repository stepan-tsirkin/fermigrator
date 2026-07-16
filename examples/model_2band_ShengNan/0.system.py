from wannierberri.grid import Path
from matplotlib import pyplot as plt
import numpy as np
from wannierberri import System_R

# import ray
# ray.init()
import pythtb


lat = np.eye(3)
orb = np.zeros((2, 3))

lattice = pythtb.Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1, 2])
my_model = pythtb.TBModel(lattice)


# set on-site energies
delta = -5
my_model.set_onsite([-delta, delta])
# set hoppings (one for each connected pair of orbitals)
# from j in R to i in 0
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(-1, 0, 0, [1, 0, 0])
my_model.set_hop(-1, 1, 1, [1, 0, 0])
my_model.set_hop(-1, 0, 0, [0, 1, 0])
my_model.set_hop(1, 1, 1, [0, 1, 0])
my_model.set_hop(-1, 0, 0, [0, 0, 1])
my_model.set_hop(1, 1, 1, [0, 0, 1])


system = System_R.from_pythtb(my_model)


system.set_pointgroup(["Inversion", "C4x", "C2z"])

system.to_npz("system-2band")
nodes_dic = {"Gamma": [0, 0, 0], "X": [1 / 2, 0, 0], "Y": [0, 1 / 2, 0], "Z": [0, 0, 1 / 2], None: None, "XY": [1 / 2, 1 / 2, 0], "XZ": [1 / 2, 0, 1 / 2], "YZ": [0, 1 / 2, 1 / 2], "M": [1 / 2, 1 / 2, 1 / 2]}
path_labales = ["Y", "X", "Gamma", "Y", None, "Z", "Gamma"]
path = Path.from_nodes(system, nodes=[nodes_dic[k] for k in path_labales],
                       labels=[k for k in path_labales if k is not None],
                       dk=0.1)

# print (len(path.K_list))
path, bands = system.get_bandstructure(return_path=True, path=path)
# print (len(path.K_list))
plt.grid(True)
bands.plot_path_fat(path=path, save_file="bands.png")

# import numpy as np
from integrateFermi.contours2D import get_contours_and_WFs
from wannierberri.models import Haldane_ptb
from wannierberri.system import System_PythTB
# import ray
# ray.init()
system = System_PythTB(Haldane_ptb())
get_contours_and_WFs(system=system,
                     grid=50,
                     save_dir="contours",
                     )
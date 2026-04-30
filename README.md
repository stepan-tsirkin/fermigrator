# integrateFermi

A Python library for computing Fermi surface contours and scattering matrices for 2D electronic systems, built on top of [WannierBerri](https://wannierberri.org).

## What it does

1. Extracts Fermi surface contours from a band structure energy grid using a triangulation (tetrahedron-like) method
2. Evaluates Bloch wavefunctions on the contour k-points
3. Transforms a scattering matrix from Bloch gauge to Wannier (R-space) representation
4. Evaluates scattering matrix elements between contour points
5. Performs multipole decomposition of the scattering vertex
6. Computes QPI (quasiparticle interference) contour intersections

Results are stored in a file-based database of `.npz` files (`ContourDatabase`).

## Dependencies

- [WannierBerri](https://wannierberri.org)
- numpy
- ray (optional, for parallel contour computation)

## Workflow

```python
import wannierberri as wb
from integrateFermi.database import ContourDatabase
from integrateFermi.contours2D import get_contours_and_WFs
from integrateFermi.scattering_matrix import ScatteringMatrix, get_chk

# 1. Set up the system from Wannier90 files
from wannierberri import System_R
system = System_R.from_npz("my-system")

db = ContourDatabase("contours")
db.set_system(system)

# 2. Compute the energy grid and extract Fermi surface contours + wavefunctions
grid = wb.Grid(system, NK=(300, 300, 1))
db.evaluate_E_grid(grid=grid)
get_contours_and_WFs(contours_db=db, Efermi_list=[0.0])

# 3. Load scattering matrix and transform to Wannier gauge
Vkk = ...  # (NK, NK, NB, NB) array from DFT/DFPT
chk = get_chk("wannier/system.chk")
scatter = ScatteringMatrix.from_Vkk(
    Vkkmn_wan=Vkk,
    mp_grid=chk.mp_grid,
    kpt_red=chk.kpt_red,
    real_lattice=chk.real_lattice,
    wannier_centers_red=system.wannier_centers_red,
)

# 4. Evaluate scattering matrix and multipole vertices on contours
scatter.get_Vkk_on_contours_all(contours_db=db)
scatter.get_multipole_on_contours_all(contours_db=db)
scatter.to_npz("scatter_RR.npz")
```

## Example

A complete graphene example using DFT data is in [`examples/graphene_dft/`](examples/graphene_dft/):

| Script | Description |
|--------|-------------|
| `0.system.py` | Build system from Wannier90, plot DOS and bands |
| `1.calc_contours.py` | Compute Fermi surface contours and wavefunctions |
| `2.scatter_contours.py` | Compute scattering matrix on contours |

## Package structure

```
integrateFermi/
  contours2D.py          # Fermi surface extraction (triangulation method)
  database.py            # File-based storage for contours and matrices
  scattering_matrix.py   # ScatteringMatrix class + Bloch→Wannier transform
  get_band_wavefunction.py  # Wavefunction evaluation on k-points
  qpi.py                 # QPI contour intersection utilities
```

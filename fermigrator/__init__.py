"""
fermigrator - Integration methods for Fermi surfaces from band structure calculations.

This package provides tools for computing and analyzing Fermi surfaces using
electronic band structure calculations with WannierBerri.
"""

from .database import ContourDatabase
from .qpi import *  # noqa: F401, F403
from .get_band_wavefunction import get_wavefunction_on_kpoints
from .scattering_matrix import *  # noqa: F401, F403

__version__ = "0.1.0"
__author__ = "Author Name"

__all__ = [
    "ContourDatabase",
    "get_segments_triangle",
    "get_wavefunction_on_kpoints",
]

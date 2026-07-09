from functools import cached_property
import os
import glob
import numpy as np
from .fermisurface import FermiSurface


class ContourDatabase:
    """File-based database for Fermi surface contours and related quantities.

    Data are stored as `.npz` files under `path/` with names of the form
    `{type}_ib={band}_EF={energy}.npz`.  A `metadata.npz` in the same
    directory records the formatting precision so filenames can be parsed
    consistently across sessions.
    """

    file_types = ["contour", "vertex", "energies_grid", "Vkk", "linewidth"]

    def __init__(self, path, num_digits_Efermi=5, num_digits_band=4):
        """Create a new database directory.

        Parameters
        ----------
        path : str
            Directory to create (or reuse).  A ``metadata.npz`` is written
            immediately so precision settings survive across sessions.
        num_digits_Efermi : int
            Decimal places used when formatting Fermi energies in filenames.
        num_digits_band : int
            Zero-padded width for band indices in filenames.
        double_spin_system : bool
            if True, the system is considered to be the same for spin-up and spin-down electrons, so summation over spin will
            be performed
        """
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.num_digits_Efermi = num_digits_Efermi
        self.num_digits_band = num_digits_band
        self.files_dicts = {}
        self.nspin_system = 1
        np.savez(os.path.join(self.path, "metadata.npz"),
                 num_digits_Efermi=num_digits_Efermi, num_digits_band=num_digits_band)

    @classmethod
    def read(cls, path):
        """Open an existing database, restoring precision settings from metadata.npz."""
        metadata = np.load(os.path.join(path, "metadata.npz"))
        num_digits_Efermi = int(metadata["num_digits_Efermi"])
        num_digits_band = int(metadata["num_digits_band"])
        instance = cls(path, num_digits_Efermi=num_digits_Efermi,
                       num_digits_band=num_digits_band)
        return instance

    def format_EF(cls, Efermi):
        """Format a Fermi energy as a fixed-width signed string for use in filenames."""
        if isinstance(Efermi, str):
            Efermi = float(Efermi)
        Efermi = round(Efermi, cls.num_digits_Efermi)
        return f"{Efermi:+.{cls.num_digits_Efermi}f}"

    def format_band(cls, band):
        """Format a band index as a zero-padded string for use in filenames."""
        return f"{int(band):0{cls.num_digits_band}d}"

    def format(self, key, value):
        if key.startswith("EF"):
            return self.format_EF(value)
        elif key.startswith("ib"):
            return self.format_band(value)
        else:
            return str(value)

    @classmethod
    def split_filename(cls, filename):
        """Parse a database filename into a dict of its key=value components.

        Example: ``contour_ib=0001_EF=+0.12345.npz`` →
        ``{"type": "contour", "ib": "0001", "EF": "+0.12345"}``.
        Values are returned as strings; callers must cast as needed.
        """
        base = os.path.basename(filename).split(".npz")[0]
        parts = base.split("_")
        res = {}
        res["type"] = parts[0]
        for part in parts[1:]:
            k, v = part.split("=")
            res[k] = v
        return res

    @cached_property
    def nspin_system(self):
        return 2 if self.double_spin_system else 1

    def get_filename(self, typ, **kwargs):
        lst = [typ] + \
            [f"{k}={self.format(k, kwargs[k])}" for k in sorted(kwargs.keys())]
        return os.path.join(self.path, "_".join(lst) + ".npz")

    def get_contour_filename(self, ib, Efermi):
        return self.get_filename("contour", ib=ib, EF=self.format_EF(Efermi))

    def has_contour(self, ib, Efermi):
        filename = self.get_contour_filename(ib, Efermi)
        return os.path.exists(filename)

    def get_key(self, typ, **info):
        assert typ in self.file_types, f"Unknown file type: {typ}"
        if info["type"] in ["contour", "vertex", "linewidth"]:
            ib = int(info["ib"])
            Efermi = info["EF"]
            return (ib, Efermi)
        elif info["type"] == "energies_grid":
            return None

    def set_data(self, typ, data, **kwargs):
        """Save a dict of arrays to the corresponding .npz file."""
        data = {k: np.asarray(v) for k, v in data.items()}
        filename = self.get_filename(typ, **kwargs)
        np.savez(filename, **data)

    def get_data(self, typ, none_if_missing=False, **kwargs):
        """Load a .npz file and return its contents as a plain dict.

        Returns None (or raises FileNotFoundError) if the file does not exist.
        """
        filename = self.get_filename(typ, **kwargs)
        if os.path.exists(filename):
            f = np.load(filename, allow_pickle=True)
            return dict(f)
        else:
            if none_if_missing:
                return None
            else:
                raise FileNotFoundError(f"File not found: {filename}")

    def get_all_Efermi(self):
        """Return the set of Fermi energy strings present in the database (from contour files)."""
        Efermi_set = set()
        for fname in glob.glob(os.path.join(self.path, "contour_*.npz")):
            EF = self.split_filename(fname)["EF"]
            Efermi_set.add(EF)
        return Efermi_set

    def get_all_Efermi_float(self):
        """Return the set of Fermi energy floats present in the database (from contour files)."""
        Efermi_set = self.get_all_Efermi()
        Efermi_set = set(float(EF) for EF in Efermi_set)
        return np.array(sorted(Efermi_set))

    def get_all_bands(self, Efermi=None):
        """Return the set of band indices of contours stored in the database.

        If `Efermi` is given, restrict to files matching that Fermi level;
        otherwise return the union over all Fermi levels.
        """
        if Efermi is None:
            return set.union(*[self.get_all_bands(Efermi) for Efermi in self.get_all_Efermi()])
        else:
            band_set = set()
            Efermi_format = self.format_EF(Efermi)
            for fname in glob.glob(os.path.join(self.path, f"contour*_EF={Efermi_format}*.npz")):
                info = self.split_filename(fname)
                band_set.add(int(info["ib"]))
            return band_set

    def get_files_Efermi(self, typ, Efermi):
        """Return all file paths of a given type matching `Efermi`."""
        file_list = []
        for fname in glob.glob(os.path.join(self.path, f"{typ}_*.npz")):
            info = self.split_filename(fname)
            if info["EF"] == self.format_EF(Efermi):
                file_list.append(fname)
        return file_list

    def set_E_grid(self, energies_grid, rec_lattice):
        self.set_data("energies_grid", {
                      "energies": energies_grid, "rec_lattice": rec_lattice})

    def get_E_grid(self):
        data = self.get_data("energies_grid")
        if data is not None:
            return data["energies"], data["rec_lattice"]
        else:
            raise FileNotFoundError(
                "Energies grid not found in database. Please run evaluate_E_grid to compute and save it.")

    def evaluate_E_grid(self, grid, ignore_existing=False, dim=2):
        """Compute band energies on a k-grid using WannierBerri and save to the database.

        Parameters
        ----------
        grid : Grid or int or tuple/list/ndarray or dict
            The k-grid to use.  An int ``N`` → (N, N, 1); a 2-tuple → (N1, N2, 1);
            a dict is passed as kwargs to ``wannierberri.Grid``.
        ignore_existing : bool
            If False (default), return the cached result when one already exists.
        """
        assert dim in [2, 3], "Only dim=2 and dim=3 are supported, but got dim={dim}"
        if not ignore_existing:
            data = self.get_data("energies_grid")
            if data is not None:
                return data
        assert grid is not None, "grid should be provided if energies_grid is not loaded from file"
        from wannierberri.grid import Grid
        from wannierberri.calculators.tabulate import TabulatorAll
        from wannierberri import run

        if isinstance(grid, Grid):
            pass
        elif isinstance(grid, int):
            grid = Grid(self.system, NK=((grid, grid) + (1,) * (3 - dim)))
        elif isinstance(grid, tuple) or isinstance(grid, list) or isinstance(grid, np.ndarray):
            grid = tuple(grid)
            if dim == 2:
                assert len(grid) == dim, "grid tuple should have same length as dim, but got grid={grid} and dim={dim}"
                grid = grid + (1,) * (3 - dim)
            grid = Grid(self.system, NK=grid)
        elif isinstance(grid, dict):
            grid = Grid(self.system, **grid)
        else:
            raise ValueError(
                "grid should be either a wannierberri.Grid/tuple/list/ndarray of NK, or a dict of parameters for Grid")
        if dim == 2:
            assert grid.dense[2] == 1, "For a 2D system, the grid should have NK[2] = 1, but got NK[2] = {grid.dense[2]}"
        calculators = {"tab": TabulatorAll(tabulators={})}
        results = run(self.system, calculators=calculators, grid=grid)
        energies_grid = results.results["tab"].results["Energy"].data.reshape(
            tuple(grid.dense[:dim]) + (-1,))
        rec_lattice = self.system.recip_lattice[:dim, :dim]
        self.set_E_grid(energies_grid, rec_lattice)
        return {"energies": energies_grid, "rec_lattice": rec_lattice}

    def set_system(self, system):
        """Serialise a WannierBerri System_R to the database directory and cache it."""
        system.to_npz(os.path.join(self.path, "system"))
        self._system = system

    @cached_property
    def system(self):
        if hasattr(self, "_system"):
            return self._system
        else:
            from wannierberri import System_R
            return System_R.from_npz(os.path.join(self.path, "system"))

    def set_fermi_surfaces(self,
                           Efermi_list=None,
                           Nfermi=101,
                           ignore_existing=False,
                           ):
        energies_grid, rec_lattice = self.get_E_grid()
        if Efermi_list is None:
            Efermi_list = np.linspace(
                np.min(energies_grid), np.max(energies_grid), Nfermi)
        for ib in range(energies_grid.shape[-1]):
            for e in Efermi_list:
                if not self.has_contour(ib, e) or ignore_existing:
                    fermisurf = FermiSurface.from_grid(
                        energies_grid[..., ib], rec_lattice, e)
                    self.set_data("contour", fermisurf.as_dict(), ib=ib, EF=e)

from functools import cached_property
import os
import glob
import numpy as np


class ContourDatabase:

    file_types = ["contour", "vertex", "energies_grid", "Vkk", "linewidth"]

    def __init__(self, path, num_digits_Efermi=5, num_digits_band=4):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        # self.read(path)
        self.num_digits_Efermi = num_digits_Efermi
        self.num_digits_band = num_digits_band
        self.files_dicts = {}
        np.savez(os.path.join(self.path, "metadata.npz"),
                 num_digits_Efermi=num_digits_Efermi, num_digits_band=num_digits_band)

    @classmethod
    def read(cls, path):
        metadata = np.load(os.path.join(path, "metadata.npz"))
        num_digits_Efermi = int(metadata["num_digits_Efermi"])
        num_digits_band = int(metadata["num_digits_band"])
        instance = cls(path, num_digits_Efermi=num_digits_Efermi,
                       num_digits_band=num_digits_band)
        return instance

    def format_EF(cls, Efermi):
        if isinstance(Efermi, str):
            Efermi = float(Efermi)
        Efermi = round(Efermi, cls.num_digits_Efermi)
        return f"{Efermi:+.{cls.num_digits_Efermi}f}"

    def format_band(cls, band):
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
        base = os.path.basename(filename).split(".npz")[0]
        parts = base.split("_")
        res = {}
        res["type"] = parts[0]
        for part in parts[1:]:
            k, v = part.split("=")
            res[k] = v
        return res

    def get_filename(self, typ, **kwargs):
        lst = [typ] + \
            [f"{k}={self.format(k, kwargs[k])}" for k in sorted(kwargs.keys())]
        return os.path.join(self.path, "_".join(lst)+".npz")

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
        filename = self.get_filename(typ, **kwargs)
        np.savez(filename, **data)

    def get_data(self, typ, none_if_missing=True, **kwargs):
        filename = self.get_filename(typ, **kwargs)
        if os.path.exists(filename):
            f = np.load(filename)
            return dict(f)
        else:
            if none_if_missing:
                return None
            else:
                raise FileNotFoundError(f"File not found: {filename}")

    def get_all_Efermi(self):
        Efermi_set = set()
        for fname in glob.glob(os.path.join(self.path, f"contour_*.npz")):
            EF = self.split_filename(fname)["EF"]
            Efermi_set.add(EF)
        return Efermi_set

    def get_all_bands(self, Efermi=None):
        if Efermi is None:
            return set.union([self.get_all_bands_Efermi(Efermi) for Efermi in self.get_all_Efermi()])
        else:
            band_set = set()
            for key in self.files_dicts.get("contour", {}).keys():
                if key["EF"] == self.format_EF(Efermi):
                    band_set.add(int(key["ib"]))
            for typ in self.file_types:
                for f in glob.glob(os.path.join(self.path, f"{typ}_ib*EF={self.format_EF(Efermi)}.npz")):
                    info = self.split_filename(f)
                    for k, val in info.items():
                        if k.startswith("ib"):
                            band_set.add(int(info[k]))
            return band_set

    def get_files_Efermi(self, typ, Efermi):
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

    def evaluate_E_grid(self, grid, ignore_existing=False):
        if not ignore_existing:
            data = self.get_data("energies_grid")
            if data is not None:
                return data
        assert grid is not None, "grid should be provided if energies_grid is not loaded from file"
        from wannierberri.grid import Grid
        from wannierberri.calculators.tabulate import TabulatorAll
        from wannierberri import run

        if isinstance(grid, int):
            grid = Grid(self.system, NK=(grid, grid, 1))
        elif isinstance(grid, tuple) or isinstance(grid, list) or isinstance(grid, np.ndarray):
            assert len(grid) == 2, "grid should have length 2 for 2D system"
            grid = Grid(self.system, NK=(tuple(grid) + (1,)))
        elif isinstance(grid, dict):
            grid = Grid(self.system, **grid)
        else:
            raise ValueError(
                "grid should be either a tuple/list/ndarray of NK, or a dict of parameters for Grid")
        assert grid.dense[2] == 1, "grid should be 2D"
        calculators = {"tab": TabulatorAll(tabulators={})}
        results = run(self.system, calculators=calculators, grid=grid)
        energies_grid = results.results["tab"].results["Energy"].data.reshape(
            tuple(grid.dense[:2]) + (-1,))
        rec_lattice = self.system.recip_lattice[:2, :2]
        self.set_E_grid(energies_grid, rec_lattice)
        return {"energies": energies_grid, "rec_lattice": rec_lattice}

    def set_system(self, system):
        system.to_npz(os.path.join(self.path, "system"))
        self._system = system

    @cached_property
    def system(self):
        if hasattr(self, "_system"):
            return self._system
        else:
            from wannierberri import System_R
            return System_R.from_npz(os.path.join(self.path, "system"))

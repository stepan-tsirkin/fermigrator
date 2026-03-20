import os
import glob
import numpy as np


class ContourDatabase:

    file_types = ["contour", "vertex", "energies_grid", "Vkk", "linewidth"]
    def __init__(self, path, num_digits_Efermi=5, num_digits_band=4):
        self.path = path
        self.read(path)
        self.num_digits_Efermi = num_digits_Efermi
        self.num_digits_band = num_digits_band
        self.files_dicts={}

            
    def format_EF(cls, Efermi):
        if isinstance(Efermi, str):
            Efermi = float(Efermi)
        Efermi = round(Efermi, cls.num_digits_Efermi)
        return f"{Efermi:+.{cls.num_digits_Efermi}f}"
    
    def format_band(cls, band):
        return f"{band:0{cls.num_digits_band}d}"
    
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
            k,v = part.split("=")
            res[k] = v
        return res
        
    def get_filename(self, typ, **kwargs):
        lst = [typ] + [f"{k}={self.format(k, kwargs[k])}" for k in sorted(kwargs.keys())]
        return os.path.join(self.path, "_".join(lst)+".npz")
    
    def read(self, path=None):
        if path is None:
            path = self.path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        all_files = glob.glob(os.path.join(self.path, "*.npz"))
        for f in all_files:
            info = self.split_filename(f)
            self.get_key(info["type"], **info)


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


    def get_all_Efermi(self):
        Efermi_set = set()    
        for key in self.files_dicts.get("contour", {}).keys():
            Efermi_set.add(key["EF"])
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
        


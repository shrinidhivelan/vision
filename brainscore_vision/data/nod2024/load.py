from pathlib import Path
import xarray as xr

PROCESSED = Path("..") / "data_nod" / "processed"

def load_nod2024_v1v2():
    ds = xr.load_dataset(PROCESSED / "nod2024_imagenet_v1v2.nc")
    # just return DataArray; shape and coords are still Brain-Score style
    return ds["responses"]

def load_nod2024_roi(region: str):
    assembly = load_nod2024_v1v2()
    mask = assembly["roi"].values == region
    return assembly.isel(neuroid=mask)
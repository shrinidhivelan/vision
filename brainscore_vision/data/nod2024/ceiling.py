from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

PROCESSED = Path("..")/"data_nod"/"processed"
SUBJECTS = ["sub-01", "sub-02", "sub-03"]

def compute_split_half_ceiling():
    dfs = []
    for sub in SUBJECTS:
        path = PROCESSED / f"nod2024_coco_{sub}_v1v2.nc"
        if not path.exists():
            continue
        ds = xr.load_dataset(path)
        da = ds["responses"]  # (run, neuroid, presentation)

        runs = da.coords["run"].values
        half1 = da.sel(run=runs[::2]).mean("run").values   # (neuroid, presentation)
        half2 = da.sel(run=runs[1::2]).mean("run").values

        n_neuroid = half1.shape[0]
        ceilings = np.zeros(n_neuroid)
        for i in range(n_neuroid):
            x, y = half1[i], half2[i]
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r = np.corrcoef(x, y)[0, 1]
                if np.isnan(r):
                    r = 0.0
            r = max(r, 0.0)
            ceilings[i] = np.sqrt(r)

        df = pd.DataFrame({
            "subject": da["subject"].values,
            "roi": da["roi"].values,
            "vertex_id": da["vertex_id"].values,
            "ceiling": ceilings,
        })
        dfs.append(df)

    ceiling_df = pd.concat(dfs, ignore_index=True)
    out_path = PROCESSED / "nod2024_ceiling_v1v2.csv"
    ceiling_df.to_csv(out_path, index=False)
    print("wrote", out_path)
    return ceiling_df

if __name__ == "__main__":
    compute_split_half_ceiling()
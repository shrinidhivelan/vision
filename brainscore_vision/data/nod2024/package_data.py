from pathlib import Path
import numpy as np
import xarray as xr
import nibabel as nib
# We import the necessary libraries

DATA_ROOT = Path("..") / "data_nod" / "raw"
OUT_ROOT = Path("..") / "data_nod" / "processed"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["sub-01", "sub-02", "sub-03"]

def load_beta(beta_path: Path) -> np.ndarray:
    img = nib.load(str(beta_path))
    data = np.asarray(img.dataobj)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Unexpected beta shape {data.shape} for {beta_path}")
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data  # (vertices, conditions)

def load_ba_exvivo_labels(sub: str):
    """Return BA_exvivo label indices per vertex as int array."""
    ba_path = DATA_ROOT / sub / "rois" / f"{sub}.BA_exvivo.32k_fs_LR.dlabel.nii"
    img = nib.load(str(ba_path))
    labels = np.squeeze(np.asarray(img.dataobj))  # (vertices,)
    return labels  # integer codes

V1_CODES = {1}   
V2_CODES = {2}   

def get_v1_v2_masks(sub: str):
    labels = load_ba_exvivo_labels(sub)
    mask_v1 = np.isin(labels, list(V1_CODES))
    mask_v2 = np.isin(labels, list(V2_CODES))
    if mask_v1.sum() == 0 or mask_v2.sum() == 0:
        raise ValueError(f"Empty V1/V2 masks for {sub}. Update V1_CODES/V2_CODES.")
    return mask_v1, mask_v2

def package_imagenet_v1v2():
    subject_arrays = []
    subject_roi_labels = []
    subject_vertex_ids = []
    subject_subjects = []

    for sub in SUBJECTS:
        beta_paths = sorted((DATA_ROOT / sub / "imagenet").glob("*_beta.dscalar.nii"))
        if not beta_paths:
            continue

        run_arrays = [load_beta(p) for p in beta_paths]
        data = np.concatenate(run_arrays, axis=1)  # (vertices, total_presentations)

        mask_v1, mask_v2 = get_v1_v2_masks(sub)
        mask = mask_v1 | mask_v2
        data = data[mask, :]  # (neuroid, presentation)

        roi_labels = np.where(mask_v1[mask], "V1", "V2")
        vertex_ids = np.where(mask)[0]

        subject_arrays.append(data)
        subject_roi_labels.append(roi_labels)
        subject_vertex_ids.append(vertex_ids)
        subject_subjects.append(np.array([sub] * data.shape[0]))

    if not subject_arrays:
        raise RuntimeError("No ImageNet beta files found")

    n_presentations = {arr.shape[1] for arr in subject_arrays}
    if len(n_presentations) != 1:
        min_pres = min(n_presentations)
        subject_arrays = [arr[:, :min_pres] for arr in subject_arrays]
    else:
        min_pres = next(iter(n_presentations))

    all_data = np.concatenate(subject_arrays, axis=0)
    all_roi = np.concatenate(subject_roi_labels, axis=0)
    all_vertex = np.concatenate(subject_vertex_ids, axis=0)
    all_subject = np.concatenate(subject_subjects, axis=0)

    da = xr.DataArray(
        all_data,
        dims=("neuroid", "presentation"),
        coords={
            "neuroid": np.arange(all_data.shape[0]),
            "presentation": np.arange(min_pres),
            "roi": ("neuroid", all_roi),
            "vertex_id": ("neuroid", all_vertex),
            "subject": ("neuroid", all_subject),
            "stimulus_id": ("presentation", np.arange(min_pres).astype(str)),
        },
        name="responses",
    )
    xr.Dataset({"responses": da}).to_netcdf(OUT_ROOT / "nod2024_imagenet_v1v2.nc")

def package_coco_v1v2():
    for sub in SUBJECTS:
        beta_paths = sorted((DATA_ROOT / sub / "coco").glob("*_beta.dscalar.nii"))
        if not beta_paths:
            continue

        mask_v1, mask_v2 = get_v1_v2_masks(sub)
        mask = mask_v1 | mask_v2
        roi_labels = np.where(mask_v1[mask], "V1", "V2")
        vertex_ids = np.where(mask)[0]

        run_arrays = []
        for path in beta_paths:
            beta = load_beta(path)        # (vertices, conditions)
            beta = beta[mask, :]          # (neuroid, presentation)
            run_arrays.append(beta)

        min_pres = min(arr.shape[1] for arr in run_arrays)
        run_arrays = [arr[:, :min_pres] for arr in run_arrays]
        data = np.stack(run_arrays, axis=0)  # (run, neuroid, presentation)

        da = xr.DataArray(
            data,
            dims=("run", "neuroid", "presentation"),
            coords={
                "run": np.arange(data.shape[0]),
                "neuroid": np.arange(data.shape[1]),
                "presentation": np.arange(data.shape[2]),
                "roi": ("neuroid", roi_labels),
                "vertex_id": ("neuroid", vertex_ids),
                "subject": ("neuroid", np.array([sub] * data.shape[1])),
                "stimulus_id": ("presentation", np.arange(data.shape[2]).astype(str)),
            },
            name="responses",
        )

        out_path = OUT_ROOT / f"nod2024_coco_{sub}_v1v2.nc"
        xr.Dataset({"responses": da}).to_netcdf(out_path)
        print("wrote", out_path)

if __name__ == "__main__":
    package_imagenet_v1v2()
    package_coco_v1v2()
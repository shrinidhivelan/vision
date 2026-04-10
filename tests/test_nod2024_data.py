from brainscore_vision.data.nod2024.load import load_nod2024_v1v2

def test_nod2024_data_loads():
    assembly = load_nod2024_v1v2()
    assert "neuroid" in assembly.dims
    assert "presentation" in assembly.dims
    assert "roi" in assembly.coords
    assert "subject" in assembly.coords

def test_nod2024_has_three_subjects_and_two_rois():
    assembly = load_nod2024_v1v2()
    subjects = set(assembly["subject"].values.tolist())
    rois = set(assembly["roi"].values.tolist())
    assert {"sub-01", "sub-02", "sub-03"}.issubset(subjects)
    assert {"V1", "V2"}.issubset(rois)
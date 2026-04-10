from brainscore_vision import data_registry
from .load import load_nod2024_v1v2

# register main loader under a dataset identifier
data_registry["NOD2024-v1v2"] = load_nod2024_v1v2
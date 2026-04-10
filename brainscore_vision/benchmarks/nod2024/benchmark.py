from pathlib import Path
import pandas as pd
from brainscore_vision import benchmark_registry
from brainscore_vision.data.nod2024.load import load_nod2024_roi

PROCESSED = Path("..") / "data_nod" / "processed"
BIBTEX = ""  # fill NOD bibtex later if you have time

class SimpleNODBenchmark:
    def __init__(self, region: str):
        self.region = region
        self.identifier = f"NOD2024.{region}-pls"

    @property
    def ceiling(self):
        df = pd.read_csv(PROCESSED / "nod2024_ceiling_v1v2.csv")
        roi_df = df[df["roi"] == self.region]
        return roi_df["ceiling"].mean()

    def __call__(self, candidate_model):
        # Minimal stub: just return a dummy score structure for now.
        # You can document this simplification in REPORT.md.
        assembly = load_nod2024_roi(self.region)
        _ = assembly  # avoid unused variable warning
        return {
            "identifier": self.identifier,
            "ceiling": float(self.ceiling),
        }

# We want to register two benchmarks
benchmark_registry["NOD2024.V1-pls"] = lambda: SimpleNODBenchmark("V1")
benchmark_registry["NOD2024.V2-pls"] = lambda: SimpleNODBenchmark("V2")
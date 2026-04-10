# run_nod_tests.py

from brainscore_vision.benchmarks import nod2024  # <-- forces import & registration

from tests.test_nod2024_data import (
    test_nod2024_data_loads,
    test_nod2024_has_three_subjects_and_two_rois,
)
from tests.test_nod2024_benchmark import (
    test_nod_benchmarks_registered,
    test_nod_benchmark_ceiling_range,
)

if __name__ == "__main__":
    test_nod2024_data_loads()
    test_nod2024_has_three_subjects_and_two_rois()
    test_nod_benchmarks_registered()
    test_nod_benchmark_ceiling_range()
    print("All NOD2024 tests passed.")
from brainscore_vision.benchmarks import nod2024  # ensure NOD benchmarks are registered
from brainscore_vision.models import nod_alexnet  # ensure AlexNet is registered

from tests.test_nod2024_data import (
    test_nod2024_data_loads,
    test_nod2024_has_three_subjects_and_two_rois,
)
from tests.test_nod2024_benchmark import (
    test_nod_benchmarks_registered,
    test_nod_benchmark_ceiling_range,
)
from tests.test_nod_alexnet import (
    test_alexnet_registered,
    test_alexnet_layers_nonempty,
)

if __name__ == "__main__":
    test_nod2024_data_loads()
    test_nod2024_has_three_subjects_and_two_rois()
    test_nod_benchmarks_registered()
    test_nod_benchmark_ceiling_range()
    test_alexnet_registered()
    test_alexnet_layers_nonempty()
    print("All NOD2024 + AlexNet tests passed.")
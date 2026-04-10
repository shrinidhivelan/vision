from brainscore_vision import benchmark_registry

def test_nod_benchmarks_registered():
    assert "NOD2024.V1-pls" in benchmark_registry
    assert "NOD2024.V2-pls" in benchmark_registry

def test_nod_benchmark_ceiling_range():
    bench = benchmark_registry["NOD2024.V1-pls"]()
    c = bench.ceiling
    assert 0.0 <= c <= 1.0
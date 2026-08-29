import pytest

from release_tools.portable_benchmark import compare_predictions, run_benchmark, summarize


def test_summarize_reports_required_percentiles():
    summary = summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert set(summary) == {"count", "mean", "p50", "p95", "p99", "max"}
    assert summary["count"] == 5
    assert summary["mean"] == 30.0
    assert summary["max"] == 50.0
    assert summary["p50"] == 30.0


def test_compare_predictions_lists_every_cpu_gpu_difference():
    gpu = {"a.png": "front", "b.png": "back"}
    cpu = {"a.png": "back", "b.png": "back"}
    assert compare_predictions(gpu, cpu) == [
        {"image": "a.png", "gpu": "front", "cpu": "back"}
    ]


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-1.0], "not-a-list"])
def test_summarize_rejects_empty_nonfinite_negative_or_wrong_input(values):
    with pytest.raises(ValueError):
        summarize(values)


@pytest.mark.parametrize("gpu,cpu", [([], {}), ({"a": "front"}, []), ({"a": 1}, {"a": "front"})])
def test_compare_predictions_rejects_invalid_maps(gpu, cpu):
    with pytest.raises(ValueError):
        compare_predictions(gpu, cpu)


def test_compare_predictions_is_sorted_and_reports_missing_entries():
    gpu = {"z": "front", "a": "back"}
    cpu = {"z": "front", "b": "back"}
    assert compare_predictions(gpu, cpu) == [
        {"image": "a", "gpu": "back", "cpu": None},
        {"image": "b", "gpu": None, "cpu": "back"},
    ]


def test_run_benchmark_requires_at_least_1000_measured_requests():
    with pytest.raises(ValueError, match="at least 1000"):
        run_benchmark("missing.zip", "missing.json", iterations=999)

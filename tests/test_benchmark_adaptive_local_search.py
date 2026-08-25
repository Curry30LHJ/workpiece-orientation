from __future__ import annotations

import importlib
from pathlib import Path
import pytest
import sys

from scripts.benchmark_adaptive_local_search import (
    _collect_environment,
    _compare_worker_payloads,
    _ensure_project_import_path,
    _render_markdown,
)


def row(
    path: str,
    label: str,
    needs_review: bool,
    stage: str,
    calls: int,
    *,
    case: str = "M1",
    actual: str = "front",
    elapsed_ms: float = 400.0,
    wall_ms: float = 410.0,
    matching_ms: float = 300.0,
) -> dict:
    return {
        "case": case,
        "image_path": path,
        "actual": actual,
        "label": label,
        "needs_review": needs_review,
        "elapsed_ms": elapsed_ms,
        "wall_ms": wall_ms,
        "timings_ms": {"local_matching": matching_ms},
        "local_search": {
            "stage": stage,
            "matched_counts": {"front": calls // 2, "back": calls // 2},
        },
        "evidence": {
            "global_scores": {"front": 0.51, "back": 0.50},
            "global_margin": 0.01,
            "local_scores": {"front": 8.0, "back": 5.0},
            "local_margin": 3.0,
            "decision_source": "global",
            "trace": {"ranked_indices": {"front": [0], "back": [0]}},
        },
    }


def payload(mode: str, rows: list[dict]) -> dict:
    return {
        "mode": mode,
        "rows": rows,
        "environment": {
            "timestamp_utc": "2026-08-25T00:00:00+00:00",
            "platform": "Windows-test",
            "python": "3.10-test",
            "paddle": "3.2.2",
            "paddle_cuda": "11.8",
            "torch": "2.7.1",
            "torch_cuda": "11.8",
            "numpy": "1.26.4",
            "gpu_name": "GPU-test",
            "gpu_driver": "test-driver",
            "git_commit": "abc123",
            "git_dirty": True,
        },
        "benchmark": {
            "warmup_per_case": 5,
            "worker_pid": 123,
            "worker_port": None,
            "worker_total_wall_ms": sum(item["wall_ms"] for item in rows),
            "m1_workpiece_id": "31f082d1a04e486b9345846f4d585033",
            "cases": {
                case: {
                    "dataset_counts": {"front": 40, "back": 40},
                    "query_counts": {"front": 20, "back": 20},
                    "template_counts": {"front": 28, "back": 28},
                }
                for case in sorted({item["case"] for item in rows})
            },
        },
    }


def test_compare_requires_per_image_label_and_review_identity():
    exhaustive = payload(
        "exhaustive", [row("a.png", "front", True, "full", 56)]
    )
    adaptive = payload(
        "adaptive", [row("a.png", "front", False, "top5", 10)]
    )

    report = _compare_worker_payloads(exhaustive, adaptive)

    assert report["gate_passed"] is False
    assert report["default_local_search_mode"] == "exhaustive"
    assert report["label_mismatches"] == []
    assert [item["image_path"] for item in report["review_mismatches"]] == [
        "a.png"
    ]
    assert report["review_mismatches"][0]["adaptive"]["evidence"]["trace"]


def test_compare_retains_every_per_image_timing_and_trace():
    exhaustive_row = row("a.png", "front", False, "full", 56)
    adaptive_row = row("a.png", "front", False, "top5", 10)

    report = _compare_worker_payloads(
        payload("exhaustive", [exhaustive_row]),
        payload("adaptive", [adaptive_row]),
    )

    assert report["worker_rows"]["exhaustive"] == [exhaustive_row]
    assert report["worker_rows"]["adaptive"] == [adaptive_row]
    assert report["worker_rows"]["adaptive"][0]["wall_ms"] == 410.0
    assert report["worker_rows"]["adaptive"][0]["timings_ms"]["local_matching"] == 300.0
    assert report["worker_rows"]["adaptive"][0]["evidence"]["trace"]


def test_compare_rejects_duplicate_or_missing_rows():
    duplicate = row("a.png", "front", False, "full", 56)
    with pytest.raises(ValueError, match="duplicate exhaustive row"):
        _compare_worker_payloads(
            payload("exhaustive", [duplicate, duplicate.copy()]),
            payload("adaptive", [duplicate]),
        )

    with pytest.raises(ValueError, match="worker row keys differ"):
        _compare_worker_payloads(
            payload("exhaustive", [row("a.png", "front", False, "full", 56)]),
            payload("adaptive", [row("b.png", "front", False, "top5", 10)]),
        )


def test_compare_reports_latency_stages_matched_counts_and_accuracy():
    exhaustive_rows = [
        row("a.png", "front", False, "full", 56, wall_ms=800.0, matching_ms=600.0),
        row(
            "b.png",
            "front",
            False,
            "full",
            56,
            actual="back",
            wall_ms=1000.0,
            matching_ms=800.0,
        ),
    ]
    adaptive_rows = [
        row("a.png", "front", False, "top5", 10, wall_ms=400.0, matching_ms=200.0),
        row(
            "b.png",
            "front",
            False,
            "top10",
            20,
            actual="back",
            wall_ms=500.0,
            matching_ms=300.0,
        ),
    ]

    report = _compare_worker_payloads(
        payload("exhaustive", exhaustive_rows), payload("adaptive", adaptive_rows)
    )

    assert report["gate_passed"] is True
    assert report["release_gate_passed"] is False
    assert report["default_local_search_mode"] == "exhaustive"
    m1 = report["case_statistics"]["M1"]
    assert m1["adaptive"]["stage_counts"] == {"top5": 1, "top10": 1, "full": 0}
    assert m1["exhaustive"]["stage_counts"] == {"top5": 0, "top10": 0, "full": 2}
    assert m1["adaptive"]["average_matched_count"] == 15.0
    assert m1["adaptive"]["wall_ms"]["p50"] == 450.0
    assert m1["adaptive"]["local_matching_ms"]["p95"] == 295.0
    assert m1["adaptive"]["accuracy"] == 0.5
    assert report["performance"]["wall_speedup"] == 2.0


def test_release_gate_requires_all_three_fixed_datasets():
    m1_only = [row("m1.png", "front", False, "top5", 10)]
    incomplete = _compare_worker_payloads(
        payload("exhaustive", m1_only), payload("adaptive", m1_only)
    )
    assert incomplete["gate_passed"] is True
    assert incomplete["release_gate_passed"] is False
    assert incomplete["missing_release_cases"] == ["M2", "M7"]
    assert incomplete["default_local_search_mode"] == "exhaustive"

    complete_rows = [
        row(f"{case.lower()}.png", "front", False, "top5", 10, case=case)
        for case in ("M1", "M2", "M7")
    ]
    complete = _compare_worker_payloads(
        payload("exhaustive", complete_rows), payload("adaptive", complete_rows)
    )
    assert complete["release_gate_passed"] is True
    assert complete["missing_release_cases"] == []
    assert complete["default_local_search_mode"] == "adaptive"


def test_markdown_reports_environment_stage_counts_latency_and_gate():
    rows = [
        row(f"{case.lower()}.png", "front", False, "top5", 10, case=case)
        for case in ("M1", "M2", "M7")
    ]
    report = _compare_worker_payloads(
        payload("exhaustive", rows), payload("adaptive", rows)
    )

    markdown = _render_markdown(report)

    assert "逐图标签一致：通过" in markdown
    assert "逐图复检状态一致：通过" in markdown
    assert "正式默认模式：`adaptive`" in markdown
    assert "GPU-test" in markdown
    assert "test-driver" in markdown
    assert "31f082d1a04e486b9345846f4d585033" in markdown
    assert "P50" in markdown
    assert "local_matching" in markdown
    assert "top5" in markdown


def test_environment_collection_does_not_load_model_runtimes(tmp_path):
    assert "paddle" not in sys.modules
    assert "torch" not in sys.modules

    environment = _collect_environment(tmp_path)

    assert environment["paddle"]
    assert environment["torch"]
    assert "paddle" not in sys.modules
    assert "torch" not in sys.modules


def test_worker_makes_project_sources_importable(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    filtered = [
        value
        for value in sys.path
        if Path(value or ".").resolve() != project_root
    ]
    monkeypatch.setattr(sys, "path", filtered)
    monkeypatch.delitem(sys.modules, "src", raising=False)
    monkeypatch.delitem(sys.modules, "src.shitu_baseline", raising=False)

    _ensure_project_import_path(project_root)

    module = importlib.import_module("src.shitu_baseline")
    assert module.split_labels.__module__ == "src.shitu_baseline"

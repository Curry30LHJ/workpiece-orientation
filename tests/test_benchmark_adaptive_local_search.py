from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import pytest
import subprocess
import sys
from types import SimpleNamespace

from scripts.benchmark_adaptive_local_search import (
    M1_WORKPIECE_ID,
    _build_case_specs,
    _build_input_fingerprint,
    _collect_environment,
    _compare_worker_payloads,
    _ensure_project_import_path,
    _fingerprint_validation_issues,
    _render_markdown,
    _run_parent,
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


def _test_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def synthetic_input_fingerprint(marker: str = "same") -> dict:
    def file_entry(path: str) -> dict:
        return {
            "path": path,
            "size": 10,
            "sha256": hashlib.sha256(f"{marker}:{path}".encode()).hexdigest(),
        }

    sources = [
        file_entry("scripts/benchmark_adaptive_local_search.py"),
        file_entry("src/geometry_profile_schema.py"),
        file_entry("src/model_execution_gate.py"),
        file_entry("src/orientation_classifier.py"),
    ]
    model_files = [file_entry("inference.pdmodel"), file_entry("inference.pdiparams")]
    selections = {}
    for case in ("M1", "M2", "M7"):
        direction_payload = {}
        template_count = 28 if case == "M1" else 20
        for direction in ("front", "back"):
            templates = [
                file_entry(f"{case}/{direction}/template-{index:02d}.png")
                for index in range(template_count)
            ]
            queries = [
                file_entry(f"{case}/{direction}/query-{index:02d}.png")
                for index in range(20)
            ]
            direction_payload[direction] = {
                "templates": templates,
                "queries": queries,
            }
        case_payload = {
            **direction_payload,
            "template_query_overlap_count": 0,
        }
        case_payload["aggregate_sha256"] = _test_sha(case_payload)
        selections[case] = case_payload
    body = {
        "schema_version": 1,
        "git": {
            "head": "a" * 40,
            "tracked_binary_diff_sha256": hashlib.sha256(marker.encode()).hexdigest(),
        },
        "project_sources": {
            "files": sources,
            "aggregate_sha256": _test_sha(sources),
        },
        "model": {
            "files": model_files,
            "aggregate_sha256": _test_sha(model_files),
        },
        "m1_artifacts": {
            "manifest": {"status": "present", "file": file_entry("M1/manifest.json")},
            "template_cache": {
                "status": "present",
                "file": file_entry("M1/.template_cache.pkl"),
            },
            "active_geometry_profile": {
                "status": "absent",
                "revision": None,
                "file": None,
            },
        },
        "selections": selections,
    }
    return {**body, "overall_sha256": _test_sha(body)}


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
        "input_fingerprint": synthetic_input_fingerprint(),
    }


def release_rows(*, case_counts: dict[str, tuple[int, int]] | None = None) -> list[dict]:
    counts = case_counts or {case: (20, 20) for case in ("M1", "M2", "M7")}
    rows = []
    for case, (front_count, back_count) in counts.items():
        for actual, count in (("front", front_count), ("back", back_count)):
            for index in range(count):
                rows.append(
                    row(
                        f"{case.lower()}-{actual}-{index:02d}.png",
                        actual,
                        False,
                        "top5",
                        10,
                        case=case,
                        actual=actual,
                    )
                )
    return rows


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


def test_compare_records_explicit_label_mismatch_with_full_evidence():
    exhaustive_row = row("a.png", "front", False, "full", 56)
    adaptive_row = row("a.png", "back", False, "top5", 10)

    report = _compare_worker_payloads(
        payload("exhaustive", [exhaustive_row]),
        payload("adaptive", [adaptive_row]),
    )

    assert report["gate_passed"] is False
    assert report["review_mismatches"] == []
    assert report["label_mismatches"][0]["image_path"] == "a.png"
    assert report["label_mismatches"][0]["exhaustive"]["evidence"]["trace"]
    assert report["label_mismatches"][0]["adaptive"]["evidence"]["trace"]


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

    complete_rows = release_rows()
    complete = _compare_worker_payloads(
        payload("exhaustive", complete_rows), payload("adaptive", complete_rows)
    )
    assert complete["release_gate_passed"] is True
    assert complete["missing_release_cases"] == []
    assert complete["default_local_search_mode"] == "adaptive"


@pytest.mark.parametrize(
    ("case_counts", "expected_code"),
    [
        ({"M1": (1, 0), "M2": (20, 20), "M7": (20, 20)}, "case_row_count"),
        ({"M1": (20, 19), "M2": (20, 20), "M7": (20, 20)}, "case_row_count"),
        ({"M1": (20, 20), "M2": (20, 19), "M7": (20, 20)}, "case_row_count"),
        ({"M1": (21, 19), "M2": (20, 20), "M7": (20, 20)}, "actual_distribution"),
    ],
)
def test_release_gate_rejects_incomplete_case_shapes(case_counts, expected_code):
    rows = release_rows(case_counts=case_counts)

    report = _compare_worker_payloads(
        payload("exhaustive", rows), payload("adaptive", rows)
    )

    assert report["gate_passed"] is True
    assert report["release_gate_passed"] is False
    assert report["default_local_search_mode"] == "exhaustive"
    assert expected_code in {item["code"] for item in report["release_scope_issues"]}


def test_release_gate_rejects_an_extra_case_even_when_pairwise_equal():
    rows = release_rows()
    rows.extend(release_rows(case_counts={"M9": (20, 20)}))

    report = _compare_worker_payloads(
        payload("exhaustive", rows), payload("adaptive", rows)
    )

    assert report["gate_passed"] is True
    assert report["release_gate_passed"] is False
    assert report["unexpected_release_cases"] == ["M9"]
    assert "unexpected_case" in {
        item["code"] for item in report["release_scope_issues"]
    }


def test_release_gate_rejects_worker_input_fingerprint_mismatch():
    rows = release_rows()
    exhaustive = payload("exhaustive", rows)
    adaptive = payload("adaptive", rows)
    adaptive["input_fingerprint"] = synthetic_input_fingerprint("changed")

    report = _compare_worker_payloads(exhaustive, adaptive)

    assert report["gate_passed"] is True
    assert report["release_gate_passed"] is False
    assert report["default_local_search_mode"] == "exhaustive"
    assert report["input_fingerprint_match"] is False
    assert "input_fingerprint_mismatch" in {
        item["code"] for item in report["input_fingerprint_issues"]
    }


def test_release_gate_rejects_missing_worker_input_fingerprint():
    rows = release_rows()
    exhaustive = payload("exhaustive", rows)
    adaptive = payload("adaptive", rows)
    adaptive.pop("input_fingerprint")

    report = _compare_worker_payloads(exhaustive, adaptive)

    assert report["gate_passed"] is True
    assert report["release_gate_passed"] is False
    assert report["input_fingerprint_match"] is False
    assert {
        (item["code"], item.get("mode"))
        for item in report["input_fingerprint_issues"]
    } >= {("missing_input_fingerprint", "adaptive")}


def test_markdown_reports_environment_stage_counts_latency_and_gate():
    rows = release_rows()
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
    assert "输入指纹" in markdown
    assert "mean" in markdown
    assert "模板/查询重叠" in markdown


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


def test_parent_writes_both_reports_and_returns_two_when_a_worker_fails(
    monkeypatch, tmp_path
):
    def fail_worker(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "worker")

    monkeypatch.setattr(subprocess, "run", fail_worker)
    output_json = tmp_path / "result.json"
    output_markdown = tmp_path / "result.md"
    args = SimpleNamespace(
        project_root=tmp_path,
        model_dir=tmp_path / "model",
        library_dir=tmp_path / "library",
        m1_workpiece_id=M1_WORKPIECE_ID,
        warmup=5,
        output_json=output_json,
        output_markdown=output_markdown,
    )

    exit_code = _run_parent(args)

    assert exit_code == 2
    assert output_json.is_file()
    assert output_markdown.is_file()
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["release_gate_passed"] is False
    assert report["default_local_search_mode"] == "exhaustive"
    assert report["input_fingerprint_match"] is False
    assert report["input_fingerprint_issues"][0]["code"] == "benchmark_execution_failed"
    assert report["error"]["type"] == "CalledProcessError"
    assert "执行错误" in output_markdown.read_text(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_BENCHMARK_INPUTS_AVAILABLE = all(
    path.exists()
    for path in (
        PROJECT_ROOT / "data" / "1_M1",
        PROJECT_ROOT / "data" / "1_M2",
        PROJECT_ROOT / "data" / "1_M7",
        PROJECT_ROOT / "runtime_library" / M1_WORKPIECE_ID,
        PROJECT_ROOT
        / "third_party"
        / "models"
        / "shiru_rec"
        / "general_PPLCNetV2_base_pretrained_v1.0_infer",
    )
)


@pytest.mark.skipif(
    not REAL_BENCHMARK_INPUTS_AVAILABLE,
    reason="local ignored M1/M2/M7 benchmark inputs are unavailable",
)
def test_real_selection_and_hash_fingerprint_are_isolated_without_loading_models():
    assert "paddle" not in sys.modules
    assert "torch" not in sys.modules
    library_dir = PROJECT_ROOT / "runtime_library"
    model_dir = (
        PROJECT_ROOT
        / "third_party"
        / "models"
        / "shiru_rec"
        / "general_PPLCNetV2_base_pretrained_v1.0_infer"
    )
    specs = _build_case_specs(PROJECT_ROOT, library_dir, M1_WORKPIECE_ID)
    assert [spec.name for spec in specs] == ["M1", "M2", "M7"]
    assert [len(spec.queries) for spec in specs] == [40, 40, 40]
    assert [
        (len(spec.templates["front"]), len(spec.templates["back"]))
        for spec in specs
    ] == [(28, 28), (20, 20), (20, 20)]

    _ensure_project_import_path(PROJECT_ROOT)
    importlib.import_module("src.orientation_classifier")
    fingerprint = _build_input_fingerprint(
        PROJECT_ROOT,
        model_dir,
        library_dir,
        M1_WORKPIECE_ID,
        specs,
    )

    assert _fingerprint_validation_issues(fingerprint) == []
    assert set(fingerprint["selections"]) == {"M1", "M2", "M7"}
    assert all(
        fingerprint["selections"][case]["template_query_overlap_count"] == 0
        for case in ("M1", "M2", "M7")
    )
    loaded_sources = {
        item["path"] for item in fingerprint["project_sources"]["files"]
    }
    assert "src/geometry_profile_schema.py" in loaded_sources
    assert "src/model_execution_gate.py" in loaded_sources
    assert fingerprint["m1_artifacts"]["manifest"]["status"] == "present"
    assert fingerprint["m1_artifacts"]["template_cache"]["status"] == "present"
    assert fingerprint["m1_artifacts"]["active_geometry_profile"]["status"] == "present"
    assert fingerprint["model"]["files"]
    assert "paddle" not in sys.modules
    assert "torch" not in sys.modules

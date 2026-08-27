from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import benchmark_adaptive_local_search as adaptive
from scripts import benchmark_fast_geometry_inference as benchmark
from scripts.benchmark_fast_geometry_inference import (
    _build_parser,
    audit_case_specs,
    combine_worker_payloads,
    evaluate_gates,
    evaluate_rule_effects,
    pairwise_accuracy,
    run_fast_measurements,
    run_isolated_workers,
    summarize_latencies,
)


def _prediction(label: str, *, elapsed: float, review: bool = False) -> dict:
    return {
        "label": label,
        "needs_review": review,
        "review_reason_codes": ["LOW_MARGIN"] if review else [],
        "geometry_status": "active",
        "decision_margin": 0.2,
        "timings_ms": {"decode": 1.0, "geometry_fit": 2.0, "total": elapsed},
    }


def _query(identity: str, expected: str = "front") -> dict:
    return {
        "query_identity": identity,
        "case": "M1",
        "dataset": "1_M1",
        "expected_orientation": expected,
        "image_path": f"C:/data/{identity}.png",
        "sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "width": 200,
        "height": 180,
    }


def _accuracy_row(identity: str, label: str, *, expected: str = "front", review: bool = False) -> dict:
    return {
        **_query(identity, expected),
        "result": _prediction(label, elapsed=10.0, review=review),
    }


def test_percentiles_use_all_samples_without_trimming():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 100.0])

    assert summary["samples"] == 5
    assert summary["mean_ms"] == pytest.approx(22.0)
    assert summary["max_ms"] == 100.0
    assert summary["p95_ms"] == pytest.approx(
        np.percentile([1.0, 2.0, 3.0, 4.0, 100.0], 95)
    )


@pytest.mark.parametrize(
    ("overrides", "failure_name"),
    [
        ({"added_errors": 1}, "added_errors"),
        ({"review_rate": 0.051}, "review_rate"),
        ({"p95_ms": 25.001}, "p95_ms"),
    ],
)
def test_gate_rejects_exact_added_error_review_and_latency_boundaries(
    overrides, failure_name
):
    values = {
        "added_errors": 0,
        "review_rate": 0.05,
        "p95_ms": 25.0,
        "rule_effects_passed": True,
    }
    passing = evaluate_gates(**values)
    failing = evaluate_gates(**{**values, **overrides})

    assert passing["passed"] is True
    assert passing["failures"] == []
    assert failing["passed"] is False
    assert [item["name"] for item in failing["failures"]] == [failure_name]


def test_report_keeps_unique_accuracy_rows_and_every_fast_repeat():
    legacy = {
        "mode": "legacy",
        "accuracy_rows": [
            _accuracy_row("q3", "front"),
            _accuracy_row("q1", "front"),
            _accuracy_row("q2", "back", expected="back"),
        ],
        "selection_inventory": {"M1": {"marker": "same"}},
    }
    fast = {
        "mode": "fast_geometry",
        "accuracy_rows": [
            _accuracy_row("q1", "front"),
            _accuracy_row("q2", "back", expected="back"),
            _accuracy_row("q3", "front"),
        ],
        "latency_rows": [
            {"query_identity": identity, "measurement_index": index, "timings_ms": {"total": 5.0 + index}}
            for index, identity in enumerate(("q1", "q2", "q3", "q1", "q2", "q3"))
        ],
        "rule_effects": [],
        "enabled_rule_directions": [],
        "selection_inventory": {"M1": {"marker": "same"}},
    }

    report = combine_worker_payloads(legacy, fast)

    assert [row["query_identity"] for row in report["accuracy_rows"]] == ["q1", "q2", "q3"]
    assert all(set(row) >= {"legacy", "fast"} for row in report["accuracy_rows"])
    assert len(report["latency_rows"]) == 6
    assert report["latency_rows"] == fast["latency_rows"]


def test_groups_use_fitted_metadata_and_published_rule_fields_not_filenames():
    legacy_row = _accuracy_row("opaque-id", "front")
    fast_row = _accuracy_row("opaque-id", "front")
    fast_row["result"].update({
        "applied_rules": [
            {"direction": "front", "mode": "inside", "name": "中心反光"}
        ],
        "fitted_angle_deg": 22.0,
        "fitted_center_offset_norm": 0.08,
    })
    legacy = {
        "mode": "legacy",
        "accuracy_rows": [legacy_row],
        "selection_inventory": {"M1": {"marker": "same"}},
    }
    fast = {
        "mode": "fast_geometry",
        "accuracy_rows": [fast_row],
        "latency_rows": [
            {"query_identity": "opaque-id", "measurement_index": 0, "timings_ms": {"total": 5.0}}
        ],
        "rule_effects": [],
        "enabled_rule_directions": [],
        "selection_inventory": {"M1": {"marker": "same"}},
    }

    report = combine_worker_payloads(legacy, fast)

    row = report["accuracy_rows"][0]
    assert row["applied_rules"][0]["name"] == "中心反光"
    assert row["fitted_angle_deg"] == 22.0
    assert row["fitted_center_offset_norm"] == 0.08
    assert set(report["groups"]) >= {
        "interference:front:inside:中心反光",
        "rotation:15-45",
        "offset:0.05-0.15",
        "crop:<=256",
    }


def test_warmups_are_excluded_from_measured_distribution():
    calls = []

    def predict(query):
        calls.append(query["query_identity"])
        elapsed = 1000.0 if len(calls) <= 5 else float(len(calls) - 5)
        return _prediction("front", elapsed=elapsed)

    result = run_fast_measurements(
        [_query("q0")],
        predict,
        warmup=5,
        repeats=1,
        minimum_measured_samples=3,
    )

    assert len(calls) == 8
    assert [row["timings_ms"]["total"] for row in result["latency_rows"]] == [1.0, 2.0, 3.0]
    assert result["latency_summary"]["samples"] == 3
    assert result["latency_summary"]["max_ms"] == 3.0


def test_minimum_samples_cycles_queries_deterministically_without_duplicate_accuracy():
    queries = [_query(f"q{index}") for index in range(3)]

    result = run_fast_measurements(
        queries,
        lambda query: _prediction("front", elapsed=4.0),
        warmup=0,
        repeats=1,
        minimum_measured_samples=8,
    )

    assert [row["query_identity"] for row in result["latency_rows"]] == [
        "q0", "q1", "q2", "q0", "q1", "q2", "q0", "q1"
    ]
    assert [row["query_identity"] for row in result["accuracy_rows"]] == ["q0", "q1", "q2"]


def test_repeats_are_all_measured_even_when_minimum_is_lower():
    queries = [_query("q0"), _query("q1")]

    result = run_fast_measurements(
        queries,
        lambda query: _prediction("front", elapsed=4.0),
        warmup=0,
        repeats=3,
        minimum_measured_samples=2,
    )

    assert [row["query_identity"] for row in result["latency_rows"]] == [
        "q0", "q1", "q0", "q1", "q0", "q1"
    ]


def test_added_error_is_only_legacy_correct_and_fast_wrong():
    rows = [
        {"query_identity": "improved", "expected_orientation": "front", "legacy": {"label": "back"}, "fast": {"label": "front"}},
        {"query_identity": "both-wrong", "expected_orientation": "front", "legacy": {"label": "back"}, "fast": {"label": "back"}},
        {"query_identity": "added", "expected_orientation": "front", "legacy": {"label": "front"}, "fast": {"label": "back"}},
    ]

    summary = pairwise_accuracy(rows)

    assert summary["added_errors"] == 1
    assert summary["added_error_identities"] == ["added"]
    assert summary["changed_prediction_identities"] == ["improved", "added"]


def _write_image(path: Path, marker: int) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (marker & 255, (marker >> 8) & 255, (marker >> 16) & 255)
    image[1, 1] = ((marker * 17) & 255, (marker * 31) & 255, (marker * 47) & 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_case_specs_are_exactly_m1_m2_m7_and_sha256_disjoint(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    library_root = tmp_path / "library"
    workpiece_id = "m1-test"
    manifest_root = library_root / workpiece_id
    manifest_root.mkdir(parents=True)
    (manifest_root / "manifest.json").write_text(
        json.dumps({"id": workpiece_id}), encoding="utf-8"
    )

    marker = 1
    for disk_label in ("0", "1"):
        for index in range(28):
            template = manifest_root / disk_label / f"template-{index:02d}.png"
            _write_image(template, marker)
            dataset_template = project_root / "data" / "1_M1" / disk_label / f"source-{index:02d}.png"
            dataset_template.parent.mkdir(parents=True, exist_ok=True)
            dataset_template.write_bytes(template.read_bytes())
            marker += 1
        _write_image(project_root / "data" / "1_M1" / disk_label / "query.png", marker)
        marker += 1

    for case in ("M2", "M7"):
        for disk_label in ("0", "1"):
            for index in range(2):
                _write_image(
                    project_root / "data" / f"1_{case}" / disk_label / f"image-{index}.png",
                    marker,
                )
                marker += 1

    monkeypatch.setattr(adaptive, "QUERY_COUNT", 1)
    monkeypatch.setattr(adaptive, "TEMPLATE_COUNT", 1)
    specs = adaptive._build_case_specs(project_root, library_root, workpiece_id)

    inventory, queries = audit_case_specs(project_root, specs)

    assert [spec.name for spec in specs] == ["M1", "M2", "M7"]
    assert list(inventory) == ["M1", "M2", "M7"]
    assert len(queries) == 6
    assert len({query["query_identity"] for query in queries}) == 6
    for case in ("M1", "M2", "M7"):
        template_hashes = {
            row["sha256"]
            for direction in ("front", "back")
            for row in inventory[case][direction]["templates"]
        }
        query_hashes = {
            row["sha256"]
            for direction in ("front", "back")
            for row in inventory[case][direction]["queries"]
        }
        assert template_hashes.isdisjoint(query_hashes)


def test_enabled_rule_effect_requires_nonempty_mask_and_embedding_change():
    enabled = ["front", "back"]
    identical = [
        {"direction": "front", "query_identity": "q1", "mask_pixels": 100, "mask_ratio": 0.1, "embedding_distance": 0.0},
        {"direction": "back", "query_identity": "q2", "mask_pixels": 0, "mask_ratio": 0.0, "embedding_distance": 2.0},
    ]

    failed = evaluate_rule_effects(enabled, identical)
    passed = evaluate_rule_effects(
        enabled,
        identical
        + [
            {"direction": "front", "query_identity": "q3", "mask_pixels": 20, "mask_ratio": 0.02, "embedding_distance": 1.1e-6},
            {"direction": "back", "query_identity": "q4", "mask_pixels": 30, "mask_ratio": 0.03, "embedding_distance": 0.3},
        ],
    )

    assert failed["passed"] is False
    assert failed["missing_directions"] == ["front", "back"]
    assert passed["passed"] is True
    assert [row["query_identity"] for row in passed["evidence"]] == ["q3", "q4"]


def test_parent_launches_separate_immutable_workers_and_joins_by_identity(tmp_path):
    commands = []

    def fake_run(command, *, check):
        commands.append(command)
        mode = command[command.index("--worker-mode") + 1]
        output = Path(command[command.index("--worker-output") + 1])
        if mode == "legacy":
            payload = {
                "mode": mode,
                "accuracy_rows": [_accuracy_row("q2", "back", expected="back"), _accuracy_row("q1", "front")],
                "selection_inventory": {"M1": {"marker": "same"}},
            }
        else:
            payload = {
                "mode": mode,
                "accuracy_rows": [_accuracy_row("q1", "front"), _accuracy_row("q2", "back", expected="back")],
                "latency_rows": [
                    {"query_identity": "q1", "measurement_index": 0, "timings_ms": {"total": 9.0}},
                    {"query_identity": "q2", "measurement_index": 1, "timings_ms": {"total": 10.0}},
                ],
                "enabled_rule_directions": [],
                "rule_effects": [],
                "selection_inventory": {"M1": {"marker": "same"}},
            }
        output.write_text(json.dumps(payload), encoding="utf-8")

    args = Namespace(
        project_root=tmp_path / "project",
        model_dir=tmp_path / "model",
        library_dir=tmp_path / "library",
        m1_workpiece_id="m1",
        warmup=2,
        repeats=3,
        minimum_measured_samples=8,
        output=tmp_path / "report.json",
        max_p95_ms=25.0,
        max_added_errors=0,
        max_review_rate=0.05,
    )

    payloads = run_isolated_workers(args, run_command=fake_run)
    report = combine_worker_payloads(payloads["legacy"], payloads["fast_geometry"])

    assert len(commands) == 2
    assert [command[command.index("--worker-mode") + 1] for command in commands] == ["legacy", "fast_geometry"]
    assert commands[0] is not commands[1]
    assert [row["query_identity"] for row in report["accuracy_rows"]] == ["q1", "q2"]
    fast_command = commands[1]
    assert fast_command[fast_command.index("--warmup") + 1] == "2"
    assert fast_command[fast_command.index("--repeats") + 1] == "3"
    assert fast_command[fast_command.index("--minimum-measured-samples") + 1] == "8"


def test_fast_m1_cache_path_never_calls_legacy_template_cache_loader(monkeypatch, tmp_path):
    spec = adaptive.CaseSpec(
        name="M1",
        dataset_dir=tmp_path / "dataset",
        templates={"front": (), "back": ()},
        queries=(),
        dataset_counts={"front": 0, "back": 0},
        cache_id="m1",
    )
    classifier = SimpleNamespace(inference_mode="fast_geometry")

    def forbidden_legacy_loader(*args, **kwargs):
        raise AssertionError("fast M1 path touched the legacy template-cache loader")

    monkeypatch.setattr(benchmark, "_load_m1_cache", forbidden_legacy_loader)
    monkeypatch.setattr(
        benchmark,
        "_load_fast_m1_cache",
        lambda classifier, spec, library_dir: ("fast-only-cache", 7),
        raising=False,
    )

    caches = benchmark._build_worker_caches(classifier, [spec], tmp_path)

    assert caches == {"M1": ("fast-only-cache", 7)}


def test_post_run_fast_worker_guard_rejects_loaded_local_stack(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.task8_probe", ModuleType("torch.task8_probe"))

    with pytest.raises(RuntimeError, match="local-feature stack"):
        benchmark._assert_fast_worker_is_local_feature_free()


def test_json_report_serializes_numpy_evidence_without_losing_values(tmp_path):
    output = tmp_path / "report.json"

    benchmark._write_json(
        output,
        {"score": np.float32(0.25), "shape": np.asarray([1, 2], dtype=np.int32)},
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "score": 0.25,
        "shape": [1, 2],
    }


def test_cli_help_runs_when_script_is_invoked_by_path():
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(project_root / "scripts" / "benchmark_fast_geometry_inference.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--minimum-measured-samples" in completed.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--warmup", "-1"],
        ["--repeats", "0"],
        ["--minimum-measured-samples", "0"],
        ["--max-p95-ms", "0"],
        ["--max-added-errors", "-1"],
        ["--max-review-rate", "-0.1"],
        ["--max-review-rate", "1.1"],
    ],
)
def test_cli_rejects_invalid_numeric_values(arguments, tmp_path):
    required = [
        "--project-root", str(tmp_path),
        "--model-dir", str(tmp_path),
        "--library-dir", str(tmp_path),
        "--m1-workpiece-id", "m1",
        "--output", str(tmp_path / "out.json"),
    ]

    with pytest.raises(SystemExit):
        _build_parser().parse_args(required + arguments)


def test_importing_benchmark_does_not_import_local_feature_runtimes():
    project_root = Path(__file__).resolve().parents[1]
    probe = (
        "import sys; import scripts.benchmark_fast_geometry_inference; "
        "forbidden=[name for name in sys.modules if name == 'torch' or "
        "name.startswith('torch.') or name == 'lightglue' or "
        "name.startswith('lightglue.') or name.startswith('aliked')]; "
        "print(forbidden); raise SystemExit(bool(forbidden))"
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"

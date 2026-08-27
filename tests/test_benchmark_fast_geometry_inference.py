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


def _fingerprint(marker: str = "same") -> dict:
    def digest(value) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def file_row(path: str) -> dict:
        return {
            "path": path,
            "size": 123,
            "sha256": digest(f"{marker}:{path}"),
        }

    source_files = [
        file_row("src/geometry_profile_schema.py"),
        file_row("src/model_execution_gate.py"),
    ]
    model_files = [file_row("inference.pdmodel")]
    selections = {}
    for case in ("M1", "M2", "M7"):
        case_payload = {}
        template_count = 28 if case == "M1" else adaptive.TEMPLATE_COUNT
        for direction in ("front", "back"):
            case_payload[direction] = {
                "templates": [
                    file_row(f"data/{case}/{direction}/template-{index:02d}.png")
                    for index in range(template_count)
                ],
                "queries": [
                    file_row(f"data/{case}/{direction}/query-{index:02d}.png")
                    for index in range(adaptive.QUERY_COUNT)
                ],
            }
        case_payload["template_query_overlap_count"] = 0
        case_payload["aggregate_sha256"] = digest(case_payload)
        selections[case] = case_payload

    body = {
        "schema_version": 1,
        "git": {
            "head": "a" * 40,
            "tracked_binary_diff_sha256": digest(f"diff:{marker}"),
            "tracked_changed_paths": [],
        },
        "project_sources": {
            "files": source_files,
            "aggregate_sha256": digest(source_files),
        },
        "model": {
            "files": model_files,
            "aggregate_sha256": digest(model_files),
        },
        "m1_artifacts": {
            "manifest": {
                "status": "present",
                "file": file_row("library/m1/manifest.json"),
            },
            "template_cache": {
                "status": "present",
                "file": file_row("library/m1/.template_cache.pkl"),
            },
            "active_geometry_profile": {
                "status": "not_configured",
                "revision": None,
                "file": None,
            },
        },
        "selections": selections,
    }
    return {**body, "overall_sha256": digest(body)}


def _refresh_outer_fingerprint(fingerprint: dict) -> dict:
    body = {
        key: value
        for key, value in fingerprint.items()
        if key != "overall_sha256"
    }
    overall = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**body, "overall_sha256": overall}


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
        "input_fingerprint": _fingerprint(),
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
        "enabled_rule_targets": [],
        "selection_inventory": {"M1": {"marker": "same"}},
        "input_fingerprint": _fingerprint(),
    }

    report = combine_worker_payloads(legacy, fast)

    assert [row["query_identity"] for row in report["accuracy_rows"]] == ["q1", "q2", "q3"]
    assert all(set(row) >= {"legacy", "fast"} for row in report["accuracy_rows"])
    assert len(report["latency_rows"]) == 6
    assert report["latency_rows"] == fast["latency_rows"]


def test_compare_rejects_different_complete_input_fingerprints_before_gates():
    legacy = {
        "mode": "legacy",
        "accuracy_rows": [_accuracy_row("q1", "front")],
        "selection_inventory": {"M1": {"marker": "same"}},
        "input_fingerprint": _fingerprint("legacy-model"),
    }
    fast = {
        "mode": "fast_geometry",
        "accuracy_rows": [_accuracy_row("q1", "front")],
        "latency_rows": [
            {"query_identity": "q1", "measurement_index": 0, "timings_ms": {"total": 5.0}}
        ],
        "enabled_rule_targets": [],
        "rule_effects": [],
        "selection_inventory": {"M1": {"marker": "same"}},
        "input_fingerprint": _fingerprint("fast-model"),
    }

    with pytest.raises(ValueError, match="input fingerprints differ"):
        combine_worker_payloads(legacy, fast)


@pytest.mark.parametrize(
    "fingerprint",
    [
        None,
        {},
        {"overall_sha256": "a" * 64},
        {**_fingerprint(), "overall_sha256": ""},
    ],
)
def test_compare_rejects_incomplete_or_invalid_input_fingerprint(fingerprint):
    legacy = {
        "mode": "legacy",
        "accuracy_rows": [_accuracy_row("q1", "front")],
        "selection_inventory": {"M1": {"marker": "same"}},
        "input_fingerprint": fingerprint,
    }
    fast = {
        "mode": "fast_geometry",
        "accuracy_rows": [_accuracy_row("q1", "front")],
        "latency_rows": [
            {"query_identity": "q1", "measurement_index": 0, "timings_ms": {"total": 5.0}}
        ],
        "enabled_rule_targets": [],
        "rule_effects": [],
        "selection_inventory": {"M1": {"marker": "same"}},
        "input_fingerprint": _fingerprint(),
    }

    with pytest.raises(ValueError, match="legacy input fingerprint"):
        combine_worker_payloads(legacy, fast)


@pytest.mark.parametrize(
    ("section", "expected_issue"),
    [
        ("project_sources", "project source aggregate sha256 is stale"),
        ("model", "model aggregate sha256 is stale"),
    ],
)
def test_compare_rejects_stale_inventory_aggregate_with_fresh_outer_hash(
    section, expected_issue
):
    fingerprint = _fingerprint()
    fingerprint[section]["aggregate_sha256"] = "f" * 64
    fingerprint = _refresh_outer_fingerprint(fingerprint)

    with pytest.raises(
        ValueError,
        match=rf"legacy input fingerprint.*{expected_issue}",
    ):
        benchmark._shared_input_fingerprint(
            {"input_fingerprint": fingerprint},
            {"input_fingerprint": json.loads(json.dumps(fingerprint))},
        )


@pytest.mark.parametrize("case", ["M1", "M2", "M7"])
def test_compare_rejects_stale_selection_aggregate_with_fresh_outer_hash(case):
    fingerprint = _fingerprint()
    fingerprint["selections"][case]["aggregate_sha256"] = "f" * 64
    fingerprint = _refresh_outer_fingerprint(fingerprint)

    with pytest.raises(
        ValueError,
        match=rf"legacy input fingerprint.*{case} selection aggregate sha256 is stale",
    ):
        benchmark._shared_input_fingerprint(
            {"input_fingerprint": fingerprint},
            {"input_fingerprint": json.loads(json.dumps(fingerprint))},
        )


def test_compare_rejects_malformed_artifact_inventory_with_fresh_outer_hash():
    fingerprint = _fingerprint()
    fingerprint["m1_artifacts"]["template_cache"] = {
        "status": "present",
        "file": {"path": "C:\\absolute-cache.pkl"},
    }
    fingerprint = _refresh_outer_fingerprint(fingerprint)

    with pytest.raises(
        ValueError,
        match=r"legacy input fingerprint.*M1 template_cache file path",
    ):
        benchmark._shared_input_fingerprint(
            {"input_fingerprint": fingerprint},
            {"input_fingerprint": json.loads(json.dumps(fingerprint))},
        )


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
        "input_fingerprint": _fingerprint(),
    }
    fast = {
        "mode": "fast_geometry",
        "accuracy_rows": [fast_row],
        "latency_rows": [
            {"query_identity": "opaque-id", "measurement_index": 0, "timings_ms": {"total": 5.0}}
        ],
        "rule_effects": [],
        "enabled_rule_targets": [],
        "selection_inventory": {"M1": {"marker": "same"}},
        "input_fingerprint": _fingerprint(),
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

    inventory, queries = audit_case_specs(specs)

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
    enabled = [
        {"case": "M1", "direction": "front"},
        {"case": "M1", "direction": "back"},
    ]
    identical = [
        {"case": "M1", "direction": "front", "query_identity": "q1", "mask_pixels": 100, "mask_ratio": 0.1, "embedding_distance": 0.0},
        {"case": "M1", "direction": "back", "query_identity": "q2", "mask_pixels": 0, "mask_ratio": 0.0, "embedding_distance": 2.0},
    ]

    failed = evaluate_rule_effects(enabled, identical)
    passed = evaluate_rule_effects(
        enabled,
        identical
        + [
            {"case": "M1", "direction": "front", "query_identity": "q3", "mask_pixels": 20, "mask_ratio": 0.02, "embedding_distance": 1.1e-6},
            {"case": "M1", "direction": "back", "query_identity": "q4", "mask_pixels": 30, "mask_ratio": 0.03, "embedding_distance": 0.3},
        ],
    )

    assert failed["passed"] is False
    assert failed["missing_targets"] == enabled
    assert passed["passed"] is True
    assert [row["query_identity"] for row in passed["evidence"]] == ["q3", "q4"]


def test_rule_effect_is_scoped_by_case_and_direction():
    targets = [
        {"case": "M1", "direction": "front"},
        {"case": "M7", "direction": "front"},
    ]
    evidence = [
        {
            "case": "M1",
            "direction": "front",
            "query_identity": "m1-front",
            "mask_pixels": 50,
            "mask_ratio": 0.05,
            "embedding_distance": 0.2,
        }
    ]

    result = evaluate_rule_effects(targets, evidence)

    assert result["passed"] is False
    assert result["missing_targets"] == [{"case": "M7", "direction": "front"}]
    assert result["evidence"][0]["case"] == "M1"


def test_parent_launches_separate_immutable_workers_and_joins_by_identity(tmp_path):
    commands = []

    def fake_run(command, *, check):
        assert check is False
        commands.append(command)
        mode = command[command.index("--worker-mode") + 1]
        output = Path(command[command.index("--worker-output") + 1])
        if mode == "legacy":
            payload = {
                "mode": mode,
                "accuracy_rows": [_accuracy_row("q2", "back", expected="back"), _accuracy_row("q1", "front")],
                "selection_inventory": {"M1": {"marker": "same"}},
                "input_fingerprint": _fingerprint(),
            }
        else:
            payload = {
                "mode": mode,
                "accuracy_rows": [_accuracy_row("q1", "front"), _accuracy_row("q2", "back", expected="back")],
                "latency_rows": [
                    {"query_identity": "q1", "measurement_index": 0, "timings_ms": {"total": 9.0}},
                    {"query_identity": "q2", "measurement_index": 1, "timings_ms": {"total": 10.0}},
                ],
                "enabled_rule_targets": [],
                "rule_effects": [],
                "selection_inventory": {"M1": {"marker": "same"}},
                "input_fingerprint": _fingerprint(),
            }
        output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0)

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
    assert "--output" not in fast_command
    assert "--max-p95-ms" not in fast_command
    assert "--max-added-errors" not in fast_command
    assert "--max-review-rate" not in fast_command


def test_failed_worker_payload_and_return_code_are_preserved_in_parent_report(
    monkeypatch, tmp_path
):
    args = Namespace(
        project_root=tmp_path / "project",
        model_dir=tmp_path / "model",
        library_dir=tmp_path / "library",
        m1_workpiece_id="m1",
        warmup=2,
        repeats=1,
        minimum_measured_samples=8,
        output=tmp_path / "report.json",
        max_p95_ms=25.0,
        max_added_errors=0,
        max_review_rate=0.05,
    )

    def fake_run(command, *, check):
        assert check is False
        mode = command[command.index("--worker-mode") + 1]
        output = Path(command[command.index("--worker-output") + 1])
        payload = {
            "mode": mode,
            "error": {
                "type": "SyntheticWorkerFailure",
                "message": "model load failed",
                "details": {"stage": "load"},
            },
        }
        output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=7)

    with pytest.raises(benchmark.WorkerProcessError) as captured:
        run_isolated_workers(args, run_command=fake_run)

    error = captured.value
    assert error.mode == "legacy"
    assert error.return_code == 7
    assert error.payload["error"]["details"] == {"stage": "load"}

    monkeypatch.setattr(benchmark, "_build_case_specs", lambda *args: [])
    monkeypatch.setattr(benchmark, "audit_case_specs", lambda specs: ({}, []))
    monkeypatch.setattr(
        benchmark,
        "run_isolated_workers",
        lambda args: (_ for _ in ()).throw(error),
    )

    assert benchmark._run_parent(args) == 2
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert report["worker_failure"] == {
        "mode": "legacy",
        "return_code": 7,
        "payload": error.payload,
    }
    assert report["error"]["type"] == "WorkerProcessError"


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
    monkeypatch.setitem(
        sys.modules,
        "src.soft_center_matcher",
        ModuleType("src.soft_center_matcher"),
    )

    with pytest.raises(RuntimeError, match="local-feature stack"):
        benchmark._assert_fast_worker_is_local_feature_free()


def test_fast_worker_guards_count_and_reject_every_local_entry_point(monkeypatch):
    monkeypatch.setattr(
        benchmark, "_assert_fast_worker_is_local_feature_free", lambda: None
    )
    classifier = SimpleNamespace(
        extractor=None,
        matcher=None,
        device=None,
        _extract_features=lambda *args, **kwargs: "aliked",
        _score_feature_pair=lambda *args, **kwargs: "lightglue",
    )
    guard = benchmark._FastWorkerLocalGuard(classifier)
    guard.install()
    try:
        assert guard.counts() == {"ALIKED": 0, "LightGlue": 0, "ORB": 0}
        with pytest.raises(RuntimeError, match="ALIKED"):
            classifier._extract_features(None)
        with pytest.raises(RuntimeError, match="LightGlue"):
            classifier._score_feature_pair(None, None, None, None)
        with pytest.raises(RuntimeError, match="ORB"):
            cv2.ORB_create()
        assert guard.counts() == {"ALIKED": 1, "LightGlue": 1, "ORB": 1}
    finally:
        guard.close()


def test_fast_worker_guards_orb_during_classifier_load_and_restores(
    monkeypatch, tmp_path
):
    from src.orientation_classifier import OrientationClassifier

    monkeypatch.setattr(
        benchmark, "_assert_fast_worker_is_local_feature_free", lambda: None
    )
    project_root = tmp_path / "project"
    model_dir = tmp_path / "model"
    library_dir = tmp_path / "library"
    for directory in (project_root, model_dir, library_dir):
        directory.mkdir()

    classifier = SimpleNamespace(
        inference_mode="fast_geometry",
        extractor=None,
        matcher=None,
        device=None,
        _extract_features=lambda *args, **kwargs: None,
        _score_feature_pair=lambda *args, **kwargs: None,
    )

    def load_with_forbidden_orb(*args, **kwargs):
        del args, kwargs
        cv2.ORB_create()
        return classifier

    monkeypatch.setattr(OrientationClassifier, "load", staticmethod(load_with_forbidden_orb))
    monkeypatch.setattr(benchmark, "_build_case_specs", lambda *args: [])
    monkeypatch.setattr(benchmark, "audit_case_specs", lambda specs: ({}, []))
    monkeypatch.setattr(benchmark, "_build_input_fingerprint", lambda *args: _fingerprint())
    monkeypatch.setattr(benchmark, "_build_worker_caches", lambda *args: {})
    original_orb_create = cv2.ORB_create
    args = Namespace(
        project_root=project_root,
        model_dir=model_dir,
        library_dir=library_dir,
        m1_workpiece_id="m1",
        worker_mode="fast_geometry",
        warmup=0,
        repeats=1,
        minimum_measured_samples=1,
    )

    with pytest.raises(RuntimeError, match="ORB") as captured:
        benchmark._run_worker(args)

    assert captured.value.local_feature_call_counts == {
        "ALIKED": 0,
        "LightGlue": 0,
        "ORB": 1,
    }
    assert cv2.ORB_create is original_orb_create


def test_worker_error_json_preserves_local_guard_counters(monkeypatch, tmp_path):
    output = tmp_path / "worker.json"
    error = benchmark._FastWorkerLocalFeatureViolation(
        "ORB",
        {"ALIKED": 0, "LightGlue": 0, "ORB": 1},
    )

    def fail_worker(args):
        del args
        raise error

    monkeypatch.setattr(benchmark, "_run_worker", fail_worker)
    monkeypatch.setattr(benchmark, "_collect_environment", lambda project_root: {})
    monkeypatch.setattr(sys, "argv", [
        "benchmark_fast_geometry_inference.py",
        "--worker-mode", "fast_geometry",
        "--worker-output", str(output),
        "--project-root", str(tmp_path / "project"),
        "--model-dir", str(tmp_path / "model"),
        "--library-dir", str(tmp_path / "library"),
        "--m1-workpiece-id", "m1",
    ])

    assert benchmark.main() == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["error"]["local_feature_call_counts"] == {
        "ALIKED": 0,
        "LightGlue": 0,
        "ORB": 1,
    }


@pytest.mark.parametrize("field", ["extractor", "matcher", "device"])
def test_fast_worker_guard_rejects_initialized_local_classifier_state(
    field, monkeypatch
):
    monkeypatch.setattr(
        benchmark, "_assert_fast_worker_is_local_feature_free", lambda: None
    )
    classifier = SimpleNamespace(
        extractor=None,
        matcher=None,
        device=None,
        _extract_features=lambda *args, **kwargs: None,
        _score_feature_pair=lambda *args, **kwargs: None,
    )
    setattr(classifier, field, object())

    with pytest.raises(RuntimeError, match=field):
        benchmark._FastWorkerLocalGuard(classifier).install()


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


def test_internal_worker_parser_does_not_require_parent_output_or_gates(tmp_path):
    args = _build_parser(worker=True).parse_args([
        "--worker-mode", "fast_geometry",
        "--worker-output", str(tmp_path / "worker.json"),
        "--project-root", str(tmp_path / "project"),
        "--model-dir", str(tmp_path / "model"),
        "--library-dir", str(tmp_path / "library"),
        "--m1-workpiece-id", "m1",
        "--warmup", "2",
        "--repeats", "3",
        "--minimum-measured-samples", "8",
    ])

    assert args.worker_mode == "fast_geometry"
    assert not hasattr(args, "output")
    assert not hasattr(args, "max_p95_ms")


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

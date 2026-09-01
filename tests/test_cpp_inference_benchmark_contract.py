import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_cpp_inference_benchmark.ps1"
CMAKE_FILE = REPO_ROOT / "native" / "ppshitu_rec_benchmark" / "CMakeLists.txt"


@pytest.fixture
def native_exe():
    configured = os.environ.get("WORKPIECE_CPP_BENCHMARK_EXE")
    if not configured:
        pytest.skip("WORKPIECE_CPP_BENCHMARK_EXE is not configured")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(
            "WORKPIECE_CPP_BENCHMARK_EXE points to a missing file: "
            f"{path}"
        )
    return path


@pytest.fixture
def native_model_dir():
    configured = os.environ.get("WORKPIECE_CPP_MODEL_DIR")
    if not configured:
        pytest.skip("WORKPIECE_CPP_MODEL_DIR is not configured")
    path = Path(configured)
    required = [path / "inference.pdmodel", path / "inference.pdiparams"]
    missing = [str(candidate) for candidate in required if not candidate.is_file()]
    if missing:
        pytest.fail("WORKPIECE_CPP_MODEL_DIR is incomplete: " + ", ".join(missing))
    return path


def test_build_script_rejects_missing_paddle_root_before_configuring(tmp_path):
    missing_paddle = tmp_path / "missing-paddle"
    missing_opencv = tmp_path / "missing-opencv"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_SCRIPT),
            "-PaddleInferenceRoot",
            str(missing_paddle),
            "-OpenCvRoot",
            str(missing_opencv),
            "-BuildDir",
            str(tmp_path / "build"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "PaddleInferenceRoot" in result.stderr
    assert not (tmp_path / "build").exists()


def test_native_cmake_graph_never_downloads_dependencies():
    content = CMAKE_FILE.read_text(encoding="utf-8")

    assert "FetchContent" not in content
    assert "ExternalProject" not in content
    assert "PADDLE_INFERENCE_ROOT" in content
    assert "OPENCV_ROOT" in content


def test_build_script_stages_only_the_vs2019_opencv_runtime(tmp_path):
    paddle_root = tmp_path / "paddle"
    (paddle_root / "paddle" / "include").mkdir(parents=True)
    (paddle_root / "paddle" / "lib").mkdir(parents=True)
    (paddle_root / "paddle" / "include" / "paddle_inference_api.h").write_text(
        "", encoding="utf-8"
    )
    (paddle_root / "paddle" / "lib" / "paddle_inference.lib").write_bytes(b"")
    (paddle_root / "paddle" / "lib" / "paddle_inference.dll").write_bytes(
        b"paddle"
    )

    opencv_root = tmp_path / "opencv"
    (opencv_root / "build").mkdir(parents=True)
    (opencv_root / "build" / "OpenCVConfig.cmake").write_text(
        "", encoding="utf-8"
    )
    vc14_bin = opencv_root / "build" / "x64" / "vc14" / "bin"
    vc15_bin = opencv_root / "build" / "x64" / "vc15" / "bin"
    vc14_bin.mkdir(parents=True)
    vc15_bin.mkdir(parents=True)
    (vc14_bin / "opencv_world460.dll").write_bytes(b"vc14")
    (vc15_bin / "opencv_world460.dll").write_bytes(b"vc15")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "cmake.cmd").write_text(
        "@echo off\n"
        "if \"%1\"==\"--build\" (\n"
        "  mkdir \"%~2\\Release\" 2>NUL\n"
        "  type NUL > \"%~2\\Release\\ppshitu_rec_benchmark.exe\"\n"
        ")\n"
        "exit /b 0\n",
        encoding="ascii",
    )
    environment = os.environ.copy()
    environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
    build_dir = tmp_path / "build"

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_SCRIPT),
            "-PaddleInferenceRoot",
            str(paddle_root),
            "-OpenCvRoot",
            str(opencv_root),
            "-BuildDir",
            str(build_dir),
        ],
        text=True,
        capture_output=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert (build_dir / "Release" / "opencv_world460.dll").read_bytes() == b"vc15"


@pytest.mark.integration
def test_native_cli_help_documents_required_inputs(native_exe):
    result = subprocess.run(
        [str(native_exe), "--help"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "--model-dir" in result.stdout
    assert "--image-list" in result.stdout
    assert "--report" in result.stdout


@pytest.mark.integration
def test_native_cli_rejects_zero_threads_with_stable_json(native_exe, tmp_path):
    report_path = tmp_path / "error.json"
    result = subprocess.run(
        [
            str(native_exe),
            "--model-dir",
            "missing",
            "--image-list",
            "missing.json",
            "--report",
            str(report_path),
            "--threads",
            "0",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["ok"] is False
    assert report["error"]["code"] == "INVALID_ARGUMENT"
    assert "threads" in report["error"]["message"]


def _write_lossless_png(path, bgr_pixels):
    ok, encoded = cv2.imencode(".png", np.asarray(bgr_pixels, dtype=np.uint8))
    assert ok
    encoded.tofile(path)


def _run_preprocess(native_exe, tmp_path, image_paths):
    image_list = tmp_path / "images.json"
    image_list.write_text(
        json.dumps([str(path) for path in image_paths], ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    dump_path = tmp_path / "inputs.f32"
    result = subprocess.run(
        [
            str(native_exe),
            "--model-dir",
            str(tmp_path / "unused-model"),
            "--image-list",
            str(image_list),
            "--report",
            str(report_path),
            "--preprocess-only",
            "--dump-inputs",
            str(dump_path),
            "--input-width",
            "2",
            "--input-height",
            "2",
            "--scale",
            str(1.0 / 255.0),
            "--mean-rgb",
            "0.485,0.456,0.406",
            "--std-rgb",
            "0.229,0.224,0.225",
        ],
        text=True,
        capture_output=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return result, report, dump_path


@pytest.mark.integration
def test_native_preprocess_matches_hand_checked_rgb_nchw_in_chinese_path(
    native_exe, tmp_path
):
    image_dir = tmp_path / "中文目录"
    image_dir.mkdir()
    image_path = image_dir / "彩色.png"
    _write_lossless_png(
        image_path,
        [
            [[0, 0, 0], [0, 0, 255]],
            [[0, 255, 0], [255, 0, 0]],
        ],
    )

    result, report, dump_path = _run_preprocess(
        native_exe, tmp_path, [image_path]
    )

    assert result.returncode == 0, result.stderr
    assert report["ok"] is True
    assert report["ordered_images"] == [str(image_path)]
    assert report["tensor_shape"] == [1, 3, 2, 2]
    actual = np.fromfile(dump_path, dtype=np.float32)
    expected_r = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    expected_g = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    expected_b = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    expected = np.concatenate(
        [
            (expected_r - 0.485) / 0.229,
            (expected_g - 0.456) / 0.224,
            (expected_b - 0.406) / 0.225,
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


@pytest.mark.integration
def test_native_preprocess_preserves_order_and_rejects_any_unreadable_image(
    native_exe, tmp_path
):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_lossless_png(first, np.zeros((2, 2, 3), dtype=np.uint8))
    _write_lossless_png(second, np.full((2, 2, 3), 255, dtype=np.uint8))

    result, report, _ = _run_preprocess(native_exe, tmp_path, [second, first])

    assert result.returncode == 0, result.stderr
    assert report["ordered_images"] == [str(second), str(first)]
    assert report["input_count"] == 2

    missing = tmp_path / "missing.png"
    failed, error_report, _ = _run_preprocess(
        native_exe, tmp_path, [first, missing, second]
    )
    assert failed.returncode == 3
    assert error_report["ok"] is False
    assert error_report["error"]["code"] == "IMAGE_UNREADABLE"
    assert "missing.png" in error_report["error"]["message"]


@pytest.mark.integration
def test_native_preprocess_rejects_malformed_image_list(native_exe, tmp_path):
    image_list = tmp_path / "images.json"
    image_list.write_text('["one.png", 2]', encoding="utf-8")
    report_path = tmp_path / "report.json"
    dump_path = tmp_path / "inputs.f32"

    result = subprocess.run(
        [
            str(native_exe),
            "--model-dir",
            str(tmp_path / "unused-model"),
            "--image-list",
            str(image_list),
            "--report",
            str(report_path),
            "--preprocess-only",
            "--dump-inputs",
            str(dump_path),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error"]["code"] == "IMAGE_LIST_INVALID"


@pytest.fixture
def five_images():
    images = sorted((REPO_ROOT / "data" / "1_M1" / "0").glob("*.png"))[:5]
    if len(images) != 5:
        pytest.fail("the native benchmark contract requires five M1 images")
    return images


@pytest.fixture
def native_run(native_exe, native_model_dir, tmp_path):
    invocation = 0

    def run(
        image_paths,
        *,
        workers=1,
        batch_size=1,
        warmup=0,
        iterations=1,
        threads=1,
    ):
        nonlocal invocation
        invocation += 1
        image_list = tmp_path / f"images-{invocation}.json"
        report_path = tmp_path / f"report-{invocation}.json"
        embeddings_path = tmp_path / f"embeddings-{invocation}.f32"
        image_list.write_text(
            json.dumps([str(path) for path in image_paths], ensure_ascii=False),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                str(native_exe),
                "--model-dir",
                str(native_model_dir),
                "--image-list",
                str(image_list),
                "--report",
                str(report_path),
                "--dump-embeddings",
                str(embeddings_path),
                "--workers",
                str(workers),
                "--batch-size",
                str(batch_size),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
                "--threads",
                str(threads),
            ],
            text=True,
            capture_output=True,
        )
        assert report_path.is_file(), result.stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert result.returncode == 0, report
        embeddings = np.fromfile(embeddings_path, dtype=np.float32).reshape(
            report["input_count"], report["feature_dimension"]
        )
        return report, embeddings

    return run


@pytest.mark.integration
def test_native_real_batch_preserves_order_and_returns_unit_embeddings(
    native_run, five_images
):
    report, embeddings = native_run(
        five_images, workers=1, batch_size=5, warmup=1, iterations=1
    )

    assert report["ok"] is True
    assert report["input_count"] == 5
    assert report["batch_size"] == 5
    assert embeddings.shape == (5, report["feature_dimension"])
    np.testing.assert_allclose(
        np.linalg.norm(embeddings, axis=1), np.ones(5), rtol=0, atol=1e-5
    )
    assert report["ordered_images"] == [str(path) for path in five_images]
    assert set(report["timings_ms"]) >= {
        "decode",
        "preprocess",
        "inference",
        "normalize",
        "total",
    }
    assert report["peak_working_set_bytes"] > 0


@pytest.mark.integration
def test_four_workers_report_four_distinct_predictor_instances(
    native_run, five_images
):
    report, embeddings = native_run(
        five_images, workers=4, batch_size=1, warmup=1, iterations=1
    )

    assert report["worker_count"] == 4
    assert len(set(report["predictor_instance_ids"])) == 4
    assert embeddings.shape[0] == 5


@pytest.mark.integration
def test_native_inference_reports_missing_model_as_model_load_failure(
    native_exe, five_images, tmp_path
):
    image_list = tmp_path / "images.json"
    image_list.write_text(json.dumps([str(five_images[0])]), encoding="utf-8")
    report_path = tmp_path / "report.json"
    embeddings_path = tmp_path / "embeddings.f32"

    result = subprocess.run(
        [
            str(native_exe),
            "--model-dir",
            str(tmp_path / "missing-model"),
            "--image-list",
            str(image_list),
            "--report",
            str(report_path),
            "--dump-embeddings",
            str(embeddings_path),
            "--iterations",
            "1",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 4
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error"]["code"] == "MODEL_LOAD_FAILED"

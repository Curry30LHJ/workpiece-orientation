import json
import os
from pathlib import Path
import subprocess

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

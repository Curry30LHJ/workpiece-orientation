from pathlib import Path
from types import SimpleNamespace

import pytest

import src.orientation_tcp_service as service


def test_prepare_windows_torch_dll_path(monkeypatch, tmp_path: Path):
    torch_lib = tmp_path / "Lib" / "site-packages" / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    monkeypatch.setattr(service.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(service.os, "name", "nt")
    monkeypatch.setenv("PATH", "original-path")
    added = []
    monkeypatch.setattr(service.os, "add_dll_directory", lambda path: added.append(path))

    service._prepare_windows_torch_dll_path()

    assert added == [str(torch_lib)]
    assert service.os.environ["PATH"].startswith(str(torch_lib))


def test_prepare_windows_numpy_dll_path_adds_frozen_runtime_dirs(monkeypatch, tmp_path: Path):
    runtime_root = tmp_path / "backend"
    internal = runtime_root / "_internal"
    # PyInstaller's onedir collector nests package-owned native libraries
    # under the package directories (the layout emitted by the release
    # builder), rather than flattening them beside ``_internal``.
    numpy_libs = internal / "numpy" / ".libs"
    paddle_libs = internal / "paddle" / "libs"
    cv2_dir = internal / "cv2"
    for path in (runtime_root, internal, numpy_libs, paddle_libs, cv2_dir):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service.sys, "frozen", True, raising=False)
    monkeypatch.setattr(service.sys, "executable", str(runtime_root / "orientation_backend.exe"))
    monkeypatch.setattr(service.os, "name", "nt")
    monkeypatch.setenv("PATH", "original-path")
    added = []
    handles = []

    def add_dll_directory(path):
        added.append(path)
        handle = object()
        handles.append(handle)
        return handle

    monkeypatch.setattr(service.os, "add_dll_directory", add_dll_directory)
    monkeypatch.setattr(service, "_windows_short_path", lambda path: None)
    monkeypatch.setattr(service, "_WINDOWS_DLL_DIRECTORY_HANDLES", [])

    service._prepare_windows_numpy_dll_path()

    assert added == [
        str(runtime_root),
        str(internal),
        str(numpy_libs),
        str(paddle_libs),
        str(cv2_dir),
    ]
    assert service.os.environ["PATH"].startswith(str(cv2_dir))
    assert service._WINDOWS_DLL_DIRECTORY_HANDLES == handles


def _required_service_args(tmp_path: Path) -> list[str]:
    return [
        "--project-root", str(tmp_path),
        "--model-dir", str(tmp_path / "models"),
        "--library-dir", str(tmp_path / "library"),
    ]


@pytest.mark.parametrize("mode", ["legacy", "fast_geometry", "compare"])
def test_parser_accepts_legacy_fast_geometry_and_compare(tmp_path: Path, mode: str):
    args = service._build_argument_parser().parse_args([
        *_required_service_args(tmp_path),
        "--inference-mode", mode,
    ])

    assert args.inference_mode == mode


def test_parser_defaults_to_legacy_inference_mode(tmp_path: Path):
    args = service._build_argument_parser().parse_args(_required_service_args(tmp_path))

    assert args.inference_mode == "legacy"


def test_parser_rejects_unknown_inference_mode(tmp_path: Path):
    with pytest.raises(SystemExit) as error:
        service._build_argument_parser().parse_args([
            *_required_service_args(tmp_path),
            "--inference-mode", "automatic",
        ])

    assert error.value.code == 2


def test_packaged_parser_accepts_identity_device_and_parent(tmp_path):
    args = service._build_argument_parser().parse_args([
        "--project-root", str(tmp_path),
        "--model-dir", str(tmp_path / "models"),
        "--data-root", str(tmp_path / "data"),
        "--paddle-config", str(tmp_path / "inference.yaml"),
        "--compute-device", "cpu",
        "--model-sha256", "a" * 64,
        "--package-version", "1.0.0",
        "--edition", "cpu",
        "--instance-token", "launch-123",
        "--parent-pid", "4321",
        "--inference-mode", "fast_geometry",
    ])
    assert args.data_root == tmp_path / "data"
    assert args.library_dir is None
    assert (args.compute_device, args.package_version, args.edition,
            args.instance_token, args.parent_pid) == (
        "cpu", "1.0.0", "cpu", "launch-123", 4321
    )


def test_parser_rejects_data_root_and_library_together(tmp_path):
    with pytest.raises(SystemExit):
        service._build_argument_parser().parse_args([
            "--project-root", str(tmp_path),
            "--model-dir", str(tmp_path / "models"),
            "--data-root", str(tmp_path / "data"),
            "--library-dir", str(tmp_path / "library"),
        ])


@pytest.mark.parametrize(
    ("instance_token", "model_sha256", "expected"),
    [
        ("", "a" * 64, "INVALID_INSTANCE_TOKEN"),
        ("launch-123", "not-a-sha", "INVALID_MODEL_SHA256"),
    ],
)
def test_packaged_main_rejects_invalid_identity_before_binding(
    monkeypatch, tmp_path, instance_token, model_sha256, expected,
):
    args = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        project_root=tmp_path,
        model_dir=tmp_path / "models",
        data_root=tmp_path / "data",
        library_dir=None,
        paddle_config=tmp_path / "inference.yaml",
        compute_device="gpu",
        model_sha256=model_sha256,
        package_version="1.0.0",
        edition="gpu",
        instance_token=instance_token,
        parent_pid=None,
        local_search_mode="adaptive",
        inference_mode="fast_geometry",
    )
    monkeypatch.setattr(
        service,
        "_build_argument_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setattr(
        service,
        "OrientationTcpServer",
        lambda *args, **kwargs: pytest.fail("listener bound before packaged validation"),
    )

    with pytest.raises(SystemExit, match=expected):
        service.main()


def test_packaged_main_creates_only_logs_before_binding(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    args = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        project_root=tmp_path,
        model_dir=tmp_path / "models",
        data_root=data_root,
        library_dir=None,
        paddle_config=tmp_path / "inference.yaml",
        compute_device="gpu",
        model_sha256="a" * 64,
        package_version="1.0.0",
        edition="gpu",
        instance_token="launch-123",
        parent_pid=None,
        local_search_mode="adaptive",
        inference_mode="fast_geometry",
    )

    class ListenerBound(Exception):
        pass

    def configure_logs(logs_dir):
        Path(logs_dir).mkdir(parents=True)

    def bind_listener(*args, **kwargs):
        assert {path.name for path in data_root.iterdir()} == {"logs"}
        raise ListenerBound

    monkeypatch.setattr(
        service,
        "_build_argument_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setattr(service, "configure_diagnostic_logging", configure_logs)
    monkeypatch.setattr(service, "_prepare_windows_numpy_dll_path", lambda: None)
    monkeypatch.setattr(service, "_prepare_windows_torch_dll_path", lambda: None)
    monkeypatch.setattr(service, "OrientationTcpServer", bind_listener)

    with pytest.raises(ListenerBound):
        service.main()

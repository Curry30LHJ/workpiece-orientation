from pathlib import Path

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

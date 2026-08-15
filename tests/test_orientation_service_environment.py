from pathlib import Path

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

from pathlib import Path


def test_release_script_exports_repo_on_python_module_path():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "$env:PYTHONPATH" in script
    assert "$repo" in script


def test_qt_build_uses_relative_project_argument_for_unicode_paths():
    script = (Path(__file__).parents[1] / "scripts" / "build_qt5.ps1").read_text(encoding="utf-8")
    assert '"..\\workpiece_orientation.pro" CONFIG+=release' in script


def test_release_script_prepends_selected_qt_bin_only_for_windeployqt():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    # Keep the machine-wide PATH untouched after deployment.  This protects a
    # Qt5 build when Qt6 happens to be earlier on PATH.
    assert "$previousPath = $env:PATH" in script
    assert '$env:PATH = "$([IO.Path]::GetFullPath($QtBin));$previousPath"' in script
    assert "$env:PATH = $previousPath" in script
    assert "finally" in script


def test_release_script_checks_each_external_build_step_and_passes_paddle_config():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    # A failed native/module invocation must stop the release rather than let a
    # later CPU iteration hide the original failure.
    for marker in (
        "Qt build failed",
        "windeployqt failed",
        "Portable backend build failed",
        "Portable package staging failed",
        "License collection failed",
        "Package audit/archive failed",
    ):
        assert marker in script
    assert "--paddle-config $configArg" in script

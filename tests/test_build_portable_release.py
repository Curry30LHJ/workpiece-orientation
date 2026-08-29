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
    assert "--paddle-config" in script
    assert "$configArg" in script


def test_release_scripts_create_and_resolve_nested_output_root():
    root = Path(__file__).parents[1]
    backend = (root / "scripts" / "build_portable_backend.ps1").read_text(encoding="utf-8")
    release = (root / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    for script in (backend, release):
        assert "Test-Path -LiteralPath $OutputRoot -PathType Container" in script
        assert "New-Item -ItemType Directory -Force -Path $OutputRoot" in script
        assert "Resolve-Path -LiteralPath $OutputRoot -ErrorAction Stop" in script
        assert ".ProviderPath" in script
        assert "DirectoryInfo" in script and ".FullName" in script
    assert "OutputRoot is not a directory" in backend
    assert "OutputRoot is not a directory" in release


def test_release_script_audits_the_extracted_zip_and_cleans_a_private_temp_root():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "audit_zip_archive" in script
    assert "ExtractToDirectory" not in script  # extraction is centralized and hardened in Python
    assert "Extracted ZIP audit failed" in script
    assert "release_artifacts" in script


def test_release_script_allows_explicit_msvc_runtime_directory():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "MsvcRuntimeDir" in script
    assert "--msvc-runtime-dir" in script


def test_backend_source_audit_filters_python_explicitly_for_windows_powershell():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_backend.ps1").read_text(encoding="utf-8")
    assert "Extension -ieq '.py'" in script
    assert "-Include '*.py'" not in script

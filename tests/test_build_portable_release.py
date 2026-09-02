from pathlib import Path
import shutil
import subprocess
import sys
import uuid


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


def test_release_script_uses_ascii_temp_copies_for_unicode_guide_and_notices():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "使用说明.txt" not in script
    assert "$asciiInputRoot = Join-Path ([IO.Path]::GetPathRoot($repo))" in script
    assert "$asciiGuide = Join-Path $asciiInputRoot 'guide.txt'" in script
    assert "$asciiNotices = Join-Path $asciiInputRoot 'notices.txt'" in script
    assert "Get-ChildItem -LiteralPath $deployDir -File -Filter '*.txt'" in script
    assert "Where-Object { $_.Name -ine 'THIRD_PARTY-NOTICES.txt' }" in script
    assert "-cne" not in script
    assert "guideCandidates.Count -eq 0" in script
    assert "guideCandidates.Count -gt 1" in script
    assert "'--guide', $asciiGuide, '--notices', $asciiNotices" in script
    assert "finally {" in script
    assert "Remove-Item -LiteralPath $asciiInputRoot -Recurse -Force" in script


def test_release_python_boundaries_use_repo_relative_paths():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "Push-Location -LiteralPath $repo" in script
    assert "'--qt-release-dir', $qtReleaseArg" in script
    assert "'--backend-dir', $backendArg" in script
    assert "'--output-root', $stagingRootArg" in script
    assert "'--repository-root', '.'" in script
    assert "Path.cwd().resolve()" in script
    assert "roots=[Path('.')" not in script
    assert "CreateProcessW" in script
    assert "$smokeReportArg = Join-Path (Join-Path $stagingRootArg 'reports')" in script
    assert "--report $smokeReportArg" in script
    assert "finally {" in script and "Pop-Location" in script


def test_release_audit_resolves_repository_root_inside_python():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "roots=[Path.cwd().resolve()," in script
    assert "roots=[Path('.')," not in script


def test_release_script_guards_pop_location_when_push_was_not_reached():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "$releaseLocationPushed = $false" in script
    assert "$releaseLocationPushed = $true" in script
    assert "if ($releaseLocationPushed) { Pop-Location }" in script


def test_smoke_report_path_join_executes_with_two_arguments():
    """The smoke report path must not call Join-Path with three positional args.

    PowerShell parses the old inline expression successfully, but fails only
    when the smoke branch executes.  Extract the production assignment and
    evaluate it with representative values so this regression catches that
    runtime-only binding error without running the full release build.
    """
    powershell = shutil.which("powershell")
    if powershell is None:
        return
    root = Path(__file__).parents[1]
    script = (root / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assignments = [line.strip() for line in script.splitlines() if line.strip().startswith("$smokeReportArg = Join-Path")]
    assert len(assignments) == 1
    assignment = assignments[0]
    harness = root / f".smoke-report-join-{uuid.uuid4().hex}.ps1"
    try:
        harness.write_text(
            "$stagingRootArg = 'release_staging\\final-1.0.0'\n"
            "$ed = 'gpu'\n"
            f"{assignment}\n"
            "if ($smokeReportArg -ne 'release_staging\\final-1.0.0\\reports\\gpu-smoke.json') { exit 2 }\n",
            encoding="utf-8-sig",
        )
        result = subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout
    finally:
        harness.unlink(missing_ok=True)


def test_windows_powershell_python_launch_preserves_unicode_argument():
    powershell = shutil.which("powershell")
    if powershell is None:
        return
    root = Path(__file__).parents[1]
    stem = f".unicode-launch-{uuid.uuid4().hex}"
    harness = root / f"{stem}.ps1"
    target = root / f"{stem}-中文结果.txt"
    try:
        harness.write_text(
            "$python = $args[0]\n"
            "$target = $args[1]\n"
            "& $python -c \"import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[1], encoding='utf-8')\" $target\n"
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n",
            encoding="utf-8-sig",
        )
        subprocess.run(
            [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness), sys.executable, str(target)],
            check=True,
        )
        assert target.read_text(encoding="utf-8") == str(target)
    finally:
        harness.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def test_backend_source_audit_filters_python_explicitly_for_windows_powershell():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_backend.ps1").read_text(encoding="utf-8")
    assert "Extension -ieq '.py'" in script
    assert "-Include '*.py'" not in script


def test_native_portable_release_script_is_explicit_and_separate():
    root = Path(__file__).parents[1]
    script_path = root / "scripts" / "build_native_portable_release.ps1"
    assert script_path.is_file()
    script = script_path.read_text(encoding="utf-8")
    assert "NativeRuntimeDir" in script
    assert "native_cpp" in script
    assert "WorkpieceOrientation-CPU-NativeCPP-x64" in script
    assert "ppshitu_rec_service.exe" in script

from pathlib import Path


def test_release_script_exports_repo_on_python_module_path():
    script = (Path(__file__).parents[1] / "scripts" / "build_portable_release.ps1").read_text(encoding="utf-8")
    assert "$env:PYTHONPATH" in script
    assert "$repo" in script

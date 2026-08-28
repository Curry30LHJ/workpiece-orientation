from pathlib import Path


SPEC = Path(__file__).parents[1] / "deploy" / "orientation_backend.spec"


def test_spec_project_root_does_not_evaluate_undefined_file():
    text = SPEC.read_text(encoding="utf-8")
    assert "Path(__file__)" not in text
    assert "WORKPIECE_PROJECT_ROOT" in text
    assert "Path.cwd()" in text


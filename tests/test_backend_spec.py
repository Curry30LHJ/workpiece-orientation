from pathlib import Path


SPEC = Path(__file__).parents[1] / "deploy" / "orientation_backend.spec"


def test_spec_project_root_does_not_evaluate_undefined_file():
    text = SPEC.read_text(encoding="utf-8")
    assert "Path(__file__)" not in text
    assert "WORKPIECE_PROJECT_ROOT" in text
    assert "Path.cwd()" in text


def test_spec_embeds_long_path_aware_manifest_for_frozen_extensions():
    text = SPEC.read_text(encoding="utf-8")
    assert "LONG_PATH_MANIFEST" in text
    assert "longPathAware" in text
    assert "manifest=LONG_PATH_MANIFEST" in text


def test_spec_uses_explicit_production_modules_and_no_loose_source_files():
    text = SPEC.read_text(encoding="utf-8")
    assert "production_src_modules" in text
    assert "*production_src_modules()" in text
    assert "src" in text
    assert "copy" not in text.lower()

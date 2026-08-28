from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "create_packaging_envs.ps1"


def test_packaging_script_clears_pip_no_index_and_checks_existing_python():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "$env:PIP_NO_INDEX = ''" in text
    assert "python --version" in text
    assert "3.10.20" in text
    assert "Packaging environment" in text


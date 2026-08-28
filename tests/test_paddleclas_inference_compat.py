from pathlib import Path
from types import SimpleNamespace

from src.paddleclas_inference_compat import create_rec_predictor, uses_legacy_model_format


def test_uses_legacy_model_format_requires_two_files_without_json(tmp_path: Path):
    assert not uses_legacy_model_format(tmp_path)
    (tmp_path / "inference.pdmodel").write_bytes(b"model")
    (tmp_path / "inference.pdiparams").write_bytes(b"params")
    assert uses_legacy_model_format(tmp_path)
    (tmp_path / "inference.json").write_text("{}", encoding="utf-8")
    assert not uses_legacy_model_format(tmp_path)


def test_legacy_predictor_temporarily_selects_old_paddle_config(tmp_path: Path):
    (tmp_path / "inference.pdmodel").write_bytes(b"model")
    (tmp_path / "inference.pdiparams").write_bytes(b"params")
    paddle = SimpleNamespace(__version__="3.2.2")
    seen = []
    config = SimpleNamespace(Global=SimpleNamespace(enable_mkldnn=True, ir_optim=True))

    def predictor(_config):
        seen.append(paddle.__version__)
        assert _config.Global.enable_mkldnn is True
        assert _config.Global.ir_optim is False
        return object()

    result = create_rec_predictor(predictor, config, paddle, tmp_path)
    assert result is not None
    assert seen == ["2.5.0"]
    assert paddle.__version__ == "3.2.2"
    assert config.Global.enable_mkldnn is True
    assert config.Global.ir_optim is True

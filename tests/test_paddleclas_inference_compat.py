from pathlib import Path
from types import SimpleNamespace

import src.paddleclas_inference_compat as compat
from src.paddleclas_inference_compat import (
    create_rec_predictor,
    prepare_paddle_model_path,
    uses_legacy_model_format,
)


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


def test_prepare_model_path_uses_ascii_copy_when_windows_path_is_unicode(tmp_path: Path, monkeypatch):
    source = tmp_path / "模型目录"
    source.mkdir()
    for name, payload in (
        ("inference.pdmodel", b"model"),
        ("inference.pdiparams", b"params"),
        ("inference.pdiparams.info", b"info"),
    ):
        (source / name).write_bytes(payload)
    monkeypatch.setattr(compat, "_windows_short_path", lambda _path: None)
    destination = prepare_paddle_model_path(source)
    try:
        # The fake Windows host used by the release tests is configured to
        # exercise the same narrow-path failure as Paddle's C++ Config API.
        if compat.os.name == "nt":
            assert all(ord(character) < 128 for character in str(destination))
            assert destination != source
            assert (destination / "inference.pdiparams").read_bytes() == b"params"
        else:
            assert destination == source
    finally:
        compat.cleanup_paddle_model_paths()

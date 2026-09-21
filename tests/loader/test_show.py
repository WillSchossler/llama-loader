from pathlib import Path

import pytest

from llama_loader.loader import Loader

from .helpers import Helper


def test_show_displays_model_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["show", "qwen"])

    qwen = helper.create_model()

    loader = Loader(args)
    loader.show("qwen")

    parameters = helper.create_capsys_out_dict(capsys)

    assert parameters["--jinja"] == ""
    assert parameters["--port"] == "9931"
    assert parameters["--host"] == "127.0.0.1"
    assert parameters["--spec-type"] == "ngram-mod,draft-mtp"
    assert parameters["--fit"] == "on"
    assert parameters["--model"] == str(qwen["model_dir"] / "model.gguf")
    assert parameters["--mmproj"] == str(qwen["model_dir"] / "mmproj.gguf")
    assert parameters["--model-draft"] == str(qwen["model_dir"] / "mtp.gguf")
    assert parameters["--chat-template-file"] == str(qwen["model_dir"] / "template.jinja")


def test_show_applies_profile_override_without_mutating_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["show", "qwen", "balanced"])

    helper.create_model()

    loader = Loader(args)
    # A copy of the model's arguments for further imutability check
    arguments_before = loader.models["qwen"].arguments.copy()

    loader.show("qwen", "balanced")

    parameters = helper.create_capsys_out_dict(capsys)

    assert parameters["--port"] == "8080"
    assert parameters["--host"] == "127.0.0.1"

    assert loader.models["qwen"].arguments == arguments_before, (
        "Model's arguments are being mutated by the show() method"
    )


def test_show_raises_for_unknown_profile_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["show", "qwen", "xhigh"])

    helper.create_model()

    loader = Loader(args)
    with pytest.raises(ValueError, match="Unknown profile"):
        loader.show("qwen", "xhigh")


def test_show_displays_profile_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["show", "default"])

    loader = Loader(args)
    loader.show("default")

    parameters = helper.create_capsys_out_dict(capsys)

    assert parameters["--jinja"] == ""
    assert parameters["--host"] == "127.0.0.1"
    assert parameters["--port"] == "9931"


def test_show_raises_for_profile_with_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["show", "default", "potato"])

    loader = Loader(args)

    with pytest.raises(ValueError, match="A profile override can only be applied when showing a model"):
        loader.show("default", "potato")


def test_show_raises_for_unknown_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["show", "potato"])

    loader = Loader(args)

    with pytest.raises(ValueError, match="is not a valid model or profile"):
        loader.show("potato")

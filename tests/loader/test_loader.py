from pathlib import Path

import pytest

from llama_loader.loader import Loader

from .helpers import Helper


def test_loader_loads_valid_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)

    args = helper.create_cli_args()
    qwen = helper.create_model(name="qwen")

    loader = Loader(args)

    assert loader.args.command == args.command

    assert loader.configs.root == helper.models_dir
    assert loader.configs.editor == helper.editor
    assert loader.configs.browser_path == helper.browser_path

    assert loader.profiles["default"]["--jinja"] == ""
    assert loader.profiles["default"]["--host"] == "127.0.0.1"
    assert loader.profiles["default"]["--port"] == 9931

    assert loader.profiles["balanced"]["--fit"] == "on"
    assert loader.profiles["balanced"]["--agent"] == ""

    assert loader.models["qwen"].name == "qwen"
    assert loader.models["qwen"].profile == qwen["profile"]

    assert loader.models["qwen"].files["--model"] == qwen["model_dir"] / "model.gguf"
    assert loader.models["qwen"].files["--mmproj"] == qwen["model_dir"] / "mmproj.gguf"
    assert loader.models["qwen"].files["--model-draft"] == qwen["model_dir"] / "mtp.gguf"
    assert loader.models["qwen"].files["--chat-template-file"] == qwen["model_dir"] / "template.jinja"

    assert loader.models["qwen"].parameters["--spec-type"] == "ngram-mod,draft-mtp"
    assert loader.models["qwen"].parameters["--fit"] == "on"
    assert loader.models["qwen"].parameters["--jinja"] == ""


def test_loader_raises_for_duplicate_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args()

    toml = """
        name = "qwen"
        profile = "default"
        [files]
        [parameters]
        """
    helper.create_model(name="gemma", custom_toml=toml)

    helper.create_model(name="qwen")

    with pytest.raises(ValueError, match="The name 'qwen' already exists"):
        Loader(args)


def test_loader_raises_for_model_name_matching_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args()

    helper.create_model(name="balanced")

    with pytest.raises(ValueError, match="The name 'balanced' is already defined as a profile"):
        Loader(args)


def test_loader_ignores_incomplete_model_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args()

    helper.create_model(name="incomplete", custom_toml='name = "incomplete"')

    loader = Loader(args)

    assert loader.models == {}

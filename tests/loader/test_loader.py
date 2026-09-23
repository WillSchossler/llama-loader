from pathlib import Path
from unittest.mock import MagicMock

import os
import pytest

from llama_loader import loader as loader_module
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


def test_print_arguments_formats_values_and_bare_flags(capsys: pytest.CaptureFixture[str]):
    arguments: dict[str, object] = {"--agent": "", "--fit": "on", "--port": 8080}

    Loader._print_arguments(arguments)

    output = capsys.readouterr().out.strip().splitlines()

    assert output == [
        "--agent",
        "--fit: on",
        "--port: 8080",
    ]


@pytest.mark.parametrize(
    "incognito",
    (
        pytest.param(False, id="standard"),
        pytest.param(True, id="incognito"),
    ),
)
def test_open_browser(monkeypatch: pytest.MonkeyPatch, incognito: bool):
    browser_path = Path("browser.exe")
    host = "127.0.0.1"
    port = 9931

    popen_mock = MagicMock()
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    Loader._open_browser(browser_path, host, port, incognito)

    expected = [browser_path, "--start-maximized", *(["--incognito"] if incognito else []), f"http://{host}:{port}"]

    popen_mock.assert_called_once_with(expected)


def test_run_server_with_llama_server_set(monkeypatch: pytest.MonkeyPatch):
    command = ["llama-server", "--model", "qwen.gguf", "--agent"]

    process_mock = MagicMock()
    process_mock.wait.side_effect = None

    popen_mock = MagicMock(return_value=process_mock)
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    Loader._run_server(command)

    popen_mock.assert_called_once_with(command)
    process_mock.wait.assert_called_once()


def test_run_server_raises_when_llama_server_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    command = ["llama-server", "--model", "qwen.gguf", "--agent"]

    popen_mock = MagicMock()
    popen_mock.side_effect = FileNotFoundError
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    with pytest.raises(SystemExit, match="Or compile your own version from source"):
        Loader._run_server(command)
    
    popen_mock.assert_called_once_with(command)

    output = capsys.readouterr().out

    if os.name == "nt":
        assert "winget" in output
    else:
        assert "brew" in output

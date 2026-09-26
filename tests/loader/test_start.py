from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llama_loader import loader as loader_module
from llama_loader.loader import Loader

from .helpers import Helper


def test_start_raises_when_model_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", "potato"])

    loader = Loader(args)

    with pytest.raises(ValueError, match="Unknown model: potato"):
        loader.start(model_name=args.model, llama_args=args.llamaargs, open_browser=args.b, incognito=args.i)


def test_start_raises_when_profile_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", "qwen", "potato"])
    helper.create_model()

    loader = Loader(args)

    with pytest.raises(ValueError, match="is not a valid profile or llama.cpp flag"):
        loader.start(model_name=args.model, llama_args=args.llamaargs, open_browser=args.b, incognito=args.i)


def test_start_with_profile_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", "qwen", "balanced"])
    qwen = helper.create_model()

    loader = Loader(args)
    selected_model = loader.models[qwen["name"]]

    fake_arguments = {"--hot": "potato"}
    fake_command = ["potato"]

    build_arguments_mock = MagicMock(return_value=fake_arguments)
    build_command_mock = MagicMock(return_value=fake_command)
    run_server_mock = MagicMock()

    monkeypatch.setattr(selected_model, "build_arguments", build_arguments_mock)
    monkeypatch.setattr(selected_model, "build_command", build_command_mock)
    monkeypatch.setattr(loader, "_run_server", run_server_mock)

    loader.start(model_name=args.model, llama_args=args.llamaargs, open_browser=args.b, incognito=args.i)

    assert selected_model.arguments == fake_arguments
    build_arguments_mock.assert_called_once_with(loader.profiles["balanced"])
    run_server_mock.assert_called_once_with(fake_command)


def test_start_applies_cli_argument_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", "qwen", "--shared", "cli", "--cli-only", "potato"])
    qwen = helper.create_model()

    loader = Loader(args)
    selected_model = loader.models[qwen["name"]]

    selected_model.arguments = {
        "--preserved": "model",
        "--shared": "model",
    }

    fake_overrides = {
        "--shared": "cli",
        "--cli-only": "potato",
    }
    fake_command = ["llama-server", "--hot", "potato"]

    args_to_dict_mock = MagicMock(return_value=fake_overrides)
    build_command_mock = MagicMock(return_value=fake_command)
    run_server_mock = MagicMock()

    monkeypatch.setattr(loader_module.CLI, "args_to_dict", args_to_dict_mock)
    monkeypatch.setattr(selected_model, "build_command", build_command_mock)
    monkeypatch.setattr(loader, "_run_server", run_server_mock)

    loader.start(model_name=args.model, llama_args=args.llamaargs, open_browser=args.b, incognito=args.i)

    assert selected_model.arguments == {
        "--preserved": "model",
        "--shared": "cli",
        "--cli-only": "potato",
    }
    args_to_dict_mock.assert_called_once_with(args.llamaargs)
    run_server_mock.assert_called_once_with(fake_command)


@pytest.mark.parametrize(
    ("browser_flag", "incognito"),
    [
        pytest.param("-b", False, id="browser"),
        pytest.param("-i", True, id="incognito"),
    ],
)
def test_start_opens_browser_with_selected_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    browser_flag: str,
    incognito: bool,
):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", browser_flag, "qwen"])
    qwen = helper.create_model()

    loader = Loader(args)
    selected_model = loader.models[qwen["name"]]

    browser_path = Path("browser.exe")
    browser_address = ("127.0.0.1", 9931)

    require_browser_mock = MagicMock(return_value=browser_path)
    require_address_mock = MagicMock(return_value=browser_address)
    open_browser_mock = MagicMock()
    run_server_mock = MagicMock()

    monkeypatch.setattr(loader.configs, "require_browser", require_browser_mock)
    monkeypatch.setattr(selected_model, "require_address", require_address_mock)
    monkeypatch.setattr(loader, "_open_browser", open_browser_mock)
    monkeypatch.setattr(loader, "_run_server", run_server_mock)

    loader.start(model_name=args.model, llama_args=args.llamaargs, open_browser=args.b, incognito=args.i)

    open_browser_mock.assert_called_once_with(browser_path, *browser_address, incognito)


def test_start_applies_argument_layer_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", "qwen", "fake_profile", "--shared", "cli", "--cli-only", "cli"])
    qwen = helper.create_model()

    fake_profile = """

    [fake_profile]
    "--profile-only" = "profile"
    "--model-profile" = "profile"
    "--shared" = "profile"
    """

    with helper.profiles_path.open("a", encoding="utf-8") as file:
        file.write(fake_profile)

    loader = Loader(args)
    selected_model = loader.models[qwen["name"]]

    fake_parameters = {"--model-only": "model", "--model-profile": "model", "--shared": "model"}
    selected_model.parameters.update(fake_parameters)

    fake_command = ["llama-server"]
    build_command_mock = MagicMock(return_value=fake_command)
    run_server_mock = MagicMock()

    monkeypatch.setattr(selected_model, "build_command", build_command_mock)
    monkeypatch.setattr(loader, "_run_server", run_server_mock)

    loader.start(model_name=args.model, llama_args=args.llamaargs, open_browser=args.b, incognito=args.i)

    assert selected_model.arguments["--profile-only"] == "profile"
    assert selected_model.arguments["--model-profile"] == "model"
    assert selected_model.arguments["--shared"] == "cli"
    assert selected_model.arguments["--model-only"] == "model"
    assert selected_model.arguments["--cli-only"] == "cli"

    run_server_mock.assert_called_once_with(fake_command)

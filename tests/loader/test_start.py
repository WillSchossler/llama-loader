from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
    
    


from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llama_loader import loader as loader_module
from llama_loader.loader import Loader

from .helpers import Helper


@pytest.mark.parametrize("name", ("configs", "profiles"))
def test_edit_opens_setting_in_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["edit", name])

    loader = Loader(args)

    popen_mock = MagicMock()
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    loader.edit(name)
    popen_mock.assert_called_once_with([helper.editor, getattr(helper, f"{name}_path")])


def test_edit_opens_model_in_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["edit", "qwen"])
    qwen = helper.create_model()

    loader = Loader(args)

    popen_mock = MagicMock()
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    loader.edit("qwen")
    popen_mock.assert_called_once_with([helper.editor, qwen["model_file"]])


def test_edit_raises_for_unknown_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["edit", "potato"])

    loader = Loader(args)

    with pytest.raises(ValueError, match="is not a valid model or configuration"):
        loader.edit("potato")


def test_edit_raises_when_file_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["edit", "qwen"])
    qwen = helper.create_model()

    loader = Loader(args)
    qwen_file: Path = qwen["model_file"]
    qwen_file.unlink()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        loader.edit("qwen")

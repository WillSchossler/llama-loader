from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llama_loader.loader import Loader
from llama_loader import loader as loader_module

from .helpers import Helper


@pytest.mark.parametrize(
    ("command", "models_only", "profiles_only"),
    (
        (["list"], False, False),
        (["list", "-m"], True, False),
        (["list", "-p"], False, True),
    ),
)
def test_run_dispatches_list_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    models_only: bool,
    profiles_only: bool,
):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(command)

    loader = Loader(args)

    roll_mock = MagicMock()
    monkeypatch.setattr(loader, "roll", roll_mock)

    loader.run()

    roll_mock.assert_called_once_with(models_only=models_only, profiles_only=profiles_only)


def test_run_dispatches_edit_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["edit", "potato"])

    loader = Loader(args)

    edit_mock = MagicMock()
    monkeypatch.setattr(loader, "edit", edit_mock)

    loader.run()
    edit_mock.assert_called_once_with(name="potato")


def test_run_dispatches_init_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["init"])

    loader = Loader(args)

    init_mock = MagicMock()
    monkeypatch.setattr(loader, "init", init_mock)

    loader.run()
    init_mock.assert_called_once_with(cwd=Path.cwd())


@pytest.mark.parametrize(("name", "profile"), (("qwen", "balanced"), ("qwen", None), ("balanced", None)))
def test_run_dispatches_show_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, profile: str):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["show", name, *([profile] if profile is not None else [])])

    loader = Loader(args)

    show_mock = MagicMock()
    monkeypatch.setattr(loader, "show", show_mock)

    loader.run()
    show_mock.assert_called_once_with(name=name, profile=profile)


@pytest.mark.parametrize(
    ("open_browser", "incognito", "llama_args"),
    (
        (False, False, []),
        (True, False, ["--fit", "off"]),
        (False, True, ["--agent", ""]),
    ),
)
def test_run_dispatches_start_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_browser: bool,
    incognito: bool,
    llama_args: list[str],
):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(
        ["start", *(["-b"] if open_browser else ["-i"] if incognito else []), "qwen", *llama_args]
    )

    loader = Loader(args)

    start_mock = MagicMock()
    monkeypatch.setattr(loader, "start", start_mock)

    loader.run()
    start_mock.assert_called_once_with(
        model_name="qwen", llama_args=llama_args, open_browser=open_browser, incognito=incognito
    )


def test_run_server_returns_normally_when_server_exits_successfully(monkeypatch: pytest.MonkeyPatch):
    process_mock = MagicMock()
    process_mock.wait.return_value = 0

    popen_mock = MagicMock(return_value=process_mock)
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    Loader._run_server(["llama-server"])

    popen_mock.assert_called_once_with(["llama-server"])
    process_mock.wait.assert_called_once_with()


@pytest.mark.parametrize("return_code", [1, -1, 42])
def test_run_server_exits_with_server_error_code(
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
):
    process_mock = MagicMock()
    process_mock.wait.return_value = return_code

    monkeypatch.setattr(
        loader_module.subprocess,
        "Popen",
        MagicMock(return_value=process_mock),
    )

    with pytest.raises(SystemExit) as exc_info:
        Loader._run_server(["llama-server"])

    assert exc_info.value.code == return_code


def test_run_server_exits_with_130_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    process_mock = MagicMock()
    process_mock.wait.side_effect = [KeyboardInterrupt(), 0]

    monkeypatch.setattr(
        loader_module.subprocess,
        "Popen",
        MagicMock(return_value=process_mock),
    )

    with pytest.raises(SystemExit) as exc_info:
        Loader._run_server(["llama-server"])

    assert exc_info.value.code == 130
    process_mock.terminate.assert_called_once_with()
    assert process_mock.wait.call_count == 2

    captured = capsys.readouterr()
    assert "Closing the server..." in captured.err
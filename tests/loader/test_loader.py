import os
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, call

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

    loader = Loader(args)

    with pytest.raises(ValueError, match="The name 'qwen' already exists"):
        _ = loader.models


def test_loader_raises_for_model_name_matching_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args()

    helper.create_model(name="balanced")

    loader = Loader(args)

    with pytest.raises(ValueError, match="The name 'balanced' is already defined as a profile"):
        _ = loader.models


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
def test_open_browser_with_correct_mode(monkeypatch: pytest.MonkeyPatch, incognito: bool):
    browser_path = Path("browser.exe")
    host = "127.0.0.1"
    port = 9931

    popen_mock = MagicMock()
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    Loader._open_browser(browser_path, host, port, incognito)

    expected = [browser_path, "--start-maximized", *(["--incognito"] if incognito else []), f"http://{host}:{port}"]

    popen_mock.assert_called_once_with(expected)


@pytest.mark.parametrize(
    ("host", "expected_url"),
    [
        pytest.param("localhost", "http://localhost:8080", id="hostname"),
        pytest.param("127.0.0.1", "http://127.0.0.1:8080", id="ipv4"),
        pytest.param("::1", "http://[::1]:8080", id="ipv6"),
        pytest.param("2001:db8::1234", "http://[2001:db8::1234]:8080", id="ipv6-full"),
    ],
)
def test_open_browser_builds_valid_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    expected_url: str,
):
    browser_path = tmp_path / "browser.exe"
    browser_path.touch()

    popen_mock = MagicMock()
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    Loader._open_browser(
        browser_path=browser_path,
        host=host,
        port=8080,
        incognito=False,
    )

    popen_mock.assert_called_once_with(
        [
            browser_path,
            "--start-maximized",
            expected_url,
        ]
    )


def test_run_server_with_llama_server_set(monkeypatch: pytest.MonkeyPatch):
    command = ["llama-server", "--model", "qwen.gguf", "--agent"]

    process_mock = MagicMock()
    process_mock.wait.return_value = 0

    popen_mock = MagicMock(return_value=process_mock)
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    Loader._run_server(command)

    popen_mock.assert_called_once_with(command)
    process_mock.wait.assert_called_once()


@pytest.mark.parametrize(
    ("os_name", "install_command"), (("nt", "winget install llama.cpp"), ("posix", "brew install llama.cpp"))
)
def test_run_server_raises_when_llama_server_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], os_name: str, install_command: str
):
    command = ["llama-server", "--model", "qwen.gguf", "--agent"]

    popen_mock = MagicMock(side_effect=FileNotFoundError)
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    # Need to change "os.name" to test for both possible outcomes
    monkeypatch.setattr(os, "name", os_name)

    with pytest.raises(SystemExit, match="Or compile your own version from source"):
        Loader._run_server(command)

    popen_mock.assert_called_once_with(command)

    output = capsys.readouterr().out

    assert "Error: llama.cpp was not found" in output
    assert install_command in output


def test_run_server_terminates_process_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    command = ["llama-server", "--model", "qwen.gguf", "--agent"]

    process_mock = MagicMock()
    process_mock.wait.return_value = 130
    process_mock.wait.side_effect = [KeyboardInterrupt, None]

    popen_mock = MagicMock(return_value=process_mock)
    monkeypatch.setattr(loader_module.subprocess, "Popen", popen_mock)

    with pytest.raises(SystemExit) as exec_info:
        Loader._run_server(command)

    output = capsys.readouterr().err.strip()

    assert exec_info.value.code == 130
    assert output == "Closing the server..."
    assert process_mock.mock_calls == [
        call.wait(),
        call.terminate(),
        call.wait(),
    ]


def test_find_model_path_ignores_malformed_unrelated_toml(tmp_path: Path):
    qwen_dir = tmp_path / "qwen"
    broken_dir = tmp_path / "broken"

    qwen_dir.mkdir()
    broken_dir.mkdir()

    qwen_file = qwen_dir / "qwen.toml"
    qwen_file.write_text(
        'name = "qwen"\n',
        encoding="utf-8",
    )

    (broken_dir / "broken.toml").write_text(
        "[broken",
        encoding="utf-8",
    )

    assert Loader._find_model_path("qwen", tmp_path) == qwen_file


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Models\Qwen\model.gguf",
        'Will\'s "Qwen" model.gguf',
        "tab\tinside",
        "line\nbreak",
        "carriage\rreturn",
        "form\ffeed",
        "back\bspace",
        "Unicode 🦙 ç 日本語",
    ],
)
def test_toml_string_round_trips(value: str):
    serialized = Loader._toml_string(value)

    parsed = tomllib.loads(f"value = {serialized}")

    assert parsed["value"] == value


@pytest.mark.parametrize("name", ["configs", "profiles"])
def test_load_model_rejects_reserved_model_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    helper = Helper(tmp_path, monkeypatch)
    helper.create_model(name=name)

    loader = Loader(helper.create_cli_args(["show", name]))

    with pytest.raises(ValueError, match=f"Model name {name!r} is reserved"):
        loader._load_model(name)


@pytest.mark.parametrize("name", ["configs", "profiles"])
def test_load_models_rejects_reserved_model_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    helper = Helper(tmp_path, monkeypatch)
    helper.create_model(name=name)

    loader = Loader(helper.create_cli_args(["list", "-m"]))

    with pytest.raises(ValueError, match=f"Model name {name!r} is reserved"):
        _ = loader.models

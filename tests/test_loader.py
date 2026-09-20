from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llama_loader import loader as loader_module
from llama_loader.cli import CLI
from llama_loader.loader import Loader


class Helper:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both "settings_dir" and "models_dir" are created here because they're used in multiple methods
        self.settings_dir: Path = tmp_path / "settings"
        self.settings_dir.mkdir()

        self.models_dir: Path = tmp_path / "models"
        self.models_dir.mkdir()

        # We need Loader's "SETTINGS_DIR" to point to our own temporary "settings_dir" folder for the tests
        monkeypatch.setattr(Loader, "SETTINGS_DIR", self.settings_dir)

        self.create_configs()
        self.create_profiles()

    def create_cli_args(self, command: list[str] | None = None) -> Namespace:
        cli = CLI()
        command = ["list"] if command is None else command

        return cli.parser.parse_args(command)

    def create_configs(self) -> None:
        self.configs_path: Path = self.settings_dir / "configs.toml"

        self.browser_path: Path = self.settings_dir / "browser.exe"
        self.browser_path.touch()

        self.editor: str = "code.cmd"

        toml: str = f"""
        root = '{self.models_dir}'
        editor = "{self.editor}"
        browser_path = '{self.browser_path}'
        """

        self.configs_path.write_text(toml, encoding="utf-8")

    def create_profiles(self) -> None:
        self.profiles_path: Path = self.settings_dir / "profiles.toml"

        toml: str = """
        [default]
        --jinja = ""
        --host = "127.0.0.1"
        --port = 9931

        [balanced]
        --agent = ""
        --fit = "on"
        --port = "8080"
        """

        self.profiles_path.write_text(toml, encoding="utf-8")

    def create_model(self, name: str = "qwen", profile: str = "default", custom_toml: str | None = None) -> dict:
        model_dir: Path = self.models_dir / name
        model_dir.mkdir()

        model_file: Path = model_dir / f"{name}.toml"

        files: list[str] = ["model.gguf", "template.jinja", "mmproj.gguf", "mtp.gguf"]

        for file in files:
            temp_file: Path = model_dir / file
            temp_file.touch()

        toml: str = (
            custom_toml
            if custom_toml is not None
            else f"""
        # {name}


        # A name used to identify the model. Must be unique.
        name = "{name}"

        # Default profile for the model. Pick one table from "profiles.toml".
        # The flags from your chosen profile will overwrite the default ones.
        profile = "{profile}"


        # Relative path of your files.
        [files]
        --model = 'model.gguf'
        --mmproj = 'mmproj.gguf'
        --model-draft = 'mtp.gguf'
        --chat-template-file = 'template.jinja'


        # Additional llama.cpp parameters.
        [parameters]
        --spec-type = "ngram-mod,draft-mtp"
        --fit = "on"
        --jinja = ""
        """
        )

        model_file.write_text(toml, encoding="utf-8")

        return {"name": name, "profile": profile, "model_dir": model_dir, "model_file": model_file, "toml": toml}

    @staticmethod
    def create_capsys_out_dict(capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
        output = capsys.readouterr().out.strip().splitlines()
        parsed_lines = [item.split(": ", maxsplit=1) for item in output]

        return {item[0]: "" if len(item) == 1 else item[1] for item in parsed_lines}


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


def test_loader_list_without_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["list"])

    helper.create_model()

    loader = Loader(args)
    loader.roll(models_only=False, profiles_only=False)

    captured = capsys.readouterr().out
    models_section, profiles_section = captured.split("Profiles:", maxsplit=1)

    assert "Name: qwen" in models_section
    assert "balanced" in profiles_section
    assert "default" not in profiles_section


def test_loader_list_models_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["list", "-m"])

    helper.create_model()

    loader = Loader(args)
    loader.roll(models_only=True, profiles_only=False)

    captured = capsys.readouterr().out

    assert "Name: qwen" in captured
    assert "balanced" not in captured


def test_loader_list_profiles_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["list", "-p"])

    helper.create_model()

    loader = Loader(args)
    loader.roll(models_only=False, profiles_only=True)

    output = capsys.readouterr().out

    assert "balanced" in output
    assert "qwen" not in output
    assert "default" not in output, "Default profile should not be listed"


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
    model_name: str,
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
        model_name=model_name, llama_args=llama_args, open_browser=open_browser, incognito=incognito
    )

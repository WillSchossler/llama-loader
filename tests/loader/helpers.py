from argparse import Namespace
from pathlib import Path

import pytest

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

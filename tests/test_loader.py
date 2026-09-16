from argparse import Namespace
from pathlib import Path

from llama_loader.cli import CLI
from llama_loader.loader import Loader


class Helper:
    def __init__(self, tmp_path: Path, cli_args: list[str] = ["list"], custom_toml=False) -> None:
        self.tmp_path = tmp_path

        self.models_dir: Path = tmp_path / "models"
        self.models_dir.mkdir()

        self.settings_dir: Path = tmp_path / "settings"
        self.settings_dir.mkdir()

        cli = CLI()
        self.args: Namespace = cli.parser.parse_args(cli_args)

        self.__create_configs_file()
        self.__create_profiles_file()
        self.__create_model_files(custom_toml)

    def __create_configs_file(self) -> None:
        self.browser_path: Path = self.settings_dir / "browser.exe"
        self.browser_path.touch()

        self.configs_path: Path = self.settings_dir / "configs.toml"

        toml: str = f"""
        root = '{self.models_dir}'
        editor = "code.cmd"
        browser_path = '{self.browser_path}'
        """

        self.configs_path.write_text(toml)

    def __create_profiles_file(self) -> None:
        self.profiles_path: Path = self.settings_dir / "profiles.toml"

        toml: str = """
        [default]
        --jinja = ""
        --port = 9931
        --host = "127.0.0.1"

        [balanced]
        --fit = "on"
        --agent = ""
        """

        self.configs_path.write_text(toml)

    def __create_model_files(self, custom_toml=False) -> None:
        toml_path: Path = self.models_dir / "models.toml"

        files: list[str] = ["model.gguf", "template.jinja", "mmproj.gguf", "mtp.gguf"]

        for file in files:
            temp_file: Path = self.models_dir / file
            temp_file.touch()

        qwen_model: Path = self.models_dir / "qwen.gguf"
        qwen_model.touch()

        toml: str = (
            custom_toml
            if custom_toml
            else """
        # models


        # A name used to identify the model. Must be unique.
        name = "models"

        # Default profile for the model. Pick one table from "profiles.toml".
        # The flags from your chosen profile will overwrite the default ones.
        profile = "default"


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

        toml_path.write_text(toml)


def test_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(Loader, "SETTINGS_DIR", tmp_path)

    helper = Helper(tmp_path, ["list"])
    args = helper.args

    Loader(args)

    """assert loader.args.command == args.command

    assert loader.configs.root == tmp_path / "models"
    assert loader.configs.editor == "code.cmd"
    assert loader.configs.browser_path == tmp_path / "browser.exe"

    assert loader.profiles["default"]["--host"] == "127.0.0.1"
    assert loader.profiles["default"]["--port"] == 9931
    assert loader.profiles["balanced"]["--temp"] == 0.6

    assert loader.models["qwen"].name == "qwen" """

from argparse import Namespace
from pathlib import Path

import pytest

from llama_loader.cli import CLI
from llama_loader.loader import Loader


class Helper:
    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        # Both "settings_dir" and "models_dir" are defined here because they're used in multiple methods
        self.settings_dir: Path = tmp_path / "settings"
        self.models_dir: Path = tmp_path / "models"

        self.settings_dir.mkdir()
        self.models_dir.mkdir()

        # We need Loader's "SETTINGS_DIR" to point to our own temporary settings_dir folder for the tests  
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

        self.configs_path.write_text(toml)

    def create_profiles(self) -> None:
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

        self.profiles_path.write_text(toml)

    def create_model(self, name: str = "qwen", profile: str = "default", custom_toml: str | None = None) -> dict[str, object]:
        model_dir: Path = self.models_dir / name
        model_dir.mkdir()

        model_file: Path = model_dir / f"{name}.toml"
    
        files: list[str] = ["model.gguf", "template.jinja", "mmproj.gguf", "mtp.gguf"]

        for file in files:
            temp_file: Path = model_dir / file
            temp_file.touch()

        toml: str = (
            custom_toml
            if custom_toml
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

        model_file.write_text(toml)
        
        return {"name": name, "profile": profile, "model_dir": model_dir, "model_file": model_file, "toml": toml}



def test_loader(tmp_path, monkeypatch):
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

def test_loader_raises_for_duplicate_model(tmp_path, monkeypatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args()

    toml = """
        name = "qwen"
        profile = "default"
        [files]
        [parameters]
        """
    gemma = helper.create_model(name="gemma", custom_toml=toml)

    qwen = helper.create_model(name="qwen")

    with pytest.raises(ValueError, match="The name 'qwen' already exists"):
        loader = Loader(args)
  
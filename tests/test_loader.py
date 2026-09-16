from argparse import Namespace
from pathlib import Path

from llama_loader.cli import CLI
from llama_loader.loader import Loader


class Helper:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

        self.models_dir: Path = tmp_path / "models"
        self.models_dir.mkdir()

        self.settings_dir: Path = tmp_path / "settings"
        self.settings_dir.mkdir()

        self.create_configs()
        self.create_profiles()

    def create_cli_args(self, command: list[str] | None = None) -> Namespace:
        cli = CLI()
        command: list[str] = ["list"] if command is None else command

        return cli.parser.parse_args(command)

    def create_configs(self) -> None:
        self.editor = "code.cmd"

        self.browser_path: Path = self.settings_dir / "browser.exe"
        self.browser_path.touch()

        self.configs_path: Path = self.settings_dir / "configs.toml"

        toml = f"""
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


    def create_model(self, custom_toml=False):
        model_path = self.models_dir / "models.toml"
        
        files: list[str] = ["model.gguf", "template.jinja", "mmproj.gguf", "mtp.gguf"]

        for file in files:
            temp_file: Path = self.models_dir / file
            temp_file.touch()

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

        model_path.write_text(toml)


def test_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(Loader, "SETTINGS_DIR", tmp_path / "settings")

    helper = Helper(tmp_path)
    helper.create_model()

    args = helper.create_cli_args()
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

    assert loader.models["models"].name == "models"
    assert loader.models["models"].profile == "default"

    assert loader.models["models"].files["--model"] == helper.models_dir / "model.gguf"
    assert loader.models["models"].files["--mmproj"] == helper.models_dir / "mmproj.gguf"
    assert loader.models["models"].files["--model-draft"] == helper.models_dir / "mtp.gguf"
    assert loader.models["models"].files["--chat-template-file"] == helper.models_dir / "template.jinja"
    
    assert loader.models["models"].parameters["--spec-type"] == "ngram-mod,draft-mtp"
    assert loader.models["models"].parameters["--fit"] == "on"
    assert loader.models["models"].parameters["--jinja"] == ""
    
    """
        [files]
        --model = 'model.gguf'
        --mmproj = 'mmproj.gguf'
        --model-draft = 'mtp.gguf'
        --chat-template-file = 'template.jinja'


        # Additional llama.cpp parameters.
        [parameters]
        --spec-type = "ngram-mod,draft-mtp"
        --fit = "on"
        --jinja = "" """
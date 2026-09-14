from pathlib import Path

from llama_loader.cli import CLI
from llama_loader.configs import Configs
from llama_loader.loader import Loader
from llama_loader.profiles import Profiles


class create_temporary:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.models_dir = tmp_path / "models"
        self.settings_dir = tmp_path / "settings"

        self.models_dir.mkdir()
        self.settings_dir.mkdir()

    def cli_args(command: list[str] = ["list"]):
        cli = CLI()
        self.args = cli.parser.parse_args(command)

        return self.args

    def model(self):
        pass



def create_model(tmp_path):
    models_path = tmp_path / "models"

    toml_path = models_path / "qwen.toml"
    
    qwen_model = models_path / "qwen.gguf"
    qwen_model.touch()

    toml = """
    # qwen

    name = "qwen"
    profile = "balanced"

    [files]
    --model = 'qwen.gguf'

    [parameters]
    --temp = 1
    """

    toml_path.write_text(toml)


def create_profiles(tmp_path: Path) -> Profiles:
    profiles_path = tmp_path / "profiles.toml"

    toml = """
    [default]
    --agent = ""
    --port = 9931
    --host = "127.0.0.1"

    [balanced]
    --temp = 0.60
    """

    profiles_path.write_text(toml)

    return Profiles(profiles_path)


def create_configs(tmp_path: Path):
    configs_path = tmp_path / "configs.toml"

    root_path = tmp_path / "models"
    root_path.mkdir()

    browser_path = tmp_path / "browser.exe"
    browser_path.touch()

    toml = f"""
    root = '{root_path}'
    editor = "code.cmd"
    browser_path = '{browser_path}'
    """

    configs_path.write_text(toml)

    return Configs(configs_path)


def test_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(Loader, "SETTINGS_DIR", tmp_path)

    args = create_cli_args(["list"])
    create_profiles(tmp_path)
    create_configs(tmp_path)
    create_model(tmp_path)

    loader = Loader(args)

    assert loader.args.command == args.command    

    assert loader.configs.root == tmp_path / "models"
    assert loader.configs.editor == "code.cmd"
    assert loader.configs.browser_path == tmp_path / "browser.exe"

    assert loader.profiles["default"]["--host"] == "127.0.0.1"
    assert loader.profiles["default"]["--port"] == 9931
    assert loader.profiles["balanced"]["--temp"] == 0.6
    
    assert loader.models == {}
    #assert loader.models["gemma"]["name"] == "gemma"
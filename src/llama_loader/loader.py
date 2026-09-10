import os
import subprocess
import textwrap
import tomllib
from argparse import Namespace
from pathlib import Path

from llama_loader.cli import CLI
from llama_loader.configs import Configs
from llama_loader.model import Model
from llama_loader.profiles import Profiles

ROOT = Path(__file__).parents[2].resolve()


class Loader:
    """
    Coordinates llama-loader's application workflow.

    The Loader acts as the main orchestration layer of the application. It loads
    the global configuration and profiles, discovers and validates model
    configurations, and dispatches parsed CLI commands to their corresponding
    operations.

    It coordinates higher-level operations such as starting llama-server,
    listing models and profiles, generating draft model configurations, opening
    configuration files for editing, displaying resolved model arguments, and
    optionally launching the configured browser.

    Model-specific validation and argument construction are delegated to Model,
    while configuration and profile validation are handled by Configs and
    Profiles respectively.

    Args:
        args: Parsed command-line arguments used to determine the requested
            operation and its options.

    Attributes:
        args: Parsed command-line arguments.
        models: Discovered models mapped by their unique names.
        configs: Validated global application configuration.
        profiles: Validated collection of available profiles.

    Methods:
        start: Configure and start llama-server for a selected model.
        list: Display the available models and profiles.
        init: Generate a draft model configuration from files in a directory.
        edit: Open an application or model configuration in the configured editor.
        show: Display a profile or the resolved arguments for a model.
        run: Dispatch the command selected through the CLI.
        open_browser: Launch the configured browser for the llama-server interface.
    """

    REQUIRED_FIELDS = frozenset({"name", "files", "profile", "parameters"})

    def __init__(self, args: Namespace):
        self.args = args
        self.models: dict[str, Model] = {}
        self.configs = Configs(ROOT / "settings" / "configs.toml")
        self.profiles = Profiles(ROOT / "settings" / "profiles.toml")

        models_root = self.configs.root
        toml_paths = models_root.rglob("*.toml")

        for toml_path in toml_paths:
            with toml_path.open("rb") as file:
                model_toml = tomllib.load(file)

            # Will consider a valid model ONLY if the .toml have a name, file, profile and parameter set
            if self.REQUIRED_FIELDS <= model_toml.keys():
                model = Model(model_toml, toml_path, toml_path.parent, self.profiles)

                if model.name in self.models:
                    raise ValueError(f"Invalid model at '{toml_path}'. The name '{model.name}' already exists")

                if model.name in self.profiles:
                    raise ValueError(f"Invalid model at '{toml_path}'. The name '{model.name}' is already defined as a profile")

                self.models[model.name] = model

    def start(
        self,
        model: str,
        llamaargs: list | None = None,
        b: bool = False,
        i: bool = False,
    ) -> None:
        """
        Configure and start llama-server for a selected model.

        The model configuration is used as the base for the final llama.cpp
        arguments. An optional profile may be provided as the first argument after
        the model, followed by arbitrary llama.cpp flags. These values are applied
        as overrides before the final command is built.

        The ``-b`` and ``-i`` options belong to llama-loader, not llama.cpp, and must
        appear before the model name. All arguments after the model are intentionally
        captured as llama.cpp arguments, allowing them to be forwarded without
        requiring llama-loader to know or define every llama.cpp option. Consequently,
        ``-b`` and ``-i`` appearing after the model are treated as llama.cpp flags.
        This ordering is intentional and is part of the CLI grammar.

        When ``-b`` is enabled, the configured browser is opened before starting
        llama-server. When ``-i`` is enabled, the browser is opened in incognito mode.
        These options are mutually exclusive at the CLI level.

        The browser is intentionally opened before the server process so the web
        interface can be used to observe the model loading state. Configuration,
        browser, and address validation are performed before llama-server is started.

        The llama-server process remains attached until it exits or is interrupted.
        A keyboard interrupt terminates the server process before returning control
        to the user.

        Args:
            model: Name of the model to start.
            llamaargs: Optional profile and llama.cpp arguments supplied after the
                model name.
            b: Whether to open the configured browser.
            i: Whether to open the configured browser in incognito mode.

        Raises:
            ValueError: If the model is unknown, an invalid profile-like argument is
                provided, or required runtime configuration is invalid.
            FileNotFoundError: If required configured resources cannot be found.
            SystemExit: If the llama-server executable cannot be found.
        """

        if model not in self.models:
            raise ValueError(f"Unknown model: {model}.")

        selected_model = self.models[model]

        if llamaargs:
            # Copy is made to prevent changes in the mutable
            llamaargs: list[str] = llamaargs.copy()
            profile_arg = llamaargs[0]

            if profile_arg in self.profiles:
                llamaargs.pop(0)
                new_profile = self.profiles[profile_arg]
                selected_model.build_arguments(new_profile)

            elif not profile_arg.startswith("-"):
                raise ValueError(f"{profile_arg} is not a valid profile or llama.cpp flag.")

            flags_dict = CLI.args_to_dict(llamaargs)

            selected_model.arguments.update(flags_dict)

        if b or i:
            browser_path = self.configs.require_browser()
            browser_host, browser_port = selected_model.require_address()

            self.open_browser(browser_path, browser_host, browser_port, i)

        command = selected_model.build_command()
        try:
            llama_process = subprocess.Popen(command)
            llama_process.wait()
        # Check if the user interrupted the process (CTRL + C) to stop the server
        except KeyboardInterrupt:
            print("\nClosing the server...")
            llama_process.terminate()
            llama_process.wait()

        except FileNotFoundError:
            print("Error: llama.cpp was not found")

            if os.name == "nt":
                print("\nInstall via winget with: 'winget install llama.cpp'")
            else:
                print("\nInstall via homebrew with: 'brew install llama.cpp'")

            raise SystemExit("\nOr compile your own version from source: See more at https://github.com/ggml-org/llama.cpp")

    def list(self, models: bool, profiles: bool) -> None:
        """
        Print the available models and profiles.

        Args:
            models: Whether to print the models section.
            profiles: Whether to print the profiles section (the default profile
                is hidden). When both are unset, both sections are shown.
        """

        def print_models(values):
            print("\nModels:")
            for model in values:
                print(f"Name: {model.name:<10}||  Profile: {model.profile:>10}  ||   Path: {model.parent.resolve()!s:<70}")

        def print_profiles(profiles):
            print("\nProfiles:")
            for profile in profiles:
                if profile != "default":
                    print(profile)

        if models:
            print_models(self.models.values())
        elif profiles:
            print_profiles(self.profiles)
        else:
            print_models(self.models.values())
            print_profiles(self.profiles)

    def init(self, cwd: Path) -> None:
        """
        Generate a draft model configuration in the current directory.

        Scans ``cwd`` for ``.gguf`` and ``.jinja`` files, classifies them into
        model / mmproj / draft / template slots, and writes a starter ``.toml``
        (named after the folder) that the user can then adjust.

        Args:
            cwd: Directory containing the model files.

        Raises:
            SystemExit: If a configuration file for the folder already exists.
        """

        stem = cwd.name.replace(" ", "-")
        model_name = "-".join(stem.split("-")[:2]).lower()
        output_name = f"{stem}.toml"
        flags = {
            "model": "",
            "mmproj": "",
            "draft": "",
            "template": "",
            "spec-type": "",
        }

        # Search only for files with ".gguf" or ".jinja" extension
        files = [file.name for file in cwd.iterdir() if file.is_file() and file.suffix.lower() in (".gguf", ".jinja")]

        for file in files.copy():
            file_lower = file.lower()

            if ".jinja" in file_lower:
                flags["template"] = f'\n\t--chat-template-file = "{file}"'
                files.remove(file)

            elif "mmproj" in file_lower:
                flags["mmproj"] = f'\n\t--mmproj = "{file}"'
                files.remove(file)

            elif "mtp" in file_lower:
                flags["draft"] = f'\n\t--model-draft = "{file}"'
                flags["spec-type"] = '\n\t--spec-type = "ngram-mod,draft-mtp"'
                files.remove(file)

            elif "dflash" in file_lower:
                flags["draft"] = f'\n\t--model-draft = "{file}"'
                flags["spec-type"] = '\n\t--spec-type = "ngram-mod,draft-dflash"'
                files.remove(file)

        # If there is only one file left, we assume it's the model
        if len(files) == 1:
            flags["model"] = f'\n\t--model = "{files[0]}"'
        else:
            flags["model"] = '\n\t--model = "DEFINE_MODEL_PATH"'

        toml = textwrap.dedent(f"""
        # {cwd.name}


        # A name used to identify the model. Must be unique.
        name = "{model_name}"

        # Default profile for the model. Pick one table from "profiles.toml".
        # The flags from your chosen profile will overwrite the default ones.
        profile = "default"


        # Relative path of your files.
        [files]{flags["model"]}{flags["mmproj"]}{flags["draft"]}{flags["template"]}


        # Aditional llama.cpp parameters.
        [parameters]{flags["spec-type"]}
        --fit = "on"
        --jinja = "" """)

        output = cwd / output_name
        if output.exists():
            raise SystemExit(f"Error: '{output.name}' already exists")

        else:
            output.write_text(toml, encoding="utf-8")

    def edit(self, file: str) -> None:
        """
        Open a configuration file in the configured editor.

        ``file`` may be one of the global config files (``configs`` / ``profiles``)
        or the name of a discovered model.

        Args:
            file: A global config name or a model name.

        Raises:
            ValueError: If the name is neither a global config nor a model.
            FileNotFoundError: If the resolved file does not exist.
        """

        if file in ("configs", "profiles"):
            path = ROOT / Path(f"{file}.toml")
        elif file in self.models:
            path = self.models[file].path
        else:
            raise ValueError(f"{file} is not a valid model or file.")

        if path.is_file():
            subprocess.Popen([self.configs.editor, str(path)])
        else:
            raise FileNotFoundError(f"{file} doesn't exist.")

    def show(self, model: str, profile: str | None = None) -> None:
        """
        Print the resolved arguments for a model, or the flags of a profile.

        When ``model`` is a profile name, that profile's flags are shown. When it
        is a model, the model's final arguments are shown; if a ``profile`` is
        also given, the arguments are recomputed against that profile first.

        Args:
            model: A model name or a profile name.
            profile: Optional profile to apply to the model.

        Raises:
            ValueError: If the given model/profile name is unknown.
        """

        if model in self.profiles:
            for key, value in self.profiles[model].items():
                if value != "":
                    print(f"{key}: {value}")
                else:
                    print(key)

        elif model in self.models:
            selected_model = self.models[model]

            if profile:
                if profile not in self.profiles:
                    raise ValueError(f"Unknown profile: {profile}")

                else:
                    selected_model.build_arguments(self.profiles[profile])

            for key, value in selected_model.arguments.items():
                if value != "":
                    print(f"{key}: {value}")
                else:
                    print(key)

        else:
            raise ValueError(f"{model} is not a valid model or profile.")

    def run(self) -> None:
        """
        Dispatch the parsed command to its operation.

        Reads the selected subcommand and its options from ``self.args`` and
        calls the matching loader method.
        """
        match self.args.command:
            case "list":
                self.list(self.args.models, self.args.profiles)
            case "edit":
                self.edit(self.args.file)
            case "init":
                self.init(Path.cwd())
            case "show":
                self.show(self.args.model, self.args.profile)
            case "start":
                self.start(
                    self.args.model,
                    self.args.llamaargs,
                    self.args.b,
                    self.args.i,
                )

    def open_browser(
        self,
        browser_path: Path,
        host: str = "127.0.0.1",
        port: int = 9931,
        incognito: bool = False,
    ) -> None:
        """
        Launch the configured browser pointed at the llama-server URL.

        Args:
            browser_path: Path to the browser executable.
            host: Server host to open.
            port: Server port to open.
            incognito: Whether to open the browser in incognito mode.
        """
        command = [str(browser_path), "--start-maximized", f"http://{host}:{port}"]
        if incognito:
            command.append("--incognito")

        subprocess.Popen(command)


def main() -> None:
    cli = CLI()
    loader = Loader(cli.parse_args())
    loader.run()

if __name__ == "__main__":
    main()

import os
import subprocess
import textwrap
import tomllib
from argparse import Namespace
from pathlib import Path

from .cli import CLI
from .configs import Configs
from .model import Model
from .profiles import Profiles


class Loader:
    """
    Coordinates llama-loader's application workflow.

    The Loader is the application's orchestration layer. It loads the global
    configuration and profiles, discovers model configurations, dispatches
    parsed CLI commands, and coordinates operations involving models, the
    configured editor and browser, and the llama-server process.

    Model-specific validation and argument construction are delegated to Model,
    while Configs and Profiles validate their own configuration data.

    Args:
        args: Parsed command-line arguments used to determine the requested
            operation and its options.

    Attributes:
        args: Parsed command-line arguments.
        configs: Validated global application configuration.
        profiles: Validated collection of available profiles.
        models: Discovered models mapped by their unique names.

    Methods:
        start: Configure and start llama-server for a selected model.
        roll: Display the available models and profiles.
        init: Generate a draft model configuration from files in a directory.
        edit: Open an application or model configuration in the configured editor.
        show: Display a profile or the resolved arguments for a model.
        run: Dispatch the command selected through the CLI.
    """

    SETTINGS_DIR = Path(__file__).parents[2].resolve() / "settings"

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.configs = Configs(self.SETTINGS_DIR / "configs.toml")
        self.profiles = Profiles(self.SETTINGS_DIR / "profiles.toml")
        self.models: dict[str, Model] = self.__load_models()

    def __load_models(self) -> dict[str, Model]:
        """
        Discover and load model configurations from the configured models root.

        TOML files that do not contain all fields required by Model are ignored.
        Model-like files are validated by Model before being added to the collection.

        Returns:
            Discovered models mapped by their unique names.

        Raises:
            ValueError: If a model is invalid, duplicates another model name, or
                uses a name already defined as a profile.
            TypeError: If a model-like configuration contains an invalid field type.
        """
        models: dict[str, Model] = {}

        for toml_path in self.configs.root.rglob("*.toml"):
            with toml_path.open("rb") as file:
                model_toml = tomllib.load(file)

            if not Model.REQUIRED_MODEL_FIELDS <= model_toml.keys():
                continue

            model = Model(model_toml, toml_path, toml_path.parent, self.profiles)

            if model.name in models:
                raise ValueError(f"Invalid model at {str(toml_path)!r}. The name {model.name!r} already exists")

            if model.name in self.profiles:
                raise ValueError(
                    f"Invalid model at {str(toml_path)!r}. The name {model.name!r} is already defined as a profile"
                )

            models[model.name] = model

        return models

    def start(
        self,
        model_name: str,
        llama_args: list[str] | None = None,
        open_browser: bool = False,
        incognito: bool = False,
    ) -> None:
        """
        Configure and start llama-server for a selected model.

        The model's resolved arguments are used as the base configuration. An
        optional profile may be supplied as the first argument after the model,
        followed by arbitrary llama.cpp flags. Profile and command-line overrides
        update the selected model's arguments before the final command is built.

        ``start`` intentionally mutates the selected model's arguments because
        starting llama-server is the terminal operation of the loader workflow.
        Inspection operations such as ``show`` resolve arguments into temporary
        mappings instead.

        The ``-b`` and ``-i`` options belong to llama-loader and must appear before
        the model name. Arguments after the model are intentionally captured as
        llama.cpp arguments, so the same flags appearing after the model are passed
        through to llama.cpp.

        When requested, the browser is opened before llama-server so its interface
        can show the model loading state.

        Args:
            model_name: Name of the model to start.
            llama_args: Optional profile and llama.cpp arguments supplied after the
                model name.
            open_browser: Whether to open the configured browser.
            incognito: Whether to open the browser in incognito mode.

        Raises:
            ValueError: If the model is unknown, an invalid profile-like argument is
                provided, or required runtime configuration is invalid.
            SystemExit: If the llama-server executable cannot be found.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        selected_model = self.models[model_name]

        if llama_args:
            # Work on a copy so parsing does not mutate the caller's argument list
            llama_args = llama_args.copy()
            first_arg = llama_args[0]

            if first_arg in self.profiles:
                llama_args.pop(0)
                selected_profile = self.profiles[first_arg]
                selected_model.arguments = selected_model.build_arguments(selected_profile)

            elif not first_arg.startswith("-"):
                raise ValueError(f"{first_arg!r} is not a valid profile or llama.cpp flag")

            overrides = CLI.args_to_dict(llama_args)
            selected_model.arguments.update(overrides)

        if open_browser or incognito:
            browser_path = self.configs.require_browser()
            browser_host, browser_port = selected_model.require_address()
            self._open_browser(browser_path, browser_host, browser_port, incognito)

        command = selected_model.build_command()
        self._run_server(command)

    def roll(self, models_only: bool, profiles_only: bool) -> None:
        """
        Print the available models and profiles.

        The default profile is omitted from profile listings. When neither filter
        is enabled, both models and profiles are printed.

        Args:
            models_only: Whether to print only the models section.
            profiles_only: Whether to print only the profiles section.
        """

        def print_models(values) -> None:
            print("\nModels:")
            for model in values:
                print(f"Name: {model.name:<15} || Profile: {model.profile:>10} || Path: {model.parent.resolve()!s:<70}")

        def print_profiles(names) -> None:
            print("\nProfiles:")
            for name in names:
                if name != "default":
                    print(name)

        if models_only:
            print_models(self.models.values())
        elif profiles_only:
            print_profiles(self.profiles)
        else:
            print_models(self.models.values())
            print_profiles(self.profiles)

    def init(self, cwd: Path) -> None:
        """
        Generate a draft model configuration in a directory.

        Scans ``cwd`` for ``.gguf`` and ``.jinja`` files, classifies recognizable
        auxiliary files, and writes a starter TOML configuration named after the
        directory. If exactly one unclassified model file remains, it is selected
        automatically; otherwise a placeholder is written for the user to replace.

        Args:
            cwd: Directory containing the model files.

        Raises:
            SystemExit: If the output configuration already exists.
        """
        stem = cwd.name.replace(" ", "-")
        model_name = "-".join(stem.split("-")[:2]).lower()
        output = cwd / f"{stem}.toml"

        if output.exists():
            raise SystemExit(f"Error: {output.name!r} already exists")

        flags: dict[str, str] = {
            "model": "",
            "mmproj": "",
            "draft": "",
            "template": "",
            "spec-type": "",
        }

        files = [file.name for file in cwd.iterdir() if file.is_file() and file.suffix.lower() in (".gguf", ".jinja")]

        for file in files.copy():
            file_lower = file.lower()

            if file_lower.endswith(".jinja"):
                flags["template"] = f"\n--chat-template-file = '{file}'"
                files.remove(file)

            elif "mmproj" in file_lower:
                flags["mmproj"] = f"\n--mmproj = '{file}'"
                files.remove(file)

            elif "mtp" in file_lower:
                flags["draft"] = f"\n--model-draft = '{file}'"
                flags["spec-type"] = '\n--spec-type = "ngram-mod,draft-mtp"'
                files.remove(file)

            elif "dflash" in file_lower:
                flags["draft"] = f"\n--model-draft = '{file}'"
                flags["spec-type"] = '\n--spec-type = "ngram-mod,draft-dflash"'
                files.remove(file)

        # If exactly one unclassified file remains, assume it is the main model. Otherwise use a placeholder
        flags["model"] = f"\n--model = '{files[0]}'" if len(files) == 1 else "\n--model = 'DEFINE_MODEL_PATH'"

        toml = textwrap.dedent(
            """
            # {folder_name}


            # A name used to identify the model. Must be unique.
            name = "{model_name}"

            # Default profile for the model. Pick one table from "profiles.toml".
            # The flags from your chosen profile will overwrite the default ones.
            profile = "default"


            # Relative path of your files.
            [files]{model}{mmproj}{draft}{template}


            # Additional llama.cpp parameters.
            [parameters]{spec_type}
            --fit = "on"
            --jinja = ""
            """
        ).format(
            folder_name=cwd.name,
            model_name=model_name,
            model=flags["model"],
            mmproj=flags["mmproj"],
            draft=flags["draft"],
            template=flags["template"],
            spec_type=flags["spec-type"],
        )

        output.write_text(toml, encoding="utf-8")

    def edit(self, name: str) -> None:
        """
        Open a configuration file in the configured editor.

        ``name`` may identify one of the global configuration files
        (``configs`` or ``profiles``) or a discovered model.

        Args:
            name: Global configuration name or model name.

        Raises:
            ValueError: If ``name`` identifies neither a global configuration nor a model.
            FileNotFoundError: If the resolved configuration file does not exist.
        """
        if name in ("configs", "profiles"):
            path = self.SETTINGS_DIR / f"{name}.toml"
        elif name in self.models:
            path = self.models[name].path
        else:
            raise ValueError(f"{name!r} is not a valid model or configuration")

        if not path.is_file():
            raise FileNotFoundError(f"{name!r} does not exist")

        subprocess.Popen([self.configs.editor, path])

    def show(self, name: str, profile: str | None = None) -> None:
        """
        Print a profile or the resolved arguments for a model.

        When ``name`` identifies a profile, that profile's flags are printed.
        When it identifies a model, arguments are resolved using either the model's
        configured profile or the optional profile override.

        Model arguments are resolved into a temporary mapping, so this operation
        does not modify the model's stored arguments.

        Args:
            name: Model name or profile name.
            profile: Optional profile to apply temporarily when displaying a model.

        Raises:
            ValueError: If the model or profile is unknown, or if a profile override
                is supplied while displaying a profile.
        """
        if name in self.profiles:
            if profile is not None:
                raise ValueError("A profile override can only be applied when showing a model")

            self._print_arguments(self.profiles[name])
            return

        if name not in self.models:
            raise ValueError(f"{name!r} is not a valid model or profile")

        selected_model = self.models[name]
        selected_profile = self.profiles[selected_model.profile]

        if profile is not None:
            if profile not in self.profiles:
                raise ValueError(f"Unknown profile: {profile}")

            selected_profile = self.profiles[profile]

        arguments = selected_model.build_arguments(selected_profile)
        self._print_arguments(arguments)

    def run(self) -> None:
        """
        Dispatch the parsed CLI command to its corresponding loader operation.
        """
        match self.args.command:
            case "list":
                self.roll(models_only=self.args.models, profiles_only=self.args.profiles)
            case "edit":
                self.edit(name=self.args.file)
            case "init":
                self.init(cwd=Path.cwd())
            case "show":
                self.show(name=self.args.model, profile=self.args.profile)
            case "start":
                self.start(
                    model_name=self.args.model,
                    llama_args=self.args.llamaargs,
                    open_browser=self.args.b,
                    incognito=self.args.i,
                )

    @staticmethod
    def _run_server(command: list[str]) -> None:
        """
        Start llama-server and keep the process attached until it exits.

        A keyboard interrupt terminates the running process before returning
        control to the caller.

        Args:
            command: Command-line tokens used to start llama-server.

        Raises:
            SystemExit: If the llama-server executable cannot be found.
        """
        try:
            llama_process = subprocess.Popen(command)
        except FileNotFoundError:
            print("Error: llama.cpp was not found")

            if os.name == "nt":
                print("\nInstall via winget with: 'winget install llama.cpp'")
            else:
                print("\nInstall via homebrew with: 'brew install llama.cpp'")

            raise SystemExit(
                "\nOr compile your own version from source: See more at https://github.com/ggml-org/llama.cpp"
            ) from None

        try:
            llama_process.wait()
        except KeyboardInterrupt:
            print("\nClosing the server...")
            llama_process.terminate()
            llama_process.wait()

    @staticmethod
    def _open_browser(
        browser_path: Path,
        host: str,
        port: int,
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
        command = [browser_path, "--start-maximized"]

        if incognito:
            command.append("--incognito")

        command.append(f"http://{host}:{port}")
        subprocess.Popen(command)

    @staticmethod
    def _print_arguments(arguments: dict[str, object]) -> None:
        """
        Print a llama.cpp argument mapping in a human-readable form.

        Arguments with an empty string value are printed as bare flags.

        Args:
            arguments: Argument mapping to print.
        """
        for key, value in arguments.items():
            if value != "":
                print(f"{key}: {value}")
            else:
                print(key)


def main() -> None:
    cli = CLI()
    loader = Loader(cli.parse_args())
    loader.run()


if __name__ == "__main__":
    main()

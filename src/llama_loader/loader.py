import argparse
import os
import subprocess
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


class CLI:
    """
    Defines and manages the command-line interface for llama-loader.

    The CLI is responsible for declaring the available commands, arguments,
    and options accepted by the application, as well as converting raw
    command-line input into structured data consumed by the application.

    The class does not perform the operations associated with the commands.
    It only defines the command-line interface and parses user input.

    Attributes:
        parser: Root ArgumentParser responsible for parsing the command-line arguments.

    Methods:
        parse_args: Parses the command-line arguments provided by the user.
        args_to_dict: Converts a sequence of llama.cpp command-line arguments
            into a dictionary of flags and values.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Load and manage llama.cpp models via llama-server.",
            prog="llama-loader",
        )
        subparsers = self.parser.add_subparsers(dest="command", required=True, help="Available commands")

        subparsers.add_parser("init", help="Generate a draft model configuration in the current directory")

        edit_parser = subparsers.add_parser("edit", help="Open a configuration file in the editor")
        edit_parser.add_argument("file", help="Model name, or 'configs' / 'profiles'")

        show_parser = subparsers.add_parser("show", help="Show a profile or a model's resolved arguments")
        show_parser.add_argument("model", help="Model or profile name")
        show_parser.add_argument("profile", nargs="?", help="Optional profile to apply to the model")

        list_parser = subparsers.add_parser("list", help="List the available models and profiles")
        list_group = list_parser.add_mutually_exclusive_group()
        list_group.add_argument("--models", "-m", action="store_true", help="List only the models")
        list_group.add_argument("--profiles", "-p", action="store_true", help="List only the profiles")

        start_parser = subparsers.add_parser("start", help="Start llama-server for a model")
        start_group = start_parser.add_mutually_exclusive_group()
        start_group.add_argument("-b", action="store_true", help="Open the browser at the server URL")
        start_group.add_argument("-i", action="store_true", help="Open the browser in incognito mode")
        start_parser.add_argument("model", help="Name of the model to start")
        start_parser.add_argument("llamaargs", nargs=argparse.REMAINDER, help="Optional profile and llama.cpp flags")

    def parse_args(self) -> argparse.Namespace:
        """
        Parses the command-line arguments provided by the user.

        Returns:
            An argparse.Namespace with the selected command, its options and
            positional values.
        """
        return self.parser.parse_args()

    @staticmethod
    def args_to_dict(args: list[str]) -> dict[str, str]:
        """
        Converts llama.cpp command-line arguments into a dictionary.

        Each flag is used as a dictionary key and its associated value is stored
        as the corresponding dictionary value. Flags without an associated value
        are represented by an empty string.

        Args:
            args: Sequence of command-line arguments to convert.

        Returns:
            A dictionary containing llama.cpp flags and their associated values.

        Raises:
            ValueError: If a value expected to represent a flag does not start with "-".
        """

        def is_flag(value: str) -> bool:
            if not value.startswith("-"):
                return False

            # Negative numeric values start with '-' but are still valid values
            try:
                float(value)
                return False
            except ValueError:
                return True

        result: dict[str, str] = {}
        i = 0

        while i < len(args):
            key = args[i]

            if not key.startswith("-"):
                raise ValueError(f"{key} is not a valid llama.cpp flag.")

            if i + 1 >= len(args):
                result[key] = ""
                break

            next_value = args[i + 1]

            if is_flag(next_value):
                result[key] = ""
                i += 1
            else:
                result[key] = next_value
                i += 2

        return result


class Configs:
    """
    Loads and validates llama-loader's user configuration.

    The configuration is read from a TOML file and exposed as validated
    attributes for use by the rest of the application.

    Args:
        configs_path: Path to the TOML configuration file.

    Attributes:
        root: Directory containing the user's LLMs.
        editor: Command used to open the configured editor.
        browser_path: Path to the browser executable, or None if not configured.

    Methods:
        require_browser: Return the configured browser path
            or raise an error if no browser is configured.
    """

    def __init__(self, configs_path: Path):
        with configs_path.open("rb") as file:
            data = tomllib.load(file)

        self.root: Path = self.__validate(field="root", validation_type="dir", data=data)
        self.editor: str = self.__validate(field="editor", validation_type="str", data=data)
        self.browser_path: Path | None = self.__validate(field="browser_path", validation_type="file", data=data, required=False)

    def require_browser(self) -> Path:
        """
        Return the path to the configured browser executable.

        Returns:
            Path to the browser executable.

        Raises:
            ValueError: If no browser path is configured.
        """

        if self.browser_path is None:
            raise ValueError("Browser is not configured")

        return self.browser_path

    def __validate(self, field: str, validation_type: str, data: dict, required: bool = True):
        """
        Validate a single field from the global configuration.

        Checks that the field is present (when required), that its value is a
        non-empty string, and that it satisfies the given validation type (a
        directory, a file, or a plain string).

        Args:
            field: Name of the field to validate.
            validation_type: One of ``"dir"``, ``"file"`` or ``"str"``.
            data: The parsed configuration dictionary.
            required: Whether the field must be present.

        Returns:
            The validated value (a ``Path`` for ``"dir"``/``"file"``), or ``None``
            when the field is optional and absent.

        Raises:
            ValueError: If the field is missing, empty, its value does not
                satisfy the validation type, or the validation type is unknown.
            TypeError: If the field is present but is not a string.
        """

        if field not in data:
            if required:
                raise ValueError(f"Field '{field}' not defined in configs.toml")
            # Optional field: absence is represented by None
            return None

        value = data[field]

        if not isinstance(value, str):
            raise TypeError(f"Field '{field}' must be a string. Got '{value}' ({type(value)})")

        if not value.strip():
            raise ValueError(f"Field '{field}' cannot be empty")

        match validation_type:
            case "dir":
                path = Path(value)
                if not path.is_dir():
                    raise ValueError(f"Value '{value}' is not a valid '{field}' directory")

                return path

            case "file":
                path = Path(value)
                if not path.is_file():
                    raise ValueError(f"Field '{field}' does not contain a valid file ({value})")

                return path

            case "str":
                return value

            case _:
                raise ValueError(f"Unknown validation type '{validation_type}'")


class Profiles:
    """
    Loads and validates the profiles defined in a TOML file.

    The class provides limited mapping-like access to validated profiles,
    allowing profile lookup and iteration without exposing the full mutable
    dictionary interface.

    The default profile must define the settings required internally by
    llama-loader.

    Args:
        path: Path to the profiles TOML file.

    Methods:
        __getitem__: Return a profile by name.
        __iter__: Iterate over profile names.
    """

    def __init__(self, path: Path):
        with path.open("rb") as file:
            data = tomllib.load(file)

        self.__validate(data)
        self.__data: dict = data

    def __getitem__(self, item: str) -> dict:
        return self.__data[item]

    def __iter__(self):
        return iter(self.__data)

    def __validate(self, data: dict) -> None:
        """
        Validate the parsed profiles data.

        Ensures the data is a dictionary that defines a ``default`` profile, that
        every profile is a table of flags (each flag starting with ``"-"``), and
        that the default profile defines ``--host`` (non-empty string) and
        ``--port`` (integer).

        Args:
            data: The parsed profiles dictionary.

        Raises:
            TypeError: If the data or a profile is not a dictionary, or if
                ``--host``/``--port`` have the wrong type.
            ValueError: If the default profile is missing, a flag name does not
                start with ``"-"``, or a required default flag is missing/empty.
        """

        if not isinstance(data, dict):
            raise TypeError("Profiles data must be a dictionary.")

        if "default" not in data:
            raise ValueError("Missing required field 'default'")

        for profile, flags in data.items():
            if not isinstance(flags, dict):
                raise TypeError(f"Profile '{profile}' must be a table")

            for flag in flags:
                if not flag.startswith("-"):
                    raise ValueError(f"'{flag}' from profile '{profile}' should start with '-'.")

        default = data["default"]
        missing = {"--host", "--port"} - default.keys()

        if missing:
            missing_flags = ", ".join(sorted(missing))
            raise ValueError(f"Default profile is missing required flags: {missing_flags}")

        host = default["--host"]
        port = default["--port"]

        if not isinstance(host, str):
            raise TypeError(f"Default '--host' must be a string. Got '{type(host).__name__}'.")

        if not host.strip():
            raise ValueError("Default '--host' cannot be empty.")

        if type(port) is not int:
            raise TypeError(f"Default '--port' must be an integer. Got '{type(port).__name__}'.")


class Model:
    """
    Represents a validated llama.cpp model configuration.

    The class validates model metadata and file references, resolves model
    file paths, combines configuration layers into the final llama.cpp
    arguments, and builds the command used to start llama-server.

    Args:
        model: Model configuration loaded from TOML.
        path: Path to the model's TOML configuration file.
        parent: Directory used to resolve relative model file paths.
        profiles: Available validated profiles.

    Attributes:
        path: Path to the model configuration file.
        parent: Base directory for model files.
        profiles: Available profiles.
        name: Unique model name.
        profile: Name of the selected profile.
        parameters: Model-specific llama.cpp parameters.
        files: Model file flags mapped to their resolved paths.
        arguments: Final llama.cpp arguments after configuration merging.

    Methods:
        require_address: Return the final host and port.
        build_arguments: Build the final argument mapping for a profile.
        build_command: Build the command used to start llama-server.
    """

    REQUIRED_FIELDS = frozenset({"name", "profile", "parameters", "files"})

    def __init__(self, model: dict, path: Path, parent: Path, profiles: Profiles):
        self.path = path
        self.parent = parent
        self.profiles = profiles

        self.__validate(model, parent, profiles)
        self.name: str = model["name"]
        self.profile: str = model["profile"]
        self.parameters: dict[str, object] = model["parameters"]
        self.files: dict[str, Path] = {flag: parent / file_path for flag, file_path in model["files"].items()}

        self.arguments: dict[str, object] = {}
        self.build_arguments(self.profiles[self.profile])

    def require_address(self) -> tuple[str, int]:
        """
        Return the effective server address for this model.

        Resolves ``--host`` and ``--port`` from the final arguments, coercing the
        port to an integer and validating its range.

        Returns:
            A ``(host, port)`` tuple.

        Raises:
            ValueError: If ``--host`` is empty or ``--port`` is not a valid port.
            TypeError: If ``--host`` is not a string or ``--port`` is not a number.
        """

        host = self.arguments["--host"]
        if not isinstance(host, str):
            raise TypeError(f"Invalid value for field --host. Expected 'str', got '{type(host).__name__}'")

        if not host.strip():
            raise ValueError("Flag '--host' cannot be empty")

        port = self.arguments["--port"]
        if isinstance(port, str):
            try:
                port = int(port)
            except ValueError:
                raise ValueError(f"Invalid port: '{port}'")

        if type(port) is not int:
            raise TypeError(f"Invalid type for field --port. Expected 'int', got '{type(port).__name__}'")

        if not 1 <= port <= 65535:
            raise ValueError(f"Port number '{port}' is not a valid value. Must be between 1 and 65535")

        return host, port

    def build_arguments(self, profile: dict) -> None:
        """
        Recompute the final llama.cpp arguments from the configuration layers.

        ``self.arguments`` is cleared and rebuilt by merging, in priority order
        (lowest to highest): the ``default`` profile, ``profile``, the model's
        own ``parameters`` and the model's ``files``.

        Args:
            profile: The profile table applied to this model.
        """
        self.arguments.clear()

        for arguments in [
            self.profiles["default"],
            profile,
            self.parameters,
            self.files,
        ]:
            self.arguments.update(arguments)

    def build_command(self) -> list[str]:
        """
        Build the command line used to start ``llama-server``.

        Flags with an empty value are emitted as bare flags; flags with a value
        are followed by that value as a separate argument.

        Returns:
            The list of command-line tokens, starting with ``llama-server``.
        """
        command = ["llama-server"]
        for parameter, value in self.arguments.items():
            command.append(parameter)
            if value != "":
                command.append(str(value))

        return command

    def __validate(self, model: dict, parent: Path, profiles: Profiles) -> None:
        """
        Validate a model configuration.

        Checks that all required fields are present with the correct types, that
        the ``profile`` references an existing profile, and that every ``files``
        entry is a path to an existing file relative to ``parent``.

        Args:
            model: The parsed model configuration.
            parent: Directory used to resolve relative file paths.
            profiles: The available validated profiles.

        Raises:
            ValueError: If a required field is missing or empty, the profile is
                unknown, a flag name is invalid, or a file does not exist.
            TypeError: If a required field has the wrong type.
        """

        missing = self.REQUIRED_FIELDS - model.keys()
        if missing:
            raise ValueError(f"Missing required model fields: {', '.join(sorted(missing))}")

        name = model["name"]
        if not isinstance(name, str):
            raise TypeError(f"Invalid type for field 'name'. Expected 'str', got '{type(name).__name__}'")

        if not name.strip():
            raise ValueError("Field 'name' cannot be empty")

        profile = model["profile"]
        if not isinstance(profile, str):
            raise TypeError(f"Invalid type for field 'profile'. Expected 'str', got '{type(profile).__name__}'")

        if not profile.strip():
            raise ValueError("Field 'profile' cannot be empty")

        if profile not in profiles:
            raise ValueError(f"Profile '{profile}' defined by model '{name}' does not exist")

        parameters = model["parameters"]
        if not isinstance(parameters, dict):
            raise TypeError(f"Invalid type for field 'parameters'. Expected 'dict', got '{type(parameters).__name__}'")

        for parameter in parameters:
            if not isinstance(parameter, str):
                raise TypeError(f"Invalid parameter type. Expected 'str'. Got '{type(parameter).__name__}'")

            if not parameter.startswith("-"):
                raise ValueError(f"Parameter '{parameter}' is not a valid llama.cpp flag")

        files = model["files"]
        if not isinstance(files, dict):
            raise TypeError(f"Invalid type for field 'files'. Expected 'dict', got '{type(files).__name__}'")

        for flag, value in files.items():
            if not isinstance(flag, str):
                raise TypeError(f"Invalid file flag type. Expected 'str', got '{type(flag).__name__}'")

            if not flag.startswith("-"):
                raise ValueError(f"File flag '{flag}' is not a valid llama.cpp flag")

            if not isinstance(value, str):
                raise TypeError(f"Invalid file path for flag '{flag}'. Expected 'str', got '{type(value).__name__}'")

            if not value.strip():
                raise ValueError(f"File path for flag '{flag}' cannot be empty")

            if not (parent / value).is_file():
                raise ValueError(f"Flag '{flag}' does not contain a valid file path: {value}")


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

    def __init__(self, args: argparse.Namespace):
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

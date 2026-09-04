import argparse
import os
import subprocess
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


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
        subparser: Subparser collection containing the application's available commands.
        init_parser: Parser for the ``init`` command.
        edit_parser: Parser for the ``edit`` command.
        show_parser: Parser for the ``show`` command.
        list_parser: Parser for the ``list`` command.
        start_parser: Parser for the ``start`` command.

    Methods:
        parse_args: Parses the command-line arguments provided by the user.
        args_to_dict: Converts a sequence of llama.cpp
        command-line arguments
            into a dictionary of flags and values.

    Raises:
        ValueError: If ``args_to_dict`` receives an argument that does not
            represent a valid llama.cpp flag.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="TODO", prog="llama-loader")
        self.subparser = self.parser.add_subparsers(dest="command", required=True, help="TODO")

        self.init_parser = self.subparser.add_parser("init", help="TODO")

        self.edit_parser = self.subparser.add_parser("edit")
        self.edit_parser.add_argument("file", help="TODO")

        self.show_parser = self.subparser.add_parser("show")
        self.show_parser.add_argument("model", help="TODO")
        self.show_parser.add_argument("profile", nargs="?", help="TODO")

        self.list_parser = self.subparser.add_parser("list")
        list_group = self.list_parser.add_mutually_exclusive_group()
        list_group.add_argument("--models", "-m", action="store_true", help="TODO")
        list_group.add_argument("--profiles", "-p", action="store_true", help="TODO")

        self.start_parser = self.subparser.add_parser("start")
        start_group = self.start_parser.add_mutually_exclusive_group()
        start_group.add_argument("-b", action="store_true", help="TODO")
        start_group.add_argument("-i", action="store_true", help="TODO")
        self.start_parser.add_argument("model", help="TODO")
        self.start_parser.add_argument("llamaargs", nargs=argparse.REMAINDER, help="TODO")

    def parse_args(self) -> argparse.Namespace:
        """
        Parses the command-line arguments provided by the user.

        Returns:
            argparse.Namespace: Namespace containing the parsed command-line
                arguments and their corresponding values.
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
            dict[str, str]: Dictionary containing llama.cpp flags
                and their associated values.

        Raises:
            ValueError: If an argument that is not a valid flag is encountered.
        """

        def is_flag(value: str) -> bool:
            """
            Determines whether a command-line argument represents a flag.

            Numeric values prefixed with ``-`` are treated as values rather than
            flags, allowing negative numeric arguments to be parsed correctly.

            Args:
                value: Command-line argument to evaluate.

            Returns:
                bool: ``True`` if the value represents a flag; otherwise, ``False``.
            """
            if not value.startswith("-"):
                return False

            # Negative numeric values start with '-' but are still valid values
            try:
                float(value)
                return False
            except ValueError:
                return True

        result = {}
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
    """TODO: Describe class"""

    def __init__(self, path: Path):
        with path.open("rb") as file:
            data = tomllib.load(file)

        self.root = self.__validate(field="root", field_type="dir", data=data)
        self.editor = self.__validate(field="editor", field_type="str", data=data)
        self.browser_path = self.__validate(field="browser_path", field_type="file", data=data, required=False)

    def require_browser(self) -> Path:
        """Returns the browser's path if available"""
        if self.browser_path is None:
            raise ValueError("Browser is not configured")

        return Path(self.browser_path)

    def __validate(self, field: str, field_type: str, data: dict, required: bool = True):
        """TODO: Describe validation"""

        # Check if the field is defined
        if field not in data:
            if required:
                raise ValueError(f"Field '{field}' not defined in configs.toml")
            # Optional field: absence is represented by None
            return None

        # Colect the value of the field
        value = data[field]

        # Check if the value is a string
        if not isinstance(value, str):
            raise TypeError(f"Field '{field}' must be a string. Got '{value}' ({type(value)})")

        # Check if the string is empty
        if not value.strip():
            raise ValueError(f"Field '{field}' cannot be empty")

        # Check field's type
        match field_type:
            case "dir":
                path = Path(value)
                if path.is_dir():
                    return path
                else:
                    raise ValueError(f"Value '{value}' is not a valid '{field}' directory")

            case "file":
                path = Path(value)
                if path.is_file():
                    return path
                else:
                    raise ValueError(f"Field '{field}' does not contain a valid file ({value})")

            case "str":
                return value

            case _:
                raise ValueError(f"Unknown validation type '{field_type}'")


class Profiles:
    """TODO: Explain class"""

    def __init__(self, path: Path):
        with path.open("rb") as file:
            data = tomllib.load(file)

        # Validate the data before saving it
        self.__validate(data)
        self.__data = data

    def __getitem__(self, item):
        """TODO: Explain method"""
        return self.__data.__getitem__(item)

    def __iter__(self):
        """TODO: Explain method"""
        return self.__data.__iter__()

    def __validate(self, data: dict) -> None:
        """TODO: Explain validation"""

        if not isinstance(data, dict):
            raise TypeError("Profiles data must be a dictionary.")

        if "default" not in data:
            raise ValueError("Missing required field 'default'")

        for profile, flags in data.items():
            if not isinstance(flags, dict):
                raise TypeError(f"Profile '{profile}' must be a table")

            for flag in flags:
                if not flag.startswith("-"):
                    raise ValueError(f"'{flag}' from profile '{profile}' is not a valid llama.cpp")


class Model:
    """Stores model data"""

    def __init__(self, model: dict, path: Path, parent: Path, profiles: Profiles):
        self.path = path
        self.parent = parent
        self.profiles = profiles

        self.__validate(model, parent, profiles)
        self.name = model["name"]
        self.profile = model["profile"]
        self.parameters = model["parameters"]
        self.files = {file: (parent / path) for (file, path) in model["files"].items()}

        self.arguments = {}
        self.build_arguments(self.profiles[self.profile])

    def require_address(self) -> tuple[str, int]:
        """Return the host and port, if available"""
        # Let the dict validate if the key is present or not
        host = self.arguments["--host"]
        port = self.arguments["--port"]

        # Check if the value of host is valid
        if not (isinstance(host, str) and host.strip()):
            raise TypeError(f"Invalid value for field --host. Expected 'str', got '{type(host).__name__}'")

        # Check for port number
        if not isinstance(port, int):
            raise TypeError(f"Invalid type for field --port. Expected 'int', got '{type(port).__name__}'")

        return host, port

    def build_arguments(self, profile: dict) -> None:
        """Builds the correct arguments dict, given a specific profile"""
        self.arguments.clear()

        for argument in [
            self.profiles["default"],
            profile,
            self.parameters,
            self.files,
        ]:
            self.arguments.update(argument)

    def build_command(self) -> list[str]:
        """Build a valid Popen list to start a llama.cpp using the model's self arguments"""
        command = ["llama-server"]
        for parameter, value in self.arguments.items():
            command.append(parameter)
            if value != "":
                command.append(str(value))

        return command

    def __validate(self, model: dict, parent: Path, profiles: Profiles) -> None:
        """TODO: Explain validation"""

        missing = {"name", "profile", "parameters", "files"} - model.keys()
        if missing:
            raise ValueError(f"Missing required profile fields: {', '.join(missing)}")

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
                raise ValueError(f"Parameter '{flag}' is not a valid llama.cpp flag")

            if not isinstance(value, str):
                raise TypeError(f"Invalid file path for flag '{flag}'. Expected 'str', got '{type(value).__name__}'")

            if not value.strip():
                raise ValueError(f"File path for flag '{flag}' cannot be empty")

            if not (parent / value).is_file():
                raise ValueError(f"Flag '{flag}' does not contain a valid file path: {value}")


class Loader:
    """TODO: Describe the class"""

    def __init__(self, args: argparse.Namespace):
        self.models = {}  # Keeps models in a dict {name: Model}
        self.args = args  # Arguments collected from the argparser
        self.configs = Configs(ROOT / "configs.toml")  # Configurations set in the configs.toml
        self.profiles = Profiles(ROOT / "profiles.toml")  # Profiles defined by the user in profiles.toml

        # Populate self.models dictionary. Looks for any ".toml" file in the root set in the configs.toml
        models_root = self.configs.root
        toml_paths = models_root.rglob("*.toml")
        required_fields = {"name", "files", "profile", "parameters"}

        for toml_path in toml_paths:
            model_toml = tomllib.load(toml_path.open("rb"))

            # Will consider a valid model ONLY if the .toml have a name, file, profile and parameter set
            if required_fields <= model_toml.keys():
                name = model_toml["name"]

                if name in self.models:  # Validation for the duplicate "name" case
                    raise ValueError(f"Invalid model at '{toml_path}'. The name '{name}' already exists")
                else:
                    self.models[model_toml["name"]] = Model(model_toml, toml_path, toml_path.parent, self.profiles)

    def start(
        self,
        model: str,
        llamaargs: list | None = None,
        b: bool = False,
        i: bool = False,
    ) -> None:
        """TODO: Describe method"""

        # Check if the model is known
        if model not in self.models:
            raise ValueError(f"Unknown model: {model}.")

        # If the model is valid, get it from the list
        selected_model = self.models[model]

        # Check for any llama.cpp flags
        if llamaargs:
            llamaargs = llamaargs.copy()
            profile_arg = llamaargs[0]

            # Check if the first argument is a profile
            if profile_arg in self.profiles:
                llamaargs.pop(0)

                # If it's a valid profile, update the selected model
                new_profile = self.profiles[profile_arg]
                selected_model.build_arguments(new_profile)

            # If the first argument is not a profile or valid llama.cpp flag
            elif not profile_arg.startswith("-"):
                raise ValueError(f"{profile_arg} is not a valid profile or llama.cpp flag.")

            # Parse the llama.cpp flags as a dict and update selected model
            flags_dict = Argparser.args_to_dict(llamaargs)
            selected_model.arguments.update(flags_dict)

        # Check if the user selected to open browser
        if b or i:
            # Get the browser path. Validation raise an error if not set
            browser_path = self.configs.require_browser()
            # Get the host address and the port. Validation is done in the method
            browser_host, browser_port = selected_model.require_address()

            # If everything is set, we open the browser
            self.open_browser(browser_path, browser_host, browser_port, i)

        # Finally, create a valid subprocess command
        command = selected_model.build_command()

        # Try to open the llama.cpp server
        try:
            llama_process = subprocess.Popen(command)
            llama_process.wait()

        # Check if the user interrupted the process (CTRL + C)
        except KeyboardInterrupt:
            print("\nClosing the server...")
            llama_process.terminate()
            llama_process.wait()

        # Check if the user have llama.cpp set as a terminal command
        except FileNotFoundError:
            print("Error: llama.cpp was not found")

            # Inform the error and suggest how to install
            if os.name == "nt":
                print("\nInstall via winget with: 'winget install llama.cpp'")
            else:
                print("\nInstall via homebrew with: 'brew install llama.cpp'")

            raise SystemExit(
                "\nOr compile your own version from source: See more at https://github.com/ggml-org/llama.cpp"
            )

    def list(self, models: bool, profiles: bool) -> None:
        if models:
            print("\nModels:")
            for model in self.models.values():
                print(
                    f"Name: {model.name:<10}||  Profile: {model.profile:>10}  ||   Path: {model.parent.resolve()!s:<70}"
                )

        elif profiles:
            print("\nProfiles:")
            for profile in self.profiles:
                if profile != "default":
                    print(profile)

        else:
            print("\nModels:")
            for model in self.models.values():
                print(
                    f"Name: {model.name:<10}||  Profile: {model.profile:>10}  ||   Path: {model.parent.resolve()!s:<70}"
                )

            print("\nProfiles:")
            for profile in self.profiles:
                if profile != "default":
                    print(profile)

    def init(self, cwd: Path):
        """TODO: Explain method"""

        # The name of the generated .toml file
        name = f"{cwd.name.replace(' ', '-')}.toml"

        # llama.cpp flags for the file
        flags = {
            "--model": "",
            "--mmproj": "",
            "--model-draft": "",
            "--chat-template-file": "",
            "--spec-type": "ngram-mod",
        }

        # Look for any valid file extensions
        files = [file.name for file in cwd.iterdir() if file.suffix.lower() in (".gguf", ".jinja")]
        # Check for any file that looks like a llama.cpp value
        for file in files.copy():
            file_lower = file.lower()

            if ".jinja" in file_lower:
                flags["--chat-template-file"] = file
                files.remove(file)

            elif "mmproj" in file_lower:
                flags["--mmproj"] = file
                files.remove(file)

            elif "mtp" in file_lower:
                flags["--model-draft"] = file
                flags["--spec-type"] += ",draft-mtp"
                files.remove(file)

            elif "dflash" in file_lower:
                flags["--model-draft"] = file
                flags["--spec-type"] += ",draft-dflash"
                files.remove(file)

        # The remaining file should be the model
        flags["--model"] = files[0] if len(files) == 1 else ""

        # Resulting .toml file
        toml = textwrap.dedent(f"""
        # {cwd.name}


        # Name of the model. Must be unique.
        name = "{"-".join(name.split("-")[0:2]).lower()}"

        # Model's profile. Pick one from "profiles.toml".
        profile = "default"


        # Relative path of your files. Delete the flag if not needed.
        [files]
        --model = "{flags["--model"]}"
        --mmproj = "{flags["--mmproj"]}"
        --model-draft = "{flags["--model-draft"]}"
        --chat-template-file = "{flags["--chat-template-file"]}"


        # Aditional llama.cpp flags.
        [parameters]
        --agent = ""
        --spec-type = "{flags["--spec-type"]}"   
        """)

        # Create the .toml file
        output = cwd / name
        # Validate existence
        if output.exists():
            raise SystemExit(f"Error: '{output.name}' already exists")
        else:
            output.write_text(toml, encoding="utf-8")

    def edit(self, file: str) -> None:
        if file in ("configs", "profiles"):
            path = ROOT / Path(f"{file}.toml")

        elif file in self.models:
            path = self.models[file].path

        else:
            raise ValueError(f"{file} is not a valid model or file.")

        if path.is_file():
            subprocess.Popen([self.configs.editor, path])
        else:
            raise FileNotFoundError(f"{file} doesn't exists.")

    def show(self, model: str, profile: str | None = None) -> None:
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

    def open_browser(self, browser_path, host="127.0.0", port="9993", incognito: bool = False) -> None:
        """Opens the browser set in configs.toml. You can choose to open in incognito"""
        command = [browser_path, "--start-maximized", f"http://{host}:{port}"]
        if incognito:
            command.append("--incognito")

        subprocess.Popen(command)


if __name__ == "__main__":
    cli = CLI()
    loader = Loader(cli.parse_args())
    loader.run()

from pathlib import Path

from .profiles import Profiles


class Model:
    """
    Represents a validated llama.cpp model configuration.

    The class validates model metadata and file references, resolves model
    file paths, combines configuration layers into resolved llama.cpp
    arguments, and builds the command used to start llama-server.

    Args:
        model: Model configuration loaded from TOML.
        path: Path to the model's TOML configuration file.
        parent: Directory used to resolve model file paths.
        profiles: Available validated profiles.

    Attributes:
        path: Path to the model configuration file.
        parent: Base directory used to resolve model file paths.
        profiles: Available profiles.
        name: Unique model name.
        profile: Name of the selected profile.
        parameters: Model-specific llama.cpp parameters.
        files: Model file flags mapped to their resolved paths.
        arguments: Resolved llama.cpp arguments using the model's selected profile.

    Methods:
        require_address: Return the effective host and port.
        build_arguments: Build and return the resolved arguments for a profile.
        build_command: Build the command used to start llama-server.
    """

    REQUIRED_MODEL_FIELDS = frozenset({"name", "profile", "parameters", "files"})

    def __init__(self, model: dict, path: Path, parent: Path, profiles: Profiles) -> None:
        self.path = path
        self.parent = parent
        self.profiles = profiles

        self.__validate(model, parent, profiles)

        self.name: str = model["name"]
        self.profile: str = model["profile"]
        self.parameters: dict[str, object] = model["parameters"]
        self.files: dict[str, Path] = {flag: parent / file_path for flag, file_path in model["files"].items()}

        self.arguments: dict[str, object] = self.build_arguments(self.profiles[self.profile])

    def require_address(self) -> tuple[str, int]:
        """
        Return the effective server address for this model.

        Resolves ``--host`` and ``--port`` from the final arguments, coercing
        the port to an integer and validating its range.

        Returns:
            A ``(host, port)`` tuple.

        Raises:
            ValueError: If ``--host`` is empty or ``--port`` is not a valid port.
            TypeError: If ``--host`` is not a string or ``--port`` is not a number.
        """
        host = self.arguments["--host"]

        if not isinstance(host, str):
            raise TypeError(f"Flag '--host' must be a string. Got {host!r} ({type(host).__name__})")

        if not host.strip():
            raise ValueError("Flag '--host' cannot be empty")

        port = self.arguments["--port"]

        if isinstance(port, str):
            try:
                port = int(port)
            except ValueError:
                raise ValueError(f"Invalid port: {port!r}") from None

        if type(port) is not int:
            raise TypeError(f"Flag '--port' must be an integer. Got {port!r} ({type(port).__name__})")

        if not 1 <= port <= 65535:
            raise ValueError(f"Port {port!r} must be between 1 and 65535")

        return host, port

    def build_arguments(self, selected_profile: dict) -> dict[str, object]:
        """
        Build the resolved llama.cpp arguments for a profile.

        Merges the configuration layers in priority order from lowest to highest:
        the ``default`` profile, the selected profile, the model's ``parameters``,
        and the model's ``files``.

        The method returns a new dictionary and does not modify
        ``self.arguments``.

        Args:
            selected_profile: Profile table applied to this model.

        Returns:
            A new dictionary containing the resolved llama.cpp arguments.
        """
        arguments: dict[str, object] = {}

        for layer in [
            self.profiles["default"],
            selected_profile,
            self.parameters,
            self.files,
        ]:
            arguments.update(layer)

        return arguments

    def build_command(self) -> list[str]:
        """
        Build the command line used to start ``llama-server``.

        Flags with an empty value are emitted as bare flags; flags with a value
        are followed by that value as a separate argument.

        Returns:
            Command-line tokens starting with ``llama-server``.
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

        Ensures that the model configuration is a dictionary containing all
        required fields with valid types, that the selected profile exists,
        and that model file entries resolve to existing files.

        Args:
            model: Parsed model configuration.
            parent: Directory used to resolve model file paths.
            profiles: Available validated profiles.

        Raises:
            ValueError: If a required field is missing or empty, the selected
                profile does not exist, a flag name is invalid, or a model file
                does not exist.
            TypeError: If the model configuration or one of its required fields
                has the wrong type.
        """
        if not isinstance(model, dict):
            raise TypeError(f"Model configuration must be a dictionary. Got {model!r} ({type(model).__name__})")

        missing = self.REQUIRED_MODEL_FIELDS - model.keys()

        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"Missing required model fields: {missing_fields}")

        name = model["name"]

        if not isinstance(name, str):
            raise TypeError(f"Field 'name' must be a string. Got {name!r} ({type(name).__name__})")

        if not name.strip():
            raise ValueError("Field 'name' cannot be empty")

        profile = model["profile"]

        if not isinstance(profile, str):
            raise TypeError(f"Field 'profile' must be a string. Got {profile!r} ({type(profile).__name__})")

        if not profile.strip():
            raise ValueError("Field 'profile' cannot be empty")

        if profile not in profiles:
            raise ValueError(f"Profile '{profile}' defined by model '{name}' does not exist")

        parameters = model["parameters"]

        if not isinstance(parameters, dict):
            raise TypeError(
                f"Field 'parameters' must be a dictionary. Got {parameters!r} ({type(parameters).__name__})"
            )

        for parameter in parameters:
            if not isinstance(parameter, str):
                raise TypeError(f"Parameter must be a string. Got {parameter!r} ({type(parameter).__name__})")

            if not parameter.startswith("-"):
                raise ValueError(f"Parameter '{parameter}' is not a valid llama.cpp flag")

        files = model["files"]

        if not isinstance(files, dict):
            raise TypeError(f"Field 'files' must be a dictionary. Got {files!r} ({type(files).__name__})")

        for flag, value in files.items():
            if not isinstance(flag, str):
                raise TypeError(f"File flag must be a string. Got {flag!r} ({type(flag).__name__})")

            if not flag.startswith("-"):
                raise ValueError(f"File flag '{flag}' is not a valid llama.cpp flag")

            if not isinstance(value, str):
                raise TypeError(f"File path for flag '{flag}' must be a string. Got {value!r} ({type(value).__name__})")

            if not value.strip():
                raise ValueError(f"File path for flag '{flag}' cannot be empty")

            if not (parent / value).is_file():
                raise ValueError(f"Flag '{flag}' does not contain a valid file path: {value}")

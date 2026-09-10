from pathlib import Path

from llama_loader.profiles import Profiles


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

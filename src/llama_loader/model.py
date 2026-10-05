from pathlib import Path
from typing import Any

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
        parameters: Model-specific llama.cpp parameters owned by this model.
            Mutating them does not modify the source configuration mapping.
        files: Model file flags mapped to their resolved paths.
        arguments: Resolved llama.cpp arguments using the model's selected profile.

    Methods:
        require_address: Return the effective host and port.
        build_arguments: Build and return the resolved arguments for a profile.
        build_command: Build the command used to start llama-server.
    """

    REQUIRED_MODEL_FIELDS: frozenset[str] = frozenset({"name", "profile", "parameters", "files"})

    def __init__(self, model_data: dict[str, Any], model_path: Path, model_dir: Path, profiles: Profiles) -> None:
        self.model_path: Path = model_path
        self.model_dir: Path = model_dir
        self.profiles: Profiles = profiles

        self.__validate(model_data, model_path, model_dir, profiles)

        self.name: str = model_data["name"]
        self.profile: str = model_data["profile"]
        self.parameters: dict[str, str | int | float] = model_data["parameters"].copy()
        self.files: dict[str, Path] = {flag: model_dir / file_path for flag, file_path in model_data["files"].items()}

        self.arguments: dict[str, str | int | float | Path] = self.build_arguments(
            selected_profile=self.profiles[self.profile]
        )

        self.model_context = f"Model '{self.name}' at '{self.model_dir}'"

    def require_address(self) -> tuple[str, int]:
        """
        Return the effective server address for this model.

        Resolves ``--host`` and ``--port`` from the final arguments, coercing
        the port to an integer and validating its range.

        Returns:
            A ``(host, port)`` tuple.

        Raises:
            ValueError: If ``--host`` is empty or ``--port`` is not a valid port.
            TypeError: If ``--host`` is not a string or ``--port`` has an invalid type.
        """
        host: Any = self.arguments["--host"]

        if not isinstance(host, str):
            raise TypeError(
                f"{self.model_context}: Flag '--host' must be a string. Got {host!r} ({type(host).__name__})."
            )

        if not host.strip():
            raise ValueError(f"{self.model_context}: Flag '--host' cannot be empty.")

        port: Any = self.arguments["--port"]

        if isinstance(port, str):
            try:
                port = int(port)
            except ValueError:
                raise ValueError(f"{self.model_context}: Invalid port {port!r}.")

        if type(port) is not int:
            raise TypeError(
                f"{self.model_context}: Flag '--port' must be an integer. Got {port!r} ({type(port).__name__})."
            )

        if not 1 <= port <= 65535:
            raise ValueError(f"{self.model_context}: Port must be between 1 and 65535. Got {port!r}.")

        return host, port

    def build_arguments(self, selected_profile: dict[str, str | int | float]) -> dict[str, str | int | float | Path]:
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
        arguments: dict[str, Any] = {}

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
        command: list[str] = ["llama-server"]

        for parameter, value in self.arguments.items():
            command.append(parameter)

            if value != "":
                command.append(str(value))

        return command

    def __validate(self, model_data: dict[str, Any], model_path: Path, model_dir: Path, profiles: Profiles) -> None:
        """
        Validate a model configuration.

        Ensures that the model configuration is a dictionary containing all
        required fields with valid types, that the selected profile exists,
        and that model file entries resolve to existing files.

        Args:
            model_data: Parsed model configuration.
            parent: Directory used to resolve model file paths.
            profiles: Available validated profiles.

        Raises:
            ValueError: If a required field is missing or empty, the selected
                profile does not exist, a flag name is invalid, or a model file
                does not exist.
            TypeError: If the model configuration or one of its required fields
                has the wrong type.
        """
        model_context: str = f"Model configuration at '{model_path}'"

        missing: frozenset[str] = self.REQUIRED_MODEL_FIELDS - model_data.keys()

        if missing:
            missing_fields: str = ", ".join(sorted(missing))
            raise ValueError(f"{model_context}: Missing required fields: {missing_fields}.")

        name: Any = model_data["name"]

        if not isinstance(name, str):
            raise TypeError(f"{model_context}: Field 'name' must be a string. Got {name!r} ({type(name).__name__}).")

        if not name.strip():
            raise ValueError(f"{model_context}: Field 'name' cannot be empty.")

        model_context = f"Model '{name}' at '{model_path}'"

        profile: Any = model_data["profile"]

        if not isinstance(profile, str):
            raise TypeError(
                f"{model_context}: Field 'profile' must be a string. Got {profile!r} ({type(profile).__name__})."
            )

        if not profile.strip():
            raise ValueError(f"{model_context}: Field 'profile' cannot be empty.")

        if profile not in profiles:
            raise ValueError(f"{model_context}: Profile '{profile}' does not exist.")

        parameters: Any = model_data["parameters"]

        if not isinstance(parameters, dict):
            raise TypeError(
                f"{model_context}: Field 'parameters' must be a dictionary. Got {parameters!r} ({type(parameters).__name__})."
            )

        for parameter, value in parameters.items():
            if not isinstance(parameter, str):
                raise TypeError(
                    f"{model_context}: Parameter '{parameter!r}' must be a string. Got '{type(parameter).__name__}'."
                )

            if not parameter.startswith("-"):
                raise ValueError(f"{model_context}: Parameter '{parameter}' is not a valid llama.cpp flag.")

            value_type: type[Any] = type(value)
            if value_type not in (str, int, float):
                raise TypeError(
                    f"{model_context}: Parameter '{parameter}' is an invalid field type: {value_type.__name__}."
                )

        files: Any = model_data["files"]

        if not isinstance(files, dict):
            raise TypeError(
                f"{model_context}: Field 'files' must be a dictionary. Got {files!r} ({type(files).__name__})."
            )

        for flag, value in files.items():
            if not isinstance(flag, str):
                raise TypeError(f"{model_context}: File flag must be a string. Got {flag!r} ({type(flag).__name__}).")

            if not flag.startswith("-"):
                raise ValueError(f"{model_context}: File flag '{flag}' is not a valid llama.cpp flag.")

            if not isinstance(value, str):
                raise TypeError(
                    f"{model_context}: File path for flag '{flag}' must be a string. Got {value!r} ({type(value).__name__})."
                )

            if not value.strip():
                raise ValueError(f"{model_context}: File path for flag '{flag}' cannot be empty.")

            if not (model_dir / value).is_file():
                raise ValueError(f"{model_context}: Flag '{flag}' does not contain a valid file path: {value}.")

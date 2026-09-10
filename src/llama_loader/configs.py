import tomllib
from pathlib import Path


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
        self.browser_path: Path | None = self.__validate(
            field="browser_path", validation_type="file", data=data, required=False
        )

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

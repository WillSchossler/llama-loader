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
        root: Directory where llama-loader searches for model configurations.
        editor: Command used to open the configured editor.
        browser_path: Path to the browser executable, or None if not configured.

    Methods:
        require_browser: Return the configured browser path
            or raise an error if no browser is configured.
    """

    def __init__(self, configs_path: Path) -> None:
        with configs_path.open("rb") as file:
            data = tomllib.load(file)

        self.root: Path = self.__validate_field(
            field="root",
            validation_type="dir",
            data=data,
        )
        self.editor: str = self.__validate_field(
            field="editor",
            validation_type="str",
            data=data,
        )
        self.browser_path: Path | None = self.__validate_field(
            field="browser_path",
            validation_type="file",
            data=data,
            required=False,
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

    def __validate_field(
        self,
        field: str,
        validation_type: str,
        data: dict,
        required: bool = True,
    ):
        """
        Validate and normalize a configuration field.

        Args:
            field: Name of the field to validate.
            validation_type: Expected field kind: "dir", "file", or "str".
            data: Parsed configuration mapping.
            required: Whether the field must be present.

        Returns:
            The validated string or Path, or None for an absent optional field.

        Raises:
            ValueError: If the field is missing, empty, invalid, or the
                validation type is unknown.
            TypeError: If the field value is not a string.
        """
        if field not in data:
            if required:
                raise ValueError(f"Field '{field}' is not defined in configs.toml")
            return None

        value = data[field]

        if not isinstance(value, str):
            raise TypeError(f"Field '{field}' must be a string. Got {value!r} ({type(value).__name__})")

        if not value.strip():
            raise ValueError(f"Field '{field}' cannot be empty")

        match validation_type:
            case "dir":
                path = Path(value)

                if not path.is_dir():
                    raise ValueError(f"Field '{field}' must point to an existing directory. Got {value!r}")

                return path

            case "file":
                path = Path(value)

                if not path.is_file():
                    raise ValueError(f"Field '{field}' must point to an existing file. Got {value!r}")

                return path

            case "str":
                return value

            case _:
                raise ValueError(f"Unknown validation type {validation_type!r}")

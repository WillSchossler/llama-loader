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
        data: dict[str, dict[str, object]],
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


"""

import tomllib
from pathlib import Path


class Configs:
    """
    Lazily load and validate llama-loader's user configuration.

    The configuration file itself is checked when Configs is created, but its
    TOML content and individual fields are only loaded and validated when they
    are first accessed.

    Args:
        configs_path: Path to the TOML configuration file.

    Attributes:
        configs_path: Path to the configuration file.
        root: Validated directory where llama-loader searches for model configurations.
        editor: Validated command used to open the configured editor.
        browser_path: Validated path to the browser executable, or None if not configured.

    Methods:
        require_browser: Return the configured browser path or raise an error if
            no browser is configured.
    """

    def __init__(self, configs_path: Path) -> None:
        if not configs_path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {configs_path}")

        self.configs_path = configs_path

        self._data: dict[str, object] | None = None

        self._root: Path | None = None
        self._editor: str | None = None

        self._browser_path: Path | None = None
        self._browser_loaded = False

    @property
    def root(self) -> Path:
        """
        Return the validated models root directory.

        The field is loaded and validated only on first access, then cached.

        Returns:
            Existing directory configured as the models root.

        Raises:
            ValueError: If the field is missing, empty, or does not point to an
                existing directory.
            TypeError: If the field is not a string.
        """
        if self._root is None:
            value = self.__get_field("root")

            assert value is not None

            path = Path(value)

            if not path.is_dir():
                raise ValueError(f"Field 'root' must point to an existing directory. Got {value!r}")

            self._root = path

        return self._root

    @property
    def editor(self) -> str:
        """
        Return the validated editor command.

        The field is loaded and validated only on first access, then cached.

        Returns:
            Configured editor command.

        Raises:
            ValueError: If the field is missing or empty.
            TypeError: If the field is not a string.
        """
        if self._editor is None:
            value = self.__get_field("editor")

            assert value is not None
            self._editor = value

        return self._editor

    @property
    def browser_path(self) -> Path | None:
        """
        Return the validated browser executable path when configured.

        The field is loaded and validated only on first access, then cached.

        Returns:
            Existing browser executable path, or None if no browser is configured.

        Raises:
            ValueError: If the configured path is empty or does not point to an
                existing file.
            TypeError: If the field is not a string.
        """
        if not self._browser_loaded:
            value = self.__get_field("browser_path", required=False)

            if value is not None:
                path = Path(value)

                if not path.is_file():
                    raise ValueError(f"Field 'browser_path' must point to an existing file. Got {value!r}")

                self._browser_path = path

            self._browser_loaded = True

        return self._browser_path

    def require_browser(self) -> Path:
        """
        Return the configured browser executable path.

        Returns:
            Validated browser executable path.

        Raises:
            ValueError: If no browser is configured or the configured path is invalid.
        """
        browser_path = self.browser_path

        if browser_path is None:
            raise ValueError("Browser is not configured")

        return browser_path

    def __load_data(self) -> dict[str, object]:
        """
        Parse the configuration TOML on first use and cache the result.

        Returns:
            Parsed configuration mapping.
        """
        if self._data is None:
            with self.configs_path.open("rb") as file:
                self._data = tomllib.load(file)

        return self._data

    def __get_field(self, field: str, required: bool = True) -> str | None:
        """
        Load and validate a string configuration field.

        Args:
            field: Name of the field to retrieve.
            required: Whether the field must be present.

        Returns:
            Validated string value, or None for an absent optional field.

        Raises:
            ValueError: If a required field is missing or the value is empty.
            TypeError: If the field value is not a string.
        """
        data = self.__load_data()

        if field not in data:
            if required:
                raise ValueError(f"Field '{field}' is not defined in configs.toml")

            return None

        value = data[field]

        if not isinstance(value, str):
            raise TypeError(f"Field '{field}' must be a string. Got {value!r} ({type(value).__name__})")

        if not value.strip():
            raise ValueError(f"Field '{field}' cannot be empty")

        return value

"""
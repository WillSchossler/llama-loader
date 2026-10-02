import os
import sys
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

            path = Path(value).expanduser()

            if not path.is_absolute():
                raise ValueError("Field 'root_path' must be an absolute path")

            if not path.is_dir():
                raise ValueError(f"Field 'root' must point to an existing directory. Got {value!r}")

            self._root = path

        return self._root

    @property
    def editor(self) -> str:
        """
        Return the configured editor command or a platform fallback.

        The field is loaded and validated only on first access, then cached.

        Returns:
            Configured editor command or a platform-specific fallback.

        Raises:
            ValueError: If the configured field is empty.
            TypeError: If the configured field is not a string.
        """
        if self._editor is None:
            value = self.__get_field("editor", required=False)

            if value is not None:
                self._editor = value
            elif os.name == "nt":
                self._editor = "notepad"
            elif sys.platform == "darwin":
                self._editor = "open"
            else:
                self._editor = "xdg-open"

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

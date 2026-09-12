import tomllib
from pathlib import Path


class Profiles:
    """
    Loads and validates profiles defined in a TOML file.

    The class provides limited mapping-like access to validated profiles,
    allowing profile lookup and iteration without exposing the internal
    profiles dictionary directly.

    The default profile must define the settings required internally by
    llama-loader.

    Args:
        profiles_path: Path to the profiles TOML file.

    Methods:
        __getitem__: Return a profile by name.
        __iter__: Iterate over profile names.
    """

    def __init__(self, profiles_path: Path) -> None:
        with profiles_path.open("rb") as file:
            data = tomllib.load(file)

        self.__validate(data)
        self.__data: dict[str, dict[str, object]] = data

    def __getitem__(self, item: str) -> dict[str, object]:
        return self.__data[item]

    def __iter__(self):
        return iter(self.__data)

    def __validate(self, data: dict[str, dict[str, object]]) -> None:
        """
        Validate the parsed profiles data.

        Ensures that the data defines a ``default`` profile, every profile is
        a table of flags, every flag starts with ``"-"``, and the default
        profile defines a non-empty ``--host`` string and an integer ``--port``.

        Args:
            data: Parsed profiles dictionary.

        Raises:
            TypeError: If the profiles data or a profile is not a dictionary,
                or if ``--host`` or ``--port`` has the wrong type.
            ValueError: If the default profile is missing, a flag does not
                start with ``"-"``, or a required default flag is missing or empty.
        """
        if not isinstance(data, dict):
            raise TypeError("Profiles data must be a dictionary")

        if "default" not in data:
            raise ValueError("Missing required field 'default'")

        for profile, flags in data.items():
            if not isinstance(flags, dict):
                raise TypeError(f"Profile '{profile}' must be a table")

            for flag in flags:
                if not flag.startswith("-"):
                    raise ValueError(f"'{flag}' from profile '{profile}' must start with '-'")

        default = data["default"]
        missing = {"--host", "--port"} - default.keys()

        if missing:
            missing_flags = ", ".join(sorted(missing))
            raise ValueError(f"Default profile is missing required flags: {missing_flags}")

        host = default["--host"]
        port = default["--port"]

        if not isinstance(host, str):
            raise TypeError(f"Default '--host' must be a string. Got {host!r} ({type(host).__name__})")

        if not host.strip():
            raise ValueError("Default '--host' cannot be empty")

        if type(port) is not int:
            raise TypeError(f"Default '--port' must be an integer. Got {port!r} ({type(port).__name__})")

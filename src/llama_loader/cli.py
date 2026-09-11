import argparse


class CLI:
    """
    Defines and parses llama-loader's command-line interface.

    The CLI declares the application's commands, arguments, and options,
    parses user input into an argparse.Namespace, and converts passthrough
    llama.cpp arguments into a flag/value mapping.

    It does not execute application commands or manage model state.

    Attributes:
        parser: Root ArgumentParser used to parse command-line arguments.

    Methods:
        parse_args: Parse the command-line arguments provided by the user.
        args_to_dict: Convert llama.cpp passthrough arguments into a flag/value mapping.
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Load and manage llama.cpp models via llama-server.",
            prog="loader",
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
        Parse the command-line arguments provided by the user.

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
                raise ValueError(f"llama.cpp flags should start with '-', got {key!r}.")

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

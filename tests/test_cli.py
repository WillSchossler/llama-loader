
import pytest

from llama_loader.cli import CLI


def test_args_to_dict_converts_flag_with_value():
    result = CLI.args_to_dict(["--ctx-size", "8192"])

    assert result == {"--ctx-size": "8192"}


def test_args_to_dict_converts_flag_without_value():
    result = CLI.args_to_dict(
        [
            "--jinja",
        ]
    )

    assert result == {"--jinja": ""}


def test_args_to_dict_accepts_negative_numeric_value():
    result = CLI.args_to_dict(["--temp", "-0.5"])

    assert result == {"--temp": "-0.5"}


def test_args_to_dict_raises_for_value_without_flag():
    with pytest.raises(ValueError, match="llama.cpp flags should start with '-'"):
        CLI.args_to_dict(["8192"])


def test_args_to_dict_converts_multiple_flags():
    result = CLI.args_to_dict(["--parallel", "-1", "--agent"])

    assert result == {"--parallel": "-1", "--agent": ""}


def test_list_command():
    cli = CLI()

    result = cli.parser.parse_args(["list"])

    assert result.command == "list"


def test_list_command_sets_models_flag():
    cli = CLI()

    result = cli.parser.parse_args(["list", "-m"])
    
    assert result.command == "list"
    assert result.models


def test_list_command_sets_profiles_flag():
    cli = CLI()

    result = cli.parser.parse_args(["list", "-p"])
    
    assert result.command == "list"
    assert result.profiles


def test_list_command_raises_for_mutually_group():
    cli = CLI()

    with pytest.raises(SystemExit):
        cli.parser.parse_args(["list", "-m", "-p"])


def test_show_command():
    cli = CLI()

    result = cli.parser.parse_args(["show", "gemma"])

    assert result.command == "show"
    assert result.model == "gemma"
    assert result.profile is None


def test_show_command_parses_optional_profile():
    cli = CLI()

    result = cli.parser.parse_args(["show", "gemma", "xhigh"])
    
    assert result.command == "show"
    assert result.model == "gemma"
    assert result.profile == "xhigh"


def test_show_command_raises_without_required_model():
    cli = CLI()

    with pytest.raises(SystemExit):
        cli.parser.parse_args(["show"])


def test_edit_command():
    cli = CLI()

    result = cli.parser.parse_args(["edit", "gemma"])

    assert result.command == "edit"
    assert result.file == "gemma"


def test_edit_command_parses_file():
    cli = CLI()

    result = cli.parser.parse_args(["edit", "gemma"])

    assert result.command == "edit"
    assert result.file == "gemma"


def test_edit_command_raises_without_required_file():
    cli = CLI()

    with pytest.raises(SystemExit):
        cli.parser.parse_args(["edit"])


def test_init_command():
    cli = CLI()

    result = cli.parser.parse_args(["init"])

    assert result.command == "init"


def test_start_command_parses_model():
    cli = CLI()

    result = cli.parser.parse_args(["start", "gemma"])

    assert result.command == "start"
    assert result.model == "gemma"
    assert result.llamaargs == []


def test_start_command_raises_without_required_model():
    cli = CLI()

    with pytest.raises(SystemExit):
        cli.parser.parse_args(["start"])


def test_start_command_parses_model_and_llamaargs():
    cli = CLI()

    result = cli.parser.parse_args(["start", "gemma", "xhigh", "--jinja", "--threads", "6"])

    assert result.command == "start"
    assert result.model == "gemma"
    assert result.llamaargs == ["xhigh", "--jinja", "--threads", "6"]


def test_start_command_parses_browser_flag():
    cli = CLI()

    result = cli.parser.parse_args(["start", "-b", "gemma", "xhigh"])

    assert result.command == "start"
    assert result.model == "gemma"
    assert result.llamaargs == ["xhigh"]
    assert result.b
    assert not result.i


def test_start_command_parses_incognito_flag():
    cli = CLI()

    result = cli.parser.parse_args(["start", "-i", "gemma"])

    assert result.i
    assert not result.b


def test_start_command_raises_for_mutually_exclusive_options():
    cli = CLI()

    with pytest.raises(SystemExit):
        cli.parser.parse_args(["start", "-b", "-i", "gemma"])


def test_start_command_treats_flags_after_model_as_llamaargs():
    cli = CLI()

    result = cli.parser.parse_args(["start", "gemma", "-b"])

    assert result.command == "start"
    assert result.model == "gemma"
    assert not result.b
    assert result.llamaargs == ["-b"]
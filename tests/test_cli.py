from llama_loader import CLI


def test_args_to_dict_converts_flag_with_value():
    result = CLI.args_to_dict(["--ctx-size", "8192"])

    assert result == {"--ctx-size": "8192"}

def test_args_to_dict_converts_flag_without_value():
    result = CLI.args_to_dict(["--jinja", ])

    assert result == {"--jinja": ""}

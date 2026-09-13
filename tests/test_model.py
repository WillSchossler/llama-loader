from pathlib import Path
from typing import Any

import pytest

from llama_loader.model import Model
from llama_loader.profiles import Profiles


def create_profiles(tmp_path: Path) -> Profiles:
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(
        """
        [default]
        --host = "127.0.0.1"
        --port = 9931
        --threads = 4
        --temp = 0.1
        --batch-size = 512
        --shared = "default"

        [fast]
        --host = "0.0.0.0"
        --port = 8080
        --threads = 8
        --temp = 0.2
        --ctx-size = 16384
        --shared = "fast"
        """
    )

    return Profiles(profiles_path)


def create_valid_model_data() -> dict:
    return {
        "name": "qwen",
        "profile": "default",
        "parameters": {
            "--temp": 0.7,
            "--jinja": "",
        },
        "files": {
            "--model": "model.gguf",
        },
    }


def create_model(tmp_path: Path, model_data: dict | None = None) -> Model:
    profiles = create_profiles(tmp_path)

    model_file = tmp_path / "model.gguf"
    model_file.touch()

    if model_data is None:
        model_data = create_valid_model_data()

    return Model(
        model=model_data,
        path=tmp_path / "qwen.toml",
        parent=tmp_path,
        profiles=profiles,
    )


def test_model_valid_configuration(tmp_path):
    model = create_model(tmp_path)

    assert model.name == "qwen"
    assert model.profile == "default"
    assert model.parameters == {
        "--temp": 0.7,
        "--jinja": "",
    }
    assert model.files == {
        "--model": tmp_path / "model.gguf",
    }
    assert model.path == tmp_path / "qwen.toml"
    assert model.parent == tmp_path


def test_model_configuration_must_be_dictionary(tmp_path):
    profiles = create_profiles(tmp_path)
    invalid_model: Any = "not a dictionary"

    with pytest.raises(TypeError, match="Model configuration must be a dictionary"):
        Model(
            model=invalid_model,
            path=tmp_path / "qwen.toml",
            parent=tmp_path,
            profiles=profiles,
        )


def test_model_requires_required_fields(tmp_path):
    model_data = create_valid_model_data()
    del model_data["name"]

    with pytest.raises(ValueError, match="Missing required model fields: name"):
        create_model(tmp_path, model_data)


def test_model_name_must_be_string(tmp_path):
    model_data = create_valid_model_data()
    model_data["name"] = 123

    with pytest.raises(TypeError, match="Field 'name' must be a string"):
        create_model(tmp_path, model_data)


def test_model_name_cannot_be_whitespace(tmp_path):
    model_data = create_valid_model_data()
    model_data["name"] = "   "

    with pytest.raises(ValueError, match="Field 'name' cannot be empty"):
        create_model(tmp_path, model_data)


def test_model_profile_must_be_string(tmp_path):
    model_data = create_valid_model_data()
    model_data["profile"] = 123

    with pytest.raises(TypeError, match="Field 'profile' must be a string"):
        create_model(tmp_path, model_data)


def test_model_profile_cannot_be_whitespace(tmp_path):
    model_data = create_valid_model_data()
    model_data["profile"] = "   "

    with pytest.raises(ValueError, match="Field 'profile' cannot be empty"):
        create_model(tmp_path, model_data)


def test_model_profile_must_exist(tmp_path):
    model_data = create_valid_model_data()
    model_data["profile"] = "missing"

    with pytest.raises(ValueError, match="Profile 'missing' defined by model 'qwen' does not exist"):
        create_model(tmp_path, model_data)


def test_model_parameters_must_be_dictionary(tmp_path):
    model_data = create_valid_model_data()
    model_data["parameters"] = "not a dictionary"

    with pytest.raises(TypeError, match="Field 'parameters' must be a dictionary"):
        create_model(tmp_path, model_data)


def test_model_parameter_name_must_be_string(tmp_path):
    model_data = create_valid_model_data()
    model_data["parameters"] = {
        123: "value",
    }

    with pytest.raises(TypeError, match="Parameter must be a string"):
        create_model(tmp_path, model_data)


def test_model_parameter_must_start_with_dash(tmp_path):
    model_data = create_valid_model_data()
    model_data["parameters"] = {
        "threads": 8,
    }

    with pytest.raises(ValueError, match="'threads' is not a valid llama.cpp flag"):
        create_model(tmp_path, model_data)


def test_model_files_must_be_dictionary(tmp_path):
    model_data = create_valid_model_data()
    model_data["files"] = "not a dictionary"

    with pytest.raises(TypeError, match="Field 'files' must be a dictionary"):
        create_model(tmp_path, model_data)


def test_model_file_flag_must_be_string(tmp_path):
    model_data = create_valid_model_data()
    model_data["files"] = {
        123: "model.gguf",
    }

    with pytest.raises(TypeError, match="File flag must be a string"):
        create_model(tmp_path, model_data)


def test_model_file_flag_must_start_with_dash(tmp_path):
    model_data = create_valid_model_data()
    model_data["files"] = {
        "model": "model.gguf",
    }

    with pytest.raises(ValueError, match="'model' is not a valid llama.cpp flag"):
        create_model(tmp_path, model_data)


def test_model_file_path_must_be_string(tmp_path):
    model_data = create_valid_model_data()
    model_data["files"] = {
        "--model": 123,
    }

    with pytest.raises(TypeError, match="File path for flag '--model' must be a string"):
        create_model(tmp_path, model_data)


def test_model_file_path_cannot_be_whitespace(tmp_path):
    model_data = create_valid_model_data()
    model_data["files"] = {
        "--model": "   ",
    }

    with pytest.raises(ValueError, match="File path for flag '--model' cannot be empty"):
        create_model(tmp_path, model_data)


def test_model_file_must_exist(tmp_path):
    model_data = create_valid_model_data()
    model_data["files"] = {
        "--model": "missing.gguf",
    }

    with pytest.raises(ValueError, match="does not contain a valid file path"):
        create_model(tmp_path, model_data)


def test_require_address_returns_valid_address(tmp_path):
    model = create_model(tmp_path)

    host, port = model.require_address()

    assert host == "127.0.0.1"
    assert port == 9931


def test_require_address_accepts_numeric_string_port(tmp_path):
    model = create_model(tmp_path)
    model.arguments["--port"] = "8080"

    host, port = model.require_address()

    assert host == "127.0.0.1"
    assert port == 8080


def test_require_address_host_must_be_string(tmp_path):
    model = create_model(tmp_path)
    model.arguments["--host"] = 123

    with pytest.raises(TypeError, match="Flag '--host' must be a string"):
        model.require_address()


def test_require_address_host_cannot_be_whitespace(tmp_path):
    model = create_model(tmp_path)
    model.arguments["--host"] = "   "

    with pytest.raises(ValueError, match="Flag '--host' cannot be empty"):
        model.require_address()


def test_require_address_rejects_non_numeric_string_port(tmp_path):
    model = create_model(tmp_path)
    model.arguments["--port"] = "batata"

    with pytest.raises(ValueError, match="Invalid port"):
        model.require_address()


def test_require_address_rejects_port_type(tmp_path):
    model = create_model(tmp_path)
    model.arguments["--port"] = True

    with pytest.raises(TypeError, match="Flag '--port' must be an integer"):
        model.require_address()


def test_require_address_rejects_port_outside_valid_range(tmp_path):
    model = create_model(tmp_path)

    model.arguments["--port"] = 0

    with pytest.raises(ValueError, match="must be between 1 and 65535"):
        model.require_address()

    model.arguments["--port"] = 65536

    with pytest.raises(ValueError, match="must be between 1 and 65535"):
        model.require_address()


def test_build_arguments_merges_layers_in_priority_order(tmp_path):
    shared_file = tmp_path / "shared.gguf"
    shared_file.touch()

    model_data = create_valid_model_data()
    model_data["parameters"] = {
        "--temp": 0.7,
        "--shared": "parameters",
        "--jinja": "",
    }
    model_data["files"] = {
        "--model": "model.gguf",
        "--shared": "shared.gguf",
    }

    model = create_model(tmp_path, model_data)

    arguments = model.build_arguments(model.profiles["fast"])

    assert arguments["--batch-size"] == 512
    assert arguments["--threads"] == 8
    assert arguments["--ctx-size"] == 16384
    assert arguments["--temp"] == 0.7
    assert arguments["--shared"] == shared_file
    assert arguments["--model"] == tmp_path / "model.gguf"


def test_build_arguments_does_not_modify_model_arguments(tmp_path):
    model = create_model(tmp_path)
    original_arguments = model.arguments.copy()

    arguments = model.build_arguments(model.profiles["fast"])

    assert model.arguments == original_arguments
    assert arguments is not model.arguments
    assert arguments["--host"] == "0.0.0.0"
    assert model.arguments["--host"] == "127.0.0.1"


def test_build_command_returns_expected_command(tmp_path):
    model = create_model(tmp_path)

    model.arguments = {
        "--threads": 16,
        "--temp": 0.7,
        "--jinja": "",
        "--model": tmp_path / "model.gguf",
    }

    command = model.build_command()

    assert command == [
        "llama-server",
        "--threads",
        "16",
        "--temp",
        "0.7",
        "--jinja",
        "--model",
        str(tmp_path / "model.gguf"),
    ]

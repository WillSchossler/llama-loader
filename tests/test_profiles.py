from pathlib import Path

import pytest

from llama_loader.profiles import Profiles


def create_profiles(tmp_path: Path, toml: str) -> Profiles:
    profiles_path = tmp_path / "profiles.toml"
    profiles_path.write_text(toml)

    return Profiles(profiles_path)


def test_profiles_loads_valid_file(tmp_path):
    toml = """
    [default]
    --agent = ""
    --port = 9931
    --host = "127.0.0.1"

    [balanced]
    --temp = 0.60
    """

    profiles = create_profiles(tmp_path, toml)

    assert profiles["default"]["--agent"] == ""
    assert profiles["default"]["--host"] == "127.0.0.1"
    assert profiles["default"]["--port"] == 9931
    assert profiles["balanced"]["--temp"] == 0.6


def test_profiles_returns_item(tmp_path):
    toml = """
    [default]
    --port = 9931
    --host = "127.0.0.1"
    """

    profiles = create_profiles(tmp_path, toml)

    assert profiles["default"]["--port"] == 9931


def test_profiles_iterates_profile_names(tmp_path):
    toml = """
    [default]
    --port = 9931
    --host = "127.0.0.1"

    [balanced]
    --temp = 0.60
    """

    profiles = create_profiles(tmp_path, toml)

    assert list(profiles) == ["default", "balanced"]


def test_profiles_raises_when_missing_default(tmp_path):
    toml = """
    [balanced]
    --agent = ""
    """

    with pytest.raises(ValueError, match="Missing required field 'default'"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_default_is_not_table(tmp_path):
    toml = """
    default = 1
    """

    with pytest.raises(TypeError, match="Profile 'default' must be a table"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_profile_is_not_table(tmp_path):
    toml = """
    balanced = 123

    [default]
    --host = "127.0.0.1"
    --port = 9931
    """

    with pytest.raises(TypeError, match="Profile 'balanced' must be a table"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_flag_has_wrong_prefix(tmp_path):
    toml = """
    [default]
    --port = 9931
    --host = "127.0.0.1"

    [balanced]
    fit = "on"
    """

    with pytest.raises(ValueError, match="'fit' from profile 'balanced' must start with '-'"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_host_is_missing(tmp_path):
    toml = """
    [default]
    --port = 9931
    """

    with pytest.raises(ValueError, match="Default profile is missing required flags: --host"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_host_is_not_string(tmp_path):
    toml = """
    [default]
    --host = 1234
    --port = 9931
    """

    with pytest.raises(TypeError, match="Default '--host' must be a string. Got 1234"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_host_is_empty(tmp_path):
    toml = """
    [default]
    --host = ""
    --port = 9931
    """

    with pytest.raises(ValueError, match="Default '--host' cannot be empty"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_host_is_whitespace(tmp_path):
    toml = """
    [default]
    --host = "   "
    --port = 9931
    """

    with pytest.raises(ValueError, match="Default '--host' cannot be empty"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_port_is_missing(tmp_path):
    toml = """
    [default]
    --host = "127.0.0.1"
    """

    with pytest.raises(ValueError, match="Default profile is missing required flags: --port"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_port_is_not_integer(tmp_path):
    toml = """
    [default]
    --host = "127.0.0.1"
    --port = "foo"
    """

    with pytest.raises(TypeError, match="Default '--port' must be an integer. Got 'foo'"):
        create_profiles(tmp_path, toml)


def test_profiles_raises_when_port_is_boolean(tmp_path):
    toml = """
    [default]
    --host = "127.0.0.1"
    --port = true
    """

    with pytest.raises(TypeError, match="Default '--port' must be an integer. Got True"):
        create_profiles(tmp_path, toml)


def test_profiles_returned_item_does_not_mutate_internal_data(tmp_path):
    toml = """
    [default]
    --host = "127.0.0.1"
    --port = 9931
    """

    profiles = create_profiles(tmp_path, toml)

    default_profile = profiles["default"]
    default_profile["--port"] = 1234

    assert profiles["default"]["--port"] == 9931

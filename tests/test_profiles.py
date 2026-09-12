from llama_loader.profiles import Profiles


def test_valid_profiles(tmp_path):
    profile_path = tmp_path / "profiles.toml"

    toml = f"""
    [default]
    --agent = ""
    --fit = "on"
    --port = 9931
    --host = "127.0.0.1"

    [balanced]
    --temp = 0.60
    """

    profile_path.write_text(toml)

    profiles = Profiles(profile_path)

    assert profiles["default"]["--agent"] == ""
    assert profiles["default"]["--fit"] == "on"
    assert profiles["default"]["--host"] == "127.0.0.1"
    assert profiles["default"]["--port"] == 9931

    assert profiles["balanced"]["--temp"] == 0.6



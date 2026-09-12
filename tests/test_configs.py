from pathlib import Path

import pytest

from llama_loader.configs import Configs


def create_configs_file(
    tmp_path: Path,
    include_root: bool = True,
    include_editor: bool = True,
    include_browser: bool = True,
):
    if include_root:
        models_path = tmp_path / "model"
        models_path.mkdir()
    else:
        models_path = None

    if include_browser:
        browser_path = tmp_path / "browser.exe"
        browser_path.touch()
    else:
        browser_path = None

    configs_path = tmp_path / "configs.toml"
    toml = ""

    if include_root:
        toml += f"root = '{models_path}'\n"

    if include_editor:
        toml += 'editor = "code.cmd"\n'

    if include_browser:
        toml += f"browser_path = '{browser_path}'\n"

    configs_path.write_text(toml)

    return configs_path, models_path, browser_path


def test_configs_loads_valid_configuration(tmp_path):
    configs_path, models_path, browser_path = create_configs_file(tmp_path)

    configs = Configs(configs_path)

    assert configs.root == models_path
    assert configs.editor == "code.cmd"
    assert configs.browser_path == browser_path


def test_configs_raises_for_missing_root_field(tmp_path):
    configs_path, _, _ = create_configs_file(tmp_path, include_root=False)

    with pytest.raises(ValueError, match="Field 'root' is not defined in configs.toml"):
        Configs(configs_path)


def test_configs_raises_for_missing_editor_field(tmp_path):
    configs_path, _, _ = create_configs_file(tmp_path, include_editor=False)

    with pytest.raises(ValueError, match="Field 'editor' is not defined in configs.toml"):
        Configs(configs_path)


def test_configs_allows_missing_browser_path(tmp_path):
    configs_path, _, _ = create_configs_file(tmp_path, include_browser=False)

    configs = Configs(configs_path)

    assert configs.browser_path is None


def test_require_browser_returns_browser_path(tmp_path):
    configs_path, _, browser_path = create_configs_file(tmp_path)

    configs = Configs(configs_path)

    assert configs.require_browser() == browser_path


def test_require_browser_raises_when_browser_is_not_configured(tmp_path):
    configs_path, _, _ = create_configs_file(tmp_path, include_browser=False)

    configs = Configs(configs_path)

    with pytest.raises(ValueError, match="Browser is not configured"):
        configs.require_browser()


def test_configs_raises_for_non_string_value(tmp_path):
    models_path = tmp_path / "model"
    models_path.mkdir()

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{models_path}'
        editor = 123
        """
    )

    with pytest.raises(TypeError, match="Field 'editor' must be a string"):
        Configs(configs_path)


def test_configs_raises_for_empty_value(tmp_path):
    models_path = tmp_path / "model"
    models_path.mkdir()

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{models_path}'
        editor = ""
        """
    )

    with pytest.raises(ValueError, match="Field 'editor' cannot be empty"):
        Configs(configs_path)


def test_configs_raises_for_nonexistent_root_directory(tmp_path):
    nonexistent_root = tmp_path / "does-not-exist"

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{nonexistent_root}'
        editor = "code.cmd"
        """
    )

    with pytest.raises(ValueError, match="Field 'root' must point to an existing directory"):
        Configs(configs_path)


def test_configs_raises_for_nonexistent_browser_file(tmp_path):
    models_path = tmp_path / "model"
    models_path.mkdir()

    nonexistent_browser = tmp_path / "browser.exe"

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{models_path}'
        editor = "code.cmd"
        browser_path = '{nonexistent_browser}'
        """
    )

    with pytest.raises(ValueError, match="Field 'browser_path' must point to an existing file"):
        Configs(configs_path)

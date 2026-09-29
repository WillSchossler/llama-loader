
import tomllib
from pathlib import Path

import pytest

from llama_loader.configs import Configs
from llama_loader import configs as configs_module


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

    configs = Configs(configs_path)

    with pytest.raises(ValueError, match="Field 'root' is not defined in configs.toml"):
        _ = configs.root


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

    configs = Configs(configs_path)

    with pytest.raises(TypeError, match="Field 'editor' must be a string"):
        _ = configs.editor


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

    configs = Configs(configs_path)

    with pytest.raises(ValueError, match="Field 'editor' cannot be empty"):
        _ = configs.editor


def test_configs_raises_for_nonexistent_root_directory(tmp_path):
    nonexistent_root = tmp_path / "does-not-exist"

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{nonexistent_root}'
        editor = "code.cmd"
        """
    )

    configs = Configs(configs_path)

    with pytest.raises(ValueError, match="Field 'root' must point to an existing directory"):
        _ = configs.root


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

    configs = Configs(configs_path)

    with pytest.raises(ValueError, match="Field 'browser_path' must point to an existing file"):
        _ = configs.browser_path


def test_invalid_browser_does_not_block_root(tmp_path: Path):
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

    configs = Configs(configs_path)

    assert configs.root == models_path


def test_invalid_root_does_not_block_editor(tmp_path: Path):
    nonexistent_root = tmp_path / "does-not-exist"

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{nonexistent_root}'
        editor = "code.cmd"
        """
    )

    configs = Configs(configs_path)

    assert configs.editor == "code.cmd"


def test_configs_does_not_parse_toml_until_field_is_accessed(tmp_path: Path):
    configs_path = tmp_path / "configs.toml"
    configs_path.write_text("[broken", encoding="utf-8")

    configs = Configs(configs_path)

    with pytest.raises(tomllib.TOMLDecodeError):
        _ = configs.root


def test_configs_returns_configured_editor(tmp_path: Path):
    configs_path, _, _ = create_configs_file(tmp_path)

    configs = Configs(configs_path)

    assert configs.editor == "code.cmd"


@pytest.mark.parametrize(
    ("os_name", "platform", "expected_editor"),
    [
        pytest.param("nt", "win32", "notepad", id="windows"),
        pytest.param("posix", "darwin", "open", id="macos"),
        pytest.param("posix", "linux", "xdg-open", id="linux"),
    ],
)
def test_configs_uses_platform_editor_when_editor_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
    platform: str,
    expected_editor: str,
):
    configs_path, _, _ = create_configs_file(tmp_path, include_editor=False)

    monkeypatch.setattr(configs_module.os, "name", os_name)
    monkeypatch.setattr(configs_module.sys, "platform", platform)

    configs = Configs(configs_path)

    assert configs.editor == expected_editor


def test_configs_raises_for_invalid_editor_type(tmp_path: Path):
    models_path = tmp_path / "model"
    models_path.mkdir()

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{models_path}'
        editor = 123
        """,
        encoding="utf-8",
    )

    configs = Configs(configs_path)

    with pytest.raises(TypeError, match="Field 'editor' must be a string"):
        _ = configs.editor

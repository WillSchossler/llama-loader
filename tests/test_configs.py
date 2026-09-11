from llama_loader.configs import Configs


def create_valid_configs_file(tmp_path):
    models_path = tmp_path / "models"
    models_path.mkdir()

    browser_path = tmp_path / "browser.exe"
    browser_path.touch()

    configs_path = tmp_path / "configs.toml"
    configs_path.write_text(
        f"""
        root = '{models_path}'
        editor = 'code'
        browser_path = '{browser_path}'
        """
    )

    return configs_path, models_path, browser_path


def test_configs_loads_valid_configuration(tmp_path):
    configs_path, models_path, browser_path = create_valid_configs_file(tmp_path)

    configs = Configs(configs_path)

    assert configs.root == models_path
    assert configs.editor == "code"
    assert configs.browser_path == browser_path


def test_configs_returns_browser(tmp_path):
    configs_path, _, browser_path = create_valid_configs_file(tmp_path)

    configs = Configs(configs_path)

    assert configs.require_browser() == browser_path
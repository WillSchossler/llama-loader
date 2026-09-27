from pathlib import Path

import pytest

from llama_loader.loader import Loader

from .helpers import Helper


def test_roll_without_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["list"])

    helper.create_model()

    loader = Loader(args)
    loader.roll(models_only=False, profiles_only=False)

    captured = capsys.readouterr().out
    models_section, profiles_section = captured.split("Profiles:", maxsplit=1)

    assert "Name: qwen" in models_section
    assert "balanced" in profiles_section
    assert "default" not in profiles_section


def test_roll_models_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["list", "-m"])

    helper.create_model()

    loader = Loader(args)
    loader.roll(models_only=True, profiles_only=False)

    captured = capsys.readouterr().out

    assert "Name: qwen" in captured
    assert "balanced" not in captured


def test_roll_profiles_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["list", "-p"])

    helper.create_model()

    loader = Loader(args)
    loader.roll(models_only=False, profiles_only=True)

    output = capsys.readouterr().out

    assert "balanced" in output
    assert "qwen" not in output
    assert "default" not in output, "Default profile should not be listed"


def test_roll_profiles_only_ignores_invalid_existing_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["list", "-p"])

    qwen = helper.create_model()
    (qwen["model_dir"] / "model.gguf").unlink()

    loader = Loader(args)
    loader.roll(models_only=False, profiles_only=True)

    output = capsys.readouterr().out

    assert "balanced" in output
    assert "default" not in output

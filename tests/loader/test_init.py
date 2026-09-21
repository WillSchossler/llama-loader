import tomllib
from pathlib import Path

import pytest

from llama_loader.loader import Loader

from .helpers import Helper


def test_init_creates_model_toml_from_detected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["init"])

    qwen = helper.create_model()
    model_file: Path = qwen["model_file"]
    model_dir: Path = qwen["model_dir"]

    # create_model() also creates the TOML, but init() must create it in this test
    model_file.unlink()

    loader = Loader(args)
    loader.init(cwd=model_dir)

    with model_file.open("rb") as file:
        model_toml = tomllib.load(file)

    assert model_toml["name"] == "qwen"
    assert model_toml["profile"] == "default"

    assert model_toml["files"]["--model"] == "model.gguf"
    assert model_toml["files"]["--mmproj"] == "mmproj.gguf"
    assert model_toml["files"]["--model-draft"] == "mtp.gguf"
    assert model_toml["files"]["--chat-template-file"] == "template.jinja"

    assert model_toml["parameters"]["--spec-type"] == "ngram-mod,draft-mtp"
    assert model_toml["parameters"]["--fit"] == "on"
    assert model_toml["parameters"]["--jinja"] == ""


def test_init_does_not_overwrite_existing_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["init"])

    qwen = helper.create_model()
    model_file: Path = qwen["model_file"]
    model_dir: Path = qwen["model_dir"]

    original_content = 'name = "DO-NOT-OVERWRITE"\n'
    model_file.write_text(original_content, encoding="utf-8")

    loader = Loader(args)

    with pytest.raises(SystemExit, match="already exists"):
        loader.init(cwd=model_dir)

    assert model_file.read_text(encoding="utf-8") == original_content


def test_init_does_not_guess_model_when_multiple_candidates_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["init"])

    qwen = helper.create_model()
    model_file: Path = qwen["model_file"]
    model_dir: Path = qwen["model_dir"]

    model_file.unlink()

    additional_candidate = model_dir / "model_2.gguf"
    additional_candidate.touch()

    loader = Loader(args)
    loader.init(cwd=model_dir)

    with model_file.open("rb") as file:
        model_toml = tomllib.load(file)

    assert model_toml["files"]["--model"] == "DEFINE_MODEL_PATH"


def test_init_ignores_unrecognized_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["init"])

    qwen = helper.create_model()
    model_file: Path = qwen["model_file"]
    model_dir: Path = qwen["model_dir"]

    model_file.unlink()

    unrecognized_files = [
        model_dir / "README.md",
        model_dir / "notes.txt",
        model_dir / "potato.png",
    ]

    for file in unrecognized_files:
        file.touch()

    loader = Loader(args)
    loader.init(cwd=model_dir)

    with model_file.open("rb") as file:
        model_toml = tomllib.load(file)

    generated_files = model_toml["files"].values()

    for file in unrecognized_files:
        assert file.name not in generated_files
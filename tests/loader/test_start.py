import pytest

from pathlib import Path

from .helpers import Helper
from unittest.mock import MagicMock, call
from llama_loader.loader import Loader


def test_start_raises_when_model_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", "potato"])
    qwen = helper.create_model()

    loader = Loader(args)

    with pytest.raises(ValueError, match="Unknown model: potato"):
        loader.start(
            model_name=args.model,
            llama_args=args.llamaargs,
            open_browser=args.b,
            incognito=args.i
        )

def test_start_with_profile_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = Helper(tmp_path, monkeypatch)
    args = helper.create_cli_args(["start", "qwen"])
    qwen = helper.create_model()

    loader = Loader(args)

    run_server_mock = MagicMock()
    monkeypatch.setattr(Loader, "_run_server", run_server_mock)

    loader.start(
            model_name=args.model,
            llama_args=args.llamaargs,
            open_browser=args.b,
            incognito=args.i
        )
    
    print(run_server_mock.mock_calls[0][1])
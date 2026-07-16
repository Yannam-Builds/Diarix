from pathlib import Path

import pytest

from backend.routes import models as model_routes


def test_cache_usage_separates_models_from_disposable_files(tmp_path: Path):
    model = tmp_path / "models--example--model" / "blobs" / "weights.safetensors"
    incomplete = tmp_path / "models--example--model" / "blobs" / "weights.incomplete"
    lock = tmp_path / ".locks" / "models--example--model" / "download.lock"
    for path, content in ((model, b"model"), (incomplete, b"partial"), (lock, b"lock")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    usage = model_routes._get_cache_usage(tmp_path)

    assert usage == {
        "total_bytes": 16,
        "model_bytes": 5,
        "temporary_bytes": 11,
        "incomplete_files": 1,
    }


@pytest.mark.asyncio
async def test_cache_cleanup_preserves_completed_model_files(tmp_path: Path, monkeypatch):
    from huggingface_hub import constants as hf_constants

    model = tmp_path / "models--example--model" / "blobs" / "weights.safetensors"
    incomplete = tmp_path / "models--example--model" / "blobs" / "weights.incomplete"
    lock = tmp_path / ".locks" / "models--example--model" / "download.lock"
    for path, content in ((model, b"model"), (incomplete, b"partial"), (lock, b"lock")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    class NoDownloads:
        @staticmethod
        def get_active_downloads():
            return []

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: NoDownloads())

    result = await model_routes.cleanup_models_cache()

    assert model.read_bytes() == b"model"
    assert not incomplete.exists()
    assert not lock.exists()
    assert result["removed_files"] == 2
    assert result["freed_bytes"] == 11
    assert result["model_bytes"] == 5
    assert result["temporary_bytes"] == 0


@pytest.mark.asyncio
async def test_cache_cleanup_ignores_dismissible_failed_downloads(
    tmp_path: Path, monkeypatch
):
    from huggingface_hub import constants as hf_constants

    incomplete = tmp_path / "models--example--model" / "blobs" / "weights.incomplete"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"partial")

    class FailedDownload:
        status = "error"

    class FailedDownloads:
        @staticmethod
        def get_active_downloads():
            return [FailedDownload()]

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: FailedDownloads())

    result = await model_routes.cleanup_models_cache()

    assert result["removed_files"] == 1
    assert not incomplete.exists()


@pytest.mark.asyncio
async def test_cache_cleanup_refuses_live_downloads(tmp_path: Path, monkeypatch):
    from fastapi import HTTPException
    from huggingface_hub import constants as hf_constants

    class LiveDownload:
        status = "downloading"

    class LiveDownloads:
        @staticmethod
        def get_active_downloads():
            return [LiveDownload()]

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: LiveDownloads())

    with pytest.raises(HTTPException) as error:
        await model_routes.cleanup_models_cache()

    assert error.value.status_code == 409

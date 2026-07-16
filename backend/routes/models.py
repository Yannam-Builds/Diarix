"""Model management endpoints."""

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import models
from ..utils.platform_detect import get_backend_type
from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

router = APIRouter()


def _shares_cache_with_map(configs) -> dict[str, list[str]]:
    """Group model configs by hf_repo_id so the UI can label shared-checkpoint
    entries instead of showing them as unrelated duplicates (e.g. WhisperX
    Large v3 and Faster-Whisper Large v3 both resolve to the same CT2
    checkpoint). Works for any future repo collision, not just this pair."""
    by_repo: dict[str, list] = {}
    for cfg in configs:
        if not cfg.hf_repo_id:
            continue
        by_repo.setdefault(cfg.hf_repo_id, []).append(cfg)

    result: dict[str, list[str]] = {}
    for cfgs in by_repo.values():
        if len(cfgs) < 2:
            continue
        for cfg in cfgs:
            result[cfg.model_name] = [
                other.display_name for other in cfgs if other.model_name != cfg.model_name
            ]
    return result


@router.get("/models/catalog", response_model=models.ModelStatusListResponse)
async def get_model_catalog():
    """Return model metadata immediately without scanning disk or loading runtimes."""
    from ..backends import get_all_model_configs

    all_configs = get_all_model_configs()
    shares_map = _shares_cache_with_map(all_configs)

    return models.ModelStatusListResponse(
        models=[
            models.ModelStatus(
                model_name=config.model_name,
                display_name=config.display_name,
                model_size=config.model_size,
                hf_repo_id=config.hf_repo_id,
                downloaded=False,
                size_mb=float(config.size_mb) or None,
                engine=config.engine,
                modality=config.modality,
                runtime_group=config.runtime_group,
                capabilities=config.capabilities,
                languages=config.languages,
                description=config.description,
                precision_options=config.precision_options,
                default_precision=config.default_precision,
                recommended=config.recommended,
                min_vram_gb=config.min_vram_gb,
                audio_input=config.audio_input,
                audio_sample_rate=(config.audio_input.sample_rate_hz if config.audio_input else None),
                audio_channels=(config.audio_input.channels if config.audio_input else None),
                audio_format=(config.audio_input.container if config.audio_input else None),
                shares_cache_with=shares_map.get(config.model_name, []),
            )
            for config in all_configs
        ]
    )


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _get_cache_usage(cache_dir: Path) -> dict[str, int]:
    """Measure durable model data separately from disposable download state."""
    model_bytes = 0
    temporary_bytes = 0
    incomplete_files = 0

    if not cache_dir.exists():
        return {
            "total_bytes": 0,
            "model_bytes": 0,
            "temporary_bytes": 0,
            "incomplete_files": 0,
        }

    for item in cache_dir.rglob("*"):
        if not item.is_file():
            continue
        try:
            size = item.stat().st_size
        except OSError:
            continue
        is_temporary = item.name.endswith(".incomplete") or ".locks" in item.parts
        if item.name.endswith(".incomplete"):
            incomplete_files += 1
        if is_temporary:
            temporary_bytes += size
        else:
            model_bytes += size

    return {
        "total_bytes": model_bytes + temporary_bytes,
        "model_bytes": model_bytes,
        "temporary_bytes": temporary_bytes,
        "incomplete_files": incomplete_files,
    }


def _copy_with_progress(src: Path, dst: Path, progress_manager, copied_so_far: int, total_bytes: int) -> int:
    """Copy a directory tree with byte-level progress tracking."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_dir():
            copied_so_far = _copy_with_progress(item, dest_item, progress_manager, copied_so_far, total_bytes)
        else:
            size = item.stat().st_size
            shutil.copy2(str(item), str(dest_item))
            copied_so_far += size
            progress_manager.update_progress(
                "migration",
                copied_so_far,
                total_bytes,
                filename=item.name,
                status="downloading",
            )
    return copied_so_far


@router.post("/models/load")
async def load_model(model_size: str = "1.7B"):
    """Manually load TTS model."""
    from ..services import tts

    try:
        tts_model = tts.get_tts_model()
        await tts_model.load_model_async(model_size)
        return {"message": f"Model {model_size} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
async def unload_model():
    """Unload the default Qwen TTS model to free memory."""
    from ..services import tts

    try:
        tts.unload_tts_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/unload")
async def unload_model_by_name(model_name: str):
    """Unload a specific model from memory without deleting it from disk."""
    from ..backends import get_model_config, unload_model_by_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        was_loaded = unload_model_by_config(config)
        if not was_loaded:
            return {"message": f"Model {model_name} is not loaded"}
        return {"message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/progress/{model_name}")
async def get_model_progress(model_name: str):
    """Get model download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe(model_name):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models/cache-dir")
async def get_models_cache_dir():
    """Get the shared model/cache path and its current disk usage."""
    from huggingface_hub import constants as hf_constants

    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    return {"path": str(cache_dir), **await asyncio.to_thread(_get_cache_usage, cache_dir)}


@router.post("/models/cache/cleanup")
async def cleanup_models_cache():
    """Remove interrupted downloads and stale lock files, never model weights."""
    from huggingface_hub import constants as hf_constants

    task_manager = get_task_manager()
    active_downloads = [
        task
        for task in task_manager.get_active_downloads()
        if task.status in {"downloading", "extracting"}
    ]
    if active_downloads:
        raise HTTPException(
            status_code=409,
            detail="Wait for active model downloads to finish or cancel them before cleaning the cache.",
        )

    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    removed_files = 0
    freed_bytes = 0
    candidates: list[Path] = []
    if cache_dir.exists():
        candidates.extend(cache_dir.rglob("*.incomplete"))
        locks_dir = cache_dir / ".locks"
        if locks_dir.exists():
            candidates.extend(item for item in locks_dir.rglob("*") if item.is_file())

    for item in candidates:
        try:
            freed_bytes += item.stat().st_size
            item.unlink()
            removed_files += 1
        except OSError:
            continue

    if cache_dir.exists():
        directories = sorted(
            (item for item in cache_dir.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                pass

    return {
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
        **await asyncio.to_thread(_get_cache_usage, cache_dir),
    }


@router.post("/models/migrate")
async def migrate_models(request: models.ModelMigrateRequest):
    """Move all downloaded models to a new directory with byte-level progress via SSE."""
    from huggingface_hub import constants as hf_constants

    source = Path(hf_constants.HF_HUB_CACHE)
    destination = Path(request.destination)

    if not source.exists():
        raise HTTPException(status_code=404, detail="Current model cache directory not found")

    if source.resolve() == destination.resolve():
        raise HTTPException(status_code=400, detail="Source and destination are the same directory")

    if destination.resolve().is_relative_to(source.resolve()):
        raise HTTPException(status_code=400, detail="Destination cannot be inside the current cache directory")

    progress_manager = get_progress_manager()
    model_dirs = [d for d in source.iterdir() if d.name.startswith("models--") and d.is_dir()]
    if not model_dirs:
        progress_manager.update_progress("migration", 1, 1, status="complete")
        progress_manager.mark_complete("migration")
        return {"moved": 0, "errors": [], "source": str(source), "destination": str(destination)}

    destination.mkdir(parents=True, exist_ok=True)

    same_fs = False
    try:
        same_fs = source.stat().st_dev == destination.stat().st_dev
    except OSError:
        pass

    async def migrate_background():
        moved = 0
        errors = []
        try:
            if same_fs:
                total = len(model_dirs)
                for i, item in enumerate(model_dirs):
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        shutil.move(str(item), str(dest_item))
                        moved += 1
                        progress_manager.update_progress(
                            "migration",
                            i + 1,
                            total,
                            filename=item.name,
                            status="downloading",
                        )
                    except Exception as e:
                        errors.append(f"{item.name}: {str(e)}")
            else:
                total_bytes = sum(_get_dir_size(d) for d in model_dirs)
                progress_manager.update_progress(
                    "migration", 0, total_bytes, filename="Calculating...", status="downloading"
                )

                copied = 0
                for item in model_dirs:
                    dest_item = destination / item.name
                    try:
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        copied = await asyncio.to_thread(
                            _copy_with_progress, item, dest_item, progress_manager, copied, total_bytes
                        )
                        await asyncio.to_thread(shutil.rmtree, str(item))
                        moved += 1
                    except Exception as e:
                        errors.append(f"{item.name}: {str(e)}")

            progress_manager.update_progress("migration", 1, 1, status="complete")
            progress_manager.mark_complete("migration")
        except Exception as e:
            progress_manager.update_progress("migration", 0, 0, status="error")
            progress_manager.mark_error("migration", str(e))

    create_background_task(migrate_background())

    return {"source": str(source), "destination": str(destination)}


@router.get("/models/migrate/progress")
async def get_migration_progress():
    """Get model migration progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("migration"):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models/status", response_model=models.ModelStatusListResponse)
async def get_model_status():
    """Get status of all available models."""
    from huggingface_hub import constants as hf_constants

    backend_type = get_backend_type()
    task_manager = get_task_manager()

    active_download_names = {task.model_name for task in task_manager.get_active_downloads()}

    try:
        from huggingface_hub import scan_cache_dir

        use_scan_cache = True
    except ImportError:
        use_scan_cache = False

    from ..backends import get_all_model_configs, check_model_loaded

    registry_configs = get_all_model_configs()
    shares_map = _shares_cache_with_map(registry_configs)
    model_configs = [
        {
            "model_name": cfg.model_name,
            "display_name": cfg.display_name,
            "hf_repo_id": cfg.hf_repo_id,
            "shares_cache_with": shares_map.get(cfg.model_name, []),
            "model_size": cfg.model_size,
            "estimated_size_mb": cfg.size_mb,
            "engine": cfg.engine,
            "modality": cfg.modality,
            "runtime_group": cfg.runtime_group,
            "capabilities": cfg.capabilities,
            "languages": cfg.languages,
            "description": cfg.description,
            "precision_options": cfg.precision_options,
            "default_precision": cfg.default_precision,
            "recommended": cfg.recommended,
            "min_vram_gb": cfg.min_vram_gb,
            "audio_input": cfg.audio_input,
            "audio_sample_rate": cfg.audio_input.sample_rate_hz if cfg.audio_input else None,
            "audio_channels": cfg.audio_input.channels if cfg.audio_input else None,
            "audio_format": cfg.audio_input.container if cfg.audio_input else None,
            "check_loaded": lambda c=cfg: check_model_loaded(c),
        }
        for cfg in registry_configs
    ]

    model_to_repo = {cfg["model_name"]: cfg["hf_repo_id"] for cfg in model_configs}
    active_download_repos = {model_to_repo.get(name) for name in active_download_names if name in model_to_repo}

    cache_info = None
    if use_scan_cache:
        try:
            cache_info = scan_cache_dir()
        except Exception:
            pass

    statuses = []

    for config in model_configs:
        try:
            downloaded = False
            size_mb = float(config["estimated_size_mb"]) or None
            loaded = False

            if cache_info:
                repo_id = config["hf_repo_id"]
                for repo in cache_info.repos:
                    if repo.repo_id == repo_id:
                        has_model_weights = False
                        for rev in repo.revisions:
                            for f in rev.files:
                                fname = f.file_name.lower()
                                if fname.endswith(
                                    (".safetensors", ".bin", ".pt", ".pth", ".npz", ".nemo")
                                ):
                                    has_model_weights = True
                                    break
                            if has_model_weights:
                                break

                        has_incomplete = False
                        try:
                            cache_dir = hf_constants.HF_HUB_CACHE
                            blobs_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "blobs"
                            if blobs_dir.exists():
                                has_incomplete = any(blobs_dir.glob("*.incomplete"))
                        except Exception:
                            pass

                        if has_model_weights and not has_incomplete:
                            downloaded = True
                            try:
                                total_size = sum(revision.size_on_disk for revision in repo.revisions)
                                size_mb = total_size / (1024 * 1024)
                            except Exception:
                                pass
                        break

            if not downloaded:
                try:
                    cache_dir = hf_constants.HF_HUB_CACHE
                    repo_cache = Path(cache_dir) / ("models--" + config["hf_repo_id"].replace("/", "--"))

                    if repo_cache.exists():
                        blobs_dir = repo_cache / "blobs"
                        has_incomplete = blobs_dir.exists() and any(blobs_dir.glob("*.incomplete"))

                        if not has_incomplete:
                            snapshots_dir = repo_cache / "snapshots"
                            has_model_files = False
                            if snapshots_dir.exists():
                                has_model_files = (
                                    any(snapshots_dir.rglob("*.bin"))
                                    or any(snapshots_dir.rglob("*.safetensors"))
                                    or any(snapshots_dir.rglob("*.pt"))
                                    or any(snapshots_dir.rglob("*.pth"))
                                    or any(snapshots_dir.rglob("*.npz"))
                                    or any(snapshots_dir.rglob("*.nemo"))
                                )

                            if has_model_files:
                                downloaded = True
                                try:
                                    total_size = sum(
                                        f.stat().st_size
                                        for f in repo_cache.rglob("*")
                                        if f.is_file() and not f.name.endswith(".incomplete")
                                    )
                                    size_mb = total_size / (1024 * 1024)
                                except Exception:
                                    pass
                except Exception:
                    pass

            try:
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False

            is_downloading = config["hf_repo_id"] in active_download_repos

            if is_downloading:
                downloaded = False
                size_mb = float(config["estimated_size_mb"]) or None

            statuses.append(
                models.ModelStatus(
                    model_name=config["model_name"],
                    display_name=config["display_name"],
                    model_size=config["model_size"],
                    hf_repo_id=config["hf_repo_id"],
                    downloaded=downloaded,
                    downloading=is_downloading,
                    size_mb=size_mb,
                    loaded=loaded,
                    engine=config["engine"],
                    modality=config["modality"],
                    runtime_group=config["runtime_group"],
                    capabilities=config["capabilities"],
                    languages=config["languages"],
                    description=config["description"],
                    precision_options=config["precision_options"],
                    default_precision=config["default_precision"],
                    recommended=config["recommended"],
                    min_vram_gb=config["min_vram_gb"],
                    audio_input=config["audio_input"],
                    audio_sample_rate=config["audio_sample_rate"],
                    audio_channels=config["audio_channels"],
                    audio_format=config["audio_format"],
                    shares_cache_with=config["shares_cache_with"],
                )
            )
        except Exception:
            try:
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False

            is_downloading = config["hf_repo_id"] in active_download_repos

            statuses.append(
                models.ModelStatus(
                    model_name=config["model_name"],
                    display_name=config["display_name"],
                    model_size=config["model_size"],
                    hf_repo_id=config["hf_repo_id"],
                    downloaded=False,
                    downloading=is_downloading,
                    size_mb=float(config["estimated_size_mb"]) or None,
                    loaded=loaded,
                    engine=config["engine"],
                    modality=config["modality"],
                    runtime_group=config["runtime_group"],
                    capabilities=config["capabilities"],
                    languages=config["languages"],
                    description=config["description"],
                    precision_options=config["precision_options"],
                    default_precision=config["default_precision"],
                    recommended=config["recommended"],
                    min_vram_gb=config["min_vram_gb"],
                    audio_input=config["audio_input"],
                    audio_sample_rate=config["audio_sample_rate"],
                    audio_channels=config["audio_channels"],
                    audio_format=config["audio_format"],
                    shares_cache_with=config["shares_cache_with"],
                )
            )

    return models.ModelStatusListResponse(models=statuses)


@router.post("/models/download")
async def trigger_model_download(request: models.ModelDownloadRequest):
    """Trigger download of a specific model."""
    from ..backends import get_model_config, get_model_load_func

    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    config = get_model_config(request.model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")

    load_func = get_model_load_func(config)

    async def download_in_background():
        try:
            result = load_func()
            if asyncio.iscoroutine(result):
                await result
            progress_manager.mark_complete(request.model_name)
            task_manager.complete_download(request.model_name)
        except Exception as e:
            task_manager.error_download(request.model_name, str(e))

    task_manager.start_download(request.model_name)

    progress_manager.update_progress(
        model_name=request.model_name,
        current=0,
        total=0,
        filename="Connecting to HuggingFace...",
        status="downloading",
    )

    create_background_task(download_in_background())

    return {"message": f"Model {request.model_name} download started"}


@router.post("/models/download/cancel")
async def cancel_model_download(request: models.ModelDownloadRequest):
    """Cancel or dismiss an errored/stale download task."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    removed = task_manager.cancel_download(request.model_name)

    progress_removed = False
    with progress_manager._lock:
        if request.model_name in progress_manager._progress:
            del progress_manager._progress[request.model_name]
            progress_removed = True

    if removed or progress_removed:
        return {"message": f"Download task for {request.model_name} cancelled"}
    return {"message": f"No active task found for {request.model_name}"}


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model from the HuggingFace cache."""
    from huggingface_hub import constants as hf_constants
    from ..backends import get_model_config, unload_model_by_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    hf_repo_id = config.hf_repo_id

    try:
        unload_model_by_config(config)

        cache_dir = hf_constants.HF_HUB_CACHE
        repo_cache_dir = Path(cache_dir) / ("models--" + hf_repo_id.replace("/", "--"))

        if not repo_cache_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in cache")

        try:
            shutil.rmtree(repo_cache_dir)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete model cache directory: {str(e)}")

        return {"message": f"Model {model_name} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

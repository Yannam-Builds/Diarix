"""
STT (Speech-to-Text) module - delegates to backend abstraction layer.
"""


import asyncio
from collections.abc import Callable

from ..backends import (
    STTBackend,
    get_stt_backend,
    get_stt_backend_for_engine,
    resolve_stt_config,
    unload_all_stt_backends,
)


def get_whisper_model() -> STTBackend:
    """
    Get STT backend instance (MLX or PyTorch based on platform).

    Returns:
        STT backend instance
    """
    return get_stt_backend()


def unload_whisper_model():
    """Unload Whisper model to free memory."""
    backend = get_stt_backend()
    backend.unload_model()


def unload_all_stt_models():
    """Unload all instantiated transcription engines."""
    unload_all_stt_backends()


def get_stt_model(model: str | None = None) -> tuple[STTBackend, object]:
    """Resolve a model selection and return its native Voicebox adapter."""
    config = resolve_stt_config(model)
    return get_stt_backend_for_engine(config.engine), config


async def await_stt_operation(operation):
    """Keep non-interruptible model threads alive before honoring cancellation."""
    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception:
            pass
        raise


async def transcribe_audio(
    audio_path: str,
    model: str | None = None,
    language: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    partial_callback: Callable[[str], None] | None = None,
    segments_callback: Callable[[list], None] | None = None,
) -> tuple[str, str]:
    """Transcribe with any registered STT engine and return text plus model id.

    ``should_stop`` is forwarded to adapters that chunk long audio so a
    cancelled batch job stops after the in-flight chunk rather than running
    every remaining chunk before the model becomes eligible to unload.
    ``partial_callback`` receives the accumulated transcript text as each
    chunk completes, for live progress display. ``segments_callback``
    receives the final timestamped segment list from engines that have one.
    """
    backend, config = get_stt_model(model)
    normalized_language = None if not language or language == "auto" else language
    text = await await_stt_operation(
        backend.transcribe(
            audio_path,
            normalized_language,
            config.model_size,
            progress_callback=progress_callback,
            should_stop=should_stop,
            partial_callback=partial_callback,
            segments_callback=segments_callback,
        )
    )
    return text, config.model_name

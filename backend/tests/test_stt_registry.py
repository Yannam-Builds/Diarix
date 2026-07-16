"""Contract tests for the unified Voicebox transcription registry."""

import asyncio
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from backend.backends import (
    QWEN3_ASR_LANGUAGE_CODES,
    WHISPER_LANGUAGE_CODES,
    get_model_load_func,
    get_stt_model_configs,
    resolve_stt_config,
)
from backend.backends.base import is_model_cached
from backend.backends.stt.qwen_backend import QWEN_LANGUAGE_NAMES


def test_legacy_whisper_sizes_still_resolve() -> None:
    for size in ("base", "small", "medium", "large", "turbo"):
        config = resolve_stt_config(size)
        assert config.engine == "whisper"
        assert config.model_size == size


def test_stt_model_names_are_unique_and_explicit() -> None:
    configs = get_stt_model_configs()
    names = [config.model_name for config in configs]
    assert len(names) == len(set(names))
    assert all(config.modality == "stt" for config in configs)
    assert all(config.description for config in configs)
    assert all(config.audio_input is not None for config in configs)
    assert all(config.audio_input.sample_rate_hz == 16_000 for config in configs)
    assert all(config.audio_input.channels == 1 for config in configs)


def test_optional_engines_use_advanced_runtime() -> None:
    configs = get_stt_model_configs()
    optional = [config for config in configs if config.engine != "whisper"]
    assert optional
    assert all(config.runtime_group == "advanced-asr" for config in optional)


def test_whisperx_declares_alignment_capabilities_without_diarization() -> None:
    config = resolve_stt_config("whisperx-large-v3")
    assert "word_timestamps" in config.capabilities
    assert "alignment" in config.capabilities
    assert "diarization" not in config.capabilities


def test_stt_language_metadata_matches_each_model_family() -> None:
    assert resolve_stt_config("whisper-turbo").languages == WHISPER_LANGUAGE_CODES
    assert resolve_stt_config("whisperx-large-v3").languages == WHISPER_LANGUAGE_CODES
    assert resolve_stt_config("whisper-distil-large-v3.5").languages == ["en"]
    assert resolve_stt_config("faster-whisper-tiny").languages == WHISPER_LANGUAGE_CODES
    assert resolve_stt_config("faster-distil-whisper-large-v3").languages == ["en"]
    assert resolve_stt_config("nvidia-canary-180m-flash").languages == ["en", "de", "fr", "es"]
    assert len(resolve_stt_config("nvidia-canary-1b-v2").languages) == 25
    canary_qwen = resolve_stt_config("nvidia-canary-qwen-2.5b")
    assert canary_qwen.languages == ["en"]
    assert canary_qwen.audio_input.sample_rate_hz == 16_000
    assert canary_qwen.audio_input.channels == 1
    assert canary_qwen.audio_input.container == "wav"
    assert resolve_stt_config("qwen3-asr-0.6b").languages == QWEN3_ASR_LANGUAGE_CODES
    assert set(QWEN_LANGUAGE_NAMES) == set(QWEN3_ASR_LANGUAGE_CODES)
    assert resolve_stt_config("ibm-granite-speech-3.3-8b").languages == ["en", "fr", "de", "es", "pt"]


def test_cached_inventory_additions_use_native_adapters() -> None:
    assert resolve_stt_config("faster-whisper-tiny").engine == "faster_whisper"
    assert resolve_stt_config("faster-distil-whisper-large-v3").engine == "faster_whisper"
    assert resolve_stt_config("nvidia-canary-180m-flash").engine == "nemo_asr"
    assert resolve_stt_config("nvidia-canary-qwen-2.5b").engine == "nemo_asr"
    assert resolve_stt_config("nvidia-parakeet-tdt-0.6b-v3").engine == "nemo_asr"


def test_nemo_archive_counts_as_cached_model(monkeypatch, tmp_path: Path) -> None:
    hub_cache = tmp_path / "hub"
    snapshot = hub_cache / "models--nvidia--canary-1b-v2" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "canary-1b-v2.nemo").write_bytes(b"model")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.constants = types.SimpleNamespace(HF_HUB_CACHE=str(hub_cache))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert is_model_cached("nvidia/canary-1b-v2") is True


def test_advanced_weights_use_voicebox_download_operation(monkeypatch) -> None:
    """Advanced weights use the model manager without importing their runtime."""
    calls: list[str] = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    @contextmanager
    def progress(model_name, is_cached):
        calls.append(f"progress:{model_name}:{is_cached}")
        yield

    fake_base = types.ModuleType("backend.backends.base")
    fake_base.is_model_cached = lambda _repo: False
    fake_base.materialize_windows_snapshot_links = lambda path: path
    fake_base.model_load_progress = progress
    monkeypatch.setitem(sys.modules, "backend.backends.base", fake_base)
    config = resolve_stt_config("nvidia-parakeet-tdt-0.6b-v3")
    asyncio.run(get_model_load_func(config)())

    assert calls == [
        "progress:nvidia-parakeet-tdt-0.6b-v3:False",
        {
            "repo_id": "nvidia/parakeet-tdt-0.6b-v3",
            "allow_patterns": ["*.nemo", "*.json", "*.model", "*.txt"],
        },
    ]


def test_model_manager_downloads_weights_without_importing_inference_runtime(monkeypatch) -> None:
    """TTS, core STT, and refinement downloads are runtime-independent."""
    calls: list[object] = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    @contextmanager
    def progress(model_name, is_cached):
        calls.append(f"progress:{model_name}:{is_cached}")
        yield

    fake_base = types.ModuleType("backend.backends.base")
    fake_base.is_model_cached = lambda _repo: False
    fake_base.materialize_windows_snapshot_links = lambda path: path
    fake_base.model_load_progress = progress
    monkeypatch.setitem(sys.modules, "backend.backends.base", fake_base)

    from backend.backends import get_model_config

    model_names = [
        "qwen-custom-voice-1.7B",
        "whisper-turbo",
        "qwen3-0.6b",
    ]
    for model_name in model_names:
        config = get_model_config(model_name)
        assert config is not None
        asyncio.run(get_model_load_func(config)())

    assert calls == [
        "progress:qwen-custom-voice-1.7B:False",
        {"repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"},
        "progress:whisper-turbo:False",
        {"repo_id": "openai/whisper-large-v3-turbo"},
        "progress:qwen3-0.6b:False",
        {"repo_id": "Qwen/Qwen3-0.6B"},
    ]

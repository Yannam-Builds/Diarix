"""Contract tests for the unified Voicebox transcription registry."""

import asyncio
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import numpy as np

from backend.backends import (
    QWEN3_ASR_LANGUAGE_CODES,
    WHISPER_LANGUAGE_CODES,
    get_model_load_func,
    get_stt_model_configs,
    is_model_config_cached,
    resolve_stt_config,
)
from backend.backends.base import is_model_cached
from backend.backends.stt.qwen_backend import QWEN_LANGUAGE_NAMES
from backend.backends.stt.transcribe_cpp_backend import TranscribeCppLiveStream
from backend.services import transcribe


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


def test_shared_downloads_are_explicit_runtime_aliases_not_duplicate_models() -> None:
    configs = get_stt_model_configs()
    by_download: dict[tuple[str, str | None], list] = {}
    for config in configs:
        by_download.setdefault((config.hf_repo_id, config.artifact_filename), []).append(config)

    shared = [group for group in by_download.values() if len(group) > 1]
    assert len(shared) == 1
    assert {config.model_name for config in shared[0]} == {
        "whisperx-large-v3",
        "faster-whisper-large-v3",
    }
    assert len({config.engine for config in shared[0]}) == len(shared[0])


def test_optional_engines_use_advanced_runtime() -> None:
    configs = get_stt_model_configs()
    optional = [config for config in configs if config.engine != "whisper"]
    assert optional
    assert all(config.runtime_group in {"advanced-asr", "native-asr"} for config in optional)
    assert any(config.engine == "transcribe_cpp" for config in optional)


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
    assert resolve_stt_config("moonshine-streaming-tiny-gguf").languages == ["en"]
    assert len(resolve_stt_config("nemotron-3.5-asr-streaming-0.6b-gguf").languages) == 28
    assert resolve_stt_config("sensevoice-small-gguf").languages == ["zh", "yue", "en", "ja", "ko"]
    assert len(resolve_stt_config("voxtral-mini-4b-realtime-gguf").languages) == 13
    assert resolve_stt_config("breeze-asr-25-gguf").languages == ["zh", "en"]
    assert resolve_stt_config("medasr-gguf").languages == ["en"]


def test_handy_catalog_additions_keep_real_capabilities_and_audio_contracts() -> None:
    voxtral = resolve_stt_config("voxtral-mini-4b-realtime-gguf")
    assert {"streaming", "language_detection", "multilingual"} <= set(voxtral.capabilities)
    assert voxtral.artifact_filename == "Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf"

    breeze = resolve_stt_config("breeze-asr-25-gguf")
    assert {"translation", "segment_timestamps", "language_detection"} <= set(
        breeze.capabilities
    )
    assert breeze.artifact_filename == "Breeze-ASR-25-Q5_K_M.gguf"

    medasr = resolve_stt_config("medasr-gguf")
    assert {"token_timestamps", "long_audio"} <= set(medasr.capabilities)
    assert medasr.source_license == "health-ai-developer-foundations"
    assert medasr.artifact_filename == "medasr-Q8_0.gguf"

    for config in (voxtral, breeze, medasr):
        assert config.audio_input.sample_rate_hz == 16_000
        assert config.audio_input.channels == 1
        assert config.audio_input.container == "wav"


def test_native_live_stream_exposes_committed_and_tentative_text() -> None:
    class Update:
        committed_changed = True
        tentative_changed = True
        input_received_ms = 100
        audio_committed_ms = 80
        revision = 2
        is_final = False

    class Text:
        full = "hello world"
        committed = "hello "
        tentative = "world"

    class FakeStream:
        def feed(self, pcm):
            assert pcm.dtype == np.float32
            assert pcm.shape == (1600,)
            return Update()

        def finalize(self):
            update = Update()
            update.is_final = True
            return update

        def text(self):
            return Text()

    class StreamContext:
        def __init__(self):
            self.stream = FakeStream()

        def __enter__(self):
            return self.stream

        def __exit__(self, *_args):
            return None

    class FakeSession:
        def __init__(self):
            self.cancelled = False
            self.closed = False

        def stream(self, **kwargs):
            assert kwargs == {"language": "en", "timestamps": "none"}
            return StreamContext()

        def cancel(self):
            self.cancelled = True

        def close(self):
            self.closed = True

    session = FakeSession()

    class FakeModel:
        def session(self):
            return session

    live = TranscribeCppLiveStream(FakeModel(), "en")
    partial = live.feed(np.zeros(1600, dtype=np.float64))
    assert partial["full"] == "hello world"
    assert partial["committed"] == "hello "
    assert partial["tentative"] == "world"
    assert partial["revision"] == 2
    assert partial["is_final"] is False

    final = live.finalize()
    assert final["is_final"] is True
    live.cancel()
    live.close()
    assert session.cancelled is True
    assert session.closed is True


def test_stt_language_contract_is_shared_by_batch_and_dictation() -> None:
    whisper = resolve_stt_config("whisper-turbo")
    assert transcribe.resolve_stt_language(whisper, "auto") == "auto"

    english_only = resolve_stt_config("moonshine-streaming-tiny-gguf")
    assert transcribe.resolve_stt_language(english_only, None) == "en"

    explicit_multilingual = resolve_stt_config("nvidia-canary-180m-flash")
    with pytest.raises(ValueError, match="requires an explicit language"):
        transcribe.resolve_stt_language(explicit_multilingual, "auto")
    with pytest.raises(ValueError, match="not supported"):
        transcribe.resolve_stt_language(explicit_multilingual, "ja")


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


def test_native_gguf_requires_its_exact_quantized_artifact(monkeypatch, tmp_path: Path) -> None:
    config = resolve_stt_config("moonshine-streaming-tiny-gguf")
    hub_cache = tmp_path / "hub"
    snapshot = (
        hub_cache
        / "models--handy-computer--moonshine-streaming-tiny-gguf"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "moonshine-streaming-tiny-Q4_K_M.gguf").write_bytes(b"other quant")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.constants = types.SimpleNamespace(HF_HUB_CACHE=str(hub_cache))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert is_model_config_cached(config) is False

    (snapshot / config.artifact_filename).write_bytes(b"configured quant")
    assert is_model_config_cached(config) is True


def test_native_gguf_download_fetches_only_the_configured_file(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []
    artifact = tmp_path / "moonshine-streaming-tiny-Q8_0.gguf"
    artifact.write_bytes(b"gguf")
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kwargs: calls.append(kwargs) or str(artifact)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    @contextmanager
    def progress(model_name, is_cached):
        calls.append(f"progress:{model_name}:{is_cached}")
        yield

    fake_base = types.ModuleType("backend.backends.base")
    fake_base.is_model_cached = lambda _repo, **_kwargs: False
    fake_base.materialize_windows_snapshot_links = lambda path: calls.append(("materialize", path))
    fake_base.model_load_progress = progress
    monkeypatch.setitem(sys.modules, "backend.backends.base", fake_base)

    config = resolve_stt_config("moonshine-streaming-tiny-gguf")
    asyncio.run(get_model_load_func(config)())

    assert calls == [
        "progress:moonshine-streaming-tiny-gguf:False",
        {
            "repo_id": "handy-computer/moonshine-streaming-tiny-gguf",
            "filename": "moonshine-streaming-tiny-Q8_0.gguf",
            "local_files_only": False,
        },
        ("materialize", tmp_path),
    ]


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

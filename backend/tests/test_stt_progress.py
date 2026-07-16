"""Model-free tests for native and duration-based STT progress adapters."""

import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

from backend.backends.stt.faster_whisper_backend import FasterWhisperSTTBackend
from backend.backends.stt.nemo_backend import NeMoASRBackend
from backend.backends.stt.qwen_backend import QwenASRBackend
from backend.backends.stt.transformers_backend import TransformersASRBackend
from backend.backends.stt.whisperx_backend import WhisperXSTTBackend


def _write_silent_wav(path: Path, duration_seconds: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16_000 * duration_seconds)


def test_faster_whisper_reports_completed_segment_timeline(tmp_path: Path) -> None:
    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return iter(
                [
                    SimpleNamespace(text="first", end=20.0),
                    SimpleNamespace(text="second", end=75.0),
                ]
            ), SimpleNamespace(duration=100.0)

    backend = FasterWhisperSTTBackend()
    backend.model = FakeModel()
    backend.model_size = "faster-whisper-tiny"
    progress: list[float] = []

    text = asyncio.run(
        backend.transcribe(
            str(tmp_path / "audio.wav"),
            model_size="faster-whisper-tiny",
            progress_callback=progress.append,
        )
    )

    assert text == "first\nsecond"
    assert progress == [0.0, 0.2, 0.75, 1.0]


def test_whisperx_maps_transcription_and_alignment_progress(monkeypatch) -> None:
    class FakeModel:
        def transcribe(self, _audio, **kwargs):
            kwargs["progress_callback"](25.0)
            kwargs["progress_callback"](100.0)
            return {"language": "en", "segments": [{"text": "hello"}]}

    fake_whisperx = SimpleNamespace(
        load_audio=lambda _path: object(),
        load_align_model=lambda *_args: (object(), object()),
        align=lambda segments, *_args, **kwargs: (
            kwargs["progress_callback"](50.0) or kwargs["progress_callback"](100.0) or {"segments": segments}
        ),
    )
    monkeypatch.setattr(
        "backend.backends.stt.whisperx_backend.require_import",
        lambda *_args: fake_whisperx,
    )
    backend = WhisperXSTTBackend()
    backend.model = FakeModel()
    backend.model_size = "large-v3"
    progress: list[float] = []

    text = asyncio.run(
        backend.transcribe(
            "audio.wav",
            model_size="large-v3",
            progress_callback=progress.append,
        )
    )

    assert text == "hello"
    assert progress == [0.0, 0.2, 0.8, 0.9, 1.0, 1.0]


def test_qwen_reports_completed_audio_chunks(tmp_path: Path) -> None:
    source = tmp_path / "qwen.wav"
    _write_silent_wav(source, 125)

    class FakeModel:
        def transcribe(self, **_kwargs):
            return SimpleNamespace(text="a useful chunk")

    backend = QwenASRBackend()
    backend.model = FakeModel()
    backend.model_size = "qwen3-asr-0.6b"
    progress: list[float] = []

    text = asyncio.run(
        backend.transcribe(
            str(source),
            model_size="qwen3-asr-0.6b",
            progress_callback=progress.append,
        )
    )

    assert text == "a useful chunk"
    assert progress[0] == 0.0
    assert len(progress) == 4
    assert 0.45 < progress[1] < 0.5
    assert progress[-1] == 1.0


def test_nemo_reports_completed_audio_chunks(tmp_path: Path) -> None:
    source = tmp_path / "nemo.wav"
    _write_silent_wav(source, 125)

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return [SimpleNamespace(text="a useful chunk")]

    backend = NeMoASRBackend()
    backend.model = FakeModel()
    backend.model_size = "nvidia-parakeet-tdt-0.6b-v3"
    progress: list[float] = []

    text = asyncio.run(
        backend.transcribe(
            str(source),
            model_size="nvidia-parakeet-tdt-0.6b-v3",
            progress_callback=progress.append,
        )
    )

    assert text == "a useful chunk"
    assert progress[0] == 0.0
    assert len(progress) == 4
    assert progress[-1] == 1.0


def test_canary_qwen_uses_official_prompt_and_non_overlapping_chunks(tmp_path: Path) -> None:
    source = tmp_path / "canary-qwen.wav"
    _write_silent_wav(source, 85)
    generated_prompts: list[list[list[dict]]] = []

    class FakeTokenIds:
        def cpu(self):
            return self

    class FakeTokenizer:
        def ids_to_text(self, _token_ids):
            return "a useful chunk"

    class FakeModel:
        audio_locator_tag = "<|audioplaceholder|>"
        tokenizer = FakeTokenizer()

        def generate(self, **kwargs):
            generated_prompts.append(kwargs["prompts"])
            assert kwargs["max_new_tokens"] == 512
            return [FakeTokenIds()]

    backend = NeMoASRBackend()
    backend.model = FakeModel()
    backend.model_size = "nvidia-canary-qwen-2.5b"
    progress: list[float] = []

    text = asyncio.run(
        backend.transcribe(
            str(source),
            language="en",
            model_size="nvidia-canary-qwen-2.5b",
            progress_callback=progress.append,
        )
    )

    assert text == "a useful chunk"
    assert len(generated_prompts) == 3
    assert generated_prompts[0][0][0]["content"] == ("Transcribe the following: <|audioplaceholder|>")
    assert generated_prompts[0][0][0]["audio"][0].endswith("audio-0000.wav")
    assert progress == [0.0, 40 / 85, 80 / 85, 1.0]


def test_transformers_asr_reports_completed_audio_chunks(tmp_path: Path) -> None:
    source = tmp_path / "granite.wav"
    _write_silent_wav(source, 65)
    backend = TransformersASRBackend()
    backend.pipeline = lambda *_args, **_kwargs: {"text": "a useful chunk"}
    backend.model_size = "ibm-granite-speech-3.3-8b"
    progress: list[float] = []

    text = asyncio.run(
        backend.transcribe(
            str(source),
            model_size="ibm-granite-speech-3.3-8b",
            progress_callback=progress.append,
        )
    )

    assert text == "a useful chunk"
    assert progress[0] == 0.0
    assert len(progress) == 4
    assert progress[-1] == 1.0

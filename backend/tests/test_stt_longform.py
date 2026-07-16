"""Model-free regression tests for complete long-form transcription."""

import asyncio
import json
import wave
from pathlib import Path

import numpy as np

from backend.backends.pytorch_backend import PyTorchSTTBackend
from backend.backends.stt.nemo_backend import (
    _canary_audio_chunks,
    _clean_canary_text,
    _merge_overlapping_text,
    _write_canary_manifest,
)


def test_core_whisper_uses_native_longform_generation(monkeypatch, tmp_path: Path) -> None:
    processor_calls: list[dict] = []
    generation_calls: list[dict] = []

    class FakeTensor:
        def to(self, *_args, **_kwargs):
            return self

    class FakeProcessor:
        def __call__(self, audio, **kwargs):
            processor_calls.append({"samples": len(audio), **kwargs})
            return {"input_features": FakeTensor(), "attention_mask": FakeTensor()}

        def batch_decode(self, _predicted_ids, **_kwargs):
            return ["opening sentence middle sentence final sentence"]

    class FakeModel:
        dtype = "float16"

        def generate(self, **kwargs):
            generation_calls.append(kwargs)
            kwargs["monitor_progress"](np.array([[1500, 3000]]))
            return object()

    backend = PyTorchSTTBackend(model_size="turbo")
    backend.model = FakeModel()
    backend.processor = FakeProcessor()
    monkeypatch.setattr(
        "backend.backends.pytorch_backend.load_audio",
        lambda *_args, **_kwargs: (np.zeros(16_000 * 95, dtype=np.float32), 16_000),
    )

    progress: list[float] = []
    text = asyncio.run(
        backend.transcribe(
            str(tmp_path / "long.wav"),
            "en",
            "turbo",
            progress_callback=progress.append,
        )
    )

    assert text.endswith("final sentence")
    assert processor_calls == [
        {
            "samples": 16_000 * 95,
            "sampling_rate": 16_000,
            "return_tensors": "pt",
            "return_attention_mask": True,
            "truncation": False,
            "padding": "max_length",
        }
    ]
    assert generation_calls[0]["return_timestamps"] is True
    assert generation_calls[0]["task"] == "transcribe"
    assert generation_calls[0]["language"] == "en"
    assert generation_calls[0]["monitor_progress"]
    assert progress == [0.0, 0.5, 1.0]


def test_canary_chunks_long_pcm_wav_below_model_limit(tmp_path: Path) -> None:
    source = tmp_path / "long.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16_000 * 65)

    chunks = _canary_audio_chunks(source, tmp_path / "chunks")

    assert len(chunks) == 3
    for chunk in chunks:
        with wave.open(str(chunk), "rb") as audio:
            assert audio.getnframes() / audio.getframerate() <= 30.0


def test_canary_manifest_and_overlap_stitching_are_explicit(tmp_path: Path) -> None:
    audio_path = tmp_path / "chunk.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16_000)
    manifest_path = tmp_path / "chunk.jsonl"
    _write_canary_manifest(manifest_path, audio_path, "de")

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert record["duration"] > 0
    assert record["source_lang"] == "de"
    assert record["target_lang"] == "de"
    assert record["taskname"] == "asr"
    assert record["pnc"] == "yes"
    assert _merge_overlapping_text(
        "one two three four", "three four five six"
    ) == "one two three four five six"
    assert _clean_canary_text("<|endoftext|> hello world.") == "hello world."

"""Focused, model-free tests for centralized FFmpeg media ingestion."""

import asyncio
import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from backend.backends import DEFAULT_STT_AUDIO_INPUT
from backend.services.media_ingestion import (
    AudioStreamNotFoundError,
    build_normalize_command,
    ingest_media,
    parse_probe_payload,
    probe_media,
    resolve_media_tool,
)


def _probe_json(*, default_index: int = 2) -> str:
    return json.dumps(
        {
            "format": {"format_name": "mov,mp4,m4a", "duration": "4.25"},
            "streams": [
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_fmt": "fltp",
                    "sample_rate": "48000",
                    "channels": 2,
                    "duration": "4.2",
                    "disposition": {"default": 0},
                },
                {
                    "index": default_index,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_fmt": "fltp",
                    "sample_rate": "44100",
                    "channels": 1,
                    "duration": "4.0",
                    "disposition": {"default": 1},
                    "tags": {"language": "en"},
                },
            ],
        }
    )


def test_probe_parser_selects_default_audio_stream(tmp_path: Path) -> None:
    parsed = parse_probe_payload(tmp_path / "meeting.mp4", _probe_json())
    assert parsed.selected_stream.index == 2
    assert parsed.selected_stream.language == "en"
    assert parsed.duration == 4.25
    assert len(parsed.audio_streams) == 2


def test_probe_parser_rejects_media_without_audio(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "format": {"format_name": "mp4", "duration": "1"},
            "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}],
        }
    )
    with pytest.raises(AudioStreamNotFoundError) as error:
        parse_probe_payload(tmp_path / "silent.mp4", payload)
    assert error.value.code == "audio_stream_not_found"


def test_normalize_command_maps_audio_and_enforces_model_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"tool")
    monkeypatch.setenv("FFMPEG_BINARY", str(ffmpeg))
    command = build_normalize_command(
        tmp_path / "input.mp4",
        tmp_path / "output.wav",
        3,
        DEFAULT_STT_AUDIO_INPUT,
    )
    assert command[0] == str(ffmpeg.resolve())
    assert command[command.index("-map") + 1] == "0:3"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert "-vn" in command


def test_tool_resolution_prefers_binary_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "custom-ffprobe.exe"
    override.write_bytes(b"tool")
    monkeypatch.setenv("FFPROBE_BINARY", str(override))
    monkeypatch.setenv("PATH", "")
    assert resolve_media_tool("ffprobe") == override.resolve()


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg tooling is not installed",
)
def test_real_mp4_is_normalized_and_job_artifacts_are_cleaned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DIARIX_MEDIA_CACHE_DIR", str(tmp_path / "jobs"))
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:d=0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.25",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )

    async def exercise() -> Path:
        async with ingest_media(source, DEFAULT_STT_AUDIO_INPUT) as normalized:
            assert normalized.converted is True
            assert normalized.audio_path.suffix == ".wav"
            assert normalized.audio_path.exists()
            verified = await probe_media(normalized.audio_path)
            assert verified.selected_stream.codec_name == "pcm_s16le"
            assert verified.selected_stream.sample_rate_hz == 16_000
            assert verified.selected_stream.channels == 1
            assert verified.duration == pytest.approx(0.25, abs=0.08)
            assert normalized.duration == pytest.approx(0.25, abs=0.08)
            return normalized.audio_path

    normalized_path = asyncio.run(exercise())
    assert not normalized_path.exists()
    assert list((tmp_path / "jobs").iterdir()) == []


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="FFprobe is not installed")
def test_matching_pcm_wav_bypasses_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DIARIX_MEDIA_CACHE_DIR", str(tmp_path / "jobs"))
    source = tmp_path / "ready.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 160)

    async def exercise() -> None:
        async with ingest_media(source, DEFAULT_STT_AUDIO_INPUT) as normalized:
            assert normalized.converted is False
            assert normalized.audio_path == source.resolve()

    asyncio.run(exercise())

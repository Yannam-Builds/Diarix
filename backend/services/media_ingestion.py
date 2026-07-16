"""Central FFprobe/FFmpeg media ingestion for every transcription surface.

The ingestion boundary accepts audio or video containers, selects a real audio
stream with FFprobe, and gives STT back a deterministic model-specific file.
Commands are always executed as argv arrays; user paths are never interpreted
by a shell.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from ..backends import AudioInputSpec, DEFAULT_STT_AUDIO_INPUT
from ..config import get_cache_dir


DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
ProgressCallback = Callable[[str, float], Optional[Awaitable[None]]]


class MediaIngestionError(RuntimeError):
    """Base class with a stable machine-readable error code."""

    code = "media_ingestion_failed"

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.detail = detail


class MediaToolUnavailableError(MediaIngestionError):
    code = "media_tool_unavailable"


class MediaProbeError(MediaIngestionError):
    code = "media_probe_failed"


class AudioStreamNotFoundError(MediaIngestionError):
    code = "audio_stream_not_found"


class MediaNormalizationError(MediaIngestionError):
    code = "media_normalization_failed"


class MediaUploadTooLargeError(MediaIngestionError):
    code = "media_upload_too_large"


@dataclass(frozen=True)
class AudioStreamInfo:
    index: int
    codec_name: str
    sample_format: str
    sample_rate_hz: int
    channels: int
    channel_layout: str | None
    duration: float | None
    language: str | None
    is_default: bool


@dataclass(frozen=True)
class MediaProbeInfo:
    source_path: Path
    format_names: tuple[str, ...]
    duration: float | None
    audio_streams: tuple[AudioStreamInfo, ...]
    selected_stream: AudioStreamInfo


@dataclass(frozen=True)
class NormalizedMedia:
    source_path: Path
    audio_path: Path
    probe: MediaProbeInfo
    spec: AudioInputSpec
    converted: bool

    @property
    def duration(self) -> float | None:
        return self.selected_duration

    @property
    def selected_duration(self) -> float | None:
        return self.probe.selected_stream.duration or self.probe.duration


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if parsed < 0 or parsed == float("inf") or parsed != parsed:
        return None
    return parsed


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _tool_candidates(tool: str) -> list[Path]:
    executable_name = f"{tool}.exe" if os.name == "nt" else tool
    candidates: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        root = Path(bundle_root)
        candidates.extend((root / "tools" / executable_name, root / executable_name))

    executable_root = Path(sys.executable).resolve().parent
    candidates.extend(
        (
            executable_root / "tools" / executable_name,
            executable_root / executable_name,
            executable_root.parent / "tools" / executable_name,
        )
    )
    return candidates


def resolve_media_tool(tool: str) -> Path:
    """Resolve FFmpeg tooling from override, app bundle, then PATH."""
    normalized = tool.lower()
    if normalized not in {"ffmpeg", "ffprobe"}:
        raise ValueError(f"Unsupported media tool: {tool}")

    override_names = (
        f"{normalized.upper()}_BINARY",
        f"DIARIX_{normalized.upper()}_PATH",
        f"VOICEBOX_{normalized.upper()}_PATH",
    )
    for variable in override_names:
        raw_path = os.environ.get(variable)
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise MediaToolUnavailableError(
            f"{normalized} was not found at the configured path.",
            detail=f"{variable}={candidate}",
        )

    configured_dir = os.environ.get("DIARIX_FFMPEG_DIR")
    if configured_dir:
        executable_name = f"{normalized}.exe" if os.name == "nt" else normalized
        candidate = (Path(configured_dir).expanduser() / executable_name).resolve()
        if candidate.is_file():
            return candidate
        raise MediaToolUnavailableError(
            f"{normalized} was not found in the configured FFmpeg directory.",
            detail=f"DIARIX_FFMPEG_DIR={configured_dir}",
        )

    for candidate in _tool_candidates(normalized):
        if candidate.is_file():
            return candidate.resolve()

    path_match = shutil.which(normalized)
    if path_match:
        return Path(path_match).resolve()
    raise MediaToolUnavailableError(
        f"{normalized} is required to ingest audio and video media.",
        detail=f"Set DIARIX_{normalized.upper()}_PATH or install {normalized} on PATH.",
    )


def get_media_job_root() -> Path:
    """Return the selected persistent cache root for short-lived media jobs."""
    explicit = os.environ.get("DIARIX_MEDIA_CACHE_DIR")
    selected_cache = os.environ.get("VOICEBOX_MODELS_DIR") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        root = Path(explicit).expanduser().resolve()
    elif selected_cache:
        root = Path(selected_cache).expanduser().resolve() / ".diarix" / "media-jobs"
    else:
        root = get_cache_dir().resolve() / "media-jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_media_job_dir(job_id: str | None = None) -> Path:
    """Create a traversal-safe work directory beneath the media cache root."""
    root = get_media_job_root().resolve()
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", job_id or uuid.uuid4().hex)
    if not safe_id:
        safe_id = uuid.uuid4().hex
    job_dir = (root / safe_id).resolve()
    if root not in job_dir.parents:
        raise MediaIngestionError("Media job path escaped the configured cache root.")
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def cleanup_media_job_dir(job_dir: str | Path) -> None:
    """Remove a media job only when it resolves beneath the selected root."""
    root = get_media_job_root().resolve()
    candidate = Path(job_dir).resolve()
    if candidate == root or root not in candidate.parents:
        raise MediaIngestionError("Refusing to remove a path outside the media job root.")
    shutil.rmtree(candidate, ignore_errors=True)


def sanitize_upload_filename(filename: str | None, fallback: str = "media") -> str:
    """Reduce a browser-supplied filename to one safe local basename."""
    normalized = (filename or "").replace("\\", "/")
    name = Path(normalized).name.strip().strip(".")
    name = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", name)
    return name[:240] or fallback


async def save_upload_to_job(
    upload,
    job_dir: str | Path,
    *,
    index: int = 0,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> tuple[Path, int]:
    """Stream an UploadFile-like object to a bounded job-local file."""
    directory = Path(job_dir).resolve()
    root = get_media_job_root().resolve()
    if directory != root and root not in directory.parents:
        raise MediaIngestionError("Upload destination is outside the media job root.")

    original_name = sanitize_upload_filename(getattr(upload, "filename", None))
    destination = directory / f"{index:04d}-{original_name}"
    total = 0
    try:
        with destination.open("xb") as output:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise MediaUploadTooLargeError(
                        f"{original_name} exceeds the {max_bytes}-byte upload limit."
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(upload, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    if total == 0:
        destination.unlink(missing_ok=True)
        raise MediaProbeError(f"{original_name} is empty.")
    return destination, total


async def _emit_progress(callback: ProgressCallback | None, stage: str, progress: float) -> None:
    if callback is None:
        return
    result = callback(stage, max(0.0, min(100.0, float(progress))))
    if inspect.isawaitable(result):
        await result


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def _run_capture(command: list[str], *, error_type: type[MediaIngestionError]) -> bytes:
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise MediaToolUnavailableError("Unable to start media tooling.", detail=str(exc)) from exc

    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        await _terminate_process(process)
        raise
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise error_type("Media tooling could not read the selected file.", detail=detail[-4000:])
    return stdout


def parse_probe_payload(source_path: str | Path, payload: bytes | str) -> MediaProbeInfo:
    """Parse FFprobe JSON and deterministically choose an audio stream."""
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MediaProbeError("FFprobe returned invalid media metadata.", detail=str(exc)) from exc

    format_data = decoded.get("format") or {}
    format_names = tuple(
        name.strip().lower()
        for name in str(format_data.get("format_name") or "").split(",")
        if name.strip()
    )
    format_duration = _safe_float(format_data.get("duration"))
    streams: list[AudioStreamInfo] = []
    for stream in decoded.get("streams") or []:
        if stream.get("codec_type") not in (None, "audio"):
            continue
        disposition = stream.get("disposition") or {}
        tags = stream.get("tags") or {}
        streams.append(
            AudioStreamInfo(
                index=_safe_int(stream.get("index")),
                codec_name=str(stream.get("codec_name") or "").lower(),
                sample_format=str(stream.get("sample_fmt") or "").lower(),
                sample_rate_hz=_safe_int(stream.get("sample_rate")),
                channels=_safe_int(stream.get("channels")),
                channel_layout=stream.get("channel_layout"),
                duration=_safe_float(stream.get("duration")),
                language=tags.get("language"),
                is_default=bool(_safe_int(disposition.get("default"))),
            )
        )

    if not streams:
        raise AudioStreamNotFoundError("The selected media does not contain an audio stream.")
    selected = next((stream for stream in streams if stream.is_default), streams[0])
    return MediaProbeInfo(
        source_path=Path(source_path).resolve(),
        format_names=format_names,
        duration=format_duration,
        audio_streams=tuple(streams),
        selected_stream=selected,
    )


async def probe_media(source_path: str | Path) -> MediaProbeInfo:
    source = Path(source_path).resolve()
    if not source.is_file():
        raise MediaProbeError("The selected media file no longer exists.")
    command = [
        str(resolve_media_tool("ffprobe")),
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        (
            "format=duration,format_name:"
            "stream=index,codec_type,codec_name,sample_fmt,sample_rate,channels,"
            "channel_layout,duration:stream_tags=language:stream_disposition=default"
        ),
        "-of",
        "json",
        str(source),
    ]
    payload = await _run_capture(command, error_type=MediaProbeError)
    return parse_probe_payload(source, payload)


def media_matches_spec(probe: MediaProbeInfo, spec: AudioInputSpec) -> bool:
    """Return true only when bypassing FFmpeg is demonstrably safe."""
    stream = probe.selected_stream
    expected_formats = {spec.sample_format.lower()}
    if spec.sample_format.lower() == "s16":
        expected_formats.add("s16p")
    return (
        spec.container.lower() in probe.format_names
        and stream.codec_name == spec.codec.lower()
        and stream.sample_format in expected_formats
        and stream.sample_rate_hz == spec.sample_rate_hz
        and stream.channels == spec.channels
    )


def build_normalize_command(
    source_path: str | Path,
    output_path: str | Path,
    stream_index: int,
    spec: AudioInputSpec,
) -> list[str]:
    """Build the deterministic, no-shell FFmpeg normalization command."""
    return [
        str(resolve_media_tool("ffmpeg")),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(Path(source_path).resolve()),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-ac",
        str(spec.channels),
        "-ar",
        str(spec.sample_rate_hz),
        "-sample_fmt",
        spec.sample_format,
        "-c:a",
        spec.codec,
        "-f",
        spec.container,
        "-progress",
        "pipe:1",
        "-nostats",
        str(Path(output_path).resolve()),
    ]


def _progress_seconds(key: str, value: str) -> float | None:
    if key in {"out_time_us", "out_time_ms"}:
        parsed = _safe_float(value)
        return parsed / 1_000_000 if parsed is not None else None
    if key == "out_time":
        match = re.fullmatch(r"(\d+):(\d+):(\d+(?:\.\d+)?)", value)
        if match:
            return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    return None


async def normalize_media(
    source_path: str | Path,
    output_path: str | Path,
    probe: MediaProbeInfo,
    spec: AudioInputSpec,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Normalize one selected audio stream while reporting real FFmpeg progress."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_normalize_command(source_path, output, probe.selected_stream.index, spec)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise MediaToolUnavailableError("Unable to start FFmpeg.", detail=str(exc)) from exc

    duration = probe.selected_stream.duration or probe.duration
    stderr_task = asyncio.create_task(process.stderr.read())
    try:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if "=" not in decoded:
                continue
            key, value = decoded.split("=", 1)
            if key == "progress" and value == "end":
                await _emit_progress(progress_callback, "normalizing", 100.0)
                continue
            seconds = _progress_seconds(key, value)
            if seconds is not None and duration and duration > 0:
                await _emit_progress(progress_callback, "normalizing", seconds / duration * 100.0)
        return_code = await process.wait()
        stderr = await stderr_task
    except asyncio.CancelledError:
        await _terminate_process(process)
        stderr_task.cancel()
        await asyncio.gather(stderr_task, return_exceptions=True)
        output.unlink(missing_ok=True)
        raise

    if return_code != 0 or not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise MediaNormalizationError(
            "FFmpeg could not normalize the selected audio stream.", detail=detail[-4000:]
        )
    return output


@asynccontextmanager
async def ingest_media(
    source_path: str | Path,
    spec: AudioInputSpec = DEFAULT_STT_AUDIO_INPUT,
    *,
    progress_callback: ProgressCallback | None = None,
    job_dir: str | Path | None = None,
    cleanup: bool = True,
):
    """Yield model-ready audio and clean temporary normalization artifacts."""
    own_job = job_dir is None
    work_dir = create_media_job_dir() if own_job else Path(job_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        await _emit_progress(progress_callback, "probing", 0.0)
        probe = await probe_media(source_path)
        await _emit_progress(progress_callback, "probing", 100.0)
        if media_matches_spec(probe, spec):
            yield NormalizedMedia(
                source_path=Path(source_path).resolve(),
                audio_path=Path(source_path).resolve(),
                probe=probe,
                spec=spec,
                converted=False,
            )
            return

        output = work_dir / f"normalized.{spec.container}"
        await normalize_media(
            source_path,
            output,
            probe,
            spec,
            progress_callback=progress_callback,
        )
        yield NormalizedMedia(
            source_path=Path(source_path).resolve(),
            audio_path=output,
            probe=probe,
            spec=spec,
            converted=True,
        )
    finally:
        if cleanup and own_job:
            cleanup_media_job_dir(work_dir)

"""Pure formatters for timestamped transcript exports (SRT/VTT/JSON).

Segments are plain ``{"start": float, "end": float, "text": str}`` dicts —
the shared shape every timestamp-capable STT adapter reports through its
``segments_callback`` (see ``backend/backends/stt/common.py``). Adapters
without a chunk/segment boundary (e.g. WhisperX's own alignment call) simply
never call it, so callers always get ``None`` rather than a partial list.
"""

from __future__ import annotations

SUPPORTED_EXPORT_FORMATS = ("txt", "srt", "vtt", "json")


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = round((seconds - whole) * 1000)
    if millis == 1000:
        whole += 1
        millis = 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    return _format_srt_timestamp(seconds).replace(",", ".")


def segments_to_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = _format_srt_timestamp(float(segment.get("start", 0.0) or 0.0))
        end = _format_srt_timestamp(float(segment.get("end", 0.0) or 0.0))
        lines.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def segments_to_vtt(segments: list[dict]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = _format_vtt_timestamp(float(segment.get("start", 0.0) or 0.0))
        end = _format_vtt_timestamp(float(segment.get("end", 0.0) or 0.0))
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"


def segments_to_json(segments: list[dict]) -> str:
    import json

    payload = [
        {
            "start": float(segment.get("start", 0.0) or 0.0),
            "end": float(segment.get("end", 0.0) or 0.0),
            "text": str(segment.get("text", "")).strip(),
        }
        for segment in segments
        if str(segment.get("text", "")).strip()
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def paragraphs_from_segments(segments: list[dict], *, gap_seconds: float = 1.2) -> str:
    """Join segment text into paragraphs, breaking on silence gaps.

    No diarization, no new model — a pure post-process over the same
    segment timestamps used for SRT/VTT export. A paragraph break is
    inserted whenever the gap between one segment's end and the next
    segment's start exceeds ``gap_seconds``.
    """
    paragraphs: list[list[str]] = []
    previous_end: float | None = None
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if previous_end is not None and (start - previous_end) > gap_seconds:
            paragraphs.append([])
        if not paragraphs:
            paragraphs.append([])
        paragraphs[-1].append(text)
        previous_end = end
    return "\n\n".join(" ".join(part) for part in paragraphs if part)

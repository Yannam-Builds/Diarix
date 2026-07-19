# Changelog

All notable Diarix changes are recorded here. Diarix began as a fork of Voicebox; earlier upstream
history remains available in the [Voicebox repository](https://github.com/jamiepine/voicebox).

## [0.1.0-alpha.1] - Unreleased

### Added

- Transcription-first dashboard with audio, folder, and video import.
- Central FFprobe inspection and FFmpeg normalization for model-specific audio contracts.
- Queued batch transcription with persistence, cancellation, export, progress, and live partial
  transcript updates where supported by the runtime.
- Whisper, Faster-Whisper, WhisperX, NVIDIA NeMo, Qwen3-ASR, and native `transcribe.cpp` adapters.
- Model-specific language selection, shared checkpoint detection, custom cache location, and model
  download management.
- Native global push-to-talk, live dictation, focus restoration, automatic paste, and lightweight
  tray controls in the existing Tauri application.
- Interchangeable compact and CUDA backend distributions using the same API and data directories.
- TTS, profiles, stories, captures, history, and local refinement as separate studio sections.
- User-controlled inference resource guard and automatic model unload policy.

### Fixed

- Long audio and MP4 transcription no longer stops after the first Whisper receptive window.
- Model-backed inference progress replaces the misleading fixed 35% stage value where supported.
- Cancellation reaches chunked adapters and unloads the active model after interruption.
- Cached models and checkpoints shared by multiple adapters report consistent download state.
- Frozen Qwen3-ASR includes Nagisa modules required by its runtime imports.
- Production startup uses the bundled server and does not open a terminal or development URL.

### Known limitations

- Windows x64 is the first supported alpha distribution.
- NVIDIA CUDA is the verified accelerated backend; ROCm and macOS builds are not alpha blockers.
- Some runtimes expose only indeterminate inference progress.
- WhisperX and other blocking adapters cannot cancel until their current native call returns.
- IBM Granite Speech 3.3 8B has not been tested on the current 12 GB GPU.
- Speaker diarization and MCP are intentionally absent from the alpha.

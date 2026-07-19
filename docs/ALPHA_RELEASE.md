# Diarix alpha release

The first public alpha is a Windows x64 release for local transcription, live dictation, voice
generation, and local refinement. It is prerelease software: model compatibility is broad, but the
installer and cold-machine experience still require explicit release-gate verification.

## Alpha 1 scope

- Transcription is the default dashboard.
- Audio and video, including MP4, use centralized FFprobe inspection and FFmpeg normalization.
- Batch transcription supports task state, cancellation, persistence, export, and live partial text
  when the engine exposes it.
- Global push-to-talk uses the native Tauri process and lightweight tray.
- TTS, profiles, stories, captures, history, models, and refinement remain separate integrated
  sections.
- The compact server and the full CUDA server use the same API, data directory, and model catalog.
- Speaker diarization and MCP are not part of the alpha.

## Distribution

| Package | Contents | Alpha sequencing |
|---|---|---|
| Diarix Core Setup | Desktop app, compact server, FFmpeg/FFprobe, empty model cache | Alpha 1 installer |
| Whisper models | Model-only downloads managed by Diarix | Available after Core sign-off |
| CUDA backend | Server core plus verified CUDA-library release parts | Available after CUDA cold-install sign-off |

The compact server already includes the native `transcribe.cpp` runtime. Models and the optional
CUDA backend are downloaded into user data; they do not create another Diarix app or Python worker.
Every package remains interchangeable against the same library, task API, and cache.

## Release gate

Run these from the repository root:

```powershell
$bun = 'Z:\Diarix Studio\Toolchains\bun\bun.cmd'
& $bun install --frozen-lockfile
& $bun run verify:alpha
& $bun run typecheck
& $bun run --cwd app build
python -m compileall -q -x 'backend[\\/](venv|\.venv|build|dist)[\\/]' backend
cargo check --manifest-path tauri/src-tauri/Cargo.toml
python -m pytest backend/tests -q
```

Then build the frozen runtime and perform a cold-install smoke test:

1. Launch the installed app without a development server or terminal.
2. Confirm `/health` reports the bundled backend variant and expected GPU.
3. Download one small STT, TTS, and refinement model from the app.
4. Transcribe a WAV and an MP4 to their full durations.
5. Confirm partial text/progress, cancellation, history, export, tray dictation, model unload, and
   restart persistence.
6. Uninstall and confirm user-selected model/cache directories are not deleted without consent.

IBM Granite Speech 3.3 8B is excluded from the current GPU verification matrix and must remain
marked unverified for this alpha.

## Ship decision

Tag `v0.1.0-alpha.1` only after every automated check passes and the exact installer artifact has
completed the cold-install smoke test. Publish it as a GitHub prerelease with the known limitations
from `CHANGELOG.md`.

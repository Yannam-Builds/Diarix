# Diarix Codex Handoff

Updated: 2026-07-16

## New-chat objective

Continue building Diarix as a transcription-first desktop speech studio based directly on the
upstream Voicebox source. Do not create a separate Diarix application beside Voicebox. Extend the
existing Voicebox architecture, model lifecycle, backend, Tauri shell, and UI components.

The next major change should make the dashboard transcription-first, restore the approved animated
wave fields on onboarding and the transcription dashboard, and add a central FFmpeg media-ingestion
pipeline so supported audio and video formats are normalized before reaching any transcription model.

## Active repository

- Repository: `Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713`
- Branch: `diarix/extensions`
- Upstream source baseline: `jamiepine/voicebox`, commit `f2cf2a7`
- Previous custom React rebuild remains preserved at:
  `Z:\Diarix Studio\diarix-studio-voicebox-rebuild-20260713`
- Previous Flutter implementation remains preserved separately. Do not delete either older project.
- Do not use `rg`; the user explicitly requested that it not be used. Use PowerShell
  `Get-ChildItem` and `Select-String`.
- Read `AGENTS.md` if one appears. The repository currently had no local `AGENTS.md`.
- Use `apply_patch` for manual edits.

## Product direction

Diarix is an independent fork built from Voicebox by Jamie Pine and its contributors.

The intended product structure is:

1. Transcription is the primary workflow and default dashboard.
2. Voicebox TTS, voice profiles, stories, history, MCP, and related functionality remain available
   in their own navigation sections.
3. Diarix-specific transcription engines integrate into Voicebox's existing model manager and
   download lifecycle. They must not use a second worker, second cache, or separate progress UI.
4. Speaker diarization has been removed from product scope. Do not reintroduce Pyannote, speaker
   labels, or Hugging Face diarization-token controls.
5. Keep the existing approved CSS/canvas wave animation language from the old onboarding and
   dashboard. The current upstream-source fork contains an untracked
   `app/src/components/SpeechMotion/` directory with preserved wave code, but it has not yet been
   connected to the active screens.
6. Avoid generic AI-dashboard styling. Preserve Voicebox's restrained native product character and
   use motion for continuity and feedback rather than decoration.

## Completed native transcription integration

The temporary standalone Diarix route, bridge service, and giant JSON-line worker were removed.
Diarix transcription is now integrated natively into the Voicebox FastAPI backend.

### Unified model catalog

`backend/backends/__init__.py` now provides an extended `ModelConfig` with:

- modality
- runtime group
- capabilities
- description
- precision options/default
- recommendation state
- minimum VRAM
- canonical and legacy model resolution

Models currently represented include:

- Whisper Base, Small, Medium, Large v3, and Large v3 Turbo
- Distil-Whisper Large v3.5
- WhisperX Large v3, without diarization
- NVIDIA Parakeet TDT 0.6B v3
- NVIDIA Canary 1B v2
- Qwen3-ASR 0.6B and 1.7B
- IBM Granite Speech 3.3 8B

### Advanced adapters

The following directory was added:

`backend/backends/stt/`

It contains lazy adapters for shared STT behavior, Transformers, NeMo, Qwen ASR, and WhisperX.
Advanced runtimes are imported only when selected.

WhisperX now performs transcription and alignment only. Its advertised capabilities are word
timestamps, alignment, VAD, and multilingual transcription. All diarization code and HF token
lookup were removed.

### Voicebox lifecycle reuse

All model weights use the existing Voicebox pathways:

- `POST /models/download`
- existing task manager
- existing progress manager and SSE progress
- selected Hugging Face/model cache
- cancel, delete, migration, installed-state scan, load, and unload behavior

`GET /models/catalog` returns metadata immediately without waiting for a disk scan.
`GET /models/status` overlays live cache/download/load state.

Captures and `/transcribe` resolve the selected model through the unified STT registry.
Legacy settings such as `base`, `small`, `medium`, `large`, and `turbo` still resolve.

### Packaging

- `backend/requirements-advanced-asr.txt` lists optional advanced ASR packages.
- `.github/workflows/release.yml` installs it only in the Windows CUDA artifact job.
- `backend/build_binary.py` includes advanced adapter hidden imports and CUDA collection metadata.
- This packages advanced support into the same Voicebox CUDA sidecar/update artifact, not a separate
  Diarix service.
- A real CUDA sidecar containing these additions has not yet been built and installed locally. The
  currently installed Voicebox sidecar is still the old upstream binary, so it cannot expose the new
  catalog at runtime.

## Frontend changes already made

- Model manager uses immediate catalog metadata and later overlays scanned status.
- Models are grouped by backend modality instead of filename prefixes.
- Capture transcription settings are populated from the catalog rather than a hardcoded Whisper list.
- Readiness checking resolves canonical model IDs correctly.
- Model metadata includes capabilities, runtime, precision, estimated size, recommendation, and
  hardware fit.

The application is not yet transcription-first. The upstream Voicebox editor remains the default
experience. Reworking this is the next priority.

## Branding completed

- Product name: Diarix
- Logo asset: `app/src/assets/diarix-logo.png`
- Logo is black in light mode and white in dark mode through `.diarix-logo` CSS.
- Logo is used in loading, navigation, About, and Windows bundle icons.
- Tauri product/window name: Diarix
- Tauri identifier: `com.diarix.app`
- Rust package and desktop executable: `diarix` / `diarix.exe`
- Self-focus detection was changed from `voicebox.exe` to `diarix.exe`.
- MCP install paths now point to the Diarix application directory, while the bundled compatibility
  binary remains named `voicebox-mcp`.
- Default generated names now use `Diarix Generation` and `Diarix Profile`.
- About and documentation explicitly state that Diarix is a fork built from Voicebox.

Keep these compatibility identifiers unchanged unless a migration is designed:

- `voicebox.*` MCP tool names
- `X-Voicebox-Client-Id`
- `voicebox-server` and `voicebox-mcp` sidecar names
- inherited archive/application-data paths needed to open existing Voicebox data

## Fork-safe updates

The inherited Voicebox updater endpoint and signing key were removed. The native updater plugin and
permission were also removed because Tauri panics at startup if the plugin is registered without a
configuration.

The UI now states that Diarix updates are unavailable until the fork publishes its own signed
release channel. Do not reconnect Diarix to Voicebox's updater feed because that could overwrite the
fork with an upstream build.

When a private Diarix GitHub repository and signing keys exist, restore updates using a Diarix-owned
release feed and key.

## Documentation

Primary logs:

- `docs/DIARIX_INTEGRATION_LOG.md`
- `Z:\#####Transcription\SessionState.md`

README and active docs were rebranded and include fork attribution. Upstream Voicebox download links
are labeled as upstream-only because Diarix does not yet publish signed installers.

The docs production build remains unverified: its first Bun dependency installation timed out before
`fumadocs-mdx` was installed. Do not claim that the docs site builds until dependencies install and
`bun run build` passes in `docs/`.

## Next implementation: transcription-first dashboard

Reorganize the existing Voicebox app instead of replacing its component system.

Recommended information architecture:

- `Transcribe` or `Dashboard`: default route and primary workspace
- `Voice Studio`: Voicebox TTS generation
- `Profiles`: Voicebox voice profiles
- `Stories`: multi-speaker/story editor
- `Models`: unified TTS and STT model manager
- `Captures` or `History`: recordings, transcripts, and generated audio
- `MCP`: agent integration
- `Settings`: server, storage, transcription defaults, voice, updates, and About

The default transcription workspace should contain:

- drag/drop and browse for files or folders
- supported-format summary
- selected transcription model
- language selection and output suffix behavior
- precision selection constrained by model capabilities
- output folder
- progress with actual percentage/stage/file state
- recent transcripts and open-output actions
- animated waveform field integrated into the layout

Do not make a marketing hero. This is a work surface.

## Next implementation: waveform motion

Restore wave motion to:

- onboarding
- transcription-first dashboard

Requirements:

- preserve the earlier closely spaced flowing line-wave visual
- motion should flow toward the primary transcription action
- use `requestAnimationFrame`, allowing the browser compositor to follow the display refresh rate
- avoid artificial FPS caps
- use CSS transforms/canvas rather than layout-changing properties
- stop animation when the document is hidden
- support `prefers-reduced-motion`
- prevent clipping by allocating a stable wave viewport and overscan
- do not add gradient blobs, decorative orbs, or generic AI visual effects

## Next implementation: media compatibility pipeline

Do not pass arbitrary source containers directly to individual models. Add one ingestion pipeline:

```text
selected media
-> FFprobe validation
-> select audio stream
-> FFmpeg extraction and normalization
-> engine-specific PCM file/stream
-> transcription backend
-> output and cleanup
```

Target accepted formats should include at least:

- WAV
- MP3
- M4A
- AAC
- FLAC
- OGG/Opus
- MP4/MOV
- WebM

Technical correction: CUDA is useful for ASR inference, but audio extraction from MP4 generally does
not benefit from CUDA. FFmpeg should use `-vn` and decode only the audio stream. Do not decode or
transcode video. The Voicebox CUDA server can coordinate the operation and provide bundled FFmpeg,
but normal audio demux/decoding should remain CPU-side. GPU inference starts after normalization.

Implementation recommendations:

- Add per-model input requirements to model metadata, including sample rate, channels, and PCM format.
- Default Whisper-family normalization: mono, 16 kHz PCM WAV or equivalent float PCM expected by the
  backend.
- Use `ffprobe` to detect missing audio streams, duration, codec, channels, and sample rate.
- Stream FFmpeg progress into the existing Voicebox task/progress system.
- Store temporary normalized files under the selected Voicebox cache/model-storage root, not `C:`.
- Use unique job directories and delete temporary media after success, cancellation, or failure.
- Preserve the original source file.
- Return useful errors for encrypted, corrupt, unsupported, or audio-less video files.
- Avoid shell-constructed commands; pass FFmpeg arguments as a structured subprocess argument list.
- Limit input size/duration where appropriate and prevent user-controlled paths from escaping the
  selected workspace/cache root.
- If the source already matches an engine's requirements, avoid unnecessary conversion.

## Security and secrets

- Never write API keys or Hugging Face tokens to source, logs, screenshots, handoff files, or exports.
- A Hugging Face token was previously pasted in chat. Treat it as exposed and do not reuse or record
  it. The user should revoke/rotate it separately.
- Preserve secure provider-key storage behavior where present.
- Keep model downloads within the user-selected cache location.

## Verification completed

- Frontend TypeScript check passed.
- Frontend Vite production build passed, with only the existing large-chunk warning.
- Backend Python compileall passed.
- STT registry/downloader tests: 5 passed.
- `cargo check --no-default-features` passed for `diarix`.
- `git diff --check` passed; line-ending warnings are informational.
- Tauri development app launched successfully as:
  `tauri\src-tauri\target\debug\diarix.exe`

At the end of the previous session, the development Diarix process was running. Recheck rather than
assuming it remains active.

## 2026-07-14 continuation state

- The one-sentence audio/video issue was fixed in core Whisper. It now uses untruncated native
  long-form generation instead of one 30-second call.
- The exact 1327.787-second user MP4 completed on CUDA with a 13,107-character transcript reaching
  the final goodbye. The retained output is under `data/transcriptions` with suffix
  `_longform-fixed.txt`.
- Distinct cached additions are registered: `faster-whisper-tiny`,
  `faster-distil-whisper-large-v3`, and `nvidia-canary-180m-flash`.
- Per-model language validation is enforced in the UI and API. Central media ingestion remains the
  only source-format boundary; all current model adapters receive mono 16 kHz PCM WAV.
- The header CTA is opaque above the wave, transcription inference uses an honest indeterminate
  progress animation, and the debug Windows shell no longer opens a console.
- The active development executable is
  `Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713\tauri\src-tauri\target\debug\diarix.exe`.

## 2026-07-16 continuation state

- Version is now **0.1.0** everywhere (bumpversion targets, Cargo.lock, backend `__version__`).
  The CUDA sidecar version gate in `main.rs` compares against the app version, so any previously
  built 0.5.0 `backends/cuda` tree will be rejected until rebuilt at 0.1.0.
- Server binaries are renamed: `diarix-server`, `diarix-server-cuda`, `diarix-server-rocm`.
  Legacy `voicebox-server-*` names are still accepted by `server.py` variant detection and by
  `main.rs` backend-directory probing. The MCP shim intentionally remains `voicebox-mcp`, and
  `voicebox.*` MCP tool names, `X-Voicebox-Client-Id`, `voicebox.db`, persistence keys, and
  `VOICEBOX_*` env vars are intentionally unchanged.
- Runtime fixes landed this session: NeMo models now actually move to CUDA; cancellation stops
  at the next chunk and unloads the model immediately (`should_stop` plumbing through every STT
  adapter + unload in `run_transcription_job`'s CancelledError handler); the self-focus
  auto-paste constant now matches `diarix.exe`/`com.diarix.app`.
- New features landed: live `partial_text` in the job SSE payload and dashboard; SRT/VTT/JSON
  export + silence-paragraph reflow (segment-capable engines only: Faster-Whisper, WhisperX);
  Captures re-transcribe-with-model dropdown; server-side transcript search on `/captures`;
  bounded `custom_instructions` in refinement settings (new nullable `capture_settings` column
  with idempotent migration).
- Toolchains live at `Z:\Diarix Studio\Toolchains` (bun 1.3.14, rustup/cargo — set
  `RUSTUP_HOME`/`CARGO_HOME` to the Toolchains subdirs). Backend Python env with PyInstaller,
  torch/cu128, and NeMo is `Z:\#####Transcription\Python311\python.exe`. No pytest is installed
  in any local env yet; `backend/tests/test_transcription_cancel.py` is written but unrun.
- Verified this session: app/web TypeScript, `cargo check` (3 pre-existing dead-code warnings in
  `keyboard_layout.rs`), backend `py_compile`, transcript formatter edge cases, and the
  `voicebox-mcp` shim rebuild. Sidecar binaries were staged under
  `tauri/src-tauri/binaries/diarix-server-x86_64-pc-windows-msvc.exe` (copied 0.5.0 CPU exe,
  placeholder until rebuilt) and `voicebox-mcp-x86_64-pc-windows-msvc.exe` (fresh build).
- Installer scaffolding: `installer/DiarixSetup.iss` is at 0.1.0
  (payload `Z:\Diarix Studio\Diarix Setup Payload 0.1.0`, output
  `Z:\Diarix Studio\Diarix Setup 0.1.0`), and `installer/build-diarix-setup.ps1` orchestrates
  CPU build → shim → CUDA onedir (+ frozen runtime self-test) → Tauri release → payload assembly
  (no starter models) → ISCC compile. **ISCC.exe (Inno Setup 6) was not found on this machine**;
  install it before the final step. The heavy PyInstaller/Tauri release builds had not been run
  yet at handoff time.

## 2026-07-16 final state (supersedes the continuation state above)

Everything in the "2026-07-16 continuation state" section above was written mid-session, before
the CUDA build was actually verified working. It's kept for history but the state below is current.

### What's actually installed and running right now

`Z:\Diarix Studio\Diarix\Diarix.exe` — a portable (non-Inno-Setup) deployment of the 0.1.0 build,
CPU + CUDA sidecars both present, CUDA confirmed live (GPU Settings page: "CUDA Backend Active —
NVIDIA GeForce RTX 5070 Ti Laptop GPU"). This is not from a compiled installer — `ISCC.exe` (Inno
Setup 6) still isn't installed on this machine, so `Diarix Setup.exe` has never been produced. The
existing model cache/downloads under that install were preserved (not wiped).

### Fully resolved this session

- MCP unwired (backend + frontend + build scripts); see the integration log's
  "MCP removed from the shipped product" entry for exactly what to restore if it comes back.
- The CUDA server's real crash cause (torch silently downgraded to a CPU build mid-venv-setup —
  see the integration log's "Real CUDA-server root cause" entry). If `diarix-cuda-venv` is ever
  rebuilt from scratch, redo the fix's last step (`pip install torch==2.8.0 torchvision==0.23.0
  torchaudio==2.8.0 --index-url .../cu128 --force-reinstall`) and verify with
  `torch.cuda.is_available()` before trusting a build — the `--runtime-self-test` import check
  does **not** catch this failure mode.
- Icon *sizing* (`icon.ico` now has all 10 real embedded sizes instead of one stretched 16×16
  frame — see `installer/resize-icon-glyph.py`).
- Disk space: both drives were critically low from repeated rebuild cycles this session; freed
  ~44GB of pure build cache/superseded output (see integration log). `diarix-cuda-venv` (9.6GB)
  was deliberately kept — expensive to rebuild, and doing so again risks re-hitting the same torch
  regression if the fix's last step is skipped.

### Explicitly unresolved — do not restart from scratch

**Icon fidelity.** Sizing is fixed; sharpness/faithfulness to the original design is not. Three
different reconstruction approaches (raster crop, vtracer+raster-dilate, vtracer+pyclipper-vector-
offset) were tried and none preserved the source's delicate double-outline/hollow-pill detail well
enough for the user — the last attempt (a clean solid badge) was visually good but rejected as too
different from the original design. **The user is providing an upscaled PNG and a real SVG source
directly** — do not re-attempt auto-tracing/reconstruction. Once those files arrive: point
`installer/resize-icon-glyph.py`'s `SOURCE` at the new PNG (or adapt it to rasterize the SVG
directly at high resolution first), regenerate, rebuild Tauri (`bun tauri build --no-bundle` from
`tauri/`), and redeploy `Diarix.exe` to both the payload and `Z:\Diarix Studio\Diarix`. The ICO
multi-size fix and the `installer/*.py` rasterization scripts are reusable as-is.

**Two-track release (whisper.cpp-lite + full PyTorch server).** Discussed, not designed or
started. Real tradeoff to resolve first: whisper.cpp/GGML only covers Whisper-family ASR, none of
NeMo/Qwen3-ASR/the TTS engines — needs a product decision on what the lightweight track actually
ships, not just a packaging change.

**Final installer compile.** `installer/DiarixSetup.iss` is ready at 0.1.0, but nothing has been
run through Inno Setup — install ISCC.exe first (the previous `Diarix Setup Payload 0.1.0` staging
directory was deleted in the disk cleanup since its contents are already deployed; regenerate it
from `backend/dist/` + `tauri/src-tauri/target/release/diarix.exe` before compiling, per
`installer/build-diarix-setup.ps1`'s payload-assembly step — note `backend/dist/` was also deleted
in cleanup and needs a rebuild via `build_binary.py` first).

**Backend test suite.** Still unverified — no pytest install exists in any local Python env this
session touched. `backend/tests/test_transcription_cancel.py` (written this session) and the rest
of the suite have never actually been run.

### Toolchain reference (see integration log for the full gotcha list)

- Bun: `Z:\Diarix Studio\Toolchains\bun\node_modules\bun\bin\bun.exe` (not `Toolchains\bun\bun.exe`
  — that's a shim script).
- Cargo/rustup: `Z:\Diarix Studio\Toolchains\cargo`/`\rustup` — set `CARGO_HOME`/`RUSTUP_HOME`.
- CPU build venv: `Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713\backend\venv`.
- CUDA build venv: `Z:\Diarix Studio\diarix-cuda-venv` (torch 2.8.0+cu128 — do not blindly
  `pip install -r requirements-advanced-asr.txt` again without reinstalling torch from the cu128
  index as the *last* step afterward, or the regression repeats).
- ASR-only reference env (never modify): `Z:\#####Transcription\Python311\python.exe`.
- No `Co-Authored-By: Claude` or Claude/Anthropic attribution in commits or GitHub contributions
  for this repo — explicit user instruction.

## Immediate first steps for the next chat

1. Read this file and `docs/DIARIX_INTEGRATION_LOG.md` (especially the 2026-07-16 continued entry).
2. Run `git status` in the repo — everything is still uncommitted working-tree changes on
   `diarix/extensions`; nothing has been committed this session either.
3. Check whether the user has delivered the upscaled PNG/SVG for the icon — that's the actual
   open thread, not a broader feature request.
4. Otherwise, pick up either the two-track release design or the Inno Setup final-compile step,
   per the "Explicitly unresolved" section above.

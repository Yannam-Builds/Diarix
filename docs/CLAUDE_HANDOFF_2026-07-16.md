# Claude Session Handoff — 2026-07-16

Complete record of everything changed in the Claude session of 2026-07-16, which took over from
Codex's 2026-07-15 state ("CUDA installer bundle, frozen ASR audit, and resource guard" in
`docs/DIARIX_INTEGRATION_LOG.md`). Written for Codex to continue from. Read this alongside:

- `docs/DIARIX_INTEGRATION_LOG.md` — the two `2026-07-16` entries summarize the same work in
  log form.
- `docs/CODEX_HANDOFF_2026-07-13.md` — its "2026-07-16 final state" section is the short version
  of the current state; this file is the long version with every touched path.

Repository: `Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713`, branch `diarix/extensions`.
**Nothing was committed this session** — everything below is uncommitted working-tree changes on
top of the already-uncommitted Codex work. When committing: NO `Co-Authored-By: Claude` or any
Claude/Anthropic attribution (explicit user instruction).

---

## Part 1: Bug fixes (backend)

### 1a. NeMo models ran on CPU even with CUDA active

`backend/backends/stt/nemo_backend.py`
- `__init__` hardcoded `self.device = "cuda"` but `_load_sync` never moved the model —
  NeMo's `from_pretrained()` loads to CPU and doesn't consult that field.
- Fix: `_load_sync` now resolves the real device via `get_torch_device()` (imported from
  `..base`), calls `self.model = self.model.cuda()` (or `.to(device)`), calls `.eval()`,
  updates `self.device` to the resolved value, and logs
  `"Loaded %s on %s"` like the other adapters. `__init__` default changed to `"cpu"`.
- Confirmed live: transcription job in the running app used the RTX 5070 Ti.

### 1b. WhisperX / Faster-Whisper Large v3 looked like duplicate models

They intentionally share one checkpoint (`Systran/faster-whisper-large-v3`). The UI made that look
like a bug (two rows, same download state).

- `backend/routes/models.py` — new helper `_shares_cache_with_map(configs)`: groups all model
  configs by `hf_repo_id`, returns `{model_name: [other display names]}` for any collision.
  Generic — any future repo collision surfaces the same way. Wired into both `/models/catalog`
  and `/models/status` (all three `ModelStatus(...)` constructions incl. the exception-fallback
  branch).
- `backend/models.py` — `ModelStatus` gained `shares_cache_with: List[str]`.
- `app/src/lib/api/types.ts`, `app/src/lib/api/models/ModelStatus.ts`,
  `app/src/lib/api/schemas/$ModelStatus.ts` — field added to all three type surfaces.
- `app/src/components/ServerSettings/ModelManagement.tsx` — renders
  `t('models.sharesCacheWith', {names})` under the model name when non-empty.
- `app/src/i18n/locales/en/translation.json` — key `models.sharesCacheWith` =
  `"Shares downloaded files with {{names}}"`.

### 1c. Cancel didn't stop transcription or free memory

Root cause: `backend/services/transcribe.py::await_stt_operation` wraps the whole multi-chunk
transcribe call in `asyncio.shield()`, so `.cancel()` raised to the caller but inference ran all
remaining chunks. And nothing ever unloaded the model after a cancel.

- `backend/services/task_queue.py` — new `is_transcription_cancel_requested(task_id)`:
  `task.cancelling() > 0` on the running asyncio task. Cooperative-poll flag for adapters.
- `backend/backends/stt/common.py` — new `stop_requested(should_stop)` helper +
  `StopCheck` type alias.
- Every chunked STT adapter gained a `should_stop: Optional[Callable[[], bool]] = None` kwarg and
  checks it between chunks:
  - `nemo_backend.py` (before each chunk in the loop)
  - `qwen_backend.py` (same)
  - `transformers_backend.py` (same)
  - `faster_whisper_backend.py` (per segment of its lazy generator — no separate chunking needed)
  - `pytorch_backend.py` (core Whisper — raises internal `_StopRequested` from inside the HF
    long-form `monitor_progress` callback, the only interruption point inside one `generate()`)
  - `whisperx_backend.py` — accepts the kwarg but documented as NOT honored (single vendored
    blocking call, no chunk boundary); `mlx_backend.py` likewise (documented, Apple-only path).
- `backend/backends/__init__.py` — `STTBackend` Protocol signature updated.
- `backend/services/transcribe.py::transcribe_audio` — passes `should_stop` through.
- `backend/services/transcription_jobs.py::run_transcription_job` — supplies
  `should_stop=lambda: task_queue.is_transcription_cancel_requested(task_id)`, and its
  `asyncio.CancelledError` handler now calls `unload_model_by_config(model_config)` (with
  logged-exception guard) so the model frees immediately once the in-flight chunk ends.
- New (unrun — no pytest env exists) regression test:
  `backend/tests/test_transcription_cancel.py`.

### 1d. Self-focus auto-paste constant was stale (found during rebrand)

`tauri/src-tauri/src/main.rs` — `VOICEBOX_BUNDLE_ID` was still `"voicebox.exe"` /
`"sh.voicebox.app"` while the shipped exe is `diarix.exe` / identifier `com.diarix.app`, so
paste-into-self detection could never match. Renamed to `APP_BUNDLE_ID` with correct values
(`com.diarix.app` macOS / `diarix.exe` Windows / `diarix` Linux).

---

## Part 2: New features

### 2a. Live per-chunk transcript under "Current job"

- `backend/backends/stt/common.py` — `report_partial_text(callback, stitched_so_far)`.
- All chunked adapters gained `partial_callback: Optional[Callable[[str], None]] = None` and call
  it with the accumulated stitched text after each chunk (faster_whisper: per segment).
  whisperx/mlx accept-but-don't-honor (documented in code).
- `backend/utils/tasks.py` — `TranscriptionTask.partial_text: str = ""`.
- `backend/models.py` — `TranscriptionJobResponse.partial_text: str = ""`.
- `backend/services/transcription_jobs.py` — `partial_text_progress()` closure publishes it via
  the existing task/SSE plumbing; reset to `""` at each file's transcribing stage.
- `app/src/lib/api/types.ts` — `TranscriptionJob.partial_text?: string`.
- `app/src/components/TranscriptionDashboard/TranscriptionDashboard.tsx` — auto-scrolling
  "Live transcript" region under the progress bar in the Current job card (`partialTextRef` +
  scroll-to-bottom effect keyed on `job?.partial_text`).

### 2b. SRT/VTT/JSON export + silence-based paragraphs

- New module `backend/services/transcript_formats.py`: `segments_to_srt/vtt/json`,
  `paragraphs_from_segments(gap_seconds=1.2)`, `SUPPORTED_EXPORT_FORMATS`. Pure functions,
  directly tested this session (timestamp rollover, clamping, empty-segment edge cases).
- `backend/backends/stt/common.py` — `report_segments(callback, segments)`.
- Segment sources: `faster_whisper_backend.py` (native segments w/ start/end) and
  `whisperx_backend.py` (aligned segments). All other adapters accept `segments_callback` but
  never call it — their 30–60s chunk boundaries are not honest subtitle timestamps (documented
  in each adapter). Core Whisper's `batch_decode(skip_special_tokens=True)` strips timestamp
  tokens, so it's also text-only (documented).
- `backend/services/transcription_jobs.py`:
  - `resolve_export_formats(str)` validator (comma list; `txt` always included).
  - `run_transcription_job(..., export_formats=None, silence_paragraphs=False)`.
  - `_unique_output_path` gained an `extension` param.
  - When segments arrived: optional paragraph reflow of the `.txt`, plus one extra file per
    requested format via the same atomic-write path; paths recorded in
    `TranscriptionResult.extra_outputs` (new field, also on the Pydantic
    `TranscriptionJobResult` and the TS type).
- `backend/routes/transcription.py` — form fields `export_formats: str | None` and
  `silence_paragraphs: bool` on `POST /transcription/jobs`.
- `app/src/lib/api/client.ts` — `createTranscriptionJob` appends both fields when set.
- Dashboard UI — SRT/VTT/JSON checkboxes + "Paragraph breaks on silence" toggle, shown only when
  the selected model's capabilities include `segment_timestamps` or `alignment` (deliberately NOT
  `word_timestamps` — NeMo declares that but its adapter can't deliver segments, so the UI gate
  matches what the backend actually produces).

### 2c. Re-transcribe with a different model (Captures)

Backend endpoint `/captures/{id}/retranscribe` and the API client method already existed — only
UI was missing.
- `app/src/components/CapturesTab/CapturesTab.tsx` — "Re-transcribe" dropdown (RefreshCw icon)
  next to "Refine locally": lists downloaded STT models (`modelStatus` query), checkmark on the
  capture's current `stt_model`, `retranscribeMutation` → toast + captures invalidation.
- i18n keys: `captures.actions.retranscribe/retranscribing/retranscribeDropdownLabel`,
  `captures.toast.retranscribed/retranscribedDetail/retranscribeFailed` (en only; other locales
  fall back).

### 2d. Server-side transcript search (full history)

- `backend/services/captures.py::list_captures` — optional `search` param; escaped
  case-insensitive `ilike` over `transcript_raw` + `transcript_refined` (LIKE wildcards `%_\`
  escaped so literal searches work).
- `backend/routes/captures.py` — `search` query param on `GET /captures` (max 200 chars).
- `app/src/lib/api/client.ts::listCaptures(limit, offset, search?)`.
- `CapturesTab.tsx` — 250ms-debounced `debouncedSearch` state; query key is now
  `['captures', debouncedSearch]`. The `capture:created` Tauri-event cache seed updated to key
  `['captures', '']` (prefix invalidation still covers filtered views). The existing client-side
  filter kept for instant keystroke feedback.

### 2e. Custom refinement instructions

- `backend/services/refinement.py` — `RefinementFlags.custom_instructions: str | None`;
  `normalize_custom_instructions()` (trim/bound, `MAX_CUSTOM_INSTRUCTIONS_CHARS = 500`);
  `build_refinement_prompt` appends it as a quoted "user preferences" section that explicitly
  cannot override the "text filter, not an assistant" base contract.
- `backend/database/models.py` — nullable `custom_instructions` TEXT column on
  `capture_settings`; idempotent migration added in `backend/database/migrations.py`
  (`_migrate_capture_settings`).
- `backend/models.py` — field on `RefinementFlagsModel` (max_length=500),
  `CaptureSettingsResponse`, `CaptureSettingsUpdate`.
- `backend/routes/captures.py` — refine endpoint passes it through in both the request-flags and
  saved-settings branches.
- `app/src/components/ServerTab/CapturesPage.tsx` — textarea under the three refinement toggles;
  local draft state, persisted on blur; disabled when auto-refine is off.
- `app/src/lib/api/types.ts` — field on `RefinementFlags` + `CaptureSettings`.
- i18n: `settings.captures.refinement.customInstructions.{title,description,placeholder}`.

---

## Part 3: Version 0.1.0 + rebrand completion

### Version

`0.5.0 → 0.1.0` in: `.bumpversion.cfg`, `package.json`, `app/package.json`, `tauri/package.json`,
`web/package.json`, `landing/package.json`, `tauri/src-tauri/tauri.conf.json`,
`tauri/src-tauri/Cargo.toml`, `tauri/src-tauri/Cargo.lock` (diarix entry),
`backend/__init__.py`. Note: `main.rs`'s CUDA/ROCm version gate compares the sidecar's
`--version` output to the app version, so any 0.5.0 `backends/` tree is rejected until rebuilt.

### Server binary rename (`voicebox-server*` → `diarix-server*`)

- `backend/build_binary.py` — `binary_name` values + help text.
- `backend/voicebox-server.spec` → renamed file `backend/diarix-server.spec`, internal
  `name='diarix-server'`; `backend/pyproject.toml` reference updated.
- `backend/server.py` — `--version` string is `diarix-server {ver}`; variant-detection regexes
  accept BOTH `(diarix|voicebox)-server-(rocm|cuda)` for legacy compat.
- `scripts/build-server.sh`, `scripts/package_cuda.py`, `scripts/package_rocm.py`, `justfile`
  (also fixed dev CUDA install path `sh.voicebox.app` → `com.diarix.app`),
  `.github/workflows/release.yml`, `.github/workflows/build-windows.yml`,
  `scripts/setup-dev-sidecar.js`.
- `tauri/src-tauri/tauri.conf.json` — `externalBin: ["binaries/diarix-server"]`.
- `tauri/src-tauri/src/main.rs` — `sidecar("diarix-server")`; ROCm/CUDA dir probing checks
  BOTH new and legacy exe names (arrays, `find_map`); process-reuse/kill checks match
  `diarix` OR `voicebox`; all log/error strings updated; window title "Diarix Dictate".

### Other naming (product-visible)

- `backend/app.py` — FastAPI `title="Diarix API"`, lifespan renamed, log lines.
- `backend/routes/health.py` — root message `"Diarix API"`; `backend/routes/__init__.py` docstring.
- Export bundles: `.voicebox.zip` → `.diarix.zip` in `backend/routes/history.py`,
  `backend/routes/profiles.py`, `app/src/lib/hooks/useHistory.ts`, `useProfiles.ts`;
  `MainEditor.tsx` import accepts BOTH extensions.
- Frontend symbols: `isLoopbackVoiceboxServerUrl` → `isLoopbackDiarixServerUrl`
  (`serverStore.ts`, `App.tsx`), `isVoiceboxHealthResponse` → `isDiarixHealthResponse`
  (`App.tsx`), `__voiceboxServerStartedByApp` → `__diarixServerStartedByApp`
  (`app/src/global.d.ts`, `App.tsx`, `tauri/src/platform/lifecycle.ts` — where the now-unneeded
  `@ts-expect-error` was removed; NEW file `tauri/src/global.d.ts` mirrors the declaration since
  the tauri workspace tsconfig only includes `tauri/src`).
- `package.json` name `diarix`, signer key path `~/.tauri/diarix.key`; `backend/pyproject.toml`
  name `diarix-backend`; `<title>Diarix</title>` in `app/index.html` + `web/index.html`;
  one i18n server-address string.

### Deliberately KEPT as voicebox (compat — do not "fix"):

`voicebox.*` MCP tool names, `X-Voicebox-Client-Id`/`VOICEBOX_CLIENT_ID`, `voicebox-mcp` shim
binary name, `voicebox.db` filename, localStorage/zustand persistence keys (`voicebox:lang`,
`voicebox-ui`, `voicebox-server`, `voicebox-audio-channels`), all `VOICEBOX_*` env vars,
upstream attribution text, `landing/` (entirely out of scope per user).

---

## Part 4: MCP removal (user request, later in session)

- `backend/app.py::create_app` — FastMCP build/mount/lifespan-composition removed; plain
  `diarix_lifespan` now passed straight to FastAPI. `ClientIdMiddleware` KEPT (used by `/speak`).
- `backend/routes/__init__.py` — `mcp_bindings` router import + registration removed.
- Deleted: `app/src/components/ServerTab/MCPPage.tsx` (was already unrouted/orphaned).
- `tauri.conf.json` `externalBin` — `binaries/voicebox-mcp` removed.
- Shim no longer built/copied in: `installer/build-diarix-setup.ps1`,
  `installer/run-throttled-build.ps1`, `installer/resume-from-cpu-server.ps1`,
  `scripts/build-server.sh`, both GitHub workflows.
- NOT deleted: all of `backend/mcp_server/`, `backend/routes/mcp_bindings.py`, the
  `mcp_client_bindings` table, `build_binary.py --shim`. Re-adding MCP = re-wiring, not rewriting.

---

## Part 5: The CUDA build saga (read before ever rebuilding the venv)

The full story is in the integration log; the operational facts:

1. A fresh full build venv now exists: `Z:\Diarix Studio\diarix-cuda-venv` (TTS + advanced-ASR +
   pyinstaller + torchcodec). The old ASR-only env `Z:\#####Transcription\Python311` was NEVER
   modified and remains the live-tested reference env.
2. Two stale pins in `backend/requirements-advanced-asr.txt` were FIXED (they made the file
   uninstallable from current PyPI): `numpy>=2.0.0,<3.0` + `numba>=0.64.0` (was `<2.0`/`<0.61`),
   and `transformers==4.57.6` (was `4.57.3`; qwen-asr 0.0.6 now pins exactly 4.57.6, nemo wants
   `~=4.57.0`). Both verified against PyPI metadata, and the frozen self-test passed 14/14
   afterward including chatterbox/tada (installed `--no-deps`; the numpy 2.x bump did not break
   their imports).
3. **The trap**: installing `requirements-advanced-asr.txt` without the cu128 index caused pip to
   silently replace cu128 torch with `2.8.0+cpu` (to satisfy whisperx's `torch~=2.8.0` from
   default PyPI). PyInstaller then bundled mismatched DLLs → `WinError 127` on `c10_cuda.dll` at
   server startup. The `--runtime-self-test` DOES NOT catch this (imports only). Fix + guard:
   `pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"
   --index-url https://download.pytorch.org/whl/cu128 --force-reinstall` as the LAST venv step,
   then verify `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
4. After the fix: direct exe invocation reached `Application startup complete` and the live app
   showed "CUDA Backend Active — NVIDIA GeForce RTX 5070 Ti Laptop GPU".

Build-orchestration scripts written this session (all in `installer/`):
`run-throttled-build.ps1` (full chain, ~70% CPU cap via BelowNormal + affinity 0x3FFF),
`resume-from-cpu-server.ps1`, `build-cuda-venv.ps1`, plus icon scripts (Part 6). PowerShell
gotchas encoded in them: raw `System.Diagnostics.Process` (never `Start-Process -PassThru` —
its `ExitCode` is unreliably `$null` on this machine), `ProcessStartInfo.Arguments` string (no
`.ArgumentList` on this .NET), env vars set via `$psi.EnvironmentVariables` (plain `$env:` didn't
propagate in one case), ASCII only (em-dashes broke parsing twice), bun's real exe is
`Toolchains\bun\node_modules\bun\bin\bun.exe`.

---

## Part 6: Icon — partially fixed, fidelity UNRESOLVED, user is providing assets

- REAL bug found and fixed: `icon.ico` contained only ONE 16×16 frame. Pillow's ICO writer
  ignores `append_images` and only emits sizes ≤ the image `.save()` is called on; the script had
  called save on the smallest frame. Windows stretched that 16px bitmap everywhere (the "mushy
  blob" titlebar icon). `installer/resize-icon-glyph.py` now saves from the large master with
  `sizes=[16,20,24,32,40,48,64,96,128,256]` — verified all 10 frames embedded by parsing the ICO
  header.
- UNRESOLVED: glyph fidelity. Attempts (raster rescale → vtracer+MaxFilter → vtracer+pyclipper
  vector offset) all failed to preserve the original's thin double-outline/hollow-pill detail;
  the last produced a clean solid badge the user rejected ("too different"). **User is supplying
  an upscaled PNG + real SVG.** When they arrive: set `resize-icon-glyph.py`'s `SOURCE` to the
  new asset (rasterize the SVG at high res first if needed), regenerate, also replace
  `app/src/assets/diarix-logo.png` (currently my crop/rescale of the original — the only version
  that exists; the file was never committed to git so there is no pristine copy), rebuild Tauri,
  redeploy. Do NOT re-attempt auto-tracing.
- `app/src/assets/diarix-logo.png` in the working tree = original artwork, glyph recentered at
  ~86% canvas fill (content itself untouched). Used by loading screen (w-48), sidebar (w-12),
  About (w-20) via `object-contain`.
- Scratch/pipeline scripts kept for reuse: `installer/resize-icon-glyph.py` (regeneration),
  `installer/rasterize-traced-logo.py`, `installer/build-icon-master.py`,
  `installer/build-icon-master-v2.py` (the traced masters they reference are rejected designs —
  only the *pipeline* is worth keeping).

---

## Part 7: Deployment state + disk cleanup

- Live install: `Z:\Diarix Studio\Diarix\` — new `Diarix.exe` (0.1.0, multi-size ICO, no-MCP),
  `diarix-server-x86_64-pc-windows-msvc.exe` (CPU sidecar — the triple-suffixed name is REQUIRED
  by Tauri's sidecar API; a plain `diarix-server.exe` copy also sits there from a failed first
  deploy attempt and is unused-but-harmless, as are the old `voicebox-server-cuda.exe` inside
  `backends/cuda/` and a `test-data-2/` dir from direct server testing), fixed CUDA onedir at
  `backends\cuda\`, user's `models/` cache preserved.
- Two stale Diarix processes from before the redeploy were killed with explicit user
  confirmation. A Windows Firewall allow prompt for `diarix-server-cuda.exe` appeared on first
  CUDA launch; the user clicked Allow themselves.
- Windows icon cache was cleared once via `ie4uinit -ClearIconCache` + explorer restart — NOTE:
  the explorer restart permanently broke the Windows-MCP `Screenshot` tool's RPC for the rest of
  the session. Fallback that works: `System.Drawing`/`CopyFromScreen` via PowerShell, then Read
  the saved PNG.
- Disk cleanup (both drives were ~95%+ full): deleted `Z:\Diarix Studio\pip-cache` (3.7G),
  `Diarix CUDA 0.5.0` (6.7G), `Diarix Setup 0.5.0` (9.1G), `Diarix Setup Payload 0.1.0` (6.9G —
  contents already deployed), repo `backend/dist` (6.9G) + `backend/build` (1.4G), and
  `C:\Users\prana\AppData\Local\pip\cache` (8.1G). Z: 21→56GB free, C: 28→37GB free.
  CONSEQUENCE: `backend/dist/` is gone — rebuilding the installer payload requires rerunning
  `build_binary.py` (CPU venv) and `build_binary.py --cuda` (diarix-cuda-venv) first.
  KEPT deliberately: `diarix-cuda-venv` (9.6G, expensive + regression-prone to rebuild),
  `backend/venv` (2.5G CPU build venv), `Toolchains`, everything under `Z:\#####Transcription\`.

---

## Part 8: Verification actually performed (and not)

Done:
- `py_compile` on every touched backend module; JSON validity on every touched config/locale.
- Frontend `tsc` clean for `app/` and `web/`. `tauri/` workspace: fixed the two errors my changes
  caused; its `@tanstack/react-query` resolution error is PRE-EXISTING and still there.
- `cargo check` clean (3 pre-existing dead-code warnings in `keyboard_layout.rs`).
- `transcript_formats.py` exercised directly (SRT/VTT/JSON output, paragraph split, timestamp
  rollover/clamp edges).
- Frozen CUDA runtime self-test 14/14; direct server startup to `Uvicorn running`; live app
  confirmed CUDA active; server logs showed continuous healthy 200s.

NOT done (be honest about these):
- No pytest run ever — no pytest in any local env. `test_transcription_cancel.py` unrun; the
  whole backend suite unrun this session.
- No end-to-end exercise of the new features against a real job (SRT files, silence paragraphs,
  re-transcribe, search, custom instructions, live partial text) — code paths are typechecked/
  compiled only. The pre-session live job that was running when work started used the OLD build.
- Icon fidelity unresolved (Part 6). Inno Setup compile never run (no ISCC on machine).

## Open threads, in priority order

1. Icon: waiting on user's upscaled PNG + SVG → wire into `resize-icon-glyph.py`, regenerate,
   rebuild, redeploy (also update `app/src/assets/diarix-logo.png` from the same source).
2. Live-test the Part 2 features end-to-end on a short real file.
3. Get pytest into `backend/venv` and run the suite (incl. the new cancel test).
4. Two-track release design (whisper.cpp-lite + full) — discussed only; real product decision
   needed on what the lite track ships (whisper.cpp covers Whisper-family ASR only, no
   NeMo/Qwen3-ASR/TTS).
5. Inno Setup: install ISCC, rebuild `backend/dist` artifacts, reassemble payload, compile
   `installer/DiarixSetup.iss`.
6. Commit the working tree (it's ALL uncommitted — this session's and Codex's prior work).
   Remember: no Claude attribution.

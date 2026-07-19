# Diarix repository guide

## Purpose

Diarix is a transcription-first native desktop studio. The shipping product is the Tauri desktop
application backed by one local Python server; do not create another desktop shell or inference
worker.

## Mental model

- `app/` is the React product interface.
- `tauri/` owns native startup, tray, global shortcuts, focus restoration, and the bundled sidecar.
- `backend/` owns media ingestion, model runtimes, tasks, downloads, cache state, captures, TTS, and
  refinement.
- `installer/` and `scripts/` own reproducible distribution work.
- `docs/` contains current product and contributor documentation. Historical session handoffs are
  not product documentation.
- `landing/` and `web/` are retained upstream surfaces, excluded from the active workspace and alpha
  release path.

Read `.context/CONTEXT.md` before changing cross-layer contracts.

## Release rules

- `main` is the only GitHub branch.
- Keep versions aligned across JavaScript, Rust, Tauri, and Python. Run `bun run verify:alpha`.
- The desktop app starts and owns the server. Development terminals are never a production
  dependency.
- User media, models, caches, databases, build output, and release payloads are never committed.
- FFprobe inspects input and FFmpeg normalizes it. Individual model adapters must not invent their
  own ingestion path.
- Model progress must be measured when the runtime exposes progress and explicitly indeterminate
  otherwise.
- Speaker diarization and MCP remain outside the alpha product.

## Working rules

- Preserve dirty work and unrelated user changes.
- Use PowerShell `Get-ChildItem` and `Select-String`; do not use ripgrep.
- Use `apply_patch` for manual edits.
- Keep TTS, profiles, stories, history, models, and settings as separate sections around the default
  transcription dashboard.
- Test the real frozen sidecar and Tauri executable before calling a package releasable.

## Verification

Run the smallest relevant checks while developing, then the alpha gate described in
`docs/ALPHA_RELEASE.md` before packaging.

# Development

## Prerequisites

- Windows 11
- Python 3.12
- Bun 1.3 or newer
- Rust stable
- FFmpeg and FFprobe available to the backend build

## Start the source app

The desktop shell owns server startup in production. During focused backend development, use the
isolated Diarix port:

```powershell
python -m uvicorn backend.main:app --reload --port 17494
```

For the full desktop development path:

```powershell
bun install
bun run dev
```

## Working boundaries

- Add model metadata and cache rules to the backend catalog, then consume them from the frontend.
- Add media formats through central ingestion, not inside an adapter.
- Add native capture and lifecycle behavior to the existing Tauri process.
- Keep optional runtimes lazy-imported so the compact server remains usable.
- Do not commit caches, models, databases, transcripts, build logs, installers, or payloads.

## Before requesting review

Run `bun run verify:alpha`, the relevant frontend/backend tests, and `git diff --check`. For release
work, follow `ALPHA_RELEASE.md` and test the frozen executable rather than only the source server.

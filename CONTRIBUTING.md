# Contributing to Diarix

Diarix is moving toward its first Windows alpha. Changes should strengthen the transcription-first
desktop product and preserve local-first behavior.

## Before changing code

Read [`AGENTS.md`](AGENTS.md), [`.context/CONTEXT.md`](.context/CONTEXT.md), and the nearest relevant
source files. The shipping system is one Tauri application and one interchangeable local backend;
do not introduce another desktop shell, Python worker, model catalog, or media-ingestion path.

## Development

```powershell
bun install
bun run verify:alpha
bun run typecheck
python -m compileall -q backend
cargo check --manifest-path tauri/src-tauri/Cargo.toml
```

Run focused tests for the code you change. Release changes must follow
[`docs/ALPHA_RELEASE.md`](docs/ALPHA_RELEASE.md) and verify the frozen executable, not only the
source server.

## Pull requests

- Keep one concern per change.
- Explain user-visible behavior and verification evidence.
- Update architecture or release documentation when a contract changes.
- Never commit models, caches, media, transcripts, databases, logs, virtual environments, or built
  installers.
- Preserve compatibility names only where migration would break existing installations.

The GitHub repository uses `main` as its sole long-lived branch.

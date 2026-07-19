# Diarix architecture context

## Contracts

The Tauri process owns exactly one local server process. The compact CPU sidecar is always
available; a compatible adjacent CUDA backend may be selected at startup. All UI features use the
same HTTP task and model services regardless of the selected backend.

Media enters through the central FFprobe/FFmpeg service. Every STT adapter declares its normalized
audio contract and receives prepared media. Originals remain untouched and are retained only when
the capture/history contract requires them.

Model identity, download status, cache location, supported languages, precision, and runtime family
come from the backend catalog. The frontend may filter or present that metadata but must not create
a second catalog.

## Architecture

`app/` talks to `backend/` through the typed API client. `tauri/` starts the sidecar, controls native
capture and global shortcuts, and surfaces lightweight tray state. Long work flows through the
shared task queue, progress manager, cancellation checks, inference lock, and capture persistence.

The compact server and CUDA server are distribution variants of the same backend code. They are not
separate applications and must remain interchangeable against the same data and cache directories.

## Release boundary

The Windows alpha contains the desktop executable, a compact sidecar, FFmpeg/FFprobe, and—depending
on edition—an adjacent optional runtime or starter model. Models are user data and normally download
after installation. Installer payloads live outside the repository or under ignored `artifacts/`.

The inherited `landing/`, `web/`, and documentation-site sources are not part of the alpha build.
Compatibility names such as legacy environment variables or database filenames may remain where
changing them would break existing installations.

## Pitfalls

- A percentage that jumps from a stage boundary is not inference progress.
- A downloaded checkpoint can serve multiple adapters; UI rows must explain shared cache state.
- Switching backend variants must not fork user data or create a second cache.
- Any server executable version mismatch causes native startup rejection; version drift is a release
  blocker.

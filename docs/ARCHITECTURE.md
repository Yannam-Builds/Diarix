# Architecture

Diarix is one native desktop application with one local backend contract. CPU, native Whisper, and
CUDA distributions are interchangeable runtime packages—not separate apps.

```mermaid
flowchart LR
    UI["React app"] --> API["Local Diarix API"]
    Native["Tauri: startup, tray, PTT"] --> UI
    Native --> API
    API --> Tasks["Tasks, progress, cancellation"]
    API --> Media["FFprobe + FFmpeg"]
    API --> Catalog["Model catalog + shared cache"]
    Tasks --> STT["STT adapters"]
    Tasks --> TTS["TTS adapters"]
    Tasks --> Refine["Local refinement"]
    STT --> Captures["Captures + exports"]
```

## Source map

| Path | Responsibility |
|---|---|
| `app/` | Product UI, typed API client, transcription dashboard, history and settings |
| `backend/` | API, media ingestion, models, downloads, inference, tasks and persistence |
| `tauri/` | Native lifecycle, sidecar selection, audio capture, global shortcuts and tray |
| `installer/` | Windows payload assembly and Inno Setup packaging |
| `scripts/` | Cross-workspace verification, code generation and release utilities |

## Invariants

- Every imported media file is inspected centrally and normalized to the selected adapter's declared
  input contract.
- The task queue owns long-running inference state; UI state is recoverable from the backend.
- Cancellation is cooperative inside chunked runtimes and always converges to model cleanup.
- Download state comes from physical cache inspection, including shared checkpoints.
- Production startup never depends on Vite, Codex, a visible console, or a separately launched
  Python process.

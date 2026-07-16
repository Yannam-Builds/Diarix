# Product

## Register

product

## Users

Diarix serves people who transcribe local audio and video, create speech, and manage voice workflows on a desktop workstation. Their primary job is to turn one or many media files into reliable transcripts with visible progress and predictable outputs, while keeping TTS, voice profiles, stories, history, models, and MCP tools close at hand.

## Product Purpose

Diarix is a transcription-first desktop speech studio built directly on Voicebox. It combines local media ingestion, multiple speech-to-text engines, Voicebox voice generation, and shared model/runtime management in one native Tauri application. Success means transcription is the immediate default workflow, every engine uses the same task, progress, cancellation, cache, download, and CUDA-server architecture, and existing Voicebox capabilities remain intact in clearly separated sections.

## Brand Personality

Restrained, capable, and focused. Diarix should feel like a dependable native workstation: quiet while idle, precise while working, and expressive only where speech motion or progress communicates useful state.

## Anti-references

Diarix must not resemble a marketing hero, a generic AI dashboard, a separate wrapper around Voicebox, or a decorative glass-and-gradient concept app. Avoid ornamental blobs and orbs, gratuitous cards, unfamiliar controls, staged page-load choreography, and any speaker-diarization UI.

## Design Principles

1. Put transcription one action away: file selection, model requirements, output choices, progress, and recent results belong in the primary workspace.
2. Extend Voicebox instead of duplicating it: one shell, one backend, one model lifecycle, and one interaction vocabulary.
3. Make system state legible: show the real file, stage, percentage, cancellation state, model readiness, and output location.
4. Let speech motion explain continuity: approved wave fields guide attention toward the primary action and settle or stop when motion is unnecessary.
5. Preserve user media and local control: originals remain untouched, temporary normalized media is bounded and cleaned up, and storage follows the selected Voicebox cache root.

## Accessibility & Inclusion

Maintain keyboard-accessible native/Radix interaction patterns, clear focus and disabled states, readable WCAG AA contrast, and status that does not rely on color alone. Every animation must respect `prefers-reduced-motion`, pause when hidden, and leave the workflow fully understandable without motion.

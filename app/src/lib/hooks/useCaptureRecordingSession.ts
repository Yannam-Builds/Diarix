import { useMutation, useQueryClient } from '@tanstack/react-query';
import { emit as tauriEmit } from '@tauri-apps/api/event';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { PillState } from '@/components/CapturePill/CapturePill';
import { apiClient } from '@/lib/api/client';
import type {
  CaptureCreateResponse,
  CaptureListResponse,
  CaptureResponse,
  CaptureSource,
} from '@/lib/api/types';
import { useAudioRecording } from '@/lib/hooks/useAudioRecording';
import { LiveCaptureTransport } from '@/lib/liveCapture';

/**
 * Broadcast to sibling Tauri webviews that the captures list has changed.
 * The main CapturesTab listens, seeds its React Query cache, and focuses the
 * new row, so uploads from the floating dictate window show up live.
 *
 * ``capture:created`` carries the full response so the sibling can seed its
 * cache before the refetch lands — otherwise the selection-guard effect
 * would snap back to ``captures[0]`` in the race window between
 * ``setSelectedId(new)`` and the list actually containing the new row.
 *
 * No-op in web mode — there are no siblings to notify.
 */
function broadcastCreated(capture: CaptureResponse) {
  tauriEmit('capture:created', { capture }).catch(() => {
    /* not running inside Tauri; nothing to sync to */
  });
}

function broadcastUpdated(id: string) {
  tauriEmit('capture:updated', { id }).catch(() => {
    /* not running inside Tauri; nothing to sync to */
  });
}

const REST_FADE_MS = 900;
// How long the green "Done" pill stays visible after refine (or transcribe,
// when auto-refine is off) completes, before the fade-out begins.
const COMPLETED_DWELL_MS = 2000;
// Long enough to read a full backend stack message and click-to-copy.
const ERROR_PILL_VISIBLE_MS = 6000;
// Short self-explanatory notices (e.g. "Recording too short, canceled") —
// there's nothing to read or copy, so clear out quickly.
const BRIEF_NOTICE_MS = 2000;
// MediaRecorder.start(100) emits its first chunk ~100ms in, but the webm
// container header isn't guaranteed to be finalised that quickly — anything
// under half a second tends to produce a blob neither AudioContext.decode
// nor ffmpeg will accept. Caught client-side and surfaced as a friendly
// "Recording too short, canceled" pill instead of bubbling up a 400.
const MIN_RECORDING_DURATION_S = 0.5;
const SHORT_RECORDING_MESSAGE = 'Recording too short, canceled';

function createOperationId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `capture-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export type CapturePillState = PillState | 'hidden';

export interface UseCaptureRecordingSessionOptions {
  /**
   * Fired after a capture row is created on the server. Callers can use this
   * to select the new capture or emit a Tauri event to a sibling window.
   */
  onCaptureCreated?: (capture: CaptureResponse) => void;
  /**
   * Fired with the final delivered text — refined if ``auto_refine`` was on
   * for this capture, raw transcript otherwise. Used by the floating
   * dictate window to hand the text off to the Rust auto-paste pipeline.
   *
   * ``allowAutoPaste`` snapshots the setting at chord-start so a refine that
   * lands after the user flips the toggle still uses the value the capture
   * was created under.
   */
  onFinalText?: (
    text: string,
    capture: CaptureResponse,
    allowAutoPaste: boolean,
  ) => void;
}

export interface UseCaptureRecordingSessionResult {
  pillState: CapturePillState;
  pillElapsedMs: number;
  errorMessage: string | null;
  isRecording: boolean;
  isUploading: boolean;
  isRefining: boolean;
  liveTranscript: string;
  startRecording: () => void;
  stopRecording: () => void;
  cancelCurrent: () => void;
  toggleRecording: () => void;
  dismissError: () => void;
  uploadFile: (file: File, source: CaptureSource) => void;
  refine: (captureId: string) => void;
}

type LiveCaptureStatus =
  | 'connecting'
  | 'active'
  | 'fallback'
  | 'succeeded'
  | 'cancelled';

interface PendingLiveCapture {
  operationId: string;
  transport: LiveCaptureTransport | null;
  status: LiveCaptureStatus;
  fallbackFile: File | null;
  fallbackStarted: boolean;
  recordingComplete: boolean;
}

/**
 * Owns the full record → transcribe → refine → rest lifecycle behind the
 * capture pill. The pill component and the Dictate/Stop button are the only
 * consumers; everything else (cache seeding, error toasts, settings reads) is
 * internal so the hook can be reused from a floating Tauri window without the
 * containing tab.
 */
export function useCaptureRecordingSession(
  options: UseCaptureRecordingSessionOptions = {},
): UseCaptureRecordingSessionResult {
  const queryClient = useQueryClient();
  // Every capture setting is resolved server-side. ``stt_model``,
  // ``llm_model`` and refine flags are read from the capture_settings table
  // inside POST /captures and /captures/*/refine, and ``auto_refine`` comes
  // back on the create response so the client decides whether to chain a
  // refine call using a value that can't go stale across sibling webviews.

  const [pillState, setPillState] = useState<CapturePillState>('hidden');
  const [frozenElapsedMs, setFrozenElapsedMs] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [liveTranscript, setLiveTranscript] = useState('');
  const restTimerRef = useRef<number | null>(null);
  const errorTimerRef = useRef<number | null>(null);
  const activeOperationIdRef = useRef<string | null>(null);
  const cancelledOperationIdsRef = useRef(new Set<string>());
  const pendingLiveCaptureRef = useRef<PendingLiveCapture | null>(null);

  // Mutation callbacks close over stale pillState otherwise.
  const pillStateRef = useRef<CapturePillState>('hidden');
  pillStateRef.current = pillState;

  const onCaptureCreatedRef = useRef(options.onCaptureCreated);
  onCaptureCreatedRef.current = options.onCaptureCreated;

  const onFinalTextRef = useRef(options.onFinalText);
  onFinalTextRef.current = options.onFinalText;

  // Snapshot of ``allow_auto_paste`` from the capture-create response —
  // held so the refine onSuccess (which only sees the plain CaptureResponse)
  // can still pass the original setting through to onFinalText.
  const allowAutoPasteRef = useRef<boolean>(true);

  const clearRestTimer = useCallback(() => {
    if (restTimerRef.current !== null) {
      window.clearTimeout(restTimerRef.current);
      restTimerRef.current = null;
    }
  }, []);

  const clearErrorTimer = useCallback(() => {
    if (errorTimerRef.current !== null) {
      window.clearTimeout(errorTimerRef.current);
      errorTimerRef.current = null;
    }
  }, []);

  const scheduleHidePill = useCallback(() => {
    clearRestTimer();
    setPillState('completed');
    // Two-hop timer: show the green "Done" pill for COMPLETED_DWELL_MS,
    // then hand off to the existing rest-fade before unmounting.
    restTimerRef.current = window.setTimeout(() => {
      setPillState('rest');
      restTimerRef.current = window.setTimeout(() => {
        setPillState('hidden');
        restTimerRef.current = null;
      }, REST_FADE_MS);
    }, COMPLETED_DWELL_MS);
  }, [clearRestTimer]);

  const showError = useCallback(
    (message: string, durationMs: number = ERROR_PILL_VISIBLE_MS) => {
      clearRestTimer();
      clearErrorTimer();
      setErrorMessage(message || 'Something went wrong');
      setPillState('error');
      errorTimerRef.current = window.setTimeout(() => {
        setPillState('hidden');
        setErrorMessage(null);
        errorTimerRef.current = null;
      }, durationMs);
    },
    [clearRestTimer, clearErrorTimer],
  );

  const dismissError = useCallback(() => {
    clearErrorTimer();
    setPillState('hidden');
    setErrorMessage(null);
  }, [clearErrorTimer]);

  useEffect(
    () => () => {
      clearRestTimer();
      clearErrorTimer();
    },
    [clearRestTimer, clearErrorTimer],
  );

  const refineMutation = useMutation({
    // Empty body — backend resolves flags and model from capture_settings.
    mutationFn: async ({
      captureId,
      operationId,
    }: {
      captureId: string;
      operationId: string;
    }) => apiClient.refineCapture(captureId, {}, operationId),
    onSuccess: (data, { captureId, operationId }) => {
      if (cancelledOperationIdsRef.current.delete(operationId)) {
        if (activeOperationIdRef.current === operationId) {
          activeOperationIdRef.current = null;
        }
        return;
      }
      if (activeOperationIdRef.current === operationId) {
        activeOperationIdRef.current = null;
      }
      queryClient.invalidateQueries({ queryKey: ['captures'] });
      broadcastUpdated(captureId);
      if (pillStateRef.current === 'refining') scheduleHidePill();
      const finalText = data.transcript_refined ?? data.transcript_raw;
      if (finalText) {
        onFinalTextRef.current?.(finalText, data, allowAutoPasteRef.current);
      }
    },
    onError: (err: Error, { operationId }) => {
      if (cancelledOperationIdsRef.current.delete(operationId)) {
        if (activeOperationIdRef.current === operationId) {
          activeOperationIdRef.current = null;
        }
        return;
      }
      if (activeOperationIdRef.current === operationId) {
        activeOperationIdRef.current = null;
      }
      showError(err.message || 'Refinement failed');
    },
  });

  function handleCaptureSuccess(
    capture: CaptureCreateResponse,
    operationId: string,
  ) {
    if (cancelledOperationIdsRef.current.delete(operationId)) {
      if (activeOperationIdRef.current === operationId) {
        activeOperationIdRef.current = null;
      }
      return;
    }
    queryClient.setQueryData<CaptureListResponse>(['captures'], (prev) => {
      if (!prev) return prev;
      if (prev.items.some((item) => item.id === capture.id)) return prev;
      return { ...prev, items: [capture, ...prev.items], total: prev.total + 1 };
    });
    queryClient.invalidateQueries({ queryKey: ['captures'] });
    broadcastCreated(capture);
    onCaptureCreatedRef.current?.(capture);
    allowAutoPasteRef.current = capture.allow_auto_paste;
    if (capture.auto_refine) {
      const refineOperationId = createOperationId();
      activeOperationIdRef.current = refineOperationId;
      setPillState('refining');
      refineMutation.mutate({
        captureId: capture.id,
        operationId: refineOperationId,
      });
    } else {
      if (activeOperationIdRef.current === operationId) {
        activeOperationIdRef.current = null;
      }
      if (pillStateRef.current === 'transcribing') scheduleHidePill();
      if (capture.transcript_raw) {
        onFinalTextRef.current?.(
          capture.transcript_raw,
          capture,
          capture.allow_auto_paste,
        );
      }
    }
  }

  const uploadMutation = useMutation({
    mutationFn: async ({
      file,
      source,
      operationId,
    }: {
      file: File;
      source: CaptureSource;
      operationId: string;
    }) => apiClient.createCapture(file, { source, operationId }),
    onSuccess: (capture, { operationId }) => {
      const pending = pendingLiveCaptureRef.current;
      if (pending?.operationId === operationId) {
        pendingLiveCaptureRef.current = null;
      }
      handleCaptureSuccess(capture, operationId);
    },
    onError: (err: Error, { operationId }) => {
      const pending = pendingLiveCaptureRef.current;
      if (pending?.operationId === operationId) {
        pendingLiveCaptureRef.current = null;
      }
      if (cancelledOperationIdsRef.current.delete(operationId)) {
        if (activeOperationIdRef.current === operationId) {
          activeOperationIdRef.current = null;
        }
        return;
      }
      setLiveTranscript('');
      if (activeOperationIdRef.current === operationId) {
        activeOperationIdRef.current = null;
      }
      // Backend's librosa-audioread fallback returns a 400 with this shape
      // for tiny/corrupt webm blobs that slip past the client guard —
      // translate it to the same friendly message so the user sees one
      // consistent cause, not an opaque decode error.
      const msg = err.message || '';
      if (/could not decode/i.test(msg) || /empty or corrupt/i.test(msg)) {
        showError(SHORT_RECORDING_MESSAGE, BRIEF_NOTICE_MS);
      } else {
        showError(msg || 'Upload failed');
      }
    },
  });

  function startFallbackUpload(pending: PendingLiveCapture) {
    if (
      pending.fallbackStarted ||
      pending.status !== 'fallback' ||
      !pending.fallbackFile
    ) {
      return;
    }
    pending.fallbackStarted = true;
    uploadMutation.mutate({
      file: pending.fallbackFile,
      source: 'dictation',
      operationId: pending.operationId,
    });
  }

  const {
    isRecording,
    duration,
    startRecording: beginAudioRecording,
    stopRecording,
    cancelRecording,
    error: recordError,
  } = useAudioRecording({
    onRecordingComplete: (blob, recordedDuration) => {
      // Trigger-happy tap — MediaRecorder hasn't emitted a usable chunk yet
      // so the blob is empty or unparseable. Surface it as a transient pill
      // so the user sees their recording was recognised and canceled.
      if (!blob.size || (recordedDuration ?? 0) < MIN_RECORDING_DURATION_S) {
        const pending = pendingLiveCaptureRef.current;
        if (pending) {
          pending.status = 'cancelled';
          pending.transport?.cancel();
          pendingLiveCaptureRef.current = null;
          if (activeOperationIdRef.current === pending.operationId) {
            activeOperationIdRef.current = null;
          }
          void apiClient.cancelCaptureOperation(pending.operationId).catch(() => {});
        }
        setLiveTranscript('');
        showError(SHORT_RECORDING_MESSAGE, BRIEF_NOTICE_MS);
        return;
      }
      setFrozenElapsedMs(Math.round((recordedDuration ?? 0) * 1000));
      setPillState('transcribing');
      const extension = blob.type.includes('wav')
        ? 'wav'
        : blob.type.includes('webm')
          ? 'webm'
          : 'bin';
      const file = new File([blob], `dictation-${Date.now()}.${extension}`, {
        type: blob.type,
      });
      const pending = pendingLiveCaptureRef.current;
      if (pending) {
        pending.recordingComplete = true;
        pending.fallbackFile = file;
        if (pending.status === 'fallback') {
          startFallbackUpload(pending);
        } else if (
          pending.status === 'succeeded' ||
          pending.status === 'cancelled'
        ) {
          pendingLiveCaptureRef.current = null;
        }
        return;
      }

      const operationId = createOperationId();
      activeOperationIdRef.current = operationId;
      uploadMutation.mutate({ file, source: 'dictation', operationId });
    },
    onPcmChunk: (pcm, sampleRate) => {
      pendingLiveCaptureRef.current?.transport?.push(pcm, sampleRate);
    },
    onPcmEnd: () => {
      pendingLiveCaptureRef.current?.transport?.stop();
    },
  });

  useEffect(() => {
    if (recordError) {
      const pending = pendingLiveCaptureRef.current;
      if (pending) {
        pending.status = 'cancelled';
        pending.transport?.cancel();
        pendingLiveCaptureRef.current = null;
        if (activeOperationIdRef.current === pending.operationId) {
          activeOperationIdRef.current = null;
        }
      }
      showError(recordError);
    }
  }, [recordError, showError]);

  const startRecording = useCallback(() => {
    // Handy-style single-session coordinator: a second shortcut press while
    // transcription/refinement is still active must not replace the captured
    // focus target or operation id. The tray remains the cancellation surface
    // until the current pipeline returns to idle.
    if (
      activeOperationIdRef.current ||
      isRecording ||
      uploadMutation.isPending ||
      refineMutation.isPending
    ) {
      return;
    }
    clearRestTimer();
    setFrozenElapsedMs(0);
    setLiveTranscript('');
    setPillState('recording');
    const operationId = createOperationId();
    activeOperationIdRef.current = operationId;
    const pending: PendingLiveCapture = {
      operationId,
      transport: null,
      status: 'connecting',
      fallbackFile: null,
      fallbackStarted: false,
      recordingComplete: false,
    };
    pendingLiveCaptureRef.current = pending;
    try {
      pending.transport = new LiveCaptureTransport(
        apiClient.getLiveCaptureWebSocketUrl(),
        operationId,
        {
          onReady: () => {
            if (pendingLiveCaptureRef.current === pending) {
              pending.status = 'active';
            }
          },
          onPartial: (text) => {
            if (pendingLiveCaptureRef.current === pending) {
              setLiveTranscript(text);
            }
          },
          onFinal: (capture, text) => {
            if (pendingLiveCaptureRef.current !== pending) return;
            pending.status = 'succeeded';
            setLiveTranscript(text);
            handleCaptureSuccess(capture, operationId);
            if (pending.recordingComplete) {
              pendingLiveCaptureRef.current = null;
            }
          },
          onUnavailable: (reason) => {
            if (pendingLiveCaptureRef.current !== pending) return;
            console.info('[dictation] using completed-audio fallback:', reason);
            pending.status = 'fallback';
            setLiveTranscript('');
            startFallbackUpload(pending);
          },
        },
      );
    } catch (error) {
      console.warn('[dictation] failed to start live transport:', error);
      pending.status = 'fallback';
    }
    // Begin loading the selected model while the user speaks. The capture
    // request later shares the same backend lifecycle lock, so it either
    // reuses the warm model or waits for this one load instead of starting a
    // duplicate.
    void apiClient.warmCaptureModel().catch((error) => {
      console.warn('[dictation] model warm-up failed:', error);
    });
    beginAudioRecording();
  }, [
    isRecording,
    uploadMutation.isPending,
    refineMutation.isPending,
    beginAudioRecording,
    clearRestTimer,
  ]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
      return;
    }
    startRecording();
  }, [isRecording, startRecording, stopRecording]);

  const cancelCurrent = useCallback(() => {
    clearRestTimer();
    clearErrorTimer();
    const pending = pendingLiveCaptureRef.current;
    if (pending) {
      pending.status = 'cancelled';
      pending.transport?.cancel();
      pendingLiveCaptureRef.current = null;
    }
    cancelRecording();
    const operationId = activeOperationIdRef.current;
    activeOperationIdRef.current = null;
    if (operationId) {
      cancelledOperationIdsRef.current.add(operationId);
      void apiClient.cancelCaptureOperation(operationId).catch((error) => {
        console.warn('[dictation] failed to cancel backend operation:', error);
      });
    }
    setFrozenElapsedMs(0);
    setLiveTranscript('');
    setErrorMessage(null);
    setPillState('hidden');
  }, [cancelRecording, clearErrorTimer, clearRestTimer]);

  const uploadFile = useCallback(
    (file: File, source: CaptureSource) => {
      const operationId = createOperationId();
      activeOperationIdRef.current = operationId;
      uploadMutation.mutate({ file, source, operationId });
    },
    [uploadMutation],
  );

  const refine = useCallback(
    (captureId: string) => {
      const operationId = createOperationId();
      activeOperationIdRef.current = operationId;
      refineMutation.mutate({ captureId, operationId });
    },
    [refineMutation],
  );

  const pillElapsedMs =
    pillState === 'recording' ? Math.round(duration * 1000) : frozenElapsedMs;

  return {
    pillState,
    pillElapsedMs,
    errorMessage,
    isRecording,
    isUploading: uploadMutation.isPending,
    isRefining: refineMutation.isPending,
    liveTranscript,
    startRecording,
    stopRecording,
    cancelCurrent,
    toggleRecording,
    dismissError,
    uploadFile,
    refine,
  };
}

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';
import { convertToWav } from '@/lib/utils/audio';

interface UseAudioRecordingOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
  onPcmChunk?: (pcm: Float32Array, sampleRate: number) => void;
  onPcmEnd?: () => void;
}

export function useAudioRecording({
  maxDurationSeconds,
  onRecordingComplete,
  onPcmChunk,
  onPcmEnd,
}: UseAudioRecordingOptions = {}) {
  const platform = usePlatform();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const cancelledRef = useRef<boolean>(false);
  const startingRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioWorkletRef = useRef<AudioWorkletNode | null>(null);
  const audioSinkRef = useRef<GainNode | null>(null);
  const onPcmChunkRef = useRef(onPcmChunk);
  const onPcmEndRef = useRef(onPcmEnd);
  onPcmChunkRef.current = onPcmChunk;
  onPcmEndRef.current = onPcmEnd;

  const stopPcmCapture = useCallback((notifyEnd: boolean) => {
    const worklet = audioWorkletRef.current;
    if (worklet) {
      worklet.port.onmessage = null;
      worklet.disconnect();
    }
    audioSourceRef.current?.disconnect();
    audioSinkRef.current?.disconnect();
    audioWorkletRef.current = null;
    audioSourceRef.current = null;
    audioSinkRef.current = null;

    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context && context.state !== 'closed') {
      void context.close();
    }
    if (notifyEnd) onPcmEndRef.current?.();
  }, []);

  const startRecording = useCallback(async () => {
    if (
      startingRef.current ||
      (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive')
    ) {
      return;
    }
    startingRef.current = true;
    try {
      setError(null);
      chunksRef.current = [];
      cancelledRef.current = false;
      setDuration(0);

      // Check if getUserMedia is available
      // In Tauri, navigator.mediaDevices might not be available immediately
      if (typeof navigator === 'undefined') {
        const errorMsg =
          'Navigator API is not available. This might be a Tauri configuration issue.';
        setError(errorMsg);
        throw new Error(errorMsg);
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        // Try waiting a bit for Tauri webview to initialize
        await new Promise((resolve) => setTimeout(resolve, 100));

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          console.error('MediaDevices check:', {
            hasNavigator: typeof navigator !== 'undefined',
            hasMediaDevices: !!navigator?.mediaDevices,
            hasGetUserMedia: !!navigator?.mediaDevices?.getUserMedia,
            isTauri: platform.metadata.isTauri,
          });

          const errorMsg = platform.metadata.isTauri
            ? 'Microphone access is not available. Please ensure:\n1. The app has microphone permissions in System Settings (macOS: System Settings > Privacy & Security > Microphone)\n2. You restart the app after granting permissions\n3. You are using Tauri v2 with a webview that supports getUserMedia'
            : 'Microphone access is not available. Please ensure you are using a secure context (HTTPS or localhost) and that your browser has microphone permissions enabled.';
          setError(errorMsg);
          throw new Error(errorMsg);
        }
      }

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;

      // Capture raw microphone frames alongside MediaRecorder. The worklet
      // never replaces the complete WAV fallback; it supplies low-latency PCM
      // only when the selected backend confirms true streaming support.
      try {
        const context = new AudioContext();
        await context.audioWorklet.addModule('/pcm-capture-worklet.js');
        const sourceNode = context.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(context, 'diarix-pcm-capture');
        const silentSink = context.createGain();
        silentSink.gain.value = 0;
        workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
          const pcm =
            event.data instanceof Float32Array
              ? event.data
              : new Float32Array(event.data);
          onPcmChunkRef.current?.(pcm, context.sampleRate);
        };
        sourceNode.connect(workletNode);
        workletNode.connect(silentSink);
        silentSink.connect(context.destination);
        await context.resume();
        audioContextRef.current = context;
        audioSourceRef.current = sourceNode;
        audioWorkletRef.current = workletNode;
        audioSinkRef.current = silentSink;
      } catch (liveError) {
        console.warn('[dictation] raw PCM capture unavailable:', liveError);
        stopPcmCapture(false);
      }

      // Create MediaRecorder with preferred MIME type
      const options: MediaRecorderOptions = {
        mimeType: 'audio/webm;codecs=opus',
      };

      // Fallback to default if webm not supported
      if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
        delete options.mimeType;
      }

      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Snapshot the cancellation flag and recorded duration immediately —
        // cancelRecording() clears chunks and sets cancelledRef synchronously
        // before this async handler runs, so we must check it first.
        const wasCancelled = cancelledRef.current;
        const recordedDuration = startTimeRef.current
          ? (Date.now() - startTimeRef.current) / 1000
          : undefined;

        const webmBlob = new Blob(chunksRef.current, { type: 'audio/webm' });

        // Stop all tracks now that we have the data
        streamRef.current?.getTracks().forEach((track) => {
          track.stop();
        });
        streamRef.current = null;

        // Don't fire completion callback if the recording was cancelled
        if (wasCancelled) return;

        // Convert to WAV format to avoid needing ffmpeg on backend
        try {
          const wavBlob = await convertToWav(webmBlob);
          onRecordingComplete?.(wavBlob, recordedDuration);
        } catch (err) {
          console.error('Error converting audio to WAV:', err);
          // Fallback to original blob if conversion fails
          onRecordingComplete?.(webmBlob, recordedDuration);
        }
      };

      mediaRecorder.onerror = (event) => {
        setError('Recording error occurred');
        console.error('MediaRecorder error:', event);
      };

      // WebKit's MediaRecorder drops the WebM EBML header from chunks when
      // started with a timeslice, so concatenated blobs fail to parse in
      // both AudioContext and ffmpeg. Starting with no timeslice produces
      // exactly one dataavailable on stop() with a valid container.
      mediaRecorder.start();
      setIsRecording(true);
      startTimeRef.current = Date.now();

      // Start timer
      timerRef.current = window.setInterval(() => {
        if (startTimeRef.current) {
          const elapsed = (Date.now() - startTimeRef.current) / 1000;
          setDuration(elapsed);

          // Auto-stop at max duration when the caller opts in — dictation
          // sessions pass undefined and run until the user releases the
          // chord or hits stop; voice-clone sample recorders pass 29s to
          // keep reference clips short.
          if (maxDurationSeconds !== undefined && elapsed >= maxDurationSeconds) {
            if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
              stopPcmCapture(true);
              mediaRecorderRef.current.stop();
              setIsRecording(false);
              if (timerRef.current !== null) {
                clearInterval(timerRef.current);
                timerRef.current = null;
              }
            }
          }
        }
      }, 100);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Failed to access microphone. Please check permissions.';
      setError(errorMessage);
      setIsRecording(false);
    } finally {
      startingRef.current = false;
    }
  }, [maxDurationSeconds, onRecordingComplete, stopPcmCapture]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      stopPcmCapture(true);
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [isRecording, stopPcmCapture]);

  const cancelRecording = useCallback(() => {
    stopPcmCapture(false);
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      cancelledRef.current = true; // Must be set before stop() triggers onstop
      chunksRef.current = [];
      recorder.stop();
      setIsRecording(false);
      setDuration(0);
    }

    // Stop all tracks
    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    streamRef.current = null;

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, [stopPcmCapture]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      stopPcmCapture(false);
      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
    };
  }, [stopPcmCapture]);

  return {
    isRecording,
    duration,
    error,
    startRecording,
    stopRecording,
    cancelRecording,
  };
}

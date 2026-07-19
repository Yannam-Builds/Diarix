import type { CaptureCreateResponse } from '@/lib/api/types';

const TARGET_SAMPLE_RATE = 16_000;
const PACKET_SAMPLES = 1_600;
const READY_TIMEOUT_MS = 60_000;

export interface LiveCaptureCallbacks {
  onReady?: () => void;
  onPartial?: (text: string) => void;
  onFinal: (capture: CaptureCreateResponse, text: string) => void;
  onUnavailable: (reason: string) => void;
}

class StreamingResampler {
  private sourceRate = 0;
  private carry = new Float32Array(0);
  private position = 0;

  push(input: Float32Array, sourceRate: number): Float32Array {
    if (!input.length) return new Float32Array(0);
    if (sourceRate <= 0) throw new Error('Invalid microphone sample rate');
    if (this.sourceRate && this.sourceRate !== sourceRate) {
      this.reset();
    }
    this.sourceRate = sourceRate;

    const combined = new Float32Array(this.carry.length + input.length);
    combined.set(this.carry);
    combined.set(input, this.carry.length);
    const ratio = sourceRate / TARGET_SAMPLE_RATE;
    const output: number[] = [];

    while (this.position + 1 < combined.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      const sample =
        combined[left] + (combined[left + 1] - combined[left]) * fraction;
      output.push(sample);
      this.position += ratio;
    }

    const consumed = Math.floor(this.position);
    this.carry = combined.slice(Math.min(consumed, combined.length));
    this.position -= consumed;
    return Float32Array.from(output);
  }

  flush(): Float32Array {
    if (!this.carry.length) return new Float32Array(0);
    const finalSample = new Float32Array([this.carry[this.carry.length - 1]]);
    this.reset();
    return finalSample;
  }

  reset() {
    this.sourceRate = 0;
    this.carry = new Float32Array(0);
    this.position = 0;
  }
}

export class LiveCaptureTransport {
  private readonly socket: WebSocket;
  private readonly callbacks: LiveCaptureCallbacks;
  private readonly resampler = new StreamingResampler();
  private readonly queuedPackets: Float32Array[] = [];
  private packetBuffer: number[] = [];
  private ready = false;
  private stopped = false;
  private terminal = false;
  private timeoutId: number | null = null;

  constructor(
    websocketUrl: string,
    operationId: string,
    callbacks: LiveCaptureCallbacks,
  ) {
    this.callbacks = callbacks;
    this.socket = new WebSocket(websocketUrl);
    this.socket.binaryType = 'arraybuffer';
    this.socket.onopen = () => {
      this.socket.send(JSON.stringify({ type: 'start', operation_id: operationId }));
    };
    this.socket.onmessage = (event) => this.handleMessage(event);
    this.socket.onerror = () => {
      this.markUnavailable('Live dictation connection failed');
    };
    this.socket.onclose = () => {
      if (!this.terminal) {
        this.markUnavailable('Live dictation connection closed');
      }
    };
    this.timeoutId = window.setTimeout(() => {
      this.timeoutId = null;
      this.markUnavailable('Live dictation model did not become ready in time');
    }, READY_TIMEOUT_MS);
  }

  push(input: Float32Array, sourceRate: number) {
    if (this.stopped || this.terminal) return;
    this.packetize(this.resampler.push(input, sourceRate));
  }

  stop() {
    if (this.stopped || this.terminal) return;
    this.stopped = true;
    this.packetize(this.resampler.flush());
    this.flushPacketBuffer();
    this.flushQueuedPackets();
    if (this.ready && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: 'stop' }));
    }
  }

  cancel() {
    if (this.terminal) return;
    this.terminal = true;
    this.clearTimeout();
    this.queuedPackets.length = 0;
    this.packetBuffer = [];
    this.resampler.reset();
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: 'cancel' }));
      window.setTimeout(() => this.socket.close(), 100);
    } else {
      this.socket.close();
    }
  }

  private packetize(samples: Float32Array) {
    for (let index = 0; index < samples.length; index += 1) {
      this.packetBuffer.push(samples[index]);
      if (this.packetBuffer.length === PACKET_SAMPLES) {
        this.enqueuePacket(Float32Array.from(this.packetBuffer));
        this.packetBuffer = [];
      }
    }
  }

  private flushPacketBuffer() {
    if (!this.packetBuffer.length) return;
    this.enqueuePacket(Float32Array.from(this.packetBuffer));
    this.packetBuffer = [];
  }

  private enqueuePacket(packet: Float32Array) {
    if (
      this.ready &&
      !this.terminal &&
      this.socket.readyState === WebSocket.OPEN
    ) {
      this.socket.send(packet);
      return;
    }
    this.queuedPackets.push(packet);
  }

  private flushQueuedPackets() {
    if (
      !this.ready ||
      this.terminal ||
      this.socket.readyState !== WebSocket.OPEN
    ) {
      return;
    }
    for (const packet of this.queuedPackets.splice(0)) {
      this.socket.send(packet);
    }
  }

  private handleMessage(event: MessageEvent) {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(String(event.data)) as Record<string, unknown>;
    } catch {
      return;
    }

    const kind = String(message.type ?? '');
    if (kind === 'ready') {
      this.ready = true;
      this.clearTimeout();
      this.flushQueuedPackets();
      this.callbacks.onReady?.();
      if (this.stopped && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ type: 'stop' }));
      }
      return;
    }
    if (kind === 'partial') {
      this.callbacks.onPartial?.(String(message.full ?? ''));
      return;
    }
    if (kind === 'final') {
      this.terminal = true;
      this.clearTimeout();
      this.callbacks.onFinal(
        message.capture as unknown as CaptureCreateResponse,
        String(message.full ?? ''),
      );
      this.socket.close();
      return;
    }
    if (kind === 'cancelled') {
      this.terminal = true;
      this.clearTimeout();
      this.socket.close();
      return;
    }
    if (kind === 'unsupported') {
      this.markUnavailable(String(message.reason ?? 'Live dictation is unavailable'));
      return;
    }
    if (kind === 'error') {
      this.markUnavailable(String(message.message ?? 'Live dictation failed'));
    }
  }

  private markUnavailable(reason: string) {
    if (this.terminal) return;
    this.terminal = true;
    this.clearTimeout();
    this.queuedPackets.length = 0;
    this.packetBuffer = [];
    this.resampler.reset();
    this.callbacks.onUnavailable(reason);
    this.socket.close();
  }

  private clearTimeout() {
    if (this.timeoutId !== null) {
      window.clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
}

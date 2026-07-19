class DiarixPcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channels = inputs[0];
    if (!channels || channels.length === 0 || channels[0].length === 0) {
      return true;
    }

    const frameCount = channels[0].length;
    const mono = new Float32Array(frameCount);
    for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
      const channel = channels[channelIndex];
      for (let index = 0; index < frameCount; index += 1) {
        mono[index] += channel[index] / channels.length;
      }
    }

    this.port.postMessage(mono, [mono.buffer]);
    return true;
  }
}

registerProcessor('diarix-pcm-capture', DiarixPcmCaptureProcessor);

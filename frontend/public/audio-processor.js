/**
 * AudioWorkletProcessor for MedScribe AI.
 * 
 * Takes input audio at the native AudioContext sample rate (e.g. 44.1 kHz, 48 kHz),
 * resamples it down to 16,000 Hz mono PCM16 using linear interpolation,
 * and posts 100 ms chunks (1,600 samples = 3,200 bytes) back to the main thread.
 */

class MedScribeAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.sourceSampleRate = sampleRate; // Global AudioWorklet sampleRate
    this.resampleRatio = this.sourceSampleRate / this.targetSampleRate;
    
    // Chunk buffer (100 ms at 16 kHz = 1600 samples)
    this.chunkSize = 1600;
    this.pcmBuffer = new Int16Array(this.chunkSize);
    this.pcmIndex = 0;
    
    // Resampler interpolation state
    this.resampleOffset = 0.0;
    this.lastSample = 0.0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channelData = input[0]; // Mono channel
    if (!channelData || channelData.length === 0) {
      return true;
    }

    const inputLen = channelData.length;
    let inputIndex = this.resampleOffset;

    while (inputIndex < inputLen) {
      const idx0 = Math.floor(inputIndex);
      const idx1 = idx0 + 1;
      const frac = inputIndex - idx0;

      const s0 = (idx0 >= 0 && idx0 < inputLen) ? channelData[idx0] : this.lastSample;
      const s1 = (idx1 < inputLen) ? channelData[idx1] : s0;

      // Linear interpolation
      const interpolated = s0 + frac * (s1 - s0);

      // Clamp and convert float32 to PCM16
      const clamped = Math.max(-1.0, Math.min(1.0, interpolated));
      const pcmSample = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
      const safePcm = Math.max(-32768, Math.min(32767, pcmSample));

      this.pcmBuffer[this.pcmIndex++] = safePcm;

      // When 100ms chunk is reached, emit to main thread
      if (this.pcmIndex >= this.chunkSize) {
        // Create an ArrayBuffer copy to transfer
        const chunkToSend = new Int16Array(this.pcmBuffer);
        this.port.postMessage(chunkToSend.buffer, [chunkToSend.buffer]);
        this.pcmIndex = 0;
      }

      inputIndex += this.resampleRatio;
    }

    // Keep leftover fraction for the next block
    this.resampleOffset = inputIndex - inputLen;
    this.lastSample = channelData[inputLen - 1];

    return true;
  }
}

registerProcessor('medscribe-audio-processor', MedScribeAudioProcessor);

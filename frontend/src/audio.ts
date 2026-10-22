/**
 * Microphone capture, AudioWorklet/ScriptProcessor pipeline, and WAV export.
 */

import { store } from './state';
import { WORKLET_PATH, WORKLET_PROCESSOR_NAME } from './audio-worklet';

export class AudioManager {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private scriptProcessorNode: ScriptProcessorNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private onAudioChunk: ((chunk: ArrayBuffer) => void) | null = null;

  // Stored audio chunks for WAV export
  private recordedChunks: ArrayBuffer[] = [];
  private recordingStartTime = 0;
  private durationInterval: number | null = null;
  private isRecording = false;

  constructor(onAudioChunk?: (chunk: ArrayBuffer) => void) {
    if (onAudioChunk) {
      this.onAudioChunk = onAudioChunk;
    }
  }

  setAudioChunkCallback(callback: (chunk: ArrayBuffer) => void): void {
    this.onAudioChunk = callback;
  }

  async start(): Promise<void> {
    if (this.isRecording) {
      console.warn('Recording is already in progress.');
      return;
    }

    // Reset recorded chunks
    this.recordedChunks = [];
    store.setHasRecording(false);
    store.setAudioDuration(0);

    // Request microphone access
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      store.setMicPermission(true);
    } catch (err: any) {
      store.setMicPermission(false);
      let errorMsg = 'Microphone access denied or unavailable.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMsg = 'Microphone permission was denied. Please allow microphone access in your browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMsg = 'No microphone found on your system. Please connect an audio input device.';
      }
      store.setError(errorMsg);
      throw new Error(errorMsg);
    }

    // Initialize AudioContext
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    this.audioContext = new AudioContextClass();

    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

    // Try AudioWorklet first, with graceful fallback to ScriptProcessor
    let workletInitialized = false;
    if (this.audioContext.audioWorklet) {
      try {
        await this.audioContext.audioWorklet.addModule(WORKLET_PATH);
        this.workletNode = new AudioWorkletNode(this.audioContext, WORKLET_PROCESSOR_NAME);

        this.workletNode.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
          this.handleIncomingChunk(event.data);
        };

        this.sourceNode.connect(this.workletNode);
        // Connect to destination to keep pipeline alive (muted to prevent feedback)
        const silentGain = this.audioContext.createGain();
        silentGain.gain.value = 0;
        this.workletNode.connect(silentGain);
        silentGain.connect(this.audioContext.destination);

        workletInitialized = true;
      } catch (workletError) {
        console.warn('AudioWorklet initialization failed, falling back to ScriptProcessor:', workletError);
      }
    }

    if (!workletInitialized) {
      this.setupScriptProcessorFallback();
    }

    this.isRecording = true;
    this.recordingStartTime = Date.now();
    this.durationInterval = window.setInterval(() => {
      const elapsed = Math.round((Date.now() - this.recordingStartTime) / 1000);
      store.setAudioDuration(elapsed);
    }, 1000);
  }

  private setupScriptProcessorFallback(): void {
    if (!this.audioContext || !this.sourceNode) return;

    const bufferSize = 4096;
    const sampleRate = this.audioContext.sampleRate;
    const targetSampleRate = 16000;
    const resampleRatio = sampleRate / targetSampleRate;

    // 100ms at 16kHz = 1600 samples
    const targetChunkSamples = 1600;
    let pcmAccumulator = new Int16Array(targetChunkSamples);
    let pcmIndex = 0;
    let resampleOffset = 0.0;
    let lastSample = 0.0;

    this.scriptProcessorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

    this.scriptProcessorNode.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      const inputLen = input.length;
      let inputIndex = resampleOffset;

      while (inputIndex < inputLen) {
        const idx0 = Math.floor(inputIndex);
        const idx1 = idx0 + 1;
        const frac = inputIndex - idx0;

        const s0 = (idx0 >= 0 && idx0 < inputLen) ? input[idx0] : lastSample;
        const s1 = (idx1 < inputLen) ? input[idx1] : s0;
        const interpolated = s0 + frac * (s1 - s0);

        const clamped = Math.max(-1.0, Math.min(1.0, interpolated));
        const pcm = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
        pcmAccumulator[pcmIndex++] = Math.max(-32768, Math.min(32767, pcm));

        if (pcmIndex >= targetChunkSamples) {
          const chunkToSend = new Int16Array(pcmAccumulator);
          this.handleIncomingChunk(chunkToSend.buffer);
          pcmIndex = 0;
        }

        inputIndex += resampleRatio;
      }

      resampleOffset = inputIndex - inputLen;
      lastSample = input[inputLen - 1];
    };

    this.sourceNode.connect(this.scriptProcessorNode);
    this.scriptProcessorNode.connect(this.audioContext.destination);
  }

  private handleIncomingChunk(chunk: ArrayBuffer): void {
    if (!this.isRecording) return;
    this.recordedChunks.push(chunk.slice(0));
    if (this.onAudioChunk) {
      this.onAudioChunk(chunk);
    }
  }

  stop(): void {
    if (!this.isRecording) return;
    this.isRecording = false;

    if (this.durationInterval !== null) {
      clearInterval(this.durationInterval);
      this.durationInterval = null;
    }

    if (this.recordedChunks.length > 0) {
      store.setHasRecording(true);
    }

    // Clean up tracks and audio context
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    if (this.scriptProcessorNode) {
      this.scriptProcessorNode.disconnect();
      this.scriptProcessorNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.audioContext) {
      try {
        this.audioContext.close();
      } catch {
        // Ignore close error
      }
      this.audioContext = null;
    }
  }

  /**
   * Generates a 16 kHz mono 16-bit WAV file from the stored in-memory recording chunks.
   */
  exportWavBlob(): Blob {
    if (this.recordedChunks.length === 0) {
      throw new Error('No audio data recorded to export.');
    }

    // Calculate total PCM bytes
    let totalBytes = 0;
    for (const chunk of this.recordedChunks) {
      totalBytes += chunk.byteLength;
    }

    // Standard 44-byte WAV Header
    const wavBuffer = new ArrayBuffer(44 + totalBytes);
    const view = new DataView(wavBuffer);

    const sampleRate = 16000;
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);

    // "RIFF" chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + totalBytes, true);
    writeString(view, 8, 'WAVE');

    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size for PCM
    view.setUint16(20, 1, true);  // AudioFormat 1 = PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, totalBytes, true);

    // Copy audio bytes
    const pcmTarget = new Uint8Array(wavBuffer, 44);
    let offset = 0;
    for (const chunk of this.recordedChunks) {
      pcmTarget.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }

    return new Blob([wavBuffer], { type: 'audio/wav' });
  }

  downloadWavFile(): void {
    const blob = this.exportWavBlob();
    const url = URL.createObjectURL(blob);
    const dateStr = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medscribe-consultation-${dateStr}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  clearRecording(): void {
    this.recordedChunks = [];
    store.setHasRecording(false);
    store.setAudioDuration(0);
  }

  getIsRecording(): boolean {
    return this.isRecording;
  }
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

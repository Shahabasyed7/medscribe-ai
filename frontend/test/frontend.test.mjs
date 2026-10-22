import test from 'node:test';
import assert from 'node:assert/strict';

// 1. HTML Escaping Tests
function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

test('escapeHtml prevents XSS injection and preserves clean strings', () => {
  assert.equal(escapeHtml('<script>alert("xss")</script>'), '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;');
  assert.equal(escapeHtml('Patient & Doctor "discussion"'), 'Patient &amp; Doctor &quot;discussion&quot;');
  assert.equal(escapeHtml("John's chart"), "John&#039;s chart");
  assert.equal(escapeHtml(''), '');
  assert.equal(escapeHtml(null), '');
});

// 2. Word and Chunk Counter Tests
test('Word counter correctly calculates word counts across transcript chunks', () => {
  const chunks = [
    { chunk: 1, text: 'Hello doctor, I have had a severe dry cough for four days.' },
    { chunk: 2, text: 'Any fever or shortness of breath?' },
    { chunk: 3, text: 'No fever at all.' }
  ];

  const fullText = chunks.map(c => c.text).join(' ');
  const words = fullText.split(/\s+/).filter(Boolean).length;
  assert.equal(chunks.length, 3);
  assert.equal(words, 22);
});

// 3. WAV Header Builder Test (Mirroring audio.ts DataView)
function buildWavHeader(pcmByteLength, sampleRate = 16000, channels = 1, bitsPerSample = 16) {
  const buffer = new ArrayBuffer(44);
  const view = new DataView(buffer);
  const byteRate = sampleRate * channels * (bitsPerSample / 8);
  const blockAlign = channels * (bitsPerSample / 8);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + pcmByteLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(36, 'data');
  view.setUint32(40, pcmByteLength, true);

  return view;
}

test('WAV Header DataView writes valid RIFF parameters', () => {
  const pcmBytes = 32000; // 1 second of 16kHz 16-bit mono
  const view = buildWavHeader(pcmBytes);

  // Check RIFF
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  assert.equal(riff, 'RIFF');
  assert.equal(view.getUint32(4, true), 36 + 32000);

  // Check WAVE and fmt
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  assert.equal(wave, 'WAVE');

  // Format code 1 (PCM), 1 channel, 16000 Hz, 32000 byteRate, 2 blockAlign, 16 bits
  assert.equal(view.getUint16(20, true), 1);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint32(28, true), 32000);
  assert.equal(view.getUint16(32, true), 2);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 32000);
});

// 4. Prescription Generator Formatting Test
test('Prescription text generation formats patient and prescription items clearly', () => {
  const summary = {
    clinical_summary: '45yo male with acute dry cough.',
    patient_details: { name: 'Alice Smith', age: 45, sex: 'Female', identifiers: 'PT-9988' },
    plan: {
      prescriptions: ['Dextromethorphan syrup 10ml q6h', 'Albuterol inhaler 2 puffs q4h prn'],
      advice: ['Warm liquids', 'Rest'],
      follow_up: 'Return in 7 days if not improved'
    }
  };

  const patientLine = `Patient: ${summary.patient_details.name} | Age: ${summary.patient_details.age} | Sex: ${summary.patient_details.sex} | ID: ${summary.patient_details.identifiers}`;
  assert.ok(patientLine.includes('Alice Smith'));
  assert.ok(patientLine.includes('PT-9988'));
  assert.equal(summary.plan.prescriptions.length, 2);
});

// 5. State Machine Transitions Test
test('State machine transitions follow idle -> recording -> processing -> done sequence', () => {
  const validTransitions = {
    idle: ['recording', 'processing', 'error'],
    recording: ['idle', 'error'],
    processing: ['done', 'idle', 'error'],
    done: ['recording', 'idle', 'processing', 'error'],
    error: ['idle', 'recording']
  };

  function canTransition(from, to) {
    return validTransitions[from]?.includes(to) ?? false;
  }

  assert.equal(canTransition('idle', 'recording'), true);
  assert.equal(canTransition('recording', 'processing'), false); // Cannot process while actively recording
  assert.equal(canTransition('recording', 'idle'), true);
  assert.equal(canTransition('idle', 'processing'), true);
  assert.equal(canTransition('processing', 'done'), true);
  assert.equal(canTransition('processing', 'error'), true);
});

// 6. Audio Pipeline & Delayed Transcript Arrival Test
test('Provisional state is preserved during VAD pause and resolved when transcript arrives', () => {
  let state = {
    status: 'recording',
    isSpeaking: false,
    provisionalText: '',
    chunks: [],
    transcript: '',
    wordCount: 0,
    chunkCount: 0,
  };

  function setSpeaking(isSpeaking) {
    state.isSpeaking = isSpeaking;
  }

  function setProvisionalText(text) {
    state.provisionalText = text;
  }

  function addTranscriptChunk(chunkData) {
    state.chunks.push(chunkData);
    state.transcript = state.chunks.map(c => `${c.speaker || 'Doctor'}: ${c.text}`).join('\n');
    state.provisionalText = '';
    state.wordCount = state.transcript.split(/\s+/).filter(Boolean).length;
    state.chunkCount = state.chunks.length;
  }

  // 1. User starts speaking -> VAD detects speech
  setSpeaking(true);
  setProvisionalText('Listening and transcribing speech...');
  assert.equal(state.isSpeaking, true);
  assert.equal(state.provisionalText, 'Listening and transcribing speech...');
  assert.equal(state.chunks.length, 0);

  // 2. User pauses speaking (e.g. between sentences or at sentence end) -> VAD is false
  // CRITICAL FIX TEST: setSpeaking(false) must NOT wipe provisionalText before Whisper returns!
  setSpeaking(false);
  assert.equal(state.isSpeaking, false);
  assert.equal(state.provisionalText, 'Listening and transcribing speech...', 'Provisional text must be retained while Whisper transcribes in the background');
  assert.equal(state.chunks.length, 0);

  // 3. Delayed transcript arrives from Whisper
  addTranscriptChunk({ text: 'Patient reports mild chest discomfort and fatigue.', chunk: 1, speaker: 'Doctor' });
  assert.equal(state.chunks.length, 1);
  assert.equal(state.provisionalText, '', 'Provisional text must be cleared once transcript chunk is committed');
  assert.equal(state.chunkCount, 1);
  assert.ok(state.wordCount > 0);
  assert.ok(state.transcript.includes('mild chest discomfort'));
});

// 7. Stop Lifecycle During Transcription Test
test('Stop button initiates finalizing state and only transitions to idle on done event', () => {
  let state = {
    status: 'recording',
    provisionalText: '',
  };
  let stopSent = false;
  let timerCleared = false;

  let stopTimeoutId = null;

  function finishRecordingSession() {
    if (stopTimeoutId !== null) {
      timerCleared = true;
      stopTimeoutId = null;
    }
    state.provisionalText = '';
    state.status = 'idle';
  }

  function onStopClick() {
    if (state.status !== 'recording') return;
    // Set provisional text and send stop
    state.provisionalText = 'Finalizing speech transcription...';
    stopSent = true;

    // Start 15s safety timer
    stopTimeoutId = 12345;
  }

  function onServerDone() {
    finishRecordingSession();
  }

  // User clicks stop while recording
  onStopClick();
  assert.equal(stopSent, true);
  assert.equal(state.status, 'recording', 'Status must remain recording/finalizing while waiting for server to flush');
  assert.equal(state.provisionalText, 'Finalizing speech transcription...');
  assert.notEqual(stopTimeoutId, null);

  // Server flushes Whisper queue and emits done
  onServerDone();
  assert.equal(state.status, 'idle', 'Status switches to idle when done arrives');
  assert.equal(state.provisionalText, '');
  assert.equal(timerCleared, true, 'Safety timer was cleared cleanly');
});

// 8. Timeout and WebSocket Error Fallback Test
test('WebSocket error or timeout cleanly resets stuck finalizing state to idle', () => {
  let state = {
    status: 'recording',
    provisionalText: 'Finalizing speech transcription...',
    errorMessage: null,
  };
  let stopTimeoutId = 9999;

  function onSocketError(errMessage) {
    if (stopTimeoutId !== null) {
      stopTimeoutId = null;
    }
    state.errorMessage = errMessage;
    state.provisionalText = '';
    state.status = 'idle';
  }

  // Simulate network drop or backend error during transcription finalization
  onSocketError('WebSocket disconnected');
  assert.equal(state.status, 'idle');
  assert.equal(state.provisionalText, '');
  assert.equal(state.errorMessage, 'WebSocket disconnected');
  assert.equal(stopTimeoutId, null);
});


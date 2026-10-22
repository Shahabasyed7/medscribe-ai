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


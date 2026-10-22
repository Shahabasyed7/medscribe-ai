/**
 * Main application entry point for MedScribe AI.
 * 
 * Orchestrates Audio, WebSocket, API, Store, and Renderer.
 */

import { store } from './state';
import { render, showToast } from './render';
import { AudioManager } from './audio';
import { AudioWebSocket } from './socket';
import { ApiService } from './api';

// Initialize audio and socket managers
let audioManager: AudioManager;
let socketManager: AudioWebSocket;
let stopTimeoutId: number | null = null;

function finishRecordingSession(): void {
  if (stopTimeoutId !== null) {
    window.clearTimeout(stopTimeoutId);
    stopTimeoutId = null;
  }
  store.setProvisionalText('');
  store.setStatus('idle');
  showToast('Recording stopped. Speech finalized.');
}

function initApp(): void {
  // Connect store to renderer
  store.subscribe((state) => {
    render(state);
  });

  // Setup WebSocket manager
  socketManager = new AudioWebSocket({
    onDone: () => {
      finishRecordingSession();
    },
    onError: (err) => {
      console.error('Socket error event:', err);
      if (stopTimeoutId !== null) {
        window.clearTimeout(stopTimeoutId);
        stopTimeoutId = null;
      }
      store.setProvisionalText('');
      store.setStatus('idle');
    },
  });

  // Setup Audio manager
  audioManager = new AudioManager((chunk) => {
    socketManager.sendAudioChunk(chunk);
  });

  attachEventListeners();

  // Background health check
  ApiService.checkHealth()
    .then((health) => {
      console.info('MedScribe AI backend connected:', health);
    })
    .catch((err) => {
      console.warn('Backend not currently reachable at /api/health:', err);
    });
}

function attachEventListeners(): void {
  const startBtn = document.getElementById('start-mic-btn');
  const stopBtn = document.getElementById('stop-mic-btn');
  const clearTranscriptBtn = document.getElementById('clear-transcript-btn');
  const saveWavBtn = document.getElementById('save-wav-btn');
  const processBtn = document.getElementById('process-ai-btn');
  const clearSummaryBtn = document.getElementById('clear-summary-btn');
  const copyPrescriptionBtn = document.getElementById('copy-prescription-btn');
  const helpBtn = document.getElementById('help-btn');
  const closeHelpBtn = document.getElementById('close-help-btn');
  const alertDismissBtn = document.getElementById('alert-dismiss-btn');
  const summaryInput = document.getElementById('summary-narrative-input') as HTMLTextAreaElement | null;
  const speakerDoctorBtn = document.getElementById('speaker-doctor-btn');
  const speakerPatientBtn = document.getElementById('speaker-patient-btn');

  // Speaker Role Toggle Handlers
  speakerDoctorBtn?.addEventListener('click', () => {
    store.setActiveSpeaker('Doctor');
    showToast('Speaker role set to Doctor');
  });

  speakerPatientBtn?.addEventListener('click', () => {
    store.setActiveSpeaker('Patient');
    showToast('Speaker role set to Patient');
  });

  // 1. Start Microphone
  startBtn?.addEventListener('click', async () => {
    const currentState = store.getState();
    if (currentState.status === 'recording' || currentState.status === 'processing') return;

    try {
      if (stopTimeoutId !== null) {
        window.clearTimeout(stopTimeoutId);
        stopTimeoutId = null;
      }
      store.setError(null);
      store.setProvisionalText('');
      // Connect WebSocket first
      await socketManager.connect();
      // Start microphone recording
      await audioManager.start();
      store.setStatus('recording');
      showToast('Microphone recording started');
    } catch (err: any) {
      console.error('Error starting recording:', err);
      store.setError(err.message || 'Failed to start microphone recording.');
      audioManager.stop();
      socketManager.close();
      store.setStatus('error');
    }
  });

  // 2. Stop Microphone
  stopBtn?.addEventListener('click', () => {
    if (store.getState().status !== 'recording') return;

    // Immediately stop microphone capture so no new audio is recorded
    audioManager.stop();

    if (socketManager && socketManager.isOpen()) {
      // Indicate to user that final audio segments are being transcribed
      store.setProvisionalText('Finalizing speech transcription...');
      socketManager.sendStop();

      // Start safety timeout fallback: if backend done message does not arrive in 15s, transition to idle
      if (stopTimeoutId !== null) {
        window.clearTimeout(stopTimeoutId);
      }
      stopTimeoutId = window.setTimeout(() => {
        console.warn('Backend finalization timed out. Forcing idle status.');
        finishRecordingSession();
      }, 15000);
    } else {
      finishRecordingSession();
    }
  });

  // 3. Clear Transcript
  clearTranscriptBtn?.addEventListener('click', () => {
    const state = store.getState();
    if (state.chunks.length > 0 || state.hasRecording) {
      const confirmClear = window.confirm(
        'Are you sure you want to clear the current consultation transcript and recording?'
      );
      if (!confirmClear) return;
    }
    audioManager.clearRecording();
    store.clearTranscript();
    showToast('Transcript cleared');
  });

  // 4. Save Recording as WAV
  saveWavBtn?.addEventListener('click', () => {
    try {
      audioManager.downloadWavFile();
      showToast('WAV recording downloaded successfully');
    } catch (err: any) {
      store.setError(err.message || 'Failed to export audio file.');
    }
  });

  // 5. Process Transcript with AI
  processBtn?.addEventListener('click', async () => {
    const state = store.getState();
    if (state.status === 'processing' || state.status === 'recording') return;

    const transcript = state.transcript.trim();
    if (!transcript) {
      store.setError('Please record or enter a consultation transcript before processing.');
      return;
    }

    try {
      store.setStatus('processing');
      store.setError(null);

      const result = await ApiService.analyzeTranscript(transcript);
      if (result.turns && result.turns.length > 0) {
        store.setTurns(result.turns);
      }
      store.setClinicalSummary(result);
      showToast('Clinical summary generated and speaker turns updated!');
    } catch (err: any) {
      console.error('Clinical analysis error:', err);
      store.setError(err.message || 'Failed to extract clinical summary from transcript.');
      store.setStatus('idle');
    }
  });

  // 6. Clear Summary
  clearSummaryBtn?.addEventListener('click', () => {
    store.clearSummary();
    showToast('Summary cleared');
  });

  // 7. Copy to Prescription
  copyPrescriptionBtn?.addEventListener('click', () => {
    const state = store.getState();
    const summary = state.clinicalSummary;
    if (!summary) return;

    const prescriptionText = generatePrescriptionText(state);
    navigator.clipboard
      .writeText(prescriptionText)
      .then(() => {
        showToast('Prescription and plan copied to clipboard!');
      })
      .catch((err) => {
        console.error('Clipboard copy failed:', err);
        showToast('Failed to copy to clipboard automatically.');
      });
  });

  // 8. Help / Safety Modal
  helpBtn?.addEventListener('click', () => {
    store.toggleHelp(true);
  });

  closeHelpBtn?.addEventListener('click', () => {
    store.toggleHelp(false);
  });

  // Close help modal when clicking outside
  const helpModal = document.getElementById('help-modal');
  helpModal?.addEventListener('click', (e) => {
    if (e.target === helpModal) {
      store.toggleHelp(false);
    }
  });

  // 9. Alert Dismiss
  alertDismissBtn?.addEventListener('click', () => {
    store.setError(null);
  });

  // 10. Editable Summary Textarea
  summaryInput?.addEventListener('input', (e) => {
    const target = e.target as HTMLTextAreaElement;
    store.setEditableSummaryText(target.value);
  });

  // 11. Settings Modal
  const settingsBtn = document.getElementById('settings-btn');
  const settingsModal = document.getElementById('settings-modal');
  const closeSettingsBtn = document.getElementById('close-settings-btn');
  const saveSettingsBtn = document.getElementById('save-settings-btn');
  const settingsProvider = document.getElementById('settings-provider') as HTMLSelectElement | null;
  const settingsApiKey = document.getElementById('settings-api-key') as HTMLInputElement | null;
  const settingsStatus = document.getElementById('settings-status');

  settingsBtn?.addEventListener('click', async () => {
    if (settingsModal) settingsModal.style.display = 'flex';
    // Load current config
    try {
      const config = await ApiService.getConfig();
      if (settingsProvider) settingsProvider.value = config.provider;
      if (settingsApiKey) settingsApiKey.value = ''; // never pre-fill key
      if (settingsStatus) settingsStatus.style.display = 'none';
    } catch {
      // Backend not reached, proceed with defaults
    }
  });

  closeSettingsBtn?.addEventListener('click', () => {
    if (settingsModal) settingsModal.style.display = 'none';
    if (settingsStatus) settingsStatus.style.display = 'none';
  });

  settingsModal?.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
      settingsModal.style.display = 'none';
    }
  });

  saveSettingsBtn?.addEventListener('click', async () => {
    const provider = settingsProvider?.value || 'gemini';
    const apiKey = settingsApiKey?.value.trim() || '';

    if (!apiKey) {
      if (settingsStatus) {
        settingsStatus.textContent = 'Please enter an API key.';
        settingsStatus.style.display = 'block';
        settingsStatus.style.background = '#fef2f2';
        settingsStatus.style.color = '#b91c1c';
        settingsStatus.style.border = '1px solid #fecaca';
      }
      return;
    }

    if (saveSettingsBtn) (saveSettingsBtn as HTMLButtonElement).disabled = true;
    if (settingsStatus) settingsStatus.style.display = 'none';

    try {
      const result = await ApiService.saveConfig(provider, apiKey);
      if (settingsStatus) {
        settingsStatus.textContent = result.message || 'API key saved successfully!';
        settingsStatus.style.display = 'block';
        settingsStatus.style.background = '#ecfdf5';
        settingsStatus.style.color = '#065f46';
        settingsStatus.style.border = '1px solid #a7f3d0';
      }
      if (settingsApiKey) settingsApiKey.value = '';
      showToast(`${provider.charAt(0).toUpperCase() + provider.slice(1)} API key configured!`);
      setTimeout(() => {
        if (settingsModal) settingsModal.style.display = 'none';
      }, 1500);
    } catch (err: any) {
      if (settingsStatus) {
        settingsStatus.textContent = err.message || 'Failed to save API key.';
        settingsStatus.style.display = 'block';
        settingsStatus.style.background = '#fef2f2';
        settingsStatus.style.color = '#b91c1c';
        settingsStatus.style.border = '1px solid #fecaca';
      }
    } finally {
      if (saveSettingsBtn) (saveSettingsBtn as HTMLButtonElement).disabled = false;
    }
  });
}

function generatePrescriptionText(state: ReturnType<typeof store.getState>): string {
  const sum = state.clinicalSummary;
  if (!sum) return '';

  const patient = sum.patient_details || {};
  const patientLine = [
    patient.name ? `Patient: ${patient.name}` : '',
    patient.age ? `Age: ${patient.age}` : '',
    patient.sex ? `Sex: ${patient.sex}` : '',
    patient.identifiers ? `ID: ${patient.identifiers}` : '',
  ]
    .filter(Boolean)
    .join(' | ');

  const narrative = state.editableSummaryText || sum.clinical_summary || '';
  const prescriptions = sum.plan?.prescriptions || [];
  const advice = sum.plan?.advice || [];
  const followUp = sum.plan?.follow_up || '';

  let lines: string[] = [];
  lines.push('=== CLINICAL PRESCRIPTION & MANAGEMENT PLAN ===');
  if (patientLine) lines.push(patientLine);
  lines.push(`Date: ${new Date().toLocaleDateString()}`);
  lines.push('');

  lines.push('CLINICAL SUMMARY:');
  lines.push(narrative);
  lines.push('');

  if (prescriptions.length > 0) {
    lines.push('PRESCRIPTIONS / RX:');
    prescriptions.forEach((rx, idx) => lines.push(` ${idx + 1}. ${rx}`));
    lines.push('');
  }

  if (advice.length > 0) {
    lines.push('PATIENT ADVICE:');
    advice.forEach((adv) => lines.push(` - ${adv}`));
    lines.push('');
  }

  if (followUp) {
    lines.push(`FOLLOW-UP: ${followUp}`);
    lines.push('');
  }

  lines.push('Notice: Verified and signed by attending clinician.');
  return lines.join('\n');
}

// Start application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

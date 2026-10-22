/**
 * Pure DOM renderer driven by the state store.
 */

import { AppState, ClinicalSummary } from './types';

export function escapeHtml(str: string | null | undefined): string {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export function showToast(message: string, durationMs = 3000): void {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast-notice';
  toast.textContent = message;

  container.innerHTML = '';
  container.appendChild(toast);
  container.style.display = 'block';

  setTimeout(() => {
    if (toast.parentElement) {
      toast.parentElement.removeChild(toast);
    }
  }, durationMs);
}

export function render(state: AppState): void {
  renderSystemStatus(state);
  renderAlertBanner(state);
  renderTranscriptionCard(state);
  renderSummaryCard(state);
  renderHelpModal(state);
}

function renderSystemStatus(state: AppState): void {
  const badge = document.getElementById('system-status-badge');
  const text = document.getElementById('system-status-text');
  if (!badge || !text) return;

  badge.className = `status-badge ${state.status}`;

  switch (state.status) {
    case 'idle':
      text.textContent = 'Ready';
      break;
    case 'recording':
      text.textContent = 'Listening Live';
      break;
    case 'processing':
      text.textContent = 'Analyzing Transcript...';
      break;
    case 'done':
      text.textContent = 'Clinical Summary Ready';
      break;
    case 'error':
      text.textContent = 'Error';
      break;
  }
}

function renderAlertBanner(state: AppState): void {
  const banner = document.getElementById('alert-banner');
  const msgElem = document.getElementById('alert-message');
  if (!banner || !msgElem) return;

  if (state.errorMessage) {
    msgElem.textContent = state.errorMessage;
    banner.style.display = 'flex';
  } else {
    banner.style.display = 'none';
  }
}

function renderTranscriptionCard(state: AppState): void {
  // 1. Speaking indicator & VAD state
  const indicator = document.getElementById('speaking-indicator');
  const speakingText = document.getElementById('speaking-text');
  if (indicator && speakingText) {
    if (state.status === 'recording') {
      if (state.isSpeaking) {
        indicator.classList.add('active');
        speakingText.textContent = 'Voice detected';
      } else {
        indicator.classList.remove('active');
        speakingText.textContent = 'Silence / Listening';
      }
    } else {
      indicator.classList.remove('active');
      speakingText.textContent = 'Microphone idle';
    }
  }

  // 2. Speaker Toggle Buttons state
  const docBtn = document.getElementById('speaker-doctor-btn');
  const patBtn = document.getElementById('speaker-patient-btn');
  if (docBtn && patBtn) {
    if (state.activeSpeaker === 'Doctor') {
      docBtn.classList.add('active');
      docBtn.setAttribute('aria-checked', 'true');
      patBtn.classList.remove('active');
      patBtn.setAttribute('aria-checked', 'false');
    } else {
      patBtn.classList.add('active');
      patBtn.setAttribute('aria-checked', 'true');
      docBtn.classList.remove('active');
      docBtn.setAttribute('aria-checked', 'false');
    }
  }

  // 3. Counters
  const wordCountElem = document.getElementById('word-count');
  const chunkCountElem = document.getElementById('chunk-count');
  if (wordCountElem) wordCountElem.textContent = state.wordCount.toLocaleString();
  if (chunkCountElem) chunkCountElem.textContent = state.chunkCount.toLocaleString();

  // 4. Transcript box & provisional live text
  const placeholder = document.getElementById('transcript-placeholder');
  const content = document.getElementById('transcript-content');
  const container = document.getElementById('transcript-container');

  if (placeholder && content && container) {
    const hasFinalChunks = state.chunks.length > 0;
    const hasProvisional = Boolean(state.provisionalText && state.provisionalText.trim());

    if (!hasFinalChunks && !hasProvisional) {
      placeholder.style.display = 'flex';
      content.style.display = 'none';
      content.innerHTML = '';
    } else {
      placeholder.style.display = 'none';
      content.style.display = 'block';

      const wasAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;

      let html = state.chunks
        .map(
          (c) => `
          <div class="transcript-chunk-item">
            <span class="speaker-badge ${c.speaker}">${escapeHtml(c.speaker)}</span>
            <span>${escapeHtml(c.text)}</span>
          </div>
        `
        )
        .join('');

      // If provisional speech is being uttered, append dynamic provisional box
      if (hasProvisional) {
        html += `
          <div class="transcript-chunk-provisional">
            <span class="provisional-pulse-dot"></span>
            <span class="speaker-badge ${state.activeSpeaker}">${escapeHtml(state.activeSpeaker)}</span>
            <span>${escapeHtml(state.provisionalText)}</span>
          </div>
        `;
      }

      content.innerHTML = html;

      if (wasAtBottom) {
        container.scrollTop = container.scrollHeight;
      }
    }
  }

  // 4. Buttons state
  const startBtn = document.getElementById('start-mic-btn') as HTMLButtonElement | null;
  const stopBtn = document.getElementById('stop-mic-btn') as HTMLButtonElement | null;
  const clearBtn = document.getElementById('clear-transcript-btn') as HTMLButtonElement | null;
  const saveWavBtn = document.getElementById('save-wav-btn') as HTMLButtonElement | null;
  const processBtn = document.getElementById('process-ai-btn') as HTMLButtonElement | null;
  const processSpinner = document.getElementById('process-spinner');
  const processIcon = document.getElementById('process-icon');
  const processText = document.getElementById('process-btn-text');

  if (startBtn) {
    startBtn.disabled = state.status === 'recording' || state.status === 'processing';
  }

  if (stopBtn) {
    stopBtn.disabled = state.status !== 'recording';
  }

  if (clearBtn) {
    clearBtn.disabled = state.status === 'recording' || (state.chunks.length === 0 && !state.hasRecording);
  }

  if (saveWavBtn) {
    saveWavBtn.disabled = state.status === 'recording' || !state.hasRecording;
  }

  if (processBtn) {
    const hasText = state.transcript.trim().length > 0;
    processBtn.disabled = !hasText || state.status === 'recording' || state.status === 'processing';

    if (state.status === 'processing') {
      if (processSpinner) processSpinner.style.display = 'inline-block';
      if (processIcon) processIcon.style.display = 'none';
      if (processText) processText.textContent = 'Extracting Clinical Summary...';
    } else {
      if (processSpinner) processSpinner.style.display = 'none';
      if (processIcon) processIcon.style.display = 'inline-block';
      if (processText) processText.textContent = 'Process Transcript with AI';
    }
  }
}

function renderSummaryCard(state: AppState): void {
  const placeholder = document.getElementById('summary-placeholder');
  const content = document.getElementById('summary-content');
  const clearSummaryBtn = document.getElementById('clear-summary-btn') as HTMLButtonElement | null;

  if (clearSummaryBtn) {
    clearSummaryBtn.disabled = !state.clinicalSummary || state.status === 'processing';
  }

  if (!state.clinicalSummary) {
    if (placeholder) placeholder.style.display = 'flex';
    if (content) content.style.display = 'none';
    return;
  }

  if (placeholder) placeholder.style.display = 'none';
  if (content) content.style.display = 'flex';

  const sum: ClinicalSummary = state.clinicalSummary;

  // Patient Details
  const nameElem = document.getElementById('summary-patient-name');
  const ageSexElem = document.getElementById('summary-patient-age-sex');
  const idElem = document.getElementById('summary-patient-id');

  if (nameElem) nameElem.textContent = sum.patient_details?.name || 'Not stated';

  if (ageSexElem) {
    const age = sum.patient_details?.age ? `${sum.patient_details.age}` : null;
    const sex = sum.patient_details?.sex || null;
    if (age && sex) ageSexElem.textContent = `${age} / ${sex}`;
    else if (age) ageSexElem.textContent = age;
    else if (sex) ageSexElem.textContent = sex;
    else ageSexElem.textContent = 'Not stated';
  }

  if (idElem) idElem.textContent = sum.patient_details?.identifiers || 'Not stated';

  // Editable Narrative Summary
  const narrativeInput = document.getElementById('summary-narrative-input') as HTMLTextAreaElement | null;
  if (narrativeInput && document.activeElement !== narrativeInput) {
    narrativeInput.value = state.editableSummaryText || sum.clinical_summary || '';
  }

  // Chief Complaint
  renderPillList('summary-chief-complaint', sum.chief_complaint);

  // History of Present Illness
  const hpiElem = document.getElementById('summary-hpi-content');
  if (hpiElem) {
    const hpi = sum.history_of_present_illness || {};
    const items: string[] = [];
    if (hpi.onset) items.push(`<strong>Onset:</strong> ${escapeHtml(hpi.onset)}`);
    if (hpi.duration) items.push(`<strong>Duration:</strong> ${escapeHtml(hpi.duration)}`);
    if (hpi.progression) items.push(`<strong>Progression:</strong> ${escapeHtml(hpi.progression)}`);
    if (hpi.aggravating_factors) items.push(`<strong>Aggravating Factors:</strong> ${escapeHtml(hpi.aggravating_factors)}`);
    if (hpi.relieving_factors) items.push(`<strong>Relieving Factors:</strong> ${escapeHtml(hpi.relieving_factors)}`);

    if (items.length > 0) {
      hpiElem.innerHTML = items.map((it) => `<li class="bullet-item">${it}</li>`).join('');
    } else {
      hpiElem.innerHTML = '<li class="bullet-item" style="color: #94a3b8; font-style: italic;">No specific trajectory stated</li>';
    }
  }

  // Symptoms (Positive & Negative Separately)
  renderPillList('summary-positive-symptoms', sum.symptoms?.positive || [], 'positive');
  renderPillList('summary-negative-symptoms', sum.symptoms?.negative || [], 'negative');

  // Past Medical History
  const pmhElem = document.getElementById('summary-pmh-content');
  if (pmhElem) {
    const pmh = sum.past_medical_history || { conditions: [], surgeries: [], hospitalisations: [] };
    const items: string[] = [];
    if (pmh.conditions && pmh.conditions.length > 0) {
      items.push(`<strong>Conditions:</strong> ${pmh.conditions.map(escapeHtml).join(', ')}`);
    }
    if (pmh.surgeries && pmh.surgeries.length > 0) {
      items.push(`<strong>Surgeries:</strong> ${pmh.surgeries.map(escapeHtml).join(', ')}`);
    }
    if (pmh.hospitalisations && pmh.hospitalisations.length > 0) {
      items.push(`<strong>Hospitalisations:</strong> ${pmh.hospitalisations.map(escapeHtml).join(', ')}`);
    }

    if (items.length > 0) {
      pmhElem.innerHTML = items.map((it) => `<li class="bullet-item">${it}</li>`).join('');
    } else {
      pmhElem.innerHTML = '<li class="bullet-item" style="color: #94a3b8; font-style: italic;">No past history noted</li>';
    }
  }

  // Medications
  const medsElem = document.getElementById('summary-medications-list');
  if (medsElem) {
    const meds = sum.medications || [];
    if (meds.length > 0) {
      medsElem.innerHTML = meds
        .map(
          (m) => `
        <div class="medication-card">
          <div>
            <div class="med-name">${escapeHtml(m.name)}</div>
            ${m.dosage ? `<div class="med-dosage">${escapeHtml(m.dosage)}</div>` : ''}
          </div>
          ${m.adherence ? `<div class="med-adherence">${escapeHtml(m.adherence)}</div>` : ''}
        </div>
      `
        )
        .join('');
    } else {
      medsElem.innerHTML = '<p style="color: #94a3b8; font-size: 0.85rem; font-style: italic;">No current medications reported</p>';
    }
  }

  // Allergies
  renderPillList('summary-allergies-list', sum.allergies || [], 'allergy');

  // Clinical Observations & Vitals
  const vitalsElem = document.getElementById('summary-vitals-grid');
  const examElem = document.getElementById('summary-exam-findings');

  if (vitalsElem) {
    const vitals = sum.clinical_observations?.vitals || {};
    const entries = Object.entries(vitals);
    if (entries.length > 0) {
      vitalsElem.innerHTML = entries
        .map(
          ([k, v]) => `
          <div class="vital-card">
            <div class="vital-name">${escapeHtml(k)}</div>
            <div class="vital-val">${escapeHtml(v)}</div>
          </div>
        `
        )
        .join('');
    } else {
      vitalsElem.innerHTML = '<p style="color: #94a3b8; font-size: 0.85rem; font-style: italic;">No vitals recorded aloud</p>';
    }
  }

  if (examElem) {
    const findings = sum.clinical_observations?.examination_findings || [];
    if (findings.length > 0) {
      examElem.innerHTML = findings.map((f) => `<li class="bullet-item">${escapeHtml(f)}</li>`).join('');
    } else {
      examElem.innerHTML = '<li class="bullet-item" style="color: #94a3b8; font-style: italic;">No physical exam findings stated</li>';
    }
  }

  // Assessment
  renderBulletList('summary-assessment-list', sum.assessment || []);

  // Plan
  const planElem = document.getElementById('summary-plan-content');
  if (planElem) {
    const plan = sum.plan || { investigations: [], prescriptions: [], advice: [], follow_up: null };
    const items: string[] = [];

    if (plan.prescriptions && plan.prescriptions.length > 0) {
      items.push(`<strong>Prescriptions:</strong> ${plan.prescriptions.map(escapeHtml).join('; ')}`);
    }
    if (plan.investigations && plan.investigations.length > 0) {
      items.push(`<strong>Investigations:</strong> ${plan.investigations.map(escapeHtml).join(', ')}`);
    }
    if (plan.advice && plan.advice.length > 0) {
      items.push(`<strong>Advice:</strong> ${plan.advice.map(escapeHtml).join(', ')}`);
    }
    if (plan.follow_up) {
      items.push(`<strong>Follow-up:</strong> ${escapeHtml(plan.follow_up)}`);
    }

    if (items.length > 0) {
      planElem.innerHTML = items.map((it) => `<li class="bullet-item">${it}</li>`).join('');
    } else {
      planElem.innerHTML = '<li class="bullet-item" style="color: #94a3b8; font-style: italic;">No specific plan voiced</li>';
    }
  }
}

function renderPillList(containerId: string, items: string[], typeClass = ''): void {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!items || items.length === 0) {
    container.innerHTML = '<span style="color: #94a3b8; font-size: 0.82rem; font-style: italic;">None reported</span>';
    return;
  }

  container.innerHTML = items
    .map((item) => `<span class="tag-pill ${typeClass}">${escapeHtml(item)}</span>`)
    .join('');
}

function renderBulletList(containerId: string, items: string[]): void {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!items || items.length === 0) {
    container.innerHTML = '<li class="bullet-item" style="color: #94a3b8; font-style: italic;">None specified</li>';
    return;
  }

  container.innerHTML = items.map((item) => `<li class="bullet-item">${escapeHtml(item)}</li>`).join('');
}

function renderHelpModal(state: AppState): void {
  const modal = document.getElementById('help-modal');
  if (!modal) return;
  modal.style.display = state.isHelpOpen ? 'flex' : 'none';
}

/**
 * Centralized typed state store with subscriber notifications.
 */

import { AppState, AppStatus, ClinicalSummary, DialogueTurn, SpeakerRole, StateListener, TranscriptChunk } from './types';

const initialState: AppState = {
  status: 'idle',
  isSpeaking: false,
  activeSpeaker: 'Doctor',
  provisionalText: '',
  transcript: '',
  chunks: [],
  wordCount: 0,
  chunkCount: 0,
  clinicalSummary: null,
  editableSummaryText: '',
  errorMessage: null,
  hasRecording: false,
  audioDurationSeconds: 0,
  micPermissionGranted: null,
  isHelpOpen: false,
};

class StateStore {
  private state: AppState = { ...initialState };
  private listeners: Set<StateListener> = new Set();

  getState(): Readonly<AppState> {
    return this.state;
  }

  subscribe(listener: StateListener): () => void {
    this.listeners.add(listener);
    // Initial notification
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify(): void {
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch (err) {
        console.error('State listener error:', err);
      }
    }
  }

  setStatus(status: AppStatus): void {
    if (this.state.status !== status) {
      this.state = { ...this.state, status };
      this.notify();
    }
  }

  setSpeaking(isSpeaking: boolean): void {
    if (this.state.isSpeaking !== isSpeaking) {
      // When speech stops, if we had provisional text, clear it as the segment closes
      const provisionalText = isSpeaking ? this.state.provisionalText : '';
      this.state = { ...this.state, isSpeaking, provisionalText };
      this.notify();
    }
  }

  setActiveSpeaker(speaker: SpeakerRole): void {
    if (this.state.activeSpeaker !== speaker) {
      this.state = { ...this.state, activeSpeaker: speaker };
      this.notify();
    }
  }

  setProvisionalText(text: string): void {
    if (this.state.provisionalText !== text) {
      this.state = { ...this.state, provisionalText: text };
      this.notify();
    }
  }

  addTranscriptChunk(chunkData: { text: string; chunk: number; speaker?: SpeakerRole }): void {
    const cleanText = chunkData.text.trim();
    if (!cleanText) return;

    const speaker = chunkData.speaker || this.state.activeSpeaker;

    const newChunk: TranscriptChunk = {
      chunk: chunkData.chunk,
      text: cleanText,
      speaker,
      timestamp: Date.now(),
    };

    const newChunks = [...this.state.chunks, newChunk];
    // Build labelled full transcript: "Doctor: ... \nPatient: ..."
    const newTranscript = newChunks.map((c) => `${c.speaker}: ${c.text}`).join('\n');
    const wordCount = newChunks.map((c) => c.text).join(' ').split(/\s+/).filter(Boolean).length;

    this.state = {
      ...this.state,
      chunks: newChunks,
      transcript: newTranscript,
      provisionalText: '', // Clear provisional text once segment finalized
      wordCount,
      chunkCount: newChunks.length,
    };
    this.notify();
  }

  setTurns(turns: DialogueTurn[]): void {
    if (!turns || turns.length === 0) return;

    const newChunks: TranscriptChunk[] = turns.map((turn, idx) => ({
      chunk: idx + 1,
      text: turn.text.trim(),
      speaker: turn.speaker || 'Unknown',
      timestamp: Date.now(),
    }));

    const newTranscript = newChunks.map((c) => `${c.speaker}: ${c.text}`).join('\n');
    const wordCount = newChunks.map((c) => c.text).join(' ').split(/\s+/).filter(Boolean).length;

    this.state = {
      ...this.state,
      chunks: newChunks,
      transcript: newTranscript,
      provisionalText: '',
      wordCount,
      chunkCount: newChunks.length,
    };
    this.notify();
  }

  setHasRecording(hasRecording: boolean): void {
    this.state = { ...this.state, hasRecording };
    this.notify();
  }

  setAudioDuration(seconds: number): void {
    this.state = { ...this.state, audioDurationSeconds: seconds };
    this.notify();
  }

  setClinicalSummary(summary: ClinicalSummary): void {
    this.state = {
      ...this.state,
      clinicalSummary: summary,
      editableSummaryText: summary.clinical_summary || '',
      status: 'done',
      errorMessage: null,
    };
    this.notify();
  }

  setEditableSummaryText(text: string): void {
    this.state = { ...this.state, editableSummaryText: text };
    this.notify();
  }

  setError(errorMessage: string | null): void {
    this.state = {
      ...this.state,
      errorMessage,
      status: errorMessage ? 'error' : this.state.status,
    };
    this.notify();
  }

  clearTranscript(): void {
    this.state = {
      ...this.state,
      transcript: '',
      chunks: [],
      provisionalText: '',
      wordCount: 0,
      chunkCount: 0,
      hasRecording: false,
      audioDurationSeconds: 0,
      isSpeaking: false,
      errorMessage: null,
      status: this.state.status === 'recording' ? 'recording' : 'idle',
    };
    this.notify();
  }

  clearSummary(): void {
    this.state = {
      ...this.state,
      clinicalSummary: null,
      editableSummaryText: '',
      errorMessage: null,
      status: this.state.status === 'done' ? 'idle' : this.state.status,
    };
    this.notify();
  }

  setMicPermission(granted: boolean): void {
    this.state = { ...this.state, micPermissionGranted: granted };
    this.notify();
  }

  toggleHelp(open?: boolean): void {
    const isHelpOpen = open !== undefined ? open : !this.state.isHelpOpen;
    this.state = { ...this.state, isHelpOpen };
    this.notify();
  }

  resetAll(): void {
    this.state = { ...initialState };
    this.notify();
  }
}

export const store = new StateStore();

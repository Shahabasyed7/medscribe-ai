/**
 * Type definitions for MedScribe AI.
 */

export type AppStatus = 'idle' | 'recording' | 'processing' | 'done' | 'error';

export interface PatientDetails {
  name?: string | null;
  age?: number | string | null;
  sex?: string | null;
  identifiers?: string | null;
}

export interface HistoryOfPresentIllness {
  onset?: string | null;
  duration?: string | null;
  progression?: string | null;
  aggravating_factors?: string | null;
  relieving_factors?: string | null;
}

export interface Symptoms {
  positive: string[];
  negative: string[];
}

export interface PastMedicalHistory {
  conditions: string[];
  surgeries: string[];
  hospitalisations: string[];
}

export interface MedicationEntry {
  name: string;
  dosage?: string | null;
  adherence?: string | null;
}

export interface ClinicalObservations {
  vitals: Record<string, string>;
  examination_findings: string[];
}

export interface Plan {
  investigations: string[];
  prescriptions: string[];
  advice: string[];
  follow_up?: string | null;
}

export interface ClinicalSummary {
  clinical_summary: string;
  patient_details: PatientDetails;
  chief_complaint: string[];
  history_of_present_illness: HistoryOfPresentIllness;
  symptoms: Symptoms;
  past_medical_history: PastMedicalHistory;
  medications: MedicationEntry[];
  allergies: string[];
  clinical_observations: ClinicalObservations;
  assessment: string[];
  plan: Plan;
}

export type SpeakerRole = 'Doctor' | 'Patient' | 'Unknown';
export type ToggleSpeakerRole = 'Doctor' | 'Patient';

export interface DialogueTurn {
  speaker: SpeakerRole;
  text: string;
}

export interface AnalyzeResponse extends ClinicalSummary {
  turns?: DialogueTurn[];
}

export interface TranscriptChunk {
  chunk: number;
  text: string;
  speaker: SpeakerRole;
  timestamp: number;
}

export interface AppState {
  status: AppStatus;
  isSpeaking: boolean;
  activeSpeaker: SpeakerRole;
  provisionalText: string;
  transcript: string;
  chunks: TranscriptChunk[];
  wordCount: number;
  chunkCount: number;
  clinicalSummary: ClinicalSummary | null;
  editableSummaryText: string;
  errorMessage: string | null;
  hasRecording: boolean;
  audioDurationSeconds: number;
  micPermissionGranted: boolean | null;
  isHelpOpen: boolean;
}

export type StateListener = (state: AppState) => void;

// WebSocket message protocols
export interface VadMessage {
  type: 'vad';
  speaking: boolean;
}

export interface TranscriptMessage {
  type: 'transcript';
  text: string;
  chunk: number;
}

export interface DoneMessage {
  type: 'done';
}

export interface ErrorMessage {
  type: 'error';
  message: string;
}

export type ServerMessage = VadMessage | TranscriptMessage | DoneMessage | ErrorMessage;

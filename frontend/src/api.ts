/**
 * HTTP API client for MedScribe AI backend.
 */

import { AnalyzeResponse } from './types';

export interface ConfigStatus {
  provider: string;
  gemini_configured: boolean;
  openai_configured: boolean;
  gemini_model: string;
  openai_model: string;
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export class ApiService {
  /**
   * Helper to build full URL based on API_BASE_URL.
   */
  private static endpoint(path: string): string {
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return API_BASE_URL ? `${API_BASE_URL}${cleanPath}` : cleanPath;
  }

  /**
   * Check backend health and readiness.
   */
  static async checkHealth(): Promise<{
    status: string;
    whisper_ready: boolean;
    llm_provider: string;
    gemini_configured: boolean;
    openai_configured: boolean;
  }> {
    const res = await fetch(this.endpoint('/api/health'));
    if (!res.ok) {
      throw new Error(`Health check failed with status ${res.status}`);
    }
    return res.json();
  }

  /**
   * Get current LLM configuration status.
   */
  static async getConfig(): Promise<ConfigStatus> {
    const res = await fetch(this.endpoint('/api/config'));
    if (!res.ok) {
      throw new Error(`Config fetch failed with status ${res.status}`);
    }
    return res.json();
  }

  /**
   * Save LLM provider and API key to backend.
   */
  static async saveConfig(provider: string, apiKey: string): Promise<{ success: boolean; message: string }> {
    const res = await fetch(this.endpoint('/api/config'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, api_key: apiKey }),
    });

    if (!res.ok) {
      let errorDetail = `Config save failed with HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errorDetail = errJson.detail;
      } catch {
        // ignore
      }
      throw new Error(errorDetail);
    }

    return res.json();
  }

  /**
   * Request LLM clinical extraction from transcript.
   */
  static async analyzeTranscript(transcript: string): Promise<AnalyzeResponse> {
    const clean = transcript.trim();
    if (!clean) {
      throw new Error('Transcript is empty. Please record or enter a consultation transcript first.');
    }

    const response = await fetch(this.endpoint('/api/analyze'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ transcript: clean }),
    });

    if (!response.ok) {
      let errorDetail = `Analysis failed with HTTP ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson.detail) {
          errorDetail = errJson.detail;
        }
      } catch {
        try {
          const errText = await response.text();
          if (errText) errorDetail = errText;
        } catch {
          // ignore
        }
      }
      throw new Error(errorDetail);
    }

    const data: AnalyzeResponse = await response.json();
    return data;
  }
}


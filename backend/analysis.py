"""LLM clinical extraction and analysis service.

Converts consultation transcripts into Pydantic-validated ClinicalSummary models
using either Google Gemini (async with automatic fallback) or OpenAI / OpenAI-compatible LLM endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from schema import AnalyzeResponse, ClinicalSummary, DialogueTurn

logger = logging.getLogger(__name__)

SPEAKER_LABEL_SYSTEM_PROMPT = """You are an expert clinical conversation segmenter and speaker tagger.
Your task is to take a raw consultation transcript and segment it into conversational dialogue turns, labelling each turn with the speaker: "Doctor", "Patient", or "Unknown".

SPEAKER IDENTIFICATION RULES:
1. Determine speaker roles primarily based on conversational context:
   - "Doctor": asks clinical questions, takes medical history, conducts/verbalizes exams, gives medical explanations, diagnostic impressions, treatment advice, prescriptions, or follow-up instructions.
   - "Patient": describes symptoms, personal feelings, history of present illness, answers clinical questions, or asks for help/clarification.
   - "Unknown": use ONLY if conversational context makes the speaker completely ambiguous or uncertain.
2. If lines already contain manual speaker prefixes like "Doctor:", "Patient:", or "Unknown:", use them ONLY as hints, but correct any obvious mislabeling or split chunks containing multiple speakers.
3. CRITICAL VERBATIM REQUIREMENT:
   - You MUST keep EVERY SINGLE WORD from the original transcript strictly verbatim.
   - DO NOT rephrase, correct grammar, fix typos, summarize, add new words, or omit any words.
   - When all turn texts are concatenated with spaces, the words must match the original words in the exact same order.
4. Output Format:
   Return ONLY a valid JSON object matching this schema:
   {
     "turns": [
       {"speaker": "Doctor", "text": "..."},
       {"speaker": "Patient", "text": "..."}
     ]
   }
   No markdown fences, preamble, or conversational commentary.
"""

SYSTEM_PROMPT = """You are an expert clinical documentation AI assistant. Your role is to convert a verbatim doctor-patient consultation transcript into a highly accurate, structured clinical summary following the exact provided schema.

STRICT MEDICAL EXTRACTION RULES:
1. Extract ONLY information explicitly stated in the supplied consultation transcript.
2. DO NOT hallucinate, assume, or infer patient demographics, symptoms, diagnoses, examination findings, or treatment.
3. Use null for absent scalar information and empty lists [] for absent collection fields. Do NOT write "Unknown", "N/A", or "None" unless explicitly uttered.
4. Keep positive symptoms explicitly reported by the patient separate from negative symptoms (symptoms explicitly denied by the patient, e.g. "no fever", "denies chest pain", "no breathing difficulty").
5. Preserve the chief complaint in the patient's own words where practical.
6. Record vitals and examination findings ONLY if stated aloud during the encounter.
7. CRITICAL ROLE RULE: Assessment, diagnostic impressions, and treatment plan (prescriptions, investigations, advice, follow-up) must come ONLY from turns labeled "Doctor". If a turn is labeled "Patient" or "Unknown", its content must NEVER be placed in assessment or plan. Never treat "Unknown" as the clinician.
8. DO NOT invent medication names, dosages, allergies, or diagnostic investigations. If a medication was taken but dosage was not stated or unknown, set dosage to null or "Not stated".
9. Distinguish a proposed examination from an examination actually performed.
10. Preserve clinical uncertainty and distinguish patient-reported history from objective clinician observations.
11. Write 'clinical_summary' in 2-4 concise, professional clinical sentences synthesizing the consultation encounter.
12. Return ONLY a single valid JSON object adhering strictly to the requested schema. No conversational filler, introductory text, or markdown code fences outside the JSON.
"""

# Ordered list of candidate Gemini models to try when experiencing high demand (503) or 404
GEMINI_CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]


def extract_words(text: str) -> List[str]:
    """Normalize text into lowercase word tokens for verbatim validation."""
    # Strip speaker prefixes like "Doctor:", "Patient:", "Unknown:" (both at line start and inline)
    cleaned = re.sub(r"\b(?:Doctor|Patient|Unknown)\s*:\s*", "", text, flags=re.IGNORECASE)
    # Extract alphanumeric words
    return re.findall(r"\b[\w'-]+\b", cleaned.lower())


def validate_verbatim(original_transcript: str, turns: List[DialogueTurn]) -> bool:
    """Validate that the concatenated turn texts equal the original transcript's words verbatim."""
    original_words = extract_words(original_transcript)
    turns_combined = " ".join(turn.text for turn in turns)
    turns_words = extract_words(turns_combined)

    if original_words == turns_words:
        return True

    logger.warning(
        f"Verbatim validation mismatch: original has {len(original_words)} words, "
        f"turns have {len(turns_words)} words."
    )
    return False


def fallback_turns_from_transcript(transcript: str) -> List[DialogueTurn]:
    """Parse original transcript into turns using existing line labels or fallback to Unknown."""
    lines = [line.strip() for line in transcript.split("\n") if line.strip()]
    fallback_turns: List[DialogueTurn] = []

    for line in lines:
        match = re.match(r"^(Doctor|Patient|Unknown)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            role_raw = match.group(1).title()
            speaker = role_raw if role_raw in ("Doctor", "Patient") else "Unknown"
            text = match.group(2).strip()
            fallback_turns.append(DialogueTurn(speaker=speaker, text=text))
        else:
            fallback_turns.append(DialogueTurn(speaker="Unknown", text=line))

    return fallback_turns


class ClinicalAnalyzer:
    """Handles LLM extraction and speaker-turn labeling for clinical consultation transcripts."""

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    def _extract_json_text(self, text: str) -> str:
        """Strip markdown code fence blocks if present."""
        clean = text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if fence_match:
            return fence_match.group(1).strip()
        first_brace = clean.find("{")
        last_brace = clean.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return clean[first_brace : last_brace + 1]
        return clean

    def _parse_turns_from_json(self, raw_json_str: str) -> List[DialogueTurn]:
        """Parse dialogue turns JSON."""
        clean_json_str = self._extract_json_text(raw_json_str)
        data = json.loads(clean_json_str)

        # Allow format {"turns": [...]} or raw list [...]
        raw_turns = data.get("turns", data) if isinstance(data, dict) else data
        if not isinstance(raw_turns, list):
            raise ValueError("Expected a list of turns in LLM output.")

        parsed_turns: List[DialogueTurn] = []
        for item in raw_turns:
            if not isinstance(item, dict):
                continue
            spk_raw = str(item.get("speaker", "Unknown")).strip().capitalize()
            spk = spk_raw if spk_raw in ("Doctor", "Patient") else "Unknown"
            txt = str(item.get("text", "")).strip()
            if txt:
                parsed_turns.append(DialogueTurn(speaker=spk, text=txt))

        return parsed_turns

    async def _call_gemini_text(self, system_prompt: str, user_prompt: str) -> str:
        """Helper to invoke Gemini API with model fallback."""
        if not self.gemini_key:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in the website Settings or backend/.env."
            )

        from google import genai
        from google.genai import types
        from google.genai.errors import APIError, ClientError, ServerError

        client = genai.Client(api_key=self.gemini_key)

        models_to_try: List[str] = [self.gemini_model]
        for m in GEMINI_CANDIDATE_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error: Optional[Exception] = None

        for model_name in models_to_try:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                return response.text or "{}"
            except (ServerError, ClientError, APIError) as api_err:
                err_msg = str(api_err)
                last_error = api_err
                if "API_KEY_INVALID" in err_msg or "400" in err_msg and "API key" in err_msg:
                    raise ValueError("Configured Gemini API key is invalid. Please check your key in Settings.")
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    logger.warning(f"Gemini quota exhausted for '{model_name}'. Trying fallback...")
                    continue
                if "404" in err_msg or "503" in err_msg or "UNAVAILABLE" in err_msg:
                    logger.warning(f"Model '{model_name}' unavailable ({err_msg[:80]}). Trying next...")
                    continue
                continue
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            raise ValueError(f"Gemini request failed: {last_error}")
        raise ValueError("Failed to get response from Gemini models.")

    async def _call_openai_text(self, system_prompt: str, user_prompt: str) -> str:
        """Helper to invoke OpenAI API."""
        if not self.openai_key:
            raise ValueError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in the website Settings or backend/.env."
            )

        from openai import AsyncOpenAI

        kwargs: Dict[str, Any] = {"api_key": self.openai_key}
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url

        client = AsyncOpenAI(**kwargs)
        completion = await client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return completion.choices[0].message.content or "{}"

    async def _call_llm_json(self, system_prompt: str, user_prompt: str) -> str:
        """Route to configured LLM provider to fetch JSON response."""
        if self.provider == "gemini":
            try:
                return await self._call_gemini_text(system_prompt, user_prompt)
            except ValueError as ve:
                if self.openai_key and "API key is not configured" in str(ve):
                    return await self._call_openai_text(system_prompt, user_prompt)
                raise ve
        elif self.provider in ("openai", "ollama", "vllm"):
            return await self._call_openai_text(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider '{self.provider}'. Use 'gemini' or 'openai'.")

    async def label_speakers(self, transcript: str) -> List[DialogueTurn]:
        """Split raw transcript into turns and label Doctor/Patient/Unknown with verbatim validation."""
        clean_transcript = transcript.strip()
        if not clean_transcript:
            return []

        user_prompt = (
            f"Segment the following consultation transcript into conversational turns with speaker labels ('Doctor', 'Patient', 'Unknown').\n"
            f"Remember: retain all words VERBATIM in exact order.\n\n"
            f'"""\n{clean_transcript}\n"""'
        )

        # Attempt 1
        try:
            raw_output = await self._call_llm_json(SPEAKER_LABEL_SYSTEM_PROMPT, user_prompt)
            turns = self._parse_turns_from_json(raw_output)
            if turns and validate_verbatim(clean_transcript, turns):
                logger.info(f"Speaker labeling succeeded on attempt 1 with {len(turns)} turns.")
                return turns
        except Exception as e:
            logger.warning(f"Speaker labeling attempt 1 failed with error: {e}")

        # Retry once on failure / verbatim mismatch
        logger.warning("Verbatim validation or parsing failed on attempt 1. Retrying once...")
        retry_prompt = (
            f"{user_prompt}\n\n"
            f"ATTENTION: Your previous output failed verbatim word validation. "
            f"You MUST preserve EVERY single word from the input without adding, deleting, replacing, or skipping any words."
        )
        try:
            raw_output_retry = await self._call_llm_json(SPEAKER_LABEL_SYSTEM_PROMPT, retry_prompt)
            turns_retry = self._parse_turns_from_json(raw_output_retry)
            if turns_retry and validate_verbatim(clean_transcript, turns_retry):
                logger.info(f"Speaker labeling succeeded on retry with {len(turns_retry)} turns.")
                return turns_retry
        except Exception as e:
            logger.warning(f"Speaker labeling retry failed with error: {e}")

        # Fallback to original transcript labels/lines
        logger.warning("Speaker labeling retry failed or violated verbatim check. Falling back to original labels.")
        return fallback_turns_from_transcript(clean_transcript)

    async def _extract_clinical_summary(self, labeled_transcript: str) -> ClinicalSummary:
        """Run clinical extraction on the speaker-labeled transcript."""
        schema_json = json.dumps(ClinicalSummary.model_json_schema(), indent=2)
        prompt = (
            f"Here is the speaker-labeled consultation transcript to analyze:\n\n"
            f'"""\n{labeled_transcript}\n"""\n\n'
            f"Extract the clinical summary strictly adhering to this JSON Schema:\n{schema_json}"
        )

        raw_output = await self._call_llm_json(SYSTEM_PROMPT, prompt)
        clean_json_str = self._extract_json_text(raw_output)
        data = json.loads(clean_json_str)
        return ClinicalSummary.model_validate(data)

    async def analyze(self, transcript: str) -> AnalyzeResponse:
        """Perform speaker-turn labeling followed by clinical extraction."""
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty or whitespace only.")

        # Re-read environment variables in case .env was updated via /api/config
        self.provider = os.getenv("LLM_PROVIDER", self.provider).lower().strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

        # Step 1: Label speakers into verbatim dialogue turns
        turns = await self.label_speakers(transcript)

        # Step 2: Format labeled transcript for clinical extraction
        labeled_transcript = "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)

        # Step 3: Extract clinical summary strictly adhering to clinician/patient role rules
        summary = await self._extract_clinical_summary(labeled_transcript)

        # Step 4: Return AnalyzeResponse envelope with both summary and turns
        return AnalyzeResponse(**summary.model_dump(), turns=turns)


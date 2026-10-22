"""ASR (Automatic Speech Recognition) service using Faster-Whisper.

Features:
- Whisper small.en model on CPU int8 compute with beam_size=5.
- Medical consultation initial_prompt containing clinical domain context, common drug names, and symptoms.
- condition_on_previous_text=False to avoid hallucination cascades.
- Hallucination and artifact filtering:
  * Rejection of URL / web address patterns.
  * Quality thresholds: no_speech_prob <= 0.6 and avg_logprob >= -1.0.
  * Exact and normalized consecutive duplicate segment suppression.
- Non-blocking execution via dedicated worker thread.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
from typing import Optional, Tuple
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Medical domain initial prompt giving context for doctor-patient consultation
MEDICAL_INITIAL_PROMPT = (
    "Doctor-patient medical consultation. Clinical dialogue covering symptoms, headache, fever, "
    "cough, dyspnea, fatigue, body pain, nausea, vomiting, hypertension, diabetes. "
    "Common medications: paracetamol, ibuprofen, amoxicillin, metformin, lisinopril, omeprazole, "
    "atorvastatin, azithromycin, ciprofloxacin, cetirizine. Dosages, adherence, allergies, and history."
)

# Regex to detect URL/website hallucinations (e.g. www.subtitles.com, http://..., etc.)
URL_REGEX = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|io|co|ai)\b(?:/\S*)?)",
    re.IGNORECASE,
)


def is_hallucinated_text(
    text: str,
    no_speech_prob: float = 0.0,
    avg_logprob: float = 0.0,
    last_text: Optional[str] = None,
) -> bool:
    """Check if transcribed segment appears to be a hallucination, silence artifact, or duplicate.

    Returns True if the text should be discarded.
    """
    cleaned = text.strip()
    if not cleaned:
        return True

    # 1. Reject if no_speech_prob > 0.6
    if no_speech_prob > 0.6:
        logger.info(f"Filtering hallucination: no_speech_prob={no_speech_prob:.2f} > 0.6 for '{cleaned}'")
        return True

    # 2. Reject if avg_logprob < -1.0
    if avg_logprob < -1.0:
        logger.info(f"Filtering low-confidence segment: avg_logprob={avg_logprob:.2f} < -1.0 for '{cleaned}'")
        return True

    # 3. Reject if text contains URLs or website links
    if URL_REGEX.search(cleaned):
        logger.info(f"Filtering URL hallucination: '{cleaned}'")
        return True

    # 4. Filter common YouTube/Whisper subtitle artifacts
    lower_text = cleaned.lower()
    subtitle_artifacts = (
        "subtitles by",
        "transcribed by",
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "all rights reserved",
    )
    if any(artifact in lower_text for artifact in subtitle_artifacts):
        logger.info(f"Filtering subtitle artifact: '{cleaned}'")
        return True

    # 5. Drop repeated identical segments
    if last_text is not None:
        norm_current = re.sub(r"[^\w\s]", "", lower_text).strip()
        norm_last = re.sub(r"[^\w\s]", "", last_text.lower()).strip()
        if norm_current and norm_current == norm_last:
            logger.info(f"Filtering duplicate consecutive segment: '{cleaned}'")
            return True

    return False


class WhisperTranscriber:
    """Faster-Whisper transcriber optimized for medical speech."""

    _instance: Optional["WhisperTranscriber"] = None

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
    ) -> None:
        self.model_size = os.getenv("WHISPER_MODEL", model_size)
        self.device = os.getenv("WHISPER_DEVICE", device)
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", compute_type)
        self.beam_size = int(os.getenv("WHISPER_BEAM_SIZE", str(beam_size)))
        self.initial_prompt = MEDICAL_INITIAL_PROMPT

        logger.info(
            f"Loading Faster-Whisper model '{self.model_size}' on {self.device} ({self.compute_type}) with beam_size={self.beam_size}..."
        )
        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Faster-Whisper model loaded successfully.")

        # Dedicated single-thread worker for sequential inference
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-asr")

    @classmethod
    def get_instance(cls) -> "WhisperTranscriber":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """Convert signed 16-bit PCM bytes to normalized float32 array in [-1.0, 1.0]."""
        if not pcm_bytes:
            return np.empty(0, dtype=np.float32)
        int16_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_samples.astype(np.float32) / 32768.0

    @staticmethod
    def sanitize_prompt(text: str, max_chars: int = 200) -> Optional[str]:
        """Sanitize and limit length of prompt text."""
        if not text:
            return None
        clean = re.sub(r"[^\w\s.,?!'-]", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) > max_chars:
            clean = clean[-max_chars:]
            first_space = clean.find(" ")
            if first_space != -1:
                clean = clean[first_space + 1 :]
        return clean if clean else None

    def _sync_transcribe(
        self, audio_data: np.ndarray, last_text: Optional[str] = None
    ) -> Tuple[str, float]:
        """Synchronous transcription execution inside worker thread with quality and hallucination filtering."""
        if audio_data.size == 0:
            return "", 0.0

        segments, info = self.model.transcribe(
            audio_data,
            beam_size=self.beam_size,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )

        valid_texts = []
        for segment in segments:
            text = segment.text.strip()
            no_speech = getattr(segment, "no_speech_prob", 0.0)
            avg_logprob = getattr(segment, "avg_logprob", 0.0)

            # Check for hallucination / low confidence / url / duplicate
            if is_hallucinated_text(text, no_speech_prob=no_speech, avg_logprob=avg_logprob, last_text=last_text):
                continue

            valid_texts.append(text)
            last_text = text

        full_text = " ".join(valid_texts).strip()
        duration = float(info.duration) if info else 0.0
        return full_text, duration

    async def transcribe_segment(
        self, pcm_bytes: bytes, last_text: Optional[str] = None
    ) -> Tuple[str, float]:
        """Asynchronously transcribe a VAD-approved PCM16 audio segment.

        Runs transcription off the main asyncio loop so audio intake remains smooth.
        """
        if not pcm_bytes:
            return "", 0.0

        audio_float = self.pcm16_to_float32(pcm_bytes)
        if len(audio_float) < 1600:  # Less than 0.1s
            return "", 0.0

        loop = asyncio.get_running_loop()

        try:
            text, duration = await loop.run_in_executor(
                self._executor, self._sync_transcribe, audio_float, last_text
            )
            return text, duration
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True)
            raise RuntimeError(f"Transcription failed: {e}") from e

    def close(self) -> None:
        """Shutdown worker executor."""
        self._executor.shutdown(wait=False)

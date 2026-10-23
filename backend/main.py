"""FastAPI application for MedScribe AI.

Serves:
- WebSocket /ws/audio for real-time PCM16 streaming, VAD state, and Whisper transcription
- POST /api/analyze for LLM clinical summary extraction
- GET /api/health for system status and readiness
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field

# Load environment variables from backend/.env if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from asr import WhisperTranscriber
from analysis import ClinicalAnalyzer
from schema import AnalyzeResponse, ClinicalSummary
from vad import SpeechSegmenter

# Configure logging (avoid logging raw transcripts or secrets)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("medscribe-ai")

# Lifespan manager to preload model once at startup
whisper_transcriber: Optional[WhisperTranscriber] = None
clinical_analyzer: Optional[ClinicalAnalyzer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_transcriber, clinical_analyzer
    logger.info("Initializing MedScribe AI backend...")
    try:
        whisper_transcriber = WhisperTranscriber.get_instance()
    except Exception as e:
        logger.error(f"Failed to pre-load Whisper model: {e}", exc_info=True)
    clinical_analyzer = ClinicalAnalyzer()
    yield
    logger.info("Shutting down MedScribe AI backend...")
    if whisper_transcriber:
        whisper_transcriber.close()


app = FastAPI(
    title="MedScribe AI Backend",
    version="1.0.0",
    description="Real-time clinical consultation speech-to-text and clinical summary extraction API",
    lifespan=lifespan,
)

# CORS configuration
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

env_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
origins = list(set(default_origins + env_origins))

# Allow Vercel preview deployments (e.g. https://*-shahabasyed7s-projects.vercel.app or custom domain)
origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*\.vercel\.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., description="Verbatim consultation transcript to analyze")


class ConfigRequest(BaseModel):
    provider: str = Field(..., description="LLM provider: 'gemini' or 'openai'")
    api_key: str = Field(..., description="API key for the selected provider")


_ENV_PATH = Path(os.path.dirname(__file__)) / ".env"


def _update_env_file(key: str, value: str) -> None:
    """Update or add a key=value pair in the .env file."""
    env_content = _ENV_PATH.read_text(encoding="utf-8") if _ENV_PATH.exists() else ""
    lines = env_content.splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")
    _ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@app.get("/api/config")
async def get_config():
    """Return current LLM configuration status (keys are masked)."""
    return {
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.post("/api/config")
async def save_config(req: ConfigRequest):
    """Save LLM provider and API key to backend/.env for persistence."""
    provider = req.provider.lower().strip()
    if provider not in ("gemini", "openai"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider '{provider}'. Use 'gemini' or 'openai'.",
        )

    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key cannot be empty.",
        )

    # Write to .env
    _update_env_file("LLM_PROVIDER", provider)
    if provider == "gemini":
        _update_env_file("GEMINI_API_KEY", api_key)
    else:
        _update_env_file("OPENAI_API_KEY", api_key)

    # Reload into current process environment so it takes effect immediately
    load_dotenv(dotenv_path=str(_ENV_PATH), override=True)

    logger.info(f"LLM configuration updated: provider={provider}")
    return {"success": True, "provider": provider, "message": f"{provider.title()} API key saved successfully."}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "whisper_ready": whisper_transcriber is not None,
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_transcript(req: AnalyzeRequest):
    """Analyze consultation transcript and return a structured ClinicalSummary with segmented turns."""
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript cannot be empty or whitespace only.",
        )

    analyzer = clinical_analyzer or ClinicalAnalyzer()

    try:
        summary = await analyzer.analyze(transcript)
        return summary
    except ValueError as ve:
        error_msg = str(ve)
        logger.warning(f"Validation or configuration error in analysis: {error_msg}")
        # If API key missing, return 400 or 502 with clear explanation
        if "API key is not configured" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Analysis failed: {error_msg}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during clinical analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to analyze transcript with upstream LLM provider.",
        )


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time PCM16 audio transcription.

    Protocol:
    Client -> Server:
      - Binary frame: 16 kHz mono 16-bit signed PCM audio bytes (~100ms chunks)
      - Text frame: "stop" -> finish recording, flush VAD buffer, complete transcription

    Server -> Client (JSON):
      - {"type": "vad", "speaking": true/false}
      - {"type": "transcript", "text": "...", "chunk": N}
      - {"type": "done"}
      - {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket client connected for audio transcription.")

    transcriber = whisper_transcriber or WhisperTranscriber.get_instance()

    # Queue for speech segments to transcribe asynchronously without blocking WebSocket intake
    segment_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    chunk_counter = 0
    full_transcript_parts: list[str] = []

    # Thread-safe VAD speaking state callback
    loop = asyncio.get_running_loop()

    def on_speaking_change(speaking: bool):
        asyncio.run_coroutine_threadsafe(
            _safe_send_json(websocket, {"type": "vad", "speaking": speaking}),
            loop,
        )

    segmenter = SpeechSegmenter(
        vad_mode=2,
        ring_buffer_size=17,
        onset_ratio=0.6,
        closure_silence_ms=700,
        post_speech_padding_ms=500,
        min_segment_seconds=0.4,
        max_segment_seconds=15.0,
        on_speaking_change=on_speaking_change,
    )

    # Worker task to process speech segments with dynamic batching of queued segments
    async def transcription_worker():
        nonlocal chunk_counter
        last_transcribed_text = None
        silence_gap = np.zeros(4800, dtype=np.float32)  # 300 ms of 16kHz silence

        while True:
            first_seg = await segment_queue.get()
            if first_seg is None:
                segment_queue.task_done()
                break

            # Collect any segments already queued without waiting, up to max ~28.0 seconds total audio
            batch_bytes: list[bytes] = [first_seg]
            first_audio = transcriber.pcm16_to_float32(first_seg)
            audio_arrays: list[np.ndarray] = [first_audio]
            durations: list[float] = [len(first_audio) / 16000.0]
            total_audio_sec = durations[0]

            while not segment_queue.empty():
                try:
                    next_item = segment_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if next_item is None:
                    # Put sentinel back to terminate after this batch
                    await segment_queue.put(None)
                    break

                next_audio = transcriber.pcm16_to_float32(next_item)
                next_dur = len(next_audio) / 16000.0
                if total_audio_sec + 0.3 + next_dur > 28.0:
                    # Exceeds max batch window; put back in front or process in next turn
                    # Because asyncio.Queue doesn't have unget, put it back into queue
                    await segment_queue.put(next_item)
                    segment_queue.task_done()
                    break

                batch_bytes.append(next_item)
                audio_arrays.append(next_audio)
                durations.append(next_dur)
                total_audio_sec += 0.3 + next_dur
                segment_queue.task_done()

            # Announce status: transcribing with current remaining queue depth
            current_queue_depth = segment_queue.qsize()
            await _safe_send_json(
                websocket,
                {
                    "type": "status",
                    "transcribing": True,
                    "pending": current_queue_depth,
                },
            )

            start_t = time.perf_counter()
            try:
                # Concatenate with 300ms silence gap between merged segments
                if len(audio_arrays) == 1:
                    combined_audio = audio_arrays[0]
                else:
                    parts = []
                    for i, arr in enumerate(audio_arrays):
                        if i > 0:
                            parts.append(silence_gap)
                        parts.append(arr)
                    combined_audio = np.concatenate(parts)

                # Compute time boundaries for each original segment in the combined audio
                segment_bounds = []
                cur_offset = 0.0
                for dur in durations:
                    segment_bounds.append((cur_offset, cur_offset + dur))
                    cur_offset += dur + 0.3  # account for 300ms silence

                # Transcribe combined audio returning detailed segments with timestamps
                whisper_segs = await transcriber.transcribe_batch(
                    combined_audio, last_text=last_transcribed_text
                )
                elapsed_s = time.perf_counter() - start_t

                # Log performance without logging transcript text
                logger.info(
                    f"ASR Batch complete: merged={len(batch_bytes)} segments, "
                    f"audio={total_audio_sec:.2f}s, took={elapsed_s:.2f}s, "
                    f"queue_depth={current_queue_depth}"
                )

                if whisper_segs:
                    # Map transcribed segments back to original segments if cleanly fit
                    assigned_chunks = [[] for _ in range(len(segment_bounds))]
                    unassigned_texts = []

                    for wseg in whisper_segs:
                        w_mid = (wseg["start"] + wseg["end"]) / 2.0
                        matched_idx = None
                        for s_idx, (s_start, s_end) in enumerate(segment_bounds):
                            # Allow 0.4s tolerance for boundary bleed
                            if (s_start - 0.4) <= w_mid <= (s_end + 0.4):
                                matched_idx = s_idx
                                break

                        if matched_idx is not None:
                            assigned_chunks[matched_idx].append(wseg["text"])
                        else:
                            unassigned_texts.append(wseg["text"])

                    # Emit chunks per segment or fallback to batch chunk
                    sent_any = False
                    for s_idx, texts in enumerate(assigned_chunks):
                        if texts:
                            seg_text = " ".join(texts).strip()
                            if seg_text:
                                chunk_counter += 1
                                last_transcribed_text = seg_text
                                full_transcript_parts.append(seg_text)
                                sent_any = True
                                await _safe_send_json(
                                    websocket,
                                    {
                                        "type": "transcript",
                                        "text": seg_text,
                                        "chunk": chunk_counter,
                                    },
                                )

                    if unassigned_texts:
                        leftover = " ".join(unassigned_texts).strip()
                        if leftover:
                            chunk_counter += 1
                            last_transcribed_text = leftover
                            full_transcript_parts.append(leftover)
                            sent_any = True
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "transcript",
                                    "text": leftover,
                                    "chunk": chunk_counter,
                                },
                            )

                    if not sent_any:
                        # Fallback: join all whisper segment texts into single chunk
                        fallback_text = " ".join(w["text"] for w in whisper_segs).strip()
                        if fallback_text:
                            chunk_counter += 1
                            last_transcribed_text = fallback_text
                            full_transcript_parts.append(fallback_text)
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "transcript",
                                    "text": fallback_text,
                                    "chunk": chunk_counter,
                                },
                            )

            except Exception as e:
                logger.error(f"Error during batch transcription: {e}", exc_info=True)
                await _safe_send_json(
                    websocket,
                    {"type": "error", "message": "Transcription error encountered."},
                )
            finally:
                segment_queue.task_done()
                # Update status after finishing batch
                await _safe_send_json(
                    websocket,
                    {
                        "type": "status",
                        "transcribing": False,
                        "pending": segment_queue.qsize(),
                    },
                )

    worker_task = asyncio.create_task(transcription_worker())

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            # Handle binary PCM16 audio chunk
            if "bytes" in message and message["bytes"]:
                audio_chunk: bytes = message["bytes"]
                completed_segments = segmenter.process_chunk(audio_chunk)
                for seg in completed_segments:
                    await segment_queue.put(seg)
                    await _safe_send_json(
                        websocket,
                        {
                            "type": "status",
                            "transcribing": True,
                            "pending": segment_queue.qsize(),
                        },
                    )

            # Handle text commands (e.g. "stop")
            elif "text" in message and message["text"]:
                cmd = message["text"].strip().lower()
                if cmd == "stop":
                    logger.info("Stop command received. Flushing remaining audio...")
                    flushed_segments = segmenter.flush()
                    for seg in flushed_segments:
                        await segment_queue.put(seg)

                    await _safe_send_json(
                        websocket,
                        {
                            "type": "status",
                            "transcribing": True,
                            "pending": segment_queue.qsize(),
                        },
                    )

                    # Send sentinel to wait for all queued segments to finish
                    await segment_queue.put(None)
                    await worker_task

                    # Notify client that transcription session is completed
                    await _safe_send_json(websocket, {"type": "done"})
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}", exc_info=True)
        await _safe_send_json(
            websocket,
            {"type": "error", "message": f"Session error: {str(e)}"},
        )
    finally:
        # Cleanup
        if not worker_task.done():
            # In case client disconnected unexpectedly
            await segment_queue.put(None)
            try:
                await asyncio.wait_for(worker_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                worker_task.cancel()
        logger.info("WebSocket audio session finalized.")


async def _safe_send_json(websocket: WebSocket, data: dict):
    """Safely send JSON message over WebSocket if connection is open."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)

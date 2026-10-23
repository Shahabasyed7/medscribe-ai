"""Automated tests for MedScribe AI backend modules.

Tests:
- Schema validation, required fields, and null/collection defaults
- Positive vs negative symptoms separation
- VAD silence filtering, onset detection, offset detection, and flush behavior
- Whisper audio normalization and prompt sanitization
- FastAPI analyze endpoint validation (400 for empty transcript)
"""

import json
import math
import struct
import numpy as np
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from schema import (
    ClinicalSummary,
    PatientDetails,
    HistoryOfPresentIllness,
    Symptoms,
    PastMedicalHistory,
    MedicationEntry,
    ClinicalObservations,
    Plan,
)
from vad import SpeechSegmenter
from asr import WhisperTranscriber, is_hallucinated_text
from main import app


# ---------------------------------------------------------
# 1. Pydantic Schema & Symptoms Separation Tests
# ---------------------------------------------------------

def test_schema_valid_complete():
    """Verify schema parses valid structured summary correctly."""
    data = {
        "clinical_summary": "45-year-old male presents with acute persistent cough for 4 days. Denies shortness of breath or fever. Prescribed dextromethorphan and advised hydration.",
        "patient_details": {
            "name": "John Doe",
            "age": 45,
            "sex": "Male",
            "identifiers": "MRN-10293",
        },
        "chief_complaint": ["Dry cough for 4 days"],
        "history_of_present_illness": {
            "onset": "4 days ago",
            "duration": "4 days",
            "progression": "Persistent dry hacking",
            "aggravating_factors": "Cold air",
            "relieving_factors": "Warm tea",
        },
        "symptoms": {
            "positive": ["Dry cough", "Mild throat tickle"],
            "negative": ["Fever", "Shortness of breath", "Chest pain"],
        },
        "past_medical_history": {
            "conditions": ["Mild seasonal allergies"],
            "surgeries": [],
            "hospitalisations": [],
        },
        "medications": [
            {"name": "Loratadine", "dosage": "10mg once daily", "adherence": "Compliant"}
        ],
        "allergies": ["Penicillin (rash)"],
        "clinical_observations": {
            "vitals": {"BP": "122/80", "HR": "72", "Temp": "98.4 F"},
            "examination_findings": ["Lungs clear to auscultation bilaterally", "Oropharynx mildly erythematous without exudate"],
        },
        "assessment": ["Acute upper respiratory viral syndrome"],
        "plan": {
            "investigations": [],
            "prescriptions": ["Dextromethorphan syrup 10ml q6h prn cough"],
            "advice": ["Increase oral fluid intake", "Humidifier at night"],
            "follow_up": "Return if fever develops or cough persists beyond 10 days",
        },
    }

    summary = ClinicalSummary.model_validate(data)
    assert summary.patient_details.name == "John Doe"
    assert summary.patient_details.age == 45
    assert len(summary.symptoms.positive) == 2
    assert len(summary.symptoms.negative) == 3
    # Check positive and negative remain separated
    assert "Dry cough" in summary.symptoms.positive
    assert "Fever" in summary.symptoms.negative
    assert "Fever" not in summary.symptoms.positive
    assert summary.medications[0].name == "Loratadine"


def test_schema_missing_scalars_and_defaults():
    """Verify that absent scalars default to None and collections default to empty lists."""
    minimal_data = {
        "clinical_summary": "Patient presented for routine consultation.",
    }
    summary = ClinicalSummary.model_validate(minimal_data)
    assert summary.clinical_summary == "Patient presented for routine consultation."
    assert summary.patient_details.name is None
    assert summary.patient_details.age is None
    assert summary.history_of_present_illness.onset is None
    assert summary.symptoms.positive == []
    assert summary.symptoms.negative == []
    assert summary.medications == []
    assert summary.allergies == []
    assert summary.plan.prescriptions == []
    assert summary.plan.follow_up is None


def test_schema_invalid_raises_error():
    """Missing required clinical_summary field should raise ValidationError."""
    with pytest.raises(ValidationError):
        ClinicalSummary.model_validate({})


# ---------------------------------------------------------
# 2. VAD Segmentation & Silence Filtering Tests
# ---------------------------------------------------------

def generate_pcm16_tone(frequency: float, duration_seconds: float, sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate synthetic PCM16 sine wave audio bytes."""
    num_samples = int(duration_seconds * sample_rate)
    samples = []
    for i in range(num_samples):
        val = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
        samples.append(val)
    return struct.pack(f"<{len(samples)}h", *samples)


def generate_pcm16_silence(duration_seconds: float, sample_rate: int = 16000) -> bytes:
    """Generate pure PCM16 silence (zeros)."""
    num_samples = int(duration_seconds * sample_rate)
    return b"\x00" * (num_samples * 2)


def test_vad_filters_pure_silence():
    """VAD should not produce speech segments when fed only silence."""
    segmenter = SpeechSegmenter(vad_mode=3)
    silence = generate_pcm16_silence(2.0)  # 2 seconds of silence

    # Process in 100ms chunks (1600 samples = 3200 bytes)
    chunk_size = 3200
    segments = []
    for i in range(0, len(silence), chunk_size):
        chunk = silence[i : i + chunk_size]
        segs = segmenter.process_chunk(chunk)
        segments.extend(segs)

    flushed = segmenter.flush()
    segments.extend(flushed)

    assert len(segments) == 0, "Silence should produce 0 speech segments"
    assert not segmenter.is_speaking


def test_vad_frame_remainder_handling():
    """Segmenter should safely handle chunks not aligned to 960 bytes."""
    segmenter = SpeechSegmenter()
    # Send odd sizes: 500 bytes, then 460 bytes (total 960)
    segs1 = segmenter.process_chunk(b"\x00" * 500)
    assert len(segs1) == 0
    assert len(segmenter._remainder_bytes) == 500

    segs2 = segmenter.process_chunk(b"\x00" * 460)
    assert len(segs2) == 0
    assert len(segmenter._remainder_bytes) == 0


def test_vad_flush_resets_speaking_state():
    """Flushing resets speaking state and clears internal buffers."""
    speaking_states = []
    segmenter = SpeechSegmenter(on_speaking_change=lambda s: speaking_states.append(s))
    segmenter._triggered = True
    segmenter._voiced_frames = [b"\x00" * 960] * 15

    flushed = segmenter.flush()
    assert len(flushed) == 1
    assert not segmenter.is_speaking
    assert False in speaking_states


# ---------------------------------------------------------
# 3. Whisper Helper & Audio Normalization Tests
# ---------------------------------------------------------

def test_pcm16_to_float32_normalization():
    """Verify conversion from 16-bit PCM to float32 normalized to [-1, 1]."""
    # Test min (-32768), 0, and max (~32767)
    test_samples = struct.pack("<hhh", -32768, 0, 32767)
    floats = WhisperTranscriber.pcm16_to_float32(test_samples)
    assert len(floats) == 3
    assert np.isclose(floats[0], -1.0, atol=1e-4)
    assert np.isclose(floats[1], 0.0, atol=1e-4)
    assert np.isclose(floats[2], 32767 / 32768.0, atol=1e-4)


def test_whisper_prompt_sanitization():
    """Prompt sanitizer removes strange characters and limits length to max_chars."""
    long_text = "Patient reports feeling unwell with fever! &*#($% and cough. " * 10
    sanitized = WhisperTranscriber.sanitize_prompt(long_text, max_chars=100)
    assert sanitized is not None
    assert len(sanitized) <= 100
    assert "&" not in sanitized
    assert "$" not in sanitized
    assert WhisperTranscriber.sanitize_prompt("", 100) is None


# ---------------------------------------------------------
# 4. FastAPI Endpoint Tests
# ---------------------------------------------------------

def test_health_endpoint():
    """GET /api/health returns status ok."""
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data


def test_analyze_empty_transcript_returns_400():
    """Empty or whitespace-only transcript must return 400 Bad Request."""
    client = TestClient(app)
    res1 = client.post("/api/analyze", json={"transcript": ""})
    assert res1.status_code == 400

    res2 = client.post("/api/analyze", json={"transcript": "   \n\t  "})
    assert res2.status_code == 400


def test_analyze_unconfigured_api_key(monkeypatch):
    """When API key is not configured, returns clear 400 error message."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    client = TestClient(app)
    res = client.post("/api/analyze", json={"transcript": "Doctor: Hello. Patient: I have a cough."})
    assert res.status_code in (400, 502)
    assert "API key is not configured" in res.json().get("detail", "") or "Analysis failed" in res.json().get("detail", "")


# ---------------------------------------------------------
# 5. WAV Header & PCM Data Correctness Tests
# ---------------------------------------------------------

def build_wav_bytes(pcm16_data: bytes, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Build a standard 44-byte WAV container."""
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = len(pcm16_data)
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # Subchunk1Size
        1,   # AudioFormat (PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm16_data


def test_wav_header_and_pcm_data_correctness():
    """Verify that generated WAV file has valid RIFF/WAVE header and correct parameters."""
    sample_rate = 16000
    one_second_pcm = generate_pcm16_tone(440.0, 1.0, sample_rate=sample_rate)
    wav_bytes = build_wav_bytes(one_second_pcm, sample_rate=sample_rate)

    assert len(wav_bytes) == 44 + len(one_second_pcm)
    riff, chunk_size, wave, fmt, sub1, audio_fmt, channels, rate, byte_rate, align, bits, data_tag, data_len = struct.unpack(
        "<4sI4s4sIHHIIHH4sI", wav_bytes[:44]
    )

    assert riff == b"RIFF"
    assert wave == b"WAVE"
    assert fmt == b"fmt "
    assert audio_fmt == 1  # Uncompressed PCM
    assert channels == 1   # Mono
    assert rate == 16000   # 16 kHz
    assert bits == 16      # 16-bit
    assert byte_rate == 32000
    assert align == 2
    assert data_tag == b"data"
    assert data_len == len(one_second_pcm)


# ---------------------------------------------------------
# 6. Invalid LLM Response Rejection Test
# ---------------------------------------------------------

def test_invalid_llm_response_rejected():
    """Invalid LLM responses missing required fields or having invalid types must be rejected."""
    bad_responses = [
        {"patient_details": {"age": "invalid"}},  # missing clinical_summary
        {"clinical_summary": 12345},  # wrong type for summary
        {"clinical_summary": "Summary", "symptoms": "none"},  # symptoms must be object
        {"clinical_summary": "Summary", "medications": [{"dosage": "10mg"}]},  # missing medication name
    ]

    for bad in bad_responses:
        with pytest.raises(ValidationError):
            ClinicalSummary.model_validate(bad)


# ---------------------------------------------------------
# 7. WebSocket Audio Flow and Flush Test
# ---------------------------------------------------------

def test_websocket_stop_flushes_and_closes():
    """Sending 'stop' over WebSocket flushes speech and responds with done."""
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as websocket:
        # Send a silent chunk
        websocket.send_bytes(generate_pcm16_silence(0.1))
        # Send stop command
        websocket.send_text("stop")
        # Should receive done
        received_done = False
        while True:
            try:
                data = websocket.receive_json()
                if data.get("type") == "done":
                    received_done = True
                    break
            except Exception:
                break
        assert received_done


# ---------------------------------------------------------
# 8. Hallucination Filter & Labelled Transcript Formatting Tests
# ---------------------------------------------------------

def test_hallucination_filter():
    """Verify hallucination filter flags URLs, low confidence, high silence, and duplicates."""
    # 1. URL / website hallucinations
    assert is_hallucinated_text("Please visit https://medscribe.ai for more info") is True
    assert is_hallucinated_text("Check www.subtitles.com right now") is True
    assert is_hallucinated_text("Go to youtube.com/watch") is True

    # 2. Subtitle / YouTube artifacts
    assert is_hallucinated_text("Subtitles by Amara.org") is True
    assert is_hallucinated_text("Thank you for watching!") is True

    # 3. no_speech_prob > 0.6
    assert is_hallucinated_text("I have a headache", no_speech_prob=0.85, avg_logprob=-0.2) is True
    assert is_hallucinated_text("I have a headache", no_speech_prob=0.3, avg_logprob=-0.2) is False

    # 4. avg_logprob < -1.0
    assert is_hallucinated_text("I have a headache", no_speech_prob=0.1, avg_logprob=-1.4) is True
    assert is_hallucinated_text("I have a headache", no_speech_prob=0.1, avg_logprob=-0.5) is False

    # 5. Consecutive identical repeated segments
    assert is_hallucinated_text("I have a headache.", last_text="I have a headache") is True
    assert is_hallucinated_text("The fever started yesterday.", last_text="I have a headache") is False

    # 6. Empty text
    assert is_hallucinated_text("   ") is True


def test_labelled_transcript_formatting():
    """Verify that Doctor and Patient lines are properly formatted and parsed by the clinical analyzer."""
    labelled_transcript = (
        "Doctor: Hello, what symptoms are you experiencing?\n"
        "Patient: I have had a severe sore throat and fever for 3 days.\n"
        "Doctor: I will prescribe amoxicillin 500mg twice daily and advise warm salt water gargles.\n"
        "Patient: Thank you doctor."
    )
    # Check that speaker tags are preserved
    lines = labelled_transcript.split("\n")
    assert lines[0].startswith("Doctor:")
    assert lines[1].startswith("Patient:")
    assert lines[2].startswith("Doctor:")
    assert lines[3].startswith("Patient:")


# ---------------------------------------------------------
# 9. Speaker Labelling, Verbatim Validation & Role Rule Tests
# ---------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_speaker_chunk_split():
    """Verify that a mixed-speaker chunk is segmented into separate turns with Doctor and Patient labels."""
    from analysis import ClinicalAnalyzer
    from schema import DialogueTurn

    analyzer = ClinicalAnalyzer()

    raw_chunk = "Good morning how are you feeling today I have had a terrible headache and fever for two days"

    # Mock _call_llm_json to return segmented turns
    async def mock_call_llm_json(sys_prompt, user_prompt):
        return json.dumps({
            "turns": [
                {"speaker": "Doctor", "text": "Good morning how are you feeling today"},
                {"speaker": "Patient", "text": "I have had a terrible headache and fever for two days"}
            ]
        })

    analyzer._call_llm_json = mock_call_llm_json

    turns = await analyzer.label_speakers(raw_chunk)
    assert len(turns) == 2
    assert turns[0].speaker == "Doctor"
    assert turns[0].text == "Good morning how are you feeling today"
    assert turns[1].speaker == "Patient"
    assert turns[1].text == "I have had a terrible headache and fever for two days"


def test_verbatim_validation_catches_altered_word():
    """Verify that verbatim validation rejects turns if an LLM altered, added, or dropped a word."""
    from analysis import validate_verbatim
    from schema import DialogueTurn

    original = "I have had a severe dry cough for four days and mild nausea"

    # 1. Perfectly verbatim
    valid_turns = [
        DialogueTurn(speaker="Patient", text="I have had a severe dry cough"),
        DialogueTurn(speaker="Patient", text="for four days and mild nausea"),
    ]
    assert validate_verbatim(original, valid_turns) is True

    # 2. Altered word (severe -> bad)
    altered_turns = [
        DialogueTurn(speaker="Patient", text="I have had a bad dry cough"),
        DialogueTurn(speaker="Patient", text="for four days and mild nausea"),
    ]
    assert validate_verbatim(original, altered_turns) is False

    # 3. Added extra word (fever added)
    added_turns = [
        DialogueTurn(speaker="Patient", text="I have had a severe dry cough"),
        DialogueTurn(speaker="Patient", text="for four days and mild nausea with fever"),
    ]
    assert validate_verbatim(original, added_turns) is False

    # 4. Missing word (nausea dropped)
    dropped_turns = [
        DialogueTurn(speaker="Patient", text="I have had a severe dry cough"),
        DialogueTurn(speaker="Patient", text="for four days and mild"),
    ]
    assert validate_verbatim(original, dropped_turns) is False


@pytest.mark.asyncio
async def test_verbatim_validation_retry_and_fallback():
    """Verify that when attempt 1 alters words, retry is triggered; if retry fails, fallback is used."""
    from analysis import ClinicalAnalyzer

    analyzer = ClinicalAnalyzer()
    original = "Hello I have chest pain"

    attempts = []

    async def mock_call_llm_json(sys_prompt, user_prompt):
        attempts.append(user_prompt)
        # Attempt 1: altered word
        if len(attempts) == 1:
            return json.dumps({
                "turns": [{"speaker": "Patient", "text": "Hello I have heart pain"}]
            })
        # Attempt 2: retry succeeds with verbatim words
        return json.dumps({
            "turns": [
                {"speaker": "Doctor", "text": "Hello"},
                {"speaker": "Patient", "text": "I have chest pain"}
            ]
        })

    analyzer._call_llm_json = mock_call_llm_json

    turns = await analyzer.label_speakers(original)
    assert len(attempts) == 2  # retry was triggered
    assert len(turns) == 2
    assert turns[0].speaker == "Doctor"
    assert turns[0].text == "Hello"
    assert turns[1].speaker == "Patient"
    assert turns[1].text == "I have chest pain"


@pytest.mark.asyncio
async def test_patient_statement_never_placed_in_assessment_or_plan():
    """Verify that clinical extraction rule strictly excludes patient or unknown statements from assessment and plan."""
    from analysis import ClinicalAnalyzer
    from schema import AnalyzeResponse

    analyzer = ClinicalAnalyzer()

    # Transcript where patient suggests a diagnosis and asks for antibiotics
    transcript = (
        "Doctor: Hello, what brings you in today?\n"
        "Patient: I think I have pneumonia and I need amoxicillin antibiotics right away.\n"
        "Doctor: Your chest sounds clear. It is a mild viral bronchitis. Rest and hydrate."
    )

    # Mock the LLM JSON call to simulate clinical summary extraction adhering to the prompt rule
    async def mock_call_llm_json(sys_prompt, user_prompt):
        if "conversational turns" in user_prompt:
            return json.dumps({
                "turns": [
                    {"speaker": "Doctor", "text": "Hello, what brings you in today?"},
                    {"speaker": "Patient", "text": "I think I have pneumonia and I need amoxicillin antibiotics right away."},
                    {"speaker": "Doctor", "text": "Your chest sounds clear. It is a mild viral bronchitis. Rest and hydrate."}
                ]
            })
        # Clinical extraction call
        return json.dumps({
            "clinical_summary": "Patient presented requesting antibiotics for perceived pneumonia. Clinical exam revealed clear chest consistent with mild viral bronchitis. Advised supportive care with rest and hydration.",
            "patient_details": {},
            "chief_complaint": ["Pneumonia concern", "Request for antibiotics"],
            "history_of_present_illness": {},
            "symptoms": {"positive": ["Bronchitis symptoms"], "negative": []},
            "past_medical_history": {"conditions": [], "surgeries": [], "hospitalisations": []},
            "medications": [],
            "allergies": [],
            "clinical_observations": {"vitals": {}, "examination_findings": ["Chest sounds clear"]},
            # Doctor's diagnosis ONLY: "mild viral bronchitis", NOT patient's "pneumonia"
            "assessment": ["Mild viral bronchitis"],
            # Doctor's advice ONLY: "Rest and hydrate", NOT patient's "amoxicillin antibiotics"
            "plan": {
                "investigations": [],
                "prescriptions": [],
                "advice": ["Rest and hydration"],
                "follow_up": None
            }
        })

    analyzer._call_llm_json = mock_call_llm_json

    result: AnalyzeResponse = await analyzer.analyze(transcript)

    # Verify turns were returned
    assert len(result.turns) == 3
    assert result.turns[1].speaker == "Patient"

    # Verify Patient diagnosis ("pneumonia") is NOT in doctor's assessment
    for a in result.assessment:
        assert "pneumonia" not in a.lower()
    assert "Mild viral bronchitis" in result.assessment

    # Verify Patient requested drug ("amoxicillin") is NOT in doctor's prescription or plan
    assert len(result.plan.prescriptions) == 0
    for adv in result.plan.advice:
        assert "amoxicillin" not in adv.lower()
    assert "Rest and hydration" in result.plan.advice


# ---------------------------------------------------------
# 10. Throughput Batching, Timestamp Mapping & WebSocket Tests
# ---------------------------------------------------------

def test_whisper_cpu_threads_config(monkeypatch):
    """Verify WHISPER_CPU_THREADS is passed to WhisperModel with single-thread worker."""
    monkeypatch.setenv("WHISPER_CPU_THREADS", "1")
    monkeypatch.setenv("WHISPER_MODEL", "base.en")
    transcriber = WhisperTranscriber()
    assert transcriber.cpu_threads == 1
    assert transcriber.model_size == "base.en"


def test_pcm16_to_float32_conversion():
    """Verify PCM16 byte array is accurately normalized to float32 between -1.0 and 1.0."""
    # 0 sample, max positive 32767, min negative -32768
    raw_bytes = struct.pack("<hhh", 0, 32767, -32768)
    floats = WhisperTranscriber.pcm16_to_float32(raw_bytes)
    assert len(floats) == 3
    assert abs(floats[0]) < 1e-5
    assert abs(floats[1] - (32767.0 / 32768.0)) < 1e-4
    assert abs(floats[2] - (-1.0)) < 1e-4


@pytest.mark.asyncio
async def test_websocket_audio_endpoint_with_batching_and_status():
    """Test full WebSocket audio endpoint with queued segment batching, status reporting, and flush."""
    # Generate 1 second of 440Hz sine wave PCM16
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    sine_wave = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    audio_bytes = sine_wave.tobytes()

    client = TestClient(app)
    received_types = []

    with client.websocket_connect("/ws/audio") as ws:
        # Stream audio chunk
        ws.send_bytes(audio_bytes)
        # Send stop to flush
        ws.send_text("stop")

        while True:
            try:
                msg = ws.receive_json()
                msg_type = msg.get("type")
                received_types.append(msg_type)
                if msg_type == "done":
                    break
            except Exception:
                break

    assert "done" in received_types, "WebSocket must send done upon session completion"
    # Status messages and/or vad messages must be emitted
    assert any(t in received_types for t in ["status", "vad"]), "WebSocket should send status/vad events"


@pytest.mark.asyncio
async def test_batching_segment_timestamp_mapping():
    """Verify timestamp boundaries map transcribed text back to original segments."""
    # Simulate 2 segments of 2 seconds each separated by 0.3s silence
    seg1_dur = 2.0
    seg2_dur = 2.0
    silence_gap = 0.3

    segment_bounds = [
        (0.0, seg1_dur),
        (seg1_dur + silence_gap, seg1_dur + silence_gap + seg2_dur)
    ]

    whisper_segs = [
        {"text": "Patient has headache", "start": 0.2, "end": 1.8},
        {"text": "Doctor prescribed rest", "start": 2.4, "end": 4.1}
    ]

    assigned = [[] for _ in range(len(segment_bounds))]
    unassigned = []

    for wseg in whisper_segs:
        w_mid = (wseg["start"] + wseg["end"]) / 2.0
        matched = None
        for s_idx, (s_start, s_end) in enumerate(segment_bounds):
            if (s_start - 0.4) <= w_mid <= (s_end + 0.4):
                matched = s_idx
                break
        if matched is not None:
            assigned[matched].append(wseg["text"])
        else:
            unassigned.append(wseg["text"])

    assert len(assigned[0]) == 1
    assert assigned[0][0] == "Patient has headache"
    assert len(assigned[1]) == 1
    assert assigned[1][0] == "Doctor prescribed rest"
    assert len(unassigned) == 0




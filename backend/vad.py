"""Voice Activity Detection (VAD) and speech segmentation using WebRTC VAD.

Processes 16 kHz mono signed 16-bit PCM audio frames into segmented speech bursts,
filtering out silence and ambient noise while avoiding clipping on speech onset.
Features configurable pre-padding, post-padding, and closure silence duration.
"""

from __future__ import annotations

import collections
from typing import Callable, Deque, List, Optional, Tuple
import webrtcvad


class SpeechSegmenter:
    """Segments continuous 16 kHz mono PCM16 audio into speech segments using WebRTC VAD.

    - Sample Rate: 16,000 Hz
    - Frame Duration: 30 ms (480 samples = 960 bytes)
    - Default VAD aggressiveness: 2 (moderate, less aggressive than 3 to avoid clipping)
    - Pre-speech padding: ~500 ms (17 frames)
    - Post-speech closure silence: ~700 ms (23-24 frames of silence required to close segment)
    - Post-speech padding: ~500 ms preserved at the end of speech segment
    """

    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2  # 16-bit signed PCM
    FRAME_DURATION_MS = 30
    SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480
    BYTES_PER_FRAME = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE  # 960

    def __init__(
        self,
        vad_mode: int = 2,
        ring_buffer_size: int = 17,  # ~510 ms ring buffer for onset detection & pre-padding
        onset_ratio: float = 0.6,    # ~60% speech frames in buffer to trigger onset smoothly
        closure_silence_ms: int = 700, # ~700 ms of silence to close segment
        post_speech_padding_ms: int = 500, # ~500 ms padding kept after speech ends
        min_segment_seconds: float = 0.4,
        max_segment_seconds: float = 15.0,
        on_speaking_change: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Initialize the segmenter.

        Args:
            vad_mode: WebRTC VAD aggressiveness mode (0 to 3, 2 is balanced).
            ring_buffer_size: Number of frames in sliding ring buffer (17 frames = 510 ms).
            onset_ratio: Threshold ratio of speech frames to trigger speech onset.
            closure_silence_ms: Duration of consecutive silence in ms required to close segment.
            post_speech_padding_ms: Duration in ms of audio preserved as post-speech padding.
            min_segment_seconds: Minimum duration for a valid speech segment.
            max_segment_seconds: Maximum duration before segment is split.
            on_speaking_change: Optional callback when speaking status transitions.
        """
        self.vad = webrtcvad.Vad(vad_mode)
        self.ring_buffer_size = ring_buffer_size
        self.onset_ratio = onset_ratio
        self.closure_silence_frames = max(1, int(closure_silence_ms / self.FRAME_DURATION_MS)) # ~23 frames
        self.post_padding_frames = max(0, int(post_speech_padding_ms / self.FRAME_DURATION_MS)) # ~16-17 frames
        self.min_segment_seconds = min_segment_seconds
        self.max_segment_seconds = max_segment_seconds
        self.on_speaking_change = on_speaking_change

        self.min_frames = int(min_segment_seconds / (self.FRAME_DURATION_MS / 1000.0))
        self.max_frames = int(max_segment_seconds / (self.FRAME_DURATION_MS / 1000.0))

        self._remainder_bytes = bytearray()
        self._ring_buffer: Deque[Tuple[bytes, bool]] = collections.deque(maxlen=self.ring_buffer_size)
        self._triggered = False
        self._voiced_frames: List[bytes] = []
        self._trailing_silence_frames: List[bytes] = []
        self._consecutive_silence_count = 0

    @property
    def is_speaking(self) -> bool:
        """Current VAD speaking state."""
        return self._triggered

    def _set_speaking_state(self, new_state: bool) -> None:
        if self._triggered != new_state:
            self._triggered = new_state
            if self.on_speaking_change:
                try:
                    self.on_speaking_change(new_state)
                except Exception:
                    pass

    def process_chunk(self, chunk: bytes) -> List[bytes]:
        """Process incoming raw PCM16 bytes and return completed speech segments if any.

        Args:
            chunk: Arbitrary size chunk of 16 kHz mono 16-bit PCM bytes.

        Returns:
            List of completed speech segments as concatenated PCM16 bytes.
        """
        if not chunk:
            return []

        self._remainder_bytes.extend(chunk)
        completed_segments: List[bytes] = []

        while len(self._remainder_bytes) >= self.BYTES_PER_FRAME:
            frame = bytes(self._remainder_bytes[: self.BYTES_PER_FRAME])
            del self._remainder_bytes[: self.BYTES_PER_FRAME]

            segment = self._process_frame(frame)
            if segment:
                completed_segments.append(segment)

        return completed_segments

    def _process_frame(self, frame: bytes) -> Optional[bytes]:
        """Process an exact 30 ms frame (960 bytes)."""
        is_speech = self.vad.is_speech(frame, self.SAMPLE_RATE)
        self._ring_buffer.append((frame, is_speech))

        if not self._triggered:
            # Check for onset: sufficient speech frames in the ring buffer
            if len(self._ring_buffer) >= min(10, self.ring_buffer_size):
                num_voiced = sum(1 for _, speech in self._ring_buffer if speech)
                if num_voiced / len(self._ring_buffer) >= self.onset_ratio:
                    self._set_speaking_state(True)
                    # Include all buffered pre-trigger frames (~500 ms padding) to avoid clipping first words
                    for buffered_frame, _ in self._ring_buffer:
                        self._voiced_frames.append(buffered_frame)
                    self._ring_buffer.clear()
                    self._trailing_silence_frames.clear()
                    self._consecutive_silence_count = 0
            return None
        else:
            # Currently in speech segment
            if is_speech:
                # If there were uncommitted silence frames between words, keep them in the utterance
                if self._trailing_silence_frames:
                    self._voiced_frames.extend(self._trailing_silence_frames)
                    self._trailing_silence_frames.clear()
                self._voiced_frames.append(frame)
                self._consecutive_silence_count = 0
            else:
                self._consecutive_silence_count += 1
                self._trailing_silence_frames.append(frame)

            # Check if segment exceeded maximum duration (split long segment)
            total_current_frames = len(self._voiced_frames) + len(self._trailing_silence_frames)
            if total_current_frames >= self.max_frames:
                segment_bytes = b"".join(self._voiced_frames + self._trailing_silence_frames)
                self._voiced_frames.clear()
                self._trailing_silence_frames.clear()
                self._ring_buffer.clear()
                self._consecutive_silence_count = 0
                return segment_bytes

            # Check for segment closure: consecutive silence reached closure threshold (~700 ms)
            if self._consecutive_silence_count >= self.closure_silence_frames:
                self._set_speaking_state(False)

                # Keep up to ~500 ms post-speech padding
                padding_to_keep = self._trailing_silence_frames[: self.post_padding_frames]
                final_frames = self._voiced_frames + padding_to_keep

                segment_bytes = None
                if len(final_frames) >= self.min_frames:
                    segment_bytes = b"".join(final_frames)

                self._voiced_frames.clear()
                self._trailing_silence_frames.clear()
                self._ring_buffer.clear()
                self._consecutive_silence_count = 0
                return segment_bytes

            return None

    def flush(self) -> List[bytes]:
        """Flush remaining buffered audio when stopping recording.

        Ensures the final sentence or speech segment is not lost.
        """
        completed: List[bytes] = []

        # If remainder bytes form at least half a frame, pad with zeros and process
        if len(self._remainder_bytes) >= self.BYTES_PER_FRAME // 2:
            padding = self.BYTES_PER_FRAME - len(self._remainder_bytes)
            padded_frame = bytes(self._remainder_bytes) + (b"\x00" * padding)
            self._remainder_bytes.clear()
            segment = self._process_frame(padded_frame)
            if segment:
                completed.append(segment)
        else:
            self._remainder_bytes.clear()

        # If currently speaking or have voiced frames accumulated
        all_pending_frames = self._voiced_frames + self._trailing_silence_frames
        if all_pending_frames:
            if len(all_pending_frames) >= self.min_frames or not completed:
                completed.append(b"".join(all_pending_frames))
            self._voiced_frames.clear()
            self._trailing_silence_frames.clear()

        self._set_speaking_state(False)
        self._ring_buffer.clear()
        self._consecutive_silence_count = 0
        return completed

    def reset(self) -> None:
        """Reset internal buffers and state."""
        self._remainder_bytes.clear()
        self._ring_buffer.clear()
        self._voiced_frames.clear()
        self._trailing_silence_frames.clear()
        self._consecutive_silence_count = 0
        self._set_speaking_state(False)

from __future__ import annotations

"""Continuous PCM mixer for Discord voice. discord.py allows one AudioSource per VoiceClient;
:class:`VoiceMixer` IS that source: installed once per guild, never stops, and every 20 ms sums its
children (looping ambient bed + one-shot speech that ducks the bed) clamped to int16. ``read`` runs on
discord.py's sender thread while children change on the asyncio loop, hence the Lock. Outgoing only."""

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

import discord

try:
    from .ffmpeg_utils import resolve_ffmpeg_executable
except ImportError:
    from ffmpeg_utils import resolve_ffmpeg_executable

if TYPE_CHECKING:  # numpy is an optional ("voice" extra) dep — never import at runtime top-level
    import numpy as np

logger = logging.getLogger(__name__)


def _require_numpy():
    """Lazy numpy import: the adapter imports this module unconditionally, so a missing
    ``voice`` extra must fail at mix time, not at import time."""
    import numpy as np  # noqa: PLC0415 — intentional lazy import
    return np

# Discord-native frame geometry (matches discord.opus.Encoder): 48 kHz, stereo, s16, 20 ms frames.
SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, FRAME_LENGTH_MS = 48000, 2, 2, 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_LENGTH_MS // 1000   # 960
FRAME_SIZE = SAMPLES_PER_FRAME * CHANNELS * SAMPLE_WIDTH    # 3840 bytes
BYTES_PER_MS = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH // 1000  # 192
SILENCE_FRAME = b"\x00" * FRAME_SIZE
STREAMING_BUFFER_BYTES = BYTES_PER_MS * 2000


class MixerChild:
    """One 48 kHz / stereo / s16le PCM stream feeding :class:`VoiceMixer`; ``read_frame``
    yields 20 ms frames, optionally looping, with per-child gain and linear fade-in."""

    __slots__ = ("_pcm", "_pos", "loop", "gain", "fade_frames", "_fade_done", "_finished")

    def __init__(self, pcm: bytes, *, loop: bool = False, gain: float = 1.0, fade_in_ms: int = 0):
        # Pad to whole frames so looping is seamless and the final partial frame doesn't click.
        self._pcm = pcm + b"\x00" * (-len(pcm) % FRAME_SIZE)
        self._pos = self._fade_done = 0
        self.loop, self.gain = loop, float(gain)
        self.fade_frames = max(0, fade_in_ms // FRAME_LENGTH_MS)
        self._finished = False

    def read_frame(self) -> "Optional[np.ndarray]":
        """Next 20 ms frame as a float32 ndarray, or None when done."""
        if self._finished:
            return None
        if self._pos >= len(self._pcm):
            if self.loop and self._pcm:
                self._pos = 0
            else:
                self._finished = True
                return None
        np = _require_numpy()
        chunk = self._pcm[self._pos:self._pos + FRAME_SIZE]
        self._pos += FRAME_SIZE
        if len(chunk) < FRAME_SIZE:
            chunk = chunk + b"\x00" * (FRAME_SIZE - len(chunk))
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        gain = self.gain
        if self.fade_frames and self._fade_done < self.fade_frames:
            self._fade_done += 1
            gain *= self._fade_done / self.fade_frames
        if gain != 1.0:
            samples = samples * gain
        return samples


class StreamingSpeechChild:
    """A speech child whose PCM grows while the TTS provider streams.

    Unlike :class:`MixerChild` (one complete clip known up front), this child
    is fed 48 kHz / stereo / s16le PCM incrementally by the gateway's
    streaming-TTS consumer while the LLM is still generating, so playback
    starts on the first synthesised sentence instead of after the whole reply.

    Threading: ``append`` / ``finish`` / ``abort`` are called from the
    asyncio event-loop thread; :meth:`read_frame` is called from discord.py's
    audio sender thread (under the mixer's lock).  The child's own lock
    guards the buffer; it never takes the mixer's lock, so the lock order is
    always mixer → child and cannot deadlock.

    Starvation (producer slower than realtime) yields silence frames instead
    of finishing, so a mid-sentence provider stall does not end the speech
    segment early and re-trigger ducking / completion bookkeeping.
    """

    __slots__ = (
        "name", "gain", "is_speech", "fade_frames", "_fade_done",
        "_buf", "_eof", "_aborted", "_lock", "_max_buffer_bytes",
    )

    def __init__(
        self,
        name: str = "speech-stream",
        *,
        gain: float = 1.0,
        fade_in_ms: int = 40,
        max_buffer_bytes: int = STREAMING_BUFFER_BYTES,
    ):
        self.name = name
        self.gain = float(gain)
        self.is_speech = True
        self.fade_frames = max(0, fade_in_ms // FRAME_LENGTH_MS)
        self._fade_done = 0
        self._buf = bytearray()
        self._eof = False
        self._aborted = False
        self._lock = threading.Condition()
        self._max_buffer_bytes = max(FRAME_SIZE, int(max_buffer_bytes))

    def append(self, pcm: bytes) -> bool:
        """Append mixer PCM, waiting for bounded capacity. False after finish/abort."""
        if not pcm:
            return True
        with self._lock:
            offset = 0
            while offset < len(pcm):
                while (
                    len(self._buf) >= self._max_buffer_bytes
                    and not self._aborted
                    and not self._eof
                ):
                    self._lock.wait()
                if self._aborted or self._eof:
                    return False
                available = self._max_buffer_bytes - len(self._buf)
                end = min(len(pcm), offset + available)
                self._buf.extend(pcm[offset:end])
                offset = end
            return True

    def finish(self) -> None:
        """Signal end-of-stream: the buffered tail plays out, then done."""
        with self._lock:
            self._eof = True
            self._lock.notify_all()

    def abort(self) -> None:
        """Drop buffered and future audio immediately (idempotent)."""
        with self._lock:
            self._aborted = True
            self._buf.clear()
            self._lock.notify_all()

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._aborted or (self._eof and not self._buf)

    def read_frame(self) -> "Optional[np.ndarray]":
        """Return the next 20 ms frame as a float32 ndarray.

        ``None`` ends the child (only after abort, or EOF + drained buffer).
        While starving mid-stream it returns silence so the mixer keeps the
        speech segment (and ambient duck) alive until more audio lands.
        """
        with self._lock:
            if self._aborted:
                return None
            np = _require_numpy()
            if len(self._buf) >= FRAME_SIZE:
                raw = bytes(self._buf[:FRAME_SIZE])
                del self._buf[:FRAME_SIZE]
                self._lock.notify_all()
            elif self._eof and self._buf:
                raw = bytes(self._buf) + b"\x00" * (FRAME_SIZE - len(self._buf))
                self._buf.clear()
                self._lock.notify_all()
            elif self._eof:
                return None
            else:
                # Starving: producer hasn't caught up to playback realtime.
                return np.zeros(SAMPLES_PER_FRAME * CHANNELS, dtype=np.float32)

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        gain = self.gain
        if self.fade_frames and self._fade_done < self.fade_frames:
            self._fade_done += 1
            gain *= self._fade_done / self.fade_frames
        if gain != 1.0:
            samples = samples * gain
        return samples


class VoiceMixer(discord.AudioSource):
    """Continuous ``discord.AudioSource`` mixing N children: :meth:`set_ambient` installs the
    looping idle bed, :meth:`play_speech` layers a one-shot clip over it (ducking the bed).
    Both are safe from the asyncio thread while discord.py drains :meth:`read`."""

    def is_opus(self) -> bool:  # pragma: no cover - trivial
        return False

    def __init__(self, *, ambient_gain: float = 0.18, duck_gain: float = 0.06, speech_gain: float = 1.0,
                 duck_release_ms: int = 400):
        self._lock = threading.Lock()
        self._ambient: Optional[MixerChild] = None
        self._speech: List[MixerChild] = []
        self._ambient_gain, self._duck_gain, self._speech_gain = float(ambient_gain), float(duck_gain), float(speech_gain)
        # When speech ends, ramp the ambient back up over this many frames instead of jumping.
        self._duck_release_frames = max(1, duck_release_ms // FRAME_LENGTH_MS)
        self._duck_release_left = 0
        self._closed = self._speech_active = False

    def set_ambient(self, pcm: Optional[bytes], *, gain: Optional[float] = None) -> None:
        """Install (or clear, with ``pcm=None``) the looping ambient bed."""
        with self._lock:
            if gain is not None:
                self._ambient_gain = float(gain)
            if not pcm:
                self._ambient = None
                return
            gain_now = self._duck_gain if self._speech_active else self._ambient_gain
            self._ambient = MixerChild(pcm, loop=True, gain=gain_now, fade_in_ms=200)

    def play_speech(self, pcm: bytes, *, gain: Optional[float] = None, fade_in_ms: int = 40) -> None:
        """Layer a one-shot speech clip over the ambient bed (ducks ambient)."""
        if not pcm:
            return
        with self._lock:
            self._speech.append(MixerChild(
                pcm, gain=self._speech_gain if gain is None else float(gain), fade_in_ms=fade_in_ms,
            ))
            self._speech_active = True
            self._duck_release_left = 0
            if self._ambient is not None:
                self._ambient.gain = self._duck_gain

    def play_speech_stream(self, child: "StreamingSpeechChild") -> None:
        """Attach a streaming speech child (grows while the provider streams).

        Same ducking semantics as :meth:`play_speech`; the child is dropped
        by :meth:`read` once it reports finished (abort, or EOF + drained).
        """
        with self._lock:
            if self._closed:
                child.abort()
                return
            self._speech.append(child)
            self._speech_active = True
            self._duck_release_left = 0
            if self._ambient is not None:
                self._ambient.gain = self._duck_gain

    @property
    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_active

    def stop_speech(self) -> None:
        """Drop any in-flight speech immediately and release the duck."""
        with self._lock:
            self._speech.clear()
            self._begin_duck_release_locked()

    def _begin_duck_release_locked(self) -> None:
        self._speech_active = False
        self._duck_release_left = self._duck_release_frames

    def read(self) -> bytes:
        """One 20 ms mixed PCM frame (always FRAME_SIZE bytes) — never b"", which would stop
        discord.py's player; the mixer must run for the lifetime of the connection."""
        with self._lock:
            if self._closed:
                return SILENCE_FRAME
            np = _require_numpy()
            acc: "Optional[np.ndarray]" = None
            # Speech children (drop exhausted ones; release duck when last ends)
            if self._speech:
                still_live: List[MixerChild] = []
                for child in self._speech:
                    frame = child.read_frame()
                    if frame is None:
                        continue
                    acc = frame if acc is None else acc + frame
                    still_live.append(child)
                self._speech = still_live
                if not self._speech and self._speech_active:
                    self._begin_duck_release_locked()
            # Ambient bed — ramp gain back up during duck-release.
            if self._ambient is not None:
                if self._duck_release_left > 0 and not self._speech_active:
                    self._duck_release_left -= 1
                    frac = 1.0 - (self._duck_release_left / self._duck_release_frames)
                    self._ambient.gain = self._duck_gain + (self._ambient_gain - self._duck_gain) * frac
                elif not self._speech_active and self._duck_release_left == 0:
                    self._ambient.gain = self._ambient_gain
                amb = self._ambient.read_frame()
                if amb is not None:
                    acc = amb if acc is None else acc + amb
            if acc is None:
                return SILENCE_FRAME
            np.clip(acc, -32768, 32767, out=acc)
            return acc.astype(np.int16).tobytes()

    def cleanup(self) -> None:  # called by discord.py when playback stops
        with self._lock:
            self._closed = True
            self._ambient = None
            for child in self._speech:
                abort = getattr(child, "abort", None)
                if abort is not None:
                    abort()
            self._speech.clear()


# ----------------------------------------------------------------------
# PCM helpers
# ----------------------------------------------------------------------

def resample_to_mixer_pcm(
    data: bytes,
    sample_rate: int,
    channels: int,
    state: Optional[bytearray] = None,
) -> bytes:
    """Convert one s16le PCM chunk to the mixer format (48 kHz stereo s16le).

    The streaming TTS providers emit 24 kHz mono s16le; the mixer speaks
    Discord-native 48 kHz stereo.  Linear interpolation (numpy) is used for
    the rate change and the channel count is normalised (stereo downmixes to
    mono first, then the mono signal is duplicated to both ears).

    ``state`` is an optional caller-owned bytearray carrying the leftover
    odd byte when a provider chunk splits a 16-bit sample; pass the same
    object on every call of one stream.  (All current streamers send
    sample-aligned chunks, but the wire format does not guarantee it.)
    """
    if state is not None:
        data = bytes(state) + data
        state.clear()
    if len(data) % 2:
        if state is not None:
            state.extend(data[-1:])
        data = data[:-1]
    if not data:
        return b""

    if sample_rate == SAMPLE_RATE and channels == CHANNELS:
        return data

    np = _require_numpy()
    samples = np.frombuffer(data, dtype=np.int16)
    if channels == 2:
        # Downmix to mono before resampling so the rate change runs once.
        samples = samples.reshape(-1, 2).mean(axis=1)
    samples = samples.astype(np.float32)

    if sample_rate != SAMPLE_RATE and len(samples) >= 2:
        n_out = max(1, round(len(samples) * SAMPLE_RATE / sample_rate))
        x_in = np.arange(len(samples), dtype=np.float64)
        # Position-based mapping: output sample j reads the input at the
        # instant it corresponds to (j * 0.5 for 24k -> 48k).
        x_out = np.arange(n_out, dtype=np.float64) * (sample_rate / SAMPLE_RATE)
        samples = np.interp(x_out, x_in, samples)

    stereo = np.repeat(samples[:, None], CHANNELS, axis=1).reshape(-1)
    np.clip(stereo, -32768, 32767, out=stereo)
    return stereo.astype(np.int16).tobytes()


def decode_to_pcm(path: str, *, timeout: float = 30.0) -> Optional[bytes]:
    """Decode any audio file to 48 kHz / stereo / s16le PCM via ffmpeg; None on failure."""
    import subprocess
    try:
        proc = subprocess.run(
            [resolve_ffmpeg_executable(), "-y", "-loglevel", "error", "-i", path, "-f", "s16le",
             "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1"],
            capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("decode_to_pcm failed for %s: %s", path, e)
        return None
    if proc.returncode != 0:
        logger.warning("ffmpeg decode failed for %s (rc=%d): %s",
                       path, proc.returncode, (proc.stderr or b"").decode("utf-8", "replace")[:200])
        return None
    return proc.stdout or None


def synth_ambient_pcm(seconds: float = 4.0) -> bytes:
    """Synthesise a subtle looping ambient bed: two detuned sine partials with a slow tremolo
    plus filtered noise; whole-cycle frequencies make the loop point click-free. Mono -> stereo."""
    np = _require_numpy()
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    def _whole_cycle_freq(target: float) -> float:
        cycles = max(1, round(target * seconds))
        return cycles / seconds
    f1 = _whole_cycle_freq(110.0)
    f2 = _whole_cycle_freq(110.5)
    trem = _whole_cycle_freq(0.5)   # ~0.5 Hz tremolo
    pad = (0.55 * np.sin(2 * np.pi * f1 * t) + 0.45 * np.sin(2 * np.pi * f2 * t))
    tremolo = 0.6 + 0.4 * (0.5 * (1 + np.sin(2 * np.pi * trem * t)))
    signal = pad * tremolo
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    kernel = np.ones(64) / 64.0
    noise = np.convolve(noise, kernel, mode="same")
    signal = signal + 0.08 * noise
    # Normalise to a modest peak (mixer applies the real ambient gain on top).
    peak = float(np.max(np.abs(signal))) or 1.0
    signal = (signal / peak) * 0.5
    mono16 = (signal * 32767.0).astype(np.int16)
    stereo16 = np.repeat(mono16[:, None], CHANNELS, axis=1).reshape(-1)
    return stereo16.tobytes()

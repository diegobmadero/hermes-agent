"""Tests for the Discord streaming-TTS sink (issue #94462).

The gateway's streaming-TTS consumer (#60671) synthesises and plays each LLM
sentence while the model is still generating, but no platform adapter
implemented the sink half of the contract — so Discord voice turns always
waited for the full reply and then synthesised one whole file.  These tests
cover the Discord sink: the PCM resampler, the appendable streaming speech
child, the adapter's contract methods, and the end-to-end consumer → mixer
flow.  No live Discord, network, or TTS credentials: the voice client and
streaming provider are fakes; the mixer and adapter logic are real.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

# numpy ships only in the optional "voice" extra (not [all,dev]); the mixer
# and resampler need it, so skip this whole module when it isn't installed.
np = pytest.importorskip("numpy")

# voice_mixer lives inside the discord plugin package dir; import by path the
# same way the adapter does.
_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import voice_mixer as vm  # noqa: E402

import gateway.run as gateway_run  # noqa: E402
from gateway.config import Platform  # noqa: E402
from gateway.platforms.base import AudioFormat  # noqa: E402
from gateway.session import SessionSource  # noqa: E402
from gateway.streaming_tts_consumer import StreamingTTSConsumer  # noqa: E402
from tools.tts_streaming import SentenceChunker  # noqa: E402


# =====================================================================
# Resampler
# =====================================================================

class TestResampleToMixerPcm:
    def test_passthrough_when_already_mixer_format(self):
        data = b"\x01\x02" * 1920
        assert vm.resample_to_mixer_pcm(data, 48000, 2) is data

    def test_24k_mono_doubles_rate_and_channels(self):
        # 100ms of 24 kHz mono s16 = 2400 samples = 4800 bytes.
        samples = (np.sin(np.arange(2400) / 50.0) * 10000).astype(np.int16)
        out = vm.resample_to_mixer_pcm(samples.tobytes(), 24000, 1)
        result = np.frombuffer(out, dtype=np.int16)
        # 2x rate × 2 channels: ~4x the sample count (interp rounding ±2).
        assert abs(len(result) - 2400 * 4) <= 2
        # Interleaved stereo: both ears carry the same signal.
        assert (result[0::2] == result[1::2]).all()
        # Interpolation preserves the signal (no clipping or wrap).
        assert result.max() <= 32767 and result.min() >= -32768
        assert result.max() > 8000  # amplitude survives

    def test_stereo_input_downmixes_before_resample(self):
        # 24 kHz stereo: left=5000, right=1000 → mono mean=3000, then 2x up.
        left = np.full(480, 5000, dtype=np.int16)
        right = np.full(480, 1000, dtype=np.int16)
        stereo = np.empty(960, dtype=np.int16)
        stereo[0::2] = left
        stereo[1::2] = right
        out = vm.resample_to_mixer_pcm(stereo.tobytes(), 24000, 2)
        result = np.frombuffer(out, dtype=np.int16).astype(np.int32)
        # 480 stereo frames → 480 mono → ~960 mono → duplicated to stereo.
        assert abs(len(result) - 480 * 4) <= 4
        # Downmixed amplitude survives (interp may wobble ±1 around 3000).
        assert np.abs(result - 3000).max() <= 2

    def test_arbitrary_chunk_boundaries_match_one_shot_after_flush(self):
        samples = np.array([0, 1000, 2000, 3000], dtype=np.int16)
        full = samples.tobytes()
        one_shot = vm.resample_to_mixer_pcm(full, 24000, 1)
        state = bytearray()
        pieces = [
            vm.resample_to_mixer_pcm(full[index:index + 1], 24000, 1, state=state)
            for index in range(len(full))
        ]
        pieces.append(vm.resample_to_mixer_pcm(b"", 24000, 1, state=state, final=True))

        assert b"".join(pieces) == one_shot
        assert state == bytearray()

    def test_silence_stays_silence(self):
        out = vm.resample_to_mixer_pcm(b"\x00\x00" * 240, 24000, 1)
        assert set(out) == {0}


# =====================================================================
# StreamingSpeechChild
# =====================================================================

class TestStreamingSpeechChild:
    def test_frames_play_in_append_order(self):
        child = vm.StreamingSpeechChild(fade_in_ms=0)
        frames = [bytes([i]) * vm.FRAME_SIZE for i in (1, 2, 3)]
        for f in frames:
            child.append(f)
        child.finish()
        for expected in frames:
            frame = child.read_frame()
            assert frame is not None
            expected_samples = np.frombuffer(expected, dtype=np.int16)
            assert (frame == expected_samples.astype(np.float32)).all()
        assert child.read_frame() is None
        assert child.finished is True

    def test_starvation_yields_silence_not_completion(self):
        child = vm.StreamingSpeechChild(fade_in_ms=0)
        assert child.read_frame() is not None  # silence frame
        assert child.finished is False
        # Appends after starvation still play.
        child.append(b"\x05\x00" * (vm.FRAME_SIZE // 2))
        child.finish()
        frame = child.read_frame()
        assert frame is not None
        assert (frame == 5.0).all()

    def test_eof_drains_padded_tail(self):
        child = vm.StreamingSpeechChild(fade_in_ms=0)
        child.append(b"\x07\x00" * 50)  # 100 bytes = 50 int16 samples
        child.finish()
        frame = child.read_frame()
        assert frame is not None
        assert (frame[:50] == 7.0).all()
        assert (frame[50:] == 0.0).all()  # zero-padded tail
        assert child.read_frame() is None

    def test_abort_drops_buffered_audio_and_is_idempotent(self):
        child = vm.StreamingSpeechChild()
        child.append(b"\x09" * vm.FRAME_SIZE * 10)
        child.abort()
        assert child.read_frame() is None
        child.abort()  # no raise
        child.append(b"\x09" * vm.FRAME_SIZE)  # dropped
        assert child.read_frame() is None
        assert child.finished is True

    def test_append_after_finish_is_dropped(self):
        child = vm.StreamingSpeechChild()
        child.finish()
        child.append(b"\x01" * vm.FRAME_SIZE)
        assert child.read_frame() is None

    def test_buffer_backpressures_and_abort_releases_blocked_writer(self):
        child = vm.StreamingSpeechChild(
            fade_in_ms=0,
            max_buffer_bytes=vm.FRAME_SIZE,
        )
        frame = b"\x01\x00" * (vm.FRAME_SIZE // 2)
        child.append(frame)
        writer_done = threading.Event()

        def write_second_frame():
            child.append(frame)
            writer_done.set()

        writer = threading.Thread(target=write_second_frame)
        writer.start()
        assert writer_done.wait(0.05) is False
        with child._lock:
            assert len(child._buf) <= vm.FRAME_SIZE

        child.abort()
        assert writer_done.wait(1.0) is True
        writer.join(timeout=1.0)
        assert child.finished is True
        assert child.read_frame() is None

    def test_mixer_cleanup_aborts_child_and_releases_blocked_writer(self):
        child = vm.StreamingSpeechChild(max_buffer_bytes=vm.FRAME_SIZE)
        frame = b"\x01\x00" * (vm.FRAME_SIZE // 2)
        child.append(frame)
        writer_done = threading.Event()
        writer = threading.Thread(
            target=lambda: (child.append(frame), writer_done.set()),
            daemon=True,
        )
        writer.start()
        assert writer_done.wait(0.05) is False

        mixer = vm.VoiceMixer()
        mixer.play_speech_stream(child)
        mixer.cleanup()

        assert writer_done.wait(1.0) is True
        writer.join(timeout=1.0)
        assert child.finished is True

    def test_stop_speech_aborts_child_and_releases_blocked_writer(self):
        child = vm.StreamingSpeechChild(max_buffer_bytes=vm.FRAME_SIZE)
        frame = b"\x01\x00" * (vm.FRAME_SIZE // 2)
        child.append(frame)
        writer_done = threading.Event()
        writer = threading.Thread(
            target=lambda: (child.append(frame), writer_done.set()),
            daemon=True,
        )
        writer.start()
        assert writer_done.wait(0.05) is False

        mixer = vm.VoiceMixer()
        mixer.play_speech_stream(child)
        mixer.stop_speech()

        assert writer_done.wait(1.0) is True
        writer.join(timeout=1.0)
        assert child.finished is True
        assert mixer.speech_active is False


# =====================================================================
# Adapter sink
# =====================================================================

def _make_adapter(*, voice_fx=None, guild_id=111, chat_id="222", connected=True):
    """Real DiscordAdapter sink methods over a fake voice client."""
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import Platform, PlatformConfig

    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_sources = {}
    adapter._voice_mixers = {}
    adapter._streaming_tts_handles = {}
    adapter._ambient_pcm_cache = None
    adapter._voice_fx_cfg = voice_fx if voice_fx is not None else {
        "enabled": False, "ambient_enabled": True, "ambient_path": "",
        "ambient_gain": 0.18, "duck_gain": 0.06, "speech_gain": 1.0,
    }
    if connected:
        vc = MagicMock()
        vc.is_connected.return_value = True
        vc.is_playing.return_value = False
        vc.disconnect = AsyncMock()
        adapter._voice_clients[guild_id] = vc
        adapter._voice_text_channels[guild_id] = int(chat_id)
    return adapter


def _drain_mixer(mixer, max_frames=10000):
    """Read frames until no speech children remain; return concatenated PCM."""
    out = bytearray()
    for _ in range(max_frames):
        with mixer._lock:
            speech_live = bool(mixer._speech)
        if not speech_live:
            break
        out.extend(mixer.read())
    return bytes(out)


class TestSinkGating:
    def test_unsupported_without_voice_binding(self):
        from plugins.platforms.discord.adapter import DiscordAdapter
        adapter = _make_adapter(connected=False)
        assert adapter.supports_streaming_tts("222", AudioFormat()) is False
        # A bare object.__new__ adapter (no voice attrs at all) is safe too.
        bare = object.__new__(DiscordAdapter)
        assert bare.supports_streaming_tts("222", AudioFormat()) is False

    def test_supported_when_bound_and_connected(self):
        adapter = _make_adapter()
        assert adapter.supports_streaming_tts("222", AudioFormat()) is True

    def test_exact_format_gate_and_lazy_first_write_start(self):
        adapter = _make_adapter()
        unsupported = AudioFormat(sample_rate=48000, channels=1, sample_width=2)
        assert adapter.supports_streaming_tts("222", unsupported) is False

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            assert handle is not None
            assert handle.started is False
            assert adapter._voice_mixers == {}
            adapter._voice_clients[111].play.assert_not_called()
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            return handle

        handle = asyncio.run(run())
        assert handle.started is True
        adapter._voice_clients[111].play.assert_called_once()
        assert adapter._streaming_tts_handles[111] is handle

    def test_unsupported_for_other_chats(self):
        adapter = _make_adapter()
        assert adapter.supports_streaming_tts("999", AudioFormat()) is False


class TestBeginStreamingTTS:
    def test_no_binding_returns_none(self):
        adapter = _make_adapter(connected=False)

        async def run():
            return await adapter.begin_streaming_tts("222", AudioFormat())

        assert asyncio.run(run()) is None

    def test_installs_bare_mixer_when_voice_fx_disabled(self):
        adapter = _make_adapter()  # voice_fx enabled=False

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            return handle

        handle = asyncio.run(run())
        assert handle is not None
        vc = adapter._voice_clients[111]
        vc.play.assert_called_once()
        mixer = adapter._voice_mixers[111]
        assert mixer is vc.play.call_args[0][0]
        # voice_fx off → no ambient bed installed.
        assert mixer._ambient is None
        assert handle.guild_id == 111

    def test_existing_mixer_is_reused(self):
        adapter = _make_adapter()
        existing = vm.VoiceMixer()
        adapter._voice_mixers[111] = existing

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            return handle

        handle = asyncio.run(run())
        assert handle is not None
        adapter._voice_clients[111].play.assert_not_called()

    def test_voice_fx_enabled_keeps_ambient_bed(self):
        adapter = _make_adapter(voice_fx={
            "enabled": True, "ambient_enabled": True, "ambient_path": "",
            "ambient_gain": 0.18, "duck_gain": 0.06, "speech_gain": 1.0,
        })

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            return handle

        handle = asyncio.run(run())
        assert handle is not None
        assert adapter._voice_mixers[111]._ambient is not None

    def test_lead_silence_precedes_first_audio(self):
        adapter = _make_adapter(voice_fx={"enabled": False, "lead_silence_ms": 100})

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            return handle

        handle = asyncio.run(run())
        # The first write queues 100ms of lead silence before converted speech.
        lead_bytes = 100 * vm.BYTES_PER_MS
        assert len(handle.child._buf) > lead_bytes
        assert set(handle.child._buf[:lead_bytes]) == {0}

    def test_replacement_aborts_stale_owner_without_clearing_new_stream(self):
        adapter = _make_adapter()

        async def run():
            stale = await adapter.begin_streaming_tts("222", AudioFormat())
            current = await adapter.begin_streaming_tts("222", AudioFormat())
            assert stale.aborted is True
            assert adapter._streaming_tts_handles[111] is current

            await adapter.write_streaming_tts(stale, b"\x10\x00" * 480)
            adapter._voice_clients[111].play.assert_not_called()
            await adapter.write_streaming_tts(current, b"\x10\x00" * 480)
            adapter._voice_clients[111].play.assert_called_once()

            await adapter.finish_streaming_tts(stale)
            assert adapter._streaming_tts_handles[111] is current
            await adapter.abort_streaming_tts(current, error="barge-in")
            assert 111 not in adapter._streaming_tts_handles

        asyncio.run(run())

    def test_first_write_never_preempts_racing_unrelated_audio(self):
        adapter = _make_adapter()
        vc = adapter._voice_clients[111]
        vc.is_playing.side_effect = [False, True]

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            with pytest.raises(RuntimeError, match="mixer"):
                await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            assert handle.audible is False

        asyncio.run(run())
        vc.stop.assert_not_called()

    def test_drain_cleanup_is_identity_safe_and_leave_aborts_owner(self):
        adapter = _make_adapter()
        adapter._cancel_voice_timeout = MagicMock()
        adapter._reset_voice_timeout = MagicMock()

        async def run():
            stale = await adapter.begin_streaming_tts("222", AudioFormat())
            await adapter.write_streaming_tts(stale, b"\x10\x00" * 480)
            stale_finish = asyncio.create_task(adapter.finish_streaming_tts(stale))
            await asyncio.sleep(0.05)
            current = await adapter.begin_streaming_tts("222", AudioFormat())
            await asyncio.wait_for(stale_finish, timeout=1.0)
            assert adapter._streaming_tts_handles[111] is current
            await adapter.write_streaming_tts(current, b"\x20\x00" * 480)

            await adapter.leave_voice_channel(111)
            assert current.aborted is True
            assert 111 not in adapter._streaming_tts_handles

        asyncio.run(run())


class TestWriteFinishAbort:
    def _begin(self, adapter):
        async def run():
            return await adapter.begin_streaming_tts("222", AudioFormat())

        return asyncio.run(run())

    def test_write_resamples_to_mixer_format(self):
        adapter = _make_adapter()
        handle = self._begin(adapter)
        # 20ms of 24 kHz mono s16 = 480 samples = 960 bytes.
        chunk = b"\x10\x00" * 480

        async def run():
            await adapter.write_streaming_tts(handle, chunk)

        asyncio.run(run())
        # Incremental interpolation retains one source frame until finish.
        assert abs(len(handle.child._buf) - 3832) <= 4

    def test_finish_flushes_resampler_endpoint_after_arbitrary_chunks(self):
        adapter = _make_adapter(voice_fx={"enabled": False, "lead_silence_ms": 0})
        handle = self._begin(adapter)
        handle.child.fade_frames = 0
        source = np.array([0, 1000, 2000, 3000], dtype=np.int16).tobytes()
        expected = vm.resample_to_mixer_pcm(source, 24000, 1)

        async def run():
            for byte in source:
                await adapter.write_streaming_tts(handle, bytes([byte]))
            finish_task = asyncio.create_task(adapter.finish_streaming_tts(handle))
            await asyncio.sleep(0.05)
            drained = _drain_mixer(adapter._voice_mixers[111])
            await asyncio.wait_for(finish_task, timeout=1.0)
            return drained

        drained = asyncio.run(run())
        assert drained[:len(expected)] == expected

    def test_write_after_abort_is_dropped(self):
        adapter = _make_adapter()
        handle = self._begin(adapter)

        async def run():
            await adapter.abort_streaming_tts(handle, error="stop")
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            await adapter.abort_streaming_tts(handle, error="again")  # idempotent

        asyncio.run(run())
        assert handle.aborted is True
        assert len(handle.child._buf) == 0

    def test_finish_marks_eof_and_drains(self):
        adapter = _make_adapter()
        handle = self._begin(adapter)

        async def run():
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            finish_task = asyncio.create_task(adapter.finish_streaming_tts(handle))
            await asyncio.sleep(0.05)
            assert handle.child.finished is False
            pcm = _drain_mixer(adapter._voice_mixers[111])
            await asyncio.wait_for(finish_task, timeout=1.0)
            return pcm

        pcm = asyncio.run(run())
        assert len(pcm) > 0
        assert handle.child.finished is True

    def test_finish_waits_for_mixer_drain_and_releases_ownership(self):
        adapter = _make_adapter()
        adapter._reset_voice_timeout = MagicMock()

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            await adapter.write_streaming_tts(handle, b"\x01\x00" * 480)
            finish_task = asyncio.create_task(adapter.finish_streaming_tts(handle))
            await asyncio.sleep(0.05)
            assert handle.child.finished is False
            assert finish_task.done() is False
            _drain_mixer(adapter._voice_mixers[111])
            await asyncio.wait_for(finish_task, timeout=1.0)
            assert handle.child.finished is True
            assert 111 not in adapter._streaming_tts_handles

        asyncio.run(run())
        adapter._reset_voice_timeout.assert_called_once_with(111)

    def test_stuck_finish_drain_aborts_and_releases_ownership(self):
        adapter = _make_adapter()
        adapter._streaming_tts_drain_timeout = 0.02
        adapter._reset_voice_timeout = MagicMock()

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            assert handle is not None
            await adapter.write_streaming_tts(handle, b"\x01\x00" * 480)
            await asyncio.wait_for(adapter.finish_streaming_tts(handle), timeout=0.2)
            assert handle.aborted is True
            assert handle.child.finished is True
            assert 111 not in adapter._streaming_tts_handles

        asyncio.run(run())
        adapter._reset_voice_timeout.assert_called_once_with(111)

    def test_receiver_pauses_on_first_audio_and_resumes_after_drain(self):
        adapter = _make_adapter()
        receiver = MagicMock()
        adapter._voice_receivers[111] = receiver
        adapter._reset_voice_timeout = MagicMock()

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            receiver.pause.assert_not_called()
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            receiver.pause.assert_called_once_with()
            receiver.resume.assert_not_called()

            finish_task = asyncio.create_task(adapter.finish_streaming_tts(handle))
            await asyncio.sleep(0.05)
            _drain_mixer(adapter._voice_mixers[111])
            await asyncio.wait_for(finish_task, timeout=1.0)
            receiver.resume.assert_called_once_with()

        asyncio.run(run())

    def test_interrupted_finish_discards_tail(self):
        adapter = _make_adapter()
        handle = self._begin(adapter)

        async def run():
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            await adapter.finish_streaming_tts(handle, interrupted=True)

        asyncio.run(run())
        assert handle.child.read_frame() is None

    def test_session_interrupt_aborts_audible_stream_after_turn_returns(self):
        adapter = _make_adapter()
        adapter._active_sessions = {}
        adapter._stop_typing_quietly = AsyncMock()

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            await adapter.interrupt_session_activity("discord:voice:222", "222")
            assert handle.aborted is True
            assert handle.child.finished is True
            assert 111 not in adapter._streaming_tts_handles

        asyncio.run(run())

    def test_gateway_stop_aborts_stream_after_worker_already_exited(self):
        adapter = _make_adapter()
        adapter._active_sessions = {}
        adapter._pending_messages = {}
        adapter._stop_typing_quietly = AsyncMock()
        runner = object.__new__(gateway_run.GatewayRunner)
        setattr(runner, "_peek_session_state", lambda _key: None)
        setattr(runner, "_invalidate_session_run_generation", lambda _key, reason=None: 2)
        setattr(runner, "_adapter_for_source", lambda _source: adapter)
        setattr(runner, "_thread_metadata_for_source", lambda _source: None)
        setattr(runner, "_release_running_agent_state", lambda _key: None)
        setattr(runner, "_evict_cached_agent", lambda _key: None)
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="222",
            chat_type="channel",
            user_id="user-1",
        )

        async def run():
            handle = await adapter.begin_streaming_tts("222", AudioFormat())
            assert handle is not None
            await adapter.write_streaming_tts(handle, b"\x10\x00" * 480)
            await runner._interrupt_and_clear_session(
                "agent:main:discord:channel:222",
                source,
                interrupt_reason=gateway_run._INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command",
            )
            assert handle.aborted is True
            assert getattr(handle, "child").finished is True
            assert 111 not in adapter._streaming_tts_handles

        asyncio.run(run())


# =====================================================================
# End-to-end: gateway consumer → real Discord sink
# =====================================================================

class _FakeStreamer:
    """Streaming provider fake: two 24 kHz mono PCM chunks per clause."""

    sample_rate = 24000
    channels = 1
    sample_width = 2

    def stream(self, text):
        yield b"\x01\x00" * 480   # 20ms
        yield b"\x02\x00" * 480   # 20ms


def _make_consumer(adapter, chat_id, loop, streamer):
    """StreamingTTSConsumer with pre-set internals (mirrors the #60671 suite)."""
    consumer = StreamingTTSConsumer.__new__(StreamingTTSConsumer)
    consumer._adapter = adapter
    consumer._chat_id = chat_id
    consumer._tts_config = {}
    consumer._loop = loop
    consumer._metadata = None
    consumer._audio_format = AudioFormat()
    consumer._streamer = streamer
    consumer._chunker = SentenceChunker()
    consumer._queue = queue.Queue(maxsize=256)
    consumer._handle = None
    consumer._started = False
    consumer._completed = False
    consumer._partial = False
    consumer._aborted = False
    consumer._finished = False
    consumer._dropped = False
    consumer._suppress_whole_file = False
    consumer._task = None
    consumer._lock = threading.Lock()
    consumer._strip_markdown = None
    return consumer


class TestConsumerToDiscordSink:
    def test_first_sentence_plays_before_full_response(self):
        """The #94462 regression contract: audio for sentence one reaches the
        mixer while the LLM is still generating — the gateway never waits for
        the complete reply before TTS starts."""
        adapter = _make_adapter()

        async def run(loop):
            consumer = _make_consumer(adapter, "222", loop, _FakeStreamer())
            consumer.start()

            first_audio_at = None
            finished_at = None

            def feed():
                nonlocal first_audio_at, finished_at
                consumer.on_delta("The first sentence is ready now. ")
                # Wait until the first clause's audio has been synthesised AND
                # written to the mixer — before finish() is ever called.
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    handle = consumer._handle
                    if handle is not None and len(handle.child._buf) > 0:
                        first_audio_at = time.monotonic()
                        break
                    time.sleep(0.01)
                # The LLM is still generating: more text arrives later.
                time.sleep(0.05)
                consumer.on_delta("And a second sentence follows it. ")
                consumer.finish()
                finished_at = time.monotonic()

            await asyncio.to_thread(feed)
            completion = asyncio.create_task(consumer.wait_complete(timeout=5.0))
            played = bytearray()
            while not completion.done():
                mixer = adapter._voice_mixers.get(111)
                if mixer is not None:
                    with mixer._lock:
                        speech_live = bool(mixer._speech)
                    if speech_live:
                        played.extend(mixer.read())
                await asyncio.sleep(0.001)
            completed = await completion

            assert completed is True
            assert first_audio_at is not None
            assert finished_at is not None
            assert first_audio_at < finished_at  # audio before end-of-text
            assert consumer.suppress_whole_file is True  # no double playback

            # Everything the provider sent landed in the mixer as 48 kHz
            # stereo s16 PCM: 2 clauses × 2 chunks × 480 samples, upsampled
            # 2x and duplicated to stereo = 15360 bytes of non-silence.
            handle = consumer._handle
            drained = bytes(played) + _drain_mixer(adapter._voice_mixers[111])
            assert len(drained) >= 15360
            assert max(drained) > 0  # real signal, not just silence frames
            assert handle.child.finished is True

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(run(loop), timeout=10.0))
        finally:
            loop.close()

    def test_abort_mid_stream_stops_playback(self):
        adapter = _make_adapter()

        class _SlowStreamer(_FakeStreamer):
            def stream(self, text):
                for _ in range(50):
                    time.sleep(0.02)
                    yield b"\x03\x00" * 480

        async def run(loop):
            consumer = _make_consumer(adapter, "222", loop, _SlowStreamer())
            consumer.start()
            consumer.on_delta("A long reply that will be interrupted. ")
            # Let some audio land, then abort (user barge-in / /stop).
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                handle = consumer._handle
                if handle is not None and len(handle.child._buf) > 0:
                    break
                await asyncio.sleep(0.01)
            consumer.abort("barge-in")
            await consumer.wait_complete(timeout=5.0)
            assert consumer.completed is False
            assert consumer._handle.child.finished is True  # aborted

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(run(loop), timeout=10.0))
        finally:
            loop.close()

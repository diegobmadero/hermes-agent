"""Gateway regression: the outer finalisation path must not NameError on
the streaming-TTS consumer (#60671).

The original commit created ``_streaming_tts_consumer`` as a local inside
``run_sync`` (executor-thread) but the outer ``_run_agent_inner`` code
referenced it during interrupt/finalisation — a cross-scope ``NameError``
on ordinary gateway turns.  The hardening moves the consumer into a
``streaming_tts_consumer_holder`` created on the event-loop thread.

This test exercises the real ``_run_agent`` → ``_run_agent_inner`` path
with a voice input message type so the streaming-TTS consumer setup
branch is entered.  The fake agent returns synchronously, the executor
finishes, and the outer finalisation code runs — proving no NameError.
"""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class _NoopAgent:
    """Minimal agent stub that returns immediately without tool calls."""

    def __init__(self, *args, **kwargs):
        self.tools = []
        self.model = kwargs.get("model", "test-model")
        self.provider = kwargs.get("provider", "test-provider")
        self.session_id = kwargs.get("session_id", "session-1")
        self.context_compressor = None
        self.is_interrupted = False

    def run_conversation(self, user_message, conversation_history=None,
                         task_id=None, persist_user_message=None,
                         persist_user_timestamp=None):
        return {
            "final_response": "Hello from the agent.",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _NoopAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None, multiplex_profiles=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._gateway_loop = None
    return runner


def _make_voice_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _setup_monkeypatches(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    (tmp_path / "config.yaml").write_text("agent:\n  model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"model": "test-model"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "test-model")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})


def test_run_agent_voice_turn_no_name_error(monkeypatch, tmp_path):
    """A voice-input turn must complete without a _streaming_tts_consumer NameError.

    The streaming-TTS consumer setup is entered (voice input + auto-TTS),
    but since no adapter is configured the consumer setup is skipped.  The
    outer finalisation path still runs and must not raise NameError when
    reading ``streaming_tts_consumer_holder``.
    """
    _setup_monkeypatches(monkeypatch, tmp_path)
    runner = _make_runner()

    # No adapter for this source → consumer setup skipped, but the outer
    # finalisation path still runs with streaming_tts_consumer_holder[0]=None.
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_adapter_for_source",
        lambda self, source: None,
    )

    async def _run():
        result = await runner._run_agent(
            message="Hello Jarvis",
            context_prompt="",
            history=[],
            source=_make_voice_source(),
            session_id="session-1",
            session_key="agent:main:telegram:dm:12345",
            message_type=MessageType.VOICE,
        )
        return result

    result = asyncio.new_event_loop().run_until_complete(_run())
    assert result["final_response"] == "Hello from the agent."


class _FinalizationConsumer:
    def __init__(self, *, audible: bool, task=None):
        self.suppress_whole_file = audible
        self.done = False
        self._task = task
        self.finish_count = 0
        self.wait_count = 0
        self.abort_reasons = []

    def finish(self):
        self.finish_count += 1

    async def wait_complete(self, timeout, idle_timeout=None):
        self.wait_count += 1
        if self._task is None:
            return self.done
        # Model the real consumer: an idle timeout caps how long a still-running task is waited on.
        wait_s = timeout if idle_timeout is None else min(timeout, idle_timeout)
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=wait_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        self.done = True
        return True

    def abort(self, reason):
        self.abort_reasons.append(reason)
        self.done = True
        if self._task is not None:
            self._task.cancel()


def test_audible_finalization_is_owned_without_turn_cleanup_abort():
    async def run():
        runner = object.__new__(gateway_run.GatewayRunner)
        runner._background_tasks = set()
        runner._draining = False
        setattr(runner, "_streaming_tts_audible_timeout", 0.1)
        deferred = asyncio.create_task(asyncio.Event().wait())
        consumer = _FinalizationConsumer(audible=True, task=deferred)
        turn_ctx = SimpleNamespace(
            streaming_tts_consumer_holder=[consumer],
            stream_consumer_holder=[None],
            session_key=None,
            run_generation=None,
        )
        adapter = SimpleNamespace(_mark_streaming_tts_completed_turn=MagicMock())

        await runner._run_agent_finalize_streaming_tts(turn_ctx, adapter)
        assert consumer.finish_count == 1
        assert consumer.wait_count == 0
        assert consumer.abort_reasons == []
        supervisor = getattr(consumer, "_hermes_audible_drain_supervisor")
        assert supervisor in runner._background_tasks
        assert supervisor is not deferred

        tracking = asyncio.create_task(asyncio.Event().wait())
        await runner._run_agent_cleanup_turn_tasks(
            turn_ctx,
            progress_task=None,
            log_task=None,
            interrupt_monitor=None,
            _notify_task=None,
            tracking_task=tracking,
            stream_task=None,
        )
        assert consumer.abort_reasons == []
        assert supervisor in runner._background_tasks

        supervisor.cancel()
        deferred.cancel()
        await asyncio.gather(supervisor, deferred, return_exceptions=True)

    asyncio.run(run())


def test_silent_finalization_retains_bounded_abort_for_fallback():
    async def run():
        runner = object.__new__(gateway_run.GatewayRunner)
        runner._background_tasks = set()
        consumer = _FinalizationConsumer(audible=False)
        turn_ctx = SimpleNamespace(
            streaming_tts_consumer_holder=[consumer],
            session_key="discord:voice:1",
            run_generation=7,
        )

        await runner._run_agent_finalize_streaming_tts(turn_ctx, adapter=None)

        assert consumer.finish_count == 1
        assert consumer.wait_count == 2
        assert consumer.abort_reasons == ["streaming TTS finalisation timeout"]
        assert runner._background_tasks == set()

    asyncio.run(run())


def test_audible_drain_counts_as_active_restart_work():
    async def run():
        runner = object.__new__(gateway_run.GatewayRunner)
        runner._background_tasks = set()
        runner._running_agents = {}
        runner._deferred_agent_workers = {}
        runner.adapters = {}
        runner._running_cron_job_count = lambda: 0
        setattr(runner, "_streaming_tts_audible_timeout", 0.1)
        drain = asyncio.create_task(asyncio.Event().wait())
        consumer = _FinalizationConsumer(audible=True, task=drain)

        runner._retain_audible_streaming_tts(consumer)

        supervisor = getattr(consumer, "_hermes_audible_drain_supervisor")
        assert getattr(supervisor, "_hermes_streaming_tts_drain", False) is True
        assert runner._active_work_count() == 1
        supervisor.cancel()
        drain.cancel()
        await asyncio.gather(supervisor, drain, return_exceptions=True)

    asyncio.run(run())


def test_audible_provider_stall_is_bounded_and_aborted():
    async def run():
        runner = object.__new__(gateway_run.GatewayRunner)
        runner._background_tasks = set()
        setattr(runner, "_streaming_tts_audible_timeout", 0.01)
        stalled = asyncio.create_task(asyncio.Event().wait())
        consumer = _FinalizationConsumer(audible=True, task=stalled)

        runner._retain_audible_streaming_tts(consumer)
        supervisor = getattr(consumer, "_hermes_audible_drain_supervisor")
        # Idle clamp is a 1s floor; the supervisor must abort well before any lifetime backstop.
        await asyncio.wait_for(supervisor, timeout=3.0)

        assert consumer.abort_reasons == ["audible streaming TTS idle timeout (no new audio for 1s)"]
        assert stalled.cancelled() is True

    asyncio.run(run())



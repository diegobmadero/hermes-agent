"""Final-pickup admission for plugin-injected queued turns (pre_plugin_injection_admit)."""
import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_plugin_admission import is_plugin_injection, plugin_injection_admitted
from gateway.session import SessionSource

KEY = "agent:main:telegram:dm:42"


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42", user_name="t")


def _plugin_event(text="BAF_RUNTIME_BATCH: " + "a" * 32 + "\nwake", plugin="baf-runtime"):
    return MessageEvent(text=text, source=_source(), internal=True, allow_gateway_control=False,
                        metadata={"hermes_plugin_id": plugin, "hermes_plugin_injection": True,
                                  "gateway_session_key": KEY, "gateway_session_id": "s-1",
                                  "gateway_session_strict": True})


@pytest.fixture
def hook(monkeypatch):
    """Install a fake plugin hook and record its invocations."""
    calls, verdict = [], {"value": None}

    async def ainvoke(name, **kwargs):
        assert name == "pre_plugin_injection_admit"
        calls.append(kwargs)
        return [verdict["value"]] if verdict["value"] is not None else []
    import hermes_cli.lifecycle as lifecycle
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_plugin_injection_admit")
    monkeypatch.setattr(lifecycle, "ainvoke_hook", ainvoke)
    return SimpleNamespace(calls=calls, verdict=verdict)


def test_hook_is_registered_name():
    from hermes_cli.plugins import VALID_HOOKS
    assert "pre_plugin_injection_admit" in VALID_HOOKS


def test_only_gateway_stamped_plugin_events_qualify():
    assert is_plugin_injection(_plugin_event())
    # Operator text that mimics a BAF marker is ordinary input, never a plugin injection.
    assert not is_plugin_injection(MessageEvent(text="BAF_RUNTIME_BATCH: " + "a" * 32, source=_source()))
    spoof = MessageEvent(text="x", source=_source(), metadata={"hermes_plugin_injection": True,
                                                              "hermes_plugin_id": "baf-runtime"})
    assert not is_plugin_injection(spoof)
    other_internal = MessageEvent(text="x", source=_source(), internal=True, metadata={})
    assert not is_plugin_injection(other_internal)


@pytest.mark.asyncio
async def test_reject_is_definite_and_allow_is_default(hook):
    event = _plugin_event()
    assert await plugin_injection_admitted(event) is True
    hook.verdict["value"] = {"action": "reject", "reason": "policy_cancelled"}
    assert await plugin_injection_admitted(event) is False
    assert hook.calls[-1]["plugin_id"] == "baf-runtime"
    assert hook.calls[-1]["session_key"] == KEY and hook.calls[-1]["session_id"] == "s-1"


@pytest.mark.asyncio
async def test_user_text_never_reaches_the_hook(hook):
    hook.verdict["value"] = {"action": "reject"}
    assert await plugin_injection_admitted(MessageEvent(text="BAF_RUNTIME_BATCH: x", source=_source())) is True
    assert hook.calls == []


@pytest.mark.asyncio
async def test_hook_failure_admits(monkeypatch):
    import hermes_cli.lifecycle as lifecycle

    async def boom(name, **kwargs):
        raise RuntimeError("plugin crashed")
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
    monkeypatch.setattr(lifecycle, "ainvoke_hook", boom)
    assert await plugin_injection_admitted(_plugin_event()) is True


def _cold_runner(event):
    runner = object.__new__(GatewayRunner)
    runner.config = None
    lease = MagicMock()
    runner._hm_estop_gate = lambda *a: None
    runner._session_key_for_source = lambda source: KEY
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    runner._hm_evict_idle_stale_agent = lambda key: None
    runner._is_session_running = lambda key: False
    runner._hm_dispatch_idle_commands = AsyncMock(return_value=(False, None))
    runner._claim_active_session_slot = lambda key, source: (lease, None)
    runner._hm_rescue_orphaned_fifo = lambda ev, src, internal, key: (ev, src, internal)
    runner._handle_message_with_agent = AsyncMock(return_value={"final_response": "ran"})
    runner._session_state = MagicMock()
    return runner, lease


@pytest.mark.asyncio
async def test_cold_start_rejection_starts_no_agent(hook):
    event = _plugin_event()
    runner, lease = _cold_runner(event)
    hook.verdict["value"] = {"action": "reject", "reason": "policy_cancelled"}
    assert await runner._handle_message(event) is None
    runner._handle_message_with_agent.assert_not_called()
    lease.release.assert_called_once()
    runner._session_state.assert_not_called()  # no sentinel/turn state was claimed


class _Adapter:
    def __init__(self, events):
        self.events = list(events)
        self._pending_messages = {}

    def get_pending_message(self, key):
        return self.events.pop(0) if self.events else None


def _drain_runner():
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    runner._promote_queued_event = lambda key, adapter, ev: ev
    runner._pending_event_audio_paths = lambda ev: []
    return runner


@pytest.mark.asyncio
async def test_drain_skips_rejected_wake_and_continues_chain(hook):
    stale, human = _plugin_event(), MessageEvent(text="human follow-up", source=_source())
    adapter = _Adapter([stale, human])
    hook.verdict["value"] = {"action": "reject", "reason": "policy_cancelled"}
    pending_event, pending = await _drain_runner()._run_agent_drain_pending(
        {"final_response": "done"}, adapter, _source(), KEY)
    assert pending_event is human and pending == "human follow-up"
    assert len(hook.calls) == 1


@pytest.mark.asyncio
async def test_drain_rejecting_only_wake_starts_nothing(hook):
    adapter = _Adapter([_plugin_event()])
    hook.verdict["value"] = {"action": "reject"}
    assert await _drain_runner()._run_agent_drain_pending(
        {"final_response": "done"}, adapter, _source(), KEY) == (None, None)


@pytest.mark.asyncio
async def test_drain_admits_authorized_wake(hook):
    wake = _plugin_event()
    pending_event, pending = await _drain_runner()._run_agent_drain_pending(
        {"final_response": "done"}, _Adapter([wake]), _source(), KEY)
    assert pending_event is wake and pending.startswith("BAF_RUNTIME_BATCH: ")

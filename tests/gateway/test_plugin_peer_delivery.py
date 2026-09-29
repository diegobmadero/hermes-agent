"""Peer injection is explicitly gated and bound to the live gateway turn."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def context():
    return PluginContext(PluginManifest(name="peer-test", key="peer-test", source="user"), PluginManager())


def test_invalid_delivery_mode_is_rejected():
    ctx = context()
    with pytest.raises(ValueError, match="delivery"):
        ctx.inject_message("notice", session_key="route", delivery="interrupt")


@pytest.mark.asyncio
async def test_busy_peer_uses_bound_agent_without_a_gateway_wake():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    entry = SessionEntry(session_key="agent:main:telegram:dm:42", session_id="session-42",
                         created_at=datetime.now(), updated_at=datetime.now(),
                         origin=source, platform=Platform.TELEGRAM)
    agent = SimpleNamespace(session_id=entry.session_id, _inflight_turn_id="turn-1",
                            queue_peer_notification=MagicMock(return_value=True))
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner._running, runner._draining = True, False
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry))
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: adapter}, {}
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=agent))
    runner._current_session_run_generation = lambda key: 4
    runner._is_session_run_current = lambda key, generation: generation == 4
    events = []

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="id-1", on_delivery=events.append,
    )

    assert accepted is True
    agent.queue_peer_notification.assert_called_once()
    adapter.handle_message.assert_not_awaited()
    assert events[0]["effective"] == "peer"
    assert events[0]["delivery_id"] == "id-1"

    # A same-process plugin reload can submit the same ID again. The gateway owns
    # the ledger, so the replacement context cannot insert or wake a second time.
    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="id-1", on_delivery=events.append,
    )
    agent.queue_peer_notification.assert_called_once()
    adapter.handle_message.assert_not_awaited()

    bound = agent.queue_peer_notification.call_args.kwargs
    runner._schedule_plugin_message_injection = MagicMock(return_value=True)
    runner._is_session_run_current = lambda key, generation: False  # /stop or /new fence
    assert bound["valid"]() is False
    bound["on_fallback"]("turn_changed")
    assert runner._schedule_plugin_message_injection.call_args.kwargs["bound_session_id"] == "session-42"
    assert events[-1]["effective"] == "queue"
    assert events[-1]["reason"] == "turn_changed"

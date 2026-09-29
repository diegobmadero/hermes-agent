"""Peer injection is explicitly gated and bound to the live gateway turn."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import hermes_yaml as yaml

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


def test_permission_off_is_explicit_queue_fallback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {
        "peer-test": {"allow_gateway_injection": True}
    }}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    ctx = context()
    sent = []

    def injector(*, session_key, content, plugin_id, on_result=None,
                 delivery="queue", delivery_id=None, on_delivery=None):
        sent.append((delivery, content))
        if on_result:
            on_result(True)
        return True

    ctx._manager.set_gateway_message_injector(object(), injector)
    events = []
    assert ctx.inject_message("peer answer", session_key="route", delivery="peer",
                              delivery_id="id-1", on_delivery=events.append)
    assert sent == [("queue", "peer answer")]
    assert events == [{"event": "routed", "delivery_id": "id-1", "session_key": "route",
                       "effective": "queue", "reason": "permission_off"}]


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
    runner._schedule_plugin_message_injection = MagicMock(
        side_effect=lambda **kwargs: kwargs["on_result"](True) or True)
    runner._is_session_run_current = lambda key, generation: False  # /stop or /new fence
    assert bound["valid"]() is False
    bound["on_fallback"]("turn_changed")
    assert runner._schedule_plugin_message_injection.call_args.kwargs["bound_session_id"] == "session-42"
    assert events[-1]["effective"] == "queue"
    assert events[-1]["reason"] == "turn_changed"
    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="id-1", on_delivery=events.append)
    assert events[-1]["effective"] == "queue"
    runner._schedule_plugin_message_injection.assert_called_once()


@pytest.mark.asyncio
async def test_idle_peer_uses_one_ordinary_turn_and_old_session_cannot_follow_new():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    entry = SessionEntry(session_key="agent:main:telegram:dm:42", session_id="session-42",
                         created_at=datetime.now(), updated_at=datetime.now(),
                         origin=source, platform=Platform.TELEGRAM)

    async def accept(event):
        event._gateway_accepted = True

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=accept))
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    lookup = AsyncMock(return_value=entry)
    runner._async_session_store = SimpleNamespace(_store=runner.session_store,
                                                  lookup_by_session_key=lookup)
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: adapter}, {}
    runner._running, runner._draining = True, False
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=None))
    runner._current_session_run_generation = lambda key: 0
    events = []
    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="id-2", on_delivery=events.append)
    adapter.handle_message.assert_awaited_once()
    assert events[0]["effective"] == "queue"
    assert events[0]["reason"] == "no_live_peer_turn"
    old_event = adapter.handle_message.await_args.args[0]
    assert old_event.metadata["gateway_session_id"] == "session-42"

    new_entry = SessionEntry(session_key=entry.session_key, session_id="new-session",
                             created_at=datetime.now(), updated_at=datetime.now(),
                             origin=source, platform=Platform.TELEGRAM)
    lookup.return_value = new_entry
    assert not await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="old peer answer", plugin_id="peer-test",
        bound_session_id="session-42")
    adapter.handle_message.assert_awaited_once()

    # The thread-safe scheduler snapshots the old session before its task gets
    # a chance to run. A replacement route cannot acquire the pending notice.
    runner.session_store.peek_session_id = lambda key: "session-42"
    runner._gateway_loop = asyncio.get_running_loop()
    runner._background_tasks = set()
    runner._is_session_run_current = lambda key, generation: True
    outcomes = []
    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="old peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="id-3", on_result=outcomes.append)
    await asyncio.gather(*list(runner._background_tasks))
    assert outcomes == [False]
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_outcome", [False, None])
async def test_removed_bound_route_retains_peer_without_claiming_queue(fallback_outcome):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    key = "agent:main:telegram:dm:42"
    def entry(session_id):
        return SessionEntry(session_key=key, session_id=session_id,
                            created_at=datetime.now(), updated_at=datetime.now(),
                            origin=source, platform=Platform.TELEGRAM)

    lookup = AsyncMock(return_value=entry("old"))
    agent = SimpleNamespace(session_id="old", _inflight_turn_id="turn-1",
                            queue_peer_notification=MagicMock(return_value=True))
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = object.__new__(GatewayRunner)
    runner._running, runner._draining = True, False
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=lookup)
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: adapter}, {}
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._session_state = lambda route: SimpleNamespace(turn=SimpleNamespace(agent=agent))
    runner._current_session_run_generation = lambda route: 4
    runner._is_session_run_current = lambda route, generation: generation == 4
    events = []
    assert await runner._dispatch_plugin_message_injection(
        session_key=key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="delivery-1", on_delivery=events.append)
    lookup.return_value = entry("new")
    scheduled = []
    def schedule(**kwargs):
        scheduled.append(kwargs)
        if fallback_outcome is None:
            kwargs["on_result"](None)
            return True
        return False
    runner._schedule_plugin_message_injection = schedule
    agent.queue_peer_notification.call_args.kwargs["on_fallback"]("turn_changed")
    assert len(scheduled) == 1
    assert scheduled[0]["bound_session_id"] == "old"
    assert not any(event["effective"] == "queue" for event in events)
    assert runner._peer_delivery_ledger[(key, "old", "peer-test", "delivery-1")] != "queue"
    assert await runner._dispatch_plugin_message_injection(
        session_key=key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="delivery-1", bound_session_id="old") is False
    adapter.handle_message.assert_not_awaited()
    lookup.return_value = entry("old")
    assert await runner._dispatch_plugin_message_injection(
        session_key=key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="delivery-1")
    assert len(scheduled) == 1
    agent.queue_peer_notification.assert_called_once()


@pytest.mark.asyncio
async def test_finalizer_fallback_before_append_returns_cannot_be_overwritten_by_peer():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    key = "agent:main:telegram:dm:42"
    entry = SessionEntry(session_key=key, session_id="old", created_at=datetime.now(),
                         updated_at=datetime.now(), origin=source, platform=Platform.TELEGRAM)
    def append_then_finalize(*args, **kwargs):
        kwargs["on_fallback"]("turn_ended")
        return True
    agent = SimpleNamespace(session_id="old", _inflight_turn_id="turn-1",
                            queue_peer_notification=append_then_finalize)
    runner = object.__new__(GatewayRunner)
    runner._running, runner._draining = True, False
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry))
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: SimpleNamespace(handle_message=AsyncMock())}, {}
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._session_state = lambda route: SimpleNamespace(turn=SimpleNamespace(agent=agent))
    runner._current_session_run_generation = lambda route: 4
    runner._is_session_run_current = lambda route, generation: generation == 4
    scheduled = []
    def schedule(**kwargs):
        scheduled.append(kwargs)
        kwargs["on_result"](True)
        return True
    runner._schedule_plugin_message_injection = schedule
    events = []
    assert await runner._dispatch_plugin_message_injection(
        session_key=key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="delivery-1", on_delivery=events.append)
    assert len(scheduled) == 1
    assert events[-1]["effective"] == "queue"
    assert runner._peer_delivery_ledger[(key, "old", "peer-test", "delivery-1")] == "queue"


@pytest.mark.asyncio
async def test_idle_peer_duplicate_waits_for_single_adapter_admission_and_refusal_can_retry():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    key = "agent:main:telegram:dm:42"
    entry = SessionEntry(session_key=key, session_id="session-42", created_at=datetime.now(),
                         updated_at=datetime.now(), origin=source, platform=Platform.TELEGRAM)
    runner = object.__new__(GatewayRunner)
    runner._running, runner._draining = True, False
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry))
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._session_state = lambda route: SimpleNamespace(turn=SimpleNamespace(agent=None))
    runner._current_session_run_generation = lambda route: 0
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def accept(event):
        calls.append(event)
        entered.set()
        await release.wait()
        event._gateway_accepted = True

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=accept))
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: adapter}, {}
    receipts = []
    kwargs = dict(session_key=key, content="peer answer", plugin_id="peer-test",
                  delivery="peer", delivery_id="delivery-1", on_delivery=receipts.append)
    first = asyncio.create_task(runner._dispatch_plugin_message_injection(**kwargs))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(runner._dispatch_plugin_message_injection(**kwargs))
        assert await asyncio.wait_for(second, 2)
        assert len(calls) == 1
        assert receipts == []
    finally:
        release.set()
        assert await first
        if second is not None:
            assert await second
    assert runner._peer_delivery_ledger[(key, "session-42", "peer-test", "delivery-1")] == "queue"
    assert [event["effective"] for event in receipts] == ["queue"]

    # A definite unstarted refusal releases this ID for the receiver's bounded retry.
    attempts = []
    async def refuse_then_accept(event):
        attempts.append(event)
        if len(attempts) == 1:
            return
        event._gateway_accepted = True
    adapter.handle_message.side_effect = refuse_then_accept
    retry = {**kwargs, "delivery_id": "delivery-2"}
    assert not await runner._dispatch_plugin_message_injection(**retry)
    assert await runner._dispatch_plugin_message_injection(**retry)
    assert len(attempts) == 2
    assert runner._peer_delivery_ledger[(key, "session-42", "peer-test", "delivery-2")] == "queue"

    async def unknown(_event):
        raise RuntimeError("adapter outcome unknown")
    adapter.handle_message.side_effect = unknown
    uncertain = {**kwargs, "delivery_id": "delivery-3"}
    with pytest.raises(RuntimeError, match="adapter outcome unknown"):
        await runner._dispatch_plugin_message_injection(**uncertain)
    assert runner._peer_delivery_ledger[(key, "session-42", "peer-test", "delivery-3")] == "retained"
    count = adapter.handle_message.await_count
    assert await runner._dispatch_plugin_message_injection(**uncertain)
    assert adapter.handle_message.await_count == count


@pytest.mark.asyncio
async def test_many_peer_delivery_ids_reuse_bounded_route_synchronization():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    key = "agent:main:telegram:dm:42"
    entry = SessionEntry(session_key=key, session_id="session-42", created_at=datetime.now(),
                         updated_at=datetime.now(), origin=source, platform=Platform.TELEGRAM)
    agent = SimpleNamespace(session_id="session-42", _inflight_turn_id="turn-1",
                            queue_peer_notification=MagicMock(return_value=True))
    runner = object.__new__(GatewayRunner)
    runner._running, runner._draining = True, False
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry))
    runner.adapters, runner._profile_adapters = {Platform.TELEGRAM: SimpleNamespace(handle_message=AsyncMock())}, {}
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._session_state = lambda route: SimpleNamespace(turn=SimpleNamespace(agent=agent))
    runner._current_session_run_generation = lambda route: 4
    runner._is_session_run_current = lambda route, generation: generation == 4
    for number in range(1000):
        assert await runner._dispatch_plugin_message_injection(
            session_key=key, content="peer answer", plugin_id="peer-test",
            delivery="peer", delivery_id=f"delivery-{number}")
    assert len(runner._peer_delivery_ledger) == 1000
    assert not hasattr(runner, "_peer_delivery_route_locks")
    assert not hasattr(runner, "_peer_delivery_route_lock")
    assert agent.queue_peer_notification.call_count == 1000
    assert await runner._dispatch_plugin_message_injection(
        session_key=key, content="peer answer", plugin_id="peer-test",
        delivery="peer", delivery_id="delivery-0")
    assert agent.queue_peer_notification.call_count == 1000

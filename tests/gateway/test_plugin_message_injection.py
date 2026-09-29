"""Tests for plugin-triggered turns in existing gateway sessions."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import hermes_yaml as yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    PlatformConfig,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _entry(*, origin=True) -> SessionEntry:
    source = None
    if origin:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="42",
            user_name="tester",
        )
    now = datetime.now()
    return SessionEntry(
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
    )


def _runner(entry: SessionEntry | None, adapter=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry)
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._is_user_authorized = MagicMock(return_value=True)
    return runner


class _RoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("network send is not expected")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_plugin_context_routes_through_live_gateway_to_existing_session(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = _entry().origin
    entry = store.get_or_create_session(source)
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    pending_user_event = MessageEvent(
        text="human follow-up",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["human.jpg"],
        media_types=["image/jpeg"],
    )
    adapter._pending_messages[entry.session_key] = pending_user_event

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._queued_events = {}
    runner._is_user_authorized = MagicMock(return_value=True)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
        manager,
    )

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert (
            context.inject_message(
                "/approve always",
                session_key=entry.session_key,
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert adapter._pending_messages[entry.session_key] is pending_user_event
        queued = runner._queued_events[entry.session_key][0]
        assert pending_user_event.text == "human follow-up"
        assert pending_user_event.media_urls == ["human.jpg"]
        assert pending_user_event.allow_gateway_control is True
        assert queued.text == "/approve always"
        assert queued.allow_gateway_control is False
        assert queued.metadata["gateway_session_id"] == entry.session_id
        adapter._message_handler.assert_not_awaited()

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False


@pytest.mark.asyncio
async def test_dispatch_uses_stored_origin_and_adapter_message_path():
    async def admit(event):
        event._gateway_accepted = True
    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=admit))
    entry = _entry()
    runner = _runner(entry, adapter)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="check the deployment",
        plugin_id="notify-plugin",
    )

    assert accepted is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "check the deployment"
    assert event.internal is True
    assert event.allow_gateway_control is False
    assert event.get_command() is None
    assert event.source == entry.origin
    assert event.source is not entry.origin
    runner._is_user_authorized.assert_called_once_with(
        event.source,
        allow_adapter_delegation=False,
    )
    assert event.metadata == {
        "hermes_plugin_id": "notify-plugin",
        "hermes_plugin_injection": True,
        "gateway_session_key": entry.session_key,
        "gateway_session_id": entry.session_id,
        "gateway_session_strict": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "with_adapter"),
    [
        (None, True),
        (_entry(origin=False), True),
        (_entry(), False),
    ],
)
async def test_dispatch_rejects_unroutable_session(entry, with_adapter):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter if with_adapter else None)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_dispatch_rechecks_current_authorization(raises):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    if raises:
        runner._is_user_authorized.side_effect = RuntimeError("config unavailable")
    else:
        runner._is_user_authorized.return_value = False

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_rejects_stored_role_only_authorization(monkeypatch):
    """A stored adapter role grant must be revalidated against current core auth."""
    for key in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.handle_message = AsyncMock()
    entry = _entry()
    entry.session_key = "agent:main:discord:dm:42"
    entry.platform = Platform.DISCORD
    source = entry.origin
    assert source is not None
    source.platform = Platform.DISCORD
    source.role_authorized = True

    runner = _runner(entry)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = GatewayConfig()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    del runner._is_user_authorized

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stops_when_gateway_drains_during_lookup():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()

    async def _lookup(_session_key):
        lookup_started.set()
        await release_lookup.wait()
        return _entry()

    runner._async_session_store.lookup_by_session_key = _lookup
    dispatch = asyncio.create_task(
        runner._dispatch_plugin_message_injection(
            session_key="agent:main:telegram:dm:42",
            content="wake up",
            plugin_id="notify-plugin",
        )
    )

    await lookup_started.wait()
    runner._draining = True
    release_lookup.set()

    assert await dispatch is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_queues_non_control_plugin_text_for_exact_session():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    source = _entry().origin
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    event = MessageEvent(
        text="/approve always",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": session_key},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._pending_messages[session_key] is event
    assert adapter._active_sessions[session_key].is_set() is False


@pytest.mark.asyncio
async def test_base_adapter_rejects_derived_session_mismatch():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    event = MessageEvent(
        text="ordinary input",
        source=_entry().origin,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": "agent:main:telegram:dm:other"},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._active_sessions == {}






@pytest.mark.asyncio
async def test_scheduler_logs_async_failure_without_callback_error():
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))
    runner._dispatch_plugin_message_injection = AsyncMock(
        side_effect=RuntimeError("adapter failed")
    )

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []






@pytest.mark.asyncio
@pytest.mark.parametrize("admission", [True, False, None])
async def test_context_reports_async_gateway_outcome_once(monkeypatch, admission):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="notify-plugin", key="notify-plugin", source="user"), manager)
    monkeypatch.setattr(context, "_gateway_injection_allowed", lambda: True)
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    if admission is None:
        runner._dispatch_plugin_message_injection = AsyncMock(side_effect=RuntimeError("unknown outcome"))
    else:
        runner._dispatch_plugin_message_injection = AsyncMock(return_value=admission)
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_message_injection)
    outcomes = []
    assert context.inject_message("wake", session_key="key", on_result=outcomes.append) is True
    await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    await asyncio.sleep(0)
    assert outcomes == [admission]


@pytest.mark.asyncio
async def test_outcome_callback_failure_does_not_change_admission(monkeypatch):
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)
    callback = MagicMock(side_effect=RuntimeError("observer failed"))
    assert runner._schedule_plugin_message_injection(
        session_key="key", content="wake", plugin_id="notify", on_result=callback) is True
    tasks = list(runner._background_tasks)
    assert await asyncio.gather(*tasks) == [True]
    await asyncio.sleep(0)
    callback.assert_called_once_with(True)


@pytest.mark.parametrize("kwargs_only", [False, True])
def test_tracked_context_keeps_legacy_injector_compatible(monkeypatch, kwargs_only):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="notify-plugin", key="notify-plugin", source="user"), manager)
    monkeypatch.setattr(context, "_gateway_injection_allowed", lambda: True)
    calls, outcomes = [], []
    def legacy(*, session_key, content, plugin_id):
        calls.append(content)
        return True
    def legacy_kwargs(**kwargs):
        calls.append(kwargs["content"])
        return True
    manager.set_gateway_message_injector(object(), legacy_kwargs if kwargs_only else legacy)
    assert context.inject_message("wake", session_key="key", on_result=outcomes.append) is True
    assert calls == ["wake"]
    assert outcomes == [None]


def test_scheduler_rejects_stopped_or_closed_gateway():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    runner._running = False

    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()

    runner._running = True
    runner._gateway_loop = None
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )

    runner._gateway_loop = loop
    loop.is_closed.return_value = True
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()


def test_scheduler_rejects_submission_failure():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _reject(coro, _target_loop, **_kwargs):
        coro.close()
        return None

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_reject):
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is False
        )


def test_install_and_clear_gateway_injector_preserves_newer_owner():
    runner = _runner(_entry())
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert manager.has_gateway_message_injector is True

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False

        runner._install_plugin_message_injector()

        newer_owner = MagicMock()
        newer_injector = MagicMock(return_value=True)
        manager.set_gateway_message_injector(newer_owner, newer_injector)
        runner._clear_plugin_message_injector()

    assert manager.has_gateway_message_injector is True
    assert manager.inject_gateway_message(value="kept") is True
    newer_injector.assert_called_once_with(value="kept")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["collision", "cap", "queued"])
async def test_dispatch_reports_real_adapter_admission(mode, caplog):
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    original = MessageEvent(text="human follow-up", source=entry.origin)
    adapter._pending_messages[entry.session_key] = original
    runner = _runner(entry, adapter)
    if mode != "collision":
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        runner._BUSY_QUEUE_MAX_PENDING = 1 if mode == "cap" else 10
    with caplog.at_level("INFO", logger="gateway.run"):
        accepted = await runner._dispatch_plugin_message_injection(
            session_key=entry.session_key, content="retained notification", plugin_id="notify-plugin",
        )
    assert accepted is (mode == "queued")
    assert adapter._pending_messages[entry.session_key] is original
    if mode != "queued":
        assert "injection dispatched" not in caplog.text
    else:
        queued = runner._queued_events[entry.session_key][0]
        assert queued._gateway_accepted is True
        assert queued.text == "retained notification"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
async def test_internal_fast_path_only_queues(mode):
    entry = _entry()
    adapter = _RoutingAdapter()
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._effective_busy_input_mode = lambda source: mode
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    runner._agent_has_active_subagents = lambda agent: False
    agent = MagicMock()
    runner._session_state(entry.session_key).turn.agent = agent
    event = MessageEvent(text="plugin wake", source=entry.origin, internal=True,
                         allow_gateway_control=False)
    await runner._hm_handle_running_session_message(event, event.source, entry.session_key)
    agent.interrupt.assert_not_called()
    agent._fold_into_running_turn.assert_not_called()
    assert adapter._pending_messages[entry.session_key] is event

@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["compression", "reset", "ended-compression"])
async def test_queued_plugin_wake_follows_only_compression(tmp_path, boundary):
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = _entry().origin
    entry = store.get_or_create_session(source)
    parent = entry.session_id
    store._db.publish_compression_child(
        parent_session_id=parent, child_session_id="compressed", source="telegram",
        messages=[{"role": "user", "content": "handoff"}], require_compression_lease=False,
    )
    entry.session_id = "manual-new" if boundary == "reset" else "compressed"
    if boundary == "ended-compression":
        store._db.end_session("compressed", "ws_orphan_reap")
    runner = _runner(entry)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._async_session_store._store = store
    runner._cache_session_source = lambda *args: None
    runner._is_telegram_topic_lane = lambda source: False
    event = MessageEvent(text="queued wake", source=source, internal=True, allow_gateway_control=False,
                         metadata={"gateway_session_key": entry.session_key,
                                   "gateway_session_id": parent, "gateway_session_strict": True})
    try:
        result = await runner._hmwa_resolve_session(event, source)
        if boundary == "compression":
            assert result is not None
            assert result[1].session_id == "compressed"
        else:
            assert result is None
    finally:
        store._db.close()

"""A fast-path correction and a parked wake must each survive interruption."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def setup_queue():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")
    key = "agent:main:telegram:dm:42"
    adapter = SimpleNamespace(_pending_messages={})
    adapter.get_pending_message = lambda session_key: adapter._pending_messages.pop(session_key, None)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._draining = False
    runner._delivery_adapter_for = lambda _: adapter
    wake = MessageEvent(text="retained notification", source=source, internal=True,
                        allow_gateway_control=False, metadata={"hermes_plugin_id": "notify"})
    runner._queue_or_replace_pending_event(key, wake)
    human = MessageEvent(text="my correction", source=source, message_id="correction-1")
    agent = SimpleNamespace(interrupt=MagicMock(), _supports_active_turn_redirect=False)
    return runner, adapter, source, key, wake, human, agent


@pytest.mark.asyncio
async def test_fast_interrupt_keeps_wake_and_original_human_event():
    runner, adapter, source, key, wake, human, agent = setup_queue()
    await runner._hm_busy_interrupt(human, source, agent, key)
    agent.interrupt.assert_called_once_with(human.text)
    result = {"interrupted": True, "interrupt_message": human.text}
    first, first_text = await runner._run_agent_drain_pending(result, adapter, source, key)
    second, second_text = await runner._run_agent_drain_pending({"completed": True}, adapter, source, key)
    assert first is wake
    assert second is human
    assert [first_text, second_text] == [wake.text, human.text]
    assert second.message_id == "correction-1"
    assert await runner._run_agent_drain_pending({"completed": True}, adapter, source, key) == (None, None)


@pytest.mark.asyncio
async def test_full_queue_does_not_interrupt_or_claim_correction_accepted():
    runner, adapter, source, key, wake, human, agent = setup_queue()
    runner._BUSY_QUEUE_MAX_PENDING = 1
    # This event may have been admitted into outer processing already; that is
    # not evidence it was retained in the busy queue.
    human._gateway_accepted = True
    reply = await runner._hm_busy_interrupt(human, source, agent, key)
    agent.interrupt.assert_not_called()
    assert reply and "full" in str(reply).lower()
    assert adapter._pending_messages[key] is wake
    assert not runner._overflow_queue(key)


@pytest.mark.asyncio
async def test_successful_redirect_is_not_also_queued():
    runner, adapter, source, key, wake, human, agent = setup_queue()
    agent._supports_active_turn_redirect = True
    agent.redirect = MagicMock()
    runner._redirect_active_turn = MagicMock(return_value=True)
    await runner._hm_busy_interrupt(human, source, agent, key)
    runner._redirect_active_turn.assert_called_once()
    agent.interrupt.assert_not_called()
    assert adapter._pending_messages[key] is wake
    assert not runner._overflow_queue(key)

@pytest.mark.asyncio
async def test_primary_busy_lane_refuses_before_interrupt_when_full():
    runner, adapter, source, key, wake, human, agent = setup_queue()
    runner._BUSY_QUEUE_MAX_PENDING = 1
    runner._is_user_authorized_for_source = lambda source: True
    runner._admit_bot_message_for_source = lambda source: True
    runner._effective_busy_input_mode = lambda source: "interrupt"
    runner._effective_busy_text_mode = lambda source: "interrupt"
    runner._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
    runner._resolve_busy_steer_or_redirect = AsyncMock(return_value=SimpleNamespace(
        effective_mode="interrupt", redirected=False, steered=False,
        demoted_for_subagents=False, demoted_for_compression=False))
    runner._session_state(key).turn.agent = agent
    runner._interrupt_running_agent_for_busy_event = AsyncMock()
    runner._compose_busy_ack_message = lambda *args, **kwargs: "queued"
    runner._send_busy_ack_reply = AsyncMock()
    assert await runner._handle_active_session_busy_message(human, key)
    runner._interrupt_running_agent_for_busy_event.assert_not_awaited()
    assert "not queued" in runner._send_busy_ack_reply.call_args.args[2]
    assert adapter._pending_messages[key] is wake


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,protected", [("queue", False), ("steer", False), ("interrupt", True)])
async def test_fast_queue_fallbacks_report_refusal(mode, protected):
    runner, adapter, source, key, wake, human, agent = setup_queue()
    runner._BUSY_QUEUE_MAX_PENDING = 1
    runner._session_state(key).turn.agent = agent
    runner._effective_busy_input_mode = lambda source: mode
    runner._agent_has_active_subagents = lambda agent: protected
    reply = await runner._hm_handle_running_session_message(human, source, key)
    assert reply and "not queued" in reply
    agent.interrupt.assert_not_called()
    assert adapter._pending_messages[key] is wake


@pytest.mark.asyncio
@pytest.mark.parametrize("incoming_internal", [True, False])
async def test_grace_window_keeps_internal_and_human_events_separate(incoming_internal):
    import time
    runner, adapter, source, key, wake, human, agent = setup_queue()
    state = runner._session_state(key)
    state.turn.agent, state.turn.started_ts = agent, time.time()
    runner._effective_busy_input_mode = lambda source: "interrupt"
    first, incoming = (human, wake) if incoming_internal else (wake, human)
    adapter._pending_messages[key] = first
    reply = await runner._hm_handle_running_session_message(incoming, source, key)
    assert reply is None
    assert adapter._pending_messages[key] is first
    assert first.text == ("my correction" if incoming_internal else "retained notification")
    assert list(runner._overflow_queue(key)) == [incoming]
    assert incoming._gateway_accepted
    agent.interrupt.assert_not_called()

"""A peer notice joins the running turn at a durable, role-safe request boundary."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.peer_notification import emit_included_peer_receipts, peer_user_row
from agent.prompt_builder import STEER_MARKER_OPEN
from agent.turn_api_call import perform_api_call
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from tools.registry import registry


TOOL = "peer_delivery_blocking_probe"
TOOL2 = "peer_delivery_second_probe"
entered = threading.Event()
release = threading.Event()


def blocking_probe(args, **kwargs):
    entered.set()
    assert release.wait(5)
    return "tool finished"


registry.register(
    name=TOOL, toolset="utility",
    schema={"name": TOOL, "description": "test probe",
            "parameters": {"type": "object", "properties": {}, "required": []}},
    handler=blocking_probe, override=True,
)
registry.register(
    name=TOOL2, toolset="utility",
    schema={"name": TOOL2, "description": "second test probe",
            "parameters": {"type": "object", "properties": {}, "required": []}},
    handler=lambda args, **kwargs: "second tool finished", override=True,
)


def test_peer_arriving_during_tool_is_included_once_in_same_turn(tmp_path):
    entered.clear()
    release.clear()
    schemas = [{"type": "function", "function": {"name": name, "description": "probe",
               "parameters": {"type": "object", "properties": {}, "required": []}}}
               for name in (TOOL, TOOL2)]
    db = SessionDB(tmp_path / "state.db")
    with (patch("model_tools.get_tool_definitions", return_value=schemas),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        session_db=db, session_id="peer-session")
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    captured = []
    tool_response = _mock_response(content="", finish_reason="tool_calls",
                                   tool_calls=[_mock_tool_call(name=TOOL), _mock_tool_call(name=TOOL2)])
    final_response = _mock_response(content="task continued", finish_reason="stop")

    def provider(**kwargs):
        captured.append(kwargs["messages"])
        return tool_response if len(captured) == 1 else final_response

    agent.client.chat.completions.create.side_effect = provider
    outcome = {}
    receipts = []

    def run():
        outcome["result"] = agent.run_conversation("original task")

    with (patch.object(agent, "_persist_session"),
          patch.object(agent, "_cleanup_task_resources")):
        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert entered.wait(5)
            assert agent.queue_peer_notification(
                "BAF-skill MESSAGE FROM AGENT TS8\nanswer envelope\n", "peer-plugin", "delivery-1",
                on_included=receipts.append)
        finally:
            release.set()
            worker.join(10)
    assert not worker.is_alive()
    assert len(captured) == 2
    rows = [row for row in outcome["result"]["messages"] if row.get("display_kind") == "peer_notification"]
    assert len(rows) == 1
    assert sum(row.get("display_kind") == "peer_notification"
               for row in db.get_messages("peer-session")) == 1
    assert rows[0]["display_metadata"]["delivery_id"] == "delivery-1"
    assert rows[0]["display_metadata"]["plugin_id"] == "peer-plugin"
    assert STEER_MARKER_OPEN not in rows[0]["content"]
    assert sum("answer envelope" in str(row.get("content")) for row in captured[1]) == 1
    roles = [row["role"] for row in captured[1]]
    assert roles[-3:] == ["tool", "tool", "user"]
    assert outcome["result"]["final_response"] == "task continued"
    assert len(receipts) == 1
    assert receipts[0]["delivery_id"] == "delivery-1"
    assert receipts[0]["session_id"] == "peer-session"
    assert receipts[0]["turn_id"] == outcome["result"]["turn_id"]
    assert receipts[0]["request_id"]
    assert agent._interrupt_requested is False
    assert not getattr(agent, "_active_children", ())


def test_peer_during_final_model_generation_queues_once_without_cancellation():
    with (patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    started, finish = threading.Event(), threading.Event()
    fallbacks = []

    def provider(**kwargs):
        started.set()
        assert finish.wait(5)
        return _mock_response(content="original task complete", finish_reason="stop")

    agent.client.chat.completions.create.side_effect = provider
    result = {}
    with (patch.object(agent, "_persist_session"),
          patch.object(agent, "_cleanup_task_resources")):
        worker = threading.Thread(target=lambda: result.setdefault(
            "turn", agent.run_conversation("original task")))
        worker.start()
        try:
            assert started.wait(5)
            assert agent._model_request_active.is_set()
            assert agent.queue_peer_notification("peer answer", "peer-plugin", "id-2",
                                                 on_fallback=fallbacks.append)
            assert agent._interrupt_requested is False
        finally:
            finish.set()
            worker.join(10)
    assert not worker.is_alive()
    assert result["turn"]["final_response"] == "original task complete"
    assert fallbacks == ["turn_ended"]
    assert all(row.get("display_kind") != "peer_notification" for row in result["turn"]["messages"])


def test_peer_buffer_never_broadcasts_to_active_children():
    agent = object.__new__(AIAgent)
    child = MagicMock()
    agent._active_children = [child]
    assert agent.queue_peer_notification("peer answer", "peer-plugin", "id-child")
    child.steer.assert_not_called()
    child.queue_peer_notification.assert_not_called()


def test_persisted_unincluded_peer_is_not_queued_a_second_time():
    agent = object.__new__(AIAgent)
    agent._interrupt_requested = False
    fallback = []
    assert agent.queue_peer_notification("peer answer", "peer-plugin", "id-persisted",
                                         on_fallback=fallback.append)
    messages = [{"role": "assistant", "tool_calls": [{"id": "call-1"}]},
                {"role": "tool", "tool_call_id": "call-1", "content": "done"}]
    assert agent._insert_pending_peer(messages)
    messages[-1]["_db_persisted"] = True
    agent._fallback_pending_peer()
    assert fallback == []
    assert messages[-1]["display_kind"] == "peer_notification"


@pytest.mark.parametrize("streaming", [False, True])
def test_cancelled_preflight_never_reports_peer_inclusion(streaming):
    receipts = []
    row = peer_user_row({"content": "peer answer\n", "plugin_id": "peer-plugin",
                         "delivery_id": "delivery-1"})
    row["_db_persisted"] = True
    notice = {"delivery_id": "delivery-1", "on_included": receipts.append}
    agent = MagicMock()
    agent.session_id = "peer-session"
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = True
    agent._peer_inserted = [(row, notice)]
    agent._model_request_active = threading.Event()
    agent._pending_redirect_lock = None
    agent._pending_redirect = None
    agent.provider = "openai"
    agent.model = "test-model"
    agent.base_url = "https://example.invalid"

    def cancelled(*args, **kwargs):
        raise InterruptedError("cancelled before provider dispatch")

    agent._interruptible_streaming_api_call.side_effect = cancelled
    agent._interruptible_api_call.side_effect = cancelled
    request = {"messages": [{"role": "user", "content": row["content"].rstrip("\n")}]}
    with (patch("agent.turn_api_call._should_stream", return_value=streaming),
          patch("hermes_cli.middleware.run_llm_execution_middleware",
                side_effect=lambda kwargs, perform, **unused: perform(kwargs)),
          patch("agent.relay_llm.execute", side_effect=lambda kwargs, dispatch, **unused: dispatch(kwargs))):
        with pytest.raises(InterruptedError):
            perform_api_call(
                agent, api_kwargs=request, _original_api_kwargs=request,
                _llm_middleware_trace=[], _moa_prepared_request=None, _retry=SimpleNamespace(),
                thinking_spinner=None, retry_count=0, api_call_count=1,
                api_request_id="request-1", effective_task_id="task-1", turn_id="turn-1",
                interrupted=False,
            )
    assert receipts == []


def test_equal_content_peer_rows_consume_distinct_wire_items_once():
    receipts = []
    agent = object.__new__(AIAgent)
    agent.session_id = "peer-session"
    agent._peer_inserted = []
    for delivery_id in ("delivery-1", "delivery-2"):
        row = peer_user_row({"content": "same payload\n", "plugin_id": "peer-plugin",
                             "delivery_id": delivery_id})
        row["_db_persisted"] = True
        agent._peer_inserted.append((row, {"delivery_id": delivery_id,
                                           "on_included": receipts.append}))
    wire = {"messages": [{"role": "user", "content": agent._peer_inserted[0][0]["content"].rstrip("\n")}]}
    emit_included_peer_receipts(agent, wire, turn_id="turn-1", request_id="request-1")
    assert [event["delivery_id"] for event in receipts] == ["delivery-1"]
    emit_included_peer_receipts(agent, wire, turn_id="turn-1", request_id="request-2")
    assert [event["delivery_id"] for event in receipts] == ["delivery-1"]
    both = {"messages": [wire["messages"][0], dict(wire["messages"][0])]}
    emit_included_peer_receipts(agent, both, turn_id="turn-1", request_id="request-3")
    assert [event["delivery_id"] for event in receipts] == ["delivery-1", "delivery-2"]


def test_streaming_dispatch_reports_persisted_peer_once_after_gate():
    from tests.agent.test_streaming import _make_stream_chunk

    row = peer_user_row({"content": "peer answer\n", "plugin_id": "peer-plugin",
                         "delivery_id": "delivery-stream"})
    row["_db_persisted"] = True
    receipts = []
    with (patch("agent.process_bootstrap.OpenAI"),
          patch("run_agent.AIAgent._create_request_openai_client") as create_client,
          patch("run_agent.AIAgent._close_request_openai_client")):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
        client = MagicMock()
        client.chat.completions.create.return_value = iter([
            _make_stream_chunk(content="done", finish_reason="stop", model="test-model")])
        create_client.return_value = client
        agent.api_mode = "chat_completions"
        agent.session_id = "peer-session"
        agent._interrupt_requested = False
        agent._peer_inserted = [(row, {"delivery_id": "delivery-stream",
                                       "on_included": receipts.append})]
        kwargs = {"model": "test-model", "messages": [{"role": "user",
                                                        "content": row["content"].rstrip("\n")}],
                  "stream": True}
        agent._interruptible_streaming_api_call(
            kwargs, on_dispatch=lambda wire: emit_included_peer_receipts(
                agent, wire, turn_id="turn-1", request_id="request-1"))
    assert [(event["delivery_id"], event["request_id"]) for event in receipts] == [
        ("delivery-stream", "request-1")]


def test_human_steer_takes_shared_tool_boundary_and_peer_falls_back_once():
    entered.clear()
    release.clear()
    schema = {"type": "function", "function": {"name": TOOL, "description": "probe",
              "parameters": {"type": "object", "properties": {}, "required": []}}}
    with (patch("model_tools.get_tool_definitions", return_value=[schema]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.process_bootstrap.OpenAI")):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    captured = []

    def provider(**kwargs):
        captured.append(kwargs["messages"])
        return (_mock_response(content="", finish_reason="tool_calls",
                               tool_calls=[_mock_tool_call(name=TOOL)]) if len(captured) == 1
                else _mock_response(content="original task continued", finish_reason="stop"))

    agent.client.chat.completions.create.side_effect = provider
    fallbacks = []
    outcome = {}
    with (patch.object(agent, "_persist_session"),
          patch.object(agent, "_cleanup_task_resources")):
        worker = threading.Thread(target=lambda: outcome.setdefault(
            "result", agent.run_conversation("original task")))
        worker.start()
        try:
            assert entered.wait(5)
            assert agent.steer("human correction")
            assert agent.queue_peer_notification("peer answer", "peer-plugin", "delivery-steer",
                                                 on_fallback=fallbacks.append)
        finally:
            release.set()
            worker.join(10)
    assert not worker.is_alive()
    assert len(captured) == 2
    assert [row["role"] for row in captured[1]][-2:] == ["tool", "user"]
    assert sum(STEER_MARKER_OPEN in str(row.get("content")) for row in captured[1]) == 1
    assert all("peer answer" not in str(row.get("content")) for row in captured[1])
    assert all(row.get("display_kind") != "peer_notification" for row in outcome["result"]["messages"])
    assert fallbacks == ["turn_ended"]

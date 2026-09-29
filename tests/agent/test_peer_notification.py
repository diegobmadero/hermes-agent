"""A peer notice joins the running turn at a durable, role-safe request boundary."""

import threading
from unittest.mock import MagicMock, patch

from agent.prompt_builder import STEER_MARKER_OPEN
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from tools.registry import registry


TOOL = "peer_delivery_blocking_probe"
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


def test_peer_arriving_during_tool_is_included_once_in_same_turn(tmp_path):
    entered.clear()
    release.clear()
    schema = {"type": "function", "function": {"name": TOOL, "description": "probe",
              "parameters": {"type": "object", "properties": {}, "required": []}}}
    db = SessionDB(tmp_path / "state.db")
    with (patch("model_tools.get_tool_definitions", return_value=[schema]),
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
                                   tool_calls=[_mock_tool_call(name=TOOL)])
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
                "BAF-skill MESSAGE FROM AGENT TS8\nanswer envelope", "peer-plugin", "delivery-1",
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
    assert outcome["result"]["final_response"] == "task continued"
    assert len(receipts) == 1
    assert receipts[0]["delivery_id"] == "delivery-1"
    assert receipts[0]["session_id"] == "peer-session"
    assert receipts[0]["turn_id"] == outcome["result"]["turn_id"]
    assert receipts[0]["request_id"]
    assert agent._interrupt_requested is False
    assert not getattr(agent, "_active_children", ())

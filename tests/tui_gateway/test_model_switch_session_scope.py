"""TUI/desktop integration contract for the model_switch tool.

Session scope must land on ``session["model_override"]`` — the same key the
TUI /model handler writes — so a rebuild (/new, resume) re-derives the
switched model and ``_sync_agent_model_with_config`` does not re-adopt the
config model at the next turn start. Turn scope must leave session state
untouched (the conversation loop reverts in-process).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tui_gateway.server as server


def _agent():
    agent = SimpleNamespace()
    agent.model = "qwen/qwen3.5-max-02-15"
    agent.provider = "openrouter"
    return agent


_OVERRIDE = {
    "model": "qwen/qwen3.5-max-02-15",
    "provider": "openrouter",
    "api_key": "sk-new",
    "base_url": "https://openrouter.ai/api/v1",
    "api_mode": "chat_completions",
}


class TestModelUpdateCallback:
    def _session(self, sid: str, agent=None) -> dict:
        return {"session_key": f"key-{sid}", "agent": agent}

    def test_session_scope_lands_on_model_override(self) -> None:
        session = self._session("s1")
        with patch.dict(server._sessions, {"s1": session}, clear=False):
            server._make_model_update_callback("s1")("old", dict(_OVERRIDE), "session")
        assert session["model_override"] == _OVERRIDE

    def test_live_agent_gets_runtime_persist_and_footer_update(self) -> None:
        agent = _agent()
        session = self._session("s2", agent=agent)
        with patch.dict(server._sessions, {"s2": session}, clear=False), \
                patch.object(server, "_persist_live_session_runtime") as persist_rt, \
                patch.object(server, "_emit") as emit:
            server._make_model_update_callback("s2")("old", dict(_OVERRIDE), "session")
        persist_rt.assert_called_once_with(session)
        assert emit.call_args.args[0] == "session.info"

    def test_turn_scope_leaves_session_untouched(self) -> None:
        session = self._session("s3")
        with patch.dict(server._sessions, {"s3": session}, clear=False), \
                patch.object(server, "_persist_live_session_runtime") as persist_rt:
            server._make_model_update_callback("s3")("old", dict(_OVERRIDE), "turn")
        assert "model_override" not in session
        persist_rt.assert_not_called()

    def test_unknown_session_is_noop(self) -> None:
        # Must not raise for a session that disappeared mid-turn.
        server._make_model_update_callback("gone")("old", dict(_OVERRIDE), "session")


class TestMakeAgentWiresModelCallback:
    def test_make_agent_passes_callback(self) -> None:
        fake_runtime = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
            "api_mode": "chat_completions",
            "command": None,
            "args": None,
            "credential_pool": None,
        }
        with (
            patch.object(server, "_load_cfg", return_value={"agent": {}}),
            patch.object(server, "_get_db", return_value=MagicMock()),
            patch.object(server, "_load_tool_progress_mode", return_value="compact"),
            patch.object(server, "_load_reasoning_config", return_value=None),
            patch.object(server, "_load_service_tier", return_value=None),
            patch.object(server, "_load_enabled_toolsets", return_value=None),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=fake_runtime,
            ),
            patch("run_agent.AIAgent") as mock_agent,
        ):
            server._make_agent("sid-cb", "key-cb")

        cb = mock_agent.call_args.kwargs.get("model_update_callback")
        assert callable(cb)

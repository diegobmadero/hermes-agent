"""Tests for AIAgent._apply_model_switch — the agent-loop model_switch core.

Hermetic: hermes_cli.model_switch.switch_model (resolution), config loading,
the cost guard, and the live swap (AIAgent.switch_model) are all patched, so
these tests pin the orchestration contract — validation order, allowlist and
cost-guard gating, turn snapshots, the per-turn budget, and the platform
callback payload — without any network or provider credentials.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.model_switch_tool import (
    MAX_SWITCHES_PER_TURN,
    TURN_REVERT_ATTR,
    TURN_SWITCH_COUNT_ATTR,
)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.model = "qwen/qwen3.5-plus-02-15"
    a.provider = "openrouter"
    return a


def _ok_result(new_model="qwen/qwen3.5-max-02-15", provider="openrouter"):
    return SimpleNamespace(
        success=True,
        new_model=new_model,
        target_provider=provider,
        api_key="sk-new",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        model_info=None,
        error_message="",
    )


def _err_result(message="Unknown model"):
    return SimpleNamespace(
        success=False,
        new_model="",
        target_provider="",
        api_key="",
        base_url="",
        api_mode="",
        model_info=None,
        error_message=message,
    )


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *a, **k: None,
    )


def _patch_resolver(monkeypatch, result):
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: result)


def _record_switch(agent):
    calls = []

    def _switch(new_model, new_provider, api_key="", base_url="", api_mode=""):
        calls.append({"new_model": new_model, "new_provider": new_provider})
        agent.model = new_model
        agent.provider = new_provider

    agent.switch_model = _switch
    return calls


class TestSessionScope:
    def test_success_applies_and_reports(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result())
        calls = _record_switch(agent)

        out = json.loads(
            agent._apply_model_switch(
                {"slug": "qwen-max", "reason": "hard proof", "scope": "session"}
            )
        )

        assert out["success"] is True
        assert out["old_model"] == "qwen/qwen3.5-plus-02-15"
        assert out["new_model"] == "qwen/qwen3.5-max-02-15"
        assert out["scope"] == "session"
        assert "prompt cache" in out["message"]
        assert calls == [
            {"new_model": "qwen/qwen3.5-max-02-15", "new_provider": "openrouter"}
        ]
        # Session scope leaves no pending turn revert.
        assert getattr(agent, TURN_REVERT_ATTR, None) is None

    def test_callback_receives_model_override_payload(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)
        seen = []
        agent.model_update_callback = lambda old, override, scope: seen.append(
            (old, override, scope)
        )

        agent._apply_model_switch({"slug": "qwen-max", "reason": "r"})

        assert seen == [
            (
                "qwen/qwen3.5-plus-02-15",
                {
                    "model": "qwen/qwen3.5-max-02-15",
                    "provider": "openrouter",
                    "api_key": "sk-new",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_mode": "chat_completions",
                },
                "session",
            )
        ]

    def test_callback_failure_does_not_break_switch(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)

        def _boom(*a, **k):
            raise RuntimeError("gateway state poisoned")

        agent.model_update_callback = _boom

        out = json.loads(agent._apply_model_switch({"slug": "qwen-max", "reason": "r"}))
        assert out["success"] is True
        assert agent.model == "qwen/qwen3.5-max-02-15"

    def test_session_switch_clears_pending_turn_revert(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)
        setattr(agent, TURN_REVERT_ATTR, {"model": "stale"})

        agent._apply_model_switch({"slug": "qwen-max", "reason": "r", "scope": "session"})

        assert getattr(agent, TURN_REVERT_ATTR) is None


class TestTurnScope:
    def test_snapshot_taken_before_mutation(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)

        out = json.loads(
            agent._apply_model_switch({"slug": "qwen-max", "reason": "r", "scope": "turn"})
        )

        assert out["scope"] == "turn"
        snapshot = getattr(agent, TURN_REVERT_ATTR)
        assert snapshot["model"] == "qwen/qwen3.5-plus-02-15"
        assert snapshot["provider"] == "openrouter"

    def test_turn_scope_skips_platform_callback(self, agent, monkeypatch):
        """Turn scope must not touch session state — the revert happens
        in-process, and a recorded session override would outlive it."""
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)
        seen = []
        agent.model_update_callback = lambda *a: seen.append(a)

        agent._apply_model_switch({"slug": "qwen-max", "reason": "r", "scope": "turn"})

        assert seen == []

    def test_second_turn_switch_keeps_original_snapshot(self, agent, monkeypatch):
        """Revert must land on the turn's STARTING runtime, not a hop."""
        _patch_resolver(monkeypatch, _ok_result("model-b"))
        _record_switch(agent)
        agent._apply_model_switch({"slug": "model-b", "reason": "r", "scope": "turn"})

        _patch_resolver(monkeypatch, _ok_result("model-c"))
        agent._apply_model_switch({"slug": "model-c", "reason": "r", "scope": "turn"})

        assert getattr(agent, TURN_REVERT_ATTR)["model"] == "qwen/qwen3.5-plus-02-15"

    def test_apply_failure_clears_created_snapshot(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result())

        def _boom(**kw):
            raise RuntimeError("bad credentials")

        agent.switch_model = _boom

        out = json.loads(
            agent._apply_model_switch({"slug": "qwen-max", "reason": "r", "scope": "turn"})
        )

        assert "error" in out
        assert getattr(agent, TURN_REVERT_ATTR, None) is None


class TestValidationAndGuards:
    def test_empty_slug_rejected_before_resolution(self, agent, monkeypatch):
        def _fail(**kw):
            raise AssertionError("resolver must not run")

        monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fail)
        out = json.loads(agent._apply_model_switch({"slug": "  ", "reason": "r"}))
        assert "error" in out

    def test_invalid_scope_rejected(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result())
        out = json.loads(
            agent._apply_model_switch({"slug": "x", "reason": "r", "scope": "forever"})
        )
        assert "error" in out
        assert out["valid_scopes"] == ["session", "turn"]

    def test_resolver_failure_is_graceful(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _err_result("Unknown model 'totally-fake'"))
        calls = _record_switch(agent)

        out = json.loads(agent._apply_model_switch({"slug": "totally-fake", "reason": "r"}))

        assert "error" in out
        assert "totally-fake" in out["error"]
        assert calls == []
        assert agent.model == "qwen/qwen3.5-plus-02-15"

    def test_no_change_short_circuits(self, agent, monkeypatch):
        _patch_resolver(
            monkeypatch, _ok_result("qwen/qwen3.5-plus-02-15", "openrouter")
        )
        calls = _record_switch(agent)

        out = json.loads(agent._apply_model_switch({"slug": "qwen", "reason": "r"}))

        assert out["no_change"] is True
        assert calls == []

    def test_allowlist_blocks_unlisted_model(self, agent, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"agent": {"model_switch_allowlist": ["sonnet"]}},
        )
        _patch_resolver(monkeypatch, _ok_result())
        calls = _record_switch(agent)

        out = json.loads(agent._apply_model_switch({"slug": "qwen-max", "reason": "r"}))

        assert "error" in out
        assert out["allowlist"] == ["sonnet"]
        assert calls == []

    def test_allowlist_matches_raw_slug_or_resolved_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"agent": {"model_switch_allowlist": ["qwen/qwen3.5-max-02-15"]}},
        )
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)

        out = json.loads(agent._apply_model_switch({"slug": "qwen-max", "reason": "r"}))
        assert out["success"] is True

    def test_allowlisted_model_skips_cost_guard(self, agent, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"agent": {"model_switch_allowlist": ["qwen/qwen3.5-max-02-15"]}},
        )

        def _guard(*a, **k):
            raise AssertionError("cost guard must not run for allowlisted targets")

        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning", _guard
        )
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)

        out = json.loads(agent._apply_model_switch({"slug": "qwen-max", "reason": "r"}))
        assert out["success"] is True

    def test_cost_guard_blocks_expensive_model_without_allowlist(self, agent, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            lambda *a, **k: SimpleNamespace(message="$$$/Mtok"),
        )
        _patch_resolver(monkeypatch, _ok_result())
        calls = _record_switch(agent)

        out = json.loads(agent._apply_model_switch({"slug": "qwen-max", "reason": "r"}))

        assert "error" in out
        assert "allowlist" in out["error"]
        assert calls == []

    def test_per_turn_budget_enforced(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _ok_result("model-b"))
        _record_switch(agent)

        assert (
            json.loads(agent._apply_model_switch({"slug": "b", "reason": "r"}))["success"]
            is True
        )
        _patch_resolver(monkeypatch, _ok_result("model-c"))
        assert (
            json.loads(agent._apply_model_switch({"slug": "c", "reason": "r"}))["success"]
            is True
        )
        _patch_resolver(monkeypatch, _ok_result("model-d"))
        out = json.loads(agent._apply_model_switch({"slug": "d", "reason": "r"}))

        assert "error" in out
        assert "budget" in out["error"]
        assert getattr(agent, TURN_SWITCH_COUNT_ATTR) == MAX_SWITCHES_PER_TURN

    def test_failed_validation_does_not_consume_budget(self, agent, monkeypatch):
        _patch_resolver(monkeypatch, _err_result())
        agent._apply_model_switch({"slug": "nope", "reason": "r"})
        assert getattr(agent, TURN_SWITCH_COUNT_ATTR, 0) == 0


class TestLoopIntegration:
    def test_invoke_tool_routes_model_switch(self, agent, monkeypatch):
        """The agent-loop interception must handle model_switch — the
        registry dispatcher only returns a stub error for loop tools."""
        _patch_resolver(monkeypatch, _ok_result())
        _record_switch(agent)

        out = json.loads(
            agent._invoke_tool(
                "model_switch",
                {"slug": "qwen-max", "reason": "complexity", "scope": "session"},
                "task-1",
            )
        )

        assert out["success"] is True
        assert agent.model == "qwen/qwen3.5-max-02-15"

    def test_guidance_present_only_with_tool(self):
        from agent.prompt_builder import MODEL_SWITCH_GUIDANCE
        from run_agent import AIAgent

        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("web_search", "model_switch"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            with_tool = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("web_search"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            without_tool = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        assert MODEL_SWITCH_GUIDANCE in with_tool._build_system_prompt()
        assert MODEL_SWITCH_GUIDANCE not in without_tool._build_system_prompt()

"""Tests for tools/model_switch_tool.py — schema, gating, turn-state helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from tools.model_switch_tool import (
    MAX_SWITCHES_PER_TURN,
    MODEL_SWITCH_SCHEMA,
    TURN_REVERT_ATTR,
    TURN_SWITCH_COUNT_ATTR,
    check_model_switch_requirements,
    model_switch_allowlist,
    reset_turn_model_switch_state,
)


class _StubAgent:
    """Records switch_model calls and mirrors the in-place swap."""

    def __init__(self):
        self.model = "qwen/qwen3.5-plus-02-15"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-old"
        self.api_mode = "chat_completions"
        self.switch_calls = []

    def switch_model(self, new_model, new_provider, api_key="", base_url="", api_mode=""):
        self.switch_calls.append(
            {
                "new_model": new_model,
                "new_provider": new_provider,
                "api_key": api_key,
                "base_url": base_url,
                "api_mode": api_mode,
            }
        )
        self.model = new_model
        self.provider = new_provider


class TestSchemaContract:
    def test_schema_requires_slug_and_reason(self):
        assert MODEL_SWITCH_SCHEMA["parameters"]["required"] == ["slug", "reason"]

    def test_scope_enum_is_session_and_turn(self):
        scope = MODEL_SWITCH_SCHEMA["parameters"]["properties"]["scope"]
        assert scope["enum"] == ["session", "turn"]

    def test_description_warns_about_prompt_cache(self):
        # The cache-bust cost is the operator's key concern — the schema
        # itself must teach it, not just the system-prompt guidance.
        assert "prompt cache" in MODEL_SWITCH_SCHEMA["description"]

    def test_description_never_offers_persistence(self):
        assert "persist" not in str(
            MODEL_SWITCH_SCHEMA["parameters"]["properties"]
        ).lower()


class TestAllowlistParser:
    def test_unset_returns_none(self):
        assert model_switch_allowlist({}) is None
        assert model_switch_allowlist(None) is None
        assert model_switch_allowlist({"agent": {}}) is None

    def test_empty_list_reads_as_unconfigured(self):
        assert model_switch_allowlist({"agent": {"model_switch_allowlist": []}}) is None

    def test_entries_normalized_lowercase(self):
        cfg = {"agent": {"model_switch_allowlist": ["  GPT-5.5 ", "sonnet"]}}
        assert model_switch_allowlist(cfg) == ["gpt-5.5", "sonnet"]

    def test_non_list_reads_as_unconfigured(self):
        assert (
            model_switch_allowlist({"agent": {"model_switch_allowlist": "sonnet"}})
            is None
        )


class TestOptInGate:
    def test_disabled_without_config_flag(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert check_model_switch_requirements() is False

    def test_enabled_with_flag(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"agent": {"allow_self_model_switch": True}},
        ):
            assert check_model_switch_requirements() is True

    def test_explicit_false_disables(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"agent": {"allow_self_model_switch": False}},
        ):
            assert check_model_switch_requirements() is False

    def test_config_error_fails_closed(self):
        with patch(
            "hermes_cli.config.load_config", side_effect=OSError("no config")
        ):
            assert check_model_switch_requirements() is False


class TestTurnStateReset:
    def test_reverts_snapshot_and_clears_it(self):
        agent = _StubAgent()
        setattr(
            agent,
            TURN_REVERT_ATTR,
            {
                "model": "small-model",
                "provider": "nous",
                "api_key": "sk-orig",
                "base_url": "https://nous.example/v1",
                "api_mode": "chat_completions",
            },
        )
        assert reset_turn_model_switch_state(agent) is True
        assert agent.switch_calls[-1]["new_model"] == "small-model"
        assert getattr(agent, TURN_REVERT_ATTR) is None
        # Idempotent: nothing pending on the second call.
        assert reset_turn_model_switch_state(agent) is False

    def test_resets_per_turn_budget(self):
        agent = _StubAgent()
        setattr(agent, TURN_SWITCH_COUNT_ATTR, MAX_SWITCHES_PER_TURN)
        reset_turn_model_switch_state(agent)
        assert getattr(agent, TURN_SWITCH_COUNT_ATTR) == 0

    def test_revert_failure_keeps_going(self):
        class _Boom(_StubAgent):
            def switch_model(self, *a, **k):
                raise RuntimeError("client rebuild failed")

        agent = _Boom()
        setattr(agent, TURN_REVERT_ATTR, {"model": "m", "provider": "p"})
        # Never raises; snapshot is consumed either way.
        assert reset_turn_model_switch_state(agent) is False
        assert getattr(agent, TURN_REVERT_ATTR) is None


class TestRegistry:
    def test_registered_with_check_fn(self):
        import importlib

        import tools.model_switch_tool as _mod
        from tools.registry import registry

        if registry.get_entry("model_switch") is None:
            importlib.reload(_mod)

        entry = registry.get_entry("model_switch")
        assert entry is not None
        assert entry.toolset == "model_switch"
        assert entry.check_fn is not None


class TestOptInFootprint:
    """model_switch is double opt-in — zero default schema footprint.

    Same contract the reasoning toolset shipped with: never in
    _HERMES_CORE_TOOLS (core tools are permanent schema cost on every
    request), off by default, enabled per platform via `hermes tools`,
    and additionally gated on agent.allow_self_model_switch via check_fn.
    """

    def test_not_in_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS

        assert "model_switch" not in _HERMES_CORE_TOOLS

    def test_toolset_off_by_default(self):
        from hermes_cli.tools_config import _DEFAULT_OFF_TOOLSETS

        assert "model_switch" in _DEFAULT_OFF_TOOLSETS

    def test_toolset_configurable(self):
        from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS

        assert any(key == "model_switch" for key, _, _ in CONFIGURABLE_TOOLSETS)

    def test_unconfigured_platform_resolution_excludes_model_switch(self):
        from hermes_cli.tools_config import _get_platform_tools

        enabled = _get_platform_tools({}, "cli", include_default_mcp_servers=False)
        assert "model_switch" not in enabled

    def test_explicit_opt_in_survives_resolution(self):
        from hermes_cli.tools_config import _get_platform_tools

        config = {"platform_toolsets": {"cli": ["hermes-cli", "model_switch"]}}
        enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
        assert "model_switch" in enabled

    def test_default_config_ships_flag_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["agent"]["allow_self_model_switch"] is False
        assert DEFAULT_CONFIG["agent"]["model_switch_allowlist"] == []

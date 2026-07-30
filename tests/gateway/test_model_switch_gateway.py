"""Gateway integration contract for the model_switch tool.

The failure mode these tests pin (the reason PR #37443 was rejected): an
in-process ``agent.switch_model`` without a session override is treated by
the gateway's post-run fallback detection as an accidental fallback —
``_is_intentional_model_switch`` returns False, the cached agent is evicted,
and the next message silently reverts to the config model. The callback must
therefore store the SAME override shape the /model handler writes, so both
the eviction guard and next-message runtime resolution see the switch.
"""

from unittest.mock import MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="hi", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(platform=platform, user_id=user_id, chat_id=chat_id)
    event = MagicMock(spec=MessageEvent)
    event.source = source
    event.text = text
    return event


def _make_runner():
    """Create a bare GatewayRunner without calling __init__ (same pattern as
    tests/gateway/test_reasoning_command.py)."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    return runner


_OVERRIDE = {
    "model": "qwen/qwen3.5-max-02-15",
    "provider": "openrouter",
    "api_key": "sk-new",
    "base_url": "https://openrouter.ai/api/v1",
    "api_mode": "chat_completions",
}


class TestModelUpdateCallback:
    def test_session_scope_stores_model_override(self):
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)

        runner._make_model_update_callback(session_key)("old-model", dict(_OVERRIDE), "session")

        assert runner._session_model_overrides[session_key] == _OVERRIDE

    def test_switched_model_counts_as_intentional(self):
        """The eviction-survival invariant: after the callback runs, the
        post-run fallback check must classify the new model as intentional."""
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)

        callback = runner._make_model_update_callback(session_key)
        assert (
            runner._is_intentional_model_switch(session_key, _OVERRIDE["model"]) is False
        )
        callback("old-model", dict(_OVERRIDE), "session")

        assert runner._is_intentional_model_switch(session_key, _OVERRIDE["model"]) is True

    def test_next_message_resolution_applies_override(self):
        """_apply_session_model_override must hand back the switched runtime
        for the next agent build — same rail /model uses."""
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)
        runner._make_model_update_callback(session_key)("old", dict(_OVERRIDE), "session")

        model, runtime_kwargs = runner._apply_session_model_override(
            session_key, "config-default-model", {}
        )

        assert model == _OVERRIDE["model"]
        assert runtime_kwargs["provider"] == "openrouter"
        assert runtime_kwargs["api_key"] == "sk-new"

    def test_turn_scope_stores_nothing(self):
        """Turn scope reverts in-process; a session override would outlive
        the revert and pin the temporary model."""
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)

        runner._make_model_update_callback(session_key)("old", dict(_OVERRIDE), "turn")

        assert session_key not in runner._session_model_overrides

    def test_isolated_between_sessions(self):
        runner = _make_runner()
        key_a = runner._session_key_for_source(_make_event(chat_id="chat-a").source)
        key_b = runner._session_key_for_source(_make_event(chat_id="chat-b").source)

        runner._make_model_update_callback(key_a)("old", dict(_OVERRIDE), "session")

        assert key_a in runner._session_model_overrides
        assert key_b not in runner._session_model_overrides

    def test_empty_session_key_is_noop(self):
        runner = _make_runner()
        runner._make_model_update_callback("")("old", dict(_OVERRIDE), "session")
        # Nothing stored anywhere; no exception raised.
        assert not dict(runner._session_model_overrides)

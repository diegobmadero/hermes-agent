#!/usr/bin/env python3
"""Model switch tool — agent-driven model routing at runtime.

Exposes the existing ``/model`` resolution pipeline
(:func:`hermes_cli.model_switch.switch_model`) as an agent-callable tool so
the model can escalate to a stronger model for hard work or downshift to a
cheaper one for routine turns, mid-session, without user intervention.

The tool itself is a thin validation layer: the actual state change lives in
the agent loop (``AIAgent._apply_model_switch``), which resolves the target,
applies it in place via ``AIAgent.switch_model`` (same rail as ``/model``),
and routes scope handling through ``model_update_callback`` so each platform
records the switch durably (gateway session model override, CLI runtime
attrs, TUI session ``model_override``). Without that callback the gateway's
post-run fallback detection would treat the new model as an accidental
fallback and evict the cached agent, silently reverting the switch.

Safety rails, in evaluation order inside ``AIAgent._apply_model_switch``:

- opt-in: the schema only appears when ``agent.allow_self_model_switch`` is
  true (check_fn) AND the ``model_switch`` toolset is enabled for the
  platform (off by default, not part of the core bundle);
- per-turn budget: at most ``MAX_SWITCHES_PER_TURN`` applied switches per
  turn, so the model cannot thrash between backends;
- allowlist: when ``agent.model_switch_allowlist`` is a non-empty list, only
  those models (raw slug or resolved id, case-insensitive) are permitted;
- cost guard: with no allowlist configured, targets that trip
  :func:`hermes_cli.model_cost_guard.expensive_model_warning` are refused —
  the interactive ``/model`` confirmation flow has no agent-side equivalent,
  so expensive targets require an explicit operator allowlist entry.

Scopes: ``session`` persists until the session resets or switches again;
``turn`` restores the pre-switch runtime at the start of the next turn (the
conversation loop calls :func:`reset_turn_model_switch_state`).
"""

import logging

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# Agent attribute holding the turn-scoped revert snapshot; read by the
# conversation loop's per-turn hook. Single source of truth here so the loop
# and the agent method cannot drift on the attribute name.
TURN_REVERT_ATTR = "_model_switch_turn_revert"

# Agent attribute counting applied switches in the current turn.
TURN_SWITCH_COUNT_ATTR = "_model_switches_this_turn"

# Applied-switch budget per turn: one escalation plus one return leg. A model
# that needs a third switch in a single turn is thrashing, not routing.
MAX_SWITCHES_PER_TURN = 2


MODEL_SWITCH_SCHEMA = {
    "name": "model_switch",
    "description": (
        "Switch your own model when the current one is clearly mismatched to "
        "the task: escalate for hard reasoning/coding, downshift for simple "
        "routine turns. The switch applies to your NEXT model request (it "
        "does not rerun the current one) and discards the provider prompt "
        "cache, so switching has real latency/cost overhead — use it only "
        "when the capability gap outweighs a cold cache, and never persist "
        "anything: session scope ends with the session. The user-facing "
        "/model command is unaffected and always wins."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Target model id or alias, resolved against the user's "
                    "configured providers exactly like the /model command "
                    "(e.g. 'qwen/qwen3.5-plus-02-15', 'gpt-5.5', 'sonnet')."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "One short sentence on why this switch is warranted; "
                    "recorded in logs for the operator."
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["session", "turn"],
                "description": (
                    "'session' (default): keep the new model for the rest of "
                    "the session. 'turn': use it for the REMAINDER of the "
                    "current turn only, then revert before the next turn."
                ),
            },
        },
        "required": ["slug", "reason"],
    },
}


def model_switch_allowlist(config: dict | None) -> list[str] | None:
    """Return the normalized ``agent.model_switch_allowlist``, or None.

    None means "no allowlist configured" (any resolvable, non-expensive model
    may be targeted); a non-empty list restricts targets to its entries.
    An empty/invalid value reads as unconfigured rather than lock-out, so a
    stray ``model_switch_allowlist: []`` in config.yaml cannot brick the tool.
    """
    agent_cfg = (config or {}).get("agent") or {}
    raw = agent_cfg.get("model_switch_allowlist") if isinstance(agent_cfg, dict) else None
    if not isinstance(raw, list):
        return None
    normalized = [str(m).strip().lower() for m in raw if str(m).strip()]
    return normalized or None


def reset_turn_model_switch_state(agent) -> bool:
    """Per-turn hook: revert a pending turn-scoped switch, reset the budget.

    Called at the start of ``run_conversation``. Returns True when a revert
    was applied. Revert failures keep the switched model and log a warning —
    the next ``/model`` or session rebuild recovers, and failing the turn
    over a cosmetic revert would be worse than running one more turn on the
    escalated model.
    """
    try:
        setattr(agent, TURN_SWITCH_COUNT_ATTR, 0)
    except Exception:
        pass
    snapshot = getattr(agent, TURN_REVERT_ATTR, None)
    if not snapshot:
        return False
    setattr(agent, TURN_REVERT_ATTR, None)
    try:
        agent.switch_model(
            new_model=snapshot.get("model", ""),
            new_provider=snapshot.get("provider", ""),
            api_key=snapshot.get("api_key", ""),
            base_url=snapshot.get("base_url", ""),
            api_mode=snapshot.get("api_mode", ""),
        )
        logger.info(
            "model_switch: reverted turn-scoped switch back to %s",
            snapshot.get("model"),
        )
        return True
    except Exception as exc:
        logger.warning("model_switch: turn-scope revert failed: %s", exc)
        return False


def check_model_switch_requirements() -> bool:
    """Schema gate: only offered when the operator opted in via config."""
    try:
        from hermes_cli.config import load_config
        from utils import is_truthy_value

        agent_cfg = (load_config() or {}).get("agent") or {}
        if not isinstance(agent_cfg, dict):
            return False
        return is_truthy_value(agent_cfg.get("allow_self_model_switch", False))
    except Exception:
        return False


# The registry stub only fires if a call bypasses the agent-loop interception
# (model_switch is in model_tools._AGENT_LOOP_TOOLS): the real handler needs
# the live agent, which the registry dispatcher does not have.
registry.register(
    name="model_switch",
    toolset="model_switch",
    schema=MODEL_SWITCH_SCHEMA,
    handler=lambda args, **kw: tool_error(
        "model_switch must run inside the agent loop"
    ),
    check_fn=check_model_switch_requirements,
    emoji="🔀",
)

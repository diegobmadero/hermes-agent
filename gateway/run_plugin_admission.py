"""Final-pickup admission for plugin-injected queued turns (``pre_plugin_injection_admit``).

A plugin wake can wait in the busy FIFO for a long time. The state that justified
scheduling it may be revoked before a model ever starts, so the owning plugin gets one
definite veto at the last point before execution. Only events the gateway itself built
in ``_dispatch_plugin_message_injection`` qualify; a user message whose text merely
resembles a plugin payload never reaches this hook.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_plugin_injection(event: Any) -> bool:
    metadata = getattr(event, "metadata", None) or {}
    return (getattr(event, "internal", False) is True
            and metadata.get("hermes_plugin_injection") is True
            and isinstance(metadata.get("hermes_plugin_id"), str))


async def plugin_injection_admitted(event: Any) -> bool:
    """False only when an owning plugin definitively rejects this unstarted injection."""
    if event is None or not is_plugin_injection(event):
        return True
    metadata = event.metadata
    try:
        from hermes_cli.lifecycle import ainvoke_hook, has_hook
        if not has_hook("pre_plugin_injection_admit"):
            return True
        results = await ainvoke_hook(
            "pre_plugin_injection_admit", event=event, plugin_id=metadata["hermes_plugin_id"],
            session_key=metadata.get("gateway_session_key", ""),
            session_id=metadata.get("gateway_session_id", ""), text=event.text or "",
        )
    except Exception:
        logger.warning("pre_plugin_injection_admit failed; admitting injection", exc_info=True)
        return True
    for result in results or ():
        if isinstance(result, dict) and result.get("action") == "reject":
            logger.info("Plugin injection rejected at final pickup: plugin=%s session=%s reason=%s",
                        metadata["hermes_plugin_id"], metadata.get("gateway_session_key", ""),
                        result.get("reason", ""))
            return False
    return True

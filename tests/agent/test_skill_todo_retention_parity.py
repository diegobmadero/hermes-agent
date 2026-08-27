"""Retention parity at the compaction boundary (#84718).

Compaction re-injects the todo list verbatim (``TODO_INJECTION_HEADER`` +
``TodoStore.format_for_injection``) while skill instructions are pruned down
to ``[SKILL_PRUNED: ...]`` markers. The imperative crosses the boundary; the
policy that governed it does not. These tests pin the coupling fix: when the
compressed transcript carries prune markers AND a todo snapshot is being
re-injected, the snapshot block must also carry an explicit instruction to
reload those skills before acting on the preserved tasks.

Invariants covered:

* the notice names every pruned skill with its exact ``skill_view`` call;
* skill guidance recovery now survives at least as well as the todo snapshot
  (same message, same strip lifecycle);
* the notice rides AFTER ``TODO_INJECTION_HEADER`` so the stale-snapshot
  strip removes both together — repeated boundaries never accumulate;
* deterministic and bounded — same input, same bytes; marker cap shared with
  the summary re-injection path;
* absent when nothing was pruned (zero recurring cost for clean sessions).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    _MAX_PRUNED_SKILL_MARKERS,
    _skill_pruned_marker,
    split_user_originated_turn,
)
from agent.conversation_compression import (
    _PRUNED_SKILL_RELOAD_NOTICE_HEADER,
    _pruned_skill_reload_notice,
    _strip_stale_todo_snapshot,
    _strip_stale_todo_snapshots,
)
from hermes_state import SessionDB
from tools.todo_tool import TODO_INJECTION_HEADER


def _build_agent_with_db(db: SessionDB, session_id: str, platform: str = "cli"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform=platform,
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = False
    return agent


def _msgs(n=20):
    # Large enough that the fake compressor's output is a genuine shrink —
    # the no-growth commit guard refuses compressions that grow the
    # transcript (see test_compression_rotation_state.py for the same shape).
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"m{i} " + "x" * 400,
        }
        for i in range(n)
    ]


class TestPrunedSkillReloadNotice:
    """Unit contract of the notice builder."""

    def test_names_every_pruned_skill_with_reload_call(self):
        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("hodle-design-system")
            + "\n"
            + _skill_pruned_marker("frontend-design")
        )
        notice = _pruned_skill_reload_notice(
            [{"role": "user", "content": summary}]
        )
        assert notice.startswith(_PRUNED_SKILL_RELOAD_NOTICE_HEADER)
        assert "skill_view(name='hodle-design-system')" in notice
        assert "skill_view(name='frontend-design')" in notice

    def test_collects_markers_from_pruned_tool_rows_in_tail(self):
        # A pruned skill_view row that survived inside the protected tail
        # carries the marker in tool-role content.
        rows = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "[skill_view] name=big-skill (18000 chars) "
                + _skill_pruned_marker("big-skill"),
            },
        ]
        notice = _pruned_skill_reload_notice(rows)
        assert "skill_view(name='big-skill')" in notice

    def test_deduplicates_and_preserves_first_seen_order(self):
        marker_a = _skill_pruned_marker("alpha")
        marker_b = _skill_pruned_marker("beta")
        rows = [
            {"role": "user", "content": f"{marker_a}\n{marker_b}\n{marker_a}"},
            {"role": "tool", "content": marker_a},
        ]
        notice = _pruned_skill_reload_notice(rows)
        assert notice.count("skill_view(name='alpha')") == 1
        assert notice.index("alpha") < notice.index("beta")

    def test_empty_when_nothing_pruned(self):
        rows = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        assert _pruned_skill_reload_notice(rows) == ""

    def test_bounded_by_shared_marker_cap(self):
        text = "\n".join(
            _skill_pruned_marker(f"skill-{i}")
            for i in range(_MAX_PRUNED_SKILL_MARKERS + 15)
        )
        notice = _pruned_skill_reload_notice([{"role": "user", "content": text}])
        assert notice.count("skill_view(name=") == _MAX_PRUNED_SKILL_MARKERS

    def test_deterministic_bytes(self):
        rows = [
            {"role": "user", "content": _skill_pruned_marker("stable-skill")}
        ]
        assert _pruned_skill_reload_notice(rows) == _pruned_skill_reload_notice(
            rows
        )

    def test_notice_does_not_feed_the_marker_extractor(self):
        """The notice must never re-trigger marker extraction on the next
        boundary — it references skills WITHOUT the canonical prefix."""
        from agent.context_compressor import _extract_pruned_skill_names

        notice = _pruned_skill_reload_notice(
            [{"role": "user", "content": _skill_pruned_marker("once")}]
        )
        assert _extract_pruned_skill_names(notice) == []


class TestSkillGuidanceSurvivesWithTodos:
    """Behavioral: skill reload guidance rides the same boundary artifact as
    the preserved todo list — retention parity, not asymmetry."""

    def _run_compaction(self, tmp_path: Path, summary_content: str):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_PARITY"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": summary_content},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": "tail"},
        ]
        agent._todo_store._items = [
            {
                "id": "remove",
                "content": "Remove the Lightning screen",
                "status": "pending",
            }
        ]
        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()
        return compressed

    def test_reload_instruction_travels_with_todo_snapshot(self, tmp_path):
        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("hodle-design-system")
        )
        compressed = self._run_compaction(tmp_path, summary)
        tail_text = str(compressed[-1]["content"])
        assert TODO_INJECTION_HEADER in tail_text
        assert "Remove the Lightning screen" in tail_text
        # Parity: the same message that preserved the imperative carries the
        # policy-recovery instruction.
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER in tail_text
        assert "skill_view(name='hodle-design-system')" in tail_text
        # Ordering: header first (the synthetic-row classifier keys on it),
        # notice after, inside the same strip window.
        assert tail_text.index(TODO_INJECTION_HEADER) < tail_text.index(
            _PRUNED_SKILL_RELOAD_NOTICE_HEADER
        )

    def test_no_notice_when_no_skills_pruned(self, tmp_path):
        compressed = self._run_compaction(
            tmp_path, "[CONTEXT COMPACTION] summary"
        )
        tail_text = str(compressed[-1]["content"])
        assert TODO_INJECTION_HEADER in tail_text
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER not in tail_text

    def test_synthetic_row_classification_unbroken(self, tmp_path):
        """A snapshot+notice appended as its own row must still classify as
        compression scaffolding, never as a real user turn."""
        from agent.context_compressor import ContextCompressor
        from agent.conversation_compression import _is_real_user_message

        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("frontend-design")
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_SYNTH"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        # Assistant tail → snapshot cannot merge; standalone flagged row.
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "acknowledged"},
        ]
        agent._todo_store._items = [
            {"id": "t1", "content": "task A", "status": "pending"}
        ]
        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()
        snapshot_rows = [
            m
            for m in compressed
            if isinstance(m, dict)
            and TODO_INJECTION_HEADER in str(m.get("content") or "")
        ]
        assert len(snapshot_rows) == 1
        row = snapshot_rows[0]
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER in str(row["content"])
        assert row.get("_todo_snapshot_synthetic") is True
        assert not _is_real_user_message(row)
        assert ContextCompressor._is_synthetic_compression_user_turn(row)


class TestNoticeStripLifecycle:
    """The notice is stripped with the stale snapshot — never accumulates."""

    def test_sweep_preserves_image_when_snapshot_text_part_is_removed(self):
        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        }
        snapshot_part = {
            "type": "text",
            "text": TODO_INJECTION_HEADER + "\n- [ ] t1. old task (pending)",
        }
        cleaned = _strip_stale_todo_snapshots(
            [{"role": "user", "content": [image_part, snapshot_part]}]
        )
        assert cleaned == [{"role": "user", "content": [image_part]}]

    def test_cleaned_multimodal_suffix_is_restored_when_compressor_omits_it(
        self, tmp_path
    ):
        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,REAL=="},
        }
        carrier = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        TODO_INJECTION_HEADER
                        + "\n- [ ] old. obsolete task (pending)\n\n"
                        + "Keep this human suffix."
                    ),
                },
                image_part,
            ],
        }
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MULTIMODAL_ANCHOR"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "compressed summary"}
        ]
        agent._todo_store._items = []
        original = [
            {
                "role": "assistant",
                "content": f"assistant segment {index} " + "x" * 400,
            }
            for index in range(20)
        ]
        original.append(carrier)

        compressed, _ = agent._compress_context(
            original, "sys", approx_tokens=120_000
        )
        db.close()

        anchors = [message for message in compressed if message.get("role") == "user"]
        assert len(anchors) == 1
        assert anchors[0]["content"] == [
            {"type": "text", "text": "Keep this human suffix."},
            image_part,
        ]

    def test_image_only_anchor_survives_stale_snapshot_stripping(self, tmp_path):
        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ONLY=="},
        }
        carrier = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": TODO_INJECTION_HEADER
                    + "\n- [ ] old. obsolete task (pending)",
                },
                image_part,
            ],
        }
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_IMAGE_ONLY_ANCHOR"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "assistant", "content": "compressed summary"}
        ]
        agent._todo_store._items = []
        original = [
            {
                "role": "assistant",
                "content": f"assistant segment {index} " + "x" * 400,
            }
            for index in range(20)
        ]
        original.append(carrier)

        compressed, _ = agent._compress_context(
            original, "sys", approx_tokens=120_000
        )
        db.close()

        anchors = [message for message in compressed if message.get("role") == "user"]
        assert len(anchors) == 1
        assert anchors[0]["content"] == [image_part]

    def test_sweep_repairs_role_alternation_before_durable_publication(self, tmp_path):
        stale_snapshot = (
            TODO_INJECTION_HEADER + "\n- [ ] old. obsolete task (pending)"
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_ROLE_REPAIR"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "first assistant segment"},
            {"role": "user", "content": stale_snapshot},
            {"role": "assistant", "content": "second assistant segment"},
            {"role": "user", "content": "latest human instruction"},
        ]
        agent._todo_store._items = [
            {"id": "fresh", "content": "continue safely", "status": "pending"}
        ]

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        durable = db.get_messages_as_conversation(agent.session_id)
        db.close()

        for transcript in (compressed, durable):
            roles = [message.get("role") for message in transcript]
            assert all(
                left != right or left == "tool"
                for left, right in zip(roles, roles[1:])
            )
            text = "\n".join(
                str(message.get("content") or "") for message in transcript
            )
            assert "first assistant segment" in text
            assert "second assistant segment" in text
            assert "obsolete task" not in text

    def test_user_summary_merges_losslessly_with_multimodal_anchor(
        self, tmp_path
    ):
        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ANCHOR=="},
        }
        carrier = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        TODO_INJECTION_HEADER
                        + "\n- [ ] old. obsolete task (pending)\n\n"
                        + "Keep the anchored instruction."
                    ),
                },
                image_part,
            ],
        }
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MULTIMODAL_USER_REPAIR"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {
                "role": "user",
                "content": f"{SUMMARY_PREFIX}\nsummary",
                COMPRESSED_SUMMARY_METADATA_KEY: True,
            }
        ]
        agent._todo_store._items = []
        original = [
            {
                "role": "assistant",
                "content": f"assistant segment {index} " + "x" * 400,
            }
            for index in range(50)
        ]
        original.append(carrier)

        compressed, _ = agent._compress_context(
            original, "sys", approx_tokens=120_000
        )
        durable = db.get_messages_as_conversation(agent.session_id)
        db.close()

        for transcript in (compressed, durable):
            roles = [message.get("role") for message in transcript]
            assert all(
                left != right or left == "tool"
                for left, right in zip(roles, roles[1:])
            )
            assert len(transcript) == 1
            content = transcript[0]["content"]
            assert isinstance(content, list)
            assert image_part in content
            text = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
            assert SUMMARY_PREFIX in text
            assert "Keep the anchored instruction." in text
            handoff, live_view = split_user_originated_turn(transcript[0])
            assert handoff is not None
            assert live_view is not None
            assert image_part in live_view["content"]

    def test_role_repair_preserves_both_multimodal_assistant_payloads(
        self, tmp_path
    ):
        first_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,FIRST=="},
        }
        second_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,SECOND=="},
        }
        stale_snapshot = (
            TODO_INJECTION_HEADER + "\n- [ ] old. obsolete task (pending)"
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MULTIMODAL_ASSISTANT_REPAIR"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first assistant segment"},
                    first_image,
                ],
            },
            {"role": "user", "content": stale_snapshot},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "second assistant segment"},
                    second_image,
                ],
            },
            {"role": "user", "content": "latest human instruction"},
        ]
        agent._todo_store._items = []

        compressed, _ = agent._compress_context(
            _msgs(50), "sys", approx_tokens=120_000
        )
        durable = db.get_messages_as_conversation(agent.session_id)
        db.close()

        for transcript in (compressed, durable):
            roles = [message.get("role") for message in transcript]
            assert all(
                left != right or left == "tool"
                for left, right in zip(roles, roles[1:])
            )
            assistant_parts = [
                part
                for message in transcript
                if message.get("role") == "assistant"
                for part in (
                    message.get("content")
                    if isinstance(message.get("content"), list)
                    else []
                )
            ]
            assert first_image in assistant_parts
            assert second_image in assistant_parts

    def test_strip_preserves_ordinary_user_header_mention(self):
        content = (
            "Can you explain "
            + TODO_INJECTION_HEADER
            + "?\nThis is ordinary user prose, not an injected task block."
        )
        assert _strip_stale_todo_snapshot(content) == content

    def test_strip_recognizes_multiline_todo_description(self):
        content = (
            TODO_INJECTION_HEADER
            + "\n- [ ] t1. first line\nsecond line (pending)\n\n"
            + "Keep this later human instruction."
        )
        assert _strip_stale_todo_snapshot(content) == (
            "Keep this later human instruction."
        )

    def test_strip_preserves_plain_human_suffix(self):
        content = (
            "summary before snapshot\n\n"
            + TODO_INJECTION_HEADER
            + "\n- [ ] t1. old task (pending)\n\n"
            + "Plain human suffix without a Markdown heading."
        )
        assert _strip_stale_todo_snapshot(content) == (
            "summary before snapshot\n\n"
            "Plain human suffix without a Markdown heading."
        )

    def test_strip_preserves_later_summary_section(self):
        content = (
            "summary before snapshot\n\n"
            + TODO_INJECTION_HEADER
            + "\n- [ ] t1. old task (pending)\n\n"
            + "## Later verified evidence\nkeep this section"
        )
        assert _strip_stale_todo_snapshot(content) == (
            "summary before snapshot\n\n"
            "## Later verified evidence\nkeep this section"
        )

    def test_strip_removes_snapshot_and_notice_together(self):
        content = (
            "real user words\n\n"
            + TODO_INJECTION_HEADER
            + "\n- [ ] t1. old task (pending)\n\n"
            + _PRUNED_SKILL_RELOAD_NOTICE_HEADER
            + "\nreload skill_view(name='old-skill') first."
        )
        stripped = _strip_stale_todo_snapshot(content)
        assert stripped == "real user words"
        assert _PRUNED_SKILL_RELOAD_NOTICE_HEADER not in stripped

    def test_repeated_boundaries_keep_single_notice(self, tmp_path):
        """Second compaction with a tail already carrying snapshot+notice
        refreshes in place instead of stacking duplicates (#26981 parity)."""
        summary = (
            "[CONTEXT COMPACTION] summary\n\n## Pruned Skills\n"
            + _skill_pruned_marker("hodle-design-system")
        )
        stale_tail = (
            "keep this human text\n\n"
            + TODO_INJECTION_HEADER
            + "\n- [ ] t0. stale task (pending)\n\n"
            + _PRUNED_SKILL_RELOAD_NOTICE_HEADER
            + "\nstale notice body skill_view(name='stale-skill')."
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_RESTRIP"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": stale_tail},
        ]
        agent._todo_store._items = [
            {"id": "t1", "content": "fresh task", "status": "pending"}
        ]
        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()
        tail_text = str(compressed[-1]["content"])
        assert tail_text.count(TODO_INJECTION_HEADER) == 1
        assert tail_text.count(_PRUNED_SKILL_RELOAD_NOTICE_HEADER) == 1
        assert "stale task" not in tail_text
        assert "stale-skill" not in tail_text
        assert "fresh task" in tail_text
        assert "skill_view(name='hodle-design-system')" in tail_text
        assert "keep this human text" in tail_text

    def test_repeated_boundaries_remove_stale_snapshot_outside_tail(self, tmp_path):
        """A prior snapshot in the middle must not survive the next boundary."""
        stale_snapshot = (
            TODO_INJECTION_HEADER
            + "\n- [ ] t0. stale task (pending)\n\n"
            + _PRUNED_SKILL_RELOAD_NOTICE_HEADER
            + "\nstale notice body skill_view(name='stale-skill')."
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_MIDDLE_RESTRIP"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] refreshed summary"},
            {"role": "user", "content": stale_snapshot},
            {"role": "assistant", "content": "continued after the first boundary"},
            {"role": "user", "content": "new human instruction"},
        ]
        agent._todo_store._items = [
            {"id": "t1", "content": "fresh task", "status": "pending"}
        ]

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()

        all_text = "\n".join(str(m.get("content") or "") for m in compressed)
        assert all_text.count(TODO_INJECTION_HEADER) == 1
        assert all_text.count(_PRUNED_SKILL_RELOAD_NOTICE_HEADER) == 0
        assert "stale task" not in all_text
        assert "stale-skill" not in all_text
        assert "fresh task" in all_text
        assert "new human instruction" in all_text

    def test_cleared_todos_remove_every_stale_snapshot(self, tmp_path):
        stale_snapshot = (
            TODO_INJECTION_HEADER
            + "\n- [>] t0. obsolete active task (in_progress)"
        )
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_SKILL_TODO_CLEARED_RESTRIP"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] refreshed summary"},
            {"role": "user", "content": stale_snapshot},
            {"role": "assistant", "content": "the task is complete"},
            {"role": "user", "content": "new human instruction"},
        ]
        agent._todo_store._items = []

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )
        db.close()

        all_text = "\n".join(str(m.get("content") or "") for m in compressed)
        assert TODO_INJECTION_HEADER not in all_text
        assert "obsolete active task" not in all_text
        assert "new human instruction" in all_text

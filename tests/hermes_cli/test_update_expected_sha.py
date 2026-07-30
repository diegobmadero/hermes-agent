"""Focused contracts for guarded ``hermes update --expected-sha``."""

from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd as hermes_update_cmd
from hermes_cli.subcommands.update import build_update_parser


EXPECTED = "a" * 40
OTHER = "b" * 40
BRANCH = "release/tested"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda args: args)
    return parser


def _setup_update(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: [])
    monkeypatch.setattr(hermes_main, "_get_origin_url", lambda *args: "https://github.com/example/repo.git")
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *args: None)
    # The update implementation moved to hermes_cli.update_cmd; helpers called
    # via its _m() indirection stay patchable on hermes_cli.main (above), but
    # module-local calls must be patched on update_cmd itself.
    monkeypatch.setattr(hermes_update_cmd, "_discard_lockfile_churn", lambda *args: None)
    monkeypatch.setattr(hermes_update_cmd, "_invalidate_update_cache", lambda: None)


def test_expected_sha_parser_normalizes_and_rejects_invalid_values():
    parsed = _parser().parse_args(["update", "--expected-sha", EXPECTED.upper()])
    assert parsed.expected_sha == EXPECTED

    for value in ("abc123", "g" * 40, "a" * 39, "a" * 41):
        with pytest.raises(SystemExit):
            _parser().parse_args(["update", "--expected-sha", value])


def test_expected_sha_mismatch_stops_before_checkout_stash_or_advance(
    monkeypatch, tmp_path, capsys
):
    _setup_update(monkeypatch, tmp_path)
    commands = []
    stash_calls = []
    monkeypatch.setattr(
        hermes_main,
        "_stash_local_changes_if_needed",
        lambda *args: stash_calls.append(args),
    )

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{OTHER}\n", stderr="")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command after mismatch: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(
            SimpleNamespace(branch=BRANCH, expected_sha=EXPECTED, yes=True)
        )

    assert stash_calls == []
    assert not any(command[1] in {"checkout", "pull", "merge", "reset"} for command in commands)
    assert "does not match expected SHA" in capsys.readouterr().out


def test_guarded_update_uses_single_fetch_ff_only_merge_and_checks_head(
    monkeypatch, tmp_path
):
    _setup_update(monkeypatch, tmp_path)
    commands = []

    class ReachedSyntaxGuard(RuntimeError):
        pass

    monkeypatch.setattr(
        hermes_update_cmd,
        "_validate_critical_files_syntax",
        lambda root: (_ for _ in ()).throw(ReachedSyntaxGuard()),
    )
    head_reads = 0

    def fake_run(cmd, **kwargs):
        nonlocal head_reads
        commands.append(cmd)
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{EXPECTED}\n", stderr="")
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{BRANCH}\n", stderr="")
        if cmd == ["git", "rev-list", f"HEAD..origin/{BRANCH}", "--count"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            head_reads += 1
            value = OTHER if head_reads == 1 else EXPECTED
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{value}\n", stderr="")
        if cmd == ["git", "merge", "--ff-only", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="Updating\n", stderr="")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(ReachedSyntaxGuard):
        hermes_main.cmd_update(
            SimpleNamespace(branch=BRANCH, expected_sha=EXPECTED, yes=True)
        )

    assert commands.count(["git", "fetch", "origin", BRANCH]) == 1
    assert ["git", "merge", "--ff-only", f"origin/{BRANCH}"] in commands
    assert not any("pull" in command or "reset" in command for command in commands)
    assert head_reads == 2


def test_guarded_ff_only_failure_never_uses_reset(monkeypatch, tmp_path):
    _setup_update(monkeypatch, tmp_path)
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{EXPECTED}\n", stderr="")
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{BRANCH}\n", stderr="")
        if cmd == ["git", "rev-list", f"HEAD..origin/{BRANCH}", "--count"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{OTHER}\n", stderr="")
        if cmd == ["git", "merge", "--ff-only", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="diverged\n")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(
            SimpleNamespace(branch=BRANCH, expected_sha=EXPECTED, yes=True)
        )

    assert not any("reset" in command for command in commands)


def test_guarded_update_rejects_post_merge_head_mismatch(monkeypatch, tmp_path, capsys):
    _setup_update(monkeypatch, tmp_path)
    head_reads = 0

    def fake_run(cmd, **kwargs):
        nonlocal head_reads
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{EXPECTED}\n", stderr="")
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{BRANCH}\n", stderr="")
        if cmd == ["git", "rev-list", f"HEAD..origin/{BRANCH}", "--count"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            head_reads += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{OTHER}\n", stderr="")
        if cmd == ["git", "merge", "--ff-only", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(
            SimpleNamespace(branch=BRANCH, expected_sha=EXPECTED, yes=True)
        )

    assert head_reads == 2
    assert "HEAD does not match expected SHA" in capsys.readouterr().out

"""Focused contracts for guarded ``hermes update --expected-sha``.

Pin mode is a strict transaction: the checkout must already be on the target
branch with a clean working tree, the fetched branch must resolve to the
pinned commit, HEAD must be an ancestor of that commit, and the only permitted
advance is a fast-forward to the immutable pinned SHA with an exact-HEAD check
inside the success region — before syntax acceptance and before any install
stage. Pinned updates never reconcile divergence, reset, switch branches,
auto-stash, synchronize upstream, or use the ZIP route.

History-transition cases run against real disposable Git graphs beneath
``tmp_path``; backup, installer, fleet, receipt-plan and network seams are
mocked. Command-boundary doubles remain for the lookup-failure edges real Git
cannot easily produce. No test runs the real host updater.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.update_receipt as update_receipt
from hermes_cli import main as hermes_main
from hermes_cli import update_cmd as hermes_update_cmd
from hermes_cli.subcommands.update import build_update_parser


EXPECTED = "a" * 40
OTHER = "b" * 40
BRANCH = "release/tested"

# Real subprocess.run, captured before any monkeypatch swaps the attribute.
_REAL_RUN = subprocess.run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda args: args)
    return parser


def _update_args(branch: str | None, expected: str | None) -> SimpleNamespace:
    return SimpleNamespace(branch=branch, expected_sha=expected, yes=True)


# ---------------------------------------------------------------------------
# Parser boundary
# ---------------------------------------------------------------------------

def test_expected_sha_parser_normalizes_and_rejects_invalid_values():
    parsed = _parser().parse_args(["update", "--expected-sha", EXPECTED.upper()])
    assert parsed.expected_sha == EXPECTED

    for value in ("abc123", "g" * 40, "a" * 39, "a" * 41):
        with pytest.raises(SystemExit):
            _parser().parse_args(["update", "--expected-sha", value])


# ---------------------------------------------------------------------------
# Command-boundary doubles (lookup failures real Git cannot easily produce)
# ---------------------------------------------------------------------------

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
    monkeypatch.setattr(hermes_update_cmd, "_begin_update_receipt_and_plan", lambda args: None)


def _clean_pin_preconditions(commands):
    """Command table entries for a clean, on-branch, ancestor pin precondition."""

    def entries(cmd):
        if cmd == ["git", "status", "--porcelain", "--untracked-files=all"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "merge-base", "--is-ancestor", "HEAD", EXPECTED]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return None

    return entries


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
        hermes_main.cmd_update(_update_args(BRANCH, EXPECTED))

    assert stash_calls == []
    assert not any(command[1] in {"checkout", "pull", "merge", "reset"} for command in commands)
    assert "does not match expected SHA" in capsys.readouterr().out


def test_expected_sha_fetched_ref_lookup_failure_stops_early(monkeypatch, tmp_path):
    """An unresolvable fetched ref is a refusal, never permission to mutate."""
    _setup_update(monkeypatch, tmp_path)
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: ambiguous argument")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command after lookup failure: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(_update_args(BRANCH, EXPECTED))

    assert not any(
        command[1] in {"checkout", "pull", "merge", "reset"}
        or command[1:3] == ["stash", "push"]
        for command in commands
    )


def test_guarded_update_uses_single_fetch_ff_only_merge_and_checks_head(
    monkeypatch, tmp_path
):
    """Pinned flow: one fetch, ff-only merge of the IMMUTABLE pinned SHA (not the
    movable tracking ref), and the exact-HEAD check lands before syntax acceptance."""
    _setup_update(monkeypatch, tmp_path)
    commands = []
    events = []

    class ReachedSyntaxGuard(RuntimeError):
        pass

    def syntax_guard(root):
        events.append(("syntax", None))
        raise ReachedSyntaxGuard()

    monkeypatch.setattr(hermes_update_cmd, "_validate_critical_files_syntax", syntax_guard)
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
        pre = _clean_pin_preconditions(commands)(cmd)
        if pre is not None:
            return pre
        if cmd == ["git", "rev-list", f"HEAD..origin/{BRANCH}", "--count"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            head_reads += 1
            # Pin precondition resolves HEAD first, then the pre-pull capture; the
            # post-merge exact-HEAD check is the first read that must see EXPECTED.
            value = OTHER if head_reads <= 2 else EXPECTED
            events.append(("head", value))
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{value}\n", stderr="")
        if cmd == ["git", "merge", "--ff-only", EXPECTED]:
            events.append(("merge", EXPECTED))
            return subprocess.CompletedProcess(cmd, 0, stdout="Updating\n", stderr="")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(ReachedSyntaxGuard):
        hermes_main.cmd_update(_update_args(BRANCH, EXPECTED))

    assert commands.count(["git", "fetch", "origin", BRANCH]) == 1
    assert ["git", "merge", "--ff-only", EXPECTED] in commands
    # The movable tracking ref must not be the merge target in pin mode.
    assert ["git", "merge", "--ff-only", f"origin/{BRANCH}"] not in commands
    assert not any("pull" in command or "reset" in command for command in commands)
    # Exact-HEAD verification is observed BEFORE syntax acceptance.
    assert events.index(("head", EXPECTED)) < events.index(("syntax", None))
    assert events.index(("merge", EXPECTED)) < events.index(("head", EXPECTED))


def test_guarded_ff_only_failure_never_reconciles_or_resets(monkeypatch, tmp_path):
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
        pre = _clean_pin_preconditions(commands)(cmd)
        if pre is not None:
            return pre
        if cmd == ["git", "rev-list", f"HEAD..origin/{BRANCH}", "--count"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{OTHER}\n", stderr="")
        if cmd == ["git", "merge", "--ff-only", EXPECTED]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="diverged\n")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(_update_args(BRANCH, EXPECTED))

    assert not any("reset" in command for command in commands)
    # Divergence reconciliation (branch probe, merge of origin ref) never runs.
    assert ["git", "branch", "--show-current"] not in commands
    assert not any(command[1] == "merge" and command[-1] != EXPECTED for command in commands if len(command) > 1)


def test_guarded_update_rejects_post_merge_head_mismatch(monkeypatch, tmp_path, capsys):
    """HEAD moved by something else after the ff: refuse before syntax/install."""
    _setup_update(monkeypatch, tmp_path)
    head_reads = 0
    syntax_calls = []
    monkeypatch.setattr(
        hermes_update_cmd,
        "_validate_critical_files_syntax",
        lambda root: syntax_calls.append(root) or (True, None, None),
    )

    def fake_run(cmd, **kwargs):
        nonlocal head_reads
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{EXPECTED}\n", stderr="")
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{BRANCH}\n", stderr="")
        if cmd == ["git", "status", "--porcelain", "--untracked-files=all"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "merge-base", "--is-ancestor", "HEAD", EXPECTED]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-list", f"HEAD..origin/{BRANCH}", "--count"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            head_reads += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{OTHER}\n", stderr="")
        if cmd == ["git", "merge", "--ff-only", EXPECTED]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(_update_args(BRANCH, EXPECTED))

    # Precondition resolve + pre-pull capture + post-merge exact-HEAD check.
    assert head_reads == 3
    assert syntax_calls == []  # refused before syntax acceptance
    assert "HEAD does not match expected SHA" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command_table,refusal_fragment",
    [
        pytest.param(
            {
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, "HEAD\n"),
            },
            "already be on",
            id="detached-head",
        ),
        pytest.param(
            {
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main\n"),
            },
            "already be on",
            id="different-branch",
        ),
        pytest.param(
            {
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, f"{BRANCH}\n"),
                ("status", "--porcelain", "--untracked-files=all"): (0, " M cli.py\n"),
            },
            "clean working tree",
            id="dirty-tree",
        ),
        pytest.param(
            {
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, f"{BRANCH}\n"),
                ("status", "--porcelain", "--untracked-files=all"): (0, ""),
                ("merge-base", "--is-ancestor", "HEAD", EXPECTED): (1, ""),
            },
            "not an ancestor",
            id="non-ancestor",
        ),
        pytest.param(
            {
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, f"{BRANCH}\n"),
                ("status", "--porcelain", "--untracked-files=all"): (0, ""),
                ("merge-base", "--is-ancestor", "HEAD", EXPECTED): (128, "fatal: bad object"),
            },
            "could not prove",
            id="ancestry-lookup-error",
        ),
    ],
)
def test_pinned_precondition_refusals_happen_before_any_mutation(
    monkeypatch, tmp_path, capsys, command_table, refusal_fragment
):
    """Branch/dirt/ancestry refusals fire before stash, checkout, merge or install."""
    _setup_update(monkeypatch, tmp_path)
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{EXPECTED}\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{OTHER}\n", stderr="")
        key = tuple(cmd[1:])
        if key in command_table:
            rc, out = command_table[key]
            return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command past refusal point: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(_update_args(BRANCH, EXPECTED))

    assert refusal_fragment in capsys.readouterr().out
    # No Git history/tree mutation; a read-only `stash list` (orphaned-autostash
    # warning) is part of the normal preflight and explicitly allowed.
    assert not any(
        command[1] in {"checkout", "pull", "merge", "reset"}
        or command[1:3] == ["stash", "push"]
        for command in commands
    )


def test_pinned_precondition_refuses_when_head_unresolvable(monkeypatch, tmp_path):
    """Indeterminate Git state is a refusal, never permission."""
    _setup_update(monkeypatch, tmp_path)

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "fetch", "origin", BRANCH]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", f"origin/{BRANCH}"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{EXPECTED}\n", stderr="")
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{BRANCH}\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: bad revision")
        if cmd and cmd[0] != "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(hermes_update_cmd.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(_update_args(BRANCH, EXPECTED))


# ---------------------------------------------------------------------------
# Real disposable Git graphs: history transitions of the pinned transaction
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _REAL_RUN(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=check)


def _git_sha(repo: Path, ref: str = "HEAD") -> str:
    return _git(repo, "rev-parse", ref).stdout.strip()


def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    _git(repo, "add", name)
    _git(
        repo,
        "-c", "user.name=Hermes Test",
        "-c", "user.email=hermes-test@example.invalid",
        "-c", "commit.gpgsign=false",
        "commit", "-m", message)
    return _git_sha(repo)


def _head_refs(repo: Path) -> str:
    return _git(repo, "for-each-ref", "refs/heads", "--format=%(refname) %(objectname)").stdout


@pytest.fixture()
def git_isolation(monkeypatch):
    """Hermetic Git: no global/system config, no credential helper prompts."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


def _build_pinned_graph(tmp_path: Path, branch: str = BRANCH) -> SimpleNamespace:
    """origin with base -> second on *branch*; work clone reset back to base."""
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(remote, "symbolic-ref", "HEAD", f"refs/heads/{branch}")

    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init")
    _git(seed, "checkout", "-b", branch)
    _git(seed, "remote", "add", "origin", str(remote))
    base = _commit_file(seed, "file.txt", "base\n", "base")
    second = _commit_file(seed, "file2.txt", "second\n", "second")
    _git(seed, "push", "-u", "origin", branch)

    work = tmp_path / "work"
    _git(tmp_path, "clone", str(remote), str(work))
    _git(work, "reset", "--hard", base)
    return SimpleNamespace(remote=remote, seed=seed, work=work, base=base, second=second, branch=branch)


@pytest.fixture()
def update_seams(monkeypatch):
    """Mock backup/installer/fleet/plan seams; Git operations stay real."""
    update_receipt._current = None
    calls = SimpleNamespace(installs=[], upstream_syncs=[], repairs=[], zips=[])

    def _begin_receipt(args):
        update_receipt.begin_update_receipt()
        return None

    monkeypatch.setattr(hermes_update_cmd, "_begin_update_receipt_and_plan", _begin_receipt)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda args: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda: [])
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *a: None)
    monkeypatch.setattr(hermes_update_cmd, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(hermes_update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    monkeypatch.setattr(hermes_update_cmd, "_sweep_bytecode_after_update", lambda *a: None)

    def _record_install(*args, **kwargs):
        calls.installs.append(
            _REAL_RUN(
                ["git", "rev-parse", "HEAD"], cwd=hermes_main.PROJECT_ROOT,
                capture_output=True, text=True).stdout.strip())

    monkeypatch.setattr(hermes_update_cmd, "_sync_python_dependencies_after_pull", _record_install)
    monkeypatch.setattr(hermes_update_cmd, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_update_cmd, "_rebuild_desktop_after_update", lambda *a, **kw: True)
    monkeypatch.setattr(hermes_update_cmd, "_run_post_update_maintenance", lambda **kw: True)
    monkeypatch.setattr(hermes_update_cmd, "_restart_gateway_fleet_after_update", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_update_cmd, "_resume_windows_gateways_and_merge_outcome", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_update_cmd, "_verify_fleet_after_update", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_update_cmd, "_apply_pending_fleet_restart_catchup", lambda: None)

    def _record_upstream_sync(*args, **kwargs):
        calls.upstream_syncs.append((args, kwargs))
        return True

    monkeypatch.setattr(hermes_main, "_sync_with_upstream_if_needed", _record_upstream_sync)

    def _record_repair(**kwargs):
        calls.repairs.append(kwargs)
        return True

    monkeypatch.setattr(hermes_update_cmd, "_repair_current_checkout", _record_repair)
    monkeypatch.setattr(
        hermes_update_cmd, "_update_via_zip",
        lambda *a, **kw: calls.zips.append((a, kw)) or True)
    yield calls
    update_receipt._current = None


def test_pinned_update_fast_forwards_to_immutable_sha_and_installs(
    git_isolation, update_seams, tmp_path, monkeypatch
):
    graph = _build_pinned_graph(tmp_path)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert _git_sha(graph.work) == graph.second
    assert _git_sha(graph.work, graph.branch) == graph.second
    # Exactly one install stage, observed at the exact pinned HEAD.
    assert update_seams.installs == [graph.second]
    assert update_seams.upstream_syncs == []
    assert update_seams.zips == []


def test_pinned_update_refuses_same_branch_divergence_without_reconciliation(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys
):
    graph = _build_pinned_graph(tmp_path)
    local = _commit_file(graph.work, "local.txt", "local work\n", "local commit")
    refs_before = _head_refs(graph.work)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert exc_info.value.code == 1
    # Branch, HEAD, local commits and files are exactly as before.
    assert _git_sha(graph.work) == local
    assert _git_sha(graph.work, graph.branch) == local
    assert _head_refs(graph.work) == refs_before
    assert (graph.work / "local.txt").read_text(encoding="utf-8") == "local work\n"
    assert update_seams.installs == [] and update_seams.repairs == []
    assert "ancestor" in capsys.readouterr().out


def test_pinned_update_refuses_detached_head_without_switching(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys
):
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "checkout", "--detach", "HEAD")
    refs_before = _head_refs(graph.work)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert exc_info.value.code == 1
    assert _git_sha(graph.work) == graph.base
    assert _git_sha(graph.work, graph.branch) == graph.base
    assert _head_refs(graph.work) == refs_before
    assert _git(graph.work, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "HEAD"
    assert update_seams.installs == [] and update_seams.repairs == []
    assert "already be on" in capsys.readouterr().out


def test_pinned_update_refuses_custom_branch_without_switch_or_merge(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys
):
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "checkout", "-b", "feature/local")
    refs_before = _head_refs(graph.work)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert exc_info.value.code == 1
    # Still on the custom branch; target branch unmoved; no new refs or merges.
    assert _git(graph.work, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "feature/local"
    assert _git_sha(graph.work, graph.branch) == graph.base
    assert _head_refs(graph.work) == refs_before
    assert update_seams.installs == [] and update_seams.repairs == []
    assert "already be on" in capsys.readouterr().out


@pytest.mark.parametrize("dirt", ["tracked", "untracked"])
def test_pinned_update_refuses_dirty_tree_without_stashing(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys, dirt
):
    graph = _build_pinned_graph(tmp_path)
    if dirt == "tracked":
        (graph.work / "file.txt").write_text("dirty edit\n", encoding="utf-8")
    else:
        (graph.work / "scratch-notes.txt").write_text("uncommitted notes\n", encoding="utf-8")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert exc_info.value.code == 1
    assert _git_sha(graph.work) == graph.base
    # The dirt itself is untouched (no auto-stash/discard).
    if dirt == "tracked":
        assert (graph.work / "file.txt").read_text(encoding="utf-8") == "dirty edit\n"
    else:
        assert (graph.work / "scratch-notes.txt").read_text(encoding="utf-8") == "uncommitted notes\n"
    assert update_seams.installs == [] and update_seams.repairs == []
    assert "clean working tree" in capsys.readouterr().out


def test_pinned_update_refuses_local_ahead_history(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys
):
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "reset", "--hard", graph.second)
    ahead = _commit_file(graph.work, "ahead.txt", "ahead of pin\n", "local ahead")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert exc_info.value.code == 1
    assert _git_sha(graph.work) == ahead
    assert update_seams.installs == [] and update_seams.repairs == []
    assert "ancestor" in capsys.readouterr().out


def test_pinned_update_already_at_expected_is_verified_noop(
    git_isolation, update_seams, tmp_path, monkeypatch
):
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "reset", "--hard", graph.second)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert _git_sha(graph.work) == graph.second
    # Existing repair machinery may run at exact HEAD; installation must not.
    assert len(update_seams.repairs) == 1
    assert update_seams.installs == []
    assert update_seams.zips == []


def test_pinned_update_merges_immutable_sha_when_tracking_ref_moves(
    git_isolation, update_seams, tmp_path, monkeypatch
):
    """A tracking ref advanced by another writer after the fetch must not redirect
    the merge: the pinned object itself is the ff-only target."""
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "checkout", "-b", "scratch", graph.second)
    third = _commit_file(graph.work, "file3.txt", "third\n", "third")
    _git(graph.work, "checkout", graph.branch)
    _git(graph.work, "reset", "--hard", graph.base)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    moved = {"done": False}

    def racing_run(cmd, **kwargs):
        result = _REAL_RUN(cmd, **kwargs)
        if not moved["done"] and list(cmd) == ["git", "rev-parse", f"origin/{graph.branch}"]:
            # Another writer advanced origin/<branch> right after our fetch.
            _git(graph.work, "update-ref", f"refs/remotes/origin/{graph.branch}", third)
            moved["done"] = True
        return result

    monkeypatch.setattr(subprocess, "run", racing_run)

    hermes_main.cmd_update(_update_args(graph.branch, graph.second))

    assert moved["done"] is True  # the race actually engaged
    assert _git_sha(graph.work) == graph.second
    assert update_seams.installs == [graph.second]


def test_pinned_update_on_fork_main_never_syncs_upstream_after_pull(
    git_isolation, update_seams, tmp_path, monkeypatch
):
    graph = _build_pinned_graph(tmp_path, branch="main")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    hermes_main.cmd_update(_update_args("main", graph.second))

    assert _git_sha(graph.work) == graph.second
    assert update_seams.installs == [graph.second]
    assert update_seams.upstream_syncs == []


def test_pinned_update_noop_on_fork_main_never_syncs_upstream(
    git_isolation, update_seams, tmp_path, monkeypatch
):
    graph = _build_pinned_graph(tmp_path, branch="main")
    _git(graph.work, "reset", "--hard", graph.second)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    hermes_main.cmd_update(_update_args("main", graph.second))

    assert len(update_seams.repairs) == 1
    assert update_seams.installs == []
    assert update_seams.upstream_syncs == []


def test_unpinned_update_on_fork_main_keeps_upstream_sync_and_origin_target(
    git_isolation, update_seams, tmp_path, monkeypatch
):
    """Control: default unpinned behavior is upstream's — origin/<branch> merge
    target and the optional fork-main upstream synchronization both remain."""
    graph = _build_pinned_graph(tmp_path, branch="main")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    hermes_main.cmd_update(_update_args("main", None))

    assert _git_sha(graph.work) == graph.second
    assert update_seams.installs == [graph.second]
    assert update_seams.upstream_syncs  # post-pull fork-main sync still offered


def test_pinned_update_syntax_failure_is_a_failed_transaction_with_receipt(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys
):
    """After a successful exact ff, invalid critical syntax rolls back to the
    captured pre-pull SHA and reports failure — never success, never install."""
    graph = _build_pinned_graph(tmp_path)
    broken = _commit_file(
        graph.seed, "cli.py",
        'x = {\n    "a": 1,\n<<<<<<< HEAD\n=======\n>>>>>>> deadbeef\n}\n',
        "break a critical file")
    _git(graph.seed, "push", "origin", graph.branch)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args(graph.branch, broken))

    assert exc_info.value.code == 1
    assert _git_sha(graph.work) == graph.base
    assert not (graph.work / "cli.py").exists()
    assert update_seams.installs == []
    out = capsys.readouterr().out
    assert "Rolling back" in out
    latest = update_receipt.read_latest_receipt()
    assert latest is not None
    assert latest["outcome"] == "failed"
    assert latest["exit_code"] == 1


# ---------------------------------------------------------------------------
# ZIP routes under a pin (service seam patched; never a real download)
# ---------------------------------------------------------------------------

def _zip_route_setup(git_isolation, update_seams, monkeypatch, tmp_path):
    fake_root = tmp_path / "zip-install"
    fake_root.mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(
        hermes_update_cmd, "_prepare_git_command", lambda **kw: (True, ["git"], False))
    return fake_root


def test_pinned_update_refuses_initial_zip_route(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys
):
    _zip_route_setup(git_isolation, update_seams, monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args("main", EXPECTED))

    assert exc_info.value.code == 1
    assert update_seams.zips == []
    assert "ZIP" in capsys.readouterr().out


def test_unpinned_update_still_uses_initial_zip_route(
    git_isolation, update_seams, tmp_path, monkeypatch
):
    _zip_route_setup(git_isolation, update_seams, monkeypatch, tmp_path)

    hermes_main.cmd_update(_update_args("main", None))

    assert len(update_seams.zips) == 1


def test_pinned_update_never_falls_back_to_zip_after_git_error(monkeypatch):
    """A Git-shaped CalledProcessError that would ordinarily qualify for the
    Windows ZIP fallback is a hard refusal under a pin."""
    import hermes_cli.main_install_repair as main_install_repair

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    zip_calls = []
    monkeypatch.setattr(
        hermes_update_cmd, "_update_via_zip", lambda *a, **kw: zip_calls.append(1) or True)
    exc = subprocess.CalledProcessError(
        1, ["git", "fetch", "origin", "main"], stderr="early EOF")

    with pytest.raises(SystemExit) as exc_info:
        hermes_update_cmd._handle_update_called_process_error(
            exc, SimpleNamespace(expected_sha=EXPECTED), False, False)

    assert exc_info.value.code == 1
    assert zip_calls == []


def test_unpinned_windows_git_error_still_falls_back_to_zip(monkeypatch):
    """Control: the unpinned ZIP fallback contract is unchanged."""
    import hermes_cli.main_install_repair as main_install_repair

    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    zip_calls = []
    monkeypatch.setattr(
        hermes_update_cmd, "_update_via_zip", lambda *a, **kw: zip_calls.append(1) or True)
    exc = subprocess.CalledProcessError(
        1, ["git", "fetch", "origin", "main"], stderr="early EOF")

    hermes_update_cmd._handle_update_called_process_error(
        exc, SimpleNamespace(expected_sha=None), False, False)

    assert zip_calls == [1]


# ---------------------------------------------------------------------------
# Pinned syntax-rollback preconditions (real graphs, direct guard calls)
# ---------------------------------------------------------------------------

def _force_syntax_failure(monkeypatch):
    monkeypatch.setattr(
        hermes_update_cmd, "_validate_critical_files_syntax",
        lambda root: (False, "cli.py", "SyntaxError: invalid syntax"))


def test_pinned_syntax_rollback_restores_pre_pull_sha(
    git_isolation, tmp_path, monkeypatch
):
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "reset", "--hard", graph.second)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)
    _force_syntax_failure(monkeypatch)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._rollback_if_pulled_syntax_error(
            ["git"], graph.base, expected_sha=graph.second)

    assert _git_sha(graph.work) == graph.base


def test_pinned_syntax_rollback_refused_when_head_moved(
    git_isolation, tmp_path, monkeypatch, capsys
):
    """A raced HEAD that is no longer the pinned commit is never reset."""
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "reset", "--hard", graph.second)
    raced = _commit_file(graph.work, "raced.txt", "foreign commit\n", "foreign")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)
    _force_syntax_failure(monkeypatch)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._rollback_if_pulled_syntax_error(
            ["git"], graph.base, expected_sha=graph.second)

    assert _git_sha(graph.work) == raced
    out = capsys.readouterr().out
    assert graph.second in out and graph.base in out  # both SHAs retained


def test_pinned_syntax_rollback_refused_when_tree_dirty(
    git_isolation, tmp_path, monkeypatch, capsys
):
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "reset", "--hard", graph.second)
    (graph.work / "file.txt").write_text("raced edit\n", encoding="utf-8")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)
    _force_syntax_failure(monkeypatch)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._rollback_if_pulled_syntax_error(
            ["git"], graph.base, expected_sha=graph.second)

    assert _git_sha(graph.work) == graph.second
    assert (graph.work / "file.txt").read_text(encoding="utf-8") == "raced edit\n"
    out = capsys.readouterr().out
    assert graph.second in out and graph.base in out


def test_unpinned_syntax_rollback_retains_existing_reset_semantics(
    git_isolation, tmp_path, monkeypatch
):
    """Control: without a pin the legacy unconditional rollback is unchanged."""
    graph = _build_pinned_graph(tmp_path)
    _git(graph.work, "reset", "--hard", graph.second)
    (graph.work / "file.txt").write_text("raced edit\n", encoding="utf-8")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)
    _force_syntax_failure(monkeypatch)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._rollback_if_pulled_syntax_error(["git"], graph.base)

    assert _git_sha(graph.work) == graph.base


# ---------------------------------------------------------------------------
# Pinned rollback vs ignored local bytes (ASTRA-R1)
# ---------------------------------------------------------------------------

def _build_ignored_collision_repo(tmp_path: Path, *, directory_obstruction: bool = False):
    """base tracks cache/recover.txt (force-added past the ignore rule) and a valid
    cli.py; pinned removes it and breaks cli.py. After the pinned checkout a local
    process recreates the now-ignored path with unrelated bytes (or as a directory
    with its own state, for the obstruction variant)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / ".gitignore").write_text("cache/\n", encoding="utf-8")
    (repo / "cache").mkdir()
    (repo / "cache" / "recover.txt").write_text("old tracked content\n", encoding="utf-8")
    (repo / "cli.py").write_text("print('valid fixture')\n", encoding="utf-8")
    _git(repo, "add", "-f", "cache/recover.txt")
    _git(repo, "add", ".gitignore", "cli.py")
    _git(repo, "-c", "user.name=Hermes Test", "-c", "user.email=hermes-test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-m", "pre-update")
    base = _git_sha(repo)
    _git(repo, "rm", "-q", "cache/recover.txt")
    (repo / "cli.py").write_text("def broken(:\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=Hermes Test", "-c", "user.email=hermes-test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-m", "pinned bad syntax")
    pinned = _git_sha(repo)
    (repo / "cache").mkdir(exist_ok=True)
    if directory_obstruction:
        (repo / "cache" / "recover.txt").mkdir()
        (repo / "cache" / "recover.txt" / "notes.txt").write_text(
            "local directory state\n", encoding="utf-8")
    else:
        (repo / "cache" / "recover.txt").write_text(
            "unrelated ignored local notes\n", encoding="utf-8")
    return SimpleNamespace(repo=repo, base=base, pinned=pinned)


def test_pinned_rollback_refuses_ignored_file_collision_and_preserves_bytes(
    git_isolation, tmp_path, monkeypatch, capsys
):
    """The rollback restores cache/recover.txt; an ignored local file already sits
    there. Automatic reset must be refused, the local bytes preserved byte-for-byte,
    HEAD left on the pinned commit, and both SHAs retained."""
    graph = _build_ignored_collision_repo(tmp_path)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.repo)
    local_bytes = (graph.repo / "cache" / "recover.txt").read_bytes()

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._rollback_if_pulled_syntax_error(
            ["git"], graph.base, expected_sha=graph.pinned)

    assert _git_sha(graph.repo) == graph.pinned
    assert (graph.repo / "cache" / "recover.txt").read_bytes() == local_bytes
    out = capsys.readouterr().out
    assert "Refusing automatic rollback" in out
    assert graph.pinned in out and graph.base in out


def test_pinned_rollback_refuses_ignored_directory_obstruction(
    git_isolation, tmp_path, monkeypatch, capsys
):
    """The restored path is occupied by an ignored local DIRECTORY: rollback must
    not delete or replace that unrelated directory state."""
    graph = _build_ignored_collision_repo(tmp_path, directory_obstruction=True)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.repo)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._rollback_if_pulled_syntax_error(
            ["git"], graph.base, expected_sha=graph.pinned)

    assert _git_sha(graph.repo) == graph.pinned
    assert (graph.repo / "cache" / "recover.txt").is_dir()
    assert (
        (graph.repo / "cache" / "recover.txt" / "notes.txt").read_text(encoding="utf-8")
        == "local directory state\n"
    )
    out = capsys.readouterr().out
    assert "Refusing automatic rollback" in out
    assert graph.pinned in out and graph.base in out


def test_pinned_rollback_allows_harmless_noncolliding_ignored_cache(
    git_isolation, tmp_path, monkeypatch
):
    """Control: an ignored file untouched by the restored tree must not disable
    the valid rollback, and must itself be preserved."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo / "cli.py").write_text("print('valid fixture')\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=Hermes Test", "-c", "user.email=hermes-test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-m", "pre-update")
    base = _git_sha(repo)
    (repo / "cli.py").write_text("def broken(:\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=Hermes Test", "-c", "user.email=hermes-test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-m", "pinned bad syntax")
    pinned = _git_sha(repo)
    (repo / "debug.log").write_text("harmless ignored cache\n", encoding="utf-8")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._rollback_if_pulled_syntax_error(["git"], base, expected_sha=pinned)

    assert _git_sha(repo) == base
    assert (repo / "cli.py").read_text(encoding="utf-8") == "print('valid fixture')\n"
    assert (repo / "debug.log").read_text(encoding="utf-8") == "harmless ignored cache\n"


def test_pinned_update_rollback_refuses_ignored_collision_at_command_boundary(
    git_isolation, update_seams, tmp_path, monkeypatch, capsys
):
    """Full production boundary: the ignored collision is created by a local process
    between the exact fast-forward and syntax validation; the failed transaction must
    keep the pinned HEAD, the local bytes, and record a failed receipt."""
    graph = _build_pinned_graph(tmp_path)
    # Rebuild the seed history with the ignore/collision shape.
    _git(graph.seed, "rm", "-q", "file2.txt")
    (graph.seed / ".gitignore").write_text("cache/\n", encoding="utf-8")
    (graph.seed / "cache").mkdir()
    (graph.seed / "cache" / "recover.txt").write_text("old tracked content\n", encoding="utf-8")
    (graph.seed / "cli.py").write_text("print('valid fixture')\n", encoding="utf-8")
    _git(graph.seed, "add", "-f", "cache/recover.txt")
    _git(graph.seed, "add", ".gitignore", "cli.py")
    _git(graph.seed, "-c", "user.name=Hermes Test", "-c", "user.email=hermes-test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-m", "pre-update state")
    base = _git_sha(graph.seed)
    _git(graph.seed, "rm", "-q", "cache/recover.txt")
    (graph.seed / "cli.py").write_text("def broken(:\n", encoding="utf-8")
    _git(graph.seed, "add", "-A")
    _git(graph.seed, "-c", "user.name=Hermes Test", "-c", "user.email=hermes-test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-m", "pinned bad syntax")
    broken = _git_sha(graph.seed)
    _git(graph.seed, "push", "-f", "origin", f"{graph.branch}:{graph.branch}")
    _git(graph.work, "fetch", "origin", graph.branch)
    _git(graph.work, "reset", "--hard", base)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", graph.work)

    def collision_seam(cmd, **kwargs):
        result = _REAL_RUN(cmd, **kwargs)
        if list(cmd)[:3] == ["git", "merge", "--ff-only"]:
            # A local process recreates the now-ignored path after the exact ff.
            (graph.work / "cache").mkdir(exist_ok=True)
            (graph.work / "cache" / "recover.txt").write_text(
                "unrelated ignored local notes\n", encoding="utf-8")
        return result

    monkeypatch.setattr(subprocess, "run", collision_seam)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(_update_args(graph.branch, broken))

    assert exc_info.value.code == 1
    assert _git_sha(graph.work) == broken  # pinned HEAD unmoved — no reset happened
    assert (
        (graph.work / "cache" / "recover.txt").read_text(encoding="utf-8")
        == "unrelated ignored local notes\n"
    )
    assert update_seams.installs == []
    out = capsys.readouterr().out
    assert "Refusing automatic rollback" in out
    assert broken in out and base in out
    latest = update_receipt.read_latest_receipt()
    assert latest is not None
    assert latest["outcome"] == "failed"
    assert latest["exit_code"] == 1


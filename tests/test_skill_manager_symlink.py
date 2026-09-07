"""Regression tests: skill_manage must resolve skills exposed via symlinks.

``pathlib.Path.rglob`` does not descend into directory symlinks, so skill
packages exposed through symlinks — the documented shared-library layout for
source-repo-owned skills — were invisible to ``_find_skill`` (and therefore
to every ``skill_manage`` action) while ``skill_view`` resolved them fine.
"""

import json
from pathlib import Path

import agent.skill_utils as skill_utils
import tools.skill_manager_tool as smt


def _make_package(root: Path, name: str) -> Path:
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\n# {name}\n")
    return pkg


def _linked_library(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Real package behind a symlink in a shared library -> (repo_pkg, shared)."""
    repo_pkg = _make_package(tmp_path / "repo", name)
    shared = tmp_path / "shared"
    shared.mkdir(exist_ok=True)
    (shared / name).symlink_to(repo_pkg)
    return repo_pkg, shared


def test_find_skill_follows_symlinked_package(tmp_path, monkeypatch):
    repo_pkg = _make_package(tmp_path / "repo", "repo-skill")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "repo-skill").symlink_to(repo_pkg)

    # Sanity: the old rglob mechanism really is blind to the link.
    assert [p for p in shared.rglob("SKILL.md") if p.parent.name == "repo-skill"] == []

    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [shared])
    found = smt._find_skill("repo-skill")
    assert found is not None
    assert found["path"].resolve() == repo_pkg.resolve()


def test_find_skill_symlink_loop_terminates(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    _make_package(shared, "pkg")
    (shared / "pkg" / "loop").symlink_to(shared)

    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [shared])
    # Must return promptly instead of recursing through the self-link.
    found = smt._find_skill("pkg")
    assert found is not None
    # Fully exhausting the iterator must also terminate: the self-link is pruned
    # by the realpath guard instead of re-walking the same directory forever.
    assert [p.name for p in list(smt._iter_skill_dirs(shared))] == ["pkg"]


def test_find_skill_dedupes_aliased_package(tmp_path):
    repo_pkg = _make_package(tmp_path / "repo", "aliased")
    shared = tmp_path / "shared"
    (shared / "cat-a").mkdir(parents=True)
    (shared / "cat-b").mkdir(parents=True)
    (shared / "cat-a" / "aliased").symlink_to(repo_pkg)
    (shared / "cat-b" / "aliased").symlink_to(repo_pkg)

    # _iter_skill_dirs yields skill DIRECTORIES; both aliases must collapse to
    # the single real package they point at.
    hits = [p for p in smt._iter_skill_dirs(shared) if p.name == "aliased"]
    assert len(hits) == 1
    assert hits[0].resolve() == repo_pkg.resolve()


def test_skill_manage_writes_through_symlinked_package(tmp_path, monkeypatch):
    """skill_manage edit/write lands in the REAL package behind the link."""
    repo_pkg, shared = _linked_library(tmp_path, "linked-skill")
    monkeypatch.setattr(smt, "SKILLS_DIR", shared)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [shared])

    patched = json.loads(smt.skill_manage(
        action="patch", name="linked-skill",
        old_string="# linked-skill", new_string="# linked-skill v2"))
    assert patched["success"] is True
    assert "# linked-skill v2" in (repo_pkg / "SKILL.md").read_text(encoding="utf-8")

    written = json.loads(smt.skill_manage(
        action="write_file", name="linked-skill",
        file_path="references/notes.md", file_content="shared notes\n"))
    assert written["success"] is True
    assert (repo_pkg / "references" / "notes.md").read_text(encoding="utf-8") == "shared notes\n"


def test_skill_manage_pinned_guard_holds_for_symlinked_package(tmp_path, monkeypatch):
    """The native pinned guard resolves the linked package too: deletion is
    refused while patch stays allowed (pin guards deletion only)."""
    from tools import skill_usage

    repo_pkg, shared = _linked_library(tmp_path, "pinned-linked")
    monkeypatch.setattr(smt, "SKILLS_DIR", shared)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [shared])
    monkeypatch.setattr(skill_usage, "get_record", lambda name: {"pinned": True})

    refused = json.loads(smt.skill_manage(action="delete", name="pinned-linked"))
    assert refused["success"] is False
    assert "pinned" in refused["error"].lower()
    assert (repo_pkg / "SKILL.md").exists()

    allowed = json.loads(smt.skill_manage(
        action="patch", name="pinned-linked",
        old_string="# pinned-linked", new_string="# pinned-linked updated"))
    assert allowed["success"] is True


def test_iter_skill_dirs_respects_exclusion_guards_with_followlinks(tmp_path, monkeypatch):
    """A package physically inside an excluded dir (dependency/cache layout) stays
    invisible — enabling followlinks must not widen the exclusion boundary."""
    shared = tmp_path / "shared"
    _make_package(shared / "node_modules", "vendored-skill")
    _make_package(shared, "visible-skill")

    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [shared])
    assert [p.name for p in smt._iter_skill_dirs(shared)] == ["visible-skill"]
    assert smt._find_skill("vendored-skill") is None

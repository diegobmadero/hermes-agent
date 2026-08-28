"""Regression tests: skill_manage must resolve skills exposed via symlinks.

``pathlib.Path.rglob`` does not descend into directory symlinks, so skill
packages exposed through symlinks — the documented shared-library layout for
source-repo-owned skills — were invisible to ``_find_skill`` (and therefore
to every ``skill_manage`` action) while ``skill_view`` resolved them fine.
"""

from pathlib import Path

import agent.skill_utils as skill_utils
import tools.skill_manager_tool as smt


def _make_package(root: Path, name: str) -> Path:
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(f"---\nname: {name}\n---\n# {name}\n")
    return pkg


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


def test_find_skill_dedupes_aliased_package(tmp_path):
    repo_pkg = _make_package(tmp_path / "repo", "aliased")
    shared = tmp_path / "shared"
    (shared / "cat-a").mkdir(parents=True)
    (shared / "cat-b").mkdir(parents=True)
    (shared / "cat-a" / "aliased").symlink_to(repo_pkg)
    (shared / "cat-b" / "aliased").symlink_to(repo_pkg)

    hits = [p for p in smt._iter_skill_md(shared) if p.parent.name == "aliased"]
    assert len(hits) == 1

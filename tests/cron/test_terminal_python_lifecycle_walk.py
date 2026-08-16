"""Terminal gateway walk must not treat .py CLIs as shell scripts.

Live 2026-08-16: Telegram `_HERMES_GATEWAY=1` blocked `~/bin/ask_agent.py`
because the referenced-script walk tokenized cache-dir string literals that
exist on disk and fail-closed on those directories. Cron already skips the
shell walk for .py (#77131); the terminal path did not.
"""

from pathlib import Path

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script


def test_python_cli_with_existing_cache_dir_literal_is_allowed(tmp_path: Path) -> None:
    cache = tmp_path / "jobs"
    cache.mkdir()
    script = tmp_path / "ask_agent.py"
    script.write_text(
        "from pathlib import Path\n"
        f'JOBS = Path("{cache}")\n'
        "print('ok')\n",
        encoding="utf-8",
    )
    assert (
        contains_gateway_lifecycle_command_or_referenced_script(
            str(script), cwd=str(tmp_path)
        )
        is False
    )


def test_python_cli_with_literal_lifecycle_command_is_still_blocked(
    tmp_path: Path,
) -> None:
    script = tmp_path / "evil.py"
    script.write_text(
        'import os\nos.system("hermes gateway restart")\n',
        encoding="utf-8",
    )
    assert (
        contains_gateway_lifecycle_command_or_referenced_script(
            str(script), cwd=str(tmp_path)
        )
        is True
    )

from __future__ import annotations

from pathlib import Path

from mackerel_road.experiments.remote import (
    REMOTE_ENVIRONMENT,
    build_remote_command,
    copy_safe_project,
    redact_destination,
)


def test_remote_destination_is_redacted() -> None:
    host = "user@department-machine.example"
    line = f"Syncing code to {host}:/tmp/job"

    assert host not in redact_destination(line, host)
    assert "[REMOTE_HOST]" in redact_destination(line, host)


def test_safe_stage_excludes_sensitive_and_generated_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='example'\n")
    (source / ".env").write_text("REMOTE_HOST=secret\n")
    (source / ".env.gemini").write_text("API_KEY=secret\n")
    (source / "tmp").mkdir()
    (source / "tmp" / "reference.txt").write_text("large reference")
    (source / "artifacts").mkdir()
    (source / "artifacts" / "run.log").write_text("generated")
    (source / "src").mkdir()
    (source / "src" / "module.py").write_text("VALUE = 1\n")

    copy_safe_project(source, destination)

    assert (destination / "pyproject.toml").is_file()
    assert (destination / "src" / "module.py").is_file()
    assert not (destination / ".env").exists()
    assert not (destination / ".env.gemini").exists()
    assert not (destination / "tmp").exists()
    assert not (destination / "artifacts").exists()


def test_remote_command_reuses_frozen_offline_environment() -> None:
    command = build_remote_command("/tmp/project", "-m package.module --device cuda")

    assert "uv sync --frozen --offline --extra cuda --no-install-project" in command
    assert f"UV_PROJECT_ENVIRONMENT={REMOTE_ENVIRONMENT}" in command
    assert f"{REMOTE_ENVIRONMENT}/bin/python -m package.module --device cuda" in command
    assert "trap 'rm -rf /tmp/project' EXIT" in command


def test_remote_command_quotes_python_arguments() -> None:
    command = build_remote_command(
        "/tmp/project", "-m package.module --name 'value; touch /tmp/unwanted'"
    )

    assert "--name 'value; touch /tmp/unwanted'" in command
    assert "--name value; touch" not in command

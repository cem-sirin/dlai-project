"""Run safely staged source in the persistent offline remote GPU environment."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

EXCLUDED = frozenset(
    {
        ".env",
        ".env.*",
        ".git",
        ".venv",
        "__pycache__",
        "artifacts",
        "checkpoints",
        "outputs",
        "runs",
        "tmp",
    }
)
REMOTE_ENVIRONMENT = "$HOME/.cache/dlai-s2-2026/venv"


def copy_safe_project(source: Path, destination: Path) -> None:
    """Copy source while excluding credentials, references, and generated state."""
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*EXCLUDED),
    )


def redact_destination(text: str, remote_host: str) -> str:
    return text.replace(remote_host, "[REMOTE_HOST]")


def build_remote_command(remote_path: str, command: str) -> str:
    python_arguments = " ".join(shlex.quote(part) for part in shlex.split(command))
    if not python_arguments:
        raise ValueError("remote Python arguments must not be empty")
    return (
        "set -e; "
        f"trap 'rm -rf {remote_path}' EXIT; "
        "mkdir -p $HOME/.cache/dlai-s2-2026; "
        f"cd {remote_path}; "
        f"UV_PROJECT_ENVIRONMENT={REMOTE_ENVIRONMENT} "
        "uv sync --frozen --offline --extra cuda --no-install-project; "
        f"PYTHONPATH={remote_path}/src "
        f"{REMOTE_ENVIRONMENT}/bin/python {python_arguments}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", help="Python arguments passed to the remote runner")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    load_dotenv(root / ".env")
    remote_host = os.environ.get("REMOTE_HOST")
    if not remote_host:
        raise SystemExit("REMOTE_HOST is not configured in the local environment or .env")
    with tempfile.TemporaryDirectory(prefix="mackerel-road-stage-") as temporary:
        stage = Path(temporary) / "project"
        copy_safe_project(root, stage)
        remote_path = f"/tmp/mackerel-road-{uuid.uuid4().hex[:8]}"
        print(f"Syncing safe stage to {redact_destination(remote_host, remote_host)}", flush=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                "ssh",
                f"{stage}/",
                f"{remote_host}:{remote_path}/",
            ],
            check=True,
        )
        remote_command = build_remote_command(remote_path, args.command)
        process = subprocess.Popen(
            ["ssh", remote_host, remote_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(redact_destination(line, remote_host), end="", flush=True)
        return_code = process.wait()
        if return_code != 0:
            raise SystemExit(return_code)


if __name__ == "__main__":
    main()

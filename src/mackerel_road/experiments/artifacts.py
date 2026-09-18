"""Extract a compressed optimization result from a sanitized remote log."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from typing import Any

RESULT_PREFIX = "MACKEREL_RESULT_JSON="
CHECKPOINT_PREFIX = "MACKEREL_CHECKPOINT_JSON="


def encode_payload(payload: dict[str, Any]) -> str:
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode("utf-8"), level=9))
    return encoded.decode("ascii")


def _decode_payload(encoded: str) -> dict[str, Any]:
    decoded = zlib.decompress(base64.b64decode(encoded)).decode("utf-8")
    result = json.loads(decoded)
    if not isinstance(result, dict) or "protocol" not in result or "runs" not in result:
        raise ValueError("result payload does not match the optimization contract")
    return result


def extract_payload(log_text: str) -> dict[str, Any]:
    final_payloads = [
        line.removeprefix(RESULT_PREFIX)
        for line in log_text.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(final_payloads) > 1:
        raise ValueError(f"expected at most one final result payload, found {len(final_payloads)}")
    if final_payloads:
        return _decode_payload(final_payloads[0])

    checkpoints = [
        line.removeprefix(CHECKPOINT_PREFIX)
        for line in log_text.splitlines()
        if line.startswith(CHECKPOINT_PREFIX)
    ]
    if not checkpoints:
        raise ValueError("expected a final result or checkpoint payload, found 0")
    return _decode_payload(checkpoints[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = extract_payload(args.log.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(payload['runs'])} runs")


if __name__ == "__main__":
    main()

"""Talk to the C command line tool (c/build/frbpe_cli) from Python.

Shared by the differential tests and scripts/diff_c_python.py. Records are
sent as 4-byte little-endian length + UTF-8 bytes, one output line comes back
per record.
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CLI = ROOT / "c" / "build" / ("frbpe_cli.exe" if os.name == "nt" else "frbpe_cli")


def cli_path() -> Path:
    return Path(os.environ.get("FRBPE_CLI", _DEFAULT_CLI))


def _run(args: list[str], texts: list[str]) -> list[str]:
    payload = b"".join(
        struct.pack("<I", len(data)) + data for data in (t.encode("utf-8") for t in texts)
    )
    result = subprocess.run(
        [str(cli_path()), *args], input=payload, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"frbpe_cli failed ({result.returncode}): {result.stderr.decode()}")
    lines = result.stdout.decode("ascii").split("\n")
    if lines[-1] != "":
        raise RuntimeError("frbpe_cli output does not end with a newline")
    lines = lines[:-1]
    if len(lines) != len(texts):
        raise RuntimeError(f"expected {len(texts)} lines from frbpe_cli, got {len(lines)}")
    return lines


def _parse(line: str) -> list[int]:
    if line.startswith("ERROR"):
        raise RuntimeError(f"frbpe_cli reported {line}")
    return [int(x) for x in line.split()] if line else []


def c_encode(model: Path, texts: list[str]) -> list[list[int]]:
    return [_parse(line) for line in _run(["encode", str(model)], texts)]


def c_split_ends(pattern: str, texts: list[str]) -> list[list[int]]:
    """Byte offset where each chunk ends, as computed by the C pre-tokenizer."""
    return [_parse(line) for line in _run(["split", pattern], texts)]


def python_split_ends(chunks: list[str]) -> list[int]:
    ends, pos = [], 0
    for chunk in chunks:
        pos += len(chunk.encode("utf-8"))
        ends.append(pos)
    return ends

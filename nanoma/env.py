"""Local environment file loading for NanoMA."""

from __future__ import annotations

import os
from pathlib import Path


_LOADED_DEFAULT = False


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load simple KEY=VALUE pairs from a .env file.

    Existing environment variables win by default. This intentionally supports
    only the common .env subset NanoMA needs: blank lines, comments, optional
    ``export``, and single/double quoted values.
    """
    global _LOADED_DEFAULT

    env_path = Path(path) if path is not None else _find_dotenv(Path.cwd())
    if path is None and _LOADED_DEFAULT:
        return env_path
    if env_path is None or not env_path.is_file():
        if path is None:
            _LOADED_DEFAULT = True
        return None

    for raw_line in env_path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            continue
        if override or key not in os.environ:
            os.environ[key] = _parse_env_value(value.strip())

    if path is None:
        _LOADED_DEFAULT = True
    return env_path


def _find_dotenv(start: Path) -> Path | None:
    current = start.resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _parse_env_value(value: str) -> str:
    quote = value[0] if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'} else ""
    if quote:
        value = value[1:-1]
        if quote == '"':
            value = bytes(value, "utf-8").decode("unicode_escape")
    return value

"""Path helpers for structured workspace tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import ToolContext


def _allowed_workspace_roots(ctx: "ToolContext") -> tuple[Path, ...]:
    roots = [ctx.workspace_root, *getattr(ctx, "workspace_extra_roots", ())]
    return tuple(root.resolve() for root in roots)


def resolve_workspace_path(path_text: str, workspace: Path, ctx: "ToolContext") -> Path:
    """Resolve a tool path, including the shared workspace alias.

    Shell commands receive SHARED in their environment. Structured workspace
    tools do not go through a shell, so they need the same alias expansion here.
    """
    raw = str(path_text)
    if raw == "$SHARED":
        path = ctx.shared_dir
    elif raw.startswith("$SHARED/"):
        path = ctx.shared_dir / raw[len("$SHARED/"):]
    elif raw == "${SHARED}":
        path = ctx.shared_dir
    elif raw.startswith("${SHARED}/"):
        path = ctx.shared_dir / raw[len("${SHARED}/"):]
    else:
        path = Path(raw)
        if not path.is_absolute():
            path = workspace / path

    resolved = path.resolve()
    for root in _allowed_workspace_roots(ctx):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError("path is outside allowed workspace roots")

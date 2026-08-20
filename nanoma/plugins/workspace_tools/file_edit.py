"""Workspace Tools Plugin - File Editing Module.

Implements the following tools:
- replace_string_in_file: 4-tier exact find and replace
- multi_replace_string_in_file: Atomic batch find and replace
- apply_patch: V4A diff/patch format application
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.plugins.workspace_tools.path_utils import resolve_workspace_path

if TYPE_CHECKING:
    from nanoma.core import ToolContext


# ─── Core matching logic ─────────────────────────────────────────────────────

def count_occurrences(text: str, search: str) -> int:
    """Count non-overlapping occurrences of search in text."""
    if not search:
        return 0
    count = 0
    pos = 0
    while True:
        pos = text.find(search, pos)
        if pos == -1:
            break
        count += 1
        pos += len(search)
    return count


def find_and_replace(
    content: str, old_string: str, new_string: str
) -> dict[str, Any]:
    """4-tier find and replace logic.

    Tier 1: Exact match
    Tier 2: Trimmed match (ignore leading/trailing whitespace)
    Tier 3: Indent-agnostic match (strip leading whitespace per line)
    Tier 4: Normalized whitespace match (collapse all whitespace)

    Returns:
        {"success": True, "new_content": str, "tier": int} on success
        {"success": False, "error": str} on failure
    """
    # Tier 1: Exact match
    exact_count = count_occurrences(content, old_string)
    if exact_count == 1:
        return {
            "success": True,
            "new_content": content.replace(old_string, new_string, 1),
            "tier": 1,
        }
    if exact_count > 1:
        return {
            "success": False,
            "error": f"Ambiguous match: oldString appears {exact_count} times in the file. It must appear exactly once.",
        }

    # Tier 2: Trimmed match
    trimmed_old = old_string.strip()
    if trimmed_old:
        trimmed_count = count_occurrences(content, trimmed_old)
        if trimmed_count == 1:
            return {
                "success": True,
                "new_content": content.replace(trimmed_old, new_string, 1),
                "tier": 2,
            }
        if trimmed_count > 1:
            return {
                "success": False,
                "error": f"Ambiguous match (tier 2, trimmed): found {trimmed_count} occurrences.",
            }

    # Tier 3: Indent-agnostic match
    tier3_result = _try_indent_agnostic_match(content, old_string, new_string)
    if tier3_result is not None:
        if tier3_result.get("ambiguous"):
            return {
                "success": False,
                "error": "Ambiguous match (tier 3, indent-agnostic): multiple matches found.",
            }
        return {"success": True, "new_content": tier3_result["new_content"], "tier": 3}

    # Tier 4: Normalized whitespace match
    tier4_result = _try_normalized_match(content, old_string, new_string)
    if tier4_result is not None:
        if tier4_result.get("ambiguous"):
            return {
                "success": False,
                "error": "Ambiguous match (tier 4, normalized whitespace): multiple matches found.",
            }
        return {"success": True, "new_content": tier4_result["new_content"], "tier": 4}

    return {
        "success": False,
        "error": "oldString not found in the file (tried all 4 matching tiers: exact, trimmed, indent-agnostic, normalized).",
    }


def _try_indent_agnostic_match(
    content: str, old_string: str, new_string: str
) -> dict[str, Any] | None:
    """Tier 3: Match ignoring leading whitespace on each line."""
    old_lines = [line.lstrip() for line in old_string.split("\n")]
    content_lines = content.split("\n")

    matches: list[int] = []
    for i in range(len(content_lines) - len(old_lines) + 1):
        match = True
        for j in range(len(old_lines)):
            if content_lines[i + j].lstrip() != old_lines[j]:
                match = False
                break
        if match:
            matches.append(i)

    if not matches:
        return None
    if len(matches) > 1:
        return {"new_content": "", "ambiguous": True}

    # Replace matched lines
    match_start = matches[0]
    new_lines = list(content_lines)
    new_lines[match_start:match_start + len(old_lines)] = new_string.split("\n")
    return {"new_content": "\n".join(new_lines)}


def _try_normalized_match(
    content: str, old_string: str, new_string: str
) -> dict[str, Any] | None:
    """Tier 4: Match after normalizing all whitespace."""

    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    normalized_old = normalize(old_string)
    content_lines = content.split("\n")
    old_line_count = len(old_string.split("\n"))

    matches: list[tuple[int, int]] = []

    for i in range(len(content_lines) - old_line_count + 1):
        # Try windows of different sizes (allow extra empty lines)
        for length in range(old_line_count, min(old_line_count + 3, len(content_lines) - i + 1)):
            window = "\n".join(content_lines[i:i + length])
            if normalize(window) == normalized_old:
                matches.append((i, i + length))
                break

    if not matches:
        return None
    if len(matches) > 1:
        return {"new_content": "", "ambiguous": True}

    start, end = matches[0]
    new_lines = list(content_lines)
    new_lines[start:end] = new_string.split("\n")
    return {"new_content": "\n".join(new_lines)}


# ─── V4A Patch parsing ───────────────────────────────────────────────────────

@dataclass
class PatchHunk:
    context_before: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    added_lines: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


@dataclass
class PatchOperation:
    action: str  # "add", "update", "delete"
    file_path: str
    new_content: str | None = None
    hunks: list[PatchHunk] = field(default_factory=list)


def parse_patch(patch_text: str) -> list[PatchOperation]:
    """Parse V4A patch format into operations.

    Format:
        *** Begin Patch
        *** Add File: path
        +content lines
        *** Update File: path
         context line (space prefix)
        -removed line
        +added line
         context line
        *** Delete File: path
        *** End Patch

    Hunks within an Update are separated by context lines. A new hunk starts
    when we see a context line AFTER having processed removed/added lines,
    and that context line is followed by more -/+ lines (indicating it's the
    before-context of the next hunk).
    """
    operations: list[PatchOperation] = []
    lines = patch_text.split("\n")
    i = 0

    # Skip until "*** Begin Patch"
    while i < len(lines) and not lines[i].startswith("*** Begin Patch"):
        i += 1
    i += 1  # Skip the "*** Begin Patch" line

    while i < len(lines):
        line = lines[i]

        if line.startswith("*** End Patch"):
            break

        if line.startswith("*** Add File:"):
            file_path = line.replace("*** Add File:", "").strip()
            i += 1
            content_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("***"):
                l = lines[i]
                content_lines.append(l[1:] if l.startswith("+") else l)
                i += 1
            operations.append(PatchOperation(
                action="add", file_path=file_path,
                new_content="\n".join(content_lines),
            ))

        elif line.startswith("*** Delete File:"):
            file_path = line.replace("*** Delete File:", "").strip()
            operations.append(PatchOperation(action="delete", file_path=file_path))
            i += 1

        elif line.startswith("*** Update File:"):
            file_path = line.replace("*** Update File:", "").strip()
            i += 1
            hunks = _parse_update_hunks(lines, i)
            # Advance past all hunk lines
            while i < len(lines) and not lines[i].startswith("***"):
                i += 1
            operations.append(PatchOperation(
                action="update", file_path=file_path, hunks=hunks,
            ))
        else:
            i += 1

    return operations


def _parse_update_hunks(lines: list[str], start: int) -> list[PatchHunk]:
    """Parse the hunk section of an Update File operation.

    Strategy: collect all lines until the next *** marker, then split into hunks.
    A hunk boundary is detected when we transition from +/- lines back to context
    lines, and then back to +/- lines again.
    """
    # Collect all lines belonging to this update section
    section_lines: list[str] = []
    i = start
    while i < len(lines) and not lines[i].startswith("***"):
        section_lines.append(lines[i])
        i += 1

    if not section_lines:
        return []

    # Parse into hunks: split on transitions from changes back to context-then-changes
    hunks: list[PatchHunk] = []
    current_hunk = PatchHunk()
    phase = "context_before"  # context_before -> changes -> context_after

    for line in section_lines:
        if line.startswith("-"):
            if phase == "context_after":
                # We were in after-context but hit more changes → start new hunk
                # The context_after of the previous hunk becomes context_before of this one
                prev_after = current_hunk.context_after[:]
                current_hunk.context_after = []
                hunks.append(current_hunk)
                current_hunk = PatchHunk()
                current_hunk.context_before = prev_after
            phase = "changes"
            current_hunk.removed_lines.append(line[1:])

        elif line.startswith("+"):
            if phase == "context_after":
                # Same as above: transition from after-context to new changes
                prev_after = current_hunk.context_after[:]
                current_hunk.context_after = []
                hunks.append(current_hunk)
                current_hunk = PatchHunk()
                current_hunk.context_before = prev_after
            phase = "changes"
            current_hunk.added_lines.append(line[1:])

        else:
            # Context line (space-prefixed or plain)
            ctx_line = line[1:] if line.startswith(" ") else line
            if phase == "changes":
                # First context line after changes → enter context_after
                phase = "context_after"
                current_hunk.context_after.append(ctx_line)
            elif phase == "context_after":
                current_hunk.context_after.append(ctx_line)
            else:
                # Still in context_before
                current_hunk.context_before.append(ctx_line)

    # Append the final hunk if it has any changes
    if current_hunk.removed_lines or current_hunk.added_lines:
        hunks.append(current_hunk)

    return hunks


def apply_hunks(content: str, hunks: list[PatchHunk]) -> str | None:
    """Apply hunks to file content using context matching.
    Returns new content or None if context doesn't match.
    """
    file_lines = content.split("\n")

    # Apply from last to first to avoid line offset issues
    for hunk in reversed(hunks):
        match_index = _find_context_match(file_lines, hunk)
        if match_index == -1:
            return None

        start_line = match_index + len(hunk.context_before)
        remove_count = len(hunk.removed_lines)

        file_lines[start_line:start_line + remove_count] = hunk.added_lines

    return "\n".join(file_lines)


def _find_context_match(lines: list[str], hunk: PatchHunk) -> int:
    """Find where a hunk's context matches in the file."""
    pattern = hunk.context_before + hunk.removed_lines

    if not pattern:
        return 0

    for i in range(len(lines) - len(pattern) + 1):
        match = True
        for j in range(len(pattern)):
            if lines[i + j].strip() != pattern[j].strip():
                match = False
                break
        if match:
            return i

    return -1


# ─── Tool Implementations ────────────────────────────────────────────────────

async def tool_replace_string(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Find and replace a string in a file (4-tier matching).

    Args:
        path: File path
        old_string: Exact text to find (must appear exactly once)
        new_string: Replacement text
    """
    file_path = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

    if not file_path or not file_path.strip():
        return {"error": "path is required"}
    if not old_string:
        return {"error": "old_string is required and must not be empty"}
    if new_string is None:
        return {"error": "new_string is required"}

    try:
        path = resolve_workspace_path(file_path, workspace, ctx)
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}

    content = path.read_text(encoding="utf-8")
    result = find_and_replace(content, old_string, new_string)

    if not result["success"]:
        return {"error": result["error"]}

    path.write_text(result["new_content"], encoding="utf-8")
    return {"match_tier": result["tier"]}


async def tool_multi_replace_string(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Execute multiple find-replace operations atomically.

    Args:
        path: File path
        replacements: List of {"old_string": str, "new_string": str}
    """
    file_path = args.get("path", "")
    replacements = args.get("replacements", [])

    if not file_path or not file_path.strip():
        return {"error": "path is required"}
    if not replacements:
        return {"error": "replacements array must not be empty"}

    try:
        path = resolve_workspace_path(file_path, workspace, ctx)
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}

    content = path.read_text(encoding="utf-8")

    # Apply each replacement sequentially (in-memory)
    for i, repl in enumerate(replacements):
        old_str = repl.get("old_string", "")
        new_str = repl.get("new_string", "")

        if not old_str:
            return {"error": f"Replacement #{i + 1}: old_string must not be empty"}

        result = find_and_replace(content, old_str, new_str)
        if not result["success"]:
            return {"error": f"Replacement #{i + 1} failed: {result['error']}. File NOT modified (all-or-nothing)."}

        content = result["new_content"]

    # All succeeded — write to file
    path.write_text(content, encoding="utf-8")
    return {"replaced_count": len(replacements)}


async def tool_apply_patch(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Apply a V4A format patch.

    Args:
        input: V4A patch content (must start with "*** Begin Patch")
        explanation: Brief description of the patch
    """
    patch_input = args.get("input", "")
    explanation = args.get("explanation", "")

    if not patch_input or not patch_input.strip():
        return {"error": "input (patch content) is required"}
    if not explanation or not explanation.strip():
        return {"error": "explanation is required"}

    if "*** Begin Patch" not in patch_input:
        return {"error": 'Patch must contain "*** Begin Patch"'}
    if "*** End Patch" not in patch_input:
        return {"error": 'Patch must contain "*** End Patch"'}

    operations = parse_patch(patch_input)
    if not operations:
        return {"error": "No file operations found in patch"}

    modified_files: list[str] = []

    for op in operations:
        # Resolve path relative to workspace
        try:
            abs_path = resolve_workspace_path(op.file_path, workspace, ctx)
        except ValueError:
            return {"error": f"Access denied: patch path '{op.file_path}' is outside workspace root"}

        if op.action == "add":
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(op.new_content or "", encoding="utf-8")
            modified_files.append(op.file_path)

        elif op.action == "delete":
            if abs_path.exists():
                abs_path.unlink()
                modified_files.append(op.file_path)

        elif op.action == "update":
            if not abs_path.exists():
                return {"error": f"Cannot update non-existent file: {op.file_path}"}
            content = abs_path.read_text(encoding="utf-8")
            new_content = apply_hunks(content, op.hunks)
            if new_content is None:
                return {"error": f"Failed to apply patch to: {op.file_path} (context mismatch)"}
            abs_path.write_text(new_content, encoding="utf-8")
            modified_files.append(op.file_path)

    return {"files_modified": modified_files, "count": len(modified_files)}

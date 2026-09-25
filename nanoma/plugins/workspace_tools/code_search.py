"""Workspace Tools Plugin - Code Search Module.

Implements the following tools:
- grep_search: Fast text/regex search across files (subprocess with Python fallback)
- code_outline: Get file symbol structure
- read_symbol: Read a specific symbol's source code
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.plugins.workspace_tools.path_utils import resolve_workspace_path

if TYPE_CHECKING:
    from nanoma.core import ToolContext


# ─── Helpers ─────────────────────────────────────────────────────────────────

BINARY_EXTENSIONS = frozenset([
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".wav", ".avi",
    ".sqlite", ".db", ".pyc", ".pyo",
])

IGNORE_DIRS = frozenset([
    "node_modules", ".git", "dist", "target", "__pycache__",
    "coverage", ".venv", "venv", ".tox", ".mypy_cache",
])

SOURCE_EXTENSIONS = frozenset([
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs",
    ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt",
])


def escape_regex(s: str) -> str:
    """Escape regex special characters."""
    return re.escape(s)


def is_binary_file(file_path: str) -> bool:
    """Check if a file is likely binary based on extension."""
    return Path(file_path).suffix.lower() in BINARY_EXTENSIONS


def find_block_end(lines: list[str], start_line: int) -> int:
    """Find the end of a code block.

    Strategy:
    1. Try brace-based detection (JS/TS/C/Rust/Java) — skips braces inside strings
    2. Fall back to Python-style indentation detection
    3. Handle single-line declarations (const x = ...; or =>)
    """
    first_line = lines[start_line]

    # Try brace-based detection (skip braces inside string literals)
    brace_count = 0
    started = False
    in_triple_quote = False
    triple_char = None

    for i in range(start_line, len(lines)):
        line = lines[i]

        # Handle triple-quoted strings (Python docstrings, JS template literals)
        if in_triple_quote:
            if triple_char * 3 in line:
                in_triple_quote = False
            continue  # skip entire line while inside triple-quote

        # Check if this line starts/contains a triple quote
        for tq in ('"""', "'''"):
            count = line.count(tq)
            if count == 1:
                # Opens a triple-quote that continues to another line
                in_triple_quote = True
                triple_char = tq[0]
                break  # skip rest of this line for brace counting
            # count >= 2 means opens and closes on same line — treat as normal

        if in_triple_quote:
            continue

        in_string = False
        string_char = None
        j = 0
        while j < len(line):
            ch = line[j]
            if in_string:
                if ch == '\\' and j + 1 < len(line):
                    j += 2  # skip escaped character
                    continue
                if ch == string_char:
                    in_string = False
            elif ch in ('"', "'", '`'):
                # Check for triple-quote on same line (already handled above)
                in_string = True
                string_char = ch
            elif ch == '/':
                # Skip // line comments
                if j + 1 < len(line) and line[j + 1] == '/':
                    break  # rest of line is comment
                # Skip # comments (Python/shell)
            elif ch == '#':
                break  # rest of line is comment (Python/shell)
            elif ch == '{':
                brace_count += 1
                started = True
            elif ch == '}':
                brace_count -= 1
                if started and brace_count == 0:
                    return i
            j += 1

        # Single-line declaration (no braces at all on first line)
        if i == start_line and not started and (';' in line or '=>' in line):
            return i

    # Python-style: use indentation to find block end
    if start_line < len(lines):
        base_indent = len(first_line) - len(first_line.lstrip())

        for i in range(start_line + 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped:  # empty line — don't end on blank lines
                continue
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            # Block ends when we find a non-empty line at same or less indentation
            if current_indent <= base_indent:
                return i - 1

        # If we reach end of file, the block extends to the last non-empty line
        for i in range(len(lines) - 1, start_line, -1):
            if lines[i].strip():
                return i

    return start_line



def _glob_to_simple_regex(pattern: str) -> re.Pattern:
    """Convert a simple glob to regex for file filtering."""
    escaped = pattern
    escaped = re.sub(r'[.+^${}()|[\]\\]', r'\\\g<0>', escaped)
    escaped = escaped.replace("**", "§§")
    escaped = escaped.replace("*", "[^/]*")
    escaped = escaped.replace("§§", ".*")
    escaped = escaped.replace("?", "[^/]")
    return re.compile(escaped)


# ─── Symbol parsing ──────────────────────────────────────────────────────────

# Patterns for common languages
SYMBOL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # TypeScript / JavaScript
    (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)"), "function"),
    (re.compile(r"^(?:export\s+)?class\s+(\w+)"), "class"),
    (re.compile(r"^(?:export\s+)?interface\s+(\w+)"), "interface"),
    (re.compile(r"^(?:export\s+)?enum\s+(\w+)"), "enum"),
    (re.compile(r"^(?:export\s+)?type\s+(\w+)"), "interface"),
    (re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)"), "variable"),
    # Python
    (re.compile(r"^(?:async\s+)?def\s+(\w+)"), "function"),
    (re.compile(r"^class\s+(\w+)"), "class"),
    # Rust
    (re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)"), "function"),
    (re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)"), "struct"),
    (re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)"), "enum"),
    (re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)"), "trait"),
    (re.compile(r"^impl(?:<[^>]*>)?\s+(\w+)"), "impl"),
    # Go
    (re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)"), "function"),
    (re.compile(r"^type\s+(\w+)\s+struct"), "struct"),
    (re.compile(r"^type\s+(\w+)\s+interface"), "interface"),
]


def parse_file_symbols(content: str, file_path: str) -> list[dict[str, Any]]:
    """Parse all top-level symbols from a file."""
    lines = content.split("\n")
    symbols: list[dict[str, Any]] = []

    for i, line in enumerate(lines):
        trimmed = line.lstrip()
        for pattern, kind in SYMBOL_PATTERNS:
            match = pattern.match(trimmed)
            if match:
                end_line = find_block_end(lines, i)
                symbols.append({
                    "name": match.group(1),
                    "kind": kind,
                    "file_path": file_path,
                    "start_line": i + 1,
                    "end_line": end_line + 1,
                })
                break

    return symbols


def find_symbol_by_name(symbols: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Find a symbol by exact name."""
    for s in symbols:
        if s["name"] == name:
            return s
    return None


# ─── Tool Implementations ────────────────────────────────────────────────────

async def tool_grep_search(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Fast text/regex search across workspace files.

    Strategy: try ripgrep (rg) first for performance, fall back to GNU grep,
    then to pure Python as last resort. Subprocess grep is 10-100x faster
    on large workspaces (SIMD, mmap, multi-threading).

    Args:
        query: Search text or regex pattern
        is_regexp: Whether query is a regex
        case_sensitive: If true, search is case-sensitive (default: false)
        include_pattern: Glob to filter files (e.g., "*.py")
        max_results: Maximum results to return (default: 100)
    """
    import asyncio
    import shutil

    query = args.get("query", "")
    is_regexp = args.get("is_regexp", False)
    case_sensitive = args.get("case_sensitive", False)
    include_pattern = args.get("include_pattern")
    max_results = min(args.get("max_results", 100), ctx.grep_max_results or 200)
    search_path = args.get("path") or args.get("root") or ""

    if not query:
        return {"error": "query is required"}

    root = workspace
    if search_path:
        try:
            root = resolve_workspace_path(str(search_path), workspace, ctx)
        except ValueError:
            return {"error": "Access denied: path is outside workspace root"}
        if not root.exists():
            return {"error": f"Path not found: {root}"}

    # Try subprocess grep for performance
    result = await _try_subprocess_grep(
        query, root, is_regexp, case_sensitive, include_pattern, max_results
    )
    if result is not None:
        return result

    # Fallback: pure Python implementation
    return _python_grep(query, root, is_regexp, case_sensitive, include_pattern, max_results)


async def _try_subprocess_grep(
    query: str, workspace: Path, is_regexp: bool, case_sensitive: bool,
    include_pattern: str | None, max_results: int,
) -> dict[str, Any] | None:
    """Try to use rg or grep subprocess. Returns None if unavailable."""
    import asyncio
    import shutil

    # Build command: prefer ripgrep, fallback to GNU grep
    rg_path = shutil.which("rg")
    grep_path = shutil.which("grep")

    if rg_path:
        cmd = [rg_path, "--no-heading", "--line-number", "--color=never",
               f"--max-count={max_results}"]
        if not case_sensitive:
            cmd.append("--ignore-case")
        if not is_regexp:
            cmd.append("--fixed-strings")
        if include_pattern:
            cmd.extend(["--glob", include_pattern])
        cmd.append(query)
        cmd.append(str(workspace))
    elif grep_path:
        cmd = [grep_path, "-rn", "--color=never", f"--max-count={max_results}"]
        if not case_sensitive:
            cmd.append("-i")
        if not is_regexp:
            cmd.append("-F")
        if include_pattern:
            cmd.extend(["--include", include_pattern])
        cmd.append(query)
        cmd.append(str(workspace))
    else:
        return None  # No subprocess available, use Python fallback

    proc: asyncio.subprocess.Process | None = None
    communicate_task: asyncio.Task[tuple[bytes, bytes]] | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        stdout, stderr = await asyncio.wait_for(communicate_task, timeout=30)
        output = stdout.decode(errors="replace")
        # Exit code 2+ means error (invalid regex, permission denied, etc.)
        # Fall back to Python for proper error reporting
        if proc.returncode and proc.returncode >= 2:
            return None
    except asyncio.TimeoutError:
        if communicate_task is not None:
            communicate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await communicate_task
        await _terminate_subprocess_group(proc)
        return None  # Fallback to Python on timeout
    except OSError:
        await _terminate_subprocess_group(proc)
        return None  # Fallback to Python on error

    # Parse output: "filepath:line:content"
    results: list[dict[str, Any]] = []
    for raw_line in output.split("\n"):
        if not raw_line or len(results) >= max_results:
            break
        # Parse "file:line:content" format
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path_str, line_no_str, content = parts[0], parts[1], parts[2]
        try:
            line_no = int(line_no_str)
        except ValueError:
            continue
        # Make path relative to workspace
        try:
            rel_path = str(Path(file_path_str).relative_to(workspace))
        except ValueError:
            rel_path = file_path_str
        results.append({
            "file": rel_path,
            "line": line_no,
            "content": content,
        })

    return {"matches": results, "count": len(results), "truncated": len(results) >= max_results}


async def _terminate_subprocess_group(proc: Any) -> None:
    if proc is None or proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    wait_task = asyncio.create_task(proc.wait())
    done, _ = await asyncio.wait({wait_task}, timeout=2)
    if not done:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await wait_task


def _python_grep(
    query: str, workspace: Path, is_regexp: bool, case_sensitive: bool,
    include_pattern: str | None, max_results: int,
) -> dict[str, Any]:
    """Pure Python grep fallback (used when no system grep is available)."""
    # Build search regex
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regexp:
            search_regex = re.compile(query, flags)
        else:
            search_regex = re.compile(escape_regex(query), flags)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    # File filter
    file_filter = _glob_to_simple_regex(include_pattern) if include_pattern else None

    results: list[dict[str, Any]] = []

    def search_file(file_path: Path):
        if len(results) >= max_results:
            return

        rel_path = str(file_path.relative_to(workspace))

        # File filter check
        if file_filter and not file_filter.search(rel_path) and not file_filter.search(file_path.name):
            return

        if is_binary_file(str(file_path)):
            return

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.split("\n")):
                if len(results) >= max_results:
                    return
                m = search_regex.search(line)
                if m:
                    results.append({
                        "file": rel_path,
                        "line": i + 1,
                        "content": line,
                        "match_start": m.start(),
                        "match_end": m.end(),
                    })
        except (OSError, UnicodeDecodeError):
            pass

    def walk_dir(dir_path: Path):
        if len(results) >= max_results:
            return
        try:
            items = sorted(dir_path.iterdir())
        except PermissionError:
            return
        for item in items:
            if len(results) >= max_results:
                return
            if item.is_dir():
                if item.name not in IGNORE_DIRS and not item.name.startswith("."):
                    walk_dir(item)
            elif item.is_file():
                search_file(item)

    walk_dir(workspace)

    return {"matches": results, "count": len(results), "truncated": len(results) >= max_results}



async def tool_code_outline(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Get the symbol outline of a source file.

    Args:
        path: Path to the source file
    """
    file_path = args.get("path", "")

    if not file_path or not file_path.strip():
        return {"error": "path is required"}

    try:
        path = resolve_workspace_path(file_path, workspace, ctx)
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": f"Cannot read file (binary or encoding issue): {path}"}

    symbols = parse_file_symbols(content, str(path))

    return {"symbols": symbols, "count": len(symbols)}


async def tool_read_symbol(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Read a specific symbol's source code from a file.

    Args:
        path: Path to the source file
        symbol_name: Exact name of the symbol to read
    """
    file_path = args.get("path", "")
    symbol_name = args.get("symbol_name", "")

    if not file_path or not symbol_name:
        return {"error": "Both path and symbol_name are required"}

    try:
        path = resolve_workspace_path(file_path, workspace, ctx)
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": f"Cannot read file: {path}"}

    symbols = parse_file_symbols(content, str(path))
    target = find_symbol_by_name(symbols, symbol_name)

    if not target:
        available = ", ".join(s["name"] for s in symbols[:20])
        return {
            "error": f'Symbol "{symbol_name}" not found in {file_path}. Available: {available}',
        }

    lines = content.split("\n")
    symbol_lines = lines[target["start_line"] - 1:target["end_line"]]

    return {
        "content": "\n".join(symbol_lines),
        "start_line": target["start_line"],
        "end_line": target["end_line"],
        "kind": target["kind"],
    }

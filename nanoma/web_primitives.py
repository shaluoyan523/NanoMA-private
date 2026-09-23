"""web/shell 命令判定原语：把一条命令归类，并读出结果里的信号。

从 core.py 抽出的 5 个常量与 10 个函数（共 242 行）。全是纯函数与常量：输入
命令串或工具结果，输出分类、签名或布尔判定，不碰任何运行时状态。

放在中立模块是因为它们有两组使用者 —— core.py 的 _update_shell_activity /
_execute_tool 等控制流，以及 web_research_guards.py 的检索护栏族。让两边单向
导入本模块，比让护栏反向导入 core.py 更干净：后者会成环。

core.py 以原名重新导出，因此既有引用无需改动。
"""

import re
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urlparse


ShellCapability = Literal["web", "python", "fs", "process", "package", "system", "unknown"]


_WEB_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)


_WEB_LOW_SIGNAL_PATTERNS = (
    "access denied",
    "attention required",
    "blocked",
    "captcha",
    "cloudflare",
    "forbidden",
    "no results",
    "not found",
    "permission denied",
    "rate limit",
    "too many requests",
    "robot check",
    "temporarily unavailable",
    "traceback",
    "unrecognized parameters",
    "unrecognized value for parameter",
    "validation-failure",
    "wikimedia error",
)


_WEB_HARD_BLOCK_PATTERNS = (
    "checking your connection",
    "verify you are human",
    "robot check",
    "captcha",
    "cf-chl-",
)


_WEB_SEARCH_DOMAINS = (
    "bing.com",
    "duckduckgo.com",
    "google.com",
    "mojeek.com",
    "search.brave.com",
    "searx.",
    "yahoo.com",
    "yandex.",
)


def _first_effective_shell_command(command: str) -> str:
    """Return the first non-comment shell line for capability classification."""
    for line in command.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = re.sub(
            r"^(?:(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*="
            r"(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)\s*(?:(?:&&|;)\s*)?)+",
            "",
            stripped,
        ).lstrip()
        stripped = re.sub(
            r"^(?:cd\s+(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)\s*&&\s*)+",
            "",
            stripped,
        ).lstrip()
        return stripped
    return command.strip()


def classify_shell_capability(command: str) -> ShellCapability:
    """Classify a shell command for internal policy pruning.

    The public tool remains a single `shell` function. This classification is
    used only by the runtime to narrow what that tool may execute under
    constraint pressure.
    """
    cmd = _first_effective_shell_command(command)
    if not cmd:
        return "unknown"

    first = re.split(r"\s+", cmd, maxsplit=1)[0].split("/")[-1]
    lowered = command.lower()

    python_network_markers = (
        "http.client",
        "requests.",
        "urllib.",
        "socket.",
        "ssl.",
        "aiohttp",
        "httpx",
        "urlopen",
        "wrap_socket",
        "create_connection",
    )

    imports_network_module = bool(
        re.search(r"\b(?:import|from)\s+(?:aiohttp|http\.client|httpx|requests|socket|ssl|urllib)\b", lowered)
        or re.search(r"\bimport\s+[A-Za-z0-9_.,\s]*(?:socket|ssl|urllib|requests|httpx|aiohttp)\b", lowered)
    )

    if (
        first in {"curl", "wget"}
        or re.search(r"https?://", lowered)
        or any(marker in lowered for marker in python_network_markers)
        or imports_network_module
    ):
        return "web"
    if first in {"python", "python3", "python2"} or re.match(r"python\d?\s*<<", lowered):
        return "python"
    if first in {
        "cat", "cd", "cp", "du", "echo", "file", "find", "head", "ls", "mkdir",
        "grep", "mv", "pwd", "realpath", "rm", "rmdir", "sed", "sort", "stat",
        "tail", "tee", "touch", "tree", "uniq", "wc",
    }:
        return "fs"
    if first in {
        "c++", "cc", "clang", "clang++", "g++", "gcc", "go", "javac", "ps",
        "pkill", "kill", "killall", "jobs", "pgrep", "rustc", "sleep", "timeout",
    }:
        return "process"
    if first in {"apt", "apt-get", "brew", "conda", "npm", "npx", "pip", "pip3", "pnpm", "yarn"}:
        return "package"
    if first in {
        "bash", "chmod", "chown", "docker", "git", "make", "node", "perl", "ruby",
        "sh", "sudo", "tar", "unzip", "xz", "zip",
    }:
        return "system"
    return "unknown"


def _extract_web_urls(command: str) -> list[str]:
    urls: list[str] = []
    for match in _WEB_URL_RE.finditer(command):
        url = match.group(0).rstrip("),.;]")
        if url:
            urls.append(url)
    return urls


def _web_command_domain(command: str) -> str:
    urls = _extract_web_urls(command)
    if not urls:
        return ""
    try:
        return (urlparse(urls[0]).netloc or "").lower()
    except Exception:
        return ""


def _web_command_writes_download(command: str) -> bool:
    """Return true when a web command explicitly persists the response to a file."""
    value = str(command or "")
    return bool(
        re.search(r"(?:^|\s)(?:-o|--output)(?:\s+|=)[^\s;&|]+", value)
        or re.search(r"(?:^|\s)(?:-O|--output-document)(?:\s+|=)[^\s;&|]+", value)
    )


def _web_command_writes_html_download(command: str) -> bool:
    """Return true when the persisted response is an HTML page, not a research asset."""
    value = str(command or "")
    matches = re.findall(
        r"(?:^|\s)(?:-o|--output|-O|--output-document)(?:\s+|=)([^\s;&|]+)",
        value,
    )
    return any(
        str(path).strip("'\"").lower().split("?", 1)[0].endswith((".html", ".htm"))
        for path in matches
    )


def _web_command_signature(command: str) -> str:
    urls = _extract_web_urls(command)
    if not urls:
        return re.sub(r"\s+", " ", command.strip().lower())[:240]

    try:
        parsed = urlparse(urls[0])
    except Exception:
        return urls[0].lower()[:240]

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_text = (
        query.get("q")
        or query.get("query")
        or query.get("search")
        or query.get("srsearch")
        or query.get("title")
        or ""
    )
    if query_text:
        if re.search(r"[{}]|\$\(|\b(?:quote|quote_plus|urlencode)\s*\(", query_text):
            return ""
        query_part = unquote(query_text).lower()
    else:
        stable_params = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in {"api_key", "apikey", "key", "token", "access_token"}
        ][:6]
        query_part = "&".join(f"{k}={v}" for k, v in stable_params).lower()

    path = parsed.path.rstrip("/") or "/"
    return re.sub(
        r"\s+",
        " ",
        f"{(parsed.netloc or '').lower()}{path.lower()}?{query_part}",
    )[:240]


def _web_result_low_signal(result: Any) -> bool:
    if not isinstance(result, dict):
        return True

    try:
        exit_code = int(result.get("exit_code", 0))
    except Exception:
        exit_code = 0
    if exit_code != 0:
        return True

    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip()
    if len(text) < 80:
        return True

    lowered = text.lower()
    if any(pattern in lowered for pattern in _WEB_LOW_SIGNAL_PATTERNS):
        return True

    if re.search(r'"(?:totalhits|count|total|num_found)"\s*:\s*0\b', lowered):
        return True
    if re.search(r'"(?:items|results|search)"\s*:\s*\[\s*\]', lowered):
        return True

    return False


def _web_result_hard_block(result: Any) -> bool:
    """Detect explicit remote denial/challenge pages, not merely weak content."""
    if not isinstance(result, dict):
        return False
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip().lower()
    if not text:
        return False
    head = text[:12000]
    headings = " ".join(re.findall(
        r"<(?:title|h1)[^>]*>(.*?)</(?:title|h1)>",
        head,
        flags=re.DOTALL,
    ))
    headings = re.sub(r"<[^>]+>", " ", headings)
    headings = re.sub(r"\s+", " ", headings).strip()
    if any(pattern in headings for pattern in _WEB_HARD_BLOCK_PATTERNS):
        return True
    if (
        ("cf-chl-" in head or "challenge-platform" in head)
        and ("just a moment" in headings or "checking your connection" in head[:2000])
    ):
        return True
    status_pattern = r"(?:\b403\b.{0,80}\bforbidden\b|\b429\b.{0,80}\btoo many requests\b)"
    if re.search(status_pattern, headings, re.DOTALL):
        return True
    return len(text) <= 6000 and bool(re.search(status_pattern, head, re.DOTALL))


def _is_search_domain(domain: str) -> bool:
    domain = domain.lower()
    return any(marker in domain for marker in _WEB_SEARCH_DOMAINS)

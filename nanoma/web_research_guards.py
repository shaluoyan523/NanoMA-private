"""检索护栏：按 shell 命令的形态给出阻断理由或纠偏指引。

从 core.py 抽出的 25 个方法（607 行），以 mixin 形式由 Runtime 继承，
因此调用点无需改动 —— 方法仍通过 self 解析。

族内每个方法都是 (agent, command[, result]) -> str 的纯判断：命中某个已知的
检索误区就返回一段话，否则返回空串。它们编码的是 arxiv、参考文献列表、PDF、
图注这些具体来源的经验，属于领域知识而非运行时机制，不该长在 core.py 里。

抽出时实测的边界：

  入向  被 core.py 中 1 个方法调用：
        _execute_tool 引用 21 个
  出向  self.config 的两个字段（system_extra_instructions 与
        web_search_failover_after_low_signal），以及 web_primitives 的判定原语
  状态  不持有实例属性；只读 agent 的 _notified_thresholds、_shell_activity、parent、task

族里还有若干 *_block_reason 孪生方法仍留在 core.py —— 它们与调用点交织更
深，未随本次一起搬。最终归属应是可插拔的领域包：这一族护栏该随任务领域装
卸，而不是常驻通用运行时。本模块先把它从通用路径里分出来。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, unquote, urlparse

from nanoma.web_primitives import (
    _extract_web_urls,
    _is_search_domain,
    _web_command_domain,
    _web_command_signature,
    _web_command_writes_download,
    _web_command_writes_html_download,
    _web_result_hard_block,
    _web_result_low_signal,
    classify_shell_capability,
)

if TYPE_CHECKING:  # Agent 定义在 core.py，而 core.py 导入本模块
    from nanoma.core import Agent


class WebResearchGuardsMixin:
    """检索护栏。仅由 Runtime 继承，不单独实例化。"""

    def _web_search_failover_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or not _is_search_domain(_web_command_domain(command))
            or not _web_result_low_signal(result)
        ):
            return ""
        signature = _web_command_signature(command)
        key = f"web_search_failover_guidance:{signature or _web_command_domain(command)}"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The public HTML search returned no usable evidence. Do not repeat this exact query. "
            "Runtime will attach a structured fallback search when available. Follow its direct links, or use "
            "a direct site/document URL or ordinary public API. Use a materially different query and do not "
            "search benchmark datasets or answer dumps."
        )

    def _web_failure_fuse_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or not _web_result_hard_block(result)
        ):
            return ""
        signature = _web_command_signature(command)
        key = f"web_failure_fuse_guidance:{signature}"
        if not signature or key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This exact URL returned an explicit 403/429 denial or human-verification challenge and is now "
            "fused. Do not fetch it again or keep parsing its challenge page. Switch to another independent "
            "source; if the task asks you to execute or verify a named deterministic procedure, perform a local "
            "deterministic reproduction and report the remote-source limitation with the local evidence."
        )

    def _direct_web_no_gain_guidance(self, agent: Agent, command: str) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or classify_shell_capability(command) != "web"
            or (
                _web_command_writes_download(command)
                and not _web_command_writes_html_download(command)
            )
            or agent._shell_activity.consecutive_web_no_gain <= 0
        ):
            return ""
        domain = _web_command_domain(command)
        signature = _web_command_signature(command)
        if not domain or _is_search_domain(domain) or not signature:
            return ""
        if signature != agent._shell_activity.last_web_signature:
            return ""
        if signature in agent._shell_activity.hard_blocked_web_signatures:
            return ""
        key = f"direct_web_no_gain_guidance:{signature}"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This direct fetch produced no task-aligned evidence. Treat the mismatch as negative evidence and "
            "do not fetch or grep the same URL again. Revisit the strongest named candidate already retrieved, "
            "or change the source or assumption. For a versioned repository, inspect the record's version "
            "history instead of inferring the relevant date from its identifier."
        )

    @staticmethod
    def _is_broad_arxiv_historical_search(command: str) -> bool:
        for raw_url in _extract_web_urls(command):
            try:
                parsed = urlparse(raw_url)
                domain = (parsed.netloc or "").lower()
            except Exception:
                continue
            if not domain.endswith("arxiv.org") or not parsed.path.rstrip("/").endswith("/search"):
                continue
            return True
        return False

    @staticmethod
    def _requires_constrained_arxiv_search(agent: Agent) -> bool:
        task = str(agent.task or "").lower()
        return (
            "runtime inherited research protocol" in task
            and "arxiv" in task
            and ("advanced html" in task or "title field" in task)
        )

    def _arxiv_broad_search_guidance(self, agent: Agent, command: str) -> str:
        if (
            not self.config.web_search_failover_after_low_signal
            or not self._requires_constrained_arxiv_search(agent)
            or not self._is_broad_arxiv_historical_search(command)
        ):
            return ""
        key = "arxiv_broad_search_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This basic arXiv search page is sorted newest-first, so changing its query text or grep/head filters "
            "cannot reliably find a historical paper or verify its title and version date. Do not repeat another "
            "basic search-page variant. Use one arXiv advanced HTML query with separate author and title fields; "
            "the required parameter shape is advanced=&terms-0-term=<URLENCODED_AUTHOR>&terms-0-field=author&"
            "terms-1-term=<URLENCODED_TITLE>&terms-1-field=title. Parse the returned IDs structurally, then "
            "inspect candidate version metadata. If empty, switch to named bibliography candidates instead of "
            "changing the local grep expression."
        )

    def _arxiv_broad_search_block_reason(self, agent: Agent, command: str) -> str:
        if (
            "arxiv_broad_search_guidance" not in agent._notified_thresholds
            or not self._requires_constrained_arxiv_search(agent)
            or not self._is_broad_arxiv_historical_search(command)
        ):
            return ""
        return (
            "Repeated basic arXiv newest-first search blocked. Use advanced search with separate author and title "
            "fields, then verify candidate version dates; do not retry a basic search page with another query or grep."
        )

    @staticmethod
    def _is_malformed_arxiv_advanced_search(command: str) -> bool:
        command_field_pairs = {
            (index, field.lower())
            for index, field in re.findall(
                r"terms-(\d+)-field\s*=\s*(author|title)",
                command,
                flags=re.IGNORECASE,
            )
        }
        command_term_indexes = set(
            re.findall(r"terms-(\d+)-term\s*=", command, flags=re.IGNORECASE)
        )
        for raw_url in _extract_web_urls(command):
            try:
                parsed = urlparse(raw_url)
                domain = (parsed.netloc or "").lower()
                query_items = parse_qsl(parsed.query, keep_blank_values=True)
            except Exception:
                continue
            if (
                not domain.endswith("arxiv.org")
                or not parsed.path.rstrip("/").endswith("/search/advanced")
            ):
                continue
            fields = {
                str(value or "").lower()
                for key, value in query_items
                if str(key).lower().endswith("-field")
            }
            keys = {str(key or "").lower() for key, _ in query_items}
            # Python often builds this URL from adjacent f-string fragments, so
            # the URL extractor sees only the first literal ending at advanced=&.
            if keys == {"advanced"}:
                fields.update(
                    field
                    for index, field in command_field_pairs
                    if index in command_term_indexes
                )
            return (
                domain not in {"arxiv.org", "www.arxiv.org"}
                or "advanced" not in keys
                or not {"author", "title"}.issubset(fields)
            )
        return False

    def _arxiv_malformed_advanced_search_block_reason(self, agent: Agent, command: str) -> str:
        if (
            not self._requires_constrained_arxiv_search(agent)
            or not self._is_malformed_arxiv_advanced_search(command)
        ):
            return ""
        return (
            "Malformed arXiv advanced-search URL blocked. The advanced endpoint does not accept an opaque terms= "
            "expression, belongs on arxiv.org rather than export.arxiv.org, and only displays the form when the "
            "hidden advanced= parameter is absent. Use https://arxiv.org/search/advanced?advanced=&terms-0-term="
            "<URLENCODED_AUTHOR>&terms-0-field=author&terms-1-term="
            "<URLENCODED_TITLE>&terms-1-field=title, with separate operator parameters if needed; parse result "
            "IDs before checking version metadata."
        )

    def _arxiv_outdated_advanced_selector_block_reason(self, agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "runtime inherited research protocol" not in task
            or "/search/advanced" not in value
            or not re.search(
                r"select(?:_one)?\s*\(\s*[\"'][^\"']*\.list-result",
                value,
            )
        ):
            return ""
        return (
            "Outdated arXiv advanced-search selector blocked. Current result records use "
            "li.arxiv-result, with the identifier link under p.list-title a[href*='/abs/']; "
            ".list-result returns an empty list even when the downloaded HTML contains results."
        )

    def _arxiv_wrong_fulltext_html_source_block_reason(self, agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "runtime inherited research protocol" not in task
            or "ar5iv" not in task
            or not re.search(r"https?://(?:www\.)?arxiv\.org/html/\d{4}\.\d+", value)
        ):
            return ""
        return (
            "Wrong full-text HTML source blocked. This inherited protocol requires the converted full paper at "
            "https://ar5iv.labs.arxiv.org/html/<ARXIV_ID>; arxiv.org/html/<ARXIV_ID> can return a short fallback "
            "page without the bibliography or figures."
        )

    def _arxiv_saved_advanced_html_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        stdout = str(result.get("stdout") or "") if isinstance(result, dict) else ""
        if (
            "runtime inherited research protocol" not in task
            or "arxiv.org/search/advanced" not in value
            or not re.search(r"(?:^|\s)(?:-o|--output)\s+[^\s;&|]+", value)
            or not re.search(r"(?:^|\D)200(?:\D|$)", stdout)
        ):
            return ""
        key = "arxiv_saved_advanced_html_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The arXiv advanced HTML download succeeded with HTTP 200 and was saved locally. Do not classify "
            "this as rate limiting, retry the network request, or deliver an incomplete search report. Parse "
            "the saved file now with BeautifulSoup using li.arxiv-result and extract each identifier from "
            "p.list-title a[href*='/abs/']; then inspect candidate version histories."
        )

    def _arxiv_advanced_batch_timeout_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        stderr = str(result.get("stderr") or "") if isinstance(result, dict) else ""
        exit_code = result.get("exit_code") if isinstance(result, dict) else None
        if (
            "runtime inherited research protocol" not in task
            or "arxiv.org/search/advanced" not in value
            or not (exit_code == -1 or "timeout" in stderr.lower())
        ):
            return ""
        key = "arxiv_advanced_batch_timeout_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The batched arXiv request timed out, but earlier loop iterations may already have saved valid HTML. "
            "Do not discard them or rerun the full batch. First list the saved search HTML files and parse every "
            "nonempty file locally with li.arxiv-result; only then fetch missing authors, preferably concurrently "
            "with bounded per-request timeouts and flushed progress output."
        )

    def _arxiv_version_history_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        output = "\n".join(
            str(result.get(key) or "") for key in ("stdout", "stderr")
        ) if isinstance(result, dict) else str(result or "")
        if (
            "runtime inherited research protocol" not in task
            or "initial or revised arxiv version" not in task
            or "/search/advanced" not in value
            or not re.search(r"arxiv\.org/abs/\d{4}\.\d+", output, flags=re.IGNORECASE)
        ):
            return ""
        key = "arxiv_version_history_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The returned arXiv identifier prefix records only the initial submission month (v1), not every "
            "revision month. Do not prioritize or reject candidates by a YYMM-looking identifier. Inspect the "
            "submission/version history for every promising title-and-author candidate before applying the target "
            "month; candidates repeated under several independently searched authors or already named in the "
            "bibliography should be checked before a merely month-matching identifier."
        )

    @staticmethod
    def _is_linewise_bibliography_filter(agent: Agent, command: str) -> bool:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        return bool(
            "runtime inherited research protocol" in task
            and "bibliography items structurally" in task
            and ("split('\\n')" in value or 'split("\\n")' in value or ".splitlines(" in value)
            and re.search(r"\bfor\b[^\n]{0,100}\bline\b", value)
            and "2020" in value
            and any(marker in value for marker in ("frb", "fast radio", "180916", "burst"))
        )

    def _bibliography_line_filter_guidance(self, agent: Agent, command: str) -> str:
        if not self._is_linewise_bibliography_filter(agent, command):
            return ""
        key = "bibliography_line_filter_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "This filter applies year/topic predicates to individual PDF text lines, but numbered citations often "
            "wrap across several lines. It can silently drop the relevant record. Do not repeat a line-oriented "
            "filter. Segment the normalized bibliography by numbered citation boundaries, join each record's "
            "continuation lines, then apply all predicates to the complete record and resolve any et al. author list "
            "through source metadata."
        )

    def _bibliography_line_filter_block_reason(self, agent: Agent, command: str) -> str:
        if (
            "bibliography_line_filter_guidance" not in agent._notified_thresholds
            or not self._is_linewise_bibliography_filter(agent, command)
        ):
            return ""
        return (
            "Repeated line-oriented bibliography filter blocked. Merge wrapped lines into complete numbered "
            "citation records before filtering by year, topic, or author."
        )

    @staticmethod
    def _partial_bibliography_scope_block_reason(agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "parse all bibliography items structurally" not in task
            or "fitz" not in value
            or not any(marker in value for marker in ("bibliography", "reference"))
        ):
            return ""
        assumes_suffix = bool(
            "last few pages" in value
            or "likely have references" in value
            or re.search(r"range\(\s*len\([^)]*\)\s*-\s*\d+", value)
        )
        if not assumes_suffix:
            return ""
        return (
            "Partial-bibliography shortcut blocked. The protocol requires all bibliography items, so do not "
            "assume an arbitrary last-N-page suffix contains the complete references. Scan page text once to "
            "locate the References/Bibliography heading, include that page and every following page, segment "
            "complete numbered records, and verify that the first parsed citation number is near the start."
        )

    @staticmethod
    def _pdf_full_page_dump_block_reason(agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "print only candidate captions and axis labels, not entire page text" not in task
            or "fitz" not in value
            or "get_text(" not in value
        ):
            return ""
        text_variables = set(re.findall(
            r"\b([a-z_]\w*)\s*=\s*(?:[a-z_]\w*\.)?get_text\(",
            value,
        ))
        dumps_page = any(
            re.search(rf"\bprint\(\s*{re.escape(variable)}\s*\)", value)
            for variable in text_variables
        ) or bool(re.search(r"\bprint\(\s*(?:[a-z_]\w*\.)?get_text\(\)\s*\)", value))
        if not dumps_page:
            return ""
        return (
            "Full-page PDF text dump blocked. The inherited protocol requires only bounded candidate captions "
            "and axis labels, and a whole page adds noise without proving plotted endpoints. Extract a short "
            "caption/label window from the already matched page, then render that page or figure crop and inspect "
            "the plot-frame borders plus tick spacing."
        )

    @staticmethod
    def _arxiv_identifier_month_assumption_block_reason(agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            "runtime inherited research protocol" not in task
            or not (
                "arxiv" in task
                and "initial or revised" in task
                and "version" in task
            )
        ):
            return ""
        searches_identifier_prefix = bool(
            re.search(r"[\"']arxiv:20\d{2}[\"']\s+in\b", value)
            or re.search(r"\bgrep\b[^\n]{0,100}arxiv[:./]?20\d{2}", value)
            or re.search(r"\.\s*startswith\s*\(\s*[rubf]*[\"']20\d{2}", value)
            or re.search(
                r"\[\s*(?:0\s*)?:\s*4\s*\]\s*(?:==|!=)\s*[rubf]*[\"']20\d{2}",
                value,
            )
            or re.search(
                r"\bre\.(?:match|search|findall|finditer)\s*\(\s*[rubf]*[\"']"
                r"[^\"'\r\n]{0,40}20\d{2}",
                value,
            )
        )
        if not searches_identifier_prefix:
            return ""
        return (
            "arXiv identifier-month inference blocked. An identifier encodes the initial submission month, "
            "while the task explicitly allows a later revision date. Shortlist papers by title/authors/subject, "
            "then inspect every candidate's version history; do not search bibliography text for a YYMM prefix."
        )

    def _arxiv_exact_title_api_block_reason(self, agent: Agent, command: str) -> str:
        task = str(agent.task or "").lower()
        value = unquote(str(command or "")).lower()
        protocol_blocks_title_api = (
            "do not add exact 'ti:frb' api filters" in task
            or (
                "advanced html" in task
                and bool(re.search(
                    r"\bdo not\b[^.\r\n]{0,160}(?:exact-title|search_query|\bti\s*:)",
                    task,
                ))
            )
        )
        if (
            not self._requires_constrained_arxiv_search(agent)
            or not protocol_blocks_title_api
            or "export.arxiv.org/api/query" not in value
        ):
            return ""
        exact_title_query = False
        for match in re.finditer(
            r"export\.arxiv\.org/api/query[^\r\n]*",
            value,
        ):
            query_line = match.group(0)
            if "search_query=" not in query_line:
                continue
            if re.search(r"\bti\s*:", query_line):
                exact_title_query = True
                break
            variable_refs = re.findall(
                r"search_query=(?:\{([a-z_]\w*)\}|\$\{?([a-z_]\w*)\}?)",
                query_line,
            )
            for groups in variable_refs:
                variable = next((name for name in groups if name), "")
                if variable and re.search(
                    rf"\b{re.escape(variable)}\s*=\s*f?[\"'][^\r\n]*\bti\s*:",
                    value,
                ):
                    exact_title_query = True
                    break
            if exact_title_query:
                break
        if not exact_title_query:
            return ""
        return (
            "Exact-title arXiv Atom API query blocked by the inherited research protocol. Use arXiv advanced "
            "HTML with separate author and title fields, or inspect ordinary metadata for a named candidate; "
            "do not retry the API with a different ti: token."
        )

    def _arxiv_atom_feed_header_parse_block_reason(self, agent: Agent, command: str) -> str:
        value = str(command or "").lower()
        if (
            not self._requires_constrained_arxiv_search(agent)
            or "export.arxiv.org/api/query" not in value
            or "id_list" not in value
        ):
            return ""
        takes_first_title = "<title>" in value and bool(
            re.search(r"\b(?:titles?|title_matches)\s*\[\s*0\s*\]", value)
        )
        takes_first_updated = "<updated>" in value and bool(
            re.search(r"\b(?:updated|updates?|updated_matches)\s*\[\s*0\s*\]", value)
        )
        if not (takes_first_title or takes_first_updated):
            return ""
        return (
            "Atom feed-header parse blocked. Both the feed and its paper entry contain title/updated fields, so "
            "the first regex match is query metadata rather than paper metadata. Parse the XML and select the Atom "
            "entry element first, then read that entry's title, published, updated, and author/name children. Use "
            "the paper's submission history when every version date is required; never use the feed-level updated "
            "timestamp as a paper revision date."
        )

    @staticmethod
    def _pdf_cli_fallback_guidance(command: str, result: Any) -> str:
        if "pdftotext" not in str(command or "").lower():
            return ""
        if isinstance(result, dict):
            result_text = "\n".join(
                str(result.get(key) or "")
                for key in ("stdout", "stderr", "error")
            )
        else:
            result_text = str(result or "")
        if not re.search(
            r"(?:pdftotext[^\n]*(?:not found|no such file)|command not found)",
            result_text,
            flags=re.IGNORECASE,
        ):
            return ""
        return (
            "The pdftotext executable is unavailable. Do not install packages or retry that CLI. Use the "
            "already available Python PDF libraries immediately: run Python with `import fitz`, open the local "
            "PDF via `fitz.open(...)`, and iterate `page.get_text()`; use PyPDF2 only if importing fitz fails."
        )

    def _figure_panel_disambiguation_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        scope_text = f"{agent.task}\n{self.config.system_extra_instructions}"
        if (
            agent.parent is None
            or classify_shell_capability(command) != "python"
            or not re.search(r"\b(?:bottom|lower)\s+panels?\b", scope_text, flags=re.IGNORECASE)
            or not isinstance(result, dict)
        ):
            return ""
        output = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr"))
        page_mentions = set(re.findall(r"\bpage\s+(\d+)\b", output, flags=re.IGNORECASE))
        if (
            len(page_mentions) < 2
            or "figure" not in output.lower()
            or "panel" not in output.lower()
        ):
            return ""
        key = "figure_panel_disambiguation_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The PDF scan found several figure/page mentions for the target. Do not select the first figure that "
            "mentions the item. Extract every matching figure caption as a complete block, then choose only the "
            "caption that satisfies all wording qualifiers from the task, especially the named item and an explicit "
            "lower/bottom panel. Treat nearby prose and a caption describing several items as ambiguous until the "
            "exact panel label is verified; render that matched page only."
        )

    def _html_regex_parser_guidance(
        self,
        agent: Agent,
        command: str,
        result: Any,
    ) -> str:
        task = str(agent.task or "").lower()
        value = str(command or "").lower()
        if (
            agent.parent is None
            or classify_shell_capability(command) != "python"
            or "bibliography" not in task
            or "structurally" not in task
            or ".html" not in value
            or not re.search(r"\bre\.(?:findall|search|finditer)\b", value)
            or not isinstance(result, dict)
        ):
            return ""
        output = "\n".join(str(result.get(key) or "") for key in ("stdout", "stderr"))
        if not re.search(
            r"(?:found\s+0\s+(?:bib|reference)|"
            r"(?:total\s+)?(?:bib(?:liography)?(?:\s+items?)?|refs?|references?)\s*:\s*0\b|"
            r"no\s+bibliography|no\s+references)",
            output,
            flags=re.IGNORECASE,
        ):
            return ""
        key = "html_regex_parser_guidance"
        if key in agent._notified_thresholds:
            return ""
        agent._notified_thresholds.add(key)
        return (
            "The tag-specific HTML regex returned zero bibliography records. Do not try another div/ol/li regex "
            "or dump every class name. Parse the saved HTML with BeautifulSoup and select the known CSS class "
            "independently of tag name (for ar5iv, `soup.select('.ltx_bibitem')`), then use each node's complete "
            "text and id/label. Inspect one selected node only if the selector itself is empty."
        )

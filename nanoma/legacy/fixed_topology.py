"""第 2 代 · 声明式固定拓扑（45 个方法，1570 行）。

按 JSON 拓扑文件预先规定节点角色、依赖、可写路径与验证门，由运行时轮询推进。
已被第 6 代 spawn judge 取代：没有任何 benchmark adapter 或 launch 脚本会设置
NANOMA_FIXED_ORCHESTRATION_PROFILE / _CONFIG，因此这些方法在当前执行路径上
不会被触达。

保留而非删除，是因为 nanoma/topologies/ 下 54 个拓扑 JSON 仍有参考价值；
要复活这一代，配置在 core.py 的 _ArchivedFixedTopologyConfig 里。

以 mixin 形式由 Runtime 继承，调用点无需改动。抽出时实测边界：
入向 23 个入口 <- core 中 13 个方法；出向依赖 core 中 12 个方法。
其中三个名字带 _fixed_ 前缀的通用文件工具（_fixed_remove_path、
_fixed_snapshot_ignore、_fixed_copy_snapshot_item）留在 core —— 它们
只是沿用了这一代的命名，实际被活跃的 merge 路径使用。
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanoma.fixed_orchestration import (
    FixedAgentSpec,
    FixedCheckpoint,
    FixedVerificationGate,
)
from nanoma.llm import ToolCall, estimate_tokens
from nanoma.tool_groups import _DELIVERY_WRITE_TOOLS

if TYPE_CHECKING:
    from nanoma.core import Agent


class FixedTopologyMixin:
    """Runtime 的固定拓扑编排行为。不可独立实例化，只作为 Runtime 的基类。"""

    def _fixed_repo_path(self) -> Path | None:
        if not self._fixed_plan:
            return None
        configured = os.path.expandvars(self._fixed_plan.task_repo).strip()
        if configured:
            return Path(configured).expanduser()
        if self.config.workspace_extra_roots:
            return Path(self.config.workspace_extra_roots[0]).expanduser()
        return None

    def _fixed_snapshot_roots(self) -> tuple[Path, ...]:
        if not self._fixed_plan:
            return ()
        roots: list[Path] = []
        for raw_path in self._fixed_plan.snapshot_paths:
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts:
                continue
            if path.as_posix() in {"", "."}:
                return ()
            roots.append(path)
        return tuple(roots)

    def _fixed_snapshot_source(self, destination: Path) -> bool:
        source = self._fixed_repo_path()
        if (
            not self._fixed_plan
            or not self._fixed_plan.snapshot_enabled
            or source is None
            or not source.is_dir()
        ):
            return False
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        roots = self._fixed_snapshot_roots()
        if roots:
            destination.mkdir(parents=True, exist_ok=True)
            copied = False
            for relative in roots:
                item = source / relative
                if not item.exists() and not item.is_symlink():
                    continue
                self._fixed_copy_snapshot_item(item, destination / relative)
                copied = True
            if not copied:
                return False
        else:
            shutil.copytree(
                source,
                destination,
                ignore=self._fixed_snapshot_ignore,
                symlinks=True,
            )
        return True

    def _fixed_initialize_green_checkpoint(self, root_id: str) -> None:
        if not self._fixed_plan:
            return
        if not self._fixed_plan.snapshot_enabled:
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "baseline_without_snapshot",
                "path": None,
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })
            return
        path = self.config.workspace_root / "fixed-green" / "baseline"
        if self._fixed_snapshot_source(path):
            self._fixed_last_green_path = path
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "baseline",
                "path": str(path),
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })

    def _fixed_record_green_checkpoint(self, root_id: str) -> bool:
        self._fixed_green_seq += 1
        path = self.config.workspace_root / "fixed-green" / f"green-{self._fixed_green_seq:03d}"
        if self._fixed_plan and not self._fixed_plan.snapshot_enabled:
            self._fixed_last_green_path = None
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._fixed_has_verified_green = True
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "verification_gate_without_snapshot",
                "seq": self._fixed_green_seq,
                "path": None,
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })
            return True
        if self._fixed_snapshot_source(path):
            self._fixed_last_green_path = path
            self._fixed_last_green_write_seq = self._fixed_source_write_seq
            self._fixed_has_verified_green = True
            self._emit(root_id, "fixed_green_checkpoint", {
                "kind": "verification_gate",
                "seq": self._fixed_green_seq,
                "path": str(path),
                "source_write_seq": self._fixed_source_write_seq,
                "elapsed_seconds": round(self.effective_elapsed(), 1),
            })
            return True
        self._emit(root_id, "fixed_green_checkpoint_skipped", {
            "reason": "snapshot_failed",
            "path": str(path),
            "source_write_seq": self._fixed_source_write_seq,
        })
        return False

    def _fixed_restore_green_checkpoint(self, root_id: str, reason: str) -> bool:
        source = self._fixed_last_green_path
        destination = self._fixed_repo_path()
        if (
            not self._fixed_plan
            or not self._fixed_plan.snapshot_enabled
            or source is None
            or destination is None
            or not source.is_dir()
        ):
            return False
        roots = self._fixed_snapshot_roots()
        if roots:
            for relative in roots:
                target = destination / relative
                self._fixed_remove_path(target)
                item = source / relative
                if item.exists() or item.is_symlink():
                    self._fixed_copy_snapshot_item(item, target)
        else:
            snapshot_files = {
                path.relative_to(source)
                for path in source.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            for path in destination.rglob("*"):
                if not (path.is_file() or path.is_symlink()):
                    continue
                relative = path.relative_to(destination)
                if any(part in self._fixed_snapshot_ignore("", [part]) for part in relative.parts):
                    continue
                if relative not in snapshot_files:
                    self._fixed_remove_path(path)
            for relative in snapshot_files:
                target = destination / relative
                self._fixed_remove_path(target)
                self._fixed_copy_snapshot_item(source / relative, target)
        self._fixed_source_write_seq = self._fixed_last_green_write_seq
        self._fixed_green_restores += 1
        self._emit(root_id, "fixed_green_restore", {
            "source": str(source),
            "reason": reason,
            "restore_count": self._fixed_green_restores,
            "source_write_seq": self._fixed_source_write_seq,
            "elapsed_seconds": round(self.effective_elapsed(), 1),
        })
        return True

    def _fixed_normalize_owned_path(self, raw_path: str) -> str | None:
        repo = self._fixed_repo_path()
        if repo is None or not raw_path.strip():
            return None
        path = Path(raw_path.strip())
        if path.is_absolute():
            try:
                return path.resolve().relative_to(repo.resolve()).as_posix()
            except ValueError:
                return None
        normalized = path.as_posix().lstrip("./")
        marker = "se-bmk-intern/combinatorial-games/"
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
        return normalized

    def _fixed_tool_write_paths(self, tc: ToolCall) -> list[str]:
        if tc.name == "ws_apply_patch":
            patch = str(tc.arguments.get("input", ""))
            return re.findall(
                r"^\*\*\* (?:Add|Update|Delete) File:\s*(.+?)\s*$",
                patch,
                flags=re.MULTILINE,
            )
        path = tc.arguments.get("path")
        return [str(path)] if path else []

    def _fixed_source_paths_for_tool(self, tc: ToolCall) -> list[str]:
        if not self._fixed_plan or tc.name not in _DELIVERY_WRITE_TOOLS:
            return []
        normalized = (
            self._fixed_normalize_owned_path(raw_path)
            for raw_path in self._fixed_tool_write_paths(tc)
        )
        return list(dict.fromkeys(path for path in normalized if path is not None))

    def _fixed_write_block_reason(self, agent: Agent, tc: ToolCall) -> str | None:
        spec = self._fixed_agent_specs.get(agent.id)
        if not self._fixed_plan:
            return None
        source_paths = self._fixed_source_paths_for_tool(tc)
        if source_paths and self._fixed_source_writes_frozen:
            return "source writes are frozen while the root validates or submits a green checkpoint"
        if (
            source_paths
            and agent.parent is None
            and not self._fixed_plan.root_source_writes
        ):
            return "the fixed topology reserves source writes for assigned worker nodes"
        if spec is None:
            return None
        if spec.read_only:
            return f"fixed role {spec.key} is read-only"
        paths = self._fixed_tool_write_paths(tc)
        if not paths:
            return "fixed orchestration requires an attributable file path for every write"
        allowed = set(spec.allowed_paths)
        for raw_path in paths:
            normalized = self._fixed_normalize_owned_path(raw_path)
            if normalized is None:
                # Private NanoMA workspace handoffs are allowed.
                continue
            if "*" not in allowed and normalized not in allowed:
                return (
                    f"fixed role {spec.key} does not own {normalized}; "
                    f"allowed_paths={sorted(allowed)}"
                )
        return None

    @staticmethod
    def _fixed_shell_redirect_scan(command: str) -> str:
        masked = list(command)

        # Heredoc bodies are interpreter input, not shell syntax. Mask them so
        # comparisons inside a Python/Ruby/Perl program are not mistaken for
        # output redirects.
        heredoc_pattern = re.compile(
            r"<<-?\s*(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)"
            r"(?P=quote)"
        )
        search_from = 0
        while match := heredoc_pattern.search(command, search_from):
            body_start = command.find("\n", match.end())
            if body_start < 0:
                break
            terminator = re.search(
                rf"(?m)^[\t ]*{re.escape(match.group('tag'))}[\t ]*(?:\n|$)",
                command[body_start + 1:],
            )
            if terminator is None:
                search_from = match.end()
                continue
            body_end = body_start + 1 + terminator.start()
            for index in range(body_start + 1, body_end):
                masked[index] = " "
            search_from = body_start + 1 + terminator.end()

        # Shell operators inside quoted arguments belong to that argument. In
        # particular, Python comparisons in `python -c "..."` are not redirects.
        quote: str | None = None
        escaped = False
        for index, char in enumerate(command):
            if masked[index] == " " and char != " ":
                continue
            if escaped:
                masked[index] = " "
                escaped = False
                continue
            if quote is not None:
                masked[index] = " "
                if char == "\\" and quote == '"':
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                masked[index] = " "
            elif char == "\\":
                escaped = True
                masked[index] = " "

        return "".join(masked)

    def _fixed_shell_block_reason(self, agent: Agent, command: str) -> str | None:
        spec = self._fixed_agent_specs.get(agent.id)
        if not self._fixed_plan:
            return None
        # Only checkpoint-bound roles inherit fixed-topology shell contracts.
        # NanoMA-created adaptive children retain the runtime's normal behavior.
        if spec is None and agent.parent is not None:
            return None
        lowered = command.lower()
        destructive_git = re.search(
            r"\bgit\s+(?:reset|restore|checkout|clean|revert|switch|merge|cherry-pick)\b",
            lowered,
        )
        if destructive_git and agent.parent is not None:
            return "destructive or integrating git commands are reserved for the root integrator"
        # Discard harmless fd merges and /dev/null sinks before looking for
        # shell redirects that could mutate benchmark source files.
        redirect_scan = self._fixed_shell_redirect_scan(command)
        write_scan = re.sub(
            r"\d*>{1,2}\s*(?:&\d+|/dev/null)(?=\s|[;|&]|$)",
            "",
            redirect_scan.lower(),
        )
        write_markers = (
            r"(?:^|\s)(?:sed\s+-i|perl\s+-pi|tee\s|cp\s|mv\s|rm\s)",
            # Block redirects to files, but permit fd merges used to inspect
            # build output, such as `lake build 2>&1 | tail -80`.
            r"(?:^|[\s;|&])\d*>{1,2}(?!&)",
        )
        python_write_markers = (
            r"\b(?:write_text|write_bytes|writelines)\s*\(",
            r"\bopen\s*\([^)]*,\s*['\"][wax+]",
            r"\.(?:save|to_csv|to_excel|to_json|to_parquet|to_pickle|dump)\s*\(",
            r"\b(?:shutil\.)?(?:copy|copy2|copyfile|move|rmtree)\s*\(",
            r"\bos\.(?:remove|unlink|rename|replace|makedirs|mkdir)\s*\(",
            r"\bpath\s*\([^)]*\)\.(?:unlink|mkdir|rename|replace|touch)\s*\(",
            r"\bapply_patch\b",
        )
        python_source_write = bool(
            re.search(r"\bpython(?:3(?:\.\d+)?)?\b", lowered)
            and any(
                re.search(pattern, lowered, flags=re.DOTALL)
                for pattern in python_write_markers
            )
        )
        if (
            any(re.search(pattern, write_scan) for pattern in write_markers)
            or python_source_write
        ):
            return (
                "shell-based source writes are disabled for fixed orchestration roles; "
                "use ws_replace_string/ws_multi_replace/ws_apply_patch on owned files"
            )
        return None

    def _fixed_finish_block_reason(self, agent: Agent, tool_name: str) -> str | None:
        if not self._fixed_plan:
            return None
        if tool_name == "submit" and agent.parent is not None:
            return "only the fixed orchestration root may submit"
        gate = self._fixed_verification_gate()
        if tool_name == "submit" and gate and gate.argv == ("sforge-submit",):
            return "official submission is owned by the Runtime verification gate"
        if agent.parent is not None:
            return None
        if self._fixed_plan.direct_build_gate and not self._fixed_has_verified_green:
            return "no Runtime-verified green checkpoint is available"
        if self._fixed_source_writes_in_flight:
            return (
                f"{self._fixed_source_writes_in_flight} source write(s) are still in flight; "
                "wait for writers, then run the Runtime verification gate"
            )
        if self._fixed_source_write_seq != self._fixed_last_green_write_seq:
            return (
                "the source tree changed after the last green checkpoint; wait for writers and "
                "obtain a successful root lake build or Runtime verification gate before "
                "finishing or submitting"
            )
        return None

    @staticmethod
    def _fixed_is_full_lake_build(command: str) -> bool:
        # Only a bare build (optionally preceded by one `cd ... &&`) preserves
        # the process exit code. Pipelines and trailing shell commands must
        # never be accepted as green evidence.
        cleaned = re.sub(r"\d*>\s*&\d+", "", command).strip()
        if (
            not cleaned
            or re.search(r"[|;\n\r]", cleaned)
            or re.search(r"(?<!&)&(?!&)", cleaned)
        ):
            return False
        parts = re.split(r"\s*&&\s*", cleaned)
        if len(parts) == 2:
            try:
                cd_tokens = shlex.split(parts[0])
            except ValueError:
                return False
            if len(cd_tokens) not in {2, 3} or cd_tokens[0] != "cd":
                return False
            if len(cd_tokens) == 3 and cd_tokens[1] != "--":
                return False
            build = parts[1]
        elif len(parts) == 1:
            build = parts[0]
        else:
            return False
        try:
            return shlex.split(build) == ["lake", "build"]
        except ValueError:
            return False

    def _fixed_observe_shell_result(
        self,
        agent: Agent,
        command: str,
        result: Any,
        write_seq_before: int | None = None,
    ) -> None:
        if not self._fixed_plan or agent.parent is not None:
            return
        if not self._fixed_is_full_lake_build(command):
            return
        if self._fixed_plan.direct_build_gate:
            self._emit(agent.id, "fixed_green_checkpoint_skipped", {
                "reason": "agent_shell_build_is_not_authoritative",
                "command": command[:500],
            })
            return
        exit_code = result.get("exit_code") if isinstance(result, dict) else None
        if exit_code == 0:
            concurrent_write = (
                write_seq_before is not None
                and write_seq_before != self._fixed_source_write_seq
            )
            if concurrent_write or self._fixed_source_writes_in_flight:
                self._emit(agent.id, "fixed_green_checkpoint_skipped", {
                    "reason": "source_changed_during_build",
                    "write_seq_before": write_seq_before,
                    "write_seq_after": self._fixed_source_write_seq,
                    "writes_in_flight": self._fixed_source_writes_in_flight,
                })
            else:
                self._fixed_record_green_checkpoint(agent.id)
        elif self._fixed_restore_green_checkpoint(agent.id, "root_lake_build_failed"):
            notice = (
                "[Runtime green restore] The root lake build failed. Source files were restored "
                "to the last successful green checkpoint. Re-read the tree before continuing."
            )
            agent.history.append({"role": "user", "content": notice})

    def _fixed_verification_gate(self) -> FixedVerificationGate | None:
        if not self._fixed_plan or not self._fixed_plan.direct_build_gate:
            return None
        return self._fixed_plan.verification_gate or FixedVerificationGate(
            argv=("lake", "build"),
            timeout_seconds=max(
                30.0,
                self.config.fixed_build_timeout_seconds
                if self.config.fixed_build_timeout_seconds is not None
                else float(os.environ.get("NANOMA_FIXED_BUILD_TIMEOUT", "900") or 900),
            ),
        )

    @staticmethod
    def _fixed_parse_gate_output(
        gate: FixedVerificationGate,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        output = f"{stdout}\n{stderr}"
        valid = exit_code == 0
        if re.search(r"^\s*Valid:\s*no\s*$", output, flags=re.IGNORECASE | re.MULTILINE):
            valid = False
        if re.search(r"^\s*[^:\n]+:\s*ERROR\s*$", output, flags=re.MULTILINE):
            valid = False

        metric = None
        metric_name = None
        score_match = re.search(
            r"^\s*Score:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
            output,
            flags=re.MULTILINE,
        )
        if score_match:
            metric = float(score_match.group(1))
            metric_name = "score"
        else:
            pass_rate_match = re.search(
                r"^\s*Pass rate:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))%\s*$",
                output,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if pass_rate_match:
                metric = float(pass_rate_match.group(1)) / 100.0
                metric_name = "pass_rate"

        if gate.acceptance == "exit-code":
            valid = exit_code == 0
        return {
            "judge_valid": valid,
            "metric": metric,
            "metric_name": metric_name,
        }

    def _fixed_gate_metric_accepted(
        self,
        gate: FixedVerificationGate,
        metric: float | None,
    ) -> bool:
        if gate.acceptance != "judge-valid-nondecreasing":
            return True
        if metric is None or self._fixed_best_gate_metric is None:
            return True
        tolerance = max(1e-12, abs(self._fixed_best_gate_metric) * 1e-9)
        if gate.score_direction == "minimize":
            return metric <= self._fixed_best_gate_metric + tolerance
        return metric + tolerance >= self._fixed_best_gate_metric

    async def _fixed_run_direct_lake_build(
        self,
        root_id: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Run the configured Runtime verification gate (legacy name retained)."""
        repo = self._fixed_repo_path()
        gate = self._fixed_verification_gate()
        if repo is None or not repo.is_dir() or gate is None:
            return {
                "accepted": False,
                "exit_code": -1,
                "reason": "verification_gate_unavailable",
            }
        gate_cwd = Path(os.path.expandvars(gate.cwd)).expanduser() if gate.cwd else repo
        if not gate_cwd.is_absolute():
            gate_cwd = repo / gate_cwd
        if not gate_cwd.is_dir():
            return {
                "accepted": False,
                "exit_code": -1,
                "reason": "verification_gate_cwd_unavailable",
                "cwd": str(gate_cwd),
            }
        async with self._fixed_build_lock:
            if self._fixed_source_writes_in_flight:
                return {
                    "accepted": False,
                    "exit_code": -1,
                    "reason": "source_write_in_flight",
                }
            write_seq_before = self._fixed_source_write_seq
            previously_frozen = self._fixed_source_writes_frozen
            self._fixed_source_writes_frozen = True
            started = time.time()
            proc: asyncio.subprocess.Process | None = None
            stdout = b""
            stderr = b""
            timed_out = False
            try:
                proc = await asyncio.create_subprocess_exec(
                    *gate.argv,
                    cwd=str(gate_cwd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=gate.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    proc.kill()
                    stdout, stderr = await proc.communicate()
            except Exception as exc:
                stderr = f"{type(exc).__name__}: {exc}".encode()
            finally:
                self._fixed_source_writes_frozen = previously_frozen

            exit_code = -1 if proc is None or timed_out else int(proc.returncode or 0)
            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")
            parsed = self._fixed_parse_gate_output(
                gate,
                exit_code,
                stdout_text,
                stderr_text,
            )
            source_changed = (
                write_seq_before != self._fixed_source_write_seq
                or bool(self._fixed_source_writes_in_flight)
            )
            valid = bool(parsed["judge_valid"])
            score_accepted = self._fixed_gate_metric_accepted(gate, parsed["metric"])
            gate_passed = valid and score_accepted and not source_changed
            accepted = gate_passed and self._fixed_record_green_checkpoint(root_id)
            if accepted:
                outcome = "accepted"
                if parsed["metric"] is not None:
                    self._fixed_best_gate_metric = float(parsed["metric"])
            else:
                restore_reason = (
                    "runtime_verification_source_changed"
                    if source_changed
                    else (
                        "runtime_verification_score_regression"
                        if valid and not score_accepted
                        else (
                            "runtime_verification_snapshot_failed"
                            if gate_passed
                            else "runtime_verification_failed"
                        )
                    )
                )
                self._fixed_restore_green_checkpoint(root_id, restore_reason)
                outcome = "restored"
            result = {
                "accepted": accepted,
                "exit_code": exit_code,
                "reason": reason,
                "outcome": outcome,
                "argv": list(gate.argv),
                "cwd": str(gate_cwd),
                "acceptance": gate.acceptance,
                "score_direction": gate.score_direction,
                "judge_valid": valid,
                "metric": parsed["metric"],
                "metric_name": parsed["metric_name"],
                "best_metric": self._fixed_best_gate_metric,
                "timed_out": timed_out,
                "source_changed": source_changed,
                "write_seq_before": write_seq_before,
                "write_seq_after": self._fixed_source_write_seq,
                "elapsed_seconds": round(time.time() - started, 3),
                "stdout_tail": stdout_text[-12000:],
                "stderr_tail": stderr_text[-12000:],
            }
            self._emit(root_id, "fixed_runtime_build_gate", result)
            self._emit(root_id, "fixed_runtime_verification_gate", result)
            return result

    def _fixed_elapsed_fraction(self) -> float:
        if self.config.time_limit <= 0:
            return 0.0
        return min(1.0, self.effective_elapsed() / self.config.time_limit)

    def _fixed_checkpoint(self, key: str) -> FixedCheckpoint | None:
        if not self._fixed_plan:
            return None
        return next((item for item in self._fixed_plan.checkpoints if item.key == key), None)

    def _fixed_missing_specs(self, checkpoint: FixedCheckpoint) -> list[FixedAgentSpec]:
        return [
            spec
            for spec in checkpoint.agents
            if spec.key not in self._fixed_agent_ids
        ]

    def _fixed_checkpoint_complete(self, key: str) -> bool:
        checkpoint = self._fixed_checkpoint(key)
        if checkpoint is None or key not in self._fixed_fired_checkpoints:
            return False
        spawned = self._fixed_checkpoint_agents.get(key, set())
        return bool(spawned) and all(
            self.agents[agent_id].status == "done"
            for agent_id in spawned
            if agent_id in self.agents
        )

    def _fixed_expire_agents(self) -> None:
        elapsed = self.effective_elapsed()
        current_task = asyncio.current_task()
        for agent_id, spec in list(self._fixed_agent_specs.items()):
            agent = self.agents.get(agent_id)
            if (
                agent is None
                or spec.end_elapsed_seconds <= 0
                or elapsed < spec.end_elapsed_seconds
                or agent.status not in {"running", "idle"}
            ):
                continue
            agent.status = "done"
            agent.result = agent.result or (
                f"[Runtime role deadline at {spec.end_elapsed_seconds:.0f}s; "
                "current candidate will be validated]"
            )
            if (
                agent._task is not None
                and agent._task is not current_task
                and not agent._task.done()
            ):
                agent._task.cancel()
            self._emit(agent_id, "fixed_agent_deadline", {
                "role": spec.key,
                "deadline_seconds": spec.end_elapsed_seconds,
                "elapsed_seconds": round(elapsed, 1),
            })

    async def _fixed_validate_completed_agents(self) -> None:
        if not self._fixed_plan or not self._fixed_plan.direct_build_gate:
            return
        gate = self._fixed_verification_gate()
        if gate is None:
            return
        root_id = self._fixed_agent_ids.get("root")
        if not root_id or root_id not in self.agents:
            return
        for agent_id, spec in list(self._fixed_agent_specs.items()):
            if agent_id in self._fixed_validated_agents:
                continue
            agent = self.agents.get(agent_id)
            if agent is None or agent.status not in {"done", "failed"}:
                continue
            if agent._task is not None and not agent._task.done():
                continue
            if spec.read_only:
                self._fixed_validated_agents.add(agent_id)
                self._fixed_candidate_validations[agent_id] = {
                    "role": spec.key,
                    "accepted": agent.status == "done",
                    "outcome": "read_only_complete" if agent.status == "done" else "read_only_failed",
                }
                continue

            other_writer_active = any(
                other_id != agent_id
                and not other_spec.read_only
                and (other := self.agents.get(other_id)) is not None
                and other.status in {"running", "idle"}
                for other_id, other_spec in self._fixed_agent_specs.items()
            )
            if other_writer_active or self._fixed_source_writes_in_flight:
                continue

            if agent.status == "failed":
                if self._fixed_source_write_seq != self._fixed_last_green_write_seq:
                    self._fixed_restore_green_checkpoint(
                        root_id,
                        f"fixed_writer_failed:{spec.key}",
                    )
                validation = {
                    "role": spec.key,
                    "accepted": False,
                    "outcome": "agent_failed_restored",
                }
            elif (
                self._fixed_source_write_seq == self._fixed_last_green_write_seq
                and not gate.always_run
            ):
                validation = {
                    "role": spec.key,
                    "accepted": self._fixed_has_verified_green,
                    "outcome": "no_source_change",
                }
            else:
                build = await self._fixed_run_direct_lake_build(
                    root_id,
                    reason=f"writer_complete:{spec.key}",
                )
                if build.get("reason") == "source_write_in_flight":
                    continue
                validation = {"role": spec.key, **build}

            self._fixed_validated_agents.add(agent_id)
            self._fixed_candidate_validations[agent_id] = validation
            accepted = bool(validation.get("accepted"))
            root = self.agents[root_id]
            root.history.append({
                "role": "user",
                "content": (
                    f"[Runtime candidate gate] {spec.key}: "
                    f"{'accepted as the new green checkpoint' if accepted else 'rejected; previous green restored'}. "
                    f"outcome={validation.get('outcome')}; "
                    f"metric={validation.get('metric')}; best={validation.get('best_metric')}."
                ),
            })
            self._emit(root_id, "fixed_candidate_validation", {
                "agent": agent_id,
                **validation,
            })

    def fixed_orchestration_completion_block_reason(self, agent: Agent) -> str | None:
        plan = self._fixed_plan
        if not plan or agent.parent is not None or not plan.completion_checkpoint:
            return None
        if self._fixed_checkpoint_complete(plan.completion_checkpoint):
            return None
        if self._fixed_elapsed_fraction() >= 0.95 or self.effective_elapsed() >= 6900:
            return None
        return (
            f"fixed orchestration is still active; wait for checkpoint "
            f"{plan.completion_checkpoint!r} and query its verifier before finishing"
        )

    def _fixed_dependency_terminal(self, agent_id: str) -> bool:
        agent = self.agents.get(agent_id)
        if agent is None or agent.status not in {"done", "failed"}:
            return False
        spec = self._fixed_agent_specs.get(agent_id)
        if (
            self._fixed_plan
            and self._fixed_plan.direct_build_gate
            and spec is not None
            and not spec.read_only
        ):
            return agent_id in self._fixed_validated_agents
        return True

    def _fixed_checkpoint_ready(self, checkpoint: FixedCheckpoint) -> bool:
        if any(dep not in self._fixed_fired_checkpoints for dep in checkpoint.depends_on):
            return False
        parent_id = self._fixed_agent_ids.get(checkpoint.parent_key)
        if not parent_id or parent_id not in self.agents:
            return False
        parent = self.agents[parent_id]
        age = max(0.0, time.time() - parent._created_at)
        elapsed = self.effective_elapsed()
        elapsed_fraction = self._fixed_elapsed_fraction()
        if (
            checkpoint.not_before_elapsed_seconds > 0
            and elapsed < checkpoint.not_before_elapsed_seconds
        ):
            return False
        if (
            checkpoint.require_parent_without_candidate
            and self._has_delivery_candidate(parent)
        ):
            return False
        if (
            checkpoint.min_parent_web_no_gain_calls > 0
            and parent._shell_activity.web_no_gain_calls
            < checkpoint.min_parent_web_no_gain_calls
        ):
            return False
        if (
            checkpoint.min_parent_unavailable_tool_calls > 0
            and parent._shell_activity.unavailable_tool_calls
            < checkpoint.min_parent_unavailable_tool_calls
        ):
            return False

        if checkpoint.mode == "settled":
            dependency_agents = set().union(*(
                self._fixed_checkpoint_agents.get(dep, set())
                for dep in checkpoint.depends_on
            )) if checkpoint.depends_on else set()
            if dependency_agents:
                return all(
                    self._fixed_dependency_terminal(agent_id)
                    for agent_id in dependency_agents
                    if agent_id in self.agents
                )
            return bool(
                (
                    checkpoint.fallback_elapsed_fraction > 0
                    and elapsed_fraction >= checkpoint.fallback_elapsed_fraction
                )
                or (
                    checkpoint.fallback_elapsed_seconds > 0
                    and elapsed >= checkpoint.fallback_elapsed_seconds
                )
            )

        if checkpoint.mode == "stalled":
            if parent.status in {"done", "failed"}:
                return False
            pending_override = self._pending_tool_overrides.get(parent_id) or {}
            if "deliver_to_parent" in set(pending_override.get("tools") or []):
                return False
            last_write_turn = parent._delivery_activity.last_write_turn
            stalled_turns = parent._turns - last_write_turn if last_write_turn > 0 else parent._turns
            stalled = stalled_turns >= max(1, checkpoint.min_stall_turns)
            progressed = parent._turns >= checkpoint.min_parent_turns
            age_fallback = (
                checkpoint.min_parent_age_seconds > 0
                and age >= checkpoint.min_parent_age_seconds
            )
            return stalled and (progressed or age_fallback)

        if parent.status in {"done", "failed"}:
            return True

        progress_signals = []
        if checkpoint.min_parent_turns > 0:
            progress_signals.append(parent._turns >= checkpoint.min_parent_turns)
        if checkpoint.min_parent_tool_calls > 0:
            progress_signals.append(parent._tool_calls >= checkpoint.min_parent_tool_calls)
        if checkpoint.min_parent_age_seconds > 0:
            progress_signals.append(age >= checkpoint.min_parent_age_seconds)
        fallback_signals = []
        if checkpoint.fallback_elapsed_seconds > 0:
            fallback_signals.append(elapsed >= checkpoint.fallback_elapsed_seconds)
        if checkpoint.fallback_elapsed_fraction > 0:
            fallback_signals.append(elapsed_fraction >= checkpoint.fallback_elapsed_fraction)
        if progress_signals:
            progress_ready = (
                all(progress_signals)
                if checkpoint.progress_logic == "all"
                else any(progress_signals)
            )
        else:
            progress_ready = not fallback_signals
        return progress_ready or any(fallback_signals)

    async def _notify_fixed_agent(self, to_id: str, message: str, mode: str = "steer") -> None:
        # Deferred: core imports this module, so its message type can only be
        # reached at call time.
        from nanoma.core import Envelope

        await self.deliver(Envelope(
            from_id="runtime-checkpoint",
            to_id=to_id,
            content=message,
            tokens=estimate_tokens(message),
            timestamp=time.time(),
            mode=mode,
        ))
        self._emit(to_id, "fixed_orchestration_message", {
            "from": "runtime-checkpoint",
            "message": message[:500],
        })

    def _fixed_dependency_handoff(self, checkpoint: FixedCheckpoint) -> str:
        handoffs: list[str] = []
        for dependency in checkpoint.depends_on:
            for agent_id in sorted(self._fixed_checkpoint_agents.get(dependency, set())):
                agent = self.agents.get(agent_id)
                if agent is None or not agent.result:
                    continue
                role = self._fixed_agent_keys.get(agent_id, agent.bio or agent_id)
                result = str(agent.result).strip()
                if not result:
                    continue
                handoffs.append(
                    f"[{dependency}/{role} from {agent_id}]\n{result[:2400]}"
                )
        if not handoffs:
            return ""
        return "\n\n".join(handoffs)[:6000]

    def _fixed_release_spawn_request(self, parent_id: str, checkpoint_key: str) -> None:
        if self._fixed_pending_checkpoint_by_parent.get(parent_id) == checkpoint_key:
            self._fixed_pending_checkpoint_by_parent.pop(parent_id, None)
        self._fixed_requested_checkpoints.discard(checkpoint_key)
        override = self._pending_tool_overrides.get(parent_id)
        if (
            override
            and (override.get("metadata") or {}).get("fixed_checkpoint") == checkpoint_key
        ):
            self._pending_tool_overrides.pop(parent_id, None)

    def _fixed_spawn_context(
        self,
        agent: Agent,
    ) -> tuple[FixedCheckpoint, list[FixedAgentSpec], dict[str, Any]] | None:
        checkpoint_key = self._fixed_pending_checkpoint_by_parent.get(agent.id)
        if not checkpoint_key:
            return None
        checkpoint = self._fixed_checkpoint(checkpoint_key)
        override = self._pending_tool_overrides.get(agent.id)
        metadata = dict((override or {}).get("metadata") or {})
        if checkpoint is None or metadata.get("fixed_checkpoint") != checkpoint_key:
            return None
        selected_keys = [str(item) for item in metadata.get("fixed_agent_keys") or []]
        specs_by_key = {spec.key: spec for spec in checkpoint.agents}
        selected = [specs_by_key[key] for key in selected_keys if key in specs_by_key]
        if not selected:
            return None
        return checkpoint, selected, metadata

    @staticmethod
    def _fixed_route_parent_task(spec: FixedAgentSpec, requested: Any) -> str | None:
        if isinstance(requested, dict):
            requested = (
                requested.get("task")
                or requested.get("description")
                or requested.get("unit")
            )
        parent_task = str(requested or "").strip()
        if not parent_task:
            return None
        runtime_contract = spec.task.strip()
        if parent_task == runtime_contract:
            return runtime_contract
        return (
            f"{runtime_contract}\n\n"
            "[Parent-authored execution plan]\n"
            "The runtime role and access contract above takes precedence over this plan.\n"
            f"{parent_task}"
        )

    def _fixed_prepare_spawn_tool_call(self, tc: ToolCall, agent: Agent) -> str | None:
        context = self._fixed_spawn_context(agent)
        if context is None:
            # A fixed topology adds checkpoint requests; it does not replace
            # NanoMA's ordinary, adaptive spawn behavior.
            return None
        checkpoint, specs, metadata = context
        required_tool = str(
            metadata.get("spawn_tool") or ("spawn_many" if len(specs) > 1 else "spawn")
        )
        if tc.name != required_tool:
            return (
                f"checkpoint {checkpoint.key} requires {required_tool}, not {tc.name}; "
                "use the single spawn tool exposed for this turn"
            )

        original_arguments = copy.deepcopy(tc.arguments)
        if tc.name == "spawn":
            if len(specs) != 1:
                return f"checkpoint {checkpoint.key} requires {len(specs)} children via spawn_many"
            routed_task = self._fixed_route_parent_task(specs[0], original_arguments)
            if routed_task is None:
                return (
                    f"checkpoint {checkpoint.key} requires the parent agent to author a non-empty "
                    "child task in NanoMA's spawn call"
                )
            tc.arguments = {"task": routed_task}
            requested_model = original_arguments.get("model")
            if requested_model:
                tc.arguments["model"] = requested_model
        else:
            requested = original_arguments.get("agents") or original_arguments.get("tasks") or []
            if not isinstance(requested, list) or len(requested) != len(specs):
                return (
                    f"checkpoint {checkpoint.key} requires exactly {len(specs)} child assignments "
                    "in one spawn_many call"
                )
            routed_agents: list[dict[str, Any]] = []
            for index, (spec, item) in enumerate(zip(specs, requested)):
                routed_task = self._fixed_route_parent_task(spec, item)
                if routed_task is None:
                    return (
                        f"checkpoint {checkpoint.key} requires a non-empty parent-authored task "
                        f"for child assignment {index + 1}"
                    )
                routed: dict[str, Any] = {"task": routed_task}
                if isinstance(item, dict) and item.get("model"):
                    routed["model"] = item["model"]
                routed_agents.append(routed)
            tc.arguments = {"agents": routed_agents}

        self._emit(agent.id, "fixed_spawn_tool_routed", {
            "checkpoint": checkpoint.key,
            "tool": tc.name,
            "agent_keys": [spec.key for spec in specs],
            "original_args_preview": str(original_arguments)[:500],
            "execution": "nanoma_tool",
            "parent_tasks_preserved": True,
            "runtime_contracts_applied": True,
        })
        return None

    def _fixed_record_observed_spawn(
        self,
        tc: ToolCall,
        parent: Agent,
        child: Agent,
        *,
        spawn_origin: str,
        checkpoint: str | None = None,
        role_key: str | None = None,
    ) -> None:
        if any(item.get("child_id") == child.id for item in self._fixed_observed_spawns):
            return
        record = {
            "parent_id": parent.id,
            "parent_role": self._fixed_agent_keys.get(parent.id),
            "child_id": child.id,
            "child_role": role_key,
            "spawn_origin": spawn_origin,
            "checkpoint": checkpoint,
            "tool": tc.name,
            "tool_call_id": tc.id,
            "elapsed_seconds": round(self.effective_elapsed(), 3),
            "depth": child.depth,
            "model": child.model,
            "bio": child.bio,
            "task_preview": child.task[:500],
        }
        self._fixed_observed_spawns.append(record)
        if spawn_origin == "autonomous_spawn":
            self._emit(parent.id, "autonomous_spawn_observed", {
                "child": child.id,
                "parent_role": record["parent_role"],
                "depth": child.depth,
                "model": child.model,
                "spawn_tool": tc.name,
                "tool_call_id": tc.id,
                "task_preview": child.task[:500],
                "topology_update": True,
            })

    def _fixed_observed_topology(self) -> dict[str, Any]:
        spawn_by_child = {
            str(record.get("child_id")): record
            for record in self._fixed_observed_spawns
            if record.get("child_id")
        }
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for agent in sorted(self.agents.values(), key=lambda item: item._created_at):
            spawn = spawn_by_child.get(agent.id)
            if agent.parent is None:
                spawn_origin = "root"
            elif spawn:
                spawn_origin = str(spawn.get("spawn_origin") or "autonomous_spawn")
            elif agent.id in self._fixed_agent_specs:
                spawn_origin = "checkpoint_spawn"
            else:
                spawn_origin = "autonomous_spawn"
            fixed_role = self._fixed_agent_keys.get(agent.id)
            nodes.append({
                "id": agent.id,
                "parent_id": agent.parent,
                "depth": agent.depth,
                "status": agent.status,
                "model": agent.model,
                "bio": agent.bio,
                "fixed_role": fixed_role,
                "spawn_origin": spawn_origin,
                "checkpoint": spawn.get("checkpoint") if spawn else None,
                "start_elapsed_seconds": round(
                    max(0.0, agent._created_at - self._start_time), 3
                ),
                "task_preview": agent.task[:500],
            })
            if agent.parent is not None:
                edges.append({
                    "from": agent.parent,
                    "to": agent.id,
                    "spawn_origin": spawn_origin,
                    "checkpoint": spawn.get("checkpoint") if spawn else None,
                    "tool": spawn.get("tool") if spawn else None,
                    "elapsed_seconds": spawn.get("elapsed_seconds") if spawn else None,
                })
        autonomous = [
            copy.deepcopy(record)
            for record in self._fixed_observed_spawns
            if record.get("spawn_origin") == "autonomous_spawn"
        ]
        checkpoint = [
            copy.deepcopy(record)
            for record in self._fixed_observed_spawns
            if record.get("spawn_origin") == "checkpoint_spawn"
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "autonomous_spawns": autonomous,
            "checkpoint_spawns": checkpoint,
            "counts": {
                "nodes": len(nodes),
                "edges": len(edges),
                "autonomous_spawns": len(autonomous),
                "checkpoint_spawns": len(checkpoint),
            },
        }

    async def _fixed_record_spawn_tool_result(
        self,
        tc: ToolCall,
        parent: Agent,
        result: Any,
    ) -> Any:
        context = self._fixed_spawn_context(parent)
        created_ids = self._spawn_tool_created_ids(tc, result)
        if context is None:
            for child_id in created_ids:
                child = self.agents.get(child_id)
                if child is not None and child.parent == parent.id:
                    self._fixed_record_observed_spawn(
                        tc,
                        parent,
                        child,
                        spawn_origin="autonomous_spawn",
                    )
            return result
        checkpoint, specs, _metadata = context

        dependency_handoff = self._fixed_dependency_handoff(checkpoint)
        registered: list[str] = []
        for spec, child_id in zip(specs, created_ids):
            child = self.agents.get(child_id)
            if child is None or child.parent != parent.id:
                continue
            child.bio = spec.bio
            self._fixed_agent_ids[spec.key] = child.id
            self._fixed_agent_keys[child.id] = spec.key
            self._fixed_agent_specs[child.id] = spec
            self._fixed_checkpoint_agents.setdefault(checkpoint.key, set()).add(child.id)
            registered.append(spec.key)
            self._fixed_record_observed_spawn(
                tc,
                parent,
                child,
                spawn_origin="checkpoint_spawn",
                checkpoint=checkpoint.key,
                role_key=spec.key,
            )
            access = (
                "read-only"
                if spec.read_only
                else f"write only: {', '.join(spec.allowed_paths) or '(no paths configured)'}"
            )
            role_message = (
                f"[Runtime role binding for {spec.key}] Your parent created you through NanoMA's "
                f"{tc.name} tool at checkpoint {checkpoint.key}. Access contract: {access}. "
                f"Assignment: {(spec.assignment or spec.bio)[:800]}. Do not work outside this contract."
            )
            if dependency_handoff:
                role_message += (
                    "\n\n[Runtime dependency handoff]\nUse these completed upstream findings as "
                    f"evidence; verify them before editing.\n\n{dependency_handoff}"
                )
            await self._notify_fixed_agent(child.id, role_message)
            self._emit(parent.id, "fixed_agent_spawned", {
                "checkpoint": checkpoint.key,
                "agent_key": spec.key,
                "child": child.id,
                "bio": spec.bio,
                "model": child.model,
                "executor": "nanoma_tool",
                "spawn_tool": tc.name,
                "tool_call_id": tc.id,
            })

        self._fixed_release_spawn_request(parent.id, checkpoint.key)
        expected_keys = {spec.key for spec in checkpoint.agents}
        spawned_keys = {
            self._fixed_agent_keys[agent_id]
            for agent_id in self._fixed_checkpoint_agents.get(checkpoint.key, set())
            if agent_id in self._fixed_agent_keys
        }
        if expected_keys.issubset(spawned_keys):
            self._fixed_fired_checkpoints.add(checkpoint.key)
            self._emit(parent.id, "fixed_checkpoint_fired", {
                "checkpoint": checkpoint.key,
                "mode": checkpoint.mode,
                "agents": sorted(spawned_keys),
                "elapsed_seconds": round(self.effective_elapsed(), 1),
                "spawn_execution": "nanoma_tool",
            })
        else:
            self._emit(parent.id, "fixed_checkpoint_spawn_incomplete", {
                "checkpoint": checkpoint.key,
                "requested": [spec.key for spec in specs],
                "registered": registered,
                "result": str(result)[:1000],
            })

        if isinstance(result, dict):
            result = {
                **result,
                "fixed_checkpoint": checkpoint.key,
                "fixed_roles_registered": registered,
                "spawn_execution": "nanoma_tool",
            }
        return result

    async def _advance_fixed_checkpoint(self, checkpoint: FixedCheckpoint) -> None:
        if not self._fixed_plan:
            return
        if (
            checkpoint.key in self._fixed_fired_checkpoints
            or checkpoint.key in self._fixed_requested_checkpoints
        ):
            return
        parent_id = self._fixed_agent_ids.get(checkpoint.parent_key)
        if not parent_id or parent_id not in self.agents:
            return
        if parent_id in self._fixed_pending_checkpoint_by_parent:
            return
        parent = self.agents[parent_id]
        if parent.status == "failed":
            return
        missing_specs = self._fixed_missing_specs(checkpoint)
        if not missing_specs:
            return

        live_agents = sum(
            other.status in {"running", "idle"}
            for other in self.agents.values()
        )
        available_slots = min(
            max(0, self._fixed_plan.max_live_agents - live_agents),
            max(0, self.config.max_agents - len(self.agents)),
        )
        if available_slots <= 0:
            return

        selected: list[FixedAgentSpec] = []
        for spec in missing_specs:
            if self._fixed_plan.direct_build_gate and not spec.read_only:
                writer_active = any(
                    not other_spec.read_only
                    and (other := self.agents.get(other_id)) is not None
                    and (
                        other.status in {"running", "idle"}
                        or (
                            other.status in {"done", "failed"}
                            and other_id not in self._fixed_validated_agents
                        )
                    )
                    for other_id, other_spec in self._fixed_agent_specs.items()
                )
                if writer_active:
                    continue
            selected.append(spec)
            if len(selected) >= available_slots:
                break
            if self._fixed_plan.direct_build_gate and not spec.read_only:
                break

        if not selected:
            return

        spawn_tool = "spawn_many" if len(selected) > 1 else "spawn"
        role_lines: list[str] = []
        for index, spec in enumerate(selected, start=1):
            access = (
                "read-only"
                if spec.read_only
                else f"write only: {', '.join(spec.allowed_paths) or '(no paths configured)'}"
            )
            role_lines.append(
                f"{index}. role_key={spec.key}; role={spec.bio}; access={access}; "
                f"assignment={(spec.assignment or spec.bio)[:1000]}"
            )
        request_message = (
            f"[Runtime checkpoint {checkpoint.key}] This checkpoint changes your next action only. "
            f"Runtime will not create these agents. You must call NanoMA's {spawn_tool} tool now "
            f"to create exactly {len(selected)} child{'ren' if len(selected) != 1 else ''}. "
            "Do not use shell, file, query, wait, send, or set_status before the spawn call. "
            "The runtime will bind the canonical assignments and access contracts to the children "
            "created by that tool call. This one call contains only the listed checkpoint roles; "
            "after it completes, your normal NanoMA spawn autonomy resumes.\n\n"
            + "\n".join(role_lines)
        )
        self._fixed_requested_checkpoints.add(checkpoint.key)
        self._fixed_pending_checkpoint_by_parent[parent_id] = checkpoint.key
        self._pending_tool_overrides[parent_id] = {
            "source": "fixed_orchestration_checkpoint",
            "action": "tool_override",
            "strategy_action": "FIXED_CHECKPOINT_SPAWN",
            "tools": [spawn_tool],
            "reason": f"fixed checkpoint {checkpoint.key}",
            "message": "",
            "once": False,
            "created_at": time.time(),
            "wrong_tool_attempts": 0,
            "metadata": {
                "fixed_checkpoint": checkpoint.key,
                "fixed_agent_keys": [spec.key for spec in selected],
                "spawn_tool": spawn_tool,
            },
        }
        self._tool_override_events.setdefault(parent_id, asyncio.Event()).set()
        self._fixed_completion_waiting.discard(parent_id)
        await self._notify_fixed_agent(parent_id, request_message)
        if parent.status == "done":
            parent.status = "running"
            if parent._task is None or parent._task.done():
                self.start_agent(parent)
            self._emit(parent_id, "fixed_checkpoint_parent_reactivated", {
                "checkpoint": checkpoint.key,
            })
        self._emit(parent_id, "fixed_checkpoint_requested", {
            "checkpoint": checkpoint.key,
            "mode": checkpoint.mode,
            "agents": [spec.key for spec in selected],
            "spawn_tool": spawn_tool,
            "elapsed_seconds": round(self.effective_elapsed(), 1),
            "executor": "parent_agent_via_nanoma_tool",
        })

    async def _poll_fixed_orchestration_once(self) -> None:
        if not self._fixed_plan:
            return
        async with self._fixed_orchestration_lock:
            self._fixed_expire_agents()
            await self._fixed_validate_completed_agents()
            for checkpoint in self._fixed_plan.checkpoints:
                if checkpoint.key in self._fixed_fired_checkpoints:
                    continue
                if self._fixed_checkpoint_ready(checkpoint):
                    await self._advance_fixed_checkpoint(checkpoint)

    async def _fixed_orchestration_monitor(self) -> None:
        if not self._fixed_plan:
            return
        try:
            while True:
                for agent_id in self._load_runtime_interventions():
                    self._tool_override_events.setdefault(
                        agent_id, asyncio.Event()
                    ).set()
                await self._poll_fixed_orchestration_once()
                await asyncio.sleep(self.config.fixed_orchestration_poll_seconds)
        except asyncio.CancelledError:
            return

    def _fixed_plan_fingerprint(self) -> str | None:
        if self._fixed_plan is None:
            return None
        payload = json.dumps(
            asdict(self._fixed_plan),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _fixed_orchestration_probe_state(self) -> dict[str, Any]:
        last_green_relative = None
        last_green_absolute = None
        if self._fixed_last_green_path is not None:
            try:
                last_green_relative = str(
                    self._fixed_last_green_path.relative_to(self.config.workspace_root)
                )
            except ValueError:
                last_green_absolute = str(self._fixed_last_green_path)
        return {
            "profile": self._fixed_plan.name if self._fixed_plan else None,
            "plan_fingerprint": self._fixed_plan_fingerprint(),
            "agent_ids": copy.deepcopy(self._fixed_agent_ids),
            "agent_keys": copy.deepcopy(self._fixed_agent_keys),
            "checkpoint_agents": {
                key: sorted(agent_ids)
                for key, agent_ids in self._fixed_checkpoint_agents.items()
            },
            "agent_specs": copy.deepcopy(self._fixed_agent_specs),
            "fired_checkpoints": sorted(self._fixed_fired_checkpoints),
            "requested_checkpoints": sorted(self._fixed_requested_checkpoints),
            "pending_checkpoint_by_parent": copy.deepcopy(
                self._fixed_pending_checkpoint_by_parent
            ),
            "observed_spawns": copy.deepcopy(self._fixed_observed_spawns),
            "completion_waiting": sorted(self._fixed_completion_waiting),
            "validated_agents": sorted(self._fixed_validated_agents),
            "candidate_validations": copy.deepcopy(self._fixed_candidate_validations),
            "best_gate_metric": self._fixed_best_gate_metric,
            "green_seq": self._fixed_green_seq,
            "last_green_path_relative": last_green_relative,
            "last_green_path_absolute": last_green_absolute,
            "has_verified_green": self._fixed_has_verified_green,
            "green_restores": self._fixed_green_restores,
            "source_write_seq": self._fixed_source_write_seq,
            "last_green_write_seq": self._fixed_last_green_write_seq,
            "source_writes_in_flight": self._fixed_source_writes_in_flight,
            "source_writes_frozen": self._fixed_source_writes_frozen,
        }

    def _validate_probe_topology(self, state: dict[str, Any]) -> None:
        saved = state.get("fixed_orchestration")
        current_profile = self._fixed_plan.name if self._fixed_plan else None
        if saved is None:
            if current_profile is not None:
                raise ValueError(
                    "probe does not contain fixed orchestration state; refusing to "
                    "restart an active topology from its root"
                )
            return

        saved_profile = saved.get("profile")
        if saved_profile != current_profile:
            raise ValueError(
                "fixed orchestration profile mismatch: "
                f"probe={saved_profile!r}, runtime={current_profile!r}"
            )
        if saved_profile is not None:
            saved_fingerprint = saved.get("plan_fingerprint")
            current_fingerprint = self._fixed_plan_fingerprint()
            if not saved_fingerprint or saved_fingerprint != current_fingerprint:
                raise ValueError(
                    "fixed orchestration plan changed since the probe was written; "
                    "refusing to resume at a different topology node"
                )

    def _restore_fixed_orchestration_probe_state(
        self,
        state: dict[str, Any],
        old_workspace_root: Path | None,
    ) -> None:
        saved = state.get("fixed_orchestration")
        if not saved or saved.get("profile") is None:
            return
        self._fixed_agent_ids = copy.deepcopy(saved.get("agent_ids") or {})
        self._fixed_agent_keys = copy.deepcopy(saved.get("agent_keys") or {})
        self._fixed_checkpoint_agents = {
            key: set(agent_ids)
            for key, agent_ids in (saved.get("checkpoint_agents") or {}).items()
        }
        self._fixed_agent_specs = copy.deepcopy(saved.get("agent_specs") or {})
        self._fixed_fired_checkpoints = set(saved.get("fired_checkpoints") or [])
        self._fixed_requested_checkpoints = set(
            saved.get("requested_checkpoints") or []
        )
        self._fixed_pending_checkpoint_by_parent = copy.deepcopy(
            saved.get("pending_checkpoint_by_parent") or {}
        )
        self._fixed_observed_spawns = copy.deepcopy(saved.get("observed_spawns") or [])
        self._fixed_completion_waiting = set(saved.get("completion_waiting") or [])
        self._fixed_validated_agents = set(saved.get("validated_agents") or [])
        self._fixed_candidate_validations = copy.deepcopy(
            saved.get("candidate_validations") or {}
        )
        metric = saved.get("best_gate_metric")
        self._fixed_best_gate_metric = float(metric) if metric is not None else None
        self._fixed_green_seq = int(saved.get("green_seq") or 0)
        relative_green = saved.get("last_green_path_relative")
        absolute_green = saved.get("last_green_path_absolute")
        if relative_green:
            self._fixed_last_green_path = self.config.workspace_root / relative_green
        elif absolute_green:
            green_path = Path(absolute_green)
            if old_workspace_root is not None:
                try:
                    green_path = self.config.workspace_root / green_path.relative_to(
                        old_workspace_root
                    )
                except ValueError:
                    pass
            self._fixed_last_green_path = green_path
        else:
            self._fixed_last_green_path = None
        self._fixed_has_verified_green = bool(saved.get("has_verified_green", False))
        self._fixed_green_restores = int(saved.get("green_restores") or 0)
        self._fixed_source_write_seq = int(saved.get("source_write_seq") or 0)
        self._fixed_last_green_write_seq = int(
            saved.get("last_green_write_seq") or 0
        )
        # A restored process has no tool handler currently writing source files.
        self._fixed_source_writes_in_flight = 0
        self._fixed_source_writes_frozen = bool(
            saved.get("source_writes_frozen", False)
        )
        self._fixed_monitor_task = None

        known_agents = set(self.agents)
        referenced_agents = (
            set(self._fixed_agent_ids.values())
            | set(self._fixed_agent_keys)
            | set(self._fixed_agent_specs)
            | set(self._fixed_pending_checkpoint_by_parent)
            | set(self._fixed_completion_waiting)
            | set(self._fixed_validated_agents)
            | {
                agent_id
                for agent_ids in self._fixed_checkpoint_agents.values()
                for agent_id in agent_ids
            }
        )
        unknown_agents = sorted(referenced_agents - known_agents)
        if unknown_agents:
            raise ValueError(
                "fixed orchestration probe references missing agents: "
                + ", ".join(unknown_agents)
            )

    async def _continue_fixed_from_probe(self) -> str:
        root_id = self._fixed_agent_ids.get("root")
        root = self.agents.get(root_id or "")
        if root is None:
            root = next((agent for agent in self.agents.values() if agent.parent is None), None)
        if root is None:
            raise ValueError("fixed orchestration probe has no root agent")
        root_id = root.id

        for agent in self.agents.values():
            if agent.status == "running" and (agent._task is None or agent._task.done()):
                self.start_agent(agent)
        self._fixed_monitor_task = asyncio.create_task(
            self._fixed_orchestration_monitor()
        )
        try:
            while True:
                while not self._rollback_requested:
                    active_tasks = [
                        agent._task
                        for agent in self.agents.values()
                        if agent.status == "running"
                        and agent._task is not None
                        and not agent._task.done()
                    ]
                    if not active_tasks:
                        break
                    done, _pending = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    await asyncio.gather(*done, return_exceptions=True)
                    root = self.agents.get(root_id) or root
                    if (
                        root._task is not None
                        and root._task.done()
                        and not self._rollback_requested
                    ):
                        break
                if not self._rollback_requested:
                    break
                await self._perform_requested_rollback()
                root = self.agents.get(root_id) or next(
                    (agent for agent in self.agents.values() if agent.parent is None),
                    None,
                )
                if root is None:
                    break
                root_id = root.id
                if root._task is None or root._task.done():
                    self.start_agent(root)
        finally:
            self._fixed_source_writes_frozen = True
            if self._fixed_monitor_task and not self._fixed_monitor_task.done():
                self._fixed_monitor_task.cancel()
                await asyncio.gather(self._fixed_monitor_task, return_exceptions=True)

        running = [
            agent
            for agent in self.agents.values()
            if agent.status in ("running", "idle") and agent.id != root.id
        ]
        if running:
            await self.shutdown()
        if self._fixed_source_write_seq != self._fixed_last_green_write_seq:
            self._fixed_restore_green_checkpoint(
                root.id,
                "runtime_finished_with_dirty_source",
            )
        return root.result or ""

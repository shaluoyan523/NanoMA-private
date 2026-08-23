"""合并/提交子系统：候选状态的克隆、比对、晋升与提交门禁。

从 core.py 抽出的 64 个方法（1479 行），以 mixin 形式由 Runtime 继承，
因此调用点无需改动 —— 方法仍通过 self 解析。

这是一次文件切分，不是依赖切分。边界仍然很宽，抽出时实测：

  入向  29 个方法被 core.py 中 27 个方法调用（其中 _merge_active、
        _merge_target、_merge_own_changes 三个占了大部分）
  出向  依赖 core.py 中 14 个方法，主要是 _emit（15 处）和
        _verify_* / _record_verified_best 这一组验证逻辑
  状态  读写 Runtime 上 8 个 _merge_* 实例属性

真正的解耦需要先收敛这三组交叉；本模块的意义是把边界显式化，
让后续收敛有个可度量的起点。类级常量（PREFLIGHT_TIMEOUT_DEFAULT、
_AGGREGATE_WAIT_SECONDS 等）仍留在 Runtime，mixin 方法通过 self 读取。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanoma.core import Agent

logger = logging.getLogger("nanoma")

# Above this size a merge diff compares size+mtime instead of hashing contents.
_MERGE_HASH_MAX_BYTES = 8 * 1024 * 1024
# At or above this size a file is hardlinked into a child's working copy instead
# of being duplicated: bulk payloads (datasets, archives, model weights) are read,
# never edited in place, so linking them keeps the copy complete but nearly free.
_MERGE_HARDLINK_MIN_BYTES = 1024 * 1024


class MergeSubmitMixin:
    """Runtime 的合并/提交行为。不可独立实例化，只作为 Runtime 的基类。"""

    def _submitted_result_text(
        self,
        agent: Agent,
        arguments: dict[str, Any] | None,
        result: Any,
    ) -> tuple[str, str]:
        submitted = ""
        shared_copy = ""
        if isinstance(result, dict):
            submitted = str(result.get("submitted") or "")
            shared_copy = str(result.get("shared_copy") or "")
        if not submitted and isinstance(arguments, dict):
            submitted = str(arguments.get("path") or "")

        candidates: list[Path] = []
        for raw in (shared_copy, submitted):
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = agent.workspace / path
            candidates.append(path)
        if agent.artifacts:
            candidates.append(agent.artifacts[-1].absolute_path)
        delivery = agent._delivery_activity
        for raw in sorted(delivery.expected_output_files | delivery.candidate_output_files):
            path = Path(raw)
            if not path.is_absolute():
                path = agent.workspace / path
            candidates.append(path)

        for path in candidates:
            try:
                if not path.exists() or not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                for key in ("answer", "final_answer", "FINAL ANSWER"):
                    value = parsed.get(key)
                    if value is not None:
                        return str(value).strip(), submitted
            if path.name.lower() == "answer.json" or len(text) <= 2000:
                return text, submitted
        return "", submitted

    def _merge_submit_path_enabled(self) -> bool:
        if self.config.merge_submit_path_enabled is not None:
            return self.config.merge_submit_path_enabled
        return os.environ.get("NANOMA_MERGE_SUBMIT_PATH") == "1"

    def _merge_size_caps(self) -> tuple[float, int]:
        """Copy-cost ceiling for a clone: (max duplicated MB, max file count)."""
        max_mb = self.config.merge_max_mb
        if max_mb is None:
            max_mb = float(os.environ.get("NANOMA_MERGE_MAX_MB", "200") or 200)
        max_files = self.config.merge_max_files
        if max_files is None:
            max_files = int(os.environ.get("NANOMA_MERGE_MAX_FILES", "40000") or 40000)
        return float(max_mb), int(max_files)

    def _merge_active(self) -> bool:
        if not self._merge_submit_path_enabled():
            return False
        if self._merge_disabled_reason:
            return False
        return self._merge_target() is not None

    def _merge_scope_roots(self) -> tuple[Path, ...]:
        """Relative subpaths that merge is restricted to (the submission paths).

        Reads NANOMA_MERGE_PATHS, else SFORGE_SUBMIT_PATHS (which EdgeBench
        already injects). Without this, a task directory that embeds datasets
        (e.g. ann-benchmarks: 509MB of `data/` for a 12KB submission) would be
        copied per child and hashed on every diff. An empty result means the
        whole tree, which is only safe for small task directories.
        """
        import os
        if self.config.merge_paths is not None:
            raw = self.config.merge_paths
        else:
            raw = os.environ.get("NANOMA_MERGE_PATHS") or os.environ.get("SFORGE_SUBMIT_PATHS") or ""
        roots: list[Path] = []
        for token in raw.replace(",", " ").split():
            token = token.strip().strip('"').strip("'")
            if not token:
                continue
            candidate = Path(token)
            if candidate.is_absolute() or ".." in candidate.parts:
                continue
            posix = candidate.as_posix().rstrip("/")
            if posix in {"", ".", "./"}:
                return ()  # "." = submit everything; fall back to the whole tree
            roots.append(Path(posix))
        return tuple(roots)

    def _merge_scope_stats(self, root: Path) -> tuple[int, int]:
        """(total_bytes, file_count) of the merge scope under `root`."""
        total = 0
        count = 0
        for path in self._merge_rel_files(root).values():
            try:
                if path.is_symlink():
                    continue
                total += path.stat().st_size
                count += 1
            except OSError:
                continue
        return total, count

    def _merge_clone_cost(self, source: Path) -> tuple[int, int]:
        """(bytes actually duplicated, file count) for one complete working copy.

        Only files below the hardlink threshold are duplicated; larger ones are
        linked, so they cost nothing. Explicit read-only dependency/cache roots
        are represented by one symlink and pruned from the walk. Stat-only walk,
        no reads.
        """
        duplicated = 0
        count = 0
        for current, dirnames, filenames in os.walk(source):
            relative_dir = Path(current).relative_to(source)
            dirnames[:] = [
                name for name in dirnames
                if not self._merge_clone_skip(name)
                and not self._merge_clone_path_is_shared(relative_dir / name)
            ]
            for name in filenames:
                if self._merge_clone_skip(name):
                    continue
                if self._merge_clone_path_is_shared(relative_dir / name):
                    continue
                path = Path(current) / name
                try:
                    if path.is_symlink():
                        continue
                    size = path.stat().st_size
                except OSError:
                    continue
                count += 1
                if size < _MERGE_HARDLINK_MIN_BYTES:
                    duplicated += size
        return duplicated, count

    def _merge_check_copy_budget(self, target: Path) -> bool:
        """Disable merge (once) if a per-child working copy is too expensive.

        Measures what a copy actually costs (duplicated bytes, not the tree size)
        so a task that is large only because of linkable datasets stays eligible.
        """
        max_mb, max_files = self._merge_size_caps()
        duplicated, count = self._merge_clone_cost(target)
        too_big = max_mb > 0 and duplicated > max_mb * 1024 * 1024
        too_many = max_files > 0 and count > max_files
        if not (too_big or too_many):
            return True
        self._merge_disabled_reason = (
            f"per-child working copy too expensive: {duplicated / (1024 * 1024):.1f}MB "
            f"duplicated / {count} files (limits {max_mb}MB / {max_files} files)"
        )
        self._emit("root", "merge_disabled", {
            "reason": self._merge_disabled_reason,
            "scope": [p.as_posix() for p in self._merge_scope_roots()] or ["<whole tree>"],
            "shared_clone_roots": [
                p.as_posix() for p in self._merge_shared_clone_roots()
            ],
            "hint": "raise NANOMA_MERGE_MAX_MB / NANOMA_MERGE_MAX_FILES to re-enable",
        })
        return False

    def _merge_target(self) -> Path | None:
        if self.config.workspace_extra_roots:
            p = Path(self.config.workspace_extra_roots[0]).expanduser()
            if p.is_dir():
                return p
        return None

    def _merge_get_lock(self) -> "asyncio.Lock":
        if self._merge_lock is None:
            self._merge_lock = asyncio.Lock()
        return self._merge_lock

    def _merge_root_dir(self) -> Path:
        return self.config.workspace_root / "_merge"

    @staticmethod
    def _merge_ignored_part(name: str) -> bool:
        return name in {
            ".git", ".lake", ".nanoma-runtime-logs", ".nanoma-task-work",
            "__pycache__", "_task_copy", "_merge",
        }

    @staticmethod
    def _merge_clone_skip(name: str) -> bool:
        """Excluded from a child's working copy: runtime artefacts only.

        Narrower than _merge_ignored_part on purpose — build metadata such as
        .git and .lake is never submitted but is often needed to run the task.
        """
        return name in {
            ".nanoma-runtime-logs", ".nanoma-task-work", "__pycache__",
            "_task_copy", "_merge",
        }

    def _merge_shared_clone_roots(self) -> tuple[Path, ...]:
        """Large immutable roots that child worktrees may reference by symlink.

        ``NANOMA_MERGE_SHARED_PATHS`` is an explicit, whitespace/comma-separated
        allowlist relative to the complete task directory. It is intended for
        dependency caches such as Lean's ``.lake/packages`` and for baseline
        trees outside the submission scope. Those trees make a runnable clone
        several gigabytes and hundreds of thousands of files even though the
        agent must never submit or modify them.

        A path is accepted only when it is outside every submitted scope or is
        already excluded from merge diffs (for example, a path below ``.lake``).
        This prevents an accidental setting from sharing editable deliverables.
        """
        raw = self.config.merge_shared_paths
        if raw is None:
            raw = os.environ.get("NANOMA_MERGE_SHARED_PATHS", "")
        if not raw.strip():
            return ()
        scopes = self._merge_scope_roots()
        roots: list[Path] = []
        for token in raw.replace(",", " ").split():
            candidate = Path(token.strip().strip('"').strip("'"))
            if (
                not candidate.parts
                or candidate.is_absolute()
                or ".." in candidate.parts
                or candidate.as_posix() in {"", ".", "./"}
            ):
                continue
            ignored_by_merge = any(
                self._merge_ignored_part(part) for part in candidate.parts
            )
            overlaps_scope = not scopes or any(
                candidate == scope
                or scope in candidate.parents
                or candidate in scope.parents
                for scope in scopes
            )
            if overlaps_scope and not ignored_by_merge:
                continue
            if candidate not in roots:
                roots.append(candidate)
        return tuple(roots)

    def _merge_clone_path_is_shared(self, relative: Path) -> bool:
        """Whether ``relative`` is the root of an explicitly shared subtree."""
        return relative in self._merge_shared_clone_roots()

    def _merge_restore_snapshot(self, snapshot: Path, target: Path) -> None:
        """Overlay a snapshot back onto the submission path, within the scope.

        Deliberately not a copy_tree: that clears the destination first, and the
        destination here holds the harness, the datasets and every child's
        working copy. Restoring by diff touches only the scoped files.
        """
        changed, deleted = self._merge_diff(snapshot, target)
        if changed or deleted:
            self._merge_apply(changed, deleted, target)

    def _merge_copy_tree(self, source: Path, dest: Path) -> bool:
        """Snapshot the merge scope of `source` into a fresh `dest`.

        For snapshots only — `dest` is cleared first, so it must never be the
        live submission path. Use _merge_restore_snapshot to go the other way.
        """
        target = self._merge_target()
        if target is not None and dest.resolve() == target.resolve():
            logger.error(f"refusing to clear the live submission path: {dest}")
            return False
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            roots = self._merge_scope_roots()
            if not roots:
                shutil.copytree(
                    source, dest,
                    ignore=self._fixed_snapshot_ignore, symlinks=True,
                )
                return True
            dest.mkdir(parents=True, exist_ok=True)
            for relative in roots:
                item = source / relative
                if not item.exists() and not item.is_symlink():
                    continue
                self._fixed_copy_snapshot_item(item, dest / relative)
            return True
        except Exception as exc:
            logger.warning(f"merge copy_tree failed ({source} -> {dest}): {exc}")
            return False

    def _merge_ensure_baseline(self) -> Path | None:
        """Snapshot the submit path as the common ancestor for a fan-out round.

        Reused while any merge-child is still active (so siblings diff against
        the same ancestor); re-snapshotted at the start of a fresh round.
        """
        target = self._merge_target()
        if target is None:
            return None
        active = [
            a for a in self.agents.values()
            if getattr(a, "_merge_copy", None)
            and getattr(a, "status", None) not in ("done", "failed", "killed")
        ]
        existing = self._merge_baseline_path
        if existing and existing.is_dir() and active:
            return existing
        if not self._merge_check_copy_budget(target):
            return None
        dest = self._merge_root_dir() / "baseline"
        if not self._merge_copy_tree(target, dest):
            return existing
        self._merge_baseline_path = dest
        self._emit("root", "merge_baseline_snapshot", {"path": str(dest), "source": str(target)})
        return dest

    def _merge_clone_workdir(self, source: Path, dest: Path) -> tuple[int, int] | None:
        """Clone the COMPLETE task directory as a child's working copy.

        Directories are always real, so files the child creates or removes stay
        local (build output, logs, benchmark results). Small files are duplicated
        so every edit a child could plausibly make is isolated; files at or above
        _MERGE_HARDLINK_MIN_BYTES are hardlinked, which keeps the tree complete
        and runnable at nearly zero cost. Explicit immutable cache/dependency
        roots are symlinked back to the common task tree. Returns
        (duplicated_bytes, private_files).
        """
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"merge clone init failed ({dest}): {exc}")
            return None
        duplicated = 0
        files = 0
        for current, dirnames, filenames in os.walk(source):
            relative = Path(current).relative_to(source)
            target_dir = dest / relative
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                dirnames[:] = []
                continue
            private_dirs: list[str] = []
            for name in dirnames:
                if self._merge_clone_skip(name):
                    continue
                child_relative = relative / name
                if self._merge_clone_path_is_shared(child_relative):
                    try:
                        os.symlink(
                            str((source / child_relative).resolve()),
                            target_dir / name,
                            target_is_directory=True,
                        )
                    except Exception as exc:
                        logger.warning(
                            "merge shared-root link failed (%s): %s",
                            child_relative,
                            exc,
                        )
                        return None
                    continue
                private_dirs.append(name)
            dirnames[:] = private_dirs
            for name in filenames:
                if self._merge_clone_skip(name):
                    continue
                src = Path(current) / name
                dst = target_dir / name
                try:
                    if self._merge_clone_path_is_shared(relative / name):
                        os.symlink(str(src.resolve()), dst)
                        continue
                    if src.is_symlink():
                        os.symlink(os.readlink(src), dst)
                        files += 1
                        continue
                    size = src.stat().st_size
                    if size >= _MERGE_HARDLINK_MIN_BYTES:
                        try:
                            os.link(src, dst)
                        except OSError:
                            # different filesystem or link limit: fall back to a copy
                            shutil.copy2(src, dst)
                            duplicated += size
                    else:
                        shutil.copy2(src, dst)
                        duplicated += size
                    files += 1
                except Exception:
                    continue
        return duplicated, files

    def _merge_seed_child_copy(self, agent: Agent) -> Path | None:
        """Give a freshly created child a complete, private, runnable task copy."""
        baseline = self._merge_ensure_baseline()
        if baseline is None:
            return None
        target = self._merge_target()
        if target is None:
            return None
        copy_dir = agent.workspace / "_task_copy"
        cost = self._merge_clone_workdir(target, copy_dir)
        if cost is None:
            return None
        # Rewind just the diffable surface to the round baseline, so the child's
        # diff carries its own work and not drift that landed after the snapshot.
        for relative in self._merge_scope_roots():
            source = baseline / relative
            if not source.exists() and not source.is_symlink():
                continue
            destination = copy_dir / relative
            self._fixed_remove_path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._fixed_copy_snapshot_item(source, destination)
        agent._merge_copy = copy_dir
        self._emit(agent.id, "merge_child_seed", {
            "copy": str(copy_dir), "duplicated_bytes": cost[0], "files": cost[1],
            "shared_clone_roots": [
                path.as_posix() for path in self._merge_shared_clone_roots()
            ],
        })
        return copy_dir

    def _merge_agent_baseline(self, agent: Agent) -> Path | None:
        """What this agent was handed, which is what its diff is measured against.

        Usually the round baseline. It differs once a copy has been refreshed to
        a later state of the submission: from then on the agent's own work is
        what it changed since that refresh, not since the run began.
        """
        own = getattr(agent, "_merge_base", None)
        if own and Path(own).is_dir():
            return Path(own)
        return self._merge_baseline_path

    def _merge_own_changes(self, agent: Agent) -> tuple[dict[str, Path], set[str]]:
        """What this agent has changed in its own copy, or nothing."""
        copy_dir = getattr(agent, "_merge_copy", None)
        baseline = self._merge_agent_baseline(agent)
        if not copy_dir or baseline is None or not Path(copy_dir).is_dir():
            return {}, set()
        try:
            return self._merge_diff(Path(copy_dir), baseline)
        except Exception:
            return {}, set()

    def _merge_rel_files(self, root: Path) -> dict[str, Path]:
        """Files under `root`, restricted to the merge scope."""
        out: dict[str, Path] = {}
        scope = self._merge_scope_roots()
        bases = [root / rel for rel in scope] if scope else [root]
        for base in bases:
            if not base.exists() and not base.is_symlink():
                continue
            if base.is_file() or base.is_symlink():
                rel = base.relative_to(root)
                if not any(self._merge_ignored_part(part) for part in rel.parts):
                    out[rel.as_posix()] = base
                continue
            for p in base.rglob("*"):
                if p.is_dir() and not p.is_symlink():
                    continue
                rel = p.relative_to(root)
                if any(self._merge_ignored_part(part) for part in rel.parts):
                    continue
                out[rel.as_posix()] = p
        return out

    @staticmethod
    def _merge_same_file(a: Path, b: Path) -> bool:
        # Diffs run repeatedly (per promote, per uniqueness check), so avoid
        # hashing large payloads: fall back to size+mtime past a threshold.
        try:
            if a.is_symlink() or b.is_symlink():
                return os.readlink(a) == os.readlink(b)
            sa, sb = a.stat(), b.stat()
            if sa.st_size != sb.st_size:
                return False
            if sa.st_size > _MERGE_HASH_MAX_BYTES:
                # nanosecond mtime: copy2/copytree preserve it exactly, so an
                # untouched file still compares equal while a same-size rewrite
                # is not mistaken for "unchanged".
                return sa.st_mtime_ns == sb.st_mtime_ns
            import hashlib
            return hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()
        except Exception:
            return False

    def _merge_diff(self, copy_dir: Path, baseline: Path) -> tuple[dict[str, Path], set[str]]:
        """(changed_or_added {rel: srcpath}, deleted {rel}) of copy vs baseline."""
        cur = self._merge_rel_files(copy_dir)
        base = self._merge_rel_files(baseline)
        changed = {
            rel: src for rel, src in cur.items()
            if rel not in base or not self._merge_same_file(src, base[rel])
        }
        deleted = set(base) - set(cur)
        return changed, deleted

    def _merge_apply(self, changed: dict[str, Path], deleted: set[str], target: Path) -> None:
        for rel, src in changed.items():
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                self._fixed_remove_path(dst)
            if src.is_symlink():
                os.symlink(os.readlink(src), dst)
            else:
                shutil.copy2(src, dst, follow_symlinks=False)
        for rel in deleted:
            self._fixed_remove_path(target / rel)

    async def _submit_as_root(
        self, args, agent: Agent, runtime, handler, *, aggregate: bool = True
    ) -> Any:
        """Fold in what is still in flight, then gate, then spend the submission.

        An official submission is a statement about the whole workspace, so it
        waits for the work still running and folds it in as one merge. Otherwise
        it scores a state that no check has passed and that nobody has finished
        writing.

        `aggregate=False` skips the wait for a caller reached after the run has
        already drained its deliveries. There, a child still marked running is
        one that was killed and will never deliver, so waiting the full budget
        would spend the end of the wall clock to learn nothing — and the closing
        submission is the one the round is scored on.
        """
        wait = (
            await self._await_outstanding_deliveries(agent)
            if aggregate
            else {"waited": [], "timed_out": False, "holding": {}}
        )
        waited = wait["waited"]
        try:
            await self._merge_promote_pending()
        except Exception as exc:
            self._emit(agent.id, "merge_promote_error", {"detail": str(exc)[:300]})
        blocked = self._submit_incomplete_block(agent, wait)
        if blocked is None:
            blocked = await self._submit_preflight(agent)
        if blocked is not None:
            if waited:
                blocked["waited_for"] = waited
            return blocked
        # Read here, not at the call site: everything above changes the state this
        # submission is about, so the aggregate is what gets sent and judged.
        self._note_state_being_submitted(agent, "submit", args)
        result = await handler(args, agent, runtime)
        if waited:
            result = dict(result) if isinstance(result, dict) else {"result": result}
            result["waited_for"] = waited
        return result

    def _submit_acting_agent(self) -> Agent | None:
        """Who a runtime-initiated submission is attributed to: the root agent."""
        for candidate in self.agents.values():
            if getattr(candidate, "parent", None) in (None, ""):
                return candidate
        return next(iter(self.agents.values()), None)

    def _merge_change_signature(self, changed: dict[str, Path], deleted: set[str]) -> frozenset:
        """Fingerprint of a diff, so an unchanged copy is not promoted twice.

        Content-hashed within the merge scope: same-size edits are common, and a
        cheaper size+mtime key would silently drop a real change.
        """
        import hashlib
        items: set[tuple] = {("-", rel) for rel in deleted}
        for rel, src in changed.items():
            try:
                stat = src.stat()
                if stat.st_size > _MERGE_HASH_MAX_BYTES:
                    items.add((rel, stat.st_size, stat.st_mtime_ns))
                else:
                    items.add((rel, hashlib.md5(src.read_bytes()).hexdigest()))
            except OSError:
                items.add((rel, "unreadable"))
        return frozenset(items)

    def _ledger_path(self) -> Path:
        return self._merge_root_dir() / "ledger.jsonl"

    def _ledger_higher_is_better(self) -> bool:
        """Direction of the judge's score, mirroring `verify`'s own parameter."""
        if self.config.official_lower_is_better is not None:
            return not self.config.official_lower_is_better
        return os.environ.get("NANOMA_OFFICIAL_LOWER_IS_BETTER") != "1"

    @staticmethod
    def _ledger_digest(signature: frozenset | None) -> str | None:
        """A short stable name for a state, so entries can be compared by value."""
        if signature is None:
            return None
        parts = sorted(repr(item) for item in signature)
        return hashlib.md5("\n".join(parts).encode()).hexdigest()[:12]

    def _ledger_snapshot_dir(self, digest: str) -> Path:
        return self._merge_root_dir() / "ledger" / digest

    def _ledger_entries(self) -> list[dict]:
        path = self._ledger_path()
        if not path.is_file():
            return []
        entries = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a torn final line must not hide the rest
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def _ledger_best(self) -> dict | None:
        """The best measurement on record, read from disk rather than memory."""
        scored = [
            e for e in self._ledger_entries()
            if isinstance(e.get("metric"), (int, float)) and e.get("counts", True)
        ]
        if not scored:
            return None
        sign = 1 if self._ledger_higher_is_better() else -1
        return max(scored, key=lambda e: sign * float(e["metric"]))

    def _ledger_note(
        self,
        source: str,
        metric: float | None,
        *,
        signature: frozenset | None,
        agent_id: str = "root",
        counts: bool = True,
        extra: dict | None = None,
    ) -> dict | None:
        """Record one (state, measurement) pair and snapshot a new best state.

        `counts=False` records an observation that must not become a state to
        fall back to — a verdict the judge rejected as invalid still says
        something true about the state, but not that it is worth returning to.
        """
        if metric is None or not self._merge_active():
            return None
        digest = self._ledger_digest(signature)
        entry = {
            "at": time.time(),
            "source": source,
            "metric": float(metric),
            "state": digest,
            "counts": bool(counts),
            "agent": agent_id,
        }
        if extra:
            entry.update(extra)

        previous_best = self._ledger_best()
        try:
            self._ledger_path().parent.mkdir(parents=True, exist_ok=True)
            with self._ledger_path().open("a") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._emit(agent_id, "ledger_error", {"detail": str(exc)[:200]})
            return None

        if counts and digest is not None:
            sign = 1 if self._ledger_higher_is_better() else -1
            improved = previous_best is None or (
                sign * float(metric) > sign * float(previous_best.get("metric", 0.0))
            )
            if improved:
                self._ledger_keep_state(digest, metric, agent_id)
        self._emit(agent_id, "ledger_note", {
            "source": source, "metric": metric, "state": digest, "counts": counts,
        })
        return entry

    def _ledger_keep_state(self, digest: str, metric: float, agent_id: str) -> None:
        """Snapshot the state a measurement applies to, so it can be restored.

        Only the best is kept. This is reached once per improvement, so keeping
        each one would grow the workspace by a full copy of the submission scope
        every time the run got better — up to NANOMA_MERGE_MAX_MB apiece, inside a
        container with a fixed disk. Superseded states stay on the record as
        numbers; `experiments` reports them as measured but not restorable.
        """
        target = self._merge_target()
        if target is None or not self._merge_snapshot_affordable(target):
            return
        destination = self._ledger_snapshot_dir(digest)
        if destination.is_dir():
            return  # this state is already kept; measuring it twice changes nothing
        if not self._merge_copy_tree(target, destination):
            return
        self._emit(agent_id, "ledger_snapshot", {"state": digest, "metric": metric})
        for stale in destination.parent.iterdir():
            if stale.is_dir() and stale.name != digest:
                shutil.rmtree(stale, ignore_errors=True)

    def _ledger_restore(self, digest: str) -> bool:
        """Put a recorded state back into the submission path."""
        source = self._ledger_snapshot_dir(digest)
        target = self._merge_target()
        if target is None or not source.is_dir():
            return False
        changed, deleted = self._merge_diff(source, target)
        if changed or deleted:
            self._merge_apply(changed, deleted, target)
        self._emit("root", "ledger_restored", {
            "state": digest, "changed": len(changed), "deleted": len(deleted),
        })
        return True

    def _ledger_state_metrics(
        self, digest: str | None, source: str | None = None
    ) -> list[float]:
        """Every metric recorded for one state, optionally from one channel only.

        Channels are not interchangeable: an official score and a local check's
        number measure different things on different scales, so pooling them
        would compare quantities that were never comparable.
        """
        if digest is None:
            return []
        return [
            float(e["metric"])
            for e in self._ledger_entries()
            if e.get("state") == digest
            and isinstance(e.get("metric"), (int, float))
            and (source is None or e.get("source") == source)
        ]

    def _measurement_spread(self, source: str) -> tuple[float, float] | None:
        """Scatter of repeat measurements of a single unchanged state.

        Returns `(relative, absolute)` — a coefficient of variation and a
        standard deviation in the metric's own units. Both, because neither
        alone survives every metric a task might report: a relative figure is
        meaningless for a metric that sits near or crosses zero (a delta, a
        margin, a signed error), while an absolute one measured on states of one
        magnitude under-protects states of another.

        Estimated from this run's own record rather than assumed, because the
        scatter is a property of the task's measurement and cannot be known in
        advance. Returns None until some state has been measured twice. A
        deterministic check measured twice gives (0, 0), which restores exact
        comparison — that is the intended answer, not a degenerate one.
        """
        by_state: dict[str, list[float]] = {}
        for entry in self._ledger_entries():
            metric, state = entry.get("metric"), entry.get("state")
            if entry.get("source") != source or state is None:
                continue
            if isinstance(metric, (int, float)):
                by_state.setdefault(state, []).append(float(metric))

        relative, absolute = [], []
        for metrics in by_state.values():
            if len(metrics) < 2:
                continue
            mean = sum(metrics) / len(metrics)
            variance = sum((m - mean) ** 2 for m in metrics) / (len(metrics) - 1)
            sigma = variance**0.5
            absolute.append(sigma)
            if abs(mean) > sigma:
                # Only where a ratio means something. A state whose repeats
                # straddle zero has a mean smaller than its own scatter, and its
                # coefficient of variation is an artefact of dividing by nearly
                # nothing — 424% for repeats of -0.01 and 0.02. Letting that into
                # the estimate would set a margin no regression could clear.
                relative.append(sigma / abs(mean))
        if not absolute:
            return None
        return (
            sum(relative) / len(relative) if relative else 0.0,
            sum(absolute) / len(absolute),
        )

    def _measurement_noise(self, source: str) -> float | None:
        """The relative half of the scatter, for reporting."""
        spread = self._measurement_spread(source)
        return None if spread is None else spread[0]

    def _noise_margin(self, source: str, at: float = 1.0) -> float | None:
        """The gap a difference must clear at magnitude `at`, in metric units."""
        spread = self._measurement_spread(source)
        if spread is None:
            return None
        try:
            k = float(
                self.config.noise_margin_sigmas
                if self.config.noise_margin_sigmas is not None
                else os.environ.get("NANOMA_NOISE_MARGIN_SIGMAS", "2") or 2
            )
        except ValueError:
            k = 2.0
        relative, absolute = spread
        # Whichever regime is worse at this magnitude. Multiplicative noise
        # dominates for a large value, additive for a small one, and taking the
        # larger keeps the margin honest without having to decide which the
        # task's measurement is.
        return k * max(relative * abs(at), absolute)

    @staticmethod
    def _official_parse(text: str) -> dict | None:
        """Read the judge's verdict out of what sforge-submit printed.

        This is the feedback channel the harness gives the agent — score, pass
        rate and the names of what failed are printed for it to read. The
        host-side auto-evals are a separate, admin-only view and are none of our
        business.
        """
        if not text or "Results" not in text:
            return None
        pass_rate = None
        m = re.search(r"Pass rate:\s+([\d.]+)%", text)
        if m:
            pass_rate = float(m.group(1)) / 100.0
        passed = re.search(r"Passed:\s+(\d+)/(\d+)", text)
        if pass_rate is None and passed and int(passed.group(2)):
            pass_rate = int(passed.group(1)) / int(passed.group(2))
        if pass_rate is None and "All tests passed!" in text:
            pass_rate = 1.0
        if pass_rate is None:
            return None
        score = None
        m = re.search(r"Score:\s+([-\d.]+)", text)
        if m:
            try:
                score = float(m.group(1))
            except ValueError:
                score = None
        if passed is None and score is None:
            # A pass rate with no tally and no score is not a verdict — reading a
            # fragment as one produced "invalid, 100% passed", which would have
            # blamed a check for a rejection nobody made.
            return None
        failed = re.findall(r"^\s+- (.+)$", text, flags=re.MULTILINE)
        round_id = None
        m = re.search(r"^\s+(\S+) Results\s*$", text, flags=re.MULTILINE)
        if m:
            round_id = m.group(1)
        return {
            "valid": "Valid:       no" not in text and "Valid: no" not in text,
            "pass_rate": pass_rate,
            "score": score,
            "failed": [f.strip() for f in failed][:20],
            "round": round_id,
        }

    def _merge_live_children(self, exclude: str = "") -> list[Agent]:
        return [
            a for a in self.agents.values()
            if a.id != exclude
            and getattr(a, "status", None) not in ("done", "failed", "killed")
            and getattr(a, "_merge_copy", None)
            and Path(a._merge_copy).is_dir()
        ]

    def _merge_refresh_idle_copies(self, exclude: str = "") -> list[str]:
        """Move copies that have nothing of their own at stake up to the submission.

        Eligibility is read off the copy rather than handed out as a role: a child
        that has not written anything loses nothing by being moved forward, and
        the moment it does write something it stops being refreshed and delivers
        its work like anyone else. That keeps an agent from reasoning about a
        state that no longer exists — measuring work that has since been replaced,
        or reporting on files nobody is going to ship.

        Called while holding the merge lock, so nobody is shown a half-applied tree.
        """
        target = self._merge_target()
        refreshed: list[str] = []
        if target is None:
            return refreshed
        for child in self._merge_live_children(exclude):
            changed, deleted = self._merge_own_changes(child)
            if changed or deleted:
                continue
            try:
                # The new baseline is taken first: a copy moved forward without
                # one reads as having changed everything the delivery brought,
                # and would re-deliver other agents' work as its own.
                base = self._merge_rebase_child(child, target)
                if base is None:
                    continue
                self._merge_restore_snapshot(target, Path(child._merge_copy))
                child._merge_base = base
                refreshed.append(child.id)
            except Exception as exc:
                self._emit(child.id, "merge_refresh_error", {"detail": str(exc)[:200]})
        return refreshed

    def _merge_rebase_child(self, agent: Agent, source: Path) -> Path | None:
        """Snapshot what an agent is being handed, to measure its own work against."""
        if not self._merge_snapshot_affordable(source):
            return None
        base = self._merge_root_dir() / "base" / agent.id
        return base if self._merge_copy_tree(source, base) else None

    async def _merge_promote_child(self, agent: Agent, reason: str = "") -> dict | None:
        """Fold a child's diff into the submission path, gated by a re-run check.

        Serialized by a lock. The child's changes are computed against the
        immutable round baseline and applied onto the (accumulating) submission
        path — and then the runtime runs the registered verification against
        that merged result. Not against the child's own copy: every party
        verifying only its own artifact is exactly how a config merged from
        three children replaced a working solution unmeasured. A merged state
        that fails, or verifies worse than the best so far, is rolled back.
        """
        if not self._merge_active():
            return None
        copy_dir = getattr(agent, "_merge_copy", None)
        baseline = self._merge_agent_baseline(agent)
        target = self._merge_target()
        if not copy_dir or not Path(copy_dir).is_dir() or baseline is None or target is None:
            return None
        changed, deleted = self._merge_diff(Path(copy_dir), baseline)
        if not changed and not deleted:
            self._emit(agent.id, "merge_promote_skip", {"reason": "no_changes"})
            return {
                "status": "nothing_to_deliver",
                "detail": f"No changes under the submission paths in {copy_dir}.",
            }
        signature = self._merge_change_signature(changed, deleted)
        if signature == getattr(agent, "_merge_promoted_sig", None):
            self._emit(agent.id, "merge_promote_skip", {"reason": "unchanged_since_last"})
            return {
                "status": "unchanged_since_last_delivery",
                "metric": self._merge_last_child_metric.get(agent.id),
                "detail": "Nothing changed since your last delivery; edit files first.",
            }
        spec = self._verification_spec_for(agent)
        async with self._merge_get_lock():
            prev = self._merge_root_dir() / "prev"
            have_prev = self._merge_copy_tree(target, prev)
            self._merge_apply(changed, deleted, target)
            verification = None
            rank = None
            if spec is not None:
                verification = await self._verify_submission_path(agent, spec)
                rank = self._verify_rank(
                    verification["ok"], verification["metric"],
                    bool(spec.get("higher_is_better", True)),
                )
            best_rank = self._merge_best_rank
            proven = self._verify_metric_is_proven(spec) if spec else False
            if verification is not None and verification.get("metric") is not None:
                # Recorded so this channel can estimate its own scatter. Never
                # counts towards the official best: a local check's number is not
                # the judge's verdict and the two are not on one scale.
                self._ledger_note(
                    "verify",
                    verification["metric"],
                    signature=self._merge_scope_signature(target),
                    agent_id=agent.id,
                    counts=False,
                    extra={"ok": bool(verification["ok"])},
                )
            keep = True
            if rank is not None:
                # A merged state that fails its check is never kept, including
                # when nothing better has been measured yet. Defaulting to keep
                # in that case let four failing merges through in one round and
                # then snapshotted the last of them as the state to fall back to.
                if not verification["ok"]:
                    keep = False
                elif best_rank is not None and proven and rank < best_rank:
                    # Worse by this check — but a gap smaller than what
                    # re-measuring the same state would move it is not a gap.
                    #
                    # Unlike the submission gate, an unknown scatter here falls
                    # back to the strict comparison. Each promotion verifies a
                    # different merged state, so this check rarely measures one
                    # state twice and a margin may never become estimable; being
                    # inert until it did would mean keeping every broken merge,
                    # which is the failure the rollback exists for. The costs are
                    # not symmetric either: a wrong rollback loses one delivery,
                    # which `prev` still holds, while a wrong refusal at the
                    # submission gate loses the round's score.
                    best_state = self._ledger_digest(self._merge_best_signature)
                    worse = self._metric_is_worse(
                        [verification["metric"]] if verification["metric"] is not None else [],
                        self._ledger_state_metrics(best_state, "verify")
                        or ([self._merge_best_metric] if self._merge_best_metric is not None else []),
                        "verify",
                    )
                    keep = worse is False
                    if keep:
                        self._emit(agent.id, "merge_keep_within_noise", {
                            "metric": verification["metric"],
                            "best": self._merge_best_metric,
                            "noise": self._measurement_noise("verify"),
                        })
            if keep and rank is not None and verification["ok"]:
                self._record_verified_best(verification["metric"], rank)
            if not keep and have_prev:
                self._merge_restore_snapshot(prev, target)
            agent._merge_promoted_sig = signature
            metric = verification["metric"] if verification else None
            self._merge_last_child_metric[agent.id] = metric
            refreshed = self._merge_refresh_idle_copies(exclude=agent.id)
            self._emit(agent.id, "merge_promote", {
                "changed": len(changed), "deleted": len(deleted),
                "verified": bool(verification and verification["ok"]),
                "metric": metric, "best": self._merge_best_metric, "kept": keep,
                "metric_proven": proven, "copies_refreshed": refreshed,
            })
        await self._notify_peers_of_delivery(
            agent, sorted(changed), keep,
            bool(verification and verification["ok"]), metric, refreshed,
        )
        if verification is None:
            note = (
                "No verification is registered, so this was merged unchecked and can "
                "never be treated as the best state. Register one with the verify tool "
                "so the runtime can prove a merged result still works."
            )
        elif keep:
            note = "Verified against the merged submission and kept."
        else:
            note = (
                "The merged submission verified worse than the best state, so it was "
                "rolled back. Your working copy is untouched — improve it and retry."
            )
        return {
            "status": "delivered",
            "verified": bool(verification and verification["ok"]),
            "metric": metric,
            "best_metric": self._merge_best_metric,
            "kept": keep,
            "metric_proven": proven,
            "changed_files": sorted(changed)[:20],
            "deleted_files": sorted(deleted)[:20],
            "note": note,
            "verification_output": (verification or {}).get("output", "")[-1500:],
        }

    def _merge_workspace_guard_segment(self, target: Path) -> str | None:
        """First path segment of the runtime workspace when it lives inside the
        task directory (EdgeBench puts it at task_cwd/.nanoma-task-work)."""
        try:
            relative = self.config.workspace_root.resolve().relative_to(target.resolve())
        except (ValueError, OSError):
            return None
        return relative.parts[0] if relative.parts else None

    def _merge_redirect_tool_args(self, agent: Agent, tc: "ToolCall") -> None:
        """Rewrite shared-task-directory paths to a merge child's private copy.

        Isolation cannot be advisory. The harness prompt names the shared task
        directory repeatedly ("cd {task_cwd} && ..."), children follow it over a
        single line of task text, and every sibling lands back on the same files
        — observed on ann_vector_search_qps, where four children edited the
        shared directory and their private copies stayed untouched. Rewriting
        the paths that filesystem tools actually receive makes the copy the only
        thing a child can reach, without needing container privileges or model
        compliance.
        """
        # Deferred: core imports this module, so the tool-name vocabulary can only
        # be read back at call time. Moving those constants to a neutral module
        # would let this become a top-level import.
        from nanoma.core import _SHELL_TOOLS

        copy_dir = getattr(agent, "_merge_copy", None)
        if not copy_dir or not self._merge_active():
            return
        if not (tc.name in _SHELL_TOOLS or tc.name.startswith("ws_")):
            return
        target = self._merge_target()
        if target is None:
            return
        source = str(target).rstrip("/")
        destination = str(copy_dir).rstrip("/")
        if not source or source == destination:
            return

        import re
        # The copy lives under the workspace, which sits inside the task
        # directory, so a naive replace would corrupt paths that already point
        # at the copy. Only rewrite references that leave the workspace alone.
        guard = self._merge_workspace_guard_segment(target)
        pattern = re.escape(source) + (rf"(?!/{re.escape(guard)}\b)" if guard else "")
        hits = 0

        def rewrite(value):
            nonlocal hits
            if isinstance(value, str):
                if source not in value:
                    return value
                new_value, count = re.subn(pattern, destination, value)
                hits += count
                return new_value
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            if isinstance(value, dict):
                return {k: rewrite(v) for k, v in value.items()}
            return value

        if not isinstance(tc.arguments, dict):
            return
        tc.arguments = rewrite(tc.arguments)
        if hits:
            seen = getattr(agent, "_merge_redirects", 0) + hits
            agent._merge_redirects = seen
            if seen <= 3:
                self._emit(agent.id, "merge_path_redirect", {
                    "tool": tc.name, "from": source, "to": destination,
                })

    def _merge_best_dir(self) -> Path:
        return self._merge_root_dir() / "best"

    def _merge_scope_signature(self, root: Path) -> frozenset:
        files = self._merge_rel_files(root)
        return self._merge_change_signature(files, set())

    def _merge_snapshot_affordable(self, target: Path) -> bool:
        max_mb, max_files = self._merge_size_caps()
        total, count = self._merge_scope_stats(target)
        return not (
            (max_mb > 0 and total > max_mb * 1024 * 1024)
            or (max_files > 0 and count > max_files)
        )

    def _merge_restore_best(self) -> None:
        """Put the best verified state back before the run hands off.

        Restores when the live state verified worse than the best, and also when
        it was edited after its last verification: an unchecked edit is worth
        less than a state that is known to work.
        """
        if not self._merge_active() or self._merge_best_rank is None:
            return
        best_dir = self._merge_best_dir()
        target = self._merge_target()
        if target is None or not best_dir.is_dir():
            return
        current = self._merge_current_rank
        if current is not None and self._merge_scored_signature is not None:
            if self._merge_scope_signature(target) != self._merge_scored_signature:
                current = None  # edited since it was last measured
        if current is not None and current >= self._merge_best_rank:
            return
        changed, deleted = self._merge_diff(best_dir, target)
        if not changed and not deleted:
            return
        self._merge_apply(changed, deleted, target)
        self._emit("root", "merge_best_restored", {
            "best": self._merge_best_metric,
            "replaced": self._merge_current_metric if current is not None else None,
            "changed": len(changed),
            "deleted": len(deleted),
        })

    async def _merge_promote_pending(self) -> None:
        """Fold in children that still hold unpromoted work, verified once.

        A child promotes from its own agent task, which shutdown cancels, so
        anything a child was still working on when the root finished is dropped
        — it only used to survive because children wrote the shared directory
        directly. Applied as a single union and verified once: the union is a
        combination nobody has ever run, which is precisely the artifact that
        needs checking, and if it fails the keep-best restore rewinds it.
        """
        if not self._merge_active():
            return
        baseline = self._merge_baseline_path
        target = self._merge_target()
        if baseline is None or target is None:
            return
        union_changed: dict[str, Path] = {}
        union_deleted: set[str] = set()
        contributors: list[str] = []
        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            copy_dir = getattr(agent, "_merge_copy", None)
            if not copy_dir or not Path(copy_dir).is_dir():
                continue
            changed, deleted = self._merge_diff(
                Path(copy_dir), self._merge_agent_baseline(agent) or baseline
            )
            if not changed and not deleted:
                continue
            if self._merge_change_signature(changed, deleted) == getattr(
                agent, "_merge_promoted_sig", None
            ):
                continue
            union_changed.update(changed)
            union_deleted |= deleted
            contributors.append(agent.id)
        if not contributors:
            return
        union_deleted -= set(union_changed)
        root = next((a for a in self.agents.values() if a.parent is None), None)
        actor = root or self.agents[contributors[0]]
        spec = self._verification_spec_for(actor)
        async with self._merge_get_lock():
            self._merge_apply(union_changed, union_deleted, target)
            verification = None
            if spec is not None:
                verification = await self._verify_submission_path(actor, spec)
                rank = self._verify_rank(
                    verification["ok"], verification["metric"],
                    bool(spec.get("higher_is_better", True)),
                )
                if rank is not None:
                    self._record_verified_best(verification["metric"], rank)
        self._emit("root", "merge_promote_pending", {
            "contributors": contributors,
            "changed": len(union_changed),
            "deleted": len(union_deleted),
            "verified": bool(verification and verification["ok"]),
            "metric": (verification or {}).get("metric"),
        })

    def _submit_preflight_command(self) -> str:
        if self.config.submit_preflight_command is not None:
            return self.config.submit_preflight_command.strip()
        return (os.environ.get("NANOMA_SUBMIT_PREFLIGHT") or "").strip()

    def _submit_requires_verified_state(self) -> bool:
        if self.config.submit_require_verified is not None:
            return self.config.submit_require_verified
        return os.environ.get("NANOMA_SUBMIT_REQUIRE_VERIFIED") == "1"

    def _submit_requires_measured_state(self) -> bool:
        """Whether a submission may ship a state worse than one on record.

        Separate from NANOMA_SUBMIT_REQUIRE_VERIFIED because it rests on
        different evidence: that gate needs the agent to have authored a check
        that passes, this one needs only a measurement the run has already seen.
        A run whose check never works is exactly the run that needs this.

        On by default, but only where the ledger can hold anything: without a
        merge target there is no state to fingerprint and no verdict to attach to
        one, so leaving it enabled there would put a gate in front of every
        submission that could never have evidence to act on.
        """
        if not self._merge_active():
            return False
        if self.config.submit_require_measured is not None:
            return self.config.submit_require_measured
        return os.environ.get("NANOMA_SUBMIT_REQUIRE_MEASURED", "1") == "1"

    def _submit_requires_complete_aggregate(self) -> bool:
        """Whether a timed-out aggregate wait may still spend a submission."""
        if self.config.submit_require_complete_aggregate is not None:
            return self.config.submit_require_complete_aggregate
        return os.environ.get("NANOMA_SUBMIT_REQUIRE_COMPLETE", "1") == "1"

    def _submit_gate_enabled(self) -> bool:
        return (
            bool(self._submit_preflight_command())
            or self._submit_requires_verified_state()
            or self._submit_requires_measured_state()
        )

    async def _run_preflight(self, command: str, workdir: Path) -> tuple[int | None, str]:
        """Run a submission preflight; the exit code is the whole verdict.

        Unlike a `verify` check this prints nothing the runtime parses — it is a
        sanity condition the task imposes, such as the submitted algorithm still
        being discoverable under the name the judge runs.
        """
        override = self.config.submit_preflight_timeout
        if override is None:
            raw = os.environ.get("NANOMA_SUBMIT_PREFLIGHT_TIMEOUT")
            override = raw if raw else self.PREFLIGHT_TIMEOUT_DEFAULT
        try:
            timeout = float(override)
        except ValueError:
            timeout = self.PREFLIGHT_TIMEOUT_DEFAULT
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workdir),
                env={**os.environ, "WORKSPACE": str(workdir)},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:
            return -1, f"{type(exc).__name__}: {exc}"
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None, f"preflight timed out after {timeout:.0f}s"
        return proc.returncode, stdout.decode(errors="replace")[-2000:]

    def _submit_incomplete_block(self, agent: Agent, wait: dict) -> dict | None:
        """Stop a timed-out aggregate wait from silently becoming a submission.

        The wait exists so a submission describes the whole workspace rather than
        "a state that no check has passed and that nobody has finished writing" —
        and when it timed out, that is exactly what got sent, because the timeout
        was invisible at the call site. Waiting out the full budget read to the
        agent as permission to go.

        Refused once, then allowed. A child can hang for the rest of the run, and
        a gate that refuses forever would turn that into a zero — the cost of
        being wrong here is the whole score. So the first attempt is refused with
        the names, the files and the ways out; submitting again proceeds. The
        point is to make the choice informed, not to make it for the agent.
        """
        if not self._submit_requires_complete_aggregate():
            return None
        if not wait.get("timed_out"):
            agent._aggregate_wait_refused = False
            return None
        holding = wait.get("holding") or {}
        if not holding:
            return None  # outstanding, but holding nothing: nothing is being lost
        if getattr(agent, "_aggregate_wait_refused", False):
            self._emit(agent.id, "submit_incomplete_allowed", {"holding": holding})
            return None

        agent._aggregate_wait_refused = True
        children = sorted(holding)
        reason = (
            f"the wait for {children} ran out after "
            f"{self._AGGREGATE_WAIT_SECONDS:.0f}s and they are still holding changes "
            f"that belong in this submission: {json.dumps(holding)}. Submitting now "
            "sends a state that is knowably unfinished. Either kill them and take "
            "what has already merged, message them for what they have, or submit "
            "again to go ahead without it — this refusal is not repeated."
        )
        self._emit(agent.id, "submit_blocked", {
            "reason": "aggregate_incomplete", "holding": holding,
            "waited_for": wait.get("waited"),
        })
        return {"submitted": False, "blocked": "aggregate_incomplete", "reason": reason}

    def _submit_regression_block(self, agent: Agent) -> dict | None:
        """Refuse to ship a state that is worse than one already measured.

        The keep-best ratchet restores a better state only at the end of the run
        and only when the agent's own check produced it. Nothing stopped a
        submission from being spent, mid-run, on a state the run had already
        measured as worse — or on one nobody had measured while a better one sat
        on record. Both happened: one run shrank the scored config to fit a local
        timeout, deleting the parameters behind its own best score, and spent its
        remaining submissions on the smaller one.

        Only refuses on evidence: without a recorded measurement there is
        nothing to be worse than, and a state that is not worse beyond the
        measurement noise passes. It used to compare this state's best single
        number against the record's best single number, which on a task whose
        repeat measurements vary by 16% meant refusing states that were fine and
        passing states that were worse, both at around chance.
        """
        if not self._submit_requires_measured_state():
            return None
        best = self._ledger_best()
        target = self._merge_target()
        if best is None or target is None or not best.get("state"):
            return None

        current = self._ledger_digest(self._merge_scope_signature(target))
        if current == best["state"]:
            return None

        best_metric = float(best["metric"])
        source = str(best.get("source") or "official")
        here = self._ledger_state_metrics(current, source)
        there = self._ledger_state_metrics(best["state"], source) or [best_metric]
        sign = 1 if self._ledger_higher_is_better() else -1
        if here and self._metric_is_worse(here, there, source) is not True:
            # Measured, and not worse by more than the noise. That includes the
            # case where the run cannot yet estimate its noise: a difference it
            # cannot distinguish from measurement scatter is not grounds to
            # spend nothing, and refusing here costs the whole score.
            return None

        measured = (
            f"this state measured {max(here, key=lambda m: sign * m):g}"
            if here
            else "this state has never been measured"
        )
        restorable = self._ledger_snapshot_dir(best["state"]).is_dir()
        reason = (
            f"a better state is already on record: {best_metric:g}"
            + (f" (round {best['round']})" if best.get("round") else "")
            + f", while {measured}. Spending a submission here would ship the worse "
            "of the two."
            + (
                f" The recorded state is kept at {self._ledger_snapshot_dir(best['state'])} "
                "— restore it, or measure this one and beat it."
                if restorable
                else " Measure this state before spending a submission on it."
            )
        )
        self._emit(agent.id, "submit_blocked", {
            "reason": "worse_than_measured",
            "best_metric": best_metric,
            "best_state": best["state"],
            "current_state": current,
            "current_metrics": here,
            "noise": self._measurement_noise(source),
            "margin": self._noise_margin(source, at=best_metric),
            "restorable": restorable,
        })
        return {"submitted": False, "blocked": "worse_than_measured", "reason": reason}

    async def _submit_preflight(self, agent: Agent) -> dict | None:
        """Why this submission must not be spent, or None to let it through.

        Deliberately inert when there is nothing to judge with: a run that
        registered no working check must still be able to submit, or a broken
        gate turns into a zero. It only refuses when the evidence to refuse on
        actually exists.
        """
        if not self._submit_gate_enabled():
            return None

        if self._submit_requires_verified_state():
            spec = self._authoritative_spec()
            target = self._merge_target()
            signature = self._merge_scope_signature(target) if target else None
            if spec is not None and signature is not None and signature != getattr(
                self, "_merge_scored_signature", None
            ):
                reason = (
                    "no check has passed the workspace as it stands. The registered "
                    f"check ({spec.get('name') or 'unnamed'}) has not been run against "
                    "this state, so the submission would score something nobody has "
                    "measured. Run `verify` on it first; if it fails, fix the state "
                    "instead of spending a submission on it."
                )
                self._emit(agent.id, "submit_blocked", {
                    "reason": "unverified_state", "check": spec.get("name"),
                })
                return {"submitted": False, "blocked": "unverified_state", "reason": reason}

        regression = self._submit_regression_block(agent)
        if regression is not None:
            return regression

        command = self._submit_preflight_command()
        if command:
            workdir = self._merge_target() or agent.workspace
            code, output = await self._run_preflight(command, Path(workdir))
            if code != 0:
                tail = " ".join((output or "").split())[-600:]
                self._emit(agent.id, "submit_blocked", {
                    "reason": "preflight_failed", "exit_code": code,
                    "output_tail": tail[-400:],
                })
                return {
                    "submitted": False,
                    "blocked": "preflight_failed",
                    "reason": (
                        "the submission preflight failed, so the judge would reject "
                        f"this state before scoring it. `{command}` exited {code}: "
                        f"{tail}"
                    ),
                }
        return None

    def _merge_wrap_submit(self, agent: Agent, tools: dict[str, dict]) -> dict[str, dict]:
        """Wrap `submit`: deliver a child's own work first, and gate what ships.

        A child works in a private copy, so a raw submit would send the parent's
        state to the judge — a number about somebody else's work. Its changes
        are merged and verified first; whether to then spend an official
        submission stays the agent's own call.

        The gate runs last, immediately before the submission is spent, so it
        judges the state that will actually be scored rather than the one that
        existed before the merge.
        """
        if "submit" not in tools:
            return tools
        original = tools["submit"]

        if not self._merge_active():
            if not self._submit_gate_enabled():
                return tools

            async def gated_submit(args, ag, runtime):
                blocked = await self._submit_preflight(ag)
                if blocked is not None:
                    return blocked
                return await original["handler"](args, ag, runtime)

            wrapped_plain = dict(original)
            wrapped_plain["handler"] = gated_submit
            patched_plain = dict(tools)
            patched_plain["submit"] = wrapped_plain
            return patched_plain

        is_child = bool(getattr(agent, "_merge_copy", None))
        if not is_child:
            async def aggregate_then_submit(args, ag, runtime):
                return await self._submit_as_root(args, ag, runtime, original["handler"])

            wrapped_parent = dict(original)
            wrapped_parent["handler"] = aggregate_then_submit
            patched_parent = dict(tools)
            patched_parent["submit"] = wrapped_parent
            return patched_parent

        async def merge_submit(args, ag, runtime):
            delivery = await self._merge_promote_child(
                ag, reason=str((args or {}).get("reason") or f"child {ag.id} submit")
            )
            if delivery is None:
                blocked = await self._submit_preflight(ag)
                if blocked is not None:
                    return blocked
                self._note_state_being_submitted(ag, "submit", args)
                return await original["handler"](args, ag, runtime)
            if not delivery.get("kept"):
                return delivery
            blocked = await self._submit_preflight(ag)
            if blocked is not None:
                return {"delivery": delivery, "submission": blocked}
            # After the child's own work went in, so the verdict is filed against
            # the state that was sent rather than the one that preceded it.
            self._note_state_being_submitted(ag, "submit", args)
            result = await original["handler"](args, ag, runtime)
            return {"delivery": delivery, "submission": result}

        wrapped = dict(original)
        wrapped["handler"] = merge_submit
        patched = dict(tools)
        patched["submit"] = wrapped
        return patched

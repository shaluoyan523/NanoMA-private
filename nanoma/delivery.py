"""Runtime-owned delivery contracts and atomic artifact publication.

Benchmarks describe *where* their final artifact tree must live and which
alternative tree layouts agents may produce.  The runtime owns the last mile:
it finds a complete candidate tree, stages it beside the official destination,
and swaps it into place only after every required file is present.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


def _relative_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {value}")
    if path.as_posix() in {"", "."}:
        raise ValueError(f"{label} cannot be empty")
    return path


@dataclass(frozen=True)
class DeliveryTree:
    """One directory tree that must be published as a unit.

    ``target`` is relative to :class:`DeliveryContract.target_root`.
    ``candidates`` are relative to candidate workspace roots.  ``required``
    paths are relative to the candidate/target tree and must be non-empty.
    """

    target: str
    candidates: tuple[str, ...]
    required: tuple[str, ...]

    def __post_init__(self) -> None:
        _relative_path(self.target, label="delivery target")
        if not self.candidates:
            raise ValueError("delivery candidates cannot be empty")
        if not self.required:
            raise ValueError("delivery required paths cannot be empty")
        for value in self.candidates:
            _relative_path(value, label="delivery candidate")
        for value in self.required:
            _relative_path(value, label="delivery required path")


@dataclass(frozen=True)
class DeliveryContract:
    """A benchmark's runtime-enforced final artifact contract."""

    target_root: Path
    trees: tuple[DeliveryTree, ...]
    block_done: bool = True
    auto_publish: bool = True

    def __post_init__(self) -> None:
        if not self.trees:
            raise ValueError("delivery contract must contain at least one tree")
        object.__setattr__(self, "target_root", Path(self.target_root).expanduser())

    def required_targets(self) -> tuple[Path, ...]:
        return tuple(
            self.target_root / tree.target / required
            for tree in self.trees
            for required in tree.required
        )


@dataclass
class DeliveryReport:
    ready: bool
    trigger: str
    published: list[dict] = field(default_factory=list)
    satisfied: list[dict] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    checked_candidates: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "trigger": self.trigger,
            "published": list(self.published),
            "satisfied": list(self.satisfied),
            "missing": list(self.missing),
            "checked_candidates": list(self.checked_candidates),
        }


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = Path(path).expanduser()
        try:
            key = str(expanded.resolve())
        except OSError:
            key = str(expanded.absolute())
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _has_required(tree_root: Path, required: Sequence[str]) -> bool:
    for raw in required:
        path = tree_root / _relative_path(raw, label="delivery required path")
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _path_endswith(path: Path, suffix: Path) -> bool:
    if len(path.parts) < len(suffix.parts):
        return False
    return path.parts[-len(suffix.parts) :] == suffix.parts


def _explicit_tree_candidates(
    explicit: Path,
    tree: DeliveryTree,
) -> list[Path]:
    """Recover a candidate tree from a submitted file or directory."""

    candidates: list[Path] = []
    path = explicit.expanduser()
    if path.is_dir():
        candidates.append(path)
        for raw in tree.candidates:
            nested = path / _relative_path(raw, label="delivery candidate")
            if nested.is_dir():
                candidates.append(nested)
    elif path.is_file():
        required_names = {
            _relative_path(raw, label="delivery required path").name
            for raw in tree.required
        }
        if path.name in required_names:
            candidates.append(path.parent)

    start = path if path.is_dir() else path.parent
    for parent in (start, *start.parents):
        for raw in tree.candidates:
            suffix = _relative_path(raw, label="delivery candidate")
            if _path_endswith(parent, suffix):
                candidates.append(parent)
    return _deduplicate_paths(candidates)


def _atomic_overlay_tree(source: Path, target: Path, required: Sequence[str]) -> None:
    """Overlay ``source`` onto ``target`` and replace the target atomically."""

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.nanoma-stage-", dir=target.parent))
    backup: Path | None = None
    try:
        if target.exists():
            if not target.is_dir():
                raise RuntimeError(f"delivery target is not a directory: {target}")
            shutil.copytree(target, stage, dirs_exist_ok=True, symlinks=True)
        shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=True)
        if not _has_required(stage, required):
            raise RuntimeError(f"staged delivery is incomplete: {stage}")

        if target.exists():
            backup = target.with_name(f".{target.name}.nanoma-backup-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        os.replace(stage, target)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
            backup = None
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def publish_delivery_contract(
    contract: DeliveryContract,
    *,
    candidate_bases: Sequence[Path],
    explicit_paths: Sequence[Path] = (),
    trigger: str,
) -> DeliveryReport:
    """Satisfy a delivery contract from complete candidate trees.

    Explicit submissions have priority, followed by benchmark/runtime-provided
    candidate bases.  A tree is never partially published: all of its required
    files must exist before staging begins and again before the atomic swap.
    """

    report = DeliveryReport(ready=True, trigger=trigger)
    bases = _deduplicate_paths(candidate_bases)
    explicit = _deduplicate_paths(explicit_paths)

    for tree in contract.trees:
        target = contract.target_root / _relative_path(tree.target, label="delivery target")
        if _has_required(target, tree.required) and not explicit:
            report.satisfied.append({
                "target": str(target),
                "source": str(target),
                "published": False,
            })
            continue

        candidates: list[Path] = []
        for path in explicit:
            candidates.extend(_explicit_tree_candidates(path, tree))
        for base in bases:
            for raw in tree.candidates:
                candidates.append(base / _relative_path(raw, label="delivery candidate"))
        candidates = _deduplicate_paths(candidates)
        report.checked_candidates.extend(str(path) for path in candidates)

        source = next(
            (path for path in candidates if path.is_dir() and _has_required(path, tree.required)),
            None,
        )
        if source is None:
            if _has_required(target, tree.required):
                report.satisfied.append({
                    "target": str(target),
                    "source": str(target),
                    "published": False,
                })
                continue
            report.ready = False
            report.missing.append({
                "target": str(target),
                "required": [str(target / item) for item in tree.required],
            })
            continue

        try:
            same_tree = source.resolve() == target.resolve()
        except OSError:
            same_tree = False
        if same_tree:
            report.satisfied.append({
                "target": str(target),
                "source": str(source),
                "published": False,
            })
            continue
        if not contract.auto_publish:
            report.ready = False
            report.missing.append({
                "target": str(target),
                "candidate": str(source),
                "reason": "auto publication disabled",
            })
            continue

        try:
            _atomic_overlay_tree(source, target, tree.required)
        except Exception as exc:
            report.ready = False
            report.missing.append({
                "target": str(target),
                "candidate": str(source),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        report.published.append({
            "target": str(target),
            "source": str(source),
            "published": True,
        })

    report.ready = report.ready and not report.missing
    return report


def copy_submission_to_shared(source: Path, destination: Path) -> None:
    """Copy a submitted file or directory to shared storage atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        stage = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.nanoma-submit-",
            dir=destination.parent,
        ))
        try:
            shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=True)
            if destination.exists():
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return

    temporary = destination.with_name(f".{destination.name}.nanoma-submit-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        if destination.exists() and destination.is_dir():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

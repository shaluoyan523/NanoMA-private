"""Runtime-owned delivery contracts and atomic artifact publication.

Benchmarks describe *where* their final artifact tree must live and which
alternative tree layouts agents may produce.  The runtime owns the last mile:
it finds a complete candidate tree, stages it beside the official destination,
and swaps it into place only after every required file is present.
"""

from __future__ import annotations

import os
import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


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
    validator: Callable[[Path], str | None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

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
    rejected: list[dict] = field(default_factory=list)
    checked_candidates: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "trigger": self.trigger,
            "published": list(self.published),
            "satisfied": list(self.satisfied),
            "missing": list(self.missing),
            "rejected": list(self.rejected),
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


def delivery_tree_digest(tree_root: Path) -> str:
    """Return a stable content digest for one complete delivery tree."""

    root = Path(tree_root).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(b"L\0")
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            continue
        if not path.is_file():
            continue
        digest.update(b"F\0")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _has_required(tree_root: Path, required: Sequence[str]) -> bool:
    for raw in required:
        path = tree_root / _relative_path(raw, label="delivery required path")
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _validation_error(tree_root: Path, tree: DeliveryTree) -> str | None:
    if not _has_required(tree_root, tree.required):
        return "required output is missing or empty"
    if tree.validator is None:
        return None
    try:
        error = tree.validator(tree_root)
    except Exception as exc:
        return f"validator raised {type(exc).__name__}: {exc}"
    return str(error).strip() if error else None


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


def _tree_candidates(
    tree: DeliveryTree,
    *,
    candidate_bases: Sequence[Path],
    explicit_paths: Sequence[Path] = (),
) -> list[Path]:
    """Enumerate candidate roots, including a workspace root itself.

    Agents commonly create the required entry directly in their private
    workspace.  Treating only ``workspace/<candidate>`` as eligible silently
    discarded those otherwise complete results.
    """

    candidates: list[Path] = []
    for path in explicit_paths:
        candidates.extend(_explicit_tree_candidates(Path(path), tree))
    for base in candidate_bases:
        base = Path(base).expanduser()
        candidates.append(base)
        for raw in tree.candidates:
            candidates.append(base / _relative_path(raw, label="delivery candidate"))
    return _deduplicate_paths(candidates)


def _make_tree_read_only(root: Path) -> None:
    """Best-effort protection against accidental mutation of review snapshots."""

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            path.chmod(0o555 if path.is_dir() else 0o444)
        except OSError:
            pass
    try:
        root.chmod(0o555)
    except OSError:
        pass


def snapshot_delivery_candidates(
    contract: DeliveryContract,
    *,
    candidate_bases: Sequence[Path],
    snapshot_root: Path,
    explicit_paths: Sequence[Path] = (),
) -> dict:
    """Validate and freeze the first complete candidate for each contract tree.

    The returned paths are content-addressed and read-only.  They can therefore
    be referenced by parent agents and final reviewers without depending on a
    child's mutable workspace.
    """

    snapshot_root = Path(snapshot_root).expanduser()
    captured: list[dict] = []
    rejected: list[dict] = []
    missing: list[dict] = []
    checked: list[str] = []

    for index, tree in enumerate(contract.trees):
        source: Path | None = None
        for path in _tree_candidates(
            tree,
            candidate_bases=() if explicit_paths else candidate_bases,
            explicit_paths=explicit_paths,
        ):
            checked.append(str(path))
            if not path.is_dir() or not _has_required(path, tree.required):
                continue
            validation_error = _validation_error(path, tree)
            if validation_error:
                rejected.append({
                    "target": tree.target,
                    "candidate": str(path),
                    "reason": validation_error,
                })
                continue
            source = path
            break
        if source is None:
            missing.append({
                "target": tree.target,
                "required": list(tree.required),
            })
            continue

        digest = delivery_tree_digest(source)
        target_label = _relative_path(tree.target, label="delivery target").as_posix().replace("/", "__")
        destination = snapshot_root / f"{index:02d}-{target_label}-{digest[:20]}"
        if not destination.is_dir() or delivery_tree_digest(destination) != digest:
            snapshot_root.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".nanoma-candidate-", dir=snapshot_root))
            try:
                shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=False)
                validation_error = _validation_error(stage, tree)
                if validation_error:
                    raise RuntimeError(f"snapshot validation failed: {validation_error}")
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(stage, destination)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            _make_tree_read_only(destination)

        manifest = []
        for path in sorted(destination.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                manifest.append({
                    "path": path.relative_to(destination).as_posix(),
                    "bytes": path.stat().st_size,
                })
        captured.append({
            "target": tree.target,
            "source": str(source),
            "snapshot": str(destination),
            "sha256": digest,
            "required": list(tree.required),
            "manifest": manifest,
            "validated": True,
        })

    composite = ""
    if captured:
        composite_input = "\n".join(
            f"{item['target']}\0{item['sha256']}" for item in captured
        )
        composite = hashlib.sha256(composite_input.encode("utf-8")).hexdigest()
    return {
        "ready": len(captured) == len(contract.trees),
        "captured": captured,
        "artifact_sha256": composite,
        "rejected": rejected,
        "missing": missing,
        "checked_candidates": checked,
    }


def _atomic_overlay_tree(source: Path, target: Path, tree: DeliveryTree) -> None:
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
        validation_error = _validation_error(stage, tree)
        if validation_error:
            raise RuntimeError(
                f"staged delivery is invalid: {stage}: {validation_error}"
            )

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
        target_error = _validation_error(target, tree)
        if target_error is None and not explicit:
            report.satisfied.append({
                "target": str(target),
                "source": str(target),
                "published": False,
            })
            continue

        candidates = _tree_candidates(
            tree,
            candidate_bases=() if explicit else bases,
            explicit_paths=explicit,
        )
        report.checked_candidates.extend(str(path) for path in candidates)

        source = None
        for path in candidates:
            if not path.is_dir() or not _has_required(path, tree.required):
                continue
            validation_error = _validation_error(path, tree)
            if validation_error:
                report.rejected.append({
                    "target": str(target),
                    "candidate": str(path),
                    "reason": validation_error,
                })
                continue
            source = path
            break
        if source is None:
            if target_error is None and not explicit:
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
                "target_validation_error": target_error,
                "reason": (
                    "explicit submission did not contain a valid complete candidate; "
                    "the previously published target was preserved"
                    if explicit and target_error is None
                    else "no valid complete candidate was found"
                ),
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
            _atomic_overlay_tree(source, target, tree)
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

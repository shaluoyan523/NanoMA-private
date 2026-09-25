"""Freeze terminal NanoMA worktrees as immutable, content-addressed candidates.

This module deliberately does not score candidates and does not feed anything
back to agents.  It only turns the private worktrees that already exist at the
end of a merge-submit run into durable artifacts for post-run evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1


def _safe_scope_roots(scope_roots: Sequence[Path | str]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw in scope_roots:
        root = Path(raw)
        if root.is_absolute() or ".." in root.parts:
            raise ValueError(f"candidate scope must be relative: {raw!s}")
        if root.as_posix() in {"", ".", "./"}:
            return ()
        roots.append(root)
    return tuple(roots)


def _iter_artifact_files(
    source: Path,
    scope_roots: Sequence[Path],
    ignored_parts: frozenset[str],
) -> Iterable[tuple[str, Path]]:
    bases = [source / relative for relative in scope_roots] if scope_roots else [source]
    found: dict[str, Path] = {}
    for base in bases:
        if not base.exists() and not base.is_symlink():
            continue
        if base.is_file() or base.is_symlink():
            relative = base.relative_to(source)
            if not any(part in ignored_parts for part in relative.parts):
                found[relative.as_posix()] = base
            continue
        for path in base.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                continue
            relative = path.relative_to(source)
            if any(part in ignored_parts for part in relative.parts):
                continue
            found[relative.as_posix()] = path
    for relative in sorted(found):
        yield relative, found[relative]


def _artifact_digest(
    files: Sequence[tuple[str, Path]], scope_roots: Sequence[Path]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"nanoma-candidate-v1\0")
    for scope in scope_roots:
        digest.update(b"scope\0")
        digest.update(scope.as_posix().encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    for relative, path in files:
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"link\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            continue
        digest.update(b"file\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_artifact(files: Sequence[tuple[str, Path]], destination: Path) -> int:
    total_bytes = 0
    destination.mkdir(parents=True, exist_ok=True)
    for relative, source in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            os.symlink(os.readlink(source), target)
            continue
        shutil.copy2(source, target, follow_symlinks=False)
        total_bytes += source.stat().st_size
    return total_bytes


def preserve_branch_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    scope_roots: Sequence[Path | str] = (),
    ignored_parts: Iterable[str] = (),
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze candidate source roots and atomically publish a manifest.

    Every logical branch remains in ``manifest["candidates"]``. Identical
    branches share one physical artifact through their SHA-256 digest, so a
    verifier can score each unique artifact once and map the result back to all
    agents that produced it.
    """

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scopes = _safe_scope_roots(scope_roots)
    ignored = frozenset(str(part) for part in ignored_parts)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    manifest_candidates: list[dict[str, Any]] = []
    artifact_records: dict[str, dict[str, Any]] = {}

    try:
        artifacts_dir = staging / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        seen_candidate_ids: set[str] = set()
        for raw_candidate in candidates:
            candidate_id = str(raw_candidate.get("candidate_id") or "").strip()
            if not candidate_id:
                raise ValueError("candidate_id is required")
            if candidate_id in seen_candidate_ids:
                raise ValueError(f"duplicate candidate_id: {candidate_id}")
            seen_candidate_ids.add(candidate_id)

            source = Path(str(raw_candidate.get("source_root") or ""))
            if not source.is_dir():
                manifest_candidates.append({
                    key: value
                    for key, value in raw_candidate.items()
                    if key != "source_root"
                } | {
                    "candidate_id": candidate_id,
                    "available": False,
                    "error": f"source root unavailable: {source}",
                })
                continue

            files = list(_iter_artifact_files(source, scopes, ignored))
            artifact_digest = _artifact_digest(files, scopes)
            artifact_relative = Path("artifacts") / artifact_digest
            if artifact_digest not in artifact_records:
                size_bytes = _copy_artifact(files, staging / artifact_relative)
                artifact_records[artifact_digest] = {
                    "digest": artifact_digest,
                    "path": artifact_relative.as_posix(),
                    "files": len(files),
                    "bytes": size_bytes,
                }

            entry = {
                key: value
                for key, value in raw_candidate.items()
                if key != "source_root"
            }
            entry.update({
                "candidate_id": candidate_id,
                "available": True,
                "artifact_digest": artifact_digest,
                "artifact_path": artifact_relative.as_posix(),
            })
            manifest_candidates.append(entry)

        selected = [
            item["candidate_id"]
            for item in manifest_candidates
            if item.get("selected") and item.get("available")
        ]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": time.time(),
            "scope_roots": [path.as_posix() for path in scopes] or ["."],
            "selected_candidate_id": selected[0] if selected else None,
            "logical_candidate_count": len(manifest_candidates),
            "available_candidate_count": sum(
                1 for item in manifest_candidates if item.get("available")
            ),
            "unique_artifact_count": len(artifact_records),
            "artifacts": list(artifact_records.values()),
            "candidates": manifest_candidates,
        }
        if run_metadata:
            manifest["run"] = dict(run_metadata)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        manifest["output_dir"] = str(destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

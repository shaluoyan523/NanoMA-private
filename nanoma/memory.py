"""Structured memory types and broker for NanoMA."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ActiveTaskCard:
    task: str
    tags: list[str] = field(default_factory=list)
    work_outline: str = ""


@dataclass
class ExperienceCard:
    summary: str
    tags: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    memory_kind: str = "experience"
    memory_source: str = "agent"
    durable: bool = False
    evidence: list[str] = field(default_factory=list)


@dataclass
class AgentPublicMemory:
    public_summary: str = ""
    active_task: ActiveTaskCard | None = None
    experience_cards: list[ExperienceCard] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    artifact_index: list[str] = field(default_factory=list)


@dataclass
class TagRegistry:
    tags: set[str] = field(default_factory=set)

    def update(self, new_tags: list[str]) -> None:
        self.tags.update(tag for tag in new_tags if tag)

    def as_list(self) -> list[str]:
        return sorted(self.tags)


@dataclass
class TagIndex:
    by_tag: dict[str, set[str]] = field(default_factory=dict)

    def update_agent(self, agent_id: str, tags: list[str]) -> None:
        for members in self.by_tag.values():
            members.discard(agent_id)
        for tag in tags:
            if not tag:
                continue
            self.by_tag.setdefault(tag, set()).add(agent_id)

    def query(self, seed_terms: list[str]) -> list[str]:
        hits: set[str] = set()
        for term in seed_terms:
            hits.update(self.by_tag.get(normalize_tag(term), set()))
        return sorted(hits)


def normalize_tag_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_.:-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.:-")
    return text


def normalize_tag(tag: Any) -> str:
    text = str(tag or "").strip()
    if not text:
        return ""
    if ":" in text:
        prefix, value = text.split(":", 1)
        prefix = normalize_tag_value(prefix)
        value = normalize_tag_value(value)
        return f"{prefix}:{value}" if prefix and value else ""
    return normalize_tag_value(text)


def normalize_tags(tags: list[Any] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for tag in tags or []:
        item = normalize_tag(tag)
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def classify_memory_kind(
    *,
    explicit_kind: Any = None,
    summary: str = "",
    tags: list[Any] | None = None,
    stop_after: bool = False,
    artifacts: list[str] | None = None,
) -> str:
    requested = normalize_tag_value(explicit_kind)
    allowed = {
        "active",
        "working",
        "summary",
        "experience",
        "durable",
        "evidence",
        "terminal",
        "handoff",
        "status",
        "recent_input",
    }
    if requested in allowed:
        return requested

    normalized_tags = set(normalize_tags(tags))
    text = f"{summary or ''}\n{' '.join(normalized_tags)}".lower()
    if stop_after:
        return "terminal"
    if artifacts or any(tag.startswith(("artifact:", "file:", "evidence:", "source:")) for tag in normalized_tags):
        return "evidence"
    if any(tag in normalized_tags for tag in {"memory:durable", "durable:true", "preference:stable", "workflow:stable"}):
        return "durable"
    if any(tag.startswith(("status:", "phase:", "progress:")) for tag in normalized_tags):
        return "status"
    if any(term in text for term in ("evidence", "source", "artifact", "citation", "found arxiv", "verified")):
        return "evidence"
    return "experience"


def memory_layer_tags(
    *,
    kind: str = "experience",
    source: str = "agent",
    durable: bool = False,
    evidence: list[Any] | None = None,
) -> list[str]:
    tags = [
        f"memory_kind:{kind or 'experience'}",
        f"memory_source:{source or 'agent'}",
    ]
    if durable:
        tags.append("memory_durable:true")
    for item in evidence or []:
        value = normalize_tag_value(item)
        if value:
            tags.append(f"evidence:{value}")
    return normalize_tags(tags)


def build_memory_card(
    summary: str,
    *,
    tags: list[Any] | None = None,
    artifacts: list[str] | None = None,
    memory_kind: Any = None,
    memory_source: Any = "agent",
    durable: bool | None = None,
    evidence: list[Any] | None = None,
    stop_after: bool = False,
) -> ExperienceCard:
    normalized_tags = normalize_tags(tags)
    kind = classify_memory_kind(
        explicit_kind=memory_kind,
        summary=summary,
        tags=normalized_tags,
        stop_after=stop_after,
        artifacts=artifacts,
    )
    is_durable = bool(durable) or kind == "durable"
    source = normalize_tag_value(memory_source) or "agent"
    card_tags = normalize_tags(
        normalized_tags
        + memory_layer_tags(kind=kind, source=source, durable=is_durable, evidence=evidence)
    )
    return ExperienceCard(
        summary=summary,
        tags=card_tags,
        artifacts=list(artifacts or []),
        memory_kind=kind,
        memory_source=source,
        durable=is_durable,
        evidence=[normalize_tag_value(item) for item in evidence or [] if normalize_tag_value(item)],
    )


def canonicalize_role(role: Any, tags: list[Any] | None = None) -> tuple[str, str, list[str]]:
    raw = str(role or "").strip()
    if not raw:
        return "", "", []
    normalized_text = normalize_tag_value(raw)
    if re.fullmatch(r"[a-z][a-z0-9]*(?:[_:-]\d+)", normalized_text):
        base = re.sub(r"[_:-]\d+$", "", normalized_text)
        extra_tags = normalize_tags([f"role:{base}"] if base else [])
        return normalized_text, "", extra_tags
    haystack = " ".join([raw.lower(), " ".join(str(tag).lower() for tag in tags or [])])
    role_map = (
        ("verifier", ("verifier", "verify", "verification", "validator", "validate")),
        ("evidence", ("evidence", "collector", "source", "citation")),
        ("researcher", ("researcher", "research", "scout", "search")),
        ("generator", ("generator", "generate", "candidate", "solver", "solution")),
        ("reviewer", ("reviewer", "review", "critic", "critique")),
        ("tester", ("tester", "test", "validation")),
        ("judge", ("judge", "comparison", "compare", "pairwise")),
        ("mutator", ("mutator", "mutation", "mutate")),
        ("integrator", ("integrator", "integration")),
        ("synthesizer", ("synthesizer", "synthesis", "summarizer", "summary")),
        ("coordinator", ("coordinator", "coordinate", "orchestrator")),
        ("worker", ("worker", "implementer")),
    )
    canonical = normalized_text
    for candidate, terms in role_map:
        if any(term in haystack for term in terms):
            canonical = candidate
            break
    extra_tags = [f"role:{canonical}"] if canonical else []
    lane_match = re.search(r"\blane\W*0*([0-9]+)\b", raw, flags=re.IGNORECASE)
    if lane_match:
        extra_tags.append(f"lane:{int(lane_match.group(1))}")
    description = raw if canonical != normalized_text else ""
    return canonical, description, normalize_tags(extra_tags)


def agent_identity_tags(
    *,
    agent_id: str,
    role: str = "",
    group_id: str = "",
    parent: str | None = None,
    created_by: str | None = None,
    create_type: str = "",
    relationship: str = "",
    workflow_prior: str = "",
    depth: int | None = None,
) -> list[str]:
    tags: list[str] = [f"agent:{agent_id}", f"id:{agent_id}"]
    if role:
        tags.append(f"role:{role}")
    if group_id:
        tags.extend([f"group:{group_id}", f"group_id:{group_id}"])
    if parent:
        tags.append(f"parent:{parent}")
    if created_by:
        tags.append(f"created_by:{created_by}")
    if create_type:
        tags.append(f"create_type:{create_type}")
    if relationship:
        tags.append(f"relationship:{relationship}")
    if workflow_prior:
        tags.append(f"workflow:{workflow_prior}")
    if depth is not None:
        tags.append(f"depth:{depth}")
    return normalize_tags(tags)


def split_experience_text(text: str, *, max_chars: int = 700, max_cards: int = 8) -> list[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []
    raw_parts = [
        part.strip()
        for part in re.split(r"\n\s*\n|^(?=#{1,6}\s+)|(?<=[。.!?])\s+(?=[A-Z0-9#*-])", stripped, flags=re.MULTILINE)
        if part.strip()
    ]
    if not raw_parts:
        raw_parts = [stripped]
    cards: list[str] = []
    for part in raw_parts:
        if len(part) > max_chars:
            for start in range(0, len(part), max_chars):
                cards.append(part[start:start + max_chars].strip())
            continue
        cards.append(part)
    return cards[:max_cards]


class MemoryBroker:
    def __init__(self) -> None:
        self.public_memories: dict[str, AgentPublicMemory] = {}
        self.tag_registry = TagRegistry()
        self.tag_index = TagIndex()

    def init_agent(self, agent_id: str, task: str, tags: list[str] | None = None) -> AgentPublicMemory:
        memory = AgentPublicMemory(active_task=ActiveTaskCard(task=task, tags=normalize_tags(tags)))
        self.public_memories[agent_id] = memory
        self.sync(agent_id)
        return memory

    def _normalize_card(self, card: ExperienceCard) -> ExperienceCard:
        kind = classify_memory_kind(
            explicit_kind=card.memory_kind,
            summary=card.summary,
            tags=card.tags,
            artifacts=card.artifacts,
        )
        source = normalize_tag_value(card.memory_source) or "agent"
        card.memory_kind = kind
        card.memory_source = source
        card.evidence = [normalize_tag_value(item) for item in card.evidence if normalize_tag_value(item)]
        card.durable = bool(card.durable) or kind == "durable"
        card.tags = normalize_tags(
            card.tags
            + memory_layer_tags(
                kind=kind,
                source=source,
                durable=card.durable,
                evidence=card.evidence,
            )
        )
        return card

    def get(self, agent_id: str) -> AgentPublicMemory | None:
        return self.public_memories.get(agent_id)

    def list(self) -> dict[str, dict[str, Any]]:
        return {agent_id: self.serialize(agent_id) for agent_id in sorted(self.public_memories)}

    def sync(self, agent_id: str) -> None:
        memory = self.public_memories.get(agent_id)
        if not memory:
            return
        tags = set(memory.tags)
        if memory.active_task:
            tags.update(memory.active_task.tags)
        for card in memory.experience_cards:
            self._normalize_card(card)
            tags.update(card.tags)
        memory.tags = sorted(tag for tag in normalize_tags(list(tags)) if tag)
        self.tag_registry.update(memory.tags)
        self.tag_index.update_agent(agent_id, memory.tags)

    def update(
        self,
        agent_id: str,
        *,
        public_summary: str | None = None,
        active_task: ActiveTaskCard | None = None,
        clear_active_task: bool = False,
        add_experience: ExperienceCard | None = None,
        add_experiences: list[ExperienceCard] | None = None,
        tags: list[str] | None = None,
        replace_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
        add_artifacts: list[str] | None = None,
    ) -> AgentPublicMemory:
        memory = self.public_memories.setdefault(agent_id, AgentPublicMemory())
        if public_summary is not None:
            memory.public_summary = public_summary
        if active_task is not None:
            active_task.tags = normalize_tags(active_task.tags)
            memory.active_task = active_task
        if clear_active_task:
            memory.active_task = None
        if add_experience is not None:
            memory.experience_cards.append(self._normalize_card(add_experience))
        if add_experiences:
            for card in add_experiences:
                memory.experience_cards.append(self._normalize_card(card))
        if replace_tags is not None:
            memory.tags = normalize_tags(replace_tags)
        if tags:
            memory.tags = sorted(set(memory.tags).union(normalize_tags(tags)))
        if remove_tags:
            blocked = set(normalize_tags(remove_tags))
            if memory.active_task:
                memory.active_task.tags = [tag for tag in memory.active_task.tags if tag not in blocked]
            for card in memory.experience_cards:
                card.tags = [tag for tag in card.tags if tag not in blocked]
            memory.tags = sorted(tag for tag in memory.tags if tag not in blocked)
        if add_artifacts:
            memory.artifact_index = sorted(set(memory.artifact_index).union(add_artifacts))
        self.sync(agent_id)
        return memory

    def read(self, intent: str = "", seed_terms: list[str] | None = None) -> dict[str, Any]:
        seed_terms = normalize_tags(seed_terms or [])
        matched_agent_ids = self.tag_index.query(seed_terms) if seed_terms else sorted(self.public_memories)
        matched = []
        for agent_id in matched_agent_ids:
            if agent_id not in self.public_memories:
                continue
            public_memory = self.serialize(agent_id)
            cards = public_memory.get("experience_cards") or []
            if seed_terms:
                card_hits = []
                for card in cards:
                    card_tags = set(card.get("tags") or [])
                    summary = str(card.get("summary") or "").lower()
                    if card_tags.intersection(seed_terms) or any(term.replace(":", " ") in summary for term in seed_terms):
                        card_hits.append(card)
            else:
                card_hits = cards
            matched.append({"agent_id": agent_id, "public_memory": public_memory, "matched_cards": card_hits[:8]})
        return {
            "intent": intent,
            "seed_terms": seed_terms,
            "matches": matched,
            "known_tags": self.tag_registry.as_list(),
        }

    def serialize(self, agent_id: str) -> dict[str, Any]:
        memory = self.public_memories.get(agent_id)
        if memory is None:
            return {}
        return asdict(memory)

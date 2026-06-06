"""Structured memory types and broker for NanoMA."""

from __future__ import annotations

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
            hits.update(self.by_tag.get(term, set()))
        return sorted(hits)


class MemoryBroker:
    def __init__(self) -> None:
        self.public_memories: dict[str, AgentPublicMemory] = {}
        self.tag_registry = TagRegistry()
        self.tag_index = TagIndex()

    def init_agent(self, agent_id: str, task: str, tags: list[str] | None = None) -> AgentPublicMemory:
        memory = AgentPublicMemory(active_task=ActiveTaskCard(task=task, tags=list(tags or [])))
        self.public_memories[agent_id] = memory
        self.sync(agent_id)
        return memory

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
            tags.update(card.tags)
        memory.tags = sorted(tag for tag in tags if tag)
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
        tags: list[str] | None = None,
        add_artifacts: list[str] | None = None,
    ) -> AgentPublicMemory:
        memory = self.public_memories.setdefault(agent_id, AgentPublicMemory())
        if public_summary is not None:
            memory.public_summary = public_summary
        if active_task is not None:
            memory.active_task = active_task
        if clear_active_task:
            memory.active_task = None
        if add_experience is not None:
            memory.experience_cards.append(add_experience)
        if tags:
            memory.tags = sorted(set(memory.tags).union(tag for tag in tags if tag))
        if add_artifacts:
            memory.artifact_index = sorted(set(memory.artifact_index).union(add_artifacts))
        self.sync(agent_id)
        return memory

    def read(self, intent: str = "", seed_terms: list[str] | None = None) -> dict[str, Any]:
        seed_terms = [term for term in (seed_terms or []) if term]
        matched_agent_ids = self.tag_index.query(seed_terms) if seed_terms else sorted(self.public_memories)
        matched = [
            {"agent_id": agent_id, "public_memory": self.serialize(agent_id)}
            for agent_id in matched_agent_ids
            if agent_id in self.public_memories
        ]
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

"""
Data models that mirror Zep Cloud's response objects.

These classes provide the same attribute interface as zep_cloud SDK
so consuming code can use them interchangeably.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ───────────────────────── Ontology helpers ─────────────────────────

@dataclass
class EpisodeData:
    """Drop-in replacement for zep_cloud.EpisodeData."""
    data: str
    type: str = "text"


@dataclass
class EntityEdgeSourceTarget:
    """Drop-in replacement for zep_cloud.EntityEdgeSourceTarget."""
    source: str = "Entity"
    target: str = "Entity"


# ───────────────────────── Graph node / edge ────────────────────────

@dataclass
class GraphNode:
    """A knowledge-graph node (entity)."""
    uuid_: str = field(default_factory=lambda: _uuid.uuid4().hex)
    name: str = ""
    labels: List[str] = field(default_factory=lambda: ["Entity"])
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Alias so both `node.uuid_` and `node.uuid` work
    @property
    def uuid(self) -> str:
        return self.uuid_

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid_": self.uuid_,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphNode":
        return cls(
            uuid_=d.get("uuid_", d.get("uuid", _uuid.uuid4().hex)),
            name=d.get("name", ""),
            labels=d.get("labels", ["Entity"]),
            summary=d.get("summary", ""),
            attributes=d.get("attributes", {}),
            created_at=d.get("created_at"),
        )


@dataclass
class GraphEdge:
    """A knowledge-graph edge (relationship / fact)."""
    uuid_: str = field(default_factory=lambda: _uuid.uuid4().hex)
    name: str = ""
    fact: str = ""
    fact_type: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = field(default_factory=lambda: datetime.utcnow().isoformat())
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)

    @property
    def uuid(self) -> str:
        return self.uuid_

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid_": self.uuid_,
            "name": self.name,
            "fact": self.fact,
            "fact_type": self.fact_type,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
            "episodes": self.episodes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEdge":
        return cls(
            uuid_=d.get("uuid_", d.get("uuid", _uuid.uuid4().hex)),
            name=d.get("name", ""),
            fact=d.get("fact", ""),
            fact_type=d.get("fact_type", ""),
            source_node_uuid=d.get("source_node_uuid", ""),
            target_node_uuid=d.get("target_node_uuid", ""),
            attributes=d.get("attributes", {}),
            created_at=d.get("created_at"),
            valid_at=d.get("valid_at"),
            invalid_at=d.get("invalid_at"),
            expired_at=d.get("expired_at"),
            episodes=d.get("episodes", []),
        )


@dataclass
class GraphEpisode:
    """An ingested text episode."""
    uuid_: str = field(default_factory=lambda: _uuid.uuid4().hex)
    data: str = ""
    type: str = "text"
    processed: bool = False
    created_at: Optional[str] = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def uuid(self) -> str:
        return self.uuid_


@dataclass
class GraphSearchResult:
    """Result returned by graph.search()."""
    edges: List[GraphEdge] = field(default_factory=list)
    nodes: List[GraphNode] = field(default_factory=list)


@dataclass
class GraphMetadata:
    """Metadata for a graph instance."""
    graph_id: str = ""
    name: str = ""
    description: str = ""
    ontology: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "description": self.description,
            "ontology": self.ontology,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphMetadata":
        return cls(
            graph_id=d.get("graph_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            ontology=d.get("ontology"),
            created_at=d.get("created_at"),
        )

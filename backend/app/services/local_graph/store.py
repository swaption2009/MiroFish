"""
JSON-based graph storage layer.

Manages graph data as JSON files on the local filesystem:
  {data_dir}/{graph_id}/
    meta.json      — graph metadata + ontology
    nodes.json     — list of node dicts
    edges.json     — list of edge dicts
    episodes.json  — list of episode dicts
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

from .models import GraphEdge, GraphEpisode, GraphMetadata, GraphNode

from ...utils.logger import get_logger

logger = get_logger("mirofish.local_graph.store")

_lock = threading.Lock()


class GraphStore:
    """Thread-safe, file-backed store for a single vault."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        os.makedirs(self._data_dir, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────

    def _graph_dir(self, graph_id: str) -> str:
        return os.path.join(self._data_dir, graph_id)

    def _read_json(self, path: str) -> Any:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # ── graph CRUD ──────────────────────────────────────────────────

    def create_graph(self, graph_id: str, name: str, description: str = "") -> GraphMetadata:
        gdir = self._graph_dir(graph_id)
        os.makedirs(gdir, exist_ok=True)

        meta = GraphMetadata(graph_id=graph_id, name=name, description=description)
        self._write_json(os.path.join(gdir, "meta.json"), meta.to_dict())
        self._write_json(os.path.join(gdir, "nodes.json"), [])
        self._write_json(os.path.join(gdir, "edges.json"), [])
        self._write_json(os.path.join(gdir, "episodes.json"), [])
        logger.info(f"Created local graph: {graph_id} ({name})")
        return meta

    def delete_graph(self, graph_id: str) -> None:
        gdir = self._graph_dir(graph_id)
        if os.path.exists(gdir):
            shutil.rmtree(gdir)
            logger.info(f"Deleted local graph: {graph_id}")

    def graph_exists(self, graph_id: str) -> bool:
        return os.path.exists(os.path.join(self._graph_dir(graph_id), "meta.json"))

    def get_metadata(self, graph_id: str) -> Optional[GraphMetadata]:
        data = self._read_json(os.path.join(self._graph_dir(graph_id), "meta.json"))
        return GraphMetadata.from_dict(data) if data else None

    # ── ontology ────────────────────────────────────────────────────

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        meta_path = os.path.join(self._graph_dir(graph_id), "meta.json")
        meta_data = self._read_json(meta_path) or {}
        meta_data["ontology"] = ontology
        self._write_json(meta_path, meta_data)
        logger.info(f"Set ontology for graph {graph_id}")

    def get_ontology(self, graph_id: str) -> Optional[Dict[str, Any]]:
        meta = self.get_metadata(graph_id)
        return meta.ontology if meta else None

    # ── nodes ───────────────────────────────────────────────────────

    def get_all_nodes(self, graph_id: str) -> List[GraphNode]:
        data = self._read_json(os.path.join(self._graph_dir(graph_id), "nodes.json")) or []
        return [GraphNode.from_dict(d) for d in data]

    def get_node(self, graph_id: str, node_uuid: str) -> Optional[GraphNode]:
        for node in self.get_all_nodes(graph_id):
            if node.uuid_ == node_uuid:
                return node
        return None

    def find_node_by_name(self, graph_id: str, name: str) -> Optional[GraphNode]:
        name_lower = name.lower().strip()
        for node in self.get_all_nodes(graph_id):
            if node.name.lower().strip() == name_lower:
                return node
        return None

    def add_nodes(self, graph_id: str, nodes: List[GraphNode]) -> None:
        path = os.path.join(self._graph_dir(graph_id), "nodes.json")
        existing = self._read_json(path) or []

        # Merge by name (case-insensitive) — update if exists, add if new
        name_idx: Dict[str, int] = {}
        for i, d in enumerate(existing):
            name_idx[d.get("name", "").lower().strip()] = i

        for node in nodes:
            key = node.name.lower().strip()
            nd = node.to_dict()
            if key in name_idx:
                # Merge: keep uuid, update other fields
                idx = name_idx[key]
                old_uuid = existing[idx].get("uuid_", nd["uuid_"])
                nd["uuid_"] = old_uuid
                # Merge attributes
                old_attrs = existing[idx].get("attributes", {})
                old_attrs.update(nd.get("attributes", {}))
                nd["attributes"] = old_attrs
                # Update summary only if new one is longer
                if len(nd.get("summary", "")) < len(existing[idx].get("summary", "")):
                    nd["summary"] = existing[idx]["summary"]
                # Merge labels
                old_labels = set(existing[idx].get("labels", []))
                old_labels.update(nd.get("labels", []))
                nd["labels"] = list(old_labels)
                existing[idx] = nd
            else:
                existing.append(nd)
                name_idx[key] = len(existing) - 1

        self._write_json(path, existing)

    # ── edges ───────────────────────────────────────────────────────

    def get_all_edges(self, graph_id: str) -> List[GraphEdge]:
        data = self._read_json(os.path.join(self._graph_dir(graph_id), "edges.json")) or []
        return [GraphEdge.from_dict(d) for d in data]

    def get_edges_for_node(self, graph_id: str, node_uuid: str) -> List[GraphEdge]:
        return [
            e for e in self.get_all_edges(graph_id)
            if e.source_node_uuid == node_uuid or e.target_node_uuid == node_uuid
        ]

    def add_edges(self, graph_id: str, edges: List[GraphEdge]) -> None:
        path = os.path.join(self._graph_dir(graph_id), "edges.json")
        existing = self._read_json(path) or []

        # Deduplicate by (source, target, name) triple
        existing_keys = set()
        for d in existing:
            key = (d.get("source_node_uuid", ""), d.get("target_node_uuid", ""), d.get("name", "").lower())
            existing_keys.add(key)

        for edge in edges:
            key = (edge.source_node_uuid, edge.target_node_uuid, edge.name.lower())
            if key not in existing_keys:
                existing.append(edge.to_dict())
                existing_keys.add(key)

        self._write_json(path, existing)

    # ── episodes ────────────────────────────────────────────────────

    def get_all_episodes(self, graph_id: str) -> List[GraphEpisode]:
        data = self._read_json(os.path.join(self._graph_dir(graph_id), "episodes.json")) or []
        eps = []
        for d in data:
            eps.append(GraphEpisode(
                uuid_=d.get("uuid_", ""),
                data=d.get("data", ""),
                type=d.get("type", "text"),
                processed=d.get("processed", False),
                created_at=d.get("created_at"),
            ))
        return eps

    def get_episode(self, graph_id: str, episode_uuid: str) -> Optional[GraphEpisode]:
        for ep in self.get_all_episodes(graph_id):
            if ep.uuid_ == episode_uuid:
                return ep
        return None

    def add_episode(self, graph_id: str, episode: GraphEpisode) -> None:
        path = os.path.join(self._graph_dir(graph_id), "episodes.json")
        existing = self._read_json(path) or []
        existing.append({
            "uuid_": episode.uuid_,
            "data": episode.data,
            "type": episode.type,
            "processed": episode.processed,
            "created_at": episode.created_at,
        })
        self._write_json(path, existing)

    def mark_episode_processed(self, graph_id: str, episode_uuid: str) -> None:
        path = os.path.join(self._graph_dir(graph_id), "episodes.json")
        data = self._read_json(path) or []
        for d in data:
            if d.get("uuid_") == episode_uuid:
                d["processed"] = True
                break
        self._write_json(path, data)

    # ── search (keyword / substring) ────────────────────────────────

    def search_edges(self, graph_id: str, query: str, limit: int = 10) -> List[GraphEdge]:
        """Simple keyword search over edge facts."""
        query_lower = query.lower()
        query_terms = query_lower.split()
        edges = self.get_all_edges(graph_id)

        scored: List[tuple] = []
        for edge in edges:
            text = f"{edge.fact} {edge.name}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, edge))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def search_nodes(self, graph_id: str, query: str, limit: int = 10) -> List[GraphNode]:
        """Simple keyword search over node names and summaries."""
        query_lower = query.lower()
        query_terms = query_lower.split()
        nodes = self.get_all_nodes(graph_id)

        scored: List[tuple] = []
        for node in nodes:
            text = f"{node.name} {node.summary}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]

    # ── graph enumeration ───────────────────────────────────────────

    def list_graphs(self) -> List[str]:
        """Return all graph IDs in the vault."""
        if not os.path.exists(self._data_dir):
            return []
        return [
            d for d in os.listdir(self._data_dir)
            if os.path.isdir(os.path.join(self._data_dir, d))
        ]

    # ── helper: find graph_id for any episode uuid ──────────────────

    def find_graph_for_episode(self, episode_uuid: str) -> Optional[str]:
        """Search all graphs to find which one contains a given episode."""
        for gid in self.list_graphs():
            ep = self.get_episode(gid, episode_uuid)
            if ep:
                return gid
        return None

    def find_graph_for_node(self, node_uuid: str) -> Optional[str]:
        """Search all graphs to find which one contains a given node."""
        for gid in self.list_graphs():
            node = self.get_node(gid, node_uuid)
            if node:
                return gid
        return None

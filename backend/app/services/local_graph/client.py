"""
LocalZepClient — drop-in replacement for ``zep_cloud.client.Zep``.

Provides the same nested attribute interface::

    client.graph.create(graph_id=..., name=..., description=...)
    client.graph.search(graph_id=..., query=..., limit=..., scope=..., reranker=...)
    client.graph.node.get_by_graph_id(graph_id, limit=..., uuid_cursor=...)
    client.graph.node.get(uuid_=...)
    client.graph.node.get_entity_edges(node_uuid=...)
    client.graph.edge.get_by_graph_id(graph_id, limit=..., uuid_cursor=...)
    client.graph.episode.get(uuid_=...)
    client.graph.add_batch(graph_id=..., episodes=[...])
    client.graph.add(graph_id=..., type=..., data=...)
    client.graph.set_ontology(graph_ids=..., entities=..., edges=...)
    client.graph.delete(graph_id=...)

All data is stored as JSON in the Obsidian vault.
Entity extraction is performed by the LLM (Gemini via OpenAI SDK).
"""

from __future__ import annotations

import os
import uuid as _uuid
from typing import Any, Dict, List, Optional

from ...config import Config
from ...utils.logger import get_logger
from .entity_extractor import EntityExtractor
from .models import (
    EpisodeData,
    GraphEdge,
    GraphEpisode,
    GraphNode,
    GraphSearchResult,
)
from .obsidian_sync import ObsidianSync
from .store import GraphStore

logger = get_logger("mirofish.local_graph.client")


# ═══════════════════════════════════════════════════════════════════
# Sub-service classes that mirror Zep's nested API
# ═══════════════════════════════════════════════════════════════════


class _NodeService:
    """Mirrors ``client.graph.node``."""

    def __init__(self, store: GraphStore):
        self._store = store

    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: Optional[str] = None,
        **_kw: Any,
    ) -> List[GraphNode]:
        """Return all nodes for a graph, with optional cursor pagination."""
        all_nodes = self._store.get_all_nodes(graph_id)

        if uuid_cursor:
            # Find the cursor position and return nodes after it
            idx = next(
                (i for i, n in enumerate(all_nodes) if n.uuid_ == uuid_cursor),
                None,
            )
            if idx is not None:
                all_nodes = all_nodes[idx + 1 :]

        return all_nodes[:limit]

    def get(self, uuid_: str, **_kw: Any) -> Optional[GraphNode]:
        """Get a single node by UUID (searches all graphs)."""
        graph_id = self._store.find_graph_for_node(uuid_)
        if graph_id:
            return self._store.get_node(graph_id, uuid_)
        return None

    def get_entity_edges(self, node_uuid: str, **_kw: Any) -> List[GraphEdge]:
        """Get all edges connected to a node (searches all graphs)."""
        graph_id = self._store.find_graph_for_node(node_uuid)
        if graph_id:
            return self._store.get_edges_for_node(graph_id, node_uuid)
        return []


class _EdgeService:
    """Mirrors ``client.graph.edge``."""

    def __init__(self, store: GraphStore):
        self._store = store

    def get_by_graph_id(
        self,
        graph_id: str,
        limit: int = 100,
        uuid_cursor: Optional[str] = None,
        **_kw: Any,
    ) -> List[GraphEdge]:
        all_edges = self._store.get_all_edges(graph_id)

        if uuid_cursor:
            idx = next(
                (i for i, e in enumerate(all_edges) if e.uuid_ == uuid_cursor),
                None,
            )
            if idx is not None:
                all_edges = all_edges[idx + 1 :]

        return all_edges[:limit]


class _EpisodeService:
    """Mirrors ``client.graph.episode``."""

    def __init__(self, store: GraphStore):
        self._store = store

    def get(self, uuid_: str, **_kw: Any) -> Optional[GraphEpisode]:
        """Get episode status (searches all graphs)."""
        graph_id = self._store.find_graph_for_episode(uuid_)
        if graph_id:
            return self._store.get_episode(graph_id, uuid_)
        # Return a "processed" stub so callers don't hang waiting
        return GraphEpisode(uuid_=uuid_, processed=True)


class _GraphService:
    """
    Mirrors ``client.graph`` — the main interface for graph operations.
    """

    def __init__(self, store: GraphStore, extractor: EntityExtractor, sync: ObsidianSync):
        self._store = store
        self._extractor = extractor
        self._sync = sync

        # Sub-services
        self.node = _NodeService(store)
        self.edge = _EdgeService(store)
        self.episode = _EpisodeService(store)

    # ── create / delete ─────────────────────────────────────────────

    def create(self, graph_id: str, name: str, description: str = "", **_kw: Any) -> None:
        self._store.create_graph(graph_id, name, description)

    def delete(self, graph_id: str, **_kw: Any) -> None:
        self._store.delete_graph(graph_id)

    # ── ontology ────────────────────────────────────────────────────

    def set_ontology(
        self,
        graph_ids: List[str],
        entities: Optional[Dict[str, Any]] = None,
        edges: Optional[Dict[str, Any]] = None,
        **_kw: Any,
    ) -> None:
        """
        Accept ontology in Zep's format (dynamic classes) and store as JSON.

        ``entities`` is a dict of {name: Pydantic EntityModel subclass}
        ``edges`` is a dict of {name: (EdgeModel subclass, [EntityEdgeSourceTarget])}
        """
        ontology_json = self._ontology_to_json(entities, edges)
        for gid in graph_ids:
            self._store.set_ontology(gid, ontology_json)

    @staticmethod
    def _ontology_to_json(
        entities: Optional[Dict[str, Any]],
        edges: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Convert Zep's dynamic Pydantic classes to a plain JSON dict."""
        result: Dict[str, Any] = {"entity_types": [], "edge_types": []}

        if entities:
            for name, cls in entities.items():
                et: Dict[str, Any] = {
                    "name": name,
                    "description": getattr(cls, "__doc__", "") or "",
                    "attributes": [],
                }
                # Extract attributes from class annotations
                annotations = getattr(cls, "__annotations__", {})
                for attr_name in annotations:
                    if attr_name.startswith("_"):
                        continue
                    field_obj = getattr(cls, attr_name, None)
                    desc = ""
                    if hasattr(field_obj, "description"):
                        desc = field_obj.description or ""
                    elif hasattr(field_obj, "field_info"):
                        desc = getattr(field_obj.field_info, "description", "") or ""
                    elif hasattr(field_obj, "json_schema_extra"):
                        desc = str(field_obj.json_schema_extra)
                    et["attributes"].append({"name": attr_name, "description": desc})
                result["entity_types"].append(et)

        if edges:
            for name, val in edges.items():
                if isinstance(val, tuple) and len(val) == 2:
                    cls, source_targets = val
                else:
                    cls = val
                    source_targets = []

                ed: Dict[str, Any] = {
                    "name": name,
                    "description": getattr(cls, "__doc__", "") or "",
                    "attributes": [],
                    "source_targets": [],
                }
                annotations = getattr(cls, "__annotations__", {})
                for attr_name in annotations:
                    if attr_name.startswith("_"):
                        continue
                    field_obj = getattr(cls, attr_name, None)
                    desc = ""
                    if hasattr(field_obj, "description"):
                        desc = field_obj.description or ""
                    ed["attributes"].append({"name": attr_name, "description": desc})

                for st in source_targets:
                    ed["source_targets"].append({
                        "source": getattr(st, "source", "Entity"),
                        "target": getattr(st, "target", "Entity"),
                    })
                result["edge_types"].append(ed)

        return result

    # ── text ingestion ──────────────────────────────────────────────

    def add_batch(
        self,
        graph_id: str,
        episodes: List[Any],
        **_kw: Any,
    ) -> List[GraphEpisode]:
        """
        Ingest a batch of text episodes.

        For each episode, runs LLM entity extraction, stores results,
        and marks the episode as processed.
        """
        ontology = self._store.get_ontology(graph_id)
        existing_nodes = self._store.get_all_nodes(graph_id)
        result_episodes: List[GraphEpisode] = []

        for ep in episodes:
            text = ep.data if hasattr(ep, "data") else str(ep)
            ep_uuid = _uuid.uuid4().hex

            episode = GraphEpisode(uuid_=ep_uuid, data=text, type="text", processed=False)
            self._store.add_episode(graph_id, episode)

            try:
                nodes, edges = self._extractor.extract(text, ontology, existing_nodes)
                if nodes:
                    # Tag edges with episode uuid
                    for edge in edges:
                        edge.episodes = [ep_uuid]
                    self._store.add_nodes(graph_id, nodes)
                    self._store.add_edges(graph_id, edges)
                    # Update existing_nodes for next iteration
                    existing_nodes = self._store.get_all_nodes(graph_id)

                self._store.mark_episode_processed(graph_id, ep_uuid)
                episode.processed = True
            except Exception as e:
                logger.warning(f"Failed to extract entities from episode {ep_uuid}: {e}")
                # Still mark as processed to avoid infinite retries
                self._store.mark_episode_processed(graph_id, ep_uuid)
                episode.processed = True

            result_episodes.append(episode)

        # Sync to Obsidian after batch completes
        try:
            self._sync.sync_graph(graph_id)
        except Exception as e:
            logger.warning(f"Obsidian sync failed: {e}")

        return result_episodes

    def add(self, graph_id: str, type: str = "text", data: str = "", **_kw: Any) -> GraphEpisode:
        """Ingest a single text episode."""
        result = self.add_batch(graph_id, [EpisodeData(data=data, type=type)])
        return result[0] if result else GraphEpisode(processed=True)

    # ── search ──────────────────────────────────────────────────────

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: Optional[str] = None,
        **_kw: Any,
    ) -> GraphSearchResult:
        """
        Search the local graph using keyword matching.

        ``scope`` can be 'edges', 'nodes', or 'both' (default).
        ``reranker`` is accepted but ignored (local search doesn't rerank).
        """
        result = GraphSearchResult()

        if scope in ("edges", "both"):
            result.edges = self._store.search_edges(graph_id, query, limit)
        if scope in ("nodes", "both"):
            result.nodes = self._store.search_nodes(graph_id, query, limit)

        return result


# ═══════════════════════════════════════════════════════════════════
# Main client class
# ═══════════════════════════════════════════════════════════════════


class LocalZepClient:
    """
    Drop-in replacement for ``Zep(api_key=...)``.

    Usage::

        from .local_graph import LocalZepClient
        client = LocalZepClient(vault_path="/path/to/MiroFish-Vault")
        client.graph.create(graph_id="g1", name="My Graph")
    """

    def __init__(self, vault_path: Optional[str] = None, **_kw: Any):
        self.vault_path = vault_path or _default_vault_path()
        self._store = GraphStore(self.vault_path)
        self._extractor = EntityExtractor()
        self._sync = ObsidianSync(self.vault_path, self._store)

        self.graph = _GraphService(self._store, self._extractor, self._sync)

        logger.info(f"LocalZepClient initialized — vault: {self.vault_path}")


# ═══════════════════════════════════════════════════════════════════
# Factory helper
# ═══════════════════════════════════════════════════════════════════


def _default_vault_path() -> str:
    """Resolve the default vault path."""
    configured = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    if configured:
        return configured
    # Fall back to MiroFish-Vault inside the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    return os.path.join(project_root, "MiroFish-Vault")


def get_graph_client(api_key: Optional[str] = None) -> Any:
    """
    Factory that returns either a Zep Cloud client or LocalZepClient
    based on the ``GRAPH_BACKEND`` env variable.

    - ``GRAPH_BACKEND=zep`` → use Zep Cloud (original behaviour)
    - ``GRAPH_BACKEND=local`` or unset → use local Obsidian vault
    """
    backend = os.environ.get("GRAPH_BACKEND", "local").lower()

    if backend == "zep":
        from zep_cloud.client import Zep
        key = api_key or Config.ZEP_API_KEY
        if not key:
            raise ValueError("ZEP_API_KEY not configured but GRAPH_BACKEND=zep")
        return Zep(api_key=key)

    return LocalZepClient()

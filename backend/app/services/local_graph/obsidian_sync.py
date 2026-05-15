"""
Obsidian Markdown Sync.

Writes graph data as human-readable markdown files with YAML frontmatter
into the Obsidian vault so users can browse entities and relationships
using Obsidian's graph view and wiki-links.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from .models import GraphEdge, GraphMetadata, GraphNode
from .store import GraphStore
from ...utils.logger import get_logger

logger = get_logger("mirofish.local_graph.obsidian_sync")


def _safe_filename(name: str) -> str:
    """Convert entity name to a safe filename."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe = safe.strip(". ")
    return safe[:100] or "unnamed"


class ObsidianSync:
    """Syncs graph data to Obsidian markdown files."""

    def __init__(self, vault_path: str, store: GraphStore):
        self.vault_path = vault_path
        self.store = store
        self._graphs_dir = os.path.join(vault_path, "Graphs")
        os.makedirs(self._graphs_dir, exist_ok=True)

    def sync_graph(self, graph_id: str) -> None:
        """Sync a full graph to Obsidian markdown."""
        meta = self.store.get_metadata(graph_id)
        if not meta:
            logger.warning(f"Cannot sync graph {graph_id}: metadata not found")
            return

        nodes = self.store.get_all_nodes(graph_id)
        edges = self.store.get_all_edges(graph_id)

        graph_name = _safe_filename(meta.name or graph_id)
        graph_dir = os.path.join(self._graphs_dir, graph_name)
        entities_dir = os.path.join(graph_dir, "Entities")
        os.makedirs(entities_dir, exist_ok=True)

        # Build node uuid→name map for edge linking
        uuid_name: Dict[str, str] = {n.uuid_: n.name for n in nodes}

        # Write graph README
        self._write_graph_readme(graph_dir, meta, nodes, edges)

        # Write entity notes
        for node in nodes:
            node_edges = [e for e in edges if e.source_node_uuid == node.uuid_ or e.target_node_uuid == node.uuid_]
            self._write_entity_note(entities_dir, node, node_edges, uuid_name)

        logger.info(f"Synced graph '{meta.name}' → {len(nodes)} entity notes")

    def _write_graph_readme(
        self,
        graph_dir: str,
        meta: GraphMetadata,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
    ) -> None:
        lines = [
            "---",
            f"graph_id: {meta.graph_id}",
            f"created_at: {meta.created_at}",
            "---",
            f"# {meta.name or meta.graph_id}",
            "",
            f"> {meta.description}" if meta.description else "",
            "",
            f"**Entities:** {len(nodes)}  |  **Relationships:** {len(edges)}",
            "",
            "## Entities",
            "",
        ]
        # Group by type
        types: Dict[str, List[GraphNode]] = {}
        for n in nodes:
            etype = next((l for l in n.labels if l not in ("Entity", "Node")), "Other")
            types.setdefault(etype, []).append(n)

        for etype, tnodes in sorted(types.items()):
            lines.append(f"### {etype} ({len(tnodes)})")
            for n in tnodes:
                safe = _safe_filename(n.name)
                lines.append(f"- [[Entities/{safe}|{n.name}]]")
            lines.append("")

        path = os.path.join(graph_dir, "README.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_entity_note(
        self,
        entities_dir: str,
        node: GraphNode,
        edges: List[GraphEdge],
        uuid_name: Dict[str, str],
    ) -> None:
        etype = next((l for l in node.labels if l not in ("Entity", "Node")), "Entity")
        lines = [
            "---",
            f"uuid: {node.uuid_}",
            f"type: {etype}",
            f"labels: [{', '.join(node.labels)}]",
            f"created_at: {node.created_at}",
            "---",
            f"# {node.name}",
            "",
        ]

        if node.summary:
            lines += [node.summary, ""]

        if node.attributes:
            lines.append("## Attributes")
            for k, v in node.attributes.items():
                if v:
                    lines.append(f"- **{k}:** {v}")
            lines.append("")

        if edges:
            lines.append("## Relationships")
            for edge in edges:
                other_uuid = edge.target_node_uuid if edge.source_node_uuid == node.uuid_ else edge.source_node_uuid
                other_name = uuid_name.get(other_uuid, other_uuid)
                safe_other = _safe_filename(other_name)
                direction = "→" if edge.source_node_uuid == node.uuid_ else "←"
                fact = edge.fact or edge.name
                lines.append(f"- {direction} [[{safe_other}|{other_name}]] — *{edge.name}*: {fact}")
            lines.append("")

        filename = _safe_filename(node.name) + ".md"
        path = os.path.join(entities_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

"""
LLM-based entity extractor.

Replaces Zep Cloud's automatic entity/relationship extraction
by sending text to Gemini with the current ontology and parsing
the structured JSON response.
"""

from __future__ import annotations

import json
import uuid as _uuid
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from ...config import Config
from ...utils.logger import get_logger
from .models import GraphEdge, GraphNode

logger = get_logger("mirofish.local_graph.entity_extractor")

# System prompt for entity extraction
_SYSTEM_PROMPT = """You are a precise knowledge-graph entity extractor.
Given a text passage and an ontology definition, extract ALL entities and
relationships mentioned in the text.

Rules:
1. Only extract entities whose type matches one of the ontology entity types.
2. Only extract relationships whose type matches one of the ontology edge types.
3. Each entity must have: name, type (from ontology), summary (1-2 sentences), attributes.
4. Each relationship must have: source entity name, target entity name, type (from ontology), fact (natural-language description of the relationship).
5. If an entity or relationship is mentioned multiple times, merge them (don't duplicate).
6. Return ONLY valid JSON, no markdown fences or extra text.
"""


def _build_extraction_prompt(text: str, ontology: Optional[Dict[str, Any]]) -> str:
    """Build the user prompt for entity extraction."""
    onto_section = ""
    if ontology:
        entity_types = ontology.get("entity_types", [])
        edge_types = ontology.get("edge_types", [])

        if entity_types:
            lines = []
            for et in entity_types:
                attrs = ", ".join(a["name"] for a in et.get("attributes", []))
                lines.append(f'  - {et["name"]}: {et.get("description", "")}  [attributes: {attrs}]')
            onto_section += "Entity types:\n" + "\n".join(lines) + "\n\n"

        if edge_types:
            lines = []
            for ed in edge_types:
                sts = "; ".join(f'{s["source"]}→{s["target"]}' for s in ed.get("source_targets", []))
                lines.append(f'  - {ed["name"]}: {ed.get("description", "")}  [{sts}]')
            onto_section += "Edge types:\n" + "\n".join(lines) + "\n\n"
    else:
        onto_section = "No predefined ontology. Extract any entities and relationships you find.\n\n"

    return f"""{onto_section}Text to analyze:
\"\"\"
{text}
\"\"\"

Return a JSON object with exactly this structure:
{{
  "entities": [
    {{
      "name": "...",
      "type": "...",
      "summary": "...",
      "attributes": {{}}
    }}
  ],
  "relationships": [
    {{
      "source": "...",
      "target": "...",
      "type": "...",
      "fact": "..."
    }}
  ]
}}
"""


class EntityExtractor:
    """Uses an LLM to extract entities and relationships from text."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def extract(
        self,
        text: str,
        ontology: Optional[Dict[str, Any]] = None,
        existing_nodes: Optional[List[GraphNode]] = None,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Extract entities and relationships from *text*.

        Returns (nodes, edges).  Edges reference nodes by uuid;
        if a node name matches an existing node the uuid is reused.
        """
        prompt = _build_extraction_prompt(text, ontology)

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                content = response.choices[0].message.content or "{}"
                parsed = self._parse_response(content)
                nodes, edges = self._to_models(parsed, existing_nodes)
                logger.info(
                    f"Extracted {len(nodes)} entities, {len(edges)} relationships "
                    f"from {len(text)} chars"
                )
                return nodes, edges

            except Exception as e:
                logger.warning(f"Entity extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    logger.error("Entity extraction failed — returning empty results")
                    return [], []

        return [], []

    # ── internal helpers ────────────────────────────────────────────

    @staticmethod
    def _parse_response(content: str) -> Dict[str, Any]:
        """Parse LLM JSON response, with fallback repair."""
        import re
        content = content.strip()
        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Attempt simple repair
            open_b = content.count("{") - content.count("}")
            open_k = content.count("[") - content.count("]")
            content += "]" * open_k + "}" * open_b
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM extraction response")
                return {"entities": [], "relationships": []}

    @staticmethod
    def _to_models(
        parsed: Dict[str, Any],
        existing_nodes: Optional[List[GraphNode]] = None,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Convert parsed JSON dicts into GraphNode / GraphEdge objects."""
        # Build name→uuid map from existing nodes
        name_uuid: Dict[str, str] = {}
        if existing_nodes:
            for n in existing_nodes:
                name_uuid[n.name.lower().strip()] = n.uuid_

        nodes: List[GraphNode] = []
        for ent in parsed.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            uid = name_uuid.get(key, _uuid.uuid4().hex)
            name_uuid[key] = uid
            etype = ent.get("type", "Entity")
            labels = ["Entity"]
            if etype and etype != "Entity":
                labels.append(etype)
            nodes.append(GraphNode(
                uuid_=uid,
                name=name,
                labels=labels,
                summary=ent.get("summary", ""),
                attributes=ent.get("attributes", {}),
            ))

        edges: List[GraphEdge] = []
        for rel in parsed.get("relationships", []):
            src_name = (rel.get("source", "")).lower().strip()
            tgt_name = (rel.get("target", "")).lower().strip()
            if not src_name or not tgt_name:
                continue

            # Ensure source/target nodes exist (create stubs if needed)
            if src_name not in name_uuid:
                uid = _uuid.uuid4().hex
                name_uuid[src_name] = uid
                nodes.append(GraphNode(uuid_=uid, name=rel.get("source", src_name)))
            if tgt_name not in name_uuid:
                uid = _uuid.uuid4().hex
                name_uuid[tgt_name] = uid
                nodes.append(GraphNode(uuid_=uid, name=rel.get("target", tgt_name)))

            edges.append(GraphEdge(
                uuid_=_uuid.uuid4().hex,
                name=rel.get("type", ""),
                fact=rel.get("fact", ""),
                fact_type=rel.get("type", ""),
                source_node_uuid=name_uuid[src_name],
                target_node_uuid=name_uuid[tgt_name],
            ))

        return nodes, edges

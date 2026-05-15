"""
Local Graph Package — Drop-in replacement for Zep Cloud.

Provides a local knowledge graph service backed by JSON files
with LLM-powered entity extraction.
"""

from .client import LocalZepClient, get_graph_client

__all__ = ["LocalZepClient", "get_graph_client"]

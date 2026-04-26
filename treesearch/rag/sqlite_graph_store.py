# -*- coding: utf-8 -*-
"""SQLite persistence for the TreeSearch GraphRAG store."""

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from treesearch.rag.graph_store import InMemoryGraphStore, _entity_id
from treesearch.rag.models import GraphEntity, GraphNodePassage, GraphRelation, StructuralEdge


class SQLiteGraphStore(InMemoryGraphStore):
    """In-memory graph store with explicit SQLite save/load support."""

    def __init__(self, db_path: str | Path):
        super().__init__()
        self.db_path = Path(db_path)

    def clear(self) -> None:
        super().clear()
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute("DELETE FROM passages")
            conn.execute("DELETE FROM entities")
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM structural_edges")

    def save(self) -> None:
        self._ensure_schema()
        with self._connect() as conn:
            conn.execute("DELETE FROM passages")
            conn.execute("DELETE FROM entities")
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM structural_edges")
            conn.executemany(
                "INSERT INTO passages(graph_node_id, payload) VALUES (?, ?)",
                [
                    (graph_node_id, json.dumps(asdict(passage), ensure_ascii=False))
                    for graph_node_id, passage in self.passages.items()
                ],
            )
            conn.executemany(
                "INSERT INTO entities(entity_id, payload) VALUES (?, ?)",
                [
                    (entity_id, json.dumps(asdict(entity), ensure_ascii=False))
                    for entity_id, entity in self.entities.items()
                ],
            )
            conn.executemany(
                "INSERT INTO relations(relation_id, payload) VALUES (?, ?)",
                [
                    (relation_id, json.dumps(asdict(relation), ensure_ascii=False))
                    for relation_id, relation in self.relations.items()
                ],
            )
            conn.executemany(
                "INSERT INTO structural_edges(payload) VALUES (?)",
                [
                    (json.dumps(asdict(edge), ensure_ascii=False),)
                    for edge in self.structural_edges
                ],
            )

    def load(self) -> None:
        self._ensure_schema()
        super().clear()
        with self._connect() as conn:
            for graph_node_id, payload in conn.execute("SELECT graph_node_id, payload FROM passages"):
                self.passages[str(graph_node_id)] = GraphNodePassage(**json.loads(payload))
            for entity_id, payload in conn.execute("SELECT entity_id, payload FROM entities"):
                self.entities[str(entity_id)] = GraphEntity(**json.loads(payload))
            for relation_id, payload in conn.execute("SELECT relation_id, payload FROM relations"):
                self.relations[str(relation_id)] = GraphRelation(**json.loads(payload))
            for (payload,) in conn.execute("SELECT payload FROM structural_edges"):
                self.structural_edges.append(StructuralEdge(**json.loads(payload)))

        for relation in self.relations.values():
            self.node_to_relation_ids[relation.graph_node_id].add(relation.relation_id)
            self.entity_to_relation_ids[_entity_id(relation.subject)].add(relation.relation_id)
            self.entity_to_relation_ids[_entity_id(relation.object)].add(relation.relation_id)

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS passages(graph_node_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS entities(entity_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS relations(relation_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS structural_edges(id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

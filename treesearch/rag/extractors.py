# -*- coding: utf-8 -*-
"""Triplet extraction interfaces for GraphRAG."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from treesearch.rag.models import GraphNodePassage, GraphRelation

TripletSpec = tuple[str, str, str] | list[str] | Mapping[str, str]


class TripletExtractor(Protocol):
    """Extract graph relations from a TreeSearch node passage."""

    def extract(self, passage: GraphNodePassage) -> list[GraphRelation]:
        ...


class RuleBasedTripletExtractor:
    """Small deterministic extractor for tests, demos, and offline smoke runs.

    It recognizes sentences like ``A is defined_in B`` and binds each extracted
    relation to the source TreeSearch node.
    """

    _RELATION_RE = re.compile(
        r"\b(?P<subject>[A-Za-z_][\w]*)\s+is\s+"
        r"(?P<predicate>[A-Za-z_][\w]*)\s+"
        r"(?P<object>[A-Za-z_][\w]*(?:\s+[A-Za-z_][\w]*)*)"
    )

    def extract(self, passage: GraphNodePassage) -> list[GraphRelation]:
        relations: list[GraphRelation] = []
        for match in self._RELATION_RE.finditer(passage.text):
            subject = match.group("subject").strip()
            predicate = match.group("predicate").strip()
            obj = match.group("object").strip().rstrip(".")
            text = f"{subject} {predicate} {obj}"
            relation_id = _relation_id(passage.graph_node_id, text)
            relations.append(
                GraphRelation(
                    relation_id=relation_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    text=text,
                    node_id=passage.node_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                )
            )
        return relations


class PreExtractedTripletExtractor:
    """Extractor backed by known triplets, useful for repeatable experiments."""

    def __init__(self, triplets_by_node_id: Mapping[str, Sequence[TripletSpec]]):
        self.triplets_by_node_id = triplets_by_node_id

    def extract(self, passage: GraphNodePassage) -> list[GraphRelation]:
        relations: list[GraphRelation] = []
        for spec in self.triplets_by_node_id.get(passage.graph_node_id, ()):
            subject, predicate, obj = _normalize_triplet_spec(spec)
            text = f"{subject} {predicate} {obj}"
            relations.append(
                GraphRelation(
                    relation_id=_relation_id(passage.graph_node_id, text),
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    text=text,
                    node_id=passage.node_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                )
            )
        return relations


def _normalize_triplet_spec(spec: TripletSpec) -> tuple[str, str, str]:
    if isinstance(spec, Mapping):
        return (
            spec["subject"].strip(),
            spec["predicate"].strip(),
            spec["object"].strip(),
        )
    subject, predicate, obj = spec
    return subject.strip(), predicate.strip(), obj.strip()


def _relation_id(node_id: str, text: str) -> str:
    payload = f"{node_id}:{text}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

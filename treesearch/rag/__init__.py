# -*- coding: utf-8 -*-
"""Optional TreeSearch-guided GraphRAG layer."""

from treesearch.rag.answer import LLMAnswerGenerator, TemplateAnswerGenerator
from treesearch.rag.evidence import EvidenceSelector, HeuristicEvidenceSelector, LLMEvidenceSelector
from treesearch.rag.eval import (
    RealRepoEvalResult,
    RealRepoSample,
    aggregate_eval_results,
    evaluate_grounded_answer,
)
from treesearch.rag.expansion import StructureConstrainedExpander
from treesearch.rag.extractors import (
    PreExtractedTripletExtractor,
    RuleBasedTripletExtractor,
    TripletExtractor,
)
from treesearch.rag.graph_builder import NodeGraphBuilder
from treesearch.rag.graph_store import InMemoryGraphStore
from treesearch.rag.llm import FakeLLMClient, LLMClient, OpenAIChatClient
from treesearch.rag.models import (
    CandidateRelation,
    EvidenceChain,
    EvidenceCitation,
    ExpansionConfig,
    GraphBuildStats,
    GraphEntity,
    GraphNodePassage,
    GraphRelation,
    GraphSeed,
    GroundedAnswer,
    StructuralEdge,
    VerificationResult,
)
from treesearch.rag.pipeline import TreeSearchGraphRAG
from treesearch.rag.sqlite_graph_store import SQLiteGraphStore

__all__ = [
    "CandidateRelation",
    "EvidenceChain",
    "EvidenceCitation",
    "EvidenceSelector",
    "ExpansionConfig",
    "FakeLLMClient",
    "GraphBuildStats",
    "GraphEntity",
    "GraphNodePassage",
    "GraphRelation",
    "GraphSeed",
    "GroundedAnswer",
    "HeuristicEvidenceSelector",
    "InMemoryGraphStore",
    "LLMAnswerGenerator",
    "LLMClient",
    "LLMEvidenceSelector",
    "NodeGraphBuilder",
    "OpenAIChatClient",
    "PreExtractedTripletExtractor",
    "RealRepoEvalResult",
    "RealRepoSample",
    "RuleBasedTripletExtractor",
    "SQLiteGraphStore",
    "StructuralEdge",
    "StructureConstrainedExpander",
    "TemplateAnswerGenerator",
    "TreeSearchGraphRAG",
    "TripletExtractor",
    "VerificationResult",
    "aggregate_eval_results",
    "evaluate_grounded_answer",
]

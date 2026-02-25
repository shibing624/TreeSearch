# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tree search over document structures — Best-First (default), MCTS,
              single-pass LLM, and the unified multi-document ``search()`` pipeline.

Key design:
  - BestFirstTreeSearch: deterministic priority-queue search with optional BM25 pre-scoring,
    LLM relevance evaluation, early stopping, budget control, and subtree caching.
  - MCTS: preserved as alternative strategy, deterministic + cache-friendly.
  - ``search()`` is the primary public API — it natively handles one or many documents.
"""
import asyncio
import hashlib
import heapq
import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .llm import achat, extract_json, DEFAULT_MODEL
from .tree import Document, flatten_tree, find_node, remove_fields

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PreFilter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PreFilter(Protocol):
    """Protocol for pre-scoring nodes before tree search.

    Implementations must provide ``score_nodes`` which returns a dict
    mapping node_id -> relevance score for a given query and document.
    Built-in implementation: ``NodeBM25Index``.
    """

    def score_nodes(self, query: str, doc_id: str) -> dict[str, float]:
        """Return {node_id: score} for nodes in the given document."""
        ...


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Result returned by :func:`search`."""
    documents: list = field(default_factory=list)
    # documents: [{'doc_id', 'doc_name', 'nodes': [{'node_id', 'title', 'text', 'score'}]}]
    query: str = ""
    total_llm_calls: int = 0
    strategy: str = ""


# ---------------------------------------------------------------------------
# Text normalizer for cache-friendly prompts
# ---------------------------------------------------------------------------

def _normalize(text: str, max_len: int = 512) -> str:
    """Collapse whitespace and truncate for stable prompt / cache keys."""
    return " ".join(text.strip().split())[:max_len]


def _query_fingerprint(query: str) -> str:
    """Short hash for cache keys."""
    return hashlib.md5(query.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Best-First Tree Search (default strategy)
# ---------------------------------------------------------------------------

class BestFirstTreeSearch:
    """
    Deterministic best-first tree search over a document tree.

    Three-layer design:
      - Layer 1: BM25 pre-scoring (optional, provides initial priority)
      - Layer 2: Best-first expansion with LLM relevance evaluation
      - Layer 3: Budget-controlled LLM calls with early stopping

    Features:
      - Priority queue driven: always expands the most promising node first
      - BM25 scores as prior: when available, nodes start with BM25-based priority
      - LLM as judge: evaluates "does this node contain the answer?" (title+summary only)
      - Early stopping: stops when top-of-queue score drops below threshold
      - Budget control: max_llm_calls limits total LLM invocations
      - Subtree cache: caches (query_fingerprint, node_id) -> relevance for reuse
    """

    # Class-level subtree cache shared across searches
    _subtree_cache: dict[str, dict] = {}

    def __init__(
        self,
        document: Document,
        query: str,
        model: str = DEFAULT_MODEL,
        max_results: int = 5,
        threshold: float = 0.3,
        max_llm_calls: int = 30,
        bm25_scores: Optional[dict[str, float]] = None,
        bm25_weight: float = 0.3,
        depth_penalty: float = 0.02,
        use_subtree_cache: bool = True,
    ):
        self.document = document
        self.query = query
        self.model = model
        self.max_results = max_results
        self.threshold = threshold
        self.max_llm_calls = max_llm_calls
        self.bm25_scores = bm25_scores or {}
        self.bm25_weight = bm25_weight
        self.depth_penalty = depth_penalty
        self.use_subtree_cache = use_subtree_cache

        self._llm_calls = 0
        self._value_cache: dict[str, float] = {}
        self._qfp = _query_fingerprint(query)

        # Build node map from document tree
        self._node_map: dict[str, dict] = {}
        self._parent_map: dict[str, Optional[str]] = {}
        self._depth_map: dict[str, int] = {}
        self._build(document.structure, parent_id=None, depth=0)

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    def _build(self, structure, parent_id: Optional[str], depth: int) -> None:
        if isinstance(structure, list):
            for item in structure:
                self._build(item, parent_id, depth)
        elif isinstance(structure, dict):
            nid = structure.get("node_id", "")
            self._node_map[nid] = structure
            self._parent_map[nid] = parent_id
            self._depth_map[nid] = depth
            for child in structure.get("nodes", []):
                self._build(child, parent_id=nid, depth=depth + 1)

    def _get_summary(self, node: dict) -> str:
        return _normalize(node.get("summary", node.get("prefix_summary", "")))

    def _initial_priority(self, node_id: str) -> float:
        """Compute initial priority from BM25 score and depth penalty."""
        bm25 = self.bm25_scores.get(node_id, 0.0)
        depth = self._depth_map.get(node_id, 0)
        # Normalize BM25 to [0,1] range approximately
        bm25_norm = min(bm25 / max(max(self.bm25_scores.values(), default=1.0), 1e-6), 1.0) if bm25 > 0 else 0.0
        return bm25_norm * self.bm25_weight - depth * self.depth_penalty

    async def _evaluate(self, node_id: str) -> float:
        """LLM scores node relevance. Returns value in [0, 1]."""
        # Check subtree cache
        if self.use_subtree_cache:
            cache_key = f"{self._qfp}::{node_id}"
            cached = BestFirstTreeSearch._subtree_cache.get(cache_key)
            if cached is not None:
                return cached.get("max_relevance", 0.0)

        # Check local cache
        if node_id in self._value_cache:
            return self._value_cache[node_id]

        node = self._node_map.get(node_id)
        if not node:
            return 0.0

        prompt = (
            f"Rate relevance of this document section to the query.\n\n"
            f"Query: {self.query}\n\n"
            f"Section title: {node.get('title', '')}\n"
            f"Section summary: {self._get_summary(node)}\n\n"
            f"Return JSON only:\n"
            f'{{"relevance": <float 0.0-1.0>}}'
        )
        response = await achat(prompt, model=self.model, temperature=0)
        result = extract_json(response)
        value = float(result.get("relevance", 0.0))

        self._value_cache[node_id] = value
        self._llm_calls += 1

        # Update subtree cache
        if self.use_subtree_cache:
            cache_key = f"{self._qfp}::{node_id}"
            BestFirstTreeSearch._subtree_cache[cache_key] = {
                "max_relevance": value,
                "depth": self._depth_map.get(node_id, 0),
            }

        return value

    async def run(self) -> list[dict]:
        """
        Run best-first tree search.

        Returns: [{'node_id': str, 'title': str, 'score': float}]
        """
        # Priority queue: (-score, node_id) for max-heap via min-heap
        pq: list[tuple[float, str]] = []
        visited: set[str] = set()
        results: list[dict] = []

        # Phase 1: evaluate root nodes and push to queue
        root_ids = [n.get("node_id", "") for n in self.document.structure if isinstance(n, dict)]
        root_tasks = []
        for rid in root_ids:
            if self._llm_calls < self.max_llm_calls:
                root_tasks.append((rid, self._evaluate(rid)))
        root_results = await asyncio.gather(*(t for _, t in root_tasks))

        for (rid, _), val in zip(root_tasks, root_results):
            priority = val + self._initial_priority(rid)
            heapq.heappush(pq, (-priority, rid))

        # Phase 2: best-first expansion
        while pq and len(results) < self.max_results and self._llm_calls < self.max_llm_calls:
            neg_score, nid = heapq.heappop(pq)
            score = -neg_score

            if nid in visited:
                continue
            visited.add(nid)

            # Early stopping: best remaining score below threshold
            if score < self.threshold:
                break

            node = self._node_map.get(nid)
            if not node:
                continue

            children = node.get("nodes", [])
            child_ids = [c.get("node_id", "") for c in children if isinstance(c, dict)]

            if not child_ids:
                # Leaf node -> collect as result
                llm_score = self._value_cache.get(nid, score)
                results.append({
                    "node_id": nid,
                    "title": node.get("title", ""),
                    "score": round(llm_score, 4),
                })
            else:
                # Non-leaf: evaluate children and push to queue
                eval_tasks = []
                eval_ids = []
                for cid in child_ids:
                    if cid not in visited and self._llm_calls < self.max_llm_calls:
                        eval_tasks.append(self._evaluate(cid))
                        eval_ids.append(cid)

                if eval_tasks:
                    child_values = await asyncio.gather(*eval_tasks)
                    for cid, cval in zip(eval_ids, child_values):
                        priority = cval + self._initial_priority(cid)
                        heapq.heappush(pq, (-priority, cid))

                # Non-leaf node itself can also be a result if highly relevant
                llm_score = self._value_cache.get(nid, score)
                if llm_score >= self.threshold:
                    results.append({
                        "node_id": nid,
                        "title": node.get("title", ""),
                        "score": round(llm_score, 4),
                    })

        results.sort(key=lambda x: (-x["score"], x["node_id"]))
        return results[:self.max_results]

    @classmethod
    def clear_subtree_cache(cls):
        """Clear the shared subtree cache."""
        cls._subtree_cache.clear()


# ---------------------------------------------------------------------------
# MCTS data structures
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """A node in the MCTS search tree (wraps a document tree node)."""
    node_id: str
    title: str
    summary: str = ""
    children_ids: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    is_expanded: bool = False
    is_terminal: bool = False

    @property
    def avg_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0


# ---------------------------------------------------------------------------
# MCTS Tree Search (deterministic + cache-friendly)
# ---------------------------------------------------------------------------

class MCTSTreeSearch:
    """
    Monte Carlo Tree Search over a document tree.

    Deterministic + cache-friendly design:
      - temperature=0 for all LLM calls
      - No "thinking" field in prompts (removes non-deterministic chain-of-thought)
      - UCB1 tie-break by node_id for stable selection
      - Per-query value cache to avoid redundant LLM calls
      - Batch evaluations sorted by node_id for reproducibility
    """

    def __init__(
        self,
        document: Document,
        query: str,
        model: str = DEFAULT_MODEL,
        exploration_weight: float = 1.0,
        max_iterations: int = 10,
        max_selected_nodes: int = 5,
        value_threshold: float = 0.3,
    ):
        self.document = document
        self.query = query
        self.model = model
        self.c = exploration_weight
        self.max_iterations = max_iterations
        self.max_selected_nodes = max_selected_nodes
        self.value_threshold = value_threshold

        self.nodes: dict[str, MCTSNode] = {}
        self.root_ids: list[str] = []
        self._value_cache: dict[str, float] = {}
        self._llm_calls = 0
        self._build(document.structure)

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    def _build(self, structure, parent_id: Optional[str] = None) -> None:
        """Convert document tree to MCTS node map."""
        if isinstance(structure, list):
            for item in structure:
                self._build(item, parent_id)
        elif isinstance(structure, dict):
            nid = structure.get("node_id", "")
            children = structure.get("nodes", [])
            child_ids = [c.get("node_id", "") for c in children if isinstance(c, dict)]

            self.nodes[nid] = MCTSNode(
                node_id=nid,
                title=structure.get("title", ""),
                summary=_normalize(structure.get("summary", structure.get("prefix_summary", ""))),
                children_ids=child_ids,
                parent_id=parent_id,
                is_terminal=(len(children) == 0),
            )
            if parent_id is None:
                self.root_ids.append(nid)
            for child in children:
                self._build(child, parent_id=nid)

    def _ucb1(self, node: MCTSNode, parent_visits: int) -> tuple[float, str]:
        """UCB1 with stable tie-break by node_id."""
        if node.visit_count == 0:
            return (float("inf"), node.node_id)
        score = node.avg_value + self.c * math.sqrt(math.log(parent_visits + 1) / node.visit_count)
        return (score, node.node_id)

    def _select(self) -> MCTSNode:
        """Selection: walk from root to a promising unexpanded/leaf node."""
        roots = [self.nodes[rid] for rid in self.root_ids]
        total_visits = sum(n.visit_count for n in roots) + 1
        current = max(roots, key=lambda n: self._ucb1(n, total_visits))

        while current.is_expanded and not current.is_terminal:
            children = [self.nodes[cid] for cid in current.children_ids if cid in self.nodes]
            if not children:
                break
            current = max(children, key=lambda n: self._ucb1(n, current.visit_count))
        return current

    async def _evaluate(self, node: MCTSNode) -> float:
        """LLM scores node relevance. Returns value in [0, 1]. Results are cached."""
        cache_key = f"{self.query}::{node.node_id}"
        if cache_key in self._value_cache:
            return self._value_cache[cache_key]

        prompt = (
            f"Rate relevance of this document section to the query.\n\n"
            f"Query: {self.query}\n\n"
            f"Section title: {node.title}\n"
            f"Section summary: {node.summary}\n\n"
            f"Return JSON only:\n"
            f'{{"relevance": <float 0.0-1.0>}}'
        )
        response = await achat(prompt, model=self.model, temperature=0)
        result = extract_json(response)
        value = float(result.get("relevance", 0.0))
        self._value_cache[cache_key] = value
        self._llm_calls += 1
        return value

    async def _evaluate_batch(self, nodes: list[MCTSNode]) -> list[float]:
        """Evaluate multiple nodes concurrently (sorted by node_id for stability)."""
        sorted_nodes = sorted(nodes, key=lambda n: n.node_id)
        values = await asyncio.gather(*(self._evaluate(n) for n in sorted_nodes))
        # Map back to original order
        value_map = {n.node_id: v for n, v in zip(sorted_nodes, values)}
        return [value_map[n.node_id] for n in nodes]

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Update visit counts and values up to root."""
        nid = node.node_id
        while nid is not None:
            n = self.nodes.get(nid)
            if n is None:
                break
            n.visit_count += 1
            n.total_value += value
            nid = n.parent_id

    async def run(self) -> list[dict]:
        """
        Run MCTS search and return ranked relevant nodes.

        Returns: [{'node_id': str, 'title': str, 'score': float, 'visits': int}]
        """
        # Phase 1: screen root-level nodes
        roots = [self.nodes[rid] for rid in self.root_ids]
        root_values = await self._evaluate_batch(roots)
        for node, value in zip(roots, root_values):
            node.visit_count += 1
            node.total_value += value
            node.is_expanded = True

        # Phase 2: MCTS iterations
        for _ in range(self.max_iterations):
            selected = self._select()

            if not selected.is_terminal and not selected.is_expanded:
                selected.is_expanded = True
                children = [self.nodes[cid] for cid in selected.children_ids if cid in self.nodes]
                if children:
                    values = await self._evaluate_batch(children)
                    for child, val in zip(children, values):
                        child.visit_count += 1
                        child.total_value += val
                    self._backpropagate(selected, max(values) if values else 0.0)
                    continue

            value = await self._evaluate(selected)
            self._backpropagate(selected, value)

        # Collect results above threshold
        results = []
        for nid, node in self.nodes.items():
            if node.visit_count > 0 and node.avg_value >= self.value_threshold:
                results.append({
                    "node_id": node.node_id,
                    "title": node.title,
                    "score": round(node.avg_value, 4),
                    "visits": node.visit_count,
                })

        results.sort(key=lambda x: (-x["score"], x["node_id"]))
        return results[: self.max_selected_nodes]


# ---------------------------------------------------------------------------
# Simple LLM tree search (single-pass, non-MCTS)
# ---------------------------------------------------------------------------

async def llm_tree_search(
    query: str,
    document: Document,
    model: str = DEFAULT_MODEL,
    expert_knowledge: str = "",
) -> list[dict]:
    """
    Single-pass LLM tree search. Faster but less thorough than MCTS.

    Returns: [{'node_id': str, 'title': str}]
    """
    tree_no_text = document.get_tree_without_text()

    prompt = (
        f"Find all sections relevant to the query.\n\n"
        f"Query: {query}\n\n"
        f"Document: {document.doc_name}\n"
        f"Tree structure: {json.dumps(tree_no_text, indent=2, ensure_ascii=False)}\n"
    )
    if expert_knowledge:
        prompt += f"\nExpert knowledge: {expert_knowledge}\n"

    prompt += (
        '\nReturn JSON only:\n'
        '{"node_list": ["node_id_1", "node_id_2"]}'
    )

    response = await achat(prompt, model=model, temperature=0)
    result = extract_json(response)
    node_ids = result.get("node_list", [])

    nodes = []
    for nid in node_ids:
        node = document.get_node_by_id(str(nid))
        if node:
            nodes.append({"node_id": nid, "title": node.get("title", "")})
    return nodes


# ---------------------------------------------------------------------------
# Document routing (pure LLM reasoning, no vector embeddings)
# ---------------------------------------------------------------------------

async def route_documents(
    query: str,
    documents: list[Document],
    model: str = DEFAULT_MODEL,
    top_k: int = 3,
) -> list[Document]:
    """
    Route query to relevant documents by LLM reasoning over descriptions.

    No vector embeddings or chunk splitting needed.
    """
    doc_list = [
        {"doc_id": d.doc_id, "doc_name": d.doc_name, "doc_description": d.doc_description}
        for d in documents
    ]
    prompt = (
        f"Select documents relevant to the query.\n\n"
        f"Query: {query}\n\n"
        f"Documents: {json.dumps(doc_list, indent=2, ensure_ascii=False)}\n\n"
        f'Return JSON only:\n'
        f'{{"selected_doc_ids": ["doc_id_1", "doc_id_2"]}}'
    )

    response = await achat(prompt, model=model, temperature=0)
    result = extract_json(response)
    selected_ids = result.get("selected_doc_ids", [])

    selected = [d for d in documents if d.doc_id in selected_ids][:top_k]
    if not selected:
        selected = documents[:top_k]
    return selected


# ---------------------------------------------------------------------------
# Unified search API
# ---------------------------------------------------------------------------

async def search(
    query: str,
    documents: list[Document],
    model: str = DEFAULT_MODEL,
    top_k_docs: int = 3,
    max_nodes_per_doc: int = 5,
    strategy: str = "best_first",
    mcts_iterations: int = 10,
    value_threshold: float = 0.3,
    max_llm_calls: int = 30,
    use_bm25: bool = True,
    pre_filter: Optional[PreFilter] = None,
    expert_knowledge: str = "",
) -> SearchResult:
    """
    Search across one or more documents using tree-structured retrieval.

    This is the primary API. It natively supports multi-document search:
      1. Route query to relevant documents (LLM reasoning, no vector DB)
      2. (Optional) Pre-filter scoring over tree nodes for initial ranking
      3. Tree search within each document (best_first / mcts / llm)
      4. Return ranked nodes with text content

    Args:
        query: user query
        documents: list of Document objects (single or multiple)
        model: LLM model name
        top_k_docs: max documents to search (routing stage)
        max_nodes_per_doc: max result nodes per document
        strategy: 'best_first' (default), 'mcts', or 'llm'
        mcts_iterations: MCTS iteration count (only for strategy='mcts')
        value_threshold: minimum relevance score
        max_llm_calls: max LLM calls per document (only for best_first)
        use_bm25: enable built-in BM25 pre-scoring (ignored if pre_filter is set)
        pre_filter: custom PreFilter instance for node pre-scoring (overrides use_bm25)
        expert_knowledge: optional domain knowledge to guide search
    """
    total_llm_calls = 0

    # Stage 1: document routing (skip for single doc)
    if len(documents) <= 1:
        selected = documents
    else:
        selected = await route_documents(query, documents, model, top_k=top_k_docs)
        total_llm_calls += 1

    logger.info("Selected %d documents: %s", len(selected), [d.doc_name for d in selected])

    # Stage 1.5: Pre-filter scoring (for best_first strategy)
    scorer = pre_filter
    if scorer is None and use_bm25 and strategy == "best_first" and selected:
        from .rank_bm25 import NodeBM25Index
        scorer = NodeBM25Index(selected)

    # Stage 2: tree search within each document (concurrent)
    async def _search_doc(doc: Document) -> dict:
        nonlocal total_llm_calls
        doc_llm_calls = 0

        if strategy == "best_first":
            bm25_scores = {}
            if scorer is not None:
                bm25_scores = scorer.score_nodes(query, doc.doc_id)

            searcher = BestFirstTreeSearch(
                document=doc,
                query=query,
                model=model,
                max_results=max_nodes_per_doc,
                threshold=value_threshold,
                max_llm_calls=max_llm_calls,
                bm25_scores=bm25_scores,
            )
            nodes = await searcher.run()
            doc_llm_calls = searcher.llm_calls

        elif strategy == "mcts":
            searcher = MCTSTreeSearch(
                document=doc,
                query=query,
                model=model,
                max_iterations=mcts_iterations,
                max_selected_nodes=max_nodes_per_doc,
                value_threshold=value_threshold,
            )
            nodes = await searcher.run()
            doc_llm_calls = searcher.llm_calls

        else:  # strategy == "llm"
            raw = await llm_tree_search(query, doc, model, expert_knowledge)
            nodes = [{"node_id": n["node_id"], "title": n["title"], "score": 1.0} for n in raw]
            nodes = nodes[:max_nodes_per_doc]
            doc_llm_calls = 1

        total_llm_calls += doc_llm_calls

        # Attach text to results
        for n in nodes:
            full = doc.get_node_by_id(str(n["node_id"]))
            if full:
                n["text"] = full.get("text", "")

        return {"doc_id": doc.doc_id, "doc_name": doc.doc_name, "nodes": nodes}

    doc_results = await asyncio.gather(*(_search_doc(d) for d in selected))

    return SearchResult(
        documents=[r for r in doc_results if r["nodes"]],
        query=query,
        total_llm_calls=total_llm_calls,
        strategy=strategy,
    )


def search_sync(query: str, documents: list[Document], **kwargs) -> SearchResult:
    """Synchronous wrapper around :func:`search`."""
    return asyncio.run(search(query, documents, **kwargs))

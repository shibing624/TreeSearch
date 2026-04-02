# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tree Searcher - Best-First Search over document trees.

Core algorithm:
1. Anchor Retrieval: use FTS5 to find high-value entry nodes
2. Tree Walk: BFS expansion from anchors along parent/child/sibling edges
3. Path Aggregation: select best root-to-leaf paths as results

No LLM or embedding dependencies. Pure structure-aware keyword search.
"""
import heapq
import logging
from dataclasses import dataclass, field

from .config import get_config
from .heuristics import (
    QueryPlan,
    build_query_plan,
    score_anchor,
    score_walk_node,
    score_path,
    check_title_match,
    check_phrase_match,
    compute_term_overlap,
    estimate_idf,
    is_generic_section,
)
from .tree import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search State for priority queue
# ---------------------------------------------------------------------------

@dataclass(order=False)
class SearchState:
    """State in the Best-First Search frontier.

    Attributes:
        doc_id: document identifier
        node_id: current node being explored
        score: heuristic score (higher = more promising)
        hop: number of hops from the anchor node
        source: how we reached this node (anchor/parent/child/sibling)
        path: list of node_ids from root to this node
        reasons: human-readable explanation of why this path was chosen
        max_ancestor_score: max FTS5 score along the current path (incremental, avoids recomputing)
    """
    doc_id: str
    node_id: str
    score: float
    hop: int
    source: str
    path: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    max_ancestor_score: float = 0.0

    def __lt__(self, other):
        """Max-heap: higher score = higher priority."""
        return self.score > other.score


# ---------------------------------------------------------------------------
# Path Result
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    """A scored root-to-answer path.

    Attributes:
        doc_id: document identifier
        doc_name: document name
        score: path-level score
        anchor_node_id: the anchor node that started the search
        target_node_id: the terminal node (answer)
        path: list of dicts with node_id and title
        reasons: list of human-readable scoring reasons
        snippet: text preview from the target node
    """
    doc_id: str
    doc_name: str
    score: float
    anchor_node_id: str
    target_node_id: str
    path: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    snippet: str = ""


# ---------------------------------------------------------------------------
# TreeSearcher
# ---------------------------------------------------------------------------

class TreeSearcher:
    """Best-First Tree Search engine.

    Usage::

        # Recommended: let TreeSearcher handle FTS5 scoring internally (fast path)
        searcher = TreeSearcher()
        paths, flat_nodes = searcher.search(query, documents)

        # With explicit FTS5 index (for benchmarks or custom indexes)
        from treesearch.fts import FTS5Index
        fts = FTS5Index()
        fts.index_documents(documents)
        searcher = TreeSearcher(fts_index=fts)
        paths, flat_nodes = searcher.search(query, documents)

        # Also works: pass pre-computed scores (for custom scoring logic)
        fts_scores = {"doc1": {"node1": 0.9, "node2": 0.3}}
        paths, flat_nodes = searcher.search(query, documents, fts_scores)
    """

    def __init__(self, fts_index=None):
        """
        Args:
            fts_index: optional FTS5Index instance. If provided, used for
                auto-scoring when fts_score_map is not passed to search().
                If None, the global singleton from get_fts_index() is used.
        """
        self.cfg = get_config()
        self._fts_index = fts_index

    def _auto_score(
        self,
        query: str,
        documents: list[Document],
    ) -> dict[str, dict[str, float]]:
        """Compute FTS5 scores via batch query (single SQL).

        This is the recommended fast path: one SQL query covers all documents
        instead of N per-doc queries. Documents must already be indexed in the
        FTS5 index (via FTS5Index.index_document or index_documents).

        Uses self._fts_index if provided (e.g., benchmark with custom index),
        otherwise falls back to the global singleton from get_fts_index().
        """
        if self._fts_index is not None:
            fts_index = self._fts_index
        else:
            from .fts import get_fts_index
            fts_index = get_fts_index(db_path=self.cfg.fts_db_path or None, weights={
                "title": self.cfg.fts_title_weight,
                "summary": self.cfg.fts_summary_weight,
                "body": self.cfg.fts_body_weight,
                "code_blocks": self.cfg.fts_code_weight,
                "front_matter": self.cfg.fts_front_matter_weight,
            })
            # Ensure all documents are indexed
            doc_map = {d.doc_id: d for d in documents}
            unindexed = fts_index.get_unindexed_doc_ids(list(doc_map.keys()))
            for doc_id in unindexed:
                fts_index.index_document(doc_map[doc_id])
        doc_ids = [d.doc_id for d in documents]
        return fts_index.score_nodes_batch(query, doc_ids=doc_ids)

    def search(
        self,
        query: str,
        documents: list[Document],
        fts_score_map: dict[str, dict[str, float]] | None = None,
    ) -> tuple[list[PathResult], list[dict]]:
        """Run tree search across documents.

        Args:
            query: user query string
            documents: list of Document objects to search
            fts_score_map: pre-computed {doc_id: {node_id: normalized_score}}.
                If None, scores are computed internally via FTS5Index.score_nodes_batch()
                (single SQL query — fast path). Passing a dict is supported for
                backward compatibility or custom scoring, but the internal path is
                preferred for performance.

        Returns:
            (paths, flat_nodes) where:
            - paths: top-K PathResult objects
            - flat_nodes: flattened node list with scores (compatible with old API)
        """
        # Auto-compute FTS5 scores if not provided (recommended fast path)
        if fts_score_map is None:
            fts_score_map = self._auto_score(query, documents)

        plan = build_query_plan(query)
        all_paths: list[PathResult] = []
        all_walked_nodes: list[tuple[str, str, float, float, int]] = []

        # Pre-cache flatten_tree results per doc — shared between search loop and _build_flat_nodes
        doc_flat_cache: dict[str, list[dict]] = {}

        # Sort documents by max FTS5 score descending so we process
        # the most relevant docs first. This enables early termination
        # and ensures the tree walk budget is spent on top docs.
        scored_docs = []
        for doc in documents:
            doc_scores = fts_score_map.get(doc.doc_id, {})
            if not doc_scores:
                continue
            max_score = max(doc_scores.values())
            scored_docs.append((max_score, doc, doc_scores))
        scored_docs.sort(key=lambda x: -x[0])

        # Limit to top docs by score: docs beyond this contribute little.
        # With 60+ scored docs, the bottom half typically have max_score < 0.1
        # and almost never produce top-K results.
        _MAX_DOCS_TO_WALK = 20
        scored_docs = scored_docs[:_MAX_DOCS_TO_WALK]

        for _, doc, doc_scores in scored_docs:
            from .tree import flatten_tree
            all_nodes = flatten_tree(doc.structure)
            doc_flat_cache[doc.doc_id] = all_nodes

            # Skip IDF for shallow/small docs (< 20 nodes) or docs with few
            # FTS5-scored nodes (< 5): IDF adds little value and costs O(N*terms).
            if plan.terms and len(all_nodes) > 20 and len(doc_scores) >= 5:
                corpus_texts = [n.get("text", "") for n in all_nodes]
                idf = estimate_idf(plan.terms, corpus_texts)
            else:
                idf = None

            # Stage 1: Anchor retrieval
            anchors = self._select_anchors(doc, doc_scores, plan, idf)
            if not anchors:
                continue

            # Stage 2: Tree walk from each anchor
            doc_paths, walked_states = self._tree_walk(doc, anchors, doc_scores, plan, idf)
            all_paths.extend(doc_paths)

            # Collect walked nodes with combined score and hop distance
            for state in walked_states:
                fts_s = doc_scores.get(state.node_id, 0.0)
                combined = 0.3 * state.score + 0.7 * fts_s
                all_walked_nodes.append((doc.doc_id, state.node_id, combined, fts_s, state.hop))

        # Stage 3: Select top paths globally
        all_paths.sort(key=lambda p: -p.score)
        top_paths = all_paths[:self.cfg.path_top_k]

        # Build flat_nodes (pass pre-cached flat data to avoid redundant flatten_tree calls)
        flat_nodes = self._build_flat_nodes(
            top_paths, all_walked_nodes, documents, fts_score_map, plan, doc_flat_cache
        )

        return top_paths, flat_nodes

    # ------------------------------------------------------------------
    # Stage 1: Anchor Retrieval
    # ------------------------------------------------------------------

    def _select_anchors(
        self,
        doc: Document,
        doc_scores: dict[str, float],
        plan: QueryPlan,
        idf: dict[str, float] | None = None,
    ) -> list[SearchState]:
        """Select top anchor nodes from FTS5 scores.

        Optimization: pre-filter by FTS5 score to avoid expensive scoring
        (compute_term_overlap, check_phrase_match) on low-relevance nodes.
        Only the top candidates by raw FTS5 score are fully scored.
        """
        # Pre-filter: only fully score the top N candidates by raw FTS5 score.
        # anchor_top_k anchors are needed; score 3x more candidates to allow
        # for dedup filtering. Nodes below this threshold are skipped.
        max_candidates = self.cfg.anchor_top_k * 3
        if len(doc_scores) > max_candidates:
            # Get the score threshold for top candidates
            sorted_scores = sorted(doc_scores.values(), reverse=True)
            threshold = sorted_scores[min(max_candidates, len(sorted_scores)) - 1]
        else:
            threshold = 0.0

        candidates = []
        for nid, fts_score in doc_scores.items():
            if fts_score < threshold:
                continue
            node = doc.get_node_by_id(nid)
            if not node:
                continue
            title = node.get("title", "")
            text = node.get("text", "")
            depth = doc.get_depth(nid)

            a_score = score_anchor(
                fts_score=fts_score,
                depth=depth,
                has_title_match=check_title_match(title, plan.terms),
                has_phrase_match=check_phrase_match(
                    title + " " + text, plan.phrases
                ),
                body_term_overlap=compute_term_overlap(text, plan.terms, idf),
            )
            candidates.append((a_score, nid, node))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])

        # Deduplicate: if two anchors are on the same root-to-leaf path,
        # keep only the higher-scoring one
        selected: list[SearchState] = []
        selected_paths: set[str] = set()

        for a_score, nid, node in candidates:
            if len(selected) >= self.cfg.anchor_top_k:
                break

            # Check path overlap
            path_to_root = doc.get_path_to_root(nid)
            path_key = ">".join(path_to_root[:3])  # use top 3 ancestors as signature
            if path_key in selected_paths:
                continue
            selected_paths.add(path_key)

            reasons = [f"FTS5 score={doc_scores.get(nid, 0):.3f}"]
            if check_title_match(node.get("title", ""), plan.terms):
                reasons.append("title match")

            state = SearchState(
                doc_id=doc.doc_id,
                node_id=nid,
                score=a_score,
                hop=0,
                source="anchor",
                path=path_to_root,
                reasons=reasons,
                max_ancestor_score=doc_scores.get(nid, 0.0),
            )
            selected.append(state)

        logger.debug(
            "Doc %s: selected %d anchors from %d candidates",
            doc.doc_id, len(selected), len(candidates),
        )
        return selected

    # ------------------------------------------------------------------
    # Stage 2: Tree Walk (Best-First Search)
    # ------------------------------------------------------------------

    def _tree_walk(
        self,
        doc: Document,
        anchors: list[SearchState],
        doc_scores: dict[str, float],
        plan: QueryPlan,
        idf: dict[str, float] | None = None,
    ) -> tuple[list[PathResult], list[SearchState]]:
        """Run Best-First Search from anchors, return scored paths and all visited states."""
        visited: set[str] = set()
        frontier: list[SearchState] = []
        best_states: list[SearchState] = []
        expansion_count = 0

        # Pre-cache term_overlap for all FTS5-scored nodes (avoids redundant text scans)
        _overlap_cache: dict[str, float] = {}
        if plan.terms:
            for nid in doc_scores:
                node = doc.get_node_by_id(nid)
                if node:
                    _overlap_cache[nid] = compute_term_overlap(
                        node.get("text", ""), plan.terms, idf
                    )

        # Initialize frontier with anchors
        for anchor in anchors:
            heapq.heappush(frontier, anchor)

        while frontier and expansion_count < self.cfg.max_expansions:
            state = heapq.heappop(frontier)

            if state.node_id in visited:
                continue
            visited.add(state.node_id)

            # Record as candidate
            best_states.append(state)
            expansion_count += 1

            # Early stop: found a very high-scoring path
            if state.score >= self.cfg.early_stop_score:
                logger.debug("Early stop: score=%.3f >= %.3f", state.score, self.cfg.early_stop_score)
                break

            # Stop expanding if frontier score is too low
            if state.score < self.cfg.min_frontier_score:
                continue

            # Don't expand beyond max hops
            if state.hop >= self.cfg.max_hops:
                continue

            # Expand neighbors
            neighbors = self._get_neighbors(doc, state)
            for nid, relation in neighbors:
                if nid in visited:
                    continue

                node = doc.get_node_by_id(nid)
                if not node:
                    continue

                title = node.get("title", "")
                text = node.get("text", "")
                lexical = doc_scores.get(nid, 0.0)

                # Use pre-cached overlap or compute on-demand for walk-only nodes
                if nid in _overlap_cache:
                    overlap = _overlap_cache[nid]
                elif plan.terms:
                    overlap = compute_term_overlap(text, plan.terms, idf)
                    _overlap_cache[nid] = overlap
                else:
                    overlap = 0.0

                # Incremental max_ancestor_score: O(1) instead of O(path_len) generator
                new_max_anc = max(state.max_ancestor_score, doc_scores.get(state.node_id, 0.0))

                w_score = score_walk_node(
                    lexical_score=lexical,
                    has_title_match=check_title_match(title, plan.terms),
                    has_phrase_match=check_phrase_match(
                        title + " " + text, plan.phrases
                    ),
                    body_term_overlap=overlap,
                    ancestor_support=new_max_anc,
                    hop=state.hop + 1,
                    is_redundant=False,
                    max_hops=self.cfg.max_hops,
                )

                # Build path: for parent/sibling, use their own path_to_root;
                # for child, extend current path
                if relation == "child":
                    new_path = state.path + [nid]
                else:
                    new_path = doc.get_path_to_root(nid)

                # Defer reasons building: only store minimal info during walk.
                # Full reasons are built in _states_to_paths for top-K states.
                new_state = SearchState(
                    doc_id=doc.doc_id,
                    node_id=nid,
                    score=w_score,
                    hop=state.hop + 1,
                    source=relation,
                    path=new_path,
                    reasons=state.reasons,  # share reference, don't copy
                    max_ancestor_score=new_max_anc,
                )
                heapq.heappush(frontier, new_state)

        logger.debug(
            "Doc %s: expanded %d nodes, found %d candidates",
            doc.doc_id, expansion_count, len(best_states),
        )

        # Convert best states to PathResults
        paths = self._states_to_paths(doc, best_states, doc_scores, plan)
        return paths, best_states

    def _get_neighbors(
        self, doc: Document, state: SearchState
    ) -> list[tuple[str, str]]:
        """Get neighbor nodes for expansion.

        Returns list of (node_id, relation) tuples.
        Expansion priority:
        - Children first (drill down into details)
        - Parent (get broader context)
        - Siblings (cover adjacent sections), limited by max_siblings
        """
        neighbors: list[tuple[str, str]] = []
        nid = state.node_id

        # Children
        children = doc.get_children_ids(nid)
        for cid in children:
            neighbors.append((cid, "child"))

        # Parent
        pid = doc.get_parent_id(nid)
        if pid:
            neighbors.append((pid, "parent"))

        # Siblings (limited)
        siblings = doc.get_sibling_ids(nid)[:self.cfg.max_siblings]
        for sid in siblings:
            neighbors.append((sid, "sibling"))

        return neighbors

    def _states_to_paths(
        self,
        doc: Document,
        states: list[SearchState],
        doc_scores: dict[str, float],
        plan: QueryPlan,
    ) -> list[PathResult]:
        """Convert search states into scored PathResults.

        Only processes top path_top_k*2 states to avoid expensive path
        construction for low-scoring states that won't make the final cut.
        """
        results: list[PathResult] = []
        seen_targets: set[str] = set()

        # Sort states by score descending
        states.sort(key=lambda s: -s.score)

        # Limit to top candidates — building full paths is expensive
        max_to_process = self.cfg.path_top_k * 2

        for state in states:
            if len(results) >= max_to_process:
                break

            # Deduplicate by target node
            if state.node_id in seen_targets:
                continue
            seen_targets.add(state.node_id)

            # Build full path from root to target
            full_path = doc.get_path_to_root(state.node_id)
            path_titles = []
            path_texts = []
            path_dicts = []
            for pid in full_path:
                pnode = doc.get_node_by_id(pid)
                title = pnode.get("title", "") if pnode else ""
                text = pnode.get("text", "") if pnode else ""
                path_titles.append(title)
                path_texts.append(text)
                path_dicts.append({"node_id": pid, "title": title})

            # Score the path (with body text and leaf FTS5 score)
            p_score = score_path(
                leaf_score=state.score,
                path_titles=path_titles,
                path_texts=path_texts,
                query_terms=plan.terms,
                path_length=len(full_path),
                leaf_fts_score=doc_scores.get(state.node_id, 0.0),
            )

            # Get snippet from target node
            target_node = doc.get_node_by_id(state.node_id)
            snippet = ""
            if target_node:
                text = target_node.get("text", "")
                snippet = text[:300] if text else ""

            # Determine anchor (first node in state.path, or the state itself if hop=0)
            anchor_id = state.path[0] if state.path else state.node_id

            result = PathResult(
                doc_id=doc.doc_id,
                doc_name=doc.doc_name,
                score=round(p_score, 4),
                anchor_node_id=anchor_id,
                target_node_id=state.node_id,
                path=path_dicts,
                reasons=state.reasons,
                snippet=snippet,
            )
            results.append(result)

        results.sort(key=lambda r: -r.score)
        return results[:self.cfg.path_top_k]

    # ------------------------------------------------------------------
    # Stage 3: Convert to flat nodes (backward compatible)
    # ------------------------------------------------------------------

    def _build_flat_nodes(
        self,
        paths: list[PathResult],
        walked_nodes: list[tuple[str, str, float, float, int]],
        documents: list[Document],
        fts_score_map: dict[str, dict[str, float]],
        plan: QueryPlan | None = None,
        doc_flat_cache: dict[str, list[dict]] | None = None,
    ) -> list[dict]:
        """Build flat node list: FTS5 base + structural reranking from tree walk.

        Optimized pipeline (P0+P1+P2):
        - P0: Pre-compute NodeContext once per node (title_lower, text_lower,
              term_hits, parent_id, children_ids, etc.) to eliminate redundant
              get_node_by_id / .lower() / substring matching across stages.
        - P1: Merge stages 1b (generic demotion) + 1c (leaf preference) +
              7 (title match boost) into a single pass over NodeContext.
        - P2: Stages 4-7 only process top-100 nodes by score to skip long tail.

        Strategy (7 stages, optimized execution):
        1. Base: FTS5 scores + merged 1b/1c/7 in one pass via NodeContext
        2. Title-prefix propagation (for ::: delimited hierarchies)
        3. Walk boost + Walk-only injection
        4-7. Parent context / Term density / Subtree evidence / Title match
             (applied to top-100 nodes only)
        """
        from .tree import flatten_tree

        doc_map = {d.doc_id: d for d in documents}
        node_scores: dict[tuple[str, str], float] = {}

        # Use pre-cached flatten_tree results to avoid redundant O(N) traversals
        def _get_flat_nodes(doc_id: str) -> list[dict]:
            if doc_flat_cache and doc_id in doc_flat_cache:
                return doc_flat_cache[doc_id]
            doc = doc_map.get(doc_id)
            return flatten_tree(doc.structure) if doc else []

        # Extract plan data once
        has_plan_terms = bool(plan and plan.terms)
        terms = plan.terms if has_plan_terms else []
        n_terms = len(terms)

        # Per-doc top-N cutoff: only keep highest-scoring nodes per document.
        # N scales with the number of documents: fewer docs → allow more nodes per doc
        # so that large single-document corpora (e.g. FinanceBench PDFs with 200+ nodes)
        # are not over-truncated, while many-doc corpora (e.g. CodeSearchNet 1000 docs)
        # still get the speed benefit.
        n_docs = max(1, len(fts_score_map))
        _MAX_NODES_PER_DOC = max(50, min(500, 2000 // n_docs))
        for doc_id, doc_scores in list(fts_score_map.items()):
            if len(doc_scores) > _MAX_NODES_PER_DOC:
                top_items = sorted(doc_scores.items(), key=lambda x: -x[1])[:_MAX_NODES_PER_DOC]
                fts_score_map[doc_id] = dict(top_items)

        # 1. Base: FTS5 scores as foundation
        for doc_id, doc_scores in fts_score_map.items():
            for nid, fts_s in doc_scores.items():
                node_scores[(doc_id, nid)] = fts_s

        # -------------------------------------------------------------------
        # P0: Pre-compute NodeContext for all scored nodes (ONE pass)
        # This replaces repeated get_node_by_id + .lower() + substring matching
        # across stages 1b, 1c, 3, 4, 5, 6, 7.
        # -------------------------------------------------------------------
        # NodeContext tuple: (title_lower, text_lower, full_lower, term_hits,
        #                     term_overlap, parent_id, children_ids, depth, text_len)
        _ctx: dict[tuple[str, str], tuple] = {}
        # Indices into the context tuple
        _TITLE = 0; _TEXT = 1; _FULL = 2; _HITS = 3; _OVERLAP = 4
        _PID = 5; _CHILDREN = 6; _DEPTH = 7; _TEXTLEN = 8

        for (doc_id, nid) in node_scores:
            doc = doc_map.get(doc_id)
            if not doc:
                continue
            node = doc.get_node_by_id(nid)
            if not node:
                continue
            title_lower = (node.get("title", "") or "").lower()
            text_lower = (node.get("text", "") or "").lower()
            full_lower = title_lower + " " + text_lower if title_lower else text_lower
            if has_plan_terms and full_lower.strip():
                hits = sum(1 for t in terms if t in full_lower)
                overlap = hits / n_terms
            else:
                hits = 0
                overlap = 0.0
            pid = doc.get_parent_id(nid)
            children = doc.get_children_ids(nid)
            depth = doc.get_depth(nid)
            text_len = len(node.get("text", ""))
            _ctx[(doc_id, nid)] = (
                title_lower, text_lower, full_lower, hits, overlap,
                pid, children, depth, text_len,
            )

        # -------------------------------------------------------------------
        # P1: Merged pass — Stage 1b (generic demotion) + 1c (leaf preference)
        # Applied in one loop over _ctx instead of 2 separate loops.
        # Stage 7 (title match boost) is deferred to the top-100 pass below.
        # -------------------------------------------------------------------
        for key, ctx in _ctx.items():
            score = node_scores[key]
            depth = ctx[_DEPTH]
            title_lower = ctx[_TITLE]
            children = ctx[_CHILDREN]
            text_len = ctx[_TEXTLEN]

            # 1b. Generic section demotion
            if depth > 0:
                # Inline is_generic_section to avoid function call overhead
                base_title = title_lower.split(" ::: ")[0].strip() if " ::: " in title_lower else title_lower.strip()
                if is_generic_section(base_title, depth):
                    # Query-aware: don't demote if query terms mention this section
                    if has_plan_terms and any(t in base_title for t in terms):
                        pass  # skip demotion
                    else:
                        score *= 0.70

            # 1c. Leaf node preference
            if not children and text_len > 100:
                score *= 1.08

            node_scores[key] = score

        # 2. Title-prefix propagation: scan ALL low-score nodes in each document.
        # In QASPER, section titles use ::: delimiter for hierarchy.
        for doc_id, doc_scores in fts_score_map.items():
            doc = doc_map.get(doc_id)
            if not doc:
                continue

            # Pre-collect high-score nodes as potential parents (cap at 20)
            parent_candidates = []
            for nid, fts_s in doc_scores.items():
                if fts_s < 0.15:
                    continue
                ctx = _ctx.get((doc_id, nid))
                if ctx and ctx[_TITLE]:
                    parent_candidates.append((nid, ctx[_TITLE], fts_s))
            if len(parent_candidates) > 20:
                parent_candidates.sort(key=lambda x: -x[2])
                parent_candidates = parent_candidates[:20]

            if not parent_candidates:
                continue

            # Propagate to low-score nodes with matching title prefix
            all_nodes = _get_flat_nodes(doc_id)
            for node_dict in all_nodes:
                nid = node_dict.get("node_id", "")
                key = (doc_id, nid)
                current_score = node_scores.get(key, 0.0)
                if current_score > 0.005:
                    continue

                title = node_dict.get("title", "")
                if not title:
                    continue
                title_lower = title.lower()

                best_parent_fts = 0.0
                for p_nid, p_title, p_fts in parent_candidates:
                    if p_nid == nid:
                        continue
                    if title_lower.startswith(p_title) and len(p_title) < len(title_lower):
                        best_parent_fts = max(best_parent_fts, p_fts)

                if best_parent_fts >= 0.15:
                    propagated = best_parent_fts * 0.30
                    node_scores[key] = max(current_score, propagated)

        # 3. Walk boost + Walk-only injection
        walked_set: set[tuple[str, str]] = set()
        for doc_id, nid, combined_score, fts_s, hop in walked_nodes:
            key = (doc_id, nid)
            walked_set.add(key)
            if key in node_scores:
                walk_bonus = 0.15 * combined_score
                node_scores[key] = node_scores[key] + walk_bonus
            elif fts_s == 0 and has_plan_terms and hop <= 3:
                # Walk-only injection: use pre-computed context if available,
                # otherwise compute on demand (walk nodes may not be in _ctx yet)
                ctx = _ctx.get(key)
                if ctx:
                    full = ctx[_FULL]
                    overlap = ctx[_OVERLAP]
                else:
                    doc = doc_map.get(doc_id)
                    if not doc:
                        continue
                    node = doc.get_node_by_id(nid)
                    if not node:
                        continue
                    text = (node.get("text", "") or "").lower()
                    title = (node.get("title", "") or "").lower()
                    full = title + " " + text
                    if full.strip():
                        hits = sum(1 for t in terms if t in full)
                        overlap = hits / n_terms
                    else:
                        overlap = 0.0
                if overlap >= 0.40:
                    hop_decay = 1.0 - 0.15 * (hop - 1)
                    inject_score = min(0.25 * overlap * hop_decay, 0.20)
                    node_scores[key] = inject_score

        # -------------------------------------------------------------------
        # P2: Top-N cutoff for expensive reranking stages (4-7)
        # N scales with number of documents: fewer docs → more nodes per doc get
        # boosted (critical for large single-doc corpora like FinanceBench PDFs).
        # -------------------------------------------------------------------
        _TOP_N_RERANK = max(100, min(500, 2000 // max(1, len(fts_score_map))))
        if len(node_scores) > _TOP_N_RERANK:
            top_keys = sorted(node_scores, key=lambda k: -node_scores[k])[:_TOP_N_RERANK]
            rerank_set = set(top_keys)
        else:
            rerank_set = set(node_scores.keys())

        # Ensure rerank nodes have context (walk-injected nodes may be missing)
        for key in rerank_set:
            if key not in _ctx:
                doc_id, nid = key
                doc = doc_map.get(doc_id)
                if not doc:
                    continue
                node = doc.get_node_by_id(nid)
                if not node:
                    continue
                title_lower = (node.get("title", "") or "").lower()
                text_lower = (node.get("text", "") or "").lower()
                full_lower = title_lower + " " + text_lower if title_lower else text_lower
                if has_plan_terms and full_lower.strip():
                    hits = sum(1 for t in terms if t in full_lower)
                    overlap = hits / n_terms
                else:
                    hits = 0
                    overlap = 0.0
                pid = doc.get_parent_id(nid)
                children = doc.get_children_ids(nid)
                depth = doc.get_depth(nid)
                text_len = len(node.get("text", ""))
                _ctx[key] = (
                    title_lower, text_lower, full_lower, hits, overlap,
                    pid, children, depth, text_len,
                )

        # 4. Parent context boost (key for financial docs)
        # Scan ALL nodes (via _get_flat_nodes) so that nodes not in FTS5 top-N
        # can still be lifted by their high-scoring parent/grandparent.
        # This is critical for FinanceBench: parent "Revenue" has high FTS5 score
        # but child nodes with the actual numbers may rank below top-N.
        if has_plan_terms:
            for doc_id, doc_scores in fts_score_map.items():
                doc = doc_map.get(doc_id)
                if not doc:
                    continue
                all_nodes = _get_flat_nodes(doc_id)
                for node_dict in all_nodes:
                    nid = node_dict.get("node_id", "")
                    key = (doc_id, nid)
                    current = node_scores.get(key, 0.0)
                    if current < 0.01:
                        continue

                    # Use pre-computed context if available, else compute inline
                    ctx = _ctx.get(key)
                    if ctx:
                        overlap = ctx[_OVERLAP]
                        pid = ctx[_PID]
                    else:
                        text = (node_dict.get("text", "") or "").lower()
                        title = (node_dict.get("title", "") or "").lower()
                        full_text = title + " " + text
                        if not full_text.strip():
                            continue
                        hits = sum(1 for t in terms if t in full_text)
                        overlap = hits / n_terms
                        pid = doc.get_parent_id(nid)

                    if overlap < 0.20:
                        continue

                    # 4a. Parent boost
                    if pid:
                        parent_fts = doc_scores.get(pid, 0.0)
                        if parent_fts > current + 0.06:
                            parent_boost = 0.50 * parent_fts * overlap
                            node_scores[key] = current + parent_boost

                    # 4b. Grandparent boost
                    grandparent_pid = None
                    if pid:
                        p_ctx = _ctx.get((doc_id, pid))
                        if p_ctx:
                            grandparent_pid = p_ctx[_PID]
                        else:
                            grandparent_pid = doc.get_parent_id(pid)
                    if grandparent_pid:
                        gp_fts = doc_scores.get(grandparent_pid, 0.0)
                        if gp_fts > current + 0.10:
                            gp_boost = 0.25 * gp_fts * overlap
                            node_scores[key] = node_scores.get(key, current) + gp_boost

        # 5. Term density boost (top-100 only)
        if has_plan_terms and n_terms >= 2:
            for key in rerank_set:
                score = node_scores[key]
                if score < 0.05:
                    continue
                ctx = _ctx.get(key)
                if not ctx:
                    continue
                overlap = ctx[_OVERLAP]
                if overlap >= 0.5:
                    density_bonus = 0.12 * overlap * score
                    node_scores[key] = score + density_bonus

        # 6. Subtree evidence aggregation (top-100 only)
        for key in rerank_set:
            score = node_scores[key]
            if score < 0.05:
                continue
            ctx = _ctx.get(key)
            if not ctx:
                continue
            doc_id, nid = key
            pid = ctx[_PID]
            parent_score = node_scores.get((doc_id, pid), 0) if pid else 0
            children_ids = ctx[_CHILDREN]
            children_scores = [node_scores.get((doc_id, cid), 0) for cid in children_ids]

            # Get sibling_ids via parent's children (avoid extra doc lookup)
            sibling_scores = []
            if pid:
                p_ctx = _ctx.get((doc_id, pid))
                if p_ctx:
                    for sid in p_ctx[_CHILDREN]:
                        if sid != nid:
                            sibling_scores.append(node_scores.get((doc_id, sid), 0))
                else:
                    doc = doc_map.get(doc_id)
                    if doc:
                        for sid in doc.get_sibling_ids(nid):
                            sibling_scores.append(node_scores.get((doc_id, sid), 0))

            best_child = max(children_scores) if children_scores else 0
            best_sibling = max(sibling_scores) if sibling_scores else 0
            context = max(parent_score, best_child, best_sibling)

            if context > score * 1.5 and context > 0.15:
                lift = 0.30 * (context - score)
                node_scores[key] = score + lift

        # 7. Title match boost (top-100 only, uses pre-computed term_hits)
        if has_plan_terms:
            for key in rerank_set:
                score = node_scores[key]
                if score < 0.05:
                    continue
                ctx = _ctx.get(key)
                if not ctx:
                    continue
                title_lower = ctx[_TITLE]
                if not title_lower:
                    continue
                # Compute title-specific hits (not full_text hits)
                title_hits = sum(1 for t in terms if t in title_lower)
                if title_hits > 0:
                    title_overlap = title_hits / n_terms
                    title_bonus = 0.15 * title_overlap * max(score, 0.10)
                    node_scores[key] = score + title_bonus

        # Build flat node dicts (uses _ctx to avoid extra get_node_by_id calls)
        flat_nodes: list[dict] = []
        for (doc_id, nid), score in node_scores.items():
            ctx = _ctx.get((doc_id, nid))
            if ctx:
                # Recover original-case title from node (ctx only has lowercase)
                doc = doc_map.get(doc_id)
                node = doc.get_node_by_id(nid) if doc else None
                title = node.get("title", "") if node else ""
                text = node.get("text", "") if node else ""
            else:
                doc = doc_map.get(doc_id)
                if not doc:
                    continue
                node = doc.get_node_by_id(nid)
                if not node:
                    continue
                title = node.get("title", "")
                text = node.get("text", "")
            flat_nodes.append({
                "node_id": nid,
                "doc_id": doc_id,
                "doc_name": doc.doc_name if doc else "",
                "title": title,
                "score": round(score, 4),
                "text": text,
            })

        flat_nodes.sort(key=lambda x: -x["score"])
        return flat_nodes

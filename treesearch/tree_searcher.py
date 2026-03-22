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
    """
    doc_id: str
    node_id: str
    score: float
    hop: int
    source: str
    path: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

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
        searcher = TreeSearcher()
        paths, flat_nodes = searcher.search(query, documents, fts_scores)
    """

    def __init__(self):
        self.cfg = get_config()

    def search(
        self,
        query: str,
        documents: list[Document],
        fts_score_map: dict[str, dict[str, float]],
    ) -> tuple[list[PathResult], list[dict]]:
        """Run tree search across documents.

        Args:
            query: user query string
            documents: list of Document objects to search
            fts_score_map: {doc_id: {node_id: normalized_score}} from FTS5

        Returns:
            (paths, flat_nodes) where:
            - paths: top-K PathResult objects
            - flat_nodes: flattened node list with scores (compatible with old API)
        """
        plan = build_query_plan(query)
        all_paths: list[PathResult] = []
        all_walked_nodes: list[tuple[str, str, float, float]] = []

        for doc in documents:
            doc_scores = fts_score_map.get(doc.doc_id, {})
            if not doc_scores:
                continue

            # Compute IDF weights for query terms from this document's nodes
            from .tree import flatten_tree
            all_nodes = flatten_tree(doc.structure)
            corpus_texts = [n.get("text", "") for n in all_nodes]
            idf = estimate_idf(plan.terms, corpus_texts) if plan.terms else None

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

        # Build flat_nodes
        flat_nodes = self._build_flat_nodes(top_paths, all_walked_nodes, documents, fts_score_map, plan)

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
        """Select top anchor nodes from FTS5 scores."""
        candidates = []
        for nid, fts_score in doc_scores.items():
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
            candidates.append((a_score, nid))

        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])

        # Deduplicate: if two anchors are on the same root-to-leaf path,
        # keep only the higher-scoring one
        selected: list[SearchState] = []
        selected_paths: set[str] = set()

        for a_score, nid in candidates:
            if len(selected) >= self.cfg.anchor_top_k:
                break

            # Check path overlap
            path_to_root = doc.get_path_to_root(nid)
            path_key = ">".join(path_to_root[:3])  # use top 3 ancestors as signature
            if path_key in selected_paths:
                continue
            selected_paths.add(path_key)

            reasons = [f"FTS5 score={doc_scores.get(nid, 0):.3f}"]
            node = doc.get_node_by_id(nid)
            if node and check_title_match(node.get("title", ""), plan.terms):
                reasons.append("title match")

            state = SearchState(
                doc_id=doc.doc_id,
                node_id=nid,
                score=a_score,
                hop=0,
                source="anchor",
                path=path_to_root,
                reasons=reasons,
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

                # Compute ancestor support: max score along current path
                ancestor_support = max(
                    (doc_scores.get(pid, 0.0) for pid in state.path),
                    default=0.0,
                )

                w_score = score_walk_node(
                    lexical_score=lexical,
                    has_title_match=check_title_match(title, plan.terms),
                    has_phrase_match=check_phrase_match(
                        title + " " + text, plan.phrases
                    ),
                    body_term_overlap=compute_term_overlap(text, plan.terms, idf),
                    ancestor_support=ancestor_support,
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

                reasons = list(state.reasons)
                reasons.append(f"{relation}: {title[:40]}")

                new_state = SearchState(
                    doc_id=doc.doc_id,
                    node_id=nid,
                    score=w_score,
                    hop=state.hop + 1,
                    source=relation,
                    path=new_path,
                    reasons=reasons,
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
        """Convert search states into scored PathResults."""
        results: list[PathResult] = []
        seen_targets: set[str] = set()

        # Sort states by score descending
        states.sort(key=lambda s: -s.score)

        for state in states:
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
    ) -> list[dict]:
        """Build flat node list: FTS5 base + structural reranking from tree walk.

        Strategy (5 stages):
        1. Base: all FTS5 scored nodes at their original score
        1b. Generic section demotion (for academic paper sections)
        2. Title-prefix propagation (for ::: delimited hierarchies)
        3. Walk boost + Walk-only injection
        4. Parent context boost: propagate relevance from structurally close
           high-scoring ancestors to low-scoring children (key for financial docs)
        5. Term density boost: rerank by query term coverage
        """
        doc_map = {d.doc_id: d for d in documents}
        node_scores: dict[tuple[str, str], float] = {}

        # 1. Base: FTS5 scores as foundation
        for doc_id, doc_scores in fts_score_map.items():
            for nid, fts_s in doc_scores.items():
                node_scores[(doc_id, nid)] = fts_s

        # 1b. Generic section demotion: reduce score of overview sections
        # (Abstract, Introduction, Conclusion, etc.) that get inflated BM25
        # scores due to broad topic coverage rather than specific answers.
        for doc_id, doc_scores in fts_score_map.items():
            doc = doc_map.get(doc_id)
            if not doc:
                continue
            for nid in doc_scores:
                node = doc.get_node_by_id(nid)
                if not node:
                    continue
                title = node.get("title", "")
                depth = doc.get_depth(nid)
                if is_generic_section(title, depth):
                    key = (doc_id, nid)
                    node_scores[key] *= 0.60

        # 2. Title-prefix propagation: scan ALL low-score nodes in each document.
        # In QASPER, section titles use ::: delimiter for hierarchy,
        # e.g. "Systems ::: Baseline" is logically under "Systems".
        # If a logical parent has high FTS5 score, propagate a fraction.
        from .tree import flatten_tree
        for doc_id, doc_scores in fts_score_map.items():
            doc = doc_map.get(doc_id)
            if not doc:
                continue

            # Pre-collect high-score nodes as potential parents
            parent_candidates = []
            for nid, fts_s in doc_scores.items():
                if fts_s < 0.15:
                    continue
                node = doc.get_node_by_id(nid)
                if not node:
                    continue
                title = node.get("title", "").lower()
                if title:
                    parent_candidates.append((nid, title, fts_s))

            if not parent_candidates:
                continue

            # Propagate to low-score nodes with matching title prefix
            all_nodes = flatten_tree(doc.structure)
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
        # Walk boost: walked nodes with FTS5 scores get structural confirmation bonus.
        # Walk-only injection: nodes FTS5 missed but structurally close to anchors.
        walked_set: set[tuple[str, str]] = set()
        for doc_id, nid, combined_score, fts_s, hop in walked_nodes:
            key = (doc_id, nid)
            walked_set.add(key)
            if key in node_scores:
                # Node found by both FTS5 and Walk — structural confirmation bonus
                walk_bonus = 0.15 * combined_score
                node_scores[key] = node_scores[key] + walk_bonus
            elif fts_s > 0:
                # Walk-discovered node with some FTS5 signal
                node_scores[key] = combined_score
            elif plan and plan.terms and hop <= 3:
                # Walk-only node (FTS5 = 0): inject if text contains query terms.
                # Relaxed threshold (0.25) to catch more FTS5 blind spots in
                # financial tables where terms are scattered across cells.
                doc = doc_map.get(doc_id)
                if doc:
                    node = doc.get_node_by_id(nid)
                    if node:
                        text = (node.get("text", "") or "").lower()
                        title = (node.get("title", "") or "").lower()
                        full = title + " " + text
                        if full.strip():
                            hits = sum(1 for t in plan.terms if t in full)
                            overlap = hits / len(plan.terms)
                            if overlap >= 0.25:
                                hop_decay = 1.0 - 0.2 * (hop - 1)  # hop 1→1.0, 2→0.8, 3→0.6
                                inject_score = 0.25 * overlap * hop_decay
                                node_scores[key] = inject_score

        # 4. Parent context boost (key for financial docs)
        # Core insight from diagnosis: 24/27 both-miss cases have relevant nodes
        # with FTS5 scores in 0.15-0.60 range, but ranked too low (rank 9-78).
        # Financial docs have the pattern: parent node (e.g., "Revenue") has high
        # FTS5 score, but the *child* containing the actual number has lower score.
        # This stage conditionally propagates parent relevance to children that
        # also contain query terms (to avoid blindly boosting unrelated children).
        if plan and plan.terms:
            for doc_id, doc_scores in fts_score_map.items():
                doc = doc_map.get(doc_id)
                if not doc:
                    continue
                all_nodes = flatten_tree(doc.structure)
                for node_dict in all_nodes:
                    nid = node_dict.get("node_id", "")
                    key = (doc_id, nid)
                    current = node_scores.get(key, 0.0)
                    if current < 0.01:
                        continue

                    # Only boost if the node itself contains query terms
                    text = (node_dict.get("text", "") or "").lower()
                    title = (node_dict.get("title", "") or "").lower()
                    full_text = title + " " + text
                    if not full_text.strip():
                        continue
                    hits = sum(1 for t in plan.terms if t in full_text)
                    overlap = hits / len(plan.terms)
                    if overlap < 0.25:
                        continue  # Node doesn't contain enough query terms

                    # 4a. Boost from parent: if parent has higher FTS5, child gets lifted
                    pid = doc.get_parent_id(nid)
                    if pid:
                        parent_fts = doc_scores.get(pid, 0.0)
                        if parent_fts > current * 1.2:
                            parent_boost = 0.18 * parent_fts * overlap
                            node_scores[key] = current + parent_boost

                    # 4b. Sibling zone boost: if ≥2 siblings have FTS5 scores, this is
                    # a relevant region (e.g., financial table with multiple matched rows)
                    sibling_ids = doc.get_sibling_ids(nid)
                    if sibling_ids:
                        scored_siblings = [
                            doc_scores.get(sid, 0.0)
                            for sid in sibling_ids
                            if doc_scores.get(sid, 0.0) > 0.05
                        ]
                        if len(scored_siblings) >= 2:
                            avg_sibling = sum(scored_siblings) / len(scored_siblings)
                            zone_boost = 0.08 * avg_sibling * overlap
                            node_scores[key] = node_scores.get(key, current) + zone_boost

        # 5. Term density boost: nodes with higher query term density get boosted.
        # This reranks nodes based on how many unique query terms appear in the text,
        # improving precision for multi-term queries (e.g., financial queries).
        if plan and plan.terms and len(plan.terms) >= 2:
            for (doc_id, nid), score in list(node_scores.items()):
                if score < 0.01:
                    continue
                doc = doc_map.get(doc_id)
                if not doc:
                    continue
                node = doc.get_node_by_id(nid)
                if not node:
                    continue
                text = (node.get("text", "") or "").lower()
                title = (node.get("title", "") or "").lower()
                combined_text = title + " " + text
                if not combined_text.strip():
                    continue
                # Count unique query terms present
                hits = sum(1 for t in plan.terms if t in combined_text)
                overlap = hits / len(plan.terms)
                if overlap >= 0.5:
                    # Boost nodes with high term coverage (lowered from 0.6 to 0.5)
                    density_bonus = 0.12 * overlap * score
                    node_scores[(doc_id, nid)] += density_bonus

        # Build flat node dicts
        flat_nodes: list[dict] = []
        for (doc_id, nid), score in node_scores.items():
            doc = doc_map.get(doc_id)
            if not doc:
                continue
            node = doc.get_node_by_id(nid)
            if not node:
                continue
            flat_nodes.append({
                "node_id": nid,
                "doc_id": doc_id,
                "doc_name": doc.doc_name,
                "title": node.get("title", ""),
                "score": round(score, 4),
                "text": node.get("text", ""),
            })

        flat_nodes.sort(key=lambda x: -x["score"])
        return flat_nodes

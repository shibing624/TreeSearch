# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tree search over document structures — FTS5 keyword matching
              and the unified multi-document ``search()`` pipeline.

              No LLM calls at search time. All scoring is done via FTS5.
"""
import asyncio
import logging
import os
import re
from typing import Optional, Protocol, runtime_checkable

from .tree import Document
from .config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PreFilter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PreFilter(Protocol):
    """Protocol for pre-scoring nodes before tree search.

    Implementations must provide ``score_nodes`` which returns a dict
    mapping node_id -> relevance score for a given query and document.
    Built-in implementation: ``FTS5Index``, ``GrepFilter``.
    """

    def score_nodes(self, query: str, doc_id: str) -> dict[str, float]:
        """Return {node_id: score} for nodes in the given document."""
        ...


class GrepFilter:
    """
    Literal string or regex matching filter.

    Provides exact matching capabilities to complement semantic search.
    When a document has a ``source_path`` pointing to an existing file and
    ``rg`` (ripgrep) is available on PATH, matching is delegated to ``rg``
    for significantly faster line-level search.  Otherwise falls back to
    pure-Python scanning over the in-memory tree.
    """

    def __init__(self, documents: list[Document], case_sensitive: bool = False, use_regex: bool = False):
        self._doc_map = {doc.doc_id: doc for doc in documents}
        self.case_sensitive = case_sensitive
        self.use_regex = use_regex

        # Build source_path -> doc_id mapping for rg mode
        self._path_to_doc: dict[str, str] = {}
        for doc in documents:
            sp = doc.metadata.get("source_path", "")
            if sp and os.path.isfile(sp):
                self._path_to_doc[sp] = doc.doc_id

    def score_nodes(self, query: str, doc_id: str) -> dict[str, float]:
        """Return {node_id: score} for nodes that contain the query literal/regex."""
        doc = self._doc_map.get(doc_id)
        if not doc:
            return {}

        sp = doc.metadata.get("source_path", "")
        # Try rg if the source file exists
        if sp and os.path.isfile(sp):
            from .ripgrep import rg_available, rg_search
            if rg_available():
                result = self._rg_score(query, doc, sp)
                if result:
                    return result
                # rg returned nothing — might be a rg error, fall through to native

        # Fallback: native Python search
        return self._native_score(query, doc)

    def _rg_score(self, query: str, doc: Document, source_path: str) -> dict[str, float]:
        """Use rg to find matching lines, map to node_ids via line ranges."""
        from .ripgrep import rg_search
        hits = rg_search(
            query, [source_path],
            case_sensitive=self.case_sensitive,
            use_regex=self.use_regex,
        )
        matched_lines = hits.get(source_path, [])
        if not matched_lines:
            return {}
        return self._lines_to_nodes(doc, matched_lines)

    @staticmethod
    def _lines_to_nodes(doc: Document, lines: list[int]) -> dict[str, float]:
        """Map matched line numbers to node_ids with hit-count-based scores."""
        from bisect import bisect_left, bisect_right
        from .tree import flatten_tree

        nodes = flatten_tree(doc.structure)
        sorted_lines = sorted(lines)
        results: dict[str, float] = {}
        for node in nodes:
            nid = node.get("node_id", "")
            ls = node.get("line_start")
            le = node.get("line_end")
            if ls is None or le is None or not nid:
                continue
            # O(log N) range count via bisect on sorted lines
            lo = bisect_left(sorted_lines, ls)
            hi = bisect_right(sorted_lines, le)
            hit_count = hi - lo
            if hit_count > 0:
                results[nid] = float(hit_count)
        # Normalize so max score = 1.0
        if results:
            max_s = max(results.values())
            if max_s > 0:
                results = {k: v / max_s for k, v in results.items()}
        return results

    def _native_score(self, query: str, doc: Document) -> dict[str, float]:
        """Pure-Python matching over in-memory tree nodes."""
        results: dict[str, float] = {}
        pattern = query if self.case_sensitive else query.lower()

        regex = None
        if self.use_regex:
            try:
                regex = re.compile(query, 0 if self.case_sensitive else re.IGNORECASE)
            except re.error:
                logger.warning("Invalid regex in GrepFilter: %s", query)

        def _scan(node):
            nid = node.get("node_id", "")
            title = node.get("title", "")
            summary = node.get("summary", node.get("prefix_summary", ""))
            text = node.get("text", "")

            # Match against title, summary, and text
            matched = False
            if regex:
                if regex.search(title) or regex.search(summary) or regex.search(text):
                    matched = True
            else:
                t_title = title if self.case_sensitive else title.lower()
                t_summary = summary if self.case_sensitive else summary.lower()
                t_text = text if self.case_sensitive else text.lower()
                if (t_title and pattern in t_title) or (t_summary and pattern in t_summary) or (t_text and pattern in t_text):
                    matched = True

            if matched:
                results[nid] = 1.0

            for child in node.get("nodes", []):
                _scan(child)

        structure = doc.structure
        if isinstance(structure, list):
            for item in structure:
                _scan(item)
        else:
            _scan(structure)

        return results


# ---------------------------------------------------------------------------
# Auto mode resolution
# ---------------------------------------------------------------------------

# Explicit mapping: source_type → whether tree walk provides meaningful benefit.
# True = has natural hierarchy (headings, nesting) that tree walk can exploit.
# False = flat or shallow structure where FTS5 alone is equally effective.
_TREE_BENEFIT: dict[str, bool] = {
    "markdown": True,   # heading hierarchy → tree walk excels
    "json": True,       # nested key/value structure → tree walk excels
    "code": False,      # flat function/class list → FTS5 keyword match suffices
    "text": False,      # no structure → FTS5 only
    "csv": False,       # tabular, no hierarchy
    "pdf": False,       # flat extracted text
    "doc": False,       # flat extracted text
    "docx": False,      # flat extracted text
    "excel": False,     # tabular, no hierarchy
    "html": False,      # parsed flat text (headings stripped by parser)
    "xml": False,       # parsed flat text
    "jsonl": False,     # line-oriented, no nesting across lines
}

# Minimum ratio of tree-benefiting docs (by count) to trigger tree mode.
# 0.3 means: if ≥30% of docs benefit from tree, use tree for all.
# Rationale: tree mode is a superset of flat — it still returns FTS5 scores
# for non-hierarchical docs, but adds path-based retrieval for the rest.
_TREE_RATIO_THRESHOLD = 0.3

# Minimum tree depth for a doc to truly benefit from tree walk.
# Docs with depth ≤ 1 (flat list of nodes) won't gain anything from BFS walk.
_MIN_TREE_DEPTH = 2


def _has_meaningful_depth(doc: Document, min_depth: int = _MIN_TREE_DEPTH) -> bool:
    """Check if a document's tree has enough depth for tree walk to help."""
    def _max_depth(nodes, current: int) -> int:
        if not nodes:
            return current
        return max(
            _max_depth(node.get("nodes", []), current + 1)
            for node in nodes
        )
    structure = doc.structure
    if isinstance(structure, list):
        if not structure:
            return False
        depth = max(_max_depth(node.get("nodes", []), 1) for node in structure)
    else:
        depth = _max_depth(structure.get("nodes", []), 1)
    return depth >= min_depth


def _resolve_auto_mode(selected: list[Document]) -> str:
    """Pick 'tree' or 'flat' based on document source types and actual structure.

    Strategy (ordered by priority):
    1. Count how many docs have a tree-benefiting source_type.
    2. For those that *claim* tree benefit, verify they actually have meaningful
       depth (≥ _MIN_TREE_DEPTH). A markdown file with no headings is effectively flat.
    3. If the ratio of truly-hierarchical docs ≥ _TREE_RATIO_THRESHOLD → tree mode.
       Otherwise → flat mode.

    This avoids the old "1 markdown among 50 code files → tree for everything" problem
    while still being generous enough to activate tree when it helps.
    """
    total = len(selected)
    tree_count = 0

    for doc in selected:
        st = doc.source_type or ""
        # Unknown source types default to flat (safe default)
        benefits_from_tree = _TREE_BENEFIT.get(st, False)
        if benefits_from_tree and _has_meaningful_depth(doc):
            tree_count += 1

    ratio = tree_count / total
    if ratio >= _TREE_RATIO_THRESHOLD:
        logger.debug(
            "Auto mode → tree: %d/%d docs (%.0f%%) have meaningful hierarchy",
            tree_count, total, ratio * 100,
        )
        return "tree"
    else:
        logger.debug(
            "Auto mode → flat: only %d/%d docs (%.0f%%) have hierarchy (threshold %.0f%%)",
            tree_count, total, ratio * 100, _TREE_RATIO_THRESHOLD * 100,
        )
        return "flat"


# ---------------------------------------------------------------------------
# Unified search API
# ---------------------------------------------------------------------------

def _get_ancestor_titles(doc: Document, node_id: str) -> list[str]:
    """Get ancestor node titles for context anchoring.

    Uses Document's cached parent map — O(depth) traversal with O(1) lookups,
    avoids redundant build_tree_maps() call.
    """
    titles = []
    pid = doc.get_parent_id(node_id)
    while pid:
        pnode = doc.get_node_by_id(pid)
        if pnode:
            titles.append(pnode.get("title", ""))
        pid = doc.get_parent_id(pid)
    titles.reverse()
    return titles


def _attach_node_fields(
    nodes: list[dict],
    doc: Document,
    text_mode: str = "full",
    include_ancestors: bool = False,
) -> None:
    """Attach full node fields to search result nodes."""
    for n in nodes:
        full = doc.get_node_by_id(str(n["node_id"]))
        if not full:
            continue

        if text_mode == "full":
            n["text"] = full.get("text", "")
        elif text_mode == "summary":
            n["text"] = full.get("summary", full.get("prefix_summary", ""))
        # text_mode == "none": no text attached

        n["summary"] = full.get("summary", full.get("prefix_summary", ""))
        n["line_start"] = full.get("line_start")
        n["line_end"] = full.get("line_end")

        if include_ancestors:
            n["ancestors"] = _get_ancestor_titles(doc, str(n["node_id"]))


def _merge_doc_results(
    doc_results: list[dict],
    merge_strategy: str = "interleave",
) -> list[dict]:
    """Apply merge strategy to multi-document results."""
    if merge_strategy == "per_doc":
        return [r for r in doc_results if r.get("nodes")]

    if merge_strategy == "global_score":
        # Flatten all nodes with doc info, sort globally by score
        all_nodes = []
        for r in doc_results:
            for node in r.get("nodes", []):
                node_copy = dict(node)
                node_copy["_doc_id"] = r.get("doc_id", "")
                node_copy["_doc_name"] = r.get("doc_name", "")
                all_nodes.append(node_copy)
        all_nodes.sort(key=lambda x: (-x.get("score", 0), x.get("node_id", "")))

        # Re-group by doc but preserve global order
        seen_docs = {}
        merged = []
        for node in all_nodes:
            did = node.pop("_doc_id", "")
            dname = node.pop("_doc_name", "")
            if did not in seen_docs:
                seen_docs[did] = {"doc_id": did, "doc_name": dname, "nodes": []}
                merged.append(seen_docs[did])
            seen_docs[did]["nodes"].append(node)
        return merged

    # Default: interleave (current behavior)
    return [r for r in doc_results if r.get("nodes")]


async def search(
    query: str,
    documents: list[Document],
    top_k_docs: Optional[int] = None,
    max_nodes_per_doc: Optional[int] = None,
    pre_filter: Optional[PreFilter] = None,
    text_mode: str = "full",
    include_ancestors: bool = False,
    merge_strategy: str = "interleave",
    search_mode: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Search across one or more documents using tree-structured retrieval.

    All parameters default to ``get_config()`` values when not explicitly set.

    Args:
        query: user query
        documents: list of Document objects (single or multiple)
        top_k_docs: max documents to search (routing stage)
        max_nodes_per_doc: max result nodes per document
        pre_filter: custom PreFilter instance for node pre-scoring
        text_mode: 'full' (default) | 'summary' | 'none' - controls text in results
        include_ancestors: attach ancestor titles for context anchoring
        merge_strategy: 'interleave' (default) | 'per_doc' | 'global_score'
        search_mode: 'auto' (default) | 'flat' | 'tree'
                     - auto: automatically selects tree or flat based on document source types
                       (tree for documents/markdown/pdf, flat for code)
                     - flat: FTS5-only ranking
                     - tree: Best-First Search over document trees (always uses tree walk)

    Returns:
        dict with 'documents' (list), 'query' (str), 'flat_nodes' (list),
        optionally 'paths' (list) when search_mode='tree'.
    """
    cfg = get_config()

    # Resolve defaults from config
    if top_k_docs is None:
        top_k_docs = cfg.top_k_docs
    if max_nodes_per_doc is None:
        max_nodes_per_doc = cfg.max_nodes_per_doc
    if search_mode is None:
        search_mode = cfg.search_mode

    # Stage 1: document routing (FTS5-based)
    if len(documents) <= 1:
        selected = documents
    else:
        from .fts import get_fts_index
        fts_index = get_fts_index(db_path=cfg.fts_db_path or None)
        doc_map = {doc.doc_id: doc for doc in documents}
        unindexed = fts_index.get_unindexed_doc_ids(list(doc_map.keys()))
        for doc_id in unindexed:
            fts_index.index_document(doc_map[doc_id])

        # Fast path for tree mode: use score_nodes_batch without doc filter
        # to get both routing AND scoring in a single FTS5 query, avoiding
        # the separate search_with_aggregation call (saves ~2.5ms/query).
        if (search_mode == "tree" or (search_mode == "auto" and documents)) and pre_filter is None:
            # Check if any doc needs grep (code source types)
            from .parsers import get_prefilters_for_source_type
            use_grep = any(
                "grep" in get_prefilters_for_source_type(doc.source_type or "text")
                for doc in documents
            )
            if not use_grep:
                weights = {
                    "title": cfg.fts_title_weight,
                    "summary": cfg.fts_summary_weight,
                    "body": cfg.fts_body_weight,
                    "code_blocks": cfg.fts_code_weight,
                    "front_matter": cfg.fts_front_matter_weight,
                }
                scorer_fts = get_fts_index(db_path=cfg.fts_db_path or None, weights=weights)
                # Batch score ALL docs in one SQL query (no doc_id filter)
                all_doc_ids = list(doc_map.keys())
                all_scores = scorer_fts.score_nodes_batch(query, doc_ids=all_doc_ids)

                # Derive routing from batch results: rank docs by max node score
                doc_max_scores = []
                for did, nscores in all_scores.items():
                    if nscores:
                        doc_max_scores.append((did, max(nscores.values())))
                doc_max_scores.sort(key=lambda x: -x[1])
                routed_ids = {did for did, _ in doc_max_scores[:top_k_docs]}

                if routed_ids:
                    selected = [d for d in documents if d.doc_id in routed_ids]
                else:
                    selected = documents[:top_k_docs]

                # Resolve mode
                effective_mode = search_mode
                if search_mode == "auto" and selected:
                    effective_mode = _resolve_auto_mode(selected)

                if effective_mode == "tree":
                    # Filter score map to selected docs only
                    fts_score_map = {did: all_scores[did] for did in routed_ids if did in all_scores}
                    result = await _search_tree_mode(
                        query, selected, scorer_fts, cfg,
                        max_nodes_per_doc=max_nodes_per_doc,
                        text_mode=text_mode,
                        include_ancestors=include_ancestors,
                        merge_strategy=merge_strategy,
                        fts_score_map=fts_score_map,
                    )
                    return result
                # else fall through to flat mode below

                # For flat mode, we still have the scorer ready
                scorer = scorer_fts

                logger.debug("Selected %d documents: %s", len(selected), [d.doc_name for d in selected])

                result = await _search_flat_mode(
                    query, selected, scorer, cfg,
                    max_nodes_per_doc=max_nodes_per_doc,
                    text_mode=text_mode,
                    include_ancestors=include_ancestors,
                    merge_strategy=merge_strategy,
                )
                return result

        # Standard routing path (for non-tree modes or grep-needing docs)
        agg = fts_index.search_with_aggregation(query, top_k=top_k_docs)
        if agg:
            relevant_ids = {a["doc_id"] for a in agg}
            selected = [d for d in documents if d.doc_id in relevant_ids]
            if not selected:
                selected = documents[:top_k_docs]
        else:
            selected = documents[:top_k_docs]

    logger.debug("Selected %d documents: %s", len(selected), [d.doc_name for d in selected])

    # Stage 1.5: Pre-filter scoring (build scorer for both modes)
    scorer = pre_filter
    if scorer is None and selected:
        from .parsers import get_prefilters_for_source_type
        use_grep = False
        for doc in selected:
            prefilters = get_prefilters_for_source_type(doc.source_type or "text")
            if "grep" in prefilters:
                use_grep = True
                break

        if use_grep:
            grep_filter = GrepFilter(selected)
            fts_scorer = _get_fts_scorer(selected, cfg)
            scorer = _CombinedScorer(grep_filter, fts_scorer) if fts_scorer else grep_filter
        else:
            scorer = _get_fts_scorer(selected, cfg)

    # Branch: resolve effective search mode
    # - "auto": picks tree vs flat based on proportion of tree-benefiting docs
    # - "tree": always uses tree walk (Best-First Walk)
    # - "flat": always uses FTS5-only
    effective_mode = search_mode
    if search_mode == "auto" and selected:
        effective_mode = _resolve_auto_mode(selected)

    if effective_mode == "tree" and scorer is not None:
        result = await _search_tree_mode(
            query, selected, scorer, cfg,
            max_nodes_per_doc=max_nodes_per_doc,
            text_mode=text_mode,
            include_ancestors=include_ancestors,
            merge_strategy=merge_strategy,
        )
        return result

    # Flat mode (original behavior, or auto-resolved from auto mode)
    result = await _search_flat_mode(
        query, selected, scorer, cfg,
        max_nodes_per_doc=max_nodes_per_doc,
        text_mode=text_mode,
        include_ancestors=include_ancestors,
        merge_strategy=merge_strategy,
    )
    return result


async def _search_tree_mode(
    query: str,
    selected: list[Document],
    scorer,
    cfg,
    max_nodes_per_doc: int = 5,
    text_mode: str = "full",
    include_ancestors: bool = False,
    merge_strategy: str = "interleave",
    fts_score_map: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Tree search mode: anchor retrieval -> tree walk -> path aggregation.

    Args:
        fts_score_map: pre-computed {doc_id: {node_id: score}}. If provided,
            skips scorer.score_nodes_batch() call (fast path when routing and
            scoring are merged). If None, scores are computed from scorer.
    """
    from .tree_searcher import TreeSearcher

    # Build FTS score maps — use pre-computed if available, else from scorer
    if fts_score_map is None:
        fts_score_map = {}
        doc_ids = [doc.doc_id for doc in selected]
        if hasattr(scorer, "score_nodes_batch"):
            fts_score_map = scorer.score_nodes_batch(query, doc_ids=doc_ids)
        else:
            for doc in selected:
                scores = scorer.score_nodes(query, doc.doc_id)
                if scores:
                    fts_score_map[doc.doc_id] = scores

    # Run tree search
    searcher = TreeSearcher()
    paths, tree_flat_nodes = searcher.search(query, selected, fts_score_map)

    # Build document-grouped results (compatible with old API)
    doc_results: list[dict] = []
    doc_nodes_map: dict[str, list[dict]] = {}
    for fn in tree_flat_nodes:
        did = fn["doc_id"]
        doc_nodes_map.setdefault(did, []).append(fn)

    for doc in selected:
        nodes = doc_nodes_map.get(doc.doc_id, [])[:max_nodes_per_doc]
        enriched_nodes = []
        for n in nodes:
            enriched_nodes.append({
                "node_id": n["node_id"],
                "title": n["title"],
                "score": n["score"],
            })
        _attach_node_fields(enriched_nodes, doc, text_mode=text_mode, include_ancestors=include_ancestors)
        if enriched_nodes:
            doc_results.append({
                "doc_id": doc.doc_id,
                "doc_name": doc.doc_name,
                "nodes": enriched_nodes,
            })

    merged = _merge_doc_results(doc_results, merge_strategy)

    # Build flat_nodes: collect all candidates first, then apply per-node char budget.
    # Strategy: sort by score globally, then truncate text per-node so high-rank nodes
    # from any document are never dropped in favor of lower-rank nodes that happen to
    # appear earlier in the doc list.
    max_result_chars = cfg.max_result_chars
    all_candidates = []
    for doc_result in merged:
        for node in doc_result.get("nodes", []):
            all_candidates.append({
                "node_id": node.get("node_id", ""),
                "doc_id": doc_result.get("doc_id", ""),
                "doc_name": doc_result.get("doc_name", ""),
                "title": node.get("title", ""),
                "score": node.get("score", 0),
                "text": node.get("text", ""),
            })
    all_candidates.sort(key=lambda x: (-x["score"], x["node_id"]))

    # Apply char budget: budget is distributed per-node (not first-come-first-served).
    # Each node gets at most (remaining_budget / remaining_nodes) chars so that all
    # high-rank nodes appear in the output, possibly truncated rather than dropped.
    flat_nodes = []
    if max_result_chars and all_candidates:
        remaining_budget = max_result_chars
        n = len(all_candidates)
        for i, node in enumerate(all_candidates):
            if remaining_budget <= 0:
                flat_nodes.append({**node, "text": ""})
                continue
            per_node_limit = max(remaining_budget // (n - i), 1)
            text = node["text"]
            if len(text) > per_node_limit:
                text = text[:per_node_limit]
            flat_nodes.append({**node, "text": text})
            remaining_budget -= len(text)
    else:
        flat_nodes = all_candidates

    # Serialize paths for output
    serialized_paths = []
    for pr in paths:
        serialized_paths.append({
            "doc_id": pr.doc_id,
            "doc_name": pr.doc_name,
            "score": pr.score,
            "anchor_node_id": pr.anchor_node_id,
            "target_node_id": pr.target_node_id,
            "path": pr.path,
            "reasons": pr.reasons,
            "snippet": pr.snippet,
        })

    return {
        "documents": merged,
        "query": query,
        "flat_nodes": flat_nodes,
        "paths": serialized_paths,
        "mode": "tree",
    }


async def _search_flat_mode(
    query: str,
    selected: list[Document],
    scorer,
    cfg,
    max_nodes_per_doc: int = 5,
    text_mode: str = "full",
    include_ancestors: bool = False,
    merge_strategy: str = "interleave",
) -> dict:
    """Flat search mode (original behavior): FTS5 scoring -> rank -> return."""
    async def _search_doc(doc: Document) -> dict:
        nodes = []
        if scorer is not None:
            score_map = scorer.score_nodes(query, doc.doc_id)
            for nid, score in sorted(score_map.items(), key=lambda x: -x[1]):
                full_node = doc.get_node_by_id(nid)
                nodes.append({
                    "node_id": nid,
                    "title": full_node.get("title", "") if full_node else "",
                    "score": round(score, 4),
                })
                if len(nodes) >= max_nodes_per_doc:
                    break

        _attach_node_fields(nodes, doc, text_mode=text_mode, include_ancestors=include_ancestors)
        return {"doc_id": doc.doc_id, "doc_name": doc.doc_name, "nodes": nodes}

    raw_results = await asyncio.gather(*(_search_doc(d) for d in selected))
    doc_results = list(raw_results)

    merged = _merge_doc_results(doc_results, merge_strategy)

    # Build flat_nodes: collect all candidates first, then apply per-node char budget.
    # See _search_tree_mode for strategy rationale.
    max_result_chars = cfg.max_result_chars
    all_candidates = []
    for doc_result in merged:
        for node in doc_result.get("nodes", []):
            all_candidates.append({
                "node_id": node.get("node_id", ""),
                "doc_id": doc_result.get("doc_id", ""),
                "doc_name": doc_result.get("doc_name", ""),
                "title": node.get("title", ""),
                "score": node.get("score", 0),
                "text": node.get("text", ""),
            })
    all_candidates.sort(key=lambda x: (-x["score"], x["node_id"]))

    flat_nodes = []
    if max_result_chars and all_candidates:
        remaining_budget = max_result_chars
        n = len(all_candidates)
        for i, node in enumerate(all_candidates):
            if remaining_budget <= 0:
                flat_nodes.append({**node, "text": ""})
                continue
            per_node_limit = max(remaining_budget // (n - i), 1)
            text = node["text"]
            if len(text) > per_node_limit:
                text = text[:per_node_limit]
            flat_nodes.append({**node, "text": text})
            remaining_budget -= len(text)
    else:
        flat_nodes = all_candidates

    return {
        "documents": merged,
        "query": query,
        "flat_nodes": flat_nodes,
        "mode": "flat",
    }


def _get_fts_scorer(documents: list[Document], cfg) -> Optional[PreFilter]:
    """Get FTS5 scorer, auto-indexing documents as needed."""
    from .fts import get_fts_index
    weights = {
        "title": cfg.fts_title_weight,
        "summary": cfg.fts_summary_weight,
        "body": cfg.fts_body_weight,
        "code_blocks": cfg.fts_code_weight,
        "front_matter": cfg.fts_front_matter_weight,
    }
    fts_index = get_fts_index(db_path=cfg.fts_db_path or None, weights=weights)
    # Batch check: only index documents not yet in the FTS5 index
    doc_map = {doc.doc_id: doc for doc in documents}
    unindexed = fts_index.get_unindexed_doc_ids(list(doc_map.keys()))
    for doc_id in unindexed:
        fts_index.index_document(doc_map[doc_id])
    return fts_index


class _CombinedScorer:
    """Combine multiple PreFilter scorers by summing normalized scores."""

    def __init__(self, *scorers):
        self._scorers = scorers

    def score_nodes(self, query: str, doc_id: str) -> dict[str, float]:
        combined: dict[str, float] = {}
        for scorer in self._scorers:
            if scorer is None:
                continue
            scores = scorer.score_nodes(query, doc_id)
            for nid, score in scores.items():
                combined[nid] = combined.get(nid, 0.0) + score
        return combined


def search_sync(query: str, documents: list[Document], **kwargs) -> dict:
    """Synchronous wrapper around :func:`search`."""
    return asyncio.run(search(query, documents, **kwargs))

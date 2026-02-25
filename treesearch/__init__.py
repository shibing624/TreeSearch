# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TreeSearch - Reasoning-based RAG with tree-structured document retrieval.

No vector embeddings. No chunk splitting. Pure LLM reasoning over document trees.
"""
__version__ = "0.2.3"
from treesearch.llm import achat, chat, count_tokens, extract_json
from treesearch.tree import (
    Document,
    INDEX_VERSION,
    assign_node_ids,
    flatten_tree,
    find_node,
    get_leaf_nodes,
    remove_fields,
    format_structure,
    save_index,
    load_index,
    print_toc,
    print_tree_json,
)
from treesearch.indexer import md_to_tree, text_to_tree, build_index
from treesearch.search import (
    PreFilter,
    SearchResult,
    BestFirstTreeSearch,
    BestFirstTreeSearch as TreeSearch,
    MCTSTreeSearch,
    llm_tree_search,
    search,
    search_sync,
    route_documents,
)
from treesearch.rank_bm25 import (
    BM25Okapi,
    NodeBM25Index,
    tokenize,
)
from treesearch.metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    reciprocal_rank,
    ndcg_at_k,
    f1_at_k,
    evaluate_query,
    evaluate_benchmark,
)

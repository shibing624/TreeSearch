# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.search module (with mocked LLM).

Tests cover: BestFirstTreeSearch, MCTSTreeSearch, llm_tree_search, route_documents,
             search(), search_sync(), SearchResult.
"""
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from treesearch.search import (
    BestFirstTreeSearch,
    MCTSTreeSearch,
    llm_tree_search,
    SearchResult,
    search,
    search_sync,
    route_documents,
)
from treesearch.tree import Document


class TestBestFirstTreeSearch:
    @pytest.mark.asyncio
    async def test_run_returns_results(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.8}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = BestFirstTreeSearch(
                document=doc,
                query="What is the backend technology?",
                max_results=5,
                use_subtree_cache=False,
            )
            results = await searcher.run()

        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "title" in r
            assert "score" in r
            assert "node_id" in r

    @pytest.mark.asyncio
    async def test_early_stopping(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.1}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = BestFirstTreeSearch(
                document=doc,
                query="Unrelated query",
                threshold=0.5,
                use_subtree_cache=False,
            )
            results = await searcher.run()

        # All scores below threshold -> no results
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_max_results_limit(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.9}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = BestFirstTreeSearch(
                document=doc,
                query="test",
                max_results=2,
                threshold=0.1,
                use_subtree_cache=False,
            )
            results = await searcher.run()

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_budget_control(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.8}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = BestFirstTreeSearch(
                document=doc,
                query="test",
                max_llm_calls=5,
                threshold=0.1,
                use_subtree_cache=False,
            )
            await searcher.run()

        assert searcher.llm_calls <= 5

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self, sample_tree_structure):
        call_count = 0

        async def mock_achat(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            scores = [0.9, 0.3, 0.7, 0.5]
            return f'{{"relevance": {scores[call_count % len(scores)]}}}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = BestFirstTreeSearch(
                document=doc,
                query="test",
                threshold=0.1,
                use_subtree_cache=False,
            )
            results = await searcher.run()

        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_with_bm25_scores(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.7}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        bm25_scores = {"0001": 2.5, "0002": 0.5, "0003": 1.8}

        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = BestFirstTreeSearch(
                document=doc,
                query="backend",
                bm25_scores=bm25_scores,
                bm25_weight=0.3,
                threshold=0.1,
                use_subtree_cache=False,
            )
            results = await searcher.run()

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_subtree_cache(self, sample_tree_structure):
        """Subtree cache should avoid redundant LLM calls across searches."""
        BestFirstTreeSearch.clear_subtree_cache()
        call_count = 0

        async def mock_achat(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return '{"relevance": 0.8}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)

        # First search
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher1 = BestFirstTreeSearch(
                document=doc, query="backend tech", use_subtree_cache=True
            )
            await searcher1.run()

        first_calls = call_count

        # Second search with same query -> should hit cache
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher2 = BestFirstTreeSearch(
                document=doc, query="backend tech", use_subtree_cache=True
            )
            await searcher2.run()

        # Second search should make fewer LLM calls due to cache
        assert call_count - first_calls < first_calls

        BestFirstTreeSearch.clear_subtree_cache()


class TestMCTSTreeSearch:
    @pytest.mark.asyncio
    async def test_run_returns_results(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.8}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = MCTSTreeSearch(
                document=doc,
                query="What is the backend technology?",
                max_iterations=3,
            )
            results = await searcher.run()

        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert "title" in r
            assert "score" in r
            assert "node_id" in r
            assert "visits" in r

    @pytest.mark.asyncio
    async def test_run_with_high_threshold(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.2}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = MCTSTreeSearch(
                document=doc,
                query="Unrelated query",
                max_iterations=2,
                value_threshold=0.9,
            )
            results = await searcher.run()

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self, sample_tree_structure):
        call_count = 0

        async def mock_achat(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            scores = [0.9, 0.3, 0.7, 0.5, 0.8, 0.6, 0.4, 0.2]
            return f'{{"relevance": {scores[call_count % len(scores)]}}}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = MCTSTreeSearch(
                document=doc,
                query="test",
                max_iterations=3,
                value_threshold=0.1,
            )
            results = await searcher.run()

        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_max_selected_nodes_limit(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.9}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = MCTSTreeSearch(
                document=doc,
                query="test",
                max_iterations=5,
                max_selected_nodes=2,
                value_threshold=0.1,
            )
            results = await searcher.run()

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_value_cache_reduces_calls(self, sample_tree_structure):
        call_count = 0

        async def mock_achat(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            return '{"relevance": 0.7}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = MCTSTreeSearch(
                document=doc,
                query="test",
                max_iterations=5,
                value_threshold=0.1,
            )
            await searcher.run()

        num_nodes = len(searcher.nodes)
        assert call_count <= num_nodes * 2

    @pytest.mark.asyncio
    async def test_llm_calls_tracked(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.7}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            searcher = MCTSTreeSearch(
                document=doc, query="test", max_iterations=2,
            )
            await searcher.run()

        assert searcher.llm_calls > 0


class TestLlmTreeSearch:
    @pytest.mark.asyncio
    async def test_llm_search_returns_nodes(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"node_list": ["0001", "0003"]}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            results = await llm_tree_search(
                query="Backend and deployment",
                document=doc,
            )

        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert "node_id" in r
            assert "title" in r

    @pytest.mark.asyncio
    async def test_llm_search_with_expert_knowledge(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            assert "Expert knowledge" in prompt
            return '{"node_list": ["0001"]}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            results = await llm_tree_search(
                query="Backend tech",
                document=doc,
                expert_knowledge="The backend uses FastAPI",
            )

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_llm_search_nonexistent_node(self, sample_tree_structure):
        async def mock_achat(prompt, **kwargs):
            return '{"node_list": ["9999"]}'

        doc = Document(doc_id="test", doc_name="Test Doc", structure=sample_tree_structure)
        with patch("treesearch.search.achat", side_effect=mock_achat):
            results = await llm_tree_search(
                query="test",
                document=doc,
            )

        assert len(results) == 0


class TestSearchResult:
    def test_fields(self):
        r = SearchResult(
            documents=[{"doc_id": "d1", "doc_name": "Doc", "nodes": []}],
            query="test",
            total_llm_calls=5,
            strategy="best_first",
        )
        assert len(r.documents) == 1
        assert r.query == "test"
        assert r.total_llm_calls == 5
        assert r.strategy == "best_first"

    def test_empty(self):
        r = SearchResult()
        assert r.documents == []
        assert r.query == ""
        assert r.total_llm_calls == 0


class TestRouteDocuments:
    @pytest.fixture
    def two_documents(self, sample_tree_structure):
        doc_a = Document(
            doc_id="a", doc_name="Doc A",
            doc_description="Architecture and deployment guide.",
            structure=sample_tree_structure,
        )
        doc_b = Document(
            doc_id="b", doc_name="Doc B",
            doc_description="Machine learning algorithms overview.",
            structure=[{
                "title": "ML Basics", "summary": "Introduction to machine learning.",
                "node_id": "0000", "text": "Machine learning is a subset of AI.",
            }],
        )
        return [doc_a, doc_b]

    @pytest.mark.asyncio
    async def test_routes_to_relevant_doc(self, two_documents):
        async def mock_achat(prompt, **kwargs):
            return '{"selected_doc_ids": ["a"]}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            selected = await route_documents(
                query="How is the system deployed?",
                documents=two_documents,
            )

        assert isinstance(selected, list)
        assert len(selected) > 0
        assert selected[0].doc_id == "a"

    @pytest.mark.asyncio
    async def test_fallback_when_no_match(self, two_documents):
        async def mock_achat(prompt, **kwargs):
            return '{"selected_doc_ids": []}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            selected = await route_documents(
                query="Unrelated query",
                documents=two_documents,
            )

        assert len(selected) > 0

    @pytest.mark.asyncio
    async def test_top_k_limit(self, two_documents):
        async def mock_achat(prompt, **kwargs):
            return '{"selected_doc_ids": ["a", "b"]}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            selected = await route_documents(
                query="test",
                documents=two_documents,
                top_k=1,
            )

        assert len(selected) <= 1


class TestSearch:
    @pytest.fixture
    def two_documents(self, sample_tree_structure):
        doc_a = Document(
            doc_id="a", doc_name="Doc A",
            doc_description="Architecture and deployment guide.",
            structure=sample_tree_structure,
        )
        doc_b = Document(
            doc_id="b", doc_name="Doc B",
            doc_description="Machine learning algorithms overview.",
            structure=[{
                "title": "ML Basics", "summary": "Introduction to machine learning.",
                "node_id": "0000", "text": "Machine learning is a subset of AI.",
            }],
        )
        return [doc_a, doc_b]

    @pytest.mark.asyncio
    async def test_end_to_end_best_first(self, two_documents):
        async def mock_achat(prompt, **kwargs):
            if "select" in prompt.lower() or "document" in prompt.lower():
                return '{"selected_doc_ids": ["a"]}'
            if "relevance" in prompt.lower() or "rate" in prompt.lower():
                return '{"relevance": 0.8}'
            return '{"node_list": ["0001"]}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            result = await search(
                query="What is the backend?",
                documents=two_documents,
                strategy="best_first",
                top_k_docs=3,
            )

        assert isinstance(result, SearchResult)
        assert result.strategy == "best_first"
        assert result.query == "What is the backend?"
        assert result.total_llm_calls > 0

    @pytest.mark.asyncio
    async def test_end_to_end_mcts(self, two_documents):
        async def mock_achat(prompt, **kwargs):
            if "select" in prompt.lower() or "document" in prompt.lower():
                return '{"selected_doc_ids": ["a"]}'
            if "relevance" in prompt.lower() or "rate" in prompt.lower():
                return '{"relevance": 0.8}'
            return '{"node_list": ["0001"]}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            result = await search(
                query="What is the backend?",
                documents=two_documents,
                strategy="mcts",
                mcts_iterations=2,
                top_k_docs=3,
            )

        assert isinstance(result, SearchResult)
        assert result.strategy == "mcts"

    @pytest.mark.asyncio
    async def test_end_to_end_llm_search(self, two_documents):
        async def mock_achat(prompt, **kwargs):
            if "select" in prompt.lower() or "document" in prompt.lower():
                return '{"selected_doc_ids": ["a"]}'
            return '{"node_list": ["0001", "0003"]}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            result = await search(
                query="Backend and deployment",
                documents=two_documents,
                strategy="llm",
                top_k_docs=3,
            )

        assert isinstance(result, SearchResult)
        assert result.strategy == "llm"

    @pytest.mark.asyncio
    async def test_single_document_skips_routing(self, sample_tree_structure):
        doc = Document(
            doc_id="only", doc_name="Only Doc",
            doc_description="The only doc.",
            structure=sample_tree_structure,
        )

        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.7}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            result = await search(
                query="test",
                documents=[doc],
                strategy="best_first",
            )

        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_with_bm25_disabled(self, two_documents):
        async def mock_achat(prompt, **kwargs):
            if "select" in prompt.lower() or "document" in prompt.lower():
                return '{"selected_doc_ids": ["a"]}'
            return '{"relevance": 0.8}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            result = await search(
                query="backend",
                documents=two_documents,
                strategy="best_first",
                use_bm25=False,
                top_k_docs=3,
            )

        assert isinstance(result, SearchResult)


class TestSearchSync:
    def test_sync_wrapper(self, sample_tree_structure):
        doc_a = Document(
            doc_id="a", doc_name="Doc A",
            doc_description="Architecture and deployment guide.",
            structure=sample_tree_structure,
        )

        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.8}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            result = search_sync(
                query="test",
                documents=[doc_a],
                strategy="best_first",
            )

        assert isinstance(result, SearchResult)

    def test_sync_mcts(self, sample_tree_structure):
        doc_a = Document(
            doc_id="a", doc_name="Doc A",
            doc_description="Architecture guide.",
            structure=sample_tree_structure,
        )

        async def mock_achat(prompt, **kwargs):
            return '{"relevance": 0.8}'

        with patch("treesearch.search.achat", side_effect=mock_achat):
            result = search_sync(
                query="test",
                documents=[doc_a],
                strategy="mcts",
                mcts_iterations=1,
            )

        assert isinstance(result, SearchResult)

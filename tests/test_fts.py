# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for SQLite FTS5 full-text search engine (treesearch/fts.py).
"""
import os
import tempfile

import pytest

from treesearch.fts import (
    FTS5Index,
    parse_md_node_text,
    get_fts_index,
    set_fts_index,
    reset_fts_index,
)
from treesearch.tree import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document():
    """Create a sample Document for testing."""
    structure = [
        {
            "title": "Introduction",
            "node_id": "0",
            "summary": "Overview of the project",
            "text": "This is the introduction to our machine learning project.",
            "line_start": 1,
            "line_end": 10,
            "nodes": [
                {
                    "title": "Background",
                    "node_id": "1",
                    "summary": "Historical context and motivation",
                    "text": "Deep learning has revolutionized natural language processing.",
                    "line_start": 11,
                    "line_end": 20,
                    "nodes": [],
                },
                {
                    "title": "Objectives",
                    "node_id": "2",
                    "summary": "Goals of this research",
                    "text": "Our objective is to build a fast retrieval system using BM25 and tree search.",
                    "line_start": 21,
                    "line_end": 30,
                    "nodes": [],
                },
            ],
        },
        {
            "title": "Methods",
            "node_id": "3",
            "prefix_summary": "Research methodology",
            "text": "We use SQLite FTS5 for full-text indexing with inverted index.",
            "line_start": 31,
            "line_end": 50,
            "nodes": [
                {
                    "title": "Data Collection",
                    "node_id": "4",
                    "summary": "How data was gathered",
                    "text": "Data was collected from academic papers and web documents.",
                    "line_start": 51,
                    "line_end": 60,
                    "nodes": [],
                },
            ],
        },
    ]
    return Document(
        doc_id="test_doc",
        doc_name="Test Document",
        structure=structure,
        doc_description="A test document for FTS5 testing",
    )


@pytest.fixture
def chinese_document():
    """Create a Chinese text Document for testing CJK support."""
    structure = [
        {
            "title": "项目简介",
            "node_id": "0",
            "summary": "关于本项目的概述",
            "text": "本项目旨在构建一个高效的语义检索系统，支持中文和英文混合搜索。",
            "line_start": 1,
            "line_end": 10,
            "nodes": [],
        },
        {
            "title": "技术方案",
            "node_id": "1",
            "summary": "采用的核心技术",
            "text": "核心技术包括：大模型驱动的查询生成、SQLite FTS5全文检索、BM25排序算法。",
            "line_start": 11,
            "line_end": 20,
            "nodes": [],
        },
    ]
    return Document(
        doc_id="cn_doc",
        doc_name="Chinese Doc",
        structure=structure,
    )


@pytest.fixture
def fts_index():
    """Create an in-memory FTS5 index."""
    idx = FTS5Index(db_path=None)
    yield idx
    idx.close()


@pytest.fixture
def persistent_fts_index(tmp_path):
    """Create a persistent FTS5 index in a temp directory."""
    db_path = str(tmp_path / "test_index.db")
    idx = FTS5Index(db_path=db_path)
    yield idx, db_path
    idx.close()


# ---------------------------------------------------------------------------
# parse_md_node_text
# ---------------------------------------------------------------------------

class TestParseMdNodeText:
    def test_empty(self):
        result = parse_md_node_text("")
        assert result["front_matter"] == ""
        assert result["body"] == ""
        assert result["code_blocks"] == ""

    def test_plain_text(self):
        text = "This is a simple paragraph.\nAnother line here."
        result = parse_md_node_text(text)
        assert "simple paragraph" in result["body"]
        assert result["code_blocks"] == ""
        assert result["front_matter"] == ""

    def test_front_matter(self):
        text = "---\ntitle: Test\nauthor: XuMing\n---\nMain content here."
        result = parse_md_node_text(text)
        assert "title: Test" in result["front_matter"]
        assert "Main content" in result["body"]

    def test_code_blocks(self):
        text = "Some text\n```python\ndef hello():\n    print('hi')\n```\nMore text"
        result = parse_md_node_text(text)
        assert "def hello" in result["code_blocks"]
        assert "def hello" not in result["body"]
        assert "Some text" in result["body"]
        assert "More text" in result["body"]

    def test_multiple_code_blocks(self):
        text = "Text\n```python\ncode1\n```\nMiddle\n```js\ncode2\n```\nEnd"
        result = parse_md_node_text(text)
        assert "code1" in result["code_blocks"]
        assert "code2" in result["code_blocks"]
        assert "Middle" in result["body"]


# ---------------------------------------------------------------------------
# FTS5Index: indexing
# ---------------------------------------------------------------------------

class TestFTS5Indexing:
    def test_index_document(self, fts_index, sample_document):
        count = fts_index.index_document(sample_document)
        assert count == 5

    def test_index_incremental(self, fts_index, sample_document):
        """Second indexing with same content should be skipped."""
        count1 = fts_index.index_document(sample_document)
        assert count1 == 5
        count2 = fts_index.index_document(sample_document)
        assert count2 == 0  # skipped due to hash match

    def test_index_force(self, fts_index, sample_document):
        """Force re-index even if hash matches."""
        fts_index.index_document(sample_document)
        count = fts_index.index_document(sample_document, force=True)
        assert count == 5

    def test_index_multiple(self, fts_index, sample_document, chinese_document):
        total = fts_index.index_documents([sample_document, chinese_document])
        assert total == 7  # 5 + 2

    def test_is_document_indexed(self, fts_index, sample_document):
        assert not fts_index.is_document_indexed("test_doc")
        fts_index.index_document(sample_document)
        assert fts_index.is_document_indexed("test_doc")

    def test_stats(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        stats = fts_index.get_stats()
        assert stats["document_count"] == 1
        assert stats["node_count"] == 5


# ---------------------------------------------------------------------------
# FTS5Index: search
# ---------------------------------------------------------------------------

class TestFTS5Search:
    def test_basic_search(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        results = fts_index.search("machine learning")
        assert len(results) > 0
        assert any(r["node_id"] == "0" for r in results)

    def test_search_with_doc_filter(self, fts_index, sample_document, chinese_document):
        fts_index.index_documents([sample_document, chinese_document])
        results = fts_index.search("machine learning", doc_id="test_doc")
        for r in results:
            assert r["doc_id"] == "test_doc"

    def test_search_no_results(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        results = fts_index.search("xyznonexistent12345")
        assert len(results) == 0

    def test_search_empty_query(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        results = fts_index.search("")
        assert len(results) == 0

    def test_search_top_k(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        results = fts_index.search("learning", top_k=2)
        assert len(results) <= 2

    def test_search_sqlite_fts5(self, fts_index, sample_document):
        """Test that SQLite FTS5 indexes content from body field."""
        fts_index.index_document(sample_document)
        results = fts_index.search("retrieval system")
        assert len(results) > 0
        # Node 0002 mentions "retrieval system" in its text
        node_ids = [r["node_id"] for r in results]
        assert "2" in node_ids

    def test_search_fts_expression(self, fts_index, sample_document):
        """Test raw FTS5 expression search."""
        fts_index.index_document(sample_document)
        results = fts_index.search("", fts_expression="machine AND learning")
        assert len(results) > 0

    def test_search_or_expression(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        results = fts_index.search("", fts_expression="sqlite OR machine")
        assert len(results) >= 2

    def test_score_nodes_prefilter(self, fts_index, sample_document):
        """Test PreFilter protocol implementation."""
        fts_index.index_document(sample_document)
        scores = fts_index.score_nodes("machine learning", "test_doc")
        assert isinstance(scores, dict)
        assert len(scores) > 0
        # Scores should be normalized to [0, 1]
        for nid, score in scores.items():
            assert 0 <= score <= 1.0

    def test_search_chinese(self, fts_index, chinese_document):
        fts_index.index_document(chinese_document)
        results = fts_index.search("语义检索")
        assert len(results) > 0

    def test_aggregation(self, fts_index, sample_document, chinese_document):
        fts_index.index_documents([sample_document, chinese_document])
        results = fts_index.search_with_aggregation("learning", group_by_doc=True)
        assert len(results) >= 1
        for r in results:
            assert "hit_count" in r
            assert "best_score" in r


# ---------------------------------------------------------------------------
# FTS5 expression builder
# ---------------------------------------------------------------------------

class TestFTS5ExpressionBuilder:
    def test_or_expression(self):
        expr = FTS5Index.build_fts_expression(["python", "async"], "OR")
        assert "OR" in expr

    def test_and_expression(self):
        expr = FTS5Index.build_fts_expression(["python", "async"], "AND")
        assert "AND" in expr

    def test_not_expression(self):
        expr = FTS5Index.build_fts_expression(["python", "java"], "NOT")
        assert "NOT" in expr

    def test_near_expression(self):
        expr = FTS5Index.build_fts_expression(["deep", "learning"], near_distance=5)
        assert "NEAR" in expr
        assert "5" in expr

    def test_column_filter(self):
        expr = FTS5Index.build_fts_expression(["machine"], column="title")
        assert "title" in expr

    def test_empty_keywords(self):
        expr = FTS5Index.build_fts_expression([])
        assert expr == ""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistent_index(self, persistent_fts_index, sample_document):
        idx, db_path = persistent_fts_index
        idx.index_document(sample_document)
        results1 = idx.search("machine learning")
        idx.close()

        # Reopen from same path
        idx2 = FTS5Index(db_path=db_path)
        results2 = idx2.search("machine learning")
        idx2.close()

        assert len(results1) == len(results2)
        assert results1[0]["node_id"] == results2[0]["node_id"]

    def test_optimize(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        fts_index.optimize()  # should not raise

    def test_clear(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        assert fts_index.get_stats()["node_count"] == 5
        fts_index.clear()
        assert fts_index.get_stats()["node_count"] == 0

    def test_save_and_load_document(self, persistent_fts_index, sample_document):
        """Test document persistence: save + load round-trip."""
        idx, db_path = persistent_fts_index
        idx.save_document(sample_document)
        idx.close()

        idx2 = FTS5Index(db_path=db_path)
        loaded = idx2.load_document("test_doc")
        idx2.close()

        assert loaded is not None
        assert loaded.doc_id == "test_doc"
        assert loaded.doc_name == "Test Document"
        assert len(loaded.structure) == 2
        assert loaded.doc_description == "A test document for FTS5 testing"

    def test_load_all_documents(self, persistent_fts_index, sample_document, chinese_document):
        idx, db_path = persistent_fts_index
        idx.save_document(sample_document)
        idx.save_document(chinese_document)
        idx.close()

        idx2 = FTS5Index(db_path=db_path)
        docs = idx2.load_all_documents()
        idx2.close()

        assert len(docs) == 2
        doc_ids = {d.doc_id for d in docs}
        assert "test_doc" in doc_ids
        assert "cn_doc" in doc_ids

    def test_remove_document(self, fts_index, sample_document):
        fts_index.index_document(sample_document)
        fts_index.save_document(sample_document)
        assert fts_index.is_document_indexed("test_doc")
        fts_index.remove_document("test_doc")
        assert not fts_index.is_document_indexed("test_doc")
        assert fts_index.load_document("test_doc") is None

    def test_index_meta(self, persistent_fts_index):
        idx, db_path = persistent_fts_index
        assert idx.get_index_meta("/tmp/test.md") is None
        idx.set_index_meta("/tmp/test.md", "abc123")
        assert idx.get_index_meta("/tmp/test.md") == "abc123"
        meta = idx.get_all_index_meta()
        assert meta["/tmp/test.md"] == "abc123"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

class TestGlobalSingleton:
    def test_get_set_reset(self):
        reset_fts_index()
        idx = FTS5Index(db_path=None)
        set_fts_index(idx)
        assert get_fts_index() is idx
        reset_fts_index()

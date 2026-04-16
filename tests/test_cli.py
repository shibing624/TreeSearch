# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.cli module.
"""
import os
import sys
import tempfile

import pytest
from treesearch.cli import (
    _build_default_parser,
    _build_index_parser,
    _build_search_parser,
    _detect_subcommand,
    _load_documents_from_dir,
)


class TestDetectSubcommand:
    def test_index(self):
        assert _detect_subcommand(["index", "--paths", "src/"]) == "index"

    def test_search(self):
        assert _detect_subcommand(["search", "--query", "hello"]) == "search"

    def test_default_mode(self):
        assert _detect_subcommand(["How does auth work?", "src/"]) is None

    def test_verbose_then_index(self):
        assert _detect_subcommand(["-v", "index", "--paths", "x.md"]) == "index"

    def test_empty(self):
        assert _detect_subcommand([]) is None

    def test_query_named_index(self):
        # "index" as first positional → detected as subcommand
        assert _detect_subcommand(["index"]) == "index"


class TestDefaultParser:
    def test_query_and_paths(self):
        p = _build_default_parser()
        args = p.parse_args(["How does auth work?", "src/", "docs/"])
        assert args.query == "How does auth work?"
        assert args.paths == ["src/", "docs/"]

    def test_query_only(self):
        p = _build_default_parser()
        args = p.parse_args(["some query"])
        assert args.query == "some query"
        assert args.paths == []

    def test_with_db(self):
        p = _build_default_parser()
        args = p.parse_args(["query", "dir/", "--db", "./my.db"])
        assert args.db == "./my.db"

    def test_with_max_nodes(self):
        p = _build_default_parser()
        args = p.parse_args(["query", "dir/", "--max-nodes", "10"])
        assert args.max_nodes == 10

    def test_no_args(self):
        p = _build_default_parser()
        args = p.parse_args([])
        assert args.query is None
        assert args.paths == []

    def test_verbose(self):
        p = _build_default_parser()
        args = p.parse_args(["-v", "my query", "src/"])
        assert args.verbose is True
        assert args.query == "my query"

    def test_with_regex_flag(self):
        p = _build_default_parser()
        args = p.parse_args(["--regex", "auth.*", "src/"])
        assert args.regex is True
        assert args.query == "auth.*"

    def test_with_fts_expression(self):
        p = _build_default_parser()
        args = p.parse_args(["--fts-expression", "auth*", "src/"])
        assert args.fts_expression == "auth*"
        assert args.query is None

    def test_with_fts_expression_and_multiple_paths(self):
        p = _build_default_parser()
        args = p.parse_args(["--fts-expression", "auth*", "src/", "docs/"])
        assert args.fts_expression == "auth*"
        assert args.query is None
        assert args.paths == ["src/", "docs/"]


class TestIndexParser:
    def test_basic(self):
        p = _build_index_parser()
        args = p.parse_args(["--paths", "test.md"])
        assert args.paths == ["test.md"]

    def test_multiple_paths(self):
        p = _build_index_parser()
        args = p.parse_args(["--paths", "docs/*.md", "paper.txt"])
        assert args.paths == ["docs/*.md", "paper.txt"]

    def test_output_dir(self):
        p = _build_index_parser()
        args = p.parse_args(["--paths", "test.md", "-o", "./out"])
        assert args.output_dir == "./out"

    def test_force(self):
        p = _build_index_parser()
        args = p.parse_args(["--paths", "test.md", "--force"])
        assert args.force is True

    def test_max_concurrency(self):
        p = _build_index_parser()
        args = p.parse_args(["--paths", "test.md", "--max-concurrency", "10"])
        assert args.max_concurrency == 10

    def test_verbose(self):
        p = _build_index_parser()
        args = p.parse_args(["-v", "--paths", "x.md"])
        assert args.verbose is True


class TestSearchParser:
    def test_basic(self):
        p = _build_search_parser()
        args = p.parse_args(["--index_dir", "./idx", "--query", "hello"])
        assert args.index_dir == "./idx"
        assert args.query == "hello"

    def test_defaults(self):
        p = _build_search_parser()
        args = p.parse_args(["--query", "test"])
        assert args.index_dir == "./indexes"
        assert args.top_k_docs == 3
        assert args.max_nodes == 5

    def test_regex_flag(self):
        p = _build_search_parser()
        args = p.parse_args(["--query", "auth.*", "--regex"])
        assert args.query == "auth.*"
        assert args.regex is True

    def test_fts_expression(self):
        p = _build_search_parser()
        args = p.parse_args(["--fts-expression", "auth*"])
        assert args.fts_expression == "auth*"


class TestLoadDocuments:
    def test_loads_from_db(self):
        from treesearch.fts import FTS5Index
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "index.db")
            fts = FTS5Index(db_path=db_path)
            for name in ["doc_a", "doc_b"]:
                from treesearch.tree import Document
                doc = Document(
                    doc_id=name,
                    doc_name=name,
                    structure=[{"title": f"{name} root", "node_id": "0"}],
                    doc_description=f"Description of {name}",
                )
                fts.save_document(doc)
            fts.close()

            docs = _load_documents_from_dir(tmpdir)
            assert len(docs) == 2
            names = {d.doc_name for d in docs}
            assert "doc_a" in names
            assert "doc_b" in names

    def test_empty_db_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "index.db")
            from treesearch.fts import FTS5Index
            fts = FTS5Index(db_path=db_path)
            fts.close()
            with pytest.raises(SystemExit):
                _load_documents_from_dir(tmpdir)

    def test_missing_db_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(SystemExit):
                _load_documents_from_dir(tmpdir)

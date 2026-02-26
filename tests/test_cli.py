# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.cli module.
"""
import os
import sys
import json
import tempfile
from unittest.mock import patch, AsyncMock

import pytest
from treesearch.cli import _build_parser, _load_documents_from_dir


class TestBuildParser:
    def test_index_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["index", "--paths", "test.md"])
        assert args.command == "index"
        assert args.paths == ["test.md"]

    def test_index_with_multiple_paths(self):
        parser = _build_parser()
        args = parser.parse_args(["index", "--paths", "docs/*.md", "paper.txt"])
        assert args.command == "index"
        assert args.paths == ["docs/*.md", "paper.txt"]

    def test_index_with_output_dir(self):
        parser = _build_parser()
        args = parser.parse_args(["index", "--paths", "test.md", "-o", "./out"])
        assert args.command == "index"
        assert args.output_dir == "./out"

    def test_index_with_api_key(self):
        parser = _build_parser()
        args = parser.parse_args(["index", "--paths", "test.md", "--api-key", "sk-test"])
        assert args.api_key == "sk-test"

    def test_index_with_force(self):
        parser = _build_parser()
        args = parser.parse_args(["index", "--paths", "test.md", "--force"])
        assert args.force is True

    def test_index_with_max_concurrency(self):
        parser = _build_parser()
        args = parser.parse_args(["index", "--paths", "test.md", "--max-concurrency", "10"])
        assert args.max_concurrency == 10

    def test_search_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "--index_dir", "./idx", "--query", "hello"])
        assert args.command == "search"
        assert args.index_dir == "./idx"
        assert args.query == "hello"

    def test_search_strategy(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "--index_dir", ".", "--query", "q", "--strategy", "mcts"])
        assert args.strategy == "mcts"

    def test_search_strategy_default(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "--index_dir", ".", "--query", "q"])
        assert args.strategy == "best_first"

    def test_search_no_bm25(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "--index_dir", ".", "--query", "q", "--no-bm25"])
        assert args.no_bm25 is True

    def test_search_with_api_key(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "--index_dir", ".", "--query", "q", "--api-key", "sk-test"])
        assert args.api_key == "sk-test"

    def test_verbose_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["-v", "index", "--paths", "x.md"])
        assert args.verbose is True

    def test_no_command_returns_none(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestLoadDocuments:
    def test_loads_json_files(self):
        from treesearch.tree import clear_doc_cache
        clear_doc_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["doc_a", "doc_b"]:
                data = {
                    "doc_name": name,
                    "structure": [{"title": f"{name} root", "node_id": "0000"}],
                    "doc_description": f"Description of {name}",
                }
                with open(os.path.join(tmpdir, f"{name}.json"), "w") as f:
                    json.dump(data, f)

            docs = _load_documents_from_dir(tmpdir)
            assert len(docs) == 2
            assert docs[0].doc_name == "doc_a"
            assert docs[1].doc_name == "doc_b"
            assert docs[0].doc_description == "Description of doc_a"
        clear_doc_cache()

    def test_skips_meta_files(self):
        """_index_meta.json should be excluded from document loading."""
        from treesearch.tree import clear_doc_cache
        clear_doc_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Normal doc
            data = {
                "doc_name": "doc_a",
                "structure": [{"title": "root", "node_id": "0000"}],
            }
            with open(os.path.join(tmpdir, "doc_a.json"), "w") as f:
                json.dump(data, f)
            # Meta file
            with open(os.path.join(tmpdir, "_index_meta.json"), "w") as f:
                json.dump({"some_file": "abc123"}, f)

            docs = _load_documents_from_dir(tmpdir)
            assert len(docs) == 1
            assert docs[0].doc_name == "doc_a"
        clear_doc_cache()

    def test_empty_directory_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(SystemExit):
                _load_documents_from_dir(tmpdir)

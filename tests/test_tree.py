# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.tree module.
"""
import json
import os
import tempfile

import pytest
from treesearch.tree import (
    Document,
    flatten_tree,
    find_node,
    get_leaf_nodes,
    assign_node_ids,
    remove_fields,
    format_structure,
    save_index,
    load_index,
    load_documents,
    clear_doc_cache,
    print_toc,
)


class TestFlattenTree:
    def test_flat_list(self, sample_tree_structure):
        nodes = flatten_tree(sample_tree_structure)
        titles = [n.get("title") for n in nodes]
        assert "Architecture" in titles
        assert "Backend" in titles
        assert "Frontend" in titles
        assert "Deployment" in titles

    def test_single_node(self):
        node = {"title": "Root", "node_id": "0000"}
        nodes = flatten_tree(node)
        assert len(nodes) == 1
        assert nodes[0]["title"] == "Root"

    def test_empty_list(self):
        assert flatten_tree([]) == []


class TestFindNode:
    def test_find_existing(self, sample_tree_structure):
        node = find_node(sample_tree_structure, "0002")
        assert node is not None
        assert node["title"] == "Frontend"

    def test_find_root(self, sample_tree_structure):
        node = find_node(sample_tree_structure, "0000")
        assert node is not None
        assert node["title"] == "Architecture"

    def test_find_nonexistent(self, sample_tree_structure):
        assert find_node(sample_tree_structure, "9999") is None


class TestGetLeafNodes:
    def test_get_leaves(self, sample_tree_structure):
        leaves = get_leaf_nodes(sample_tree_structure)
        titles = [l["title"] for l in leaves]
        assert "Backend" in titles
        assert "Frontend" in titles
        assert "Deployment" in titles
        # Architecture is a parent, not a leaf
        assert "Architecture" not in titles

    def test_single_leaf(self):
        node = {"title": "Leaf", "node_id": "0000"}
        leaves = get_leaf_nodes(node)
        assert len(leaves) == 1


class TestAssignNodeIds:
    def test_assigns_ids(self):
        tree = [
            {"title": "A", "nodes": [{"title": "B"}, {"title": "C"}]},
            {"title": "D"},
        ]
        assign_node_ids(tree)
        flat = flatten_tree(tree)
        ids = [n["node_id"] for n in flat]
        assert ids == ["0000", "0001", "0002", "0003"]


class TestRemoveFields:
    def test_remove_text(self):
        data = {"title": "A", "text": "long content", "summary": "short"}
        result = remove_fields(data, fields=["text"])
        assert "text" not in result
        assert result["summary"] == "short"

    def test_nested_removal(self, sample_tree_structure):
        result = remove_fields(sample_tree_structure, fields=["text"])
        for node in flatten_tree(result):
            assert "text" not in node


class TestFormatStructure:
    def test_reorder_keys(self):
        data = {"text": "content", "title": "A", "summary": "B"}
        result = format_structure(data, order=["title", "summary", "text"])
        keys = list(result.keys())
        assert keys == ["title", "summary", "text"]


class TestSaveLoadIndex:
    def test_round_trip(self, sample_tree_structure):
        clear_doc_cache()
        index = {"doc_name": "test", "structure": sample_tree_structure, "source_path": "/tmp/test.md"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_index(index, path)
            doc = load_index(path)
            assert isinstance(doc, Document)
            assert doc.doc_name == "test"
            assert len(doc.structure) == 2
            assert doc.metadata.get("source_path") == "/tmp/test.md"
        finally:
            clear_doc_cache()
            os.unlink(path)

    def test_cache_returns_same_object(self, sample_tree_structure):
        clear_doc_cache()
        index = {"doc_name": "cached", "structure": sample_tree_structure}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_index(index, path)
            doc1 = load_index(path)
            doc2 = load_index(path)
            assert doc1 is doc2
        finally:
            clear_doc_cache()
            os.unlink(path)

    def test_load_documents_from_dir(self, sample_tree_structure):
        clear_doc_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["alpha", "beta"]:
                index = {"doc_name": name, "structure": sample_tree_structure}
                save_index(index, os.path.join(tmpdir, f"{name}_structure.json"))
            # Add a meta file that should be skipped
            with open(os.path.join(tmpdir, "_index_meta.json"), "w") as f:
                json.dump({"meta": True}, f)

            docs = load_documents(tmpdir)
            assert len(docs) == 2
            assert docs[0].doc_name == "alpha"
            assert docs[1].doc_name == "beta"
        clear_doc_cache()

    def test_creates_directory(self, sample_tree_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "out.json")
            save_index({"data": 1}, path)
            assert os.path.isfile(path)


class TestDocument:
    def test_get_tree_without_text(self, sample_tree_structure):
        doc = Document(
            doc_id="d1", doc_name="test", structure=sample_tree_structure
        )
        tree = doc.get_tree_without_text()
        for node in flatten_tree(tree):
            assert "text" not in node
        # original should still have text
        flat_orig = flatten_tree(doc.structure)
        texts = [n.get("text") for n in flat_orig if "text" in n]
        assert len(texts) > 0

    def test_get_node_by_id(self, sample_tree_structure):
        doc = Document(
            doc_id="d1", doc_name="test", structure=sample_tree_structure
        )
        node = doc.get_node_by_id("0001")
        assert node is not None
        assert node["title"] == "Backend"

    def test_get_node_by_id_missing(self, sample_tree_structure):
        doc = Document(
            doc_id="d1", doc_name="test", structure=sample_tree_structure
        )
        assert doc.get_node_by_id("9999") is None

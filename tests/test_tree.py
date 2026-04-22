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
        node = {"title": "Root", "node_id": "0"}
        nodes = flatten_tree(node)
        assert len(nodes) == 1
        assert nodes[0]["title"] == "Root"

    def test_empty_list(self):
        assert flatten_tree([]) == []


class TestFindNode:
    def test_find_existing(self, sample_tree_structure):
        node = find_node(sample_tree_structure, "2")
        assert node is not None
        assert node["title"] == "Frontend"

    def test_find_root(self, sample_tree_structure):
        node = find_node(sample_tree_structure, "0")
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
        node = {"title": "Leaf", "node_id": "0"}
        leaves = get_leaf_nodes(node)
        assert len(leaves) == 1


class TestAssignNodeIds:
    def test_assigns_ids_unique_and_stable(self):
        tree = [
            {"title": "A", "nodes": [{"title": "B"}, {"title": "C"}]},
            {"title": "D"},
        ]
        assign_node_ids(tree)
        flat = flatten_tree(tree)
        ids = [n["node_id"] for n in flat]
        # All four nodes have an id
        assert len(ids) == 4
        # IDs are stable 16-char hex strings (blake2b digest)
        assert all(len(i) == 16 and all(c in "0123456789abcdef" for c in i) for i in ids)
        # Unique within the tree
        assert len(set(ids)) == 4

    def test_ids_are_deterministic(self):
        """Same structure → identical node_ids across runs (foundation of diff)."""
        tree1 = [
            {"title": "Alpha", "nodes": [{"title": "Beta"}, {"title": "Gamma"}]},
        ]
        tree2 = [
            {"title": "Alpha", "nodes": [{"title": "Beta"}, {"title": "Gamma"}]},
        ]
        assign_node_ids(tree1)
        assign_node_ids(tree2)
        ids1 = [n["node_id"] for n in flatten_tree(tree1)]
        ids2 = [n["node_id"] for n in flatten_tree(tree2)]
        assert ids1 == ids2

    def test_text_edit_does_not_change_id(self):
        """Editing text/summary/etc. must not perturb node_id."""
        tree_a = [{"title": "Section", "text": "v1", "nodes": [{"title": "Sub", "text": "old"}]}]
        tree_b = [{"title": "Section", "text": "v2", "nodes": [{"title": "Sub", "text": "new"}]}]
        assign_node_ids(tree_a)
        assign_node_ids(tree_b)
        ids_a = [n["node_id"] for n in flatten_tree(tree_a)]
        ids_b = [n["node_id"] for n in flatten_tree(tree_b)]
        assert ids_a == ids_b

    def test_duplicate_sibling_titles_disambiguated_by_ordinal(self):
        tree = [
            {"title": "Item", "nodes": []},
            {"title": "Item", "nodes": []},
            {"title": "Item", "nodes": []},
        ]
        assign_node_ids(tree)
        ids = [n["node_id"] for n in flatten_tree(tree)]
        assert len(set(ids)) == 3  # all distinct via ordinal

    def test_normalization_collapses_whitespace_and_case(self):
        """Cosmetic title edits (extra space, casing) should not change id."""
        tree_a = [{"title": "Hello World"}]
        tree_b = [{"title": "  hello   WORLD  "}]
        assign_node_ids(tree_a)
        assign_node_ids(tree_b)
        assert tree_a[0]["node_id"] == tree_b[0]["node_id"]


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
        index = {"doc_name": "test", "structure": sample_tree_structure, "source_path": "/tmp/test.md"}
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            save_index(index, db_path, doc_id="test")
            doc = load_index(db_path, doc_id="test")
            assert isinstance(doc, Document)
            assert doc.doc_name == "test"
            assert len(doc.structure) == 2
            assert doc.metadata.get("source_path") == "/tmp/test.md"

    def test_load_returns_fresh_object(self, sample_tree_structure):
        index = {"doc_name": "cached", "structure": sample_tree_structure}
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            save_index(index, db_path, doc_id="cached")
            doc1 = load_index(db_path, doc_id="cached")
            doc2 = load_index(db_path, doc_id="cached")
            # Each call returns a new Document
            assert doc1 is not doc2
            assert doc1.doc_name == doc2.doc_name

    def test_load_documents_from_db(self, sample_tree_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            for name in ["alpha", "beta"]:
                index = {"doc_name": name, "structure": sample_tree_structure}
                save_index(index, db_path, doc_id=name)

            docs = load_documents(db_path)
            assert len(docs) == 2
            names = [d.doc_name for d in docs]
            assert "alpha" in names
            assert "beta" in names

    def test_creates_directory(self, sample_tree_structure):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sub", "dir", "out.db")
            save_index({"doc_name": "test", "structure": sample_tree_structure}, db_path, doc_id="test")
            assert os.path.isfile(db_path)


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
        node = doc.get_node_by_id("1")
        assert node is not None
        assert node["title"] == "Backend"

    def test_get_node_by_id_missing(self, sample_tree_structure):
        doc = Document(
            doc_id="d1", doc_name="test", structure=sample_tree_structure
        )
        assert doc.get_node_by_id("9999") is None

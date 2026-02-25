# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.indexer module (with mocked LLM).
"""
import json
import os
from unittest.mock import patch, AsyncMock

import pytest
from treesearch.indexer import (
    md_to_tree,
    text_to_tree,
    _extract_md_headings,
    _detect_headings,
    _build_tree,
    _cut_md_text,
    _needs_llm_fallback,
)


class TestExtractMdHeadings:
    def test_basic_headings(self, sample_md_file):
        with open(sample_md_file, "r") as f:
            content = f.read()
        headings, lines = _extract_md_headings(content)
        titles = [h["title"] for h in headings]
        assert "Overview" in titles
        assert "Architecture" in titles
        assert "Backend" in titles

    def test_heading_levels(self, sample_md_file):
        with open(sample_md_file, "r") as f:
            content = f.read()
        headings, lines = _extract_md_headings(content)
        level_map = {h["title"]: h["level"] for h in headings}
        assert level_map["Overview"] == 1
        assert level_map["Architecture"] == 2
        assert level_map["Backend"] == 3

    def test_ignores_code_blocks(self):
        content = "# Real Heading\n\n```\n# Not a heading\n```\n\n## Another Heading"
        headings, lines = _extract_md_headings(content)
        titles = [h["title"] for h in headings]
        assert "Real Heading" in titles
        assert "Not a heading" not in titles
        assert "Another Heading" in titles

    def test_empty_content(self):
        headings, lines = _extract_md_headings("")
        assert headings == []


class TestDetectHeadings:
    def test_all_caps_with_blank_neighbor(self):
        lines = ["", "CHAPTER ONE", "", "Some text here.", "", "CHAPTER TWO", "", "More text."]
        markers = _detect_headings(lines)
        heading_markers = [m for m in markers if m.get("title")]
        assert len(heading_markers) >= 2

    def test_numeric_pattern(self):
        lines = ["1.1 First Item", "Description.", "1.2 Second Item", "Details."]
        markers = _detect_headings(lines)
        heading_markers = [m for m in markers if m.get("title")]
        assert len(heading_markers) >= 2

    def test_hierarchical_numeric(self):
        lines = ["1. Top Level", "content", "1.1 Sub Level", "content", "1.1.1 Sub Sub", "content"]
        markers = _detect_headings(lines)
        levels = {m["title"]: m["level"] for m in markers}
        assert levels.get("1. Top Level", 0) == 1
        assert levels.get("1.1 Sub Level", 0) == 2
        assert levels.get("1.1.1 Sub Sub", 0) == 3

    def test_chinese_chapter(self):
        lines = ["第一章 绪论", "内容", "第二章 方法", "内容"]
        markers = _detect_headings(lines)
        assert len(markers) >= 2
        assert markers[0]["level"] == 1

    def test_roman_numeral(self):
        lines = ["I. Introduction", "content", "II. Methods", "content"]
        markers = _detect_headings(lines)
        assert len(markers) >= 2

    def test_no_pattern(self):
        lines = ["Just a sentence.", "Another sentence.", "Nothing special."]
        markers = _detect_headings(lines)
        heading_markers = [m for m in markers if m.get("title")]
        assert len(heading_markers) == 0

    def test_ignores_code_blocks(self):
        lines = ["```", "1.1 Not a heading", "```", "1.1 Real Heading", "content"]
        markers = _detect_headings(lines)
        heading_markers = [m for m in markers if m.get("title")]
        assert len(heading_markers) == 1
        assert "Real Heading" in heading_markers[0]["title"]


class TestBuildTree:
    def test_flat_nodes(self):
        nodes = [
            {"title": "A", "level": 1, "text": "a", "line_num": 1},
            {"title": "B", "level": 1, "text": "b", "line_num": 5},
        ]
        tree = _build_tree(nodes)
        assert len(tree) == 2
        assert tree[0]["title"] == "A"
        assert tree[1]["title"] == "B"

    def test_nested_nodes(self):
        nodes = [
            {"title": "Parent", "level": 1, "text": "p", "line_num": 1},
            {"title": "Child", "level": 2, "text": "c", "line_num": 3},
        ]
        tree = _build_tree(nodes)
        assert len(tree) == 1
        assert tree[0]["title"] == "Parent"
        assert len(tree[0]["nodes"]) == 1
        assert tree[0]["nodes"][0]["title"] == "Child"

    def test_empty_input(self):
        assert _build_tree([]) == []


class TestNeedsLlmFallback:
    def test_too_few_headings(self):
        markers = [{"level": 1}, {"level": 2}]
        assert _needs_llm_fallback(markers, [""] * 100) is True

    def test_enough_headings(self):
        markers = [{"level": 1}, {"level": 2}, {"level": 2}, {"level": 1}]
        assert _needs_llm_fallback(markers, [""] * 50) is False


class TestMdToTree:
    @pytest.mark.asyncio
    async def test_basic_structure(self, sample_md_file):
        async def mock_achat(prompt, **kwargs):
            return "This is a mock summary."

        with patch("treesearch.indexer.achat", side_effect=mock_achat):
            result = await md_to_tree(
                md_path=sample_md_file,
                if_add_node_summary=True,
                if_add_node_id=True,
            )

        assert "doc_name" in result
        assert "structure" in result
        assert len(result["structure"]) > 0

        from treesearch.tree import flatten_tree
        nodes = flatten_tree(result["structure"])
        assert all("node_id" in n for n in nodes)

    @pytest.mark.asyncio
    async def test_no_summary(self, sample_md_file):
        result = await md_to_tree(
            md_path=sample_md_file,
            if_add_node_summary=False,
            if_add_node_id=True,
        )
        assert "structure" in result

    @pytest.mark.asyncio
    async def test_with_description(self, sample_md_file):
        async def mock_achat(prompt, **kwargs):
            if "description" in prompt.lower() or "one sentence" in prompt.lower():
                return "A document about system architecture."
            return "Mock summary."

        with patch("treesearch.indexer.achat", side_effect=mock_achat):
            result = await md_to_tree(
                md_path=sample_md_file,
                if_add_node_summary=True,
                if_add_doc_description=True,
            )
        assert "doc_description" in result

    @pytest.mark.asyncio
    async def test_from_content(self):
        async def mock_achat(prompt, **kwargs):
            return "Summary."

        content = "# Title\n\nSome content.\n\n## Section\n\nMore content."
        with patch("treesearch.indexer.achat", side_effect=mock_achat):
            result = await md_to_tree(
                md_content=content,
                if_add_node_summary=True,
            )
        assert result["doc_name"] == "untitled"
        assert len(result["structure"]) > 0

    @pytest.mark.asyncio
    async def test_both_path_and_content_raises(self, sample_md_file):
        with pytest.raises(ValueError, match="only one"):
            await md_to_tree(md_path=sample_md_file, md_content="# Test")


class TestTextToTree:
    @pytest.mark.asyncio
    async def test_basic_structure(self, sample_text_file):
        async def mock_achat(prompt, **kwargs):
            return "Mock summary."

        with patch("treesearch.indexer.achat", side_effect=mock_achat):
            result = await text_to_tree(
                text_path=sample_text_file,
                fallback_to_llm="no",
                if_add_node_summary=True,
                if_add_node_id=True,
            )

        assert "doc_name" in result
        assert "structure" in result
        assert len(result["structure"]) > 0

    @pytest.mark.asyncio
    async def test_no_summary(self, sample_text_file):
        result = await text_to_tree(
            text_path=sample_text_file,
            fallback_to_llm="no",
            if_add_node_summary=False,
        )
        assert "structure" in result

    @pytest.mark.asyncio
    async def test_from_content(self):
        content = "1.1 Section A\n\nContent A.\n\n1.2 Section B\n\nContent B."
        result = await text_to_tree(
            text_content=content,
            fallback_to_llm="no",
            if_add_node_summary=False,
        )
        assert result["doc_name"] == "untitled"
        assert len(result["structure"]) > 0

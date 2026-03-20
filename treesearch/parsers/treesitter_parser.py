# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tree-sitter based multi-language code structure parser.

Uses tree-sitter to parse source code into AST and extract classes, functions,
structs, interfaces, etc. Supports all major programming languages.

Requires: pip install tree-sitter-languages
"""
import logging
import os
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

# File extension -> tree-sitter language name mapping
EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "c",
    ".c": "c",
    ".cs": "c_sharp",
    ".php": "php",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".bash": "bash",
    ".sh": "bash",
    ".el": "commonlisp",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".jl": "julia",
    ".ml": "ocaml",
    ".pl": "perl",
    ".m": "objc",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".css": "css",
    ".html": "html",
    ".htm": "html",
    ".dockerfile": "dockerfile",
    ".mk": "make",
}

# -----------------------------------------------------------------------
# Per-language node type -> (level, title_extractor) definitions
# -----------------------------------------------------------------------
# level 1 = top-level constructs (class, struct, interface, module, enum)
# level 2 = member-level constructs (function, method)
# level 3 = nested constructs (inner class, lambda, closure)

# Node types that represent "class-like" constructs (level 1)
_CLASS_NODE_TYPES: set[str] = {
    # Python
    "class_definition",
    # Java / Kotlin
    "class_declaration", "interface_declaration", "enum_declaration",
    "annotation_type_declaration", "record_declaration",
    "object_declaration", "companion_object",
    # C++ / C
    "struct_specifier", "class_specifier", "enum_specifier", "union_specifier",
    "namespace_definition",
    # C#
    "struct_declaration", "namespace_declaration",
    # Go
    "type_declaration",
    # Rust
    "struct_item", "enum_item", "trait_item", "impl_item", "mod_item",
    # Ruby
    "class", "module",
    # PHP
    "trait_declaration",
    # Scala
    "trait_definition", "class_definition", "object_definition",
    # TypeScript / JavaScript
    # (class_declaration already included above)
}

# Node types that represent "function-like" constructs (level 2)
_FUNC_NODE_TYPES: set[str] = {
    # Python
    "function_definition",
    # Java / Kotlin
    "method_declaration", "constructor_declaration",
    "function_declaration",
    # C / C++
    "function_definition", "function_declarator",
    # C#
    "method_declaration", "constructor_declaration",
    "property_declaration", "indexer_declaration",
    # Go
    "function_declaration", "method_declaration",
    # Rust
    "function_item",
    # Swift
    "function_declaration", "init_declaration",
    # Ruby
    "method", "singleton_method",
    # PHP
    "method_declaration", "function_definition",
    # Lua
    "function_declaration", "function_definition_statement",
    # TypeScript / JavaScript
    "function_declaration", "method_definition",
    "arrow_function",  # top-level only
    # Elixir
    # handled specially via call node "def"/"defp"
    # SQL
    "create_function_statement",
    # Bash
    "function_definition",
}


def _get_node_name(node, source_bytes: bytes, lang: str) -> str:
    """Extract the name/title from an AST node."""
    # Try to find a "name" child node
    name_node = node.child_by_field_name("name")
    if name_node:
        return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

    # Go type_declaration: type Name struct/interface
    if node.type == "type_declaration" and lang == "go":
        for child in node.children:
            if child.type == "type_spec":
                spec_name = child.child_by_field_name("name")
                if spec_name:
                    type_text = source_bytes[spec_name.start_byte:spec_name.end_byte].decode("utf-8", errors="replace")
                    # Append type kind (struct/interface)
                    type_node = child.child_by_field_name("type")
                    if type_node:
                        kind = type_node.type.replace("_type", "")
                        return f"{type_text} ({kind})"
                    return type_text
        return _first_line_text(node, source_bytes)

    # Rust impl_item: impl Trait for Type
    if node.type == "impl_item" and lang == "rust":
        return _first_line_text(node, source_bytes)

    # Ruby module/class
    if node.type in ("class", "module") and lang == "ruby":
        for child in node.children:
            if child.type in ("constant", "scope_resolution"):
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return _first_line_text(node, source_bytes)

    # Fallback: first line of the node text
    return _first_line_text(node, source_bytes)


def _first_line_text(node, source_bytes: bytes, max_len: int = 100) -> str:
    """Get the first line of node text, truncated."""
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    first_line = text.split("\n", 1)[0].strip()
    if len(first_line) > max_len:
        return first_line[:max_len - 3] + "..."
    return first_line


def _build_title(node, source_bytes: bytes, lang: str) -> str:
    """Build a descriptive title for a code structure node."""
    name = _get_node_name(node, source_bytes, lang)
    node_type = node.type

    # Classify node type for title prefix
    type_prefixes = {
        # Python
        "class_definition": "class",
        "function_definition": "def",
        # Java / C# / TypeScript
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "method_declaration": "method",
        "constructor_declaration": "constructor",
        "record_declaration": "record",
        "annotation_type_declaration": "@interface",
        # C / C++
        "struct_specifier": "struct",
        "class_specifier": "class",
        "enum_specifier": "enum",
        "union_specifier": "union",
        "namespace_definition": "namespace",
        # Go
        "type_declaration": "type",
        "function_declaration": "func",
        # Rust
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
        "mod_item": "mod",
        "function_item": "fn",
        # Ruby
        "class": "class",
        "module": "module",
        "method": "def",
        "singleton_method": "def self.",
        # PHP
        "trait_declaration": "trait",
        # Kotlin
        "object_declaration": "object",
        "companion_object": "companion object",
        # Scala
        "trait_definition": "trait",
        "object_definition": "object",
        # JS/TS
        "method_definition": "method",
        "arrow_function": "=>",
        # SQL
        "create_function_statement": "CREATE FUNCTION",
    }

    prefix = type_prefixes.get(node_type, "")
    if prefix and name and not name.startswith(prefix):
        return f"{prefix} {name}"
    return name or _first_line_text(node, source_bytes)


def parse_treesitter_structure(source: str, ext: str) -> list[dict]:
    """Parse source code using tree-sitter and return a flat heading list.

    Args:
        source: source code string
        ext: file extension (e.g. ".py", ".go")

    Returns:
        list of {"title": str, "line_num": int, "level": int}
        Returns empty list if tree-sitter is not available or language unsupported.
    """
    if not source or not source.strip():
        return []

    try:
        from tree_sitter_languages import get_parser as ts_get_parser
    except ImportError:
        logger.debug("tree-sitter-languages not installed, skipping tree-sitter parsing")
        return []

    lang_name = EXT_TO_LANGUAGE.get(ext.lower())
    if not lang_name:
        logger.debug("No tree-sitter language mapping for extension: %s", ext)
        return []

    try:
        # Suppress FutureWarning from tree_sitter_languages using deprecated Language(path, name) API
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            parser = ts_get_parser(lang_name)
    except Exception as e:
        logger.debug("Failed to get tree-sitter parser for %s: %s", lang_name, e)
        return []

    source_bytes = source.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.debug("tree-sitter parse failed for %s: %s", lang_name, e)
        return []

    headings: list[dict] = []
    _walk_tree(tree.root_node, source_bytes, lang_name, headings, depth=0)

    # Sort by line number
    headings.sort(key=lambda h: h["line_num"])

    # Deduplicate by line_num (some nodes overlap)
    seen_lines: set[int] = set()
    unique = []
    for h in headings:
        if h["line_num"] not in seen_lines:
            seen_lines.add(h["line_num"])
            unique.append(h)

    logger.debug("tree-sitter extracted %d headings for %s", len(unique), ext)
    return unique


def _walk_tree(
    node,
    source_bytes: bytes,
    lang: str,
    headings: list[dict],
    depth: int,
) -> None:
    """Recursively walk tree-sitter AST and collect class/function nodes."""
    node_type = node.type

    is_class = node_type in _CLASS_NODE_TYPES
    is_func = node_type in _FUNC_NODE_TYPES

    # Special handling for languages with unique patterns
    if lang == "go" and node_type == "type_declaration":
        is_class = True
    elif lang == "rust" and node_type == "impl_item":
        is_class = True

    if is_class or is_func:
        title = _build_title(node, source_bytes, lang)
        # line_num is 1-based
        line_num = node.start_point[0] + 1

        if is_class:
            level = 1
        else:
            # If inside a class-like node (depth > 0 in class context), it's a method
            level = 2

        headings.append({
            "title": title,
            "line_num": line_num,
            "level": level,
        })

    # Recurse into children
    for child in node.children:
        _walk_tree(child, source_bytes, lang, headings, depth + 1)


async def treesitter_code_to_tree(
    code_path: str,
    *,
    model: Optional[str] = None,
    if_thinning: bool = False,
    min_thinning_chars: int = 15000,
    if_add_node_summary: bool = True,
    summary_chars_threshold: int = 600,
    if_add_doc_description: bool = False,
    if_add_node_text: bool = False,
    if_add_node_id: bool = True,
) -> dict:
    """Build a tree index from a code file using tree-sitter.

    Uses tree-sitter for accurate multi-language AST parsing.
    Falls back to regex-based code_to_tree if tree-sitter is unavailable.

    Returns:
        {"doc_name": str, "structure": list, "source_path": str}
    """
    from ..indexer import (
        _build_tree, _cut_md_text, _update_char_counts, _thin_tree,
        generate_summaries, generate_doc_description, code_to_tree,
    )
    from ..tree import assign_node_ids, format_structure

    with open(code_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    doc_name = os.path.splitext(os.path.basename(code_path))[0]
    ext = os.path.splitext(code_path)[1].lower()
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Try tree-sitter first
    headings = parse_treesitter_structure(text, ext)

    if not headings:
        # Fallback to regex-based parser
        logger.debug("tree-sitter returned no results for %s, falling back to regex parser", code_path)
        return await code_to_tree(
            code_path=code_path,
            model=model,
            if_thinning=if_thinning,
            min_thinning_chars=min_thinning_chars,
            if_add_node_summary=if_add_node_summary,
            summary_chars_threshold=summary_chars_threshold,
            if_add_doc_description=if_add_doc_description,
            if_add_node_text=if_add_node_text,
            if_add_node_id=if_add_node_id,
        )

    logger.debug("tree-sitter parsed %s: %d structures in %d lines", ext, len(headings), len(lines))

    markers = [{"title": h["title"], "line_num": h["line_num"], "level": h["level"]} for h in headings]

    if not markers:
        markers = [{"title": doc_name, "line_num": 1, "level": 1}]

    nodes = _cut_md_text(markers, lines)

    if if_thinning and min_thinning_chars:
        nodes = _update_char_counts(nodes)
        logger.debug("Thinning tree (threshold=%d chars)...", min_thinning_chars)
        nodes = _thin_tree(nodes, min_thinning_chars)

    logger.debug("Building tree from %d nodes...", len(nodes))
    tree = _build_tree(nodes)

    if if_add_node_id:
        assign_node_ids(tree)

    base_order = ["title", "node_id", "summary", "prefix_summary"]
    text_fields = ["text"] if if_add_node_text or if_add_node_summary else []
    tail_fields = ["line_start", "line_end", "nodes"]
    order = base_order + text_fields + tail_fields

    tree = format_structure(tree, order=order)

    if if_add_node_summary:
        logger.debug("Generating summaries...")
        tree = generate_summaries(tree, threshold=summary_chars_threshold)
        if not if_add_node_text:
            order_no_text = [f for f in order if f != "text"]
            tree = format_structure(tree, order=order_no_text)

    result = {"doc_name": doc_name, "structure": tree, "source_path": os.path.abspath(code_path)}

    if if_add_doc_description:
        logger.debug("Generating document description...")
        result["doc_description"] = generate_doc_description(tree)

    return result

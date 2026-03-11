# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Python AST-based code structure parser.

Uses the built-in ``ast`` module to extract classes and functions from
Python source files, replacing regex-based detection for `.py` files.
Falls back to regex if AST parsing fails (e.g. syntax errors).
"""
import ast
import logging
import warnings

logger = logging.getLogger(__name__)


def parse_python_structure(source: str) -> list[dict]:
    """Parse Python source code using AST and return a flat heading list.

    Returns:
        list of {'title': str, 'line_num': int, 'level': int}
        level 1 = class, level 2 = function/method
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError as e:
        logger.debug("AST parse failed (syntax error at line %s), falling back to regex", e.lineno)
        return []

    headings = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Build class signature with bases
            bases = ", ".join(_name_of(b) for b in node.bases) if node.bases else ""
            title = f"class {node.name}({bases})" if bases else f"class {node.name}"
            headings.append({
                "title": title,
                "line_num": node.lineno,
                "level": 1,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args_str = _format_args(node.args)
            # Return annotation
            ret = ""
            if node.returns:
                ret = f" -> {_name_of(node.returns)}"
            title = f"{prefix} {node.name}({args_str}){ret}"
            # Truncate long signatures
            if len(title) > 120:
                title = title[:117] + "..."
            headings.append({
                "title": title,
                "line_num": node.lineno,
                "level": 2,
            })

    # Sort by line number (ast.walk doesn't guarantee order)
    headings.sort(key=lambda h: h["line_num"])
    return headings


def _name_of(node) -> str:
    """Extract a readable name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Subscript):
        return f"{_name_of(node.value)}[{_name_of(node.slice)}]"
    elif isinstance(node, ast.Tuple):
        return ", ".join(_name_of(e) for e in node.elts)
    elif isinstance(node, ast.List):
        return "[" + ", ".join(_name_of(e) for e in node.elts) + "]"
    elif isinstance(node, ast.Starred):
        return f"*{_name_of(node.value)}"
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f"{_name_of(node.left)} | {_name_of(node.right)}"
    return "..."


def _format_args(args: ast.arguments) -> str:
    """Format function arguments to a readable string."""
    parts = []

    # Positional-only
    for a in args.posonlyargs:
        parts.append(a.arg)
    if args.posonlyargs:
        parts.append("/")

    # Regular args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, a in enumerate(args.args):
        name = a.arg
        if a.annotation:
            name += f": {_name_of(a.annotation)}"
        di = i - defaults_offset
        if di >= 0 and di < len(args.defaults):
            name += f"={_name_of(args.defaults[di])}"
        parts.append(name)

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    # Keyword-only
    for i, a in enumerate(args.kwonlyargs):
        name = a.arg
        if a.annotation:
            name += f": {_name_of(a.annotation)}"
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            name += f"={_name_of(args.kw_defaults[i])}"
        parts.append(name)

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    result = ", ".join(parts)
    # Skip 'self' / 'cls' at the beginning for readability
    for prefix in ("self, ", "cls, "):
        if result.startswith(prefix):
            result = result[len(prefix):]
            break
    if result in ("self", "cls"):
        result = ""

    return result

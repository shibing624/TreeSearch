# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Path resolution utilities — glob, directory walk, .gitignore support.

Resolves a mix of files, globs, and directories into a flat list of file paths,
with configurable ignore rules and optional .gitignore support via ``pathspec``.
"""
import glob as globmod
import logging
import os
logger = logging.getLogger(__name__)

# Default directories to skip during recursive walk
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".eggs",
})

# Patterns for directory names (matched with fnmatch-style)
_IGNORE_DIR_PATTERNS: frozenset[str] = frozenset({
    "*.egg-info",
})

MAX_DIR_FILES: int = 10_000


def _get_default_extensions() -> set[str]:
    """Return all file extensions supported by the parser registry + SOURCE_TYPE_MAP."""
    from .parsers.registry import ParserRegistry, SOURCE_TYPE_MAP

    exts: set[str] = set()
    exts.update(ParserRegistry.supported_extensions())
    exts.update(SOURCE_TYPE_MAP.keys())
    return exts


def _find_gitignore(start_dir: str) -> str | None:
    """Find the nearest .gitignore by searching *start_dir* and its parents.

    Stops searching when a ``.git`` directory is found (project root) or the
    filesystem root is reached.  Returns the path to the ``.gitignore`` file,
    or ``None`` if not found.
    """
    curr = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(curr, ".gitignore")
        if os.path.isfile(candidate):
            return candidate
        # Stop at project root (.git present) or filesystem root
        if os.path.isdir(os.path.join(curr, ".git")):
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return None


def _load_gitignore_spec(root: str):
    """Try to load the nearest .gitignore as a ``pathspec`` matcher.

    Searches *root* and its parent directories (up to the project root).
    Returns a ``(pathspec.PathSpec, base_dir)`` tuple, or ``(None, root)``
    if the file is not found or ``pathspec`` is not installed.
    """
    gitignore_path = _find_gitignore(root)
    if not gitignore_path:
        return None, root
    import pathspec
    base_dir = os.path.dirname(gitignore_path)
    with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
        return pathspec.PathSpec.from_lines("gitwildmatch", f), base_dir


def _should_ignore_dir(dirname: str, ignore_dirs: frozenset[str]) -> bool:
    """Check if a directory name should be ignored."""
    if dirname in ignore_dirs:
        return True
    import fnmatch
    for pat in _IGNORE_DIR_PATTERNS:
        if fnmatch.fnmatch(dirname, pat):
            return True
    return False


def _walk_directory(
    directory: str,
    *,
    allowed_extensions: set[str] | None = None,
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
    respect_gitignore: bool = True,
    max_files: int = MAX_DIR_FILES,
    follow_symlinks: bool = False,
) -> list[str]:
    """Recursively walk *directory* and return matching file paths."""
    directory = os.path.abspath(directory)
    gitignore_spec, gitignore_base = _load_gitignore_spec(directory) if respect_gitignore else (None, directory)

    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=follow_symlinks):
        # Filter out ignored directories (in-place to prevent os.walk descent)
        dirnames[:] = [
            d for d in dirnames
            if not _should_ignore_dir(d, ignore_dirs)
        ]
        dirnames.sort()

        for fname in sorted(filenames):
            # Extension filter
            if allowed_extensions is not None:
                _, ext = os.path.splitext(fname)
                if ext.lower() not in allowed_extensions:
                    continue

            full_path = os.path.join(dirpath, fname)

            # .gitignore filter
            if gitignore_spec is not None:
                rel = os.path.relpath(full_path, gitignore_base)
                if gitignore_spec.match_file(rel):
                    continue

            results.append(full_path)
            if len(results) > max_files:
                raise ValueError(
                    f"Directory '{directory}' contains more than {max_files} matching files. "
                    f"Use a more specific path or increase max_files."
                )

    return results


def resolve_paths(
    patterns: list[str],
    *,
    allowed_extensions: set[str] | None = None,
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
    respect_gitignore: bool = True,
    max_files: int = MAX_DIR_FILES,
    follow_symlinks: bool = False,
) -> list[str]:
    """Resolve a mix of files, globs, and directories into file paths.

    Resolution rules:
    1. Glob pattern (contains ``*`` or ``?``) -> ``glob.glob(recursive=True)``
    2. Regular file -> include directly
    3. Directory -> recursive walk, filter by *allowed_extensions*,
       skip *ignore_dirs*, optionally respect ``.gitignore``

    Args:
        patterns: list of file paths, glob patterns, or directories.
        allowed_extensions: set of lowercase extensions (e.g. ``{".py", ".md"}``).
            Defaults to all extensions known to the parser registry.
            Only applied to directory walks — explicit files and globs are not filtered.
        ignore_dirs: directory names to skip during recursive walk.
        respect_gitignore: if True and ``pathspec`` is installed, honour
            ``.gitignore`` files found at the root of walked directories.
        max_files: safety cap on total files from a single directory walk.
        follow_symlinks: follow symbolic links during directory walk.

    Returns:
        List of resolved file paths (deduplicated, order-preserved).
    """
    if allowed_extensions is None:
        allowed_extensions = _get_default_extensions()

    resolved: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        abspath = os.path.abspath(path)
        if abspath not in seen:
            seen.add(abspath)
            resolved.append(path)

    for p in patterns:
        if "*" in p or "?" in p:
            # Glob pattern
            for match in sorted(globmod.glob(p, recursive=True)):
                if os.path.isfile(match):
                    _add(match)
        elif os.path.isdir(p):
            # Directory → recursive walk
            for fp in _walk_directory(
                p,
                allowed_extensions=allowed_extensions,
                ignore_dirs=ignore_dirs,
                respect_gitignore=respect_gitignore,
                max_files=max_files,
                follow_symlinks=follow_symlinks,
            ):
                _add(fp)
        elif os.path.isfile(p):
            # Regular file
            _add(p)
        else:
            logger.warning("Path not found: %s", p)

    return resolved

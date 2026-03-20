# -*- coding: utf-8 -*-
"""Tests for treesearch.pathutil — resolve_paths utility."""
import os
import pytest
from treesearch.pathutil import (
    resolve_paths,
    DEFAULT_IGNORE_DIRS,
    MAX_DIR_FILES,
    _should_ignore_dir,
)


class TestResolvePathsBasic:
    """Basic resolution: files, globs, missing paths."""

    def test_single_file(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("print(1)")
        result = resolve_paths([str(f)])
        assert len(result) == 1
        assert os.path.abspath(str(f)) in {os.path.abspath(r) for r in result}

    def test_glob_pattern(self, tmp_path):
        for name in ("a.py", "b.py", "c.txt"):
            (tmp_path / name).write_text("x")
        pattern = str(tmp_path / "*.py")
        result = resolve_paths([pattern])
        assert len(result) == 2

    def test_missing_path_warns(self, tmp_path):
        result = resolve_paths([str(tmp_path / "nonexistent.py")])
        assert result == []

    def test_deduplication(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x")
        result = resolve_paths([str(f), str(f)])
        assert len(result) == 1


class TestResolvePathsDirectory:
    """Directory walking."""

    def test_directory_walk(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("y")
        result = resolve_paths([str(tmp_path)], allowed_extensions={".py"})
        assert len(result) == 2

    def test_directory_filters_extensions(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.dat").write_text("y")
        result = resolve_paths([str(tmp_path)], allowed_extensions={".py"})
        assert len(result) == 1
        assert result[0].endswith(".py")

    def test_directory_skips_ignore_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        nd = tmp_path / "node_modules"
        nd.mkdir()
        (nd / "b.py").write_text("y")
        result = resolve_paths([str(tmp_path)], allowed_extensions={".py"})
        assert len(result) == 1

    def test_directory_skips_git(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        gd = tmp_path / ".git"
        gd.mkdir()
        (gd / "config").write_text("z")
        result = resolve_paths([str(tmp_path)], allowed_extensions={".py"})
        assert len(result) == 1

    def test_max_files_limit(self, tmp_path):
        for i in range(5):
            (tmp_path / f"{i}.py").write_text("x")
        with pytest.raises(ValueError, match="more than 3"):
            resolve_paths([str(tmp_path)], allowed_extensions={".py"}, max_files=3)

    def test_egg_info_ignored(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        ei = tmp_path / "foo.egg-info"
        ei.mkdir()
        (ei / "PKG-INFO").write_text("y")
        result = resolve_paths([str(tmp_path)], allowed_extensions={".py"})
        assert len(result) == 1


class TestShouldIgnoreDir:
    def test_known_dirs(self):
        assert _should_ignore_dir(".git", DEFAULT_IGNORE_DIRS)
        assert _should_ignore_dir("node_modules", DEFAULT_IGNORE_DIRS)
        assert _should_ignore_dir("__pycache__", DEFAULT_IGNORE_DIRS)

    def test_normal_dir_not_ignored(self):
        assert not _should_ignore_dir("src", DEFAULT_IGNORE_DIRS)

    def test_egg_info_pattern(self):
        assert _should_ignore_dir("mypackage.egg-info", DEFAULT_IGNORE_DIRS)


class TestGitignoreSupport:
    """Test .gitignore integration."""

    def test_gitignore_respected(self, tmp_path):
        (tmp_path / ".gitignore").write_text("ignored.py\n")
        (tmp_path / "kept.py").write_text("x")
        (tmp_path / "ignored.py").write_text("y")
        result = resolve_paths(
            [str(tmp_path)],
            allowed_extensions={".py"},
            respect_gitignore=True,
        )
        names = [os.path.basename(r) for r in result]
        assert "kept.py" in names
        assert "ignored.py" not in names

    def test_gitignore_disabled(self, tmp_path):
        (tmp_path / ".gitignore").write_text("ignored.py\n")
        (tmp_path / "kept.py").write_text("x")
        (tmp_path / "ignored.py").write_text("y")
        result = resolve_paths(
            [str(tmp_path)],
            allowed_extensions={".py"},
            respect_gitignore=False,
        )
        names = [os.path.basename(r) for r in result]
        assert "kept.py" in names
        assert "ignored.py" in names


class TestMixedPatterns:
    """Mix of files, globs, and directories in one call."""

    def test_mixed(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        (tmp_path / "c.md").write_text("z")
        result = resolve_paths(
            [str(sub), str(tmp_path / "b.py"), str(tmp_path / "*.md")],
            allowed_extensions={".py", ".md"},
        )
        assert len(result) == 3

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Shared fixtures for TreeSearch tests.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ---------------------------------------------------------------------------
# Isolate tests from environment
# ---------------------------------------------------------------------------
# This autouse fixture saves and removes all TREESEARCH_* and OPENAI_*
# vars before each test, then restores them after.
# ---------------------------------------------------------------------------

_ENV_PREFIXES = ("TREESEARCH_", "OPENAI_")


@pytest.fixture(autouse=True)
def _isolate_env_and_config():
    """Remove .env-injected vars before each test, restore after."""
    from treesearch.config import reset_config
    from treesearch.fts import reset_fts_index
    from treesearch.tokenizer import reset_jieba
    saved = {}
    for k in list(os.environ):
        if k.startswith(_ENV_PREFIXES):
            saved[k] = os.environ.pop(k)
    reset_config()
    reset_fts_index()
    reset_jieba()
    yield
    # Restore
    for k in list(os.environ):
        if k.startswith(_ENV_PREFIXES):
            os.environ.pop(k, None)
    os.environ.update(saved)
    reset_config()
    reset_fts_index()
    reset_jieba()


@pytest.fixture
def sample_md_file():
    """Create a temp Markdown file for testing."""
    content = """\
# Overview

This document describes the system.

## Architecture

The system uses microservices architecture.

### Backend

Built with Python and FastAPI.

### Frontend

Built with React and TypeScript.

## Deployment

Deployed on Kubernetes with Helm charts.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def sample_text_file():
    """Create a temp plain text file for testing."""
    content = """\
Chapter 1 Introduction

This is the introduction section.

1.1 Background

Some background information here.

1.2 Motivation

The motivation for this work.

Chapter 2 Methods

Description of methods used.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def sample_tree_structure():
    """A pre-built tree structure for testing search/retrieval."""
    return [
        {
            "title": "Architecture",
            "summary": "System architecture overview.",
            "node_id": "0",
            "nodes": [
                {
                    "title": "Backend",
                    "summary": "Python FastAPI backend.",
                    "node_id": "1",
                    "text": "The backend is built with Python and FastAPI. It handles REST API requests.",
                },
                {
                    "title": "Frontend",
                    "summary": "React TypeScript frontend.",
                    "node_id": "2",
                    "text": "The frontend uses React with TypeScript for type safety.",
                },
            ],
        },
        {
            "title": "Deployment",
            "summary": "Kubernetes deployment.",
            "node_id": "3",
            "text": "The app is deployed on Kubernetes using Helm charts and ArgoCD.",
        },
    ]

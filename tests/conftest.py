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
            "node_id": "0000",
            "nodes": [
                {
                    "title": "Backend",
                    "summary": "Python FastAPI backend.",
                    "node_id": "0001",
                    "text": "The backend is built with Python and FastAPI. It handles REST API requests.",
                },
                {
                    "title": "Frontend",
                    "summary": "React TypeScript frontend.",
                    "node_id": "0002",
                    "text": "The frontend uses React with TypeScript for type safety.",
                },
            ],
        },
        {
            "title": "Deployment",
            "summary": "Kubernetes deployment.",
            "node_id": "0003",
            "text": "The app is deployed on Kubernetes using Helm charts and ArgoCD.",
        },
    ]

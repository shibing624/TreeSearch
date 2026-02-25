# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Plain text indexing demo with automatic heading detection.

Demonstrates TreeSearch's ability to parse unstructured text into a tree:
  - Rule-based heading detection (numeric, Chinese, Roman numeral, etc.)
  - Optional LLM fallback for documents without clear heading patterns
  - No special formatting required -- works with any plain text

Usage:
    python examples/02_text_indexing.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import text_to_tree, save_index, print_toc

SAMPLE_TEXT = """\
1. Introduction

This chapter introduces the fundamental concepts of distributed systems.
A distributed system is a collection of independent computers that appears
to its users as a single coherent system.

1.1 Definition and Goals

The primary goals of distributed systems include:
- Resource sharing across networked computers
- Transparency in distribution
- Openness and scalability

1.2 Types of Distributed Systems

There are several categories:
- Distributed computing systems (cluster, grid, cloud)
- Distributed information systems (transaction processing)
- Distributed pervasive systems (IoT, sensor networks)

2. Architecture

This chapter covers architectural styles for distributed systems.

2.1 Layered Architecture

Software is organized into layers where each layer provides services
to the layer above and uses services from the layer below.
Common examples include the OSI model and three-tier web architecture.

2.2 Peer-to-Peer Architecture

In P2P systems, all nodes are equal and can act as both client and server.
Examples include BitTorrent and blockchain networks.

3. Communication

3.1 Remote Procedure Call

RPC allows a program to call procedures on remote machines as if they
were local. Modern implementations include gRPC and Thrift.

3.2 Message-Oriented Communication

Systems communicate by exchanging messages through middleware
such as RabbitMQ, Apache Kafka, or ZeroMQ.
"""


async def main():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_TEXT)
        text_path = f.name

    try:
        # Build tree with rule-based heading detection (no LLM needed for this step)
        print("=== Building tree index from plain text ===")
        print("Using rule-based heading detection (no LLM fallback)\n")

        result = await text_to_tree(
            text_path=text_path,
            fallback_to_llm="no",
            if_add_node_summary=True,
            if_add_node_id=True,
        )

        print("Table of Contents:")
        print_toc(result["structure"])

        output_path = "indexes/distributed_systems_structure.json"
        save_index(result, output_path)
        print(f"\nIndex saved to: {output_path}")

    finally:
        os.unlink(text_path)


if __name__ == "__main__":
    asyncio.run(main())

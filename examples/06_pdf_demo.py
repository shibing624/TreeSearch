
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch.indexer import build_index
from treesearch.search import search

async def pdf_demo():
    # 1. Build index for PDF files

    PDF_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
    print(f"Building index for PDFs in: {PDF_DATA_DIR}")
    
    # We use a temporary index directory
    index_dir = "./pdf_indexes"
    
    # Build index (this will use our fixed pdf_parser)
    # Note: we disable LLM-based summaries by default in treesearch as per current implementation
    documents = await build_index(
        paths=[PDF_DATA_DIR],
        output_dir=index_dir,
        force=True,
        if_add_node_summary=True # In treesearch, this uses character-length heuristic, not LLM
    )
    
    print(f"Successfully indexed {len(documents)} documents.")
    for doc in documents:
        print(f" - {doc.doc_name} ({len(doc.structure)} root nodes)")

    # 2. Perform a search
    query = "Earth Mover"
    print(f"\nSearching for: '{query}'")
    result_dict = await search(query, documents=documents)
    flat_nodes = result_dict.get("flat_nodes", [])
    
    print(f"\nFound {len(flat_nodes)} matches:")
    for i, res in enumerate(flat_nodes[:5], 1):
        print(f"{i}. [{res['score']:.4f}] {res['doc_name']} > {res['title']}")
        # Summary is not in flat_nodes by default, but we can check doc structure
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(pdf_demo())

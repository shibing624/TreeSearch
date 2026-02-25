# API Reference

## Indexing

### `md_to_tree`

```python
async def md_to_tree(
    md_path: str = None,
    md_content: str = None,
    *,
    model: str = "gpt-4o-2024-11-20",
    if_thinning: bool = False,
    min_token_threshold: int = 5000,
    if_add_node_summary: bool = True,
    summary_token_threshold: int = 200,
    if_add_doc_description: bool = False,
    if_add_node_text: bool = False,
    if_add_node_id: bool = True,
) -> dict
```

Build a tree index from a Markdown file or string.

**Returns**: `{'doc_name': str, 'structure': list, 'doc_description'?: str}`

### `text_to_tree`

```python
async def text_to_tree(
    text_path: str = None,
    text_content: str = None,
    *,
    model: str = "gpt-4o-2024-11-20",
    fallback_to_llm: str = "auto",  # "auto" | "yes" | "no"
    if_thinning: bool = False,
    ...
) -> dict
```

Build a tree index from plain text. Supports rule-based heading detection with optional LLM fallback.

**Heading patterns supported**: Numeric (`1.1`, `1.2.3`), Chinese chapters (`第一章`), Roman numerals (`I.`, `II.`), RST underlines, ALL CAPS, and more.

### `build_index`

```python
async def build_index(
    paths: list[str],
    output_dir: str = "./indexes",
    *,
    model: str = "gpt-4o-2024-11-20",
    if_add_node_summary: bool = True,
    if_add_doc_description: bool = True,
    if_add_node_text: bool = True,
    if_add_node_id: bool = True,
    **kwargs,
) -> list[dict]
```

Build tree indexes for multiple files concurrently. Accepts glob patterns (e.g. `["docs/*.md", "paper.txt"]`). Auto-detects `.md`/`.markdown` vs text by extension.

**Returns**: list of result dicts (same format as `md_to_tree` / `text_to_tree`)

---

## Search

### `search`

```python
async def search(
    query: str,
    documents: list[Document],
    model: str = "gpt-4o-2024-11-20",
    top_k_docs: int = 3,
    max_nodes_per_doc: int = 5,
    strategy: str = "best_first",
    mcts_iterations: int = 10,
    value_threshold: float = 0.3,
    max_llm_calls: int = 30,
    use_bm25: bool = True,
    expert_knowledge: str = "",
) -> SearchResult
```

Search across one or more documents using tree-structured retrieval. This is the primary API — it natively supports multi-document search:

1. Route query to relevant documents (LLM reasoning, skipped for single doc)
2. (Optional) BM25 pre-scoring over tree nodes for initial ranking
3. Tree search within each document (best_first / mcts / llm)
4. Return ranked nodes with text content

**Args**:

| Parameter | Description |
|-----------|-------------|
| `query` | User query string |
| `documents` | List of `Document` objects (single or multiple) |
| `strategy` | `"best_first"` (default), `"mcts"`, or `"llm"` |
| `max_llm_calls` | Max LLM calls per document (best_first only, default: 30) |
| `use_bm25` | Enable BM25 pre-scoring for best_first strategy (default: True) |
| `mcts_iterations` | MCTS iteration count (mcts only, default: 10) |
| `value_threshold` | Minimum relevance score (default: 0.3) |
| `top_k_docs` | Max documents to search in routing stage (default: 3) |
| `max_nodes_per_doc` | Max result nodes per document (default: 5) |
| `expert_knowledge` | Optional domain knowledge to guide search |

### `search_sync`

Synchronous wrapper: `search_sync(query, documents, **kwargs) -> SearchResult`

### `BestFirstTreeSearch`

```python
class BestFirstTreeSearch:
    def __init__(
        self,
        document: Document,
        query: str,
        model: str = "gpt-4o-2024-11-20",
        max_results: int = 5,
        threshold: float = 0.3,
        max_llm_calls: int = 30,
        bm25_scores: dict[str, float] = None,
        bm25_weight: float = 0.3,
        depth_penalty: float = 0.02,
        use_subtree_cache: bool = True,
    )

    async def run(self) -> list[dict]

    @property
    def llm_calls(self) -> int

    @classmethod
    def clear_subtree_cache(cls)
```

**Default search strategy.** Deterministic best-first tree search with three-layer design:

- **Layer 1**: BM25 pre-scoring (optional, provides initial priority via `bm25_scores`)
- **Layer 2**: Priority queue expansion with LLM relevance evaluation (title + summary only)
- **Layer 3**: Budget-controlled LLM calls with early stopping

**Features**:
- Priority queue driven: always expands the most promising node first
- BM25 warm start: when provided, nodes start with BM25-based priority
- Early stopping: stops when top-of-queue score drops below `threshold`
- Budget control: `max_llm_calls` limits total LLM invocations
- Subtree cache: class-level `(query_fingerprint, node_id) -> relevance` cache for reuse across searches

**Returns**: `[{'node_id', 'title', 'score'}]`

### `MCTSTreeSearch`

```python
class MCTSTreeSearch:
    def __init__(
        self,
        document: Document,
        query: str,
        model: str = "gpt-4o-2024-11-20",
        exploration_weight: float = 1.0,
        max_iterations: int = 10,
        max_selected_nodes: int = 5,
        value_threshold: float = 0.3,
    )

    async def run(self) -> list[dict]

    @property
    def llm_calls(self) -> int
```

MCTS-based tree search over a single document. Deterministic + cache-friendly design (temperature=0, UCB tie-break by node_id, value cache per query×node).

**Returns**: `[{'node_id', 'title', 'score', 'visits'}]`

### `llm_tree_search`

```python
async def llm_tree_search(
    query: str,
    document: Document,
    model: str = "gpt-4o-2024-11-20",
    expert_knowledge: str = "",
) -> list[dict]
```

Single-pass LLM tree search. Fastest but least thorough — sends full tree structure to LLM in one call.

**Returns**: `[{'node_id', 'title'}]`

### `route_documents`

```python
async def route_documents(
    query: str,
    documents: list[Document],
    model: str = "gpt-4o-2024-11-20",
    top_k: int = 3,
) -> list[Document]
```

Select relevant documents by LLM reasoning over descriptions. No vector embeddings needed.

---

## BM25

### `tokenize`

```python
def tokenize(text: str) -> list[str]
```

Tokenize text for BM25 indexing. Supports Chinese and English:
- **Chinese**: jieba word segmentation (falls back to character-level if jieba not installed)
- **English**: lowercase + split on non-word characters
- **Mixed**: handled correctly

### `BM25Okapi`

```python
class BM25Okapi:
    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    )

    def get_scores(self, query: list[str]) -> list[float]
    def get_top_n(self, query: list[str], n: int = 5) -> list[tuple[int, float]]
```

Pure Python BM25 Okapi implementation. No numpy dependency.

### `NodeBM25Index`

```python
class NodeBM25Index:
    def __init__(
        self,
        documents: list[Document],
        title_weight: float = 1.0,
        summary_weight: float = 0.7,
        body_weight: float = 0.3,
        ancestor_decay: float = 0.5,
        tokenizer: Callable = None,
    )

    def search(self, query: str, top_k: int = 20, propagate: bool = True) -> list[dict]
    def get_node_scores_for_doc(self, query: str, doc_id: str, top_k: int = 50) -> dict[str, float]
```

Structure-aware BM25 index over document tree nodes. No LLM needed — instant results.

**Features**:
- **Hierarchical field weighting**: `title × 1.0 + summary × 0.7 + body × 0.3`
- **Ancestor score propagation**: `parent.score += alpha × max(child.score)` — fixes BM25's structural blindness
- **Chinese + English** via jieba tokenizer
- **Multi-document** support with per-document filtering

**`search()` returns**: `[{'node_id', 'doc_id', 'title', 'summary', 'bm25_score'}]`

---

## Data Models

### `Document`

```python
@dataclass
class Document:
    doc_id: str
    doc_name: str
    structure: list       # tree structure (list of root nodes)
    doc_description: str = ""
    metadata: dict = field(default_factory=dict)

    def get_tree_without_text(self) -> list
    def get_node_by_id(self, node_id: str) -> Optional[dict]
```

### `SearchResult`

```python
@dataclass
class SearchResult:
    documents: list   # [{'doc_id', 'doc_name', 'nodes': [{'node_id', 'title', 'text', 'score'}]}]
    query: str = ""
    total_llm_calls: int = 0
    strategy: str = ""
```

---

## Tree Utilities

| Function | Description |
|---|---|
| `flatten_tree(structure)` | Flatten tree into list of node dicts |
| `find_node(structure, node_id)` | Find node by ID |
| `get_leaf_nodes(structure)` | Get all leaf nodes |
| `assign_node_ids(data)` | Assign zero-padded IDs to all nodes |
| `remove_fields(data, fields)` | Remove specified fields recursively |
| `save_index(index, path)` | Save tree index to JSON |
| `load_index(path)` | Load tree index from JSON |
| `print_toc(tree)` | Print tree as table of contents |

---

## CLI

```bash
# Build index from Markdown (single file)
treesearch index --paths document.md

# Build indexes from glob pattern (multiple files)
treesearch index --paths "docs/*.md" --add-description

# Build index from mixed file types
treesearch index --paths docs/*.md paper.txt -o ./indexes

# Search with Best-First strategy (default, BM25 + LLM)
treesearch search --index_dir ./indexes/ --query "How does authentication work?"

# Search with MCTS strategy
treesearch search --index_dir ./indexes/ --query "deployment" --strategy mcts

# Search with single-pass LLM (fastest, less thorough)
treesearch search --index_dir ./indexes/ --query "config" --strategy llm

# Search without BM25 pre-scoring
treesearch search --index_dir ./indexes/ --query "auth" --no-bm25

# Control LLM budget
treesearch search --index_dir ./indexes/ --query "auth" --max-llm-calls 10
```

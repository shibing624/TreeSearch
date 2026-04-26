# TreeSearch Code V2: Structure-Preserving Graph RAG 改造方案

## 0. 目标

本文档描述如何把当前 TreeSearch 从 structure-aware sparse retrieval library 改造成论文 V2 所需的：

> TreeSearch-Guided Structure-Preserving Graph RAG

目标不是把 `vector-graph-rag` 原样搬进来，而是在 TreeSearch 现有能力上新增一个可选 GraphRAG 层：

1. TreeSearch 继续负责文件解析、结构节点、FTS5/Grep 检索、line-level grounding。
2. 新增 graph layer，把 TreeSearch nodes 转换为 entity-relation-node graph。
3. 新增 structure-constrained expansion，用 TreeSearch 的结构信号约束 GraphRAG 子图扩展。
4. 新增 evidence chain selector，用一次结构化 LLM 调用选择可验证证据链。
5. 新增 verifier，保证 relation、node、citation 都能映射回真实 TreeSearch node。
6. 新增 benchmark 与 smoke demo，为论文实验服务。

## 1. 当前 repo 可复用的能力

当前 TreeSearch 已经具备 GraphRAG 改造所需的底座。

### 1.1 结构节点

文件：`treesearch/tree.py`

已有能力：

1. `Document`
2. stable `node_id`
3. `source_type`
4. `metadata`
5. `line_start` / `line_end`
6. `get_node_by_id()`
7. `get_parent_id()`
8. `get_children_ids()`
9. `get_sibling_ids()`
10. `get_path_to_root()`
11. `get_subtree_node_ids()`

这些能力可以直接作为 graph substrate。

### 1.2 检索 seed

文件：`treesearch/search.py`

已有能力：

1. FTS5 node scoring
2. GrepFilter for code / regex
3. `search_mode=auto/tree/flat`
4. `include_ancestors`
5. `line_start` / `line_end` result fields
6. `flat_nodes`
7. multi-document routing

这些能力可以直接作为 GraphRAG seed retrieval。

### 1.3 FTS5 和持久化

文件：`treesearch/fts.py`

已有能力：

1. SQLite `documents` table
2. SQLite `nodes` table
3. FTS5 `fts_nodes`
4. node-level incremental diff
5. source_path / source_type persistence
6. document structure JSON persistence

这些能力可以复用，也可以扩展新 graph tables。

### 1.4 Benchmark 基础

目录：`examples/benchmark/`

已有：

1. QASPER
2. FinanceBench
3. CodeSearchNet
4. HotpotQA
5. metrics
6. benchmark_utils

后续新增 GraphRAG benchmark 时应复用这里的样式。

## 2. 总体架构

新增模块建议放在 `treesearch/rag/` 下，避免污染现有轻量检索 API。

```text
treesearch/
  rag/
    __init__.py
    models.py
    node_graph.py
    extractors.py
    graph_store.py
    graph_builder.py
    seed.py
    expansion.py
    evidence.py
    verifier.py
    answer.py
    pipeline.py
```

职责：

| 文件 | 责任 |
|---|---|
| `models.py` | GraphRAG 数据结构 |
| `node_graph.py` | TreeSearch node 到 graph passage 的转换 |
| `extractors.py` | entity/relation/triplet 抽取接口与 mock extractor |
| `graph_store.py` | SQLite graph tables 或内存 graph store |
| `graph_builder.py` | 从 Document 构建 node-level graph |
| `seed.py` | 调用 TreeSearch search 获取 seed nodes |
| `expansion.py` | structure-constrained subgraph expansion |
| `evidence.py` | single-pass evidence chain selection |
| `verifier.py` | relation/node/citation/verifiable chain 检查 |
| `answer.py` | grounded answer generation |
| `pipeline.py` | 面向用户的 TreeSearchGraphRAG API |

先做 SQLite / in-memory store，不要一上来强依赖 Milvus。向量检索可以作为 optional backend。

## 3. 新增数据模型

文件：`treesearch/rag/models.py`

建议定义：

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class GraphNodePassage:
    node_id: str
    doc_id: str
    doc_name: str
    source_path: str
    source_type: str
    title: str
    text: str
    path_titles: tuple[str, ...]
    line_start: int | None
    line_end: int | None
    parent_node_id: str | None = None
    child_node_ids: tuple[str, ...] = ()
    sibling_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEntity:
    entity_id: str
    text: str
    node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphRelation:
    relation_id: str
    subject: str
    predicate: str
    object: str
    text: str
    node_id: str
    doc_id: str
    source_type: str


@dataclass(frozen=True)
class StructuralEdge:
    src_node_id: str
    dst_node_id: str
    edge_type: Literal["parent", "child", "sibling", "same_doc"]


@dataclass(frozen=True)
class GraphSeed:
    node_id: str
    doc_id: str
    score: float
    source: Literal["fts5", "grep", "tree"]


@dataclass(frozen=True)
class CandidateRelation:
    relation: GraphRelation
    semantic_score: float = 0.0
    sparse_seed_score: float = 0.0
    structure_score: float = 0.0
    source_type_score: float = 0.0
    grounding_score: float = 0.0

    @property
    def total_score(self) -> float:
        return (
            self.semantic_score
            + self.sparse_seed_score
            + self.structure_score
            + self.source_type_score
            + self.grounding_score
        )


@dataclass(frozen=True)
class EvidenceCitation:
    node_id: str
    doc_id: str
    source_path: str
    line_start: int | None
    line_end: int | None
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceChain:
    query: str
    bridge_entities: tuple[str, ...]
    selected_relation_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    reasoning_chain: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...]
    evidence_sufficiency: bool
```

注意：

1. 不要用 `getattr(self, "_xxx", default)` 这种假防御。
2. 所有字段在 dataclass 明确定义。
3. 先用 tuple 保持不可变，便于测试和缓存。

## 4. TreeSearch node 到 graph passage

文件：`treesearch/rag/node_graph.py`

目标：把 `Document` 的每个 node 转成 `GraphNodePassage`。

核心函数：

```python
from treesearch.tree import Document, flatten_tree
from treesearch.rag.models import GraphNodePassage, StructuralEdge


def document_to_node_passages(document: Document) -> list[GraphNodePassage]:
    passages: list[GraphNodePassage] = []
    source_path = str(document.metadata.get("source_path", ""))

    for node in flatten_tree(document.structure):
        node_id = str(node.get("node_id", ""))
        if not node_id:
            continue

        path_ids = document.get_path_to_root(node_id)
        path_titles = tuple(
            p.get("title", "")
            for p in (document.get_node_by_id(pid) for pid in path_ids)
            if p
        )

        passages.append(
            GraphNodePassage(
                node_id=node_id,
                doc_id=document.doc_id,
                doc_name=document.doc_name,
                source_path=source_path,
                source_type=document.source_type,
                title=str(node.get("title", "")),
                text=str(node.get("text", "")),
                path_titles=path_titles,
                line_start=node.get("line_start"),
                line_end=node.get("line_end"),
                parent_node_id=document.get_parent_id(node_id),
                child_node_ids=tuple(document.get_children_ids(node_id)),
                sibling_node_ids=tuple(document.get_sibling_ids(node_id)),
            )
        )

    return passages
```

结构边：

```python
def document_to_structural_edges(document: Document) -> list[StructuralEdge]:
    edges: list[StructuralEdge] = []
    for node in flatten_tree(document.structure):
        node_id = str(node.get("node_id", ""))
        if not node_id:
            continue

        parent_id = document.get_parent_id(node_id)
        if parent_id:
            edges.append(StructuralEdge(parent_id, node_id, "child"))
            edges.append(StructuralEdge(node_id, parent_id, "parent"))

        for sibling_id in document.get_sibling_ids(node_id):
            edges.append(StructuralEdge(node_id, sibling_id, "sibling"))

    return edges
```

测试：

1. Markdown tree 能输出 path_titles。
2. code tree 能保留 line_start / line_end。
3. parent/child/sibling edges 正确。
4. 空 node_id 被跳过。

## 5. Triplet 抽取接口

文件：`treesearch/rag/extractors.py`

先不要强绑 OpenAI，定义协议和两个实现：

1. `RuleBasedTripletExtractor`：用于 tests 和 smoke demo。
2. `LLMTripletExtractor`：后续接 OpenAI/Hunyuan/本地模型。

接口：

```python
from typing import Protocol

from treesearch.rag.models import GraphNodePassage, GraphRelation


class TripletExtractor(Protocol):
    def extract(self, passage: GraphNodePassage) -> list[GraphRelation]:
        ...
```

Rule-based 版本支持简单测试数据：

```text
max_concurrency is defined in TreeSearchConfig
TreeSearchConfig is documented in Runtime Settings
```

抽取成：

```python
GraphRelation(
    relation_id="...",
    subject="max_concurrency",
    predicate="defined_in",
    object="TreeSearchConfig",
    text="max_concurrency defined_in TreeSearchConfig",
    node_id=passage.node_id,
    doc_id=passage.doc_id,
    source_type=passage.source_type,
)
```

后续 LLM 抽取要求：

1. 输出 JSON。
2. 每个 relation 绑定 `node_id`。
3. 不生成跨 node 的虚假 citation。
4. 抽取失败直接抛异常或记录在调用边界，不在内部吞异常。

## 6. Graph Store

文件：`treesearch/rag/graph_store.py`

先实现内存 store，方便单元测试：

```python
class InMemoryGraphStore:
    def __init__(self):
        self.passages: dict[str, GraphNodePassage] = {}
        self.entities: dict[str, GraphEntity] = {}
        self.relations: dict[str, GraphRelation] = {}
        self.node_to_relation_ids: dict[str, set[str]] = {}
        self.entity_to_relation_ids: dict[str, set[str]] = {}
        self.structural_edges: list[StructuralEdge] = []
```

必须提供：

1. `add_passages(passages)`
2. `add_relations(relations)`
3. `add_structural_edges(edges)`
4. `get_passage(node_id)`
5. `get_relations_by_node(node_id)`
6. `get_relations_by_entity(entity_text)`
7. `get_neighbor_node_ids(node_id, edge_types)`
8. `get_relations_by_ids(relation_ids)`

后续再加 SQLiteGraphStore：

1. `rag_passages`
2. `rag_entities`
3. `rag_relations`
4. `rag_entity_relations`
5. `rag_structural_edges`

不要一开始引入 Milvus 作为强依赖。Milvus/vector backend 放 optional extra。

## 7. Graph Builder

文件：`treesearch/rag/graph_builder.py`

职责：

1. 输入 `list[Document]`
2. 转 node passages
3. 抽 triplets
4. 写入 graph store
5. 写 structural edges
6. 返回 stats

接口：

```python
@dataclass(frozen=True)
class GraphBuildStats:
    documents: int
    passages: int
    relations: int
    structural_edges: int


class NodeGraphBuilder:
    def __init__(self, extractor: TripletExtractor, store: InMemoryGraphStore):
        self.extractor = extractor
        self.store = store

    def build(self, documents: list[Document]) -> GraphBuildStats:
        ...
```

测试：

1. 一个文档两个节点能写入两个 passages。
2. relation 的 node_id/doc_id/source_type 正确。
3. stats 正确。
4. rebuild 行为清晰：MVP 可以先 clear 再 build，不做增量。

## 8. TreeSearch Seed Retrieval

文件：`treesearch/rag/seed.py`

职责：复用现有 `search()` 输出 GraphRAG seeds。

接口：

```python
from treesearch.search import search


async def retrieve_seed_nodes(
    query: str,
    documents: list[Document],
    top_k: int = 10,
) -> list[GraphSeed]:
    result = await search(
        query,
        documents,
        top_k_docs=5,
        max_nodes_per_doc=top_k,
        include_ancestors=True,
        merge_strategy="global_score",
    )
    seeds = []
    for node in result.get("flat_nodes", []):
        seeds.append(
            GraphSeed(
                node_id=str(node["node_id"]),
                doc_id=str(node["doc_id"]),
                score=float(node.get("score", 0.0)),
                source="tree" if result.get("mode") == "tree" else "fts5",
            )
        )
    return seeds
```

如果当前 `flat_nodes` 没有 `doc_id`，需要在 search 输出补齐。改造点：

文件：`treesearch/search.py`

要求：

1. `flat_nodes` 中必须包含 `doc_id`
2. `flat_nodes` 中必须包含 `doc_name`
3. `flat_nodes` 中必须包含 `source_type`
4. 保持现有 `documents` 输出兼容

测试：

1. `TreeSearch.search(..., merge_strategy="global_score")` 的 `flat_nodes` 每个节点都有 doc metadata。
2. 不破坏现有 tests。

## 9. Structure-Constrained Expansion

文件：`treesearch/rag/expansion.py`

这是论文最核心方法。

接口：

```python
@dataclass(frozen=True)
class ExpansionConfig:
    max_relations: int = 50
    max_hops: int = 1
    structure_weight: float = 1.0
    seed_weight: float = 1.0
    source_type_weight: float = 0.5
    grounding_weight: float = 0.5


class StructureConstrainedExpander:
    def __init__(self, store: InMemoryGraphStore, config: ExpansionConfig):
        self.store = store
        self.config = config

    def expand(self, query: str, seeds: list[GraphSeed]) -> list[CandidateRelation]:
        ...
```

MVP scoring：

```python
def score_relation(self, relation: GraphRelation, seeds: list[GraphSeed]) -> CandidateRelation:
    seed_by_node = {seed.node_id: seed for seed in seeds}
    sparse_seed_score = seed_by_node.get(relation.node_id).score if relation.node_id in seed_by_node else 0.0
    structure_score = self._structure_score(relation.node_id, seeds)
    source_type_score = self._source_type_score(relation.source_type, query)
    grounding_score = self._grounding_score(relation.node_id)

    return CandidateRelation(
        relation=relation,
        sparse_seed_score=sparse_seed_score,
        structure_score=structure_score,
        source_type_score=source_type_score,
        grounding_score=grounding_score,
    )
```

Structure score：

1. same node: 1.0
2. parent/child: 0.8
3. sibling: 0.6
4. same doc: 0.3
5. otherwise: 0.0

Grounding score：

1. node has source_path + line_start + line_end: 1.0
2. node has source_path but no line: 0.5
3. otherwise: 0.0

Source type score：

1. query contains code-like token and source_type == code: 0.5
2. query contains config-like token and source_type in code/json/yaml/toml: 0.5
3. query is natural language and source_type in markdown/pdf/docx/text: 0.3

不要一开始做复杂 learned model。先用可解释 rule-based scoring，便于消融和论文分析。

测试：

1. same-node relation 分数最高。
2. sibling relation 高于 unrelated relation。
3. 有 line range 的 relation grounding_score 更高。
4. max_relations 生效。

## 10. Evidence Chain Selection

文件：`treesearch/rag/evidence.py`

先定义协议：

```python
class EvidenceSelector(Protocol):
    def select(
        self,
        query: str,
        candidates: list[CandidateRelation],
        store: InMemoryGraphStore,
    ) -> EvidenceChain:
        ...
```

实现两个版本：

1. `HeuristicEvidenceSelector`：用于测试，无 LLM。
2. `LLMEvidenceSelector`：后续接 LLM JSON 输出。

Heuristic 版本：

1. 按 `CandidateRelation.total_score` 排序。
2. 取 top-k relation。
3. 从 relation node 生成 citations。
4. `reasoning_chain` 使用 relation text。
5. `evidence_sufficiency=True` 当 relation 非空。

LLM 版本 prompt 要求：

1. 只返回 JSON。
2. relation id 必须从候选中选择。
3. node id 必须从候选 relation 的 node_id 中选择。
4. citation 必须来自 node metadata。
5. 不允许编造 line range。

## 11. Verifier

文件：`treesearch/rag/verifier.py`

职责：把 LLM 输出拉回确定性约束。

接口：

```python
@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    errors: tuple[str, ...] = ()


class EvidenceVerifier:
    def __init__(self, store: InMemoryGraphStore):
        self.store = store

    def verify(self, chain: EvidenceChain) -> VerificationResult:
        ...
```

检查项：

1. `selected_relation_ids` 都存在。
2. `selected_node_ids` 都存在。
3. 每个 citation 的 node_id 存在。
4. citation 的 line range 与 passage 一致。
5. selected relations 的 node_id 至少有一个在 selected_node_ids 中。
6. 如果有多个 relation，至少共享一个 entity 或位于结构邻近节点。

测试：

1. 正确 chain 通过。
2. 编造 relation id 失败。
3. 编造 node id 失败。
4. citation line range 不匹配失败。
5. 空 evidence_sufficiency 失败。

## 12. Answer Generation

文件：`treesearch/rag/answer.py`

先实现一个简单 answer generator：

```python
@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    answer: str
    evidence_chain: EvidenceChain
    verification: VerificationResult


class TemplateAnswerGenerator:
    def generate(self, query: str, chain: EvidenceChain) -> GroundedAnswer:
        ...
```

MVP 可以不调用 LLM，只把 evidence chain 拼成答案，方便 smoke demo 和 tests。

后续 `LLMAnswerGenerator`：

1. 输入只包含 verified evidence。
2. 输出 answer + citation ids。
3. 不允许引用未 selected 的 evidence。

## 13. Pipeline API

文件：`treesearch/rag/pipeline.py`

用户 API：

```python
class TreeSearchGraphRAG:
    def __init__(
        self,
        tree_search: TreeSearch,
        graph_builder: NodeGraphBuilder,
        expander: StructureConstrainedExpander,
        selector: EvidenceSelector,
        verifier: EvidenceVerifier,
        answer_generator: TemplateAnswerGenerator,
    ):
        self.tree_search = tree_search
        self.graph_builder = graph_builder
        self.expander = expander
        self.selector = selector
        self.verifier = verifier
        self.answer_generator = answer_generator

    def build_graph(self) -> GraphBuildStats:
        ...

    def query(self, query: str) -> GroundedAnswer:
        ...
```

同步 `query()` 内部可以调用 `asyncio.run()`，但要和现有 TreeSearch 的 sync wrapper 风格一致。

Public API 可在 `treesearch/__init__.py` 导出：

```python
from .rag.pipeline import TreeSearchGraphRAG
```

## 14. 对现有代码的必要改动

### 14.1 `treesearch/search.py`

目标：让 GraphRAG seed 能拿到完整 node metadata。

改动：

1. `flat_nodes` 每个节点补 `doc_id`
2. `flat_nodes` 每个节点补 `doc_name`
3. `flat_nodes` 每个节点补 `source_type`
4. `flat_nodes` 每个节点补 `source_path`

风险：

1. 不改变 `documents` 输出结构。
2. 保持已有 benchmark 结果一致。

### 14.2 `treesearch/tree.py`

可选新增：

```python
def get_path_titles(self, node_id: str) -> list[str]:
    ...
```

这不是必须，但能减少 `node_graph.py` 重复逻辑。

不要用 `getattr` 假防御，直接基于已有字段和方法。

### 14.3 `treesearch/fts.py`

MVP 不改。后续如果做 SQLiteGraphStore，可以在同一个 DB 中新增 `rag_*` tables。

建议先不要修改 FTS5Index，避免影响现有检索稳定性。

### 14.4 `pyproject.toml`

MVP 不新增强依赖。

后续 optional extras：

```toml
graphrag = ["openai>=1.0", "tenacity>=8.0"]
vector = ["pymilvus>=2.4"]
```

但第一阶段先用 mock / rule-based extractor。

## 15. 测试计划

新增测试目录：

```text
tests/rag/
  test_node_graph.py
  test_extractors.py
  test_graph_store.py
  test_graph_builder.py
  test_seed.py
  test_expansion.py
  test_evidence.py
  test_verifier.py
  test_pipeline.py
```

### 15.1 Unit tests

必须覆盖：

1. node passage conversion
2. structural edge construction
3. rule-based triplet extraction
4. graph store indexing and lookup
5. TreeSearch seed conversion
6. expansion scoring
7. evidence selection
8. verifier failure cases
9. pipeline query

### 15.2 Integration tests

构造临时 repo：

```text
tmp_repo/
  treesearch/config.py
  docs/config.md
```

内容：

1. `config.py` 定义 `TreeSearchConfig.max_concurrency`
2. `docs/config.md` 解释 `max_concurrency`

query：

> max_concurrency 默认值在哪里定义，文档里怎么解释？

期望：

1. seed 命中 config.py 和 docs/config.md
2. graph expansion 找到两条 relation
3. evidence chain 包含两个 node_id
4. citations 包含两个 source_path
5. verifier 通过

### 15.3 测试命令

每次改实现后运行：

```bash
python ~/.agents/rules/check_ast.py .
pytest tests/rag/ -v --tb=short
pytest tests/ -v --tb=short
```

涉及 OpenAI/LLM 的测试必须 mock API key 和 LLM 响应，不能真实调用 OpenAI。

## 16. Smoke Demo

用户规则要求大块功能改完后补 end-to-end smoke demo，放在 repo root 的 `./tmp/` 下。

建议：

```text
tmp/graphrag_smoke_demo.py
tmp/graphrag_demo_repo/
  treesearch/config.py
  docs/runtime.md
```

demo 做真实用户路径：

1. 创建 demo repo 文件。
2. `TreeSearch` index demo repo。
3. `TreeSearchGraphRAG` build graph。
4. query: “max_concurrency 默认值在哪里定义，文档如何解释？”
5. 打印 grounded answer。
6. 打印 evidence chain。
7. 打印 citations with line ranges。

运行命令：

```bash
python tmp/graphrag_smoke_demo.py
```

期望输出包含：

1. `TreeSearchConfig`
2. `max_concurrency`
3. `docs/runtime.md`
4. `treesearch/config.py`
5. line range
6. verifier ok

## 17. 实验代码改造

新增：

```text
examples/graphrag/
  real_repo_bench.py
  hotpotqa_graphrag.py
  musique_graphrag.py
  twowiki_graphrag.py
  metrics.py
  baselines.py
```

### 17.1 RealRepoBench

数据格式：

```json
{
  "query_id": "rrb_001",
  "query": "max_concurrency 默认值在哪里定义，文档里怎么解释？",
  "query_type": "code_doc",
  "gold_node_ids": ["..."],
  "gold_source_paths": ["treesearch/config.py", "docs/config.md"],
  "gold_answer": "...",
  "requires_cross_source": true,
  "needs_line_grounding": true
}
```

指标：

1. node recall@k
2. source path recall@k
3. citation precision
4. citation recall
5. line accuracy
6. answer correctness
7. task success

### 17.2 Public multi-hop

先接 HotpotQA，因为现有已有 benchmark。后续补：

1. MuSiQue
2. 2WikiMultiHopQA

对 public benchmark，TreeSearch 结构优势未必明显，因此重点指标是：

1. answer quality
2. evidence recall
3. LLM calls
4. latency
5. cost

不要把 public benchmark 作为唯一主结果。论文主角应是真实异构仓库任务。

## 18. 实施阶段

### Phase 1: Graph substrate MVP

目标：

1. 新增 `treesearch/rag/models.py`
2. 新增 `node_graph.py`
3. 新增 `extractors.py`
4. 新增 `graph_store.py`
5. 新增 `graph_builder.py`

完成标准：

1. 可以从 `Document` 构建 node graph。
2. rule-based extractor 能跑通。
3. tests/rag 基础测试通过。

### Phase 2: Seed + expansion

目标：

1. 新增 `seed.py`
2. 修改 `search.py` 的 `flat_nodes` metadata
3. 新增 `expansion.py`

完成标准：

1. query 能产生 GraphSeed。
2. expansion 能按结构分数排序 CandidateRelation。
3. 消融开关可控。

### Phase 3: Evidence + verifier + answer

目标：

1. 新增 `evidence.py`
2. 新增 `verifier.py`
3. 新增 `answer.py`
4. 新增 `pipeline.py`

完成标准：

1. 端到端 query 返回 GroundedAnswer。
2. verifier 能拒绝伪造 citation。
3. smoke demo 可运行。

### Phase 4: Benchmark

目标：

1. RealRepoBench 数据格式
2. metrics
3. baseline runners
4. public multi-hop adapter

完成标准：

1. 能跑 Ours vs TreeSearch-only vs VectorGraph baseline。
2. 能输出论文主表所需指标。

### Phase 5: LLM / vector optional backend

目标：

1. LLMTripletExtractor
2. LLMEvidenceSelector
3. LLMAnswerGenerator
4. optional vector graph store

完成标准：

1. 所有 LLM 测试 mock。
2. demo 可用真实 LLM 开关运行。
3. 默认安装仍然不强依赖 OpenAI/Milvus。

## 19. 最小实现路线

为了尽快产出论文可验证原型，建议最小路线：

1. 不接 Milvus。
2. 不接真实 OpenAI。
3. 使用 in-memory graph store。
4. 使用 rule-based triplet extractor。
5. 使用 heuristic evidence selector。
6. 先在真实 repo 小样本上验证 structure expansion 和 grounding。
7. 再接 LLM extractor/selector。

这样可以先证明 TreeSearch 的结构能力真的能形成 graph substrate 和 grounded evidence chain。

## 20. 完成后的用户 API 示例

理想 API：

```python
from treesearch import TreeSearch
from treesearch.rag import TreeSearchGraphRAG
from treesearch.rag.extractors import RuleBasedTripletExtractor

ts = TreeSearch("src/", "docs/", db_path=None)
ts.index("src/", "docs/")

rag = TreeSearchGraphRAG.from_tree_search(
    ts,
    extractor=RuleBasedTripletExtractor(),
)
rag.build_graph()

answer = rag.query("max_concurrency 默认值在哪里定义，文档里怎么解释？")

print(answer.answer)
for citation in answer.evidence_chain.citations:
    print(citation.source_path, citation.line_start, citation.line_end)
```

输出：

```text
max_concurrency is defined in TreeSearchConfig and documented in Runtime Settings.

treesearch/config.py:12-34
docs/runtime.md:42-58
verification: ok
```

## 21. 不建议做的事

第一版不要做：

1. 不要引入 Neo4j。
2. 不要强依赖 Milvus。
3. 不要把所有 LLM 调用写死成 OpenAI。
4. 不要把 GraphRAG 逻辑塞进 `search.py`。
5. 不要改坏现有 TreeSearch API。
6. 不要为了兼容旧未发布 GraphRAG 设计写复杂 fallback。
7. 不要过度 try/except 吞掉 triplet extraction 或 verifier 错误。

## 22. 论文实验对应代码开关

为了支持 ablation，代码中要保留显式配置：

```python
@dataclass(frozen=True)
class GraphRAGConfig:
    use_treesearch_seed: bool = True
    use_structure_proximity: bool = True
    use_source_type_prior: bool = True
    use_grounding_score: bool = True
    use_verifier: bool = True
    max_seed_nodes: int = 10
    max_candidate_relations: int = 50
```

对应消融：

| Ablation | Config |
|---|---|
| no TreeSearch seed | `use_treesearch_seed=False` |
| no structure proximity | `use_structure_proximity=False` |
| no source type prior | `use_source_type_prior=False` |
| no grounding score | `use_grounding_score=False` |
| no verifier | `use_verifier=False` |

## 23. 交付标准

MVP 完成标准：

1. `tests/rag/` 全部通过。
2. `pytest tests/ -v --tb=short` 通过。
3. `python ~/.agents/rules/check_ast.py .` 通过。
4. `tmp/graphrag_smoke_demo.py` 能跑通。
5. demo 输出 grounded answer、evidence chain、citation line ranges。
6. 至少有一个小型 RealRepoBench runner。

论文原型完成标准：

1. 至少 100 条 RealRepoBench query。
2. 至少 3 个 baseline。
3. 至少 5 个消融。
4. 输出 citation precision / line accuracy。
5. 输出 LLM calls / latency / cost。
6. 能生成 `benchmark_results/graphrag/*.json`。

## 24. 最终建议

代码改造要服务论文核心，不要泛化成一个大而全 GraphRAG 框架。

优先实现：

1. TreeSearch node graph
2. structure-constrained expansion
3. verifiable evidence chain
4. line-level citation
5. RealRepoBench

这五个模块直接对应论文最强 novelty：

> structure-preserving graph substrate + TreeSearch-guided expansion constraint + single-pass verifiable evidence chain selection.

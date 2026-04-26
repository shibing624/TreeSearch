# TreeSearch Paper V2: Structure-Preserving Graph RAG

## 0. 这份文档是什么

这份文档是基于 `docs/new_idea.md` 重新整理后的论文指导版本。它替代 `docs/paper_zh.md` 里偏 `dense + TreeSearch hybrid retrieval` 的旧主线，把 TreeSearch 的论文定位升级为：

> 面向异构代码-文档仓库的结构保持型 GraphRAG 系统。

新的核心判断是：

1. 不建议单独写 `single-pass CoT rerank`，它容易被认为是 prompt engineering。
2. 不建议只写 `TreeSearch + dense retrieval`，它容易被认为是工程拼装。
3. 最推荐写 `TreeSearch-Guided Structure-Preserving Graph RAG`：TreeSearch 提供结构节点、精确 seed、结构约束和 grounding，GraphRAG 提供跨节点、跨文档、多跳推理。

这篇论文应该回答的问题是：

> 在代码、Markdown、配置、PDF、JSON 等异构仓库中，如何构建一个既能多跳推理、又不破坏原始结构、还能输出可验证引用位置的 RAG 系统？

## 1. 推荐论文标题

首选：

**TreeSearch-Guided Graph RAG with Structure-Preserving Grounding**

备选：

1. **Structure-Preserving Hybrid Graph RAG for Heterogeneous Repositories**
2. **Structure-Constrained Graph RAG for Code and Document Repositories**
3. **Grounded Graph RAG over Tree-Structured Repository Nodes**

如果投 Industry Track，标题可以更系统化：

**A Structure-Preserving Graph RAG System for Heterogeneous Enterprise Repositories**

## 2. 一句话定位

英文版：

> We present a structure-preserving Graph RAG system for heterogeneous repositories, where TreeSearch converts files into repository-native structural nodes, guides graph expansion with sparse and structural signals, and grounds generated answers back to exact sections, paths, and line ranges.

中文版：

> 我们提出一个面向异构仓库的结构保持型 GraphRAG 系统：TreeSearch 将代码、文档和配置解析为可 grounding 的结构节点，并用这些节点指导图构建、子图扩展和证据链选择，最终生成带有精确路径和行号引用的答案。

## 3. 论文主张

旧主张是：

> TreeSearch 是 dense retrieval 的结构化 sparse complement。

新主张应改成：

> TreeSearch is a structural substrate for GraphRAG.

也就是说，TreeSearch 不只是另一路 sparse retrieval，而是提供 GraphRAG 的基础表示和约束：

1. 定义 graph passage 的基本单位：不是 arbitrary chunk，而是 repository-native structural node。
2. 定义 graph seed：用 FTS5/Grep/tree search 找高精度结构节点，再从节点进入实体关系图。
3. 定义 expansion constraint：用 parent/child/sibling、section path、line proximity、source_type 约束子图扩展。
4. 定义 final grounding：最终答案引用 `doc path + section path + node_id + line range + relation chain`。

## 4. 核心创新点排序

论文主贡献建议压成三条，不要把所有想法都并列讲。

### Contribution 1: Structure-Preserving Graph Substrate

已有 GraphRAG 通常从 chunks 或 passages 抽取实体和关系。这对普通百科 QA 可以接受，但对异构仓库有明显问题：

1. 代码函数、类和方法边界会被 chunk 切断。
2. Markdown heading hierarchy 会丢失。
3. JSON/config 的 key hierarchy 会丢失。
4. PDF/DOCX 的章节定位很难映射回原文。
5. answer citation 难以落到可执行的位置。

TreeSearch 已经具备稳定结构节点：

1. `Document`
2. `node_id`
3. `source_type`
4. title / summary / text
5. line_start / line_end
6. parent / child / sibling
7. path-to-root

论文贡献应写成：

> We construct the knowledge graph over TreeSearch nodes rather than arbitrary chunks, preserving repository-native boundaries such as sections, functions, classes, config subtrees, and line ranges.

这是 TreeSearch 和 LightRAG / HippoRAG / Microsoft GraphRAG 拉开差异的根基。

### Contribution 2: TreeSearch-Guided Graph Expansion

这是最像方法创新的部分，优先级最高。

传统 GraphRAG 的 expansion 主要依赖：

1. entity-relation edges
2. vector similarity
3. graph traversal
4. PPR / random walk
5. iterative retrieval

TreeSearch 可以引入结构约束：

1. sparse seed score：FTS5/Grep/tree search 命中的节点更可信。
2. structure proximity：同 section、父子节点、兄弟节点更可信。
3. source proximity：同文件、同模块、同配置树更可信。
4. source-type prior：code/doc/config/query 类型匹配时加权。
5. grounding confidence：有 line range、stable node_id、明确 path 的证据优先。

可以将子图候选关系评分写成：

```text
score(r | q) =
  alpha * semantic_score(q, r)
+ beta  * sparse_seed_score(q, node(r))
+ gamma * structure_proximity(seed_node, node(r))
+ delta * source_type_prior(q, node(r))
+ eta   * grounding_confidence(node(r))
```

主张：

> TreeSearch does not merely retrieve extra documents; it constrains graph expansion to reduce noisy multi-hop context.

这个贡献最容易通过消融验证：

1. no TreeSearch seed
2. no structure proximity
3. no source_type prior
4. no grounding confidence
5. vector-only expansion

### Contribution 3: Single-Pass Verifiable Evidence Chain Selection

不要写成 “single-pass CoT rerank”。更好的术语是：

> single-pass verifiable evidence chain selection

输入：

1. query
2. TreeSearch seed nodes
3. expanded candidate relations
4. structural metadata
5. relation-to-node links

输出：

1. bridge entities
2. selected relations
3. reasoning chain
4. supporting node IDs
5. citation line ranges
6. evidence sufficiency flag

它的作用不是简单排序，而是在一次结构化 LLM 调用中完成：

1. bridge entity identification
2. evidence compression
3. relation-chain selection
4. citation grounding
5. sufficiency checking

示例输出：

```json
{
  "question_type": "cross-source 2-hop",
  "bridge_entities": ["TreeSearchConfig", "max_concurrency"],
  "selected_relations": [
    "max_concurrency defined_in TreeSearchConfig",
    "TreeSearchConfig documented_in Runtime Settings"
  ],
  "supporting_node_ids": ["config_py_abc123", "docs_runtime_def456"],
  "citations": [
    {"path": "treesearch/config.py", "line_start": 12, "line_end": 34},
    {"path": "docs/config.md", "line_start": 42, "line_end": 58}
  ],
  "evidence_sufficiency": true
}
```

这个模块要配一个 verifier：

1. selected relation 是否存在于图中。
2. relation 是否链接到真实 TreeSearch node。
3. node_id 是否存在。
4. line range 是否存在。
5. relation chain 是否连通或共享 bridge entity。
6. answer 引用是否只来自 selected evidence。

## 5. 非核心但保留的系统点

下面两个点不要作为主贡献，但可以放到 System 或 Analysis。

### Sparse-to-Graph Seeding

这是 Contribution 2 的一部分。它实用，但单独 novelty 不够，因为 reviewer 可能认为这只是 BM25 seed + graph expansion。

写法：

> We use TreeSearch as a high-precision seed selector for symbol-heavy and structure-heavy queries.

### Adaptive Hybrid RAG Policy

这是工业系统价值，但不建议作为主创新。除非做 learned router 或 cost-quality Pareto optimization，否则 rule-based policy 容易被认为工程策略。

写法：

> A lightweight policy avoids expensive GraphRAG computation for simple symbol or location queries, and triggers graph expansion only for cross-node or cross-source questions.

## 6. 论文系统架构

图 1 建议画成：

```text
Heterogeneous Repository
  -> TreeSearch Parser
      -> structural nodes
      -> node_id / path / source_type / line range
  -> Node-Level Graph Builder
      -> entities
      -> relations
      -> node-linked passages
      -> structural edges
  -> Dual Index
      -> SQLite FTS5 / Grep
      -> vector entity-relation index

Query
  -> Query Analyzer
  -> TreeSearch Seed Retrieval
  -> Structure-Constrained Graph Expansion
  -> Single-Pass Evidence Chain Selection
  -> Verifier
  -> Grounded Answer
```

图中必须突出：

1. `chunk` 被替换为 `TreeSearch node`。
2. Graph expansion 不是纯 graph/vector，而是受 TreeSearch 结构约束。
3. 最终答案有 verifiable evidence chain 和 line-level citation。

## 7. 方法章节建议

### 7.1 Problem Setting

定义异构仓库：

```text
R = {files}
file types = code, markdown, json, yaml, pdf, docx, html, text
```

每个文件被 TreeSearch 解析为 tree：

```text
D = (V_T, E_T)
```

其中 `V_T` 是结构节点，`E_T` 是 parent-child / sibling / path relation。

目标：

给定 query `q`，返回：

1. answer `a`
2. evidence chain `C`
3. grounded citations `G`

其中 `G` 必须映射到真实 `source_path + line range`。

### 7.2 Structure-Preserving Graph Construction

从每个 TreeSearch node 抽取：

1. entities
2. relations
3. node passage
4. structural edges

Graph schema：

```text
Entity(id, text, node_ids)
Relation(id, subject, predicate, object, node_id, source_type)
NodePassage(node_id, doc_id, title, text, path, line_start, line_end)
StructuralEdge(src_node_id, dst_node_id, edge_type)
```

边类型：

1. entity-relation
2. relation-node
3. node-parent
4. node-child
5. node-sibling
6. node-same-file
7. node-cross-source

### 7.3 TreeSearch-Guided Seed Retrieval

先跑 TreeSearch：

```text
S = TreeSearch(q)
```

得到 top structural nodes。再从这些 nodes 找 graph seeds：

```text
E_seed = entities linked to S
R_seed = relations linked to S
```

这样可以处理：

1. symbol query
2. config query
3. path query
4. code API query
5. section title query

### 7.4 Structure-Constrained Expansion

从 seeds 开始扩展：

```text
G_q = Expand(E_seed, R_seed, budget=B)
```

Expansion 不只看 entity-relation 邻接，还看 TreeSearch 结构：

```text
priority(candidate) =
  semantic_similarity
+ sparse_seed_score
+ tree_distance_score
+ source_type_score
+ grounding_score
```

其中 `tree_distance_score` 可由下面信号组成：

1. same node
2. same parent
3. ancestor-descendant
4. sibling
5. same source path
6. line distance bucket

### 7.5 Evidence Chain Selection

把候选关系和节点 metadata 输入 LLM：

```text
SelectEvidence(q, candidate_relations, node_metadata) -> EvidenceChain
```

LLM 输出必须是 JSON，不要自由文本：

1. `bridge_entities`
2. `selected_relation_ids`
3. `selected_node_ids`
4. `reasoning_chain`
5. `citation_spans`
6. `evidence_sufficiency`

### 7.6 Verification and Answering

Verifier 做确定性检查：

1. relation id 是否存在。
2. node id 是否存在。
3. citation line range 是否存在。
4. relation chain 是否连通。
5. selected evidence 是否覆盖 answer 所需实体。

通过 verifier 后再生成答案。生成 prompt 中只提供 selected evidence，不提供整个 expanded subgraph。

## 8. 实验设计

### 8.1 主实验问题

论文必须回答 5 个问题：

1. 结构节点 graph 是否优于 chunk graph？
2. TreeSearch-guided expansion 是否优于 vector-only / PPR expansion？
3. 单次 evidence chain selection 是否能接近或超过 iterative retrieval？
4. line-level grounded answer 是否比普通 passage citation 更可靠？
5. 在质量相当时，系统是否降低 LLM calls、latency 和成本？

### 8.2 Baselines

必须比较：

1. Dense RAG
2. BM25 / FTS5 RAG
3. Dense + BM25 Hybrid RAG
4. Vector Graph RAG
5. HippoRAG / HippoRAG 2
6. LightRAG
7. IRCoT
8. RankGPT / listwise rerank over retrieved passages
9. TreeSearch-only
10. TreeSearch-guided GraphRAG

如果实现所有 baseline 工作量太大，最小可投版本至少包括：

1. Dense RAG
2. FTS5 / TreeSearch RAG
3. Vector Graph RAG
4. LightRAG 或 HippoRAG 之一
5. IRCoT
6. Ours

### 8.3 Datasets

公开 benchmark：

1. HotpotQA
2. MuSiQue
3. 2WikiMultiHopQA
4. QASPER
5. FinanceBench
6. CodeSearchNet

真实异构仓库 benchmark：

`RealRepoBench` 或 `HeteroRepoBench`，至少 100-300 条 query。

任务类型：

1. code locate
2. config lookup
3. doc QA
4. code-doc consistency
5. troubleshooting
6. cross-source multi-hop
7. implementation tracing

典型 query：

1. “`search_mode=auto` 的路由逻辑在哪里，README 是怎么描述的？”
2. “`max_concurrency` 默认值在哪里定义，文档里有没有解释？”
3. “CLI 参数 `--regex` 从入口到搜索管线经过哪些函数？”
4. “JSON 文件解析出的 node 如何被 FTS5 建索引？”
5. “tree mode 的 parent boost 对哪些 benchmark 有帮助？”

### 8.4 Metrics

Retrieval：

1. MRR
2. Recall@K
3. Hit@K
4. NDCG@K
5. supporting evidence recall

Answer：

1. EM / F1
2. LLM-judge correctness
3. answer faithfulness
4. hallucination rate

Evidence：

1. citation precision
2. citation recall
3. line-level citation accuracy
4. evidence chain validity
5. bridge entity coverage

System：

1. LLM calls/query
2. tokens/query
3. P50 / P95 latency
4. cost per 1k queries
5. index build time
6. incremental update time
7. index size

### 8.5 Ablations

最小消融：

1. `Ours`
2. `Ours w/o TreeSearch seed`
3. `Ours w/o structure proximity`
4. `Ours w/o structural edges`
5. `Ours w/o evidence verifier`
6. `Ours w/o single-pass chain selection`
7. `Chunk GraphRAG` vs `TreeSearch-node GraphRAG`

扩展消融：

1. expansion degree 0/1/2
2. relation budget 20/50/100/200
3. LLM model: GPT-4o-mini / GPT-4o / Qwen / Llama
4. no CoT vs CoT vs structured JSON chain
5. answer with full subgraph vs selected evidence only

## 9. 主结果表设计

### 表 1：Multi-hop QA 主结果

| Method | HotpotQA F1 | MuSiQue F1 | 2Wiki F1 | Evidence Recall@5 | LLM Calls | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Dense RAG | | | | | | |
| IRCoT | | | | | | |
| HippoRAG | | | | | | |
| LightRAG | | | | | | |
| Vector Graph RAG | | | | | | |
| Ours | | | | | | |

### 表 2：真实异构仓库任务

| Method | Task Success@3 | Citation Precision | Line Accuracy | Time-to-Answer | LLM Calls |
|---|---:|---:|---:|---:|---:|
| Dense RAG | | | | | |
| TreeSearch RAG | | | | | |
| Vector Graph RAG | | | | | |
| Ours | | | | | |

### 表 3：消融

| Variant | Answer F1 | Evidence Precision | Chain Validity | Cost |
|---|---:|---:|---:|---:|
| Ours | | | | |
| w/o TreeSearch seed | | | | |
| w/o structure proximity | | | | |
| w/o verifier | | | | |
| chunk graph | | | | |

### 表 4：RealRepoBench-mini pilot 结果

当前代码已经可以生成一个 30-query `RealRepoBench-mini` pilot。这个 pilot 只用于验证实验链路和表格生成，不作为论文最终主结果；正式论文需要扩展到 100-300 条跨 repo 标注样本。

运行命令：

```bash
python examples/graphrag/real_repo_bench.py \
  --paths examples/graphrag/fixtures/repo \
  --queries examples/graphrag/fixtures/queries.json \
  --triplets examples/graphrag/fixtures/triplets.json \
  --baseline all \
  --graph-store sqlite \
  --graph-store-path tmp/graphrag_mini_graph.db \
  --output tmp/graphrag_mini_all_results.json \
  --markdown-output tmp/graphrag_mini_all_results.md \
  --latex-output tmp/graphrag_mini_all_results.tex
```

Pilot 表：

| Method | Count | Node Recall | Source Recall | Citation Precision | Citation Recall | Line Accuracy | Task Success | Avg Latency | LLM Calls/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TreeSearch-guided GraphRAG | 30 | 1.000 | 1.000 | 0.633 | 1.000 | 1.000 | 1.000 | 0.015s | 0.000 |
| TreeSearch RAG | 30 | 1.000 | 1.000 | 0.661 | 1.000 | 1.000 | 1.000 | 0.013s | 0.000 |
| FTS5 RAG | 30 | 1.000 | 1.000 | 0.661 | 1.000 | 1.000 | 1.000 | 0.014s | 0.000 |
| Dense lexical proxy | 30 | 1.000 | 1.000 | 0.444 | 1.000 | 1.000 | 1.000 | <0.001s | 0.000 |
| Hybrid lexical proxy | 30 | 1.000 | 1.000 | 0.444 | 1.000 | 1.000 | 1.000 | 0.013s | 0.000 |
| GraphRAG + real LLM selector/answer | 30 | 0.967 | 0.967 | 1.000 | 0.967 | 1.000 | 0.933 | 3.629s | 2.000 |

Pilot 观察：

1. 30 条样本目前过于简单，所有非 LLM 方法 task success 都接近满分，不能用于论文 claim。
2. LLM selector/answer 的 citation precision 更高，但 task success 略低，说明 evidence selection prompt 需要在正式实验前调优。
3. 当前 `dense` 和 `hybrid` 是无外部依赖的 lexical proxy，只用于 harness smoke；正式论文必须替换为 OpenAI/BGE embedding + FAISS 或 SQLite vector cache。
4. 这个 pilot 已验证 JSON、Markdown、LaTeX 表格链路、SQLite graph store、真实 LLM API、line grounding metrics 和 baseline adapter 能跑通。

### 表 5：正式实验执行矩阵

为了达到可投稿标准，最终实验必须分三层：

| Layer | Purpose | Dataset | Methods | Required Output |
|---|---|---|---|---|
| RealRepoBench-main | 主结果，验证异构仓库结构与 grounding | 100-300 queries, 3-5 repos | Dense RAG, FTS5 RAG, Hybrid RAG, TreeSearch RAG, Vector Graph RAG, Ours | Table 2, per-type breakdown, case studies |
| Public multi-hop | 对齐 GraphRAG / IRCoT / LightRAG 文献 | HotpotQA, MuSiQue, 2Wiki | Dense RAG, IRCoT, LightRAG/HippoRAG, Vector Graph RAG, Ours | Table 1, evidence recall, cost |
| Ablation suite | 证明核心创新点有效 | RealRepoBench-main + selected public split | Ours variants | Table 3, sensitivity plots |

正式 RealRepoBench-main 标注协议：

1. 每条 query 必须有 `gold_node_ids`、`gold_source_paths`、`gold_line_ranges`、`gold_answer`。
2. 每条 query 标注 `query_type`、`difficulty`、`requires_cross_source`、`needs_line_grounding`。
3. 每个 repo 至少覆盖 code、docs、config、CLI/API、troubleshooting 五类问题。
4. 至少 40% query 必须跨两个以上 source type。
5. 至少 50% query 必须要求 line-level citation。

正式 baseline 替换要求：

1. `dense`：使用真实 embedding baseline，推荐 `text-embedding-3-small` 或 BGE-M3，缓存到 SQLite/FAISS。
2. `hybrid`：FTS5/TreeSearch seed + dense rerank，不能再使用 lexical proxy。
3. `Vector Graph RAG`：复用 `/Users/xuming/Documents/Codes/vector-graph-rag`，固定同一 LLM/embedding 配置。
4. `IRCoT`：实现 2-3 step iterative retrieval，记录 LLM calls 和 latency。
5. `LightRAG/HippoRAG`：至少接入一个公开实现作为 external baseline。

正式论文 claim 只有在上述实验完成后才能写成 strong claim；当前 pilot 只能写成 implementation validation。

## 10. 6 页正文结构

### Page 1

1. Title
2. Abstract
3. Introduction
4. Problem motivation: GraphRAG over heterogeneous repositories breaks structure and grounding

### Page 2

1. Related Work
2. Problem Formulation
3. System Overview figure

### Page 3

1. Structure-Preserving Graph Construction
2. TreeSearch-guided Seed Retrieval
3. Structure-Constrained Expansion

### Page 4

1. Single-Pass Evidence Chain Selection
2. Verification and Grounded Answering
3. Complexity / cost discussion

### Page 5

1. Experimental Setup
2. Baselines
3. Datasets
4. Main Results

### Page 6

1. Ablations
2. Error Analysis
3. Deployment / cost analysis
4. Conclusion

Appendix：

1. prompts
2. verifier rules
3. benchmark annotation protocol
4. full tables
5. examples
6. failure cases

## 11. 摘要草稿

英文摘要初稿：

> Graph-based retrieval-augmented generation improves multi-hop reasoning by connecting entities and relations across documents. However, existing GraphRAG systems typically construct graphs over arbitrary chunks or passages, which breaks repository-native structure such as code functions, configuration subtrees, document sections, and line-level provenance. This makes graph expansion noisy and final answers difficult to ground. We present TreeSearch-Guided Graph RAG, a structure-preserving retrieval system for heterogeneous repositories. TreeSearch parses code, documents, and structured files into stable structural nodes with source types, paths, parent-child relations, and line ranges. We construct a knowledge graph over these nodes, use sparse and structural signals to guide subgraph expansion, and employ a single structured LLM call to select verifiable multi-hop evidence chains. A deterministic verifier checks relation connectivity and citation validity before answer generation. Experiments on multi-hop QA benchmarks and real repository tasks evaluate retrieval quality, answer correctness, citation precision, latency, and cost. Our results show that structure-preserving graph construction and TreeSearch-guided expansion improve grounded evidence quality while reducing unnecessary iterative retrieval.

## 12. 审稿风险与应对

### 风险 1：这是不是又一个 GraphRAG 工程系统？

应对：

1. 强调 graph substrate 从 arbitrary chunks 改为 TreeSearch structural nodes。
2. 用 chunk graph vs node graph 的消融证明差异。
3. 用 line-level citation accuracy 证明普通 GraphRAG 做不到。

### 风险 2：TreeSearch-guided expansion 是否只是 BM25 seed？

应对：

1. 不只用 seed，还用 structure proximity、source_type prior、line grounding confidence。
2. 做 `w/o structure proximity` 消融。
3. 展示 code/config/doc 混合任务中 BM25 seed 只能找点，结构扩展才能连链。

### 风险 3：single-pass evidence selection 是 prompt engineering？

应对：

1. 输出结构化 evidence chain，不是自由文本 CoT。
2. 加 deterministic verifier。
3. 与 IRCoT、RankGPT、listwise rerank 比较。
4. 报告 chain validity 和 citation precision。

### 风险 4：公开 multi-hop benchmark 不能体现 TreeSearch 优势

应对：

1. 公开 benchmark 只作为通用能力验证。
2. 主角放在 RealRepoBench / HeteroRepoBench。
3. 公开 benchmark 上强调质量-成本 tradeoff，真实仓库上强调 grounding 和 task success。

## 13. 投稿定位和采纳率判断

现实判断：

1. 如果只实现 single-pass rerank：workshop 或弱 Findings。
2. 如果实现 TreeSearch + GraphRAG 但无强消融：Industry Track / Findings 中等概率。
3. 如果完整实现 structure-preserving node graph、structure-constrained expansion、verifiable evidence chain，并补真实仓库任务：Findings / Industry Track 强稿，main 有机会。

预估：

| 版本 | ACL/EMNLP Main | Findings | Industry Track | Workshop |
|---|---:|---:|---:|---:|
| single-pass rerank only | 10%-20% | 25%-35% | 30%-45% | 50%+ |
| TreeSearch + GraphRAG pipeline | 15%-25% | 35%-45% | 45%-60% | 60%+ |
| full structure-preserving GraphRAG | 25%-40% | 50%-65% | 55%-70% | 75%+ |
| full system + strong experiments | 40%-55% | 65%-80% | 70%-85% | 85%+ |

## 14. 最终执行建议

优先实现和验证的顺序：

1. Node-level graph schema：先把 TreeSearch nodes 变成 graph passages。
2. Triplet extraction with node links：每个 relation 必须能回到 node_id。
3. TreeSearch seed retrieval：用现有 search 输出 top nodes。
4. Structure-constrained expansion：先做 rule-based scoring。
5. Evidence chain selector：结构化 JSON 输出。
6. Verifier：确定性检查 relation/node/citation。
7. RealRepoBench：构造 100-300 条真实仓库任务。
8. Public benchmark：补 HotpotQA / MuSiQue / 2Wiki。

最小能写论文的系统必须包含：

1. TreeSearch-node graph construction
2. TreeSearch-guided expansion
3. evidence chain selection
4. citation verifier
5. chunk graph vs node graph 消融
6. real repository benchmark

如果缺少第 5 和第 6，论文仍容易退化成系统想法。

# TreeSearch 论文新方向：Structure-Preserving Hybrid Graph RAG

## 1. 总体判断

当前 `docs/paper_zh.md` 的方向是对的：不要把 TreeSearch 写成 dense retrieval 的替代品，而是写成异构仓库中的 structure-aware sparse channel、routing channel 和 grounding channel。

但如果要写成更强的论文，现在还不够。当前方案更像一个 hybrid retrieval system blueprint，方法创新主要集中在 TreeSearch + dense retrieval + routing/fusion。这个主张稳，但容易被 reviewer 认为是工程组合，缺少更强的算法机制和端到端 RAG 效果闭环。

`vector-graph-rag` 可以补上的关键能力是：GraphRAG、多跳检索、实体关系图、subgraph expansion、LLM rerank、answer generation 和公开 multi-hop benchmark 评测。更重要的是，它可以帮助 TreeSearch 从“结构化 sparse 检索工具”升级成“结构保持型 Hybrid Graph RAG 系统”。

推荐的新论文主线：

> Structure-Preserving Hybrid Graph Retrieval for Heterogeneous Repositories

核心一句话：

> TreeSearch preserves document structure and provides precise sparse grounding, while graph-based vector retrieval provides semantic multi-hop expansion. Together, they form a low-cost, interpretable, and structure-preserving RAG system for heterogeneous repositories.

## 2. 当前论文草稿的不足

### 2.1 技术创新偏系统集成

当前草稿中的主要系统是：

1. dense retrieval channel
2. TreeSearch retrieval channel
3. adaptive routing
4. fusion / rerank
5. grounding output

这个方向适合 industry track，但如果没有更强的方法模块，容易被质疑为“把 BM25/FTS5 和 embedding 拼起来”。

### 2.2 RAG 闭环不足

当前重点仍然是 retrieval quality、latency、cost 和 grounding。论文还缺少完整 RAG 闭环：

1. query understanding
2. evidence retrieval
3. evidence chain construction
4. reranking
5. answer generation
6. citation / grounding verification

TreeSearch 现在能返回 node、path、line range，但还没有把这些结果组织成可解释的答案生成证据链。

### 2.3 多跳能力不足

TreeSearch 擅长：

1. symbol / config / path 查询
2. section 定位
3. 结构化文档检索
4. line-level grounding
5. 低成本增量索引

但 TreeSearch 本身不是 GraphRAG。对于下面这类 query，单纯 tree/sparse 检索还不够：

1. “这个参数默认值在哪里定义，文档里如何解释？”
2. “某个 API 的行为在哪个 issue、设计文档和代码里同时出现？”
3. “A 模块依赖的配置项最终影响了哪个功能？”
4. “论文中的方法、实验、数据集和结论之间是什么关系？”

这些需要跨节点、跨文档、跨实体的 multi-hop retrieval。

### 2.4 实验证据还不够完整

当前已有 QASPER、FinanceBench、CodeSearchNet、HotpotQA 等 benchmark 结果，但如果要支撑论文，需要补：

1. 更强 dense / hybrid / GraphRAG baseline
2. multi-hop benchmark
3. end-to-end RAG answer 指标
4. citation / grounding quality
5.真实异构仓库任务集
6. TreeSearch 在 hybrid / GraphRAG 中的边际贡献分析

## 3. vector-graph-rag 能带来的关键增强

`vector-graph-rag` 的核心流程是：

```text
Documents
  -> LLM triplet extraction
  -> Entities + Relations + Passages
  -> Embedding into Milvus
  -> Query entity extraction
  -> Entity / relation vector search
  -> Subgraph expansion
  -> Single-pass LLM rerank
  -> Answer generation
```

它对 TreeSearch 的增强主要有三点。

### 3.1 补 GraphRAG 和 multi-hop retrieval

TreeSearch 是 tree-structured retrieval，`vector-graph-rag` 是 entity-relation graph retrieval。两者结合后，可以形成：

> tree-structured sparse retrieval + vectorized knowledge graph expansion

这能把论文从 hybrid retrieval 升级成 hybrid Graph RAG。

### 3.2 补端到端 RAG 效果

TreeSearch 当前更像 retrieval library。GraphRAG 能补上：

1. evidence chain
2. relation-level reranking
3. answer generation
4. retrieved evidence to final answer 的闭环
5. multi-hop QA benchmark

### 3.3 补更强的论文 novelty

如果只是把 dense retrieval 接入 TreeSearch，创新点不够强。更好的方向是：

> 用 TreeSearch 的结构节点作为 GraphRAG 的基本 passage 单位，用 TreeSearch 的 path / node_id / line range 约束 graph construction、subgraph expansion 和 answer grounding。

这不是简单“加一个向量库”，而是结构保持型 GraphRAG。

## 4. 最值得写成论文贡献的创新点

### 4.1 Structure-Preserving Graph Construction

传统 RAG 和 GraphRAG 往往从 fixed-size chunks 抽取 triplets。这个做法会破坏文档结构：

1. section 边界丢失
2. heading hierarchy 丢失
3. code function / class 边界丢失
4. JSON / config 层级丢失
5. passage 与原文件位置难以精确对齐

TreeSearch 可以提供不同的 graph construction 单位：

1. Markdown section node
2. code function / class node
3. JSON subtree node
4. PDF section / page node
5. DOCX heading node
6. config key subtree

每个 node 都有：

1. stable `node_id`
2. `source_type`
3. title
4. path
5. parent / child / sibling relation
6. line range
7. original text

因此可以提出：

> Instead of extracting a knowledge graph from arbitrary chunks, we construct a structure-preserving graph over TreeSearch nodes.

这是最重要的技术创新之一。

### 4.2 Sparse-to-Graph Seeding

`vector-graph-rag` 默认用 query entity extraction + vector search 找 seed entities / relations。对于代码仓库和异构技术文档，这不一定最稳，因为很多 query 是 symbol-heavy 或 config-heavy：

1. `OPENAI_API_KEY`
2. `TreeSearchConfig`
3. `search_mode`
4. `auth.middleware.ts`
5. `max_concurrency`
6. `BM25`

这类 query 用 TreeSearch 的 FTS5 / Grep 更准。

可以提出：

> TreeSearch first retrieves high-precision structural nodes, then uses entities and relations attached to those nodes as seeds for graph expansion.

流程：

```text
Query
  -> TreeSearch sparse / grep / structure retrieval
  -> top structural nodes
  -> node-linked entities and relations
  -> subgraph expansion
  -> relation rerank
  -> grounded answer
```

这个创新点特别适合解释为什么 TreeSearch 不是普通 BM25 baseline，而是 GraphRAG 的 seed selector 和 grounding layer。

### 4.3 Tree-Constrained Subgraph Expansion

普通 GraphRAG 的 expansion 主要按 entity-relation edge 扩展，容易引入噪声。TreeSearch 可以加入结构约束：

1. 同一 section 内的关系优先
2. parent / child / sibling section 优先
3. 同一 source file / module 优先
4. line distance 近的关系优先
5. source_type 匹配的关系优先
6. code-doc-config 跨源关系作为显式 bridge edge

可以把 subgraph scoring 写成：

```text
score = semantic_score
      + sparse_seed_score
      + structure_proximity
      + source_type_prior
      + grounding_confidence
```

对应论文主张：

> TreeSearch does not merely add sparse recall; it constrains graph expansion using document structure.

这能显著增强方法创新。

### 4.4 Adaptive Hybrid RAG Policy

不是所有 query 都应该跑 GraphRAG。可以设计 query-aware policy：

| Query 类型 | 推荐路径 |
|---|---|
| symbol / path / config lookup | TreeSearch-only |
| natural language semantic query | dense / vector channel |
| hierarchical document query | TreeSearch tree mode |
| cross-document multi-hop query | TreeSearch-guided GraphRAG |
| answer generation query | GraphRAG + grounded answer |
| IDE jump / debugging query | TreeSearch + line grounding |

这样论文可以从“所有 query 统一重检索”升级为：

> adaptive retrieval-time computation

即简单 query 不浪费 LLM 和 vector graph，复杂 query 才触发 GraphRAG。

### 4.5 Grounded Graph RAG Answering

普通 GraphRAG 通常只能给 passage 或 relation。TreeSearch 可以让最终答案引用：

1. document path
2. section title
3. section path
4. stable node_id
5. line range
6. relation chain
7. source_type

最终输出可以是：

```text
Answer:
  ...

Evidence:
  1. docs/config.md > Runtime Settings > max_concurrency, lines 42-58
  2. treesearch/config.py > TreeSearchConfig, lines 10-35
  3. Relation chain:
     max_concurrency -> defined_in -> TreeSearchConfig
     TreeSearchConfig -> documented_in -> Runtime Settings
```

这会让论文在 RAG trustworthiness / citation / agent usability 上更强。

## 5. 可以形成的新系统架构

推荐系统图：

```text
Heterogeneous Repository
  -> TreeSearch Parser
      -> structure nodes
      -> node_id / path / line range / source_type
  -> Structure-Preserving Graph Builder
      -> entities
      -> relations
      -> node-linked passages
      -> structural edges
  -> Dual Index
      -> SQLite FTS5 / Grep / TreeSearch
      -> Vector entity-relation-passage index

Query
  -> Query Analyzer
  -> Adaptive Policy
      -> TreeSearch-only
      -> Dense-only
      -> Hybrid
      -> TreeSearch-guided GraphRAG
  -> Structure-Constrained Subgraph Expansion
  -> Single-pass Rerank
  -> Grounded Answer
```

## 6. 推荐的新论文贡献

可以把 contribution 改成三条：

### Contribution 1

提出一种面向异构仓库的 structure-preserving graph construction 方法。系统不从固定 chunk 构图，而是从 TreeSearch 解析出的结构节点构建 entity-relation-passage graph，并保留 section path、source type 和 line-level grounding。

### Contribution 2

提出 TreeSearch-guided Hybrid Graph RAG。TreeSearch 作为 high-precision sparse seed selector、structure-aware expansion constraint 和 grounding layer；vector graph retrieval 负责 semantic multi-hop expansion；adaptive policy 根据 query 类型决定是否触发 GraphRAG。

### Contribution 3

提出一套面向真实仓库 RAG 的评测协议，覆盖 retrieval quality、multi-hop recall、answer quality、citation correctness、grounding quality、latency、cost 和 incremental update。

## 7. 推荐实验设计

### 7.1 主方法对比

需要至少比较：

1. Dense RAG
2. TreeSearch only
3. Vector Graph RAG
4. Naive Hybrid RAG
5. TreeSearch-guided Graph RAG
6. Adaptive Hybrid Graph RAG

### 7.2 公开 benchmark

建议保留现有 benchmark，并补 multi-hop：

1. QASPER
2. FinanceBench
3. CodeSearchNet
4. HotpotQA
5. MuSiQue
6. 2WikiMultiHopQA

### 7.3 真实异构仓库任务集

构造 `RealRepoBench` 或 `HeteroRepoBench`：

1. code locate
2. config lookup
3. doc QA
4. code-doc consistency
5. troubleshooting
6. cross-source multi-hop

典型 query：

1. “这个配置项默认值在哪里定义，文档里怎么解释？”
2. “搜索模式 auto 的路由逻辑在哪里，实现和 README 描述是否一致？”
3. “某个 CLI 参数从 parser 到最终 search 调用经过哪些函数？”
4. “某个 API 报错时，文档和代码分别给了什么处理方式？”

### 7.4 指标

Retrieval 指标：

1. MRR
2. Recall@K
3. Hit@K
4. NDCG@K
5. supporting evidence recall

RAG 指标：

1. answer correctness
2. citation precision
3. citation recall
4. grounded answer rate
5. hallucination rate

系统指标：

1. P50 / P95 latency
2. LLM calls per query
3. dense calls per query
4. cost per 1k queries
5. index build time
6. incremental update time
7. index size

### 7.5 消融实验

至少做：

1. 去掉 TreeSearch seed
2. 去掉 structure constraint
3. 去掉 graph expansion
4. 去掉 LLM rerank
5. 去掉 adaptive policy
6. 去掉 line-level grounding
7. chunk-based graph construction vs TreeSearch-node graph construction

这些消融能回答：

1. TreeSearch 是不是只提供 sparse recall？
2. 结构约束是否降低 graph expansion 噪声？
3. node-level graph construction 是否优于 chunk-level graph construction？
4. adaptive policy 是否降低成本且保持质量？
5. line-level grounding 是否提升用户任务完成率？

## 8. 论文标题备选

### 8.1 推荐标题

**Structure-Preserving Hybrid Graph Retrieval for Heterogeneous Repositories**

### 8.2 更偏 RAG

**Structure-Preserving Hybrid Graph RAG for Code and Document Repositories**

### 8.3 更偏工业系统

**Adaptive Hybrid Graph RAG for Heterogeneous Enterprise Repositories**

### 8.4 更突出 TreeSearch

**TreeSearch-Guided Graph RAG with Structure-Preserving Grounding**

## 9. 最终建议

不要把 `vector-graph-rag` 作为一个独立竞品或简单 baseline 加进来。更好的方式是吸收它的 GraphRAG 机制，并用 TreeSearch 的结构能力重新定义它：

1. `vector-graph-rag` 提供 multi-hop graph retrieval 思路
2. TreeSearch 提供结构节点、稳定 ID、路径、source type 和 line range
3. 二者结合形成 structure-preserving GraphRAG

最强主张：

> TreeSearch is not only a sparse retrieval engine. It is a structural substrate for Graph RAG: it defines retrieval units, constrains graph expansion, routes computation, and grounds final answers back to exact repository locations.

如果按这个方向改，论文会从“TreeSearch 加 dense 的 hybrid system”升级成“结构保持型 Hybrid Graph RAG”，技术创新、RAG 效果和 industry story 都会更强。

## 10. 从投稿角度的核心判断

### 10.1 是否建议把 single-pass CoT rerank 和 TreeSearch 结合

建议结合，而且不建议把 single-pass CoT rerank 单独作为论文主创新。

原因是：

1. 单独的 single-pass CoT rerank 很容易被审稿人看成 listwise LLM reranking 或 prompt engineering。
2. GraphRAG、HippoRAG、LightRAG、RankRAG、RankGPT、FIRST 等相关工作已经很多，单独说“一次 LLM 调用选关系”不够新。
3. TreeSearch 的结构化节点、稳定 `node_id`、source path、line range、parent/child/sibling 关系，才是能把这个方法和已有 GraphRAG 拉开差异的部分。

因此最强写法不是：

> We use a single CoT prompt to rerank graph relations.

而是：

> We use TreeSearch to construct and constrain a structure-preserving evidence graph, then use a single structured LLM call to select a compact, verifiable evidence chain for multi-hop RAG.

也就是说，single-pass CoT rerank 应该被重新定位为：

> Single-pass structured evidence-chain selection over a TreeSearch-grounded graph.

这样它不再只是 reranking，而是一个面向结构化仓库 RAG 的 evidence selection 模块。

### 10.2 最核心的论文创新点

如果只保留一个最核心创新点，建议写成：

> TreeSearch turns heterogeneous repository files into structure-preserving retrieval units, and these units become the substrate for GraphRAG: graph nodes are not arbitrary chunks, graph expansion is constrained by document structure, and final answers are grounded back to exact repository locations.

展开成三个 contribution：

### Innovation 1: Structure-Preserving Graph Substrate

已有 GraphRAG 大多从 chunk 或 passage 抽取实体关系。问题是 chunk 破坏原始结构，尤其不适合代码、配置和长文档。

TreeSearch 可以把异构文件解析成天然结构节点：

1. Markdown section
2. code function / class
3. JSON subtree
4. config key subtree
5. PDF / DOCX section
6. HTML / XML node

这些节点带有稳定 `node_id`、路径、父子关系、source type 和 line range。GraphRAG 不再建立在 arbitrary chunks 上，而是建立在 repository-native structure 上。

审稿价值：

1. 比“无 graph database”更像研究贡献。
2. 和 LightRAG / HippoRAG 的 chunk/passage graph 有明确差异。
3. 能自然连接 code/document/config 异构仓库场景。

### Innovation 2: TreeSearch-Guided Graph Expansion

传统 GraphRAG expansion 主要依赖 entity-relation edge、PPR、向量相似度或图遍历。TreeSearch 可以加入结构约束：

1. sparse hit 的节点作为高精度 graph seed
2. 同 section / parent / child / sibling 关系优先扩展
3. 同文件、同模块、同配置树的 relation 权重更高
4. code-doc-config 之间的 cross-source edge 单独建模
5. line proximity 和 section path distance 作为 rerank feature

核心主张：

> TreeSearch is not only an additional sparse retriever; it constrains graph expansion and reduces noisy multi-hop context.

审稿价值：

1. 回答了“TreeSearch 比 BM25 多什么”。
2. 回答了“为什么不是普通 GraphRAG”。
3. 可以通过消融直接证明：去掉 structure constraint 后 evidence precision 和 answer quality 下降。

### Innovation 3: Single-Pass Verifiable Evidence Chain Selection

不要把这个模块写成普通 rerank。更好的形式是：

输入：

1. query
2. TreeSearch seed nodes
3. expanded relation candidates
4. node path / source type / line range
5. relation-to-node links

输出：

1. bridge entities
2. selected relations
3. reasoning chain
4. supporting node ids
5. citation line ranges
6. evidence sufficiency flag

这样 single-pass CoT 的作用不是“排序”，而是：

> 在一次 LLM 调用中完成 bridge entity identification、evidence compression、relation-chain selection 和 citation grounding。

审稿价值：

1. 比 prompt reranking 更强。
2. 可以和 IRCoT 的多轮 retrieve-reason 对比成本。
3. 可以和 HippoRAG/PPR 对比 evidence chain precision。
4. 可以做 verifier，检查 chain 是否连通、是否引用真实 node、是否覆盖必要证据。

## 11. 审稿人采纳率评估

下面是按不同论文定位的现实判断。

### 11.1 单独写 single-pass CoT rerank

采纳率判断：低。

适合目标：

1. workshop
2. demo
3. 工程报告
4. 弱 Findings

主要风险：

1. 被认为是 prompt engineering。
2. 和 RankGPT / RankRAG / listwise reranking 工作差异不够。
3. GraphRAG 场景下只做 relation rerank，不足以构成完整方法创新。
4. 如果提升主要是 1 个百分点以内，很难说服审稿人。

预估：

| 目标 | 粗略采纳概率 |
|---|---:|
| ACL / EMNLP main | 10%-20% |
| ACL / EMNLP Findings | 25%-35% |
| RAG / NLP workshop | 50%-70% |

### 11.2 TreeSearch + GraphRAG，但只做系统集成

采纳率判断：中低。

如果只是把 TreeSearch、vector graph、LLM rerank 拼成 pipeline，创新仍然偏工程。

主要风险：

1. reviewer 会问与 LightRAG、HippoRAG、Microsoft GraphRAG 的本质区别。
2. reviewer 会认为结构节点只是更细粒度的 passage。
3. 如果没有真实异构仓库实验，应用动机不够强。

预估：

| 目标 | 粗略采纳概率 |
|---|---:|
| ACL / EMNLP main | 15%-25% |
| ACL / EMNLP Findings | 35%-45% |
| Industry Track | 45%-60% |
| workshop | 60%-75% |

### 11.3 TreeSearch-Guided Structure-Preserving Graph RAG

采纳率判断：中高，是最推荐方向。

这个方向需要把 novelty 压在三个地方：

1. structure-preserving graph substrate
2. TreeSearch-guided expansion constraint
3. single-pass verifiable evidence chain selection

如果实验完整，有机会成为 Findings 或 Industry Track 强稿。

关键前提：

1. 必须有 strong baselines：Dense RAG、Hybrid RAG、HippoRAG、LightRAG、IRCoT、RankGPT/listwise rerank。
2. 必须有 end-to-end answer quality，不只 Recall@5。
3. 必须有 evidence / citation quality，不只 retrieval。
4. 必须有真实异构仓库任务，否则 TreeSearch 的结构优势无法充分体现。
5. 必须有清晰消融，证明 TreeSearch 不是普通 sparse channel。

预估：

| 目标 | 粗略采纳概率 |
|---|---:|
| ACL / EMNLP main | 25%-40% |
| ACL / EMNLP Findings | 50%-65% |
| EMNLP Industry Track | 55%-70% |
| COLING / NAACL Findings | 55%-70% |
| RAG / IR workshop | 75%+ |

### 11.4 如果实验做得非常强

如果能做到下面几点，ACL/EMNLP main 才有更现实的竞争力：

1. 在 2-3 个 multi-hop QA benchmark 上 answer EM/F1 明显超过 HippoRAG / LightRAG / IRCoT，或者质量相当但成本显著下降。
2. 在真实异构仓库任务上显著提升 task completion、citation correctness 和 time-to-answer。
3. 证明 chunk-level GraphRAG 在 code/config/doc 混合仓库中会破坏 grounding，而 TreeSearch-node graph 明显更好。
4. 提供结构化 evidence verifier，降低 hallucinated citation。
5. 做充分错误分析：失败来自 triplet extraction、missing edge、LLM chain selection 还是 grounding mismatch。

达到这个强度后，可以把采纳率估为：

| 目标 | 粗略采纳概率 |
|---|---:|
| ACL / EMNLP main | 40%-55% |
| ACL / EMNLP Findings | 65%-80% |
| Industry Track | 70%-85% |

## 12. 推荐的最终论文定位

最推荐的论文定位：

> A structure-preserving GraphRAG system for heterogeneous repositories, where TreeSearch provides repository-native retrieval units, sparse graph seeds, structural expansion constraints, and line-level grounding, while a single structured LLM call selects verifiable multi-hop evidence chains.

对应中文定位：

> 面向异构代码-文档仓库的结构保持型 GraphRAG：TreeSearch 负责把仓库解析成可 grounding 的结构节点，并用这些节点指导图构建、图扩展和证据链选择；GraphRAG 负责跨节点、跨文档、多跳语义推理；最终系统用一次结构化 LLM 调用生成可验证证据链和 grounded answer。

这比“TreeSearch + dense retrieval”更强，也比“single-pass CoT rerank”更像论文。

## 13. 最小可投版本

如果要控制工作量，最小可投版本可以这样收敛：

### 方法

1. TreeSearch parses repository files into structural nodes.
2. LLM extracts triplets from each structural node.
3. TreeSearch retrieves sparse seed nodes for each query.
4. The system expands a graph around seed nodes using entity-relation links and structural edges.
5. A single structured LLM call selects a compact evidence chain.
6. The answer is generated with node-level citation and line grounding.

### 实验

1. Public multi-hop QA：HotpotQA、MuSiQue、2WikiMultiHopQA。
2. Document QA：QASPER。
3. RealRepoBench：至少 100-300 条真实或半真实 code/doc/config query。
4. Metrics：Recall@5、Answer F1/LLM judge、citation precision、latency、LLM calls、cost。
5. Ablation：no TreeSearch seed、no structural edge、no evidence-chain output、no verifier、chunk graph vs node graph。

### 最重要的结果表

主表建议比较：

1. Dense RAG
2. Hybrid dense+sparse
3. Vector Graph RAG
4. HippoRAG / LightRAG
5. TreeSearch-guided GraphRAG
6. TreeSearch-guided GraphRAG + evidence verifier

如果主表能显示：

1. multi-hop QA 上质量持平或小幅领先；
2. 真实仓库任务上明显领先；
3. citation correctness 明显更高；
4. LLM calls / latency / cost 明显低于 iterative methods；

这篇论文就有比较健康的投稿竞争力。

## 14. 一句话结论

最该写的不是“单次 CoT 重排序”论文，而是：

> TreeSearch-Guided Structure-Preserving Graph RAG: 用 TreeSearch 把异构仓库变成可 grounding 的结构图，用单次结构化 LLM 调用选择可验证证据链，从而在多跳 RAG 中兼顾质量、成本和可解释性。

这个方向的论文采纳率明显高于单独写 rerank，也更符合 TreeSearch 这个 repo 的独特价值。

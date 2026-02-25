

## 一、TreeSearch 当前代码重构建议

### 1. `llm.py` — LLM 客户端模块

| 问题 | 说明 |
|------|------|
| **每次调用都创建 `AsyncOpenAI` 客户端** | `achat()` 和 `achat_with_finish_reason()` 中 `async with openai.AsyncOpenAI(...)` 在每次请求都新建+销毁连接，高并发场景性能极差。应改为**模块级单例 client**，复用连接池。 |
| **两个函数 90% 代码重复** | `achat` 和 `achat_with_finish_reason` 几乎一样，只差返回值。应抽成一个内部 `_achat_impl()` 返回完整 response，外面两个函数做不同的返回值解包。 |
| **`count_tokens` 每次调用都创建 encoder** | `tiktoken.encoding_for_model(model)` 有内部缓存但仍有开销。建议维护 `_encoder_cache: dict[str, tiktoken.Encoding]` 显式缓存。 |
| **缺少 max_tokens / response_format 参数** | 没有 `max_tokens` 控制，也不支持 `response_format={"type":"json_object"}`（即 OpenAI JSON mode），这在 JSON 输出场景可以省掉后续 `extract_json` 解析的复杂度。 |
| **`extract_json` 太脆弱** | 用字符串 `replace("None", "null")` 会破坏正文包含 "None" 的合法文本。应只在 JSON 文本内做替换，或直接用 `json_mode`。 |
| **缺少 LLM 调用的可观测性** | 没有 token 用量统计（input/output tokens）、调用延迟统计、成本估算。生产环境必备。 |

### 2. `tree.py` — 树结构模块

| 问题 | 说明 |
|------|------|
| **`Document.get_node_by_id` 每次都 `flatten_tree` 全遍历** | O(n) 线性搜索。应在 Document 初始化时构建 `_node_map: dict[str, dict]`，O(1) 查找。 |
| **缺少 `Document` 的序列化/反序列化** | `Document` 和 JSON 索引文件之间没有统一的 `from_index()` / `to_index()` 方法，CLI 中手动拼装。 |
| **`flatten_tree` 在多处被重复调用** | `search.py` 和 `indexer.py` 都反复 flatten，应该在 Document 级别缓存。 |
| **缺少树的深度/节点数等元信息** | 没有 `depth`, `total_nodes`, `total_tokens` 等统计属性。 |

### 3. `indexer.py` — 索引构建模块

| 问题 | 说明 |
|------|------|
| **`_children()` 函数在两处重复定义** | `_update_token_counts` 和 `_thin_tree` 里各定义了一遍 `_children()`，应抽出为模块级函数。 |
| **`build_index` 缺少增量索引能力** | 每次都全量重建，没有检测文件是否变化（hash/mtime）跳过已索引的文件。 |
| **`build_index` 无并发控制** | `asyncio.gather` 同时发起所有文件的索引，如果有 100 个文件会同时发起 100+ LLM 并发请求。应加 `asyncio.Semaphore` 限流。 |
| **`_llm_generate_toc` 的 `_locate()` 函数定位不够健壮** | fuzzy match 只取前 4 个词，对长关键词不友好。可结合 `difflib.SequenceMatcher` 做更好的模糊匹配。 |
| **文本预处理太简单** | 没有处理 HTML 标签、特殊 Unicode 字符、表格等。 |

### 4. `search.py` — 搜索模块 ✅ 已完成重构

| 项目 | 状态 |
|------|------|
| Best-First Tree Search 作为默认策略 | ✅ 已实现 `BestFirstTreeSearch` 类 |
| BM25 预评分 + LLM 精排三层架构 | ✅ 已实现 |
| Early stopping + budget control | ✅ `max_llm_calls` + threshold 停止 |
| Subtree cache（query-aware） | ✅ `(query_fingerprint, node_id)` 缓存 |
| SearchResult 丰富元信息 | ✅ 新增 `query`, `total_llm_calls`, `strategy` |
| MCTS 保留为可选策略 | ✅ `strategy="mcts"` |
| search() API 支持策略切换 | ✅ `strategy` 参数 |

### 5. `rank_bm25.py` — BM25 模块 ✅ 已完成实现

| 项目 | 状态 |
|------|------|
| 纯 Python BM25Okapi（无 numpy） | ✅ 已实现 |
| 中英文分词（jieba + regex） | ✅ jieba lazy-load，英文 regex 分词 |
| NodeBM25Index 节点级索引 | ✅ 分层加权（title > summary > body） |
| 祖先分数回传 | ✅ `ancestor_decay` 参数控制 |
| 多文档索引 + 按文档过滤 | ✅ `get_node_scores_for_doc()` |

### 6. `cli.py` — 命令行模块 ✅ 已更新

| 项目 | 状态 |
|------|------|
| `--strategy best_first\|mcts\|llm` | ✅ 替代旧 `--no-mcts` |
| `--max-llm-calls` 参数 | ✅ |
| `--no-bm25` 参数 | ✅ |

### 7. 整体架构待完善

| 缺失 | 说明 | 优先级 |
|------|------|--------|
| **无配置文件系统** | PageIndex 有 `config.yaml`，TreeSearch 全部硬编码 | P2 |
| **无日志系统** | PageIndex 有 `JsonLogger` 做结构化日志，TreeSearch 只用 `logging` | P2 |
| **无版本化索引格式** | 索引 JSON 没有版本号字段，后续格式变更会不兼容 | P2 |

---

## 二、对比 PageIndex 原版 cookbook 的差距总结

| 能力 | PageIndex (原版) | TreeSearch (当前) |
|------|-----------------|-------------------|
| Markdown 索引 | 有 | 有，基本对等 |
| 纯文本索引 | 有 | 有，基本对等 |
| MCTS 树搜索 | 有 | ✅ 有，保留为可选策略 |
| Best-First 搜索 | 无 | ✅ 新增，作为默认策略 |
| BM25 节点级索引 | 无 | ✅ 新增，结构感知 |
| 中英文分词 | 无 | ✅ 新增（jieba + regex） |
| 增量索引 | 有缓存检测 | 无 |
| 索引持久化格式 | 有标准格式 | 无版本控制 |
| 结构化日志 | JsonLogger | 无 |
| Token 统计/成本 | 部分 | 无 |
| 并发控制 | 基本 | 无限并发 |

---

## 三、MCTS vs Best-First Tree Search 对比分析

### 3.1 核心问题：MCTS 适合 TreeSearch 这个场景吗？

**MCTS 的设计初衷**是解决**不确定性博弈**问题（如围棋），其核心假设：
- 每一步的结果是**随机的**（需要多次模拟取平均）
- 状态空间极大，需要在**探索 vs 利用**之间权衡
- 单次评估不可靠，需要**多次 rollout** 来获取可靠信号

**但 TreeSearch 场景的特点**：
- LLM `temperature=0`，评估结果是**确定性的**
- 节点相关性是**静态的**（同一 query+node 的相关性不变）
- 已有 `_value_cache` 避免重复评估 — 但 MCTS 仍在用 visit_count/total_value 做统计平均，这对确定性结果是**多余计算**

### 3.2 当前 MCTS 实现的实际行为分析

当前代码中，`_evaluate()` 使用 `temperature=0` 并有 cache，所以：
- **同一节点的 value 永远不变**：`avg_value = total_value / visit_count = (value * N) / N = value`
- **backpropagate 传递的是子节点 max value**，但 `avg_value` 其实就是被稀释后的值
- **UCB1 的探索项** `c * sqrt(ln(N_parent) / N_child)` 会随 visit 增加而衰减，最终退化为按 avg_value 排序

结论：**当前 MCTS 在确定性 LLM 下，本质上是一个带额外开销的 Best-First Search**。

### 3.3 Best-First Tree Search 更适合的原因

| 维度 | MCTS（当前） | Best-First（建议） |
|------|-------------|-------------------|
| 确定性 | 需要 temperature=0 + cache + tie-break 多处 hack 才能保证 | 天然确定性，无需额外措施 |
| 计算开销 | 维护 visit_count/total_value，backpropagate 链式更新 | 只维护 priority score，优先队列 O(log n) |
| 评估次数 | 同一节点可能被重复访问（虽有 cache 避免重复 LLM 调用，但循环开销仍在） | 每个节点最多评估一次 |
| Early stopping | 当前未实现，不知道何时该停 | 天然支持：队列中最高分 < 阈值即可停止 |
| BM25 分数集成 | 不兼容：MCTS 需要 rollout-based 的 value | 天然兼容：BM25 score 直接作为 priority 的一部分 |
| 代码复杂度 | UCB1 + backpropagate + expand/select 四阶段 | 一个 while 循环 + 优先队列 |
| 可解释性 | 输出 visits + avg_value，用户难理解 | 输出 score + path，直观清晰 |
| 百万级扩展 | 状态维护成本高（每个节点需要 visit/value） | 只需 visited set + priority queue，内存极低 |

### 3.4 设计建议

**保留 MCTS 作为可选策略**（毕竟已经写好了），但**默认改为 Best-First**。✅ 已实现。

### 3.5 结论

在 TreeSearch 的确定性 LLM 评估场景下，**Best-First Search 严格优于 MCTS**：
- 更简单、更快、更省 LLM 调用
- 天然支持 early stopping 和 budget control
- 天然兼容 BM25 分数作为先验 priority
- MCTS 的优势（处理不确定性、探索-利用权衡）在 `temperature=0` + cache 下完全用不上

---

## 四、面向百万级文档的三层架构

### 4.1 核心思路

TreeSearch 的定位：**轻量 pypi 库，开箱即用，不集成 embedding/向量检索/PDF 解析**。

在这个约束下，要打赢 BM25 和 PK GraphRAG 的路线：

> BM25 负责"找线索"，Tree Search + LLM 负责"理解结构、做决策"

关键优势：**文档天然是树**，不需要像 GraphRAG 那样强行从无结构数据构建图。

### 4.2 三层架构 ✅ 已实现

```
Query
  |
  v
[Layer 1] BM25 节点级召回（百万 -> 千）           ✅ NodeBM25Index
  |       - 分层加权：title(1.0) + summary(0.7) + body(0.3)
  |       - 祖先分数回传：parent += alpha * max(child)
  |       - 中英文分词（jieba + regex）
  v
[Layer 2] Best-First Tree Search（千 -> 十）      ✅ BestFirstTreeSearch
  |       - 优先队列驱动，BM25 分数作为初始 priority
  |       - LLM 评估（带 budget 控制 + early stopping）
  |       - Subtree cache 跨搜索复用
  v
[Layer 3] LLM 精排（十 -> 2~4）                   ✅ 内置于 BestFirstTreeSearch
          - LLM 只判断 "这个节点是否包含答案"
          - 不给全文，只给 title + summary + query
          - temperature=0，结果缓存
```

### 4.3 Layer 1 实现细节（`rank_bm25.py`）

**库内内置（不依赖外部服务）**：

1. **`BM25Okapi`**：纯 Python BM25 实现，无 numpy 依赖
2. **`NodeBM25Index`**：节点级结构感知索引
   - 分层加权 BM25：`title(1.0) + summary(0.7) + body(0.3)`
   - 祖先分数回传：`parent.score += alpha * max(child.score)`
   - 支持多文档索引 + 按文档过滤
3. **`tokenize()`**：中英文自动检测分词
   - 中文：jieba（可选依赖，未安装则降级到字符级切分）
   - 英文：正则分词 + 小写化

### 4.4 Layer 2 实现细节（`search.py`）

`BestFirstTreeSearch` 核心设计：
- **Priority = LLM relevance + BM25 prior + depth penalty**
- **Early stopping**：队列最高分 < threshold 即停
- **Budget control**：`max_llm_calls` 限制总 LLM 调用次数
- **Subtree cache**：`(query_fingerprint, node_id) -> relevance`，类级别共享

### 4.5 Layer 3 实现细节

LLM 的角色严格限定为 **judge**，不是 generator：
- 输入：title + summary + query
- 输出：`{"relevance": 0.0~1.0}`
- temperature=0（确定性）+ 结果缓存

### 4.6 为什么这套方案能 PK GraphRAG

| 维度 | GraphRAG | TreeSearch 方案 |
|------|----------|----------------|
| 离线构建成本 | 高（需要构建知识图谱） | 低（树天然存在） |
| 更新复杂度 | 高（图结构增量更新难） | 低（单文档重建树即可） |
| 可解释性 | 一般（图路径复杂） | 极强（树路径天然可解释） |
| 查询延迟 | 高 | 可控（budget 机制） |
| 工程复杂度 | 高（图DB + embedding + LLM） | 中（纯 LLM + 文本匹配） |
| 中文/垂直领域 | 不稳定（依赖 embedding 质量） | 可定制（LLM prompt 可调） |
| 向量依赖 | 必须 | 不需要 |

核心论点：**GraphRAG 解决的是"没结构的数据如何强行建结构"，TreeSearch 解决的是"有结构的数据如何真正用好结构"。在文档树场景下，TreeSearch 是更工程、更便宜、更可控的方案。**

---

## 五、落地计划

### Phase 1：核心重构 ✅ 已完成

- [x] 实现 Best-First Tree Search，作为默认搜索策略（保留 MCTS 可选）
- [x] 实现节点级 BM25 召回（内置关键词匹配 + 分层加权）
- [x] 祖先分数回传机制
- [x] LLM 调用 budget 控制 + early stopping
- [x] query-aware subtree 缓存
- [x] search() API 支持 `strategy` 参数切换
- [x] CLI 更新（`--strategy`, `--max-llm-calls`, `--no-bm25`）
- [x] 中英文分词（jieba 可选依赖）
- [x] 测试 115 个全部通过
- [x] 新增 example 05_best_first_search.py

### Phase 1.1：API 完善 ✅ 已完成

- [x] `BestFirstTreeSearch as TreeSearch` 别名导出（`__init__.py`），与项目名对应
- [x] Examples 更新：01/02/04 使用 `strategy="best_first"` 新 API
- [x] `docs/architecture.md` 重写：三层架构、Best-First vs MCTS 对比、GraphRAG 对比
- [x] `docs/api.md` 重写：新增 BestFirstTreeSearch、BM25、tokenize 等完整 API 文档
- [x] `README.md` 更新新 slogan：Structure-aware document retrieval without embeddings
- [x] `pyproject.toml` 更新 description

### Phase 2：工程完善 ✅ 已完成

- [x] `PreFilter` 协议接口（`search.py`，支持用户注入外部召回，`NodeBM25Index` 已实现协议）
- [x] LLM 单例客户端（`llm.py`，`_get_async_client()` 复用连接池，按 api_key+base_url 缓存）
- [x] `_achat_impl()` 消除 `achat` / `achat_with_finish_reason` 90% 重复代码
- [x] `count_tokens()` encoder 缓存（`_encoder_cache` 避免重复创建）
- [x] Document `_node_map` 缓存（`tree.py`，`__post_init__` 构建，`get_node_by_id` O(1)）
- [x] 索引格式版本化（`INDEX_VERSION = "1.0"`，`save_index` 写入，`load_index` 检测不兼容）
- [x] 增量索引（`_index_meta.json` 记录文件 MD5 hash，未变更文件跳过）
- [x] 并发控制（`asyncio.Semaphore(max_concurrency)`，默认 5 并发）
- [x] `_children()` 函数去重（`indexer.py` 抽出为模块级 `_children_indices()`）
- [x] CLI 增强：`--api-key`、`--force`、`--max-concurrency`、耗时显示
- [x] CLI `_load_documents()` 跳过 `_index_meta.json` 等 `_` 开头文件
- [x] 测试 123 个全部通过

### Phase 3：对比评测 ✅ 已完成

- [x] `treesearch/metrics.py`：评估指标模块（Precision@K、Recall@K、MRR、NDCG@K、Hit@K、F1@K）
- [x] `examples/benchmark_ground_truth.json`：20 条 ground truth 查询（easy/medium/hard 三档难度）
- [x] `examples/benchmark.py`：完整 benchmark 脚本，支持 4 种策略对比
  - BM25-only：零 LLM 成本，MRR=0.71，NDCG@3=0.63
  - BestFirst（默认）：BM25 + LLM tree search，精度明显提升
  - MCTS：蒙特卡洛树搜索 + LLM 评估
  - LLM single-pass：单次 LLM 调用，最少 API 开销
- [x] benchmark 支持 `--bm25-only`（无需 API key）、`--strategies`、`--force-index`、`--top-k`
- [x] 结果输出到 `results/benchmark_results.json`
- [x] 对比表格：Metric / Strategy 交叉对比 + 按难度分组统计
- [x] 卖点验证：不用 embedding、天然支持结构层级、BM25 作为零成本 baseline
- [x] 测试 153 个全部通过

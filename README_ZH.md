[**🌐English**](https://github.com/shibing624/TreeSearch/blob/main/README.md) | [**🇨🇳中文**](https://github.com/shibing624/TreeSearch/blob/main/README_ZH.md)

<div align="center">
  <a href="https://github.com/shibing624/TreeSearch">
    <img src="https://raw.githubusercontent.com/shibing624/TreeSearch/main/docs/logo.svg" height="150" alt="Logo">
  </a>
</div>

-----------------

# TreeSearch: 结构感知的文档检索
[![PyPI version](https://badge.fury.io/py/pytreesearch.svg)](https://badge.fury.io/py/pytreesearch)
[![Downloads](https://static.pepy.tech/badge/pytreesearch)](https://pepy.tech/project/pytreesearch)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/TreeSearch.svg)](https://github.com/shibing624/TreeSearch/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

**TreeSearch** 是一个结构感知的文档检索库。无需向量嵌入，无需分块，基于 BM25 + LLM 推理在文档树结构上进行检索。

## 安装

```bash
pip install -U pytreesearch
```

## 快速开始

```python
import asyncio
from treesearch import build_index, load_index, Document, search

async def main():
    # 构建 Markdown 文件索引
    await build_index(paths=["docs/*.md"], output_dir="./indexes")

    # 加载索引文档
    import os
    documents = []
    for fp in sorted(os.listdir("./indexes")):
        if not fp.endswith(".json"):
            continue
        data = load_index(os.path.join("./indexes", fp))
        documents.append(Document(
            doc_id=fp, doc_name=data["doc_name"],
            structure=data["structure"],
            doc_description=data.get("doc_description", ""),
        ))

    # 使用 Best-First 策略搜索（BM25 + LLM）
    result = await search(query="认证系统如何工作？", documents=documents)
    for doc_result in result.documents:
        for node in doc_result["nodes"]:
            print(f"[{node['score']:.2f}] {node['title']}")

asyncio.run(main())
```

需要先设置 API Key：

```bash
export OPENAI_API_KEY="sk-..."
# 可选：自定义 endpoint
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

## 为什么选择 TreeSearch？

传统 RAG 系统将文档切分为固定大小的块，通过向量相似度检索。这种方式**破坏了文档结构**，丢失了标题层级关系，且无法处理需要推理的查询。

TreeSearch 采用完全不同的方法——根据文档的自然标题层级将其解析为**树结构**，然后使用 **BM25 + LLM 推理**在树上导航，找到最相关的章节。

| | 传统 RAG | TreeSearch |
|---|---|---|
| **预处理** | 分块 + 向量嵌入 | 解析标题 → 构建树 |
| **检索** | 向量相似度搜索 | BM25 预打分 + LLM 树搜索 |
| **多文档** | 需要向量数据库路由 | LLM 根据文档描述路由 |
| **结构** | 分块后丢失 | 完整保留为树形层级 |
| **依赖** | 向量数据库 + 嵌入模型 | 仅 LLM（无嵌入、无向量库） |
| **零成本基线** | 无 | BM25 独立搜索（无需 LLM） |

### 核心优势

- **无需向量嵌入** — 不需要训练、部署或付费使用嵌入模型
- **无需分块** — 文档保留自然的标题层级结构
- **无需向量数据库** — 不需要 Pinecone、Milvus 或 Chroma
- **树感知检索** — 标题层级引导搜索，而非任意的分块边界
- **BM25 零成本基线** — 即时关键词搜索，无需 API 调用，可独立使用或作为预过滤
- **LLM 预算控制** — 设定每次查询的最大 LLM 调用次数，置信度高时提前停止

## 功能特性

- **三层搜索** — BM25 预打分 → Best-First 树搜索 → LLM 相关性评估
- **树结构索引** — Markdown 和纯文本文档被解析为层级树
- **BM25 节点级索引** — 结构感知评分，层级字段加权（标题 > 摘要 > 正文）和祖先传播
- **Best-First 搜索**（默认） — 优先队列驱动，确定性搜索，支持提前停止和预算控制
- **MCTS 搜索** — 蒙特卡洛树搜索，LLM 作为价值函数
- **LLM 单次调用** — 每个文档一次 LLM 调用，成本最低
- **多文档搜索** — 通过 LLM 推理在文档集合间路由查询
- **中英文支持** — 内置 jieba 中文分词和英文正则分词
- **批量索引** — `build_index()` 支持 glob 模式并发多文件处理
- **评估指标** — 内置 Precision@K、Recall@K、MRR、NDCG@K、Hit@K、F1@K
- **异步优先** — 所有核心函数均为异步，提供同步适配器
- **CLI 命令** — `treesearch index` 和 `treesearch search` 命令

## BM25 独立搜索（无需 LLM）

```python
from treesearch import NodeBM25Index, Document, load_index

data = load_index("indexes/my_doc.json")
doc = Document(doc_id="doc1", doc_name=data["doc_name"], structure=data["structure"])

index = NodeBM25Index([doc])
results = index.search("认证配置", top_k=5)
for r in results:
    print(f"[{r['bm25_score']:.4f}] {r['title']}")
```

## CLI

```bash
# 从 glob 模式构建索引
treesearch index --paths "docs/*.md" --add-description

# 使用 Best-First 搜索（默认，BM25 + LLM）
treesearch search --index_dir ./indexes/ --query "认证系统如何工作？"

# 使用 MCTS 策略搜索
treesearch search --index_dir ./indexes/ --query "部署" --strategy mcts

# 控制 LLM 调用预算
treesearch search --index_dir ./indexes/ --query "认证" --max-llm-calls 10
```

## 工作原理

```
输入文档 (MD/TXT)
        │
        ▼
   ┌──────────┐
   │  Indexer  │  解析标题 → 构建树 → 生成摘要
   └────┬─────┘    (build_index 支持 glob 批量处理)
        │  JSON 索引文件
        ▼
   ┌──────────┐
   │  search   │  BM25 预打分 → 文档路由 → 树搜索
   └────┬─────┘
        │  SearchResult
        ▼
  带分数和文本的排序节点
```

**第一层 — BM25 预打分**：`NodeBM25Index` 使用结构感知的 BM25 对所有树节点评分，采用层级字段加权（标题 > 摘要 > 正文）和祖先分数传播。即时完成，无需 LLM。

**第二层 — Best-First 树搜索**：`BestFirstTreeSearch` 使用优先队列展开最有潜力的节点。LLM 仅评估节点的标题 + 摘要的相关性。当最高分低于阈值时提前停止。

**第三层 — 结果输出**：预算控制的 LLM 调用，支持子树缓存以便跨查询复用。

### 搜索策略

| 策略 | 描述 | LLM 调用 | 适用场景 |
|------|------|----------|----------|
| `best_first`（默认） | BM25 预打分 + 优先队列 + LLM 评估 | 中等（预算控制） | 通用场景，准确率最高 |
| `mcts` | 蒙特卡洛树搜索，LLM 作为价值函数 | 较高 | 复杂推理查询 |
| `llm` | 每个文档一次 LLM 调用 | 最少 | 低成本、简单查询 |
| BM25-only | `NodeBM25Index.search()` 独立使用 | 零 | 即时关键词搜索，无需 API Key |

## 示例

| 示例 | 描述 |
|------|------|
| [`01_index_and_search.py`](examples/01_index_and_search.py) | 单文档索引 + BestFirst 搜索 |
| [`02_text_indexing.py`](examples/02_text_indexing.py) | 纯文本 → 树索引，自动标题检测 |
| [`03_cli_workflow.py`](examples/03_cli_workflow.py) | CLI 工作流：构建索引 + 策略搜索 |
| [`04_multi_doc_search.py`](examples/04_multi_doc_search.py) | 多文档 BM25 + BestFirst + 策略对比 + 中文 |
| [`05_benchmark.py`](examples/05_benchmark.py) | 基准测试：BM25 / BestFirst / MCTS / LLM 全指标评估 |

## 项目结构

```
treesearch/
├── llm.py            # 异步 LLM 客户端，支持重试和 JSON 提取
├── tree.py           # Document 数据类、树操作、持久化
├── indexer.py        # Markdown / 纯文本 → 树结构，批量 build_index()
├── search.py         # Best-First、MCTS、LLM 搜索，文档路由，统一 search() API
├── rank_bm25.py      # BM25Okapi、NodeBM25Index、中英文分词器
├── metrics.py        # 评估指标：Precision@K、Recall@K、MRR、NDCG@K、Hit@K、F1@K
└── cli.py            # CLI 入口（index / search）
```

## 文档

- [架构设计](https://github.com/shibing624/TreeSearch/blob/main/docs/architecture.md) — 设计原则和三层架构
- [API 参考](https://github.com/shibing624/TreeSearch/blob/main/docs/api.md) — 完整 API 文档

## 社区与支持

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/TreeSearch/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "llm"，加入技术交流群

<img src="https://github.com/shibing624/TreeSearch/blob/main/docs/wechat.jpeg" width="200" />

## 引用

如果您在研究中使用了 TreeSearch，请引用：

```bibtex
@software{xu2026treesearch,
  author = {Xu, Ming},
  title = {TreeSearch: Structure-Aware Document Retrieval Without Embeddings},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shibing624/TreeSearch}
}
```

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎贡献！请提交 [Pull Request](https://github.com/shibing624/TreeSearch/pulls)。

## 致谢

- [BM25 (Okapi BM25)](https://en.wikipedia.org/wiki/Okapi_BM25) — 经典的概率排序函数
- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — 为结构化索引与检索提供了启发

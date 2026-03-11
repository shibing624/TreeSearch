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

**TreeSearch** 是一个结构感知的文档检索库。将文档解析为树结构，然后通过 FTS5 关键词匹配进行检索。支持 Markdown、纯文本、代码文件（Python AST + 正则、Java/Go/JS/C++ 等）、HTML、XML、JSON、CSV、PDF 和 DOCX。无需向量embedding，无需分chunk。

毫秒检索万级文档和大型代码库，并保留文档结构，避免上下文丢失。

## 安装

```bash
pip install -U pytreesearch
```

## 快速开始

```python
from treesearch import TreeSearch

# 懒加载索引 —— 首次搜索时自动构建索引
ts = TreeSearch("docs/*.md", "src/*.py")
results = ts.search("认证系统如何工作？")
for doc in results["documents"]:
    for node in doc["nodes"]:
        print(f"[{node['score']:.2f}] {node['title']}")
        print(f"  {node['text'][:200]}")
```


## 为什么选择 TreeSearch？

**一句话**：保留文档结构的智能检索库，避免传统 RAG 把文档切碎导致的上下文丢失。

**核心差异**：
```
传统 RAG 方案：
文档 → 切成 chunks → 向量化 → 检索 → ❌ 上下文断裂

TreeSearch：
文档 → 解析为树结构（章节层级）→ 结构化检索 → ✅ 保留完整语义
```

### 传统 RAG vs TreeSearch

| | 传统 RAG | TreeSearch |
|---|---|---|
| **预处理** | 分块 + 向量嵌入 | 解析标题 → 构建树 |
| **检索** | 向量相似度搜索 | FTS5 关键词匹配（无需 LLM） |
| **多文档** | 需要向量数据库路由 | FTS5 跨文档打分 |
| **结构** | 分块后丢失 | 完整保留为树形层级 |
| **依赖** | 向量数据库 + 嵌入模型 | 仅 SQLite（无嵌入、无向量库） |

### 核心优势

- **结构感知** — 不是"找字符串"，而是"找章节/类/函数"
- **零成本** — 不需要 LLM，不需要 API Key
- **快速** — 毫秒级响应，不需要 Embedding
- **精准** — 带章节标题作为锚点，上下文清晰
- **无需向量嵌入** — 不需要训练、部署或付费使用嵌入模型
- **无需分块** — 文档保留自然的标题层级结构
- **无需向量数据库** — 不需要 Pinecone、Milvus 或 Chroma

## 功能特性

- **FTS5 搜索** — 零 LLM 调用，毫秒级 FTS5 关键词匹配，无需 API Key
- **SQLite FTS5 引擎** — 持久化倒排索引，WAL 模式，增量更新，MD 结构感知列（标题/摘要/正文/代码/前言），列权重加权，CJK 分词
- **树结构索引** — Markdown、纯文本、代码文件（Python AST + 正则、Java/Go/JS/C++/PHP）、HTML、XML、JSON、CSV、PDF 和 DOCX 均被解析为层级树
- **解析器注册表** — 可扩展的 `ParserRegistry`，内置解析器自动注册；支持 `ParserRegistry.register()` 注册自定义解析器
- **Python AST 解析** — `ast` 模块提取类/函数的完整签名（参数、返回值类型）；语法错误时回退正则
- **PDF/DOCX/HTML 解析器** — 可选解析器，通过 `pageindex`、`python-docx`、`beautifulsoup4` 实现（`pip install pytreesearch[all]`）
- **GrepFilter 精准匹配** — 支持字面量/正则表达式匹配，精准定位代码符号和关键词
- **Source-type 路由** — 根据文件类型自动选择预过滤器（如代码文件使用 GrepFilter + FTS5）
- **中英文支持** — 内置 jieba 中文分词和英文正则分词
- **批量索引** — `build_index()` 支持 glob 模式并发多文件处理
- **异步优先** — 所有核心函数均为异步，提供同步适配器
- **配置驱动默认值** — `search()` 和 `build_index()` 从 `get_config()` 读取默认值，支持按调用覆盖
- **CLI 命令** — `treesearch index` 和 `treesearch search` 命令

## FTS5 独立搜索

```python
from treesearch import FTS5Index, Document, load_index

data = load_index("indexes/my_doc.json")
doc = Document(doc_id="doc1", doc_name=data["doc_name"], structure=data["structure"])

fts = FTS5Index(db_path="indexes/fts.db")  # 持久化, 或不传为内存模式
fts.index_documents([doc])

# 简单关键词搜索
results = fts.search("认证配置", top_k=5)
for r in results:
    print(f"[{r['fts_score']:.4f}] {r['title']}")

# 高级 FTS5 查询语法
results = fts.search("认证", fts_expression='title:认证 AND body:配置', top_k=5)

# 按文档聚合统计
agg = fts.search_with_aggregation("认证", group_by_doc=True)
for doc_agg in agg:
    print(f"{doc_agg['doc_name']}: {doc_agg['hit_count']} 命中, 最高分={doc_agg['best_score']:.4f}")
```

## CLI

```bash
# 从 glob 模式构建索引
treesearch index --paths "docs/*.md" --add-description

# 使用 FTS5 搜索
treesearch search --index_dir ./indexes/ --query "认证系统如何工作？" --fts

# 持久化 FTS5 数据库
treesearch search --index_dir ./indexes/ --query "认证" --fts --fts-db ./indexes/fts.db
```

## 工作原理

```
输入文档 (MD/TXT/Code/JSON/CSV/HTML/XML/PDF/DOCX)
        │
        ▼
   ┌──────────┐
   │  Indexer  │  ParserRegistry 分派 → 解析结构 → 构建树 → 生成摘要
   └────┬─────┘    (build_index 支持 glob 批量处理)
        │  JSON 索引文件
        ▼
   ┌──────────┐
   │  search   │  FTS5/Grep 预过滤 → 跨文档打分 → 排序结果
   └────┬─────┘
        │  dict 结果
        ▼
  带分数和文本的排序节点
```

**FTS5 预打分**：`FTS5Index` 使用 SQLite FTS5 倒排索引，MD 结构感知列（标题/摘要/正文/代码/前言）和列权重加权实现快速打分。即时返回结果，无需 LLM。

**Source-Type 路由**：对于代码文件，自动组合 `GrepFilter` + `FTS5` 实现精准符号匹配。预过滤器根据文件类型通过 `PREFILTER_ROUTING` 自动选择。

## 适用场景

### 场景 1：技术文档问答（最强场景）

**问题**：公司内部有 100+ 份技术文档（API 文档、设计文档、RFC），传统搜索找不准。

```python
from treesearch import build_index, search

# 1. 构建索引（只需运行一次）
docs = await build_index(
    paths=["docs/*.md", "specs/*.txt"],
    output_dir="./indexes"
)

# 2. 搜索 — 毫秒级响应
result = await search(
    query="如何配置 Redis 集群？",
    documents=docs,
)

# 3. 结果 — 完整章节，不是碎片
for doc in result["documents"]:
    print(f"文档: {doc['doc_name']}")
    for node in doc["nodes"]:
        print(f"  章节: {node['title']}")
        print(f"  内容: {node['text'][:200]}...")
```

**为什么比传统 RAG 好？**
- 找到的是**完整章节**，不是碎片
- 带上**章节标题**作为上下文锚点
- 支持"查看父章节/子章节"的层级导航

### 场景 2：代码库检索

**问题**：想在大型代码库中搜索"登录相关的类和方法"，但 grep 只能找行，看不到结构。

```python
# 索引代码库
docs = await build_index(
    paths=["src/**/*.py", "lib/**/*.java"],
    output_dir="./code_indexes"
)

# 搜索 — 自动识别代码文件，用 AST 解析 + GrepFilter
result = await search(
    query="用户登录 authentication",
    documents=docs,
)

# 结果示例：
# 文档: auth_service.py
#   class UserAuthenticator
#     def login(username, password)
#     def verify_token(token)
```

**为什么比 grep/IDE 搜索好？**
- **语义理解**：不只是关键字匹配，能理解"登录"="authentication"
- **结构感知**：找到的是完整的类/方法，带 docstring
- **精准定位**：直接定位到代码行号

### 场景 3：长文本 QA（论文/书籍）

**问题**：有一篇 50 页的论文，想问"作者在第 3 章提到的实验方法是什么？"

```python
docs = await build_index(paths=["paper.pdf"])

result = await search(
    query="实验方法 methodology",
    documents=docs,
)

# 自动找到 "3.2 实验设计" 这一节的内容
```

**为什么比 Ctrl+F 好？**
- **语义匹配**：找的是"实验方法"的同义词段落
- **章节定位**：告诉你在第几章第几节
- **可扩展到多文档**：同时搜索 10 篇论文

### 实际案例对比

**案例**：在公司文档中查找"如何申请 GPU 机器"

**传统方式（Ctrl+F）**：
```
搜索 "GPU" → 找到 47 处匹配 → 手工翻阅 → 10 分钟
```

**TreeSearch 方式**：
```python
result = await search("如何申请 GPU 机器", docs)
# 直接返回 "资源申请指南 > GPU 申请流程" 章节
# 耗时：< 100ms
```

**效率提升**：**100x**

### 与其他方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Ctrl+F** | 简单直接 | 无语义理解，结果碎片化 | 已知关键字 |
| **传统 RAG** | 语义理解好 | 切片破坏上下文，响应慢 | 纯文本 QA |
| **向量数据库** | 相似度搜索 | 需要 Embedding 预处理，成本高 | 大规模语义检索 |
| **TreeSearch** | 保留结构 + 快速 + 零成本 | 需要结构化文档 | 技术文档/代码库 |

## Benchmark 评测

### 文档检索（QASPER）

基于 [QASPER](https://huggingface.co/datasets/allenai/qasper) 数据集评测（50 个 query，18 篇学术论文）：

| 指标 | Embedding (zhipu-embedding-3) | TreeSearch FTS5 |
|------|-----------------------------------|-----------------|
| **MRR** | 0.4235 | 0.3863 |
| **Precision@1** | 0.2553 | 0.1915 |
| **Recall@5** | 0.4259 | **0.5514** |
| **NDCG@3** | 0.3053 | 0.2836 |
| **F1@3** | 0.2196 | 0.2207 |
| **索引时间** | 22.8s | **0.1s** |
| **平均查询时间** | 199.7ms | **0.9ms** |

**核心结论**：
- Embedding MRR 高 9.6% — 自然语言查询的语义理解更强
- TreeSearch Recall@5 高 29% — 结构保留有助于召回更多相关内容
- TreeSearch 查询速度快 **217x** — 亚毫秒级 vs 百毫秒级
- TreeSearch 索引速度快 **228x** — 无需 Embedding API 调用

### 代码检索（CodeSearchNet）

基于 [CodeSearchNet](https://huggingface.co/datasets/code_search_net) 数据集评测（50 个 query，500 个 Python corpus）：

| 指标 | Embedding (zhipu-embedding-3) | TreeSearch FTS5 |
|------|-----------------------------------|-----------------|
| **MRR** | 0.8483 | 0.8433 |
| **Precision@1** | 0.7800 | **0.8000** |
| **Recall@5** | **0.9400** | 0.9000 |
| **Hit@1** | 0.7800 | **0.8000** |
| **索引时间** | 33.8s | **3.5s** |
| **平均查询时间** | 179.0ms | **2.4ms** |

**核心结论**：
- TreeSearch MRR 几乎持平 Embedding（0.84 vs 0.85）— BM25 在代码搜索中词汇重叠度高，表现出色
- TreeSearch **Precision@1 胜出**（0.80 vs 0.78）— 精确关键词匹配在代码搜索中更强
- TreeSearch 查询速度快 **74x** — 毫秒级 vs 百毫秒级
- TreeSearch 索引速度快 **10x** — 无需 Embedding API 调用

### 总结

> TreeSearch 不是要替代 Embedding 检索，而是提供一个**零成本、极速**的选择。在代码搜索场景中，查询与代码共享大量词汇，TreeSearch 表现与 Embedding 持平。在自然语言查询文档场景中，Embedding 在精度上略有优势，而 TreeSearch 在召回率上更胜一筹。

自行运行评测：
```bash
# 文档检索（QASPER）
python examples/benchmark/qasper_benchmark.py --max-samples 50 --max-papers 20 --with-embedding

# 代码检索（CodeSearchNet）
python examples/benchmark/codesearchnet_benchmark.py --max-samples 50 --max-corpus 500 --with-embedding
```

## 文档

- [架构设计](https://github.com/shibing624/TreeSearch/blob/main/docs/architecture.md) — 设计原则和架构
- [API 参考](https://github.com/shibing624/TreeSearch/blob/main/docs/api.md) — 完整 API 文档

## 社区与支持

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/TreeSearch/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "nlp"，加入技术交流群

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

- [SQLite FTS5](https://www.sqlite.org/fts5.html) — TreeSearch 的全文搜索引擎
- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — 为结构化索引与检索提供了启发

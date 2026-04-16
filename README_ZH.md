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

### Rust CLI

Rust CLI 与 Python 包分开发布。如果你只想安装独立命令行工具，不想带上 Python 依赖，可以直接：

```bash
cargo install treesearch
ts --help
```

如果本机没有 Rust，也可以直接从 [GitHub Releases](https://github.com/shibing624/TreeSearch/releases) 下载预编译二进制，当前覆盖：

- `x86_64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`
- `x86_64-pc-windows-msvc`

每次推送类似 `v0.1.0` 的 tag 后，GitHub Actions 会自动构建并上传这些平台的二进制：

```bash
git tag v0.1.0
git push origin v0.1.0
```

工作流完成后，直接到 GitHub Releases 下载对应平台的压缩包，解压后运行 `ts` 即可。

## 快速开始

```python
from treesearch import TreeSearch

# 直接传入目录 —— 自动递归发现所有支持的文件
ts = TreeSearch("project_root/", "docs/")
results = ts.search("认证系统如何工作？")
for doc in results["documents"]:
    for node in doc["nodes"]:
        print(f"[{node['score']:.2f}] {node['title']}")
        print(f"  {node['text'][:200]}")
```

目录递归扫描带有智能默认值：
- 自动发现 `.py`、`.md`、`.json`、`.jsonl`、`.java`、`.go`、`.ts`、`.pdf`、`.docx` 等文件
- 自动跳过 `.git`、`node_modules`、`__pycache__`、`.venv`、`dist`、`build` 等
- 安装 [`pathspec`](https://pypi.org/project/pathspec/) 后自动尊重 `.gitignore` 规则(`pip install pathspec`)
- 单目录安全上限 10,000 文件（可通过 `max_files` 配置）

也可以自由混合目录、文件和 glob 模式：

```python
# 三种输入类型可自由组合
ts = TreeSearch("src/", "docs/*.md", "README.md")
results = ts.search("认证配置")
```

### 内存模式

快速搜索、脚本或临时使用场景，设置 `db_path=None` 即可跳过写入 `.db` 文件：

```python
# 内存模式 — 不生成 index.db，所有索引保存在内存中
ts = TreeSearch("docs/", db_path=None)
results = ts.search("语音通话")
```

即使处理数千个文档性能也很出色（5,000 个文档 < 10ms）。代价是进程退出后索引丢失。如需持久化增量索引，使用默认 `db_path` 或指定文件路径。

### Tree 模式（最适合论文和文档）

对于学术论文、长文档和深层标题层级的技术文档，使用 **tree 模式** 进行结构感知的最佳优先搜索：

```python
from treesearch import TreeSearch

# Tree 模式：锚点检索 → 树遍历 → 路径聚合
ts = TreeSearch("papers/", "docs/")
results = ts.search("实验方法", search_mode="tree")

# Tree 模式返回排序节点（与 flat 模式相同）
for doc in results["documents"]:
    for node in doc["nodes"]:
        print(f"[{node['score']:.2f}] {node['title']}")

# 额外返回：树遍历路径，展示结果之间的层级关系
for path in results["paths"]:
    chain = " > ".join(p["title"] for p in path["path"])
    print(f"[{path['score']:.2f}] {chain}")
    print(f"  {path['snippet'][:200]}")
```

**何时使用哪种模式？**
| 模式 | 最适合 | MRR 优势 |
|------|--------|---------|
| `"auto"` (默认) | 自动根据文档类型选择 | **智能默认，全场景最优** |
| `"tree"` | 学术论文、有标题层级的技术文档 | QASPER 最优 (MRR 0.50, +25% vs FTS5) |
| `"flat"` | 代码搜索、关键词密集查询 | CodeSearchNet 最优 (MRR 0.91) |

**Auto Mode** (`search_mode="auto"`, 默认): 智能选择 tree vs flat，三层策略：
1. **类型映射** — 每种 `source_type` 有明确的 tree 收益标识（`_TREE_BENEFIT`）
2. **深度校验** — 只有实际树深度 ≥ 2 的文档才算真正有层级
3. **比例阈值** — ≥ 30% 的文档真正受益于 tree → `tree` 模式；否则 → `flat`

这避免了旧版"50 个代码文件中混了 1 个 markdown 就全走 tree"的问题。

| 文档类型 | Tree 收益？ | 深度检查 | Auto Mode |
|---|---|---|---|
| Markdown (.md) | ✅ 是 | 必须有标题层级 (depth ≥ 2) | `tree`（如有层级） |
| JSON (.json) | ✅ 是 | 必须有嵌套 (depth ≥ 2) | `tree`（如有嵌套） |
| Code (.py/.js/.go...) | ❌ 否 | — | `flat` |
| PDF (.pdf) | ❌ 否 | — | `flat` |
| DOCX (.docx) | ❌ 否 | — | `flat` |
| CSV (.csv) | ❌ 否 | — | `flat` |
| Text (.txt) | ❌ 否 | — | `flat` |
| JSONL (.jsonl) | ❌ 否 | — | `flat` |
| 未知类型 | ❌ 否（安全默认） | — | `flat` |


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

- **智能目录发现** — `ts.index("src/")` 自动递归发现所有支持的文件；跳过 `.git`/`node_modules`/`__pycache__`；尊重 `.gitignore`
- **FTS5 搜索** — 零 LLM 调用，毫秒级 FTS5 关键词匹配，无需 API Key
- **SQLite FTS5 引擎** — 持久化倒排索引，WAL 模式，增量更新，MD 结构感知列（标题/摘要/正文/代码/前言），列权重加权，CJK 分词
- **树结构索引** — Markdown、纯文本、代码文件（Python AST + 正则、Java/Go/JS/C++/PHP）、HTML、XML、JSON、CSV、PDF 和 DOCX 均被解析为层级树
- **Ripgrep 加速 GrepFilter** — 自动调用系统 `rg` 进行快速行级匹配，未安装时透明降级为纯 Python；基于命中次数的评分让多次命中的节点排名更高
- **解析器注册表** — 可扩展的 `ParserRegistry`，内置解析器自动注册；支持 `ParserRegistry.register()` 注册自定义解析器
- **Python AST 解析** — `ast` 模块提取类/函数的完整签名（参数、返回值类型）；语法错误时回退正则
- **PDF/DOCX/HTML 解析器** — 可选解析器，通过 `PyMuPDF`、`python-docx`、`beautifulsoup4` 实现（`pip install pytreesearch[all]`）
- **GrepFilter 精准匹配** — 支持字面量/正则表达式匹配，精准定位代码符号和关键词
- **Source-type 路由** — 根据文件类型自动选择预过滤器（如代码文件使用 GrepFilter + FTS5）
- **中英文支持** — 内置 jieba 中文分词和英文正则分词
- **批量索引** — `build_index()` 支持 glob 模式、文件路径和目录，并发多文件处理
- **异步优先** — 所有核心函数均为异步，提供同步适配器
- **配置驱动默认值** — `search()` 和 `build_index()` 从 `get_config()` 读取默认值，支持按调用覆盖
- **CLI 命令** — `treesearch "查询" 路径/` 一键搜索；`treesearch index` 和 `treesearch search` 支持高级工作流

## FTS5 独立搜索

```python
from treesearch import FTS5Index, Document, load_index, save_index, md_to_tree
import asyncio

# 方式 1：从 Markdown 构建并保存到 DB
result = asyncio.run(md_to_tree(md_path="docs/voice-call.md", if_add_node_summary=True))
save_index(result, "indexes/voice-call.db")

# 方式 2：从 DB 加载已保存的文档
doc = load_index("indexes/voice-call.db")  # 返回 Document 对象

# 创建 FTS5 索引并搜索
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
# 默认模式：一条命令完成所有操作（懒索引 + 搜索）
treesearch "认证系统如何工作？" src/ docs/
treesearch "配置 Redis" project/

# 带参数
treesearch "认证" src/ --max-nodes 10 --db ./my_index.db

# 高级：单独构建索引（适合大型代码库）
treesearch index --paths src/ docs/ --add-description
treesearch index --paths "docs/*.md" "src/**/*.py" --add-description

# 高级：搜索已构建的索引
treesearch search --index_dir ./indexes/ --query "认证系统如何工作？"
```

## 工作原理

```
输入文档 (MD/TXT/Code/JSON/CSV/HTML/XML/PDF/DOCX)
        │
        ▼
   ┌──────────┐
   │  Indexer  │  ParserRegistry 分派 → 解析结构 → 构建树 → 生成摘要
   └────┬─────┘    (build_index 支持 glob 批量处理)
        │  SQLite DB (FTS5)
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

# 1. 构建索引 — 直接传目录（只需运行一次）
docs = await build_index(
    paths=["docs/", "specs/"],
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
# 直接索引目录 —— 自动发现 .py, .java, .go 等文件
docs = await build_index(
    paths=["src/", "lib/"],
    output_dir="./code_indexes"
)

# 搜索 — 自动识别代码文件，用 AST 解析 + GrepFilter（ripgrep 加速）
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

### 场景 3：长文本 QA（论文/书籍）— Tree 模式

**问题**：有一篇 50 页的论文，想问"作者在第 3 章提到的实验方法是什么？"

```python
docs = await build_index(paths=["paper.pdf"])

# 论文场景使用 tree 模式 — 沿标题层级遍历，效果更好
result = await search(
    query="实验方法 methodology",
    documents=docs,
    search_mode="tree",  # 学术论文上 MRR 比 FTS5 高 25%，比 Embedding 高 19%
)

# 标准排序结果
# 自动找到 "3.2 实验设计" 这一节的内容

# Tree 模式还返回遍历路径
for path in result["paths"]:
    chain = " > ".join(p["title"] for p in path["path"])
    print(f"[{path['score']:.2f}] {chain}")
    # 例如 [0.82] 论文标题 > 3. 方法 > 3.2 实验设计
```

**为什么比 Ctrl+F 好？**
- **树感知排序**：沿标题层级（章 → 节 → 小节）遍历，找到最优路径
- **章节定位**：返回完整路径如 `3. 方法 > 3.2 实验设计`
- **可扩展到多文档**：同时搜索 10 篇论文
- 学术论文上 **MRR 比 Embedding 高 19%**（QASPER 评测，Auto/Tree MRR=0.50）

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
| **向量数据库** | 相似度搜索 | 需要 Embedding 预处理，成本高 | 大规模语义检索 |
| **TreeSearch** | 保留结构 + 快速 + 零成本 | 需要结构化文档 | 技术文档/代码库 |

## Benchmark 评测

### 文档检索（QASPER）

基于 [QASPER](https://huggingface.co/datasets/allenai/qasper) 数据集评测（50 个 query，学术论文）：

| 指标 | Embedding (zhipu emb-3) | FTS5 | Tree | **Auto** |
|------|------------------------|------|------|---------|
| **MRR** | 0.4235 | 0.4033 | 0.5046 | **0.5046** |
| **R@5** | 0.4259 | 0.5337 | **0.5812** | **0.5812** |
| **R@10** | 0.6075 | 0.8372 | **0.8674** | **0.8674** |
| **Hit@5** | 0.6383 | 0.7021 | **0.7660** | **0.7660** |
| **Hit@10** | 0.8085 | 0.9574 | **0.9787** | **0.9787** |
| **索引时间** | 88.3s | **0.1s** | **0.1s** | **0.1s** |
| **平均查询时间** | 154.8ms | 0.7ms | 1.0ms | **1.0ms** |

**核心结论**：
- 🏆 **Tree / Auto MRR 最优**（0.50 vs 0.42 Embedding）— 结构感知树遍历大幅提升排序质量
- R@5：Tree 0.58 vs Embedding 0.43，**+35% 召回率**
- Auto 自动路由到 Tree（markdown 深层树），效果完全等价，速度快 **155x** vs Embedding

### 金融文档检索（FinanceBench）

基于 [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) 数据集评测（50 个 query，SEC 财报文件）：

| 指标 | Embedding (zhipu-embedding-3) | FTS5 | Tree | **Auto** |
|------|-------------------------------|------|------|---------|
| **MRR** | 0.2206 | 0.2420 | **0.2512** | 0.2420 |
| **R@5** | 0.2782 | 0.2067 | **0.2344** | 0.2067 |
| **索引时间** | 406.0s | **0.24s** | **0.24s** | **0.24s** |
| **平均查询时间** | 154.3ms | 5.7ms | 23.5ms | **5.4ms** |

**核心结论**：
- Tree 模式 MRR（0.2512）和 R@5（0.2344）均最优
- Auto 路由到 FTS5（PDF 扁平结构），速度最快（5.4ms）且效果无损
- TreeSearch 索引速度快 **1692x** — 无需 Embedding API

### 代码检索（CodeSearchNet）

基于 [CodeSearchNet](https://huggingface.co/datasets/code_search_net) 数据集评测（100 个 query，Python corpus）：

| 指标 | Embedding (zhipu-embedding-3) | FTS5 | Tree | **Auto** |
|------|-------------------------------|------|------|---------|
| **MRR** | 0.8483 | 0.9050 | 0.2833 | **0.9100** |
| **R@5** | **0.9400** | 0.9200 | 0.3000 | 0.9200 |
| **索引时间** | 33.8s | **2.8s** | 2.8s | **2.8s** |
| **平均查询时间** | 166.0ms | 4.5ms | 30.2ms | **4.5ms** |

**核心结论**：
- 🏆 **Auto MRR 最优**（0.91 vs 0.85 Embedding），甚至略超 FTS5（0.91 vs 0.905）
- Auto 自动路由到 FTS5（code 扁平结构），彻底规避 Tree 在代码上的严重退化（MRR 0.28）
- TreeSearch 查询速度快 **37x** — 毫秒级 vs 百毫秒级

### 多跳推理（HotpotQA）

基于 [HotpotQA](https://huggingface.co/datasets/hotpot_qa) 数据集评测（50 queries，多跳问答）：

| 指标 | FTS5 | Tree | **Auto** |
|------|------|------|---------|
| **MRR** | 0.9712 | 0.9115 | **1.0000** |
| **SP-Recall@3** | 0.9939 | 0.9879 | **1.0000** |
| **2-hop-Cov@3** | 0.9939 | 0.9879 | **1.0000** |
| **SP-Recall@5** | 1.0000 | 1.0000 | **1.0000** |
| **平均查询时间** | 6ms | 3ms | 13ms |

**核心结论**：
- 🏆 **Auto MRR 满分 1.0** — 自动路由到 FTS5（浅树文本），完美覆盖所有多跳问题
- Tree 在浅树文档（句子列表）上略低于 FTS5 — 无结构信号时 reranking 引入轻微噪声，符合预期
- Auto 完全规避 Tree 在浅树上的退化

### 总结

> **Auto Mode 是生产环境的最优选择**：自动识别文档类型，永远走最优路径，零配置、不踩坑。

| 评测 | 最优模式 | MRR | vs Embedding | 查询速度 |
|------|----------|-----|-------------|---------|
| **QASPER**（学术论文） | Auto = Tree | **0.5046** | +19% | **190x 更快** |
| **FinanceBench**（SEC 财报） | Auto = FTS5 | **0.2420** | +10% | **29x 更快** |
| **CodeSearchNet**（Python） | Auto = FTS5 | **0.9100** | +7% | **37x 更快** |
| **HotpotQA**（多跳推理） | Auto = FTS5 | **1.0000** | — | 极速 |

自行运行评测：
```bash
# 文档检索（QASPER）
python examples/benchmark/qasper_benchmark.py --max-samples 50 --max-papers 20 --with-embedding

# 金融文档检索（FinanceBench）
python examples/benchmark/financebench_benchmark.py --max-samples 50 --with-embedding

# 代码检索（CodeSearchNet）
python examples/benchmark/codesearchnet_benchmark.py --max-samples 50 --max-corpus 500 --with-embedding

# 多跳推理（HotpotQA）
python examples/benchmark/hotpotqa_benchmark.py --max-samples 50
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

# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自定义 jieba 词表 demo —— 提升中文领域术语检索准确率。

适用场景：默认 jieba 词表不认识业务术语 / 品牌名 / 多字实体，会把
"图神经网络"切成"图 + 神经网络"，"知识蒸馏"切成"知识 + 蒸馏"，
导致包含"神经网络"的无关文档反而被召回到前面。

本 demo 演示三种注入自定义词表的方式：
  1. TreeSearchConfig.jieba_user_words —— 内存中直接给词
  2. TreeSearchConfig.jieba_user_dict_paths —— jieba 标准词典文件
  3. 环境变量 TREESEARCH_JIEBA_USER_DICT / TREESEARCH_JIEBA_USER_WORDS

词表文件格式（jieba 原生）：每行 `词语 [词频] [词性]`，例如:
    图神经网络 9000 n
    知识蒸馏 9000 n
    向量数据库 8000 n

Usage:
    python examples/09_jieba_userdict_demo.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import TreeSearch
from treesearch.config import TreeSearchConfig, set_config, reset_config
from treesearch.tokenizer import reset_jieba, tokenize


DOCS = {
    "doc_gnn.md": """\
# 图神经网络在推荐系统中的应用

本文系统介绍图神经网络如何建模用户-物品交互图。
图神经网络通过消息传递机制聚合邻居节点信息,
在节点分类与链接预测任务上表现出色。
""",
    "doc_distill.md": """\
# 知识蒸馏综述

知识蒸馏是一种模型压缩技术, 通过让学生网络模仿教师网络的
软标签输出, 在保持精度的同时显著降低推理成本。
""",
    "doc_general_nn.md": """\
# 神经网络基础

神经网络是深度学习的核心。常见的神经网络包括多层感知机、
卷积神经网络、循环神经网络等, 用于图像、语音和文本任务。
""",
    "doc_general_kd.md": """\
# 知识管理与蒸馏工艺

本文与机器学习无关, 主要讨论知识管理体系以及化工蒸馏工艺
中的常见操作步骤。
""",
}


def make_workspace() -> tuple[str, str]:
    """创建一份临时语料 + 一个独立的 user dict 文件。"""
    workdir = tempfile.mkdtemp(prefix="ts_jieba_example_")
    corpus = os.path.join(workdir, "docs")
    os.makedirs(corpus, exist_ok=True)
    for name, text in DOCS.items():
        with open(os.path.join(corpus, name), "w", encoding="utf-8") as f:
            f.write(text)

    user_dict_path = os.path.join(workdir, "domain_terms.txt")
    with open(user_dict_path, "w", encoding="utf-8") as f:
        f.write("图神经网络 9000 n\n")
        f.write("知识蒸馏 9000 n\n")
    return corpus, user_dict_path


def show_search(label: str, query: str, ts: TreeSearch, top_k: int = 3) -> None:
    print(f"\n[{label}] query={query!r}")
    print(f"  分词结果: {tokenize(query)}")
    result = ts.search(query, top_k_docs=top_k)
    for i, doc in enumerate(result["documents"], 1):
        name = os.path.basename(doc["doc_id"])
        best = max((n.get("score", 0.0) for n in doc.get("nodes", [])), default=0.0)
        print(f"  #{i}  {name:25s}  best_node_score={best:.4f}")


def demo_baseline(corpus: str) -> None:
    print("=" * 60)
    print("Demo 1: 默认 jieba（无自定义词表）")
    print("=" * 60)
    reset_config()
    reset_jieba()
    ts = TreeSearch(corpus, db_path=None)
    show_search("baseline", "图神经网络", ts)
    show_search("baseline", "知识蒸馏", ts)


def demo_inline_words(corpus: str) -> None:
    print("\n" + "=" * 60)
    print("Demo 2: 内存中注入词条（jieba_user_words） —— 三种写法都支持")
    print("=" * 60)
    reset_config()
    reset_jieba()
    set_config(TreeSearchConfig(
        jieba_user_words=[
            "图神经网络",                # 纯词，等价于 jieba.add_word("图神经网络")
            "知识蒸馏 9000",             # 词 + 词频
            "消息传递机制 8000 n",        # 词 + 词频 + 词性
        ],
    ))
    ts = TreeSearch(corpus, db_path=None)
    show_search("inline-words", "图神经网络", ts)
    show_search("inline-words", "知识蒸馏", ts)


def demo_dict_file(corpus: str, user_dict_path: str) -> None:
    print("\n" + "=" * 60)
    print("Demo 3: 加载词典文件（jieba_user_dict_paths）")
    print("=" * 60)
    print(f"  词典文件: {user_dict_path}")
    reset_config()
    reset_jieba()
    set_config(TreeSearchConfig(jieba_user_dict_paths=[user_dict_path]))
    ts = TreeSearch(corpus, db_path=None)
    show_search("dict-file", "图神经网络", ts)
    show_search("dict-file", "知识蒸馏", ts)


def demo_env_var(corpus: str, user_dict_path: str) -> None:
    print("\n" + "=" * 60)
    print("Demo 4: 通过环境变量启用（TREESEARCH_JIEBA_USER_DICT）")
    print("=" * 60)
    os.environ["TREESEARCH_JIEBA_USER_DICT"] = user_dict_path
    os.environ["TREESEARCH_JIEBA_USER_WORDS"] = "向量数据库 8000 n,大模型 8000 n"
    reset_config()
    reset_jieba()
    ts = TreeSearch(corpus, db_path=None)
    show_search("env-var", "图神经网络", ts)
    show_search("env-var", "向量数据库选型对比", ts)
    os.environ.pop("TREESEARCH_JIEBA_USER_DICT", None)
    os.environ.pop("TREESEARCH_JIEBA_USER_WORDS", None)


def demo_jieba_official_snippet() -> None:
    """复刻 jieba 官方文档示例：load_userdict + add_word + del_word。"""
    print("\n" + "=" * 60)
    print("Demo 5: 复刻 jieba 官方示例（add_word / del_word / load_userdict）")
    print("=" * 60)

    workdir = tempfile.mkdtemp(prefix="ts_jieba_official_")
    userdict_path = os.path.join(workdir, "userdict.txt")
    with open(userdict_path, "w", encoding="utf-8") as f:
        f.write("云计算 5\n")
        f.write("李小福 2 nr\n")
        f.write("创新办 3 i\n")
        f.write("easy_install 3 eng\n")
        f.write("好用 300\n")
        f.write("韩玉赏鉴 3 nz\n")
        f.write("八一双鹿 3 nz\n")
        f.write("台中\n")           # 纯词，无词频/词性
        f.write("自定义词\n")        # 纯词，稍后会被 del_word 删掉

    test_sent = (
        "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
        "例如我输入一个带\u201c韩玉赏鉴\u201d的标题，在自定义词库中也增加了此词为N类\n"
        "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
    )

    reset_config()
    reset_jieba()
    set_config(TreeSearchConfig(
        jieba_user_dict_paths=[userdict_path],
        jieba_user_words=["石墨烯", "凱特琳"],   # 纯加词
        jieba_del_words=["自定义词"],            # 删掉
    ))

    print("\n>> tokenize() 结果：")
    print("/".join(tokenize(test_sent, use_stemmer=False, remove_stopwords=False)))

    shutil.rmtree(workdir, ignore_errors=True)


def demo_token_diff() -> None:
    print("\n" + "=" * 60)
    print("Demo 6: 分词效果对比（核心原理）")
    print("=" * 60)
    queries = ["图神经网络在推荐系统中的应用", "知识蒸馏综述", "向量数据库选型"]

    reset_config()
    reset_jieba()
    print("\n>> 默认 jieba")
    for q in queries:
        print(f"  {q}  ->  {tokenize(q)}")

    set_config(TreeSearchConfig(
        jieba_user_words=[
            "图神经网络 9000 n",
            "知识蒸馏 9000 n",
            "向量数据库 8000 n",
            "推荐系统 8000 n",
        ],
    ))
    reset_jieba()
    print("\n>> 启用自定义词表后")
    for q in queries:
        print(f"  {q}  ->  {tokenize(q)}")


def main() -> None:
    corpus, user_dict_path = make_workspace()
    try:
        demo_baseline(corpus)
        demo_inline_words(corpus)
        demo_dict_file(corpus, user_dict_path)
        demo_env_var(corpus, user_dict_path)
        demo_jieba_official_snippet()
        demo_token_diff()
    finally:
        shutil.rmtree(os.path.dirname(corpus), ignore_errors=True)
        reset_config()
        reset_jieba()


if __name__ == "__main__":
    main()

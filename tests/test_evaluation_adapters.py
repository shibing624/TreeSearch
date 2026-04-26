import json
import asyncio


def test_graphrag_bench_adapter_writes_official_compatible_rows(tmp_path):
    from evaluation.graphrag_bench import run_graphrag_bench

    bench_dir = tmp_path / "bench"
    corpus_dir = bench_dir / "Datasets" / "Corpus"
    question_dir = bench_dir / "Datasets" / "Questions"
    corpus_dir.mkdir(parents=True)
    question_dir.mkdir(parents=True)

    (corpus_dir / "medical.json").write_text(
        json.dumps(
            {
                "corpus_name": "Medical",
                "context": "# Basal cell carcinoma\nBasal cell carcinoma is the most common type of skin cancer.",
            }
        ),
        encoding="utf-8",
    )
    (question_dir / "medical_questions.json").write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "source": "Medical",
                    "question": "What is the most common type of skin cancer?",
                    "answer": "Basal cell carcinoma is the most common type of skin cancer.",
                    "question_type": "Fact Retrieval",
                    "evidence": "Basal cell carcinoma is the most common type of skin cancer.",
                    "evidence_relations": "Basal cell carcinoma is the most common type of skin cancer.",
                }
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "predictions.json"
    summary = run_graphrag_bench(
        benchmark_dir=bench_dir,
        subset="medical",
        output_path=output_path,
        work_dir=tmp_path / "work",
        limit=1,
    )

    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["num_questions"] == 1
    assert summary["metrics"]["accuracy"] == 1.0
    assert summary["metrics"]["r"] == 1.0
    assert summary["metrics"]["ar"] == 1.0
    assert rows[0]["ground_truth"] == rows[0]["answer"]
    assert rows[0]["generated_answer"]
    assert rows[0]["context"]
    assert rows[0]["contexts"] == rows[0]["context"]
    assert rows[0]["evidences"] == [rows[0]["evidence"]]


def test_zhipu_langchain_embeddings_wraps_local_client():
    from evaluation.graphrag_bench import ZhipuLangChainEmbeddings

    class FakeEmbeddingClient:
        def embed(self, texts, batch_size=10):
            return [[float(len(text))] for text in texts]

    embeddings = ZhipuLangChainEmbeddings(client=FakeEmbeddingClient())

    assert embeddings.embed_documents(["a", "abcd"]) == [[1.0], [4.0]]
    assert embeddings.embed_query("abc") == [3.0]
    assert asyncio.run(embeddings.aembed_query("abcde")) == [5.0]


def test_graphrag_bench_extractive_answer_skips_headings_and_bad_first_sentence():
    from evaluation.graphrag_bench import _extractive_answer

    contexts = [
        "Medical chunk 9\n\nBiochemical tests measure hormones. "
        "Squamous cell skin cancer is the second most common type of skin cancer, after basal cell carcinoma.",
        "Medical chunk 0\n\nBasal cell skin cancer, also known as basal cell carcinoma, "
        "is the most common type of skin cancer.",
    ]

    answer = _extractive_answer("What is the most common type of skin cancer?", contexts, fallback="")

    assert answer.startswith("Basal cell skin cancer")


def test_repoqa_snf_adapter_outputs_compute_score_compatible_rows(tmp_path):
    from evaluation.repoqa_snf import run_repoqa_snf

    dataset = {
        "python": [
            {
                "repo": "demo/repo",
                "commit_sha": "abc",
                "entrypoint_path": ".",
                "topic": "math",
                "content": {
                    "math_utils.py": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n",
                },
                "dependency": {"math_utils.py": []},
                "functions": {
                    "math_utils.py": [
                        {
                            "name": "add",
                            "start_line": 0,
                            "end_line": 3,
                            "start_byte": 0,
                            "end_byte": 61,
                            "description": "Add two numbers and return the sum.",
                        }
                    ]
                },
                "needles": [
                    {
                        "name": "add",
                        "path": "math_utils.py",
                        "start_line": 0,
                        "end_line": 3,
                        "start_byte": 0,
                        "end_byte": 61,
                        "description": "Add two numbers and return the sum.",
                    }
                ],
            }
        ]
    }
    dataset_path = tmp_path / "repoqa.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    output_path = tmp_path / "repoqa_outputs.jsonl"
    summary = run_repoqa_snf(
        dataset_path=dataset_path,
        output_path=output_path,
        work_dir=tmp_path / "work",
        limit=1,
        use_official_score=False,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["num_tasks"] == 1
    assert summary["retrieval"]["hit@1"] == 1.0
    assert summary["official_score"]["threshold"] == 0.8
    assert rows[0]["language"] == "python"
    assert rows[0]["repo"] == "demo/repo"
    assert rows[0]["name"] == "add"
    assert rows[0]["output"][0].startswith("```python")


def test_codesearchnet_wrapper_uses_evaluation_cache_dir():
    from evaluation.codesearchnet_eval import default_paths

    paths = default_paths("python")

    assert "evaluation/data/codesearchnet_cache" in str(paths.cache_dir)
    assert "evaluation/output/codesearchnet" in str(paths.output_dir)
    assert "evaluation/output/indexes/codesearchnet" in str(paths.index_dir)

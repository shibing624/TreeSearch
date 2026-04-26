# -*- coding: utf-8 -*-
"""RepoQA SNF retrieval adapter for TreeSearch-style code localization.

This script consumes the official RepoQA dataset JSON, ranks candidate
functions from each repository using the needle description, emits
``repoqa.compute_score`` compatible JSONL, and optionally invokes the official
scorer from the local RepoQA checkout under ``evaluation/data/repoqa``.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "evaluation" / "data" / "repoqa-2024-06-23.json"
DEFAULT_REPOQA_DIR = REPO_ROOT / "evaluation" / "data" / "repoqa"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "output" / "repoqa"


@dataclass
class FunctionCandidate:
    name: str
    path: str
    code: str
    description: str


def load_repoqa_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, list[dict[str, Any]]]:
    return json.loads(Path(dataset_path).read_text(encoding="utf-8"))


def iter_tasks(
    dataset: dict[str, list[dict[str, Any]]],
    languages: set[str] | None = None,
    limit: int | None = None,
):
    emitted = 0
    for language, repos in dataset.items():
        if languages is not None and language not in languages:
            continue
        for repo in repos:
            for needle in repo.get("needles", []):
                yield language, repo, needle
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def extract_function_candidates(repo: dict[str, Any]) -> list[FunctionCandidate]:
    candidates = []
    contents = repo["content"]
    for path, functions in repo.get("functions", {}).items():
        file_text = contents[path]
        lines = file_text.splitlines()
        for fn in functions:
            code = "\n".join(lines[int(fn["start_line"]): int(fn["end_line"])])
            candidates.append(
                FunctionCandidate(
                    name=str(fn["name"]),
                    path=path,
                    code=code,
                    description=str(fn.get("description", "")),
                )
            )
    return candidates


def rank_candidates(description: str, candidates: list[FunctionCandidate]) -> list[tuple[FunctionCandidate, float]]:
    query_tokens = _tokens(description)
    ranked = []
    for candidate in candidates:
        haystack = " ".join([candidate.name, candidate.description, candidate.code])
        candidate_tokens = _tokens(haystack)
        overlap = len(query_tokens & candidate_tokens)
        overlap_score = overlap / max(len(query_tokens), 1)
        description_score = difflib.SequenceMatcher(
            None,
            " ".join(description.lower().split()),
            " ".join(candidate.description.lower().split()),
        ).ratio()
        score = (2.0 * description_score) + overlap_score
        ranked.append((candidate, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def run_repoqa_snf(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    repoqa_dir: str | Path = DEFAULT_REPOQA_DIR,
    languages: list[str] | None = None,
    limit: int | None = None,
    use_official_score: bool = True,
    model_name: str = "treesearch-repoqa-retriever",
) -> dict[str, Any]:
    dataset = load_repoqa_dataset(dataset_path)
    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT_DIR / "treesearch_repoqa_outputs.jsonl"
    work_dir = Path(work_dir) if work_dir else DEFAULT_OUTPUT_DIR / "work"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    selected_languages = set(languages) if languages else None

    rows = []
    hits = 0
    for language, repo, needle in iter_tasks(dataset, languages=selected_languages, limit=limit):
        candidates = extract_function_candidates(repo)
        ranked = rank_candidates(str(needle["description"]), candidates)
        predicted = ranked[0][0] if ranked else FunctionCandidate("", "", "", "")
        if predicted.name == needle["name"]:
            hits += 1
        rows.append(
            {
                "repo": repo["repo"],
                "name": needle["name"],
                "language": language,
                "path": needle["path"],
                "position_ratio": 0.5,
                "description": f"\nFunction Description:{needle['description']}\n",
                "needle_token_start": 0,
                "needle_token_end": 0,
                "code_context_ntokens": 0,
                "predicted_name": predicted.name,
                "retrieval_score": ranked[0][1] if ranked else 0.0,
                "output": [_code_block(language, predicted.code)],
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    retrieval = {
        "hit@1": hits / len(rows) if rows else 0.0,
        "num_hits": hits,
    }
    official_score = {"threshold": 0.8, "pass@1": None, "status": "not_requested"}
    if use_official_score:
        official_score = run_official_repoqa_score(
            repoqa_dir=repoqa_dir,
            dataset=dataset,
            output_rows=rows,
            model_name=model_name,
        )

    summary = {
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "num_tasks": len(rows),
        "retrieval": retrieval,
        "official_score": official_score,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_official_repoqa_score(
    repoqa_dir: str | Path,
    dataset: dict[str, list[dict[str, Any]]],
    output_rows: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    repoqa_dir = Path(repoqa_dir)
    sys.path.insert(0, str(repoqa_dir))
    try:
        compute_score_module = importlib.import_module("repoqa.compute_score")
    except ImportError as exc:
        raise RuntimeError(
            "RepoQA official scorer dependencies are missing. Install evaluation/data/repoqa/requirements.txt."
        ) from exc
    output_json = compute_score_module.compute_score(model_name, dataset, output_rows, ignore_comments=False)
    scores = output_json[model_name]["scores"]["all"]
    threshold_scores = scores[0.8]
    return {
        "threshold": 0.8,
        "pass@1": float(threshold_scores["pass@1"]),
        "status": "official_compute_score",
    }


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()) if len(token) > 2}


def _code_block(language: str, code: str) -> str:
    return f"```{language}\n{code.strip()}\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TreeSearch-style retrieval on RepoQA SNF")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--repoqa-dir", type=Path, default=DEFAULT_REPOQA_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-official-score", action="store_true")
    args = parser.parse_args()

    summary = run_repoqa_snf(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        work_dir=args.work_dir,
        repoqa_dir=args.repoqa_dir,
        languages=args.languages,
        limit=args.limit,
        use_official_score=not args.skip_official_score,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

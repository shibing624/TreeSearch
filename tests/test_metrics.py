# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.metrics evaluation module.
"""
import pytest
from treesearch.metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    reciprocal_rank,
    ndcg_at_k,
    f1_at_k,
    evaluate_query,
    evaluate_benchmark,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["a", "x", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)

    def test_k_greater_than_retrieved(self):
        assert precision_at_k(["a"], ["a", "b"], 3) == pytest.approx(1 / 3)

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], 3) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], ["a", "b", "c"], 2) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        assert recall_at_k(["a"], [], 3) == 0.0


class TestHitAtK:
    def test_hit(self):
        assert hit_at_k(["x", "a", "y"], ["a"], 3) is True

    def test_miss(self):
        assert hit_at_k(["x", "y", "z"], ["a"], 3) is False

    def test_hit_at_1(self):
        assert hit_at_k(["a", "x"], ["a"], 1) is True

    def test_miss_at_1(self):
        assert hit_at_k(["x", "a"], ["a"], 1) is False


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5

    def test_not_found(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_multiple_relevant(self):
        assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


class TestNdcgAtK:
    def test_perfect_ndcg(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_partial_ndcg(self):
        val = ndcg_at_k(["x", "a"], ["a", "b"], 2)
        assert 0.0 < val < 1.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a"], [], 3) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], ["a"], 0) == 0.0


class TestF1AtK:
    def test_perfect_f1(self):
        assert f1_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_zero_f1(self):
        assert f1_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_partial_f1(self):
        val = f1_at_k(["a", "x", "y"], ["a", "b"], 3)
        assert 0.0 < val < 1.0


class TestEvaluateQuery:
    def test_basic(self):
        result = evaluate_query(["a", "b", "c"], ["a", "c"], [1, 3])
        assert "mrr" in result
        assert "precision@1" in result
        assert "recall@3" in result
        assert result["mrr"] == 1.0
        assert result["recall@3"] == 1.0

    def test_empty_retrieved(self):
        result = evaluate_query([], ["a"], [1])
        assert result["mrr"] == 0.0
        assert result["hit@1"] == 0.0


class TestEvaluateBenchmark:
    def test_average(self):
        qr = [
            {"retrieved": ["a", "b"], "relevant": ["a"]},
            {"retrieved": ["x", "a"], "relevant": ["a"]},
        ]
        avg = evaluate_benchmark(qr, [1, 3])
        assert avg["mrr"] == pytest.approx(0.75)  # (1.0 + 0.5) / 2
        assert avg["hit@1"] == pytest.approx(0.5)  # (1 + 0) / 2

    def test_empty(self):
        assert evaluate_benchmark([]) == {}

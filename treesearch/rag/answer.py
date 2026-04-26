# -*- coding: utf-8 -*-
"""Grounded answer generation for GraphRAG."""

from treesearch.rag.models import EvidenceChain, GroundedAnswer, VerificationResult


class TemplateAnswerGenerator:
    """No-LLM answer generator for deterministic tests and smoke demos."""

    def generate(
        self,
        query: str,
        chain: EvidenceChain,
        verification: VerificationResult,
    ) -> GroundedAnswer:
        answer = "\n".join(chain.reasoning_chain)
        return GroundedAnswer(
            query=query,
            answer=answer,
            evidence_chain=chain,
            verification=verification,
        )

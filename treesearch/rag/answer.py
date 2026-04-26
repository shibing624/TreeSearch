# -*- coding: utf-8 -*-
"""Grounded answer generation for GraphRAG."""

from treesearch.rag.models import EvidenceChain, GroundedAnswer, VerificationResult
from treesearch.rag.llm import LLMClient, OpenAIChatClient


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


class LLMAnswerGenerator:
    """Generate a grounded answer from verified evidence with an OpenAI-compatible LLM."""

    def __init__(self, llm_client: LLMClient | None = None, model: str | None = None):
        self.llm_client = llm_client or OpenAIChatClient()
        self.model = model

    def generate(
        self,
        query: str,
        chain: EvidenceChain,
        verification: VerificationResult,
    ) -> GroundedAnswer:
        evidence = "\n".join(chain.reasoning_chain)
        citation_text = "\n".join(
            f"- {citation.source_path}:{citation.line_start}-{citation.line_end}"
            for citation in chain.citations
        )
        answer = self.llm_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided evidence. "
                        "Keep the answer concise and grounded in the citations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{query}\n\nEvidence:\n{evidence}\n\nCitations:\n{citation_text}",
                },
            ],
            model=self.model,
        )
        return GroundedAnswer(
            query=query,
            answer=answer,
            evidence_chain=chain,
            verification=verification,
        )

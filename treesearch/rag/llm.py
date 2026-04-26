# -*- coding: utf-8 -*-
"""LLM client interfaces for GraphRAG reranking and answer generation."""

import json
import os
import urllib.request
from pathlib import Path
from typing import Protocol


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        ...


class FakeLLMClient:
    """Deterministic LLM client for tests."""

    def __init__(self, response: str | list[str]):
        self.response = response
        self.calls: list[dict] = []

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        self.calls.append({"messages": messages, "model": model})
        if isinstance(self.response, list):
            index = len(self.calls) - 1
            return self.response[index]
        return self.response


class OpenAIChatClient:
    """Small OpenAI-compatible chat client backed by the repository .env file."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        env_path: str | Path | None = None,
    ):
        env_path = env_path or Path(__file__).resolve().parents[2] / ".env"
        env = load_env_file(env_path)
        self.api_key = api_key or env.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"]
        self.base_url = (
            base_url
            or env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or env.get("OPENAI_MODEL") or env.get("OPENAI_API_MODEL") or "gpt-4o-mini"

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values

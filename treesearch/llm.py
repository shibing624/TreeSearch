# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async-first LLM client with singleton connection pool, retry, token counting, and JSON extraction.
"""
import asyncio
import json
import logging
import os
from typing import Optional, Any

import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global defaults from env
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEFAULT_MODEL = os.getenv("TREESEARCH_MODEL", "gpt-4o-mini")
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Singleton async client (connection pool reuse)
# ---------------------------------------------------------------------------
_async_clients: dict[str, openai.AsyncOpenAI] = {}


def _get_async_client(api_key: Optional[str] = None) -> openai.AsyncOpenAI:
    """Return a singleton AsyncOpenAI client keyed by (api_key, base_url)."""
    key = api_key or OPENAI_API_KEY or ""
    cache_key = f"{key}::{OPENAI_BASE_URL or ''}"
    if cache_key not in _async_clients:
        kw: dict[str, Any] = {"api_key": key}
        if OPENAI_BASE_URL:
            kw["base_url"] = OPENAI_BASE_URL
        _async_clients[cache_key] = openai.AsyncOpenAI(**kw)
    return _async_clients[cache_key]


# ---------------------------------------------------------------------------
# Token counting (with encoder cache)
# ---------------------------------------------------------------------------
_encoder_cache: dict[str, tiktoken.Encoding] = {}


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count tokens using tiktoken with cached encoder."""
    if not text:
        return 0
    if model not in _encoder_cache:
        _encoder_cache[model] = tiktoken.encoding_for_model(model)
    return len(_encoder_cache[model].encode(text))


# ---------------------------------------------------------------------------
# Core async LLM call (single implementation)
# ---------------------------------------------------------------------------

async def _achat_impl(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0,
    messages: Optional[list[dict]] = None,
) -> openai.types.chat.ChatCompletion:
    """Internal: async chat completion with retry. Returns raw response."""
    if messages is not None:
        msgs = list(messages) + [{"role": "user", "content": prompt}]
    else:
        msgs = [{"role": "user", "content": prompt}]

    client = _get_async_client(api_key)
    for attempt in range(MAX_RETRIES):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("LLM retry %d/%d: %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(2 ** attempt, 8))
            else:
                logger.error("LLM max retries reached")
                raise


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def achat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0,
    messages: Optional[list[dict]] = None,
) -> str:
    """Async chat completion. Returns the assistant message content."""
    resp = await _achat_impl(prompt, model=model, api_key=api_key,
                             temperature=temperature, messages=messages)
    return resp.choices[0].message.content


async def achat_with_finish_reason(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0,
    messages: Optional[list[dict]] = None,
) -> tuple[str, str]:
    """Async chat returning (content, finish_reason). finish_reason is 'finished' or 'max_output_reached'."""
    resp = await _achat_impl(prompt, model=model, api_key=api_key,
                             temperature=temperature, messages=messages)
    content = resp.choices[0].message.content
    reason = "max_output_reached" if resp.choices[0].finish_reason == "length" else "finished"
    return content, reason


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

def chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0,
    messages: Optional[list[dict]] = None,
) -> str:
    """Sync wrapper around :func:`achat`."""
    return asyncio.run(achat(prompt, model=model, api_key=api_key,
                             temperature=temperature, messages=messages))


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(content: str) -> dict | list:
    """
    Extract JSON from LLM response.

    Handles ```json fences, trailing commas, Python None literals.
    """
    if not content:
        return {}
    try:
        # Strip ```json ... ``` fences
        start = content.find("```json")
        if start != -1:
            start += 7
            end = content.rfind("```")
            text = content[start:end].strip()
        else:
            text = content.strip()

        text = text.replace("None", "null")
        text = text.replace("\n", " ").replace("\r", " ")
        text = " ".join(text.split())
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            text = text.replace(",]", "]").replace(",}", "}")
            return json.loads(text)
        except Exception:
            logger.error("Failed to parse JSON from LLM response")
            return {}
    except Exception:
        return {}

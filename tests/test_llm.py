# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.llm module.
"""
import json
import os
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from treesearch.llm import achat, achat_with_finish_reason, chat, count_tokens, extract_json


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        count = count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hello")
        long = count_tokens("Hello, this is a much longer sentence with many words.")
        assert long > short

    def test_encoder_cache(self):
        """Repeated calls should reuse cached encoder."""
        from treesearch.llm import _encoder_cache
        count_tokens("test", model="gpt-4o-mini")
        assert "gpt-4o-mini" in _encoder_cache


class TestExtractJson:
    def test_plain_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = '```json\n{"items": [1, 2, 3]}\n```'
        result = extract_json(text)
        assert result == {"items": [1, 2, 3]}

    def test_json_with_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        result = extract_json(text)
        assert result == {"a": 1, "b": 2}

    def test_empty_string(self):
        assert extract_json("") == {}

    def test_invalid_json(self):
        result = extract_json("not json at all")
        assert result == {}

    def test_json_array(self):
        text = '[{"id": 1}, {"id": 2}]'
        result = extract_json(text)
        assert isinstance(result, list)
        assert len(result) == 2


class TestAchat:
    @pytest.mark.asyncio
    async def test_achat_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LLM response"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("treesearch.llm._get_async_client", return_value=mock_client):
            result = await achat("test prompt", api_key="fake-key")
            assert result == "LLM response"

    @pytest.mark.asyncio
    async def test_achat_with_messages(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response with context"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("treesearch.llm._get_async_client", return_value=mock_client):
            result = await achat(
                "follow up",
                api_key="fake-key",
                messages=[{"role": "system", "content": "You are helpful."}],
            )
            assert result == "response with context"


class TestAchatWithFinishReason:
    @pytest.mark.asyncio
    async def test_finished(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "complete response"
        mock_response.choices[0].finish_reason = "stop"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("treesearch.llm._get_async_client", return_value=mock_client):
            content, reason = await achat_with_finish_reason("test", api_key="fake-key")
            assert content == "complete response"
            assert reason == "finished"

    @pytest.mark.asyncio
    async def test_max_output_reached(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "truncated..."
        mock_response.choices[0].finish_reason = "length"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("treesearch.llm._get_async_client", return_value=mock_client):
            content, reason = await achat_with_finish_reason("test", api_key="fake-key")
            assert reason == "max_output_reached"


class TestChatSync:
    def test_chat_calls_achat(self):
        with patch("treesearch.llm.achat", new_callable=AsyncMock, return_value="sync result"):
            result = chat("test prompt", api_key="fake-key")
            assert result == "sync result"


class TestSingletonClient:
    def test_same_key_returns_same_client(self):
        from treesearch.llm import _get_async_client, _async_clients
        _async_clients.clear()
        c1 = _get_async_client(api_key="test-key-1")
        c2 = _get_async_client(api_key="test-key-1")
        assert c1 is c2
        _async_clients.clear()

    def test_different_keys_return_different_clients(self):
        from treesearch.llm import _get_async_client, _async_clients
        _async_clients.clear()
        c1 = _get_async_client(api_key="key-a")
        c2 = _get_async_client(api_key="key-b")
        assert c1 is not c2
        _async_clients.clear()

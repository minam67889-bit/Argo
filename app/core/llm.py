"""LLM client wrapper. Thin layer over the OpenAI SDK to support any
OpenAI-compatible endpoint (OpenRouter, Ollama, vLLM, llama.cpp server, etc.)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAIError

from app.core.config import settings


class LLMClient:
    """Async client. One instance can serve many concurrent requests."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = base_url or settings.LLM_BASE_URL
        self.timeout = timeout or settings.LLM_TIMEOUT
        if not self.api_key:
            self.api_key = "sk-no-key"
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=2,
        )

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Non-streaming chat completion. Returns the full message dict.

        tools: list of OpenAI-format tool schemas (function calling).
        tool_choice: "auto" | "none" | {"type": "function", "function": {"name": ...}}
        """
        kwargs: Dict[str, Any] = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.AGENT_TEMPERATURE,
            "top_p": top_p if top_p is not None else 0.9,
            "max_tokens": max_tokens or settings.AGENT_MAX_TOKENS,
            "stream": False,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

        msg = resp.choices[0].message
        # Extract native tool_calls if present
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })

        return {
            "role": msg.role,
            "content": msg.content or "",
            "reasoning": getattr(msg, "reasoning_content", None),
            "finish_reason": resp.choices[0].finish_reason,
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            } if resp.usage else None,
        }

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming chat completion. Yields token deltas.

        Yields dicts like:
          {"type": "delta", "content": "..."}
          {"type": "reasoning", "content": "..."}
          {"type": "tool_call", "name": "...", "arguments": {...}}
          {"type": "done", "finish_reason": "...", "usage": {...}}
          {"type": "error", "message": "..."}
        """
        kwargs: Dict[str, Any] = {
            "model": model or settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.AGENT_TEMPERATURE,
            "top_p": top_p if top_p is not None else 0.9,
            "max_tokens": max_tokens or settings.AGENT_MAX_TOKENS,
            "stream": True,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason
                if getattr(delta, "reasoning_content", None):
                    yield {"type": "reasoning", "content": delta.reasoning_content}
                if delta.content:
                    yield {"type": "delta", "content": delta.content}
                # Native tool calls in streaming
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        try:
                            args = tc.function.arguments
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args) if args else {}
                                except Exception:
                                    args = {"raw": args}
                        except Exception:
                            args = {}
                        yield {
                            "type": "tool_call",
                            "name": tc.function.name or "",
                            "arguments": args or {},
                            "id": getattr(tc, "id", None),
                        }
                if finish:
                    yield {
                        "type": "done",
                        "finish_reason": finish,
                        "usage": None,
                    }
        except OpenAIError as e:
            yield {"type": "error", "message": str(e)}

    async def list_models(self) -> List[Dict[str, Any]]:
        """Try to fetch model list. Many providers don't support this — we fall back."""
        try:
            resp = await self._client.models.list()
            return [{"id": m.id, "owned_by": getattr(m, "owned_by", "")} for m in resp.data]
        except Exception:
            return []


def safe_json_loads(text: str) -> Optional[Any]:
    """Tolerant JSON parser. Returns None on failure."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

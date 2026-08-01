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
            # OpenAI SDK requires a non-empty key; use a placeholder for local servers.
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
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Non-streaming chat completion. Returns the full message dict."""
        try:
            resp = await self._client.chat.completions.create(
                model=model or settings.LLM_MODEL,
                messages=messages,
                temperature=temperature if temperature is not None else settings.AGENT_TEMPERATURE,
                max_tokens=max_tokens or settings.AGENT_MAX_TOKENS,
                stream=False,
            )
        except OpenAIError as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

        msg = resp.choices[0].message
        return {
            "role": msg.role,
            "content": msg.content or "",
            "reasoning": getattr(msg, "reasoning_content", None),
            "finish_reason": resp.choices[0].finish_reason,
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
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming chat completion. Yields token deltas.

        Yields dicts like:
          {"type": "delta", "content": "..."}
          {"type": "reasoning", "content": "..."}
          {"type": "done", "finish_reason": "...", "usage": {...}}
          {"type": "error", "message": "..."}
        """
        try:
            stream = await self._client.chat.completions.create(
                model=model or settings.LLM_MODEL,
                messages=messages,
                temperature=temperature if temperature is not None else settings.AGENT_TEMPERATURE,
                max_tokens=max_tokens or settings.AGENT_MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason
                if getattr(delta, "reasoning_content", None):
                    yield {"type": "reasoning", "content": delta.reasoning_content}
                if delta.content:
                    yield {"type": "delta", "content": delta.content}
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
    # Strip code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # drop first ```json / ``` and last ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

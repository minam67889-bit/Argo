"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CreateChatRequest(BaseModel):
    title: str = Field("چت جدید", max_length=200)
    mode: Literal["chat", "agent"] = "chat"
    model: Optional[str] = None
    workspace: Optional[str] = None


class UpdateChatRequest(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    workspace: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str] = None  # for tool messages
    tool_calls: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None


class SendMessageRequest(BaseModel):
    content: str
    mode: Optional[Literal["chat", "agent"]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    auto_approve: Optional[bool] = None
    workspace: Optional[str] = None


class SettingsUpdate(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    auto_approve: Optional[bool] = None
    max_steps: Optional[int] = None

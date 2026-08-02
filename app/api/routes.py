"""REST API routes."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.agent import AgentLoop
from app.api.schemas import (
    ChatMessage,
    CreateChatRequest,
    SendMessageRequest,
    SettingsUpdate,
    UpdateChatRequest,
)
from app.core import storage
from app.core.config import settings
from app.core.llm import LLMClient


router = APIRouter()


# ---- File upload helpers ----

# Allowed file extensions (text + code + zip + common docs)
ALLOWED_EXTS = {
    # Text
    ".txt", ".md", ".rst", ".log",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".xml", ".svg", ".sql",
    ".sh", ".bash", ".zsh", ".fish",
    ".java", ".kt", ".scala", ".groovy",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".go", ".rs", ".rb", ".php", ".pl", ".pm",
    ".swift", ".m", ".mm", ".dart", ".lua", ".r",
    # Archives
    ".zip",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB per file


def get_chat_uploads_dir(chat_id: str) -> Path:
    """Each chat has its own upload directory."""
    d = settings.ARGO_DB.parent / "uploads" / chat_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- Health & info ----

@router.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "model": settings.LLM_MODEL,
        "base_url": settings.LLM_BASE_URL,
        "has_api_key": bool(settings.LLM_API_KEY),
    }


@router.get("/api/models")
async def list_models():
    """Try to fetch model list from the configured provider."""
    try:
        client = LLMClient()
        models = await client.list_models()
        return {"models": models, "current": settings.LLM_MODEL}
    except Exception as e:
        return {"models": [], "current": settings.LLM_MODEL, "error": str(e)}


# ---- Settings ----

_runtime_settings: Dict[str, Any] = {}


@router.get("/api/settings")
async def get_settings():
    return {
        "model": settings.LLM_MODEL,
        "base_url": settings.LLM_BASE_URL,
        "has_api_key": bool(settings.LLM_API_KEY),
        "temperature": settings.AGENT_TEMPERATURE,
        "max_tokens": settings.AGENT_MAX_TOKENS,
        "max_steps": settings.AGENT_MAX_STEPS,
        "auto_approve": settings.AGENT_AUTO_APPROVE,
        "default_workspace": str(settings.ARGO_WORKSPACE),
        "runtime": _runtime_settings,
    }


@router.post("/api/settings")
async def update_settings(body: SettingsUpdate):
    """Update runtime settings. Persists to env on the fly (process scope)."""
    import os
    if body.api_key is not None:
        os.environ["LLM_API_KEY"] = body.api_key
        settings.LLM_API_KEY = body.api_key
        _runtime_settings["api_key_set"] = True
    if body.base_url is not None:
        os.environ["LLM_BASE_URL"] = body.base_url
        settings.LLM_BASE_URL = body.base_url
        _runtime_settings["base_url"] = body.base_url
    if body.model is not None:
        os.environ["LLM_MODEL"] = body.model
        settings.LLM_MODEL = body.model
    if body.temperature is not None:
        os.environ["AGENT_TEMPERATURE"] = str(body.temperature)
        settings.AGENT_TEMPERATURE = body.temperature
    if body.max_tokens is not None:
        os.environ["AGENT_MAX_TOKENS"] = str(body.max_tokens)
        settings.AGENT_MAX_TOKENS = body.max_tokens
    if body.auto_approve is not None:
        os.environ["AGENT_AUTO_APPROVE"] = "1" if body.auto_approve else "0"
        settings.AGENT_AUTO_APPROVE = body.auto_approve
    if body.max_steps is not None:
        os.environ["AGENT_MAX_STEPS"] = str(body.max_steps)
        settings.AGENT_MAX_STEPS = body.max_steps
    return {"ok": True}


# ---- Chats ----

@router.get("/api/chats")
async def api_list_chats():
    return {"chats": storage.list_chats()}


@router.post("/api/chats")
async def api_create_chat(body: CreateChatRequest):
    chat = storage.create_chat(
        title=body.title, mode=body.mode,
        model=body.model, workspace=body.workspace,
    )
    return chat


@router.get("/api/chats/{chat_id}")
async def api_get_chat(chat_id: str):
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    chat["messages"] = storage.list_messages(chat_id)
    return chat


@router.patch("/api/chats/{chat_id}")
async def api_update_chat(chat_id: str, body: UpdateChatRequest):
    storage.update_chat(chat_id, **body.model_dump(exclude_none=True))
    return storage.get_chat(chat_id)


@router.delete("/api/chats/{chat_id}")
async def api_delete_chat(chat_id: str):
    storage.delete_chat(chat_id)
    return {"ok": True}


# ---- Send a message (streaming SSE) ----

@router.post("/api/chats/{chat_id}/messages")
async def api_send_message(chat_id: str, body: SendMessageRequest, request: Request):
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")

    mode = body.mode or chat.get("mode") or "chat"
    model = body.model or chat.get("model")
    workspace = body.workspace or chat.get("workspace")
    if not workspace:
        workspace = str(settings.ARGO_WORKSPACE)
    workdir = Path(workspace).resolve()
    if not workdir.exists():
        workdir.mkdir(parents=True, exist_ok=True)

    # Link uploaded files into the workdir under "uploads/"
    # so the agent can access them via bash/read_file.
    uploads_dir = get_chat_uploads_dir(chat_id)
    workdir_uploads = workdir / "uploads"
    try:
        if uploads_dir.exists() and any(uploads_dir.iterdir()):
            workdir_uploads.mkdir(parents=True, exist_ok=True)
            # Symlink each file (so updates to uploads are visible)
            for src in uploads_dir.iterdir():
                if src.is_file():
                    link = workdir_uploads / src.name
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(src.resolve())
    except OSError:
        # Best effort — uploads may be on a different filesystem (Drive)
        # In that case, just copy
        try:
            workdir_uploads.mkdir(parents=True, exist_ok=True)
            for src in uploads_dir.iterdir():
                if src.is_file():
                    link = workdir_uploads / src.name
                    if link.exists():
                        link.unlink()
                    import shutil
                    shutil.copy2(src, link)
        except Exception:
            pass

    # Save the user message immediately
    user_msg = storage.add_message(chat_id, "user", body.content)

    # Build the agent loop with full history
    client = LLMClient()
    # Detect if endpoint supports native tools. Local llama.cpp / ollama (older) don't.
    base = settings.LLM_BASE_URL.lower()
    supports_native_tools = None  # auto-detect on first error
    if any(host in base for host in ("openrouter.ai", "api.openai.com", "api.deepseek.com")):
        supports_native_tools = True
    elif "localhost" in base or "127.0.0.1" in base:
        # Local servers typically don't support tools (llama.cpp, ollama)
        supports_native_tools = False

    agent = AgentLoop(
        llm=client,
        workdir=workdir,
        mode=mode,
        model=model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        auto_approve=body.auto_approve if body.auto_approve is not None else settings.AGENT_AUTO_APPROVE,
        bash_timeout=settings.AGENT_BASH_TIMEOUT,
        supports_native_tools=supports_native_tools,
    )
    # Load prior messages from DB (skip the one we just added; we'll add it manually)
    prior = storage.list_messages(chat_id)[:-1]  # exclude the just-added user msg
    for m in prior:
        msg: Dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("name"):
            msg["name"] = m["name"]
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        # Drop system messages from DB — agent builds its own
        if msg["role"] != "system":
            agent.messages.append(msg)
    # Add the current user message
    agent.add_user_message(body.content)

    # Update chat title from first user message if it's the default
    if chat.get("title") in ("چت جدید", "New chat", "") and body.content.strip():
        title = body.content.strip()[:60]
        if len(body.content.strip()) > 60:
            title += "..."
        storage.update_chat(chat_id, title=title)

    async def event_stream() -> AsyncIterator[bytes]:
        full_text = ""
        tool_calls_log: List[Dict[str, Any]] = []
        tool_results_log: List[Dict[str, Any]] = []
        last_assistant_msg_id: Optional[str] = None

        # Initial event with the user message id so the UI can update
        yield f"data: {json.dumps({'type': 'user_msg', 'id': user_msg['id']}, ensure_ascii=False)}\n\n".encode()

        try:
            async for ev in agent.run(body.content):
                if await request.is_disconnected():
                    break

                # Mirror events to the database: assistant text and tool calls
                if ev.type == "text":
                    full_text += ev.data.get("content", "")
                elif ev.type == "tool_call":
                    tool_calls_log.append({
                        "name": ev.data["name"],
                        "arguments": ev.data.get("arguments", {}),
                    })
                elif ev.type == "tool_result":
                    tool_results_log.append({
                        "name": ev.data["name"],
                        "output": ev.data.get("output", ""),
                        "error": ev.data.get("error", False),
                    })
                elif ev.type == "done":
                    # Save the assistant message (if any text or tool calls)
                    if full_text.strip() or tool_calls_log:
                        meta = {
                            "steps": ev.data.get("steps"),
                            "elapsed": ev.data.get("elapsed"),
                            "tokens": ev.data.get("tokens"),
                            "max_steps_reached": ev.data.get("max_steps_reached", False),
                            "tool_results": tool_results_log,
                        }
                        saved = storage.add_message(
                            chat_id, "assistant", full_text,
                            tool_calls=tool_calls_log if tool_calls_log else None,
                            meta=meta,
                        )
                        last_assistant_msg_id = saved["id"]
                        ev.data["message_id"] = saved["id"]

                yield ev.to_sse().encode()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            err_ev = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_ev, ensure_ascii=False)}\n\n".encode()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---- Workspaces ----

@router.get("/api/workspaces")
async def api_list_workspaces():
    """List directories under the parent of the default workspace."""
    base = settings.ARGO_WORKSPACE.parent
    if not base.exists():
        return {"workspaces": [str(settings.ARGO_WORKSPACE)]}
    workspaces = sorted(
        [str(p) for p in base.iterdir() if p.is_dir()],
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return {"workspaces": workspaces}


@router.post("/api/workspaces/create")
async def api_create_workspace(body: ChatMessage):
    """body.content is the path. Creates the directory."""
    path = Path(body.content)
    if not path.is_absolute():
        path = settings.ARGO_WORKSPACE.parent / path
    path = path.resolve()
    # Allow anywhere writable but require an absolute path
    if path == Path("/") or str(path) in ("/", str(Path.home())):
        raise HTTPException(400, "refusing to create workspace at that path")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(400, f"failed: {e}")
    return {"workspace": str(path)}


@router.get("/api/workspaces/browse")
async def api_browse_workspace(path: str = "."):
    """List a workspace's top-level contents (for the file browser in UI)."""
    p = Path(path)
    if not p.is_absolute():
        p = settings.ARGO_WORKSPACE / p
    p = p.resolve()
    if not p.exists():
        raise HTTPException(404, "not found")
    if not p.is_dir():
        raise HTTPException(400, "not a directory")
    try:
        entries = []
        for x in sorted(p.iterdir(), key=lambda y: (not y.is_dir(), y.name.lower())):
            try:
                entries.append({
                    "name": x.name,
                    "path": str(x.relative_to(settings.ARGO_WORKSPACE)) if x.is_relative_to(settings.ARGO_WORKSPACE) else str(x),
                    "is_dir": x.is_dir(),
                    "size": x.stat().st_size if x.is_file() else None,
                })
            except OSError:
                continue
        return {"path": str(p), "entries": entries}
    except PermissionError as e:
        raise HTTPException(403, str(e))


# ---- File upload ----

@router.post("/api/chats/{chat_id}/files")
async def api_upload_files(chat_id: str, files: List[UploadFile] = File(...)):
    """Upload one or more files to this chat. Files are saved under
    data/uploads/{chat_id}/ and become available to the agent as part
    of its workspace (via symlink or copy).
    """
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")

    uploads_dir = get_chat_uploads_dir(chat_id)
    saved = []
    errors = []

    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            errors.append({"filename": f.filename, "error": f"extension {ext} not allowed"})
            continue

        # Sanitize filename — strip path components
        safe_name = Path(f.filename).name
        # Avoid collisions
        dest = uploads_dir / safe_name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 1
            while dest.exists():
                dest = uploads_dir / f"{stem}_{i}{suffix}"
                i += 1

        try:
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                errors.append({"filename": safe_name, "error": f"too large (>{MAX_FILE_SIZE//1024//1024}MB)"})
                continue
            dest.write_bytes(content)
            saved.append({
                "filename": dest.name,
                "size": len(content),
                "path": str(dest.relative_to(settings.ARGO_DB.parent)),
            })
        except Exception as e:
            errors.append({"filename": safe_name, "error": str(e)})

    return {"saved": saved, "errors": errors}


@router.get("/api/chats/{chat_id}/files")
async def api_list_files(chat_id: str):
    """List files uploaded to this chat."""
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    uploads_dir = get_chat_uploads_dir(chat_id)
    if not uploads_dir.exists():
        return {"files": []}
    files = []
    for p in sorted(uploads_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file():
            try:
                st = p.stat()
                files.append({
                    "filename": p.name,
                    "size": st.st_size,
                    "modified": st.st_mtime,
                })
            except OSError:
                continue
    return {"files": files}


@router.delete("/api/chats/{chat_id}/files/{filename}")
async def api_delete_file(chat_id: str, filename: str):
    """Delete a single uploaded file."""
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    uploads_dir = get_chat_uploads_dir(chat_id)
    target = (uploads_dir / filename).resolve()
    try:
        target.relative_to(uploads_dir.resolve())
    except ValueError:
        raise HTTPException(400, "invalid filename")
    if not target.exists():
        raise HTTPException(404, "file not found")
    target.unlink()
    return {"ok": True}


@router.get("/api/chats/{chat_id}/files/{filename}/content")
async def api_get_file_content(chat_id: str, filename: str):
    """Read a file's text content (for display in the UI)."""
    chat = storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "chat not found")
    uploads_dir = get_chat_uploads_dir(chat_id)
    target = (uploads_dir / filename).resolve()
    try:
        target.relative_to(uploads_dir.resolve())
    except ValueError:
        raise HTTPException(400, "invalid filename")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "file not found")
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(500, f"failed to read: {e}")
    return {"filename": filename, "content": content, "size": len(content)}

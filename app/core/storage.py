"""Lightweight SQLite storage for chats + messages. No external deps."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings


_LOCK = threading.Lock()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'chat',
    model TEXT,
    workspace TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    meta TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.ARGO_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_CONN: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        settings.ARGO_DB.parent.mkdir(parents=True, exist_ok=True)
        _CONN = _connect()
        with _LOCK:
            _CONN.executescript(_SCHEMA)
            _CONN.commit()
    return _CONN


def init_db() -> None:
    _get_conn()


# ----- Chat CRUD -----

def create_chat(
    title: str = "چت جدید",
    mode: str = "chat",
    model: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    cid = str(uuid.uuid4())
    now = time.time()
    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO chats (id, title, mode, model, workspace, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, title, mode, model, workspace, now, now),
        )
        conn.commit()
    return {
        "id": cid, "title": title, "mode": mode,
        "model": model, "workspace": workspace,
        "created_at": now, "updated_at": now,
    }


def list_chats(limit: int = 100) -> List[Dict[str, Any]]:
    with _LOCK:
        rows = _get_conn().execute(
            "SELECT * FROM chats ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        row = _get_conn().execute(
            "SELECT * FROM chats WHERE id = ?", (chat_id,),
        ).fetchone()
    return dict(row) if row else None


def update_chat(chat_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [chat_id]
    with _LOCK:
        conn = _get_conn()
        conn.execute(f"UPDATE chats SET {cols} WHERE id = ?", vals)
        conn.commit()


def delete_chat(chat_id: str) -> None:
    with _LOCK:
        conn = _get_conn()
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()


# ----- Messages -----

def add_message(
    chat_id: str,
    role: str,
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    tool_name: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mid = str(uuid.uuid4())
    now = time.time()
    with _LOCK:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, tool_calls, tool_name, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mid, chat_id, role, content,
                json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                tool_name,
                json.dumps(meta, ensure_ascii=False) if meta else None,
                now,
            ),
        )
        # Bump chat's updated_at
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
        conn.commit()
    return {
        "id": mid, "chat_id": chat_id, "role": role, "content": content,
        "tool_calls": tool_calls, "tool_name": tool_name, "meta": meta,
        "created_at": now,
    }


def list_messages(chat_id: str) -> List[Dict[str, Any]]:
    with _LOCK:
        rows = _get_conn().execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("tool_calls"):
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except Exception:
                d["tool_calls"] = None
        if d.get("meta"):
            try:
                d["meta"] = json.loads(d["meta"])
            except Exception:
                d["meta"] = None
        out.append(d)
    return out

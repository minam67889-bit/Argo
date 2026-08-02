# Argo — Architecture

## Layers

```
┌──────────────────────────────────────────────┐
│  Frontend (Vanilla JS + CSS)                 │
│  - Sidebar (chats)                           │
│  - Main (messages, streaming)                │
│  - Settings modal                            │
└──────────────────────────────────────────────┘
                  │ HTTP/SSE
                  ▼
┌──────────────────────────────────────────────┐
│  FastAPI (app/main.py, app/api/routes.py)    │
│  - /api/chats/*   (CRUD chats)               │
│  - /api/chats/{id}/messages  (SSE stream)    │
│  - /api/settings  (runtime config)           │
│  - /api/workspaces                           │
└──────────────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
┌──────────────┐    ┌──────────────────┐
│  SQLite      │    │  AgentLoop        │
│  (storage)   │    │  (app/agent)      │
│              │    │   ↓ parser        │
│  - chats     │    │   ↓ loop          │
│  - messages  │    │   ↓ tools         │
└──────────────┘    └──────────────────┘
                            │
                  ┌─────────┼─────────┐
                  ▼         ▼         ▼
            ┌────────┐ ┌────────┐ ┌────────┐
            │ LLM    │ │ Tools  │ │ Config │
            │ client │ │ (6x)   │ │        │
            └────────┘ └────────┘ └────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  OpenAI-compat   │
        │  (OpenRouter,    │
        │   Ollama, vLLM,  │
        │   llama.cpp, …)  │
        └──────────────────┘
```

## Agent loop details

```
user_input
  ↓
add to messages
  ↓
for step in 1..MAX_STEPS:
  manage_context (sliding window)
  ↓
  call LLM (streaming or non-streaming)
  ↓
  parse response (parser tries 9 formats in order):
    1. <tool_call name="X">{...}</tool_call>           (Qwen-style XML, closed)
    2. <|tool_call name="X">{...}</tool_call|>           (Mistral/Llama-3 pipe-style, closed)
    3. <|tool_call name="X">...                          (pipe-style, unclosed)
    4. <tool_call name="X">...  (unclosed, until </think>)
    5. <tool_call>{json}</tool_call>                  (no name attribute, Qwen3 alt)
    6. <|tool_call|>{json}</tool_call|>                 (pipe-style no name, closed)
    7. <|tool_call|>{json}<|tool_call|>                 (pipe-style no name, unclosed)
    8. <tool_code>...```json```...</tool_code>          (multi-call wrapper)
    9. ```json { "name": ..., "arguments": ... }```     (generic fenced)
   10. ReAct: Action: / Action Input:
   11. Bare JSON with "name"/"arguments" anywhere
  ↓
  strip incomplete special tokens (e.g. "<|im_start|>tool_call")
  ↓
  if no tool calls → emit "done", return
  ↓
  for each ToolCall:
    - check loop detection (same call N times → stop)
    - run tool (with sandbox)
    - append <tool_result> to messages
  ↓
  (loop)
```

## Why this is better than the old project

| Aspect | Old | Argo |
|---|---|---|
| UI | HTML base64 inline, 2-line | Modern, sidebar, multi-chat, markdown, dark/light |
| Tool parser | regex with multiple fallbacks, often broken | 4 formats + native, all tested |
| Context mgmt | None (just dropped oldest) | Sliding window with char budget |
| Loop detection | None | 3-strike rule with same call detection |
| Streaming | None (or broken) | SSE events: text, reasoning, tool_call, tool_result, step, done, error |
| Tools | 5 (some buggy) | 6 (all tested): bash, read/write/edit, list, search |
| Storage | None (browser localStorage) | SQLite (chats + messages persistent) |
| Settings | None | Live update of LLM endpoint, model, params |
| Modes | Mixed (1 mode with 5 tools always) | chat / agent (separate prompts) |
| Backend deps | Mixed, Colab-specific | Pure Python, runs anywhere |
| CLI | Single file, basic | Color, REPL, /reset, /mode |
| Tests | None | Unit tests for parser + tools |
| Config | Env vars only | Env + API + per-chat override |

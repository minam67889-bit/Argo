#!/usr/bin/env python3
"""Tiny OpenAI-compatible server around llama-cpp-python.

Designed for Google Colab: uses prebuilt wheels (no compile), runs on GPU.
Exposes /v1/chat/completions and /v1/models for Argo (and any OpenAI client).
"""
from __future__ import annotations

import os
import time
import uuid
import json
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Load model once at startup
# Free VRAM from any previous model (e.g. the GPU sanity check)
try:
    import gc
    gc.collect()
except Exception:
    pass
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except Exception:
    pass
import time as _time
_time.sleep(1)  # let the OS settle

from llama_cpp import Llama

MODEL_PATH = os.environ.get("MODEL_PATH", "/content/models/qwen3-14b-abliterated.Q4_K_M.gguf")
N_CTX = int(os.environ.get("N_CTX", "12288"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = all on GPU
CHAT_FORMAT = os.environ.get("CHAT_FORMAT", "chatml")
PORT = int(os.environ.get("LLAMA_PORT", "8080"))
HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3-14b-abliterated")

print(f"[server] loading {MODEL_PATH} (n_ctx={N_CTX}, n_gpu_layers={N_GPU_LAYERS})...", flush=True)
t0 = time.time()
LLM = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    chat_format=CHAT_FORMAT,
    verbose=False,
    n_threads=int(os.environ.get("N_THREADS", "4")),
    n_batch=int(os.environ.get("N_BATCH", "512")),
    use_mmap=True,
    use_mlock=False,
    flash_attn=True,
)
print(f"[server] model loaded in {time.time()-t0:.0f}s", flush=True)


app = FastAPI(title="llama-cpp-server (Colab)")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    stream: bool = False
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


def _to_chat_messages(msgs: List[Message]) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in msgs]


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = _to_chat_messages(req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        resp = LLM.create_chat_completion(
            messages=messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
            seed=req.seed,
            stream=False,
        )
        return JSONResponse(resp)

    # Streaming
    def event_stream():
        stream = LLM.create_chat_completion(
            messages=messages,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop,
            seed=req.seed,
            stream=True,
        )
        for chunk in stream:
            # Convert to OpenAI streaming format
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            finish_reason = choice.get("finish_reason")
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content} if content else {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if finish_reason:
                yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    # Try the requested port; if busy, find a free one
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((HOST, PORT))
        s.close()
        chosen_port = PORT
    except OSError:
        # Find a free port in range
        for p in range(PORT + 1, PORT + 100):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((HOST, p))
                s.close()
                chosen_port = p
                print(f"[server] port {PORT} busy, using {p} instead", flush=True)
                # Save the actual port for downstream consumers
                with open('/content/llama.port', 'w') as f:
                    f.write(str(p))
                break
            except OSError:
                continue
        else:
            print(f"[server] no free port found near {PORT}!", flush=True)
            raise SystemExit(1)
    else:
        with open('/content/llama.port', 'w') as f:
            f.write(str(chosen_port))
    uvicorn.run(app, host=HOST, port=chosen_port, log_level="warning")

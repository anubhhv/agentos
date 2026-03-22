import os
import json
import uuid
import asyncio
from typing import Optional
from pathlib import Path
from datetime import timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent
from auth import (
    register_user, authenticate_user, get_user,
    create_access_token, decode_token, EXPIRE_MINUTES
)

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="AgentOS API", version="1.0.0")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500,http://localhost:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session stores ─────────────────────────────────────────────────
# session_id -> list of message dicts
conversation_store: dict[str, list] = {}

# session_id -> dict of filename -> bytes
file_store: dict[str, dict] = {}


# ── Auth helpers ─────────────────────────────────────────────────────────────
def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Extract user from Bearer token. Returns None if no/invalid token."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        return None
    return get_user(payload.get("sub", ""))


def require_user(authorization: Optional[str] = Header(None)) -> dict:
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── Pydantic models ──────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# ── Auth routes ──────────────────────────────────────────────────────────────
@app.post("/auth/register")
async def register(req: RegisterRequest):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    result = register_user(req.email, req.password, req.name)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    token = create_access_token({"sub": req.email}, timedelta(minutes=EXPIRE_MINUTES))
    return {"token": token, "user": {"email": result["email"], "name": result["name"]}}


@app.post("/auth/login")
async def login(req: LoginRequest):
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    token = create_access_token({"sub": req.email}, timedelta(minutes=EXPIRE_MINUTES))
    return {"token": token, "user": user}


@app.get("/auth/me")
async def me(user: dict = Depends(require_user)):
    return user


# ── Session routes ────────────────────────────────────────────────────────────
@app.post("/session/new")
async def new_session():
    session_id = str(uuid.uuid4())
    conversation_store[session_id] = []
    file_store[session_id] = {}
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    conversation_store.pop(session_id, None)
    file_store.pop(session_id, None)
    return {"cleared": True}


@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    history = conversation_store.get(session_id, [])
    return {"session_id": session_id, "history": history, "turns": len(history)}


# ── File upload ───────────────────────────────────────────────────────────────
@app.post("/session/{session_id}/upload")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    MAX_SIZE = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTS = {".pdf", ".csv", ".json", ".txt", ".md", ".log", ".py", ".js", ".html", ".css"}

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"File type '{ext}' not supported. Allowed: {sorted(ALLOWED_EXTS)}")

    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(400, f"File too large ({len(content)//1024}KB). Max 20MB.")

    if session_id not in file_store:
        file_store[session_id] = {}
    file_store[session_id][file.filename] = content

    return {
        "filename": file.filename,
        "size_kb": round(len(content) / 1024, 1),
        "type": ext,
        "session_id": session_id,
        "message": f"File '{file.filename}' uploaded. You can now ask the agent about it."
    }


@app.get("/session/{session_id}/files")
async def list_files(session_id: str):
    files = file_store.get(session_id, {})
    return {
        "session_id": session_id,
        "files": [
            {"filename": name, "size_kb": round(len(data) / 1024, 1)}
            for name, data in files.items()
        ]
    }


# ── Main agent chat endpoint (SSE streaming) ──────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Stream agent responses as Server-Sent Events.
    Each event is a JSON object on a data: line.

    Event types:
      thinking    — agent reasoning text
      tool_call   — tool being invoked
      tool_result — tool output
      final_answer — complete answer
      error       — something went wrong
      done        — agent finished, includes iteration count
    """
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in conversation_store:
        conversation_store[session_id] = []
    if session_id not in file_store:
        file_store[session_id] = {}

    history = conversation_store[session_id]
    files = file_store[session_id]

    # We'll collect assistant messages to append to history after streaming
    collected_answer = []

    async def event_stream():
        final_text = ""
        iterations = 0

        try:
            async for event in run_agent(req.message, list(history), files):
                # Stream every event to the client
                yield f"data: {json.dumps(event)}\n\n"

                if event["type"] == "final_answer":
                    final_text = event["text"]
                if event["type"] == "done":
                    iterations = event["iterations"]

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        # Persist conversation turn after streaming
        if final_text:
            history.append({"role": "user", "content": req.message})
            history.append({"role": "assistant", "content": final_text})
            # Keep history bounded at 20 turns
            if len(history) > 40:
                conversation_store[session_id] = history[-40:]

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-ID": session_id
        }
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "tools": ["web_search", "web_fetch", "run_python", "calculate", "get_weather", "read_file"],
        "model": "claude-sonnet-4-20250514"
    }


@app.get("/")
async def root():
    return {
        "name": "AgentOS API",
        "docs": "/docs",
        "health": "/health"
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

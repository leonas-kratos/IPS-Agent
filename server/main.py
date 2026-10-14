import os
import json
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="IPS-UWB Central AI Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True
    model: str = "default"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "IPS-UWB Central AI Server",
        "message": "Máy chủ AI luôn online phục vụ repo IPS UWB",
        "has_groq": bool(GROQ_API_KEY),
        "has_gemini": bool(GEMINI_API_KEY)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

def stream_groq(messages: List[ChatMessage]):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": True,
        "temperature": 0.4
    }
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except Exception:
                        continue

def stream_gemini(messages: List[ChatMessage]):
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"

    contents = []
    system_instruction = None
    for m in messages:
        if m.role == "system":
            system_instruction = {"parts": [{"text": m.content}]}
        else:
            role = "user" if m.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.3}
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    with requests.post(url, headers={"Content-Type": "application/json"}, json=payload, stream=True, timeout=60) as r:
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        candidates = chunk.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                if "text" in p:
                                    yield f"data: {json.dumps({'content': p['text']})}\n\n"
                    except Exception:
                        continue

def stream_ollama(messages: List[ChatMessage]):
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": "qwen2.5:3b",
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": True
    }
    with requests.post(url, json=payload, stream=True, timeout=120) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    # Ưu tiên Groq (siêu nhanh, miễn phí) -> Gemini -> Ollama
    if GROQ_API_KEY:
        generator = stream_groq(req.messages)
    elif GEMINI_API_KEY:
        generator = stream_gemini(req.messages)
    else:
        generator = stream_ollama(req.messages)

    return StreamingResponse(generator, media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

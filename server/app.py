import os
import json
import requests
from typing import List
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import gradio as gr

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True

def stream_groq(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    all_msgs = [{"role": m.role if hasattr(m, 'role') else m["role"], 
                 "content": m.content if hasattr(m, 'content') else m["content"]} for m in messages]
    
    groq_candidates = [
        ("openai/gpt-oss-120b", 1200),
        ("qwen/qwen3.8-27b", 800),
        ("openai/gpt-oss-20b", 1500)
    ]
    
    success = False
    for model_cand, max_tok in groq_candidates:
        payload = {
            "model": model_cand,
            "messages": all_msgs,
            "stream": True,
            "max_tokens": max_tok,
            "temperature": 0.3
        }
        try:
            r = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
            if r.status_code == 200:
                success = True
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    content = choices[0]["delta"].get("content", "")
                                    if content:
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                            except Exception:
                                continue
                break
            elif r.status_code == 429:
                continue
        except Exception:
            continue

    if not success:
        yield from stream_free_unlimited(messages)

def stream_gemini(messages):
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"

    contents = []
    system_instruction = None
    for m in messages:
        role_val = m.role if hasattr(m, 'role') else m["role"]
        content_val = m.content if hasattr(m, 'content') else m["content"]
        if role_val == "system":
            system_instruction = {"parts": [{"text": content_val}]}
        else:
            r = "user" if role_val == "user" else "model"
            contents.append({"role": r, "parts": [{"text": content_val}]})

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

def stream_free_unlimited(messages):
    url = "https://text.pollinations.ai/openai/chat/completions"
    payload = {
        "model": "openai",
        "messages": [{"role": m.role if hasattr(m, 'role') else m["role"], 
                      "content": m.content if hasattr(m, 'content') else m["content"]} for m in messages],
        "stream": True,
        "temperature": 0.3
    }
    with requests.post(url, json=payload, stream=True, timeout=60) as r:
        for line in r.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            content = choices[0]["delta"].get("content", "")
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                    except Exception:
                        continue

# Giao diện Web Gradio
def web_chat(message, history):
    msgs = [{"role": "system", "content": "Bạn là Chuyên gia Nghiên cứu Kỹ thuật Hệ thống Định vị Trong nhà (IPS) UWB."}]
    for h in history:
        msgs.append({"role": "user", "content": h[0]})
        msgs.append({"role": "assistant", "content": h[1]})
    msgs.append({"role": "user", "content": message})

    if GROQ_API_KEY:
        gen = stream_groq(msgs)
    elif GEMINI_API_KEY:
        gen = stream_gemini(msgs)
    else:
        gen = stream_free_unlimited(msgs)

    full_resp = ""
    for chunk in gen:
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:].strip())
                full_resp += data.get("content", "")
                yield full_resp
            except Exception:
                pass

demo = gr.ChatInterface(
    fn=web_chat,
    title="📡 IPS-UWB AI Assistant Web & API",
    description="Máy chủ AI 24/7 phục vụ nghiên cứu và lập trình IPS UWB. Hỗ trợ cả Web Chat và Terminal CLI!"
)

# Gắn API route trực tiếp vào ứng dụng FastAPI của Gradio
@demo.app.get("/health")
def health():
    return {"status": "ok", "service": "IPS-UWB AI Server"}

@demo.app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if GROQ_API_KEY:
        generator = stream_groq(req.messages)
    elif GEMINI_API_KEY:
        generator = stream_gemini(req.messages)
    else:
        generator = stream_free_unlimited(req.messages)

    return StreamingResponse(generator, media_type="text/event-stream")

# Chạy trực tiếp demo với Gradio
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

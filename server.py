# -*- coding: utf-8 -*-
"""
GemProxy for GitHub Codespaces
OpenAI 兼容 API 服务，支持文字对话、图片生成、视频生成
"""

import asyncio
import json
import uuid
import time
import os
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from gemini_webapi import GeminiClient

# ============ 配置 ============
DATA_DIR = Path("/workspaces/gemproxy")
COOKIE_FILE = DATA_DIR / "cookies.json"
API_KEY_FILE = DATA_DIR / "api_key.txt"
PORT = int(os.getenv("PORT", 8080))

# ============ 请求模型 ============
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "gemini-3-pro"
    messages: List[ChatMessage]
    stream: bool = False

class ImageRequest(BaseModel):
    prompt: str
    model: str = "gemini"

class VideoRequest(BaseModel):
    prompt: str
    model: str = "gemini"

class CookieSetup(BaseModel):
    secure_1psid: str
    secure_1psidts: str

# ============ 全局客户端 ============
client: Optional[GeminiClient] = None

def load_cookies():
    if COOKIE_FILE.exists():
        cookies = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
        return cookies.get("__Secure-1PSID", ""), cookies.get("__Secure-1PSIDTS", "")
    return "", ""

def load_api_key():
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text(encoding="utf-8").strip()
    return None

API_KEY = load_api_key()

async def get_client() -> GeminiClient:
    global client
    if client is None:
        secure_1psid, secure_1psidts = load_cookies()
        if not secure_1psid or not secure_1psidts:
            raise HTTPException(status_code=500, detail="Cookie 未配置，请访问 /setup 设置")
        client = GeminiClient(secure_1psid, secure_1psidts, proxy=None)
        await client.init(timeout=30, auto_close=False, close_delay=300, auto_refresh=True)
        print("[INFO] Gemini client initialized!")
    return client

# ============ 认证依赖 ============
from fastapi import Header, Depends

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return True
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ============ FastAPI 应用 ============
app = FastAPI(title="GemProxy", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head><title>GemProxy</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto;">
        <h1>GemProxy 服务运行中</h1>
        <p>OpenAI 兼容 API，支持 Gemini 对话、生图、生视频</p>
        <hr>
        <h3>快速设置 Cookie</h3>
        <form action="/setup" method="post">
            <label>__Secure-1PSID:</label><br>
            <input name="secure_1psid" style="width: 100%; margin-bottom: 10px;"><br>
            <label>__Secure-1PSIDTS:</label><br>
            <input name="secure_1psidts" style="width: 100%; margin-bottom: 10px;"><br>
            <button type="submit" style="padding: 10px 20px;">保存</button>
        </form>
        <hr>
        <p><a href="/docs">API 文档</a> | <a href="/v1/models">模型列表</a></p>
    </body>
    </html>
    """

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-3-pro", "object": "model", "owned_by": "google"},
            {"id": "gemini-3-flash", "object": "model", "owned_by": "google"},
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, _: bool = Depends(verify_api_key)):
    try:
        gemini = await get_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    user_message = request.messages[-1].content if request.messages else ""
    
    if request.stream:
        async def generate_stream():
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            try:
                async for chunk in gemini.generate_content_stream(user_message, model=request.model):
                    delta = {"content": chunk.text_delta}
                    data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                final = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        try:
            response = await gemini.generate_content(user_message, model=request.model)
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response.text}, "finish_reason": "stop"}]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/generations")
async def image_generations(request: ImageRequest, _: bool = Depends(verify_api_key)):
    gemini = await get_client()
    try:
        response = await gemini.generate_content(f"生成图片: {request.prompt}")
        images = []
        if response.images:
            for i, img in enumerate(response.images):
                img_path = DATA_DIR / "output" / f"img_{uuid.uuid4().hex[:8]}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                await img.save(path=str(img_path.parent), filename=img_path.name)
                images.append({"url": f"/output/{img_path.name}"})
        return {"created": int(time.time()), "data": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/videos/generations")
async def video_generations(request: VideoRequest, _: bool = Depends(verify_api_key)):
    gemini = await get_client()
    try:
        response = await gemini.generate_content(f"生成视频: {request.prompt}")
        videos = []
        if response.videos:
            for video in response.videos:
                vid_path = DATA_DIR / "output" / f"video_{uuid.uuid4().hex[:8]}.mp4"
                vid_path.parent.mkdir(parents=True, exist_ok=True)
                await video.save(path=str(vid_path.parent), verbose=True)
                videos.append({"url": f"/output/{vid_path.name}", "status": "completed"})
        return {"created": int(time.time()), "data": videos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup")
async def setup_cookies(secure_1psid: str = "", secure_1psidts: str = ""):
    if not secure_1psid or not secure_1psidts:
        return HTMLResponse("<h3>请填写完整的 Cookie 信息</h3><a href='/'>返回</a>")
    
    cookies = {
        "__Secure-1PSID": secure_1psid,
        "__Secure-1PSIDTS": secure_1psidts
    }
    COOKIE_FILE.write_text(json.dumps(cookies), encoding="utf-8")
    return HTMLResponse("<h3>Cookie 已保存！</h3><p>服务将自动重启...</p><a href='/'>返回首页</a>")

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print(f"""
========================================
 GemProxy 服务启动中...
 地址: http://localhost:{PORT}
========================================
    """)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "output").mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)

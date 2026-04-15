# -*- coding: utf-8 -*-
"""
GemProxy for GitHub Codespaces - 无 API Key 验证版本
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

# ============ 全局客户端 ============
client: Optional[GeminiClient] = None

def load_cookies():
 if COOKIE_FILE.exists():
 cookies = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
 return cookies.get("__Secure-1PSID", ""), cookies.get("__Secure-1PSIDTS", "")
 return "", ""

async def get_client() -> GeminiClient:
 global client
 if client is None:
 secure_1psid, secure_1psidts = load_cookies()
 if not secure_1psid or not secure_1psidts:
 raise HTTPException(status_code=500, detail="Cookie 未配置")
 client = GeminiClient(secure_1psid, secure_1psidts, proxy=None)
 await client.init(timeout=30, auto_close=False, close_delay=300, auto_refresh=True)
 print("[INFO] Gemini client initialized!")
 return client

# ============ FastAPI 应用 ============
app = FastAPI(title="GemProxy", version="1.0.0")

@app.get("/")
async def root():
 return {"message": "GemProxy 运行中", "docs": "/docs"}

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
async def chat_completions(request: ChatRequest):
 try:
 gemini = await get_client()
 except Exception as e:
 raise HTTPException(status_code=500, detail=str(e))
 
 user_message = request.messages[-1].content if request.messages else ""
 
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
async def image_generations(request: ImageRequest):
 gemini = await get_client()
 try:
 response = await gemini.generate_content(f"生成图片: {request.prompt}")
 images = []
 if response.images:
 for img in response.images:
 img_path = DATA_DIR / "output" / f"img_{uuid.uuid4().hex[:8]}.png"
 img_path.parent.mkdir(parents=True, exist_ok=True)
 await img.save(path=str(img_path.parent), filename=img_path.name)
 images.append({"url": f"/output/{img_path.name}"})
 return {"created": int(time.time()), "data": images}
 except Exception as e:
 raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/videos/generations")
async def video_generations(request: VideoRequest):
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

@app.get("/health")
async def health():
 return {"status": "ok"}

if __name__ == "__main__":
 print(f"\n========================================\n GemProxy 服务启动中...\n 地址: http://localhost:{PORT}\n========================================\n")
 DATA_DIR.mkdir(parents=True, exist_ok=True)
 (DATA_DIR / "output").mkdir(parents=True, exist_ok=True)
 uvicorn.run(app, host="0.0.0.0", port=PORT)

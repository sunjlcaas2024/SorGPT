# -*- coding: utf-8 -*-
"""
SorGPT API Server - 带用户认证和隔离
修复：使用单例管道，所有用户共享一个实例
"""
import os
import sys
import uuid
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import SorghumRAGPipeline
from config import *

# ============== 全局共享的 RAG 管道 ==============
# 修复：只创建一次全局管道，所有用户共享
_global_pipeline: Optional[SorghumRAGPipeline] = None

def get_global_pipeline() -> SorghumRAGPipeline:
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = SorghumRAGPipeline()
    return _global_pipeline

# ============== 用户认证系统 ==============
class UserSession:
    """用户会话管理"""
    def __init__(self, user_id: str, api_key: str):
        self.user_id = user_id
        self.api_key = api_key
        self.query_count = 0
        self.last_query_time = 0
        self.created_at = time.time()

users_db: Dict[str, UserSession] = {}
api_keys: Dict[str, str] = {}

RATE_LIMIT_SECONDS = 10
MAX_QUERY_PER_DAY = 500

def generate_api_key() -> str:
    return f"sk-{uuid.uuid4().hex[:32]}"

def create_user(username: str) -> tuple:
    user_id = str(uuid.uuid4())
    api_key = generate_api_key()
    session = UserSession(user_id, api_key)
    users_db[user_id] = session
    api_keys[api_key] = user_id
    return user_id, api_key

def verify_api_key(api_key: str) -> Optional[UserSession]:
    user_id = api_keys.get(api_key)
    if user_id:
        return users_db.get(user_id)
    return None

# ============== API 模型 ==============
class QuestionRequest(BaseModel):
    question: str
    stream: bool = False

class QuestionResponse(BaseModel):
    query: str
    query_type: str
    answer: str
    references: List[str]
    user_id: str

class HealthResponse(BaseModel):
    status: str
    users_online: int
    version: str

# ============== FastAPI 应用 ==============
def _normalize_citations(answer: str, references: list):
    import re
    # Find all citation numbers in answer, in order of first appearance
    seen = set()
    order = []
    for m in re.finditer(r'\[([\d,\s]+)\]', answer):
        for n in m.group(1).split(','):
            n = n.strip()
            if n.isdigit() and n not in seen:
                seen.add(n)
                order.append(n)
    if not order:
        return answer, references
    
    # Build old-ref lookup
    old_refs = {}
    for r in references:
        m = re.match(r'\[(\d+)\]', r)
        if m: old_refs[m.group(1)] = r
    
    # Build new references: only keep cited ones, in citation order
    new_refs = []
    mapping = {}
    for new_id, old_id in enumerate(order, 1):
        mapping[old_id] = str(new_id)
        if old_id in old_refs:
            new_refs.append(re.sub(r'^\[\d+\]', f'[{new_id}]', old_refs[old_id]))
    
    # Remap citations in answer
    def _remap(m):
        nums = [n.strip() for n in m.group(1).split(',') if n.strip().isdigit()]
        new_nums = [mapping.get(n, '') for n in nums]
        new_nums = [n for n in new_nums if n]
        return '[' + ','.join(new_nums) + ']' if new_nums else ''
    
    answer = re.sub(r'\[([\d,\s]+)\]', _remap, answer)
    answer = re.sub(r'\[\s*\]', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer, new_refs

app = FastAPI(title="SorGPT API", description="高粱科研问答 RAG 系统 API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============== 依赖注入 ==============
async def get_current_user(x_api_key: str = Header(..., alias="X-API-Key")) -> UserSession:
    session = verify_api_key(x_api_key)
    if not session:
        raise HTTPException(status_code=401, detail="无效的 API Key")

    current_time = time.time()
    if current_time - session.last_query_time < RATE_LIMIT_SECONDS:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请等待 {RATE_LIMIT_SECONDS} 秒")

    session.last_query_time = current_time
    session.query_count += 1
    return session

# ============== API 路由 ==============
@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", users_online=len(users_db), version="1.0.1")

@app.post("/register")
async def register_user(username: str = "anonymous"):
    user_id, api_key = create_user(username)
    return {"user_id": user_id, "api_key": api_key, "message": "请妥善保存 API Key"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, session: UserSession = Depends(get_current_user)):
    try:
        pipeline = get_global_pipeline()
        result = pipeline.ask(request.question)
        # v8: proper one-pass citation normalization
        import re as _re8
        a = result["answer"]
        refs = result.get("references", [])
        # Collect unique cited numbers in order of first appearance
        seen = set()
        order = []
        for m in _re8.finditer(r'\[([\d,\s]+)\]', a):
            for n in m.group(1).split(','):
                n = n.strip()
                if n.isdigit() and n not in seen:
                    seen.add(n)
                    order.append(n)
        if order and refs:
            # old_id -> new_id mapping
            new_id = {}
            for ni, oi in enumerate(order, 1):
                new_id[oi] = str(ni)
            # One-pass replace: for each bracket group, remap all numbers
            def _remap(m):
                nums = [x.strip() for x in m.group(1).split(',') if x.strip().isdigit()]
                mapped = [new_id.get(n, '') for n in nums]
                mapped = [x for x in mapped if x]
                return '[' + ','.join(mapped) + ']' if mapped else ''
            a = _re8.sub(r'\[([\d,\s]+)\]', _remap, a)
            a = _re8.sub(r'\[\s*\]', '', a)
            a = _re8.sub(r'  +', ' ', a).strip()
            # Rebuild refs
            old_r = {}
            for r in refs:
                m2 = _re8.match(r'\[(\d+)\]', r)
                if m2: old_r[m2.group(1)] = r
            new_refs = []
            for oi in order:
                if oi in old_r:
                    new_refs.append(_re8.sub(r'\[\d+\]', '[' + new_id[oi] + ']', old_r[oi], 1))
            result["answer"] = a
            result["references"] = new_refs
        return QuestionResponse(
            query=result["query"],
            query_type=result["query_type"],
            answer=result["answer"],
            references=result["references"],
            user_id=session.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# 支持流式响应
async def stream_generator(pipeline: SorghumRAGPipeline, question: str):
    for chunk in pipeline.ask_stream(question):
        yield chunk

async def _stream_with_cleanup(pipeline: SorghumRAGPipeline, question: str):
    full = []
    async for chunk in stream_generator(pipeline, question):
        full.append(chunk)
        yield chunk
    # Can't easily clean streaming output retroactively, skip for now
    # The METADATA block already has cleaned references from pipeline
    pass

@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest, session: UserSession = Depends(get_current_user)):
    try:
        pipeline = get_global_pipeline()
        return StreamingResponse(
            _stream_with_cleanup(pipeline, request.question),
            media_type="text/plain",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache, no-transform",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/stats")
async def get_stats(session: UserSession = Depends(get_current_user)):
    return {
        "user_id": session.user_id,
        "query_count": session.query_count,
        "uptime_seconds": int(time.time() - session.created_at)
    }

# ============== 预加载全局管道 ==============
if __name__ == "__main__":
    print("预加载 RAG 管道模型和索引...")
    pipeline = get_global_pipeline()
    print("RAG 管道加载完成，启动 API 服务器")

    # 使用固定的admin API key，避免重启后变化
    FIXED_ADMIN_KEY = "sk-6a687dfd1e7d4cd09ebe9afc612965a5"
    if not api_keys:
        admin_id = str(uuid.uuid4())
        api_key = FIXED_ADMIN_KEY
        session = UserSession(admin_id, api_key)
        users_db[admin_id] = session
        api_keys[api_key] = admin_id
        print("\n" + "="*60)
        print("默认管理员已创建:")
        print(f"  User ID: {admin_id}")
        print(f"  API Key: {api_key}")
        print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

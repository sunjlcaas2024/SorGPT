# SorGPT - 高粱科研问答系统

基于 RAG (检索增强生成) 的高粱科研文献智能问答 API 服务。

## 项目结构

```
sorghum_rag/
├── api_server.py          # FastAPI 服务（用户认证+隔离）
├── config.py              # 全局配置
├── pipeline.py            # RAG 主流程
├── static/
│   └── index.html         # Web 前端测试页面
├── bge-m3/                # Embedding 模型 (2.2GB)
├── Qwen/                  # Qwen2.5-7B 模型 (15GB)
├── faiss_v3_*/            # FAISS 向量索引 (~5GB)
├── db/
│   └── sorghum_genes.db   # 基因注释数据库 (72MB)
└── *.py                   # 核心模块
```

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 开发模式（前台运行）
python api_server.py

# 或使用启动脚本
./start.sh
```

### 3. 访问

- **Web 前端**: http://服务器IP:8000
- **API 文档**: http://服务器IP:8000/docs

## API 使用

### 注册用户

```bash
curl -X POST "http://localhost:8000/register?username=test"
```

返回:
```json
{
  "user_id": "xxx",
  "api_key": "sk-xxxxx...",
  "message": "请妥善保存 API Key"
}
```

### 问答

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 你的API密钥" \
  -d '{"question": "SbDW3基因的功能是什么？"}'
```

## 开放端口

### 方式一：直接运行（防火墙开放 8000）

```bash
# 确保云服务器安全组/防火墙开放 8000 端口
# 阿里云/腾讯云：在控制台安全组添加入站规则
# 端口范围：8000，协议：TCP

python api_server.py
```

### 方式二：使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 方式三：使用 uvicorn 指定 host

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 用户隔离

- 每个用户有独立的 API Key
- 同一用户有请求速率限制（默认 10 秒间隔）
- 每日查询配额：500 次
- 用户会话独立管理

## 配置

编辑 `config.py` 修改：
- 模型路径
- API 地址
- 检索参数
- 速率限制

# SorGPT 项目全貌

## 服务器连接

```
生产服务器: 10.122.1.1 (内网，需跳板机)
跳板机:     10.122.14.119, 端口 1056, 用户 sunjlcaas2024
SSH 方式:   ssh -J cluster server
生产路径:   /vol/sunjilin/website/data/agent/sorghum_rag
Conda 环境: sorghum_rag (miniforge3)
服务启动:   cd /vol/sunjilin/website/data/agent/sorghum_rag && conda activate sorghum_rag && python api_server.py
服务端口:   8000
API Key:    请求头 X-API-Key: sk-6a687dfd1e7d4cd09ebe9afc612965a5
健康检查:   ssh -J cluster server "curl -s http://localhost:8000/"
```

## 本地项目路径

```
/Users/jilinsun/项目/高粱在线网站/SorGPT/           ← 文档和评估
/Users/jilinsun/website/sorghum-database/           ← Vue 前端
/Users/jilinsun/项目/高粱在线网站/SorGPT/技术评估报告_2026-06-21/  ← 评估报告和脚本
```

## GitHub

```
https://github.com/sunjlcaas2024/SorGPT (私有仓库)
SSH Key: ~/.ssh/id_ed25519 (sorghum-database@github)
分支: main
推送: git push origin main
```

## 核心架构

SorGPT 是高粱科研文献 RAG 问答系统：

```
FastAPI → query_classifier (9类型路由) → metadata FAISS → fulltext FAISS (4粒度)
→ reranker (粒度+章节+引用次数+BM25 gate) → prompt_builder → DeepSeek API → 答案+引用
```

## 关键模块

- `api_server.py`: FastAPI 入口，/ask 和 /ask/stream，用户认证+限流
- `pipeline.py`: RAG 总控，ask() 和 ask_stream()，引用重排 _reorder_refs_by_appearance v5
- `config.py`: TOP_META_K=120, USE_FAISS_GPU=False, META_INDEX_PATHS, FULLTEXT_INDEX_PATHS
- `retriever.py`: 元数据检索+全文检索，BM25 v2，Dynamic TOP_K，std/large去重，双语检索
- `reranker.py`: 粒度/section/引用次数/期刊加权，BM25 Density Gate，diversify_and_trim
- `query_classifier.py`: boundary/locate/兜底修复（v2），多标签路由
- `prompt_builder.py`: 中英双语 prompt，基因DB注入，所有 DB 标记改为圆括号
- `generator.py`: DeepSeek Reasoner API，流式 enable_thinking=False
- `db/gene_db_query.py`: 基因注释查询 SQLite
- `db/omics_query.py`: 多组学/QTL/已知基因查询

## 已完成的修改

### 分类器 (query_classifier.py)
- _is_boundary() 新增30+边界触发词
- _is_locate() 新增15+定位触发词
- locate 优先规则：含"请提供作者/DOI"+"基因名"→强制 locate
- 兜底从 mechanism 改为 review

### 管线 (pipeline.py)  
- _format_locate_answer() 高粱过滤+去重+期刊排序
- _format_gene_count_answer() SQLite 基因统计 (36,924 genes)
- _reorder_refs_by_appearance v5: 引用重排+幽灵引用过滤
- ask() 和 ask_stream() 出口层引用归一化
- 流式 enable_thinking=False (快速输出)

### 检索 (retriever.py)
- Dynamic TOP_K: count/review/gene_list ×2
- std/large 去重: >70% 重叠→保留最佳粒度
- BM25 Density Gate: gene_list需≥0.5或≥3chunk, gene_function≥0.3
- 双语检索: EN+ZH平行，英语跳过中文索引

### 配置 (config.py)
- TOP_META_K: 300→120
- USE_FAISS_GPU: False
- 中文 meta/CSV 路径 (中文全文索引因 MKL 损坏暂禁用)

### 数据库标记
- 全部 DB 标记从方括号改为圆括号: (GeneDB)(KnownGenes)(KnownGene)(QTLDB)(PfamDB)(DBCount)(PhenomeDB)(MetabolomeDB)
- 涉及文件: pipeline.py, prompt_builder.py, gene_db_query.py, omics_query.py

### 前端 (Vue)
- `/Users/jilinsun/website/sorghum-database/src/views/SorGPT.vue`
- 流式渲染 /ask/stream
- 引用重排: 信任服务端 citationMap
- heading 正则: 确保 markdown ### 前有换行
- 基准字号 14px, Generating answer 跳动动画

### 评估 (eval_runner.py)
- 5阶段自动评估: Claim提取→事实验证→引用质量→答案质量→聚合
- 200题双语评估集 (eval_questions_clean_200.json)
- 引用 AutoNuggetizer+RAGChecker+RAGEval 三篇 CCF-A 论文

## 已知问题

- MKL 库损坏，无法重建中文 IVF-PQ 索引
- 中文全文索引为 IndexFlat (无压缩，GPU OOM)
- DeepSeek Reasoner 思考阶段有固有静默期
- 流式输出状态消息可能有 pyc 缓存不更新问题(需清 __pycache__ 后重启)

## 数据资产

- FAISS 索引: v3 English fine/std/large/para + meta (5GB), Chinese fine/std/large + meta (3GB)
- 数据库: sorghum_genes.db (72MB), known_genes.db (41KB), qtl.db (193MB), omics.db (127MB)
- 模型: BGE-M3 (2.2GB), Qwen2.5-7B (15GB, 已注释)
- 文献: English CSV 25K, Chinese CSV 17K, Chinese PDFs 17K
- BM25 IDF: bm25_idf.pkl (30MB), citation_cache.db (5.3MB)

## 常用命令

```bash
# 连接服务器
ssh -J cluster server

# 查看服务状态
ps aux | grep api_server | grep -v grep
curl -s http://localhost:8000/

# 查看日志
tail -50 /vol/sunjilin/website/data/agent/sorghum_rag/app.log

# 重启服务 (改代码后)
kill $(ps aux | grep 'python api_server' | grep -v grep | awk '{print $2}')
cd /vol/sunjilin/website/data/agent/sorghum_rag
source /home/sunjilin/miniforge3/etc/profile.d/conda.sh
conda activate sorghum_rag
nohup python api_server.py > app.log 2>&1 &

# 清缓存重启 (pyc 问题)
rm -rf __pycache__/pipeline*.pyc && 重启

# 测试 API
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-6a687dfd1e7d4cd09ebe9afc612965a5" \
  -d '{"question":"What genes control plant height in sorghum?"}'

# Git
cd /vol/sunjilin/website/data/agent/sorghum_rag
git status
git add -A && git commit -m "message" && git push
```

## 备份文件

服务器上有 `.bak` 文件记录各阶段修改：
- query_classifier.py.bak.v2
- pipeline.py.bak.v2, .bak.v4, .bak.v5
- retriever.py.bak.v2
- config.py.bak.v3
- api_server.py.bak
- static/index.html.bak

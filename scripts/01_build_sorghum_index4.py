import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
import pandas as pd
import fitz  # PyMuPDF
import torch
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# =========================================
# 1. 核心参数与梯度配置
# =========================================
MODEL_PATH = "./models/bge-m3"
# 梯度定义：(Chunk_Size, Overlap, Embed_Batch)
# 这里的 Embed_Batch 随 Size 增大而减小，确保显存安全
SCALES = [
    {"name": "fine",   "size": 512,  "overlap": 50,  "batch": 256}, # 精细粒度
    {"name": "std",    "size": 1000, "overlap": 100, "batch": 128}, # 标准段落
    {"name": "large",  "size": 1500, "overlap": 200, "batch": 64}   # 宏观上下文
]

SOURCE_CONFIG = {
    "english": {
        "csv": "/vol/sunjilin/website/data/publication/english_content.csv",
        "pdf_dir": "/vol/sunjilin/website/data/publication/english/"
    },
    "chinese": {
        "csv": "/vol/sunjilin/website/data/publication/chinese_content.csv",
        "pdf_dir": "/vol/sunjilin/website/data/publication/chinese/"
    }
}

# =========================================
# 2. 增强型 Embedding 类
# =========================================
from langchain_core.embeddings import Embeddings

class FastBgeEmbeddings(Embeddings):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = SentenceTransformer(MODEL_PATH, device="cuda")
        self.model.half() # 3090 必开 FP16 加速

    def embed_documents(self, texts):
        with torch.no_grad():
            return self.model.encode(
                texts, batch_size=self.batch_size, 
                normalize_embeddings=True, show_progress_bar=False
            ).tolist()

    def __call__(self, text):
        return self.embed_query(text)

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()

# =========================================
# 3. 多进程解析 Worker
# =========================================
def parse_pdf_worker(task):
    fpath, meta, size, overlap = task
    try:
        fname = os.path.basename(fpath)
        with fitz.open(fpath) as doc:
            text = " ".join([page.get_text() for page in doc])
        if len(text.strip()) < 50: return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        
        return [Document(
            page_content=c,
            metadata={
                "source": fname,
                "title": str(meta.get("names", fname)),
                "authors": str(meta.get("Author Full Names", "Unknown")),
                "journal": str(meta.get("Source Title", "Unknown")),
                "doi": str(meta.get("doi", "")),
                "scale": size
            }
        ) for c in chunks]
    except: return []

# =========================================
# 4. 主循环流程
# =========================================
def run_indexing():
    for scale in SCALES:
        print(f"\n{'='*20} 正在构建梯度: {scale['name'].upper()} (Size:{scale['size']}) {'='*20}")
        save_path = f"./faiss_index_{scale['name']}"
        embed_model = FastBgeEmbeddings(scale['batch'])
        vector_db = None

        for lang, config in SOURCE_CONFIG.items():
            if not os.path.exists(config["csv"]): continue
            
            # 加载 CSV 映射表
            try:
                df = pd.read_csv(config["csv"], encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(config["csv"], encoding="gb18030")
                except:
                    df = pd.read_csv(config["csv"], encoding="utf-8", encoding_errors="ignore")
            # 自动识别最后一列作为文件名列
            filename_col = df.columns[-1]
            df = df.drop_duplicates(subset=[filename_col])
            meta_map = df.set_index(filename_col).to_dict("index"); meta_map = {k: {**v, "names": k} for k, v in meta_map.items()}
            pdf_files = [f for f in os.listdir(config["pdf_dir"]) if f.endswith('.pdf')]
            
            # 分批解析与向量化（每 100 个文件存一次盘，保护显存）
            FILE_BATCH = 100
            with tqdm(total=len(pdf_files), desc=f"Progress ({lang})", unit="file") as pbar:
                for i in range(0, len(pdf_files), FILE_BATCH):
                    current_files = pdf_files[i : i + FILE_BATCH]
                    tasks = [(os.path.join(config["pdf_dir"], f), meta_map.get(f, {}), 
                              scale['size'], scale['overlap']) for f in current_files]

                    # CPU 多进程解析
                    batch_docs = []
                    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
                        results = list(executor.map(parse_pdf_worker, tasks))
                        for res in results: batch_docs.extend(res)

                    # GPU 向量化
                    if batch_docs:
                        pbar.set_postfix(gpu="Active", chunks=len(batch_docs))
                        if vector_db is None:
                        if os.path.exists(save_path):
                            vector_db = FAISS.load_local(save_path, embed_model, allow_dangerous_deserialization=True)
                        else:
                            vector_db = FAISS.from_documents(batch_docs, embed_model)
                        else:
                            vector_db.add_documents(batch_docs)
                        
                        vector_db.save_local(save_path)
                    
                    pbar.update(len(current_files))
                    torch.cuda.empty_cache()
                    gc.collect()

        # 释放当前梯度的模型，为下一个梯度腾出显存
        del embed_model
        torch.cuda.empty_cache()
        time.sleep(5) 

if __name__ == "__main__":
    import time
    run_indexing()

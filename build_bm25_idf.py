#!/usr/bin/env python
"""Build BM25 IDF dictionary from FAISS fulltext indexes.
Usage: python build_bm25_idf.py
Output: bm25_idf.pkl
"""
import sys, os, math, pickle, re
from collections import Counter

# Add project dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FULLTEXT_INDEX_PATHS, MODEL_PATH
from embeddings import BgeEmbeddingsWrapper
from langchain_community.vectorstores import FAISS

INDEX_KEYS = ["en_fine", "en_std", "en_large", "en_para"]

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","can","shall","to","of","in","for","on","with","at",
    "by","from","as","into","through","during","before","after","about",
    "and","or","not","but","if","then","else","when","this","that",
    "these","those","it","its","we","they","he","she","which","who",
    "also","than","more","less","very","too","just","only","such","each",
    "all","both","few","most","other","some","any","no","nor","so","thus",
    "therefore","however","although","because","since","while","where","how",
    "what","here","there","et","al","between","under","over","above","below",
}

def tokenize(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s\-_\.]", " ", text)
    tokens = [t.strip(".-_") for t in text.split() if t.strip(".-_")]
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]

print("Loading embedding model (BGE-M3)...")
embed_model = BgeEmbeddingsWrapper()

all_texts = []
seen = set()

for key in INDEX_KEYS:
    path = FULLTEXT_INDEX_PATHS.get(key)
    if not path or not os.path.isdir(path):
        print(f"SKIP {key}: path not found ({path})")
        continue
    
    print(f"Loading FAISS index: {key} ({path})...")
    store = FAISS.load_local(path, embed_model, allow_dangerous_deserialization=True)
    
    docstore = store.docstore
    idx_to_id = store.index_to_docstore_id
    
    count = 0
    n_total = store.index.ntotal
    print(f"  Index size: {n_total} vectors")
    
    for i in range(n_total):
        doc_id = idx_to_id.get(i)
        if doc_id is None:
            continue
        
        result = docstore.search(doc_id)
        if result is None:
            continue
        
        text = result.page_content if hasattr(result, "page_content") else str(result)
        th = text[:200]
        if th not in seen and len(text) > 50:
            seen.add(th)
            all_texts.append(text)
            count += 1
        
        if count % 50000 == 0:
            print(f"  {count} unique chunks extracted...")
    
    print(f"  {key}: {count} chunks (from {n_total} vectors)")

print(f"\nTotal unique chunks: {len(all_texts)}")

# IDF computation
N = len(all_texts)
df = Counter()
doc_lengths = []

print("Computing document frequencies...")
for i, text in enumerate(all_texts):
    tokens = tokenize(text)
    for t in set(tokens):
        df[t] += 1
    doc_lengths.append(len(tokens))
    if (i+1) % 50000 == 0:
        print(f"  {i+1}/{N}...")

avgdl = sum(doc_lengths) / N if N > 0 else 1.0

print("Computing IDF...")
idf = {}
for t, dft in df.items():
    idf[t] = math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)

print(f"IDF built: N={N}, vocab={len(idf)}, avgdl={avgdl:.1f}")

# Show some stats
print("\nTop 20 highest IDF terms:")
for t, v in sorted(idf.items(), key=lambda x: -x[1])[:20]:
    print(f"  {t}: idf={v:.2f} (df={df[t]})")

print("\nSample mid-IDF terms:")
mid_items = sorted(idf.items(), key=lambda x: x[1])[len(idf)//2:len(idf)//2+10]
for t, v in mid_items:
    print(f"  {t}: idf={v:.2f} (df={df[t]})")

# Save
output = {"k1": 1.2, "b": 0.75, "idf": dict(idf), "avgdl": avgdl, "N": N}
out_path = os.path.join(os.path.dirname(__file__), "bm25_idf.pkl")
with open(out_path, "wb") as f:
    pickle.dump(output, f)

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")

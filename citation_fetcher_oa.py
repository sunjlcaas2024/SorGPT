#!/usr/bin/env python
"""Fetch citation counts from OpenAlex API and cache in SQLite.
OpenAlex is free, no API key required, 10 req/sec rate limit.
Supports pipe-separated DOI batch filtering.
"""
import sys, os, json, time, sqlite3, re, csv
from typing import Dict, List, Optional
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CSV_PATHS

OA_URL = "https://api.openalex.org/works"
BATCH_SIZE = 50  # DOIs per request (pipe-separated filter)
SLEEP = 0.12     # ~8 req/sec (safe below 10/sec limit)
DB_PATH = os.path.join(os.path.dirname(__file__), "citation_cache.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS citation_cache (
            doi TEXT PRIMARY KEY,
            citation_count INTEGER NOT NULL DEFAULT 0,
            title TEXT,
            year INTEGER,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_citation_count ON citation_cache(citation_count)")
    conn.commit()
    return conn

def normalize_doi(doi):
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi

def extract_dois_from_csv(csv_paths):
    dois = set()
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"WARNING: CSV not found: {csv_path}")
            continue
        print(f"Reading: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doi = row.get("doi", "") or row.get("DOI", "") or row.get("doi_link", "")
                doi = normalize_doi(doi)
                if doi and len(doi) > 5 and "/" in doi:
                    dois.add(doi)
    print(f"Total unique DOIs: {len(dois)}")
    return sorted(dois)

def fetch_batch_oa(dois):
    """Fetch citation counts using OpenAlex pipe-separated DOI filter."""
    doi_filter = "|".join(quote(d) for d in dois)
    url = f"{OA_URL}?filter=doi:{doi_filter}&select=doi,cited_by_count,publication_year,title&per_page={BATCH_SIZE}"
    try:
        resp = __import__("requests").get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        elif resp.status_code == 429:
            print("  Rate limited, sleeping 5s...")
            time.sleep(5)
            return fetch_batch_oa(dois)
        else:
            print(f"  API error {resp.status_code}")
            return []
    except Exception as e:
        print(f"  Request error: {e}")
        time.sleep(2)
        return []

def fetch_all_citations(conn, dois, force=False):
    if not force:
        cached = set()
        for (doi,) in conn.execute("SELECT doi FROM citation_cache").fetchall():
            cached.add(doi)
        new_dois = [d for d in dois if d not in cached]
        print(f"Cached: {len(cached)}, need: {len(new_dois)}")
    else:
        new_dois = dois
    
    if not new_dois:
        print("All cached.")
        return
    
    total = len(new_dois)
    batches = total // BATCH_SIZE + (1 if total % BATCH_SIZE else 0)
    
    for i in range(0, total, BATCH_SIZE):
        batch = new_dois[i:i+BATCH_SIZE]
        bn = i // BATCH_SIZE + 1
        print(f"Batch {bn}/{batches} ({i+1}-{min(i+BATCH_SIZE, total)} of {total})...", end=" ", flush=True)
        
        results = fetch_batch_oa(batch)
        found = 0
        for item in results:
            doi = normalize_doi(item.get("doi", ""))
            if not doi:
                continue
            count = item.get("cited_by_count", 0) or 0
            title = item.get("title", "")
            year = item.get("publication_year")
            conn.execute(
                "INSERT OR REPLACE INTO citation_cache(doi, citation_count, title, year) VALUES(?,?,?,?)",
                (doi, count, title, year)
            )
            found += 1
        
        conn.commit()
        print(f"{found} found")
        time.sleep(SLEEP)
    
    print("Done.")

def get_stats(conn):
    total = conn.execute("SELECT COUNT(*) FROM citation_cache").fetchone()[0]
    with_data = conn.execute("SELECT COUNT(*) FROM citation_cache WHERE citation_count > 0").fetchone()[0]
    avg = conn.execute("SELECT AVG(citation_count) FROM citation_cache WHERE citation_count > 0").fetchone()[0] or 0
    print(f"DOI cache: {total} total, {with_data} with citations (>0), avg={avg:.1f}")
    
    print("\nCitation distribution:")
    bins = [(0,0), (1,5), (6,20), (21,50), (51,100), (101,500), (501,9999)]
    for lo, hi in bins:
        if lo == 0 and hi == 0:
            cnt = conn.execute("SELECT COUNT(*) FROM citation_cache WHERE citation_count = 0").fetchone()[0]
        else:
            cnt = conn.execute("SELECT COUNT(*) FROM citation_cache WHERE citation_count >= ? AND citation_count <= ?", (lo, hi)).fetchone()[0]
        pct = cnt/total*100 if total > 0 else 0
        print(f"  {lo}-{hi if hi<9999 else inf}: {cnt} ({pct:.1f}%)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()
    
    conn = init_db()
    
    if args.stats:
        get_stats(conn)
    elif args.fetch:
        dois = extract_dois_from_csv(CSV_PATHS)
        if dois:
            fetch_all_citations(conn, dois, force=args.force)
            get_stats(conn)
    
    conn.close()

#!/usr/bin/env python
"""Fetch citation counts from Semantic Scholar API and cache in SQLite.
Usage:
    python citation_fetcher.py --fetch      # Fetch all DOIs
    python citation_fetcher.py --stats      # Show cache stats
    python citation_fetcher.py --update     # Incremental update (new DOIs only)
"""
import sys, os, json, time, sqlite3, re
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CSV_PATHS

# Semantic Scholar batch API
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
BATCH_SIZE = 500  # Max per batch request
SLEEP_BETWEEN_BATCHES = 1.1  # Rate limit: ~100 requests/5min

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

def normalize_doi(doi: str) -> str:
    """Normalize DOI to lowercase, stripped."""
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi

def extract_dois_from_csv(csv_paths: List[str]) -> List[str]:
    """Extract and normalize unique DOIs from CSV files."""
    import csv
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
                if doi and len(doi) > 5:
                    dois.add(doi)
    print(f"Total unique DOIs: {len(dois)}")
    return sorted(dois)

def fetch_batch(dois: List[str]) -> List[Dict]:
    """Fetch citation counts for a batch of DOIs from Semantic Scholar."""
    payload = {"ids": [d for d in dois if d]}
    try:
        resp = __import__("requests").post(
            S2_BATCH_URL,
            params={"fields": "citationCount,title,year"},
            json=payload,
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json() or []
        elif resp.status_code == 429:
            print(f"  Rate limited, sleeping 60s...")
            time.sleep(60)
            return fetch_batch(dois)  # Retry once
        else:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"  Request error: {e}")
        return []

def fetch_all_citations(conn, dois: List[str], force: bool = False):
    """Fetch citation counts for all DOIs, caching in SQLite."""
    # Skip already-cached DOIs
    if not force:
        cached = set()
        rows = conn.execute("SELECT doi FROM citation_cache").fetchall()
        for (doi,) in rows:
            cached.add(doi)
        new_dois = [d for d in dois if d not in cached]
        print(f"Already cached: {len(cached)}, need to fetch: {len(new_dois)}")
    else:
        new_dois = dois
    
    if not new_dois:
        print("All DOIs already cached.")
        return
    
    total = len(new_dois)
    for i in range(0, total, BATCH_SIZE):
        batch = new_dois[i:i+BATCH_SIZE]
        print(f"Fetching batch {i//BATCH_SIZE + 1}/{(total+BATCH_SIZE-1)//BATCH_SIZE} "
              f"({i+1}-{min(i+BATCH_SIZE, total)} of {total})...")
        
        results = fetch_batch(batch)
        
        for item in results:
            item_doi = ""
            if isinstance(item, dict):
                ext_ids = item.get("externalIds") or {}
                item_doi = normalize_doi(ext_ids.get("DOI", ""))
                if not item_doi:
                    # Try paperId lookup fallback
                    continue
                count = item.get("citationCount", 0) or 0
                title = item.get("title", "")
                year = item.get("year")
                
                conn.execute(
                    "INSERT OR REPLACE INTO citation_cache(doi, citation_count, title, year) VALUES(?,?,?,?)",
                    (item_doi, count, title, year)
                )
        
        conn.commit()
        time.sleep(SLEEP_BETWEEN_BATCHES)
    
    print("Done fetching.")

def get_citation_count(conn, doi: str) -> Optional[int]:
    """Look up citation count for a DOI."""
    doi = normalize_doi(doi)
    if not doi:
        return None
    row = conn.execute(
        "SELECT citation_count FROM citation_cache WHERE doi=?",
        (doi,)
    ).fetchone()
    return row[0] if row else None

def get_citation_stats(conn):
    """Print cache statistics."""
    total = conn.execute("SELECT COUNT(*) FROM citation_cache").fetchone()[0]
    with_data = conn.execute(
        "SELECT COUNT(*) FROM citation_cache WHERE citation_count > 0"
    ).fetchone()[0]
    avg = conn.execute(
        "SELECT AVG(citation_count) FROM citation_cache WHERE citation_count > 0"
    ).fetchone()[0] or 0
    
    print(f"Cache stats: {total} DOIs total, {with_data} with citations (>0)")
    print(f"Average citations (non-zero): {avg:.1f}")
    
    # Distribution
    print("\nCitation distribution:")
    bins = [(0,0), (1,5), (6,20), (21,50), (51,100), (101,500), (501,9999)]
    for lo, hi in bins:
        if lo == 0 and hi == 0:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM citation_cache WHERE citation_count = 0"
            ).fetchone()[0]
            label = "0"
        else:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM citation_cache WHERE citation_count >= ? AND citation_count <= ?",
                (lo, hi)
            ).fetchone()[0]
            label = f"{lo}-{hi}"
        pct = cnt/total*100 if total > 0 else 0
        print(f"  {label}: {cnt} ({pct:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Citation count fetcher for SorGPT")
    parser.add_argument("--fetch", action="store_true", help="Fetch citations for all DOIs in CSVs")
    parser.add_argument("--update", action="store_true", help="Incremental update (new DOIs only)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--lookup", type=str, help="Look up a specific DOI")
    args = parser.parse_args()
    
    conn = init_db()
    
    if args.stats:
        get_citation_stats(conn)
    elif args.lookup:
        doi = normalize_doi(args.lookup)
        count = get_citation_count(conn, doi)
        print(f"DOI: {doi}")
        print(f"Citation count: {count if count is not None else not found}")
    elif args.fetch or args.update:
        dois = extract_dois_from_csv(CSV_PATHS)
        if dois:
            fetch_all_citations(conn, dois, force=args.force)
            get_citation_stats(conn)
    else:
        parser.print_help()
    
    conn.close()

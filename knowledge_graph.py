"""
SorGPT Knowledge Graph Builder
从 4 个 SQLite 数据库构建高粱基因知识图谱
Nodes: genes, GO terms, Pfam domains, QTLs, traits, orthologs
Edges: gene→GO, gene→Pfam, QTL→gene, gene→ortholog, gene→function
"""
import sqlite3, os, sys, pickle, time
import networkx as nx
from collections import defaultdict

DB_DIR = "/vol/sunjilin/website/data/agent/sorghum_rag/db"
GRAPH_PATH = "/vol/sunjilin/website/data/agent/sorghum_rag/knowledge_graph.pkl"

def build_graph():
    G = nx.MultiDiGraph()
    _genes = os.path.join(DB_DIR, "sorghum_genes.db")
    _known = os.path.join(DB_DIR, "known_genes.db")
    _qtl   = os.path.join(DB_DIR, "qtl.db")

    # ============================================================
    # 1. Gene nodes
    # ============================================================
    print("[1/6] Loading genes...", end=" ", flush=True)
    conn = sqlite3.connect(_genes)
    rows = conn.execute("SELECT gene_id, chr, start, end, strand FROM genes").fetchall()
    for gid, ch, st, ed, strand in rows:
        G.add_node(gid, type="gene", chr=ch, start=st, end=ed, strand=strand)
    conn.close()
    print(f"{len(rows):,} gene nodes")

    # ============================================================
    # 2. GO annotation edges
    # ============================================================
    print("[2/8] Loading GO annotations...", end=" ", flush=True)
    conn = sqlite3.connect(_genes)
    rows = conn.execute("SELECT gene_id, go_id, go_name, namespace FROM go_annotation").fetchall()
    for gid, go_id, go_name, ns in rows:
        G.add_node(go_id, type="go_term", name=go_name, namespace=ns)
        G.add_edge(gid, go_id, relation="has_go")
    conn.close()
    go_count = len(set(r[1] for r in rows))
    print(f"{len(rows):,} edges, {go_count:,} GO nodes")

    # ============================================================
    # 3. Pfam domain edges
    # ============================================================
    print("[3/8] Loading Pfam domains...", end=" ", flush=True)
    conn = sqlite3.connect(_genes)
    rows = conn.execute("SELECT gene_id, pfam_id, pfam_name FROM pfam").fetchall()
    for gid, pfam_id, pfam_name in rows:
        G.add_node(pfam_id, type="pfam", name=pfam_name)
        G.add_edge(gid, pfam_id, relation="has_pfam")
    conn.close()
    p_count = len(set(r[1] for r in rows))
    print(f"{len(rows):,} edges, {p_count:,} Pfam nodes")

    # ============================================================
    # 4. Functional annotation edges
    # ============================================================
    print("[4/8] Loading functional annotations...", end=" ", flush=True)
    conn = sqlite3.connect(_genes)
    rows = conn.execute(
        "SELECT gene_id, db_name, term_id, term_name, category FROM func_annotation"
    ).fetchall()
    for gid, db_name, term_id, term_name, cat in rows:
        func_node = f"{db_name}:{term_id}"
        G.add_node(func_node, type="function", db=db_name, name=term_name, category=cat)
        G.add_edge(gid, func_node, relation="has_function")
    conn.close()
    print(f"{len(rows):,} edges")

    # ============================================================
    # 5. Ortholog edges
    # ============================================================
    print("[5/8] Loading orthologs...", end=" ", flush=True)
    conn = sqlite3.connect(_genes)
    rows = conn.execute(
        "SELECT DISTINCT btx623_id, genome, ortho_id, position FROM orthologs"
    ).fetchall()
    for gid, genome, ortho_id, pos in rows:
        ortho_node = f"{genome}:{ortho_id}"
        G.add_node(ortho_node, type="ortholog", genome=genome, position=pos)
        G.add_edge(gid, ortho_node, relation="has_ortholog")
    conn.close()
    o_count = len(set(r[2] for r in rows))
    print(f"{len(rows):,} edges, {o_count:,} ortholog nodes")

    # ============================================================
    # 5. Variety/Accession nodes + Phenotype + Metabolite edges (omics.db)
    # ============================================================
    _omics = os.path.join(DB_DIR, "omics.db")
    if os.path.exists(_omics):
        print("[5/8] Loading phenotypes & metabolites...", end=" ", flush=True)
        conn = sqlite3.connect(_omics)

        # Phenotype traits → variety edges
        rows = conn.execute("SELECT sample_id, trait, value, unit FROM phenotype_quant").fetchall()
        for sid, trait, value, unit in rows:
            G.add_node(sid, type="accession")
            trait_node = f"TRAIT:{trait}"
            G.add_node(trait_node, type="phenotype", name=trait, unit=unit)
            G.add_edge(sid, trait_node, relation="has_phenotype", value=value)

        rows = conn.execute("SELECT sample_id, trait, value_info FROM phenotype_qual").fetchall()
        for sid, trait, val_info in rows:
            G.add_node(sid, type="accession")
            trait_node = f"TRAIT:{trait}"
            G.add_node(trait_node, type="phenotype", name=trait)
            G.add_edge(sid, trait_node, relation="has_phenotype", value=val_info)

        # Metabolite → variety edges
        rows = conn.execute("SELECT sample_id, metabolite, intensity FROM metabolite_sample").fetchall()
        for sid, met, intensity in rows:
            G.add_node(sid, type="accession")
            met_node = f"MET:{met}"
            G.add_node(met_node, type="metabolite", name=met)
            G.add_edge(sid, met_node, relation="has_metabolite", intensity=intensity)

        # Metabolite → pathway edges
        rows = conn.execute("SELECT name, superclass, class, pathway FROM metabolite_meta").fetchall()
        for name, sclass, mclass, pathway in rows:
            met_node = f"MET:{name}"
            G.add_node(met_node, type="metabolite", name=name, superclass=sclass, mclass=mclass)
            if pathway:
                pw_node = f"PATH:{pathway}"
                G.add_node(pw_node, type="pathway", name=pathway)
                G.add_edge(met_node, pw_node, relation="belongs_to_pathway")

        conn.close()
        acc_count = len([n for n,d in G.nodes(data=True) if d.get("type")=="accession"])
        phe_count = len([n for n,d in G.nodes(data=True) if d.get("type")=="phenotype"])
        met_count = len([n for n,d in G.nodes(data=True) if d.get("type")=="metabolite"])
        pw_count = len([n for n,d in G.nodes(data=True) if d.get("type")=="pathway"])
        print(f"{acc_count} accessions, {phe_count} traits, {met_count} metabolites, {pw_count} pathways")

    # ============================================================
    # 6. QTL → gene edges
    # ============================================================
    print("[7/8] Loading QTL associations...", end=" ", flush=True)
    conn = sqlite3.connect(_qtl)
    qtl_meta = {}
    for qid, trait, pub, pop, v3_chr in conn.execute(
        "SELECT qtl_id, trait, publication, population, v3_chr FROM qtl_loci"
    ).fetchall():
        qtl_meta[qid] = {"trait": trait, "publication": pub, "population": pop, "chr": v3_chr}

    # Add all QTL nodes first
    for qid, meta in qtl_meta.items():
        G.add_node(qid, type="qtl", trait=meta.get("trait",""),
                   publication=meta.get("publication",""), chr=meta.get("chr",""))
    # Add QTL→gene edges (all 1.48M)
    rows = conn.execute("SELECT qtl_id, gene_id FROM qtl_genes").fetchall()
    for qid, gid in rows:
        if G.has_node(qid) and G.has_node(gid):
            G.add_edge(qid, gid, relation="contains_gene")
    conn.close()
    q_count = len(qtl_meta)
    edge_count = sum(1 for _,_,d in G.edges(data=True) if d.get("relation")=="contains_gene")
    print(f"{edge_count:,} edges, {q_count:,} QTL nodes")

    # ============================================================
    # 6. Known gene nodes (from known_genes.db)
    # ============================================================
    print("[8/8] Loading known genes...", end=" ", flush=True)
    conn = sqlite3.connect(_known)
    rows = conn.execute(
        "SELECT gene_name, gene_id, trait, annotation, first_author, full_citation FROM known_genes"
    ).fetchall()
    for gname, gid, trait, annot, author, citation in rows:
        if gid and G.has_node(gid):
            G.nodes[gid]["known"] = True
            G.nodes[gid]["gene_name"] = gname
            G.nodes[gid]["trait"] = trait
            G.nodes[gid]["function"] = annot
            G.nodes[gid]["evidence"] = f"{author}: {citation[:80] if citation else ''}"
    conn.close()
    has_known = sum(1 for n in G.nodes if G.nodes[n].get("known"))
    print(f"{len(rows)} known genes annotated, {has_known} matched to graph")

    # ============================================================
    # 9. Paper nodes from publication CSV + citation counts
    # ============================================================
    _csv_dir = "/vol/sunjilin/website/data/publication"
    import csv as _csv

    # Pre-extract known gene info for paper matching
    known_info = {}
    for gid, gdata in G.nodes(data=True):
        if gdata.get("type") == "gene" and gdata.get("known"):
            evidence = (gdata.get("evidence") or "").lower()
            gene_name = (gdata.get("gene_name") or "").lower()
            known_info[gid] = {"evidence": evidence, "gene_name": gene_name}

    paper_count = 0
    gene_paper_edges = 0
    for csv_file in ["english_content_merged.csv"]:
        csv_path = os.path.join(_csv_dir, csv_file)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                title = (row.get("filename") or row.get("title") or row.get("Article Title") or "").strip()
                if not title or len(title) < 5:
                    continue
                doi = (row.get("doi") or row.get("DOI") or row.get("DOI Link") or "").strip()
                authors = (row.get("Author Full Names") or row.get("Authors") or row.get("authors") or "").strip()
                journal = (row.get("Source Title") or row.get("journal") or row.get("Journal") or "").strip()
                year = (row.get("Publication Year") or row.get("year") or row.get("Year") or "").strip()

                paper_id = f"PAPER:{title[:120]}"
                G.add_node(paper_id, type="paper", title=title, authors=authors,
                          journal=journal, year=year, doi=doi)
                paper_count += 1

                # Link known genes to their papers via author + gene name
                authors_lower = (authors or "").lower()
                title_lower = title.lower()
                doi_lower = (doi or "").lower()
                for gid, info in known_info.items():
                    ev = info["evidence"]
                    gn = info["gene_name"]
                    # Match by: first author last name in evidence, OR gene name in title, OR DOI in evidence
                    first_author = authors_lower.split(",")[0].split(";")[0].strip() if authors_lower else ""
                    if (first_author and len(first_author) > 2 and first_author in ev) or \
                       (gn and gn in title_lower) or \
                       (doi_lower and doi_lower in ev):
                        G.add_edge(gid, paper_id, relation="cited_in")
                        gene_paper_edges += 1
                        break

    print(f"[9/9] {paper_count:,} papers, {gene_paper_edges} gene-paper links")

    # Save
    print(f"\nSaving graph ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)...", end=" ", flush=True)
    t0 = time.time()
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(GRAPH_PATH) / (1024**2)
    print(f"{mb:.0f}MB, {time.time()-t0:.0f}s")
    return G


# ============================================================
# Query functions
# ============================================================
class GeneKB:
    def __init__(self, graph_path=GRAPH_PATH):
        print(f"Loading graph from {graph_path}...", end=" ", flush=True)
        t0 = time.time()
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)
        print(f"{self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges, {time.time()-t0:.0f}s")

    def paper_search(self, keyword, max_results=10):
        """按关键词搜索论文（标题/作者/期刊/DOI），支持多词空格分隔"""
        matches = []
        terms = keyword.lower().split()
        for n, data in self.G.nodes(data=True):
            if data.get("type") != "paper":
                continue
            title = (data.get("title") or "").lower()
            authors = (data.get("authors") or "").lower()
            journal = (data.get("journal") or "").lower()
            doi = (data.get("doi") or "").lower()
            score = 0
            for kw in terms:
                score += (5 if kw in title else 0) + (3 if kw in authors else 0) + \
                         (2 if kw in journal else 0) + (4 if kw in doi else 0)
            if score > 0:
                matches.append((score, data))
        matches.sort(key=lambda x: -x[0])
        info = [f"Papers matching '{keyword}': {len(matches)} found"]
        for score, data in matches[:max_results]:
            info.append(f"  {data.get('title','')[:120]}")
            info.append(f"    {data.get('authors','')[:80]} | {data.get('journal','')} ({data.get('year','')})")
            if data.get("doi"):
                info.append(f"    DOI: {data.get('doi','')}")
            # Genes linked to this paper
            paper_id = f"PAPER:{data.get('title','')[:120]}"
            genes = []
            for gid, _, edata in self.G.in_edges(paper_id, data=True):
                if edata.get("relation") == "cited_in":
                    gdata = self.G.nodes[gid]
                    genes.append(f"{gid} ({gdata.get('gene_name','')})")
            if genes:
                info.append(f"    Linked genes: {', '.join(genes[:5])}")
        return "\n".join(info)

    def gene_info(self, gene_id):
        """查询单个基因的完整注释"""
        if gene_id not in self.G:
            return f"Gene {gene_id} not found"
        node = self.G.nodes[gene_id]
        info = [f"Gene: {gene_id}", f"  Location: {node.get('chr','?')}:{node.get('start','?')}-{node.get('end','?')}"]
        if node.get("known"):
            info.append(f"  Known: {node.get('gene_name','')} — {node.get('trait','')}")
            info.append(f"  Function: {node.get('function','')}")
            if node.get("evidence"):
                info.append(f"  Evidence: {node.get('evidence','')[:100]}")
        # QTL intervals containing this gene
        qtls_all = []
        for qid, _, edata in self.G.in_edges(gene_id, data=True):
            if edata.get("relation") == "contains_gene":
                qdata = self.G.nodes[qid]
                qtls_all.append((qid, qdata.get("trait",""), qdata.get("chr","")))
        if qtls_all:
            info.append(f"  QTL intervals ({len(qtls_all)}):")
            for q, t, c in qtls_all[:10]:
                info.append(f"    {q} [{c}] → {t}")
            if len(qtls_all) > 10:
                info.append(f"    ... and {len(qtls_all)-10} more")
        # Orthologs
        orthologs = []
        for _, neighbor, data in self.G.out_edges(gene_id, data=True):
            if data["relation"] == "has_ortholog":
                orthologs.append((neighbor, self.G.nodes[neighbor].get("genome","")))
        if orthologs:
            # Group by genome
            by_genome = {}
            for o, g in orthologs:
                by_genome.setdefault(g, []).append(o)
            info.append(f"  Orthologs ({len(orthologs)}):")
            for genome, ids in sorted(by_genome.items()):
                info.append(f"    {genome}: {', '.join(ids[:3])}{'...' if len(ids)>3 else ''}")
        # Functional annotations (InterPro, KEGG, etc.)
        funcs = defaultdict(list)
        for _, neighbor, data in self.G.out_edges(gene_id, data=True):
            if data["relation"] == "has_function":
                funcs[data.get("db","?")].append(f"{neighbor}: {data.get('name','')}")
        if funcs:
            info.append(f"  Functional annotations:")
            for db, items in sorted(funcs.items()):
                info.append(f"    {db}: {items[0]}")
                if len(items) > 1:
                    info.append(f"      ... and {len(items)-1} more {db} terms")
        # GO terms
        go_terms = []
        for _, neighbor, data in self.G.out_edges(gene_id, data=True):
            if data["relation"] == "has_go":
                go_terms.append(f"    {neighbor}: {self.G.nodes[neighbor].get('name','')}")
        if go_terms:
            info.append(f"  GO terms ({len(go_terms)}):")
            info.extend(go_terms[:10])
        # Pfam
        pfams = []
        for _, neighbor, data in self.G.out_edges(gene_id, data=True):
            if data["relation"] == "has_pfam":
                pfams.append(f"    {neighbor}: {self.G.nodes[neighbor].get('name','')}")
        if pfams:
            info.append(f"  Pfam domains ({len(pfams)}):")
            info.extend(pfams[:5])
        return "\n".join(info)

    def gene_network(self, gene_id, max_hops=2):
        """查询基因的关联网络（N跳邻居）"""
        if gene_id not in self.G:
            return f"Gene {gene_id} not found"
        sub = nx.ego_graph(self.G, gene_id, radius=max_hops, undirected=True)
        nodes_by_type = defaultdict(int)
        for n in sub.nodes:
            nodes_by_type[self.G.nodes[n].get("type", "unknown")] += 1
        info = [f"Network around {gene_id} ({max_hops} hops): {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges"]
        for t, c in sorted(nodes_by_type.items()):
            info.append(f"  {t}: {c}")
        return "\n".join(info)

    def trait_genes(self, trait_keyword):
        """查询与性状相关的已知基因 + QTL 共定位基因"""
        info = []
        # 1. Find known genes matching the trait
        known_matches = []
        for n, data in self.G.nodes(data=True):
            if data.get("type") == "gene" and data.get("known"):
                t = (data.get("trait") or "").lower()
                if trait_keyword.lower() in t:
                    known_matches.append((n, data))
        if known_matches:
            info.append(f"Known genes for '{trait_keyword}': {len(known_matches)}")
            for gid, data in known_matches:
                name = data.get("gene_name", "")
                func = (data.get("function") or "")[:80]
                evidence = (data.get("evidence") or "")[:60]
                info.append(f"  {gid} | {name} | {func}")
                if evidence:
                    info.append(f"    Evidence: {evidence}")
                # Find QTLs containing this gene
                qtls = []
                for qid, _, edata in self.G.in_edges(gid, data=True):
                    if edata.get("relation") == "contains_gene":
                        qdata = self.G.nodes[qid]
                        qtls.append((qid, qdata.get("trait",""), qdata.get("chr","")))
                if qtls:
                    # Show QTLs sorted by whether trait matches
                    matching = [(q, t, c) for q, t, c in qtls if trait_keyword.lower() in t.lower()]
                    other = [(q, t, c) for q, t, c in qtls if trait_keyword.lower() not in t.lower()]
                    if matching:
                        info.append(f"    QTLs (matching '{trait_keyword}'): {', '.join(q for q,_,_ in matching[:5])}")
                    if other:
                        info.append(f"    QTLs (other traits, same locus): {', '.join(q for q,_,_ in other[:5])}")
        else:
            info.append(f"No known genes found for '{trait_keyword}'")
        return "\n".join(info)

    def co_qtl_genes(self, trait_keyword, max_qtls=10, max_genes=50):
        """查询与某个性状相关的所有基因（通过QTL共定位）"""
        matches = []
        for n, data in self.G.nodes(data=True):
            if data.get("type") == "qtl" and trait_keyword.lower() in (data.get("trait","") or "").lower():
                genes = []
                for _, neighbor, edata in self.G.out_edges(n, data=True):
                    if edata["relation"] == "contains_gene":
                        genes.append(neighbor)
                # Parse chromosome number for sorting
                chr_str = (data.get("chr","") or "").replace("Chr","").replace("chr","")
                try:
                    chr_num = int(chr_str)
                except:
                    chr_num = 99
                matches.append((n, data.get("trait",""), chr_num, genes))
        if not matches:
            return f"No QTL found for: {trait_keyword}"
        # Sort by chromosome then gene count
        matches.sort(key=lambda x: (x[2], -len(x[3])))
        info = [f"QTLs related to '{trait_keyword}': {len(matches)} found"]
        # Show breakdown by chromosome
        chr_counts = defaultdict(int)
        for _, _, c, gs in matches:
            chr_counts[c] += 1
        info.append("  By chromosome: " + ", ".join(f"Chr{c}:{cnt}" for c,cnt in sorted(chr_counts.items()) if c != 99))
        seen_genes = set()
        qtls_shown = 0
        for qid, trait, chrom, genes in matches:
            if qtls_shown >= max_qtls:
                break
            info.append(f"  {qid} [Chr{chrom}]: {trait} → {len(genes)} genes")
            for g in genes[:8]:
                if g not in seen_genes:
                    gdata = self.G.nodes[g]
                    known_str = f" ★{gdata.get('gene_name','')}" if gdata.get('known') else ""
                    info.append(f"    {g}{known_str}")
                    seen_genes.add(g)
            qtls_shown += 1
            if len(seen_genes) >= max_genes:
                break
        return "\n".join(info)

    def shared_pathway(self, gene_a, gene_b):
        """查找两个基因的共享通路/GO/Pfam"""
        if gene_a not in self.G or gene_b not in self.G:
            return "Gene not found"
        a_neighbors = set()
        for _, n, d in self.G.out_edges(gene_a, data=True):
            if d["relation"] in ("has_go", "has_pfam"):
                a_neighbors.add((n, d["relation"], self.G.nodes[n].get("name","")))
        b_neighbors = set()
        for _, n, d in self.G.out_edges(gene_b, data=True):
            if d["relation"] in ("has_go", "has_pfam"):
                b_neighbors.add((n, d["relation"], self.G.nodes[n].get("name","")))
        shared = a_neighbors & b_neighbors
        info = [f"Shared annotations between {gene_a} and {gene_b}: {len(shared)}"]
        for n, rel, name in sorted(shared, key=lambda x: x[1]):
            info.append(f"  [{rel}] {n}: {name}")
        return "\n".join(info)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true", help="Build knowledge graph from databases")
    p.add_argument("--gene", type=str, help="Query gene info")
    p.add_argument("--network", type=str, help="Query gene network (N-hop ego graph)")
    p.add_argument("--trait", type=str, help="Find genes by trait keyword (QTL co-localization)")
    p.add_argument("--shared", nargs=2, help="Find shared annotations between two genes")
    p.add_argument("--paper", type=str, help="Search papers by keyword (title/author/journal/DOI)")
    p.add_argument("--stats", action="store_true", help="Print graph statistics")
    args = p.parse_args()

    if args.build:
        build_graph()
    elif os.path.exists(GRAPH_PATH):
        kb = GeneKB(GRAPH_PATH)
        if args.stats:
            types = defaultdict(int)
            for n, d in kb.G.nodes(data=True):
                types[d.get("type","?")] += 1
            print("Node types:")
            for t, c in sorted(types.items(), key=lambda x: -x[1]):
                print(f"  {t}: {c:,}")
        elif args.gene:
            print(kb.gene_info(args.gene))
        elif args.network:
            print(kb.gene_network(args.network))
        elif args.trait:
            print(kb.co_qtl_genes(args.trait))
        elif args.shared:
            print(kb.shared_pathway(args.shared[0], args.shared[1]))
        elif args.paper:
            print(kb.paper_search(args.paper))
    else:
        print("Graph not built. Run with --build first.")

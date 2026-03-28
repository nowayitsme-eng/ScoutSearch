# search_engine_barrels.py
# BARREL-OPTIMIZED SEARCH ENGINE - Loads only required barrels per query
import csv
import json
import math
import os
import re
import time
from collections import defaultdict

# ---------- CONFIG & PATHS ----------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "index")
BARREL_DIR = os.path.join(INDEX_DIR, "barrels")
LEXICON_PATH = os.path.join(INDEX_DIR, "lexicon_complete.json")
FORWARD_INDEX_PATH = os.path.join(INDEX_DIR, "forward_index_termid.json")
TERM_TO_BARREL_MAP_PATH = os.path.join(BARREL_DIR, "term_to_barrel_map.json")
MARKET_VALUE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "player_latest_market_value", "player_latest_market_value.csv")
PROFILE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "complete_player_profiles.json")

# BM25 parameters
K1 = 1.2
B = 0.75

# Scoring boosts
NAME_TOKEN_WEIGHT = 0.75
NAME_PREFIX_BONUS = 1.25
EXACT_NAME_BONUS = 3.0
RAW_SUBSTRING_BONUS = 0.25
MARKET_VALUE_WEIGHT = 12.0
PROFILE_LENGTH_WEIGHT = 4.0
NON_NAME_MATCH_PENALTY = 1.5

# ---------- TEXT NORMALIZATION ----------

COMPREHENSIVE_STOP_WORDS = {
    "the", "and", "in", "for", "with", "on", "at", "from", "by", "as", "is", "was",
    "are", "were", "be", "been", "have", "has", "had", "to", "of", "a", "an", "that",
    "this", "these", "those", "it", "its", "or", "but", "not", "what", "which", "who",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "player", "club", "team", "football",
    "soccer", "match", "game", "season", "league", "cup", "champions", "premier", "la",
    "bundesliga", "serie", "current", "main", "position", "nationality", "birth", "place",
    # Universal terms that appear in ALL documents (filtering for memory/performance)
    "comprehensive", "international", "performance", "transfermarkt", "injury",
    "summary", "market", "history", "database", "value",
    # Stemmed versions and other universal terms
    "data", "teammat", "sourc", "career", "assist", "app", "minut",
    "available", "national", "significant", "teammate", "transfer", "goal"
}

def simple_stemmer(word: str) -> str:
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        return word[:-2]
    elif word.endswith("es") and len(word) > 4:
        return word[:-2]
    elif word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word

def normalize_and_tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    result = []
    for w in tokens:
        if w in COMPREHENSIVE_STOP_WORDS or len(w) <= 2:
            continue
        result.append(simple_stemmer(w))
    return result

def normalize_name_tokens(value: str):
    if not isinstance(value, str):
        return []
    tokens = re.findall(r"[a-z]+", value.lower())
    return [simple_stemmer(tok) for tok in tokens if tok]

def build_name_metadata(name: str):
    tokens = normalize_name_tokens(name)
    token_set = set(tokens)
    normalized = " ".join(tokens)
    return {
        "tokens": tokens,
        "token_set": token_set,
        "normalized": normalized,
        "raw_lower": name.lower() if isinstance(name, str) else "",
    }

def load_market_values(path: str):
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    player_id = int(row.get("player_id", ""))
                except (TypeError, ValueError):
                    continue
                raw_value = row.get("value")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                date_key = row.get("date_unix", "") or ""
                current = values.get(player_id)
                if current is None or date_key > current[0]:
                    values[player_id] = (date_key, value)
    except FileNotFoundError:
        print(f"[warn] Market value file not found at {path}")
        return {}
    return {pid: info[1] for pid, info in values.items()}

def load_profile_lengths(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        print(f"[warn] Profile data file not found at {path}")
        return {}

    lengths = {}
    for entry in data:
        player_id = entry.get("player_id")
        if not isinstance(player_id, int):
            continue
        detailed = entry.get("detailed_content")
        if isinstance(detailed, str) and detailed:
            lengths[player_id] = len(detailed)
    return lengths

# ---------- LOAD STATIC INDEXES (NOT INVERTED INDEX) ----------

print("[init] Loading lexicon...")
with open(LEXICON_PATH, "r", encoding="utf-8") as f:
    lexicon_entries = json.load(f)
token_to_id = {entry["token"]: entry["term_id"] for entry in lexicon_entries}
termid_to_token = {entry["term_id"]: entry["token"] for entry in lexicon_entries}
term_document_frequency = {entry["term_id"]: entry["df"] for entry in lexicon_entries}
print(f"[done] Lexicon loaded: {len(token_to_id):,} tokens")

print("[init] Loading forward index...")
with open(FORWARD_INDEX_PATH, "r", encoding="utf-8") as f:
    forward_index = json.load(f)
doc_by_id = {doc["player_id"]: doc for doc in forward_index}
N = len(doc_by_id)
avg_doc_len = sum(d["total_terms"] for d in forward_index) / N if N > 0 else 0.0
name_metadata = {doc_id: build_name_metadata(doc.get("player_name"))
                 for doc_id, doc in doc_by_id.items()}
print(f"[done] Forward index: {N:,} documents (avg_len={avg_doc_len:.2f})")

print("[init] Loading term-to-barrel mapping...")
with open(TERM_TO_BARREL_MAP_PATH, "r", encoding="utf-8") as f:
    term_to_barrel = json.load(f)
print(f"[done] Term-to-barrel map loaded: {len(term_to_barrel):,} mappings")

print("[init] Loading market values...")
player_market_value = load_market_values(MARKET_VALUE_PATH)
max_market_value = max(player_market_value.values(), default=0.0)
market_value_log_max = math.log1p(max_market_value) if max_market_value > 0 else 1.0
print(f"[done] Market values loaded for {len(player_market_value):,} players")

print("[init] Loading profile metadata...")
profile_length_by_id = load_profile_lengths(PROFILE_DATA_PATH)
max_profile_length = max(profile_length_by_id.values(), default=0)
profile_length_log_max = math.log1p(max_profile_length) if max_profile_length > 0 else 1.0
print(f"[done] Profile metadata loaded for {len(profile_length_by_id):,} players")

# ---------- BARREL CACHE (LRU-like) ----------

barrel_cache = {}
MAX_CACHED_BARRELS = 10  # Keep only 10 barrels in memory at once

def load_barrel(barrel_name: str):
    """Load a barrel file and cache it. Implements simple LRU eviction."""
    if barrel_name in barrel_cache:
        return barrel_cache[barrel_name]
    
    barrel_path = os.path.join(BARREL_DIR, f"{barrel_name}.json")
    try:
        with open(barrel_path, "r", encoding="utf-8") as f:
            barrel_data = json.load(f)
        
        # Cache management
        if len(barrel_cache) >= MAX_CACHED_BARRELS:
            # Remove oldest (first) entry
            oldest_key = next(iter(barrel_cache))
            del barrel_cache[oldest_key]
        
        barrel_cache[barrel_name] = barrel_data
        return barrel_data
    except FileNotFoundError:
        print(f"[error] Barrel file not found: {barrel_path}")
        return None

# ---------- QUERY TO TERM IDs ----------

def tokens_to_term_ids(tokens):
    seen = set()
    unique_term_ids = []
    for tok in tokens:
        tid = token_to_id.get(tok)
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        unique_term_ids.append(tid)
    return unique_term_ids

# ---------- BM25 SCORING ----------

def bm25_score(tf, df, doc_len, N, avg_doc_len, k1=K1, b=B):
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    denom = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
    return idf * (tf * (k1 + 1) / denom)

# ---------- BARREL-BASED SEARCH ----------

def search(query: str, top_k: int = 10, verbose: bool = True):
    start_time = time.perf_counter()
    
    log = print if verbose else (lambda *args, **kwargs: None)
    
    log(f"\n[query] {query}")
    query_tokens = normalize_and_tokenize(query)
    term_ids = tokens_to_term_ids(query_tokens)
    
    if not term_ids:
        elapsed = (time.perf_counter() - start_time) * 1000
        log(f"No query terms found in lexicon. (took {elapsed:.2f} ms)")
        return []
    
    log("Query tokens -> term_ids:",
        [(termid_to_token.get(tid, "?"), tid) for tid in term_ids])
    
    # **KEY OPTIMIZATION: Determine which barrels to load**
    required_barrels = set()
    for tid in term_ids:
        barrel_name = term_to_barrel.get(str(tid))
        if barrel_name:
            required_barrels.add(barrel_name)
    
    log(f"[barrels] Loading {len(required_barrels)} barrel(s): {sorted(required_barrels)}")
    
    # Load only required barrels
    barrel_load_start = time.perf_counter()
    loaded_barrels = {}
    for barrel_name in required_barrels:
        barrel_data = load_barrel(barrel_name)
        if barrel_data:
            loaded_barrels[barrel_name] = barrel_data
    barrel_load_time = (time.perf_counter() - barrel_load_start) * 1000
    log(f"[barrels] Loaded in {barrel_load_time:.2f} ms")
    
    # BM25 scoring using barrel data
    scores = defaultdict(float)
    
    for tid in term_ids:
        df = term_document_frequency.get(tid, 0)
        if df == 0:
            continue
        
        # Get barrel for this term
        barrel_name = term_to_barrel.get(str(tid))
        if not barrel_name or barrel_name not in loaded_barrels:
            continue
        
        barrel_data = loaded_barrels[barrel_name]
        inverted_index_part = barrel_data.get("inverted_index", {})
        
        # Get postings for this term
        term_data = inverted_index_part.get(str(tid))
        if not term_data:
            continue
        
        postings = term_data.get("postings", {})
        
        for doc_id_str, info in postings.items():
            doc_id = int(doc_id_str)
            tf = info["tf"]
            doc_len = doc_by_id[doc_id]["total_terms"]
            scores[doc_id] += bm25_score(tf, df, doc_len, N, avg_doc_len)
    
    # Metadata boosting (same as before)
    if scores:
        query_name_tokens = normalize_name_tokens(query)
        query_name = " ".join(query_name_tokens)
        raw_query_lower = query.lower().strip()
        
        for doc_id in scores:
            boost = 0.0
            meta = name_metadata.get(doc_id)
            has_name_match = False
            match_count = 0
            
            if meta:
                if query_tokens:
                    match_count = sum(1 for tok in query_tokens if tok in meta["token_set"])
                    if match_count:
                        boost += NAME_TOKEN_WEIGHT * match_count
                        has_name_match = True
                
                if query_name:
                    if meta["normalized"] == query_name:
                        boost += EXACT_NAME_BONUS
                        has_name_match = True
                    elif meta["normalized"].startswith(query_name):
                        boost += NAME_PREFIX_BONUS
                        has_name_match = True
                
                if raw_query_lower and raw_query_lower in meta["raw_lower"]:
                    boost += RAW_SUBSTRING_BONUS
                    has_name_match = True
            
            if not has_name_match and query_tokens:
                boost -= NON_NAME_MATCH_PENALTY
            
            if has_name_match:
                value = player_market_value.get(doc_id)
                if value and market_value_log_max > 0.0:
                    boost += MARKET_VALUE_WEIGHT * (math.log1p(value) / market_value_log_max)
                
                length = profile_length_by_id.get(doc_id)
                if length and profile_length_log_max > 0.0:
                    boost += PROFILE_LENGTH_WEIGHT * (math.log1p(length) / profile_length_log_max)
            
            scores[doc_id] += boost
    
    if not scores:
        elapsed = (time.perf_counter() - start_time) * 1000
        log(f"No documents matched these terms. (took {elapsed:.2f} ms)")
        return []
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        doc = doc_by_id[doc_id]
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "player_id": doc["player_id"],
            "player_name": doc["player_name"],
            "score": score,
            "market_value": player_market_value.get(doc_id),
        })
    
    elapsed = (time.perf_counter() - start_time) * 1000
    
    log("\n[top] Results:")
    for r in results:
        value = r["market_value"]
        length = profile_length_by_id.get(r["doc_id"])
        extras = []
        if value:
            extras.append(f"market_value~{value:,.0f} EUR")
        if length:
            extras.append(f"profile_chars={length}")
        extra_text = f" [{', '.join(extras)}]" if extras else ""
        log(f"{r['rank']:2d}. [{r['score']:.3f}] {r['player_name']} (player_id={r['player_id']}){extra_text}")
    
    log(f"\n[time] {elapsed:.2f} ms (barrel_load={barrel_load_time:.2f} ms)")
    log(f"[memory] {len(barrel_cache)} barrels cached, {len(required_barrels)} loaded for this query")
    
    if elapsed < 500:
        log("[perf]Under 500 ms goal")
    else:
        log("[perf]Above 500 ms goal")
    
    return results

# ---------- CLI ----------

if __name__ == "__main__":
    print("\n[ready] BARREL-OPTIMIZED search engine ready.")
    print(f"[info] System loads only required barrels per query (max {MAX_CACHED_BARRELS} cached)")
    print("[info] Type a query or press Enter to exit.\n")
    
    while True:
        q = input("Query> ").strip()
        if not q:
            break
        search(q, top_k=10)
    
    print("\n[exit] Exiting search engine.")

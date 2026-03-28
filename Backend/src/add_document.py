# add_document.py
# DYNAMIC DOCUMENT ADDITION - Incrementally add new players without full rebuild
import csv
import json
import math
import os
import re
import time
from collections import defaultdict

# ---------- PATHS ----------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INDEX_DIR = os.path.join(PROJECT_ROOT, 'data', 'index')
BARREL_DIR = os.path.join(INDEX_DIR, 'barrels')

LEXICON_PATH = os.path.join(INDEX_DIR, 'lexicon_complete.json')
FORWARD_INDEX_PATH = os.path.join(INDEX_DIR, 'forward_index_termid.json')
INVERTED_INDEX_PATH = os.path.join(INDEX_DIR, 'inverted_index_termid.json')
TERM_TO_BARREL_MAP_PATH = os.path.join(BARREL_DIR, 'term_to_barrel_map.json')

# ---------- TEXT NORMALIZATION (MUST MATCH BUILD PIPELINE) ----------

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

# ---------- LOAD EXISTING INDEXES ----------

def load_indexes():
    """Load all existing indexes into memory."""
    print("[load] Loading lexicon...")
    with open(LEXICON_PATH, 'r', encoding='utf-8') as f:
        lexicon = json.load(f)
    token_to_entry = {entry["token"]: entry for entry in lexicon}
    max_term_id = max(entry["term_id"] for entry in lexicon)
    print(f"[done] Loaded {len(lexicon):,} tokens (max_term_id={max_term_id})")
    
    print("[load] Loading forward index...")
    with open(FORWARD_INDEX_PATH, 'r', encoding='utf-8') as f:
        forward_index = json.load(f)
    doc_by_id = {doc["player_id"]: doc for doc in forward_index}
    print(f"[done] Loaded {len(forward_index):,} documents")
    
    print("[load] Loading term-to-barrel mapping...")
    with open(TERM_TO_BARREL_MAP_PATH, 'r', encoding='utf-8') as f:
        term_to_barrel = json.load(f)
    print(f"[done] Loaded {len(term_to_barrel):,} mappings")
    
    return {
        'lexicon': lexicon,
        'token_to_entry': token_to_entry,
        'max_term_id': max_term_id,
        'forward_index': forward_index,
        'doc_by_id': doc_by_id,
        'term_to_barrel': term_to_barrel
    }

# ---------- ADD NEW DOCUMENT ----------

def add_document(player_data: dict, indexes: dict):
    """
    Add a new player document to the search engine.
    
    player_data format:
    {
        "player_id": 999999,
        "player_name": "New Player",
        "detailed_content": "Long text with player bio, stats, etc...",
        # ... other metadata fields
    }
    
    Returns: dict with statistics about the update
    """
    start_time = time.perf_counter()
    
    player_id = player_data.get("player_id")
    player_name = player_data.get("player_name", "")
    detailed_content = player_data.get("detailed_content", "")
    
    if not player_id or not player_name:
        return {"error": "Missing required fields: player_id, player_name"}
    
    if player_id in indexes['doc_by_id']:
        return {"error": f"Player ID {player_id} already exists"}
    
    print(f"\n[add] Adding player: {player_name} (ID={player_id})")
    
    # 1. Tokenize content
    print("[step 1/5] Tokenizing content...")
    all_text = f"{player_name} {detailed_content}"
    tokens = normalize_and_tokenize(all_text)
    
    if not tokens:
        return {"error": "No valid tokens found in document"}
    
    # Count term frequencies
    term_freq = defaultdict(int)
    for token in tokens:
        term_freq[token] += 1
    
    total_terms = len(tokens)
    unique_terms = len(term_freq)
    print(f"   Found {total_terms} tokens, {unique_terms} unique")
    
    # 2. Update lexicon (assign term_ids to new tokens)
    print("[step 2/5] Updating lexicon...")
    new_tokens = []
    next_term_id = indexes['max_term_id'] + 1
    
    for token, tf in term_freq.items():
        if token not in indexes['token_to_entry']:
            # New token - add to lexicon
            new_entry = {
                "token": token,
                "df": 1,  # This document is the first
                "term_id": next_term_id
            }
            indexes['lexicon'].append(new_entry)
            indexes['token_to_entry'][token] = new_entry
            new_tokens.append(token)
            next_term_id += 1
        else:
            # Existing token - increment document frequency
            indexes['token_to_entry'][token]["df"] += 1
    
    indexes['max_term_id'] = next_term_id - 1
    print(f"   Added {len(new_tokens)} new tokens to lexicon")
    
    # 3. Update forward index
    print("[step 3/5] Updating forward index...")
    term_ids_in_doc = {}
    for token, tf in term_freq.items():
        entry = indexes['token_to_entry'][token]
        term_id = entry["term_id"]
        term_ids_in_doc[term_id] = {
            "token": token,
            "tf": tf
        }
    
    forward_entry = {
        "player_id": player_id,
        "player_name": player_name,
        "total_terms": total_terms,
        "unique_terms": unique_terms,
        "terms": term_ids_in_doc
    }
    indexes['forward_index'].append(forward_entry)
    indexes['doc_by_id'][player_id] = forward_entry
    print(f"   Added document to forward index")
    
    # 4. Update barrels (inverted index distributed)
    print("[step 4/5] Updating barrels...")
    barrels_updated = set()
    
    for token, tf in term_freq.items():
        entry = indexes['token_to_entry'][token]
        term_id = entry["term_id"]
        term_id_str = str(term_id)
        
        # Determine which barrel this term belongs to
        barrel_name = indexes['term_to_barrel'].get(term_id_str)
        
        if not barrel_name:
            # New term - assign to a barrel (use simple mod distribution)
            num_barrels = max(int(bn.split('_')[1]) for bn in set(indexes['term_to_barrel'].values())) + 1
            barrel_idx = term_id % num_barrels
            barrel_name = f"barrel_{barrel_idx:03d}"
            indexes['term_to_barrel'][term_id_str] = barrel_name
        
        # Load barrel, update, save back
        barrel_path = os.path.join(BARREL_DIR, f"{barrel_name}.json")
        
        if os.path.exists(barrel_path):
            with open(barrel_path, 'r', encoding='utf-8') as f:
                barrel_data = json.load(f)
        else:
            barrel_data = {
                'metadata': {
                    'term_count': 0,
                    'posting_count': 0,
                    'barrel_name': barrel_name
                },
                'inverted_index': {}
            }
        
        # Update postings for this term
        if term_id_str not in barrel_data['inverted_index']:
            barrel_data['inverted_index'][term_id_str] = {
                'token': token,
                'df': entry['df'],
                'postings': {}
            }
        
        # Add this document to postings
        barrel_data['inverted_index'][term_id_str]['postings'][str(player_id)] = {
            "tf": tf
        }
        
        # Update df in barrel
        barrel_data['inverted_index'][term_id_str]['df'] = entry['df']
        
        # Update metadata
        barrel_data['metadata']['term_count'] = len(barrel_data['inverted_index'])
        barrel_data['metadata']['posting_count'] = sum(
            len(term_data['postings']) 
            for term_data in barrel_data['inverted_index'].values()
        )
        
        # Save barrel
        with open(barrel_path, 'w', encoding='utf-8') as f:
            json.dump(barrel_data, f, ensure_ascii=False)
        
        barrels_updated.add(barrel_name)
    
    print(f"   Updated {len(barrels_updated)} barrels: {sorted(barrels_updated)}")
    
    # 5. Save updated indexes
    print("[step 5/5] Saving updated indexes...")
    
    # Save lexicon
    with open(LEXICON_PATH, 'w', encoding='utf-8') as f:
        json.dump(indexes['lexicon'], f, ensure_ascii=False)
    
    # Save forward index
    with open(FORWARD_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(indexes['forward_index'], f, ensure_ascii=False)
    
    # Save term-to-barrel mapping
    with open(TERM_TO_BARREL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(indexes['term_to_barrel'], f, ensure_ascii=False)
    
    print(f"   Saved all indexes")
    
    elapsed = time.perf_counter() - start_time
    
    stats = {
        "success": True,
        "player_id": player_id,
        "player_name": player_name,
        "total_terms": total_terms,
        "unique_terms": unique_terms,
        "new_tokens_added": len(new_tokens),
        "barrels_updated": len(barrels_updated),
        "time_seconds": elapsed,
        "meets_requirement": elapsed < 60  # Must be under 1 minute
    }
    
    print(f"\n[done] Document added in {elapsed:.2f} seconds")
    if stats["meets_requirement"]:
        print("[perf]Under 1 minute requirement")
    else:
        print("[perf]Exceeded 1 minute requirement")
    
    return stats

# ---------- CLI ----------

if __name__ == "__main__":
    print("=" * 60)
    print("DYNAMIC DOCUMENT ADDITION SYSTEM")
    print("=" * 60)
    
    # Load indexes
    indexes = load_indexes()
    
    print("\n[ready] System ready to add new documents.")
    print("[info] Enter player data in JSON format or type 'exit' to quit.\n")
    
    # Example usage
    print("Example player data format:")
    example = {
        "player_id": 999999,
        "player_name": "Test Player",
        "detailed_content": "This is a test player from Manchester United. He plays as a striker and has won multiple trophies."
    }
    print(json.dumps(example, indent=2))
    print("\n" + "-" * 60 + "\n")
    
    while True:
        print("Enter player data (JSON) or 'exit':")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        try:
            player_data = json.loads(user_input)
            result = add_document(player_data, indexes)
            print("\n[result]")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print(f"[error] Invalid JSON: {e}")
        except Exception as e:
            print(f"[error] {e}")
        
        print("\n" + "-" * 60 + "\n")
    
    print("\n[exit] Exiting document addition system.")

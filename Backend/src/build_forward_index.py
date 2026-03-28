import json
import re
from collections import defaultdict

print(" BUILDING FORWARD INDEX (TERM IDs)...")
print("=" * 50)

# ---------- Load documents ----------
print(" Loading search documents...")
with open("data/processed/complete_player_profiles.json", "r", encoding="utf-8") as f:
    search_documents = json.load(f)
print(f" Loaded {len(search_documents)} documents")

# ---------- Load lexicon (term_id mapping) ----------
print(" Loading lexicon (term IDs)...")
with open("data/index/lexicon_complete.json", "r", encoding="utf-8") as f:
    lexicon_entries = json.load(f)

token_to_id = {entry["token"]: entry["term_id"] for entry in lexicon_entries}
print(f" Loaded {len(token_to_id):,} tokens in lexicon")

print(" Building forward index with term IDs...")

forward_index = []  # list of doc objects

for doc_idx, doc in enumerate(search_documents):
    player_id = doc.get("player_id")          # this is your doc identifier
    player_name = doc.get("player_name", "")

    # Text source
    text = doc.get("detailed_content", "")
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()

    # Tokenize
    words = re.findall(r"\b[a-z]+\b", text)

    # Count term frequencies and positions using tokens first
    token_tf = defaultdict(int)
    token_positions = defaultdict(list)

    for position, word in enumerate(words):
        token_tf[word] += 1
        token_positions[word].append(position)

    term_entries = []
    total_terms = 0

    for token, tf in token_tf.items():
        term_id = token_to_id.get(token)
        if term_id is None:
            continue

        positions = token_positions[token]
        term_entries.append({
            "term_id": term_id,
            "tf": tf,
            "positions": positions[:10],  # first 10 positions
        })
        total_terms += tf

    doc_entry = {
        "player_id": player_id,          # acts as doc id
        "player_name": player_name,
        "terms": term_entries,
        "total_terms": total_terms,
        "unique_terms": len(term_entries),
    }

    forward_index.append(doc_entry)

    if (doc_idx + 1) % 10000 == 0:
        print(f" Processed {doc_idx + 1} documents...")

print("\n FORWARD INDEX STATISTICS (TERM IDs):")
print("=" * 40)
doc_count = len(forward_index)
total_terms_all = sum(d["total_terms"] for d in forward_index)
total_unique_terms_all = sum(d["unique_terms"] for d in forward_index)
avg_terms_per_doc = total_terms_all // doc_count if doc_count else 0
avg_unique_terms_per_doc = total_unique_terms_all // doc_count if doc_count else 0

print(f"  Documents indexed: {doc_count:,}")
print(f"  Total terms: {total_terms_all:,}")
print(f"  Total unique terms (per-doc sum): {total_unique_terms_all:,}")
print(f"  Avg terms per document: {avg_terms_per_doc}")
print(f"  Avg unique terms per document: {avg_unique_terms_per_doc}")

# ---------- Save forward index (compact but readable) ----------
print("\n Saving forward index (TERM IDs)...")
with open("data/index/forward_index_termid.json", "w", encoding="utf-8") as f:
    json.dump(forward_index, f, ensure_ascii=False, indent=1, separators=(",", ":"))

print(" FORWARD INDEX WITH TERM IDs BUILT!")
print(" Saved: data/index/forward_index_termid.json")

# ---------- Sample output ----------
print("\n SAMPLE FORWARD INDEX ENTRIES (TERM IDs):")
print("=" * 50)

for doc in forward_index[:3]:
    print(f"\n Player / Doc: {doc['player_id']}")
    print(f"  Player name: {doc['player_name']}")
    print(f"  Total terms: {doc['total_terms']}")
    print(f"  Unique terms: {doc['unique_terms']}")
    sample_terms = doc["terms"][:5]
    print("  Sample terms (term_id:tf@positions):")
    for t in sample_terms:
        print(f"    {t['term_id']}:{t['tf']}@{t['positions']}")

print("\n FORWARD INDEX READY FOR INVERTED INDEX BUILDING!")

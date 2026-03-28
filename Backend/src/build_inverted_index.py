# build_inverted_index_termid.py
import json
from collections import defaultdict

print(" BUILDING MINIMAL INVERTED INDEX (TERM IDs)...")
print("=" * 50)

# Load forward index with term_ids
print(" Loading forward index (TERM IDs)...")
with open("data/index/forward_index_termid.json", "r", encoding="utf-8") as f:
    forward_index = json.load(f)
print(f" Loaded {len(forward_index):,} documents from forward index")

# Optional: load lexicon for term_id -> token mapping (only for debugging / printing)
print(" Loading lexicon for term_id -> token mapping...")
with open("data/index/lexicon_complete.json", "r", encoding="utf-8") as f:
    lexicon_entries = json.load(f)

termid_to_token = {entry["term_id"]: entry["token"] for entry in lexicon_entries}
print(f" Loaded {len(termid_to_token):,} term IDs")

print(" Building inverted index (TERM IDs)...")

# Inverted index: term_id -> {doc_id: {tf, positions}}
inverted_index = defaultdict(lambda: defaultdict(dict))

# DF: how many docs contain each term_id
term_document_frequency = defaultdict(int)

for doc_data in forward_index:
    doc_id = doc_data["player_id"]          # this is your document identifier
    terms = doc_data["terms"]               # list of {term_id, tf, positions}

    for term_entry in terms:
        term_id = term_entry["term_id"]
        tf = term_entry["tf"]
        positions = term_entry["positions"]

        inverted_index[term_id][doc_id] = {
            "tf": tf,
            "positions": positions,
        }
        term_document_frequency[term_id] += 1

print(" Saving minimal inverted index (TERM IDs)...")

# Convert nested defaultdicts to plain dicts and serialize
inverted_index_dict = {int(tid): dict(docs) for tid, docs in inverted_index.items()}
output = {
    "term_document_frequency": {int(tid): df for tid, df in term_document_frequency.items()},
    "inverted_index": inverted_index_dict,
}

with open("data/index/inverted_index_termid.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=1, separators=(",", ":"))

print(" MINIMAL INVERTED INDEX (TERM IDs) BUILT!")
print(" Saved: data/index/inverted_index_termid.json")

# Optional: small debug sample
print("\n SAMPLE TERMS:")
for term_id in list(inverted_index_dict.keys())[:5]:
    token = termid_to_token.get(term_id, f"<term_{term_id}>")
    docs = inverted_index_dict[term_id]
    print(f"  term_id={term_id}, token='{token}', docs={len(docs)}")

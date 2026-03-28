import json
import math
import os
from collections import defaultdict

print("BUILDING BARRELS SYSTEM...")
print("=" * 50)

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INDEX_DIR = os.path.join(PROJECT_ROOT, 'data', 'index')
BARREL_DIR = os.path.join(INDEX_DIR, 'barrels')

# Create barrels directory
os.makedirs(BARREL_DIR, exist_ok=True)

# Load lexicon
print("Loading lexicon...")
lexicon_path = os.path.join(INDEX_DIR, 'lexicon_complete.json')
with open(lexicon_path, 'r', encoding='utf-8') as f:
    lexicon = json.load(f)

print(f"Loaded {len(lexicon):,} tokens from lexicon")

# Load inverted index (term_id based)
print("Loading inverted index (this may take a moment)...")
inverted_index_path = os.path.join(INDEX_DIR, 'inverted_index_termid.json')
with open(inverted_index_path, 'r', encoding='utf-8') as f:
    inv_data = json.load(f)

# Extract the actual inverted index from the nested structure
inverted_index = inv_data.get('inverted_index', {})
term_document_frequency = inv_data.get('term_document_frequency', {})

print(f"Loaded inverted index with {len(inverted_index):,} entries")

# Calculate optimal number of barrels
total_terms = len(lexicon)
target_terms_per_barrel = 4000
num_barrels = math.ceil(total_terms / target_terms_per_barrel)

print(f"Barrel Configuration:")
print(f"Total terms: {total_terms:,}")
print(f"Target terms per barrel: {target_terms_per_barrel:,}")
print(f"Number of barrels: {num_barrels}")

# Create barrels
barrels = defaultdict(dict)
barrel_stats = defaultdict(lambda: {'term_count': 0, 'posting_count': 0})
term_to_barrel = {}  # Mapping for quick lookup

print(f"\nDistributing terms into {num_barrels} barrels...")

# Strategy: Distribute by term_id (already sorted and efficient)
sorted_lexicon = sorted(lexicon, key=lambda x: x['term_id'])
terms_per_barrel = math.ceil(len(sorted_lexicon) / num_barrels)

for barrel_idx in range(num_barrels):
    start_idx = barrel_idx * terms_per_barrel
    end_idx = min(start_idx + terms_per_barrel, len(sorted_lexicon))
    barrel_terms = sorted_lexicon[start_idx:end_idx]
    
    barrel_name = f"barrel_{barrel_idx:03d}"
    
    for term_entry in barrel_terms:
        term_id = str(term_entry['term_id'])
        token = term_entry['token']
        
        # Store term mapping
        term_to_barrel[term_id] = barrel_name
        
        # Add to barrel if exists in inverted index
        if term_id in inverted_index:
            barrels[barrel_name][term_id] = {
                'token': token,
                'df': term_entry['df'],
                'postings': inverted_index[term_id]
            }
            barrel_stats[barrel_name]['term_count'] += 1
            barrel_stats[barrel_name]['posting_count'] += len(inverted_index[term_id])
    
    print(f"{barrel_name}: {len(barrel_terms):>4} terms, {barrel_stats[barrel_name]['posting_count']:>6,} postings")

print(f"\nBARREL SYSTEM STATISTICS:")
print("=" * 40)

total_barrel_terms = sum(stats['term_count'] for stats in barrel_stats.values())
total_barrel_postings = sum(stats['posting_count'] for stats in barrel_stats.values())

print(f"   Total barrels: {len(barrels)}")
print(f"   Total terms in barrels: {total_barrel_terms:,}")
print(f"   Total postings in barrels: {total_barrel_postings:,}")
print(f"   Avg terms per barrel: {total_barrel_terms // len(barrels) if len(barrels) > 0 else 0:,}")
print(f"   Avg postings per barrel: {total_barrel_postings // len(barrels) if len(barrels) > 0 else 0:,}")

# Save barrels
print(f"\nSaving barrels to {BARREL_DIR}...")
for barrel_name, barrel_data in barrels.items():
    filename = os.path.join(BARREL_DIR, f"{barrel_name}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'term_count': barrel_stats[barrel_name]['term_count'],
                'posting_count': barrel_stats[barrel_name]['posting_count'],
                'barrel_name': barrel_name
            },
            'inverted_index': barrel_data
        }, f, ensure_ascii=False, indent=2)

# Save term-to-barrel mapping for quick lookup
mapping_path = os.path.join(BARREL_DIR, 'term_to_barrel_map.json')
print(f"Saving term-to-barrel mapping...")
with open(mapping_path, 'w', encoding='utf-8') as f:
    json.dump(term_to_barrel, f, ensure_ascii=False, indent=2)

print(f"BARRELS SYSTEM BUILT!")
print(f"Saved {len(barrels)} barrel files to {BARREL_DIR}")
print(f"Saved term-to-barrel mapping")


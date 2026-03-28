import pandas as pd
import json

print("🔧 FIXING INCORRECT MAPPINGS...")

# Load the enhanced mapping
with open('player_mapping_enhanced.json', 'r') as f:
    mapping = json.load(f)

# Load datasets for verification
fifa_df = pd.read_csv("players_22.csv", low_memory=False)
transfermarkt_docs = []
with open('search_engine_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        transfermarkt_docs.append(json.loads(line))

# Create a TM ID to player lookup
tm_lookup = {doc['player_id']: doc for doc in transfermarkt_docs}

# Manual corrections for top players
manual_corrections = {
    # FIFA_ID: Correct_TM_ID
    "158023": "283",      # L. Messi -> Lionel Messi
    "192985": "88755",    # K. De Bruyne -> Kevin De Bruyne  
    "167495": "7442",     # M. Neuer -> Manuel Neuer
    "200389": "96341",    # M. ter Stegen -> Marc-André ter Stegen
    "202126": "132098",   # H. Kane -> Harry Kane
}

print("Applying manual corrections...")
for fifa_id, correct_tm_id in manual_corrections.items():
    fifa_id_str = str(fifa_id)
    if fifa_id_str in mapping:
        old_tm_id = mapping[fifa_id_str]
        old_player = tm_lookup.get(old_tm_id, {})
        new_player = tm_lookup.get(correct_tm_id, {})
        
        print(f"🔧 Fixed: FIFA {fifa_df[fifa_df['sofifa_id'] == int(fifa_id)].iloc[0]['short_name']}")
        print(f"   FROM: {old_player.get('player_name', 'Unknown')}")
        print(f"   TO: {new_player.get('player_name', 'Unknown')}")
        
        mapping[fifa_id_str] = correct_tm_id

# Save the corrected mapping
with open('player_mapping_corrected.json', 'w') as f:
    json.dump(mapping, f, indent=2)

print(f"\n✅ CORRECTED MAPPING SAVED!")
print(f"Total mapped players: {len(mapping)}")
print(f"Mapping rate: {(len(mapping)/len(fifa_df))*100:.1f}%")

# Verify corrections
print(f"\n👀 VERIFIED MAPPINGS:")
top_players_to_check = ["158023", "192985", "167495", "188545", "20801"]
for fifa_id in top_players_to_check:
    if fifa_id in mapping:
        fifa_player = fifa_df[fifa_df['sofifa_id'] == int(fifa_id)].iloc[0]
        tm_player = tm_lookup.get(mapping[fifa_id], {})
        print(f"✅ {fifa_player['short_name']} -> {tm_player.get('player_name', 'Unknown')}")
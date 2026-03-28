import pandas as pd
import os

# Configuration
TARGET_PLAYERS = 10000
RAW_DATA_DIR = os.path.join('..', 'data', 'raw')
OUTPUT_DIR = os.path.join('..', 'data', 'raw_reduced')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Reducing dataset to {TARGET_PLAYERS} players...")

# Step 1: Load player_profiles and select 10,000 players
print("\nStep 1: Loading player_profiles.csv...")
player_profiles_path = os.path.join(RAW_DATA_DIR, 'player_profiles', 'player_profiles.csv')
df_profiles = pd.read_csv(player_profiles_path)
print(f"Original player count: {len(df_profiles)}")

# Select first 10,000 players (you can change this to random sampling if needed)
df_profiles_reduced = df_profiles.head(TARGET_PLAYERS)
selected_player_ids = set(df_profiles_reduced['player_id'].values)
print(f"Selected {len(selected_player_ids)} unique player IDs")

# Save reduced player_profiles
output_profiles_dir = os.path.join(OUTPUT_DIR, 'player_profiles')
os.makedirs(output_profiles_dir, exist_ok=True)
output_path = os.path.join(output_profiles_dir, 'player_profiles.csv')
df_profiles_reduced.to_csv(output_path, index=False)
print(f"Saved: {output_path}")

# Step 2: Filter all other CSV files based on selected player IDs
csv_files = [
    ('player_injuries', 'player_injuries.csv', 'player_id'),
    ('player_latest_market_value', 'player_latest_market_value.csv', 'player_id'),
    ('player_market_value', 'player_market_value.csv', 'player_id'),
    ('player_national_performances', 'player_national_performances.csv', 'player_id'),
    ('player_performances', 'player_performances.csv', 'player_id'),
    ('player_teammates_played_with', 'player_teammates_played_with.csv', 'player_id'),
    ('transfer_history', 'transfer_history.csv', 'player_id'),
]

print("\nStep 2: Filtering related player data files...")
for folder, filename, player_col in csv_files:
    input_path = os.path.join(RAW_DATA_DIR, folder, filename)
    
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found, skipping...")
        continue
    
    print(f"\nProcessing {filename}...")
    df = pd.read_csv(input_path)
    print(f"  Original rows: {len(df)}")
    
    # Filter by selected player IDs
    df_filtered = df[df[player_col].isin(selected_player_ids)]
    print(f"  Filtered rows: {len(df_filtered)}")
    
    # Save filtered data
    output_folder = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    df_filtered.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

# Step 3: Copy team-related CSV files (not filtered by players)
print("\nStep 3: Copying team-related files...")
team_files = [
    ('team_children', 'team_children.csv'),
    ('team_competitions_seasons', 'team_competitions_seasons.csv'),
    ('team_details', 'team_details.csv'),
]

for folder, filename in team_files:
    input_path = os.path.join(RAW_DATA_DIR, folder, filename)
    
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found, skipping...")
        continue
    
    print(f"\nCopying {filename}...")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}")
    
    # Save to output directory
    output_folder = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

print("\n" + "="*60)
print(f"Dataset reduction complete!")
print(f"Reduced data saved to: {OUTPUT_DIR}")
print(f"Player count reduced from ~93,000 to {TARGET_PLAYERS}")
print("="*60)

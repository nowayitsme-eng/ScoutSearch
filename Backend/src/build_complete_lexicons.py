import json
import re
from collections import defaultdict

print("BUILDING COMPLETE LEXICON (ONE FILE WITH DF + TERM ID)...")
print("=" * 50)

# Load search documents (array JSON)
print("Loading search documents...")
with open("data/processed/complete_player_profiles.json", "r", encoding="utf-8") as f:
    search_documents = json.load(f)
print(f"Loaded {len(search_documents)} documents")

# Comprehensive stop words list
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
    """Basic stemming for common suffixes"""
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        return word[:-2]
    elif word.endswith("es") and len(word) > 4:
        return word[:-2]
    elif word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word

print("Building ONE COMPLETE lexicon with DF and term IDs...")

# token -> document frequency
lexicon_df = defaultdict(int)

for doc_idx, doc in enumerate(search_documents):
    # Use detailed_content as text
    text = doc.get("detailed_content", "")
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()

    # Tokens appearing in this document
    doc_tokens = set()

    # From detailed_content
    words = re.findall(r"\b[a-z]+\b", text)
    for word in words:
        if len(word) <= 2:
            continue
        stemmed = simple_stemmer(word)
        if stemmed in COMPREHENSIVE_STOP_WORDS:
            continue
        doc_tokens.add(stemmed)

    # From metadata.current_club
    metadata = doc.get("metadata", {}) or {}
    current_club = metadata.get("current_club")
    if isinstance(current_club, str):
        club_text = current_club.lower()
        club_words = re.findall(r"\b[a-z]+\b", club_text)
        for word in club_words:
            if len(word) > 2:
                stemmed = simple_stemmer(word)
                if stemmed not in COMPREHENSIVE_STOP_WORDS:
                    doc_tokens.add(stemmed)

    # Update DF counts (once per doc)
    for token in doc_tokens:
        lexicon_df[token] += 1

    if (doc_idx + 1) % 10000 == 0:
        print(f"Processed {doc_idx + 1} documents...")

print("\nLEXICON STATISTICS:")
print("=" * 40)
print(f"Total unique tokens: {len(lexicon_df):,}")
print(f"Total DF sum: {sum(lexicon_df.values()):,}")

# Build list of entries and sort by DF desc
entries = [
    {"token": token, "df": df}
    for token, df in lexicon_df.items()
]
entries.sort(key=lambda x: x["df"], reverse=True)

# Assign term_id to each token (0-based; use +1 for 1-based if you prefer)
for idx, entry in enumerate(entries):
    entry["term_id"] = idx

# Save ONLY the entries list; perfect for forward + inverted index building
with open("data/index/lexicon_complete.json", "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print("\nONE COMPLETE LEXICON WITH DF + TERM IDs BUILT!")
print("Saved: data/index/lexicon_complete.json")
print(f"{len(entries):,} tokens with term IDs ready for indexing!")

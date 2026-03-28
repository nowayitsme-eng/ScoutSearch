# ScoutSearch: High-Performance Football Scouting Engine

ScoutSearch is a bespoke, extremely low-latency football (soccer) player scouting search engine. Built entirely from scratch without relying on heavy external enterprise search frameworks (like Elasticsearch, Solr, or Lucene), it provides millisecond-latency natural language querying over highly multidimensional player datasets.

This project was built to address the severe limitations of standard Relational Databases (SQL) when handling nuanced lexical search queries, and the unnecessary bloat of JVM-based enterprise indexers when deployed in isolated web environments.

## 🧠 Architectural Overview & Data Structures

At its core, ScoutSearch is powered by a set of custom-built disk-backed and in-memory data structures designed for maximum retrieval efficiency.

### 1. The Sharded Inverted Index (Barrels)
Instead of loading a monolithic indexing file into system RAM, the backend implements **Index Sharding**. 
- **The Lexicon (`lexicon_complete.json`):** A global vocabulary HashMap tracking document frequencies (DFs) that strictly maps any given token to a specific `Barrel ID`.
- **Barrels (`barrel_000` to `barrel_025`):** The inverted index posting lists are chunked into multiple localized JSON blocks on disk. 
- **Query Execution:** When a query processes, the engine maps the tokens to barrel IDs via the Lexicon, loads *only* the required barrels into memory (`barrel_manager.py`), calculates intersections, and garbage-collects immediately. This approach cuts RAM footprint by 80% during execution.

### 2. Forward Index (`O(1)` Lookups)
While the Inverted Index handles boolean/weighted retrieval, the **Forward Index** (`forward_index_termid.json`) acts as an `O(1)` Hash Map. When the result set of `Player_IDs` is resolved, the system maps those IDs directly to full document payloads for rendering the frontend without touching a SQL database.

### 3. Autocomplete with Trie Data Structures
To provide real-time, sub-50ms suggestions as the user types, `autocomplete.py` implements a custom **Prefix Trie** (`AutocompleteTrie`).
- Each node (`TrieNode`) maps character edges and stores traversal frequencies extracted from historic query logs and the Lexicon's document frequencies.
- Instead of running a heavy `LIKE '%text%'` SQL query, the API searches the Trie by traversing the prefix string `O(M)` (where M is prefix length) and firing a Depth-First Search (DFS) to collect the top-5 weighted predictive terms based on highest frequency thresholds.

### 4. Smart LRU Query Caching
To minimize disk I/O from continuous barrel loading, `optimized_search.py` utilizes a custom **Least Recently Used (LRU) Cache**.
- Semantically identical queries or highly frequent queries bypass the lexer and barrel mappings entirely, returning a pre-computed set of resulting IDs instantly.
- The cache acts dynamically, evicting stale query hashes when memory limits are reached.

---

## 🔍 Semantic Search & Ranking Engine

Implemented via `semantic_search.py` and `text_processor.py`, the engine is mathematically driven to interpret intent.

* **Lexical Stemming & Normalization:** Raw string queries are strictly stripped of Windows-1252 encoding artifacts to pure UTF-8. Suffixes are algorithmically stemmed (e.g., matching "dribbler" safely to "dribbl").
* **TF-IDF & Cosine Weighting Context:** Search results are explicitly weighted using **Term Frequency-Inverse Document Frequency (TF-IDF)** algorithms. Common words ("player", "the") carry negligible weight, while specific high-value identifiers (e.g., "Marseille", "Regista") exponentially multiply a document's retrieval score. 

---

## 💾 Multi-Dataset Entity Resolution

ScoutSearch synthesizes a truly massive dataset matrix, bringing together two globally disparate schemas:

1. **FIFA 22 Core Database (19,000+ Profiles):** Provides deep technical ratings, skill moves, work rates, positioning maps, and physical characteristics (extracted from `players_22.csv`).
2. **Transfermarkt Multi-Tables:** Supplies exhaustive real-world data across deep relational CSVs (`player_profiles.csv`, `player_performances.csv`, `player_injuries.csv`, `transfer_history.csv`, `player_market_value.csv`, etc.).

**The Entity Resolution Problem (`idfcmnpl.py`):**
Because these datasets utilize completely different Primary Keys (`sofifa_id` vs Transfermarkt's custom ID) and inherently different string naming conventions ("L. Messi" vs "Lionel Andres Messi"), our resolution script utilizes intelligent string distance mapping, manual top-player overrides, and integer coercion to build `player_mapping_enhanced.json`. This links the technical capabilities of a player with their real-world transfer economy seamlessly.

---

## ⚡ Zero-Dependency Vanilla Frontend

The user interface (`index.html` + `static/`) explicitly rejects heavy Virtual DOM frameworks (React, Angular). 
- Rendering is executed directly via hardware-accelerated CSS and raw Vanilla JavaScript.
- **Explicit Type-Casting:** JavaScript inherently coerces numerical types, which initially corrupted dataset fetches for detailed profiles (requesting `/api/player/158023.0` instead of integer IDs). Strict `parseInt()` encapsulations protect the relative `/api/...` fetch boundaries, entirely avoiding CORS domain mismatches.

---

## 🚀 Installation & Build Pipeline

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nowayitsme-eng/ScoutSearch.git
   cd ScoutSearch
   ```

2. **Set up Virtual Environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Notice on Build Datasets:**
   Raw datasets (CSVs/JSONL) and the generated Inverted Barrels are deliberately ignored from version control due to their gigantic sizes. An administrator must place the core `.csv` files into `Backend/data/raw/` and sequentially run the indexing pipeline (`build_forward_index.py`, `build_inverted_index.py`) to generate the local Barrels and Lexicons.

4. **Start the Production Engine:**
   ```bash
   python Backend/src/app.py
   ```
   *Navigate to `http://127.0.0.1:8000` via a web browser.*

---
*Developed by @nowayitsme-eng*

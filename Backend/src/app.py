from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import sys
import requests
from flask import Response
import time
import re
from functools import lru_cache

# Add src directory to path for imports (needed for Azure deployment)
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import new modules
from barrel_manager import BarrelManager
from autocomplete import initialize_autocomplete, SmartAutocomplete
from semantic_search import initialize_semantic_search
from performance_monitor import performance_monitor, track_query
from dynamic_indexer import DynamicIndexer
from optimized_search import OptimizedSearchEngine

# Get base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(base_dir)
static_dir = os.path.join(project_root, 'static')

# EXTREMELY IMPORTANT: Extract payload sequentially before ANYTHING is instantiated globally
_zip_path = os.path.join(base_dir, 'data', 'scoutsearch_data.zip')
_data_dir = os.path.join(base_dir, 'data')
if os.path.exists(_zip_path) and not os.path.exists(os.path.join(_data_dir, 'raw', 'players_22.csv')):
    print(f"[STARTUP] Unzipping payload {_zip_path} as early step...")
    try:
        import zipfile
        with zipfile.ZipFile(_zip_path, 'r') as zipf:
            zipf.extractall(_data_dir)
        print("[STARTUP] Raw Dataset & Indexes extracted successfully before engine instantiation.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Zip extraction failed: {e}")

app = Flask(__name__, static_folder=static_dir, static_url_path='/static')
CORS(app)  # Enable CORS for all routes

# Initialize dynamic_indexer at module level (will be set in init_advanced_components)
dynamic_indexer = None


def sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to native Python types for JSON serialization."""
    # Import here to avoid circular issues in some environments
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    # numpy types
    try:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return [sanitize_for_json(v) for v in obj.tolist()]
    except Exception:
        pass
    return obj

# Add CSP headers to all responses
@app.after_request
def set_csp_headers(response):
    # Allow images from SoFIFA CDN and our own server
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "connect-src 'self' http://localhost:5000 http://127.0.0.1:5000; "
        "img-src 'self' data: blob: https://cdn.sofifa.net https://via.placeholder.com; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com;"
    )
    return response

class TextSearchEngine:
    def __init__(self, dataset_path=None, mapping_path=None, 
                 inverted_index_path=None, lexicon_path=None):
        # Build absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path = dataset_path or os.path.join(base_dir, 'data', 'raw', 'search_engine_dataset.jsonl')
        self.mapping_path = mapping_path or os.path.join(base_dir, 'data', 'raw', 'player_mapping_enhanced.json')
        self.inverted_index_path = inverted_index_path or os.path.join(base_dir, 'data', 'index', 'inverted_index.json')
        self.lexicon_path = lexicon_path or os.path.join(base_dir, 'data', 'index', 'lexicon_complete.json')
        
        # Initialize barrel manager for scalable index access
        barrel_dir = os.path.join(base_dir, 'data', 'index', 'barrels')
        self.barrel_manager = BarrelManager(barrel_dir, self.lexicon_path)
        
        # Initialize semantic search
        from semantic_search import semantic_engine
        self.semantic_engine = semantic_engine
        
        self.documents = {}
        self.player_mapping = {}
        self.inverted_index = None  # Lazy load (fallback)
        self.word_doc_freq = None   # Lazy load
        self.total_docs = 0
        self.index_loaded = False
        self.load_data()
    
    def load_data(self):
        """Load Transfermarkt dataset and player mapping"""
        try:
            # Load player mapping
            with open(self.mapping_path, 'r') as f:
                self.player_mapping = json.load(f)
            print(f"[OK] Player mapping loaded: {len(self.player_mapping)} mappings")
            
            # Load Transfermarkt documents
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    pid = str(doc.get('player_id', ''))
                    # Store only what is needed or minimal representation
                    self.documents[pid] = doc
            self.total_docs = len(self.documents)
            print(f"[OK] Text documents loaded: {self.total_docs} documents")
            
        except Exception as e:
            print(f"[ERROR] Error loading text search data: {e}")
    
    def ensure_index_loaded(self):
        """Lazy load inverted index on first search"""
        if not self.index_loaded:
            try:
                print(" Loading inverted index...")
                with open(self.inverted_index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    self.inverted_index = index_data.get('inverted_index', {})
                    self.word_doc_freq = index_data.get('word_document_frequency', {})
                self.index_loaded = True
                print(f"[OK] Inverted index loaded: {len(self.inverted_index)} terms")
            except Exception as e:
                print(f"[WARNING] Could not load inverted index: {e}")
                self.inverted_index = {}
                self.word_doc_freq = {}
                self.index_loaded = True
    
    def tokenize(self, text):
        """Tokenize and normalize text"""
        import re
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        # Keep hyphens in words like "left-back"
        tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
        return tokens
    
    def calculate_tf_idf(self, term, player_id):
        """Calculate TF-IDF score for a term in a document"""
        # TF: term frequency in document
        postings = self.inverted_index.get(term, {})
        player_key = f"player_{player_id}"
        
        # Get term frequency
        if player_key in postings:
            posting_data = postings[player_key]
            if isinstance(posting_data, dict):
                tf = posting_data.get('frequency', 0)
            else:
                tf = posting_data
        else:
            tf = 0
        
        if tf == 0:
            return 0
        
        # IDF: inverse document frequency
        df = self.word_doc_freq.get(term, 0)
        if df == 0:
            return 0
        
        import math
        idf = math.log(self.total_docs / df)
        
        return tf * idf
    
    def search_text(self, query, limit=50):
        """Advanced text search with TF-IDF ranking + barrel system + semantic expansion"""
        if not self.documents or not query.strip():
            return []
        
        try:
            # Expand query with semantic synonyms
            expanded_terms = [query.lower()]
            if self.semantic_engine:
                expanded_terms = self.semantic_engine.expand_query(query, max_expansions=2)
            
            # Tokenize all expanded queries
            all_query_terms = []
            for term in expanded_terms:
                all_query_terms.extend(self.tokenize(term))
            
            # Remove duplicates while preserving order
            query_terms = list(dict.fromkeys(all_query_terms))
            
            if not query_terms:
                return []
            
            # Score documents using barrel manager (memory efficient!)
            doc_scores = {}
            
            # Use barrel manager to get postings (only loads needed barrels)
            for term in query_terms:
                postings = self.barrel_manager.get_postings(term)
                
                for player_key, posting_data in postings.items():
                    try:
                        # Extract player_id
                        if isinstance(posting_data, dict):
                            player_id = posting_data.get('player_id')
                            tf = posting_data.get('frequency', 1)
                        else:
                            # Parse from key
                            if isinstance(player_key, str) and player_key.startswith('player_'):
                                player_id = int(player_key.replace('player_', ''))
                            else:
                                player_id = int(player_key)
                            tf = posting_data
                        
                        if player_id not in doc_scores:
                            doc_scores[player_id] = 0
                        
                        # Get document frequency from barrel manager
                        df = self.barrel_manager.get_term_df(term)
                        if df > 0:
                            import math
                            idf = math.log(self.total_docs / df)
                            doc_scores[player_id] += tf * idf
                        else:
                            doc_scores[player_id] += tf
                            
                    except (ValueError, TypeError, AttributeError) as e:
                        continue
            
            # Phrase matching bonus
            original_query_terms = self.tokenize(query.lower())
            if len(original_query_terms) > 1:
                query_lower = query.lower()
                for player_id in list(doc_scores.keys()):
                    doc = self.documents.get(str(player_id))
                    if doc:
                        doc_text = doc.get('text_content', '').lower()
                        if query_lower in doc_text:
                            doc_scores[player_id] *= 2.5
                        elif all(term in doc_text for term in original_query_terms):
                            doc_scores[player_id] *= 1.5

            # Sort and get results
            ranked_player_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

            results = []
            for player_id in ranked_player_ids[:limit]:
                doc = self.documents.get(str(player_id))
                if doc:
                    results.append(doc)

            return results

        except Exception as e:
            print(f"[WARNING] Error in text search: {e}")
            import traceback
            traceback.print_exc()
            return self.simple_search(query, limit)

    def simple_search(self, query, limit=50):
        """Fallback simple substring search"""
        query_lower = query.lower()
        results = []

        for doc in self.documents.values():
            text_content = doc.get('text_content', '').lower()
            if query_lower in text_content:
                results.append(doc)
                if len(results) >= limit:
                    break
        
        return results

class ScoutSearchEngine:
    def __init__(self, data_path=None, detailed_profiles_path=None):
        # Build absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_path or os.path.join(base_dir, 'data', 'raw', 'players_22.csv')
        self.detailed_profiles_path = detailed_profiles_path or os.path.join(base_dir, 'data', 'processed', 'complete_player_profiles.json')
        self.df = None
        self.text_search_engine = TextSearchEngine()
        self.detailed_profiles = {}
        
        # Performance optimization: LRU cache for search results
        self._search_cache = {}
        self._cache_max_size = 100
        
        # Pre-computed data for faster searches
        self._normalized_names = None
        self._position_masks = {}
        
        self.load_data()
        self.load_detailed_profiles()
        self._precompute_search_data()
    
    def load_data(self):
        """Load the FIFA 22 dataset"""
        # Only load the columns we actually need to save massive amounts of RAM on Render
        needed_cols = [
            'sofifa_id', 'short_name', 'long_name', 'player_positions', 
            'overall', 'potential', 'value_eur', 'wage_eur', 'age', 
            'height_cm', 'club_name', 'nationality_name', 'preferred_foot', 
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 
            'player_face_url', 'club_logo_url', 'nation_flag_url', 'work_rate',
            'skill_moves', 'weak_foot'
        ]
        try:
            # Check which columns actually exist to avoid KeyError
            import csv
            with open(self.data_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = next(csv.reader(f))
            actual_cols = [c for c in needed_cols if c in header]
            
            self.df = pd.read_csv(self.data_path, usecols=actual_cols, encoding='utf-8', low_memory=False)
            print(f"[OK] Dataset loaded: {len(self.df)} players, {len(self.df.columns)} columns")
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.data_path, usecols=actual_cols, encoding='latin-1', low_memory=False)
            print(f"[OK] Dataset loaded with latin-1: {len(self.df)} players")
        except FileNotFoundError:
            print(f"[ERROR] File {self.data_path} not found!")
            return
        
        # Clean the data
        self.clean_data()
    
    def _precompute_search_data(self):
        """Pre-compute normalized names and position masks for faster searching"""
        if self.df is None or self.df.empty:
            return
        
        try:
            from text_processor import get_text_processor
            tp = get_text_processor()
            
            # Pre-compute normalized names (huge speedup for text search)
            self.df['_norm_long_name'] = self.df['long_name'].apply(lambda x: tp.normalize_text(str(x)) if pd.notna(x) else '')
            self.df['_norm_short_name'] = self.df['short_name'].apply(lambda x: tp.normalize_text(str(x)) if pd.notna(x) else '')
            self.df['_norm_first_name'] = self.df['long_name'].apply(lambda x: tp.normalize_text(str(x).split()[0]) if pd.notna(x) and str(x).strip() else '')
            self.df['_norm_last_name'] = self.df['long_name'].apply(lambda x: tp.normalize_text(str(x).split()[-1]) if pd.notna(x) and str(x).strip() else '')
            
            # Pre-compute lowercase club and nationality for faster matching
            self.df['_club_lower'] = self.df['club_name'].str.lower().fillna('')
            self.df['_nationality_lower'] = self.df['nationality_name'].str.lower().fillna('')
            
            print(f"[OK] Pre-computed search data for {len(self.df)} players")
        except Exception as e:
            print(f"[WARNING] Could not pre-compute search data: {e}")
    
    def clean_data(self):
        """Clean and prepare the data"""
        # Ensure numeric columns are properly formatted
        numeric_columns = ['overall', 'potential', 'age', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'value_eur', 'skill_moves', 'weak_foot']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Fill missing image URLs with empty string
        image_columns = ['player_face_url', 'nation_flag_url', 'club_logo_url']
        for col in image_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')
        
        # Fill missing text columns
        if 'preferred_foot' in self.df.columns:
            self.df['preferred_foot'] = self.df['preferred_foot'].fillna('Right')
            
        # Parse work_rate into attacking and defensive
        if 'work_rate' in self.df.columns and 'attacking_work_rate' not in self.df.columns:
            try:
                rates = self.df['work_rate'].str.split('/', expand=True)
                if len(rates.columns) == 2:
                    self.df['attacking_work_rate'] = rates[0].str.strip()
                    self.df['defensive_work_rate'] = rates[1].str.strip()
                else:
                    self.df['attacking_work_rate'] = 'Medium'
                    self.df['defensive_work_rate'] = 'Medium'
            except:
                self.df['attacking_work_rate'] = 'Medium'
                self.df['defensive_work_rate'] = 'Medium'
    
    def search_players(self, filters, sort_by='overall', ascending=False, limit=50):
        """Main search function for attribute-based search"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        results = self.df.copy()
        results = self.apply_filters(results, filters)
        
        # If we have very few results, relax the filters slightly
        if len(results) < 5:
            print(f"[WARNING] Only {len(results)} players found with current filters. Consider relaxing search criteria.")
        
        # Prioritize main club matches if club filter is specified
        # For example: "barcelona" should prioritize "FC Barcelona" over "RCD Espanyol de Barcelona"
        if 'club' in filters and filters['club'] and len(results) > 0:
            club_filter = filters['club'].lower().strip()
            
            # Calculate priority score for each club
            # Highest priority: major club with common prefix (FC Barcelona, Real Madrid, etc.)
            # Medium priority: starts with search term (Barcelona SC)  
            # Low priority: contains search term elsewhere (RCD Espanyol de Barcelona)
            def club_priority(club_name):
                club_lower = str(club_name).lower()
                # Exact match
                if club_lower == club_filter:
                    return 4
                # Common European club prefix + search term (FC Barcelona, Real Madrid, etc.)
                # This handles the major clubs correctly
                major_prefixes = ['fc ', 'real ', 'atletico ']
                for prefix in major_prefixes:
                    if club_lower.startswith(prefix) and club_lower[len(prefix):].startswith(club_filter):
                        return 3
                # Starts with search term directly
                if club_lower.startswith(club_filter + ' ') or (club_lower.startswith(club_filter) and not any(club_lower.endswith(suffix) for suffix in [' de ' + club_filter])):
                    return 2
                # Less common prefix + search term (RCD, Athletic, Club, etc.)
                other_prefixes = ['rcd ', 'athletic ', 'club ', 'ca ', 'cd ']
                for prefix in other_prefixes:
                    if club_lower.startswith(prefix) and club_lower[len(prefix):].startswith(club_filter):
                        return 2
                # Contains search term anywhere (e.g., "RCD Espanyol de Barcelona")
                if club_filter in club_lower:
                    return 1
                return 0
            
            results['_club_priority'] = results['club_name'].apply(club_priority)
            # Sort by priority first, then by the specified sort column
            results = results.sort_values(by=['_club_priority', sort_by], ascending=[False, ascending])
            results = results.drop('_club_priority', axis=1)
        else:
            # Sort the results normally
            if sort_by in results.columns:
                results = results.sort_values(by=sort_by, ascending=ascending)
        
        # Always try to return at least some results if the dataset has them
        if len(results) == 0:
            print("[ERROR] No players match the specified criteria")
        else:
            print(f"[OK] Found {len(results)} players matching criteria, returning top {min(limit, len(results))}")
        
        return results.head(limit)
    
    def search_players_text(self, query, limit=50):
        """Enhanced text-based search using FIFA dataset directly with intelligent matching"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        # Check cache first for exact query match
        cache_key = f"{query.lower().strip()}_{limit}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key].copy()
        
        # Import text processor for normalization
        try:
            from text_processor import get_text_processor
            text_processor = get_text_processor()
            query_lower = text_processor.normalize_text(query)
        except:
            query_lower = query.lower().strip()
        
        query_words = query_lower.split()
        original_query_words = query_words.copy()  # Keep original for parsing
        
        # ======================================================================
        # PARSE COMPARISON OPERATORS FIRST (before number extraction)
        # ======================================================================
        age_min, age_max = None, None
        ovr_min, ovr_max = None, None
        
        # Parse "between X and Y" patterns
        if 'between' in query_lower and 'and' in query_lower:
            between_idx = original_query_words.index('between') if 'between' in original_query_words else -1
            if between_idx >= 0 and between_idx + 3 < len(original_query_words):
                try:
                    val1 = int(original_query_words[between_idx + 1])
                    val2 = int(original_query_words[between_idx + 3])  # Skip "and"
                    min_val, max_val = min(val1, val2), max(val1, val2)
                    
                    # Determine if it's age or ovr based on context
                    context_words = original_query_words[:between_idx]
                    if any(w in context_words for w in ['age', 'years', 'old']):
                        # "between" is inclusive, but filter uses <, so add 1 to max
                        age_min, age_max = min_val, max_val + 1
                    elif any(w in context_words for w in ['ovr', 'overall', 'rating', 'rated']):
                        # "between" is inclusive, but filter uses <=, so keep as is
                        ovr_min, ovr_max = min_val, max_val
                    else:
                        # Default to ovr if no context - "between" is inclusive
                        ovr_min, ovr_max = min_val, max_val
                    
                    # Remove parsed words from working list
                    query_words = [w for w in query_words if w not in ['between', str(val1), 'and', str(val2)]]
                except (ValueError, IndexError):
                    pass
        
        # Parse "greater than", "more than", "above", "over" patterns
        comparison_operators = {
            'greater': ('>', 1), 'more': ('>', 1), 'above': ('>', 1), 'over': ('>', 1),
            'less': ('<', 0), 'fewer': ('<', 0), 'under': ('<', 0), 'below': ('<', 0)
        }
        
        for operator, (op_symbol, offset) in comparison_operators.items():
            if operator in query_lower:
                op_idx = original_query_words.index(operator) if operator in original_query_words else -1
                if op_idx >= 0:
                    # Look for "than" after operator
                    than_idx = op_idx + 1 if op_idx + 1 < len(original_query_words) and original_query_words[op_idx + 1] == 'than' else op_idx
                    num_idx = than_idx + 1
                    
                    if num_idx < len(original_query_words):
                        try:
                            value = int(original_query_words[num_idx])
                            
                            # Determine if it's age or ovr based on context
                            context_words = original_query_words[:op_idx]
                            is_age = any(w in context_words or w in original_query_words for w in ['age', 'years', 'old'])
                            is_ovr = any(w in context_words or w in original_query_words for w in ['ovr', 'overall', 'rating', 'rated'])
                            
                            if is_age:
                                if op_symbol == '>':
                                    age_min = value + offset
                                else:
                                    age_max = value + offset
                                query_words = [w for w in query_words if w not in [operator, 'than', str(value)]]
                            elif is_ovr:
                                if op_symbol == '>':
                                    ovr_min = value + offset
                                else:
                                    ovr_max = value + offset
                                query_words = [w for w in query_words if w not in [operator, 'than', str(value)]]
                            else:
                                # Default: check if value looks like age (15-45) or rating (40-99)
                                if 15 <= value <= 45:
                                    if op_symbol == '>':
                                        age_min = value + offset
                                    else:
                                        age_max = value + offset
                                elif 40 <= value <= 99:
                                    if op_symbol == '>':
                                        ovr_min = value + offset
                                    else:
                                        ovr_max = value + offset
                                query_words = [w for w in query_words if w not in [operator, 'than', str(value)]]
                            break
                        except (ValueError, IndexError):
                            pass
        
        # ======================================================================
        # NOW EXTRACT NUMBERS FOR RESULT LIMITS
        # ======================================================================
        custom_limit = limit
        number_keywords = ['top', 'best', 'worst', 'first', 'last', 'lowest', 'highest', 'cheapest', 'fastest', 'slowest', 'tallest', 'shortest']
        
        # Check for "keyword NUMBER" pattern (e.g., "top 10")
        for i, word in enumerate(query_words):
            if word in number_keywords and i + 1 < len(query_words):
                try:
                    custom_limit = int(query_words[i + 1])
                    query_words = [w for w in query_words if w != str(custom_limit)]
                    break
                except ValueError:
                    pass
        
        # Check for "NUMBER keyword" pattern (e.g., "10 best")
        if custom_limit == limit:  # Only if not already found
            for i, word in enumerate(query_words):
                if word.isdigit() and i + 1 < len(query_words):
                    next_word = query_words[i + 1]
                    if next_word in number_keywords or next_word in ['players', 'strikers', 'defenders', 'midfielders', 'goalkeepers', 'wingers', 'forwards']:
                        try:
                            custom_limit = int(word)
                            query_words = [w for w in query_words if w != str(custom_limit)]
                            break
                        except ValueError:
                            pass
        
        # Check for standalone numbers at start/end (e.g., "show me 15 young talents")
        if custom_limit == limit:  # Only if not already found
            for word in query_words:
                if word.isdigit():
                    num = int(word)
                    if 1 <= num <= 100:  # Reasonable range for result count
                        custom_limit = num
                        query_words = [w for w in query_words if w != str(custom_limit)]
                        break
        
        # Keyword categorization with synonym mapping
        quality_keywords = ['best', 'top', 'elite', 'world', 'class', 'great', 'good', 'worst', 'bad', 'poor', 'lowest', 'highest', 'cheap', 'expensive', 'valuable']
        age_keywords = ['young', 'old', 'veteran', 'experienced', 'talent', 'promising']
        attribute_keywords_list = ['fast', 'quick', 'speedy', 'pacey', 'strong', 'physical', 'shooter', 'finisher', 'clinical', 'passer', 'playmaker', 'creative', 'dribbler', 'skilled', 'technical', 'defensive', 'tackler', 'tall', 'short']
        position_keywords_list = ['striker', 'forward', 'winger', 'midfielder', 'defender', 'goalkeeper', 'keeper', 'fullback', 'wingback', 'centre', 'center', 'attacking', 'defensive']
        rating_patterns = ['rating', 'rated', 'overall', 'ovr']
        
        # Synonym mapping for query normalization
        keyword_synonyms = {
            'lowest': 'worst',
            'cheapest': 'cheap', 
            'expensive': 'valuable',
            'highest': 'best',
            'fastest': 'fast',
            'slowest': 'slow',
            'tallest': 'tall',
            'shortest': 'short'
        }
        
        # Normalize query with synonyms
        normalized_query = query_lower
        for synonym, target in keyword_synonyms.items():
            if synonym in normalized_query:
                normalized_query = normalized_query.replace(synonym, target)
                if target not in query_lower:
                    query_words.append(target)
        
        all_keywords = quality_keywords + age_keywords + attribute_keywords_list + position_keywords_list + number_keywords + rating_patterns
        non_keyword_words = [w for w in original_query_words if w not in all_keywords and not w.isdigit()]
        
        # Detect single name query
        is_single_name_query = len(non_keyword_words) == 1 and len(original_query_words) <= 3
        
        # Create scoring dataframe
        results = self.df.copy()
        results['search_score'] = 0.0
        
        # Position keywords mapping (expanded)
        position_keywords = {
            'striker': ['ST', 'CF'],
            'forward': ['ST', 'CF', 'LW', 'RW'],
            'winger': ['LW', 'RW', 'LM', 'RM'],
            'left winger': ['LW', 'LM'],
            'right winger': ['RW', 'RM'],
            'midfielder': ['CM', 'CDM', 'CAM', 'LM', 'RM'],
            'central midfielder': ['CM'],
            'defensive midfielder': ['CDM'],
            'attacking midfielder': ['CAM'],
            'defender': ['CB', 'LB', 'RB', 'LWB', 'RWB'],
            'centre back': ['CB'],
            'center back': ['CB'],
            'fullback': ['LB', 'RB'],
            'left back': ['LB'],
            'right back': ['RB'],
            'wingback': ['LWB', 'RWB'],
            'goalkeeper': ['GK'],
            'keeper': ['GK']
        }
        
        # Attribute keywords mapping (expanded)
        attribute_keywords = {
            'fast': ('pace', 85),
            'quick': ('pace', 85),
            'speedy': ('pace', 85),
            'pacey': ('pace', 85),
            'strong': ('physic', 80),
            'physical': ('physic', 80),
            'shooter': ('shooting', 80),
            'finisher': ('shooting', 85),
            'clinical': ('shooting', 85),
            'passer': ('passing', 80),
            'playmaker': ('passing', 85),
            'creative': ('passing', 80),
            'dribbler': ('dribbling', 80),
            'skilled': ('dribbling', 85),
            'technical': ('dribbling', 80),
            'defensive': ('defending', 75),
            'tackler': ('defending', 80)
        }
        
        # League/Competition keywords
        league_keywords = {
            'premier league': ['England', 'English', 'Manchester', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham'],
            'la liga': ['Spain', 'Spanish', 'Real Madrid', 'Barcelona', 'Atletico'],
            'serie a': ['Italy', 'Italian', 'Juventus', 'Milan', 'Inter', 'Roma', 'Napoli'],
            'bundesliga': ['Germany', 'German', 'Bayern', 'Dortmund', 'Leipzig'],
            'ligue 1': ['France', 'French', 'PSG', 'Paris', 'Lyon', 'Marseille']
        }
        
        # Single name query - exact name matching (using pre-computed columns)
        if is_single_name_query:
            search_name = non_keyword_words[0]
            
            # Use pre-computed normalized names if available
            if '_norm_first_name' in results.columns:
                exact_match = (results['_norm_first_name'] == search_name) | (results['_norm_last_name'] == search_name)
                results.loc[exact_match, 'search_score'] += 200
                
                partial_match = (results['_norm_first_name'].str.contains(search_name, na=False)) | (results['_norm_last_name'].str.contains(search_name, na=False))
                results.loc[partial_match & ~exact_match, 'search_score'] += 100
                
                short_match = results['_norm_short_name'].str.contains(search_name, na=False)
                results.loc[short_match, 'search_score'] += 50
            else:
                # Fallback to runtime computation
                try:
                    from text_processor import get_text_processor
                    tp = get_text_processor()
                    results['first_name'] = results['long_name'].apply(lambda x: tp.normalize_text(str(x).split()[0]) if pd.notna(x) else '')
                    results['last_name'] = results['long_name'].apply(lambda x: tp.normalize_text(str(x).split()[-1]) if pd.notna(x) else '')
                    results['norm_short_name'] = results['short_name'].apply(lambda x: tp.normalize_text(str(x)) if pd.notna(x) else '')
                except:
                    results['first_name'] = results['long_name'].str.split().str[0].str.lower()
                    results['last_name'] = results['long_name'].str.split().str[-1].str.lower()
                    results['norm_short_name'] = results['short_name'].str.lower()
                
                exact_match = (results['first_name'] == search_name) | (results['last_name'] == search_name)
                results.loc[exact_match, 'search_score'] += 200
                
                partial_match = (results['first_name'].str.contains(search_name, na=False)) | (results['last_name'].str.contains(search_name, na=False))
                results.loc[partial_match & ~exact_match, 'search_score'] += 100
                
                short_match = results['norm_short_name'].str.contains(search_name, na=False)
                results.loc[short_match, 'search_score'] += 50
                
                results = results.drop(columns=['first_name', 'last_name', 'norm_short_name'])
        else:
            # Multi-word query - enhanced name matching (using pre-computed columns)
            if '_norm_long_name' not in results.columns:
                try:
                    from text_processor import get_text_processor
                    tp = get_text_processor()
                    results['norm_long_name'] = results['long_name'].apply(lambda x: tp.normalize_text(str(x)) if pd.notna(x) else '')
                    results['norm_short_name'] = results['short_name'].apply(lambda x: tp.normalize_text(str(x)) if pd.notna(x) else '')
                except:
                    results['norm_long_name'] = results['long_name'].str.lower()
                    results['norm_short_name'] = results['short_name'].str.lower()
            else:
                results['norm_long_name'] = results['_norm_long_name']
                results['norm_short_name'] = results['_norm_short_name']
            
            for word in query_words:
                if len(word) > 2 and word not in all_keywords and not word.isdigit():
                    # Check if word is in any part of normalized name
                    name_parts = results['norm_long_name'].str.split()
                    exact_name_part = name_parts.apply(lambda parts: word in parts if isinstance(parts, list) else False)
                    results.loc[exact_name_part, 'search_score'] += 150
                    
                    name_match = results['norm_long_name'].str.contains(word, na=False)
                    results.loc[name_match & ~exact_name_part, 'search_score'] += 100
                    
                    short_match = results['norm_short_name'].str.contains(word, na=False)
                    results.loc[short_match, 'search_score'] += 80
            
            results = results.drop(columns=['norm_long_name', 'norm_short_name'])
        
        # Nationality/Country matching (using pre-computed lowercase if available)
        nationality_col = '_nationality_lower' if '_nationality_lower' in results.columns else 'nationality_name'
        if nationality_col == 'nationality_name':
            nationality_lower = results['nationality_name'].str.lower()
        else:
            nationality_lower = results['_nationality_lower']
        
        # Map nationality keywords to proper country names
        nationality_map = {
            'brazilian': 'brazil',
            'argentinian': 'argentina',
            'french': 'france',
            'spanish': 'spain',
            'german': 'germany',
            'english': 'england',
            'italian': 'italy',
            'portuguese': 'portugal',
            'dutch': 'netherlands',
            'belgian': 'belgium'
        }
        
        for word in query_words:
            # Check if word is a nationality adjective and map it
            search_word = nationality_map.get(word, word)
            
            if word not in all_keywords and not word.isdigit() and len(word) > 3:
                nationality_match = nationality_lower.str.contains(search_word, na=False)
                results.loc[nationality_match, 'search_score'] += 60
        
        # Club matching (using pre-computed lowercase if available)
        club_col = '_club_lower' if '_club_lower' in results.columns else 'club_name'
        if club_col == 'club_name':
            club_lower = results['club_name'].str.lower()
        else:
            club_lower = results['_club_lower']
        
        for word in query_words:
            if word not in all_keywords and not word.isdigit() and len(word) > 2:
                club_match = club_lower.str.contains(word, na=False)
                results.loc[club_match, 'search_score'] += 50
        
        # League matching
        for league, keywords in league_keywords.items():
            if league in query_lower:
                for keyword in keywords:
                    club_match = results['club_name'].str.contains(keyword, case=False, na=False)
                    results.loc[club_match, 'search_score'] += 40
        
        # Position matching with strict filtering
        position_matched = False
        strict_position_filter = False
        for keyword, positions in position_keywords.items():
            if keyword in query_lower:
                position_matched = True
                # Check if player has ANY of the positions for this keyword
                any_pos_match = pd.Series([False] * len(results), index=results.index)
                for pos in positions:
                    pos_match = results['player_positions'].str.contains(pos, case=False, na=False)
                    any_pos_match |= pos_match
                
                # Award points for matching the position group
                results.loc[any_pos_match, 'search_score'] += 80
                
                # Apply penalty only ONCE if player doesn't match ANY position in the group
                if keyword in ['goalkeeper', 'keeper', 'striker', 'winger']:
                    strict_position_filter = True
                    results.loc[~any_pos_match, 'search_score'] -= 150
        
        # Attribute matching with thresholds
        for keyword, (attr_col, threshold) in attribute_keywords.items():
            if keyword in query_lower and attr_col in results.columns:
                excellent_attr = results[attr_col] >= threshold + 5
                high_attr = (results[attr_col] >= threshold) & (results[attr_col] < threshold + 5)
                medium_attr = (results[attr_col] >= threshold - 10) & (results[attr_col] < threshold)
                
                results.loc[excellent_attr, 'search_score'] += 60
                results.loc[high_attr, 'search_score'] += 40
                results.loc[medium_attr, 'search_score'] += 20
        
        # Quality modifiers (using normalized query)
        if any(word in normalized_query for word in ['best', 'top', 'elite', 'world class', 'great', 'highest']):
            results['search_score'] += (results['overall'] - 70) * 3.5
            results.loc[results['overall'] >= 88, 'search_score'] += 80
            results.loc[results['overall'] >= 85, 'search_score'] += 50
        
        if any(word in normalized_query for word in ['worst', 'bad', 'poor']):
            results['search_score'] += (75 - results['overall']) * 2
            results.loc[results['overall'] <= 65, 'search_score'] += 50
        
        # Value-based modifiers
        if 'cheap' in normalized_query and 'value_eur' in results.columns:
            results.loc[results['value_eur'] <= 1000000, 'search_score'] += 60
            results.loc[results['value_eur'] <= 500000, 'search_score'] += 40
        
        if 'valuable' in normalized_query and 'value_eur' in results.columns:
            results.loc[results['value_eur'] >= 50000000, 'search_score'] += 60
            results.loc[results['value_eur'] >= 100000000, 'search_score'] += 40
        
        # Physical attribute modifiers
        if 'tall' in normalized_query and 'height_cm' in results.columns:
            results.loc[results['height_cm'] >= 190, 'search_score'] += 60
            results.loc[results['height_cm'] >= 185, 'search_score'] += 30
        
        if 'short' in normalized_query and 'height_cm' in results.columns:
            results.loc[results['height_cm'] <= 170, 'search_score'] += 60
            results.loc[results['height_cm'] <= 175, 'search_score'] += 30
        
        # Age-based filtering
        if 'young' in query_lower or 'talent' in query_lower or 'promising' in query_lower:
            young_talent = (results['age'] <= 23) & (results['potential'] >= 80)
            results.loc[young_talent, 'search_score'] += 70
            results['pot_diff'] = results['potential'] - results['overall']
            results.loc[results['pot_diff'] > 15, 'search_score'] += 50
            results.loc[results['pot_diff'] > 10, 'search_score'] += 30
            results = results.drop(columns=['pot_diff'])
        
        if 'old' in query_lower or 'veteran' in query_lower or 'experienced' in query_lower:
            veteran = results['age'] >= 32
            results.loc[veteran, 'search_score'] += 50
        
        # Apply age range filters
        if age_min is not None:
            results = results[results['age'] >= age_min]
        if age_max is not None:
            results = results[results['age'] < age_max]
        
        # Apply overall rating filters
        if ovr_min is not None:
            results = results[results['overall'] >= ovr_min]
        if ovr_max is not None:
            results = results[results['overall'] <= ovr_max]
        
        # Foot preference
        if 'left footed' in query_lower or 'left foot' in query_lower:
            results.loc[results['preferred_foot'] == 'Left', 'search_score'] += 40
        if 'right footed' in query_lower or 'right foot' in query_lower:
            results.loc[results['preferred_foot'] == 'Right', 'search_score'] += 40
        
        # Work rate matching
        if 'high attacking' in query_lower:
            results.loc[results['attacking_work_rate'] == 'High', 'search_score'] += 30
        if 'high defensive' in query_lower:
            results.loc[results['defensive_work_rate'] == 'High', 'search_score'] += 30
        
        # Filter by score threshold
        # Special case: if query only has filters (age/ovr) and no meaningful search terms, don't filter by score
        has_only_filters = (age_min is not None or age_max is not None or ovr_min is not None or ovr_max is not None)
        
        # Check if we have meaningful search terms (excluding common words like 'players', 'with', comparison operators, etc.)
        common_words = {'players', 'player', 'with', 'from', 'in', 'at', 'of', 'the', 'a', 'an', 'and',
                        'above', 'below', 'over', 'under', 'between', 'greater', 'less', 'than',
                        'more', 'fewer', 'higher', 'lower', 'good', 'bad', 'age', 'years', 'old',
                        'ovr', 'overall', 'rating', 'rated'}
        meaningful_search_terms = [w for w in original_query_words 
                                    if w not in all_keywords 
                                    and w not in common_words
                                    and not w.isdigit() 
                                    and len(w) > 2]
        has_meaningful_search = len(meaningful_search_terms) > 0
        
        if any(word in normalized_query for word in ['best', 'top', 'elite', 'highest']):
            if not position_matched and not has_meaningful_search:
                results = results.sort_values(by='overall', ascending=False)
                return results.head(custom_limit)
            else:
                if has_meaningful_search:
                    results = results[results['search_score'] > -100]
        elif any(word in normalized_query for word in ['worst', 'poor', 'bad']):
            # For "worst/lowest" queries, sort by overall ascending
            if results['search_score'].max() > 0:
                results = results[results['search_score'] > 0]
            results = results.sort_values(by=['search_score', 'overall'], ascending=[False, True])
            return results.head(custom_limit)
        else:
            # Only filter by score if there were meaningful search terms (names/clubs/nationalities)
            if has_meaningful_search:
                results = results[results['search_score'] > 0]
            elif has_only_filters:
                # If we have filters but no search terms, keep all results (filters already applied)
                pass
            else:
                # If no filters and no meaningful search terms, filter by score
                results = results[results['search_score'] > 0]
        
        # Fallback search if no results found
        if results.empty or len(results) == 0:
            # Try a broader text search
            combined_text = (
                self.df['long_name'].fillna('') + ' ' +
                self.df['player_positions'].fillna('') + ' ' +
                self.df['nationality_name'].fillna('') + ' ' +
                self.df['club_name'].fillna('')
            ).str.lower()
            
            search_terms = [w for w in query_words if w not in all_keywords and not w.isdigit() and len(w) > 2]
            if search_terms:
                match_mask = combined_text.str.contains('|'.join(search_terms), na=False, regex=True)
                results = self.df[match_mask].copy()
                results['search_score'] = results['overall']
            
            # If still no results, return top players as fallback
            if results.empty or len(results) == 0:
                results = self.df.copy()
                results['search_score'] = results['overall']
                results = results.sort_values(by='overall', ascending=False).head(custom_limit)
        
        # Sort by score then overall rating
        results = results.sort_values(by=['search_score', 'overall'], ascending=[False, False])
        
        final_results = results.head(custom_limit)
        
        # Store in cache (LRU-style: remove oldest if cache is full)
        if len(self._search_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
        
        self._search_cache[cache_key] = final_results.copy()
        
        return final_results
    
    def apply_filters(self, df, filters):
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Overall rating
        if 'overall_min' in filters:
            filtered_df = filtered_df[filtered_df['overall'] >= filters['overall_min']]
        if 'overall_max' in filters:
            filtered_df = filtered_df[filtered_df['overall'] <= filters['overall_max']]
        
        # Potential
        if 'potential_min' in filters:
            filtered_df = filtered_df[filtered_df['potential'] >= filters['potential_min']]
        if 'potential_max' in filters:
            filtered_df = filtered_df[filtered_df['potential'] <= filters['potential_max']]
        
        # Age
        if 'age_min' in filters:
            filtered_df = filtered_df[filtered_df['age'] >= filters['age_min']]
        if 'age_max' in filters:
            filtered_df = filtered_df[filtered_df['age'] <= filters['age_max']]
        
        # Position
        if 'position' in filters and filters['position']:
            filtered_df = filtered_df[filtered_df['player_positions'].str.contains(filters['position'], case=False, na=False)]
        
        # Attributes
        attribute_mapping = {
            'pace': 'pace',
            'shooting': 'shooting', 
            'passing': 'passing',
            'dribbling': 'dribbling',
            'defending': 'defending',
            'physicality': 'physic'
        }
        
        for attr_key, data_col in attribute_mapping.items():
            min_key = f"{attr_key}_min"
            max_key = f"{attr_key}_max"
            
            if min_key in filters and data_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[data_col] >= filters[min_key]]
            if max_key in filters and data_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[data_col] <= filters[max_key]]
        
        # Text filters with improved matching
        if 'nationality' in filters and filters['nationality']:
            nat_filter = filters['nationality'].strip()
            if nat_filter:
                filtered_df = filtered_df[filtered_df['nationality_name'].str.contains(nat_filter, case=False, na=False)]
        
        if 'club' in filters and filters['club']:
            club_filter = filters['club'].strip()
            if club_filter:
                # Club aliases for common abbreviations and variations
                club_aliases = {
                    'psg': 'paris saint-germain',
                    'barca': 'barcelona',
                    'real': 'real madrid',
                    'atletico': 'atl',  # Matches "Atltico" (partial match)
                    'munchen': 'bayern',  # Mnchen alternative
                    'munich': 'bayern',
                    'man utd': 'manchester united',
                    'man city': 'manchester city',
                    'juve': 'juventus',
                    'spurs': 'tottenham',
                    'arsenal': 'arsenal',
                    'inter': 'inter',
                    'ac milan': 'ac milan',
                }
                
                # Check if the filter is an alias
                club_search = club_aliases.get(club_filter.lower(), club_filter)
                
                # Also try to normalize accented characters for matching
                # Create a normalized version of club names for matching
                try:
                    import unicodedata
                    def normalize_text(text):
                        if pd.isna(text):
                            return ''
                        # Normalize unicode and remove accents
                        normalized = unicodedata.normalize('NFD', str(text))
                        return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn').lower()
                    
                    # Try exact match first
                    match = filtered_df['club_name'].str.contains(club_search, case=False, na=False)
                    
                    # If no matches, try normalized search
                    if match.sum() == 0:
                        normalized_clubs = filtered_df['club_name'].apply(normalize_text)
                        normalized_search = normalize_text(club_search)
                        match = normalized_clubs.str.contains(normalized_search, na=False)
                    
                    filtered_df = filtered_df[match]
                except:
                    filtered_df = filtered_df[filtered_df['club_name'].str.contains(club_search, case=False, na=False)]
        
        return filtered_df

    def get_alternative_image_url(self, player_row):
        """Get player image from alternative source using SoFIFA ID"""
        sofifa_id = player_row.get('sofifa_id', '')
        if sofifa_id:
            # Format: https://cdn.sofifa.net/players/158/023/22_120.png
            sofifa_str = str(int(sofifa_id))
            if len(sofifa_str) >= 6:
                part1 = sofifa_str[:-3]
                part2 = sofifa_str
                return f"https://cdn.sofifa.net/players/{part1}/{part2}/22_120.png"
        return ""

    def fix_image_url(self, url):
        """Fix common image URL issues"""
        if not url:
            return ""
        
        # Fix relative URLs
        if url.startswith('//'):
            return 'https:' + url
        elif url.startswith('/'):
            return 'https://cdn.sofifa.net' + url
        elif not url.startswith('http'):
            return 'https://cdn.sofifa.net' + url
        
        return url

    def get_player_card_data(self, player_row):
        """Extract player data for frontend display"""
        def safe_int(value, default=0):
            """Safely convert value to int, handling NaN and None"""
            try:
                if pd.isna(value):
                    return default
                return int(value)
            except (ValueError, TypeError):
                return default
        
        def safe_str(value, default=''):
            """Safely convert value to string, handling NaN and None"""
            try:
                if pd.isna(value):
                    return default
                return str(value)
            except (ValueError, TypeError):
                return default
        
        # Get original URL and fix it
        original_url = safe_str(player_row.get('player_face_url', ''))
        fixed_original_url = self.fix_image_url(original_url)
        
        # Get alternative URL
        alternative_url = self.get_alternative_image_url(player_row)
        
        # Choose the best available URL
        photo_url = fixed_original_url if fixed_original_url else alternative_url
        
        # Get nation flag URL and club logo URL from dataset
        nation_flag_url = self.fix_image_url(safe_str(player_row.get('nation_flag_url', '')))
        club_logo_url = self.fix_image_url(safe_str(player_row.get('club_logo_url', '')))
        
        return {
            'id': safe_str(player_row.get('sofifa_id', player_row.get('player_id', ''))),
            'name': safe_str(player_row.get('long_name', ''), 'Unknown'),
            'short_name': safe_str(player_row.get('short_name', ''), 'Unknown'),
            'overall': safe_int(player_row.get('overall', 0)),
            'potential': safe_int(player_row.get('potential', 0)),
            'position': safe_str(player_row.get('player_positions', ''), 'SUB'),
            'age': safe_int(player_row.get('age', 0)),
            'club': safe_str(player_row.get('club_name', ''), 'Free Agent'),
            'nationality': safe_str(player_row.get('nationality_name', ''), 'Unknown'),
            'photo_url': photo_url,
            'nation_flag_url': nation_flag_url,
            'club_logo_url': club_logo_url,
            'preferred_foot': safe_str(player_row.get('preferred_foot', ''), 'Right'),
            'skill_moves': safe_int(player_row.get('skill_moves', 0)),
            'weak_foot': safe_int(player_row.get('weak_foot', 0)),
            'attributes': {
                'pace': safe_int(player_row.get('pace', 0)),
                'shooting': safe_int(player_row.get('shooting', 0)),
                'passing': safe_int(player_row.get('passing', 0)),
                'dribbling': safe_int(player_row.get('dribbling', 0)),
                'defending': safe_int(player_row.get('defending', 0)),
                'physicality': safe_int(player_row.get('physic', 0))
            },
            'value_eur': safe_int(player_row.get('value_eur', 0)),
            'wage_eur': safe_int(player_row.get('wage_eur', 0))
        }

    def load_detailed_profiles(self):
        """Load optional detailed profiles (if available) and cache them."""
        try:
            if os.path.exists(self.detailed_profiles_path):
                with open(self.detailed_profiles_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Normalize to mapping by player_id (Transfermarkt ID)
                if isinstance(data, dict):
                    # Keys are already player_id strings
                    self.detailed_profiles = {str(k): v for k, v in data.items()}
                elif isinstance(data, list):
                    # If it's a list of profiles, index by player_id
                    mapping = {}
                    for item in data:
                        pid = None
                        if isinstance(item, dict):
                            if 'player_id' in item:
                                pid = item.get('player_id')
                            elif 'id' in item:
                                pid = item.get('id')
                        if pid is not None:
                            mapping[str(pid)] = item
                    self.detailed_profiles = mapping

                print(f"[OK] Loaded detailed profiles: {len(self.detailed_profiles)} entries")
            else:
                self.detailed_profiles = {}
        except Exception as e:
            print(f"[ERROR] Error loading detailed profiles: {e}")
            self.detailed_profiles = {}

# Initialize the search engine
search_engine = ScoutSearchEngine()

# Initialize optimized search engine (with pre-built indices)
optimized_search = None
try:
    optimized_search = OptimizedSearchEngine(search_engine.df)
except Exception as e:
    print(f"[WARNING] Optimized search not available: {e}")


@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return send_from_directory(base_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files directly from root (like 7070065.jpg, etc.)"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return send_from_directory(base_dir, filename)

# NEW: Text Search Endpoint
@app.route('/api/text-search', methods=['POST', 'OPTIONS'])
def text_search_players():
    """API endpoint for text-based player search with semantic expansion"""
    if request.method == 'OPTIONS':
        return '', 200
    
    start_time = time.time()
    query = ""
    success = True
    result_count = 0
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        limit = data.get('limit', 50)
        use_semantic = data.get('semantic', True)  # Enable by default
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400
        
        print(f" Text search query: '{query}' (semantic={use_semantic})")
        
        # Apply semantic expansion if enabled
        expanded_query = query
        if use_semantic:
            try:
                from semantic_search import semantic_engine
                expanded_terms = semantic_engine.expand_query(query, max_expansions=2)
                expanded_query = ' '.join(expanded_terms)
                print(f" Expanded query: '{expanded_query}'")
            except Exception as e:
                print(f"[WARNING] Semantic expansion failed: {e}")
        
        # Perform text search with expanded query
        results_df = search_engine.search_players_text(expanded_query, limit=limit)
        result_count = len(results_df)
        
        print(f" Found {len(results_df)} results")
        
        # Convert to frontend format
        players_data = []
        for _, player in results_df.iterrows():
            players_data.append(search_engine.get_player_card_data(player))
        
        # Sanitize for JSON serialization
        players_data = sanitize_for_json(players_data)
        
        return jsonify({
            'success': True,
            'players': players_data,
            'count': len(players_data),
            'query': query,
            'expanded_query': expanded_query if use_semantic else None
        })
        
    except Exception as e:
        success = False
        print(f"[ERROR] Text search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    finally:
        # Track performance
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        word_count = len(query.split()) if query else 0
        query_type = 'text_search_single' if word_count == 1 else f'text_search_{word_count}word'
        track_query(query, response_time, result_count, query_type, success)

# Your existing endpoints
@app.route('/api/search', methods=['POST', 'OPTIONS'])
def search_players():
    """API endpoint for attribute-based player search"""
    if request.method == 'OPTIONS':
        return '', 200
    
    start_time = time.time()
    success = True
    result_count = 0
    
    try:
        data = request.get_json()
        
        # Extract filters from request
        filters = {
            'overall_min': data.get('overallMin', 0),
            'overall_max': data.get('overallMax', 99),
            'potential_min': data.get('potentialMin', 0),
            'potential_max': data.get('potentialMax', 99),
            'position': data.get('position', ''),
            'pace_min': data.get('paceMin', 0),
            'pace_max': data.get('paceMax', 99),
            'shooting_min': data.get('shootingMin', 0),
            'shooting_max': data.get('shootingMax', 99),
            'passing_min': data.get('passingMin', 0),
            'passing_max': data.get('passingMax', 99),
            'dribbling_min': data.get('dribblingMin', 0),
            'dribbling_max': data.get('dribblingMax', 99),
            'nationality': data.get('nationality', ''),
            'club': data.get('club', '')
        }
        
        # Debug logging for attribute search
        if filters.get('club') and filters.get('position'):
            print(f"[DEBUG ATTR] Club='{filters['club']}' Position='{filters['position']}'")
        
        # Only apply age filters if explicitly provided
        if 'ageMin' in data and data['ageMin'] is not None:
            filters['age_min'] = data['ageMin']
        if 'ageMax' in data and data['ageMax'] is not None:
            filters['age_max'] = data['ageMax']
        
        sort_by = data.get('sortBy', 'overall')
        limit = data.get('limit', 50)
        
        # Get query if provided (for name filtering)
        query = data.get('query', '').strip()
        
        # USE OPTIMIZED SEARCH if available and query is provided
        if optimized_search and query:
            opt_filters = {
                'overallMin': filters.get('overall_min', 0),
                'overallMax': filters.get('overall_max', 99),
                'ageMin': filters.get('age_min', None),
                'ageMax': filters.get('age_max', None),
                'position': filters.get('position', ''),
                'paceMin': filters.get('pace_min', 0),
                'shootingMin': filters.get('shooting_min', 0)
            }
            
            results = optimized_search.search(query, opt_filters, limit)
            result_count = len(results)
            
            # Convert optimized results to frontend format (matching get_player_card_data)
            players_data = []
            for player in results:
                card_data = {
                    'id': '',  # Optimized search doesn't have sofifa_id in preprocessed data
                    'name': player['long_name'],
                    'short_name': player['short_name'],
                    'overall': int(player['overall']),
                    'potential': int(player['potential']),
                    'position': player['player_positions'],
                    'age': int(player['age']),
                    'club': player['club_name'],
                    'nationality': player['nationality_name'],
                    'photo_url': '',  # Not in optimized index
                    'nation_flag_url': '',
                    'club_logo_url': '',
                    'preferred_foot': 'Right',
                    'skill_moves': 0,
                    'weak_foot': 0,
                    'attributes': {
                        'pace': int(player['pace']),
                        'shooting': int(player['shooting']),
                        'passing': int(player['passing']),
                        'dribbling': int(player['dribbling']),
                        'defending': int(player['defending']),
                        'physicality': int(player['physic'])
                    },
                    'value_eur': float(player['value_eur']),
                    'wage_eur': float(player['wage_eur'])
                }
                players_data.append(card_data)
        else:
            # Fallback to original search
            results_df = search_engine.search_players(filters, sort_by=sort_by, limit=limit)
            result_count = len(results_df)
            
            # Debug logging: show what clubs are in the results
            if filters.get('club') and filters.get('position') and len(results_df) > 0:
                clubs_in_results = results_df['club_name'].value_counts()
                print(f"[DEBUG ATTR] Results by club: {dict(clubs_in_results.head(5))}")
                barcelona_only = results_df[results_df['club_name'].str.contains('FC Barcelona', case=False, na=False)]
                print(f"[DEBUG ATTR] FC Barcelona players: {len(barcelona_only)}")
                if len(barcelona_only) > 0:
                    print(f"[DEBUG ATTR] FC Barcelona strikers: {list(barcelona_only['short_name'].head(5))}")
            
            # Convert to frontend format
            players_data = []
            for _, player in results_df.iterrows():
                players_data.append(search_engine.get_player_card_data(player))
        
        # Sanitize for JSON serialization
        players_data = sanitize_for_json(players_data)
        
        return jsonify({
            'success': True,
            'players': players_data,
            'count': len(players_data),
            'message': f'Found {len(players_data)} players' if len(players_data) > 0 else 'No players found matching your criteria. Try adjusting the filters.'
        })
        
    except Exception as e:
        success = False
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    finally:
        # Track performance
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        track_query('attribute_search', response_time, result_count, 'attribute_search', success)


@app.route('/api/test-search', methods=['GET'])
def test_search():
    """Test endpoint to verify search logic works"""
    try:
        # Test with overall <= 76
        test_filters = {
            'overall_min': 0,
            'overall_max': 76
        }
        
        print(f" Testing with filters: {test_filters}")
        results_df = search_engine.search_players(filters=test_filters, limit=20)
        print(f" Got {len(results_df)} results")
        
        players_list = []
        for _, player in results_df.iterrows():
            players_list.append({
                'name': player.get('short_name', 'Unknown'),
                'overall': int(player.get('overall', 0)),
                'age': int(player.get('age', 0)),
                'club': player.get('club_name', 'Unknown')
            })
        
        return jsonify({
            'success': True,
            'total_results': len(results_df),
            'filters_used': test_filters,
            'players': players_list,
            'dataset_size': len(search_engine.df)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/players/top', methods=['GET', 'OPTIONS'])
def get_top_players():
    """Get top players for initial display"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        filters = {'overall_min': 80}
        results_df = search_engine.search_players(filters, limit=20)
        
        players_data = []
        for _, player in results_df.iterrows():
            players_data.append(search_engine.get_player_card_data(player))
        
        # Sanitize for JSON serialization
        players_data = sanitize_for_json(players_data)
        
        return jsonify({
            'success': True,
            'players': players_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug/images', methods=['GET'])
def debug_images():
    """Debug endpoint to check image URLs"""
    try:
        # Get top 5 players with their image URLs
        filters = {'overall_min': 85}
        results_df = search_engine.search_players(filters, limit=10)
        
        debug_info = []
        for _, player in results_df.iterrows():
            player_data = search_engine.get_player_card_data(player)
            
            debug_info.append({
                'name': player.get('long_name', ''),
                'sofifa_id': player.get('sofifa_id', ''),
                'original_url': player.get('player_face_url', ''),
                'final_url': player_data['photo_url'],
                'has_original': bool(player.get('player_face_url', '')),
                'alternative_url': search_engine.get_alternative_image_url(player)
            })
        
        return jsonify({
            'success': True,
            'debug_info': debug_info,
            'total_players_checked': len(debug_info)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug/columns', methods=['GET'])
def debug_columns():
    """Debug endpoint to check available columns"""
    try:
        if search_engine.df is None:
            return jsonify({'success': False, 'error': 'Data not loaded'})
        
        # Get all columns
        all_columns = list(search_engine.df.columns)
        
        # Get image-related columns
        image_columns = [col for col in all_columns if any(keyword in col.lower() for keyword in 
                        ['url', 'logo', 'flag', 'badge', 'face', 'image'])]
        
        # Get first player sample to see actual data
        sample_player = search_engine.df.iloc[0] if len(search_engine.df) > 0 else {}
        sample_data = {}
        
        for col in image_columns:
            if col in sample_player and pd.notna(sample_player[col]):
                sample_data[col] = sample_player[col]
        
        return jsonify({
            'success': True,
            'total_columns': len(all_columns),
            'image_columns': image_columns,
            'sample_image_data': sample_data,
            'first_5_columns': all_columns[:5]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug/test-player', methods=['GET'])
def debug_test_player():
    """Test endpoint to check one player's data"""
    try:
        if search_engine.df is None or search_engine.df.empty:
            return jsonify({'success': False, 'error': 'Data not loaded'})
        
        # Get first player
        player = search_engine.df.iloc[0]
        player_data = search_engine.get_player_card_data(player)
        
        return jsonify({
            'success': True,
            'player': player_data,
            'has_nation_flag': bool(player_data['nation_flag_url']),
            'has_club_logo': bool(player_data['club_logo_url']),
            'has_skill_moves': 'skill_moves' in player_data,
            'has_preferred_foot': 'preferred_foot' in player_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/player/<int:player_id>', methods=['GET'])
def get_player_details(player_id):
    """Get detailed information for a specific player"""
    try:
        player_df = search_engine.df[search_engine.df['sofifa_id'] == player_id]
        
        if player_df.empty:
            return jsonify({
                'success': False,
                'error': 'Player not found'
            }), 404
        
        player = player_df.iloc[0]
        player_data = search_engine.get_player_card_data(player)

        # Attempt to attach richer detailed profile if available
        detailed = {}
        try:
            # Map sofifa_id to transfermarkt player_id, then lookup detailed profile
            if hasattr(search_engine, 'detailed_profiles') and hasattr(search_engine.text_search_engine, 'player_mapping'):
                tm_player_id = search_engine.text_search_engine.player_mapping.get(str(player_id))
                if tm_player_id:
                    detailed = search_engine.detailed_profiles.get(str(tm_player_id), {})
        except Exception as e:
            print(f"Warning: Could not load detailed profile for {player_id}: {e}")
            detailed = {}

        # If detailed exists, attach under `details` to avoid colliding with core fields
        if detailed and isinstance(detailed, dict):
            player_data_enriched = dict(player_data)
            player_data_enriched['details'] = detailed
        else:
            player_data_enriched = player_data

        # Sanitize numpy/pandas types for JSON
        player_data_enriched = sanitize_for_json(player_data_enriched)

        return jsonify({
            'success': True,
            'player': player_data_enriched
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/filters/options', methods=['GET'])
def get_filter_options():
    """Get available options for filters (positions, nationalities, clubs)"""
    try:
        if search_engine.df is None:
            return jsonify({'success': False, 'error': 'Data not loaded'})
        
        # Get unique positions
        all_positions = []
        for positions in search_engine.df['player_positions'].dropna():
            if isinstance(positions, str):
                all_positions.extend([pos.strip() for pos in positions.split(',')])
        
        unique_positions = sorted(list(set(all_positions)))
        
        # Get unique nationalities (top 50)
        nationalities = search_engine.df['nationality_name'].dropna().unique()
        top_nationalities = sorted(nationalities)[:50]
        
        # Get unique clubs (top 50)
        clubs = search_engine.df['club_name'].dropna().unique()
        top_clubs = sorted(clubs)[:50]
        
        return jsonify({
            'success': True,
            'positions': unique_positions,
            'nationalities': top_nationalities.tolist(),
            'clubs': top_clubs.tolist()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        if search_engine.df is None:
            return jsonify({'success': False, 'error': 'Data not loaded'})
        
        total_players = len(search_engine.df)
        players_with_images = search_engine.df['player_face_url'].notna().sum()
        avg_overall = search_engine.df['overall'].mean()
        avg_age = search_engine.df['age'].mean()
        
        # Top 5 nationalities
        top_nationalities = search_engine.df['nationality_name'].value_counts().head(5).to_dict()
        
        # Top 5 clubs
        top_clubs = search_engine.df['club_name'].value_counts().head(5).to_dict()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_players': total_players,
                'players_with_images': int(players_with_images),
                'image_coverage': f"{(players_with_images / total_players * 100):.1f}%",
                'average_rating': f"{avg_overall:.1f}",
                'average_age': f"{avg_age:.1f}",
                'top_nationalities': top_nationalities,
                'top_clubs': top_clubs
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/image-proxy')
def image_proxy():
    """Proxy images to avoid CORS and CSP issues"""
    try:
        image_url = request.args.get('url')
        if not image_url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Fix the URL if needed
        if image_url.startswith('//'):
            image_url = 'https:' + image_url
        elif image_url.startswith('/'):
            image_url = 'https://cdn.sofifa.net' + image_url
        
        # Fetch the image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Return the image with proper headers
        return Response(
            response.content,
            content_type=response.headers.get('Content-Type', 'image/jpeg'),
            headers={
                'Cache-Control': 'public, max-age=86400',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        print(f"Image proxy error for {image_url}: {e}")
        # Return a transparent pixel as fallback
        from io import BytesIO
        try:
            from PIL import Image
            img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            img_io = BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            return Response(img_io.getvalue(), content_type='image/png')
        except ImportError:
            # If PIL is not available, return empty response
            return Response(b'', content_type='image/png')


# NEW ENDPOINTS FOR REQUIREMENTS

@app.route('/api/autocomplete', methods=['GET', 'OPTIONS'])
def autocomplete_suggestions():
    """Get autocomplete suggestions for query prefix"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        from autocomplete import autocomplete_engine
        
        prefix = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 5))
        
        if not autocomplete_engine:
            return jsonify({'suggestions': []})
        
        suggestions = autocomplete_engine.get_smart_suggestions(prefix, limit=limit)
        
        return jsonify({
            'suggestions': suggestions,
            'prefix': prefix
        })
        
    except Exception as e:
        print(f"Autocomplete error: {e}")
        return jsonify({'suggestions': [], 'error': str(e)}), 500


@app.route('/api/performance', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics and requirement compliance"""
    try:
        stats = performance_monitor.get_statistics()
        
        return jsonify({
            'success': True,
            'metrics': stats,
            'report': performance_monitor.get_performance_report().split('\n')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/barrel/stats', methods=['GET'])
def get_barrel_stats():
    """Get barrel manager statistics"""
    try:
        text_engine = search_engine.text_search_engine
        barrel_stats = text_engine.barrel_manager.get_statistics()
        
        return jsonify({
            'success': True,
            'barrel_stats': barrel_stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/semantic/expand', methods=['POST'])
def expand_query_semantic():
    """Expand query with semantic synonyms"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        max_expansions = data.get('max_expansions', 3)
        
        from semantic_search import semantic_engine
        
        if not semantic_engine:
            return jsonify({'success': False, 'error': 'Semantic engine not initialized'})
        
        expanded = semantic_engine.expand_query(query, max_expansions=max_expansions)
        
        return jsonify({
            'success': True,
            'original_query': query,
            'expanded_terms': expanded
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/player/add', methods=['POST'])
def add_new_player():
    """
    Add a new player dynamically with full indexing (REQUIREMENT #10)
    Updates lexicon, forward index, inverted index, and barrels
    Makes player immediately searchable without blocking existing searches
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['short_name', 'overall', 'age', 'nationality_name']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing)}'
            }), 400
        
        # Generate unique player ID
        new_player_id = len(search_engine.df) + 1
        data['player_id'] = new_player_id
        
        # Set defaults for missing optional fields
        defaults = {
            'long_name': data.get('short_name', 'Unknown'),
            'player_positions': data.get('player_positions', 'SUB'),
            'club_name': data.get('club_name', 'Free Agent'),
            'league_name': data.get('league_name', 'Unknown'),
            'potential': data.get('potential', data.get('overall', 70)),
            'value_eur': data.get('value_eur', 100000),
            'wage_eur': data.get('wage_eur', 1000),
            'preferred_foot': data.get('preferred_foot', 'Right'),
            'weak_foot': data.get('weak_foot', 3),
            'skill_moves': data.get('skill_moves', 3),
            'work_rate': data.get('work_rate', 'Medium/Medium'),
            'body_type': data.get('body_type', 'Normal'),
            'pace': data.get('pace', 70),
            'shooting': data.get('shooting', 70),
            'passing': data.get('passing', 70),
            'dribbling': data.get('dribbling', 70),
            'defending': data.get('defending', 70),
            'physic': data.get('physic', 70),
            'player_face_url': data.get('player_face_url', ''),
            'club_logo_url': data.get('club_logo_url', ''),
            'nation_flag_url': data.get('nation_flag_url', '')
        }
        
        # Apply defaults
        for key, value in defaults.items():
            if key not in data:
                data[key] = value
        
        # Add player to FIFA dataset (in-memory)
        new_row = pd.DataFrame([data])
        search_engine.df = pd.concat([search_engine.df, new_row], ignore_index=True)
        
        # Update optimized search engine if available
        if optimized_search is not None:
            try:
                # Add to name index
                name_tokens = data['short_name'].lower().split()
                for token in name_tokens:
                    if token not in optimized_search.name_index:
                        optimized_search.name_index[token] = set()
                    optimized_search.name_index[token].add(new_player_id)
                
                # Add to club index
                club = data.get('club_name', 'Free Agent')
                if club not in optimized_search.club_index:
                    optimized_search.club_index[club] = set()
                optimized_search.club_index[club].add(new_player_id)
                
                # Add to nationality index
                nationality = data.get('nationality_name', 'Unknown')
                if nationality not in optimized_search.nationality_index:
                    optimized_search.nationality_index[nationality] = set()
                optimized_search.nationality_index[nationality].add(new_player_id)
            except Exception as idx_error:
                print(f"[WARNING] Could not update optimized search: {idx_error}")
        
        # Create text content for indexing
        text_content = f"{data['short_name']} {data.get('long_name', '')} {data.get('player_positions', '')} {data.get('nationality_name', '')} {data.get('club_name', '')} {data.get('league_name', '')}".lower()
        
        # Check if dynamic_indexer is available
        if dynamic_indexer is None:
            return jsonify({
                'success': False,
                'error': 'Dynamic indexer not initialized. Server may need restart.'
            }), 500
        
        # Index the document using DynamicIndexer
        doc_id, success, message = dynamic_indexer.add_document(
            doc_content=text_content,
            doc_metadata={
                'player_id': new_player_id,
                'type': 'player',
                'source': 'user_added'
            }
        )
        
        if not success:
            return jsonify({
                'success': False,
                'error': f'Indexing failed: {message}'
            }), 500
        
        # Reload barrel manager to include new terms
        try:
            from barrel_manager import barrel_manager
            barrel_manager.reload_mappings()
        except:
            pass  # Not critical
        
        elapsed = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'message': f'Player "{data["short_name"]}" added and indexed successfully',
            'player_id': new_player_id,
            'doc_id': doc_id,
            'indexing_time_ms': round(elapsed, 2),
            'total_players': len(search_engine.df),
            'index_stats': dynamic_indexer.get_stats()
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def init_advanced_components():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 0. Extract compressed dataset if running on cloud
    zip_path = os.path.join(base_dir, 'data', 'scoutsearch_data.zip')
    data_dir = os.path.join(base_dir, 'data')
    
    if os.path.exists(zip_path):
        print(f"[STARTUP] Found dataset payload {zip_path}, unzipping...")
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # We changed the zip structure to extract straight into 'data' rather than 'data/index'
                zipf.extractall(data_dir)
            print("[STARTUP] Dataset extracted successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to extract dataset payload: {e}")
            
    print("\n Initializing components...")
    
    # 1. Autocomplete System
    try:
        lexicon_path = os.path.join(base_dir, 'data', 'index', 'lexicon_complete.json')
        if os.path.exists(lexicon_path):
            from autocomplete import autocomplete_engine
            initialize_autocomplete(lexicon_path)
            print("[OK] Autocomplete engine initialized")
        else:
            print("[WARNING] Lexicon not found, autocomplete disabled")
    except Exception as e:
        print(f"[WARNING] Autocomplete initialization failed: {e}")
    
    # 2. Semantic Search
    try:
        from semantic_search import semantic_engine
        initialize_semantic_search()  # Loads Word2Vec or custom synonyms
        print("[OK] Semantic search initialized with Word2Vec embeddings")
    except Exception as e:
        print(f"[WARNING] Semantic search initialization failed: {e}")
    
    # 3. Dynamic Indexer
    try:
        index_dir_path = os.path.join(base_dir, 'data', 'index')
        # Use a local reference to avoid global keyword issue at module level
        _di = DynamicIndexer(data_dir=index_dir_path)
        # Update the module-level variable via globals()
        globals()['dynamic_indexer'] = _di
        print(f"[OK] Dynamic indexer initialized ({_di.get_stats()['total_terms']:,} terms)")
    except Exception as e:
        print(f"[WARNING] Dynamic indexer initialization failed: {e}")
        
# Run initialization automatically for WSGI environments (like gunicorn)
init_advanced_components()

if __name__ == '__main__':
    # Check if required files exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_files = {
        os.path.join(base_dir, 'data', 'raw', 'players_22.csv'): 'FIFA 22 dataset',
        os.path.join(base_dir, 'data', 'raw', 'search_engine_dataset.jsonl'): 'Transfermarkt text dataset', 
        os.path.join(base_dir, 'data', 'raw', 'player_mapping_enhanced.json'): 'Player mapping'
    }
    
    missing_files = []
    for file, description in required_files.items():
        if not os.path.exists(file):
            missing_files.append(f"{file} ({description})")
    
    if missing_files:
        print("[ERROR] Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("=" * 60)
        print("[STARTUP] STARTING ENHANCED SCOUTSEARCH SERVER")
        print("=" * 60)
        
        # 4. Performance Monitor (already initialized globally)
        print("[OK] Performance monitor active")
        
        print("\n Components loaded:")
        print("   - FIFA 22 dataset")
        print("   - Text search engine with barrel system")
        print("   - Autocomplete with Trie")
        print("   - Semantic search")
        print("   - Performance monitoring")
        
        print("\n Server running at: http://localhost:5000")
        print("=" * 60)
        print("\n Available endpoints:")
        print("   GET  /                    - Main frontend")
        print("   POST /api/search          - Attribute search")
        print("   POST /api/text-search     - Text search with barrels & semantic")
        print("   POST /api/player/add      - Add new player (DYNAMIC INDEXING)")
        print("   GET  /api/players/top     - Get top players")
        print("   GET  /api/autocomplete    - Autocomplete suggestions")
        print("   POST /api/semantic/expand - Semantic query expansion")
        print("   GET  /api/performance     - System performance stats")
        print("   GET  /api/stats           - System statistics")
        print("   GET  /api/player/<id>     - Get player details")
        print("   GET  /api/debug/*         - Debug endpoints")
        print("")
        
        port = int(os.environ.get('PORT', 8000))
        app.run(debug=False, host='0.0.0.0', port=port)
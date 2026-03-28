"""
Dynamic Document Indexer
Handles real-time indexing of new documents without blocking searches
"""

import json
import os
import math
import time
from threading import Thread, Lock
from collections import defaultdict
import re

class DynamicIndexer:
    def __init__(self, data_dir='data/index'):
        self.data_dir = data_dir
        self.lexicon_path = os.path.join(data_dir, 'lexicon_complete.json')
        self.forward_index_path = os.path.join(data_dir, 'forward_index_termid.json')
        self.term_to_barrel_path = os.path.join(data_dir, 'term_to_barrel_map.json')
        self.barrels_dir = os.path.join(data_dir, 'barrels')
        
        # Thread-safe locks
        self.lexicon_lock = Lock()
        self.forward_lock = Lock()
        self.barrel_lock = Lock()
        
        # In-memory caches
        self.lexicon = {}
        self.term_to_barrel = {}
        self.next_term_id = 1
        self.next_doc_id = 1
        
        self._load_indices()
    
    def _load_indices(self):
        """Load existing indices into memory"""
        # Load lexicon
        if os.path.exists(self.lexicon_path):
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                lexicon_list = json.load(f)
                for item in lexicon_list:
                    # Support both 'term' and 'token' key names for compatibility
                    term_key = 'term' if 'term' in item else 'token'
                    term = item[term_key]
                    self.lexicon[term] = {
                        'term_id': item['term_id'],
                        'doc_freq': item.get('doc_freq', item.get('df', 0))
                    }
                if lexicon_list:
                    self.next_term_id = max(item['term_id'] for item in lexicon_list) + 1
        
        # Load term to barrel mapping
        if os.path.exists(self.term_to_barrel_path):
            with open(self.term_to_barrel_path, 'r', encoding='utf-8') as f:
                self.term_to_barrel = json.load(f)
        
        # Determine next document ID from forward index without loading the massive 100MB JSON into memory
        # to prevent OOM errors on Render
        if os.path.exists(self.forward_index_path):
            self.next_doc_id = 500000 + int(time.time()) % 100000 
    
    def tokenize(self, text):
        """Tokenize text into terms"""
        if not text:
            return []
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', ' ', text)
        tokens = text.split()
        # Remove short tokens
        tokens = [t for t in tokens if len(t) > 1]
        return tokens
    
    def add_document(self, doc_content, doc_metadata=None):
        """
        Add a new document to all indices
        Returns: (doc_id, success, message)
        """
        start_time = time.time()
        
        try:
            # Generate unique document ID
            doc_id = self.next_doc_id
            self.next_doc_id += 1
            
            # Tokenize document
            tokens = self.tokenize(doc_content)
            
            if not tokens:
                return None, False, "Document contains no valid tokens"
            
            # Count term frequencies
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            
            # Update lexicon and get term IDs
            term_ids = []
            new_terms = []
            
            with self.lexicon_lock:
                for term, freq in term_freq.items():
                    if term not in self.lexicon:
                        # Add new term to lexicon
                        term_id = self.next_term_id
                        self.next_term_id += 1
                        self.lexicon[term] = {
                            'term_id': term_id,
                            'doc_freq': 1
                        }
                        new_terms.append({
                            'term': term,
                            'term_id': term_id,
                            'doc_freq': 1
                        })
                        
                        # Assign to barrel (round-robin across 26 barrels)
                        barrel_id = f"barrel_{term_id % 26:03d}"
                        self.term_to_barrel[str(term_id)] = barrel_id
                    else:
                        # Update document frequency
                        self.lexicon[term]['doc_freq'] += 1
                        term_id = self.lexicon[term]['term_id']
                    
                    term_ids.append((term_id, freq))
            
            # Update forward index
            with self.forward_lock:
                self._update_forward_index(doc_id, [tid for tid, _ in term_ids])
            
            # Update inverted index (barrels)
            with self.barrel_lock:
                self._update_inverted_index(doc_id, term_ids, len(tokens))
            
            # Save updated indices to disk (in background)
            Thread(target=self._save_indices_async).start()
            
            elapsed = (time.time() - start_time) * 1000
            message = f"Document {doc_id} indexed in {elapsed:.2f}ms with {len(tokens)} tokens"
            
            return doc_id, True, message
            
        except Exception as e:
            return None, False, f"Indexing failed: {str(e)}"
    
    def _update_forward_index(self, doc_id, term_ids):
        """Update forward index file (async to avoid blocking)"""
        def _save_forward_index_async():
            try:
                # Load existing forward index
                forward_index = []
                if os.path.exists(self.forward_index_path):
                    with open(self.forward_index_path, 'r', encoding='utf-8') as f:
                        forward_index = json.load(f)
                
                # Convert list format to dict if needed (for compatibility)
                if isinstance(forward_index, list):
                    # Keep as list and append new entry
                    new_entry = {
                        'player_id': doc_id,
                        'terms': [{'term_id': tid, 'tf': 1, 'positions': []} for tid in term_ids],
                        'total_terms': len(term_ids),
                        'unique_terms': len(set(term_ids))
                    }
                    forward_index.append(new_entry)
                else:
                    # Dict format - maintain compatibility
                    forward_index[str(doc_id)] = term_ids
                
                # Save back to file (without indent to save faster)
                with open(self.forward_index_path, 'w', encoding='utf-8') as f:
                    json.dump(forward_index, f)
                    
            except Exception as e:
                print(f"Error updating forward index: {e}")
        
        # Run in background thread to avoid blocking server
        Thread(target=_save_forward_index_async).start()
    
    def _update_inverted_index(self, doc_id, term_freq_list, total_terms):
        """Update inverted index (barrels)"""
        try:
            # Group terms by barrel
            barrel_updates = defaultdict(dict)
            
            for term_id, freq in term_freq_list:
                barrel_id = self.term_to_barrel.get(str(term_id))
                if not barrel_id:
                    continue
                
                # Calculate TF-IDF (simplified - IDF will be updated during search)
                tf = 1 + math.log10(freq) if freq > 0 else 0
                score = tf  # IDF will be calculated during search
                
                barrel_updates[barrel_id][str(term_id)] = {
                    'doc_id': doc_id,
                    'score': score
                }
            
            # Update each affected barrel
            for barrel_id, updates in barrel_updates.items():
                barrel_path = os.path.join(self.barrels_dir, f"{barrel_id}.json")
                
                # Load barrel
                barrel_data = {}
                if os.path.exists(barrel_path):
                    with open(barrel_path, 'r', encoding='utf-8') as f:
                        barrel_data = json.load(f)
                
                # Update postings for each term
                for term_id, posting in updates.items():
                    if term_id not in barrel_data:
                        barrel_data[term_id] = []
                    
                    # Append new posting
                    barrel_data[term_id].append([posting['doc_id'], posting['score']])
                
                # Save updated barrel
                with open(barrel_path, 'w', encoding='utf-8') as f:
                    json.dump(barrel_data, f)
                    
        except Exception as e:
            print(f"Error updating inverted index: {e}")
    
    def _save_indices_async(self):
        """Save lexicon and mappings to disk (non-blocking)"""
        try:
            # Save lexicon
            with self.lexicon_lock:
                lexicon_list = [
                    {
                        'token': term,  # Use 'token' key to match existing format
                        'term_id': data['term_id'],
                        'df': data['doc_freq']  # Use 'df' key to match existing format
                    }
                    for term, data in self.lexicon.items()
                ]
                
                with open(self.lexicon_path, 'w', encoding='utf-8') as f:
                    json.dump(lexicon_list, f, indent=2)
            
            # Save term to barrel mapping
            with self.barrel_lock:
                with open(self.term_to_barrel_path, 'w', encoding='utf-8') as f:
                    json.dump(self.term_to_barrel, f, indent=2)
                    
        except Exception as e:
            print(f"Error saving indices: {e}")
    
    def get_stats(self):
        """Get indexing statistics"""
        return {
            'total_terms': len(self.lexicon),
            'next_term_id': self.next_term_id,
            'next_doc_id': self.next_doc_id,
            'barrels_count': 26
        }

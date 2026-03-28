"""
Barrel Manager: Efficient on-demand loading of inverted index barrels
Implements scalable search by loading only required barrel chunks
"""

import json
import os
from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarrelManager:
    """
    Manages barrel-based inverted index loading for scalable search.
    Only loads barrels containing query terms (memory efficient).
    """
    
    def __init__(self, barrel_dir: str, lexicon_path: str):
        """
        Initialize BarrelManager
        
        Args:
            barrel_dir: Directory containing barrel JSON files
            lexicon_path: Path to lexicon file for term lookups
        """
        self.barrel_dir = barrel_dir
        self.lexicon_path = lexicon_path
        
        # Cache loaded barrels
        self.loaded_barrels: Dict[str, dict] = {}
        
        # Term to barrel mapping
        self.term_to_barrel: Dict[str, str] = {}
        
        # Lexicon for term_id lookups
        self.term_to_id: Dict[str, str] = {}
        self.id_to_term: Dict[str, str] = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.barrels_loaded = 0
        
        # Initialize mappings
        self._load_mappings()
        
        logger.info(f"BarrelManager initialized with {len(self.term_to_barrel):,} terms")
    
    def _load_mappings(self):
        """Load term-to-barrel mapping and lexicon"""
        try:
            # Load term-to-barrel mapping
            mapping_path = os.path.join(self.barrel_dir, 'term_to_barrel_map.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.term_to_barrel = json.load(f)
                logger.info(f"Loaded term-to-barrel mapping: {len(self.term_to_barrel):,} entries")
            else:
                logger.warning(f"Term-to-barrel mapping not found at {mapping_path}")
            
            # Load lexicon for term lookups
            if os.path.exists(self.lexicon_path):
                with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                    lexicon = json.load(f)
                    
                for entry in lexicon:
                    term_id = str(entry['term_id'])
                    token = entry['token']
                    self.term_to_id[token] = term_id
                    self.id_to_term[term_id] = token
                
                logger.info(f"Loaded lexicon: {len(self.term_to_id):,} tokens")
            else:
                logger.warning(f"Lexicon not found at {self.lexicon_path}")
                
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
    
    def get_term_id(self, token: str) -> Optional[str]:
        """Convert token to term_id"""
        return self.term_to_id.get(token.lower())
    
    def get_barrel_for_term(self, term_id: str) -> Optional[str]:
        """Get barrel name for a term_id"""
        return self.term_to_barrel.get(str(term_id))
    
    def load_barrel(self, barrel_name: str) -> dict:
        """
        Load a specific barrel from disk
        
        Args:
            barrel_name: Name of barrel (e.g., 'barrel_000')
            
        Returns:
            Dictionary containing barrel's inverted index
        """
        if barrel_name in self.loaded_barrels:
            self.cache_hits += 1
            return self.loaded_barrels[barrel_name]
        
        # Cache miss - load from disk
        self.cache_misses += 1
        barrel_path = os.path.join(self.barrel_dir, f"{barrel_name}.json")
        
        try:
            with open(barrel_path, 'r', encoding='utf-8') as f:
                barrel_data = json.load(f)
                self.loaded_barrels[barrel_name] = barrel_data.get('inverted_index', {})
                self.barrels_loaded += 1
                logger.debug(f"Loaded barrel: {barrel_name} ({len(self.loaded_barrels[barrel_name])} terms)")
                return self.loaded_barrels[barrel_name]
        except FileNotFoundError:
            logger.error(f"Barrel not found: {barrel_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading barrel {barrel_name}: {e}")
            return {}
    
    def get_postings(self, token: str) -> Dict[str, any]:
        """
        Get postings list for a token (loads barrel on-demand)
        
        Args:
            token: Search term (will be converted to term_id)
            
        Returns:
            Postings dictionary: {doc_id: {tf, positions, ...}}
        """
        # Convert token to term_id
        term_id = self.get_term_id(token)
        if not term_id:
            logger.debug(f"Token '{token}' not in lexicon")
            return {}
        
        # Find which barrel contains this term
        barrel_name = self.get_barrel_for_term(term_id)
        if not barrel_name:
            logger.debug(f"No barrel found for term_id {term_id}")
            return {}
        
        # Load barrel and get postings
        barrel = self.load_barrel(barrel_name)
        term_data = barrel.get(term_id, {})
        
        if isinstance(term_data, dict):
            return term_data.get('postings', {})
        return {}
    
    def get_postings_batch(self, tokens: List[str]) -> Dict[str, Dict[str, any]]:
        """
        Efficiently get postings for multiple tokens
        Loads all required barrels in batch
        
        Args:
            tokens: List of search terms
            
        Returns:
            Dictionary mapping tokens to their postings
        """
        results = {}
        
        # Group tokens by barrel to minimize loads
        barrel_groups = defaultdict(list)
        
        for token in tokens:
            term_id = self.get_term_id(token)
            if term_id:
                barrel_name = self.get_barrel_for_term(term_id)
                if barrel_name:
                    barrel_groups[barrel_name].append((token, term_id))
        
        # Load barrels and extract postings
        for barrel_name, term_list in barrel_groups.items():
            barrel = self.load_barrel(barrel_name)
            
            for token, term_id in term_list:
                term_data = barrel.get(term_id, {})
                if isinstance(term_data, dict):
                    results[token] = term_data.get('postings', {})
                else:
                    results[token] = {}
        
        return results
    
    def get_term_df(self, token: str) -> int:
        """Get document frequency for a term"""
        term_id = self.get_term_id(token)
        if not term_id:
            return 0
        
        barrel_name = self.get_barrel_for_term(term_id)
        if not barrel_name:
            return 0
        
        barrel = self.load_barrel(barrel_name)
        term_data = barrel.get(term_id, {})
        
        if isinstance(term_data, dict):
            return term_data.get('df', 0)
        return 0
    
    def clear_cache(self):
        """Clear loaded barrels from memory"""
        count = len(self.loaded_barrels)
        self.loaded_barrels.clear()
        logger.info(f"Cleared {count} barrels from cache")
    
    def get_statistics(self) -> dict:
        """Get barrel manager statistics"""
        return {
            'total_terms': len(self.term_to_barrel),
            'loaded_barrels': len(self.loaded_barrels),
            'total_barrels': len(set(self.term_to_barrel.values())),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses) 
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            'memory_efficiency': (
                f"{len(self.loaded_barrels)}/{len(set(self.term_to_barrel.values()))}"
            )
        }
    
    def preload_common_terms(self, common_terms: List[str]):
        """
        Preload barrels containing common search terms
        Useful for warming up the cache
        """
        logger.info(f"Preloading barrels for {len(common_terms)} common terms...")
        
        barrels_to_load = set()
        for term in common_terms:
            term_id = self.get_term_id(term)
            if term_id:
                barrel_name = self.get_barrel_for_term(term_id)
                if barrel_name:
                    barrels_to_load.add(barrel_name)
        
        for barrel_name in barrels_to_load:
            self.load_barrel(barrel_name)
        
        logger.info(f"Preloaded {len(barrels_to_load)} barrels")    
    def reload_mappings(self):
        """Reload mappings from disk (for dynamic updates)"""
        logger.info("Reloading barrel mappings...")
        self.term_to_barrel.clear()
        self.term_to_id.clear()
        self.id_to_term.clear()
        self._load_mappings()
        logger.info("Mappings reloaded successfully")
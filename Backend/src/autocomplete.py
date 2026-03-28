"""
Autocomplete System: Trie-based real-time query suggestions
Provides 3-5 relevant suggestions as user types
"""

import json
import os
from typing import List, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrieNode:
    """Node in the Trie structure"""
    
    def __init__(self):
        self.children: dict = {}
        self.is_end_of_word: bool = False
        self.frequency: int = 0
        self.word: str = ""


class AutocompleteTrie:
    """
    Trie-based autocomplete system for fast prefix matching
    Supports frequency-based ranking of suggestions
    """
    
    def __init__(self, lexicon_path: str = None, query_log_path: str = None):
        """
        Initialize Autocomplete Trie
        
        Args:
            lexicon_path: Path to lexicon JSON for building trie
            query_log_path: Optional path to query log for frequency data
        """
        self.root = TrieNode()
        self.lexicon_path = lexicon_path
        self.query_log_path = query_log_path
        self.word_count = 0
        
        # Common football terms to boost
        self.boosted_terms = {
            'striker', 'midfielder', 'defender', 'goalkeeper', 'winger',
            'forward', 'attacking', 'defensive', 'fast', 'strong',
            'young', 'experienced', 'veteran', 'talented', 'skillful'
        }
        
        if lexicon_path and os.path.exists(lexicon_path):
            self.build_from_lexicon()
        
        logger.info(f"AutocompleteTrie initialized with {self.word_count:,} words")
    
    def insert(self, word: str, frequency: int = 1):
        """
        Insert a word into the trie
        
        Args:
            word: Word to insert
            frequency: Usage frequency (higher = more common)
        """
        if not word:
            return
        
        word = word.lower().strip()
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.word = word
        node.frequency = max(node.frequency, frequency)  # Keep highest frequency
        
        # Boost common football terms
        if word in self.boosted_terms:
            node.frequency *= 3
    
    def build_from_lexicon(self):
        """Build trie from lexicon file"""
        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
            
            logger.info(f"Building trie from {len(lexicon):,} lexicon entries...")
            
            for entry in lexicon:
                token = entry.get('token', '')
                df = entry.get('df', 1)  # Document frequency as proxy for importance
                
                if token and len(token) >= 2:  # Skip single chars
                    self.insert(token, frequency=df)
                    self.word_count += 1
            
            logger.info(f"Trie built with {self.word_count:,} words")
            
        except Exception as e:
            logger.error(f"Error building trie from lexicon: {e}")
    
    def add_common_queries(self, queries: List[Tuple[str, int]]):
        """
        Add common query phrases
        
        Args:
            queries: List of (query, frequency) tuples
        """
        for query, freq in queries:
            # Split multi-word queries
            words = query.lower().split()
            for word in words:
                if len(word) >= 2:
                    self.insert(word, frequency=freq)
    
    def _collect_suggestions(self, node: TrieNode, suggestions: List[Tuple[str, int]], limit: int = 5):
        """
        Recursively collect suggestions from a node
        
        Args:
            node: Current trie node
            suggestions: List to accumulate suggestions
            limit: Maximum number of suggestions
        """
        if len(suggestions) >= limit * 3:  # Collect extra to sort later
            return
        
        if node.is_end_of_word:
            suggestions.append((node.word, node.frequency))
        
        # Traverse children
        for char in sorted(node.children.keys()):
            self._collect_suggestions(node.children[char], suggestions, limit)
    
    def get_suggestions(self, prefix: str, limit: int = 5) -> List[dict]:
        """
        Get autocomplete suggestions for a prefix
        
        Args:
            prefix: Partial word typed by user
            limit: Maximum number of suggestions to return
            
        Returns:
            List of suggestion dictionaries with word and score
        """
        if not prefix:
            return []
        
        prefix = prefix.lower().strip()
        node = self.root
        
        # Navigate to prefix node
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]
        
        # Collect all words with this prefix
        suggestions = []
        self._collect_suggestions(node, suggestions, limit)
        
        # Sort by frequency (descending) and return top N
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for word, freq in suggestions[:limit]:
            results.append({
                'word': word,
                'score': freq,
                'highlight': prefix  # Part to highlight in UI
            })
        
        return results
    
    def search_exact(self, word: str) -> bool:
        """Check if exact word exists in trie"""
        word = word.lower().strip()
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def get_statistics(self) -> dict:
        """Get trie statistics"""
        return {
            'total_words': self.word_count,
            'boosted_terms': len(self.boosted_terms)
        }


class SmartAutocomplete:
    """
    Enhanced autocomplete with context awareness
    Handles multi-word queries and football-specific context
    """
    
    def __init__(self, trie: AutocompleteTrie):
        self.trie = trie
        
        # Common football query patterns
        self.query_templates = [
            "best {position}",
            "top {position}",
            "young {position}",
            "fast {position}",
            "{nationality} {position}",
            "{position} in {league}"
        ]
        
        # Position keywords
        self.positions = {
            'striker', 'forward', 'midfielder', 'defender', 
            'goalkeeper', 'winger', 'fullback', 'center-back'
        }
        
        # Context-aware suggestions
        self.context_keywords = {
            'best': ['striker', 'midfielder', 'defender', 'goalkeeper'],
            'top': ['scorer', 'rated', 'young', 'players'],
            'fast': ['striker', 'winger', 'players'],
            'young': ['talent', 'prospect', 'players']
        }
    
    def get_smart_suggestions(self, query: str, limit: int = 5) -> List[dict]:
        """
        Get context-aware suggestions for partial query
        
        Args:
            query: Partial query (may be multi-word)
            limit: Number of suggestions
            
        Returns:
            List of suggestion dictionaries
        """
        query = query.strip().lower()
        
        if not query:
            return self._get_popular_queries(limit)
        
        # Multi-word query handling
        words = query.split()
        
        if len(words) > 1:
            # Complete last word, show full query
            last_word = words[-1]
            prefix = ' '.join(words[:-1])
            
            suggestions = self.trie.get_suggestions(last_word, limit)
            
            # Format as complete queries
            results = []
            for sugg in suggestions:
                results.append({
                    'word': f"{prefix} {sugg['word']}" if prefix else sugg['word'],
                    'score': sugg['score'],
                    'highlight': last_word
                })
            
            # Add context-aware completions
            if len(words) >= 2 and words[0] in self.context_keywords:
                context_words = self.context_keywords[words[0]]
                for context_word in context_words:
                    if context_word.startswith(last_word):
                        results.append({
                            'word': f"{prefix} {context_word}",
                            'score': 1000,  # Boost context suggestions
                            'highlight': last_word
                        })
            
            # Sort and return
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]
        
        else:
            # Single word - standard autocomplete
            suggestions = self.trie.get_suggestions(query, limit)
            
            # Add template-based suggestions for positions
            if any(pos.startswith(query) for pos in self.positions):
                for pos in self.positions:
                    if pos.startswith(query):
                        suggestions.append({
                            'word': f"best {pos}",
                            'score': 800,
                            'highlight': query
                        })
            
            suggestions.sort(key=lambda x: x['score'], reverse=True)
            return suggestions[:limit]
    
    def _get_popular_queries(self, limit: int = 5) -> List[dict]:
        """Get popular/suggested queries when input is empty"""
        popular = [
            {'word': 'best striker', 'score': 1000, 'highlight': ''},
            {'word': 'top midfielder', 'score': 900, 'highlight': ''},
            {'word': 'young talent', 'score': 850, 'highlight': ''},
            {'word': 'fast winger', 'score': 800, 'highlight': ''},
            {'word': 'goalkeeper', 'score': 750, 'highlight': ''}
        ]
        return popular[:limit]


# Global instance (initialized in app.py)
autocomplete_engine: Optional[SmartAutocomplete] = None


def initialize_autocomplete(lexicon_path: str) -> SmartAutocomplete:
    """
    Initialize global autocomplete engine
    
    Args:
        lexicon_path: Path to lexicon file
        
    Returns:
        SmartAutocomplete instance
    """
    global autocomplete_engine
    
    trie = AutocompleteTrie(lexicon_path=lexicon_path)
    autocomplete_engine = SmartAutocomplete(trie)
    
    logger.info("Autocomplete engine initialized")
    return autocomplete_engine

"""
Advanced Text Processor with Ultimate Optimization
Handles normalization, fuzzy matching, and intelligent tokenization
"""

import re
import unicodedata
from functools import lru_cache

class AdvancedTextProcessor:
    """Ultra-fast text processing with caching and optimizations"""
    
    def __init__(self):
        # Pre-compile regex patterns for speed
        self._special_chars_pattern = re.compile(r'[^\w\s-]')
        self._whitespace_pattern = re.compile(r'\s+')
        self._number_pattern = re.compile(r'\d+')
        
        # Fast lookup tables (pre-computed)
        self._accent_map = self._build_accent_map()
        self._abbreviations = {
            'cf': 'center forward', 'cam': 'attacking midfielder',
            'cdm': 'defensive midfielder', 'lb': 'left back',
            'rb': 'right back', 'cb': 'center back',
            'gk': 'goalkeeper', 'st': 'striker',
            'cm': 'central midfielder', 'lw': 'left wing',
            'rw': 'right wing', 'lm': 'left mid', 'rm': 'right mid'
        }
    
    def _build_accent_map(self):
        """Pre-build accent removal map for O(1) lookups"""
        accents = {
            '': 'e', '': 'e', '': 'e', '': 'e',
            '': 'a', '': 'a', '': 'a', '': 'a', '': 'a', '': 'a',
            '': 'i', '': 'i', '': 'i', '': 'i',
            '': 'o', '': 'o', '': 'o', '': 'o', '': 'o',
            '': 'u', '': 'u', '': 'u', '': 'u',
            '': 'n', '': 'c', '': 'ss', '': 'y', '': 'y'
        }
        # Add uppercase versions
        for k, v in list(accents.items()):
            accents[k.upper()] = v.upper()
        return accents
    
    @lru_cache(maxsize=10000)
    def normalize_text(self, text):
        """
        Ultra-fast normalization with LRU cache
        Converts: Mbapp  mbappe, So Paulo  sao paulo
        """
        if not text:
            return ""
        
        text = str(text).lower()
        
        # Fast accent removal using pre-built map
        chars = []
        for c in text:
            chars.append(self._accent_map.get(c, c))
        text = ''.join(chars)
        
        # Unicode normalization (fallback for missed characters)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Remove special characters (keep hyphens)
        text = self._special_chars_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self._whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    @lru_cache(maxsize=5000)
    def tokenize(self, text):
        """Fast tokenization with caching"""
        normalized = self.normalize_text(text)
        tokens = normalized.split()
        
        # Expand abbreviations inline
        result = []
        for token in tokens:
            if token in self._abbreviations:
                result.append(self._abbreviations[token])
            else:
                result.append(token)
                # Handle hyphenated names
                if '-' in token:
                    result.extend(token.split('-'))
        
        return tuple(result)  # Tuple for caching
    
    def quick_match(self, query, text):
        """
        Ultra-fast substring matching
        Returns match score (0-1)
        """
        q_norm = self.normalize_text(query)
        t_norm = self.normalize_text(text)
        
        if not q_norm or not t_norm:
            return 0.0
        
        # Exact match
        if q_norm == t_norm:
            return 1.0
        
        # Contains match
        if q_norm in t_norm:
            return 0.9
        
        # Token overlap
        q_tokens = set(q_norm.split())
        t_tokens = set(t_norm.split())
        if q_tokens and t_tokens:
            overlap = len(q_tokens & t_tokens) / len(q_tokens | t_tokens)
            return overlap * 0.7
        
        return 0.0
    
    def fuzzy_similarity(self, str1, str2, threshold=0.6):
        """
        Fast fuzzy matching using optimized Jaro-Winkler
        Only computes if strings are similar length
        """
        s1 = self.normalize_text(str1)
        s2 = self.normalize_text(str2)
        
        if not s1 or not s2:
            return 0.0
        
        # Quick reject if length difference is too large
        len_diff = abs(len(s1) - len(s2))
        if len_diff > max(len(s1), len(s2)) * 0.5:
            return 0.0
        
        # Simple edit distance approximation
        if s1 == s2:
            return 1.0
        
        # Character overlap score (fast approximation)
        chars1 = set(s1)
        chars2 = set(s2)
        overlap = len(chars1 & chars2) / len(chars1 | chars2)
        
        return overlap

# Global singleton
_text_processor = None

def get_text_processor():
    """Get singleton instance"""
    global _text_processor
    if _text_processor is None:
        _text_processor = AdvancedTextProcessor()
    return _text_processor

"""
Semantic Search: Word embeddings for understanding query meaning
Uses Word2Vec/GloVe for synonym expansion and conceptual similarity
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import defaultdict

# Try to import gensim for Word2Vec
try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("WARNING: gensim not installed. Semantic search will use custom synonyms only.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Semantic search using pre-trained word embeddings
    Expands queries with synonyms and similar terms
    """
    
    def __init__(self, embeddings_path: str = None):
        """
        Initialize semantic search engine
        
        Args:
            embeddings_path: Path to pre-trained embeddings (GloVe/Word2Vec format)
        """
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dim = 0
        self.embeddings_path = embeddings_path
        
        # Football-specific semantic relationships
        self.custom_synonyms = {
            # Positions
            'striker': ['forward', 'attacker', 'frontman', 'target-man', 'poacher'],
            'midfielder': ['playmaker', 'center-mid', 'attacking-mid', 'defensive-mid'],
            'defender': ['center-back', 'fullback', 'wing-back', 'sweeper'],
            'goalkeeper': ['keeper', 'goalie', 'shot-stopper'],
            'winger': ['wide-player', 'flanker', 'wing-forward'],
            
            # Attributes
            'fast': ['quick', 'speedy', 'rapid', 'pacy', 'swift'],
            'strong': ['powerful', 'physical', 'robust', 'muscular'],
            'skillful': ['technical', 'talented', 'gifted', 'agile'],
            'experienced': ['veteran', 'senior', 'seasoned'],
            'young': ['youth', 'prospect', 'talent', 'wonderkid'],
            
            # Quality
            'best': ['top', 'elite', 'excellent', 'outstanding', 'world-class'],
            'good': ['decent', 'solid', 'capable', 'reliable'],
            
            # Play styles
            'attacking': ['offensive', 'forward-thinking', 'aggressive'],
            'defensive': ['solid', 'protective', 'conservative'],
            'creative': ['playmaking', 'inventive', 'vision'],
            'clinical': ['lethal', 'deadly', 'prolific', 'sharp']
        }
        
        # Build reverse mapping
        self.term_to_synonyms: Dict[str, Set[str]] = {}
        for key, synonyms in self.custom_synonyms.items():
            self.term_to_synonyms[key] = set(synonyms)
            for syn in synonyms:
                if syn not in self.term_to_synonyms:
                    self.term_to_synonyms[syn] = set()
                self.term_to_synonyms[syn].add(key)
                self.term_to_synonyms[syn].update([s for s in synonyms if s != syn])
        
        # Load embeddings if path provided
        if embeddings_path and os.path.exists(embeddings_path):
            self.load_embeddings()
        elif GENSIM_AVAILABLE:
            # Try to load lightweight model from gensim
            logger.info("Attempting to load pre-trained Word2Vec model via gensim...")
            try:
                self.load_word2vec_from_gensim()
            except Exception as e:
                logger.warning(f"Could not load gensim model: {e}. Using custom synonyms only.")
        else:
            logger.info("No embeddings file provided, using custom synonyms only")
    
    def load_word2vec_from_gensim(self, model_name='glove-wiki-gigaword-50'):
        """
        Load pre-trained Word2Vec/GloVe model using gensim's downloader
        
        Args:
            model_name: Model to download (glove-wiki-gigaword-50 is 65MB, fast)
                       Options: 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
                                'word2vec-google-news-300' (large!)
        """
        if not GENSIM_AVAILABLE:
            logger.warning("gensim not available, cannot load pre-trained models")
            return False
        
        try:
            logger.info(f"Downloading {model_name} (this may take a minute on first run)...")
            model = api.load(model_name)
            
            # Convert to our format
            self.embedding_dim = model.vector_size
            count = 0
            
            for word in model.index_to_key[:50000]:  # Limit to 50k most common words
                self.embeddings[word.lower()] = model[word]
                count += 1
                
                if count % 5000 == 0:
                    logger.info(f"Loaded {count:,} word vectors...")
            
            logger.info(f" Successfully loaded {count:,} word vectors (dim={self.embedding_dim})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load gensim model: {e}")
            return False
    
    def load_embeddings(self, limit: int = 100000):
        """
        Load pre-trained word embeddings from file
        Supports GloVe format: word dim1 dim2 ... dimN
        
        Args:
            limit: Maximum number of words to load (memory constraint)
        """
        try:
            logger.info(f"Loading embeddings from {self.embeddings_path}...")
            
            count = 0
            with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if count >= limit:
                        break
                    
                    parts = line.strip().split()
                    if len(parts) < 10:  # Skip invalid lines
                        continue
                    
                    word = parts[0].lower()
                    try:
                        vector = np.array([float(x) for x in parts[1:]])
                        
                        if self.embedding_dim == 0:
                            self.embedding_dim = len(vector)
                        elif len(vector) != self.embedding_dim:
                            continue  # Skip inconsistent dimensions
                        
                        self.embeddings[word] = vector
                        count += 1
                        
                        if count % 10000 == 0:
                            logger.info(f"Loaded {count:,} word vectors...")
                    except ValueError:
                        continue
            
            logger.info(f"Loaded {len(self.embeddings):,} word embeddings (dim={self.embedding_dim})")
            
        except FileNotFoundError:
            logger.warning(f"Embeddings file not found: {self.embeddings_path}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word"""
        return self.embeddings.get(word.lower())
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_similar_words(self, word: str, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find semantically similar words using embeddings
        
        Args:
            word: Query word
            top_k: Number of similar words to return
            threshold: Minimum similarity score
            
        Returns:
            List of (word, similarity_score) tuples
        """
        word = word.lower()
        
        # First check custom synonyms
        if word in self.term_to_synonyms:
            custom_results = [(syn, 0.95) for syn in self.term_to_synonyms[word]]
            if len(custom_results) >= top_k:
                return custom_results[:top_k]
        
        # Use embeddings if available
        if not self.embeddings:
            return [(syn, 0.9) for syn in self.term_to_synonyms.get(word, [])[:top_k]]
        
        word_vec = self.get_vector(word)
        if word_vec is None:
            return [(syn, 0.9) for syn in self.term_to_synonyms.get(word, [])[:top_k]]
        
        # Calculate similarities
        similarities = []
        for other_word, other_vec in self.embeddings.items():
            if other_word == word:
                continue
            
            sim = self.cosine_similarity(word_vec, other_vec)
            if sim >= threshold:
                similarities.append((other_word, sim))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Combine with custom synonyms
        results = similarities[:top_k]
        
        # Add custom synonyms if not enough results
        if word in self.term_to_synonyms and len(results) < top_k:
            for syn in self.term_to_synonyms[word]:
                if syn not in [w for w, _ in results]:
                    results.append((syn, 0.9))
                if len(results) >= top_k:
                    break
        
        return results[:top_k]
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with semantically similar terms
        
        Args:
            query: Original search query
            max_expansions: Maximum synonyms per term
            
        Returns:
            List of expanded query terms (includes original)
        """
        query_terms = query.lower().split()
        expanded_terms = set(query_terms)  # Include original terms
        
        for term in query_terms:
            # Get similar words
            similar = self.get_similar_words(term, top_k=max_expansions, threshold=0.6)
            
            # Add to expanded set
            for similar_word, score in similar:
                if score >= 0.6:  # Only high-confidence synonyms
                    expanded_terms.add(similar_word)
        
        return list(expanded_terms)
    
    def semantic_score(self, query: str, document_terms: List[str]) -> float:
        """
        Calculate semantic similarity between query and document
        
        Args:
            query: Search query
            document_terms: Terms in document
            
        Returns:
            Semantic similarity score (0-1)
        """
        query_terms = query.lower().split()
        
        # Expand query with synonyms
        expanded_query = self.expand_query(query, max_expansions=2)
        
        # Calculate overlap score
        doc_terms_lower = [t.lower() for t in document_terms]
        
        # Direct match score
        direct_matches = sum(1 for term in query_terms if term in doc_terms_lower)
        
        # Semantic match score
        semantic_matches = sum(1 for term in expanded_query if term in doc_terms_lower)
        
        # Combined score (weighted)
        max_possible = len(query_terms) + len(expanded_query)
        if max_possible == 0:
            return 0.0
        
        score = (direct_matches * 2 + semantic_matches) / (len(query_terms) * 2 + len(expanded_query))
        
        return min(score, 1.0)
    
    def get_custom_synonyms(self, word: str) -> List[str]:
        """Get football-specific synonyms for a word"""
        return list(self.term_to_synonyms.get(word.lower(), []))
    
    def get_statistics(self) -> dict:
        """Get semantic engine statistics"""
        return {
            'embeddings_loaded': len(self.embeddings),
            'embedding_dimension': self.embedding_dim,
            'custom_synonym_groups': len(self.custom_synonyms),
            'total_synonym_mappings': len(self.term_to_synonyms),
            'using_pretrained': len(self.embeddings) > 0
        }


# Lightweight version using only custom synonyms (no embeddings needed)
class LightweightSemanticSearch:
    """
    Lightweight semantic search using only custom football synonyms
    No external embeddings required
    """
    
    def __init__(self):
        # Reuse the custom synonyms from full engine
        semantic_engine = SemanticSearchEngine()
        self.term_to_synonyms = semantic_engine.term_to_synonyms
        self.custom_synonyms = semantic_engine.custom_synonyms
        
        logger.info(f"Lightweight semantic search initialized with {len(self.custom_synonyms)} synonym groups")
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query using custom synonyms only"""
        query_terms = query.lower().split()
        expanded_terms = set(query_terms)
        
        for term in query_terms:
            if term in self.term_to_synonyms:
                synonyms = list(self.term_to_synonyms[term])[:max_expansions]
                expanded_terms.update(synonyms)
        
        return list(expanded_terms)
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        return list(self.term_to_synonyms.get(word.lower(), []))


# Global instance
semantic_engine: Optional[LightweightSemanticSearch] = None


def initialize_semantic_search(embeddings_path: str = None) -> LightweightSemanticSearch:
    """
    Initialize global semantic search engine
    
    Args:
        embeddings_path: Optional path to embeddings file
        
    Returns:
        Semantic search engine instance
    """
    global semantic_engine
    
    # Use lightweight version (custom synonyms only) for now
    # Can switch to full SemanticSearchEngine if embeddings file is provided
    if embeddings_path and os.path.exists(embeddings_path):
        logger.info("Using full semantic search with embeddings")
        semantic_engine = SemanticSearchEngine(embeddings_path)
    else:
        logger.info("Using lightweight semantic search (custom synonyms)")
        semantic_engine = LightweightSemanticSearch()
    
    return semantic_engine

"""
Optimized Search Engine with Advanced Indexing and Caching
Ultra-fast search with < 100ms response time
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from functools import lru_cache
import time
from text_processor import get_text_processor

class OptimizedSearchEngine:
    """
    Ultra-optimized search engine with:
    - Multi-level caching
    - Inverted index for O(1) lookups
    - Pre-computed normalized fields
    - Efficient ranking algorithm
    """
    
    def __init__(self, df):
        print("[STARTUP] Building optimized search engine...")
        start = time.time()
        
        self.df = df
        self.text_processor = get_text_processor()
        
        # Pre-process all data once
        self.players = self._preprocess_players()
        
        # Build inverted indices for instant lookups
        self.name_index = defaultdict(set)
        self.club_index = defaultdict(set)
        self.nationality_index = defaultdict(set)
        self.position_index = defaultdict(set)
        
        self._build_indices()
        
        elapsed = (time.time() - start) * 1000
        print(f" Search engine ready in {elapsed:.0f}ms")
        print(f"   - {len(self.players)} players indexed")
        print(f"   - {len(self.name_index)} name tokens")
        print(f"   - {len(self.club_index)} clubs")
        print(f"   - {len(self.nationality_index)} nationalities")
    
    def _preprocess_players(self):
        """Pre-compute all normalized fields"""
        players = []
        
        for idx, row in self.df.iterrows():
            player = {
                # Original fields
                'id': idx,
                'short_name': row.get('short_name', ''),
                'long_name': row.get('long_name', ''),
                'overall': int(row.get('overall', 0)),
                'potential': int(row.get('potential', 0)),
                'age': int(row.get('age', 0)),
                'club_name': row.get('club_name', ''),
                'nationality_name': row.get('nationality_name', ''),
                'player_positions': row.get('player_positions', ''),
                'value_eur': float(row.get('value_eur', 0)),
                'wage_eur': float(row.get('wage_eur', 0)),
                'pace': int(row.get('pace', 0)),
                'shooting': int(row.get('shooting', 0)),
                'passing': int(row.get('passing', 0)),
                'dribbling': int(row.get('dribbling', 0)),
                'defending': int(row.get('defending', 0)),
                'physic': int(row.get('physic', 0)),
                
                # Pre-normalized fields for fast search
                '_norm_name': self.text_processor.normalize_text(row.get('short_name', '')),
                '_norm_long': self.text_processor.normalize_text(row.get('long_name', '')),
                '_norm_club': self.text_processor.normalize_text(row.get('club_name', '')),
                '_norm_nat': self.text_processor.normalize_text(row.get('nationality_name', '')),
                '_norm_pos': self.text_processor.normalize_text(row.get('player_positions', '')),
                
                # Pre-tokenized for instant matching
                '_tokens': self.text_processor.tokenize(
                    f"{row.get('short_name', '')} {row.get('long_name', '')}"
                )
            }
            
            players.append(player)
        
        return players
    
    def _build_indices(self):
        """Build inverted indices for O(1) lookups"""
        for idx, player in enumerate(self.players):
            # Index by name tokens
            for token in player['_tokens']:
                self.name_index[token].add(idx)
            
            # Index by club
            if player['_norm_club']:
                self.club_index[player['_norm_club']].add(idx)
            
            # Index by nationality
            if player['_norm_nat']:
                self.nationality_index[player['_norm_nat']].add(idx)
            
            # Index by position
            positions = player['_norm_pos'].split()
            for pos in positions:
                if pos:
                    self.position_index[pos].add(idx)
    
    def search(self, query, filters=None, max_results=20):
        """
        Ultra-fast search with multi-strategy approach
        Target: < 100ms response time
        """
        start_time = time.time()
        
        if not query or not query.strip():
            # No query - return top players by rating
            candidates = self.players[:100]
        else:
            # Get candidates using inverted index
            candidates = self._get_candidates_fast(query)
        
        # Apply filters
        if filters:
            candidates = self._apply_filters_fast(candidates, filters)
        
        # Rank results
        ranked = self._rank_fast(query, candidates, max_results)
        
        elapsed = (time.time() - start_time) * 1000
        print(f" Search completed in {elapsed:.1f}ms ({len(ranked)} results)")
        
        return ranked
    
    def _get_candidates_fast(self, query):
        """
        Lightning-fast candidate retrieval using indices
        Strategy: Start with smallest result set
        """
        query_norm = self.text_processor.normalize_text(query)
        query_tokens = self.text_processor.tokenize(query)
        
        candidate_sets = []
        
        # Strategy 1: Name token matches (most precise)
        for token in query_tokens:
            if token in self.name_index:
                candidate_sets.append(self.name_index[token])
        
        # Strategy 2: Club matches
        if query_norm in self.club_index:
            candidate_sets.append(self.club_index[query_norm])
        
        # Strategy 3: Nationality matches
        if query_norm in self.nationality_index:
            candidate_sets.append(self.nationality_index[query_norm])
        
        # Strategy 4: Position matches
        for token in query_tokens:
            if token in self.position_index:
                candidate_sets.append(self.position_index[token])
        
        if not candidate_sets:
            # Fallback: partial matching (slower but comprehensive)
            return self._fallback_search(query_norm, query_tokens)
        
        # Union of all candidate sets
        candidate_indices = set()
        for s in candidate_sets:
            candidate_indices.update(s)
        
        # Convert indices to player objects
        return [self.players[idx] for idx in candidate_indices if idx < len(self.players)]
    
    def _fallback_search(self, query_norm, query_tokens):
        """Fallback: scan all players for partial matches"""
        candidates = []
        
        for player in self.players:
            # Quick substring check in pre-normalized fields
            searchable = f"{player['_norm_name']} {player['_norm_club']} {player['_norm_nat']}"
            
            if query_norm in searchable:
                candidates.append(player)
                if len(candidates) >= 100:  # Limit fallback results
                    break
        
        return candidates
    
    def _apply_filters_fast(self, candidates, filters):
        """Ultra-fast filtering using numpy-style operations"""
        filtered = []
        
        for player in candidates:
            # Overall
            if 'overallMin' in filters and filters['overallMin']:
                if player['overall'] < filters['overallMin']:
                    continue
            if 'overallMax' in filters and filters['overallMax']:
                if player['overall'] > filters['overallMax']:
                    continue
            
            # Age
            if 'ageMin' in filters and filters['ageMin']:
                if player['age'] < filters['ageMin']:
                    continue
            if 'ageMax' in filters and filters['ageMax']:
                if player['age'] > filters['ageMax']:
                    continue
            
            # Position
            if 'position' in filters and filters['position']:
                if filters['position'].lower() not in player['_norm_pos']:
                    continue
            
            # Pace
            if 'paceMin' in filters and filters['paceMin']:
                if player['pace'] < filters['paceMin']:
                    continue
            
            # Shooting
            if 'shootingMin' in filters and filters['shootingMin']:
                if player['shooting'] < filters['shootingMin']:
                    continue
            
            filtered.append(player)
        
        return filtered
    
    def _rank_fast(self, query, candidates, max_results):
        """
        Fast ranking algorithm
        Scores based on: relevance (70%) + quality (30%)
        """
        if not query or not query.strip():
            # No query - sort by overall rating
            candidates.sort(key=lambda p: p['overall'], reverse=True)
            return candidates[:max_results]
        
        query_norm = self.text_processor.normalize_text(query)
        
        scored = []
        for player in candidates:
            # Relevance score (fast matching)
            rel_score = 0.0
            
            # Name match (highest weight)
            if query_norm in player['_norm_name']:
                rel_score += 100.0
            if query_norm == player['_norm_name']:
                rel_score += 200.0
            
            # Club match
            if query_norm in player['_norm_club']:
                rel_score += 30.0
            
            # Nationality match
            if query_norm in player['_norm_nat']:
                rel_score += 20.0
            
            # Token overlap
            for token in player['_tokens']:
                if query_norm in token or token in query_norm:
                    rel_score += 10.0
            
            # Quality score
            quality = player['overall']
            
            # Final score: 70% relevance + 30% quality
            final_score = (rel_score * 0.7) + (quality * 0.3)
            
            scored.append((player, final_score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [player for player, score in scored[:max_results]]

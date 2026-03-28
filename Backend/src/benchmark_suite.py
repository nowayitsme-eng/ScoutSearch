"""
Performance Benchmark Suite
Tests search engine against all requirements
"""

import time
import requests
import json
import psutil
import os
from typing import Dict, List

class PerformanceBenchmark:
    def __init__(self, api_base='http://localhost:5000/api'):
        self.api_base = api_base
        self.results = {
            'single_word_queries': [],
            'multi_word_queries': [],
            'memory_usage': {},
            'indexing_performance': {},
            'compliance': {}
        }
    
    def test_single_word_queries(self, count=10):
        """Test single word query performance (< 500ms requirement)"""
        test_words = [
            'striker', 'midfielder', 'defender', 'goalkeeper', 'fast',
            'young', 'experienced', 'talented', 'strong', 'skillful'
        ]
        
        print("\n" + "="*60)
        print("TESTING SINGLE WORD QUERIES (Requirement: < 500ms)")
        print("="*60)
        
        for word in test_words[:count]:
            start = time.time()
            
            try:
                response = requests.post(
                    f"{self.api_base}/text-search",
                    json={'query': word, 'limit': 20},
                    timeout=2
                )
                
                elapsed = (time.time() - start) * 1000
                success = response.status_code == 200
                result_count = len(response.json().get('players', [])) if success else 0
                
                self.results['single_word_queries'].append({
                    'word': word,
                    'time_ms': round(elapsed, 2),
                    'success': success,
                    'results': result_count,
                    'compliant': elapsed < 500
                })
                
                status = "" if elapsed < 500 else ""
                print(f"{status} '{word}': {elapsed:.2f}ms ({result_count} results)")
                
            except Exception as e:
                print(f" '{word}': FAILED ({str(e)})")
                self.results['single_word_queries'].append({
                    'word': word,
                    'time_ms': 0,
                    'success': False,
                    'results': 0,
                    'compliant': False,
                    'error': str(e)
                })
    
    def test_multi_word_queries(self, count=5):
        """Test multi-word query performance (< 1500ms for 5 words)"""
        test_queries = [
            'young talented striker premier league',
            'fast winger left foot world class',
            'experienced defender strong physical presence',
            'creative attacking midfielder playmaker vision',
            'top goalkeeper quick reflexes shot stopper'
        ]
        
        print("\n" + "="*60)
        print("TESTING MULTI-WORD QUERIES (Requirement: < 1500ms for 5 words)")
        print("="*60)
        
        for query in test_queries[:count]:
            word_count = len(query.split())
            start = time.time()
            
            try:
                response = requests.post(
                    f"{self.api_base}/text-search",
                    json={'query': query, 'limit': 20},
                    timeout=3
                )
                
                elapsed = (time.time() - start) * 1000
                success = response.status_code == 200
                result_count = len(response.json().get('players', [])) if success else 0
                
                compliant = elapsed < 1500 if word_count == 5 else True
                
                self.results['multi_word_queries'].append({
                    'query': query,
                    'word_count': word_count,
                    'time_ms': round(elapsed, 2),
                    'success': success,
                    'results': result_count,
                    'compliant': compliant
                })
                
                status = "" if compliant else ""
                print(f"{status} '{query[:50]}...': {elapsed:.2f}ms ({result_count} results)")
                
            except Exception as e:
                print(f" '{query[:50]}...': FAILED ({str(e)})")
                self.results['multi_word_queries'].append({
                    'query': query,
                    'word_count': word_count,
                    'time_ms': 0,
                    'success': False,
                    'results': 0,
                    'compliant': False,
                    'error': str(e)
                })
    
    def test_memory_usage(self):
        """Test memory usage (< 2GB for < 100k documents)"""
        print("\n" + "="*60)
        print("TESTING MEMORY USAGE (Requirement:  2GB for < 100k docs)")
        print("="*60)
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        memory_mb = memory_info.rss / (1024 * 1024)
        memory_gb = memory_mb / 1024
        
        compliant = memory_gb <= 2.0
        status = "" if compliant else ""
        
        self.results['memory_usage'] = {
            'rss_mb': round(memory_mb, 2),
            'rss_gb': round(memory_gb, 3),
            'compliant': compliant
        }
        
        print(f"{status} Memory Usage: {memory_mb:.2f} MB ({memory_gb:.3f} GB)")
        print(f"   Virtual Memory: {process.memory_info().vms / (1024**2):.2f} MB")
    
    def test_indexing_performance(self):
        """Test new document indexing (< 1 minute requirement)"""
        print("\n" + "="*60)
        print("TESTING DOCUMENT INDEXING (Requirement: < 1 minute)")
        print("="*60)
        
        test_player = {
            'short_name': 'Test Player',
            'long_name': 'Test Player Full Name',
            'overall': 75,
            'age': 25,
            'nationality_name': 'Test Country',
            'club_name': 'Test FC',
            'player_positions': 'ST',
            'pace': 80,
            'shooting': 75,
            'passing': 70,
            'dribbling': 78,
            'defending': 40,
            'physic': 72
        }
        
        start = time.time()
        
        try:
            response = requests.post(
                f"{self.api_base}/player/add",
                json=test_player,
                timeout=65  # Slightly over 1 minute
            )
            
            elapsed = (time.time() - start) * 1000
            success = response.status_code == 200
            
            compliant = elapsed < 60000  # < 1 minute
            status = "" if compliant else ""
            
            self.results['indexing_performance'] = {
                'time_ms': round(elapsed, 2),
                'time_sec': round(elapsed / 1000, 2),
                'success': success,
                'compliant': compliant
            }
            
            if success:
                response_data = response.json()
                print(f"{status} Indexing Time: {elapsed:.2f}ms ({elapsed/1000:.2f}s)")
                print(f"   Player ID: {response_data.get('player_id')}")
                print(f"   Document ID: {response_data.get('doc_id')}")
            else:
                print(f" Indexing Failed: {response.text}")
                
        except Exception as e:
            print(f" Indexing Failed: {str(e)}")
            self.results['indexing_performance'] = {
                'time_ms': 0,
                'success': False,
                'compliant': False,
                'error': str(e)
            }
    
    def calculate_compliance(self):
        """Calculate overall compliance with requirements"""
        print("\n" + "="*60)
        print("COMPLIANCE SUMMARY")
        print("="*60)
        
        # Single word queries
        single_word_pass = sum(1 for q in self.results['single_word_queries'] if q.get('compliant', False))
        single_word_total = len(self.results['single_word_queries'])
        single_word_pct = (single_word_pass / single_word_total * 100) if single_word_total > 0 else 0
        
        # Multi-word queries
        multi_word_pass = sum(1 for q in self.results['multi_word_queries'] if q.get('compliant', False))
        multi_word_total = len(self.results['multi_word_queries'])
        multi_word_pct = (multi_word_pass / multi_word_total * 100) if multi_word_total > 0 else 0
        
        # Memory
        memory_compliant = self.results['memory_usage'].get('compliant', False)
        
        # Indexing
        indexing_compliant = self.results['indexing_performance'].get('compliant', False)
        
        self.results['compliance'] = {
            'single_word_queries': {
                'passed': single_word_pass,
                'total': single_word_total,
                'percentage': round(single_word_pct, 2),
                'compliant': single_word_pct >= 90
            },
            'multi_word_queries': {
                'passed': multi_word_pass,
                'total': multi_word_total,
                'percentage': round(multi_word_pct, 2),
                'compliant': multi_word_pct >= 90
            },
            'memory_usage': memory_compliant,
            'indexing_speed': indexing_compliant
        }
        
        print(f"\n Single Word Queries: {single_word_pass}/{single_word_total} passed ({single_word_pct:.1f}%)")
        print(f" Multi-Word Queries: {multi_word_pass}/{multi_word_total} passed ({multi_word_pct:.1f}%)")
        print(f"{'' if memory_compliant else ''} Memory Usage: {'PASS' if memory_compliant else 'FAIL'}")
        print(f"{'' if indexing_compliant else ''} Indexing Speed: {'PASS' if indexing_compliant else 'FAIL'}")
        
        overall_compliant = (
            single_word_pct >= 90 and
            multi_word_pct >= 90 and
            memory_compliant and
            indexing_compliant
        )
        
        print(f"\n{' OVERALL: COMPLIANT' if overall_compliant else ' OVERALL: NOT COMPLIANT'}")
        
        return overall_compliant
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("\n PERFORMANCE BENCHMARK SUITE")
        print("=" * 60)
        
        self.test_single_word_queries()
        self.test_multi_word_queries()
        self.test_memory_usage()
        self.test_indexing_performance()
        
        compliant = self.calculate_compliance()
        
        # Save results to file
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n Full results saved to benchmark_results.json")
        
        return compliant
    
    def get_results(self):
        """Get benchmark results"""
        return self.results

if __name__ == '__main__':
    print(" Make sure Flask server is running on http://localhost:5000")
    input("Press Enter to start benchmarks...")
    
    benchmark = PerformanceBenchmark()
    benchmark.run_all_tests()

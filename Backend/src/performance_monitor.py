"""
Performance Monitoring: Track query times, memory usage, and system metrics
Provides real-time performance statistics and benchmarking
"""

import time
import psutil
import os
from functools import wraps
from typing import Dict, List, Callable
from collections import defaultdict, deque
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Centralized performance monitoring system
    Tracks query response times, memory usage, and system health
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize performance monitor
        
        Args:
            history_size: Number of recent queries to keep in history
        """
        self.history_size = history_size
        
        # Query performance tracking
        self.query_times: deque = deque(maxlen=history_size)
        self.query_counts = defaultdict(int)
        self.query_types = defaultdict(list)
        
        # Memory tracking
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        
        # System statistics
        self.total_queries = 0
        self.failed_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance requirements tracking
        self.single_word_times = deque(maxlen=100)
        self.multi_word_times = defaultdict(lambda: deque(maxlen=100))
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
        
        logger.info("Performance monitor initialized")
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage in MB"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': self.process.memory_percent()
        }
    
    def record_query(self, query: str, response_time: float, 
                    result_count: int, query_type: str = 'unknown',
                    success: bool = True):
        """
        Record a query's performance metrics
        
        Args:
            query: Search query text
            response_time: Time taken in milliseconds
            result_count: Number of results returned
            query_type: Type of query (single_word, multi_word, semantic, etc.)
            success: Whether query succeeded
        """
        with self.lock:
            # Record query time
            self.query_times.append({
                'query': query,
                'time_ms': response_time,
                'results': result_count,
                'type': query_type,
                'timestamp': time.time(),
                'success': success
            })
            
            # Update counters
            self.total_queries += 1
            if not success:
                self.failed_queries += 1
            
            self.query_counts[query_type] += 1
            self.query_types[query_type].append(response_time)
            
            # Track by word count for requirement compliance
            word_count = len(query.split())
            if word_count == 1:
                self.single_word_times.append(response_time)
            else:
                self.multi_word_times[word_count].append(response_time)
    
    def get_statistics(self) -> dict:
        """Get comprehensive performance statistics"""
        with self.lock:
            # Calculate averages
            if self.query_times:
                recent_times = [q['time_ms'] for q in self.query_times]
                avg_time = sum(recent_times) / len(recent_times)
                max_time = max(recent_times)
                min_time = min(recent_times)
            else:
                avg_time = max_time = min_time = 0
            
            # Memory stats
            current_memory = self.get_memory_usage()
            memory_delta = current_memory['rss_mb'] - self.initial_memory['rss_mb']
            
            # Requirement compliance
            compliance = self._check_compliance()
            
            return {
                'total_queries': self.total_queries,
                'failed_queries': self.failed_queries,
                'success_rate': (
                    (self.total_queries - self.failed_queries) / self.total_queries * 100
                    if self.total_queries > 0 else 100
                ),
                'average_response_time_ms': round(avg_time, 2),
                'max_response_time_ms': round(max_time, 2),
                'min_response_time_ms': round(min_time, 2),
                'memory_usage_mb': round(current_memory['rss_mb'], 2),
                'memory_delta_mb': round(memory_delta, 2),
                'memory_percent': round(current_memory['percent'], 2),
                'query_types': dict(self.query_counts),
                'recent_queries': list(self.query_times)[-10:],  # Last 10 queries
                'compliance': compliance,
                'cache_hit_rate': (
                    self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                    if (self.cache_hits + self.cache_misses) > 0 else 0
                )
            }
    
    def _check_compliance(self) -> dict:
        """Check compliance with performance requirements"""
        compliance = {
            'single_word_query': {
                'requirement': '< 500ms',
                'current_avg': 0,
                'passing': False,
                'sample_size': 0
            },
            'five_word_query': {
                'requirement': '< 1500ms',
                'current_avg': 0,
                'passing': False,
                'sample_size': 0
            },
            'memory_usage': {
                'requirement': ' 2GB (datasets < 100k docs)',
                'current_mb': 0,
                'passing': False
            }
        }
        
        # Single word queries
        if self.single_word_times:
            avg_single = sum(self.single_word_times) / len(self.single_word_times)
            compliance['single_word_query']['current_avg'] = round(avg_single, 2)
            compliance['single_word_query']['passing'] = avg_single < 500
            compliance['single_word_query']['sample_size'] = len(self.single_word_times)
        
        # Five word queries
        if 5 in self.multi_word_times and self.multi_word_times[5]:
            avg_five = sum(self.multi_word_times[5]) / len(self.multi_word_times[5])
            compliance['five_word_query']['current_avg'] = round(avg_five, 2)
            compliance['five_word_query']['passing'] = avg_five < 1500
            compliance['five_word_query']['sample_size'] = len(self.multi_word_times[5])
        
        # Memory usage
        current_memory = self.get_memory_usage()
        compliance['memory_usage']['current_mb'] = round(current_memory['rss_mb'], 2)
        compliance['memory_usage']['passing'] = current_memory['rss_mb'] <= 2048
        
        return compliance
    
    def get_performance_report(self) -> str:
        """Generate human-readable performance report"""
        stats = self.get_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"\nQUERY STATISTICS:")
        report.append(f"  Total Queries: {stats['total_queries']:,}")
        report.append(f"  Failed Queries: {stats['failed_queries']}")
        report.append(f"  Success Rate: {stats['success_rate']:.2f}%")
        report.append(f"\nRESPONSE TIMES:")
        report.append(f"  Average: {stats['average_response_time_ms']}ms")
        report.append(f"  Max: {stats['max_response_time_ms']}ms")
        report.append(f"  Min: {stats['min_response_time_ms']}ms")
        report.append(f"\nMEMORY USAGE:")
        report.append(f"  Current: {stats['memory_usage_mb']}MB")
        report.append(f"  Delta from Start: {stats['memory_delta_mb']:+.2f}MB")
        report.append(f"  Percent: {stats['memory_percent']:.2f}%")
        report.append(f"\nREQUIREMENT COMPLIANCE:")
        
        for req_name, req_data in stats['compliance'].items():
            status = " PASS" if req_data['passing'] else " FAIL"
            report.append(f"  {req_name}: {status}")
            report.append(f"    Requirement: {req_data['requirement']}")
            if 'current_avg' in req_data:
                report.append(f"    Current Avg: {req_data['current_avg']}ms")
                report.append(f"    Sample Size: {req_data['sample_size']}")
            elif 'current_mb' in req_data:
                report.append(f"    Current: {req_data['current_mb']}MB")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def performance_tracked(query_type: str = 'unknown'):
    """
    Decorator to track function performance
    
    Usage:
        @performance_tracked('text_search')
        def search_function(query):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result_count = 0
            query = ""
            
            try:
                result = func(*args, **kwargs)
                
                # Extract query and result count
                if args and isinstance(args[0], str):
                    query = args[0]
                elif 'query' in kwargs:
                    query = kwargs['query']
                
                if isinstance(result, dict) and 'results' in result:
                    result_count = len(result['results'])
                elif isinstance(result, list):
                    result_count = len(result)
                
                return result
                
            except Exception as e:
                success = False
                raise
            
            finally:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Record in global monitor
                if hasattr(wrapper, 'monitor'):
                    wrapper.monitor.record_query(
                        query=query,
                        response_time=response_time,
                        result_count=result_count,
                        query_type=query_type,
                        success=success
                    )
        
        return wrapper
    return decorator


# Global performance monitor instance
performance_monitor: PerformanceMonitor = PerformanceMonitor()


def track_query(query: str, response_time: float, result_count: int, 
                query_type: str = 'unknown', success: bool = True):
    """
    Manually track a query's performance
    
    Args:
        query: Search query
        response_time: Response time in milliseconds
        result_count: Number of results
        query_type: Type of query
        success: Whether successful
    """
    performance_monitor.record_query(
        query=query,
        response_time=response_time,
        result_count=result_count,
        query_type=query_type,
        success=success
    )


def get_performance_stats() -> dict:
    """Get current performance statistics"""
    return performance_monitor.get_statistics()


def get_performance_report() -> str:
    """Get formatted performance report"""
    return performance_monitor.get_performance_report()

"""
Benchmarking Module
===================

Comprehensive benchmarking and competitive analysis systems.

Components:
- ComprehensiveBenchmarkSuite: Multi-baseline competitive benchmarking
- ARC Evaluator: ARC-specific evaluation metrics
- Consciousness Metrics: Consciousness-specific measurement systems
"""

try:
    from .comprehensive_benchmark_suite import ComprehensiveBenchmarkSuite
except ImportError:
    pass

__all__ = ['ComprehensiveBenchmarkSuite']
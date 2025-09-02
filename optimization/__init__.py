"""
Optimization Module
===================

Performance optimization systems for hyperparameters and architecture.

Components:
- ConsciousnessHyperparameterOptimizer: Bayesian hyperparameter optimization
- ArchitectureOptimizer: Model compression and quantization
- Inference Optimizer: Runtime performance optimization
"""

try:
    from .hyperparameter_optimizer import ConsciousnessHyperparameterOptimizer
    from .architecture_optimizer import ArchitectureOptimizer
except ImportError:
    pass

__all__ = ['ConsciousnessHyperparameterOptimizer', 'ArchitectureOptimizer']
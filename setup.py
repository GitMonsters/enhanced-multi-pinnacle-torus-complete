#!/usr/bin/env python3
"""
Enhanced Multi-PINNACLE Consciousness System Setup
===================================================

Installation script for the Enhanced Multi-PINNACLE Consciousness System,
a revolutionary AI system that integrates multiple consciousness frameworks
for solving abstract reasoning challenges.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the README file
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Enhanced Multi-PINNACLE Consciousness System"

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering out comments and options"""
    requirements = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Remove inline comments
                    requirement = line.split('#')[0].strip()
                    if requirement:
                        requirements.append(requirement)
    except FileNotFoundError:
        pass
    return requirements

# Core requirements (always installed)
install_requires = [
    "torch>=2.0.0",
    "numpy>=1.24.0", 
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "optuna>=3.4.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "psutil>=5.9.0",
    "pyyaml>=6.0.0",
    "tqdm>=4.65.0",
]

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0", 
        "black>=23.3.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "line_profiler>=4.1.0",
        "memory_profiler>=0.61.0",
    ],
    "notebooks": [
        "jupyter>=1.0.0",
        "ipython>=8.14.0",
        "notebook>=6.5.0",
        "plotly>=5.15.0",
    ],
    "cloud": [
        "boto3>=1.28.0",
        "google-cloud-storage>=2.10.0", 
        "azure-storage-blob>=12.17.0",
    ],
    "api": [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    "visualization": [
        "plotly>=5.15.0",
        "bokeh>=3.2.0",
        "dash>=2.11.0",
    ]
}

# All optional dependencies
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="enhanced-multi-pinnacle",
    version="1.0.0",
    author="Enhanced Multi-PINNACLE Team",
    author_email="contact@enhanced-multi-pinnacle.ai",
    description="Advanced Consciousness-Based AI System for Abstract Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/enhanced_multi_pinnacle_complete",
    
    # Package configuration
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "enhanced-pinnacle=core.enhanced_multi_pinnacle:main",
            "pinnacle-train=training.advanced_training_pipeline:main",
            "pinnacle-validate=validation.real_world_arc_validator:main", 
            "pinnacle-benchmark=benchmarking.comprehensive_benchmark_suite:main",
            "pinnacle-optimize=optimization.hyperparameter_optimizer:main",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md", "*.cfg"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-username/enhanced_multi_pinnacle_complete/issues",
        "Source": "https://github.com/your-username/enhanced_multi_pinnacle_complete",
        "Documentation": "https://enhanced-multi-pinnacle.readthedocs.io/",
        "Research Paper": "https://arxiv.org/abs/xxxx.xxxxx",
    },
    
    # Keywords for PyPI
    keywords=[
        "artificial intelligence",
        "consciousness",
        "abstract reasoning", 
        "neural networks",
        "machine learning",
        "cognitive science",
        "ARC challenge",
        "AGI",
        "multi-modal AI",
        "transformer models"
    ],
    
    # Zip safe
    zip_safe=False,
)
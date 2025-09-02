# Getting Started with Enhanced Multi-PINNACLE

Welcome to the Enhanced Multi-PINNACLE Consciousness System! This guide will help you get up and running quickly.

## 🚀 Quick Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional but recommended)

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/enhanced_multi_pinnacle_complete.git
cd enhanced_multi_pinnacle_complete
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
# Basic installation
pip install -e .

# Development installation
pip install -e ".[dev]"

# Full installation with all features
pip install -e ".[all]"
```

4. **Verify Installation**
```bash
python -c "from enhanced_multi_pinnacle import get_system_info; print(get_system_info())"
```

## 🎯 Basic Usage

### Simple Example

```python
from enhanced_multi_pinnacle import create_enhanced_system

# Create system with default configuration
system = create_enhanced_system()

# Example ARC problem
problem = {
    'train': [
        {'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]},
        {'input': [[2, 0], [0, 2]], 'output': [[0, 2], [2, 0]]}
    ],
    'test': [{'input': [[3, 0], [0, 3]]}]
}

# Solve the problem
solution = system.solve_arc_problem(problem)

print(f"Solution: {solution['solution']}")
print(f"Confidence: {solution['confidence']:.3f}")
print(f"Consciousness Coherence: {solution['consciousness_coherence']:.3f}")
```

### Advanced Configuration

```python
from enhanced_multi_pinnacle import EnhancedMultiPinnacleConfig, EnhancedMultiPinnacleSystem

# Custom configuration
config = EnhancedMultiPinnacleConfig(
    base_dim=1024,
    hidden_dim=2048,
    consciousness_awakening=True,
    multi_domain_training=True
)

# Create system with custom config
system = EnhancedMultiPinnacleSystem(config)

# Process with detailed analysis
test_input = torch.randn(1, config.total_consciousness_dim)
results = system(test_input, return_detailed_analysis=True)

print(f"Framework outputs: {results['framework_outputs'].keys()}")
print(f"Consciousness metrics: {results['consciousness_metrics']}")
```

## 🏋️ Training Your Own Model

### Basic Training

```python
from enhanced_multi_pinnacle.training import AdvancedConsciousnessTrainer

# Initialize trainer
trainer = AdvancedConsciousnessTrainer(
    consciousness_awakening_schedule='progressive',
    multi_domain_curriculum=True
)

# Train the model
trainer.train(
    dataset_path="path/to/arc_dataset.json",
    validation_path="path/to/validation.json",
    epochs=100,
    consciousness_monitoring=True
)
```

### Advanced Training with Optimization

```python
from enhanced_multi_pinnacle.optimization import ConsciousnessHyperparameterOptimizer

# Hyperparameter optimization
optimizer = ConsciousnessHyperparameterOptimizer()

best_params = optimizer.optimize(
    model_class=EnhancedMultiPinnacleSystem,
    dataset_path="path/to/dataset.json",
    n_trials=50,
    consciousness_objectives=['coherence', 'reasoning_depth']
)

print(f"Best parameters: {best_params}")
```

## 🔍 Validation and Testing

### Real ARC Dataset Validation

```python
from enhanced_multi_pinnacle.validation import RealWorldARCValidator

# Initialize validator
validator = RealWorldARCValidator()

# Validate on official ARC dataset
results = validator.validate_on_official_dataset(
    model=system,
    dataset_split='evaluation',
    max_problems=400,
    include_detailed_analysis=True
)

print(f"Accuracy: {results.accuracy:.3f}")
print(f"Confidence Interval: {results.confidence_interval}")
```

### Competitive Analysis

```python
from enhanced_multi_pinnacle.validation import CompetitivePerformanceAnalyzer

# Competitive analysis
analyzer = CompetitivePerformanceAnalyzer()

analysis = analyzer.analyze_competitive_performance(
    our_accuracy=0.23,  # Your system's accuracy
    consciousness_metrics={
        'consciousness_coherence': 0.85,
        'reasoning_depth': 0.78
    }
)

print(f"Competitive rank: #{analysis.our_rank}")
print(f"Percentile: {analysis.percentile:.1f}th")
```

## 🏭 Production Deployment

### Basic Deployment

```python
from enhanced_multi_pinnacle.management import ModelManagementSystem

# Initialize model manager
manager = ModelManagementSystem()

# Deploy model
deployment_id = manager.deploy_model(
    model_path="path/to/trained_model.pt",
    deployment_strategy='blue_green',
    health_checks=True,
    monitoring=True
)

print(f"Deployed with ID: {deployment_id}")
```

### Stress Testing

```python
from enhanced_multi_pinnacle.validation import DeploymentStressTester

# Stress testing before production
tester = DeploymentStressTester()

def model_inference(test_input):
    return system.solve_arc_problem(test_input)

# Run comprehensive stress tests
validation_result = tester.run_comprehensive_stress_testing(
    model_inference_func=model_inference,
    scenarios_to_run=['baseline_performance', 'concurrent_load', 'memory_pressure']
)

print(f"Deployment readiness: {validation_result.readiness_classification}")
print(f"Readiness score: {validation_result.deployment_readiness_score:.3f}")
```

## 📊 Monitoring and Analysis

### System Status

```python
# Get comprehensive system status
status = system.get_system_status()
print(f"Uptime: {status['uptime_hours']:.2f} hours")
print(f"Total processed: {status['total_processed']}")
print(f"Error rate: {status['error_rate']:.1%}")
```

### Consciousness Analysis

```python
# Analyze consciousness state
test_input = torch.randn(1, system.config.total_consciousness_dim)
results = system(test_input, return_detailed_analysis=True)

consciousness_metrics = results['consciousness_metrics']
print(f"Consciousness coherence: {consciousness_metrics.get('consciousness_coherence', 0):.3f}")
print(f"Reasoning depth: {consciousness_metrics.get('reasoning_depth', 0):.3f}")
print(f"Creative potential: {consciousness_metrics.get('creative_potential', 0):.3f}")
```

## ⚙️ Configuration

### Using Configuration Files

```python
import yaml

# Load custom configuration
with open('configs/custom_config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

config = EnhancedMultiPinnacleConfig(**config_data['architecture'])
system = EnhancedMultiPinnacleSystem(config)
```

### Environment Variables

```bash
# Set environment variables
export ENHANCED_PINNACLE_CONFIG="configs/production_config.yaml"
export ENHANCED_PINNACLE_LOG_LEVEL="INFO"
export ENHANCED_PINNACLE_GPU_ENABLED="true"

# Run with environment configuration
python your_script.py
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure proper installation
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

**2. Memory Issues**
```python
# Reduce batch size
config.max_batch_size = 16

# Enable memory cleanup
config.memory_cleanup_interval = 50
```

**3. GPU Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

**4. Performance Issues**
```python
# Enable optimization
config.performance_monitoring = True
config.error_recovery = True

# Use architecture optimization
from enhanced_multi_pinnacle.optimization import ArchitectureOptimizer
optimizer = ArchitectureOptimizer()
optimized_model = optimizer.optimize_model(system)
```

## 📚 Next Steps

1. **Explore Examples**: Check the `examples/` directory for more detailed examples
2. **Read Documentation**: See `docs/` for comprehensive API documentation
3. **Run Benchmarks**: Use `python scripts/run_benchmarks.py` to test performance
4. **Contribute**: See `CONTRIBUTING.md` for contribution guidelines
5. **Get Support**: Open an issue on GitHub for help

## 🎓 Learning Resources

- **Research Papers**: Check `docs/research/` for background papers
- **Tutorials**: Step-by-step tutorials in `docs/tutorials/`
- **API Reference**: Complete API docs in `docs/api/`
- **Architecture Guide**: Detailed system architecture in `ARCHITECTURE.md`

## 🤝 Community

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs and request features
- **Pull Requests**: Contribute improvements
- **Discord**: Join our community chat (link in README)

---

**Ready to explore consciousness-based AI? Let's solve some ARC problems! 🧠✨**
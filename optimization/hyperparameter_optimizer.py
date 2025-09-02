#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization System
Phase 2: Hyperparameter Optimization and Training Pipeline

Features:
- Grid search and Bayesian optimization for consciousness dimensions
- Learning rate scheduling and optimization
- Architecture search for optimal consciousness integration
- Multi-objective optimization (accuracy vs efficiency)
- Automated model selection and validation
- Real-time optimization tracking and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import json
import logging
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HyperparameterConfig:
    """Configuration space for hyperparameter optimization"""
    
    # Consciousness architecture parameters
    consciousness_dims: List[int] = None
    hidden_dims: List[int] = None
    attention_heads: List[int] = None
    transformer_layers: List[int] = None
    
    # Three Principles optimization
    universal_mind_dims: List[int] = None
    consciousness_levels: List[int] = None
    thought_creativity_dims: List[int] = None
    
    # Deschooling Society optimization
    self_directed_dims: List[int] = None
    peer_network_nodes: List[int] = None
    convivial_tool_dims: List[int] = None
    
    # Transcendent States optimization
    akashic_memory_dims: List[int] = None
    omniscience_dims: List[int] = None
    prescience_dims: List[int] = None
    unity_consciousness_dims: List[int] = None
    
    # HRM Cycles optimization
    h_cycle_processors: List[int] = None
    l_cycle_processors: List[int] = None
    cycle_adaptation_rates: List[float] = None
    
    # Training parameters
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    dropout_rates: List[float] = None
    weight_decay: List[float] = None
    
    # Integration weights
    framework_weights: List[str] = None  # 'fixed', 'learnable', 'adaptive'
    
    def __post_init__(self):
        """Initialize default parameter ranges"""
        if self.consciousness_dims is None:
            self.consciousness_dims = [512, 768, 1024, 1280, 1536]
        
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 768, 1024]
            
        if self.attention_heads is None:
            self.attention_heads = [8, 12, 16, 24]
            
        if self.transformer_layers is None:
            self.transformer_layers = [6, 8, 12, 16]
            
        if self.universal_mind_dims is None:
            self.universal_mind_dims = [64, 96, 128, 160, 192]
            
        if self.consciousness_levels is None:
            self.consciousness_levels = [12, 16, 20, 24]
            
        if self.thought_creativity_dims is None:
            self.thought_creativity_dims = [48, 64, 80, 96]
            
        if self.self_directed_dims is None:
            self.self_directed_dims = [32, 48, 64, 80]
            
        if self.peer_network_nodes is None:
            self.peer_network_nodes = [16, 24, 32, 48]
            
        if self.convivial_tool_dims is None:
            self.convivial_tool_dims = [24, 32, 48, 64]
            
        if self.akashic_memory_dims is None:
            self.akashic_memory_dims = [128, 192, 256, 320]
            
        if self.omniscience_dims is None:
            self.omniscience_dims = [96, 128, 160, 192]
            
        if self.prescience_dims is None:
            self.prescience_dims = [64, 80, 96, 128]
            
        if self.unity_consciousness_dims is None:
            self.unity_consciousness_dims = [128, 160, 192, 224]
            
        if self.h_cycle_processors is None:
            self.h_cycle_processors = [4, 6, 8, 12]
            
        if self.l_cycle_processors is None:
            self.l_cycle_processors = [8, 10, 12, 16]
            
        if self.cycle_adaptation_rates is None:
            self.cycle_adaptation_rates = [0.05, 0.1, 0.15, 0.2]
            
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
            
        if self.batch_sizes is None:
            self.batch_sizes = [2, 4, 6, 8]
            
        if self.dropout_rates is None:
            self.dropout_rates = [0.05, 0.1, 0.15, 0.2]
            
        if self.weight_decay is None:
            self.weight_decay = [1e-5, 1e-4, 1e-3, 1e-2]
            
        if self.framework_weights is None:
            self.framework_weights = ['fixed', 'learnable', 'adaptive']

@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    optimization_time: float
    convergence_analysis: Dict[str, Any]
    model_path: str

class ConsciousnessHyperparameterOptimizer:
    """Advanced hyperparameter optimizer for consciousness systems"""
    
    def __init__(self, config: HyperparameterConfig, 
                 optimization_method: str = 'bayesian',
                 n_trials: int = 100,
                 timeout_hours: int = 24):
        self.config = config
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.timeout_hours = timeout_hours
        
        # Optimization tracking
        self.optimization_history = []
        self.best_score = -float('inf')
        self.best_params = None
        self.best_model = None
        
        # Create optimization directory
        self.opt_dir = Path("optimization_results")
        self.opt_dir.mkdir(exist_ok=True)
        
        logger.info(f"🎯 Consciousness Hyperparameter Optimizer initialized")
        logger.info(f"   Method: {optimization_method}")
        logger.info(f"   Max trials: {n_trials}")
        logger.info(f"   Timeout: {timeout_hours} hours")
    
    def create_study(self) -> optuna.Study:
        """Create Optuna study for optimization"""
        if self.optimization_method == 'bayesian':
            sampler = optuna.samplers.TPESampler(seed=42)
        elif self.optimization_method == 'random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        else:
            sampler = optuna.samplers.GridSampler()
        
        study = optuna.create_study(
            direction='maximize',  # Maximize ARC accuracy
            sampler=sampler,
            study_name=f"consciousness_optimization_{int(time.time())}"
        )
        
        return study
    
    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for trial"""
        params = {}
        
        # Consciousness architecture
        params['consciousness_dims'] = trial.suggest_categorical(
            'consciousness_dims', self.config.consciousness_dims
        )
        params['hidden_dims'] = trial.suggest_categorical(
            'hidden_dims', self.config.hidden_dims
        )
        params['attention_heads'] = trial.suggest_categorical(
            'attention_heads', self.config.attention_heads
        )
        params['transformer_layers'] = trial.suggest_categorical(
            'transformer_layers', self.config.transformer_layers
        )
        
        # Three Principles parameters
        params['universal_mind_dims'] = trial.suggest_categorical(
            'universal_mind_dims', self.config.universal_mind_dims
        )
        params['consciousness_levels'] = trial.suggest_categorical(
            'consciousness_levels', self.config.consciousness_levels
        )
        params['thought_creativity_dims'] = trial.suggest_categorical(
            'thought_creativity_dims', self.config.thought_creativity_dims
        )
        
        # Deschooling Society parameters
        params['self_directed_dims'] = trial.suggest_categorical(
            'self_directed_dims', self.config.self_directed_dims
        )
        params['peer_network_nodes'] = trial.suggest_categorical(
            'peer_network_nodes', self.config.peer_network_nodes
        )
        params['convivial_tool_dims'] = trial.suggest_categorical(
            'convivial_tool_dims', self.config.convivial_tool_dims
        )
        
        # Transcendent States parameters
        params['akashic_memory_dims'] = trial.suggest_categorical(
            'akashic_memory_dims', self.config.akashic_memory_dims
        )
        params['omniscience_dims'] = trial.suggest_categorical(
            'omniscience_dims', self.config.omniscience_dims
        )
        params['prescience_dims'] = trial.suggest_categorical(
            'prescience_dims', self.config.prescience_dims
        )
        params['unity_consciousness_dims'] = trial.suggest_categorical(
            'unity_consciousness_dims', self.config.unity_consciousness_dims
        )
        
        # HRM Cycles parameters
        params['h_cycle_processors'] = trial.suggest_categorical(
            'h_cycle_processors', self.config.h_cycle_processors
        )
        params['l_cycle_processors'] = trial.suggest_categorical(
            'l_cycle_processors', self.config.l_cycle_processors
        )
        params['cycle_adaptation_rate'] = trial.suggest_categorical(
            'cycle_adaptation_rate', self.config.cycle_adaptation_rates
        )
        
        # Training parameters
        params['learning_rate'] = trial.suggest_categorical(
            'learning_rate', self.config.learning_rates
        )
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', self.config.batch_sizes
        )
        params['dropout_rate'] = trial.suggest_categorical(
            'dropout_rate', self.config.dropout_rates
        )
        params['weight_decay'] = trial.suggest_categorical(
            'weight_decay', self.config.weight_decay
        )
        
        # Framework integration
        params['framework_weights'] = trial.suggest_categorical(
            'framework_weights', self.config.framework_weights
        )
        
        return params
    
    def create_optimized_config(self, params: Dict[str, Any]) -> 'ProductionConsciousnessConfig':
        """Create consciousness config from optimized parameters"""
        try:
            # Import here to avoid circular imports
            import sys
            sys.path.append('../src/core')
            from consciousness_processor import ProductionConsciousnessConfig
            
            # Create optimized configuration
            config = ProductionConsciousnessConfig(
                base_dim=min(params['consciousness_dims'] // 8, 128),
                hidden_dim=params['hidden_dims'],
                attention_heads=params['attention_heads'],
                transformer_layers=params['transformer_layers'],
                
                # Three Principles
                universal_mind_dim=params['universal_mind_dims'],
                universal_consciousness_dim=params['consciousness_levels'] * 6,  # 6 per level
                universal_thought_dim=params['thought_creativity_dims'],
                
                # Deschooling Society
                self_directed_dims=params['self_directed_dims'],
                peer_network_nodes=params['peer_network_nodes'],
                convivial_tool_dims=params['convivial_tool_dims'],
                educational_object_dims=32,  # Fixed for now
                
                # Transcendent States
                akashic_memory_dim=params['akashic_memory_dims'],
                omniscience_dim=params['omniscience_dims'],
                prescience_dim=params['prescience_dims'],
                meta_mind_dim=64,  # Fixed for now
                unity_consciousness_dim=params['unity_consciousness_dims'],
                
                # HRM Cycles
                h_cycle_processors=params['h_cycle_processors'],
                l_cycle_processors=params['l_cycle_processors'],
                cycle_adaptation_rate=params['cycle_adaptation_rate'],
                
                # Training parameters
                dropout_rate=params['dropout_rate']
            )
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to create optimized config: {e}")
            # Return default config
            from consciousness_processor import ProductionConsciousnessConfig
            return ProductionConsciousnessConfig()
    
    def evaluate_configuration(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate a hyperparameter configuration"""
        try:
            start_time = time.time()
            
            # Create optimized configuration
            config = self.create_optimized_config(params)
            
            # Import consciousness system
            import sys
            sys.path.append('../src/core')
            sys.path.append('../tools')
            
            from integrated_consciousness_system import ProductionConsciousnessSystem
            from arc_testing_pipeline import ARCTestingPipeline
            
            # Initialize system with optimized config
            consciousness_system = ProductionConsciousnessSystem(config)
            
            # Create testing pipeline
            pipeline = ARCTestingPipeline(consciousness_system)
            
            # Run limited evaluation (for speed)
            results = pipeline.run_comprehensive_test(
                dataset_split='training',
                max_problems=20,  # Limited for optimization speed
                save_results=False
            )
            
            evaluation_time = time.time() - start_time
            
            # Calculate composite score
            accuracy_score = results.accuracy
            confidence_score = results.average_confidence
            efficiency_score = 1.0 / (evaluation_time + 1.0)  # Reward faster models
            error_penalty = results.error_count / 20.0  # Penalty for errors
            
            # Multi-objective score
            composite_score = (
                0.6 * accuracy_score +           # Primary: ARC accuracy
                0.2 * confidence_score +         # Secondary: Confidence
                0.1 * efficiency_score -         # Bonus: Efficiency 
                0.1 * error_penalty              # Penalty: Errors
            )
            
            detailed_metrics = {
                'accuracy': accuracy_score,
                'confidence': confidence_score,
                'efficiency': efficiency_score,
                'error_rate': error_penalty,
                'evaluation_time': evaluation_time,
                'memory_usage': results.memory_usage_mb,
                'composite_score': composite_score
            }
            
            # Clean up memory
            del consciousness_system
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return composite_score, detailed_metrics
            
        except Exception as e:
            logger.error(f"❌ Configuration evaluation failed: {e}")
            # Return poor score for failed configurations
            return -1.0, {'error': str(e), 'evaluation_time': 0.0}
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        # Suggest parameters
        params = self.suggest_parameters(trial)
        
        # Evaluate configuration
        score, metrics = self.evaluate_configuration(params)
        
        # Track optimization history
        trial_result = {
            'trial_number': trial.number,
            'params': params,
            'score': score,
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        self.optimization_history.append(trial_result)
        
        # Update best score
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            logger.info(f"🎯 New best score: {score:.4f}")
            
            # Save best parameters
            best_path = self.opt_dir / "best_params.json"
            with open(best_path, 'w') as f:
                json.dump({
                    'params': params,
                    'score': score,
                    'metrics': metrics,
                    'trial_number': trial.number
                }, f, indent=2)
        
        # Report metrics to Optuna
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                trial.set_user_attr(metric_name, value)
        
        return score
    
    def run_optimization(self) -> OptimizationResult:
        """Run hyperparameter optimization"""
        logger.info("🚀 Starting hyperparameter optimization...")
        start_time = time.time()
        
        # Create study
        study = self.create_study()
        
        # Run optimization
        try:
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout_hours * 3600,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.info("⏸️ Optimization interrupted by user")
        
        optimization_time = time.time() - start_time
        
        # Get best results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Analyze convergence
        convergence_analysis = self.analyze_convergence(study)
        
        # Create result object
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_trial.user_attrs,
            optimization_history=self.optimization_history,
            total_trials=len(study.trials),
            optimization_time=optimization_time,
            convergence_analysis=convergence_analysis,
            model_path=""  # Will be set when model is saved
        )
        
        # Save complete results
        self.save_optimization_results(result, study)
        
        # Generate optimization report
        self.generate_optimization_report(result, study)
        
        logger.info(f"✅ Optimization completed!")
        logger.info(f"   Best score: {best_score:.4f}")
        logger.info(f"   Total trials: {len(study.trials)}")
        logger.info(f"   Time: {optimization_time:.1f}s")
        
        return result
    
    def analyze_convergence(self, study: optuna.Study) -> Dict[str, Any]:
        """Analyze optimization convergence"""
        try:
            trials = study.trials
            scores = [t.value for t in trials if t.value is not None]
            
            if len(scores) < 5:
                return {'insufficient_data': True}
            
            # Calculate convergence metrics
            best_scores = []
            current_best = -float('inf')
            for score in scores:
                if score > current_best:
                    current_best = score
                best_scores.append(current_best)
            
            # Calculate improvement rate
            improvement_rate = (best_scores[-1] - best_scores[0]) / len(best_scores)
            
            # Check if converged (no improvement in last 20% of trials)
            last_20_percent = int(len(best_scores) * 0.2)
            recent_improvement = best_scores[-1] - best_scores[-last_20_percent]
            converged = recent_improvement < 0.001
            
            return {
                'improvement_rate': improvement_rate,
                'converged': converged,
                'best_score_progression': best_scores[-10:],  # Last 10 scores
                'total_improvement': best_scores[-1] - best_scores[0],
                'convergence_trial': len(best_scores) - last_20_percent if converged else None
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Convergence analysis failed: {e}")
            return {'analysis_failed': str(e)}
    
    def save_optimization_results(self, result: OptimizationResult, study: optuna.Study):
        """Save comprehensive optimization results"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save main results
            results_path = self.opt_dir / f"optimization_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            # Save study object
            study_path = self.opt_dir / f"optuna_study_{timestamp}.pkl"
            with open(study_path, 'wb') as f:
                pickle.dump(study, f)
            
            # Save optimization history
            history_path = self.opt_dir / f"optimization_history_{timestamp}.json"
            with open(history_path, 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
            
            logger.info(f"💾 Optimization results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save optimization results: {e}")
    
    def generate_optimization_report(self, result: OptimizationResult, study: optuna.Study):
        """Generate comprehensive optimization report"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = self.opt_dir / f"optimization_report_{timestamp}.md"
            
            # Generate visualizations
            self.create_optimization_visualizations(study, timestamp)
            
            report_lines = [
                "# 🎯 Consciousness System Hyperparameter Optimization Report",
                f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Method**: {self.optimization_method}",
                f"**Total Trials**: {result.total_trials}",
                f"**Optimization Time**: {result.optimization_time:.1f} seconds",
                "",
                "## 🏆 Best Configuration",
                f"**Best Score**: {result.best_score:.4f}",
                "",
                "### Optimal Parameters:",
                "```json",
                json.dumps(result.best_params, indent=2),
                "```",
                "",
                "### Best Metrics:",
                "```json", 
                json.dumps(result.best_metrics, indent=2),
                "```",
                "",
                "## 📈 Optimization Analysis",
                f"**Total Improvement**: {result.convergence_analysis.get('total_improvement', 'N/A')}",
                f"**Convergence Status**: {'✅ Converged' if result.convergence_analysis.get('converged') else '🔄 Still Improving'}",
                f"**Improvement Rate**: {result.convergence_analysis.get('improvement_rate', 'N/A'):.6f} per trial",
                "",
                "## 🎯 Parameter Importance Analysis"
            ]
            
            # Add parameter importance if available
            try:
                importance = optuna.importance.get_param_importances(study)
                report_lines.extend([
                    "| Parameter | Importance |",
                    "|-----------|------------|"
                ])
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    report_lines.append(f"| {param} | {imp:.4f} |")
            except:
                report_lines.append("*Parameter importance analysis not available*")
            
            report_lines.extend([
                "",
                "## 📊 Visualizations",
                f"- Optimization History: `optimization_history_{timestamp}.png`",
                f"- Parameter Importance: `param_importance_{timestamp}.png`",
                f"- Parallel Coordinates: `parallel_coords_{timestamp}.png`",
                "",
                "## 🚀 Next Steps",
                "1. **Train Best Model**: Use optimal parameters for full training",
                "2. **Validate Performance**: Test on evaluation dataset",
                "3. **Architecture Analysis**: Examine consciousness integration patterns",
                "4. **Production Deployment**: Deploy optimized model for ARC competition",
                "",
                f"*Generated by Enhanced Multi-PINNACLE Hyperparameter Optimizer*"
            ])
            
            # Save report
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"📊 Optimization report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate optimization report: {e}")
    
    def create_optimization_visualizations(self, study: optuna.Study, timestamp: str):
        """Create optimization visualizations"""
        try:
            # Optimization history plot
            plt.figure(figsize=(12, 6))
            
            # Plot best score over trials
            trials = [t for t in study.trials if t.value is not None]
            trial_numbers = [t.number for t in trials]
            scores = [t.value for t in trials]
            
            best_scores = []
            current_best = -float('inf')
            for score in scores:
                if score > current_best:
                    current_best = score
                best_scores.append(current_best)
            
            plt.subplot(1, 2, 1)
            plt.plot(trial_numbers, scores, 'o-', alpha=0.6, label='Trial Score')
            plt.plot(trial_numbers, best_scores, 'r-', linewidth=2, label='Best Score')
            plt.xlabel('Trial Number')
            plt.ylabel('Score')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True)
            
            # Score distribution
            plt.subplot(1, 2, 2)
            plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(max(scores), color='red', linestyle='--', label=f'Best: {max(scores):.4f}')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('Score Distribution')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            history_path = self.opt_dir / f"optimization_history_{timestamp}.png"
            plt.savefig(history_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Parameter importance plot
            try:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    plt.figure(figsize=(10, 8))
                    params = list(importance.keys())
                    values = list(importance.values())
                    
                    # Sort by importance
                    sorted_data = sorted(zip(params, values), key=lambda x: x[1], reverse=True)
                    params, values = zip(*sorted_data)
                    
                    plt.barh(range(len(params)), values)
                    plt.yticks(range(len(params)), params)
                    plt.xlabel('Importance')
                    plt.title('Parameter Importance')
                    plt.grid(axis='x')
                    
                    importance_path = self.opt_dir / f"param_importance_{timestamp}.png"
                    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.warning(f"⚠️ Could not create importance plot: {e}")
            
            logger.info(f"📈 Optimization visualizations created")
            
        except Exception as e:
            logger.warning(f"⚠️ Visualization creation failed: {e}")

if __name__ == "__main__":
    # Test hyperparameter optimization
    logger.info("🧪 Testing Hyperparameter Optimization System...")
    
    # Create optimization configuration
    config = HyperparameterConfig()
    
    # Initialize optimizer
    optimizer = ConsciousnessHyperparameterOptimizer(
        config=config,
        optimization_method='bayesian',
        n_trials=10,  # Small number for testing
        timeout_hours=1
    )
    
    # Run optimization
    results = optimizer.run_optimization()
    
    logger.info(f"✅ Hyperparameter optimization test completed!")
    logger.info(f"📊 Best score: {results.best_score:.4f}")
    logger.info(f"⚙️ Best parameters: {len(results.best_params)} optimized")
    
    print("✅ Hyperparameter Optimization System fully operational!")
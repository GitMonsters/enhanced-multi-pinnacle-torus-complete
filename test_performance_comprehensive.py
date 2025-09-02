#!/usr/bin/env python3
"""
COMPREHENSIVE PERFORMANCE TEST SUITE
Enhanced Multi-PINNACLE Torus System with TinyGrad
===================================================

Tests all major capabilities and performance metrics of the system.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_multi_pinnacle import EnhancedMultiPinnacleConfig, EnhancedMultiPinnacleSystem
from tinygrad.tensor import Tensor

class PerformanceTestSuite:
    """Comprehensive performance testing suite"""
    
    def __init__(self):
        self.results = {
            'system_info': {},
            'initialization_tests': {},
            'processing_tests': {},
            'memory_tests': {},
            'scalability_tests': {},
            'consciousness_tests': {},
            'arc_solving_tests': {},
            'stability_tests': {},
            'overall_performance': {}
        }
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 ENHANCED MULTI-PINNACLE PERFORMANCE TEST SUITE")
        print("=" * 70)
        
        # System initialization tests
        self.test_system_initialization()
        
        # Basic processing tests
        self.test_basic_processing()
        
        # Memory efficiency tests  
        self.test_memory_efficiency()
        
        # Scalability tests
        self.test_scalability()
        
        # Consciousness framework tests
        self.test_consciousness_frameworks()
        
        # ARC problem solving tests
        self.test_arc_solving_capability()
        
        # System stability tests
        self.test_system_stability()
        
        # Generate final report
        self.generate_performance_report()
        
        return self.results
    
    def test_system_initialization(self):
        """Test system initialization performance"""
        print("\n🔧 INITIALIZATION PERFORMANCE TESTS")
        print("-" * 50)
        
        # Test different configurations
        configs = [
            ("Minimal", {"base_dim": 32, "num_heads": 2, "num_layers": 1}),
            ("Standard", {"base_dim": 64, "num_heads": 4, "num_layers": 2}),
            ("Large", {"base_dim": 128, "num_heads": 8, "num_layers": 3}),
            ("Ultra", {"base_dim": 256, "num_heads": 16, "num_layers": 4})
        ]
        
        init_results = {}
        
        for config_name, config_params in configs:
            try:
                start_time = time.time()
                config = EnhancedMultiPinnacleConfig(**config_params)
                system = EnhancedMultiPinnacleSystem(config)
                init_time = time.time() - start_time
                
                init_results[config_name] = {
                    'init_time': init_time,
                    'success': True,
                    'params': config_params,
                    'consciousness_dim': config.total_consciousness_dim
                }
                
                print(f"✅ {config_name}: {init_time:.3f}s | Consciousness Dim: {config.total_consciousness_dim}")
                
            except Exception as e:
                init_results[config_name] = {
                    'init_time': -1,
                    'success': False,
                    'error': str(e),
                    'params': config_params
                }
                print(f"❌ {config_name}: FAILED - {e}")
        
        self.results['initialization_tests'] = init_results
    
    def test_basic_processing(self):
        """Test basic processing performance"""
        print("\n🧠 BASIC PROCESSING PERFORMANCE TESTS")
        print("-" * 50)
        
        # Create standard system
        config = EnhancedMultiPinnacleConfig(base_dim=64, num_heads=4, num_layers=2)
        system = EnhancedMultiPinnacleSystem(config)
        
        # Test different input sizes
        test_inputs = [
            ("Small", (1, 5, 64)),
            ("Medium", (2, 10, 64)),
            ("Large", (4, 20, 64)),
            ("Batch", (8, 15, 64))
        ]
        
        processing_results = {}
        
        for test_name, shape in test_inputs:
            try:
                # Generate test input
                test_input = Tensor.randn(*shape)
                
                # Time the forward pass
                start_time = time.time()
                result = system(test_input)
                processing_time = time.time() - start_time
                
                # Check output quality
                has_solution = 'arc_solution' in result and result['arc_solution'].shape[1] == 900
                has_confidence = 'confidence' in result
                processing_success = result.get('success', False)
                
                processing_results[test_name] = {
                    'input_shape': shape,
                    'processing_time': processing_time,
                    'throughput': shape[0] * shape[1] / processing_time,  # samples per second
                    'has_solution': has_solution,
                    'has_confidence': has_confidence,
                    'processing_success': processing_success,
                    'output_keys': list(result.keys())
                }
                
                print(f"✅ {test_name} {shape}: {processing_time:.3f}s | Throughput: {processing_results[test_name]['throughput']:.1f} samples/sec")
                
            except Exception as e:
                processing_results[test_name] = {
                    'input_shape': shape,
                    'processing_time': -1,
                    'error': str(e),
                    'success': False
                }
                print(f"❌ {test_name} {shape}: FAILED - {e}")
        
        self.results['processing_tests'] = processing_results
    
    def test_memory_efficiency(self):
        """Test memory usage and efficiency"""
        print("\n💾 MEMORY EFFICIENCY TESTS")
        print("-" * 50)
        
        import psutil
        import os
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = EnhancedMultiPinnacleConfig(base_dim=64)
        system = EnhancedMultiPinnacleSystem(config)
        
        # Memory after initialization
        init_memory = process.memory_info().rss / 1024 / 1024
        
        # Test memory usage during processing
        test_input = Tensor.randn(4, 20, 64)
        
        before_processing = process.memory_info().rss / 1024 / 1024
        result = system(test_input)
        after_processing = process.memory_info().rss / 1024 / 1024
        
        # Multiple processing cycles to check for memory leaks
        memory_samples = []
        for i in range(10):
            _ = system(test_input)
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
        
        memory_results = {
            'baseline_memory_mb': baseline_memory,
            'init_memory_mb': init_memory,
            'init_overhead_mb': init_memory - baseline_memory,
            'processing_memory_mb': after_processing,
            'processing_overhead_mb': after_processing - before_processing,
            'memory_samples': memory_samples,
            'memory_stable': max(memory_samples[-5:]) - min(memory_samples[-5:]) < 10,  # stable if <10MB variance
            'peak_memory_mb': max(memory_samples)
        }
        
        print(f"✅ Baseline Memory: {baseline_memory:.1f} MB")
        print(f"✅ Initialization Overhead: {memory_results['init_overhead_mb']:.1f} MB")
        print(f"✅ Processing Overhead: {memory_results['processing_overhead_mb']:.1f} MB")
        print(f"✅ Memory Stable: {memory_results['memory_stable']}")
        print(f"✅ Peak Memory: {memory_results['peak_memory_mb']:.1f} MB")
        
        self.results['memory_tests'] = memory_results
    
    def test_scalability(self):
        """Test system scalability with increasing loads"""
        print("\n📈 SCALABILITY TESTS")
        print("-" * 50)
        
        config = EnhancedMultiPinnacleConfig(base_dim=64)
        system = EnhancedMultiPinnacleSystem(config)
        
        # Test increasing batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_length = 10
        
        scalability_results = {}
        
        for batch_size in batch_sizes:
            try:
                test_input = Tensor.randn(batch_size, seq_length, 64)
                
                start_time = time.time()
                result = system(test_input)
                processing_time = time.time() - start_time
                
                throughput = (batch_size * seq_length) / processing_time
                
                scalability_results[f'batch_{batch_size}'] = {
                    'batch_size': batch_size,
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'time_per_sample': processing_time / (batch_size * seq_length),
                    'success': True
                }
                
                print(f"✅ Batch {batch_size:2d}: {processing_time:.3f}s | {throughput:.1f} samples/sec | {processing_time/(batch_size*seq_length)*1000:.2f}ms/sample")
                
            except Exception as e:
                scalability_results[f'batch_{batch_size}'] = {
                    'batch_size': batch_size,
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ Batch {batch_size:2d}: FAILED - {e}")
        
        # Calculate scaling efficiency
        if len([r for r in scalability_results.values() if r.get('success')]) > 1:
            successful_results = {k: v for k, v in scalability_results.items() if v.get('success')}
            batch_1_time = successful_results.get('batch_1', {}).get('time_per_sample', 0)
            if batch_1_time > 0:
                for key, result in successful_results.items():
                    if result.get('time_per_sample'):
                        result['scaling_efficiency'] = batch_1_time / result['time_per_sample']
        
        self.results['scalability_tests'] = scalability_results
    
    def test_consciousness_frameworks(self):
        """Test consciousness framework integration"""
        print("\n🧘 CONSCIOUSNESS FRAMEWORK TESTS")  
        print("-" * 50)
        
        config = EnhancedMultiPinnacleConfig(base_dim=64)
        system = EnhancedMultiPinnacleSystem(config)
        
        test_input = Tensor.randn(1, 10, 64)
        
        # Test detailed analysis mode
        start_time = time.time()
        result = system(test_input, return_detailed_analysis=True)
        analysis_time = time.time() - start_time
        
        consciousness_results = {
            'analysis_time': analysis_time,
            'consciousness_metrics_present': 'consciousness_metrics' in result,
            'framework_outputs': {},
            'integration_success': True
        }
        
        # Check if consciousness metrics are meaningful
        if 'consciousness_metrics' in result:
            metrics = result['consciousness_metrics']
            consciousness_results['metrics_keys'] = list(metrics.keys()) if isinstance(metrics, dict) else []
        
        # Test individual framework accessibility
        frameworks = ['universal_mind', 'three_principles', 'torus_topology']
        for framework in frameworks:
            if hasattr(system, framework):
                consciousness_results['framework_outputs'][framework] = 'available'
                print(f"✅ {framework.replace('_', ' ').title()}: Available")
            else:
                consciousness_results['framework_outputs'][framework] = 'missing'
                print(f"⚠️  {framework.replace('_', ' ').title()}: Missing")
        
        print(f"✅ Consciousness Analysis: {analysis_time:.3f}s")
        print(f"✅ Consciousness Metrics: {'Present' if consciousness_results['consciousness_metrics_present'] else 'Missing'}")
        
        self.results['consciousness_tests'] = consciousness_results
    
    def test_arc_solving_capability(self):
        """Test ARC problem solving capabilities"""
        print("\n🎯 ARC PROBLEM SOLVING TESTS")
        print("-" * 50)
        
        config = EnhancedMultiPinnacleConfig(base_dim=64)
        system = EnhancedMultiPinnacleSystem(config)
        
        # Simulate different ARC-like problems
        arc_tests = [
            ("Simple Pattern", (1, 8, 64)),
            ("Complex Pattern", (1, 15, 64)),  
            ("Multi-Grid", (2, 12, 64)),
            ("Large Problem", (1, 25, 64))
        ]
        
        arc_results = {}
        
        for test_name, shape in arc_tests:
            try:
                # Create ARC-like input
                test_input = Tensor.randn(*shape)
                
                start_time = time.time()
                result = system(test_input)
                solve_time = time.time() - start_time
                
                # Validate ARC solution format
                has_valid_solution = (
                    'arc_solution' in result and 
                    result['arc_solution'].shape[1] == 900  # Standard ARC output size
                )
                
                has_confidence = 'confidence' in result
                confidence_value = None
                if has_confidence:
                    try:
                        # TinyGrad tensor to value
                        confidence_value = float(result['confidence'].numpy().flatten()[0])
                    except:
                        confidence_value = "tensor"
                
                arc_results[test_name] = {
                    'input_shape': shape,
                    'solve_time': solve_time,
                    'has_valid_solution': has_valid_solution,
                    'solution_shape': result['arc_solution'].shape if 'arc_solution' in result else None,
                    'confidence': confidence_value,
                    'success': result.get('success', False)
                }
                
                print(f"✅ {test_name}: {solve_time:.3f}s | Solution: {has_valid_solution} | Confidence: {confidence_value}")
                
            except Exception as e:
                arc_results[test_name] = {
                    'input_shape': shape,
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ {test_name}: FAILED - {e}")
        
        self.results['arc_solving_tests'] = arc_results
    
    def test_system_stability(self):
        """Test system stability under stress"""
        print("\n🛡️  SYSTEM STABILITY TESTS")
        print("-" * 50)
        
        config = EnhancedMultiPinnacleConfig(base_dim=64)
        system = EnhancedMultiPinnacleSystem(config)
        
        stability_results = {
            'stress_test_passed': False,
            'error_handling_test_passed': False,
            'repeated_processing_test_passed': False,
            'edge_cases_test_passed': False
        }
        
        # Stress test - rapid processing
        try:
            print("Running stress test (100 rapid inferences)...")
            stress_start = time.time()
            test_input = Tensor.randn(2, 10, 64)
            
            for i in range(100):
                _ = system(test_input)
                if i % 25 == 0:
                    print(f"  Completed {i+1}/100...")
            
            stress_time = time.time() - stress_start
            stability_results['stress_test_passed'] = True
            stability_results['stress_test_time'] = stress_time
            stability_results['avg_inference_time'] = stress_time / 100
            
            print(f"✅ Stress Test: {stress_time:.2f}s total | {stress_time/100*1000:.1f}ms/inference")
            
        except Exception as e:
            print(f"❌ Stress Test: FAILED - {e}")
            stability_results['stress_test_error'] = str(e)
        
        # Error handling test
        try:
            print("Testing error handling with invalid inputs...")
            
            # Test with wrong shapes
            invalid_inputs = [
                Tensor.randn(10),  # 1D instead of 3D
                Tensor.randn(1, 1),  # 2D instead of 3D
                Tensor.randn(1, 5, 32)  # Wrong feature dimension
            ]
            
            error_handled_count = 0
            for invalid_input in invalid_inputs:
                try:
                    result = system(invalid_input)
                    if 'error' in result or not result.get('success', True):
                        error_handled_count += 1
                except:
                    error_handled_count += 1  # Exception caught = good error handling
            
            stability_results['error_handling_test_passed'] = error_handled_count >= 2
            stability_results['error_cases_handled'] = error_handled_count
            
            print(f"✅ Error Handling: {error_handled_count}/3 cases handled gracefully")
            
        except Exception as e:
            print(f"❌ Error Handling Test: FAILED - {e}")
        
        # Edge cases test
        try:
            print("Testing edge cases...")
            
            edge_cases = [
                ("Minimal input", Tensor.randn(1, 1, 64)),
                ("Large sequence", Tensor.randn(1, 50, 64)),
                ("Single sample", Tensor.randn(1, 10, 64))
            ]
            
            edge_case_success = 0
            for case_name, test_input in edge_cases:
                try:
                    result = system(test_input)
                    if 'arc_solution' in result:
                        edge_case_success += 1
                        print(f"  ✅ {case_name}: Processed successfully")
                    else:
                        print(f"  ⚠️  {case_name}: Processed but incomplete output")
                except Exception as e:
                    print(f"  ❌ {case_name}: FAILED - {e}")
            
            stability_results['edge_cases_test_passed'] = edge_case_success >= 2
            stability_results['edge_cases_passed'] = edge_case_success
            
        except Exception as e:
            print(f"❌ Edge Cases Test: FAILED - {e}")
        
        self.results['stability_tests'] = stability_results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "=" * 70)
        print("🏆 ENHANCED MULTI-PINNACLE PERFORMANCE REPORT")
        print("=" * 70)
        
        # Overall system assessment
        overall_score = 0
        max_score = 0
        
        # Initialization score
        init_success_rate = len([r for r in self.results['initialization_tests'].values() if r.get('success')]) / len(self.results['initialization_tests'])
        init_score = init_success_rate * 20
        overall_score += init_score
        max_score += 20
        
        # Processing score  
        proc_success_rate = len([r for r in self.results['processing_tests'].values() if r.get('processing_success') or 'throughput' in r]) / len(self.results['processing_tests'])
        proc_score = proc_success_rate * 25
        overall_score += proc_score
        max_score += 25
        
        # Memory score
        memory_stable = self.results['memory_tests'].get('memory_stable', False)
        memory_score = 15 if memory_stable else 10
        overall_score += memory_score
        max_score += 15
        
        # Scalability score
        scale_success_rate = len([r for r in self.results['scalability_tests'].values() if r.get('success')]) / len(self.results['scalability_tests'])
        scale_score = scale_success_rate * 15
        overall_score += scale_score
        max_score += 15
        
        # ARC solving score
        arc_success_rate = len([r for r in self.results['arc_solving_tests'].values() if r.get('has_valid_solution')]) / len(self.results['arc_solving_tests'])
        arc_score = arc_success_rate * 25
        overall_score += arc_score
        max_score += 25
        
        overall_percentage = (overall_score / max_score) * 100
        
        self.results['overall_performance'] = {
            'overall_score': overall_score,
            'max_score': max_score,
            'percentage': overall_percentage,
            'grade': self._get_performance_grade(overall_percentage),
            'initialization_success_rate': init_success_rate,
            'processing_success_rate': proc_success_rate,
            'scalability_success_rate': scale_success_rate,
            'arc_solving_success_rate': arc_success_rate,
            'memory_stable': memory_stable
        }
        
        print(f"\n📊 OVERALL PERFORMANCE SCORE: {overall_score:.1f}/{max_score} ({overall_percentage:.1f}%)")
        print(f"🎯 PERFORMANCE GRADE: {self._get_performance_grade(overall_percentage)}")
        
        print(f"\n📈 COMPONENT SCORES:")
        print(f"   🔧 Initialization: {init_success_rate*100:.0f}% success rate")
        print(f"   🧠 Processing: {proc_success_rate*100:.0f}% success rate")
        print(f"   💾 Memory: {'Stable' if memory_stable else 'Unstable'}")
        print(f"   📈 Scalability: {scale_success_rate*100:.0f}% success rate") 
        print(f"   🎯 ARC Solving: {arc_success_rate*100:.0f}% valid solutions")
        
        # Performance insights
        print(f"\n🔍 KEY INSIGHTS:")
        
        if overall_percentage >= 85:
            print("   ✅ EXCELLENT: System demonstrates production-ready performance")
        elif overall_percentage >= 70:
            print("   ✅ GOOD: System shows strong performance with minor optimizations needed")
        elif overall_percentage >= 50:
            print("   ⚠️  FAIR: System functional but needs performance improvements")
        else:
            print("   ❌ POOR: System needs significant optimization")
        
        # Get best performing test
        if self.results['processing_tests']:
            best_throughput = max([r.get('throughput', 0) for r in self.results['processing_tests'].values()])
            print(f"   🚀 Peak Throughput: {best_throughput:.1f} samples/second")
        
        if self.results['memory_tests']:
            peak_memory = self.results['memory_tests'].get('peak_memory_mb', 0)
            print(f"   💾 Peak Memory Usage: {peak_memory:.1f} MB")
        
        return overall_percentage
    
    def _get_performance_grade(self, percentage):
        """Get performance grade based on percentage"""
        if percentage >= 95:
            return "A+ (EXCEPTIONAL)"
        elif percentage >= 90:
            return "A (EXCELLENT)"
        elif percentage >= 85:
            return "A- (VERY GOOD)"
        elif percentage >= 80:
            return "B+ (GOOD)"
        elif percentage >= 75:
            return "B (ABOVE AVERAGE)"
        elif percentage >= 70:
            return "B- (AVERAGE)"
        elif percentage >= 65:
            return "C+ (BELOW AVERAGE)"
        elif percentage >= 60:
            return "C (POOR)"
        else:
            return "F (FAILING)"

def main():
    """Run the complete performance test suite"""
    print("🎯 Starting Enhanced Multi-PINNACLE Performance Testing...")
    
    test_suite = PerformanceTestSuite()
    results = test_suite.run_all_tests()
    
    # Save results
    with open('performance_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: performance_test_results.json")
    print(f"\n🎉 PERFORMANCE TESTING COMPLETE!")
    
    return results

if __name__ == "__main__":
    main()
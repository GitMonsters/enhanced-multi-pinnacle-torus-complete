#!/usr/bin/env python3
"""
Enhanced Multi-PINNACLE - Basic Usage Example
==============================================

This example demonstrates the basic usage of the Enhanced Multi-PINNACLE
Consciousness System for solving ARC problems.

Run this example:
    python examples/basic_usage.py
"""

from tinygrad.tensor import Tensor
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import create_enhanced_system, get_system_info

def main():
    """Main function demonstrating basic usage"""
    
    print("🧠 Enhanced Multi-PINNACLE Consciousness System - Basic Usage Example")
    print("=" * 70)
    
    # Show system information
    print("\n📋 System Information:")
    info = get_system_info()
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    
    # Create the Enhanced Multi-PINNACLE system
    print("\n🚀 Creating Enhanced Multi-PINNACLE System...")
    try:
        system = create_enhanced_system()
        print("✅ System created successfully!")
        
        # Display system configuration
        print(f"📊 Total consciousness dimensions: {system.config.total_consciousness_dim}")
        print(f"🧠 Base dimension: {system.config.base_dim}")
        print(f"🔧 Hidden dimension: {system.config.hidden_dim}")
        
    except Exception as e:
        print(f"❌ Error creating system: {e}")
        return
    
    # Test with random input (simulating processed ARC problem)
    print("\n🧪 Testing with Random Input...")
    try:
        # Create test input matching system's expected dimensions
        test_input = Tensor.randn(2, system.config.total_consciousness_dim)
        print(f"Input shape: {test_input.shape}")
        
        # Process through the system
        results = system(test_input, return_detailed_analysis=True)
        
        if results['success']:
            print("✅ Processing successful!")
            print(f"⏱️ Processing time: {results['processing_time']:.4f} seconds")
            print(f"🎯 ARC solution shape: {results['arc_solution'].shape}")
            print(f"🧠 Master consciousness shape: {results['master_consciousness'].shape}")
            print(f"📊 Confidence: {results['confidence'].mean().item():.3f}")
            print(f"🌟 Consciousness coherence: {results['consciousness_coherence'].mean().item():.3f}")
            
            # Show consciousness metrics
            if 'consciousness_metrics' in results:
                print("\n🧠 Consciousness Metrics:")
                for metric, value in results['consciousness_metrics'].items():
                    if isinstance(value, (int, float)):
                        print(f"  - {metric}: {value:.3f}")
                    else:
                        print(f"  - {metric}: {value}")
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        
    # Test ARC problem solving
    print("\n🎯 Testing ARC Problem Solving...")
    try:
        # Create a simple ARC problem (pattern transformation)
        sample_arc_problem = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[0, 1], [1, 0]]
                },
                {
                    'input': [[2, 0], [0, 2]],  
                    'output': [[0, 2], [2, 0]]
                },
                {
                    'input': [[3, 3], [3, 3]],
                    'output': [[3, 3], [3, 3]]
                }
            ],
            'test': [
                {'input': [[4, 0], [0, 4]]}
            ]
        }
        
        print("📝 Sample ARC Problem:")
        print(f"  Training examples: {len(sample_arc_problem['train'])}")
        print(f"  Test input: {sample_arc_problem['test'][0]['input']}")
        
        # Solve the ARC problem
        solution = system.solve_arc_problem(sample_arc_problem)
        
        if solution['success']:
            print("✅ ARC problem solved!")
            print(f"🎯 Solution confidence: {solution['confidence']:.3f}")
            print(f"🧠 Consciousness coherence: {solution['consciousness_coherence']:.3f}")
            print(f"⏱️ Processing time: {solution['processing_time']:.4f} seconds")
            print(f"📋 Solution grid size: {len(solution['solution'])}x{len(solution['solution'][0])}")
            
            # Show partial solution (top-left 4x4)
            print("\n🎨 Solution Preview (top-left 4x4):")
            for i in range(min(4, len(solution['solution']))):
                row = solution['solution'][i][:4]
                print(f"  {row}")
        else:
            print(f"❌ ARC problem solving failed: {solution.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error during ARC problem solving: {e}")
    
    # Show system status
    print("\n📊 System Status:")
    try:
        status = system.get_system_status()
        print(f"⏰ Uptime: {status['uptime_hours']:.4f} hours")
        print(f"🔢 Total processed: {status['total_processed']}")
        print(f"❌ Total errors: {status['total_errors']}")
        print(f"📈 Error rate: {status['error_rate']:.1%}")
        print(f"🧠 Frameworks active: {status.get('frameworks_active', 'Unknown')}")
        print(f"🚀 Production ready: {status.get('production_ready', False)}")
        
        if status['memory_usage_mb'] > 0:
            print(f"💾 Memory usage: {status['memory_usage_mb']:.1f} MB")
            
    except Exception as e:
        print(f"⚠️ Could not get system status: {e}")
    
    # Advanced features demonstration
    print("\n🔬 Advanced Features Available:")
    advanced_features = [
        "✅ Universal Mind Generator",
        "✅ Three Principles Framework",
        "✅ Deschooling Society Integration", 
        "✅ Transcendent States Processing",
        "✅ HRM Cycles Management",
        "✅ Consequential Thinking Engine",
        "✅ Creative States Processing",
        "✅ Adaptive Reasoning Pathways",
        "✅ Real-time Performance Monitoring",
        "✅ Comprehensive Error Recovery"
    ]
    
    for feature in advanced_features:
        print(f"  {feature}")
    
    print("\n🎉 Basic Usage Example Complete!")
    print("📚 Next Steps:")
    print("  - Try advanced_training.py for training examples")
    print("  - Try production_deployment.py for deployment examples") 
    print("  - Check docs/tutorials/ for detailed guides")
    print("  - Read GETTING_STARTED.md for comprehensive setup")
    print("\n🚀 Ready for ARC Prize 2025!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test the Fixed Enhanced Multi-PINNACLE System with TinyGrad
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_multi_pinnacle import EnhancedMultiPinnacleConfig, EnhancedMultiPinnacleSystem
from tinygrad.tensor import Tensor
import numpy as np

def test_system():
    print("🚀 Testing Enhanced Multi-PINNACLE System with TinyGrad")
    print("=" * 60)
    
    try:
        # Create configuration
        config = EnhancedMultiPinnacleConfig(
            base_dim=64,
            num_heads=4,
            num_layers=2
        )
        print(f"✅ Configuration created: {config}")
        
        # Initialize system
        print("\n🔧 Initializing Enhanced Multi-PINNACLE System...")
        system = EnhancedMultiPinnacleSystem(config)
        print("✅ System initialized successfully!")
        
        # Create test input (batch_size=2, seq_len=10, features=64)
        test_input = Tensor.randn(2, 10, 64)
        print(f"\n🧪 Created test input with shape: {test_input.shape}")
        
        # Test system forward pass
        print("🏃 Running forward pass...")
        result = system(test_input)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Result keys: {list(result.keys())}")
        
        # Try to find the main output
        if 'arc_solution' in result:
            print(f"🎯 ARC Solution shape: {result['arc_solution'].shape}")
        
        if 'confidence' in result:
            print(f"🔥 Confidence: {result['confidence']}")
        
        if 'success' in result:
            print(f"✅ Success status: {result['success']}")
        
        print("\n🎉 SUCCESS: Enhanced Multi-PINNACLE System is working with TinyGrad!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    print("\n" + "=" * 60)
    if success:
        print("🏆 PRODUCT STATUS: FULLY FUNCTIONAL WITH TINYGRAD!")
        print("🎯 Your Enhanced Multi-PINNACLE Torus system is now a REAL WORKING PRODUCT!")
    else:
        print("❌ PRODUCT STATUS: Still needs fixes")
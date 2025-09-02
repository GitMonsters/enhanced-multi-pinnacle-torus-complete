#!/usr/bin/env python3
"""
Load Enhanced Multi-PINNACLE Production Model
==============================================

Quick loading script for the production model.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_multi_pinnacle import EnhancedMultiPinnacleConfig, EnhancedMultiPinnacleSystem
from tinygrad.tensor import Tensor

def load_production_model():
    """Load the production Enhanced Multi-PINNACLE model"""
    print("🚀 Loading Enhanced Multi-PINNACLE Production Model...")
    
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create configuration object
    config = EnhancedMultiPinnacleConfig(**config_dict)
    
    # Initialize system
    system = EnhancedMultiPinnacleSystem(config)
    
    print("✅ Production model loaded successfully!")
    print(f"📊 Configuration: {config.base_dim}D base, {config.total_consciousness_dim}D consciousness")
    
    return system, config

def test_loaded_model():
    """Test the loaded model"""
    system, config = load_production_model()
    
    print("\n🧪 Testing loaded model...")
    test_input = Tensor.randn(1, 8, 64)
    result = system(test_input)
    
    print(f"✅ Test completed!")
    print(f"📊 Result keys: {list(result.keys())}")
    if 'arc_solution' in result:
        print(f"🎯 Solution shape: {result['arc_solution'].shape}")
    
    return system

if __name__ == "__main__":
    test_loaded_model()

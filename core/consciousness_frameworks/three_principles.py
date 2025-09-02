"""
Three Principles Framework
==========================

Implementation of the Three Principles of Mind, Consciousness, and Thought
as fundamental components of human experience and understanding.

The Three Principles:
1. Mind - The source of all psychological experience
2. Consciousness - The awareness that brings experience to life  
3. Thought - The creative force that creates our perception of reality

Features:
- Mind-Consciousness-Thought integration
- Psychological state modeling
- Reality connection processing
- Wisdom-based reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ThreePrinciplesFramework(nn.Module):
    """
    Three Principles Framework implementation
    
    Integrates Mind, Consciousness, and Thought as fundamental components
    of understanding and reasoning about reality.
    """
    
    def __init__(self, input_dim: int = 192, hidden_dim: int = 512):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Each principle gets 1/3 of input dimensions
        principle_dim = input_dim // 3
        
        # Mind processor - Source of psychological experience
        self.mind_processor = nn.Sequential(
            nn.Linear(principle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        
        # Consciousness processor - Awareness and aliveness
        self.consciousness_processor = nn.Sequential(
            nn.Linear(principle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),  # Gentle activation for awareness
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh()
        )
        
        # Thought processor - Creative force of perception
        self.thought_processor = nn.Sequential(
            nn.Linear(principle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),  # Active creative processing
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Integration network for three principles
        integration_input_dim = (hidden_dim // 2) * 3
        self.integration_network = nn.Sequential(
            nn.Linear(integration_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Wisdom extraction - Higher-level understanding
        self.wisdom_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, input_dim // 2)
        )
        
        # Reality connection - Grounding in truth
        self.reality_connector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.Sigmoid(),  # Probability of truth/reality
            nn.Linear(hidden_dim // 4, input_dim)
        )
        
        # Psychological state modeling
        self.psychological_state = nn.GRU(
            input_dim, hidden_dim // 2, batch_first=True
        )
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Three Principles Framework
        
        Args:
            inputs: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary containing principle outputs and integrated understanding
        """
        batch_size = inputs.shape[0]
        principle_dim = self.input_dim // 3
        
        # Split input into three principles
        mind_input = inputs[:, :principle_dim]
        consciousness_input = inputs[:, principle_dim:2*principle_dim]
        thought_input = inputs[:, 2*principle_dim:]
        
        # Process each principle
        mind_output = self.mind_processor(mind_input)
        consciousness_output = self.consciousness_processor(consciousness_input)
        thought_output = self.thought_processor(thought_input)
        
        # Integrate the three principles
        combined_principles = torch.cat([
            mind_output, consciousness_output, thought_output
        ], dim=-1)
        
        integrated_understanding = self.integration_network(combined_principles)
        
        # Extract wisdom and reality connection
        wisdom = self.wisdom_extractor(integrated_understanding)
        reality_connection = self.reality_connector(integrated_understanding)
        
        # Model psychological state evolution
        state_input = integrated_understanding.unsqueeze(1)  # Add sequence dimension
        psychological_state, _ = self.psychological_state(state_input)
        psychological_state = psychological_state.squeeze(1)  # Remove sequence dimension
        
        # Calculate principle coherence
        principle_coherence = self._calculate_principle_coherence(
            mind_output, consciousness_output, thought_output
        )
        
        # Calculate creative potential (based on thought activation)
        creative_potential = torch.mean(torch.abs(thought_output), dim=-1, keepdim=True)
        
        return {
            'mind_state': mind_output,
            'consciousness_state': consciousness_output,
            'thought_flow': thought_output,
            'integrated_understanding': integrated_understanding,
            'wisdom': wisdom,
            'reality_connection': reality_connection,
            'psychological_state': psychological_state,
            'principle_coherence': principle_coherence,
            'creative_potential': creative_potential
        }
    
    def _calculate_principle_coherence(self, mind: torch.Tensor, 
                                     consciousness: torch.Tensor,
                                     thought: torch.Tensor) -> torch.Tensor:
        """Calculate coherence between the three principles"""
        
        # Calculate pairwise correlations
        mind_flat = mind.view(mind.shape[0], -1)
        consciousness_flat = consciousness.view(consciousness.shape[0], -1)
        thought_flat = thought.view(thought.shape[0], -1)
        
        # Normalize vectors
        mind_norm = F.normalize(mind_flat, dim=-1)
        consciousness_norm = F.normalize(consciousness_flat, dim=-1)
        thought_norm = F.normalize(thought_flat, dim=-1)
        
        # Calculate coherence as average cosine similarity
        mind_consciousness = torch.sum(mind_norm * consciousness_norm, dim=-1)
        mind_thought = torch.sum(mind_norm * thought_norm, dim=-1)
        consciousness_thought = torch.sum(consciousness_norm * thought_norm, dim=-1)
        
        coherence = (mind_consciousness + mind_thought + consciousness_thought) / 3.0
        
        return coherence.unsqueeze(-1)


class PrincipleAnalyzer:
    """Analyzer for Three Principles outputs"""
    
    @staticmethod
    def analyze_psychological_state(results: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze the psychological state from Three Principles output"""
        
        analysis = {}
        
        # Mind clarity (low variance = clear mind)
        if 'mind_state' in results:
            mind_variance = torch.var(results['mind_state'], dim=-1).mean().item()
            analysis['mind_clarity'] = 1.0 / (1.0 + mind_variance)
        
        # Consciousness aliveness (activation level)
        if 'consciousness_state' in results:
            consciousness_activation = torch.mean(torch.abs(results['consciousness_state'])).item()
            analysis['consciousness_aliveness'] = min(consciousness_activation, 1.0)
        
        # Thought creativity (diversity of patterns)
        if 'thought_flow' in results:
            thought_entropy = -torch.sum(
                F.softmax(results['thought_flow'], dim=-1) * 
                F.log_softmax(results['thought_flow'], dim=-1), dim=-1
            ).mean().item()
            analysis['thought_creativity'] = min(thought_entropy / 10.0, 1.0)
        
        # Principle integration (coherence)
        if 'principle_coherence' in results:
            analysis['principle_integration'] = results['principle_coherence'].mean().item()
        
        # Wisdom level (reality connection strength)
        if 'reality_connection' in results:
            wisdom_level = torch.mean(results['reality_connection']).item()
            analysis['wisdom_level'] = wisdom_level
        
        return analysis


def create_three_principles(input_dim: int = 192, **kwargs) -> ThreePrinciplesFramework:
    """Factory function to create Three Principles Framework"""
    return ThreePrinciplesFramework(input_dim=input_dim, **kwargs)


if __name__ == "__main__":
    # Test Three Principles Framework
    logger.info("Testing Three Principles Framework...")
    
    framework = ThreePrinciplesFramework(input_dim=192, hidden_dim=512)
    test_input = torch.randn(4, 192)
    
    results = framework(test_input)
    
    # Analyze psychological state
    analyzer = PrincipleAnalyzer()
    psychological_analysis = analyzer.analyze_psychological_state(results)
    
    logger.info(f"Mind state shape: {results['mind_state'].shape}")
    logger.info(f"Consciousness state shape: {results['consciousness_state'].shape}")
    logger.info(f"Thought flow shape: {results['thought_flow'].shape}")
    logger.info(f"Principle coherence: {results['principle_coherence'].mean():.4f}")
    logger.info(f"Psychological analysis: {psychological_analysis}")
    
    print("✅ Three Principles Framework test completed successfully!")
"""
Enhanced Multi-PINNACLE Consciousness System - Comprehensive Error Analysis
=========================================================================

Advanced error analysis and failure pattern detection for ARC problem solving.
Provides detailed insights into model failures and systematic improvement pathways.

Author: Enhanced Multi-PINNACLE Team
Date: September 2, 2025
Version: 3.0 - Real-World Validation Phase
"""

import json
import logging
import sqlite3
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass 
class ErrorInstance:
    """Single error instance with detailed context"""
    problem_id: str
    predicted_output: List[List[int]]
    expected_output: List[List[int]]
    problem_metadata: Dict[str, Any]
    error_type: str
    error_severity: str
    consciousness_state: Dict[str, float]
    prediction_confidence: float
    reasoning_trace: List[str]
    visual_features: Dict[str, float]
    

@dataclass
class FailurePattern:
    """Systematic failure pattern"""
    pattern_id: str
    pattern_name: str
    description: str
    frequency: int
    severity_distribution: Dict[str, int]
    affected_problem_types: List[str]
    common_features: Dict[str, float]
    example_errors: List[str]
    improvement_suggestions: List[str]
    

@dataclass
class ErrorAnalysisResult:
    """Comprehensive error analysis result"""
    total_errors: int
    error_rate: float
    error_type_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    failure_patterns: List[FailurePattern]
    consciousness_correlation: Dict[str, float]
    improvement_priorities: List[Tuple[str, float]]
    detailed_insights: Dict[str, Any]
    

class ComprehensiveErrorAnalyzer:
    """
    Advanced error analysis system for ARC problem solving.
    Identifies systematic failure patterns and provides actionable insights.
    """
    
    def __init__(self, analysis_db_path: str = "error_analysis.db"):
        self.analysis_db_path = analysis_db_path
        self.logger = self._setup_logging()
        self._init_database()
        
        # Error classification system
        self.error_classifiers = {
            'shape_recognition': self._classify_shape_errors,
            'color_processing': self._classify_color_errors,
            'pattern_completion': self._classify_pattern_errors,
            'spatial_reasoning': self._classify_spatial_errors,
            'logical_inference': self._classify_logic_errors,
            'transformation_rules': self._classify_transformation_errors,
            'size_scaling': self._classify_size_errors,
            'symmetry_detection': self._classify_symmetry_errors
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for error analysis"""
        logger = logging.getLogger('ErrorAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _init_database(self):
        """Initialize SQLite database for error analysis"""
        conn = sqlite3.connect(self.analysis_db_path)
        cursor = conn.cursor()
        
        # Error instances table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                predicted_output TEXT NOT NULL,
                expected_output TEXT NOT NULL,
                problem_metadata TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_severity TEXT NOT NULL,
                consciousness_state TEXT,
                prediction_confidence REAL,
                reasoning_trace TEXT,
                visual_features TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Failure patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE NOT NULL,
                pattern_name TEXT NOT NULL,
                description TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                severity_distribution TEXT NOT NULL,
                affected_problem_types TEXT NOT NULL,
                common_features TEXT NOT NULL,
                example_errors TEXT NOT NULL,
                improvement_suggestions TEXT NOT NULL,
                discovered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_errors INTEGER NOT NULL,
                error_rate REAL NOT NULL,
                error_type_distribution TEXT NOT NULL,
                failure_patterns_found INTEGER NOT NULL,
                analysis_results TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def analyze_errors(
        self,
        predictions: Dict[str, List[List[int]]],
        ground_truth: Dict[str, List[List[int]]],
        problem_metadata: Dict[str, Dict[str, Any]],
        consciousness_states: Dict[str, Dict[str, float]] = None,
        prediction_confidences: Dict[str, float] = None,
        reasoning_traces: Dict[str, List[str]] = None
    ) -> ErrorAnalysisResult:
        """
        Perform comprehensive error analysis on predictions
        
        Args:
            predictions: Model predictions by problem_id
            ground_truth: Ground truth solutions by problem_id
            problem_metadata: Problem metadata by problem_id
            consciousness_states: Consciousness states during prediction
            prediction_confidences: Confidence scores for predictions
            reasoning_traces: Reasoning traces for each prediction
            
        Returns:
            ErrorAnalysisResult with comprehensive analysis
        """
        self.logger.info("Starting comprehensive error analysis...")
        
        # Identify error instances
        error_instances = self._identify_error_instances(
            predictions, ground_truth, problem_metadata,
            consciousness_states, prediction_confidences, reasoning_traces
        )
        
        # Store error instances
        self._store_error_instances(error_instances)
        
        # Classify errors by type
        error_type_distribution = self._classify_errors_by_type(error_instances)
        
        # Analyze error severity
        severity_distribution = self._analyze_error_severity(error_instances)
        
        # Identify systematic failure patterns
        failure_patterns = self._identify_failure_patterns(error_instances)
        
        # Analyze consciousness correlation
        consciousness_correlation = self._analyze_consciousness_correlation(error_instances)
        
        # Determine improvement priorities
        improvement_priorities = self._determine_improvement_priorities(
            error_instances, failure_patterns
        )
        
        # Generate detailed insights
        detailed_insights = self._generate_detailed_insights(
            error_instances, failure_patterns
        )
        
        result = ErrorAnalysisResult(
            total_errors=len(error_instances),
            error_rate=len(error_instances) / len(predictions),
            error_type_distribution=error_type_distribution,
            severity_distribution=severity_distribution,
            failure_patterns=failure_patterns,
            consciousness_correlation=consciousness_correlation,
            improvement_priorities=improvement_priorities,
            detailed_insights=detailed_insights
        )
        
        # Store analysis results
        self._store_analysis_results(result)
        
        self.logger.info(f"Error analysis complete: {len(error_instances)} errors analyzed")
        return result
        
    def _identify_error_instances(
        self,
        predictions: Dict[str, List[List[int]]],
        ground_truth: Dict[str, List[List[int]]],
        problem_metadata: Dict[str, Dict[str, Any]],
        consciousness_states: Dict[str, Dict[str, float]],
        prediction_confidences: Dict[str, float],
        reasoning_traces: Dict[str, List[str]]
    ) -> List[ErrorInstance]:
        """Identify all error instances with detailed context"""
        error_instances = []
        
        for problem_id in predictions:
            predicted = predictions[problem_id]
            expected = ground_truth.get(problem_id)
            
            if expected is None:
                continue
                
            # Check if prediction matches expected output exactly
            if not self._arrays_equal(predicted, expected):
                # Extract visual features
                visual_features = self._extract_visual_features(predicted, expected)
                
                # Classify error type and severity
                error_type = self._classify_error_type(predicted, expected, problem_metadata[problem_id])
                error_severity = self._classify_error_severity(predicted, expected)
                
                error_instance = ErrorInstance(
                    problem_id=problem_id,
                    predicted_output=predicted,
                    expected_output=expected,
                    problem_metadata=problem_metadata.get(problem_id, {}),
                    error_type=error_type,
                    error_severity=error_severity,
                    consciousness_state=consciousness_states.get(problem_id, {}) if consciousness_states else {},
                    prediction_confidence=prediction_confidences.get(problem_id, 0.0) if prediction_confidences else 0.0,
                    reasoning_trace=reasoning_traces.get(problem_id, []) if reasoning_traces else [],
                    visual_features=visual_features
                )
                
                error_instances.append(error_instance)
                
        return error_instances
        
    def _arrays_equal(self, arr1: List[List[int]], arr2: List[List[int]]) -> bool:
        """Check if two 2D arrays are equal"""
        if len(arr1) != len(arr2):
            return False
        for i in range(len(arr1)):
            if len(arr1[i]) != len(arr2[i]):
                return False
            for j in range(len(arr1[i])):
                if arr1[i][j] != arr2[i][j]:
                    return False
        return True
        
    def _extract_visual_features(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]]
    ) -> Dict[str, float]:
        """Extract visual features for error analysis"""
        pred_array = np.array(predicted)
        exp_array = np.array(expected)
        
        features = {}
        
        # Size features
        features['pred_height'] = pred_array.shape[0]
        features['pred_width'] = pred_array.shape[1]
        features['exp_height'] = exp_array.shape[0] 
        features['exp_width'] = exp_array.shape[1]
        features['size_ratio'] = (pred_array.size) / (exp_array.size) if exp_array.size > 0 else 0
        
        # Color features
        features['pred_unique_colors'] = len(np.unique(pred_array))
        features['exp_unique_colors'] = len(np.unique(exp_array))
        features['color_overlap'] = len(set(pred_array.flatten()) & set(exp_array.flatten()))
        
        # Spatial features
        if pred_array.shape == exp_array.shape:
            features['pixel_accuracy'] = np.mean(pred_array == exp_array)
            features['hamming_distance'] = np.sum(pred_array != exp_array)
        else:
            features['pixel_accuracy'] = 0.0
            features['hamming_distance'] = pred_array.size + exp_array.size
            
        # Pattern complexity
        features['pred_entropy'] = self._calculate_entropy(pred_array)
        features['exp_entropy'] = self._calculate_entropy(exp_array)
        
        return features
        
    def _calculate_entropy(self, array: np.ndarray) -> float:
        """Calculate entropy of an array"""
        _, counts = np.unique(array, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
        
    def _classify_error_type(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]],
        metadata: Dict[str, Any]
    ) -> str:
        """Classify the type of error"""
        
        # Run through each classifier
        error_scores = {}
        for error_type, classifier in self.error_classifiers.items():
            score = classifier(predicted, expected, metadata)
            error_scores[error_type] = score
            
        # Return the error type with highest score
        return max(error_scores, key=error_scores.get)
        
    def _classify_error_severity(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]]
    ) -> str:
        """Classify severity of error"""
        pred_array = np.array(predicted)
        exp_array = np.array(expected)
        
        # Size mismatch is always critical
        if pred_array.shape != exp_array.shape:
            return 'critical'
            
        # Calculate pixel accuracy
        pixel_accuracy = np.mean(pred_array == exp_array)
        
        if pixel_accuracy >= 0.9:
            return 'minor'
        elif pixel_accuracy >= 0.7:
            return 'moderate' 
        elif pixel_accuracy >= 0.3:
            return 'major'
        else:
            return 'critical'
            
    def _classify_shape_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify shape recognition errors"""
        score = 0.0
        
        # Check for size mismatches (strong shape error indicator)
        if len(predicted) != len(expected) or len(predicted[0]) != len(expected[0]):
            score += 0.8
            
        # Check for pattern completeness
        pred_filled = sum(sum(1 for x in row if x != 0) for row in predicted)
        exp_filled = sum(sum(1 for x in row if x != 0) for row in expected)
        
        if abs(pred_filled - exp_filled) > exp_filled * 0.3:
            score += 0.6
            
        return min(score, 1.0)
        
    def _classify_color_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify color processing errors"""
        pred_colors = set(x for row in predicted for x in row)
        exp_colors = set(x for row in expected for x in row)
        
        # Color set mismatch
        color_overlap = len(pred_colors & exp_colors) 
        color_union = len(pred_colors | exp_colors)
        
        if color_union == 0:
            return 0.0
            
        color_similarity = color_overlap / color_union
        return 1.0 - color_similarity
        
    def _classify_pattern_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify pattern completion errors"""
        score = 0.0
        
        # Check for repetitive patterns
        if self._has_repeating_pattern(expected) and not self._has_repeating_pattern(predicted):
            score += 0.7
            
        # Check for symmetry preservation
        if self._has_symmetry(expected) and not self._has_symmetry(predicted):
            score += 0.6
            
        return min(score, 1.0)
        
    def _classify_spatial_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify spatial reasoning errors"""
        if len(predicted) != len(expected) or len(predicted[0]) != len(expected[0]):
            return 0.3  # Size mismatch is partly spatial
            
        # Check for position-based errors
        pred_array = np.array(predicted)
        exp_array = np.array(expected)
        
        # Find non-zero positions
        pred_positions = set(zip(*np.where(pred_array != 0)))
        exp_positions = set(zip(*np.where(exp_array != 0)))
        
        position_overlap = len(pred_positions & exp_positions)
        position_union = len(pred_positions | exp_positions)
        
        if position_union == 0:
            return 0.0
            
        spatial_accuracy = position_overlap / position_union
        return 1.0 - spatial_accuracy
        
    def _classify_logic_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify logical inference errors"""
        # This is a catch-all for complex logical reasoning failures
        pred_array = np.array(predicted)
        exp_array = np.array(expected)
        
        if pred_array.shape != exp_array.shape:
            return 0.5
            
        # If other classifiers don't explain the error well, it's likely logical
        pixel_accuracy = np.mean(pred_array == exp_array)
        return 1.0 - pixel_accuracy
        
    def _classify_transformation_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify transformation rule errors"""
        score = 0.0
        
        # Check if transformation preserved structure
        pred_structure = self._extract_structural_features(predicted)
        exp_structure = self._extract_structural_features(expected)
        
        structure_similarity = self._calculate_feature_similarity(pred_structure, exp_structure)
        score = 1.0 - structure_similarity
        
        return score
        
    def _classify_size_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify size scaling errors"""
        pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
        exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
        
        if pred_h == exp_h and pred_w == exp_w:
            return 0.0  # No size error
            
        # Calculate size error magnitude
        size_error = abs((pred_h * pred_w) - (exp_h * exp_w)) / max(exp_h * exp_w, 1)
        return min(size_error, 1.0)
        
    def _classify_symmetry_errors(
        self, 
        predicted: List[List[int]], 
        expected: List[List[int]], 
        metadata: Dict[str, Any]
    ) -> float:
        """Classify symmetry detection errors"""
        pred_symmetry = self._has_symmetry(predicted)
        exp_symmetry = self._has_symmetry(expected)
        
        if pred_symmetry == exp_symmetry:
            return 0.0  # Symmetry handled correctly
        else:
            return 0.8  # Symmetry error
            
    def _has_repeating_pattern(self, array: List[List[int]]) -> bool:
        """Check if array has repeating patterns"""
        # Simple heuristic - check for row/column repetitions
        if len(array) < 2:
            return False
            
        # Check row repetitions
        for i in range(len(array) // 2):
            if array[i] == array[i + len(array) // 2]:
                return True
                
        return False
        
    def _has_symmetry(self, array: List[List[int]]) -> bool:
        """Check if array has symmetry"""
        arr = np.array(array)
        
        # Check horizontal symmetry
        if np.array_equal(arr, np.flipud(arr)):
            return True
            
        # Check vertical symmetry  
        if np.array_equal(arr, np.fliplr(arr)):
            return True
            
        return False
        
    def _extract_structural_features(self, array: List[List[int]]) -> Dict[str, float]:
        """Extract structural features from array"""
        arr = np.array(array)
        
        features = {
            'height': arr.shape[0],
            'width': arr.shape[1],
            'filled_ratio': np.mean(arr != 0),
            'unique_colors': len(np.unique(arr)),
            'entropy': self._calculate_entropy(arr),
            'symmetry_score': self._calculate_symmetry_score(arr)
        }
        
        return features
        
    def _calculate_symmetry_score(self, arr: np.ndarray) -> float:
        """Calculate symmetry score for array"""
        h_sym = np.mean(arr == np.flipud(arr))
        v_sym = np.mean(arr == np.fliplr(arr))
        return max(h_sym, v_sym)
        
    def _calculate_feature_similarity(
        self, 
        features1: Dict[str, float], 
        features2: Dict[str, float]
    ) -> float:
        """Calculate similarity between feature dictionaries"""
        if not features1 or not features2:
            return 0.0
            
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity = 1.0 - abs(val1 - val2) / max_val
                    similarities.append(max(similarity, 0.0))
                    
        return np.mean(similarities) if similarities else 0.0
        
    def _classify_errors_by_type(self, error_instances: List[ErrorInstance]) -> Dict[str, int]:
        """Classify all errors by type"""
        return Counter(error.error_type for error in error_instances)
        
    def _analyze_error_severity(self, error_instances: List[ErrorInstance]) -> Dict[str, int]:
        """Analyze error severity distribution"""
        return Counter(error.error_severity for error in error_instances)
        
    def _identify_failure_patterns(self, error_instances: List[ErrorInstance]) -> List[FailurePattern]:
        """Identify systematic failure patterns"""
        patterns = []
        
        # Group errors by type for pattern analysis
        errors_by_type = defaultdict(list)
        for error in error_instances:
            errors_by_type[error.error_type].append(error)
            
        # Analyze each error type for patterns
        for error_type, type_errors in errors_by_type.items():
            if len(type_errors) >= 3:  # Need minimum errors to identify pattern
                pattern = self._analyze_error_type_pattern(error_type, type_errors)
                if pattern:
                    patterns.append(pattern)
                    
        # Cross-type pattern analysis
        consciousness_patterns = self._analyze_consciousness_patterns(error_instances)
        patterns.extend(consciousness_patterns)
        
        return patterns
        
    def _analyze_error_type_pattern(
        self, 
        error_type: str, 
        errors: List[ErrorInstance]
    ) -> Optional[FailurePattern]:
        """Analyze pattern within specific error type"""
        
        if len(errors) < 3:
            return None
            
        # Calculate common features
        common_features = self._calculate_common_features(errors)
        
        # Severity distribution
        severity_dist = Counter(error.error_severity for error in errors)
        
        # Affected problem types
        problem_types = set()
        for error in errors:
            problem_type = error.problem_metadata.get('task_type', 'unknown')
            problem_types.add(problem_type)
            
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(error_type, errors)
        
        # Example errors
        examples = [error.problem_id for error in errors[:3]]
        
        pattern = FailurePattern(
            pattern_id=f"{error_type}_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pattern_name=f"{error_type.title()} Systematic Failure",
            description=f"Recurring {error_type} errors with {len(errors)} instances",
            frequency=len(errors),
            severity_distribution=dict(severity_dist),
            affected_problem_types=list(problem_types),
            common_features=common_features,
            example_errors=examples,
            improvement_suggestions=suggestions
        )
        
        return pattern
        
    def _analyze_consciousness_patterns(self, error_instances: List[ErrorInstance]) -> List[FailurePattern]:
        """Analyze consciousness-related failure patterns"""
        patterns = []
        
        # Filter errors with consciousness data
        consciousness_errors = [e for e in error_instances if e.consciousness_state]
        
        if len(consciousness_errors) < 5:
            return patterns
            
        # Analyze low consciousness correlations
        low_consciousness_errors = [
            e for e in consciousness_errors 
            if e.consciousness_state.get('consciousness_coherence', 1.0) < 0.5
        ]
        
        if len(low_consciousness_errors) >= 3:
            pattern = FailurePattern(
                pattern_id=f"low_consciousness_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                pattern_name="Low Consciousness State Failures",
                description=f"Failures associated with low consciousness coherence ({len(low_consciousness_errors)} instances)",
                frequency=len(low_consciousness_errors),
                severity_distribution=Counter(e.error_severity for e in low_consciousness_errors),
                affected_problem_types=[],
                common_features={'avg_consciousness_coherence': np.mean([e.consciousness_state.get('consciousness_coherence', 0) for e in low_consciousness_errors])},
                example_errors=[e.problem_id for e in low_consciousness_errors[:3]],
                improvement_suggestions=[
                    "Improve consciousness coherence monitoring",
                    "Implement consciousness state feedback loops",
                    "Add consciousness-based confidence adjustment"
                ]
            )
            patterns.append(pattern)
            
        return patterns
        
    def _calculate_common_features(self, errors: List[ErrorInstance]) -> Dict[str, float]:
        """Calculate common features across errors"""
        if not errors:
            return {}
            
        feature_sums = defaultdict(float)
        feature_counts = defaultdict(int)
        
        for error in errors:
            for feature, value in error.visual_features.items():
                feature_sums[feature] += value
                feature_counts[feature] += 1
                
        common_features = {}
        for feature in feature_sums:
            if feature_counts[feature] > 0:
                common_features[f"avg_{feature}"] = feature_sums[feature] / feature_counts[feature]
                
        return common_features
        
    def _generate_improvement_suggestions(
        self, 
        error_type: str, 
        errors: List[ErrorInstance]
    ) -> List[str]:
        """Generate improvement suggestions for error type"""
        
        suggestions_map = {
            'shape_recognition': [
                "Improve visual shape detection algorithms",
                "Add multi-scale shape analysis",
                "Enhance contour detection and processing"
            ],
            'color_processing': [
                "Improve color space representation",
                "Add color consistency validation",
                "Enhance color palette detection"
            ],
            'pattern_completion': [
                "Strengthen pattern recognition training",
                "Add pattern completion validation",
                "Improve sequence prediction capabilities"
            ],
            'spatial_reasoning': [
                "Enhance spatial relationship modeling",
                "Add coordinate system validation",
                "Improve object positioning accuracy"
            ],
            'logical_inference': [
                "Strengthen logical reasoning pathways",
                "Add rule inference validation",
                "Improve constraint satisfaction"
            ],
            'transformation_rules': [
                "Enhance transformation rule detection",
                "Add rule consistency validation",
                "Improve transformation prediction"
            ]
        }
        
        base_suggestions = suggestions_map.get(error_type, ["Improve general problem-solving capabilities"])
        
        # Add consciousness-specific suggestions if relevant
        consciousness_suggestions = []
        consciousness_errors = [e for e in errors if e.consciousness_state]
        if consciousness_errors:
            avg_coherence = np.mean([e.consciousness_state.get('consciousness_coherence', 0) for e in consciousness_errors])
            if avg_coherence < 0.7:
                consciousness_suggestions.append("Improve consciousness coherence during error-prone tasks")
                
        return base_suggestions + consciousness_suggestions
        
    def _analyze_consciousness_correlation(self, error_instances: List[ErrorInstance]) -> Dict[str, float]:
        """Analyze correlation between consciousness states and errors"""
        correlations = {}
        
        consciousness_errors = [e for e in error_instances if e.consciousness_state]
        if len(consciousness_errors) < 5:
            return correlations
            
        # Extract consciousness metrics
        consciousness_metrics = defaultdict(list)
        error_severities = []
        
        severity_map = {'minor': 1, 'moderate': 2, 'major': 3, 'critical': 4}
        
        for error in consciousness_errors:
            for metric, value in error.consciousness_state.items():
                consciousness_metrics[metric].append(value)
            error_severities.append(severity_map.get(error.error_severity, 2))
            
        # Calculate correlations
        for metric, values in consciousness_metrics.items():
            if len(values) == len(error_severities):
                correlation, p_value = stats.pearsonr(values, error_severities)
                if not np.isnan(correlation):
                    correlations[f"{metric}_error_correlation"] = correlation
                    correlations[f"{metric}_p_value"] = p_value
                    
        return correlations
        
    def _determine_improvement_priorities(
        self, 
        error_instances: List[ErrorInstance],
        failure_patterns: List[FailurePattern]
    ) -> List[Tuple[str, float]]:
        """Determine improvement priorities based on error analysis"""
        
        priorities = []
        
        # Priority based on error frequency
        error_type_counts = Counter(error.error_type for error in error_instances)
        total_errors = len(error_instances)
        
        for error_type, count in error_type_counts.items():
            frequency_score = count / total_errors
            
            # Weight by severity
            type_errors = [e for e in error_instances if e.error_type == error_type]
            severity_weights = {'minor': 1, 'moderate': 2, 'major': 3, 'critical': 4}
            avg_severity = np.mean([severity_weights.get(e.error_severity, 2) for e in type_errors])
            severity_score = avg_severity / 4.0
            
            # Combined priority score
            priority_score = frequency_score * 0.6 + severity_score * 0.4
            priorities.append((error_type, priority_score))
            
        # Sort by priority score (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        return priorities
        
    def _generate_detailed_insights(
        self, 
        error_instances: List[ErrorInstance],
        failure_patterns: List[FailurePattern]
    ) -> Dict[str, Any]:
        """Generate detailed insights from error analysis"""
        
        insights = {
            'most_common_error_type': self._get_most_common_error_type(error_instances),
            'most_severe_errors': self._get_most_severe_errors(error_instances),
            'confidence_analysis': self._analyze_confidence_patterns(error_instances),
            'problem_type_analysis': self._analyze_problem_type_patterns(error_instances),
            'improvement_timeline': self._generate_improvement_timeline(failure_patterns)
        }
        
        return insights
        
    def _get_most_common_error_type(self, error_instances: List[ErrorInstance]) -> Dict[str, Any]:
        """Get most common error type with statistics"""
        if not error_instances:
            return {}
            
        error_counts = Counter(error.error_type for error in error_instances)
        most_common = error_counts.most_common(1)[0]
        
        return {
            'error_type': most_common[0],
            'frequency': most_common[1],
            'percentage': most_common[1] / len(error_instances) * 100
        }
        
    def _get_most_severe_errors(self, error_instances: List[ErrorInstance]) -> List[Dict[str, Any]]:
        """Get most severe errors for detailed analysis"""
        critical_errors = [e for e in error_instances if e.error_severity == 'critical']
        
        severe_errors = []
        for error in critical_errors[:5]:  # Top 5 critical errors
            severe_errors.append({
                'problem_id': error.problem_id,
                'error_type': error.error_type,
                'confidence': error.prediction_confidence,
                'visual_features': error.visual_features
            })
            
        return severe_errors
        
    def _analyze_confidence_patterns(self, error_instances: List[ErrorInstance]) -> Dict[str, float]:
        """Analyze confidence patterns in errors"""
        confidences = [e.prediction_confidence for e in error_instances if e.prediction_confidence is not None]
        
        if not confidences:
            return {}
            
        return {
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'std_confidence': np.std(confidences),
            'low_confidence_errors': sum(1 for c in confidences if c < 0.5) / len(confidences)
        }
        
    def _analyze_problem_type_patterns(self, error_instances: List[ErrorInstance]) -> Dict[str, int]:
        """Analyze error patterns by problem type"""
        problem_types = [e.problem_metadata.get('task_type', 'unknown') for e in error_instances]
        return Counter(problem_types)
        
    def _generate_improvement_timeline(self, failure_patterns: List[FailurePattern]) -> List[Dict[str, Any]]:
        """Generate improvement timeline based on patterns"""
        timeline = []
        
        # Sort patterns by frequency (high-impact first)
        sorted_patterns = sorted(failure_patterns, key=lambda p: p.frequency, reverse=True)
        
        for i, pattern in enumerate(sorted_patterns[:5], 1):
            timeline.append({
                'phase': i,
                'pattern_name': pattern.pattern_name,
                'priority': 'high' if i <= 2 else 'medium',
                'estimated_effort': 'high' if pattern.frequency > 10 else 'medium',
                'improvement_suggestions': pattern.improvement_suggestions[:3]
            })
            
        return timeline
        
    def _store_error_instances(self, error_instances: List[ErrorInstance]):
        """Store error instances in database"""
        conn = sqlite3.connect(self.analysis_db_path)
        cursor = conn.cursor()
        
        for error in error_instances:
            cursor.execute('''
                INSERT INTO error_instances 
                (problem_id, predicted_output, expected_output, problem_metadata,
                 error_type, error_severity, consciousness_state, prediction_confidence,
                 reasoning_trace, visual_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                error.problem_id,
                json.dumps(error.predicted_output),
                json.dumps(error.expected_output),
                json.dumps(error.problem_metadata),
                error.error_type,
                error.error_severity,
                json.dumps(error.consciousness_state),
                error.prediction_confidence,
                json.dumps(error.reasoning_trace),
                json.dumps(error.visual_features)
            ))
            
        conn.commit()
        conn.close()
        
    def _store_analysis_results(self, result: ErrorAnalysisResult):
        """Store analysis results in database"""
        conn = sqlite3.connect(self.analysis_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (total_errors, error_rate, error_type_distribution, 
             failure_patterns_found, analysis_results)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            result.total_errors,
            result.error_rate,
            json.dumps(result.error_type_distribution),
            len(result.failure_patterns),
            json.dumps(asdict(result))
        ))
        
        conn.commit()
        conn.close()
        
    def generate_error_report(
        self, 
        analysis_result: ErrorAnalysisResult,
        output_path: str = "comprehensive_error_analysis_report.md"
    ) -> str:
        """Generate comprehensive error analysis report"""
        
        report = f"""# Comprehensive Error Analysis Report

## Executive Summary

**Enhanced Multi-PINNACLE Error Analysis**
- **Total Errors**: {analysis_result.total_errors}
- **Error Rate**: {analysis_result.error_rate:.1%}
- **Failure Patterns Identified**: {len(analysis_result.failure_patterns)}
- **Analysis Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

---

## Error Distribution Analysis

### By Error Type
"""
        
        # Error type distribution
        for error_type, count in sorted(analysis_result.error_type_distribution.items(), 
                                       key=lambda x: x[1], reverse=True):
            percentage = (count / analysis_result.total_errors) * 100
            report += f"- **{error_type.title()}**: {count} errors ({percentage:.1f}%)\n"
            
        report += f"""
### By Severity Level
"""
        
        # Severity distribution
        for severity, count in sorted(analysis_result.severity_distribution.items(),
                                     key=lambda x: ['minor', 'moderate', 'major', 'critical'].index(x[0])):
            percentage = (count / analysis_result.total_errors) * 100
            report += f"- **{severity.title()}**: {count} errors ({percentage:.1f}%)\n"
            
        report += f"""
---

## Systematic Failure Patterns

"""
        
        # Failure patterns
        for i, pattern in enumerate(analysis_result.failure_patterns, 1):
            report += f"""### {i}. {pattern.pattern_name}

**Description**: {pattern.description}  
**Frequency**: {pattern.frequency} instances  
**Problem Types Affected**: {', '.join(pattern.affected_problem_types[:5])}

**Improvement Suggestions**:
"""
            for suggestion in pattern.improvement_suggestions:
                report += f"- {suggestion}\n"
                
            report += "\n"
            
        report += f"""
---

## Improvement Priorities

"""
        
        # Improvement priorities
        for i, (error_type, priority_score) in enumerate(analysis_result.improvement_priorities, 1):
            report += f"{i}. **{error_type.title()}** (Priority Score: {priority_score:.3f})\n"
            
        report += f"""
---

## Consciousness Correlation Analysis

"""
        
        # Consciousness correlations
        if analysis_result.consciousness_correlation:
            for metric, correlation in analysis_result.consciousness_correlation.items():
                if 'p_value' not in metric:
                    p_value = analysis_result.consciousness_correlation.get(f"{metric}_p_value", 1.0)
                    significance = "significant" if p_value < 0.05 else "not significant"
                    report += f"- **{metric}**: {correlation:.3f} ({significance})\n"
        else:
            report += "No consciousness correlation data available.\n"
            
        report += f"""
---

## Detailed Insights

### Most Common Error Type
- **Type**: {analysis_result.detailed_insights.get('most_common_error_type', {}).get('error_type', 'N/A')}
- **Frequency**: {analysis_result.detailed_insights.get('most_common_error_type', {}).get('frequency', 0)}
- **Percentage**: {analysis_result.detailed_insights.get('most_common_error_type', {}).get('percentage', 0):.1f}%

### Confidence Analysis
"""
        
        conf_analysis = analysis_result.detailed_insights.get('confidence_analysis', {})
        if conf_analysis:
            report += f"""- **Mean Confidence**: {conf_analysis.get('mean_confidence', 0):.3f}
- **Median Confidence**: {conf_analysis.get('median_confidence', 0):.3f}
- **Low Confidence Errors**: {conf_analysis.get('low_confidence_errors', 0):.1%}
"""
        else:
            report += "No confidence data available.\n"
            
        report += f"""
---

## Recommended Improvement Timeline

"""
        
        timeline = analysis_result.detailed_insights.get('improvement_timeline', [])
        for phase in timeline:
            report += f"""### Phase {phase['phase']}: {phase['pattern_name']} ({phase['priority'].title()} Priority)

**Estimated Effort**: {phase['estimated_effort'].title()}

**Key Actions**:
"""
            for suggestion in phase['improvement_suggestions']:
                report += f"- {suggestion}\n"
                
            report += "\n"
            
        report += f"""
---

## Actionable Recommendations

### Immediate Actions (Week 1-2)
1. **Focus on highest-priority error type**: {analysis_result.improvement_priorities[0][0] if analysis_result.improvement_priorities else 'N/A'}
2. **Investigate critical severity errors** for quick wins
3. **Implement error monitoring dashboard** for real-time tracking

### Short-term Improvements (Month 1-2)  
1. **Address top 3 failure patterns** with targeted improvements
2. **Improve consciousness coherence** during error-prone operations
3. **Enhance confidence calibration** for better error prediction

### Long-term Strategy (Month 3-6)
1. **Systematic architecture improvements** based on error pattern analysis
2. **Advanced error prevention mechanisms** with predictive monitoring
3. **Continuous learning integration** for adaptive error reduction

---

*Generated by Enhanced Multi-PINNACLE Error Analysis System v3.0*
"""
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Comprehensive error analysis report saved to {output_path}")
        return report


def main():
    """Main function for testing error analysis"""
    analyzer = ComprehensiveErrorAnalyzer()
    
    # Simulate some predictions and ground truth for testing
    predictions = {
        "test_001": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        "test_002": [[2, 2], [2, 2]],
        "test_003": [[3, 0, 3, 0], [0, 3, 0, 3]]
    }
    
    ground_truth = {
        "test_001": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],  # Correct
        "test_002": [[1, 1], [1, 1]],                   # Color error
        "test_003": [[3, 0, 3], [0, 3, 0]]              # Size error
    }
    
    problem_metadata = {
        "test_001": {"task_type": "pattern_completion"},
        "test_002": {"task_type": "color_transformation"},
        "test_003": {"task_type": "size_scaling"}
    }
    
    consciousness_states = {
        "test_002": {"consciousness_coherence": 0.3, "reasoning_depth": 0.4},
        "test_003": {"consciousness_coherence": 0.6, "reasoning_depth": 0.7}
    }
    
    prediction_confidences = {
        "test_001": 0.9,
        "test_002": 0.4,
        "test_003": 0.7
    }
    
    # Perform error analysis
    result = analyzer.analyze_errors(
        predictions=predictions,
        ground_truth=ground_truth,
        problem_metadata=problem_metadata,
        consciousness_states=consciousness_states,
        prediction_confidences=prediction_confidences
    )
    
    # Generate report
    report = analyzer.generate_error_report(result)
    
    print(f"Error Analysis Complete!")
    print(f"Total Errors: {result.total_errors}")
    print(f"Error Rate: {result.error_rate:.1%}")
    print(f"Failure Patterns: {len(result.failure_patterns)}")


if __name__ == "__main__":
    main()